"""Season-long goalie projections for the upcoming season (2026-27).

Goaltending is the part of a hockey team that is most obviously a closed system, and the
old version of this module ignored every one of the constraints. It projected 114
appearances per team (a team has 84 games), 1643 league wins (32 teams x 84 games / 2 =
1344 exactly), and a "GAA" that was goals divided by APPEARANCES rather than by ice time
-- reading the NHL feed's `timeOnIce`, which is in seconds, as though a goalie's game
were a fixed unit. It also had no age term at all, so a 38-year-old and a 28-year-old
with the same three seasons behind them came out identical, and it double-counted team
strength: a goalie's own win rate already contains his old team, and that number was then
multiplied by his NEW team's quality on top.

So the module now works the way the skater module does -- a CLAIM per goalie, settled
against the team's budget -- and the chain is built in the order the constraints bind:

    starts  ->  relief appearances  ->  minutes  ->  shots against  ->  goals against
                                                                    ->  wins, shutouts

Each link is settled against something measured (budgets.goalie_budgets), and the
downstream numbers are DERIVED rather than projected separately, which is what makes them
consistent with each other for the first time:

    save percentage = 1 - goals against / shots against
    GAA             = goals against x 60 / MINUTES
    saves           = shots against - goals against

Three of the budgets are accounting identities rather than estimates: 84 starts, 60.1
minutes of goaltending per team-game, and half the league's games won. What is left over
is a genuine claim about the goalie: how many of his team's starts he takes, and how many
of the shots he faces go in. That second one carries the age curve (age_curves.
build_goalie_age_curves, measured on goals allowed per shot rather than on save
percentage, and damped by C.GOALIE_AGE_STRENGTH).

Team quality now enters in exactly ONE place -- the budget -- so it cannot be counted
twice. Within a team, the better goalie wins a slightly larger share of his starts than
the worse one, and that is the only thing the win claim says.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import age_curves as ac
import allocate as al
import budgets as bg
import config as C
import data_layer as dl
import overrides as ov

FREE_AGENT = "FA"

# A relief appearance is not a game: measured against the identity below it averages
# about half of one. Kept explicit because it is the only part of the minutes chain that
# is not forced by the clock.
RELIEF_MINUTES = 27.0

# Shots faced per 60 is a property of the TEAM, not of the goalie -- it is his skaters'
# defending, and for a goalie who changed teams his own history is actively misleading.
# So it is regressed toward the league far harder than his save rate is, and the level
# comes from the new team's shot-suppression budget instead.
SA_RATE_REGRESS_MULT = 3.0

# Season totals a scenario can state outright; each is honoured exactly and comes out of
# the team budget before anyone else is allocated anything.
LOCKABLE = ("starts", "gp", "minutes", "wins", "shutouts", "shots_against",
            "goals_against")


# --------------------------------------------------------------------------- #
# history                                                                     #
# --------------------------------------------------------------------------- #
def _prep_goalie_seasons() -> pd.DataFrame:
    """One row per goalie-season, with every rate the projection needs.

    The sample gate is ICE TIME, not shots faced. The old `shotsAgainst >= 100` filter
    silently discarded a goalie's short seasons, which are exactly the seasons that tell
    you he is a third-stringer -- and since the workload claim is now only a statement
    about how a team's starts SPLIT, throwing them away biased that split.
    """
    gs = dl.load_nhl_goalie_summary()
    keep = ["playerId", "goalieFullName", "teamAbbrevs", "mp_season_year",
            "gamesPlayed", "gamesStarted", "wins", "losses", "otLosses",
            "savePct", "shutouts", "saves", "shotsAgainst", "goalsAgainst", "timeOnIce"]
    g = gs[keep].copy()
    g = g[(g["timeOnIce"] >= C.MIN_ICETIME_SEC) & (g["shotsAgainst"] > 0)].copy()

    g["minutes"] = g["timeOnIce"] / 60.0          # the feed is SECONDS
    g["ga_per_shot"] = g["goalsAgainst"] / g["shotsAgainst"]
    g["sa_per_60"] = g["shotsAgainst"] * 60.0 / g["minutes"]
    starts = g["gamesStarted"].clip(lower=1)
    g["relief_per_start"] = ((g["gamesPlayed"] - g["gamesStarted"]) / starts).clip(0, 3)
    g["so_per_start"] = g["shutouts"] / starts
    g["win_rate"] = g["wins"] / g["gamesPlayed"]
    g["start_share"] = (g["gamesStarted"] / g["gamesPlayed"]).clip(upper=1.0)

    births = dl.goalie_birthdates(dl.load_nhl_goalie_bios())[["playerId", "birthDate"]]
    g = g.merge(births, on="playerId", how="left")
    bd = pd.to_datetime(g["birthDate"], errors="coerce", utc=True).dt.tz_localize(None)
    ref = pd.to_datetime((g["mp_season_year"] + 1).astype(str) + "-02-01", errors="coerce")
    g["age"] = (ref - bd).dt.days / 365.25
    return g


def _goalie_age(age: float | None) -> float:
    """Clamp to the range the goalie curves are measured over (20-42)."""
    if age is None or not np.isfinite(age):
        return float(ac.GOALIE_PEAK_ANCHOR)
    return float(np.clip(age, ac.GOALIE_AGE_MIN, ac.GOALIE_AGE_MAX))


def _age_mult(curves: dict, stat: str, from_age: float, to_age: float) -> float:
    c = curves.get(stat)
    if not c:
        return 1.0
    fa = int(round(_goalie_age(from_age)))
    ta = int(round(_goalie_age(to_age)))
    denom = c.get(fa, 1.0)
    return (c.get(ta, 1.0) / denom) if denom > 0 else 1.0


# --------------------------------------------------------------------------- #
# stage 1: claims                                                             #
# --------------------------------------------------------------------------- #
def _claim_rows(pool: pd.DataFrame, curves: dict, lvl: dict, regress_k: float,
                target: int, max_gp: float) -> list[dict]:
    """One unconstrained claim per goalie with a real NHL history."""
    years = sorted(pool["mp_season_year"].unique(), reverse=True)
    rate_w = dict(zip([target - 1, target - 2, target - 3], C.GOALIE_RECENCY_WEIGHTS))
    lg_ga_per_shot = 1.0 - lvl["sv_pct"]
    season_scale = max_gp / 82.0
    rows = []

    for pid, g in pool.groupby("playerId"):
        g = g.sort_values("mp_season_year")
        latest = g.iloc[-1]
        g = g.assign(rec_w=g["mp_season_year"].map(rate_w).fillna(0.0))
        shots = float(g["shotsAgainst"].sum())
        rw = (g["rec_w"] * g["shotsAgainst"]).to_numpy(dtype=float)
        if rw.sum() <= 0 or shots <= 0:
            continue

        birth = latest["birthDate"]
        bd = pd.to_datetime(birth, errors="coerce", utc=True)
        target_age = _goalie_age(
            ((pd.Timestamp(f"{target + 1}-02-01", tz="UTC") - bd).days / 365.25)
            if pd.notna(bd) else np.nan)
        ref_age = _goalie_age(float(np.average(g["age"].fillna(target_age), weights=rw)))

        # Shot-stopping: blend recent goals-allowed-per-shot, weighted by recency AND by
        # shots faced (a 2000-shot season says far more than a 400-shot one), regress hard
        # toward the league -- save percentage is the most luck-dominated rate in the sport
        # -- then age it. A multiplier above 1.0 means he lets in more, i.e. is worse.
        blended = float(np.average(g["ga_per_shot"], weights=rw))
        rate_ga = (blended * shots + lg_ga_per_shot * regress_k) / (shots + regress_k)
        rate_ga *= _age_mult(curves, "ga_per_shot", ref_age, target_age)
        rate_ga = float(np.clip(rate_ga, 0.055, 0.16))

        # Workload. Only the SHAPE matters now: the level is the team's 84 starts, so this
        # decides how a depth chart splits rather than how many starts exist. Last season
        # dominates, because a depth chart is a recent fact.
        ww = C.GOALIE_WORKLOAD_WEIGHTS
        num = den = 0.0
        for i, yr in enumerate([y for y in years if y in set(g["mp_season_year"])][:len(ww)]):
            w = ww[i]
            num += w * float(g.loc[g["mp_season_year"] == yr, "gamesStarted"].iloc[0])
            den += w
        claim_starts = (num / den) * season_scale if den > 0 else 1.0
        claim_starts *= _age_mult(curves, "workload", ref_age, target_age)
        claim_starts = float(np.clip(claim_starts, 1.0, C.MAX_GOALIE_STARTS))

        gw = (g["rec_w"] * g["gamesPlayed"]).to_numpy(dtype=float)
        gw = gw if gw.sum() > 0 else np.ones(len(g))
        relief = float(np.average(g["relief_per_start"], weights=gw))
        so_rate = float(np.average(g["so_per_start"], weights=gw))
        sa60 = float(np.average(g["sa_per_60"], weights=rw))
        k_sa = regress_k * SA_RATE_REGRESS_MULT
        rate_sa60 = (sa60 * shots + lvl["sa_per_60"] * k_sa) / (shots + k_sa)

        rows.append({
            "playerId": int(pid), "name": latest["goalieFullName"],
            "prior_team": str(latest["teamAbbrevs"]).split(",")[0],
            "target_age": round(target_age, 1), "ref_age": round(ref_age, 1),
            "seasons": int(len(g)), "sample_shots": shots, "rookie": False,
            "sample_starts": float(latest["gamesStarted"]),
            # How much workload EVIDENCE he has, which is what decides how far the
            # depth-chart prior is allowed to pull his claim (see _depth_chart).
            "start_history": float(g["gamesStarted"].sum()),
            "sample_save_pct": round(1.0 - blended, 4),
            "rate_ga_per_shot": rate_ga, "rate_sa_per_60": rate_sa60,
            "rate_relief_per_start": relief,
            "rate_so_per_start": 0.6 * so_rate + 0.4 * lvl["shutouts_per_start"],
            "claim_starts": claim_starts,
        })
    return rows


def _rookie_rows(roster: pd.DataFrame, known: set[int], curves: dict,
                 lvl: dict, target: int) -> list[dict]:
    """Rostered goalies with no NHL history, projected from the league prior.

    Five of the 2026-27 rostered goalies had never played an NHL game and were dropped
    entirely, which handed their starts to the veteran ahead of them. A third-stringer's
    share of a season is small but it is not zero, and if he is actually the plan the app
    can say so.
    """
    rows = []
    for _, p in roster[roster["roster_group"] == "G"].iterrows():
        pid = int(p["playerId"])
        if pid in known:
            continue
        bd = pd.to_datetime(p.get("birthDate"), errors="coerce", utc=True)
        target_age = _goalie_age(
            ((pd.Timestamp(f"{target + 1}-02-01", tz="UTC") - bd).days / 365.25)
            if pd.notna(bd) else np.nan)
        anchor = float(ac.GOALIE_PEAK_ANCHOR)
        rows.append({
            "playerId": pid, "name": p.get("fullName"), "prior_team": p["team"],
            "target_age": round(target_age, 1), "ref_age": anchor, "seasons": 0,
            "sample_shots": 0.0, "rookie": True, "sample_starts": 0.0,
            "start_history": 0.0, "sample_save_pct": np.nan,
            # No history means no opinion beyond the one thing that IS measurable about a
            # goalie who has never played: a debut season allows 2.1% more goals per shot
            # than the league. The age curve is deliberately not used here -- extrapolating
            # it seven years back from its anchor made a 20-year-old the best prior in the
            # league (see age_curves.build_goalie_age_curves).
            "rate_ga_per_shot": (1.0 - lvl["sv_pct"]) * C.GOALIE_DEBUT_PENALTY,
            "rate_sa_per_60": lvl["sa_per_60"],
            "rate_relief_per_start": lvl["appearances_per_start"] - 1.0,
            "rate_so_per_start": lvl["shutouts_per_start"],
            "claim_starts": C.SEASON_GAMES * C.ROOKIE_GOALIE_START_SHARE,
        })
    return rows


# --------------------------------------------------------------------------- #
# scenario edits                                                              #
# --------------------------------------------------------------------------- #
def _init_edit_cols(out: pd.DataFrame) -> None:
    out["edited"] = False
    out["lock_sv"] = False
    for col in LOCKABLE:
        out[f"lock_{col}"] = False
        out[f"fixed_{col}"] = np.nan


def _apply_edits(out: pd.DataFrame, sc: ov.Scenario | None) -> None:
    """Scenario edits. Inputs flow through the chain; totals are honoured exactly.

    `save_pct` is the interesting one: it is a RATE, so stating it changes goals against,
    saves, GAA and the team's implied save percentage together, and the goalie's teammates
    absorb the goals he no longer allows. That is what makes it an argument about hockey
    rather than a patch on a spreadsheet cell.
    """
    _init_edit_cols(out)
    if sc is None or not sc.goalies:
        return
    row_of = {int(p): i for i, p in enumerate(out["playerId"])}
    for pid_s, edits in sc.goalies.items():
        try:
            i = row_of[int(pid_s)]
        except (KeyError, ValueError):
            continue
        out.at[i, "edited"] = True
        if "team" in edits:
            out.at[i, "team"] = str(edits["team"])
            out.at[i, "on_roster"] = True
        if "on_roster" in edits:
            keep = bool(edits["on_roster"])
            out.at[i, "on_roster"] = keep
            if not keep:
                out.at[i, "team"] = FREE_AGENT
        if "save_pct" in edits:
            sv = float(np.clip(edits["save_pct"], 0.5, 0.999))
            out.at[i, "rate_ga_per_shot"] = 1.0 - sv
            out.at[i, "lock_sv"] = True
        if "starts" in edits:
            out.at[i, "claim_starts"] = float(np.clip(edits["starts"], 0, C.SEASON_GAMES))
            out.at[i, "lock_starts"] = True
            out.at[i, "fixed_starts"] = out.at[i, "claim_starts"]
        for field in ("gp", "minutes", "wins", "shutouts", "shots_against",
                      "goals_against"):
            if field in edits:
                out.at[i, f"lock_{field}"] = True
                out.at[i, f"fixed_{field}"] = float(max(edits[field], 0.0))
        # The remaining ratings, stated directly. These are the same three numbers the
        # model measures off his history, so overwriting one is an argument about the
        # goalie rather than about his totals: shots faced per 60 flows into shots
        # against and saves, shutouts per start into shutouts, relief appearances per
        # start into games played.
        for field, lo, hi in (("rate_sa_per_60", 0.0, 60.0),
                              ("rate_so_per_start", 0.0, 0.5),
                              ("rate_relief_per_start", 0.0, 5.0)):
            if field in edits:
                out.at[i, field] = float(np.clip(edits[field], lo, hi))


# --------------------------------------------------------------------------- #
# stage 2: settlement                                                         #
# --------------------------------------------------------------------------- #
def _depth_chart(out: pd.DataFrame, gb: pd.DataFrame) -> pd.Series:
    """Turn each team's claims into a share of its 84 starts, and return the coverage.

    A goalie's own history is a good guide to WHERE he is on a depth chart and a poor one
    to how much that position plays: it cannot see a 24-year-old promoted to starter, and
    it keeps projecting 60 starts for a proven starter who has just been traded into a
    tandem. So the claims are ranked within the team, and the resulting share is blended
    with the league's measured share for that rank (C.GOALIE_RANK_SHARES), which is
    measured by CLAIM rank precisely so that it can be used here without smuggling in
    hindsight.

    How hard that prior pulls is per goalie and set by EVIDENCE, not taste: a goalie with
    three 60-start seasons behind him barely moves, a debutant IS the prior. Both this and
    the flatter-than-proportional own share are calibrated against 269 historical
    team-seasons and hold up out of sample -- see the constants in config.

    Then the better goalie is given the net: the share is tilted by shot-stopping relative
    to his own teammates, which is a real and measured effect the workload history is blind
    to, and the single biggest improvement to this allocation.

    The cumulative rank shares are also the team's coverage -- the fraction of its
    goaltending its listed goalies account for -- so the split and the shortfall come from
    one measured curve instead of two assumptions that could disagree.

    Writes `claim_start_share` and rewrites `claim_starts` as a share of 84; returns the
    per-team coverage.
    """
    ranks = np.asarray(C.GOALIE_RANK_SHARES, dtype=float)
    dial = float(C.GOALIE_RANK_PRIOR_WEIGHT)
    k = float(C.GOALIE_RANK_PRIOR_STARTS)
    hist = out["start_history"].fillna(0.0).to_numpy(dtype=float)
    pull = dial * k / (k + np.clip(hist, 0.0, None))
    beta = float(C.GOALIE_QUALITY_START_TILT)
    log_ga = np.log(out["rate_ga_per_shot"].clip(lower=0.04).to_numpy(dtype=float))
    claim = out["claim_starts"].to_numpy(dtype=float)
    team = out["team"].to_numpy()
    on = (out["on_roster"] & out["team"].isin(gb.index)).to_numpy()
    share = np.zeros(len(out), dtype=float)
    cov = {}
    for t in gb.index:
        pos = np.flatnonzero(on & (team == t))
        if pos.size == 0:
            cov[t] = float(ranks[:1].sum())
            continue
        c = claim[pos]
        prior = np.zeros(pos.size)
        for r, i in enumerate(np.argsort(-c)):
            prior[i] = ranks[r] if r < len(ranks) else 0.0
        cov[t] = float(prior.sum())
        # Flatter than proportional: summed claims over-count, most of all for the depth
        # goalie whose recent starts came from someone else's injury (C.GOALIE_CLAIM_SHARE_EXP).
        cw = c ** float(C.GOALIE_CLAIM_SHARE_EXP)
        own = (cw / cw.sum() * prior.sum()) if cw.sum() > 0 else prior
        p = pull[pos]
        blend = (1.0 - p) * own + p * prior
        # Per-goalie weights do not preserve the team total the way one shared weight
        # does, so the blend is renormalised back onto the coverage. Without this a team
        # of veterans would quietly claim more of the season than a team of unknowns.
        s = blend * (cov[t] / blend.sum()) if blend.sum() > 0 else prior.copy()
        # The better goalie gets the net. Centred on the team's own mean, so this decides
        # WHO starts and never how many starts exist; a goalie is compared with the
        # partner he is actually competing with rather than with the league.
        if beta and pos.size > 1:
            lq = log_ga[pos]
            s = np.clip(s - beta * cov[t] * (lq - lq.mean()), 0.005 * cov[t], None)
            s *= cov[t] / s.sum()
        share[pos] = s
    out["claim_start_share"] = np.round(share, 4)
    games = gb["games"].reindex(pd.Index(team)).to_numpy(dtype=float)
    # Off-roster goalies keep their own claim: they have no team budget to share. A stated
    # number of starts also survives -- it was what ranked the depth chart in the first
    # place, and the settlement honours it out of the budget before anyone else is served.
    adj = np.where(on & np.isfinite(games), share * games, claim)
    fixed = out["fixed_starts"].to_numpy(dtype=float)
    out["claim_starts"] = np.where(np.isfinite(fixed), fixed, adj)
    return pd.Series(cov, dtype=float).reindex(gb.index).fillna(1.0)


def _settle(out: pd.DataFrame, gb: pd.DataFrame, tilt: float, lvl: dict) -> None:
    """Settle the chain, each link against the budget the link above it leaves."""
    cov = _depth_chart(out, gb)
    gb["coverage"] = cov

    def budget(col: str, scale: bool = True) -> dict:
        if col not in gb:
            return {}
        b = gb[col] * cov if scale else gb[col]
        return b.to_dict()

    # 1. Starts. A team makes exactly 84 of them, and its listed goalies take the share of
    #    those the depth chart says they do -- which `_depth_chart` has already written
    #    into the claim, so this settlement only has to honour locks and the ceiling.
    out["proj_starts"] = al.settle_frame(
        out, "claim_starts", budget("starts"), tilt=tilt, lock_col="lock_starts",
        cap=float(C.MAX_GOALIE_STARTS))

    # 2. Relief appearances, settled separately from starts so that appearances can never
    #    come out below starts -- which is what a single appearances budget would allow.
    relief_budget = ((gb["appearances"] - gb["starts"]) * cov).to_dict()
    out["claim_relief"] = out["proj_starts"] * out["rate_relief_per_start"]
    out["proj_relief"] = al.settle_frame(out, "claim_relief", relief_budget, tilt=1.0)
    out["proj_gp"] = out["proj_starts"] + out["proj_relief"]
    lock_gp = out["lock_gp"].to_numpy(dtype=bool)
    if lock_gp.any():
        fixed = out["fixed_gp"].to_numpy(dtype=float)
        out["proj_gp"] = np.where(lock_gp, np.maximum(fixed, out["proj_starts"]),
                                  out["proj_gp"])
        out["proj_relief"] = (out["proj_gp"] - out["proj_starts"]).clip(lower=0.0)

    # 3. Minutes. There is one goalie on the ice, so a team's 60.1 minutes per game are
    #    not a projection: they follow from who starts. A start is worth `start_min` and a
    #    relief appearance about half a game, and `start_min` is solved from the budget
    #    identity rather than guessed, so the minutes claim already sums to the budget and
    #    the settlement below is a consistency check plus a home for a stated total.
    start_min = float((gb["minutes"] - (gb["appearances"] - gb["starts"]) * RELIEF_MINUTES)
                      .div(gb["starts"]).mean())
    out["claim_minutes"] = out["proj_starts"] * start_min + out["proj_relief"] * RELIEF_MINUTES
    out["proj_minutes"] = al.settle_frame(
        out, "claim_minutes", budget("minutes"), tilt=1.0, lock_col="lock_minutes",
        cap=(out["proj_gp"].to_numpy() * 68.0))
    out["min_per_gp"] = np.where(out["proj_gp"] > 0,
                                 out["proj_minutes"] / out["proj_gp"].replace(0, np.nan), 0.0)

    # 4. Shots against: the team's, divided among its goalies by the minutes they play.
    out["claim_shots_against"] = out["proj_minutes"] * out["rate_sa_per_60"] / 60.0
    out["proj_shots_against"] = al.settle_frame(
        out, "claim_shots_against", budget("shots_against"), tilt=1.0,
        lock_col="lock_shots_against")

    # 5. Goals against, given the shots. This is where the goalie's own skill lives, and
    #    a stated save percentage is turned into a locked total HERE, now that the shots
    #    he will face are known -- an edit of ".920" means the ratio, not a count.
    out["claim_goals_against"] = out["proj_shots_against"] * out["rate_ga_per_shot"]
    sv_lock = out["lock_sv"].to_numpy(dtype=bool)
    if sv_lock.any():
        out["lock_goals_against"] = out["lock_goals_against"] | out["lock_sv"]
        out["fixed_goals_against"] = np.where(
            sv_lock & out["fixed_goals_against"].isna(),
            out["claim_goals_against"], out["fixed_goals_against"])
    fixed_ga = out["fixed_goals_against"]
    out["claim_goals_against"] = np.where(fixed_ga.notna(), fixed_ga,
                                          out["claim_goals_against"])
    out["proj_goals_against"] = al.settle_frame(
        out, "claim_goals_against", budget("goals_against"), tilt=1.0,
        lock_col="lock_goals_against")

    # 6. Wins. Team quality is in the BUDGET, so the claim only says how a team's wins
    #    split: in proportion to starts, tilted by how much better than league the goalie
    #    stops the puck. The old model multiplied a win rate that already contained his
    #    former team by his new team's quality and never reconciled the league total.
    lg_ga = 1.0 - lvl["sv_pct"]
    qual = (lg_ga / out["rate_ga_per_shot"].clip(lower=0.02)) ** C.GOALIE_WIN_QUALITY_EXP
    out["quality"] = qual.round(3)
    out["claim_wins"] = out["proj_starts"] * 0.5 * qual
    out["proj_wins"] = al.settle_frame(out, "claim_wins", budget("wins"), tilt=1.0,
                                       lock_col="lock_wins",
                                       cap=out["proj_gp"].to_numpy())
    out["claim_shutouts"] = out["proj_starts"] * out["rate_so_per_start"] * qual
    out["proj_shutouts"] = al.settle_frame(out, "claim_shutouts", budget("shutouts"),
                                           tilt=1.0, lock_col="lock_shutouts",
                                           cap=out["proj_starts"].to_numpy())

    # 7. Everything else is derived, which is the point: these used to be projected
    #    separately and could contradict each other.
    sa = out["proj_shots_against"].to_numpy(dtype=float)
    ga = out["proj_goals_against"].to_numpy(dtype=float)
    mins = out["proj_minutes"].to_numpy(dtype=float)
    out["proj_saves"] = sa - ga
    out["proj_save_pct"] = np.where(sa > 0, 1.0 - ga / np.maximum(sa, 1e-9), lvl["sv_pct"])
    out["proj_gaa"] = np.where(mins > 0, ga * 60.0 / np.maximum(mins, 1e-9), np.nan)
    # A record needs three numbers. Overtime losses are a league rate on decisions, not a
    # goalie skill, so they are taken at that rate and the regulation losses are what is
    # left of his starts.
    otl_share = lvl.get("otl_per_start", 0.0)
    out["proj_otl"] = (out["proj_starts"] * otl_share).clip(lower=0.0)
    out["proj_losses"] = (out["proj_starts"] - out["proj_wins"] - out["proj_otl"]).clip(lower=0.0)

    _record_coverage(out, gb)


def _record_coverage(out: pd.DataFrame, gb: pd.DataFrame) -> None:
    """What the listed goalies were allocated, and what is left for goalies not yet named."""
    on = out[out["on_roster"] & out["team"].isin(gb.index)]
    for alloc, bcol in (("proj_starts", "starts"), ("proj_gp", "appearances"),
                        ("proj_minutes", "minutes"), ("proj_wins", "wins"),
                        ("proj_shots_against", "shots_against"),
                        ("proj_goals_against", "goals_against"),
                        ("proj_shutouts", "shutouts")):
        got = on.groupby("team")[alloc].sum().reindex(gb.index).fillna(0.0)
        gb[f"{bcol}_allocated"] = got
        gb[f"depth_{bcol}"] = (gb[bcol] - got).clip(lower=0.0)
    gb["sv_pct_allocated"] = 1.0 - gb["goals_against_allocated"] / \
        gb["shots_against_allocated"].replace(0, np.nan)


# --------------------------------------------------------------------------- #
# prediction intervals                                                        #
# --------------------------------------------------------------------------- #
# Empirically-calibrated goalie interval widths (backtest 2022-25, 293 goalie-seasons).
# WINS behave like count data: residual std ~ sqrt(proj_wins), and unlike skaters the
# normalized std is uniform across workload (start_share doesn't separate goalies who
# play enough to project), so a single coefficient suffices. coef 2.2 at +-1.28 std
# gives ~82% coverage of an 80% target. SV% is a bounded, roughly homoscedastic rate:
# a flat additive std of 0.0151 gives ~79% coverage. p10/p90 (80% band) matches skaters.
PI_WINS_COEF = 2.2
PI_SVPCT_STD = 0.0151
_PI_Z = 1.2816  # 80% central interval


def _add_prediction_intervals(out: pd.DataFrame) -> None:
    """p10/p90 bands. Volume bands are right-skewed; the rate band is symmetric.

    Save percentage is the one goalie number whose band should NOT be lognormal: it is a
    bounded rate sitting near .900 with no meaningful skew, and it is measured directly.
    Wins, starts and appearances are counts, so they get the same mean-preserving
    lognormal the skater totals use -- a symmetric band on a backup's 8 wins ran below
    zero and had to be clipped, which quietly broke the coverage it was calibrated for.
    """
    def band(col: str, coef: float, floor: float, locked: str | None = None) -> None:
        mean = out[col].to_numpy(dtype=float)
        sigma = coef * np.sqrt(np.maximum(mean, floor))
        if locked and locked in out:
            sigma = np.where(out[locked].to_numpy(dtype=bool), 0.0, sigma)
        safe = np.maximum(mean, 1e-9)
        s2 = np.log1p((sigma / safe) ** 2)
        s = np.sqrt(s2)
        median = safe / np.exp(s2 / 2.0)
        stem = col.replace("proj_", "")
        out[f"{stem}_p10"] = np.round(np.where(mean > 0, median * np.exp(-_PI_Z * s), 0.0), 1)
        out[f"{stem}_p90"] = np.round(np.where(mean > 0, median * np.exp(+_PI_Z * s), 0.0), 1)

    band("proj_wins", PI_WINS_COEF, 2.0, "lock_wins")
    band("proj_saves", PI_WINS_COEF, 100.0)

    # Starts get the measured quantile table instead of a shape (C.GOALIE_START_BAND):
    # a backup's season is bimodal and no parametric band can be honest about it.
    tbl = C.GOALIE_START_BAND
    scale = float(C.SEASON_GAMES) / float(tbl["measured_season"])
    st = out["proj_starts"].to_numpy(dtype=float)
    x = np.asarray(tbl["proj"], dtype=float) * scale
    p10 = np.interp(st, x, np.asarray(tbl["p10"], dtype=float) * scale)
    p90 = np.interp(st, x, np.asarray(tbl["p90"], dtype=float) * scale)
    # A stated number of starts is not uncertain, and the band must not cross the
    # projection itself when a goalie sits outside the table's range.
    locked = out["lock_starts"].to_numpy(dtype=bool) if "lock_starts" in out else False
    out["starts_p10"] = np.round(np.where(locked, st, np.minimum(p10, st)), 1)
    out["starts_p90"] = np.round(np.where(locked, st, np.maximum(p90, st)), 1)

    sv_half = _PI_Z * PI_SVPCT_STD
    locked_sv = out["lock_sv"].to_numpy(dtype=bool) if "lock_sv" in out else False
    half = np.where(locked_sv, 0.0, sv_half)
    out["save_pct_p10"] = (out["proj_save_pct"] - half).clip(0, 1).round(4)
    out["save_pct_p90"] = (out["proj_save_pct"] + half).clip(0, 1).round(4)
    # GAA follows from the save-percentage band at the shot volume he is projected to
    # face, so the two can never tell a reader different stories: p10 GAA is the GOOD end.
    sa_per_60 = np.where(out["proj_minutes"] > 0,
                         out["proj_shots_against"] * 60.0 /
                         out["proj_minutes"].replace(0, np.nan), np.nan)
    out["gaa_p10"] = np.round(sa_per_60 * (1.0 - out["save_pct_p90"]), 2)
    out["gaa_p90"] = np.round(sa_per_60 * (1.0 - out["save_pct_p10"]), 2)


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def project_goalies(scenario: ov.Scenario | None = None, verbose: bool = False,
                    with_budgets: bool = False):
    """Season-long goalie projections, settled against team budgets.

    With `with_budgets`, returns (goalies, goalie_budgets); the budget frame carries each
    team's 84 starts, what its listed goalies covered and what is left for the goalies
    nobody has named yet, which is what the app shows to explain a number.
    """
    sc = scenario
    cfg = sc.settings() if sc is not None else C.league_defaults()
    tilt = float(cfg["budget_tilt"])
    enforce = bool(cfg["enforce_budgets"])
    max_gp = float(cfg["season_games"])
    regress_k = float(cfg["goalie_regress_shots"])

    target = C.TARGET_SEASON
    hist = _prep_goalie_seasons()
    curves = ac.build_goalie_age_curves()
    lvl = bg.goalie_league_level()
    lvl.setdefault("otl_per_start", _otl_per_start(hist))

    recent = [target - 1, target - 2, target - 3]
    pool = hist[hist["mp_season_year"].isin(recent)].copy()
    roster = dl.load_rosters(target)

    rows = _claim_rows(pool, curves, lvl, regress_k, target, max_gp)
    known = {r["playerId"] for r in rows}
    rows += _rookie_rows(roster, known, curves, lvl, target)
    out = pd.DataFrame(rows).reset_index(drop=True)

    # Team for the season being projected comes from the published roster; a goalie on no
    # roster is NOT left on last year's team, which is what used to happen to 29 of 106.
    cur_team = dict(zip(roster.loc[roster["roster_group"] == "G", "playerId"],
                        roster.loc[roster["roster_group"] == "G", "team"]))
    out["on_roster"] = out["playerId"].isin(cur_team)
    out["team"] = out["playerId"].map(cur_team).fillna(FREE_AGENT)

    # A September roster page lists the camp crease -- measured 4.3 goalies a team, one team
    # with six -- and the rank-share table only reaches five because seasons happen, not
    # because a team ever carries five. Left alone the extra bodies take the projected
    # starter under 40 starts (median top-goalie starts fell to 41.4 from the high forties).
    # Keep the top few by claim; the rest stay projected, marked `camp`, and giving one a
    # team in the app puts him back in a crease.
    out["camp"] = False
    out["camp_team"] = ""
    listed = out["on_roster"] & (out["team"] != FREE_AGENT)
    cut = []
    for _t, idx in out[listed].groupby("team").groups.items():
        if len(idx) > C.ROSTER_GOALIE_CAP:
            order = out.loc[idx, "claim_starts"].sort_values(ascending=False).index
            cut.extend(order[C.ROSTER_GOALIE_CAP:])
    if cut:
        out.loc[cut, "camp"] = True
        out.loc[cut, "camp_team"] = out.loc[cut, "team"]
        out.loc[cut, "on_roster"] = False
        out.loc[cut, "team"] = FREE_AGENT

    _apply_edits(out, sc)
    out["status"] = np.where(out["on_roster"], "roster",
                             np.where(out["camp"], "camp", "unsigned"))

    # The unconstrained numbers, kept so the app can show what settlement did.
    out["unc_starts"] = out["claim_starts"]
    out["unc_gp"] = out["claim_starts"] * (1.0 + out["rate_relief_per_start"])
    out["unc_minutes"] = out["unc_gp"] * lvl["min_per_appearance"]
    out["unc_shots_against"] = out["unc_minutes"] * out["rate_sa_per_60"] / 60.0
    out["unc_goals_against"] = out["unc_shots_against"] * out["rate_ga_per_shot"]
    out["unc_save_pct"] = (1.0 - out["rate_ga_per_shot"]).round(4)

    gb = None
    if not enforce:
        _no_budget_projection(out, lvl, max_gp)
    else:
        teams = sorted(t for t in out.loc[out["on_roster"], "team"].unique()
                       if t != FREE_AGENT)
        claims = out[out["on_roster"]].groupby("team")[
            ["unc_goals_against", "unc_shots_against"]].sum()
        claims.columns = ["goals_against", "shots_against"]
        gb = bg.goalie_budgets(target, teams=teams, games=int(max_gp), claims=claims,
                               shrink=float(cfg["team_rating_shrink"]))
        gb = _apply_team_edits(gb, sc)
        _settle(out, gb, tilt, lvl)
        if verbose:
            print(al.budget_report(out[out["on_roster"]], "claim_starts", "proj_starts",
                                   gb["starts"].to_dict()).round(2).to_string())

    _add_prediction_intervals(out)

    for col in out.columns:
        if col.startswith(("proj_", "claim_", "unc_")) and \
                pd.api.types.is_float_dtype(out[col]) and not col.endswith(
                    ("save_pct", "gaa")):
            out[col] = out[col].round(1)
    out["proj_save_pct"] = out["proj_save_pct"].round(4)
    out["proj_gaa"] = out["proj_gaa"].round(2)
    out = out.sort_values("proj_wins", ascending=False).reset_index(drop=True)
    out.attrs["scenario"] = "baseline" if sc is None else sc.name
    out.attrs["league_sv_pct"] = lvl["sv_pct"]
    return (out, gb) if with_budgets else out


def _otl_per_start(hist: pd.DataFrame) -> float:
    """League overtime losses per start, trend-weighted the same way the levels are."""
    last = C.LAST_COMPLETED_SEASON
    num = den = 0.0
    for w, season in zip(C.LEAGUE_RATE_WEIGHTS, [last - i for i in range(3)]):
        g = hist[hist["mp_season_year"] == season]
        if g.empty or g["gamesStarted"].sum() <= 0:
            continue
        num += w * float(g["otLosses"].sum()) / float(g["gamesStarted"].sum())
        den += w
    return num / den if den > 0 else 0.0


def _no_budget_projection(out: pd.DataFrame, lvl: dict, max_gp: float) -> None:
    """The old unconstrained behaviour, kept so the budgets can be switched off and seen."""
    out["proj_starts"] = out["claim_starts"].clip(0, max_gp)
    out["proj_relief"] = out["proj_starts"] * out["rate_relief_per_start"]
    out["proj_gp"] = out["proj_starts"] + out["proj_relief"]
    out["proj_minutes"] = out["proj_gp"] * lvl["min_per_appearance"]
    out["min_per_gp"] = lvl["min_per_appearance"]
    out["proj_shots_against"] = out["proj_minutes"] * out["rate_sa_per_60"] / 60.0
    out["proj_goals_against"] = out["proj_shots_against"] * out["rate_ga_per_shot"]
    out["proj_saves"] = out["proj_shots_against"] - out["proj_goals_against"]
    out["proj_save_pct"] = 1.0 - out["rate_ga_per_shot"]
    out["proj_gaa"] = out["proj_goals_against"] * 60.0 / out["proj_minutes"]
    out["quality"] = 1.0
    out["proj_wins"] = out["proj_starts"] * 0.5
    out["proj_shutouts"] = out["proj_starts"] * out["rate_so_per_start"]
    out["proj_otl"] = out["proj_starts"] * lvl.get("otl_per_start", 0.0)
    out["proj_losses"] = (out["proj_starts"] - out["proj_wins"] - out["proj_otl"]).clip(lower=0)


def _apply_team_edits(gb: pd.DataFrame, sc: ov.Scenario | None) -> pd.DataFrame:
    """A stated team goalie total replaces the budget; save percentage follows from it."""
    if sc is None or not sc.teams:
        return gb
    for team, edits in sc.teams.items():
        if team not in gb.index:
            continue
        for col, val in edits.items():
            key = col[7:] if col.startswith("goalie_") else col
            if key in ("shots_against", "goals_against", "wins", "shutouts",
                       "appearances", "minutes"):
                gb.at[team, key] = float(val)
        gb.at[team, "sv_pct"] = 1.0 - gb.at[team, "goals_against"] / gb.at[team, "shots_against"]
        gb.at[team, "gaa"] = gb.at[team, "goals_against"] * 60.0 / gb.at[team, "minutes"]
    return gb


if __name__ == "__main__":
    out, gb = project_goalies(with_budgets=True)
    on = out[out["on_roster"]]
    print(f"Projected {len(out)} goalies for {C.TARGET_SEASON}-{C.TARGET_SEASON+1} "
          f"({len(on)} on an NHL roster, {len(out) - len(on)} unsigned)\n")
    cols = ["name", "team", "target_age", "proj_starts", "proj_gp", "proj_minutes",
            "proj_wins", "proj_losses", "proj_otl", "proj_save_pct", "proj_gaa",
            "proj_shutouts", "wins_p10", "wins_p90"]
    print(on[cols].head(20).to_string(index=False))

    teams = len(gb)
    print("\nAccounting identities (allocated | budget | budget x coverage):")
    checks = [("starts", 84.0 * teams), ("wins", 42.0 * teams),
              ("appearances", None), ("minutes", None),
              ("shots_against", None), ("goals_against", None), ("shutouts", None)]
    alloc = {"starts": "proj_starts", "wins": "proj_wins", "appearances": "proj_gp",
             "minutes": "proj_minutes", "shots_against": "proj_shots_against",
             "goals_against": "proj_goals_against", "shutouts": "proj_shutouts"}
    for key, hard in checks:
        got = on[alloc[key]].sum()
        full = gb[key].sum()
        eff = (gb[key] * gb["coverage"]).sum()
        flag = "" if abs(got - eff) < max(0.01 * eff, 0.05) else "  <-- MISS"
        extra = f"   (identity {hard:.0f})" if hard else ""
        print(f"  {key:15s} {got:10.1f} | {full:10.1f} | {eff:10.1f}{flag}{extra}")
    cov = float(gb["coverage"].mean())
    print(f"\nListed goalies cover {cov:.1%} of the goaltending "
          f"({len(on) / teams:.2f} listed per team, 3.11 play in a real season); "
          f"{gb['depth_starts'].sum():.0f} starts left for goalies not yet named")
    print(f"League save percentage allocated "
          f"{1 - on['proj_goals_against'].sum() / on['proj_shots_against'].sum():.4f} "
          f"vs level {out.attrs['league_sv_pct']:.4f}")
    print(f"Minutes per appearance {on['proj_minutes'].sum() / on['proj_gp'].sum():.1f} "
          f"(league 57.1); busiest goalie {on['proj_starts'].max():.1f} starts "
          f"(cap {C.MAX_GOALIE_STARTS})")
    rk = out[out["rookie"]]
    print(f"\n{len(rk)} rostered goalies with no NHL history, now projected instead of dropped:")
    print(rk.nlargest(5, "proj_starts")[
        ["name", "team", "target_age", "proj_starts", "proj_save_pct", "proj_wins"]
    ].to_string(index=False))

    path = C.OUTPUT / f"goalie_projections_{C.TARGET_SEASON}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {path}")
