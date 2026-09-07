"""Season-long skater projections.

Two stages, and the separation between them is the whole design.

STAGE 1 -- THE CLAIM. What a player's own record says he would do if nothing
constrained him. Marcel-style, on a per-60 RATE basis so an injury-shortened season
does not read as lost talent:

  1. RATE BASE: blend the player's last five seasons of per-60 rates, front-loaded
     (RECENCY_WEIGHTS), each season further weighted by its own ice time.
  2. REGRESSION: pull that toward the mean of players in the same POSITION and USAGE
     TIER, by sampled ice time -- a first-line centre regresses toward other first-line
     centres, not toward a mean that includes fourth-liners.
  3. AGE: scale by the empirical delta-method age curve for that stat.
  4. VOLUME: project ice time per game (own history, age-trended) and games played.

  Rates that belong to a situation are measured against that situation's clock. Power
  play points per 60 minutes OF POWER PLAY is a skill that travels with a player; PP
  points per 60 minutes of total ice time -- which is what this model used to compute --
  is mostly a statement about how much power-play time his last coach gave him, so a
  promotion to the first unit could not show up in the projection at all.

STAGE 2 -- THE SETTLEMENT. A season is a closed system and the claims do not fit in it.
Every team has ~297 skater-minutes a night, dresses 18 skaters, and scores about as many
goals as its offence is worth. Before this stage existed the projections violated all
three, unevenly by team: 354 skater-minutes per game against a budget of 297 (and 278
for one team against 415 for another), 3.70 goals per team-game against 3.08, 35.1 shots
against 27.8. A reader could not see that error and could not fix it by overriding
players one at a time, because it is a property of the team.

So the claims are settled against the team's budget (see budgets.py), in the order the
constraints actually bind: games played, then ice time given the games, then production
given the ice time. Settlement takes the overshoot from the smallest claims first
(allocate.settle), because the surplus on a deep roster is bottom-of-the-lineup ice time
that will be scratched, not first-line ice time.

An override is honoured exactly and comes out of the budget FIRST, so stating that a
player scores 50 goals redistributes his teammates rather than being quietly rescaled
away. That is the difference between a projection you can read and one you can argue
with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_layer as dl
import age_curves as ac
import allocate as al
import budgets as bg
import live
import overrides as ov

# Rates measured against ALL of a player's ice time.
EVEN_STATS = ["goals", "primaryAssists", "secondaryAssists", "shots", "ixg",
              "blocks", "hits", "pim", "faceoffs_won"]
# Rates measured against one situation's ice time: stat -> situation prefix.
SIT_STATS = {"pp_points": "pp", "sh_points": "sh"}
RATE_STATS = EVEN_STATS + list(SIT_STATS)
# Stats that get a season total settled against a team budget.
COUNT_STATS = EVEN_STATS + list(SIT_STATS)

# MoneyPuck all-situations column for each even-strength-clock stat.
RAW_COLS = {
    "goals": "I_F_goals",
    "primaryAssists": "I_F_primaryAssists",
    "secondaryAssists": "I_F_secondaryAssists",
    "shots": "I_F_shotsOnGoal",
    "ixg": "I_F_xGoals",
    "blocks": "shotsBlockedByPlayer",
    "hits": "I_F_hits",
    "pim": "penalityMinutes",
    "faceoffs_won": "faceoffsWon",
}
# Which age curve a stat borrows when it has none of its own.
AGE_CURVE_FOR = {"secondaryAssists": "assists", "pp_points": "points",
                 "sh_points": "points"}
# Situations whose ice time is projected per game and then settled.
SIT_SOURCES = (("pp", "5on4"), ("sh", "4on5"))
SIT_PREFIXES = tuple(p for p, _ in SIT_SOURCES)


# --------------------------------------------------------------------------- #
# history                                                                     #
# --------------------------------------------------------------------------- #
def _situation_table(situation: str, prefix: str, with_live: bool = False) -> pd.DataFrame:
    """Per player-season points and ice time in one on-ice situation."""
    sit = dl.load_moneypuck_situation(situation)
    if with_live:
        sit = _append_live(sit, dl.load_live_skaters(situation=situation))
    pts = (sit["I_F_goals"] + sit["I_F_primaryAssists"] + sit["I_F_secondaryAssists"])
    return pd.DataFrame({
        "playerId": sit["playerId"].to_numpy(),
        "mp_season_year": sit["mp_season_year"].to_numpy(),
        f"{prefix}_points_raw": pts.to_numpy(),
        f"{prefix}_toi_min": sit["icetime"].to_numpy() / 60.0,
    })


def _append_live(hist: pd.DataFrame, live_rows: pd.DataFrame) -> pd.DataFrame:
    """Stack the season in progress onto the completed seasons.

    The live file is the same file, for a season that is not over. So it needs no special
    handling anywhere downstream: it is one more player-season, weighted by the ice time it
    contains, which after five games is about 5% of a full season and by March is all of it.
    Only the columns the history already has are carried over, so a new column appearing in
    the source cannot quietly change the shape of the frame the model is built on.
    """
    if live_rows is None or live_rows.empty:
        return hist
    cols = [c for c in hist.columns if c in live_rows.columns]
    return pd.concat([hist, live_rows[cols]], ignore_index=True)


def _prep_skater_seasons(with_live: bool = False) -> pd.DataFrame:
    """One row per player-season: per-60 rates on the right clock, ice time, age, position.

    `with_live` adds the season in progress. Off by default so that anything measuring the
    model against history -- the backtest above all -- cannot accidentally see a season that
    is not finished.
    """
    df = dl.load_moneypuck_skaters()
    if with_live:
        df = _append_live(df, dl.load_live_skaters(situation="all"))
    births = dl.player_birthdates(dl.load_nhl_skater_bios())
    df = df.merge(births[["playerId", "birthDate", "positionCode", "fullName"]],
                  on="playerId", how="left")
    # Prefer the NHL API's accented name over MoneyPuck's ASCII-stripped one.
    df["name"] = df["fullName"].fillna(df["name"])
    for prefix, situation in SIT_SOURCES:
        df = df.merge(_situation_table(situation, prefix, with_live=with_live),
                      on=["playerId", "mp_season_year"], how="left")
        df[f"{prefix}_points_raw"] = df[f"{prefix}_points_raw"].fillna(0.0)
        df[f"{prefix}_toi_min"] = df[f"{prefix}_toi_min"].fillna(0.0)

    df = df[df["icetime"] >= C.MIN_ICETIME_SEC].copy()
    df["toi_min"] = df["icetime"] / 60.0
    per60 = 60.0 / df["toi_min"]

    for stat, col in RAW_COLS.items():
        df[f"{stat}_raw"] = df[col]
        df[stat] = df[col] * per60
    df["points_raw"] = df["I_F_points"]
    df["points"] = df["I_F_points"] * per60
    df["assists_raw"] = df["primaryAssists_raw"] + df["secondaryAssists_raw"]
    df["assists"] = df["assists_raw"] * per60

    # Situation rates are per 60 minutes OF THAT SITUATION. A season with no power-play
    # time contributes a rate of zero AND a blend weight of zero, so it neither drags a
    # specialist down nor invents a rate for someone who never took a shift.
    for stat, prefix in SIT_STATS.items():
        toi = df[f"{prefix}_toi_min"].to_numpy(dtype=float)
        raw = df[f"{prefix}_points_raw"].to_numpy(dtype=float)
        df[stat] = np.where(toi > 0, raw * 60.0 / np.where(toi > 0, toi, 1.0), 0.0)
        df[f"{prefix}_toi_per_gp"] = toi / df["games_played"].to_numpy(dtype=float)

    pos = df["positionCode"].fillna(df["position"])
    df["pos_group"] = np.where(pos.isin(["D"]), "D", "F")
    df["toi_per_gp"] = df["toi_min"] / df["games_played"]

    bd = pd.to_datetime(df["birthDate"], errors="coerce")
    ref = pd.to_datetime((df["mp_season_year"] + 1).astype(str) + "-02-01", errors="coerce")
    df["age"] = (ref - bd).dt.days / 365.25
    return df


def _positional_means(hist: pd.DataFrame) -> dict:
    """Regression targets by position group AND usage tier, ice-time weighted.

    A single global positional mean is the wrong prior for a star: regressing a
    first-line centre toward an average that includes fourth-liners systematically
    under-projects elite players (backtest: ~-0.46 pts/60 of rate compression). The
    prior is instead the mean of players in the same TOI/game quartile.

    Situation rates use that situation's own ice time as the weight, so the power-play
    prior is set by players who actually play on the power play.
    """
    recent = hist[hist["mp_season_year"] >= C.LAST_COMPLETED_SEASON - 2].copy()
    means: dict = {}
    for grp, g in recent.groupby("pos_group"):
        w = g["toi_min"]
        gm = {stat: float(np.average(g[stat], weights=w)) for stat in EVEN_STATS}
        gm["points"] = float(np.average(g["points"], weights=w))
        for stat, prefix in SIT_STATS.items():
            sw = g[f"{prefix}_toi_min"]
            gm[stat] = float(np.average(g[stat], weights=sw)) if sw.sum() > 0 else 0.0
        gm["toi_per_gp"] = float(np.average(g["toi_per_gp"], weights=g["games_played"]))
        for prefix in SIT_PREFIXES:
            gm[f"{prefix}_toi_per_gp"] = float(
                np.average(g[f"{prefix}_toi_per_gp"], weights=g["games_played"]))
        gm["ref_age"] = float(np.average(g["age"].fillna(g["age"].mean()), weights=w))

        qs = g["toi_per_gp"].quantile([0.25, 0.50, 0.75]).to_numpy()
        gm["_toipg_q"] = qs
        gm["_tier"] = {}
        tier = np.searchsorted(qs, g["toi_per_gp"].to_numpy())
        for t in range(4):
            sub = g[tier == t]
            if len(sub) < 20:
                continue
            sw_all = sub["toi_min"]
            entry = {stat: float(np.average(sub[stat], weights=sw_all)) for stat in EVEN_STATS}
            entry["points"] = float(np.average(sub["points"], weights=sw_all))
            for stat, prefix in SIT_STATS.items():
                sw = sub[f"{prefix}_toi_min"]
                entry[stat] = float(np.average(sub[stat], weights=sw)) if sw.sum() > 0 else 0.0
            entry["toi_per_gp"] = float(np.average(sub["toi_per_gp"], weights=sub["games_played"]))
            for prefix in SIT_PREFIXES:
                entry[f"{prefix}_toi_per_gp"] = float(
                    np.average(sub[f"{prefix}_toi_per_gp"], weights=sub["games_played"]))
            entry["ref_age"] = float(np.average(sub["age"].fillna(sub["age"].mean()),
                                                weights=sw_all))
            gm["_tier"][t] = entry
        means[grp] = gm
    return means


def _tier_of(pm: dict, toipg: float) -> int:
    return int(np.searchsorted(pm["_toipg_q"], toipg))


def _tier_target(pm: dict, toipg: float, stat: str) -> float:
    """Regression target for `stat`: the usage-tier mean where there is one."""
    tier = pm["_tier"].get(_tier_of(pm, toipg))
    if tier is not None and stat in tier:
        return tier[stat]
    return pm.get(stat, 0.0)


# --------------------------------------------------------------------------- #
# stage 1: claims                                                             #
# --------------------------------------------------------------------------- #
def _project_gp(g: pd.DataFrame) -> tuple[float, float]:
    """Project games played from recent GP, recency-weighted, capped at MAX_GP.

    Returns (proj_gp, gp_reliability), the second in [0,1] reflecting how predictable
    this player's availability is: high for a consistently-healthy skater, low for an
    injury-prone or thin-history one. Backtest: durable players (min recent GP>=72) have
    GP MAE ~10 and actual-GP std ~13; injury-prone (<60) have MAE ~17 and std ~25. That
    difference is real and predictable, so it drives the interval width -- not the point
    estimate, because even iron men regress.
    """
    gp = dict(zip(g["mp_season_year"].tolist(), g["games_played"]))
    num = den = 0.0
    for i, w in enumerate(C.GP_RECENCY_WEIGHTS):
        yr = C.TARGET_SEASON - 1 - i
        if yr in gp:
            num += w * gp[yr]
            den += w
    if den == 0:
        return 60.0, 0.3
    base = num / den
    # Mild regression toward a full-season durability prior. Backtest-confirmed: even
    # players with a spotless recent record regress ~8 GP, so the hedge is correct.
    proj = 0.80 * base + 0.20 * 70.0
    recent_vals = [gp[C.TARGET_SEASON - 1 - i] for i in range(C.GP_RELIABILITY_SEASONS)
                   if (C.TARGET_SEASON - 1 - i) in gp]
    floor = min(recent_vals) if recent_vals else base
    spread = np.std(recent_vals) if len(recent_vals) >= 2 else 15.0
    rel = np.clip((floor - 45) / 35.0, 0, 1) * np.clip(1 - spread / 20.0, 0.2, 1.0)
    return float(np.clip(proj, 1, C.MAX_GP)), float(rel)


def _live_availability(hist_share: float, act_gp: float, team_played: float) -> float:
    """A player's share of the games AHEAD of him, given how many he has already missed.

    `hist_share` is what his history says (his projected full season over 84). The season in
    progress is evidence about the same quantity and is blended in at
    `n / (n + C.IN_SEASON_GP_K)` with K = 20 team-games: a player who has played 8 of his
    team's 20 is moved halfway toward being that player, not all the way. Injuries arrive in
    blocks of twenty games rather than as independent draws, so the naive per-game Bernoulli
    weight (K around 6) would read one six-week absence as a permanent condition; there is no
    injury feed here, so this is a prior and the honest thing to do with it is to move slowly
    and let a reader state the games on the player's page when he knows more than the model.
    """
    if team_played <= 0:
        return hist_share
    w = float(team_played / (team_played + C.IN_SEASON_GP_K))
    live_share = float(np.clip(act_gp / team_played, 0.0, 1.0))
    return float(np.clip((1.0 - w) * hist_share + w * live_share, 0.0, 1.0))


def _claim_rows(pool: pd.DataFrame, pos_means: dict, curves: dict,
                toi_curve: dict, regress_k: float,
                acts: dict | None = None, team_played: float = 0.0) -> list[dict]:
    """One unconstrained claim per player with NHL history.

    `acts` maps playerId -> (games, ice time) already played this season, and `team_played`
    is the league mean team-games so far. Given those, two quantities stop coming from
    history alone: ice time per game, which moves fast, and availability, which is returned
    as a SHARE of the games ahead rather than a count -- how many games are ahead depends on
    the player's team, which is not known until the roster is joined on.
    """
    target = C.TARGET_SEASON
    wmap = dict(zip([target - i for i in range(1, C.SKATER_HISTORY_SEASONS + 1)],
                    C.RECENCY_WEIGHTS))
    # The season in progress is the most recent season, so it earns the most recent season's
    # weight; the ice-time weighting inside the blend is what keeps a five-game sample from
    # behaving as though it were a five-hundred-minute one.
    wmap[target] = C.RECENCY_WEIGHTS[C.LIVE_RECENCY_WEIGHT_INDEX]
    acts = acts or {}
    rows = []
    for pid, g in pool.groupby("playerId"):
        g = g.sort_values("mp_season_year")
        latest = g.iloc[-1]
        pm = pos_means[latest["pos_group"]]

        rec_w = g["mp_season_year"].map(wmap).fillna(0.0).to_numpy(dtype=float)
        blend_w = rec_w * g["toi_min"].to_numpy(dtype=float)
        if blend_w.sum() <= 0:
            continue
        total_toi = float(g["toi_min"].sum())

        mean_age = float(np.average(g["age"], weights=blend_w))
        mean_year = float(np.average(g["mp_season_year"], weights=blend_w))
        target_age = mean_age + (target - mean_year)

        # Ice time per game from the player's own history, then age-trended: young
        # players earn bigger roles and veterans lose them, and projecting from history
        # alone under-projects risers (~-34% at 18-21) and over-projects fading vets.
        #
        # Measured on COMPLETED seasons only, deliberately. Ice time is the fastest-moving
        # thing in a season -- a promotion to the first line is visible in three games -- and
        # the ice-time weighting that is right for a per-60 rate is far too slow for the
        # minutes themselves: after eight games a 20-minute player has ~8% of the blend's
        # weight, where the measured answer (C.IN_SEASON_TOI_K = 8 player-games) is nearer
        # half. So the live season is left out here and blended in explicitly below.
        done = g[g["mp_season_year"] < target]
        if len(done):
            dw = blend_w[(g["mp_season_year"] < target).to_numpy()]
            sample_toipg = float(np.average(done["toi_per_gp"], weights=dw)) \
                if dw.sum() > 0 else float(done["toi_per_gp"].mean())
        else:
            sample_toipg = float(np.average(g["toi_per_gp"], weights=blend_w))
        toipg = sample_toipg * ac.age_multiplier({"toi": toi_curve}, "toi",
                                                 mean_age, target_age)
        act = acts.get(int(pid))
        live_gp = float(act["gp"]) if act else 0.0
        toi_w = live_gp / (live_gp + C.IN_SEASON_TOI_K) if live_gp > 0 else 0.0
        if toi_w > 0:
            toipg = (1.0 - toi_w) * toipg + toi_w * (act["toi"] / live_gp)

        row = {
            "playerId": int(pid), "name": latest["name"],
            "prior_team": latest["team"], "position": latest["position"],
            "pos_group": latest["pos_group"], "target_age": round(target_age, 1),
            "seasons": int(len(g)), "sample_toi_min": round(total_toi, 1),
            "sample_toi_per_gp": round(sample_toipg, 2),
            "usage_tier": _tier_of(pm, sample_toipg), "rookie": False,
        }

        for stat in EVEN_STATS + ["points"]:
            blended = float(np.average(g[stat], weights=blend_w))
            reg = (blended * total_toi + _tier_target(pm, sample_toipg, stat) * regress_k) \
                / (total_toi + regress_k)
            mult = ac.age_multiplier(curves, AGE_CURVE_FOR.get(stat, stat),
                                     mean_age, target_age)
            row[f"rate_{stat}"] = reg * mult

        # Situation rates: weighted by that situation's ice time, regressed on it, and
        # the ice time itself projected per game so a change of role can move the total.
        for stat, prefix in SIT_STATS.items():
            sit_toi = g[f"{prefix}_toi_min"].to_numpy(dtype=float)
            sw = rec_w * sit_toi
            sit_total = float(sit_toi.sum())
            blended = float(np.average(g[stat], weights=sw)) if sw.sum() > 0 else 0.0
            k = max(regress_k * 0.25, 1.0)   # far less situation ice time exists to sample
            reg = (blended * sit_total + _tier_target(pm, sample_toipg, stat) * k) \
                / (sit_total + k)
            mult = ac.age_multiplier(curves, AGE_CURVE_FOR.get(stat, stat),
                                     mean_age, target_age)
            row[f"rate_{stat}"] = reg * mult
            # Power-play ice time moves for the same reason total ice time does, and faster:
            # a promotion to the first unit is a decision, not a trend. Same explicit blend.
            sit_toipg = float(np.average(g[f"{prefix}_toi_per_gp"], weights=blend_w))
            if toi_w > 0:
                sit_toipg = (1.0 - toi_w) * sit_toipg + toi_w * (act[f"{prefix}_toi"] / live_gp)
            row[f"claim_{prefix}_toi_per_gp"] = sit_toipg

        proj_gp, gp_rel = _project_gp(g)
        row["claim_gp"] = proj_gp
        row["gp_reliability"] = gp_rel
        # Availability as a SHARE of whatever games are ahead. In a full-season run this is
        # just proj_gp/84 and multiplying it back out changes nothing; in season it is the
        # number that has to be used, because 84 games are not ahead of anybody.
        row["avail_share"] = _live_availability(proj_gp / float(C.SEASON_GAMES),
                                                live_gp, team_played)
        row["claim_toi_per_gp"] = toipg
        rows.append(row)
    return rows


def _rookie_rows(roster: pd.DataFrame, known: set, pos_means: dict,
                 curves: dict, toi_curve: dict) -> list[dict]:
    """Rostered skaters with no NHL history, projected from the tier prior alone.

    These used to be dropped, which was worse than a rough projection: their ice time
    and their points were silently handed to the veterans on the same team, so a rookie
    on the roster made the whole team's projection wrong rather than merely absent.
    """
    target = C.TARGET_SEASON
    sk = roster[(roster["roster_group"] != "G") & (~roster["playerId"].isin(known))]
    ref = pd.Timestamp(f"{target + 1}-02-01")
    rows = []
    for _, p in sk.iterrows():
        pos = p.get("positionCode") or "C"
        grp = "D" if pos == "D" else "F"
        pm = pos_means[grp]
        tier = pm["_tier"].get(C.ROOKIE_TOI_TIER) or pm
        bd = pd.to_datetime(p.get("birthDate"), errors="coerce")
        target_age = float((ref - bd).days / 365.25) if pd.notna(bd) else 23.0
        ref_age = tier.get("ref_age", pm["ref_age"])

        row = {"playerId": int(p["playerId"]), "name": p.get("fullName"),
               "prior_team": p["team"], "position": pos, "pos_group": grp,
               "target_age": round(target_age, 1), "seasons": 0,
               "sample_toi_min": 0.0, "usage_tier": C.ROOKIE_TOI_TIER, "rookie": True}
        for stat in EVEN_STATS + ["points"] + list(SIT_STATS):
            base = tier.get(stat, pm.get(stat, 0.0))
            row[f"rate_{stat}"] = base * ac.age_multiplier(
                curves, AGE_CURVE_FOR.get(stat, stat), ref_age, target_age)
        toipg = tier.get("toi_per_gp", pm["toi_per_gp"]) * ac.age_multiplier(
            {"toi": toi_curve}, "toi", ref_age, target_age)
        row["claim_toi_per_gp"] = toipg
        row["sample_toi_per_gp"] = round(toipg, 2)
        for prefix in SIT_PREFIXES:
            row[f"claim_{prefix}_toi_per_gp"] = tier.get(f"{prefix}_toi_per_gp", 0.0)
        row["claim_gp"] = C.ROOKIE_GP
        row["gp_reliability"] = C.ROOKIE_GP_RELIABILITY
        row["avail_share"] = C.ROOKIE_GP / float(C.SEASON_GAMES)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# scenario edits                                                              #
# --------------------------------------------------------------------------- #
def _init_edit_cols(out: pd.DataFrame) -> None:
    out["edited"] = False
    for col in ("gp", "toi", "pp_toi", "sh_toi", *COUNT_STATS):
        out[f"lock_{col}"] = False
        if col in COUNT_STATS:
            out[f"fixed_{col}"] = np.nan


def _apply_input_edits(out: pd.DataFrame, sc: ov.Scenario | None,
                       gp_csv: dict) -> pd.DataFrame:
    """Structural and input-side edits: team, availability, ice time, rates.

    These are causes, so they are applied BEFORE the season totals are derived and they
    flow all the way through -- raising a player's ice time raises his goals, shots and
    blocks together, and costs his teammates the minutes he gained.
    """
    _init_edit_cols(out)
    row_of = {int(p): i for i, p in enumerate(out["playerId"])}

    # The legacy games-played CSV, which was the only editable input the workbook had.
    if gp_csv:
        lowered = {str(n).strip().lower(): i for i, n in enumerate(out["name"])}
        for key, val in gp_csv.items():
            i = row_of.get(key) if isinstance(key, int) else lowered.get(key)
            if i is None:
                continue
            out.at[i, "claim_gp"] = val
            out.at[i, "lock_gp"] = True
            out.at[i, "gp_reliability"] = 1.0
            out.at[i, "edited"] = True

    if sc is None:
        return out
    for pid_s, edits in sc.players.items():
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
        if "gp" in edits:
            out.at[i, "claim_gp"] = float(np.clip(edits["gp"], 0, C.MAX_GP))
            out.at[i, "lock_gp"] = True
            # A stated status is certain, so it earns the tight interval.
            out.at[i, "gp_reliability"] = 1.0
        if "toi_per_gp" in edits:
            out.at[i, "claim_toi_per_gp"] = float(max(edits["toi_per_gp"], 0.0))
            out.at[i, "lock_toi"] = True
        for prefix, field in (("pp", "pp_toi_per_gp"),):
            if field in edits:
                out.at[i, f"claim_{prefix}_toi_per_gp"] = float(max(edits[field], 0.0))
                out.at[i, f"lock_{prefix}_toi"] = True
        for stat in RATE_STATS:
            key = ov.RATE_PREFIX + stat
            if key in edits:
                out.at[i, f"rate_{stat}"] = float(max(edits[key], 0.0))
    return out


def _apply_total_locks(out: pd.DataFrame, sc: ov.Scenario | None, lvl: dict) -> None:
    """Season totals stated outright. Honoured exactly, out of the budget first.

    `assists` and `points` are not stats the model allocates directly, so a lock on
    either is decomposed into the parts that are: assists at the league primary/secondary
    split, points across the player's OWN claimed goals and assists. That keeps the edit
    predictable -- a locked 100 points does not change what kind of player he is.
    """
    if sc is None:
        return
    row_of = {int(p): i for i, p in enumerate(out["playerId"])}
    p_share = lvl["primary_per_goal"] / (lvl["primary_per_goal"] + lvl["secondary_per_goal"])
    for pid_s, edits in sc.players.items():
        try:
            i = row_of[int(pid_s)]
        except (KeyError, ValueError):
            continue
        for stat in COUNT_STATS:
            if stat in edits:
                out.at[i, f"lock_{stat}"] = True
                out.at[i, f"fixed_{stat}"] = float(max(edits[stat], 0.0))
        if "assists" in edits:
            a = float(max(edits["assists"], 0.0))
            for stat, share in (("primaryAssists", p_share), ("secondaryAssists", 1 - p_share)):
                out.at[i, f"lock_{stat}"] = True
                out.at[i, f"fixed_{stat}"] = a * share
        if "points" in edits:
            want = float(max(edits["points"], 0.0))
            parts = ["goals", "primaryAssists", "secondaryAssists"]
            have = float(sum(out.at[i, f"claim_{s}"] for s in parts))
            for s in parts:
                share = (out.at[i, f"claim_{s}"] / have) if have > 0 else (1 / 3)
                out.at[i, f"lock_{s}"] = True
                out.at[i, f"fixed_{s}"] = want * share
    for stat in COUNT_STATS:
        fixed = out[f"fixed_{stat}"]
        out[f"claim_{stat}"] = np.where(fixed.notna(), fixed, out[f"claim_{stat}"])


# --------------------------------------------------------------------------- #
# stage 2: settlement                                                         #
# --------------------------------------------------------------------------- #
FREE_AGENT = "FA"


def _derive_totals(out: pd.DataFrame, toi: str, pp_toi: str, sh_toi: str,
                   into: str = "claim_") -> None:
    """Season totals from rates and ice time. Each stat on its own clock."""
    for stat in EVEN_STATS + ["points"]:
        out[f"{into}{stat}"] = out[f"rate_{stat}"] * out[toi] / 60.0
    for stat, prefix in SIT_STATS.items():
        col = {"pp": pp_toi, "sh": sh_toi}[prefix]
        out[f"{into}{stat}"] = out[f"rate_{stat}"] * out[col] / 60.0


def _apply_sos(tb: pd.DataFrame, sos: dict) -> pd.DataFrame:
    """Tilt the offensive budgets by strength of schedule, conserving the league total.

    An unbalanced (divisional) schedule is a real effect on a season total, but it is
    zero-sum across the league: an easier schedule for one team is a harder one for the
    teams it beats up on. So the factors are re-centred and each budget renormalised,
    which is why this belongs on the team budget and not on the player -- it is a
    property of who a team plays, and applying it per player let it inflate the league.
    """
    f = pd.Series(sos, dtype=float).reindex(tb.index).fillna(1.0)
    f = f / f.mean()
    for col in ("goals", "ixg", "shots", "pp_points", "sh_points"):
        if col in tb:
            scaled = tb[col] * f
            total = tb[col].sum()
            tb[col] = scaled * (total / scaled.sum()) if scaled.sum() > 0 else tb[col]
    return tb


def _cut_camp_bodies(out: pd.DataFrame) -> None:
    """Trim a training-camp roster down to a plausible active roster, in place.

    The roster endpoint lists whoever is in the building. In September that is the camp and
    the AHL affiliate -- 39 players a team where an active roster is 23 -- and since every
    listed player claims from the same fixed pool of skater-games and minutes, the extras
    are paid for by the regulars: 1856 claimed skater-games against a 1512 budget clawed
    ~19% off everybody, McDavid included.

    Who to keep is decided by claimed minutes, which is the model's own read on who wins a
    camp job and the only ranking available before a season starts. Everyone cut keeps his
    projection and is marked `camp`; he is simply not on a team, which is the same treatment
    an unsigned player already gets, and giving him a team in the app undoes this entirely.
    """
    minutes = out["claim_gp"] * out["claim_toi_per_gp"]
    listed = out["on_roster"] & (out["team"] != FREE_AGENT)
    cut = []
    for (_team, grp), idx in out[listed].groupby(["team", "pos_group"]).groups.items():
        cap = C.ROSTER_SKATER_CAP.get(grp)
        if cap is None or len(idx) <= cap:
            continue
        # Ties broken by claimed games, so two players with identical minutes are ordered
        # by the more durable one rather than by row order.
        order = (minutes.loc[idx] + 1e-6 * out.loc[idx, "claim_gp"]).sort_values(
            ascending=False).index
        cut.extend(order[cap:])
    if not cut:
        return
    out.loc[cut, "camp"] = True
    out.loc[cut, "camp_team"] = out.loc[cut, "team"]      # whose camp, for the app to show
    out.loc[cut, "on_roster"] = False
    out.loc[cut, "team"] = FREE_AGENT


# --------------------------------------------------------------------------- #
# the season in progress                                                      #
# --------------------------------------------------------------------------- #
# Everything the model settles -- ice time, games, goals -- is settled over a WINDOW. The
# window used to be the season, because the season had not started. These three functions
# are the whole of the difference: they narrow the window to the games that are left, keep a
# stated season total meaning a season total, and then add back what is already banked so
# every other column in the frame carries on meaning what it always meant.
def _to_rest_of_season_gp(out: pd.DataFrame, window: np.ndarray, acts: dict) -> None:
    """Rewrite the games-played claim as games REMAINING, in place.

    Preseason `claim_gp` is a count out of 84. In season the count that matters is out of
    what is left, and the durable part of the claim is the SHARE (`avail_share`, which the
    claim step already blended with how many games he has actually played). A locked games
    total is a statement about the season, so what it asks of the remaining games is the
    stated number minus the games he has played -- and if he has already passed it, zero.
    """
    banked = out["playerId"].map(lambda p: acts.get(int(p), {}).get("gp", 0.0)) \
        .astype(float).to_numpy()
    share = out["avail_share"].fillna(0.0).to_numpy(dtype=float) \
        if "avail_share" in out else np.zeros(len(out))
    ros = share * window
    locked = out["lock_gp"].to_numpy(dtype=bool) if "lock_gp" in out else np.zeros(len(out), bool)
    stated = out["claim_gp"].to_numpy(dtype=float)
    out["claim_gp"] = np.clip(np.where(locked, stated - banked, ros), 0.0, window)
    out["team_games_left"] = np.round(window, 1)


def _locks_to_rest_of_season(out: pd.DataFrame, acts: pd.DataFrame) -> None:
    """Turn every stated season total into what is left of it, in place."""
    if acts is None or acts.empty:
        return
    by_id = acts.set_index("playerId")
    for stat in COUNT_STATS:
        col = f"act_{stat}"
        if col not in by_id:
            continue
        banked = out["playerId"].map(by_id[col]).fillna(0.0).to_numpy(dtype=float)
        fixed = out[f"fixed_{stat}"].to_numpy(dtype=float)
        left = np.maximum(fixed - banked, 0.0)
        out[f"fixed_{stat}"] = np.where(np.isfinite(fixed), left, fixed)
        out[f"claim_{stat}"] = np.where(np.isfinite(fixed), left,
                                        out[f"claim_{stat}"].to_numpy(dtype=float))


# Every quantity that is a SUM over games, and so can be banked and added to. Rates
# (per-60s, per-game ice time) are not here on purpose: they are recomputed from the totals
# below, because a season-to-date average is the ratio of the two halves and not their sum.
FOLDABLE = COUNT_STATS + ["assists", "points", "gp", "toi", "pp_toi", "sh_toi"]


def _fold_in_actuals(out: pd.DataFrame, acts: pd.DataFrame, state) -> None:
    """`proj_* = act_* + ros_*`, in place, with the bands carried up with the mean.

    The bands are the reason this order matters. They were drawn on the rest of the season,
    where the uncertainty actually lives, so shifting them up by a banked total narrows them
    exactly as much as the season has narrowed them: a player 60 games into a season has 24
    games of spread left around a number that is mostly already decided. Drawing them on the
    full-season total instead would keep publishing a September-width band in March.
    """
    empty = pd.Series(0.0, index=out.index)
    by_id = acts.set_index("playerId") if acts is not None and not acts.empty else None

    def banked(name: str) -> pd.Series:
        if by_id is None or f"act_{name}" not in by_id:
            return empty
        return out["playerId"].map(by_id[f"act_{name}"]).fillna(0.0).astype(float)

    for name in FOLDABLE:
        proj, unc = f"proj_{name}", f"unc_{name}"
        if proj not in out:
            continue
        act = banked(name)
        out[f"act_{name}"] = act.round(1)
        out[f"ros_{name}"] = out[proj].round(1)
        out[proj] = (out[proj] + act).round(1)
        if unc in out:
            out[unc] = (out[unc] + act).round(1)
        for tail in ("p10", "p90"):
            b = f"{name}_{tail}"
            if b in out:
                out[f"ros_{b}"] = out[b].round(1)
                out[b] = (out[b] + act).round(1)

    # Per-game numbers are ratios of the two halves, so they are recomputed rather than
    # added: what a reader wants next to a season total is the season's average ice time.
    gp = out["proj_gp"].to_numpy(dtype=float)
    safe = np.where(gp > 0, gp, 1.0)
    for stem in ("toi", "pp_toi", "sh_toi"):
        col = f"proj_{stem}_per_gp"
        if col in out and f"proj_{stem}" in out:
            out[col] = np.where(gp > 0, out[f"proj_{stem}"].to_numpy(dtype=float) / safe, 0.0)
    if "points_rate_check" in out:
        out["points_rate_check"] = (out.get("act_points", empty)
                                    + out["rate_points"] * out["ros_toi"] / 60.0).round(1)
    out["season_frac_played"] = round(float(state.frac_played), 3)


COVERAGE_SCALED = ["goals", "ixg", "shots", "blocks", "hits", "pim", "faceoffs_won",
                   "primaryAssists", "secondaryAssists", "assists", "points"]


def _record_coverage(out: pd.DataFrame, tb: pd.DataFrame) -> None:
    """What share of each team's budget the listed roster accounts for, and what is left.

    The remainder is not an error to be hidden: it is the ice time and production that
    will go to players nobody has named yet -- call-ups, waiver claims, the depth forward
    signed in October. Naming it is what stops it being quietly added to the stars.
    """
    on = out[out["on_roster"] & out["team"].isin(tb.index)]
    for alloc, bcol, name in (("proj_toi", "toi_min", "toi"),
                              ("proj_gp", "skater_games", "gp"),
                              ("proj_pp_toi", "pp_toi_min_skaters", "pp"),
                              ("proj_sh_toi", "sh_toi_min_skaters", "sh")):
        if bcol not in tb:
            tb[f"{name}_coverage"] = 1.0
            continue
        got = on.groupby("team")[alloc].sum().reindex(tb.index).fillna(0.0)
        tb[f"{name}_allocated"] = got
        tb[f"{name}_coverage"] = (got / tb[bcol]).clip(upper=1.0).fillna(1.0)
        tb[f"depth_{name}"] = (tb[bcol] - got).clip(lower=0.0)


def _settle(out: pd.DataFrame, tb: pd.DataFrame, tilt: float, lvl: dict, max_gp) -> None:
    """Settle every claim against its team budget, in the order the constraints bind.

    Volume settles DOWNWARD ONLY. A team's ice time is played by ~28 skaters over a
    season and a pre-season roster lists ~22, so the listed players honestly account for
    only about 91% of it; filling the rest would hand fourth-line minutes to the stars.
    The shortfall is recorded per team instead (`depth_*` on the budget frame) and shown
    in the app as ice time not yet accounted for.

    Production then settles against a budget scaled BY THAT COVERAGE, which is the part
    that does the work: conditional on the minutes actually allocated, a team's goals,
    assists, blocks and hits are determined. Measured on this run the listed rosters'
    per-minute scoring rates were already right (goals 1.005 of their share, shots
    0.990), but blocks came in 7% hot, penalty minutes 10% hot and assists 4% high
    relative to goals -- errors that are invisible player-by-player and that this step
    removes exactly.
    """
    def budget(col, frame=None):
        f = tb if frame is None else frame
        return f[col].to_dict() if col in f else {}

    # 1. Games played. 18 skaters dress every night, so a team has exactly 18 x 84
    #    skater-games to hand out however deep or thin its roster is.
    # `max_gp` is the WINDOW: 84 preseason, and each player's team's remaining games once
    # the season is under way, which is why it is allowed to be per player rather than one
    # number for the league.
    out["proj_gp"] = al.settle_frame(out, "claim_gp", budget("skater_games"), tilt=tilt,
                                    lock_col="lock_gp", cap=max_gp, only_reduce=True)
    out["proj_gp"] = np.clip(out["proj_gp"].to_numpy(dtype=float), 0.0, max_gp)

    # 2. Ice time, given the games. The hardest constraint in the sport: 297.4
    #    skater-minutes per team-game, measured to within two tenths of a percent every
    #    season since 2021, because it is the clock and not a tendency.
    out["claim_toi"] = out["proj_gp"] * out["claim_toi_per_gp"]
    cap_toi = out["proj_gp"] * out["pos_group"].map(C.MAX_TOI_PER_GP).fillna(24.0)
    out["proj_toi"] = al.settle_frame(out, "claim_toi", budget("toi_min"), tilt=tilt,
                                     lock_col="lock_toi", cap=cap_toi.to_numpy(),
                                     only_reduce=True)
    gp = out["proj_gp"].to_numpy(dtype=float)
    out["proj_toi_per_gp"] = np.where(gp > 0, out["proj_toi"] / np.where(gp > 0, gp, 1.0), 0.0)

    # 3. Special-teams ice time. The budget is the team's power-play minutes times the
    #    five skaters who are on the ice for them.
    for prefix, cap_pg, bcol in (("pp", C.MAX_PP_TOI_PER_GP, "pp_toi_min_skaters"),
                                 ("sh", C.MAX_SH_TOI_PER_GP, "sh_toi_min_skaters")):
        claim = f"claim_{prefix}_toi"
        out[claim] = out["proj_gp"] * out[f"claim_{prefix}_toi_per_gp"]
        cap = np.minimum(out["proj_gp"].to_numpy() * cap_pg, out["proj_toi"].to_numpy())
        out[f"proj_{prefix}_toi"] = al.settle_frame(
            out, claim, budget(bcol), tilt=tilt, lock_col=f"lock_{prefix}_toi", cap=cap,
            only_reduce=True)
        out[f"proj_{prefix}_toi_per_gp"] = np.where(
            gp > 0, out[f"proj_{prefix}_toi"] / np.where(gp > 0, gp, 1.0), 0.0)

    # 4. How much of each team's real budget the listed roster actually accounts for.
    _record_coverage(out, tb)
    eff = tb.copy()
    for col in COVERAGE_SCALED:
        if col in eff:
            eff[col] = tb[col] * tb["toi_coverage"]
    for stat, prefix in SIT_STATS.items():
        if stat in eff:
            eff[stat] = tb[stat] * tb[f"{prefix}_coverage"]

    # 5. Production, given the ice time. The claims are re-derived off the SETTLED ice
    #    time first -- a player who lost minutes should not still claim the production
    #    that came with them.
    _derive_totals(out, "proj_toi", "proj_pp_toi", "proj_sh_toi")
    for stat in COUNT_STATS:
        fixed = out[f"fixed_{stat}"]
        out[f"claim_{stat}"] = np.where(fixed.notna(), fixed, out[f"claim_{stat}"])
        out[f"proj_{stat}"] = al.settle_frame(out, f"claim_{stat}", budget(stat, eff),
                                             tilt=tilt, lock_col=f"lock_{stat}")
    for col in COVERAGE_SCALED + list(SIT_STATS):
        if col in eff:
            tb[f"eff_{col}"] = eff[col]

    # Assists and points are sums, not separate allocations, so they are consistent with
    # the team budget by construction: team points = team goals x (1 + 0.935 + 0.751).
    out["proj_assists"] = out["proj_primaryAssists"] + out["proj_secondaryAssists"]
    out["proj_points"] = out["proj_goals"] + out["proj_assists"]
    out["claim_assists"] = out["claim_primaryAssists"] + out["claim_secondaryAssists"]
    # The directly-projected points rate, kept as a CHECK rather than silently discarded
    # as it used to be: a large gap between it and goals+assists means the parts of a
    # player's scoring disagree about him, which is worth seeing.
    out["points_rate_check"] = (out["rate_points"] * out["proj_toi"] / 60.0).round(1)


# --------------------------------------------------------------------------- #
# prediction intervals                                                        #
# --------------------------------------------------------------------------- #
# Empirically-calibrated counting-stat spreads (calibrate_intervals.py, backtest
# 2022-25, n=3136). For every stat the residual std scales as ~sqrt(projection) (count
# behaviour), with a coefficient that depends on how PREDICTABLE the player's
# availability is: normalised residual std is markedly larger for injury-prone or
# thin-history skaters (rel<0.33) than for durable ones (rel>0.66). So per stat
#   sigma = coef*sqrt(max(proj, floor)),  coef = C_LO - (C_LO-C_HI)*gp_reliability.
PI_COEFS = {
    "points":   (2.99, 1.75),
    "goals":    (2.05, 1.37),
    "assists":  (2.47, 1.56),
    "shots":    (4.83, 2.49),
    "pp_points": (2.99, 1.75),   # points shape; small totals governed by the floor
    "sh_points": (2.99, 1.75),
    # Not separately calibrated yet. These are volume/usage stats, so they borrow the
    # coefficient of the measured stat that behaves most like them (shots) rather than
    # the narrower scoring bands -- an honest over-estimate of the width beats a
    # precise-looking guess. backtest.py will replace these once it scores them.
    "blocks": (4.83, 2.49), "hits": (4.83, 2.49), "pim": (4.83, 2.49),
    "faceoffs_won": (4.83, 2.49), "ixg": (2.05, 1.37),
}
PI_FLOOR = {"points": 4.0, "goals": 2.0, "assists": 3.0, "shots": 15.0,
            "pp_points": 2.0, "sh_points": 1.0, "blocks": 10.0, "hits": 10.0,
            "pim": 6.0, "faceoffs_won": 15.0, "ixg": 2.0}
_PI_Z = 1.2816     # 80% central interval


def _add_prediction_intervals(out: pd.DataFrame, gp_cap=None) -> None:
    """p10/p90 for every stat, plus games played, right-skewed and non-negative.

    Two changes from the symmetric normal band this used to draw. First, season counting
    totals are right-skewed -- the distance from a 30-goal projection up to 45 is not the
    distance down to 15, and a symmetric band on a small total ran below zero and had to
    be clipped, which quietly broke the coverage it was calibrated for. The band is now
    drawn from a lognormal with the SAME mean and standard deviation, which is skewed the
    right way and cannot go negative. Second, games played gets its own interval: it is
    the largest single source of season-total error, and publishing a points range while
    hiding the availability range behind it was the wrong way round.
    """
    rel = out["gp_reliability"].fillna(0.3) if "gp_reliability" in out else 0.3
    for stat, (c_lo, c_hi) in PI_COEFS.items():
        col = f"proj_{stat}"
        if col not in out:
            continue
        mean = out[col].to_numpy(dtype=float)
        coef = (c_lo - (c_lo - c_hi) * rel).to_numpy(dtype=float)
        sigma = coef * np.sqrt(np.maximum(mean, PI_FLOOR[stat]))
        safe = np.maximum(mean, 1e-9)
        s2 = np.log1p((sigma / safe) ** 2)          # lognormal shape^2 matching the cv
        s = np.sqrt(s2)
        median = safe / np.exp(s2 / 2.0)            # preserve the mean
        lo = np.where(mean > 0, median * np.exp(-_PI_Z * s), 0.0)
        hi = np.where(mean > 0, median * np.exp(+_PI_Z * s), 0.0)
        out[f"{stat}_p10"] = np.round(np.maximum(lo, 0.0), 1)
        out[f"{stat}_p90"] = np.round(hi, 1)

    # Games played: measured actual-GP std is ~13 for durable skaters and ~25 for
    # injury-prone ones, so the width is interpolated on the same reliability score. In
    # season the window is shorter, and the spread scales with its length: a player cannot
    # miss twenty of the twelve games he has left. sqrt, because availability over a window
    # behaves like a count of games missed and not like a fixed fraction of them.
    cap = np.full(len(out), float(C.MAX_GP)) if gp_cap is None \
        else np.broadcast_to(np.asarray(gp_cap, dtype=float), (len(out),)).astype(float)
    gp = out["proj_gp"].to_numpy(dtype=float)
    gp_sigma = (25.0 - 12.0 * rel).to_numpy(dtype=float) * np.sqrt(
        np.clip(cap / float(C.MAX_GP), 0.0, 1.0))
    locked = out["lock_gp"].to_numpy(dtype=bool) if "lock_gp" in out else np.zeros(len(out), bool)
    gp_sigma = np.where(locked, 0.0, gp_sigma)
    out["gp_p10"] = np.round(np.clip(gp - _PI_Z * gp_sigma, 0, cap), 1)
    out["gp_p90"] = np.round(np.clip(gp + _PI_Z * gp_sigma, 0, cap), 1)


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
ROUND_1 = ["proj_gp", "proj_toi_per_gp", "proj_pp_toi_per_gp", "proj_sh_toi_per_gp",
           "proj_assists", "proj_points", "proj_toi"]


def project_skaters(scenario: ov.Scenario | None = None, verbose: bool = False,
                    with_budgets: bool = False, in_season: bool | None = None):
    """Season-long skater projections, settled against team budgets.

    `scenario` carries the user's edits (see overrides.py); None is the pure model.
    With `with_budgets`, returns (skaters, team_budgets) -- the budget frame carries each
    team's allowance, what the roster covered and what is left for depth, which is what
    the app needs to show a reader why a number moved.

    Once the season starts this becomes `act_* + ros_*`: what a player has already banked
    plus the ordinary model run against the games his team has left. `in_season=False`
    forces the preseason projection (which is how the two get compared); the default
    follows `live.season_state()`.
    """
    sc = scenario
    cfg = sc.settings() if sc is not None else C.league_defaults()
    tilt = float(cfg["budget_tilt"])
    enforce = bool(cfg["enforce_budgets"])
    max_gp = float(cfg["season_games"])
    regress_k = float(cfg["skater_regress_toi_min"])
    xg_weight = float(cfg["goals_xg_weight"])

    state = live.season_state()
    use_live = state.live if in_season is None else bool(in_season and state.live)

    hist = _prep_skater_seasons(with_live=use_live)
    curves = ac.build_skater_age_curves()
    toi_curve = ac.build_toi_age_curve()
    pos_means = _positional_means(hist)

    target = C.TARGET_SEASON
    recent_years = [target - i for i in range(1, C.SKATER_HISTORY_SEASONS + 1)]
    if use_live:
        # A player whose only NHL season is the one being played belongs in the claim pool,
        # not in the rookie prior: three weeks of his own ice time beats a tier average.
        recent_years.append(target)
    pool = hist[hist["mp_season_year"].isin(recent_years)].copy()

    acts = live.skater_actuals() if use_live else pd.DataFrame()
    act_map = {}
    if not acts.empty:
        act_map = {int(r.playerId): {"gp": float(r.act_gp), "toi": float(r.act_toi),
                                     "pp_toi": float(r.act_pp_toi),
                                     "sh_toi": float(r.act_sh_toi)}
                   for r in acts.itertuples()}

    roster = dl.load_rosters(target)
    rows = _claim_rows(pool, pos_means, curves, toi_curve, regress_k, acts=act_map,
                       team_played=state.team_games_played if use_live else 0.0)
    known = {r["playerId"] for r in rows}
    rows += _rookie_rows(roster, known, pos_means, curves, toi_curve)
    out = pd.DataFrame(rows)

    # Team for the season being projected comes from the published roster. A player who
    # is on no NHL roster is NOT quietly left on last year's team, which is what used to
    # happen: 696 of 1397 skaters were assigned to a prior team, including 15 to the
    # relocated Arizona Coyotes, and between them they claimed 6,295 projected points
    # that no team was ever going to produce. They are marked unsigned instead, kept out
    # of every team budget, and can be signed to a team in the app.
    cur_team = dict(zip(roster["playerId"], roster["team"]))
    out["on_roster"] = out["playerId"].isin(cur_team)
    out["team"] = out["playerId"].map(cur_team).fillna(FREE_AGENT)
    out["camp"] = False
    out["camp_team"] = ""
    _cut_camp_bodies(out)
    out["status"] = np.where(out["on_roster"], "roster",
                             np.where(out["camp"], "camp", "unsigned"))

    # Expected goals into the goal rate. Shot QUALITY repeats far better than finishing
    # does, so a rate built partly on xG is the more honest base for next season -- it is
    # also why `proj_ixg` is worth showing next to `proj_goals`: the gap between them is
    # exactly how much of a projection is being carried by a player's shooting hands.
    # The scale factor puts xG on the goals scale so the blend cannot shift the level.
    recent = hist[hist["mp_season_year"] >= C.LAST_COMPLETED_SEASON - 2]
    g_rate = float(np.average(recent["goals"], weights=recent["toi_min"]))
    x_rate = float(np.average(recent["ixg"], weights=recent["toi_min"]))
    xg_scale = (g_rate / x_rate) if x_rate > 0 else 1.0
    out["rate_goals_own"] = out["rate_goals"]
    out["rate_goals"] = ((1.0 - xg_weight) * out["rate_goals"]
                         + xg_weight * out["rate_ixg"] * xg_scale)

    out = out.reset_index(drop=True)
    out = _apply_input_edits(out, sc, ov.gp_override_csv())
    # Edits run AFTER the camp cut on purpose: naming a team for a player the cut dropped
    # is how a reader says "he made the team", and it has to win over the model's guess.
    out["status"] = np.where(out["on_roster"], "roster",
                             np.where(out.get("camp", False), "camp", "unsigned"))

    # How many games each player still has in front of him. Preseason that is the whole
    # season for everybody; in season it is his team's remaining games, which differ -- teams
    # are routinely five games apart in November. Off-roster players get the league mean,
    # because signing one in the app should not also change how long his season is.
    if use_live:
        rem_by_team = state.remaining
        league_rem = float(rem_by_team.mean()) if len(rem_by_team) else max_gp
        window = out["team"].map(rem_by_team).fillna(league_rem).to_numpy(dtype=float)
        _to_rest_of_season_gp(out, window, act_map)
    else:
        window = np.full(len(out), max_gp, dtype=float)

    # Unconstrained totals, used for two different things: to shape the team budgets (a
    # team whose roster changed should carry a budget its NEW roster justifies) and, in
    # the app, to show what settlement did to each player.
    out["claim_toi"] = out["claim_gp"] * out["claim_toi_per_gp"]
    for prefix in ("pp", "sh"):
        out[f"claim_{prefix}_toi"] = out["claim_gp"] * out[f"claim_{prefix}_toi_per_gp"]
    _derive_totals(out, "claim_toi", "claim_pp_toi", "claim_sh_toi")
    # Keep the unconstrained numbers under their own names. The app shows claim next to
    # allocation, which is the only way a reader can see that settlement happened at all
    # and judge whether it took the minutes off the right players.
    for stat in COUNT_STATS + ["gp", "toi"]:
        out[f"unc_{stat}"] = out[f"claim_{stat}"]
    out["unc_assists"] = out["unc_primaryAssists"] + out["unc_secondaryAssists"]
    out["unc_points"] = out["unc_goals"] + out["unc_assists"]

    lvl = bg.league_level()
    teams = sorted(t for t in out.loc[out["on_roster"], "team"].unique() if t != FREE_AGENT)
    _apply_total_locks(out, sc, lvl)
    if use_live:
        # A stated season total is a statement about the SEASON, not about the games left, so
        # what has to be honoured over the remaining games is the stated number minus what he
        # has already banked. Without this, locking "50 goals" in February would ask for 50
        # more of them.
        _locks_to_rest_of_season(out, acts)
    tb = None
    if not enforce:
        _no_budget_projection(out, window)
    else:
        import context as ctx
        claims = out[out["on_roster"]].groupby("team")[
            [f"claim_{s}" for s in COUNT_STATS]].sum()
        claims.columns = COUNT_STATS
        tb = bg.team_budgets(target, teams=teams, claims=claims,
                             shrink=float(cfg["team_rating_shrink"]),
                             games=(state.remaining if use_live else int(max_gp)),
                             live_rates=live.live_team_rates() if use_live else None)
        sos = ctx.schedule_context(target).groupby("team")["sos_factor"].first().to_dict()
        tb = _apply_sos(tb, sos)
        tb = _apply_team_edits(tb, sc, lvl)
        _settle(out, tb, tilt, lvl, window)
        if verbose:
            report = al.budget_report(out[out["on_roster"]], "claim_toi", "proj_toi",
                                      tb["toi_min"].to_dict())
            print(report.round(3).to_string())

    _add_prediction_intervals(out, gp_cap=window)
    if use_live:
        # Everything above this line is the REST of the season. Add what is banked and every
        # column in the frame means the same thing it meant preseason -- a full-season number
        # -- so no page downstream has to know the difference. The `ros_*` and `act_*` columns
        # are kept because the performance page has to score a projection against the window
        # it was actually made for.
        _fold_in_actuals(out, acts, state)
        # The fold inserts three columns per stat one at a time, which leaves the frame
        # badly fragmented; one copy here is cheaper than every later access paying for it.
        out = out.copy()

    for col in out.columns:
        if col.startswith(("proj_", "claim_", "unc_", "rate_")) and \
                pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(3 if col.startswith("rate_") else 1)
    out = out.sort_values("proj_points", ascending=False).reset_index(drop=True)
    # attrs last, and SCALARS ONLY: pandas compares `attrs` for equality when it
    # finalises a concat, so a DataFrame parked in here raises "truth value is
    # ambiguous" the first time anyone calls nlargest on the result. The budget table is
    # returned instead.
    out.attrs["league_level"] = lvl
    out.attrs["scenario"] = "baseline" if sc is None else sc.name
    out.attrs["coverage"] = float(tb["toi_coverage"].mean()) if tb is not None else 1.0
    out.attrs["in_season"] = bool(use_live)
    out.attrs["team_games_played"] = float(state.team_games_played) if use_live else 0.0
    out.attrs["frac_played"] = float(state.frac_played) if use_live else 0.0
    out.attrs["as_of"] = str(state.as_of.date())
    out.attrs["window"] = state.label()
    return (out, tb) if with_budgets else out


def _no_budget_projection(out: pd.DataFrame, max_gp) -> None:
    """The old unconstrained behaviour, kept so the budgets can be switched off and seen.

    `enforce_budgets = False` in a scenario reproduces what the model did before team
    accounting existed. It is wrong -- that is the point of being able to look at it.
    """
    out["proj_gp"] = out["claim_gp"].clip(lower=0.0, upper=pd.Series(
        np.broadcast_to(np.asarray(max_gp, dtype=float), (len(out),)), index=out.index))
    out["proj_toi"] = out["claim_toi"]
    out["proj_toi_per_gp"] = out["claim_toi_per_gp"]
    for prefix in ("pp", "sh"):
        out[f"proj_{prefix}_toi"] = out[f"claim_{prefix}_toi"]
        out[f"proj_{prefix}_toi_per_gp"] = out[f"claim_{prefix}_toi_per_gp"]
    for stat in COUNT_STATS:
        out[f"proj_{stat}"] = out[f"claim_{stat}"]
    out["proj_assists"] = out["proj_primaryAssists"] + out["proj_secondaryAssists"]
    out["proj_points"] = out["proj_goals"] + out["proj_assists"]
    out["points_rate_check"] = (out["rate_points"] * out["proj_toi"] / 60.0).round(1)


def _apply_team_edits(tb: pd.DataFrame, sc: ov.Scenario | None, lvl: dict) -> pd.DataFrame:
    """A stated team total replaces the budget, and the assist budgets follow its goals.

    Editing a team budget is the honest answer to a roster change too recent for either
    the team's history or its published roster to know about. Note that this deliberately
    does NOT renormalise the league: if a user says a team scores 300 goals, that is what
    they said, and silently taking it back off the other 31 teams would be worse.
    """
    if sc is None or not sc.teams:
        return tb
    for team, edits in sc.teams.items():
        if team not in tb.index:
            continue
        for col, val in edits.items():
            if col in tb.columns:
                tb.at[team, col] = float(val)
        if "goals" in edits:
            tb.at[team, "primaryAssists"] = tb.at[team, "goals"] * lvl["primary_per_goal"]
            tb.at[team, "secondaryAssists"] = tb.at[team, "goals"] * lvl["secondary_per_goal"]
            tb.at[team, "assists"] = tb.at[team, "primaryAssists"] + tb.at[team, "secondaryAssists"]
            tb.at[team, "points"] = tb.at[team, "goals"] + tb.at[team, "assists"]
    return tb


if __name__ == "__main__":
    out, tb = project_skaters(with_budgets=True)
    cols = ["name", "team", "position", "target_age", "proj_gp", "proj_toi_per_gp",
            "proj_points", "proj_goals", "proj_assists", "proj_shots", "proj_pp_points",
            "proj_blocks", "proj_hits", "points_p10", "points_p90"]
    on = out[out["on_roster"]]
    print(f"Projected {len(out)} skaters for {C.TARGET_SEASON}-{C.TARGET_SEASON+1} "
          f"({len(on)} on an NHL roster, {len(out) - len(on)} unsigned)\n")
    print(on[cols].head(25).to_string(index=False))

    games, lvl = C.SEASON_GAMES, out.attrs["league_level"]
    tg = len(on["team"].unique()) * games
    cov = float(tb["toi_coverage"].mean())
    print(f"\nListed rosters account for {cov:.1%} of team ice time "
          f"({tb['depth_toi'].sum() / tg:.1f} min per team-game left for call-ups; "
          f"{len(on) / len(tb):.1f} skaters listed per team, ~28 play in a real season)")
    print("Per team-game, on-roster skaters (allocated | full budget | budget x coverage):")
    for stat in ("toi", "goals", "assists", "shots", "blocks", "hits", "pim",
                 "faceoffs_won", "pp_points", "sh_points"):
        key = {"toi": "toi_min"}.get(stat, stat)
        got = on[f"proj_{stat}"].sum() / tg
        full = lvl.get(key, float("nan"))
        eff = tb[f"eff_{stat}"].sum() / tg if f"eff_{stat}" in tb else full * cov
        flag = "" if abs(got - eff) < max(0.02 * eff, 0.01) else "  <-- MISS"
        print(f"  {stat:14s} {got:8.2f} | {full:8.2f} | {eff:8.2f}{flag}")
    print(f"  {'skater games':14s} {on['proj_gp'].sum() / tg:8.2f} | {lvl['dressed']:8.2f} |"
          f" {lvl['dressed'] * float(tb['gp_coverage'].mean()):8.2f}")
    print(f"  {'assists/goal':14s} "
          f"{on['proj_assists'].sum() / on['proj_goals'].sum():8.3f} | "
          f"{lvl['primary_per_goal'] + lvl['secondary_per_goal']:8.3f} |  (ratio, unscaled)")
    print(f"\nBusiest projected ice times (cap {C.MAX_TOI_PER_GP}):")
    print(on.nlargest(5, "proj_toi_per_gp")[
        ["name", "team", "pos_group", "proj_gp", "proj_toi_per_gp", "proj_points"]
    ].to_string(index=False))
    rk = out[out["rookie"]]
    print(f"\n{len(rk)} rostered players with no NHL history, now projected from the "
          f"tier prior instead of dropped:")
    print(rk.nlargest(5, "proj_points")[
        ["name", "team", "position", "target_age", "proj_gp", "proj_toi_per_gp",
         "proj_points"]].to_string(index=False))

    path = C.OUTPUT / f"skater_projections_{C.TARGET_SEASON}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {path}")
