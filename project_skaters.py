"""Season-long skater projections for the upcoming season (2026-27).

Method (Marcel + age curve, on a per-60 RATE basis so injury-shortened seasons
don't distort a player's true talent):

  1. RATE BASE: for each stat, blend the player's last up-to-3 seasons of per-60
     rates, weighting recent seasons heaviest (RECENCY_WEIGHTS), each season further
     weighted by that season's TOI (a full season counts more than a cameo).
  2. REGRESSION: pull the blended rate toward the positional league-average rate,
     weighted by total sampled TOI (thin history => regress hard; established stars
     barely move). SKATER_REGRESS_TOI_MIN controls the strength.
  3. AGE: scale each rate by the age-curve multiplier from the player's weighted
     mean age to their target-season age.
  4. VOLUME: project target-season TOI (recency-weighted, regressed) and games
     played (recency + durability), then rates * TOI -> counting stats, distributed
     across projected games.

Outputs projected G / A / points / shots / PP points and projected GP + TOI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_layer as dl
import age_curves as ac

RATE_STATS = ["points", "goals", "assists", "primaryAssists", "secondaryAssists",
              "shots", "pp_points"]


def _prep_skater_seasons() -> pd.DataFrame:
    """One row per player-season with per-60 rates, TOI, age, position."""
    sk = dl.load_moneypuck_skaters()
    pp = dl.load_moneypuck_skaters_pp()
    bios = dl.load_nhl_skater_bios()
    births = dl.player_birthdates(bios)

    # Power-play points per player-season (goals + total assists at 5on4).
    pp_pts = (pp["I_F_goals"] + pp["I_F_primaryAssists"] + pp["I_F_secondaryAssists"])
    pp_tbl = pd.DataFrame({
        "playerId": pp["playerId"],
        "mp_season_year": pp["mp_season_year"],
        "pp_points_raw": pp_pts.values,
    })

    df = sk.merge(births[["playerId", "birthDate", "positionCode", "fullName"]],
                  on="playerId", how="left")
    # Prefer the NHL API's clean, accented name over MoneyPuck's ASCII-stripped one.
    df["name"] = df["fullName"].fillna(df["name"])
    df = df.merge(pp_tbl, on=["playerId", "mp_season_year"], how="left")
    df["pp_points_raw"] = df["pp_points_raw"].fillna(0.0)
    df = df[df["icetime"] >= C.MIN_ICETIME_SEC].copy()

    df["toi_min"] = df["icetime"] / 60.0
    per60 = 60.0 / df["toi_min"]
    df["points_raw"] = df["I_F_points"]
    df["goals_raw"] = df["I_F_goals"]
    df["assists_raw"] = df["I_F_primaryAssists"] + df["I_F_secondaryAssists"]
    df["primaryAssists_raw"] = df["I_F_primaryAssists"]
    df["secondaryAssists_raw"] = df["I_F_secondaryAssists"]
    df["shots_raw"] = df["I_F_shotsOnGoal"]

    for stat in RATE_STATS:
        df[stat] = df[f"{stat}_raw"] * per60

    # Position group: forwards vs defense (for positional regression means).
    pos = df["positionCode"].fillna(df["position"])
    df["pos_group"] = np.where(pos.isin(["D"]), "D", "F")

    # Age as of Feb 1 of the season.
    bd = pd.to_datetime(df["birthDate"], errors="coerce")
    ref = pd.to_datetime((df["mp_season_year"] + 1).astype(str) + "-02-01", errors="coerce")
    df["age"] = (ref - bd).dt.days / 365.25
    return df


def _positional_means(hist: pd.DataFrame) -> dict:
    """Regression targets by position group AND usage tier, TOI-weighted, recent seasons.

    A single global positional mean is the WRONG prior for a star: regressing a
    first-line center toward an average that includes 4th-liners systematically
    under-projects elite players (backtest confirmed ~-0.46 pts/60 rate compression).
    Instead we regress toward the mean of players in the SAME usage tier (TOI/game
    quartile within position) — a top-6 forward's prior is "other top-6 forwards."
    Backtest: this + lighter regression cuts elite rate bias ~38% AND lowers overall MAE.
    """
    recent = hist[hist["mp_season_year"] >= C.LAST_COMPLETED_SEASON - 2].copy()
    recent["toipg"] = recent["toi_min"] / recent["games_played"]
    means = {}
    for grp, g in recent.groupby("pos_group"):
        w = g["toi_min"]
        gm = {stat: np.average(g[stat], weights=w) for stat in RATE_STATS}
        gm["toi_per_gp"] = np.average(g["toipg"], weights=g["games_played"])
        # Usage-tier breakpoints (TOI/game quartiles) + per-tier rate means.
        qs = g["toipg"].quantile([0.25, 0.50, 0.75]).to_numpy()
        gm["_toipg_q"] = qs
        gm["_tier"] = {}
        tier = np.searchsorted(qs, g["toipg"].to_numpy())
        for t in range(4):
            sub = g[tier == t]
            if len(sub) >= 20:
                gm["_tier"][t] = {stat: np.average(sub[stat], weights=sub["toi_min"])
                                  for stat in RATE_STATS}
        means[grp] = gm
    return means


def _tier_target(pm: dict, toipg: float, stat: str) -> float:
    """Regression target for `stat`: the usage-tier mean if available, else the
    position-group mean. Tier is the player's TOI/game quartile."""
    t = int(np.searchsorted(pm["_toipg_q"], toipg))
    tier = pm["_tier"].get(t)
    if tier is not None:
        return tier[stat]
    return pm[stat]


def project_skaters() -> pd.DataFrame:
    hist = _prep_skater_seasons()
    curves = ac.build_skater_age_curves()
    toi_curve = ac.build_toi_age_curve()
    pos_means = _positional_means(hist)

    target = C.TARGET_SEASON
    # A player is projectable if they played in any of the last N seasons (N = window).
    recent_years = [target - i for i in range(1, C.SKATER_HISTORY_SEASONS + 1)]
    pool = hist[hist["mp_season_year"].isin(recent_years)].copy()

    # Current-season team from the published roster (captures trades / FA moves that
    # the prior-season stats, tied to the OLD team, cannot). + season SOS factor.
    import context as ctx
    roster = dl.load_rosters(target)
    cur_team = dict(zip(roster["playerId"], roster["team"]))
    sos = ctx.schedule_context(target).groupby("team")["sos_factor"].first().to_dict()
    gp_overrides = _load_gp_overrides()

    rows = []
    for pid, g in pool.groupby("playerId"):
        g = g.sort_values("mp_season_year")
        latest = g.iloc[-1]
        pos_group = latest["pos_group"]
        pm = pos_means[pos_group]

        # Recency weight per season (t-1 heaviest), times that season's TOI.
        wmap = dict(zip(recent_years, C.RECENCY_WEIGHTS))
        g = g.assign(rec_w=g["mp_season_year"].map(wmap).fillna(0.0))
        g = g.assign(blend_w=g["rec_w"] * g["toi_min"])
        total_w = g["blend_w"].sum()
        total_toi = g["toi_min"].sum()
        if total_w <= 0:
            continue

        # Weighted mean age of the sample, advanced to the target season.
        mean_age = np.average(g["age"], weights=g["blend_w"])
        target_age = mean_age + (target - np.average(g["mp_season_year"], weights=g["blend_w"]))

        # Team for the UPCOMING season: prefer the published roster; fall back to the
        # last team the player appeared for. `on_roster` flags whether the player is
        # actually rostered for the target season (vs. projected-but-unsigned).
        team = cur_team.get(pid, latest["team"])
        on_roster = pid in cur_team
        proj = {"playerId": pid, "name": latest["name"], "team": team,
                "prior_team": latest["team"], "on_roster": on_roster,
                "position": latest["position"], "pos_group": pos_group,
                "target_age": round(target_age, 1)}

        # ---- volume: projected TOI/GP and games played ----
        # Compute the sampled TOI/game FIRST: it selects the usage tier the player is
        # regressed toward (a top-line forward should regress toward top-line rates,
        # not the global positional mean that includes 4th-liners / bottom-pairing D).
        sample_toipg = np.average(g["toi_min"] / g["games_played"], weights=g["blend_w"])
        tp = C.SKATER_TOI_PRIOR_MIN
        toi_per_gp = (sample_toipg * total_toi + pm["toi_per_gp"] * tp) / (total_toi + tp) \
            if (total_toi + tp) > 0 else sample_toipg
        # Age-trend the ice time: young players earn bigger roles, vets lose them.
        # Projecting TOI from a player's own (smaller-role) history otherwise under-
        # projects risers (~-34% at 18-21) and over-projects fading vets (~+8% at 33+).
        toi_per_gp *= ac.age_multiplier({"toi": toi_curve}, "toi", mean_age, target_age)

        # ---- rate projection per stat ----
        proj_rate = {}
        for stat in RATE_STATS:
            blended = np.average(g[stat], weights=g["blend_w"])
            # Regress toward the usage-tier mean (by sampled TOI/game) rather than the
            # global positional mean, then by sampled total TOI.
            k = C.SKATER_REGRESS_TOI_MIN
            reg_target = _tier_target(pm, sample_toipg, stat)
            regressed = (blended * total_toi + reg_target * k) / (total_toi + k)
            # Age adjustment (use the matching curve; pp_points/secondary reuse points/assists).
            curve_stat = {"pp_points": "points", "secondaryAssists": "assists"}.get(stat, stat)
            mult = ac.age_multiplier(curves, curve_stat, mean_age, target_age)
            proj_rate[stat] = regressed * mult

        proj_gp, gp_rel = _project_gp(g)
        # User override wins if present (keyed by id or exact name). A known status
        # makes availability certain, so treat an override as fully reliable.
        override = gp_overrides.get(pid, gp_overrides.get(str(latest["name"]).strip().lower()))
        if override is not None:
            proj_gp, gp_rel = override, 1.0
        proj_toi = toi_per_gp * proj_gp

        # Season strength-of-schedule factor for the player's target-season team.
        # This is the genuine, non-zero-sum shift from an unbalanced (divisional)
        # schedule; it is tiny for skaters (~<1%) by design and by the data.
        sos_factor = sos.get(team, 1.0)

        proj["proj_gp"] = round(proj_gp, 1)
        proj["gp_reliability"] = round(gp_rel, 2)
        proj["gp_override"] = override is not None
        proj["proj_toi_per_gp"] = round(toi_per_gp, 2)
        proj["sos_factor"] = round(sos_factor, 4)
        for stat in RATE_STATS:
            total = proj_rate[stat] * proj_toi / 60.0 * sos_factor
            proj[f"proj_{stat}"] = round(total, 1)
        rows.append(proj)

    out = pd.DataFrame(rows)
    # Consistency: assists = primary + secondary; points ~ goals + assists.
    out["proj_assists"] = (out["proj_primaryAssists"] + out["proj_secondaryAssists"]).round(1)
    out["proj_points"] = (out["proj_goals"] + out["proj_assists"]).round(1)
    _add_prediction_intervals(out)
    out = out.sort_values("proj_points", ascending=False).reset_index(drop=True)
    return out


# Empirically-calibrated counting-stat spreads (calibrate_intervals.py, backtest
# 2022-25, n=3136, production-faithful). For EVERY stat the residual std scales as
# ~sqrt(projection) (count-data behaviour), and the coefficient depends on how PREDICTABLE
# the player's availability is: the normalized resid std is markedly larger for injury-
# prone / thin histories (rel<0.33) than for durable skaters (rel>0.66). So per stat
#   width = coef*sqrt(max(proj, floor)),  coef = C_LO - (C_LO-C_HI)*gp_reliability.
# ±1.28 std => an 80% central band; measured coverage ~0.85 (slightly conservative, the
# safe side for a floor/ceiling). Each stat is calibrated on its OWN residuals — a shots
# band is ~2x a goals band — rather than borrowing the points width. pp_points has no
# separate backtest actuals wired, so it reuses the points shape (same 5on4 count
# behaviour); its small totals are governed by the floor. This EXPOSES the irreducible
# games-played/injury uncertainty AND shrinks it where a track record earns it.
PI_COEFS = {              # (C_LO @ low reliability, C_HI @ high reliability)
    "points":   (2.99, 1.75),
    "goals":    (2.05, 1.37),
    "assists":  (2.47, 1.56),
    "shots":    (4.83, 2.49),
    "pp_points": (2.99, 1.75),  # reuse points shape (no separate backtest actuals)
}
PI_FLOOR = {"points": 4.0, "goals": 2.0, "assists": 3.0, "shots": 15.0, "pp_points": 2.0}
_PI_Z = 1.2816     # 80% central interval


def _add_prediction_intervals(out: pd.DataFrame) -> None:
    """Add p10/p90 floor/ceiling columns for every counting stat, in place,
    GP-reliability-aware (predictable availability => tighter band)."""
    rel = out["gp_reliability"].fillna(0.3) if "gp_reliability" in out else 0.3
    for stat, (c_lo, c_hi) in PI_COEFS.items():
        col = f"proj_{stat}"
        if col not in out:
            continue
        coef = c_lo - (c_lo - c_hi) * rel
        std = coef * np.sqrt(out[col].clip(lower=PI_FLOOR[stat]))
        out[f"{stat}_p10"] = (out[col] - _PI_Z * std).clip(lower=0).round(1)
        out[f"{stat}_p90"] = (out[col] + _PI_Z * std).round(1)


def _project_gp(g: pd.DataFrame) -> tuple[float, float]:
    """Project games played from recent GP, recency-weighted, capped at MAX_GP.

    Returns (proj_gp, gp_reliability) where gp_reliability in [0,1] reflects how
    predictable this player's availability is from their track record: high for a
    consistently-healthy skater (tight recent GP, all high), low for an injury-prone or
    thin-history one. Backtest: durable players (min recent GP>=72) have GP MAE ~10 and
    actual-GP std ~13; injury-prone (<60) have MAE ~17 and std ~25. That difference is
    real and predictable, so it drives a per-player interval width (not the point
    estimate — even iron-men regress, so the prior stays).
    """
    years = g["mp_season_year"].tolist()
    gp = dict(zip(years, g["games_played"]))
    num = den = 0.0
    for i in range(len(C.GP_RECENCY_WEIGHTS)):
        yr = C.TARGET_SEASON - 1 - i
        if yr in gp:
            w = C.GP_RECENCY_WEIGHTS[i]
            num += w * gp[yr]
            den += w
    if den == 0:
        return 60.0, 0.3
    base = num / den
    # Mild regression toward a full-season durability prior (70 GP). Backtest-confirmed:
    # even players with a spotless recent record regress ~8 GP, so this hedge is correct.
    proj = 0.80 * base + 0.20 * 70.0
    # Reliability: high when the RECENT floor is high and the recent seasons agree.
    # Availability is a recent signal — measure it over the last GP_RELIABILITY_SEASONS
    # only. (The point estimate above still uses the full weighted window; but a healthy
    # iron-man shouldn't be judged "unpredictable" for an injury four seasons ago, which
    # is exactly what a floor over the whole 5-yr window would do.)
    recent_vals = [gp[C.TARGET_SEASON - 1 - i] for i in range(C.GP_RELIABILITY_SEASONS)
                   if (C.TARGET_SEASON - 1 - i) in gp]
    floor = min(recent_vals) if recent_vals else base
    spread = np.std(recent_vals) if len(recent_vals) >= 2 else 15.0
    rel = np.clip((floor - 45) / 35.0, 0, 1) * np.clip(1 - spread / 20.0, 0.2, 1.0)
    return float(np.clip(proj, 1, C.MAX_GP)), float(rel)


def _load_gp_overrides() -> dict:
    """User-supplied games-played overrides keyed by playerId (int) and lowercased name."""
    path = C.GP_OVERRIDE_FILE
    if not path.exists():
        return {}
    try:
        ov = pd.read_csv(path, comment="#", skip_blank_lines=True)
    except Exception:
        return {}
    if ov.empty:
        return {}
    out = {}
    for _, r in ov.iterrows():
        val = r.get("games_played")
        if pd.isna(val):
            continue
        val = float(np.clip(val, 0, C.MAX_GP))
        if "playerId" in ov.columns and pd.notna(r.get("playerId")):
            out[int(r["playerId"])] = val
        if "name" in ov.columns and pd.notna(r.get("name")):
            out[str(r["name"]).strip().lower()] = val
    return out


if __name__ == "__main__":
    out = project_skaters()
    cols = ["name", "team", "position", "target_age", "proj_gp",
            "proj_points", "proj_goals", "proj_assists", "proj_shots", "proj_pp_points"]
    print(f"Projected {len(out)} skaters for {C.TARGET_SEASON}-{C.TARGET_SEASON+1}\n")
    print(out[cols].head(30).to_string(index=False))
    path = C.OUTPUT / f"skater_projections_{C.TARGET_SEASON}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {path}")
