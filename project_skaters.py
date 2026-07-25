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
    """League-average per-60 rate by position group, TOI-weighted, recent seasons."""
    recent = hist[hist["mp_season_year"] >= C.LAST_COMPLETED_SEASON - 2]
    means = {}
    for grp, g in recent.groupby("pos_group"):
        w = g["toi_min"]
        means[grp] = {stat: np.average(g[stat], weights=w) for stat in RATE_STATS}
    # Also a TOI/GP mean per group for volume regression.
    for grp, g in recent.groupby("pos_group"):
        means[grp]["toi_per_gp"] = np.average(g["toi_min"] / g["games_played"],
                                              weights=g["games_played"])
    return means


def project_skaters() -> pd.DataFrame:
    hist = _prep_skater_seasons()
    curves = ac.build_skater_age_curves()
    pos_means = _positional_means(hist)

    target = C.TARGET_SEASON
    # A player is projectable if they played in any of the last 3 seasons.
    recent_years = [target - 1, target - 2, target - 3]
    pool = hist[hist["mp_season_year"].isin(recent_years)].copy()

    rows = []
    for pid, g in pool.groupby("playerId"):
        g = g.sort_values("mp_season_year")
        latest = g.iloc[-1]
        pos_group = latest["pos_group"]
        pm = pos_means[pos_group]

        # Recency weight per season (t-1 heaviest), times that season's TOI.
        wmap = {recent_years[0]: C.RECENCY_WEIGHTS[0],
                recent_years[1]: C.RECENCY_WEIGHTS[1],
                recent_years[2]: C.RECENCY_WEIGHTS[2]}
        g = g.assign(rec_w=g["mp_season_year"].map(wmap).fillna(0.0))
        g = g.assign(blend_w=g["rec_w"] * g["toi_min"])
        total_w = g["blend_w"].sum()
        total_toi = g["toi_min"].sum()
        if total_w <= 0:
            continue

        # Weighted mean age of the sample, advanced to the target season.
        mean_age = np.average(g["age"], weights=g["blend_w"])
        target_age = mean_age + (target - np.average(g["mp_season_year"], weights=g["blend_w"]))

        proj = {"playerId": pid, "name": latest["name"], "team": latest["team"],
                "position": latest["position"], "pos_group": pos_group,
                "target_age": round(target_age, 1)}

        # ---- rate projection per stat ----
        proj_rate = {}
        for stat in RATE_STATS:
            blended = np.average(g[stat], weights=g["blend_w"])
            # Regress toward positional mean by sampled TOI.
            k = C.SKATER_REGRESS_TOI_MIN
            regressed = (blended * total_toi + pm[stat] * k) / (total_toi + k)
            # Age adjustment (use the matching curve; pp_points/secondary reuse points/assists).
            curve_stat = {"pp_points": "points", "secondaryAssists": "assists"}.get(stat, stat)
            mult = ac.age_multiplier(curves, curve_stat, mean_age, target_age)
            proj_rate[stat] = regressed * mult

        # ---- volume: projected TOI/GP and games played ----
        toi_per_gp = np.average(g["toi_min"] / g["games_played"], weights=g["blend_w"])
        toi_per_gp = (toi_per_gp * total_toi + pm["toi_per_gp"] * 600.0) / (total_toi + 600.0)

        proj_gp = _project_gp(g)
        proj_toi = toi_per_gp * proj_gp

        proj["proj_gp"] = round(proj_gp, 1)
        proj["proj_toi_per_gp"] = round(toi_per_gp, 2)
        for stat in RATE_STATS:
            total = proj_rate[stat] * proj_toi / 60.0
            proj[f"proj_{stat}"] = round(total, 1)
        rows.append(proj)

    out = pd.DataFrame(rows)
    # Consistency: assists = primary + secondary; points ~ goals + assists.
    out["proj_assists"] = (out["proj_primaryAssists"] + out["proj_secondaryAssists"]).round(1)
    out["proj_points"] = (out["proj_goals"] + out["proj_assists"]).round(1)
    out = out.sort_values("proj_points", ascending=False).reset_index(drop=True)
    return out


def _project_gp(g: pd.DataFrame) -> float:
    """Project games played from recent GP, recency-weighted, capped at MAX_GP."""
    years = g["mp_season_year"].tolist()
    gp = dict(zip(years, g["games_played"]))
    num = den = 0.0
    for i, yr in enumerate([C.TARGET_SEASON - 1, C.TARGET_SEASON - 2, C.TARGET_SEASON - 3]):
        if yr in gp:
            w = C.GP_RECENCY_WEIGHTS[i]
            num += w * gp[yr]
            den += w
    if den == 0:
        return 60.0
    base = num / den
    # Mild regression toward a full-season durability prior (70 GP).
    proj = 0.80 * base + 0.20 * 70.0
    return float(np.clip(proj, 1, C.MAX_GP))


if __name__ == "__main__":
    out = project_skaters()
    cols = ["name", "team", "position", "target_age", "proj_gp",
            "proj_points", "proj_goals", "proj_assists", "proj_shots", "proj_pp_points"]
    print(f"Projected {len(out)} skaters for {C.TARGET_SEASON}-{C.TARGET_SEASON+1}\n")
    print(out[cols].head(30).to_string(index=False))
    path = C.OUTPUT / f"skater_projections_{C.TARGET_SEASON}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {path}")
