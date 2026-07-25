"""Backtest the skater points projection against naive baselines.

For a held-out target season, we project it using ONLY data available before it,
then score MAE / RMSE against actual results for players who actually played.
Baselines:
  - naive1  : last season's totals = this season's projection
  - naive3  : simple 3-year average of totals
  - rate1   : last season's per-60 rate * actual TOI (isolates rate skill from volume)
  - model   : our Marcel+age projection

The model should beat naive1/naive3 on totals. We report per-60-rate accuracy too,
since usage/injury noise in totals can mask real rate-projection skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_layer as dl
import age_curves as ac


def _seasons_frame() -> pd.DataFrame:
    sk = dl.load_moneypuck_skaters()
    bios = dl.load_nhl_skater_bios()
    births = dl.player_birthdates(bios)[["playerId", "birthDate", "positionCode"]]
    df = sk.merge(births, on="playerId", how="left")
    df = df[df["icetime"] >= C.MIN_ICETIME_SEC].copy()
    df["toi_min"] = df["icetime"] / 60.0
    df["points"] = df["I_F_points"]
    df["pts_per60"] = df["points"] * 60.0 / df["toi_min"]
    pos = df["positionCode"].fillna(df["position"])
    df["pos_group"] = np.where(pos.isin(["D"]), "D", "F")
    bd = pd.to_datetime(df["birthDate"], errors="coerce")
    ref = pd.to_datetime((df["mp_season_year"] + 1).astype(str) + "-02-01", errors="coerce")
    df["age"] = (ref - bd).dt.days / 365.25
    return df


def backtest_season(df: pd.DataFrame, curves: dict, target: int) -> pd.DataFrame:
    recent_years = [target - 1, target - 2, target - 3]
    hist = df[df["mp_season_year"].isin(recent_years)]
    actual = df[df["mp_season_year"] == target][
        ["playerId", "points", "pts_per60", "toi_min", "games_played"]
    ].rename(columns={"points": "act_points", "pts_per60": "act_rate",
                      "toi_min": "act_toi", "games_played": "act_gp"})

    # Positional mean rate from the pre-target window.
    pm = hist.groupby("pos_group").apply(
        lambda g: np.average(g["pts_per60"], weights=g["toi_min"]), include_groups=False
    ).to_dict()

    wmap = dict(zip(recent_years, C.RECENCY_WEIGHTS))
    rows = []
    for pid, g in hist.groupby("playerId"):
        g = g.sort_values("mp_season_year")
        latest = g.iloc[-1]
        pos_group = latest["pos_group"]
        g = g.assign(rec_w=g["mp_season_year"].map(wmap).fillna(0.0))
        g = g.assign(blend_w=g["rec_w"] * g["toi_min"])
        if g["blend_w"].sum() <= 0:
            continue
        total_toi = g["toi_min"].sum()

        blended_rate = np.average(g["pts_per60"], weights=g["blend_w"])
        k = C.SKATER_REGRESS_TOI_MIN
        regressed = (blended_rate * total_toi + pm[pos_group] * k) / (total_toi + k)

        mean_age = np.average(g["age"], weights=g["blend_w"])
        target_age = mean_age + (target - np.average(g["mp_season_year"], weights=g["blend_w"]))
        mult = ac.age_multiplier(curves, "points", mean_age, target_age)
        model_rate = regressed * mult

        # Volume projection (same as production).
        toi_per_gp = np.average(g["toi_min"] / g["games_played"], weights=g["blend_w"])
        gp = dict(zip(g["mp_season_year"], g["games_played"]))
        num = den = 0.0
        for i, yr in enumerate(recent_years):
            if yr in gp:
                num += C.GP_RECENCY_WEIGHTS[i] * gp[yr]; den += C.GP_RECENCY_WEIGHTS[i]
        proj_gp = float(np.clip(0.80 * (num / den) + 0.20 * 70.0, 1, C.MAX_GP)) if den else 60.0
        model_pts = model_rate * (toi_per_gp * proj_gp) / 60.0

        # Baselines.
        last = g[g["mp_season_year"] == target - 1]
        naive1 = float(last["points"].iloc[0]) if len(last) else np.nan
        naive3 = float(g["points"].mean())
        last_rate = float(last["pts_per60"].iloc[0]) if len(last) else blended_rate

        rows.append({"playerId": pid, "model_pts": model_pts, "naive1": naive1,
                     "naive3": naive3, "model_rate": model_rate, "last_rate": last_rate})

    pred = pd.DataFrame(rows).merge(actual, on="playerId", how="inner")
    return pred


def _score(pred: pd.DataFrame, col: str, actual_col: str) -> tuple[float, float]:
    d = pred.dropna(subset=[col, actual_col])
    err = d[col] - d[actual_col]
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


if __name__ == "__main__":
    df = _seasons_frame()
    curves = ac.build_skater_age_curves()
    test_seasons = [2022, 2023, 2024, 2025]

    print("Backtest: season-long POINTS (players with a real target-season sample)\n")
    print(f"{'season':>7} {'n':>5} | {'model MAE':>10} {'naive1 MAE':>11} {'naive3 MAE':>11} "
          f"| {'model RMSE':>11} {'naive1 RMSE':>12}")
    agg = {"model": [], "naive1": [], "naive3": []}
    rate_agg = {"model": [], "last": []}
    for target in test_seasons:
        pred = backtest_season(df, curves, target)
        m_mae, m_rmse = _score(pred, "model_pts", "act_points")
        n1_mae, n1_rmse = _score(pred, "naive1", "act_points")
        n3_mae, n3_rmse = _score(pred, "naive3", "act_points")
        # Rate skill: predict per-60, scored on players with >=300 target TOI.
        rp = pred[pred["act_toi"] >= 300]
        rm_mae, _ = _score(rp, "model_rate", "act_rate")
        rl_mae, _ = _score(rp, "last_rate", "act_rate")
        agg["model"].append(m_mae); agg["naive1"].append(n1_mae); agg["naive3"].append(n3_mae)
        rate_agg["model"].append(rm_mae); rate_agg["last"].append(rl_mae)
        print(f"{target:>7} {len(pred):>5} | {m_mae:>10.2f} {n1_mae:>11.2f} {n3_mae:>11.2f} "
              f"| {m_rmse:>11.2f} {n1_rmse:>12.2f}")

    print(f"\n{'MEAN':>7}       | {np.mean(agg['model']):>10.2f} {np.mean(agg['naive1']):>11.2f} "
          f"{np.mean(agg['naive3']):>11.2f}")
    print(f"\nPer-60 RATE MAE (>=300 TOI):  model {np.mean(rate_agg['model']):.3f}  "
          f"vs last-season {np.mean(rate_agg['last']):.3f}")
