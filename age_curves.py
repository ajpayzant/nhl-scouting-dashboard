"""Empirical NHL age curves via the delta method.

The delta method avoids survivor bias: instead of averaging a raw stat at each age
(which is biased because only good old players keep playing), we measure how the SAME
player's per-60 rate changes from age a to age a+1, average those paired deltas across
all players, then chain them into a multiplicative curve.

Output: for each stat, a dict {age -> multiplier} normalised so peak age = 1.0.
A player's projected rate is scaled by curve[target_age] / curve[current_age].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_layer as dl

# Per-60 stats we build curves for (skaters). Goalies use a separate, gentler curve.
SKATER_RATE_STATS = ["points", "goals", "assists", "shots", "primaryAssists"]

AGE_MIN, AGE_MAX = 18, 42
PEAK_ANCHOR = 24    # normalise curve so this age = 1.0
DECLINE_AFTER = 29  # curve is forced non-increasing past the peak found within age<=this


def _skater_rates() -> pd.DataFrame:
    """Player-season per-60 rates joined with age. One row per player-season."""
    sk = dl.load_moneypuck_skaters()
    bios = dl.load_nhl_skater_bios()
    births = dl.player_birthdates(bios)[["playerId", "birthDate"]]

    df = sk.merge(births, on="playerId", how="left")
    df = df[df["icetime"] >= C.MIN_ICETIME_SEC].copy()

    toi_min = df["icetime"] / 60.0
    per60 = 60.0 / toi_min
    df["points"] = df["I_F_points"] * per60
    df["goals"] = df["I_F_goals"] * per60
    df["assists"] = (df["I_F_primaryAssists"] + df["I_F_secondaryAssists"]) * per60
    df["primaryAssists"] = df["I_F_primaryAssists"] * per60
    df["shots"] = df["I_F_shotsOnGoal"] * per60

    # Age as of Feb 1 of the season (mid-season reference), season start-year = mp_season_year.
    bd = pd.to_datetime(df["birthDate"], errors="coerce")
    ref = pd.to_datetime((df["mp_season_year"] + 1).astype("Int64").astype(str) + "-02-01",
                         errors="coerce")
    df["age"] = (ref - bd).dt.days / 365.25
    df["age_int"] = np.floor(df["age"]).astype("Int64")
    df["toi_min"] = toi_min
    return df.dropna(subset=["age_int"])


def build_skater_age_curves() -> dict[str, dict[int, float]]:
    df = _skater_rates()
    curves: dict[str, dict[int, float]] = {}

    for stat in SKATER_RATE_STATS:
        # Pair each player's season with their next season.
        a = df[["playerId", "age_int", stat, "toi_min"]].copy()
        b = a.copy()
        b["age_int"] = b["age_int"] - 1  # b's age+1 aligns to a's age
        pair = a.merge(b, on=["playerId", "age_int"], suffixes=("", "_next"))
        # Weight each delta by the smaller of the two TOI samples (harmonic-ish).
        w = np.minimum(pair["toi_min"], pair["toi_min_next"])
        # Multiplicative change, guarding against divide-by-zero.
        valid = (pair[stat] > 0.05) & (pair[f"{stat}_next"] >= 0)
        pair = pair[valid]
        w = w[valid]
        ratio = pair[f"{stat}_next"] / pair[stat]
        # Trim extreme ratios (small-sample noise) before averaging.
        lo, hi = ratio.quantile(0.02), ratio.quantile(0.98)
        keep = (ratio >= lo) & (ratio <= hi)

        deltas = {}
        for age in range(AGE_MIN, AGE_MAX):
            m = keep & (pair["age_int"] == age)
            if m.sum() < 20:  # too few pairs to trust
                deltas[age] = np.nan
                continue
            deltas[age] = np.average(ratio[m], weights=w[m])

        # Chain the age->age+1 multipliers into a level curve.
        ages = list(range(AGE_MIN, AGE_MAX + 1))
        curve = {AGE_MIN: 1.0}
        for age in range(AGE_MIN, AGE_MAX):
            step = deltas.get(age)
            if step is None or np.isnan(step):
                step = 1.0  # flat where data is thin
            curve[age + 1] = curve[age] * step

        # Regularise the survivor-biased tail. Scoring rates peak empirically at
        # 27-29; past that the delta method over-credits the few elite players who
        # play into their late 30s, producing an implausible upward drift. Find the
        # peak within a plausible window, then force a non-increasing curve after it.
        window = {a: v for a, v in curve.items() if a <= DECLINE_AFTER}
        peak_age = max(window, key=window.get)
        prev = curve[peak_age]
        for age in range(peak_age + 1, AGE_MAX + 1):
            curve[age] = min(curve[age], prev)
            prev = curve[age]

        # Normalise so PEAK_ANCHOR = 1.0.
        anchor = curve.get(PEAK_ANCHOR, 1.0)
        curve = {age: v / anchor for age, v in curve.items()}
        curves[stat] = curve

    return curves


def age_multiplier(curves: dict[str, dict[int, float]], stat: str,
                   from_age: float, to_age: float) -> float:
    """Multiplicative adjustment moving a `stat` rate from from_age to to_age."""
    c = curves.get(stat) or curves.get("points")
    if from_age is None or to_age is None or np.isnan(from_age) or np.isnan(to_age):
        return 1.0  # unknown age -> no adjustment
    fa = int(np.clip(round(from_age), AGE_MIN, AGE_MAX))
    ta = int(np.clip(round(to_age), AGE_MIN, AGE_MAX))
    denom = c.get(fa, 1.0)
    if denom <= 0:
        return 1.0
    return c.get(ta, 1.0) / denom


if __name__ == "__main__":
    curves = build_skater_age_curves()
    print("Skater age curves (multiplier vs age 24 peak):\n")
    hdr = "age " + " ".join(f"{s[:5]:>7}" for s in SKATER_RATE_STATS)
    print(hdr)
    for age in range(18, 41):
        row = f"{age:>3} " + " ".join(f"{curves[s].get(age, float('nan')):7.3f}"
                                       for s in SKATER_RATE_STATS)
        print(row)
