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

# Per-60 stats we build curves for (skaters). Goalies get their own curves below.
# The physical stats are on this list for the same reason the scoring ones are: hits and
# blocked shots have their own age shape, and borrowing the points curve for them (which
# is what happens to any stat without one) would be an assumption nobody checked.
SKATER_RATE_STATS = ["points", "goals", "assists", "shots", "primaryAssists",
                     "ixg", "blocks", "hits", "pim", "faceoffs_won"]

AGE_MIN, AGE_MAX = 18, 42
PEAK_ANCHOR = 24    # normalise curve so this age = 1.0
DECLINE_AFTER = 29  # curve is forced non-increasing past the peak found within age<=this

# Stats whose late-career shape is NOT forced to decline. Scoring rates do decline and
# the survivor-biased tail has to be regularised; a hit rate or a block rate is a role,
# not a talent, and a 36-year-old defenceman genuinely blocks more shots than he did at
# 24 because of where his coach plays him. Forcing those curves down would be inventing
# a decline the data does not show.
NO_FORCED_DECLINE = {"blocks", "hits", "pim", "faceoffs_won"}

# A pair only counts if BOTH seasons have a meaningful rate, expressed as a fraction of
# the stat's own league mean. A flat threshold does not work across stats: 0.05 points
# per 60 is negligible, but 0.05 faceoff wins per 60 is a winger who took two draws all
# season, and dividing next year's two draws by this year's one produces a ratio of 2.0
# that has nothing to do with ageing. Left unfiltered this made the faceoff curve claim
# a 39-year-old wins fourteen times as many draws as he did at 24.
MIN_RATE_FRACTION = 0.25

# No genuine one-year age effect on a per-60 rate is larger than this. Young players do
# improve fast (the points curve moves ~10% a year at 19-21, which is real), so the cap
# is loose enough to let that through and tight enough that one thin age cell cannot
# inject a 30% step that then chains into everything above it.
MAX_YEARLY_STEP = 0.15


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
    df["ixg"] = df["I_F_xGoals"] * per60
    df["blocks"] = df["shotsBlockedByPlayer"] * per60
    df["hits"] = df["I_F_hits"] * per60
    df["pim"] = df["penalityMinutes"] * per60
    df["faceoffs_won"] = df["faceoffsWon"] * per60

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
        # Both seasons must show a real rate for the ratio between them to mean
        # anything -- see MIN_RATE_FRACTION.
        floor = MIN_RATE_FRACTION * float(np.average(df[stat], weights=df["toi_min"]))
        valid = (pair[stat] > floor) & (pair[f"{stat}_next"] > 0)
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

        # Chain the age->age+1 multipliers into a level curve. Only the LOCAL slope of
        # this curve is ever used (a projection moves a player one or two years), so
        # each step is capped: a single thin cell should not be able to bend the curve.
        curve = {AGE_MIN: 1.0}
        for age in range(AGE_MIN, AGE_MAX):
            step = deltas.get(age)
            if step is None or np.isnan(step):
                step = 1.0  # flat where data is thin
            step = float(np.clip(step, 1.0 - MAX_YEARLY_STEP, 1.0 + MAX_YEARLY_STEP))
            curve[age + 1] = curve[age] * step

        # Regularise the survivor-biased tail. Scoring rates peak empirically at
        # 27-29; past that the delta method over-credits the few elite players who
        # play into their late 30s, producing an implausible upward drift. Find the
        # peak within a plausible window, then force a non-increasing curve after it.
        # Role stats are exempt (NO_FORCED_DECLINE): a veteran defenceman really does
        # block more shots than he used to, and that is usage, not survivor bias.
        if stat not in NO_FORCED_DECLINE:
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

    # ── Assists late-career decline ──────────────────────────────────────────
    # The raw delta method leaves the ASSISTS tail flat (frozen ~1.24 from 29-38)
    # because dropout is severe among aging playmakers (share not returning next
    # season climbs 20%->51% from age 26->38) — only the elite survive to produce
    # a delta, so the survivors don't decline. Backtest on aging skaters confirms
    # the flat tail OVER-projects 34+ playmakers (bias +1.9 assists). Points and
    # goals DO decline in the same data; a dropout-imputation estimator overcorrected
    # (tested, bias -3.2). The stable, data-grounded fix is to make assists decline
    # at the SAME rate these players' POINTS decline — a real, cleanly-measured slope,
    # not an arbitrary constant. Backtest: 34+ assists bias +1.9->+0.9, MAE 6.43->6.32,
    # and the full head-to-head shows no stat regresses.
    #
    # This has to be applied to BOTH assist curves. `project_skaters` projects primary
    # and secondary assists separately and adds them, so it reads the primaryAssists
    # curve directly -- and that one has the flattest tail of all (frozen at 1.297 from
    # age 30 to 42, i.e. the model claiming a 39-year-old sets up as many goals as he
    # did at 29). Fixing only the combined `assists` curve, as this originally did, left
    # the broken tail in the curve that actually reaches a projection.
    if "points" in curves:
        pts = curves["points"]
        for name in ("assists", "primaryAssists"):
            ast = curves.get(name)
            if ast is None:
                continue
            window = {a: v for a, v in ast.items() if a <= DECLINE_AFTER}
            peak = max(window, key=window.get)
            for age in range(peak + 1, AGE_MAX + 1):
                pts_slope = pts.get(age, 1.0) / pts.get(age - 1, 1.0)  # <=1 in decline
                ast[age] = ast[age - 1] * min(pts_slope, 1.0)

    return curves


def build_toi_age_curve() -> dict[int, float]:
    """Empirical TOI/GAME age curve (delta method), normalised so PEAK_ANCHOR = 1.0.

    Ice time is NOT flat with age: young players earn bigger roles (~+5-6%/yr in their
    early 20s) and aging players lose them (~-2 to -5%/yr past 30). Projecting a
    player's TOI purely from his own recent history therefore under-projects young
    risers (~34% too low at 18-21 in backtest) and over-projects fading vets (~+8% at
    33+). This curve captures that role trajectory; it is chained then smoothed to be
    monotone up to the peak and monotone down after (role growth then decline).
    """
    df = _skater_rates()
    df = df.dropna(subset=["games_played"]) if "games_played" in df else df
    df["toipg"] = df["toi_min"] / df["games_played"]

    a = df[["playerId", "age_int", "toipg", "toi_min"]].copy()
    b = a.copy()
    b["age_int"] = b["age_int"] - 1
    pair = a.merge(b, on=["playerId", "age_int"], suffixes=("", "_next"))
    w = np.minimum(pair["toi_min"], pair["toi_min_next"])
    valid = pair["toipg"] > 3.0
    pair, w = pair[valid], w[valid]
    ratio = pair["toipg_next"] / pair["toipg"]
    lo, hi = ratio.quantile(0.02), ratio.quantile(0.98)
    keep = (ratio >= lo) & (ratio <= hi)

    deltas = {}
    for age in range(AGE_MIN, AGE_MAX):
        m = keep & (pair["age_int"] == age)
        deltas[age] = np.average(ratio[m], weights=w[m]) if m.sum() >= 20 else np.nan

    curve = {AGE_MIN: 1.0}
    for age in range(AGE_MIN, AGE_MAX):
        step = deltas.get(age)
        if step is None or np.isnan(step):
            step = 1.0
        curve[age + 1] = curve[age] * step

    # Enforce the shape we trust: non-decreasing up to the peak, non-increasing after.
    window = {a_: v for a_, v in curve.items() if a_ <= DECLINE_AFTER}
    peak_age = max(window, key=window.get)
    prev = -np.inf
    for age in range(AGE_MIN, peak_age + 1):
        curve[age] = max(curve[age], prev)
        prev = curve[age]
    prev = curve[peak_age]
    for age in range(peak_age + 1, AGE_MAX + 1):
        curve[age] = min(curve[age], prev)
        prev = curve[age]

    anchor = curve.get(PEAK_ANCHOR, 1.0)
    return {age: v / anchor for age, v in curve.items()}


GOALIE_AGE_MIN, GOALIE_AGE_MAX = 20, 42
GOALIE_PEAK_ANCHOR = 27


def build_goalie_age_curves() -> dict[str, dict[int, float]]:
    """Delta-method age curves for goalies: shot-stopping and workload.

    The goalie model previously had no age term at all, so a 38-year-old and a
    28-year-old with the same three seasons behind them were projected identically.

    Shot-stopping is measured on GOALS ALLOWED PER SHOT (1 - SV%), not on SV% itself,
    and this is the important choice. SV% is a bounded rate hovering around .900, so a
    multiplicative curve on it is numerically flat -- the whole career range fits inside
    two percent -- and rounds away the signal. The same skill expressed as goals per
    shot ranges from about .085 to .115, so a 5% deterioration is a 5% move in the
    number and survives the arithmetic. A multiplier ABOVE 1.0 therefore means a goalie
    lets in more, i.e. is worse.

    Workload is a straight GP curve: it captures a young goalie earning the starts and
    an old one losing them, which is a real and large effect the recency blend alone
    treats as noise.
    """
    gs = dl.load_nhl_goalie_summary()
    births = dl.goalie_birthdates(dl.load_nhl_goalie_bios())[["playerId", "birthDate"]]
    df = gs.merge(births, on="playerId", how="left")
    df = df[df["shotsAgainst"] >= 400].copy()          # a real sample only
    df["ga_per_shot"] = df["goalsAgainst"] / df["shotsAgainst"]
    df["workload"] = df["gamesPlayed"]
    bd = pd.to_datetime(df["birthDate"], errors="coerce")
    ref = pd.to_datetime((df["mp_season_year"] + 1).astype(str) + "-02-01", errors="coerce")
    df["age_int"] = np.floor((ref - bd).dt.days / 365.25).astype("Int64")
    df = df.dropna(subset=["age_int"])

    curves: dict[str, dict[int, float]] = {}
    for stat, weight_col in (("ga_per_shot", "shotsAgainst"), ("workload", "gamesPlayed")):
        a = df[["playerId", "age_int", stat, weight_col]].copy()
        b = a.copy()
        b["age_int"] = b["age_int"] - 1
        pair = a.merge(b, on=["playerId", "age_int"], suffixes=("", "_next"))
        pair = pair[pair[stat] > 0]
        if pair.empty:
            continue
        w = np.minimum(pair[weight_col], pair[f"{weight_col}_next"])
        ratio = pair[f"{stat}_next"] / pair[stat]
        lo, hi = ratio.quantile(0.02), ratio.quantile(0.98)
        keep = (ratio >= lo) & (ratio <= hi)

        curve = {GOALIE_AGE_MIN: 1.0}
        for age in range(GOALIE_AGE_MIN, GOALIE_AGE_MAX):
            m = keep & (pair["age_int"] == age)
            # Goalie samples are an order of magnitude thinner than skater samples
            # (about 90 goalies a season, not 900), so the minimum pair count has to be
            # lower or every curve comes out flat -- and a flat curve is the bug.
            step = np.average(ratio[m], weights=w[m]) if m.sum() >= 8 else 1.0
            step = float(np.clip(step, 1.0 - MAX_YEARLY_STEP, 1.0 + MAX_YEARLY_STEP))
            curve[age + 1] = curve[age] * step

        # Smooth the tail the way the evidence supports: shot-stopping does not come
        # back, so goals allowed per shot is forced non-decreasing after its best age,
        # and workload non-increasing after its peak.
        window = {k: v for k, v in curve.items() if k <= 32}
        if stat == "ga_per_shot":
            best = min(window, key=window.get)
            prev = curve[best]
            for age in range(best + 1, GOALIE_AGE_MAX + 1):
                curve[age] = max(curve[age], prev)
                prev = curve[age]
        else:
            peak = max(window, key=window.get)
            prev = curve[peak]
            for age in range(peak + 1, GOALIE_AGE_MAX + 1):
                curve[age] = min(curve[age], prev)
                prev = curve[age]

        anchor = curve.get(GOALIE_PEAK_ANCHOR, 1.0) or 1.0
        curve = {age: v / anchor for age, v in curve.items()}

        # Damp the shot-stopping curve, because the two ways of measuring it disagree
        # and the delta method is the one with a known bias here. Paired year-over-year
        # changes say a goalie loses 2-3 points of save percentage a year relative to
        # league from 26 on; the cross-section says roughly flat until 36. Both are
        # wrong in a known direction -- the cross-section because bad old goalies stop
        # getting starts, the pairs because a goalie with a big sample this season was
        # partly lucky in it and regresses next season for reasons that are not age.
        # Save percentage is far more luck-dominated than any skater rate, so that
        # second bias is large. C.GOALIE_AGE_STRENGTH raises the multiplier to a power:
        # 1.0 believes the pairs, 0.0 ignores age, and the value is set by backtest.
        # Workload is left alone -- games started is not luck, so it is cleanly measured.
        if stat == "ga_per_shot":
            # And the curve is held FLAT before the peak. Left as measured, the paired
            # method says a 20-year-old stops 9.5% more of what he faces than the same
            # goalie will at 27 and gets monotonically worse from there, which is not
            # credible and is a bias with a known source: a goalie only gets a 400-shot
            # season at 20 by playing above his true level, so his age-21 season regresses
            # for reasons that have nothing to do with ageing, and the entry threshold is
            # most selective exactly where the curve is steepest. The cross-section over
            # the same seasons is flat from 20 to 36 (within +-3%, n=4 to 103 per age) and
            # only turns up after 37, so the two methods agree on the decline and disagree
            # about a rise that neither can separate from selection. Flat before the peak
            # is what is left when the artefact is removed -- and it matters: extrapolated
            # across the seven years from the anchor, the raw curve made a 20-year-old with
            # no NHL history the best goaltending prior in the league.
            curve = {age: (1.0 if age <= GOALIE_PEAK_ANCHOR else max(v, 1.0))
                     for age, v in curve.items()}
            s = float(C.GOALIE_AGE_STRENGTH)
            curve = {age: float(v) ** s for age, v in curve.items()}
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
