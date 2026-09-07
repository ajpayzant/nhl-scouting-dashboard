"""Team strength ratings and per-game schedule context.

Provides two things the season projection consumes:

  1. team_defense_ratings() -> per-team OFFENSE and DEFENSE strength relative to the
     league (recent seasons), used to build opponent (strength-of-schedule) multipliers.
     A defense rating >1.0 means the team SUPPRESSES scoring (opponents score less);
     for a skater, facing that team should reduce expected production.

  2. schedule_context() -> for every (team, game) in the target schedule, a set of
     mean-1.0-normalized multipliers (opponent, home/away, back-to-back rest). Because
     they are normalized to average 1.0 across each team's 84 games, applying them to a
     season total REDISTRIBUTES it across games without changing the sum — UNLESS the
     schedule is unbalanced (divisional weighting), in which case a genuine, small SOS
     shift in the season total is allowed to flow through (see apply_sos()).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_layer as dl


def team_defense_ratings() -> pd.DataFrame:
    """Per-team offense/defense strength vs league, averaged over recent seasons.

    def_rating > 1.0  => team allows FEWER goals than average (strong defense).
    off_rating > 1.0  => team scores MORE goals than average (strong offense).
    Uses a blend of actual goals and expected goals (xG) for stability.
    """
    teams = dl.load_moneypuck_teams()
    recent = teams[teams["mp_season_year"] >= C.LAST_COMPLETED_SEASON - C.TEAM_STRENGTH_SEASONS + 1].copy()

    recent["gf_pg"] = recent["goalsFor"] / recent["games_played"]
    recent["ga_pg"] = recent["goalsAgainst"] / recent["games_played"]
    recent["xgf_pg"] = recent["xGoalsFor"] / recent["games_played"]
    recent["xga_pg"] = recent["xGoalsAgainst"] / recent["games_played"]
    # Blend actual and expected (xG is more stable / predictive of future).
    recent["off_pg"] = 0.5 * recent["gf_pg"] + 0.5 * recent["xgf_pg"]
    recent["def_pg"] = 0.5 * recent["ga_pg"] + 0.5 * recent["xga_pg"]

    agg = recent.groupby("team").agg(off_pg=("off_pg", "mean"),
                                     def_pg=("def_pg", "mean")).reset_index()
    lg_off = agg["off_pg"].mean()
    lg_def = agg["def_pg"].mean()
    # Offense rating: >1 scores more than league. Defense rating: >1 allows fewer.
    agg["off_rating"] = agg["off_pg"] / lg_off
    agg["def_rating"] = lg_def / agg["def_pg"]
    return agg[["team", "off_rating", "def_rating"]]


def _opponent_scoring_multiplier(def_rating: float) -> float:
    """How a skater's expected offense scales when facing a team with this def_rating.

    def_rating > 1 (strong defense) => opponent allows fewer goals => multiplier < 1.
    The effect is softened by SKATER_OPP_STRENGTH so skater SOS stays a gentle tweak.
    """
    raw = 1.0 / def_rating          # facing a strong D suppresses your output
    return 1.0 + C.SKATER_OPP_STRENGTH * (raw - 1.0)


def schedule_context(season: int = C.TARGET_SEASON) -> pd.DataFrame:
    """One row per (team, game) with normalized skater production multipliers.

    Columns added to the schedule:
      opp_mult      : opponent-strength multiplier (mean 1.0 within team by construction
                      only if schedule balanced; unbalanced schedules leave a residual)
      home_mult     : home/away multiplier
      b2b_mult      : back-to-back rest multiplier
      game_weight   : combined per-game weight, normalized so the team's 84 weights average 1.0
      sos_factor    : the team's season-level SOS factor = mean opp_mult (captures the
                      genuine, non-zero-sum shift from an unbalanced schedule)
    """
    sched = dl.load_schedule(season).copy()
    ratings = team_defense_ratings().set_index("team")
    lg_def_mean = 1.0  # def_rating is already league-normalized

    # Opponent multiplier from opponent defense rating.
    def opp_mult(opp):
        if opp in ratings.index:
            return _opponent_scoring_multiplier(ratings.loc[opp, "def_rating"])
        return 1.0
    sched["opp_mult"] = sched["opponent"].map(opp_mult)

    # Home/away: +boost at home, symmetric -boost away (mean ~1.0 over balanced schedule).
    sched["home_mult"] = np.where(sched["is_home"], 1.0 + C.HOME_ICE_BOOST,
                                  1.0 - C.HOME_ICE_BOOST)

    # Back-to-back: penalty on the 2nd of games on consecutive calendar days.
    sched["gdate"] = pd.to_datetime(sched["gameDate"])
    sched = sched.sort_values(["team", "gdate"])
    prev = sched.groupby("team")["gdate"].shift(1)
    is_b2b = (sched["gdate"] - prev).dt.days == 1
    sched["b2b_mult"] = np.where(is_b2b.fillna(False), 1.0 - C.B2B_PENALTY, 1.0)

    # Per-game raw weight = product of the three effects.
    sched["raw_weight"] = sched["opp_mult"] * sched["home_mult"] * sched["b2b_mult"]

    # Season SOS factor: the mean opponent multiplier a team faces, RE-CENTRED so the
    # league averages exactly 1.0. The centring is not cosmetic. opp_mult is built from
    # 1/def_rating, and the mean of a reciprocal exceeds the reciprocal of the mean
    # (Jensen), so the raw average came out above 1.0 for nearly every team -- strength
    # of schedule was quietly handing the whole league a bonus of a few tenths of a
    # percent instead of describing who has it easier than whom. Only the differences
    # between teams are real; the level belongs to the league budget.
    team_sos = sched.groupby("team")["opp_mult"].mean()
    team_sos = (team_sos / team_sos.mean()).rename("sos_factor")
    sched = sched.merge(team_sos, on="team")

    # game_weight: normalize raw weights so each team's games average 1.0. This makes
    # home/rest purely redistributive; the opponent's non-zero-sum part is carried
    # separately in sos_factor so it isn't normalized away.
    team_mean = sched.groupby("team")["raw_weight"].transform("mean")
    sched["game_weight"] = sched["raw_weight"] / team_mean

    return sched


if __name__ == "__main__":
    r = team_defense_ratings().sort_values("def_rating", ascending=False)
    print("Team strength ratings (def_rating>1 = strong defense):\n")
    print(r.head(8).to_string(index=False))
    print("...")
    print(r.tail(4).to_string(index=False))

    sc = schedule_context()
    print(f"\nSchedule context: {len(sc)} team-games")
    sos = sc.groupby("team")["sos_factor"].first().sort_values()
    print("\nToughest schedules (lowest SOS factor = faces strong defenses most):")
    print(sos.head(5).to_string())
    print("\nEasiest schedules:")
    print(sos.tail(5).to_string())
