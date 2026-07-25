"""Season-long goalie projections for the upcoming season (2026-27).

Goalies are notoriously hard to project, so the model is deliberately conservative:

  1. SV% BASE: blend the last up-to-3 seasons of save percentage, recency-weighted
     and further weighted by shots faced (a 2000-shot season counts far more than 400).
  2. REGRESSION: pull SV% hard toward the league mean by shots faced (GOALIE_REGRESS_SHOTS).
     Save-percentage skill is low-signal year-to-year, so this regression is strong.
  3. VOLUME: project games started / games played (recency-weighted), shots-against per
     game from recent workload, and team save context.
  4. DERIVE: saves = shots * SV%; GAA and wins from projected workload and quality.

Wins depend heavily on team strength; we anchor to the goalie's recent win rate per
start, lightly regressed, since we're not modelling full team schedules here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_layer as dl


def _prep_goalie_seasons() -> pd.DataFrame:
    """One row per goalie-season combining NHL summary (W/SO/GAA) + MoneyPuck (xG)."""
    gs = dl.load_nhl_goalie_summary()
    gs = gs.rename(columns={"playerId": "playerId"})
    # NHL summary carries the real W/L/SV%/GAA/SO we want to project.
    keep = ["playerId", "goalieFullName", "teamAbbrevs", "mp_season_year",
            "gamesPlayed", "gamesStarted", "wins", "losses", "otLosses",
            "savePct", "goalsAgainstAverage", "shutouts", "saves", "shotsAgainst",
            "timeOnIce"]
    g = gs[keep].copy()
    g = g[g["shotsAgainst"] >= 100]  # real sample only
    g["sa_per_gp"] = g["shotsAgainst"] / g["gamesPlayed"]
    g["win_rate"] = g["wins"] / g["gamesPlayed"]
    g["so_rate"] = g["shutouts"] / g["gamesPlayed"]
    g["start_share"] = (g["gamesStarted"] / g["gamesPlayed"]).clip(upper=1.0)
    return g


def project_goalies() -> pd.DataFrame:
    hist = _prep_goalie_seasons()
    target = C.TARGET_SEASON
    recent_years = [target - 1, target - 2, target - 3]

    # League-average SV% and shots-against/GP over recent seasons (shot-weighted).
    recent = hist[hist["mp_season_year"] >= C.LAST_COMPLETED_SEASON - 2]
    lg_svpct = np.average(recent["savePct"], weights=recent["shotsAgainst"])
    lg_sa_gp = np.average(recent["sa_per_gp"], weights=recent["gamesPlayed"])

    pool = hist[hist["mp_season_year"].isin(recent_years)].copy()
    wmap = dict(zip(recent_years, C.GP_RECENCY_WEIGHTS))

    # Current-season team + team strength. For goalies, team context is FIRST-ORDER:
    # a strong-defense team suppresses shots-against (fewer saves, better GAA) and lifts
    # the win rate. We apply the NEW team's ratings, not the goalie's prior team.
    import context as ctx
    roster = dl.load_rosters(target)
    cur_team = dict(zip(roster["playerId"], roster["team"]))
    ratings = ctx.team_defense_ratings().set_index("team")

    rows = []
    for pid, g in pool.groupby("playerId"):
        g = g.sort_values("mp_season_year")
        latest = g.iloc[-1]
        g = g.assign(rec_w=g["mp_season_year"].map(wmap).fillna(0.0))
        # SV% weighted by recency * shots faced.
        sv_w = g["rec_w"] * g["shotsAgainst"]
        if sv_w.sum() <= 0:
            continue
        total_shots = g["shotsAgainst"].sum()

        blended_sv = np.average(g["savePct"], weights=sv_w)
        k = C.GOALIE_REGRESS_SHOTS
        proj_sv = (blended_sv * total_shots + lg_svpct * k) / (total_shots + k)

        # Games played: recency-weighted, regressed toward a starter/backup-neutral prior.
        gp_w = g["rec_w"]
        blended_gp = np.average(g["gamesPlayed"], weights=gp_w)
        proj_gp = float(np.clip(0.75 * blended_gp + 0.25 * 40.0, 1, 70))

        # Team context for the UPCOMING season (first-order for goalies).
        team = cur_team.get(pid, latest["teamAbbrevs"])
        on_roster = pid in cur_team
        # Use the first listed team abbrev if the goalie was traded mid-history.
        team_key = str(team).split(",")[0]
        def_rating = ratings.loc[team_key, "def_rating"] if team_key in ratings.index else 1.0

        # Shots against per game: recent workload blended toward what THIS team's defense
        # allows. A strong defense (def_rating>1) suppresses shots faced.
        blended_sa = np.average(g["sa_per_gp"], weights=g["rec_w"] * g["gamesPlayed"])
        team_sa_gp = lg_sa_gp / def_rating  # strong D => fewer shots against
        # 60% the goalie's own recent workload, 40% the new team's shot-suppression profile.
        proj_sa_gp = 0.6 * blended_sa + 0.4 * team_sa_gp
        proj_shots = proj_sa_gp * proj_gp

        # Win rate per game: recent form, regressed toward a start-neutral prior, then
        # nudged by team quality (better team => more wins independent of the goalie).
        blended_wr = np.average(g["win_rate"], weights=g["rec_w"] * g["gamesPlayed"])
        proj_wr = (blended_wr * total_shots + 0.42 * 1500.0) / (total_shots + 1500.0)
        team_quality = (ratings.loc[team_key, "off_rating"] * def_rating) if team_key in ratings.index else 1.0
        proj_wr = float(np.clip(proj_wr * (0.5 + 0.5 * team_quality), 0.05, 0.85))

        # Shutout rate scales with save quality vs league.
        blended_so = np.average(g["so_rate"], weights=g["rec_w"] * g["gamesPlayed"])
        proj_so_rate = 0.6 * blended_so + 0.4 * (recent["so_rate"].mean())

        proj_saves = proj_shots * proj_sv
        proj_ga = proj_shots * (1 - proj_sv)
        proj_gaa = proj_ga / proj_gp  # goals against per appearance (~ per game)

        rows.append({
            "playerId": pid,
            "name": latest["goalieFullName"],
            "team": team_key,
            "prior_team": latest["teamAbbrevs"],
            "on_roster": on_roster,
            "proj_gp": round(proj_gp, 1),
            "proj_wins": round(proj_wr * proj_gp, 1),
            "proj_save_pct": round(proj_sv, 4),
            "proj_gaa": round(proj_gaa, 2),
            "proj_saves": round(proj_saves, 0),
            "proj_shutouts": round(proj_so_rate * proj_gp, 1),
            "proj_shots_against": round(proj_shots, 0),
        })

    out = pd.DataFrame(rows).sort_values("proj_wins", ascending=False).reset_index(drop=True)
    return out


if __name__ == "__main__":
    out = project_goalies()
    print(f"Projected {len(out)} goalies for {C.TARGET_SEASON}-{C.TARGET_SEASON+1}\n")
    cols = ["name", "team", "proj_gp", "proj_wins", "proj_save_pct",
            "proj_gaa", "proj_saves", "proj_shutouts"]
    print(out[cols].head(25).to_string(index=False))
    path = C.OUTPUT / f"goalie_projections_{C.TARGET_SEASON}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {path}")
