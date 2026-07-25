"""Game-by-game skater projections via TOP-DOWN decomposition.

Per the modeling research: simulating games and summing does NOT improve season-total
accuracy (linearity of expectation — the mean is unchanged). So instead of building the
season up from per-game sims, we take the validated season projection as an ANCHOR and
distribute it across the team's real 84-game schedule using normalized per-game weights
(opponent strength, home/away, back-to-back rest).

Because the weights are normalized to average 1.0 across each team's games, the per-game
projections sum EXACTLY back to the season total (conservation):  Σ game_i = season.
This yields useful per-game lines (matchup, home/away, rest-adjusted) without degrading
the season number.

Optionally attaches a per-game distribution (Poisson mean = the game's expected value)
so you can read P(2+ points), etc., for props — again without changing the mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import context as ctx
import project_skaters as ps

# Season-total columns to distribute across games.
DIST_STATS = ["proj_points", "proj_goals", "proj_assists", "proj_shots", "proj_pp_points"]


def project_games(season: int = C.TARGET_SEASON,
                  season_proj: pd.DataFrame | None = None) -> pd.DataFrame:
    """Expand season skater projections into one row per (player, game).

    Returns columns: playerId, name, team, gameId, gameDate, opponent, is_home,
    plus per-game expected stats (points/goals/assists/shots/pp_points) that sum to
    the player's season projection.
    """
    if season_proj is None:
        season_proj = ps.project_skaters()

    sched = ctx.schedule_context(season)  # team, game rows with game_weight
    # Games played per season are projected as a fraction of the full schedule; we
    # scale each game's weight so the player's expected games-played is preserved:
    # a player projected for 70 of 84 games contributes 70/84 of a "full" game each night.

    out_rows = []
    # Index schedule by team for fast lookup.
    sched_by_team = {t: g.sort_values("gameDate") for t, g in sched.groupby("team")}

    for _, p in season_proj.iterrows():
        team = p["team"]
        tsched = sched_by_team.get(team)
        if tsched is None or len(tsched) == 0:
            continue
        n_games = len(tsched)
        # Availability fraction: projected GP spread across the schedule.
        avail = float(np.clip(p["proj_gp"] / n_games, 0.0, 1.0))

        w = tsched["game_weight"].to_numpy()
        w = w / w.mean()  # ensure mean 1.0 within this team's games

        for stat in DIST_STATS:
            season_total = p[stat]
            # Per-game expectation = (season_total / n_games) * game_weight.
            # Summed over games this is exactly season_total (weights average 1.0),
            # regardless of availability (availability is already baked into proj_gp,
            # which set season_total). game_weight only reshapes the distribution.
            per_game = (season_total / n_games) * w
            tsched = tsched.assign(**{f"g_{stat}": per_game})

        for _, gm in tsched.iterrows():
            out_rows.append({
                "playerId": p["playerId"], "name": p["name"], "team": team,
                "gameId": gm["gameId"], "gameDate": gm["gameDate"],
                "opponent": gm["opponent"], "is_home": gm["is_home"],
                "exp_points": round(gm["g_proj_points"], 3),
                "exp_goals": round(gm["g_proj_goals"], 3),
                "exp_assists": round(gm["g_proj_assists"], 3),
                "exp_shots": round(gm["g_proj_shots"], 3),
                "exp_pp_points": round(gm["g_proj_pp_points"], 3),
            })

    return pd.DataFrame(out_rows)


def add_prop_probabilities(games: pd.DataFrame) -> pd.DataFrame:
    """Attach common per-game prop probabilities using a Poisson on the expected value.

    P(1+ points), P(2+ points), P(1+ goals), P(2+ shots... ) etc. The Poisson mean is
    the game's expected value, so these are consistent with the (unchanged) totals.
    """
    from scipy.stats import poisson
    g = games.copy()
    g["p_1plus_points"] = 1 - poisson.cdf(0, g["exp_points"])
    g["p_2plus_points"] = 1 - poisson.cdf(1, g["exp_points"])
    g["p_1plus_goals"] = 1 - poisson.cdf(0, g["exp_goals"])
    g["p_anytime_goal"] = g["p_1plus_goals"]  # alias for the common market name
    return g


def verify_conservation(games: pd.DataFrame, season_proj: pd.DataFrame) -> pd.DataFrame:
    """Sanity check: per-game sums must equal the season projection (within rounding)."""
    s = games.groupby("playerId")[["exp_points", "exp_goals", "exp_assists"]].sum()
    s = s.merge(season_proj.set_index("playerId")[["name", "proj_points", "proj_goals"]],
                left_index=True, right_index=True)
    s["points_err"] = (s["exp_points"] - s["proj_points"]).abs()
    s["goals_err"] = (s["exp_goals"] - s["proj_goals"]).abs()
    return s.sort_values("points_err", ascending=False)


if __name__ == "__main__":
    season_proj = ps.project_skaters()
    games = project_games(season_proj=season_proj)
    print(f"Game-by-game rows: {len(games)}  "
          f"({games['playerId'].nunique()} players x their schedules)\n")

    # Show a star player's first few games.
    mcd = games[games["name"] == "Connor McDavid"].head(6)
    print("Connor McDavid — first 6 games:")
    print(mcd[["gameDate", "opponent", "is_home", "exp_points", "exp_goals",
               "exp_assists", "exp_shots"]].to_string(index=False))

    # Conservation check.
    chk = verify_conservation(games, season_proj)
    print(f"\nConservation check (per-game sums vs season totals):")
    print(f"  max points error across all players: {chk['points_err'].max():.4f}")
    print(f"  max goals  error across all players: {chk['goals_err'].max():.4f}")

    path = C.OUTPUT / f"skater_game_projections_{C.TARGET_SEASON}.csv"
    games.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(games)} game rows -> {path}")
