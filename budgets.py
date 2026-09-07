"""Team budgets: how much there is to go round.

Everything a skater does comes out of something his team only has a fixed amount of.
There are ~297 skater-minutes in a team's game whatever the roster looks like; a team
scores about as many goals as its offence is worth; every goal carries 0.935 primary
assists and 0.751 secondary assists, league-wide, every single season. These are the
constraints the old model did not know about, and they are strong: measured across
2021-2025 the team ice-time budget varies by less than two tenths of a percent from
season to season, and the assists-per-goal ratios are stable to the third decimal.

Two separate jobs here, and they are worth keeping apart:

  1. THE LEAGUE LEVEL -- how much a typical team-game is worth. This is measured, not
     assumed, and it is TRENDING, which matters: shots on goal have fallen from 31.6
     to 27.8 per team-game since 2021 and hits from 22.9 to 20.4. A flat average over
     five seasons projects a league that no longer exists, so the level is a
     front-loaded blend of the last three (LEAGUE_RATE_WEIGHTS).

  2. THE TEAM'S SHARE of it. Ice time is shared exactly equally -- it is a property of
     the clock, not the team -- but goals are not, so offensive budgets are scaled by a
     team rating and then renormalised so the league total is conserved to the goal.

A team rating is deliberately not believed in full. Two seasons of team scoring is
part talent and part luck, and the honest weight is measurable: `rating_persistence()`
regresses a season's team rating on its own prior two-season blend, and the slope of
that regression IS the fraction worth carrying forward (TEAM_RATING_SHRINK). A rating
is also blended with what the team's own projected players claim, because the rating
is about last year's roster and the claims are about this year's -- that is what lets
a team that traded for a scorer carry a bigger budget than its history alone implies.

Team totals are read off the TEAM file, never off the player file. MoneyPuck gives a
traded player a single season row on one of his teams, so aggregating players by team
scatters the ice-time budget across a 252-325 range that does not exist in reality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_layer as dl

# Player stat -> the team-file column that measures the same thing. `blocks` is the
# one that reads backwards: blockedShotAttemptsAgainst is the opponent's attempts that
# THIS team blocked, which is the team's block total.
TEAM_SOURCE = {
    "goals": "goalsFor",
    "ixg": "xGoalsFor",
    "shots": "shotsOnGoalFor",
    "blocks": "blockedShotAttemptsAgainst",
    "hits": "hitsFor",
    "pim": "penalityMinutesFor",
    "faceoffs_won": "faceOffsWonFor",
}
# Ratings for these are stabilised by averaging with the expected-goals version, which
# is the more repeatable half of a scoring rate.
RATING_XG_PARTNER = {"goals": "xGoalsFor"}


# --------------------------------------------------------------------------- #
# the league level                                                            #
# --------------------------------------------------------------------------- #
def _team_games(season: int) -> pd.Series:
    t = dl.load_moneypuck_teams()
    return t[t["mp_season_year"] == season].set_index("team")["games_played"]


def _skater_league_totals(season: int) -> dict:
    """League totals per team-game from the player file, for one season.

    Summed league-wide, so the trade artefact that makes per-team player aggregates
    unusable cancels out exactly.
    """
    sk = dl.load_moneypuck_skaters()
    s = sk[sk["mp_season_year"] == season]
    if s.empty:
        return {}
    team_games = float(_team_games(season).sum())
    if team_games <= 0:
        return {}
    goals = float(s["I_F_goals"].sum())
    out = {
        # Skaters dressed per game. Measured 17.990 / 17.990 / 17.996 / 17.997 / 17.992
        # over 2021-25, because it is a rule and not a tendency: a team may carry 20
        # players and two of them are goaltenders. This is what makes a fringe forward's
        # games played a claim on something finite rather than a free parameter.
        "dressed": float(s["games_played"].sum()) / team_games,
        "toi_min": float(s["icetime"].sum()) / 60.0 / team_games,
        "goals": goals / team_games,
        "shots": float(s["I_F_shotsOnGoal"].sum()) / team_games,
        "ixg": float(s["I_F_xGoals"].sum()) / team_games,
        "blocks": float(s["shotsBlockedByPlayer"].sum()) / team_games,
        "hits": float(s["I_F_hits"].sum()) / team_games,
        "pim": float(s["penalityMinutes"].sum()) / team_games,
        "faceoffs_won": float(s["faceoffsWon"].sum()) / team_games,
        # Per GOAL, not per game: these two ratios are the most stable numbers in the
        # sport (0.934-0.937 and 0.748-0.755 across five seasons) and they are what
        # ties a team's assist budget to its own goal budget.
        "primary_per_goal": float(s["I_F_primaryAssists"].sum()) / goals,
        "secondary_per_goal": float(s["I_F_secondaryAssists"].sum()) / goals,
    }
    # Special teams. The situation files carry each player's own ice time, so a team's
    # MINUTES on the power play is the skater-minutes divided by the skaters on the ice
    # -- five at 5on4, four at 4on5. Getting that divisor wrong would misprice every
    # per-PP-60 rate by a quarter.
    for situation, key, on_ice in (("5on4", "pp", 5.0), ("4on5", "sh", 4.0)):
        try:
            sit = dl.load_moneypuck_situation(situation)
        except (FileNotFoundError, ValueError):
            continue
        ss = sit[sit["mp_season_year"] == season]
        if ss.empty:
            continue
        pts = (ss["I_F_goals"] + ss["I_F_primaryAssists"] + ss["I_F_secondaryAssists"]).sum()
        out[f"{key}_points"] = float(pts) / team_games
        out[f"{key}_toi_min"] = float(ss["icetime"].sum()) / 60.0 / team_games / on_ice
    return out


def league_level(last: int | None = None) -> dict:
    """League production per team-game, trend-weighted over the last few seasons."""
    last = C.LAST_COMPLETED_SEASON if last is None else last
    weights = C.LEAGUE_RATE_WEIGHTS
    seasons = [last - i for i in range(len(weights))]
    frames = [(w, _skater_league_totals(s)) for w, s in zip(weights, seasons)]
    frames = [(w, d) for w, d in frames if d]
    if not frames:
        raise RuntimeError("no seasons of skater data to measure the league level from")
    keys = set().union(*(d.keys() for _, d in frames))
    out = {}
    for k in keys:
        num = sum(w * d[k] for w, d in frames if k in d)
        den = sum(w for w, d in frames if k in d)
        out[k] = num / den
    return out


def goalie_league_level(last: int | None = None) -> dict:
    """Goalie budgets per team and the league save-percentage level.

    Three of these are not estimates at all. A team plays 84 games, so it makes exactly
    84 goalie STARTS and its goalies win exactly as many games as the team does; the
    league wins 32 x 84 / 2 = 1344 of them and not one more. And there is one goalie on
    the ice for the whole game, so a team's goalies play 60.1 minutes per team-game
    (measured 60.05 / 59.96 / 60.19 over 2023-25 -- sixty for the clock, plus overtime,
    less the minute or two of a pulled net). Appearances exceed starts only by relief
    work, 1.055 per start. The old model projected 114 appearances and 1643 league wins.

    Everything is expressed PER START, which is the same thing as per team-game because
    starts per team-game is exactly 1.000. That keeps the arithmetic checkable.
    """
    last = C.LAST_COMPLETED_SEASON if last is None else last
    weights = C.LEAGUE_RATE_WEIGHTS
    gs = dl.load_nhl_goalie_summary()
    rows = []
    for w, season in zip(weights, [last - i for i in range(len(weights))]):
        g = gs[gs["mp_season_year"] == season]
        if g.empty:
            continue
        games = float(_team_games(season).mean())
        starts = float(g["gamesStarted"].sum())
        rows.append((w, {
            # appearances per start: relief work, the only part that is not fixed
            "appearances_per_start": float(g["gamesPlayed"].sum()) / starts,
            "shutouts_per_start": float(g["shutouts"].sum()) / starts,
            "sa_per_start": float(g["shotsAgainst"].sum()) / starts,
            "ga_per_start": float(g["goalsAgainst"].sum()) / starts,
            # timeOnIce in this feed is SECONDS, not minutes. Reading it as minutes is
            # what made the old proj_gaa divide by the wrong quantity entirely.
            "min_per_start": float(g["timeOnIce"].sum()) / 60.0 / starts,
            "sv_pct": 1.0 - float(g["goalsAgainst"].sum()) / float(g["shotsAgainst"].sum()),
            "min_per_appearance": float(g["timeOnIce"].sum()) / 60.0 / float(g["gamesPlayed"].sum()),
            "sa_per_60": float(g["shotsAgainst"].sum()) * 3600.0 / float(g["timeOnIce"].sum()),
            "games": games,
        }))
    if not rows:
        raise RuntimeError("no goalie seasons to measure from")
    keys = set().union(*(d.keys() for _, d in rows))
    out = {k: sum(w * d[k] for w, d in rows if k in d) / sum(w for w, d in rows if k in d)
           for k in keys}
    out["starts_per_team"] = float(C.SEASON_GAMES)
    out["appearances_per_team"] = C.SEASON_GAMES * out["appearances_per_start"]
    out["wins_per_team"] = C.SEASON_GAMES / 2.0
    out["shutouts_per_team"] = C.SEASON_GAMES * out["shutouts_per_start"]
    out["minutes_per_team"] = C.SEASON_GAMES * out["min_per_start"]
    return out


# --------------------------------------------------------------------------- #
# team ratings                                                                #
# --------------------------------------------------------------------------- #
def _rating_for(stat: str, seasons: list[int], weights: list[float]) -> pd.Series:
    """Team rating for one stat: its per-game level over the league's, recency-weighted."""
    col = TEAM_SOURCE.get(stat)
    if col is None:
        return pd.Series(dtype=float)
    t = dl.load_moneypuck_teams()
    parts, ws = [], []
    for w, season in zip(weights, seasons):
        s = t[t["mp_season_year"] == season]
        if s.empty or col not in s:
            continue
        per_game = (s.set_index("team")[col] / s.set_index("team")["games_played"])
        partner = RATING_XG_PARTNER.get(stat)
        if partner and partner in s:
            xg = s.set_index("team")[partner] / s.set_index("team")["games_played"]
            per_game = 0.5 * per_game + 0.5 * xg
        parts.append(per_game / per_game.mean())
        ws.append(w)
    if not parts:
        return pd.Series(dtype=float)
    stacked = pd.concat(parts, axis=1)
    raw = (stacked * np.asarray(ws)).sum(axis=1) / np.where(
        stacked.notna(), np.asarray(ws), 0.0).sum(axis=1)
    return raw


def _suppression_rating(col: str, seasons: list[int], weights: list[float],
                        xg_partner: str | None = None) -> pd.Series:
    """How well a team SUPPRESSES `col` (a team-file "against" column). >1 = suppresses.

    The reciprocal of a rate rather than the rate itself, so it composes the same way
    the offensive ratings do: a budget is divided by nothing and multiplied by this.
    """
    t = dl.load_moneypuck_teams()
    parts, ws = [], []
    for w, season in zip(weights, seasons):
        s = t[t["mp_season_year"] == season]
        if s.empty or col not in s:
            continue
        s = s.set_index("team")
        per_game = s[col] / s["games_played"]
        if xg_partner and xg_partner in s:
            per_game = 0.5 * per_game + 0.5 * (s[xg_partner] / s["games_played"])
        parts.append(per_game.mean() / per_game)      # invert: fewer allowed => >1
        ws.append(w)
    if not parts:
        return pd.Series(dtype=float)
    stacked = pd.concat(parts, axis=1)
    return (stacked * np.asarray(ws)).sum(axis=1) / np.where(
        stacked.notna(), np.asarray(ws), 0.0).sum(axis=1)


def team_ratings(season: int | None = None, stats: list[str] | None = None,
                 shrink: float | None = None) -> pd.DataFrame:
    """Per-team, per-stat rating for the target season, shrunk toward league average.

    A rating of 1.08 for goals means "this team's offence has been 8% above league".
    How much of that carries into next season is measured per stat, not chosen -- see
    TEAM_RATING_PERSISTENCE and `rating_persistence()`. `shrink` is a global dial on
    those measurements: 1.0 believes them as measured, 0.0 flattens every team to
    league average.
    """
    season = C.TARGET_SEASON if season is None else season
    shrink = C.TEAM_RATING_SHRINK if shrink is None else shrink
    stats = list(TEAM_SOURCE) if stats is None else stats
    n = C.TEAM_STRENGTH_SEASONS
    seasons = [season - 1 - i for i in range(n)]
    weights = [0.6, 0.4, 0.2, 0.15][:n] or [1.0]
    out = {}
    for stat in stats:
        raw = _rating_for(stat, seasons, weights)
        if raw.empty:
            continue
        keep = shrink * C.TEAM_RATING_PERSISTENCE.get(
            stat, C.TEAM_RATING_PERSISTENCE_DEFAULT)
        out[stat] = 1.0 + keep * (raw - 1.0)
    return pd.DataFrame(out)


def rating_persistence(stats: list[str] | None = None) -> pd.DataFrame:
    """How much of a team rating survives into the next season -- the honest shrink.

    For each past season, regress the team's ACTUAL rating on the rating a projection
    would have had (the prior two seasons, unshrunk). The OLS slope through 1.0 is the
    fraction of a deviation that persists, i.e. exactly what TEAM_RATING_SHRINK should
    be. Run as a diagnostic; the config constant is set from its output.
    """
    stats = list(TEAM_SOURCE) if stats is None else stats
    n = C.TEAM_STRENGTH_SEASONS
    rows = []
    for stat in stats:
        xs, ys = [], []
        for season in range(C.FIRST_SEASON + n + 4, C.LAST_COMPLETED_SEASON + 1):
            prior = _rating_for(stat, [season - 1 - i for i in range(n)],
                                [0.6, 0.4, 0.2, 0.15][:n])
            actual = _rating_for(stat, [season], [1.0])
            both = pd.concat([prior.rename("x"), actual.rename("y")], axis=1).dropna()
            if len(both) < 20:
                continue
            xs.extend((both["x"] - 1.0).tolist())
            ys.extend((both["y"] - 1.0).tolist())
        if len(xs) < 50:
            continue
        x, y = np.asarray(xs), np.asarray(ys)
        slope = float((x * y).sum() / (x * x).sum())
        rows.append({"stat": stat, "n": len(x), "persistence": round(slope, 3),
                     "corr": round(float(np.corrcoef(x, y)[0, 1]), 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the budgets themselves                                                      #
# --------------------------------------------------------------------------- #
def team_budgets(season: int | None = None, teams: list[str] | None = None,
                 claims: pd.DataFrame | None = None, shrink: float | None = None,
                 claim_weight: float | None = None, games: int | None = None
                 ) -> pd.DataFrame:
    """One row per team: how much of each quantity there is to allocate this season.

    `claims`, if given, is the per-team sum of the players' own unconstrained
    projections. It is blended with the historical rating so that a team whose roster
    changed carries a budget its new roster justifies -- the rating knows about last
    year's team, the claims know about this year's. The result is renormalised so the
    LEAGUE total is exactly the measured league level, which is the part that has to
    hold no matter how the shares are argued over.
    """
    season = C.TARGET_SEASON if season is None else season
    games = int(C.SEASON_GAMES if games is None else games)
    claim_weight = C.TEAM_CLAIM_WEIGHT if claim_weight is None else claim_weight
    lvl = league_level()
    if teams is None:
        teams = sorted(dl.load_schedule(season)["team"].unique())
    idx = pd.Index(teams, name="team")
    ratings = team_ratings(season, shrink=shrink).reindex(idx)

    out = pd.DataFrame(index=idx)
    out["games"] = games
    # Ice time and the number of skaters dressed are the clock and the rule book, not
    # the team: every team gets exactly the same budget for both.
    out["toi_min"] = lvl["toi_min"] * games
    out["skater_games"] = lvl["dressed"] * games

    scaled = ["goals", "ixg", "shots", "blocks", "hits", "pim", "faceoffs_won"]
    for stat in scaled:
        if stat not in lvl:
            continue
        rating = ratings[stat] if stat in ratings else pd.Series(1.0, index=idx)
        rating = rating.fillna(1.0)
        if claims is not None and stat in claims:
            c = claims[stat].reindex(idx).fillna(0.0)
            if c.sum() > 0:
                claim_rating = c / c.mean()
                # Geometric blend: a rating is multiplicative, and the geometric mean
                # of two multipliers is the one that does not favour the larger.
                rating = (rating.clip(lower=0.2) ** (1.0 - claim_weight)) * \
                         (claim_rating.clip(lower=0.2) ** claim_weight)
        raw = lvl[stat] * games * rating
        out[stat] = raw * (lvl[stat] * games * len(idx) / raw.sum())

    # Assists follow each team's OWN goals, at the league ratio. This is what makes
    # points internally consistent: team points = team goals x (1 + 0.935 + 0.751).
    out["primaryAssists"] = out["goals"] * lvl["primary_per_goal"]
    out["secondaryAssists"] = out["goals"] * lvl["secondary_per_goal"]
    out["assists"] = out["primaryAssists"] + out["secondaryAssists"]
    out["points"] = out["goals"] + out["assists"]

    # Special teams: a team's PP point budget tracks its power-play goal scoring, so
    # scale by the goal rating rather than inventing a separate one.
    for stat in ("pp_points", "sh_points"):
        if stat not in lvl:
            continue
        rating = ratings["goals"].fillna(1.0) if "goals" in ratings else pd.Series(1.0, index=idx)
        raw = lvl[stat] * games * rating
        out[stat] = raw * (lvl[stat] * games * len(idx) / raw.sum())
    # Two different quantities, and conflating them would misprice every power-play
    # rate by a factor of five. `pp_toi_min` is how long the TEAM spends on the power
    # play; `pp_toi_min_skaters` is the sum of what its individual skaters accumulate,
    # which is five times larger because five of them are on the ice at once. Player
    # claims settle against the second.
    for key, on_ice in (("pp_toi_min", 5.0), ("sh_toi_min", 4.0)):
        if key in lvl:
            out[key] = lvl[key] * games
            out[f"{key}_skaters"] = out[key] * on_ice
    return out


def goalie_budgets(season: int | None = None, teams: list[str] | None = None,
                   games: int | None = None, claims: pd.DataFrame | None = None,
                   shrink: float | None = None, claim_weight: float | None = None
                   ) -> pd.DataFrame:
    """Per-team goalie budgets: starts, appearances, minutes, shots/goals against, wins.

    Three of these are accounting identities rather than projections -- 84 starts, half
    the games won, and 60.1 minutes of goaltending per team-game -- and the old model
    violated all three (114 appearances per team, 1643 league wins, and a "GAA" that was
    goals divided by appearances). Wins are allocated by team strength and then
    renormalised to the league's 1344, so a projection can be optimistic about a team
    only by being pessimistic about its opponents.

    Shots against and goals against are deliberately given SEPARATE ratings. Shots
    against come from how well a team suppresses shots (its shotsOnGoalAgainst rate);
    goals against come from how well it suppresses goals. The gap between the two IS the
    team's goaltending, and it is real: implied team save percentage ranged from .872 to
    .908 in 2025-26. Holding the two apart is what lets a team's save percentage be a
    consequence of the goalies projected into it rather than an assumption.

    `claims` (per-team sums of the goalies' own unconstrained goals-against claims) is
    blended into the goals-against rating for the same reason it is on the skater side:
    the rating knows about last year's goalies, the claims know about this year's.
    """
    season = C.TARGET_SEASON if season is None else season
    games = int(C.SEASON_GAMES if games is None else games)
    shrink = C.TEAM_RATING_SHRINK if shrink is None else shrink
    claim_weight = C.TEAM_CLAIM_WEIGHT if claim_weight is None else claim_weight
    lvl = goalie_league_level()
    if teams is None:
        teams = sorted(dl.load_schedule(season)["team"].unique())
    idx = pd.Index(teams, name="team")

    import context as ctx
    r = ctx.team_defense_ratings().set_index("team").reindex(idx)
    off = r["off_rating"].fillna(1.0)
    # Same measured persistence as the skater budgets: a team's scoring rating is worth
    # about 0.70 of itself a season later, so believing it in full would over-separate
    # the good teams from the bad ones.
    keep = shrink * C.TEAM_RATING_PERSISTENCE["goals"]
    n = C.TEAM_STRENGTH_SEASONS
    seasons = [season - 1 - i for i in range(n)]
    weights = [0.6, 0.4, 0.2, 0.15][:n] or [1.0]

    def shrunk(s: pd.Series) -> pd.Series:
        return 1.0 + keep * (s.reindex(idx).fillna(1.0) - 1.0)

    off = shrunk(off)
    dfn = shrunk(r["def_rating"])
    shot_sup = shrunk(_suppression_rating("shotsOnGoalAgainst", seasons, weights,
                                          xg_partner="xOnGoalAgainst"))
    goal_sup = shrunk(_suppression_rating("goalsAgainst", seasons, weights,
                                          xg_partner="xGoalsAgainst"))

    out = pd.DataFrame(index=idx)
    out["games"] = games
    out["starts"] = float(games)
    out["appearances"] = games * lvl["appearances_per_start"]
    # One goalie on the ice at a time: the minutes budget is the clock, so it is the same
    # for every team, exactly like the skaters' 297.4.
    out["minutes"] = games * lvl["min_per_start"]

    def conserve(raw: pd.Series, per_team: float) -> pd.Series:
        """Scale a rating-tilted budget so the LEAGUE total is the measured level."""
        total = per_team * len(idx)
        return raw * (total / raw.sum()) if raw.sum() > 0 else pd.Series(per_team, index=idx)

    out["shots_against"] = conserve(lvl["sa_per_start"] * games / shot_sup,
                                    lvl["sa_per_start"] * games)

    ga_rating = 1.0 / goal_sup.clip(lower=0.2)      # >1 = allows more than average
    if claims is not None and {"goals_against", "shots_against"} <= set(claims):
        # A RATE, not a total. The goalies a team has listed claim a number of goals
        # against that depends mostly on how many goalies are listed -- a team with one
        # goalie on its published roster claims about half a season of them -- so a rating
        # built on the total handed that team an absurd save percentage (.931 for the
        # Penguins on the first run). Goals allowed PER SHOT is the part of the claim that
        # is actually about the goaltending.
        ga = claims["goals_against"].reindex(idx)
        sa = claims["shots_against"].reindex(idx).replace(0.0, np.nan)
        lg = float(ga.sum() / sa.sum()) if sa.sum() > 0 else 0.0
        if lg > 0:
            rate = (ga / sa / lg).fillna(1.0).clip(lower=0.6, upper=1.6)
            ga_rating = (ga_rating ** (1.0 - claim_weight)) * (rate ** claim_weight)
    out["goals_against"] = conserve(lvl["ga_per_start"] * games * ga_rating,
                                     lvl["ga_per_start"] * games)

    # Win share from team quality, normalised so the league wins exactly half its games.
    league_wins = games / 2.0 * len(idx)
    quality = (off * dfn) ** 1.6      # a goal ratio converts to a win ratio super-linearly
    w = quality * (league_wins / quality.sum())
    w = w.clip(upper=games * 0.78)    # nobody has ever won 78% of an NHL season
    out["wins"] = w * (league_wins / w.sum())

    out["shutouts"] = conserve(lvl["shutouts_per_team"] * (goal_sup ** 2),
                                lvl["shutouts_per_team"])
    # Implied, and worth carrying: this is the save percentage the team's budgets say its
    # goaltending has to post, and a reader should be able to see it next to the goalies.
    out["sv_pct"] = 1.0 - out["goals_against"] / out["shots_against"]
    out["gaa"] = out["goals_against"] * 60.0 / out["minutes"]
    out["min_per_appearance"] = lvl["min_per_appearance"]
    return out


if __name__ == "__main__":
    lvl = league_level()
    print("League production per team-game (trend-weighted):")
    for k in sorted(lvl):
        print(f"  {k:22s} {lvl[k]:8.3f}")
    print("\nGoalie league level / per-team budgets:")
    for k, v in sorted(goalie_league_level().items()):
        print(f"  {k:22s} {v:8.3f}")
    print("\nHow much of a team rating persists (sets TEAM_RATING_SHRINK):")
    print(rating_persistence().to_string(index=False))
    tb = team_budgets()
    print(f"\nTeam budgets ({len(tb)} teams, {C.SEASON_GAMES} games):")
    show = ["toi_min", "goals", "assists", "points", "shots", "pp_points", "blocks", "hits"]
    print(tb[show].sort_values("goals", ascending=False).head(6).round(1).to_string())
    print("  ...")
    print(tb[show].sort_values("goals").head(3).round(1).to_string())
    gb = goalie_budgets()
    print(f"\nGoalie budgets: league starts {gb['starts'].sum():.0f}, "
          f"wins {gb['wins'].sum():.0f}, appearances {gb['appearances'].sum():.0f}, "
          f"minutes/team-game {gb['minutes'].mean() / C.SEASON_GAMES:.2f}")
    print(gb[["starts", "appearances", "minutes", "wins", "shutouts", "shots_against",
              "goals_against", "sv_pct", "gaa"]]
          .sort_values("wins", ascending=False).head(5).round(3).to_string())
    print("  implied team save percentage spans "
          f"{gb['sv_pct'].min():.4f} - {gb['sv_pct'].max():.4f} (goaltending, not shots)")
