"""The season in progress: what is already banked, and how much is left.

A preseason projection is a statement about 84 unplayed games. Once the season starts it
stops being that, and a tool that keeps serving it is wrong in a way that gets worse every
night. So from opening night the projection becomes two pieces:

    projection = what he has already done  +  what he does with the games that are left

The first piece is not a projection at all, it is a fact, and this module fetches it. The
second is the ordinary model run against a smaller season -- a team's budget is its
per-game budget times the games it has LEFT, which is the same arithmetic as a full season
with a smaller number in it. Nothing about the settlement changes.

Three things the season in progress tells the model, in decreasing order of how fast they
should move it:

  1. ICE TIME PER GAME (C.IN_SEASON_TOI_K = 8 player-games). The fastest signal in the
     sport and the one a reader most wants noticed: a promotion to the first line is
     visible in three games, and the model's own history cannot see it at all.
  2. AVAILABILITY (C.IN_SEASON_GP_K = 20 team-games). A player who has missed 12 of his
     team's 20 games is a different bet for the rest of the season than his history says.
  3. PRODUCTION RATES. These enter through the ordinary multi-season rate blend, where the
     live season is weighted by the ice time it contains -- so after five games it moves a
     player's per-60 rates by about 5%, and by March it owns them. That is the correct
     speed for a rate and it needs no extra dial: scoring rates are the part of a season
     most contaminated by luck, and chasing them is the single easiest way to make an
     in-season projection worse than the preseason one it replaced.

WHY THE ACTUALS ARE NOT JUST "THE PROJECTION SO FAR". The banked half carries no
uncertainty, so the prediction bands narrow on their own as the season runs: by February a
player's remaining 20 games can move his point total far less than his remaining 84 could
in September. The bands are drawn on the rest-of-season piece and then shifted up by what
is banked, which is what makes that happen without a separate rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as C
import data_layer as dl

# MoneyPuck's own situation labels for the two special-teams clocks, matching
# project_skaters.SIT_SOURCES. Kept here as data rather than imported so this module has no
# dependency on the projection modules (they depend on it).
SIT_SOURCES = (("pp", "5on4"), ("sh", "4on5"))


# --------------------------------------------------------------------------- #
# where the season stands                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeasonState:
    """How much of the target season is history, per team.

    `played` is deliberately measured from the STATS file rather than from the schedule,
    because it has to agree with the actuals: if the schedule knows about last night's game
    and MoneyPuck's file does not yet, counting that game as played would leave its
    production in neither half of the projection.
    """
    season: int
    as_of: pd.Timestamp
    scheduled: pd.Series          # games on the schedule, per team
    played: pd.Series             # games the banked actuals cover, per team
    started: bool                 # has a game been played at all
    live: bool                    # enough of the season to use it (IN_SEASON_MIN_TEAM_GAMES)
    source: str = ""              # where `played` came from, for the app to show

    @property
    def remaining(self) -> pd.Series:
        return (self.scheduled - self.played).clip(lower=0.0)

    @property
    def team_games_played(self) -> float:
        """League mean team-games played -- the sample size the shrink constants use."""
        return float(self.played.mean()) if len(self.played) else 0.0

    @property
    def frac_played(self) -> float:
        total = float(self.scheduled.sum())
        return float(self.played.sum()) / total if total > 0 else 0.0

    @property
    def frac_remaining(self) -> float:
        return 1.0 - self.frac_played

    def label(self) -> str:
        """One line for the app: what window this projection actually covers."""
        if not self.started:
            return f"preseason — the full {int(round(self.scheduled.mean()))}-game season"
        if not self.live:
            n = self.team_games_played
            return (f"{n:.0f} team-games played, below the {C.IN_SEASON_MIN_TEAM_GAMES}-game "
                    "threshold — still the preseason projection")
        return (f"{self.frac_played:.0%} of the season played "
                f"({self.played.mean():.1f} games a team) — "
                f"banked totals plus {self.remaining.mean():.1f} games to come")


def _schedule_counts(season: int, refresh: bool = False) -> tuple[pd.Series, pd.Series]:
    """(games scheduled, games with a final result) per team, from one load."""
    sched = dl.load_schedule(season, refresh=refresh)
    if sched.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    scheduled = sched.groupby("team").size().astype(float)
    if "gameState" not in sched:
        return scheduled, pd.Series(0.0, index=scheduled.index)
    done = sched[sched["gameState"].isin(["FINAL", "OFF"])]
    return scheduled, done.groupby("team").size().astype(float).reindex(
        scheduled.index).fillna(0.0)


def season_state(season: int | None = None, refresh: bool = False) -> SeasonState:
    """Where the season stands right now. Cheap enough to call from anywhere."""
    season = C.TARGET_SEASON if season is None else season
    scheduled, done = _schedule_counts(season, refresh=refresh)
    teams = dl.load_live_teams(season, refresh=refresh)
    source = "MoneyPuck team file"
    if not teams.empty and "games_played" in teams:
        played = teams.set_index("team")["games_played"].astype(float)
        played = played.reindex(scheduled.index).fillna(0.0)
    else:
        played, source = done, "schedule results"
        if played.empty:
            played = pd.Series(0.0, index=scheduled.index)
            source = "nothing played"
    started = bool(played.sum() > 0)
    live = bool(started and played.mean() >= C.IN_SEASON_MIN_TEAM_GAMES)
    return SeasonState(season=season, as_of=pd.Timestamp.now().normalize(),
                       scheduled=scheduled, played=played, started=started, live=live,
                       source=source)


# --------------------------------------------------------------------------- #
# what is already banked: skaters                                             #
# --------------------------------------------------------------------------- #
def skater_actuals(season: int | None = None, refresh: bool = False) -> pd.DataFrame:
    """Season-to-date totals per skater, in the projection's own stat vocabulary.

    One row per player, columns `act_*`, so a projection can be written as
    `proj_x = act_x + ros_x` and every page in the app can show both halves. A traded
    player has one MoneyPuck row per team, so the rows are summed by player id -- his
    banked production belongs to him wherever he earned it, and the team he plays for now
    is whatever the roster says.
    """
    season = C.TARGET_SEASON if season is None else season
    # Imported here, not at module scope: project_skaters imports this module, and the stat
    # vocabulary belongs to the model rather than being duplicated into the data layer.
    from project_skaters import RAW_COLS

    allsit = dl.load_live_skaters(season, refresh=refresh, situation="all")
    if allsit is None or allsit.empty:
        return pd.DataFrame()

    cols = {"act_gp": allsit["games_played"].astype(float),
            "act_toi": allsit["icetime"].astype(float) / 60.0}
    for stat, col in RAW_COLS.items():
        cols[f"act_{stat}"] = allsit[col].astype(float)
    act = pd.DataFrame(cols)
    act["playerId"] = allsit["playerId"].astype(int).to_numpy()
    act["live_team"] = allsit["team"].astype(str).to_numpy()
    act = act.groupby("playerId", as_index=False).agg(
        {**{c: "sum" for c in cols}, "live_team": "last"})

    # Special teams, each on its own clock. Points, not goals plus assists separately:
    # the model projects pp_points and sh_points as single quantities.
    for prefix, situation in SIT_SOURCES:
        sit = dl.load_live_skaters(season, refresh=False, situation=situation)
        if sit is None or sit.empty:
            act[f"act_{prefix}_points"] = 0.0
            act[f"act_{prefix}_toi"] = 0.0
            continue
        s = pd.DataFrame({
            "playerId": sit["playerId"].astype(int).to_numpy(),
            f"act_{prefix}_points": (sit["I_F_goals"] + sit["I_F_primaryAssists"]
                                     + sit["I_F_secondaryAssists"]).astype(float).to_numpy(),
            f"act_{prefix}_toi": (sit["icetime"].astype(float) / 60.0).to_numpy(),
        }).groupby("playerId", as_index=False).sum()
        act = act.merge(s, on="playerId", how="left")
        act[f"act_{prefix}_points"] = act[f"act_{prefix}_points"].fillna(0.0)
        act[f"act_{prefix}_toi"] = act[f"act_{prefix}_toi"].fillna(0.0)

    act["act_assists"] = act["act_primaryAssists"] + act["act_secondaryAssists"]
    act["act_points"] = act["act_goals"] + act["act_assists"]
    act["act_toi_per_gp"] = np.where(act["act_gp"] > 0,
                                     act["act_toi"] / act["act_gp"].replace(0, np.nan), 0.0)
    return act


def goalie_actuals(season: int | None = None, refresh: bool = False) -> pd.DataFrame:
    """Season-to-date totals per goalie, in the projection's own vocabulary.

    A traded goalie's row in this feed lists every team he has played for and the split is
    not in the source data, so the totals are summed by player id for the same reason the
    skaters' are.
    """
    season = C.TARGET_SEASON if season is None else season
    g = dl.load_live_goalies(season, refresh=refresh)
    if g is None or g.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "playerId": g["playerId"].astype(int).to_numpy(),
        "act_starts": g["gamesStarted"].astype(float).to_numpy(),
        "act_gp": g["gamesPlayed"].astype(float).to_numpy(),
        "act_wins": g["wins"].astype(float).to_numpy(),
        "act_losses": g["losses"].astype(float).to_numpy(),
        "act_otl": g["otLosses"].astype(float).to_numpy(),
        "act_shutouts": g["shutouts"].astype(float).to_numpy(),
        "act_saves": g["saves"].astype(float).to_numpy(),
        "act_shots_against": g["shotsAgainst"].astype(float).to_numpy(),
        "act_goals_against": g["goalsAgainst"].astype(float).to_numpy(),
        # The feed's timeOnIce is SECONDS. Reading it as minutes is the bug that made the
        # old GAA divide by the wrong quantity entirely; it is worth restating here.
        "act_minutes": g["timeOnIce"].astype(float).to_numpy() / 60.0,
    }).groupby("playerId", as_index=False).sum()
    out["act_relief"] = (out["act_gp"] - out["act_starts"]).clip(lower=0.0)
    sa = out["act_shots_against"].to_numpy()
    out["act_save_pct"] = np.where(sa > 0, 1.0 - out["act_goals_against"] / np.maximum(sa, 1e-9),
                                   np.nan)
    return out


# --------------------------------------------------------------------------- #
# blending the season in progress into a projection                           #
# --------------------------------------------------------------------------- #
def shrink_weight(n: float | np.ndarray, k: float) -> float | np.ndarray:
    """`n / (n + k)`: the weight a sample of size n earns against a prior.

    The one piece of arithmetic every in-season blend in this system shares, written once
    so the three constants that feed it (C.IN_SEASON_TEAM_K, _GP_K, _TOI_K) can be argued
    about on their own terms. Each was measured, not chosen -- see config.
    """
    n = np.asarray(n, dtype=float)
    w = np.where(n > 0, n / (n + k), 0.0)
    return float(w) if w.ndim == 0 else w


def live_team_rates(season: int | None = None, stats: list[str] | None = None
                    ) -> pd.DataFrame:
    """Per-team, per-game rates for the season in progress, relative to the league.

    Returns a frame indexed by team with one column per stat, each a rating on the same
    scale as `budgets.team_ratings` (1.08 = 8% above this season's league average) plus a
    `games` column so the caller knows how much to believe it. Empty before the season
    starts, which is what makes the budget blend a no-op then.
    """
    from budgets import TEAM_SOURCE

    season = C.TARGET_SEASON if season is None else season
    stats = list(TEAM_SOURCE) if stats is None else stats
    t = dl.load_live_teams(season)
    if t is None or t.empty or "games_played" not in t:
        return pd.DataFrame()
    t = t.set_index("team")
    gp = t["games_played"].astype(float)
    out = pd.DataFrame({"games": gp})
    for stat in stats:
        col = TEAM_SOURCE.get(stat)
        if col is None or col not in t:
            continue
        per_game = t[col].astype(float) / gp.replace(0, np.nan)
        mean = float(per_game.mean())
        out[stat] = (per_game / mean) if mean > 0 else 1.0
    return out


if __name__ == "__main__":
    st = season_state()
    print(f"Season {st.season}-{st.season + 1} as of {st.as_of.date()}")
    print(f"  {st.label()}")
    print(f"  played from: {st.source}; started={st.started} live={st.live}")
    if len(st.played):
        print(f"  scheduled/team {st.scheduled.mean():.1f}, played/team "
              f"{st.played.mean():.2f}, remaining/team {st.remaining.mean():.2f}")
    sk = skater_actuals()
    print(f"\nskater actuals: {len(sk)} rows")
    if not sk.empty:
        print(sk.nlargest(5, "act_points")[
            ["playerId", "live_team", "act_gp", "act_toi_per_gp", "act_goals",
             "act_assists", "act_points"]].to_string(index=False))
    g = goalie_actuals()
    print(f"\ngoalie actuals: {len(g)} rows")
    if not g.empty:
        print(g.nlargest(5, "act_starts")[
            ["playerId", "act_starts", "act_wins", "act_save_pct"]].to_string(index=False))
    tr = live_team_rates()
    print(f"\nlive team rates: {len(tr)} teams")
    if not tr.empty:
        print(tr.round(3).head().to_string())
