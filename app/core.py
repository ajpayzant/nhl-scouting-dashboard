"""Shared plumbing for the Streamlit app: projections, the live scenario, formatting.

Two rules shape this file.

First, a projection is expensive (the engine rebuilds age curves and five seasons of
rates), so it is cached on the CONTENT of the scenario rather than recomputed per click,
and the skater and goalie sides are cached separately -- editing a goalie's starts has no
business re-running 900 skaters.

Second, the scenario on disk is the truth. Every edit is written to `scenarios/working.json`
immediately, so closing the browser loses nothing and the file can be read, diffed or
mailed on its own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C           # noqa: E402
import data_layer as dl      # noqa: E402
import live as lv            # noqa: E402
import overrides as ov       # noqa: E402
import project_goalies as pg  # noqa: E402
import project_skaters as ps  # noqa: E402

SEASON_LABEL = f"{C.TARGET_SEASON}-{str(C.TARGET_SEASON + 1)[-2:]}"

# Is this app being used by more than one person at once?
#
# It matters because the scenario is the app's only mutable state. Run locally it belongs
# on disk: `scenarios/working.json` is written after every edit, so closing the browser
# loses nothing. Deployed, that same file would be SHARED -- Streamlit gives every visitor
# a session but only one container and one filesystem, so one person's lock on McDavid
# would rewrite what everybody else is reading, and a redeploy would wipe the lot.
#
# So when the app is shared, the working scenario lives in session memory instead: each
# visitor gets his own, starting from the model, and keeps it by saving it to the library
# (`library.py`) or downloading the JSON. Streamlit Community Cloud checks out the repo
# under /mount/src, which is the detection; `multiuser` in secrets overrides either way.
_ON_CLOUD = "/mount/src" in ROOT.as_posix()


def _secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:                                          # noqa: BLE001
        return default


MULTIUSER = bool(_secret("multiuser", _ON_CLOUD))


# --------------------------------------------------------------------------- #
# the live scenario                                                           #
# --------------------------------------------------------------------------- #
def scenario() -> ov.Scenario:
    if "sc" not in st.session_state:
        # Shared: this visitor's own blank scenario. Local: whatever was left on disk.
        st.session_state.sc = ov.Scenario(name="working") if MULTIUSER else ov.live()
    return st.session_state.sc


def commit(sc: ov.Scenario, toast: str | None = None) -> None:
    """Adopt an edited scenario, persist it if it is ours to persist, and redraw."""
    st.session_state.sc = sc
    if not MULTIUSER:
        ov.save_live(sc)
    if toast:
        st.toast(toast, icon="✅")
    st.rerun()


def edit_player(pid, **fields) -> None:
    commit(scenario().set_player(pid, **fields))


def edit_goalie(pid, **fields) -> None:
    commit(scenario().set_goalie(pid, **fields))


def edit_team(team, **fields) -> None:
    commit(scenario().set_team(team, **fields))


# --------------------------------------------------------------------------- #
# the season in progress                                                      #
# --------------------------------------------------------------------------- #
# Once the season starts, a projection is only as current as the box scores behind it, so
# the app pulls last night's numbers itself rather than waiting for the weekly data commit.
# Hourly is the right cadence: it is three HTTP requests and about a second, and nothing in
# a season-long projection moves faster than that. The schedule is deliberately NOT part of
# it -- that is 32 requests and the only thing in it that changes is results, which the
# stats file already carries.
LIVE_TTL = 3600


@st.cache_data(show_spinner="Checking the season so far ...", ttl=LIVE_TTL)
def _live_pull() -> str:
    """Refresh the live season at most once an hour; return a token for the cache keys.

    The token is the point. Projections are cached on the CONTENT of the scenario, so
    without something in the key that moves when the data moves, a container that has been
    up since Tuesday would keep serving Tuesday's numbers forever.
    """
    try:
        counts = dl.refresh_live(schedule=False)
    except Exception:                                          # noqa: BLE001
        counts = {}                    # a failed pull is not a broken app: use what we have
    state = lv.season_state()
    return (f"{state.as_of.date()}|{float(state.played.sum()):.0f}|"
            f"{counts.get('skater_rows', 0)}")


def live_token() -> str:
    return _live_pull()


@st.cache_data(show_spinner=False, ttl=LIVE_TTL)
def season_state(token: str):
    """Where the season stands, for any page that wants to say so. Cached on the token."""
    return lv.season_state()


def window_label() -> str:
    """One line: what window the projections on screen actually cover."""
    return season_state(live_token()).label()


# --------------------------------------------------------------------------- #
# projections                                                                 #
# --------------------------------------------------------------------------- #
# The cache key is the scenario itself, reduced to the parts that can change the
# answer -- which is why these take a JSON string rather than a Scenario: Streamlit
# hashes the arguments, so the argument has to BE the identity of the request.
def _skater_key(sc: ov.Scenario) -> str:
    return json.dumps({"players": sc.players, "teams": sc.teams, "league": sc.league},
                      sort_keys=True)


def _goalie_key(sc: ov.Scenario) -> str:
    return json.dumps({"goalies": sc.goalies, "teams": sc.teams, "league": sc.league},
                      sort_keys=True)


def _from_key(key: str) -> ov.Scenario:
    d = json.loads(key)
    return ov.Scenario(name="working", players=d.get("players", {}),
                       goalies=d.get("goalies", {}), teams=d.get("teams", {}),
                       league=d.get("league", {}))


# `stamp` is never read: it is in the signature so that new box scores invalidate the cache.
@st.cache_data(show_spinner="Projecting skaters ...", max_entries=6)
def _skaters(key: str, stamp: str = ""):
    return ps.project_skaters(_from_key(key), with_budgets=True)


@st.cache_data(show_spinner="Projecting goalies ...", max_entries=6)
def _goalies(key: str, stamp: str = ""):
    return pg.project_goalies(_from_key(key), with_budgets=True)


def skaters(sc: ov.Scenario | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(skaters, team budgets) for a scenario. Cached; safe to call from every view."""
    return _skaters(_skater_key(sc or scenario()), live_token())


def goalies(sc: ov.Scenario | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _goalies(_goalie_key(sc or scenario()), live_token())


def baseline_skaters() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The pure model, for showing what an edit changed."""
    return _skaters(_skater_key(ov.Scenario()), live_token())


def baseline_goalies() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _goalies(_goalie_key(ov.Scenario()), live_token())


# The counting stats a reader compares season to season, and the clock each one is
# measured against. Keeping the pairing here means the history tables and the projection
# agree on what "per 60" means -- even-strength-clock stats are per 60 minutes of TOTAL
# ice time (which is what the model regresses), and power-play points are per 60 minutes
# OF POWER PLAY, because that is the skill that survives a change of team.
HISTORY_STATS = ["goals", "assists", "points", "shots", "ixg", "blocks", "hits", "pim",
                 "faceoffs_won"]
HISTORY_SIT_STATS = {"pp_points": "pp", "sh_points": "sh"}


@st.cache_data(show_spinner="Loading history ...")
def skater_history() -> pd.DataFrame:
    """One row per player-season: totals, ice time, and the per-60 rates behind them."""
    sk = dl.load_moneypuck_skaters()
    keep = ["playerId", "name", "mp_season_year", "team", "position", "games_played",
            "icetime", "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
            "I_F_points", "I_F_shotsOnGoal", "I_F_xGoals", "shotsBlockedByPlayer",
            "I_F_hits", "penalityMinutes", "faceoffsWon"]
    df = sk[[c for c in keep if c in sk.columns]].copy()
    df["toi_min"] = df["icetime"] / 60.0
    df["toi_per_gp"] = df["toi_min"] / df["games_played"].replace(0, pd.NA)
    df["assists"] = df["I_F_primaryAssists"] + df["I_F_secondaryAssists"]
    df = df.rename(columns={"mp_season_year": "season", "games_played": "gp",
                            "I_F_goals": "goals", "I_F_points": "points",
                            "I_F_shotsOnGoal": "shots", "I_F_xGoals": "ixg",
                            "shotsBlockedByPlayer": "blocks", "I_F_hits": "hits",
                            "penalityMinutes": "pim", "faceoffsWon": "faceoffs_won"})

    for stat, prefix in HISTORY_SIT_STATS.items():
        sit = dl.load_moneypuck_situation({"pp": "5on4", "sh": "4on5"}[prefix])
        part = pd.DataFrame({
            "playerId": sit["playerId"].to_numpy(),
            "season": sit["mp_season_year"].to_numpy(),
            stat: (sit["I_F_goals"] + sit["I_F_primaryAssists"]
                   + sit["I_F_secondaryAssists"]).to_numpy(),
            f"{prefix}_toi_min": sit["icetime"].to_numpy() / 60.0,
        })
        df = df.merge(part, on=["playerId", "season"], how="left")
        df[stat] = df[stat].fillna(0.0)
        df[f"{prefix}_toi_min"] = df[f"{prefix}_toi_min"].fillna(0.0)
        df[f"{prefix}_toi_per_gp"] = df[f"{prefix}_toi_min"] / df["gp"].replace(0, pd.NA)

    toi = df["toi_min"].replace(0, pd.NA)
    for stat in HISTORY_STATS:
        df[f"rate_{stat}"] = df[stat] * 60.0 / toi
    for stat, prefix in HISTORY_SIT_STATS.items():
        df[f"rate_{stat}"] = df[stat] * 60.0 / df[f"{prefix}_toi_min"].replace(0, pd.NA)
    return df


@st.cache_data(show_spinner="Loading history ...")
def goalie_history() -> pd.DataFrame:
    """One row per goalie-season: totals plus the four rates the projection is built on."""
    gs = dl.load_nhl_goalie_summary()
    df = gs.copy()
    df["minutes"] = df["timeOnIce"] / 60.0
    df["save_pct"] = 1.0 - df["goalsAgainst"] / df["shotsAgainst"].replace(0, pd.NA)
    df["gaa"] = df["goalsAgainst"] * 60.0 / df["minutes"].replace(0, pd.NA)
    df = df.rename(columns={"mp_season_year": "season", "goalieFullName": "name",
                            "teamAbbrevs": "team", "gamesPlayed": "gp",
                            "gamesStarted": "starts", "otLosses": "otl",
                            "shotsAgainst": "shots_against",
                            "goalsAgainst": "goals_against", "shutouts": "shutouts"})
    starts = df["starts"].replace(0, pd.NA)
    df["rate_ga_per_shot"] = 1.0 - df["save_pct"]
    df["rate_sa_per_60"] = df["shots_against"] * 60.0 / df["minutes"].replace(0, pd.NA)
    df["rate_so_per_start"] = df["shutouts"] / starts
    # Appearances that were not starts: the relief work a backup picks up.
    df["rate_relief_per_start"] = (df["gp"] - df["starts"]).clip(lower=0) / starts
    return df


# Franchises whose three-letter code in the source changes part-way through the history,
# either because MoneyPuck renamed it (LAK, NJD, SJS, TBL all switched from the dotted
# form in 2021) or because the franchise moved (Atlanta -> Winnipeg in 2011, Arizona ->
# Utah in 2024). Without this a team page shows five seasons of Los Angeles history and
# none at all before the move, which reads as a data gap rather than a rename.
FRANCHISE_ALIASES = {
    "LAK": ["L.A"], "NJD": ["N.J"], "SJS": ["S.J"], "TBL": ["T.B"],
    "UTA": ["ARI", "PHX"], "WPG": ["ATL"],
}


@st.cache_data(show_spinner="Loading team history ...")
def team_history() -> pd.DataFrame:
    """One row per team-season since 2008, with the rates a budget is judged against.

    `team` is the CURRENT franchise code so a page can select on it; `as_named` keeps the
    code the season was actually played under, because a Utah page that silently labels
    2019 as Utah is lying about where those games were played.
    """
    t = dl.load_moneypuck_teams()
    # The source carries its own `season` and a duplicate `team.1`; drop both first or the
    # rename below produces two columns called `season` and every sort on it fails.
    t = t.drop(columns=[c for c in ("season", "team.1") if c in t.columns])
    df = t.rename(columns={"mp_season_year": "season", "games_played": "gp"}).copy()
    df["as_named"] = df["team"]
    back = {old: new for new, olds in FRANCHISE_ALIASES.items() for old in olds}
    df["team"] = df["team"].map(lambda x: back.get(x, x))

    gp = df["gp"].replace(0, pd.NA)
    for src, name in (("goalsFor", "gf"), ("goalsAgainst", "ga"),
                      ("xGoalsFor", "xgf"), ("xGoalsAgainst", "xga"),
                      ("shotsOnGoalFor", "sf"), ("shotsOnGoalAgainst", "sa"),
                      ("highDangerShotsFor", "hdf"), ("highDangerShotsAgainst", "hda"),
                      ("hitsFor", "hits"), ("penalityMinutesFor", "pim"),
                      ("takeawaysFor", "takeaways"), ("giveawaysFor", "giveaways")):
        df[name] = df[src]
        df[f"{name}_pg"] = df[src] / gp
    df["gdiff_pg"] = df["gf_pg"] - df["ga_pg"]
    df["xgdiff_pg"] = df["xgf_pg"] - df["xga_pg"]
    df["shoot_pct"] = df["goalsFor"] / df["shotsOnGoalFor"].replace(0, pd.NA)
    df["save_pct"] = 1.0 - df["goalsAgainst"] / df["shotsOnGoalAgainst"].replace(0, pd.NA)
    # PDO: shooting plus saving. It has almost no year-to-year persistence, which is
    # exactly why it belongs on this page -- a team well above 1.000 outscored its own
    # chances and is the one whose repeat a projection should doubt.
    df["pdo"] = df["shoot_pct"] + df["save_pct"]
    fo = (df["faceOffsWonFor"] + df["faceOffsWonAgainst"]).replace(0, pd.NA)
    df["fow_pct"] = df["faceOffsWonFor"] / fo
    # The source's own xGoalsPercentage / corsiPercentage / fenwickPercentage are rounded
    # to two decimals, which is coarse enough to hide the difference between a 51.4% team
    # and a 50.5% one. They are exact ratios of columns that are already here, so compute
    # them rather than display someone else's rounding as if it were precision.
    for name, fcol, acol in (
            ("xg_share", "xGoalsFor", "xGoalsAgainst"),
            ("corsi", "shotAttemptsFor", "shotAttemptsAgainst"),
            ("fenwick", "unblockedShotAttemptsFor", "unblockedShotAttemptsAgainst")):
        if fcol in df.columns and acol in df.columns:
            df[name] = df[fcol] / (df[fcol] + df[acol]).replace(0, pd.NA)
    keep = ["season", "team", "as_named", "gp", "gf", "ga", "gf_pg", "ga_pg", "gdiff_pg",
            "xgf", "xga", "xgf_pg", "xga_pg", "xgdiff_pg", "sf_pg", "sa_pg", "shoot_pct",
            "save_pct", "pdo", "hdf_pg", "hda_pg", "xg_share", "corsi", "fenwick",
            "hits_pg", "pim_pg", "fow_pct", "takeaways_pg", "giveaways_pg"]
    return df[[c for c in keep if c in df.columns]].sort_values(
        ["team", "season"], ascending=[True, False]).reset_index(drop=True)


@st.cache_data(show_spinner="Reading the schedule ...")
def schedule() -> pd.DataFrame:
    """Per-team, per-game schedule context for the target season (opponent, home, SOS)."""
    import context as ctx
    return ctx.schedule_context(C.TARGET_SEASON)


@st.cache_data(show_spinner="Rating teams ...")
def team_ratings() -> pd.DataFrame:
    import context as ctx
    return ctx.team_defense_ratings().set_index("team")


@st.cache_data(show_spinner="Projecting games ...", max_entries=3)
def games(key: str, stamp: str = "") -> pd.DataFrame:
    import project_games as pgm
    sk, _ = _skaters(key, stamp)
    return pgm.project_games(season_proj=sk)


# --------------------------------------------------------------------------- #
# formatting                                                                  #
# --------------------------------------------------------------------------- #
def team_options(df: pd.DataFrame) -> list[str]:
    return sorted(t for t in df["team"].dropna().unique() if t != "FA")


def pct(x) -> str:
    return "-" if pd.isna(x) else f"{x:.1%}"


def sv(x) -> str:
    """Save percentage the way hockey writes it: .912, not 0.912."""
    return "-" if pd.isna(x) else f"{x:.4f}".lstrip("0")


def num(x, nd: int = 1) -> str:
    return "-" if pd.isna(x) else f"{x:,.{nd}f}"


def band(lo, hi, nd: int = 0) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return "-"
    return f"{lo:,.{nd}f} - {hi:,.{nd}f}"


def edit_badge(n: int) -> str:
    return f"{n} edit{'s' if n != 1 else ''}"


def roster_label(row) -> str:
    """The first thing on a player's caption: his age, or why he has no team.

    Three states, not two. "Unsigned" and "in somebody's camp" are different situations
    and telling a reader a Blackhawks prospect is unsigned is simply wrong -- he is on the
    roster page, he just lost the model's read on the last job. Both are off every team
    budget, and giving him a team on this page is what puts him back on one.
    """
    if row["on_roster"]:
        return f"age {row['target_age']:.0f}"
    camp = row.get("camp_team") or ""
    return f"in {camp} camp" if row.get("camp") and camp else "unsigned"


def header(title: str, sub: str | None = None) -> None:
    st.markdown(f"### {title}")
    if sub:
        st.caption(sub)


def scenario_bar() -> None:
    """The sidebar: what scenario is loaded, how far it has been argued with, and out."""
    sc = scenario()
    n = sc.count()
    with st.sidebar:
        st.markdown(f"**{SEASON_LABEL} projections**")
        # What window the numbers cover. Before opening night this says "preseason"; after
        # it, it is the single most important thing on the screen -- a season total that
        # already contains 24 games of hockey is a different claim from one that contains
        # none, and a reader who does not know which he is looking at cannot use either.
        st.caption(window_label())
        st.caption("Baseline model" if sc.is_baseline
                   else f"{edit_badge(n['edits'])} · "
                        f"{n['players']} skaters, {n['goalies']} goalies, {n['teams']} teams")
        if not sc.is_baseline and st.button("Clear all edits", width="stretch"):
            commit(sc.clear_all(), "Back to the baseline model")
        if MULTIUSER:
            st.caption("Your edits are yours alone — nobody else's view changes, and "
                       "nothing you do here is saved automatically. Publish them on the "
                       "**Scenario** page to let others open them, or download the JSON.")
        st.divider()
