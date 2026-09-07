"""How the projections are actually doing this season.

Every other page in this app is the model talking. This one is the season talking back.

It works off dated snapshots (`snapshots.py`): each time the data refreshes, the baseline
projection files a copy of itself, and a copy that cannot be edited afterwards is the only
honest basis for a scorecard. Each snapshot is scored on the games played SINCE it was
taken -- not on the season total, which nobody knows until April -- so there is something
to say from the second week of October onward.

Four questions, four tabs:

  Scorecard      per stat, how big the miss was and in which direction, ranked by error as
                 a SHARE of what happened so points and hits can sit in the same table.
  Players        who the model was most wrong about, both ways, with availability error
                 separated from rate error: a projection can be right about a player and
                 wrong about how often he plays, and those are different fixes.
  Vintages       is the mid-season projection actually better than the preseason one? Read
                 per game, because the vintages are judged over different windows.
  On pace        the chance each player still reaches the total he was given in September,
                 from the model's own band rather than a new assumption.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import core
import snapshots as sn

# Plain-English names, so a table does not make a reader translate `faceoffs_won`.
LABEL = {"points": "Points", "goals": "Goals", "assists": "Assists", "shots": "Shots",
         "gp": "Games played", "toi": "Ice time (min)", "pp_points": "PP points",
         "sh_points": "SH points", "blocks": "Blocks", "hits": "Hits", "pim": "PIM",
         "ixg": "Expected goals", "faceoffs_won": "Faceoffs won",
         "wins": "Wins", "starts": "Starts", "saves": "Saves", "minutes": "Minutes",
         "shots_against": "Shots against", "goals_against": "Goals against",
         "shutouts": "Shutouts", "losses": "Losses"}


def _label(stat: str) -> str:
    return LABEL.get(stat, stat.replace("_", " ").capitalize())


@st.cache_data(show_spinner="Scoring the projections ...", ttl=core.LIVE_TTL)
def _scored(kind: str, day: str, token: str) -> pd.DataFrame:
    return sn.score_players(kind, day)


@st.cache_data(show_spinner=False, ttl=core.LIVE_TTL)
def _vintages(kind: str, stat: str, token: str) -> pd.DataFrame:
    return sn.by_vintage(kind, stat)


@st.cache_data(show_spinner="Working out who is on pace ...", ttl=core.LIVE_TTL)
def _reach(kind: str, stat: str, day: str, token: str) -> pd.DataFrame:
    proj = core.baseline_skaters()[0] if kind == "skaters" else core.baseline_goalies()[0]
    return sn.reach(kind, stat, target_day=day, current=proj)


# --------------------------------------------------------------------------- #
# tabs                                                                        #
# --------------------------------------------------------------------------- #
def _scorecard(kind: str, day: str, pl: pd.DataFrame) -> None:
    stats = sn.score_stats(kind, day, players=pl)
    if stats.empty:
        st.info("Not enough hockey has been played since that snapshot to score it.")
        return
    window = float(pl["team_games_since"].mean())
    # The per-stat rows below need a player to have appeared five times in the window,
    # otherwise a fourth-liner's two games dominate a normalised error. Report the number
    # that actually feeds the table rather than the number of rows scored.
    used = int((pl["obs_gp_since"] >= 5.0).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Snapshot", day, help="the projection being graded")
    c2.metric("Games since", f"{window:.0f}", help="team games played since it was filed")
    c3.metric("Players scored", f"{used:,}",
              help=f"of {len(pl):,} with any appearance since; five in the window is the "
                   "minimum for a stat row")

    show = stats.copy()
    show["Stat"] = show["stat"].map(_label)
    show["Direction"] = show["bias"].map(
        lambda b: "under-projected" if b > 0 else ("over-projected" if b < 0 else "even"))
    cols = {"Stat": "Stat", "observed_per_player": "Actual per player",
            "projected_per_player": "Projected per player", "mae": "Avg miss",
            "nmae": "Miss as % of actual", "bias": "Bias", "Direction": "Direction",
            "rate_mae": "Avg miss, per-game rate only",
            "corr": "Correlation", "band_coverage": "In its 80% band"}
    view = show[[c for c in cols if c in show]].rename(columns=cols)
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "Actual per player": st.column_config.NumberColumn(format="%.2f"),
        "Projected per player": st.column_config.NumberColumn(format="%.2f"),
        "Avg miss": st.column_config.NumberColumn(
            format="%.2f", help="mean absolute error over the window since the snapshot"),
        "Miss as % of actual": st.column_config.NumberColumn(
            format="percent",
            help="the ranking column: error relative to how much of the stat there is"),
        "Bias": st.column_config.NumberColumn(
            format="%+.2f", help="positive = players did MORE than projected"),
        "Avg miss, per-game rate only": st.column_config.NumberColumn(
            format="%.2f",
            help="the same error charged only for the games he actually played, which "
                 "takes availability out of it"),
        "Correlation": st.column_config.NumberColumn(format="%.2f"),
        "In its 80% band": st.column_config.NumberColumn(
            format="percent", help="should sit near 80%; well above means the bands are "
                                   "too wide, well below means too narrow")})
    worst, best = view.iloc[0], view.iloc[-1]
    st.caption(
        f"Least accurate: **{worst['Stat']}** (miss of "
        f"{worst['Miss as % of actual']:.0%} of what happened). Most accurate: "
        f"**{best['Stat']}** ({best['Miss as % of actual']:.0%}). Sorted on the share, not "
        "the raw miss — otherwise the list is just the biggest stats first.")
    st.caption(
        "Two error columns, because they are different failures. The **avg miss** includes "
        "getting a player's availability wrong; the **per-game rate** column charges the "
        "model only for the games he really played. A wide gap between them says the model "
        "understands the player and not his health.")


def _players(kind: str, day: str, pl: pd.DataFrame) -> None:
    opts = [s for s in sn.HEADLINE[kind] if f"err_{s}" in pl]
    if not opts:
        st.info("Nothing scorable yet.")
        return
    c1, c2 = st.columns([1, 3])
    stat = c1.selectbox("Stat", opts, format_func=_label, key="perf_miss_stat")
    n = int(c2.slider("How many each way", 5, 40, 15, key="perf_miss_n"))
    under, over = sn.misses(kind, day, stat=stat, n=n, players=pl)
    rename = {"name": "Player", "team": "Team", "position": "Pos",
              "obs_gp_since": "GP since", "exp_gp_since": "GP projected",
              f"obs_{stat}": f"{_label(stat)} actual",
              f"exp_{stat}": f"{_label(stat)} projected",
              f"err_{stat}": "Miss", f"rate_err_{stat}": "Miss, rate only"}
    cfg = {"Miss": st.column_config.NumberColumn(format="%+.1f"),
           "Miss, rate only": st.column_config.NumberColumn(
               format="%+.1f", help="the miss he would have had if the model had known how "
                                    "many games he would play"),
           "GP since": st.column_config.NumberColumn(format="%.0f"),
           "GP projected": st.column_config.NumberColumn(format="%.0f"),
           f"{_label(stat)} actual": st.column_config.NumberColumn(format="%.1f"),
           f"{_label(stat)} projected": st.column_config.NumberColumn(format="%.1f")}
    left, right = st.columns(2)
    with left:
        st.markdown("**Under-projected** — did more than the model said")
        st.dataframe(under.rename(columns=rename), hide_index=True, width="stretch",
                     column_config=cfg)
    with right:
        st.markdown("**Over-projected** — did less")
        st.dataframe(over.rename(columns=rename), hide_index=True, width="stretch",
                     column_config=cfg)
    st.caption("Ranked on the size of the miss, not on the ratio: a player given four "
               "points who scored eleven is a more useful failure than one given 0.2 who "
               "got 0.6, and the ratio version of this table is always fourth-liners.")


def _vintage_tab(kind: str) -> None:
    stat = st.selectbox("Stat", sn.HEADLINE[kind], format_func=_label, key="perf_vint_stat")
    vt = _vintages(kind, stat, core.live_token())
    if vt.empty:
        st.info("Two snapshots with games played between them are needed before vintages "
                "can be compared. That takes a week or two of the season.")
        return
    show = vt.rename(columns={
        "snap_date": "Snapshot", "frac_played_then": "Season played then",
        "window_games": "Games judged over", "players": "Players",
        "mae": "Avg miss", "mae_per_game": "Avg miss per game",
        "nmae": "Miss as % of actual", "bias": "Bias", "band_coverage": "In its 80% band"})
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "Season played then": st.column_config.NumberColumn(format="percent"),
        "Games judged over": st.column_config.NumberColumn(format="%.0f"),
        "Avg miss": st.column_config.NumberColumn(format="%.2f"),
        "Avg miss per game": st.column_config.NumberColumn(
            format="%.3f", help="the comparable column: the vintages cover windows of "
                                "different lengths, so only the per-game error can be put "
                                "side by side"),
        "Miss as % of actual": st.column_config.NumberColumn(format="percent"),
        "Bias": st.column_config.NumberColumn(format="%+.2f"),
        "In its 80% band": st.column_config.NumberColumn(format="percent")})
    if len(vt) > 1:
        first, last = vt.iloc[0], vt.iloc[-1]
        delta = last["mae_per_game"] - first["mae_per_game"]
        better = "better" if delta < 0 else "worse"
        st.caption(
            f"The {last['snap_date']} projection is {abs(delta):.3f} {_label(stat).lower()} "
            f"per game {better} than the {first['snap_date']} one. That is the whole case "
            "for updating during the season: if this number does not fall, the in-season "
            "signal is not paying for itself.")
    st.caption("Each snapshot is scored only on the games played after it was filed, so no "
               "vintage is credited with knowing something it could not have known.")


def _pace(kind: str) -> None:
    days = sn.vintages(kind)
    if not days:
        st.info("No snapshots on file yet.")
        return
    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
    stat = c1.selectbox("Stat", sn.HEADLINE[kind], format_func=_label, key="perf_pace_stat")
    day = c2.selectbox("Projection to measure against", days, index=0, key="perf_pace_day",
                       help="the earliest snapshot is the preseason projection")
    df = _reach(kind, stat, day, core.live_token())
    if df.empty:
        st.info("Nothing to compare yet.")
        return
    df = df[df["on_roster"]] if "on_roster" in df else df
    floor = float(c3.number_input(f"Only players projected {_label(stat).lower()} above",
                                 min_value=0.0, value=float(round(df["target"].quantile(0.9))),
                                 step=1.0, key="perf_pace_floor"))
    df = df[df["target"] >= floor].copy()
    hit = float(df["on_pace"].mean()) if len(df) else float("nan")
    c1, c2 = st.columns(2)
    c1.metric("On pace to reach it", core.pct(hit),
              help="ahead of the projection they were given, on today's numbers")
    c2.metric("Average chance", core.pct(float(df["p_reach"].mean())) if len(df) else "-",
              help="the model's own probability, from its rest-of-season band")
    rename = {"name": "Player", "team": "Team", "position": "Pos", "target": "Projected then",
              f"act_{stat}": "Banked", f"ros_{stat}": "Rest of season",
              "pace": "Projected now", "p_reach": "Chance of reaching it"}
    cols = [c for c in ("name", "team", "position", "target", f"act_{stat}", f"ros_{stat}",
                        "pace", "p_reach") if c in df]
    st.dataframe(df[cols].rename(columns=rename), hide_index=True, width="stretch",
                 column_config={
                     "Projected then": st.column_config.NumberColumn(format="%.1f"),
                     "Banked": st.column_config.NumberColumn(format="%.1f"),
                     "Rest of season": st.column_config.NumberColumn(format="%.1f"),
                     "Projected now": st.column_config.NumberColumn(format="%.1f"),
                     "Chance of reaching it": st.column_config.ProgressColumn(
                         format="percent", min_value=0.0, max_value=1.0)})
    st.caption(
        "The chance comes from the model's own p10-p90 band on the games that are left, so "
        "it needs no new assumption: an 80% band implies a standard deviation, and the "
        "question is simply whether banked plus rest clears the target. A player already "
        "past it is at 100% as arithmetic, not as an estimate.")


# --------------------------------------------------------------------------- #
# page                                                                        #
# --------------------------------------------------------------------------- #
def page() -> None:
    core.scenario_bar()
    core.header("Model performance",
                "How the projections filed earlier this season have actually done. "
                "Baseline model only — a scorecard for somebody's edits would not answer "
                "the question.")
    state = core.season_state(core.live_token())
    days = sn.vintages("skaters")
    if not days:
        st.info("No snapshots have been filed yet. The daily refresh files one "
                "automatically (`python run.py --snapshot`); the first one becomes the "
                "preseason projection every later one is compared against.")
        if st.button("File one now"):
            with st.spinner("Projecting and filing ..."):
                sn.take()
            st.cache_data.clear()
            st.rerun()
        return
    if not state.started:
        st.info(f"{len(days)} snapshot(s) on file and the season has not started, so there "
                "is nothing to score them against yet. This page fills in from the second "
                "week of October.")
        st.dataframe(sn.index(), hide_index=True, width="stretch")
        return

    kind = st.radio("Who", ["skaters", "goalies"], horizontal=True, key="perf_kind",
                    format_func=lambda k: k.capitalize(), label_visibility="collapsed")
    scorable = sn.latest_scorable(kind)
    all_days = sn.vintages(kind)
    day = st.selectbox(
        "Projection being graded", all_days,
        index=all_days.index(scorable) if scorable in all_days else 0,
        key="perf_day",
        help="scored on the games played since it was filed; the newest snapshot has no "
             "window behind it yet, which is why the default is the newest one that does")
    pl = _scored(kind, day, core.live_token())
    if pl.empty:
        st.warning("No games have been played since that snapshot. Pick an earlier one.")
        return

    t1, t2, t3, t4 = st.tabs(["Scorecard", "Who it missed", "Preseason vs now", "On pace"])
    with t1:
        _scorecard(kind, day, pl)
    with t2:
        _players(kind, day, pl)
    with t3:
        _vintage_tab(kind)
    with t4:
        _pace(kind)

    with st.expander("Snapshots on file"):
        st.dataframe(sn.index(), hide_index=True, width="stretch")
        st.caption(f"Baseline only, filed by `python run.py --snapshot`, which the daily "
                   f"data refresh runs — so the record builds itself. The cadence is "
                   f"measured in hockey rather than days: a new one once "
                   f"{sn.MIN_GAP_GAMES:.0f} more team-games have been played, which is "
                   f"about weekly, so September does not fill up with identical copies of "
                   f"the preseason projection.")
