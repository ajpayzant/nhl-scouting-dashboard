"""Game by game: the season projection spread across the real schedule.

Nothing here is a new opinion. The season total is the anchor and each night is a share
of it, reshaped by opponent, home ice and rest, with the shares averaging one so the
games sum back to the season exactly. That is deliberate: simulating games and adding
them up does not make a season total more accurate, it only makes it slower.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import core

STATS = {"exp_points": "PTS", "exp_goals": "G", "exp_assists": "A",
         "exp_shots": "SOG", "exp_pp_points": "PPP"}


def _slate(gm: pd.DataFrame) -> None:
    dates = sorted(pd.to_datetime(gm["gameDate"]).dt.date.unique())
    c1, c2 = st.columns([1, 3])
    day = c1.selectbox("Date", dates, format_func=lambda d: d.strftime("%a %d %b %Y"))
    stat = c2.radio("Stat", list(STATS), horizontal=True,
                    format_func=lambda s: STATS[s], label_visibility="collapsed")

    d = gm[pd.to_datetime(gm["gameDate"]).dt.date == day]
    if d.empty:
        st.info("No games that day.")
        return
    show = pd.DataFrame({
        "Player": d["name"], "Team": d["team"],
        "Matchup": np.where(d["is_home"], "vs " + d["opponent"], "@ " + d["opponent"]),
        **{label: d[col] for col, label in STATS.items()},
    }).sort_values(STATS[stat], ascending=False).head(200)
    st.caption(f"{d['team'].nunique()} teams playing · {len(d)} player-games")
    st.dataframe(show, hide_index=True, width="stretch", height=420, column_config={
        label: st.column_config.NumberColumn(format="%.2f") for label in STATS.values()})
    st.download_button("Download this slate (CSV)",
                       d.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"slate_{day}.csv", mime="text/csv")


def _player_log(gm: pd.DataFrame) -> None:
    names = sorted(gm["name"].unique())
    who = st.selectbox("Player", names, key="gm_player")
    d = gm[gm["name"] == who].sort_values("gameDate")
    show = pd.DataFrame({
        "Date": pd.to_datetime(d["gameDate"]).dt.strftime("%a %d %b"),
        "Matchup": np.where(d["is_home"], "vs " + d["opponent"], "@ " + d["opponent"]),
        **{label: d[col] for col, label in STATS.items()},
    })
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(show, hide_index=True, height=400, width="stretch", column_config={
            label: st.column_config.NumberColumn(format="%.2f")
            for label in STATS.values()})
    with c2:
        st.bar_chart(show.set_index("Date")["PTS"], height=400)
    st.caption(f"Sums to {float(d['exp_points'].sum()):.1f} points over "
               f"{len(d)} games — the same number the season page shows.")


def page() -> None:
    core.scenario_bar()
    core.header(f"Game by game · {core.SEASON_LABEL}",
                "The season projection distributed over the real schedule. Opponent, home "
                "ice and rest reshape each night; the totals do not move.")

    if not st.session_state.get("games_on"):
        st.info("Building the game-by-game table takes about a minute (60,000 rows). "
                "It is rebuilt only when the projection changes.")
        if st.button("Build game-by-game projections", type="primary"):
            st.session_state.games_on = True
            st.rerun()
        return

    gm = core.games(core._skater_key(core.scenario()))
    tab_slate, tab_player = st.tabs(["A night's slate", "One player's schedule"])
    with tab_slate:
        _slate(gm)
    with tab_player:
        _player_log(gm)
