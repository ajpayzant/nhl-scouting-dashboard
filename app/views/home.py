"""Overview: where the season stands and where to go next.

Deliberately short. A landing page that tries to be a dashboard ends up being a worse
version of every other page, so this one answers three things -- is the projection built,
what does it currently say at the top, and what have I changed -- and then gets out of
the way.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import core


def _leaders(df: pd.DataFrame, cols: dict[str, str], sort: str, n: int = 12,
             fmt: dict | None = None) -> None:
    show = df.sort_values(sort, ascending=False).head(n)[list(cols)].rename(columns=cols)
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config=fmt or {}, height=35 * n + 40)


def page() -> None:
    core.scenario_bar()
    sk, tb = core.skaters()
    g, gb = core.goalies()
    sc = core.scenario()
    on_sk, on_g = sk[sk["on_roster"]], g[g["on_roster"]]

    core.header(f"{core.SEASON_LABEL} season projections",
                "Every skater and goalie on an NHL roster, projected from five seasons of "
                "rates and settled against what each team has to give.")

    m = st.columns(5)
    m[0].metric("Skaters", f"{len(on_sk):,}",
                help=f"{len(sk) - len(on_sk)} more are projected but on no roster — "
                     "camp bodies and unsigned players, all off every team budget")
    m[1].metric("Goalies", f"{len(on_g):,}")
    m[2].metric("Teams", len(tb))
    m[3].metric("Roster covers", f"{float(tb['toi_coverage'].mean()):.0%}",
                help="share of league ice time the published rosters account for; the rest "
                     "is held back for players not yet named")
    m[4].metric("Your edits", sc.count()["edits"],
                help="the model's own opinion everywhere you have not disagreed")

    if sc.is_baseline:
        st.info("This is the model with nothing overridden. Open **Skaters** or **Goalies** "
                "to state what you know that it does not — a depth chart, a training-camp "
                "line, a signing — and every teammate resettles around it.")
    else:
        n = sc.count()
        st.success(f"{core.edit_badge(n['edits'])} applied across {n['players']} skaters, "
                   f"{n['goalies']} goalies and {n['teams']} team budgets. **Scenario** "
                   "lists them all, and any one of them can be undone exactly.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Points**")
        _leaders(on_sk, {"name": "Player", "team": "Team", "proj_gp": "GP",
                         "proj_points": "PTS", "points_p10": "Floor",
                         "points_p90": "Ceiling"}, "proj_points", fmt={
            "GP": st.column_config.NumberColumn(format="%.0f"),
            "PTS": st.column_config.NumberColumn(format="%.1f"),
            "Floor": st.column_config.NumberColumn(format="%.0f"),
            "Ceiling": st.column_config.NumberColumn(format="%.0f")})
    with c2:
        st.markdown("**Goals**")
        _leaders(on_sk, {"name": "Player", "team": "Team", "proj_goals": "G",
                         "goals_p10": "Floor", "goals_p90": "Ceiling",
                         "proj_shots": "SOG"}, "proj_goals", fmt={
            "G": st.column_config.NumberColumn(format="%.1f"),
            "Floor": st.column_config.NumberColumn(format="%.0f"),
            "Ceiling": st.column_config.NumberColumn(format="%.0f"),
            "SOG": st.column_config.NumberColumn(format="%.0f")})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Goalie wins**")
        _leaders(on_g, {"name": "Goalie", "team": "Team", "proj_starts": "GS",
                        "proj_wins": "W", "proj_save_pct": "SV%", "proj_gaa": "GAA"},
                 "proj_wins", fmt={
                     "GS": st.column_config.NumberColumn(format="%.1f"),
                     "W": st.column_config.NumberColumn(format="%.1f"),
                     "SV%": st.column_config.NumberColumn(format="%.4f"),
                     "GAA": st.column_config.NumberColumn(format="%.2f")})
        st.caption("A projected starter takes about 47 of 84 starts, not 60: the number is "
                   "an expectation, and it already prices in the injury and the bad "
                   "November that cost him the net. State a depth chart on his page if you "
                   "want a specific plan instead.")
    with c2:
        st.markdown("**Team scoring**")
        rows = pd.DataFrame({
            "Team": tb.index,
            "Budget": tb["goals"].to_numpy(),
            "Projected": [float(on_sk.loc[on_sk["team"] == t, "proj_goals"].sum())
                          for t in tb.index],
            "Roster covers": tb["toi_coverage"].to_numpy(),
        })
        _leaders(rows, {"Team": "Team", "Budget": "Goals budget",
                        "Projected": "Projected", "Roster covers": "Roster covers"},
                 "Budget", fmt={
                     "Goals budget": st.column_config.NumberColumn(format="%.0f"),
                     "Projected": st.column_config.NumberColumn(format="%.0f"),
                     "Roster covers": st.column_config.ProgressColumn(
                         format="%.1f%%", min_value=0.0, max_value=1.0)})
        st.caption("Projected sits below budget by exactly the part of the roster that has "
                   "not been named yet. **Teams** shows that arithmetic per stat.")
