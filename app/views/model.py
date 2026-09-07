"""Diagnostics: does the projection still add up, and how old is the data.

Two questions only, both answerable by looking. The identities are arithmetic that must
hold -- 32 teams playing 84 games start 2,688 goalies and win 1,344 games, whatever the
model thinks of anybody -- so if a row here is off, the projection is wrong regardless of
how reasonable the player pages look. Everything else on this page is a number that was
measured rather than assumed, listed so the next person can re-measure it.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

import core

CONSTANTS = [
    ("Goalie depth-chart shares", "GOALIE_RANK_SHARES",
     "Share of a team's goalie minutes by depth-chart rank, measured on 269 clean "
     "team-seasons (2012-2025) using the rank the MODEL can see. Using hindsight ranks "
     "instead would build the model's own error in as knowledge."),
    ("Own-share flattening", "GOALIE_CLAIM_SHARE_EXP",
     "Summed goalie claims over-count, because each goalie's recent starts were partly "
     "someone else's injury. Shares go as claim to this power, not proportionally."),
    ("Quality tilt on starts", "GOALIE_QUALITY_START_TILT",
     "A coach gives the net to whoever stops the puck: within a team, a goalie 5% better "
     "than his partner takes about 5 more starts of 82. Applied team-centred so the "
     "shifts cancel and the team's starts stay exact."),
    ("Budget tilt", "BUDGET_TILT",
     "How a team's overshoot is taken back. Below 1 protects high-usage players, who are "
     "the least likely to be the reason the team is oversubscribed."),
    ("Expected-goals weight", "GOALS_XG_WEIGHT",
     "How far a goal projection leans on shot quality rather than the player's own "
     "finishing, which regresses much harder."),
    ("Skater regression", "SKATER_REGRESS_TOI_MIN",
     "Ice time a skater needs before his own per-60 rates outweigh his position's prior."),
    ("Goalie regression", "GOALIE_REGRESS_SHOTS",
     "Shots a goalie needs before his own save percentage outweighs the league's."),
]


def _identities(sk: pd.DataFrame, tb: pd.DataFrame, g: pd.DataFrame,
                gb: pd.DataFrame) -> None:
    teams, games = len(tb), float(core.C.SEASON_GAMES)
    on_sk, on_g = sk[sk["on_roster"]], g[g["on_roster"]]

    st.markdown("**Goaltending — these are arithmetic, not opinion**")
    rows = []
    for field, label, identity in [
            ("starts", "Starts", games * teams),
            ("wins", "Wins", games * teams / 2.0),
            ("gp", "Appearances", None), ("minutes", "Minutes", None),
            ("shots_against", "Shots against", None),
            ("goals_against", "Goals against", None),
            ("shutouts", "Shutouts", None)]:
        bcol = {"gp": "appearances"}.get(field, field)
        budget = float(gb[bcol].sum())
        expected = float((gb[bcol] * gb["coverage"]).sum())
        got = float(on_g[f"proj_{field}"].sum())
        rows.append({"": label, "Full budget": budget,
                     "Identity": identity if identity is not None else float("nan"),
                     "Budget x coverage": expected, "Allocated": got,
                     "Off by": got - expected})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Full budget": st.column_config.NumberColumn(format="%.0f"),
        "Identity": st.column_config.NumberColumn(
            format="%.0f", help="what the schedule forces, independent of the model"),
        "Budget x coverage": st.column_config.NumberColumn(format="%.0f"),
        "Allocated": st.column_config.NumberColumn(format="%.0f"),
        "Off by": st.column_config.NumberColumn(format="%+.1f")})

    st.markdown("**Skaters, per team-game**")
    tg = teams * games
    rows = []
    for stat, label in [("toi", "Ice time (min)"), ("goals", "Goals"),
                        ("assists", "Assists"), ("points", "Points"), ("shots", "Shots"),
                        ("blocks", "Blocks"), ("hits", "Hits"), ("pim", "PIM"),
                        ("faceoffs_won", "Faceoffs won")]:
        bcol = "toi_min" if stat == "toi" else stat
        if bcol not in tb.columns:
            continue
        full = float(tb[bcol].sum()) / tg
        eff = (float(tb[f"eff_{bcol}"].sum()) if f"eff_{bcol}" in tb.columns
               else float((tb[bcol] * tb["toi_coverage"]).sum())) / tg
        got = float(on_sk[f"proj_{stat}"].sum()) / tg
        rows.append({"": label, "Full budget": full, "Budget x coverage": eff,
                     "Allocated": got, "Off by": got - eff})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Full budget": st.column_config.NumberColumn(format="%.2f"),
        "Budget x coverage": st.column_config.NumberColumn(format="%.2f"),
        "Allocated": st.column_config.NumberColumn(format="%.2f"),
        "Off by": st.column_config.NumberColumn(format="%+.3f")})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Skaters listed per team", f"{len(on_sk) / teams:.1f}",
              help="about 28 skaters play a real team-season")
    c2.metric("Roster covers", f"{float(tb['toi_coverage'].mean()):.1%}",
              help="share of league ice time the listed rosters account for")
    c3.metric("Goalies listed per team", f"{len(on_g) / teams:.1f}",
              help="about 3.1 goalies play a real team-season")
    c4.metric("League SV% allocated", core.sv(
        1.0 - on_g["proj_goals_against"].sum() / on_g["proj_shots_against"].sum()),
        help=f"league level {core.sv(float((gb['sv_pct'] * gb['shots_against']).sum() / gb['shots_against'].sum()))}")
    st.caption(
        "The gap between the full budget and budget-x-coverage is the part of the season "
        "held back for players not on a published roster. It is not a rounding error: a "
        "listed roster is about 22 skaters and 2.6 goalies, and a real season uses 28 and "
        "3.1. Handing the listed players the whole budget is exactly the mistake that "
        "makes a projection look generous in July and wrong in April.")


def _constants() -> None:
    st.caption("Each of these was measured against completed seasons and checked out of "
               "sample. The measurement is worth re-running after a data refresh.")
    for label, name, why in CONSTANTS:
        value = getattr(core.C, name, None)
        if isinstance(value, list):
            shown = ", ".join(f"{v:.3f}" for v in value)
        elif isinstance(value, float):
            shown = f"{value:g}"
        else:
            shown = str(value)
        st.markdown(f"**{label}** — `{name}` = {shown}")
        st.caption(why)

    st.markdown("**Goalie starts floor and ceiling**")
    tbl = core.C.GOALIE_START_BAND
    scale = float(core.C.SEASON_GAMES) / float(tbl["measured_season"])
    st.dataframe(pd.DataFrame({
        "Projected starts": [v * scale for v in tbl["proj"]],
        "Floor (p10)": [v * scale for v in tbl["p10"]],
        "Ceiling (p90)": [v * scale for v in tbl["p90"]],
    }), hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.0f")
        for c in ("Projected starts", "Floor (p10)", "Ceiling (p90)")})
    st.caption("Measured quantiles, interpolated, not a symmetric error bar. A backup's "
               "season is bimodal — eight starts, or forty because the man ahead of him "
               "got hurt — so no smooth distribution fits it. A parametric band covered "
               "62% of real outcomes against an 80% target; this table covers 82%.")


def _freshness() -> None:
    files = sorted(core.C.DATA_RAW.glob("*.parquet")) + \
        sorted(core.C.DATA_RAW.glob("*.json")) + \
        sorted(core.C.DATA_RAW.glob("*.csv"))
    if not files:
        st.warning("No cached data files found.")
        return
    today = dt.date.today()
    rows = [{"File": p.name,
             "Cached": dt.datetime.fromtimestamp(p.stat().st_mtime).date(),
             "Days old": (today - dt.datetime.fromtimestamp(p.stat().st_mtime).date()).days,
             "MB": p.stat().st_size / 1e6} for p in files]
    df = pd.DataFrame(rows).sort_values("Days old", ascending=False)
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "MB": st.column_config.NumberColumn(format="%.1f")})
    roster_age = int(df.loc[df["File"].str.contains("roster"), "Days old"].max()) \
        if df["File"].str.contains("roster").any() else int(df["Days old"].max())
    if roster_age > 14:
        st.warning(f"The cached rosters are {roster_age} days old. Rosters move constantly "
                   "in the weeks before a season, and a stale one puts traded and cut "
                   "players on the wrong team — which costs a reader more manual work than "
                   "anything else in here.")
    c1, c2 = st.columns([1.4, 3])
    if c1.button("Refresh rosters and schedule", width="stretch"):
        with st.spinner("Asking the NHL for all 32 rosters ..."):
            core.dl.load_rosters(refresh=True)
            core.dl.load_schedule(refresh=True)
        st.cache_data.clear()
        st.success("Rosters and schedule refreshed. The projection will rebuild.")
        st.rerun()
    c2.caption("Season history is a bigger download and is only worth refreshing once a "
               "season has finished: run `python run.py --refresh` in the repo for that.")
    st.caption(f"Projecting {core.SEASON_LABEL} from history through "
               f"{core.C.LAST_COMPLETED_SEASON}-"
               f"{str(core.C.LAST_COMPLETED_SEASON + 1)[-2:]}.")


def page() -> None:
    core.scenario_bar()
    sk, tb = core.skaters()
    g, gb = core.goalies()
    core.header("Model check",
                "Whether the projection still adds up, what was measured rather than "
                "assumed, and how old the data underneath it is.")
    tab_id, tab_const, tab_data = st.tabs(
        ["Does it add up", "Measured constants", "Data freshness"])
    with tab_id:
        _identities(sk, tb, g, gb)
    with tab_const:
        _constants()
    with tab_data:
        _freshness()
