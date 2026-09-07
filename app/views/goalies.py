"""Goalies: the starts split first, everything else downstream of it.

A goalie season is a depth-chart decision before it is a talent question -- the same
.910 goalie is worth 30 wins or 12 depending on how many nights he gets the net. So the
board leads with starts and shows the measured floor-to-ceiling band beside them, because
a backup's season is genuinely bimodal: he starts 8 games, or the man ahead of him gets
hurt and he starts 40. The `starts` edit is how a reader states a depth chart the model
cannot know.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import core

LOCKS = [("wins", "Wins", 82.0), ("shutouts", "Shutouts", 20.0),
         ("shots_against", "Shots against", 3000.0),
         ("goals_against", "Goals against", 300.0), ("minutes", "Minutes", 5200.0),
         ("gp", "Appearances", 84.0)]


def _filters(g: pd.DataFrame) -> pd.DataFrame:
    c1, c2, c3 = st.columns([2.2, 1.7, 1.6])
    q = c1.text_input("Search", placeholder="Search a name", label_visibility="collapsed")
    teams = c2.multiselect("Team", core.team_options(g), placeholder="All teams",
                           label_visibility="collapsed")
    who = c3.selectbox("Who", ["On a roster", "Everyone", "In a camp", "Unsigned only",
                               "Edited only"], label_visibility="collapsed")
    df = g.copy()
    df["_camp"] = df["camp"].fillna(False) if "camp" in df else False
    ct = df["camp_team"].fillna("") if "camp_team" in df else ""
    df["_home"] = df["team"].mask(df["_camp"] & (ct != ""), ct)
    if q:
        df = df[df["name"].str.contains(q, case=False, na=False)]
    if teams:
        df = df[df["_home"].isin(teams)]      # a camp goalie answers to his camp's name
    if who == "On a roster":
        df = df[df["on_roster"]]
    elif who == "In a camp":
        df = df[df["_camp"]]
    elif who == "Unsigned only":
        df = df[~df["on_roster"] & ~df["_camp"]]
    elif who == "Edited only":
        df = df[df["edited"]]
    return df


def _grid(df: pd.DataFrame) -> int | None:
    show = pd.DataFrame({
        " ": np.where(df["edited"], "✎", ""),
        "Goalie": df["name"], "Team": df["team"],
        "GP": df["proj_gp"], "GS": df["proj_starts"],
        "GS floor": df["starts_p10"], "GS ceiling": df["starts_p90"],
        "W": df["proj_wins"], "L": df["proj_losses"], "OTL": df["proj_otl"],
        "SV%": df["proj_save_pct"], "GAA": df["proj_gaa"], "SO": df["proj_shutouts"],
        "Saves": df["proj_saves"], "SA/60": df["rate_sa_per_60"],
    })
    ev = st.dataframe(
        show, hide_index=True, height=440, width="stretch",
        on_select="rerun", selection_mode="single-row",
        column_config={
            " ": st.column_config.TextColumn(" ", width="small", help="edited"),
            "Goalie": st.column_config.TextColumn(width="medium"),
            "Team": st.column_config.TextColumn(width="small"),
            "GP": st.column_config.NumberColumn(format="%.1f", help="appearances"),
            "GS": st.column_config.NumberColumn(format="%.1f", help="starts"),
            "GS floor": st.column_config.NumberColumn(format="%.0f"),
            "GS ceiling": st.column_config.NumberColumn(format="%.0f"),
            "W": st.column_config.NumberColumn(format="%.1f"),
            "L": st.column_config.NumberColumn(format="%.1f"),
            "OTL": st.column_config.NumberColumn(format="%.1f"),
            "SV%": st.column_config.NumberColumn(format="%.4f"),
            "GAA": st.column_config.NumberColumn(format="%.2f"),
            "SO": st.column_config.NumberColumn(format="%.1f"),
            "Saves": st.column_config.NumberColumn(format="%.0f"),
            "SA/60": st.column_config.NumberColumn(
                format="%.1f", help="shots he is expected to face per 60 minutes"),
        })
    rows = ev.selection.get("rows") or []
    return int(rows[0]) if rows else None


# --------------------------------------------------------------------------- #
def _goalie(row: pd.Series, g: pd.DataFrame, gb: pd.DataFrame,
            base: pd.DataFrame) -> None:
    pid = int(row["playerId"])
    edits = core.scenario().goalie(pid)
    b = base[base["playerId"] == pid]
    b = b.iloc[0] if len(b) else None

    st.markdown(f"#### {row['name']} · {row['team']}")
    bits = [core.roster_label(row),
            f"{int(row['seasons'])} seasons", f"{row['start_history']:.0f} career starts"]
    if row.get("rookie"):
        bits.append("rookie prior")
    if edits:
        bits.append(core.edit_badge(len(edits)))
    st.caption(" · ".join(bits))

    metrics = [("Starts", core.num(row["proj_starts"], 1), "proj_starts"),
               ("Wins", core.num(row["proj_wins"], 1), "proj_wins"),
               ("SV%", core.sv(row["proj_save_pct"]), None),
               ("GAA", core.num(row["proj_gaa"], 2), None),
               ("Shutouts", core.num(row["proj_shutouts"], 1), None),
               ("Saves", core.num(row["proj_saves"], 0), None)]
    for col, (label, value, key) in zip(st.columns(len(metrics)), metrics):
        delta = None
        if key and b is not None and abs(float(row[key]) - float(b[key])) >= 0.05:
            delta = f"{float(row[key]) - float(b[key]):+.1f} vs model"
        col.metric(label, value, delta=delta)
    st.caption(
        f"Starts {core.band(row['starts_p10'], row['starts_p90'])} · wins "
        f"{core.band(row['wins_p10'], row['wins_p90'])} · SV% "
        f"{core.sv(row['save_pct_p10'])} to {core.sv(row['save_pct_p90'])} · GAA "
        f"{core.num(row['gaa_p90'], 2)} to {core.num(row['gaa_p10'], 2)}. The starts band "
        "is a measured quantile, not a symmetric error bar — it prices in both losing the "
        "job and inheriting it.")

    tab_edit, tab_room, tab_hist, tab_how = st.tabs(
        ["Edit", "The room", "History", "How it was built"])
    with tab_edit:
        _edit_form(row, edits, g)
    with tab_room:
        _room(row, g, gb)
    with tab_hist:
        _history(pid)
    with tab_how:
        _provenance(row, gb)


def _edit_form(row: pd.Series, edits: dict, g: pd.DataFrame) -> None:
    pid = int(row["playerId"])
    st.markdown("**Starts and rate.** Starts is the depth-chart decision — set it and his "
                "teammates split what is left of the team's 84 games. Save percentage "
                "flows through to goals against, saves and GAA together.")
    c1, c2, c3 = st.columns([1, 1, 1.4])
    starts = c1.number_input("Starts", 0.0, float(core.C.SEASON_GAMES),
                             float(row["proj_starts"]), 1.0, key=f"gs{pid}")
    svp = c2.number_input("Save percentage", 0.820, 0.960,
                          float(row["proj_save_pct"]), 0.001, format="%.4f",
                          key=f"sv{pid}")
    teams = core.team_options(g)
    cur_team = str(row["team"])
    idx = teams.index(cur_team) + 1 if cur_team in teams else 0
    team = c3.selectbox("Team", ["(not on a roster)"] + teams, index=idx, key=f"gtm{pid}")

    st.markdown("**Season totals.** Stated outright and honoured exactly, taken out of the "
                "team's goaltending budget before the others are served.")
    locks: dict[str, float | None] = {}
    cols = st.columns(3)
    for i, (field, label, cap) in enumerate(LOCKS):
        cur = edits.get(field)
        proj = float(row.get(f"proj_{field}", 0.0))
        locks[field] = cols[i % 3].number_input(
            label, min_value=0.0, max_value=cap,
            value=float(cur) if cur is not None else None,
            step=1.0, placeholder=f"model says {proj:.0f}", key=f"glk{field}{pid}")

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Save edits", type="primary", key=f"gsave{pid}"):
        patch: dict = {}
        if abs(starts - float(row["proj_starts"])) >= 0.01:
            patch["starts"] = float(starts)
        if abs(svp - float(row["proj_save_pct"])) >= 0.0001:
            patch["save_pct"] = float(svp)
        want_team = None if team.startswith("(") else team
        if want_team != (cur_team if cur_team in teams else None):
            patch["team"] = want_team or cur_team
            patch["on_roster"] = want_team is not None
        for field, v in locks.items():
            old = edits.get(field)
            if v is None and old is not None:
                patch[field] = None
            elif v is not None and (old is None or abs(float(old) - float(v)) >= 0.01):
                patch[field] = float(v)
        if patch:
            core.edit_goalie(pid, **patch)
        else:
            st.info("Nothing to save.")
    if edits and b2.button("Back to the model", key=f"gclr{pid}"):
        core.commit(core.scenario().clear_goalie(pid), f"{row['name']} back to the model")
    if edits:
        st.caption("Currently overriding: "
                   + ", ".join(f"{k} = {v}" for k, v in sorted(edits.items())))


def _room(row: pd.Series, g: pd.DataFrame, gb: pd.DataFrame) -> None:
    """Who else is in the crease. A goalie edit is really an argument about this table."""
    team = str(row["team"])
    mates = g[(g["team"] == team) & g["on_roster"]].sort_values(
        "proj_starts", ascending=False)
    if mates.empty:
        st.info("Not on a roster, so there is no depth chart to split.")
        return
    show = pd.DataFrame({
        "Goalie": mates["name"], "Share of team starts": mates["claim_start_share"],
        "Starts": mates["proj_starts"], "Floor": mates["starts_p10"],
        "Ceiling": mates["starts_p90"], "SV%": mates["proj_save_pct"],
        "Wins": mates["proj_wins"],
    })
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "Share of team starts": st.column_config.ProgressColumn(
            format="%.3f", min_value=0.0, max_value=1.0),
        "Starts": st.column_config.NumberColumn(format="%.1f"),
        "Floor": st.column_config.NumberColumn(format="%.0f"),
        "Ceiling": st.column_config.NumberColumn(format="%.0f"),
        "SV%": st.column_config.NumberColumn(format="%.4f"),
        "Wins": st.column_config.NumberColumn(format="%.1f")})
    if team in gb.index:
        row_b = gb.loc[team]
        left = float(row_b["depth_starts"])
        st.caption(
            f"{team} has {float(row_b['games']):.0f} games to give. Listed goalies cover "
            f"{float(row_b['coverage']):.1%} of them; {left:.0f} starts are held back for "
            "goalies not yet on the roster — that is the model refusing to hand a whole "
            "season to two names when a real team uses three.")
        if float(row_b["coverage"]) < 0.75:
            st.warning(f"{team} lists too few goalies for a full season, so every listed "
                       "goalie here is projected light. Add the missing goalie on his own "
                       "page (set his team) or state starts directly.")


def _history(pid: int) -> None:
    h = core.goalie_history()
    h = h[h["playerId"] == pid].sort_values("season", ascending=False)
    if h.empty:
        st.info("No NHL history — this projection comes from the rookie prior.")
        return
    show = pd.DataFrame({
        "Season": h["season"].astype(int).astype(str) + "-"
                  + (h["season"].astype(int) + 1).astype(str).str[-2:],
        "Team": h["team"], "GP": h["gp"], "GS": h["starts"], "W": h["wins"],
        "L": h["losses"], "OTL": h["otl"], "SV%": h["save_pct"], "GAA": h["gaa"],
        "SO": h["shutouts"], "SA": h["shots_against"], "GA": h["goals_against"],
        "Min": h["minutes"],
    })
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "SV%": st.column_config.NumberColumn(format="%.4f"),
        "GAA": st.column_config.NumberColumn(format="%.2f"),
        "Min": st.column_config.NumberColumn(format="%.0f")})
    st.caption("A traded goalie's row lists every team he played for; the split is not in "
               "the source data.")


def _provenance(row: pd.Series, gb: pd.DataFrame) -> None:
    st.caption("Same two steps as a skater: an unconstrained claim from his own rates, "
               "then settlement against the team's 84 games and its share of league wins.")
    rows = []
    for field, label in [("starts", "Starts"), ("gp", "Appearances"),
                         ("minutes", "Minutes"), ("shots_against", "Shots against"),
                         ("goals_against", "Goals against"), ("save_pct", "Save %")]:
        unc, proj = row.get(f"unc_{field}"), row.get(f"proj_{field}")
        if unc is None or proj is None or pd.isna(unc) or pd.isna(proj):
            continue
        rows.append({"Stat": label, "Claim": float(unc), "Projection": float(proj),
                     "Change": float(proj) - float(unc)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Claim": st.column_config.NumberColumn(format="%.2f"),
        "Projection": st.column_config.NumberColumn(format="%.2f"),
        "Change": st.column_config.NumberColumn(format="%+.2f")})
    st.caption(
        f"Rates behind it — goals per shot {row['rate_ga_per_shot']:.4f} (SV% "
        f"{1 - float(row['rate_ga_per_shot']):.4f}), shots against per 60 "
        f"{row['rate_sa_per_60']:.1f}, shutouts per start {row['rate_so_per_start']:.3f}, "
        f"relief appearances per start {row['rate_relief_per_start']:.3f}. Quality index "
        f"{row['quality']:.3f}, and quality is what moves starts within a team: a goalie "
        "5% better than his partner takes about 5 more starts of 82. Sample "
        f"{row['sample_shots']:,.0f} shots at {core.sv(row['sample_save_pct'])}, "
        f"{row['min_per_gp']:.1f} minutes per appearance. His projected save percentage sits "
        "a point or two off his own rate because it is settled against what his team is "
        "expected to concede — that gap is the shot quality he actually faces.")


# --------------------------------------------------------------------------- #
def page() -> None:
    core.scenario_bar()
    g, gb = core.goalies()
    on = g[g["on_roster"]]
    core.header(f"Goalies · {core.SEASON_LABEL}",
                f"{len(on)} on an NHL roster covering "
                f"{float(gb['coverage'].mean()):.0%} of the league's goaltending. "
                "Click a row to open a goalie.")

    df = _filters(g).sort_values("proj_wins", ascending=False).reset_index(drop=True)
    pos = _grid(df)

    c1, c2 = st.columns([1.2, 4])
    c1.download_button("Download this view (CSV)",
                       df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"goalies_{core.SEASON_LABEL}.csv", mime="text/csv",
                       width="stretch")
    c2.caption(f"{len(df)} goalies shown · {int(df['edited'].sum())} of them edited · "
               f"projected starts total {float(g['proj_starts'].sum()):,.0f} of "
               f"{float(gb['starts'].sum()):,.0f} available")

    thin = gb[gb["coverage"] < 0.75]
    if len(thin):
        st.caption("Thin creases (fewer listed goalies than a season needs): "
                   + ", ".join(f"{t} {float(gb.at[t, 'coverage']):.0%}"
                               for t in thin.index))

    st.divider()
    if pos is None:
        st.info("Select a goalie above to see his history, his depth chart and edit him.")
        return
    base, _ = core.baseline_goalies()
    _goalie(df.loc[pos], g, gb, base)
