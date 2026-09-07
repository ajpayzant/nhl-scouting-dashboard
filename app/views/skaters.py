"""The skater board, and the player page it opens onto.

This is the workbook sheet people actually used, with the two things it could not do:
it filters, and it can be argued with. Editing lives in two places on purpose -- a
spreadsheet-style grid for a run of games-played or ice-time changes, and a player page
for the one player worth thinking about carefully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import core

VIEWS: dict[str, list[str]] = {
    "Scoring": ["proj_gp", "proj_toi_per_gp", "proj_points", "points_p10", "points_p90",
                "proj_goals", "proj_assists", "proj_shots", "proj_pp_points"],
    "Per 60": ["proj_toi_per_gp", "rate_points", "rate_goals", "rate_primaryAssists",
               "rate_secondaryAssists", "rate_shots", "rate_ixg", "rate_pp_points"],
    "Physical": ["proj_gp", "proj_toi_per_gp", "proj_blocks", "proj_hits", "proj_pim",
                 "proj_faceoffs_won", "proj_points"],
    "Usage": ["proj_gp", "proj_toi_per_gp", "proj_pp_toi_per_gp", "proj_sh_toi_per_gp",
              "proj_toi", "proj_points", "proj_pp_points", "proj_sh_points"],
    "Shooting": ["proj_gp", "proj_shots", "proj_goals", "proj_ixg", "proj_points",
                 "proj_toi_per_gp"],
}

LABELS = {
    "proj_gp": "GP", "proj_toi_per_gp": "TOI/GP", "proj_toi": "TOI",
    "proj_pp_toi_per_gp": "PP/GP", "proj_sh_toi_per_gp": "SH/GP",
    "proj_points": "PTS", "points_p10": "PTS floor", "points_p90": "PTS ceiling",
    "proj_goals": "G", "proj_assists": "A", "proj_shots": "SOG", "proj_ixg": "ixG",
    "proj_pp_points": "PPP", "proj_sh_points": "SHP", "proj_blocks": "BLK",
    "proj_hits": "HIT", "proj_pim": "PIM", "proj_faceoffs_won": "FOW",
    # The per-60 rates the projection is actually built from.
    "rate_points": "PTS/60", "rate_goals": "G/60", "rate_primaryAssists": "A1/60",
    "rate_secondaryAssists": "A2/60", "rate_shots": "SOG/60", "rate_ixg": "ixG/60",
    "rate_pp_points": "PPP/60",
}
ONE_DP = {"proj_gp", "proj_toi_per_gp", "proj_pp_toi_per_gp", "proj_sh_toi_per_gp",
          "proj_ixg"}

# Season totals a reader can state outright. Games and ice time are inputs and get their
# own controls; these are the "he scores 40" edits, honoured exactly.
LOCKS = ["goals", "assists", "points", "shots", "pp_points", "sh_points",
         "blocks", "hits", "pim", "faceoffs_won"]


def _filters(sk: pd.DataFrame) -> pd.DataFrame:
    c1, c2, c3, c4 = st.columns([2.2, 1.7, 1.4, 1.5])
    q = c1.text_input("Search", placeholder="Search a name",
                      label_visibility="collapsed")
    teams = c2.multiselect("Team", core.team_options(sk), placeholder="All teams",
                           label_visibility="collapsed")
    pos = c3.multiselect("Position", ["C", "L", "R", "D"], placeholder="All positions",
                         label_visibility="collapsed")
    who = c4.selectbox("Who", ["On a roster", "Everyone", "In a camp", "Unsigned only",
                               "Edited only"], label_visibility="collapsed")

    df = sk.copy()
    df["_camp"] = df["camp"].fillna(False) if "camp" in df else False
    # A camp player answers to his camp's name in the team filter. He has no team in the
    # projection, but "show me EDM" should still find the players Edmonton's depth chart
    # pushed off the roster -- they are exactly who a reader looks for when he disagrees.
    ct = df["camp_team"].fillna("") if "camp_team" in df else ""
    df["_home"] = df["team"].mask(df["_camp"] & (ct != ""), ct)
    if q:
        df = df[df["name"].str.contains(q, case=False, na=False)]
    if teams:
        df = df[df["_home"].isin(teams)]
    if pos:
        df = df[df["position"].isin(pos)]
    if who == "On a roster":
        df = df[df["on_roster"]]
    elif who == "In a camp":
        df = df[df["_camp"]]
    elif who == "Unsigned only":
        df = df[~df["on_roster"] & ~df["_camp"]]
    elif who == "Edited only":
        df = df[df["edited"]]
    return df


def _grid(df: pd.DataFrame, cols: list[str]) -> int | None:
    show = df[["name", "team", "position", *cols]].copy()
    show.insert(0, " ", np.where(df["edited"], "✎", ""))
    cfg = {" ": st.column_config.TextColumn(" ", width="small", help="edited"),
           "name": st.column_config.TextColumn("Player", width="medium"),
           "team": st.column_config.TextColumn("Team", width="small"),
           "position": st.column_config.TextColumn("Pos", width="small")}
    for c in cols:
        fmt = "%.2f" if c.startswith("rate_") else "%.1f" if c in ONE_DP else "%.0f"
        cfg[c] = st.column_config.NumberColumn(LABELS.get(c, c), format=fmt)
    ev = st.dataframe(show, hide_index=True, column_config=cfg, height=440,
                      width="stretch", on_select="rerun", selection_mode="single-row")
    rows = ev.selection.get("rows") or []
    return int(rows[0]) if rows else None


# --------------------------------------------------------------------------- #
# bulk edits                                                                  #
# --------------------------------------------------------------------------- #
def _bulk_editor(df: pd.DataFrame) -> None:
    """Spreadsheet-style edits over the players currently filtered, applied on a click."""
    sc = core.scenario()
    n = st.slider("How many rows", 10, 200, 40, 10, key="bulk_n")
    src = df.head(n)
    grid = pd.DataFrame({
        "Player": src["name"].to_numpy(),
        "Team": src["team"].to_numpy(),
        "GP": src["proj_gp"].round(0).to_numpy(),
        "TOI/GP": src["proj_toi_per_gp"].round(2).to_numpy(),
        "PP/GP": src["proj_pp_toi_per_gp"].round(2).to_numpy(),
        "Lock PTS": [sc.player(p).get("points") for p in src["playerId"]],
        "PTS now": src["proj_points"].round(1).to_numpy(),
    })
    edited = st.data_editor(
        grid, hide_index=True, key="bulk", width="stretch", height=320,
        disabled=["Player", "Team", "PTS now"],
        column_config={
            "GP": st.column_config.NumberColumn(
                min_value=0.0, max_value=float(core.C.MAX_GP), step=1.0, format="%.0f"),
            "TOI/GP": st.column_config.NumberColumn(
                min_value=0.0, max_value=30.0, step=0.25, format="%.2f"),
            "PP/GP": st.column_config.NumberColumn(
                min_value=0.0, max_value=8.0, step=0.25, format="%.2f"),
            "Lock PTS": st.column_config.NumberColumn(
                min_value=0.0, max_value=200.0, step=1.0, format="%.0f",
                help="state a season total outright; it comes out of the team's budget "
                     "first and everyone else settles around what is left"),
            "PTS now": st.column_config.NumberColumn(
                format="%.1f", help="the current projection, for reference"),
        })

    if not st.button("Apply grid edits", type="primary", key="bulk_apply"):
        return
    fields = {"GP": "gp", "TOI/GP": "toi_per_gp", "PP/GP": "pp_toi_per_gp",
              "Lock PTS": "points"}
    changed, sc2 = 0, sc
    for i, pid in enumerate(src["playerId"].to_numpy()):
        patch = {}
        for col, field in fields.items():
            old, new = grid.at[i, col], edited.at[i, col]
            if _same(old, new):
                continue
            patch[field] = None if pd.isna(new) else float(new)
        if patch:
            sc2 = sc2.set_player(int(pid), **patch)
            changed += 1
    if changed:
        core.commit(sc2, f"Applied edits to {changed} player{'s' if changed != 1 else ''}")
    else:
        st.info("Nothing in the grid changed.")


def _same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) < 1e-9


# --------------------------------------------------------------------------- #
# player page                                                                 #
# --------------------------------------------------------------------------- #
def _player(row: pd.Series, base: pd.DataFrame) -> None:
    pid = int(row["playerId"])
    edits = core.scenario().player(pid)
    b = base[base["playerId"] == pid]
    b = b.iloc[0] if len(b) else None

    st.markdown(f"#### {row['name']} · {row['team']} {row['position']}")
    bits = [core.roster_label(row),
            f"{int(row['seasons'])} seasons of history"]
    if row.get("rookie"):
        bits.append("rookie prior")
    if edits:
        bits.append(core.edit_badge(len(edits)))
    st.caption(" · ".join(bits))

    metrics = [("Games", "proj_gp", 0), ("TOI/GP", "proj_toi_per_gp", 1),
               ("Points", "proj_points", 1), ("Goals", "proj_goals", 1),
               ("Assists", "proj_assists", 1), ("Shots", "proj_shots", 0)]
    for col, (label, key, nd) in zip(st.columns(len(metrics)), metrics):
        delta = None
        if b is not None and abs(float(row[key]) - float(b[key])) >= 0.05:
            delta = f"{float(row[key]) - float(b[key]):+.1f} vs model"
        col.metric(label, core.num(row[key], nd), delta=delta)
    st.caption(f"Points floor to ceiling {core.band(row['points_p10'], row['points_p90'])}"
               f" · games {core.band(row['gp_p10'], row['gp_p90'])}")

    tab_edit, tab_hist, tab_how = st.tabs(["Edit", "History", "How it was built"])
    with tab_edit:
        _edit_form(row, edits)
    with tab_hist:
        _history(pid)
    with tab_how:
        _provenance(row)


def _edit_form(row: pd.Series, edits: dict) -> None:
    pid = int(row["playerId"])
    st.markdown("**Inputs.** These flow through the model: raise his ice time and his "
                "goals, shots and blocks all move, and his teammates give up the minutes "
                "he gained.")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
    gp = c1.number_input("Games played", 0.0, float(core.C.MAX_GP),
                         float(row["proj_gp"]), 1.0, key=f"gp{pid}")
    toi = c2.number_input("TOI per game", 0.0, 30.0, float(row["proj_toi_per_gp"]),
                          0.25, key=f"toi{pid}")
    pp = c3.number_input("PP TOI per game", 0.0, 8.0, float(row["proj_pp_toi_per_gp"]),
                         0.25, key=f"pp{pid}")
    teams = core.team_options(core.skaters()[0])
    cur_team = str(row["team"])
    idx = teams.index(cur_team) + 1 if cur_team in teams else 0
    team = c4.selectbox("Team", ["(not on a roster)"] + teams, index=idx, key=f"tm{pid}",
                        help="a signing or a trade the published roster has not caught up "
                             "with yet")

    st.markdown("**Season totals.** A number stated here is honoured exactly and comes out "
                "of the team's budget before anyone else is served. Leave one blank for "
                "the model's own opinion.")
    locks: dict[str, float | None] = {}
    cols = st.columns(5)
    for i, stat in enumerate(LOCKS):
        cur = edits.get(stat)
        proj = float(row.get(f"proj_{stat}", 0.0))
        locks[stat] = cols[i % 5].number_input(
            stat.replace("_", " ").title(), min_value=0.0, max_value=400.0,
            value=float(cur) if cur is not None else None, step=1.0,
            placeholder=f"model says {proj:.0f}", key=f"lk{stat}{pid}")

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Save edits", type="primary", key=f"save{pid}"):
        patch: dict = {}
        for value, field, now in [(gp, "gp", row["proj_gp"]),
                                  (toi, "toi_per_gp", row["proj_toi_per_gp"]),
                                  (pp, "pp_toi_per_gp", row["proj_pp_toi_per_gp"])]:
            if abs(float(value) - float(now)) >= 0.01:
                patch[field] = float(value)
        want_team = None if team.startswith("(") else team
        if want_team != (cur_team if cur_team in teams else None):
            patch["team"] = want_team or cur_team
            patch["on_roster"] = want_team is not None
        for stat, v in locks.items():
            old = edits.get(stat)
            if v is None and old is not None:
                patch[stat] = None
            elif v is not None and (old is None or abs(float(old) - float(v)) >= 0.01):
                patch[stat] = float(v)
        if patch:
            core.edit_player(pid, **patch)
        else:
            st.info("Nothing to save.")
    if edits and b2.button("Back to the model", key=f"clr{pid}"):
        core.commit(core.scenario().clear_player(pid),
                    f"{row['name']} back to the model")
    if edits:
        st.caption("Currently overriding: "
                   + ", ".join(f"{k} = {v}" for k, v in sorted(edits.items())))


def _history(pid: int) -> None:
    h = core.skater_history()
    h = h[h["playerId"] == pid].sort_values("season", ascending=False)
    if h.empty:
        st.info("No NHL history — this projection comes from the rookie prior.")
        return
    cols = ["season", "team", "gp", "toi_per_gp", "goals", "assists", "points", "shots",
            "ixg", "blocks", "hits", "pim", "faceoffs_won"]
    show = h[[c for c in cols if c in h.columns]].copy()
    show["season"] = (show["season"].astype(int).astype(str) + "-"
                      + (show["season"].astype(int) + 1).astype(str).str[-2:])
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "season": st.column_config.TextColumn("Season"),
        "team": st.column_config.TextColumn("Team"),
        "gp": st.column_config.NumberColumn("GP", format="%.0f"),
        "toi_per_gp": st.column_config.NumberColumn("TOI/GP", format="%.1f"),
        "goals": st.column_config.NumberColumn("G", format="%.0f"),
        "assists": st.column_config.NumberColumn("A", format="%.0f"),
        "points": st.column_config.NumberColumn("PTS", format="%.0f"),
        "shots": st.column_config.NumberColumn("SOG", format="%.0f"),
        "ixg": st.column_config.NumberColumn("ixG", format="%.1f"),
        "blocks": st.column_config.NumberColumn("BLK", format="%.0f"),
        "hits": st.column_config.NumberColumn("HIT", format="%.0f"),
        "pim": st.column_config.NumberColumn("PIM", format="%.0f"),
        "faceoffs_won": st.column_config.NumberColumn("FOW", format="%.0f")})
    st.caption("Weighted by recency and adjusted for age before it becomes a rate; the "
               "most recent season carries the most weight.")


def _provenance(row: pd.Series) -> None:
    """Claim, then settlement — the two numbers whose difference is the team budget."""
    st.caption("A projection is a CLAIM — what this player's own rates ask for — settled "
               "against what his team has to give. The gap is the budget at work, and a "
               "big negative gap means his team is oversubscribed at that stat.")
    rows = []
    for stat, label in [("gp", "Games"), ("toi", "Ice time (min)"), ("goals", "Goals"),
                        ("primaryAssists", "Primary assists"),
                        ("secondaryAssists", "Secondary assists"), ("points", "Points"),
                        ("shots", "Shots"), ("ixg", "Expected goals"),
                        ("pp_points", "PP points"), ("sh_points", "SH points"),
                        ("blocks", "Blocks"), ("hits", "Hits"), ("pim", "PIM"),
                        ("faceoffs_won", "Faceoffs won")]:
        unc, proj = row.get(f"unc_{stat}"), row.get(f"proj_{stat}")
        if unc is None or proj is None or pd.isna(unc) or pd.isna(proj):
            continue
        unc, proj = float(unc), float(proj)
        locked = bool(row.get(f"lock_{stat}", False))
        rows.append({"Stat": label + (" (locked)" if locked else ""),
                     "Claim": unc, "Projection": proj, "Change": proj - unc,
                     "%": (proj / unc - 1.0) * 100.0 if unc else np.nan})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Claim": st.column_config.NumberColumn(format="%.1f"),
        "Projection": st.column_config.NumberColumn(format="%.1f"),
        "Change": st.column_config.NumberColumn(format="%+.1f"),
        "%": st.column_config.NumberColumn("Change %", format="%+.1f%%")})
    st.caption(
        f"Per-60 rates behind it — goals {row['rate_goals']:.2f}, primary assists "
        f"{row['rate_primaryAssists']:.2f}, shots {row['rate_shots']:.2f}, blocks "
        f"{row['rate_blocks']:.2f}, hits {row['rate_hits']:.2f}. Usage tier "
        f"{int(row['usage_tier'])} of 4 · games-played reliability "
        f"{row['gp_reliability']:.2f} · sample {row['sample_toi_min']:,.0f} minutes "
        f"at {row['sample_toi_per_gp']:.1f}/game.")


# --------------------------------------------------------------------------- #
def page() -> None:
    core.scenario_bar()
    sk, _ = core.skaters()
    camp = int(sk.get("camp", pd.Series(False, index=sk.index)).sum())
    core.header(f"Skaters · {core.SEASON_LABEL}",
                f"{int(sk['on_roster'].sum())} on an NHL roster, {camp} in a camp who did "
                f"not make one, {int((~sk['on_roster']).sum()) - camp} unsigned. "
                "Click a row to open a player.")

    view = st.radio("Columns", list(VIEWS), horizontal=True, label_visibility="collapsed")
    df = _filters(sk)
    df = df.sort_values("rate_points" if view == "Per 60" else "proj_points",
                        ascending=False).reset_index(drop=True)
    pos = _grid(df, VIEWS[view])

    c1, c2 = st.columns([1.2, 4])
    c1.download_button("Download this view (CSV)",
                       df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"skaters_{core.SEASON_LABEL}.csv", mime="text/csv",
                       width="stretch")
    c2.caption(f"{len(df)} players shown · {int(df['edited'].sum())} of them edited")

    with st.expander("Bulk edit these players (games, ice time, a points lock)"):
        _bulk_editor(df)

    st.divider()
    if pos is None:
        st.info("Select a player above to see his history and edit him.")
        return
    base, _ = core.baseline_skaters()
    _player(df.loc[pos], base)
