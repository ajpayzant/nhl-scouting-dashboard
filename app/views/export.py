"""Export: the workbook demoted from interface to output.

The spreadsheet was never the problem -- people genuinely want a file they can mail, sort
and paste into a slide. What it could not do was recompute, so an override meant editing a
number whose dependents did not move. So it stays, as an export of whatever the app is
currently showing, edits and all.
"""
from __future__ import annotations

import datetime as dt
from io import BytesIO

import pandas as pd
import streamlit as st

import core

SKATER_COLS = [
    ("name", "Player", 22, None), ("team", "Team", 7, None),
    ("position", "Pos", 6, None), ("target_age", "Age", 6, "0.0"),
    ("proj_gp", "GP", 7, "0.0"), ("proj_toi_per_gp", "TOI/GP", 9, "0.0"),
    ("proj_goals", "G", 7, "0.0"), ("proj_assists", "A", 7, "0.0"),
    ("proj_points", "PTS", 8, "0.0"), ("points_p10", "PTS floor", 10, "0"),
    ("points_p90", "PTS ceiling", 11, "0"), ("proj_shots", "SOG", 8, "0"),
    ("proj_ixg", "ixG", 8, "0.0"), ("proj_pp_points", "PPP", 8, "0.0"),
    ("proj_sh_points", "SHP", 8, "0.0"), ("proj_blocks", "BLK", 8, "0"),
    ("proj_hits", "HIT", 8, "0"), ("proj_pim", "PIM", 8, "0"),
    ("proj_faceoffs_won", "FOW", 8, "0"),
    # The rates the projection is built from, so a reader can see the engine's view of the
    # player and not just its arithmetic.
    ("rate_points", "PTS/60", 9, "0.00"), ("rate_goals", "G/60", 8, "0.00"),
    ("rate_shots", "SOG/60", 9, "0.00"), ("rate_ixg", "ixG/60", 9, "0.00"),
    ("edited", "Edited", 8, None),
]
GOALIE_COLS = [
    ("name", "Goalie", 22, None), ("team", "Team", 7, None),
    ("target_age", "Age", 6, "0.0"), ("proj_starts", "GS", 7, "0.0"),
    ("starts_p10", "GS floor", 10, "0"), ("starts_p90", "GS ceiling", 11, "0"),
    ("proj_gp", "GP", 7, "0.0"), ("proj_wins", "W", 7, "0.0"),
    ("wins_p10", "W floor", 9, "0"), ("wins_p90", "W ceiling", 10, "0"),
    ("proj_losses", "L", 7, "0.0"), ("proj_otl", "OTL", 7, "0.0"),
    ("proj_save_pct", "SV%", 8, "0.000"), ("proj_gaa", "GAA", 8, "0.00"),
    ("proj_shutouts", "SO", 7, "0.0"), ("proj_saves", "SV", 8, "0"),
    ("proj_shots_against", "SA", 8, "0"), ("claim_start_share", "Start share", 12, "0.000"),
    ("rate_sa_per_60", "SA/60", 9, "0.00"), ("edited", "Edited", 8, None),
]


def _sheet(wb, name: str, df: pd.DataFrame, cols, fmts) -> None:
    ws = wb.add_worksheet(name)
    ws.freeze_panes(1, 1)
    for i, (col, header, width, numfmt) in enumerate(cols):
        ws.set_column(i, i, width, fmts.get(numfmt))
        ws.write(0, i, header, fmts["head"])
    for r, (_, row) in enumerate(df.iterrows(), start=1):
        for i, (col, _h, _w, _n) in enumerate(cols):
            v = row.get(col)
            if pd.isna(v):
                ws.write_blank(r, i, None)
            elif isinstance(v, (bool,)):
                ws.write(r, i, "yes" if v else "")
            else:
                ws.write(r, i, v)
    ws.autofilter(0, 0, len(df), len(cols) - 1)


def _workbook(sk: pd.DataFrame, tb: pd.DataFrame, g: pd.DataFrame,
              gb: pd.DataFrame) -> bytes:
    import xlsxwriter
    buf = BytesIO()
    wb = xlsxwriter.Workbook(buf, {"nan_inf_to_errors": True, "in_memory": True})
    base = {"font_name": "Calibri", "font_size": 11}
    fmts = {
        "head": wb.add_format({**base, "bold": True, "font_color": "white",
                               "bg_color": "#1F3864", "align": "left", "bottom": 1}),
        "title": wb.add_format({**base, "bold": True, "font_size": 16,
                                "font_color": "#1F3864"}),
        "note": wb.add_format({**base, "text_wrap": True, "valign": "top"}),
        None: wb.add_format(base),
    }
    for nf in ("0", "0.0", "0.00", "0.000"):
        fmts[nf] = wb.add_format({**base, "num_format": nf})

    sc = core.scenario()
    n = sc.count()
    ws = wb.add_worksheet("Read me")
    ws.set_column(0, 0, 100)
    ws.write(0, 0, f"NHL season projections · {core.SEASON_LABEL}", fmts["title"])
    lines = [
        f"Exported {dt.datetime.now():%d %b %Y %H:%M} from the projection app.",
        "",
        f"Scenario: {'the baseline model, no edits' if sc.is_baseline else str(n['edits']) + ' edits'}"
        + ("" if sc.is_baseline else
           f" — {n['players']} skaters, {n['goalies']} goalies, {n['teams']} team budgets."),
        "",
        "Every projection is a claim settled against a team budget: a player's own per-60 "
        "rates say what he asks for, and his team's expected totals say what there is to "
        "give. Floors and ceilings are the 10th and 90th percentile, so a player should "
        "beat his ceiling about one season in ten.",
        "",
        "Listed rosters do not cover a whole season — a real team uses about 28 skaters and "
        "3 goalies where a published roster lists 22 and 2.6 — so part of every team's "
        "budget is deliberately held back for players not yet named. That is why these "
        "totals sit below what a full-season workload would imply.",
        "",
        "This file is an export. It cannot recompute: changing a number here changes "
        "nothing else. Make edits in the app, which resettles the whole team, and export "
        "again.",
    ]
    for i, line in enumerate(lines, start=2):
        ws.write(i, 0, line, fmts["note"])

    on_sk = sk[sk["on_roster"]].sort_values("proj_points", ascending=False)
    on_g = g[g["on_roster"]].sort_values("proj_wins", ascending=False)
    _sheet(wb, "Skaters", on_sk, SKATER_COLS, fmts)
    _sheet(wb, "Goalies", on_g, GOALIE_COLS, fmts)

    teams = pd.DataFrame({
        "team": tb.index,
        "games": tb["games"].to_numpy(),
        "goals_budget": tb["goals"].to_numpy(),
        "points_budget": tb["points"].to_numpy(),
        "toi_coverage": tb["toi_coverage"].to_numpy(),
        "minutes_held_back": tb["depth_toi"].to_numpy(),
        "goalie_coverage": gb["coverage"].reindex(tb.index).to_numpy(),
        "starts_held_back": gb["depth_starts"].reindex(tb.index).to_numpy(),
        "team_sv_pct": gb["sv_pct"].reindex(tb.index).to_numpy(),
    })
    teams["goals_projected"] = [float(on_sk.loc[on_sk["team"] == t, "proj_goals"].sum())
                                for t in tb.index]
    teams["wins_projected"] = [float(on_g.loc[on_g["team"] == t, "proj_wins"].sum())
                               for t in tb.index]
    _sheet(wb, "Teams", teams.sort_values("points_budget", ascending=False), [
        ("team", "Team", 8, None), ("games", "Games", 8, "0"),
        ("goals_budget", "Goals budget", 13, "0"),
        ("goals_projected", "Goals projected", 15, "0"),
        ("points_budget", "Points budget", 14, "0"),
        ("toi_coverage", "Roster covers", 14, "0.000"),
        ("minutes_held_back", "Minutes held back", 18, "0"),
        ("goalie_coverage", "Crease covers", 14, "0.000"),
        ("starts_held_back", "Starts held back", 17, "0"),
        ("wins_projected", "Wins projected", 15, "0.0"),
        ("team_sv_pct", "Team SV%", 10, "0.000")], fmts)

    if not sc.is_baseline:
        rows = []
        names = dict(zip(sk["playerId"].astype(str), sk["name"]))
        names.update(dict(zip(g["playerId"].astype(str), g["name"])))
        for bucket, label in [("players", "Skater"), ("goalies", "Goalie"),
                              ("teams", "Team")]:
            for key, fields in getattr(sc, bucket).items():
                for field, value in sorted(fields.items()):
                    rows.append({"kind": label, "who": names.get(key, key),
                                 "field": field, "value": value})
        for key, value in sorted(sc.league.items()):
            rows.append({"kind": "League", "who": "all teams", "field": key,
                         "value": value})
        _sheet(wb, "Edits", pd.DataFrame(rows), [
            ("kind", "Kind", 10, None), ("who", "Who", 24, None),
            ("field", "Field", 18, None), ("value", "Value", 14, None)], fmts)

    wb.close()
    return buf.getvalue()


def page() -> None:
    core.scenario_bar()
    sk, tb = core.skaters()
    g, gb = core.goalies()
    sc = core.scenario()
    core.header("Export",
                "Take the current projection out — edits included — as a workbook or as "
                "flat CSVs.")

    st.markdown("**Workbook**")
    st.caption("Read me, skaters, goalies, team budgets, and a list of every edit in the "
               "scenario. Opens in Excel or Google Sheets; no formulas, so nothing breaks "
               "on the way across.")
    stamp = dt.datetime.now().strftime("%Y%m%d")
    tag = "baseline" if sc.is_baseline else "edited"
    if st.button("Build the workbook", type="primary"):
        with st.spinner("Writing the workbook ..."):
            st.session_state.xlsx = _workbook(sk, tb, g, gb)
    if st.session_state.get("xlsx"):
        st.download_button(
            "Download the workbook (.xlsx)", st.session_state.xlsx,
            file_name=f"NHL_projections_{core.SEASON_LABEL}_{tag}_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()
    st.markdown("**CSVs**")
    st.caption("Every column the model produces, for anything that wants to read the "
               "projection as data.")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Skaters (all columns)",
                       sk.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"skaters_{core.SEASON_LABEL}_{tag}.csv",
                       mime="text/csv", width="stretch")
    c2.download_button("Goalies (all columns)",
                       g.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"goalies_{core.SEASON_LABEL}_{tag}.csv",
                       mime="text/csv", width="stretch")
    c3.download_button("Team budgets",
                       tb.join(gb, rsuffix="_goalie").to_csv().encode("utf-8-sig"),
                       file_name=f"team_budgets_{core.SEASON_LABEL}_{tag}.csv",
                       mime="text/csv", width="stretch")
    st.caption("The repo also writes these on every `python run.py`, to the `output` "
               "folder, without any scenario edits applied.")
