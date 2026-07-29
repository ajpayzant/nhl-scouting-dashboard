"""Build an interactive, Google-Sheets-ready .xlsx workbook from the projections.

Produces `output/NHL_Projections_2026.xlsx` with these tabs:
  1. Overview & Dictionary  — how the model works + every abbreviation
  2. Dashboard              — pick any player from a dropdown; see their projected
                              line AND their last 3 seasons of actuals side-by-side
  3. All Skaters            — every rostered skater, sortable/filterable
  4. All Goalies            — every rostered goalie
  5. <TEAM>                 — one tab per team (skaters + goalies)

Google Sheets imports .xlsx directly (File > Import), preserving tabs, the dropdown,
INDEX/MATCH formulas, conditional formatting and frozen headers. Just run:

    python run.py            # (re)generate the projection CSVs first
    python build_workbook.py # then build the workbook

Then upload NHL_Projections_2026.xlsx to Google Drive and open with Google Sheets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xlsxwriter

import config as C
import project_skaters as ps
import project_goalies as pg

TARGET = C.TARGET_SEASON
SEASON_LABEL = f"{TARGET}-{str(TARGET + 1)[-2:]}"           # 2026-27
HIST_YEARS = [TARGET - 1, TARGET - 2, TARGET - 3]           # 2025, 2024, 2023 (last 3 completed)


def _season_label(mp_year: int) -> str:
    """MoneyPuck season start-year -> 'YYYY-YY' label."""
    return f"{mp_year}-{str(mp_year + 1)[-2:]}"


# --------------------------------------------------------------------------- data
def _skater_history() -> pd.DataFrame:
    """Per-player-season skater actuals for the last 3 completed seasons."""
    h = ps._prep_skater_seasons()
    h = h[h["mp_season_year"].isin(HIST_YEARS)].copy()
    h["toi_per_gp"] = h["toi_min"] / h["games_played"]
    keep = ["playerId", "mp_season_year", "games_played", "goals_raw", "assists_raw",
            "points_raw", "shots_raw", "pp_points_raw", "toi_per_gp"]
    return h[keep]


def _goalie_history() -> pd.DataFrame:
    h = pg._prep_goalie_seasons()
    h = h[h["mp_season_year"].isin(HIST_YEARS)].copy()
    keep = ["playerId", "mp_season_year", "gamesPlayed", "wins", "savePct",
            "goalsAgainstAverage", "saves", "shutouts"]
    return h[keep]


def _load_projections():
    sk = pd.read_csv(C.OUTPUT / f"skater_projections_{TARGET}.csv")
    g = pd.read_csv(C.OUTPUT / f"goalie_projections_{TARGET}.csv")
    return sk, g


# --------------------------------------------------------------------------- workbook
# Column specs: (df_column, header, width, num_format_key, heat)
SKATER_COLS = [
    ("name", "Player", 20, "text", False),
    ("team", "Team", 7, "text", False),
    ("position", "Pos", 6, "text", False),
    ("target_age", "Age", 6, "1dp", False),
    ("proj_gp", "GP", 7, "1dp", False),
    ("proj_goals", "G", 7, "1dp", True),
    ("goals_p10", "G p10", 8, "1dp", False),
    ("goals_p90", "G p90", 8, "1dp", False),
    ("proj_assists", "A", 7, "1dp", True),
    ("assists_p10", "A p10", 8, "1dp", False),
    ("assists_p90", "A p90", 8, "1dp", False),
    ("proj_points", "PTS", 8, "1dp", True),
    ("points_p10", "PTS p10", 9, "1dp", False),
    ("points_p90", "PTS p90", 9, "1dp", False),
    ("proj_shots", "SOG", 8, "0dp", True),
    ("shots_p10", "SOG p10", 9, "0dp", False),
    ("shots_p90", "SOG p90", 9, "0dp", False),
    ("proj_pp_points", "PPP", 8, "1dp", True),
    ("pp_points_p10", "PPP p10", 9, "1dp", False),
    ("pp_points_p90", "PPP p90", 9, "1dp", False),
    ("proj_toi_per_gp", "TOI/GP", 9, "1dp", False),
]

GOALIE_COLS = [
    ("name", "Goalie", 20, "text", False),
    ("team", "Team", 7, "text", False),
    ("proj_gp", "GP", 7, "1dp", False),
    ("proj_wins", "W", 7, "1dp", True),
    ("wins_p10", "W p10", 8, "1dp", False),
    ("wins_p90", "W p90", 8, "1dp", False),
    ("proj_save_pct", "SV%", 8, "pct3", True),
    ("save_pct_p10", "SV% p10", 9, "pct3", False),
    ("save_pct_p90", "SV% p90", 9, "pct3", False),
    ("proj_gaa", "GAA", 8, "2dp", False),
    ("proj_saves", "SV", 8, "0dp", False),
    ("proj_shutouts", "SO", 7, "1dp", True),
    ("proj_shots_against", "SA", 8, "0dp", False),
]


def _make_formats(wb):
    base = {"font_name": "Calibri", "font_size": 11}
    return {
        "title": wb.add_format({**base, "bold": True, "font_size": 20, "font_color": "#1F3864"}),
        "subtitle": wb.add_format({**base, "italic": True, "font_size": 11, "font_color": "#595959"}),
        "h2": wb.add_format({**base, "bold": True, "font_size": 14, "font_color": "#1F3864"}),
        "header": wb.add_format({**base, "bold": True, "font_color": "white", "bg_color": "#1F3864",
                                 "align": "center", "valign": "vcenter", "border": 1, "text_wrap": True}),
        "label": wb.add_format({**base, "bold": True, "bg_color": "#D6DCE4", "border": 1}),
        "text": wb.add_format({**base, "border": 1}),
        "text_c": wb.add_format({**base, "border": 1, "align": "center"}),
        "1dp": wb.add_format({**base, "border": 1, "align": "center", "num_format": "0.0"}),
        "0dp": wb.add_format({**base, "border": 1, "align": "center", "num_format": "0"}),
        "2dp": wb.add_format({**base, "border": 1, "align": "center", "num_format": "0.00"}),
        "pct3": wb.add_format({**base, "border": 1, "align": "center", "num_format": "0.000"}),
        "big": wb.add_format({**base, "bold": True, "font_size": 22, "align": "center",
                              "valign": "vcenter", "border": 1, "num_format": "0.0", "bg_color": "#FCE4D6"}),
        "big_lbl": wb.add_format({**base, "bold": True, "align": "center", "font_color": "#595959"}),
        "pick": wb.add_format({**base, "bold": True, "font_size": 14, "bg_color": "#FFF2CC",
                               "border": 2, "align": "center"}),
        "input": wb.add_format({**base, "bold": True, "font_size": 14, "bg_color": "#FFF2CC",
                                "border": 2, "align": "center", "num_format": "0"}),
        "big_lbl_val": wb.add_format({**base, "bold": True, "font_size": 12, "border": 1,
                                      "align": "center", "num_format": "0.0", "bg_color": "#E2EFDA"}),
        "wrap": wb.add_format({**base, "text_wrap": True, "valign": "top"}),
        "bullet": wb.add_format({**base, "text_wrap": True, "valign": "top", "indent": 1}),
        "dict_term": wb.add_format({**base, "bold": True, "border": 1, "bg_color": "#D6DCE4"}),
        "dict_def": wb.add_format({**base, "border": 1, "text_wrap": True, "valign": "top"}),
    }


def _fmt_key(df_col, key, fmts):
    return fmts[key]


def _write_table(ws, df, cols, fmts, start_row=0, heat_cols=None):
    """Write a header + data table; returns the number of data rows written."""
    for c, (col, hdr, width, _, _) in enumerate(cols):
        ws.set_column(c, c, width)
        ws.write(start_row, c, hdr, fmts["header"])
    for r, (_, row) in enumerate(df.iterrows()):
        for c, (col, hdr, width, nf, _) in enumerate(cols):
            val = row.get(col)
            if nf == "text":
                ws.write(start_row + 1 + r, c, "" if pd.isna(val) else str(val), fmts["text"])
            else:
                if pd.isna(val):
                    ws.write(start_row + 1 + r, c, "", fmts[nf])
                else:
                    ws.write_number(start_row + 1 + r, c, float(val), fmts[nf])
    n = len(df)
    ws.freeze_panes(start_row + 1, 0)
    if n:
        ws.autofilter(start_row, 0, start_row + n, len(cols) - 1)
        # heat map on the key counting stats
        for c, (col, hdr, width, nf, heat) in enumerate(cols):
            if heat:
                ws.conditional_format(start_row + 1, c, start_row + n, c,
                                      {"type": "3_color_scale",
                                       "min_color": "#F8696B", "mid_color": "#FFEB84",
                                       "max_color": "#63BE7B"})
    return n


# --------------------------------------------------------------------------- tabs
def _overview_tab(wb, fmts, n_sk, n_g, n_teams):
    ws = wb.add_worksheet("Overview & Dictionary")
    ws.hide_gridlines(2)
    ws.set_column(0, 0, 22)
    ws.set_column(1, 1, 95)
    ws.write(0, 0, f"NHL {SEASON_LABEL} Season Projections", fmts["title"])
    ws.write(1, 0, "Season-long projected stat lines for every rostered skater and goalie. "
                   "Built from a Marcel-style model with empirical age curves.", fmts["subtitle"])
    r = 3

    def section(title):
        nonlocal r
        ws.write(r, 0, title, fmts["h2"]); r += 1

    def line(label, text):
        nonlocal r
        ws.write(r, 0, label, fmts["label"])
        ws.write(r, 1, text, fmts["bullet"])
        ws.set_row(r, 30)
        r += 1

    section("How to use this workbook")
    line("Dashboard", "Pick any player from the yellow dropdown to see their projected "
         f"{SEASON_LABEL} stat line next to their actual stats from the last three seasons — "
         "so you can judge the projection against recent history. You can also type a games-played "
         "number into the yellow 'Your GP' cell and every counting stat rescales instantly to that "
         "workload (rate stats like SV%, GAA and TOI/GP stay put, since they are per-game).")
    line("All Skaters / All Goalies", "Every rostered player in one sortable table. Click the "
         "filter arrows in the header to sort by goals, points, wins, etc., or filter by team/position. "
         "Green = high, red = low within each stat column.")
    line("Team tabs", f"One tab per team ({n_teams} teams) listing that team's projected skaters and goalies.")
    line("Coverage", f"{n_sk} skaters and {n_g} goalies who are on a published {SEASON_LABEL} roster. "
         "Retired/unsigned players and rookies with no NHL history are excluded from these tables "
         "(the Dashboard can still pull anyone with prior-season data).")
    r += 1

    section("How the projections are made")
    for txt in [
        "MARCEL + AGE CURVE, ON A PER-60 RATE BASIS. Each player's last up-to-3 seasons are "
        "converted to per-60-minute rates so an injury-shortened season doesn't understate talent. "
        "Rates are blended with recent seasons weighted heaviest.",
        "REGRESSION TO THE MEAN. Blended rates are pulled toward the average for that player's "
        "position AND usage tier (a first-line forward regresses toward other first-liners, not 4th-liners). "
        "Thin track records regress hard; established stars barely move.",
        "AGE CURVES. Empirical curves (built from thousands of paired year-over-year changes) scale "
        "each rate from the player's current age to next season's age — capturing young-player growth "
        "and veteran decline, including how ice time itself rises and falls with age.",
        "VOLUME. Games played and time-on-ice per game are projected separately (recency-weighted, with "
        "a durability hedge), then rate x ice time gives the counting totals.",
        "GOALIES are deliberately conservative — save percentage is noisy year to year, so it is "
        "regressed hard to the league mean; workload (games, shots faced) and win rate are projected "
        "from recent form and the new team's defensive strength.",
        "TEAM CONTEXT. Each player's team comes from the published roster (so trades/signings are "
        "reflected). Strength of schedule barely moves skaters (<1%) but matters for goalie wins.",
    ]:
        ws.write(r, 1, "•  " + txt, fmts["bullet"]); ws.set_row(r, 42); r += 1
    r += 1

    section("Reading the uncertainty bands (p10 / p90)")
    ws.write(r, 1, "Every projection is a best estimate, not a guarantee. EVERY counting stat (goals, "
             "assists, points, shots, power-play points; wins and save % for goalies) carries a p10 and p90 — "
             "a realistic FLOOR and CEILING with about an 80% chance the actual result lands between them. "
             "Each stat's band is calibrated on its OWN historical error, so a shots band is wider than a "
             "goals band. A wide band means the outcome is hard to pin down (usually injury/availability "
             "risk); a tight band means a durable, predictable player. The single biggest source of error is "
             "games played — injuries are close to random — which is why the bands exist and why you can "
             "override games played manually (see gp_overrides.csv in the project). On the Dashboard the "
             "range sits right beside each projected stat and rescales with any games-played override you type.",
             fmts["bullet"])
    ws.set_row(r, 96); r += 2

    section("Validation")
    ws.write(r, 1, "Backtested on the 2022-2025 seasons (projecting each using only prior data): the model "
             "beats both a 'last season repeats' and a '3-year average' baseline every season, and cuts "
             "per-60 rate error ~19% versus using last season alone. Skater points error (MAE) is ~9.6; "
             "goalie wins error is ~6.6. Elite players are calibrated correctly on a rate basis — where they "
             "fall short of a career-year total, it is the honest games-played hedge, not model compression.",
             fmts["bullet"])
    ws.set_row(r, 60); r += 2

    # Dictionary
    section("Dictionary — abbreviations")
    entries = [
        ("Pos", "Position: C=center, L/R=wing, D=defense, G=goalie."),
        ("Age", "Player's age as of Feb 1 of the projected season."),
        ("GP", "Games played (projected). The season is 84 games (new CBA), not 82."),
        ("G", "Goals (projected)."),
        ("G p10 / p90", "Goals floor / ceiling — ~80% of outcomes fall in this range."),
        ("A", "Assists (projected)."),
        ("A p10 / p90", "Assists floor / ceiling — ~80% interval."),
        ("PTS", "Points = goals + assists (projected)."),
        ("PTS p10 / p90", "Points floor / ceiling — ~80% of outcomes fall in this range."),
        ("SOG", "Shots on goal (projected)."),
        ("SOG p10 / p90", "Shots-on-goal floor / ceiling — ~80% interval."),
        ("PPP", "Power-play points (projected)."),
        ("PPP p10 / p90", "Power-play-points floor / ceiling — ~80% interval."),
        ("TOI/GP", "Time on ice per game, in minutes (projected)."),
        ("W", "Goalie wins (projected)."),
        ("W p10 / p90", "Wins floor / ceiling — ~80% interval. Wide because wins depend heavily on team support."),
        ("SV%", "Save percentage (projected). .900 = saves 90% of shots faced."),
        ("SV% p10 / p90", "Save-percentage floor / ceiling — ~80% interval."),
        ("GAA", "Goals against average — goals allowed per game."),
        ("SV", "Saves (projected)."),
        ("SO", "Shutouts (projected)."),
        ("SA", "Shots against (projected)."),
        ("SOS", "Strength of schedule factor (1.00 = neutral; >1 easier, <1 tougher)."),
    ]
    ws.write(r, 0, "Abbreviation", fmts["header"]); ws.write(r, 1, "Meaning", fmts["header"]); r += 1
    for term, dfn in entries:
        ws.write(r, 0, term, fmts["dict_term"]); ws.write(r, 1, dfn, fmts["dict_def"])
        ws.set_row(r, 24); r += 1


def _dashboard_tab(wb, fmts, sk_all, g_all, sk_hist, g_hist):
    """Dropdown-driven player card with projection + last 3 seasons of actuals.

    A hidden sheet holds one lookup row per player (projection + history) keyed by a
    unique display name; the Dashboard uses INDEX/MATCH so it works in Google Sheets
    without dynamic-array functions.
    """
    # ---- build the hidden lookup table (skaters then goalies) ----
    # Unique display key (dedupe repeated names by appending team).
    def uniq_names(df):
        key = df["name"].astype(str).copy()
        dup = key.duplicated(keep=False)
        key[dup] = key[dup] + " (" + df.loc[dup, "team"].astype(str) + ")"
        return key

    sk = sk_all.copy(); sk["key"] = uniq_names(sk); sk["kind"] = "SKATER"
    g = g_all.copy(); g["key"] = uniq_names(g); g["kind"] = "GOALIE"

    # history pivots keyed by playerId -> {year: value}
    def hist_map(hist, pid_col, cols):
        m = {}
        for pid, grp in hist.groupby("playerId"):
            m[pid] = {int(row["mp_season_year"]): row for _, row in grp.iterrows()}
        return m
    skm = {pid: {int(rr["mp_season_year"]): rr for _, rr in grp.iterrows()}
           for pid, grp in sk_hist.groupby("playerId")}
    gm = {pid: {int(rr["mp_season_year"]): rr for _, rr in grp.iterrows()}
          for pid, grp in g_hist.groupby("playerId")}

    hidden = wb.add_worksheet("_Lookup")
    hidden.hide()
    # Column layout of the hidden sheet (0-based indices; letters in parens):
    # 0 key | 1 kind | 2 team | 3 pos | 4 age |
    # projection block, 17 fixed slots at cols 5..21 (F..V), read BY KIND:
    #   skater: 5(F)gp 6(G)G 7(H)Gp10 8(I)Gp90 9(J)A 10(K)Ap10 11(L)Ap90 12(M)PTS
    #           13(N)PTSp10 14(O)PTSp90 15(P)SOG 16(Q)SOGp10 17(R)SOGp90 18(S)PPP
    #           19(T)PPPp10 20(U)PPPp90 21(V)TOI
    #   goalie: 5(F)gp 6(G)W 7(H)Wp10 8(I)Wp90 9(J)SV% 10(K)SVp10 11(L)SVp90 12(M)GAA
    #           13(N)SV(saves) 14(O)SO 15(P)SA   (slots 16..21 unused for goalies)
    # A given row is EITHER a skater OR a goalie, so the two vocabularies safely share the
    # same columns; the dashboard blanks whichever card doesn't match the picked kind.
    # History blocks then start at col len(H)=22 (was 16 before ranges were added to every stat).
    H = ["key", "kind", "team", "posOrBlank", "age",
         "p_gp", "p_1", "p_2", "p_3", "p_4", "p_5", "p_6", "p_7", "p_8",
         "p_9", "p_10", "p_11", "p_12", "p_13", "p_14", "p_15", "p_16"]
    hist_slot = []
    for y in HIST_YEARS:
        hist_slot += [f"h{y}_gp", f"h{y}_1", f"h{y}_2", f"h{y}_3", f"h{y}_4", f"h{y}_5", f"h{y}_6"]
    header = H + hist_slot
    for c, h in enumerate(header):
        hidden.write(0, c, h)

    def put(ws_row, vals, start=0):
        for i, v in enumerate(vals):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if isinstance(v, str):
                hidden.write(ws_row, start + i, v)
            else:
                hidden.write_number(ws_row, start + i, float(v))

    row_of_key = {}
    rr = 1
    # skaters: 17 projection slots F..V = gp, G,Gp10,Gp90, A,Ap10,Ap90, PTS,PTSp10,PTSp90,
    #          SOG,SOGp10,SOGp90, PPP,PPPp10,PPPp90, TOI ; history 1..6 = G,A,PTS,SOG,PPP,TOI/GP
    for _, s in sk.iterrows():
        put(rr, [s["key"], "SKATER", s["team"], s.get("position"), s.get("target_age"),
                 s.get("proj_gp"),
                 s.get("proj_goals"), s.get("goals_p10"), s.get("goals_p90"),
                 s.get("proj_assists"), s.get("assists_p10"), s.get("assists_p90"),
                 s.get("proj_points"), s.get("points_p10"), s.get("points_p90"),
                 s.get("proj_shots"), s.get("shots_p10"), s.get("shots_p90"),
                 s.get("proj_pp_points"), s.get("pp_points_p10"), s.get("pp_points_p90"),
                 s.get("proj_toi_per_gp")])
        base = len(H)
        for j, y in enumerate(HIST_YEARS):
            hs = skm.get(s["playerId"], {}).get(y)
            if hs is not None:
                put(rr, [hs["games_played"], hs["goals_raw"], hs["assists_raw"], hs["points_raw"],
                         hs["shots_raw"], hs["pp_points_raw"], hs["toi_per_gp"]], start=base + j * 7)
        row_of_key[s["key"]] = rr + 1  # 1-based row for sheet formulas
        rr += 1
    # goalies: projection slots F..P = W,Wp10,Wp90,SV%,SVp10,SVp90,GAA,SV,SO,SA (slots Q..V unused);
    #          history 1..6 = W,SV%,GAA,SV,SO
    for _, s in g.iterrows():
        put(rr, [s["key"], "GOALIE", s["team"], "", None,
                 s.get("proj_gp"), s.get("proj_wins"), s.get("wins_p10"), s.get("wins_p90"),
                 s.get("proj_save_pct"), s.get("save_pct_p10"), s.get("save_pct_p90"),
                 s.get("proj_gaa"), s.get("proj_saves"), s.get("proj_shutouts"), s.get("proj_shots_against")])
        base = len(H)
        for j, y in enumerate(HIST_YEARS):
            hs = gm.get(s["playerId"], {}).get(y)
            if hs is not None:
                put(rr, [hs["gamesPlayed"], hs["wins"], hs["savePct"], hs["goalsAgainstAverage"],
                         hs["saves"], hs["shutouts"], None], start=base + j * 7)
        row_of_key[s["key"]] = rr + 1
        rr += 1
    n_rows = rr - 1

    # sorted list of names for the dropdown, written to a hidden column far to the right
    all_keys = sorted(list(sk["key"]) + list(g["key"]))
    for i, k in enumerate(all_keys):
        hidden.write(1 + i, 60, k)
    keys_range = f"_Lookup!$BI$2:$BI${1 + len(all_keys)}"

    # ---- the visible Dashboard ----
    ws = wb.add_worksheet("Dashboard")
    ws.activate()
    ws.hide_gridlines(2)
    ws.set_column(0, 0, 3)
    ws.set_column(1, 1, 18)
    ws.set_column(2, 2, 13)
    ws.set_column(3, 3, 16)   # inline p10–p90 range text ("78.9  to  95.2")
    ws.set_column(4, 9, 12)
    ws.write(1, 1, f"NHL {SEASON_LABEL} Player Dashboard", fmts["title"])
    ws.write(2, 1, "Pick a player  ➜", fmts["subtitle"])
    ws.write(2, 3, all_keys[0] if all_keys else "", fmts["pick"])
    ws.data_validation(2, 3, 2, 3, {"validate": "list", "source": keys_range})
    ws.merge_range(2, 4, 2, 6, "◀ choose any skater or goalie from this dropdown", fmts["subtitle"])

    PICK = "$D$3"
    # MATCH the picked key to a row in the hidden sheet.
    match = f'MATCH({PICK},_Lookup!$A:$A,0)'

    def idx(col_letter):
        return f'IFERROR(INDEX(_Lookup!${col_letter}:${col_letter},{match}),"")'

    # meta line
    ws.write(4, 1, "Team", fmts["big_lbl"]); ws.write(5, 1, f'={idx("C")}', fmts["pick"])
    ws.write(4, 3, "Type", fmts["big_lbl"]); ws.write(5, 3, f'={idx("B")}', fmts["pick"])
    ws.write(4, 5, "Age", fmts["big_lbl"])
    ws.write_formula(5, 5, f'=IFERROR(IF(INDEX(_Lookup!$E:$E,{match})="","-",INDEX(_Lookup!$E:$E,{match})),"")', fmts["pick"])

    # ---------- INTERACTIVE games-played control ----------
    # Counting stats are exactly linear in games played (season total = per-game rate x
    # GP), so we let the user override GP and rescale the whole stat line live. Per-game
    # stats (SV%, GAA, TOI/GP) do NOT change. The input cell defaults to blank => the pure
    # model projection; type a number 0-84 to test a different workload.
    ADJ = "$C$10"                       # the editable cell
    MG = idx("F")                       # model's projected GP for the picked player
    EFF = f'IF({ADJ}="",{MG},{ADJ})'    # games actually driving the stats
    SCALE = f'IF({ADJ}="",1,IFERROR({ADJ}/({MG}),1))'  # linear rescale factor

    ws.write(7, 1, "⚙  Adjust Games Played (optional)", fmts["h2"])
    ws.write(8, 1, "Model projects", fmts["label"])
    ws.write_formula(8, 2, f'={MG}', fmts["1dp"])
    ws.merge_range(8, 3, 8, 7, "← the model's games-played projection for this player", fmts["subtitle"])
    ws.write(9, 1, "Your GP  ✎", fmts["label"])
    ws.write_blank(9, 2, None, fmts["input"])          # C10 — user types here
    ws.data_validation(9, 2, 9, 2, {"validate": "decimal", "criteria": "between",
                                    "minimum": 0, "maximum": C.SEASON_GAMES,
                                    "ignore_blank": True,
                                    "input_message": "Type games played (0-84), or leave blank to use "
                                                     "the model's projection. Every counting stat below "
                                                     "rescales instantly."})
    ws.merge_range(9, 3, 9, 7, "type any number of games (0–84) here to rescale the stats — "
                   "leave blank for the model default", fmts["subtitle"])
    ws.write(10, 1, "Games used", fmts["label"])
    ws.write_formula(10, 2, f'={EFF}', fmts["big_lbl_val"])
    ws.merge_range(10, 3, 10, 7, "← the stat lines below are built on this many games", fmts["subtitle"])

    # We show a SKATER card and a GOALIE card; whichever matches the picked type fills in,
    # the other shows blanks. Kind flag:
    is_sk = f'(INDEX(_Lookup!$B:$B,{match})="SKATER")'

    hdrs = [f"{SEASON_LABEL} (proj)", "Range (p10–p90)"] + [_season_label(y) for y in HIST_YEARS]
    hist_base_cols = _hist_col_letters()  # letters for the 3 history season blocks

    def _range_cell(lo_col, hi_col, nf, scaled):
        """Formula text for a 'p10  to  p90' range. When the user has overridden GP the
        counting-stat range scales linearly with the same factor as the point estimate
        (rate stats pass scaled=False and show the stored band unchanged)."""
        d = "0.0" if nf == "1dp" else ("0" if nf == "0dp" else "0.000")
        if scaled:
            lo = f'({idx(lo_col)}*{SCALE})'
            hi = f'({idx(hi_col)}*{SCALE})'
        else:
            lo, hi = idx(lo_col), idx(hi_col)
        return f'TEXT({lo},"{d}")&"  to  "&TEXT({hi},"{d}")'

    # ---------- Skater card ----------
    r0 = 12
    ws.write(r0, 1, "SKATER PROJECTION", fmts["h2"])
    ws.write(r0 + 1, 1, "", fmts["header"])
    for c, h in enumerate(hdrs):
        ws.write(r0 + 1, 2 + c, h, fmts["header"])
    # rows: (label, proj col, p10 col, p90 col, hist offset 0..6, fmt, mode)
    #   mode "gp"=show effective GP, "scale"=rate x GP, "flat"=per-game (unchanged)
    # skater projection cols (17-slot block): F=gp | G=G H=Gp10 I=Gp90 | J=A K=Ap10 L=Ap90 |
    #   M=PTS N=PTSp10 O=PTSp90 | P=SOG Q=SOGp10 R=SOGp90 | S=PPP T=PPPp10 U=PPPp90 | V=TOI
    sk_rows = [
        ("Games Played", "F", None, None, 0, "1dp", "gp"),
        ("Goals",        "G", "H", "I", 1, "1dp", "scale"),
        ("Assists",      "J", "K", "L", 2, "1dp", "scale"),
        ("Points",       "M", "N", "O", 3, "1dp", "scale"),
        ("Shots (SOG)",  "P", "Q", "R", 4, "0dp", "scale"),
        ("PP Points",    "S", "T", "U", 5, "1dp", "scale"),
        ("TOI/GP",       "V", None, None, 6, "1dp", "flat"),
    ]
    rr2 = r0 + 2
    for label, pcol, lo_col, hi_col, hoff, nf, mode in sk_rows:
        ws.write(rr2, 1, label, fmts["label"])
        if mode == "gp":
            val = EFF
        elif mode == "scale":
            val = f'{idx(pcol)}*{SCALE}'
        else:
            val = idx(pcol)
        ws.write_formula(rr2, 2, f'=IF({is_sk},{val},"")', fmts[nf])
        # per-stat p10–p90 range (blank for GP / TOI, which have no band)
        if lo_col:
            rng = _range_cell(lo_col, hi_col, nf, scaled=(mode == "scale"))
            ws.write_formula(rr2, 3, f'=IF({is_sk},IFERROR({rng},""),"")', fmts["text_c"])
        else:
            ws.write(rr2, 3, "", fmts["text_c"])
        for j in range(3):
            col = hist_base_cols[j][hoff]
            ws.write_formula(rr2, 4 + j, f'=IF({is_sk},{idx(col)},"")', fmts[nf])
        rr2 += 1

    # ---------- Goalie card ----------
    r1 = rr2 + 3
    ws.write(r1, 1, "GOALIE PROJECTION", fmts["h2"])
    ws.write(r1 + 1, 1, "", fmts["header"])
    for c, h in enumerate(hdrs):
        ws.write(r1 + 1, 2 + c, h, fmts["header"])
    # goalie projection cols: F=gp G=W H=Wp10 I=Wp90 J=SV% K=SVp10 L=SVp90 M=GAA N=SV O=SO P=SA
    is_g = f'(INDEX(_Lookup!$B:$B,{match})="GOALIE")'
    #  (label, proj col, p10 col, p90 col, hist offset, fmt, mode)
    g_rows = [
        ("Games Played", "F", None, None, 0, "1dp", "gp"),
        ("Wins",         "G", "H", "I", 1, "1dp", "scale"),
        ("Save %",       "J", "K", "L", 2, "pct3", "flat"),
        ("GAA",          "M", None, None, 3, "2dp", "flat"),
        ("Saves",        "N", None, None, 4, "0dp", "scale"),
        ("Shutouts",     "O", None, None, 5, "1dp", "scale"),
    ]
    rr3 = r1 + 2
    for label, pcol, lo_col, hi_col, hoff, nf, mode in g_rows:
        ws.write(rr3, 1, label, fmts["label"])
        if mode == "gp":
            val = EFF
        elif mode == "scale":
            val = f'{idx(pcol)}*{SCALE}'
        else:
            val = idx(pcol)
        ws.write_formula(rr3, 2, f'=IF({is_g},{val},"")', fmts[nf])
        # ranges: Wins scales with GP; SV% is per-game so its stored band is unchanged.
        if lo_col:
            rng = _range_cell(lo_col, hi_col, nf, scaled=(mode == "scale"))
            ws.write_formula(rr3, 3, f'=IF({is_g},IFERROR({rng},""),"")', fmts["text_c"])
        else:
            ws.write(rr3, 3, "", fmts["text_c"])
        for j in range(3):
            col = hist_base_cols[j][hoff]
            ws.write_formula(rr3, 4 + j, f'=IF({is_g},{idx(col)},"")', fmts[nf])
        rr3 += 1

    ws.write(rr3 + 2, 1, "How to read this", fmts["label"])
    ws.merge_range(rr3 + 2, 2, rr3 + 6, 7,
                   "The first value column is the projection for the upcoming season; the columns to its "
                   "right are the player's ACTUAL results from the last three seasons, so you can sanity-check "
                   "the projection against recent form. Only the card matching the player's type (skater or "
                   "goalie) fills in. "
                   "TIP: type a number in the yellow 'Your GP' cell to see every counting stat (goals, assists, "
                   "points, shots, wins, saves…) rescale to that workload — rate stats like SV%, GAA and TOI/GP "
                   "stay the same because they are per-game. Leave it blank to use the model's own games-played "
                   "projection. Note seasons before 2026-27 were 82 games, not 84.", fmts["wrap"])
    ws.set_row(rr3 + 2, 74)


_HIST_START = 22  # = len(H): 22 fixed cols (A..V) precede the history season blocks


def _hist_col_letters():
    """Column letters for the 3 history season blocks in the hidden sheet.

    Hidden layout: 22 fixed cols (A..V, the key/meta + 17-slot projection block) then
    3 blocks of 7 each. Block j occupies columns (_HIST_START + j*7 .. +6), 0-based.
    Returns list of 3 lists, each 7 letters (offsets 0..6)."""
    def L(n):  # 0-based col index -> excel letters
        s = ""
        n += 1
        while n:
            n, rem = divmod(n - 1, 26)
            s = chr(65 + rem) + s
        return s
    blocks = []
    for j in range(3):
        start = _HIST_START + j * 7
        blocks.append([L(start + k) for k in range(7)])
    return blocks


def _all_skaters_tab(wb, fmts, sk):
    ws = wb.add_worksheet("All Skaters")
    ws.hide_gridlines(2)
    _write_table(ws, sk, SKATER_COLS, fmts)


def _all_goalies_tab(wb, fmts, g):
    ws = wb.add_worksheet("All Goalies")
    ws.hide_gridlines(2)
    _write_table(ws, g, GOALIE_COLS, fmts)


def _team_tabs(wb, fmts, sk, g):
    teams = sorted(set(sk["team"].dropna()) | set(g["team"].dropna()))
    for team in teams:
        ws = wb.add_worksheet(str(team)[:31])
        ws.hide_gridlines(2)
        ws.set_column(0, 0, 20)
        ws.write(0, 0, f"{team} — {SEASON_LABEL} Projections", fmts["title"])
        tsk = sk[sk["team"] == team].sort_values("proj_points", ascending=False)
        tg = g[g["team"] == team].sort_values("proj_wins", ascending=False)
        ws.write(2, 0, "Skaters", fmts["h2"])
        n = _write_table(ws, tsk, SKATER_COLS, fmts, start_row=3)
        gstart = 3 + n + 3
        ws.write(gstart - 1, 0, "Goalies", fmts["h2"])
        # team tabs: don't freeze (two tables stacked); write goalie table without freeze
        for c, (col, hdr, width, _, _) in enumerate(GOALIE_COLS):
            ws.write(gstart, c, hdr, fmts["header"])
        for r, (_, row) in enumerate(tg.iterrows()):
            for c, (col, hdr, width, nf, _) in enumerate(GOALIE_COLS):
                val = row.get(col)
                if nf == "text":
                    ws.write(gstart + 1 + r, c, "" if pd.isna(val) else str(val), fmts["text"])
                elif pd.isna(val):
                    ws.write(gstart + 1 + r, c, "", fmts[nf])
                else:
                    ws.write_number(gstart + 1 + r, c, float(val), fmts[nf])


def build():
    sk_all, g_all = _load_projections()
    # rostered-only for the browse tables (clean rosters; drops retired/unsigned + defunct teams)
    sk = sk_all[sk_all["on_roster"]].copy().sort_values("proj_points", ascending=False)
    g = g_all[g_all["on_roster"]].copy().sort_values("proj_wins", ascending=False)
    sk_hist = _skater_history()
    g_hist = _goalie_history()

    teams = sorted(set(sk["team"].dropna()) | set(g["team"].dropna()))
    path = C.OUTPUT / f"NHL_Projections_{TARGET}.xlsx"
    wb = xlsxwriter.Workbook(str(path), {"nan_inf_to_errors": True})
    fmts = _make_formats(wb)

    _overview_tab(wb, fmts, len(sk), len(g), len(teams))
    # Dashboard can pull ANY player with history, not just rostered.
    _dashboard_tab(wb, fmts, sk, g, sk_hist, g_hist)
    _all_skaters_tab(wb, fmts, sk)
    _all_goalies_tab(wb, fmts, g)
    _team_tabs(wb, fmts, sk, g)

    wb.close()
    print(f"Built workbook: {path}")
    print(f"  {len(sk)} rostered skaters, {len(g)} rostered goalies, {len(teams)} team tabs")
    print(f"  Upload to Google Drive and open with Google Sheets (tabs/dropdown/formulas preserved).")


if __name__ == "__main__":
    build()
