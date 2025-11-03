# app.py — Combined NBA Player Scouting + Team Dashboard
# NOTE: Per your request, ONLY the "Player Projection Summary" section was changed.
# All other logic, layout, endpoints, and data handling remain the same.

import time
import datetime
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import re
from zoneinfo import ZoneInfo  # ET cutoff for season-to-date

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import (
    playergamelog,
    playercareerstats,
    leaguedashteamstats,
    LeagueDashPlayerStats,
    commonplayerinfo,
    leaguegamefinder,                # for opponent-specific all-season logs
    teamdashboardbygeneralsplits,    # for per-team fallback diagnostics
)

# ----------------------- Streamlit Setup -----------------------
st.set_page_config(page_title="NBA Dashboards", layout="wide")

# ----------------------- Config -----------------------
CACHE_HOURS = 12           # general caches
TEAM_CTX_TTL_SECONDS = 300 # 5 min TTL for opponent metrics
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2

def _retry_api(endpoint_cls, kwargs, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES, sleep=0.8):
    last_err = None
    for i in range(retries + 1):
        try:
            obj = endpoint_cls(timeout=timeout, **kwargs)
            return obj.get_data_frames()
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(sleep * (i + 1))
    raise last_err

def _season_labels(start=2010, end=None):
    if end is None:
        end = datetime.datetime.utcnow().year
    def lab(y): return f"{y}-{str((y+1)%100).zfill(2)}"
    return [lab(y) for y in range(end, start-1, -1)]

SEASONS = _season_labels(2015, datetime.datetime.utcnow().year)

def _prev_season_label(season_label: str) -> str:
    """Turn '2025-26' -> '2024-25' safely."""
    try:
        y0 = int(season_label.split("-")[0])
        return f"{y0-1}-{str((y0)%100).zfill(2)}"
    except Exception:
        return season_label

# ----------------------- Utils -----------------------
def numeric_format_map(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    return {c: "{:.2f}" for c in num_cols}

def _auto_height(df, row_px=34, header_px=38, max_px=900):
    rows = max(len(df), 1)
    return min(max_px, header_px + row_px * rows + 8)

def _fmt1(v):
    try:
        return f"{float(v):.1f}"
    except Exception:
        return "—"

# --- Parsing MATCHUP opponent token (used only in legacy fallback) ---
_punct_re = re.compile(r"[^\w]")
def parse_opp_from_matchup(matchup_str: str):
    if not isinstance(matchup_str, str):
        return None
    parts = matchup_str.split()
    if len(parts) < 3:
        return None
    token = parts[-1].upper().strip()
    token = _punct_re.sub("", token)
    return token

def add_shot_breakouts(df):
    for col in ["MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB"]:
        if col not in df.columns:
            df[col] = 0
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["2PM"] = df["FGM"] - df["FG3M"]
    df["2PA"] = df["FGA"] - df["FG3A"]
    keep_order = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","PRA","2PM","2PA","FG3M","FG3A","FTM","FTA","OREB","DREB"]
    existing = [c for c in keep_order if c in df.columns]
    return df[existing]

def format_record(w, l):
    try:
        return f"{int(w)}–{int(l)}"
    except Exception:
        return "—"

def append_average_row(df: pd.DataFrame, label: str = "Average") -> pd.DataFrame:
    """Append an 'Average' row to the end of a box score table (mean over numeric cols)."""
    out = df.copy()
    if out.empty:
        return out
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return out
    avg_vals = out[num_cols].mean(numeric_only=True)
    avg_row = {c: np.nan for c in out.columns}
    for c in num_cols:
        avg_row[c] = float(avg_vals.get(c, np.nan))
    # Set identifiers to a clean label
    if "GAME_DATE" in out.columns:
        avg_row["GAME_DATE"] = pd.NaT
    if "MATCHUP" in out.columns:
        avg_row["MATCHUP"] = label
    if "WL" in out.columns:
        avg_row["WL"] = ""
    out = pd.concat([out, pd.DataFrame([avg_row])], ignore_index=True)
    return out

# --- Opponent abbrev & TEAM_ID resolution (for fallback and GameFinder) ---
def _build_static_maps():
    teams_df = pd.DataFrame(static_teams.get_teams())
    by_full = dict(zip(teams_df["full_name"].astype(str), teams_df["abbreviation"].astype(str)))
    id_by_full = dict(zip(teams_df["full_name"].astype(str), teams_df["id"].astype(int)))

    nick_map = {
        "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
        "LA Lakers": "LAL", "Los Angeles Lakers": "LAL",
        "NY Knicks": "NYK", "New York Knicks": "NYK",
        "GS Warriors": "GSW", "Golden State Warriors": "GSW",
        "SA Spurs": "SAS", "San Antonio Spurs": "SAS",
        "NO Pelicans": "NOP", "New Orleans Pelicans": "NOP",
        "OKC Thunder": "OKC", "Oklahoma City Thunder": "OKC",
        "PHX Suns": "PHX", "Phoenix Suns": "PHX",
        "POR Trail Blazers": "POR", "Portland Trail Blazers": "POR",
        "UTA Jazz": "UTA", "Utah Jazz": "UTA",
        "WAS Wizards": "WAS", "Washington Wizards": "WAS",
        "CLE Cavaliers": "CLE", "Cleveland Cavaliers": "CLE",
        "MIN Timberwolves": "MIN", "Minnesota Timberwolves": "MIN",
        "CHA Hornets": "CHA", "Charlotte Hornets": "CHA",
        "BRK Nets": "BKN", "Brooklyn Nets": "BKN",
        "PHI 76ers": "PHI", "Philadelphia 76ers": "PHI",
    }
    alias_map = {"PHO":"PHX","BRK":"BKN","NJN":"BKN","NOH":"NOP","NOK":"NOP","CHO":"CHA","CHH":"CHA","SEA":"OKC","WSB":"WAS","VAN":"MEM"}

    by_full_cf = {k.casefold(): v for k, v in by_full.items()}
    nick_cf = {k.casefold(): v for k, v in nick_map.items()}
    alias_up = {k.upper(): v.upper() for k, v in alias_map.items()}
    return by_full_cf, nick_cf, alias_up, id_by_full

BY_FULL_CF, NICK_CF, ABBR_ALIAS, TEAMID_BY_FULL = _build_static_maps()

def normalize_abbr(abbr: str | None) -> str | None:
    if not isinstance(abbr, str) or not abbr:
        return None
    a = abbr.upper().strip()
    return ABBR_ALIAS.get(a, a)

def resolve_team_abbrev(team_name: str, team_ctx_row: pd.Series | None = None) -> str | None:
    if team_ctx_row is not None and "TEAM_ABBREVIATION" in team_ctx_row.index:
        v = str(team_ctx_row.get("TEAM_ABBREVIATION", "")).strip().upper()
        if 2 <= len(v) <= 4:
            return normalize_abbr(v)
    if isinstance(team_name, str):
        cf = team_name.casefold().strip()
        if cf in BY_FULL_CF: return normalize_abbr(BY_FULL_CF[cf])
        if cf in NICK_CF:    return normalize_abbr(NICK_CF[cf])
    return None

def resolve_team_id(team_name: str, team_ctx_row: pd.Series | None = None) -> int | None:
    if team_ctx_row is not None and "TEAM_ID" in team_ctx_row.index:
        try:
            return int(team_ctx_row["TEAM_ID"])
        except Exception:
            pass
    return TEAMID_BY_FULL.get(team_name)

# ----------------------- Cached data (shared) -----------------------
@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_season_player_index(season):
    try:
        frames = _retry_api(LeagueDashPlayerStats, {
            "season": season,
            "per_mode_detailed": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        })
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    keep = ["PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION","TEAM_NAME","GP","MIN"]
    for c in keep:
        if c not in df.columns: df[c] = 0
    return df[keep].drop_duplicates(subset=["PLAYER_ID"]).sort_values(["TEAM_NAME","PLAYER_NAME"]).reset_index(drop=True)

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_player_logs(player_id, season):
    try:
        frames = _retry_api(playergamelog.PlayerGameLog, {
            "player_id": player_id,
            "season": season,
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        })
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    if df.empty: return df
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_player_career(player_id):
    try:
        frames = _retry_api(playercareerstats.PlayerCareerStats, {"player_id": player_id})
        return frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_common_player_info(player_id):
    try:
        frames = _retry_api(commonplayerinfo.CommonPlayerInfo, {"player_id": player_id})
        return frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ----------------------- Team context (Advanced=PerGame, Base=Totals) -----------------------
@st.cache_data(ttl=TEAM_CTX_TTL_SECONDS, show_spinner=False)
def get_team_context_regular_season_to_date(season: str, cutoff_date_et: str, _refresh_key: int = 0):
    """
    NBA-only team context THROUGH cutoff_date_et (ET).
    Advanced = PerGame (PACE/ratings), Base = Totals (MIN matches NBA.com).
    """
    common = dict(
        season=season,
        season_type_all_star="Regular Season",
        league_id_nullable="00",
        date_from_nullable=None,
        date_to_nullable=cutoff_date_et,
        po_round_nullable=None,
    )

    def _safe_frames(ep_cls, kwargs):
        try:
            frames = _retry_api(ep_cls, kwargs)
            return frames[0] if frames else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # Advanced (PerGame)
    adv = _safe_frames(
        leaguedashteamstats.LeagueDashTeamStats,
        dict(common, measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame"),
    )
    # Base (Totals) so MIN matches NBA.com (48 * GP)
    base = _safe_frames(
        leaguedashteamstats.LeagueDashTeamStats,
        dict(common, measure_type_detailed_defense="Base", per_mode_detailed="Totals"),
    )

    def _nba_only(df):
        if df is None or df.empty or "TEAM_ID" not in df.columns:
            return pd.DataFrame()
        return df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()

    adv = _nba_only(adv)
    base = _nba_only(base)

    # De-dup just in case
    for df in (adv, base):
        if not df.empty:
            df.sort_values(["TEAM_ID"], inplace=True)
            df.drop_duplicates(subset=["TEAM_ID"], keep="first", inplace=True)

    # Keep cols
    adv_cols = ["TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION","PACE","OFF_RATING","DEF_RATING","NET_RATING"]
    for c in adv_cols:
        if c not in adv.columns: adv[c] = np.nan
    adv = adv[adv_cols].copy()

    base_cols = ["TEAM_ID","GP","W","L","W_PCT","MIN"]  # MIN totals
    for c in base_cols:
        if c not in base.columns: base[c] = np.nan
    base = base[base_cols].copy()

    # Type coercion
    for c in ["PACE","OFF_RATING","DEF_RATING","NET_RATING"]:
        adv[c] = pd.to_numeric(adv[c], errors="coerce")
    for c in ["GP","W","L","W_PCT","MIN"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    # Merge
    df = pd.merge(adv, base, on="TEAM_ID", how="inner")

    # Fill missing abbreviations from static teams (rare)
    teams_df = pd.DataFrame(static_teams.get_teams())
    abbr_map = dict(zip(teams_df["id"], teams_df["abbreviation"]))
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].fillna(df["TEAM_ID"].map(abbr_map))

    # Validate suspicious rows; fallback to team dashboard if needed
    def _fix_row(r):
        bad_def = pd.isna(r["DEF_RATING"]) or not (90 <= float(r["DEF_RATING"]) <= 130)
        bad_pace = pd.isna(r["PACE"]) or not (90 <= float(r["PACE"]) <= 110)
        if not (bad_def or bad_pace):
            return r
        try:
            td = _retry_api(
                teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits,
                dict(
                    team_id=int(r["TEAM_ID"]),
                    season=season,
                    season_type_all_star="Regular Season",
                    league_id_nullable="00",
                    date_from_nullable=None,
                    date_to_nullable=cutoff_date_et,
                    measure_type_detailed_defense="Advanced",
                    per_mode_detailed="PerGame",
                ),
            )
            dash = td[0] if td else pd.DataFrame()
            if not dash.empty:
                for k in ["OFF_RATING","DEF_RATING","NET_RATING","PACE","GP","W","L","W_PCT"]:
                    if k in dash.columns:
                        r[k] = pd.to_numeric(dash.iloc[0][k], errors="coerce")
        except Exception:
            pass
        return r

    if not df.empty:
        df = df.apply(_fix_row, axis=1)

    # Ranks (1 = best)
    df["DEF_RANK"] = df["DEF_RATING"].rank(ascending=True,  method="min").astype("Int64")
    df["PACE_RANK"] = df["PACE"].rank(ascending=False, method="min").astype("Int64")
    df["NET_RANK"]  = df["NET_RATING"].rank(ascending=False, method="min").astype("Int64")

    df.sort_values("TEAM_NAME", inplace=True)
    fetched_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return df.reset_index(drop=True), fetched_at, cutoff_date_et

# --- Fallback: collect all-season player logs and filter by parsed matchup token ---
@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_all_player_logs_all_seasons(player_id, season_labels):
    frames = []
    for s in season_labels:
        df = get_player_logs(player_id, s)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0, ignore_index=True)
    return out.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

# --- Primary: use LeagueGameFinder to fetch player's games vs a specific opponent across ALL seasons ---
@st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
def get_vs_opponent_games(player_id: int, opp_team_id: int):
    """
    Reliable per-game box scores vs opponent across all seasons (Regular Season only).
    """
    try:
        frames = _retry_api(
            leaguegamefinder.LeagueGameFinder,
            {
                "player_or_team_abbreviation": "P",   # player mode
                "player_id_nullable": player_id,
                "vs_team_id_nullable": opp_team_id,
                "season_type_nullable": "Regular Season",
                "league_id_nullable": "00",
            },
        )
        df = frames[0] if frames else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    wanted = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB"]
    for c in wanted:
        if c not in df.columns:
            df[c] = 0
    return df[wanted]

# =====================================================================
# Player Dashboard
# =====================================================================
def player_dashboard():
    st.title("NBA Player Scouting Dashboard")

    # ----------------------- Sidebar (Season, Player, Recency + Refresh) -----------------------
    with st.sidebar:
        st.header("Player Filters")
        season = st.selectbox("Season", SEASONS, index=0, key="season_sel")

        col_r1, col_r2 = st.columns([1,1])
        with col_r1:
            if st.button("Refresh metrics"):
                st.session_state["team_ctx_refresh_key"] = st.session_state.get("team_ctx_refresh_key", 0) + 1
        with col_r2:
            if st.button("Hard clear cache"):
                st.cache_data.clear()
                st.session_state["team_ctx_refresh_key"] = st.session_state.get("team_ctx_refresh_key", 0) + 1

    # ET cutoff for "to date"
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    cutoff_date_et = now_et.strftime("%m/%d/%Y")

    # Load team & player context
    refresh_key = st.session_state.get("team_ctx_refresh_key", 0)
    with st.spinner("Loading league context..."):
        team_ctx, fetched_at, cutoff_used = get_team_context_regular_season_to_date(season, cutoff_date_et, refresh_key)

    if team_ctx.empty:
        st.error("Unable to load team context for this season.")
        st.stop()

    team_list = team_ctx["TEAM_NAME"].tolist()

    with st.sidebar:
        with st.spinner("Loading players..."):
            season_players = get_season_player_index(season)

        q = st.text_input("Search player", key="player_search").strip()
        filtered_players = season_players if not q else season_players[season_players["PLAYER_NAME"].str.contains(q, case=False, na=False)]

        if filtered_players.empty:
            st.info("No players match your search.")
            st.stop()

        default_idx = 0
        if "player_sel" in st.session_state:
            if st.session_state["player_sel"] in filtered_players["PLAYER_NAME"].tolist():
                default_idx = filtered_players["PLAYER_NAME"].tolist().index(st.session_state["player_sel"])

        player_name = st.selectbox("Player", filtered_players["PLAYER_NAME"].tolist(), index=default_idx, key="player_sel")
        player_row = filtered_players[filtered_players["PLAYER_NAME"] == player_name].iloc[0]
        player_id  = int(player_row["PLAYER_ID"])

        n_recent = st.selectbox("Recency window", ["Season", 5, 10, 15, 20], index=1, key="recent_sel")

    # ----------------------- Fetch Player Data -----------------------
    with st.spinner("Fetching player logs & info..."):
        logs = get_player_logs(player_id, season)
        if logs.empty:
            st.error("No game logs for this player/season.")
            st.stop()
        career_df = get_player_career(player_id)
        cpi = get_common_player_info(player_id)

    # ----------------------- Header + Opponent Selector -----------------------
    left, right = st.columns([2, 1])

    with left:
        st.subheader(f"{player_name} — {season}")
        team_name_disp = (cpi["TEAM_NAME"].iloc[0] if ("TEAM_NAME" in cpi.columns and not cpi.empty) else player_row.get("TEAM_NAME","Unknown"))
        pos = (cpi["POSITION"].iloc[0] if ("POSITION" in cpi.columns and not cpi.empty) else "N/A")
        exp = (cpi["SEASON_EXP"].iloc[0] if ("SEASON_EXP" in cpi.columns and not cpi.empty) else "N/A")
        gp = len(logs)
        st.caption(f"**Team:** {team_name_disp} • **Position:** {pos} • **Seasons:** {exp} • **Games Played:** {gp}")

    with right:
        opponent = st.selectbox("Opponent", team_list, index=0, key="opponent_sel")

    # Opponent row + record + metrics + freshness + cutoff date
    opp_row = team_ctx.loc[team_ctx["TEAM_NAME"] == opponent].iloc[0]
    opp_record = format_record(opp_row.get("W", np.nan), opp_row.get("L", np.nan))

    st.markdown(f"### Opponent: **{opponent}** ({opp_record})")
    st.caption(f"Opponent metrics last updated: {fetched_at} • Season-to-date through (ET): {cutoff_used}")
    c1, c2, c3 = st.columns(3)
    c1.metric("DEF Rating", _fmt1(opp_row.get("DEF_RATING", np.nan)))
    c1.caption(f"Rank: {int(opp_row['DEF_RANK'])}/30" if pd.notna(opp_row.get("DEF_RANK")) else "Rank: —")
    c2.metric("PACE", _fmt1(opp_row.get("PACE", np.nan)))
    c2.caption(f"Rank: {int(opp_row['PACE_RANK'])}/30" if pd.notna(opp_row.get("PACE_RANK")) else "Rank: —")
    c3.metric("NET Rating", _fmt1(opp_row.get("NET_RATING", np.nan)))
    c3.caption(f"Rank: {int(opp_row['NET_RANK'])}/30" if pd.notna(opp_row.get("NET_RANK")) else "Rank: —")

    # ----------------------- Recent Averages (tiles) -----------------------
    for col in ["MIN","PTS","REB","AST","FG3M"]:
        if col not in logs.columns:
            logs[col] = 0
    window_df = logs if st.session_state.get("recent_sel","Season") == "Season" else logs.head(int(st.session_state["recent_sel"]))
    recent_avg = window_df[["MIN","PTS","REB","AST","FG3M"]].mean(numeric_only=True)

    st.markdown("### Recent Averages")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MIN", _fmt1(recent_avg.get("MIN", np.nan)))
    m2.metric("PTS", _fmt1(recent_avg.get("PTS", np.nan)))
    m3.metric("REB", _fmt1(recent_avg.get("REB", np.nan)))
    m4.metric("AST", _fmt1(recent_avg.get("AST", np.nan)))
    m5.metric("3PM", _fmt1(recent_avg.get("FG3M", np.nan)))

    # ----------------------- Trends -----------------------
    st.markdown(f"### Trends (Last {st.session_state.get('recent_sel','Season')} Games)")
    if "PRA" not in logs.columns:
        logs["PRA"] = logs.get("PTS", 0) + logs.get("REB", 0) + logs.get("AST", 0)
    trend_cols = [c for c in ["MIN","PTS","REB","AST","PRA","FG3M"] if c in logs.columns]
    n_recent_val = st.session_state.get("recent_sel","Season")
    trend_df = logs[["GAME_DATE"] + trend_cols].head(int(n_recent_val) if n_recent_val != "Season" else len(logs)).copy()
    trend_df = trend_df.sort_values("GAME_DATE")
    if "GAME_DATE" in trend_df.columns and len(trend_cols) > 0 and len(trend_df) > 0:
        for s in trend_cols:
            chart = (
                alt.Chart(trend_df)
                .mark_line(point=True)
                .encode(x="GAME_DATE:T", y=alt.Y(s, title=s))
                .properties(height=160)
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No trend data available to chart.")

    # ----------------------- Compare Windows -----------------------
    st.markdown("### Compare Windows (Career / Prev Season / Current Season / L5 / L20)")

    def avg(df, n):
        if df.empty: return pd.Series(dtype=float)
        if n == "Season": return df.mean(numeric_only=True)
        return df.head(int(n)).mean(numeric_only=True)

    def career_per_game(career_df, cols=("MIN","PTS","REB","AST","FG3M")):
        if career_df.empty or "GP" not in career_df.columns:
            return pd.Series({c: np.nan for c in cols}, dtype=float)
        needed = list(set(cols) | {"GP"})
        for c in needed:
            if c not in career_df.columns: career_df[c] = 0
        total_gp = pd.to_numeric(career_df["GP"], errors="coerce").sum()
        if total_gp == 0:
            return pd.Series({c: np.nan for c in cols}, dtype=float)
        out = {c: pd.to_numeric(career_df[c], errors="coerce").sum() / total_gp for c in cols}
        return pd.Series(out).astype(float)

    # Ensure 3PM column exists in logs for means
    if "FG3M" not in logs.columns:
        logs["FG3M"] = 0

    # Compute each slice
    metrics_order = ["MIN","PTS","REB","AST","FG3M"]

    career_pg = career_per_game(career_df, cols=metrics_order)

    # Previous season
    prev_season = _prev_season_label(season)
    prev_logs = get_player_logs(player_id, prev_season)
    for col in metrics_order:
        if col not in prev_logs.columns:
            prev_logs[col] = 0
    prev_season_pg = prev_logs[metrics_order].mean(numeric_only=True)

    # Current season
    for col in metrics_order:
        if col not in logs.columns:
            logs[col] = 0
    current_season_pg = logs[metrics_order].mean(numeric_only=True)

    # Last 5 and Last 20
    l5_pg = logs[metrics_order].head(5).mean(numeric_only=True)
    l20_pg = logs[metrics_order].head(20).mean(numeric_only=True)

    # Build table with requested order and columns
    cmp_df = pd.DataFrame({
        "Career Avg": career_pg,
        "Prev Season Avg": prev_season_pg,
        "Current Season Avg": current_season_pg,
        "Last 5 Avg": l5_pg,
        "Last 20 Avg": l20_pg,
    }, index=metrics_order).round(2)

    st.dataframe(
        cmp_df.style.format(numeric_format_map(cmp_df)),
        use_container_width=True,
        height=_auto_height(cmp_df)
    )

    # ----------------------- Last 5 Games (current season) -----------------------
    st.markdown("### Last 5 Games")
    cols_base = ["GAME_DATE","MATCHUP","WL","MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB"]
    last5 = logs[cols_base].head(5).copy()
    last5 = add_shot_breakouts(last5)
    # Append average row
    last5 = append_average_row(last5, label="Average (Last 5)")
    num_fmt = {c: "{:.1f}" for c in last5.select_dtypes(include=[np.number]).columns if c != "GAME_DATE"}
    st.dataframe(last5.style.format(num_fmt), use_container_width=True, height=_auto_height(last5))

    # ----------------------- Last 5 vs Opponent (All Seasons) -----------------------
    st.markdown(f"### Last 5 Games vs {opponent}")

    # Primary: LeagueGameFinder using opponent TEAM_ID (most robust)
    opp_team_id = resolve_team_id(opponent, opp_row)
    vs_opp_df = pd.DataFrame()
    if opp_team_id:
        vs_opp_df = get_vs_opponent_games(player_id, opp_team_id)

    # Fallback: if nothing returned (rare), use all-season logs + parsed MATCHUP token
    if vs_opp_df.empty:
        opp_abbrev = resolve_team_abbrev(opponent, opp_row)
        if "SEASON" in career_df.columns and not career_df.empty:
            season_labels = list(career_df["SEASON"].dropna().unique())
            def _yr(s):
                try: return int(s.split("-")[0])
                except: return -1
            season_labels = sorted(season_labels, key=_yr, reverse=True)
        else:
            season_labels = SEASONS

        if opp_abbrev:
            all_logs = get_all_player_logs_all_seasons(player_id, season_labels)
            if not all_logs.empty and "MATCHUP" in all_logs.columns:
                all_logs = all_logs.copy()
                all_logs["OPP_ABBR"] = all_logs["MATCHUP"].apply(parse_opp_from_matchup)
                # light aliasing for safety
                all_logs["OPP_ABBR"] = all_logs["OPP_ABBR"].apply(lambda x: ABBR_ALIAS.get(x, x) if isinstance(x, str) else x)
                vs_opp_df = all_logs[all_logs["OPP_ABBR"] == opp_abbrev][cols_base].copy() if opp_abbrev else pd.DataFrame(columns=cols_base)

    # Render last 5 vs opponent (if any)
    if vs_opp_df.empty:
        st.info(f"No historical games vs {opponent}.")
    else:
        vs_opp5 = add_shot_breakouts(vs_opp_df.sort_values("GAME_DATE", ascending=False).head(5).copy())
        # Append average row
        vs_opp5 = append_average_row(vs_opp5, label="Average (Last 5 vs Opp)")
        num_fmt2 = {c: "{:.1f}" for c in vs_opp5.select_dtypes(include=[np.number]).columns if c != "GAME_DATE"}
        st.dataframe(vs_opp5.style.format(num_fmt2), use_container_width=True, height=_auto_height(vs_opp5))

    # ----------------------- Projections (UPDATED: normalized 0–1 weights + correct CI) -----------------------
    with st.expander("Player Projection Summary"):
        st.markdown(
            """
            #### How this projection works
            This model blends multiple sources on a per-game basis—**Recent**, **Current Season**, **Previous Season**, **Career**, and (if available) **vs. Opponent**—then applies **defense** and **pace** adjustments for the selected opponent.  
            - Use the **weights** (0.00–1.00 each). We’ll **normalize** your inputs to sum to 1.00 automatically.  
            - Adjust **Projected MIN** to see projections scale accordingly.  
            - Toggle the **Confidence Band** and choose **90% / 80% / 70%** intervals to see uncertainty bounds.
            """
        )

        try:
            # --- Weight sliders: all on 0..1 scale, normalized later ---
            wc1, wc2, wc3, wc4, wc5 = st.columns(5)
            with wc1:
                w_recent_in = st.slider("Recent (0–1)", 0.00, 1.00, 0.45, 0.05, help="Last N games (see Recency window above)")
            with wc2:
                w_season_in = st.slider("Season (0–1)", 0.00, 1.00, 0.25, 0.05, help="Current season per-game")
            with wc3:
                w_prev_in   = st.slider("Prev (0–1)",   0.00, 1.00, 0.10, 0.05, help="Previous season per-game")
            with wc4:
                w_career_in = st.slider("Career (0–1)", 0.00, 1.00, 0.10, 0.05, help="Career per-game average")
            with wc5:
                w_vsopp_in  = st.slider("Vs Opp (0–1)", 0.00, 1.00, 0.10, 0.05, help="Last 5 vs this opponent (if available)")

            # --- Confidence & model knobs ---
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                z_level = st.selectbox("Confidence Interval", ["90%", "80%", "70%"], index=0)
                z_map = {"70%": 1.04, "80%": 1.28, "90%": 1.64}
                z = z_map[z_level]
            with cc2:
                rel_cap = st.slider("CI cap (±%)", 0.10, 0.40, 0.25, 0.05, help="Limits CI spread to a % around the point estimate")
            with cc3:
                alpha_min_vol = st.slider("Minutes volatility sensitivity", 0.0, 1.0, 0.5, 0.05, help="Higher widens CI when minutes vary")

            # --- Build sources (existing logic preserved) ---
            METRICS = ["PTS", "REB", "AST", "FG3M", "MIN"]
            for c in METRICS:
                if c not in logs.columns:
                    logs[c] = 0

            recent_sel = st.session_state.get("recent_sel", "Season")
            recent_n = 10 if recent_sel == "Season" else int(recent_sel)

            src = {}
            src["recent"] = logs[METRICS].head(recent_n).mean(numeric_only=True)
            src["season"] = logs[METRICS].mean(numeric_only=True)

            prev_label = _prev_season_label(season)
            prev_logs_local = get_player_logs(player_id, prev_label)
            for c in METRICS:
                if c not in prev_logs_local.columns:
                    prev_logs_local[c] = 0
            src["prev"] = (
                prev_logs_local[METRICS].mean(numeric_only=True)
                if not prev_logs_local.empty else pd.Series({m: np.nan for m in METRICS})
            )

            def _career_pg_fast(cdf, cols):
                if cdf.empty or "GP" not in cdf.columns:
                    return pd.Series({k: np.nan for k in cols})
                tot_gp = pd.to_numeric(cdf["GP"], errors="coerce").sum()
                if tot_gp == 0:
                    return pd.Series({k: np.nan for k in cols})
                out = {k: pd.to_numeric(cdf.get(k, 0), errors="coerce").sum() / tot_gp for k in cols}
                return pd.Series(out)

            src["career"] = _career_pg_fast(career_df, METRICS)

            if 'vs_opp_df' in locals() and not vs_opp_df.empty:
                tmp = vs_opp_df.copy()
                for c in METRICS:
                    if c not in tmp.columns: tmp[c] = 0
                src["vsopp"] = tmp.sort_values("GAME_DATE", ascending=False).head(5)[METRICS].mean(numeric_only=True)
            else:
                src["vsopp"] = pd.Series({m: np.nan for m in METRICS})

            # --- Normalize weights (handles any 0..1 inputs) ---
            raw_weights = {
                "recent": w_recent_in,
                "season": w_season_in,
                "prev":   w_prev_in,
                "career": w_career_in,
                "vsopp":  w_vsopp_in
            }
            # Keep sources that have at least some non-NaN values
            valid_sources = {k: v for k, v in src.items() if v.notna().any()}
            # If all sliders are zero, default to recent=1
            total_raw = sum(raw_weights[k] for k in valid_sources.keys()) if valid_sources else 0.0
            if total_raw <= 0:
                norm_w = {k: (1.0 if k == "recent" else 0.0) for k in valid_sources.keys()}
            else:
                norm_w = {k: raw_weights[k] / total_raw for k in valid_sources.keys()}

            if not valid_sources:
                st.info("Not enough data to generate a projection.")
                raise RuntimeError("No projection sources")

            # --- Blended baseline ---
            blend = sum(norm_w[k] * valid_sources[k] for k in valid_sources.keys())
            blend = blend.reindex(METRICS)

            # --- Opponent / pace adjustments on per-minute rates ---
            eps = 1e-9
            per_min = {}
            for m in ["PTS", "REB", "AST", "FG3M"]:
                per_min[m] = (blend[m] / max(blend["MIN"], eps)) if pd.notna(blend[m]) and pd.notna(blend["MIN"]) and blend["MIN"] > 0 else np.nan
            per_min = pd.Series(per_min)

            league_def = team_ctx["DEF_RATING"].mean()
            opp_def   = opp_row.get("DEF_RATING", np.nan)
            def_factor = float(league_def) / float(opp_def) if pd.notna(league_def) and pd.notna(opp_def) and opp_def > 0 else 1.0
            def_factor = float(np.clip(def_factor, 0.90, 1.10))  # tighter
            league_pace = team_ctx["PACE"].mean()
            opp_pace    = opp_row.get("PACE", np.nan)
            pace_factor = float(opp_pace) / float(league_pace) if pd.notna(league_pace) and pd.notna(opp_pace) and league_pace > 0 else 1.0
            pace_factor = float(np.sqrt(np.clip(pace_factor, 0.90, 1.10)))  # gentler
            adj_rate = per_min * def_factor * pace_factor

            # --- Projected minutes (user override) ---
            min_recent = valid_sources.get("recent", pd.Series()).get("MIN", np.nan)
            min_season = valid_sources.get("season", pd.Series()).get("MIN", np.nan)
            if pd.isna(min_recent) and pd.isna(min_season):
                min_proj_seed = blend["MIN"] if pd.notna(blend["MIN"]) else 30.0
            elif pd.isna(min_recent):
                min_proj_seed = float(min_season)
            elif pd.isna(min_season):
                min_proj_seed = float(min_recent)
            else:
                min_proj_seed = 0.65 * float(min_recent) + 0.35 * float(min_season)
            min_proj_seed = float(np.clip(min_proj_seed if pd.notna(min_proj_seed) else 30.0, 10.0, 42.0))

            min_proj = st.number_input(
                "Projected MIN (editable)", min_value=5.0, max_value=48.0,
                value=float(min_proj_seed), step=0.5,
                help="Adjust minutes to scale PTS/REB/AST/3PM and PRA outputs"
            )

            # --- Final projection counts ---
            proj = pd.Series(index=["PTS", "REB", "AST", "FG3M", "MIN"], dtype=float)
            for m in ["PTS", "REB", "AST", "FG3M"]:
                proj[m] = float(adj_rate.get(m, np.nan) * min_proj) if pd.notna(adj_rate.get(m, np.nan)) else np.nan
            proj["MIN"] = float(min_proj)
            proj["PRA"] = (proj["PTS"] if pd.notna(proj["PTS"]) else 0) + \
                          (proj["REB"] if pd.notna(proj["REB"]) else 0) + \
                          (proj["AST"] if pd.notna(proj["AST"]) else 0)

            # --- Confidence band (uses selected z from 90/80/70 exactly) ---
            show_ci = st.checkbox("Show confidence band", value=True)
            ci_df = None
            if show_ci:
                hist = logs.head(15).copy()
                for c in ["PTS", "REB", "AST", "FG3M", "MIN"]:
                    if c not in hist.columns: hist[c] = 0
                hist = hist[hist["MIN"] > 0]
                if not hist.empty:
                    # Per-minute distributions
                    pm = pd.DataFrame({
                        "PTS":  hist["PTS"]/hist["MIN"],
                        "REB":  hist["REB"]/hist["MIN"],
                        "AST":  hist["AST"]/hist["MIN"],
                        "FG3M": hist["FG3M"]/hist["MIN"],
                    })
                    # Winsorize
                    q05 = pm.quantile(0.05, numeric_only=True)
                    q95 = pm.quantile(0.95, numeric_only=True)
                    pm = pm.clip(lower=q05, upper=q95, axis=1)

                    # Robust SD via MAD
                    med = pm.median(numeric_only=True)
                    mad = (pm - med).abs().median(numeric_only=True)
                    robust_sd = 1.4826 * mad
                    fallback_sd = pm.std(numeric_only=True).fillna(0.0)
                    sd_pm = robust_sd.fillna(fallback_sd)

                    # Small-sample shrinkage
                    n = len(pm)
                    shrink_k = 10
                    sd_pm = sd_pm * (n / (n + shrink_k))

                    # Minutes volatility multiplier
                    min_std  = float(hist["MIN"].std() or 0.0)
                    min_mean = float(hist["MIN"].mean() or 1.0)
                    vol_ratio = min_std / max(min_mean, 1.0)
                    ci_mult = 1.0 + alpha_min_vol * vol_ratio

                    # Error per minute -> counts
                    err_pm = z * sd_pm
                    err = err_pm * np.sqrt(max(min_proj, 1.0)) * ci_mult

                    base = proj[["PTS","REB","AST","FG3M"]]
                    rate_est = adj_rate[["PTS","REB","AST","FG3M"]]
                    lo_raw = (rate_est - err.clip(lower=0)) * min_proj
                    hi_raw = (rate_est + err.clip(lower=0)) * min_proj

                    # Relative caps around point estimate
                    lo_cap = base * (1.0 - rel_cap)
                    hi_cap = base * (1.0 + rel_cap)
                    lo = pd.concat([lo_raw, lo_cap], axis=1).max(axis=1)
                    hi = pd.concat([hi_raw, hi_cap], axis=1).min(axis=1)

                    lo = np.minimum(lo, base)
                    hi = np.maximum(hi, base)

                    ci_df = pd.DataFrame({"Low": lo.round(2), "Proj": base.round(2), "High": hi.round(2)})

            # --- Render outputs (with 3PM label) ---
            out = proj[["MIN", "PTS", "REB", "AST", "PRA", "FG3M"]].to_frame("Projection").T.round(2)
            out = out.rename(columns={"FG3M": "3PM"})
            st.dataframe(out, use_container_width=True, height=90)

            # Show weight summary (normalized)
            st.caption(
                "Normalized blend weights → "
                + ", ".join([f"{k.capitalize()}: {norm_w.get(k,0):.2f}" for k in ["recent","season","prev","career","vsopp"]])
                + f" • CI: {z_level} • Defense adj: {def_factor:.3f} • Pace adj: {pace_factor:.3f}"
            )

            if ci_df is not None and not ci_df.empty:
                st.markdown("**Confidence Band (counts):**")
                # Reorder rows to match our display order
                ci_disp = ci_df.reindex(["PTS","REB","AST","FG3M"])
                ci_disp = ci_disp.rename(index={"FG3M":"3PM"})
                st.dataframe(ci_disp, use_container_width=True, height=170)

            st.markdown(
                """
                *Notes:*  
                • Weights are **normalized** automatically—set any combination between 0 and 1.  
                • If all weights are set to 0, the model defaults to **Recent = 1.00**.  
                • **Projected MIN** scales all counting stats.  
                • CI uses robust per-minute variance with minutes-volatility adjustment and a relative cap for stability.
                """
            )

        except Exception as e:
            st.info(f"Projection temporarily unavailable: {e}")

    # ----------------------- Footer -----------------------
    st.caption("Notes: Opponent metrics are NBA-only ‘Regular Season’ through today’s ET date (5-min cache). MIN reflects totals from Base (Totals); PACE/ratings from Advanced (PerGame). Opponent last-5 uses LeagueGameFinder with a robust fallback. Average rows are computed over the shown 5 games.")

# =====================================================================
# Team Dashboard (as provided; unchanged)
# =====================================================================
def team_dashboard():
    st.title("NBA Team Dashboard")

    # ----------------------- Data Fetchers (cached) -----------------------
    @st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=False)
    def get_teams_df():
        t = pd.DataFrame(static_teams.get_teams())
        t = t.rename(columns={"id": "TEAM_ID", "full_name": "TEAM_NAME", "abbreviation": "TEAM_ABBREVIATION"})
        t["TEAM_ID"] = t["TEAM_ID"].astype(int)
        return t[["TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION"]]

    @st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
    def fetch_league_team_traditional(season: str) -> pd.DataFrame:
        frames = _retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                measure_type_detailed_defense="Base",
                per_mode_detailed="PerGame",
            ),
        )
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return df
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        # normalize dtypes
        for c in df.columns:
            if c not in ("TEAM_NAME","TEAM_ABBREVIATION"):
                df[c] = pd.to_numeric(df[c], errors="ignore")
        return df.reset_index(drop=True)

    @st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
    def fetch_league_team_advanced(season: str) -> pd.DataFrame:
        frames = _retry_api(
            leaguedashteamstats.LeagueDashTeamStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
            ),
        )
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return df
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        for c in df.columns:
            if c not in ("TEAM_NAME","TEAM_ABBREVIATION"):
                df[c] = pd.to_numeric(df[c], errors="ignore")
        return df.reset_index(drop=True)

    @st.cache_data(ttl=CACHE_HOURS*3600, show_spinner=True)
    def fetch_league_players_pg(season: str, last_n_games: int) -> pd.DataFrame:
        frames = _retry_api(
            LeagueDashPlayerStats,
            dict(
                season=season,
                season_type_all_star="Regular Season",
                league_id_nullable="00",
                per_mode_detailed="PerGame",
                last_n_games=last_n_games,   # 0=season, 5, 15
            ),
        )
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return df
        df = df[df["TEAM_ID"].astype(str).str.startswith("161061")].copy()
        return df.reset_index(drop=True)

    # ----------------------- Helpers -----------------------
    def _fmt(v, pct=False, d=1):
        if pd.isna(v):
            return "—"
        if pct:
            return f"{float(v)*100:.{d}f}%"
        return f"{float(v):.{d}f}"

    def _rank_series(df: pd.DataFrame, col: str, ascending: bool) -> pd.Series:
        if col not in df.columns:
            return pd.Series([np.nan]*len(df))
        return df[col].rank(ascending=ascending, method="min")

    def _add_fg2(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "FG2M" not in out.columns:
            out["FG2M"] = pd.to_numeric(out.get("FGM", 0), errors="coerce") - pd.to_numeric(out.get("FG3M", 0), errors="coerce")
        if "FG2A" not in out.columns:
            out["FG2A"] = pd.to_numeric(out.get("FGA", 0), errors="coerce") - pd.to_numeric(out.get("FG3A", 0), errors="coerce")
        return out

    def _select_roster_columns(df: pd.DataFrame) -> pd.DataFrame:
        colmap = {
            "TEAM_ABBREVIATION": "TEAM",
            "PLAYER_NAME": "PLAYER_NAME",
            "AGE": "AGE",
            "GP": "GP",
            "MIN": "MIN",
            "PTS": "PTS",
            "REB": "REB",
            "AST": "AST",
            "FG2M": "FG2M",
            "FG2A": "FG2A",
            "FG3M": "FG3M",
            "FG3A": "FG3A",
            "FTM": "FTM",
            "FTA": "FTA",
            "OREB": "OREB",
            "DREB": "DREB",
            "STL": "STL",
            "BLK": "BLK",
            "TOV": "TOV",
            "PF": "PF",
            "PLUS_MINUS": "PLUS_MINUS",
        }
        for c in colmap.keys():
            if c not in df.columns:
                df[c] = np.nan
        out = df[list(colmap.keys())].copy()
        out.columns = list(colmap.values())
        return out

    def _auto_height(df: pd.DataFrame, row_px=34, header_px=38, max_px=900):
        rows = max(len(df), 1)
        return min(max_px, header_px + row_px * rows + 8)

    def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Guarantee columns exist (fill NaN if absent) to avoid KeyError on selection."""
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out

    # ----------------------- Sidebar -----------------------
    with st.sidebar:
        st.header("Team Filters")
        season = st.selectbox("Season (Team Tab)", _season_labels(2015, dt.datetime.utcnow().year), index=0, key="team_season_sel")
        teams_df = get_teams_df()
        team_name = st.selectbox("Team", sorted(teams_df["TEAM_NAME"].tolist()))
        team_row = teams_df[teams_df["TEAM_NAME"] == team_name].iloc[0]
        team_id = int(team_row["TEAM_ID"])
        team_abbr = team_row["TEAM_ABBREVIATION"]

    # ----------------------- Load league data -----------------------
    with st.spinner("Loading league team stats..."):
        trad = fetch_league_team_traditional(season)
        adv = fetch_league_team_advanced(season)

    if trad.empty or adv.empty:
        st.error("Could not load team stats. Try refreshing or changing the season.")
        st.stop()

    # Columns we want from each table (guarded)
    TRAD_WANTED = [
        "TEAM_ID","TEAM_NAME","TEAM_ABBREVIATION","GP","W","L","W_PCT",
        "MIN","PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT",
        "OREB","DREB","REB","AST","STL","BLK","TOV","PLUS_MINUS"
    ]
    ADV_WANTED = ["TEAM_ID","OFF_RATING","DEF_RATING","NET_RATING","PACE"]

    trad_g = _ensure_cols(trad, TRAD_WANTED)[TRAD_WANTED].copy()
    adv_g  = _ensure_cols(adv,  ADV_WANTED)[ADV_WANTED].copy()

    # Merge traditional + advanced
    merged = pd.merge(trad_g, adv_g, on="TEAM_ID", how="left")

    # League ranks (1 = best); compute only if column exists
    def _safe_rank(col, ascending):
        return _rank_series(merged, col, ascending=ascending)

    ranks = pd.DataFrame({"TEAM_ID": merged["TEAM_ID"]})
    ranks["PTS"]         = _safe_rank("PTS", ascending=False)
    ranks["OFF_RATING"]  = _safe_rank("OFF_RATING", ascending=False)
    ranks["DEF_RATING"]  = _safe_rank("DEF_RATING", ascending=True)
    ranks["NET_RATING"]  = _safe_rank("NET_RATING", ascending=False)
    ranks["PACE"]        = _safe_rank("PACE", ascending=False)
    ranks["FG_PCT"]      = _safe_rank("FG_PCT", ascending=False)
    ranks["FGA"]         = _safe_rank("FGA", ascending=False)
    ranks["FG3_PCT"]     = _safe_rank("FG3_PCT", ascending=False)
    ranks["FG3A"]        = _safe_rank("FG3A", ascending=False)
    ranks["FT_PCT"]      = _safe_rank("FT_PCT", ascending=False)
    ranks["FTM"]         = _safe_rank("FTM", ascending=False)
    ranks["STL"]         = _safe_rank("STL", ascending=False)
    ranks["BLK"]         = _safe_rank("BLK", ascending=False)
    ranks["TOV"]         = _safe_rank("TOV", ascending=True)   # lower is better
    ranks["PLUS_MINUS"]  = _safe_rank("PLUS_MINUS", ascending=False)

    n_teams = len(merged)

    # Selected team row
    sel = merged[merged["TEAM_ID"] == team_id]
    if sel.empty:
        st.error("Selected team not found in this season dataset.")
        st.stop()

    tr = sel.iloc[0]
    rr = ranks[ranks["TEAM_ID"] == team_id].iloc[0]
    record = (
        f"{int(tr['W'])}–{int(tr['L'])}"
        if pd.notna(tr.get("W")) and pd.notna(tr.get("L"))
        else "—"
    )

    # ----------------------- Header -----------------------
    st.subheader(f"{tr['TEAM_NAME']} — {season}")

    def _metric(col, label, value, rank, pct=False, d=1):
        val = _fmt(value, pct=pct, d=d)
        delta = f"Rank {int(rank)}/{n_teams}" if pd.notna(rank) else None
        col.metric(label, val, delta=delta)

    # First line: Record
    c_rec, _, _, _, _ = st.columns(5)
    c_rec.metric("Record", record)

    # Row 1: Scoring / ratings / pace
    c1, c2, c3, c4, c5 = st.columns(5)
    _metric(c1, "PTS",        tr.get("PTS"),        rr.get("PTS"))
    _metric(c2, "NET Rating", tr.get("NET_RATING"), rr.get("NET_RATING"))
    _metric(c3, "OFF Rating", tr.get("OFF_RATING"), rr.get("OFF_RATING"))
    _metric(c4, "DEF Rating", tr.get("DEF_RATING"), rr.get("DEF_RATING"))
    _metric(c5, "PACE",       tr.get("PACE"),       rr.get("PACE"))

    # Row 2: FG / 3P / FT
    c6, c7, c8, c9, c10 = st.columns(5)
    _metric(c6,  "FG%",  tr.get("FG_PCT"),  rr.get("FG_PCT"),  pct=True)
    _metric(c7,  "FGA",  tr.get("FGA"),     rr.get("FGA"))
    _metric(c8,  "3P%",  tr.get("FG3_PCT"), rr.get("FG3_PCT"), pct=True)
    _metric(c9,  "3PA",  tr.get("FG3A"),    rr.get("FG3A"))
    _metric(c10, "FT%",  tr.get("FT_PCT"),  rr.get("FT_PCT"),  pct=True)

    # Row 3: Makes + defense/misc
    c11, c12, c13, c14, c15 = st.columns(5)
    _metric(c11, "FTM",       tr.get("FTM"),        rr.get("FTM"))
    _metric(c12, "STL",       tr.get("STL"),        rr.get("STL"))
    _metric(c13, "BLK",       tr.get("BLK"),        rr.get("BLK"))
    _metric(c14, "TOV",       tr.get("TOV"),        rr.get("TOV"))
    _metric(c15, "+/-",       tr.get("PLUS_MINUS"), rr.get("PLUS_MINUS"))

    st.caption("Ranks are relative to all NBA teams (1 = best). Shooting % tiles display percentage; volume tiles show per-game counts.")

    # ----------------------- Roster tables (stacked) -----------------------
    with st.spinner("Loading roster per-game (season / last 5 / last 15)..."):
        season_pg = fetch_league_players_pg(season, last_n_games=0)
        last5_pg  = fetch_league_players_pg(season, last_n_games=5)
        last15_pg = fetch_league_players_pg(season, last_n_games=15)

    def _prep_roster(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        out = df[df["TEAM_ID"] == team_id].copy()
        if out.empty:
            return out
        num_like = ["AGE","GP","MIN","PTS","REB","AST","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB","STL","BLK","TOV","PF","PLUS_MINUS"]
        for c in num_like:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out = _add_fg2(out)
        out = _select_roster_columns(out)
        if "MIN" in out.columns:
            out = out.sort_values("MIN", ascending=False).reset_index(drop=True)
        return out

    def _num_fmt_map(df: pd.DataFrame):
        fmts = {}
        for c in df.columns:
            if c in ("TEAM","PLAYER_NAME"):
                continue
            fmts[c] = "{:.1f}"
        return fmts

    season_tbl = _prep_roster(season_pg, team_id)
    last5_tbl  = _prep_roster(last5_pg, team_id)
    last15_tbl = _prep_roster(last15_pg, team_id)

    st.markdown("### Roster — Season Per-Game")
    if season_tbl.empty:
        st.info("No season per-game data for this team.")
    else:
        st.dataframe(
            season_tbl.style.format(_num_fmt_map(season_tbl)),
            use_container_width=True,
            height=_auto_height(season_tbl),
        )

    st.markdown("### Roster — Last 5 Games (Per-Game)")
    if last5_tbl.empty:
        st.info("No Last 5 per-game data for this team.")
    else:
        st.dataframe(
            last5_tbl.style.format(_num_fmt_map(last5_tbl)),
            use_container_width=True,
            height=_auto_height(last5_tbl),
        )

    st.markdown("### Roster — Last 15 Games (Per-Game)")
    if last15_tbl.empty:
        st.info("No Last 15 per-game data for this team.")
    else:
        st.dataframe(
            last15_tbl.style.format(_num_fmt_map(last15_tbl)),
            use_container_width=True,
            height=_auto_height(last15_tbl),
        )

    # ----------------------- Footer -----------------------
    st.caption(
        "Notes: Team stats from NBA.com LeagueDashTeamStats (Traditional & Advanced, Per-Game). "
        "Player roster per-game from LeagueDashPlayerStats with last_n_games filters (0/5/15). "
        "FG2M/FG2A are computed as (FGM−FG3M)/(FGA−FG3A). Tables are sorted by MIN."
    )

# =====================================================================
# App Tabs
# =====================================================================
tab1, tab2 = st.tabs(["Player Dashboard", "Team Dashboard"])
with tab1:
    player_dashboard()
with tab2:
    team_dashboard()
