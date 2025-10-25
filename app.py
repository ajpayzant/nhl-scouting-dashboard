# app.py — NHL Player Scouting Dashboard (club-stats only; no standings)
# ----------------------------------------------------------------------
# Run locally:  pip install -r requirements.txt && streamlit run app.py
# requirements.txt:
#   streamlit
#   pandas
#   numpy
#   requests
#   plotly

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import plotly.express as px

st.set_page_config(page_title="NHL Player Scouting Dashboard", layout="wide")
pd.set_option("display.max_columns", 0)

TEAM_ABBRS = [
    "ANA","ARI","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET","EDM","FLA","LAK",
    "MIN","MTL","NSH","NJD","NYI","NYR","OTT","PHI","PIT","SEA","SJS","STL","TBL","TOR",
    "VAN","VGK","WPG","WSH"
]

# ----------------------------- Utils -----------------------------
def get_json(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def nhl_season_label_for_date(d=None) -> int:
    d = d or dt.date.today()
    start = d.year if d.month >= 7 else d.year - 1
    return int(f"{start}{start+1}")

def season_options(num=8):
    """[current, prev1, prev2, ...] as ints like 20252026."""
    cur = nhl_season_label_for_date()
    start = int(str(cur)[:4])
    return [int(f"{start-i}{start-i+1}") for i in range(num)]

def safe_div(a, b, default=np.nan):
    try:
        if b in (0, None) or (isinstance(b, float) and np.isnan(b)):
            return default
        v = a / b
        return float(v) if np.isfinite(v) else default
    except Exception:
        return default

def blend_vals(r, s, c, w_recent=0.55, w_season=0.30, w_career=0.15):
    tot = max(w_recent + w_season + w_career, 1e-9)
    return (w_recent/tot)*r + (w_season/tot)*s + (w_career/tot)*c

def to_minutes(time_str):
    """Parse 'MM:SS' or 'H:MM:SS' to float minutes."""
    if pd.isna(time_str):
        return np.nan
    s = str(time_str)
    parts = s.split(":")
    try:
        if len(parts) == 2:
            mm, ss = int(parts[0]), int(parts[1])
            return mm + ss/60.0
        elif len(parts) == 3:
            hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
            return hh*60 + mm + ss/60.0
    except:
        return np.nan
    return np.nan

def auto_height(df, row_px=35, min_px=120, max_px=600):
    rows = max(1, len(df))
    return int(np.clip(rows * row_px + 60, min_px, max_px))

# ----------------------------- API wrappers -----------------------------
def search_players(q: str, limit=25) -> pd.DataFrame:
    data = get_json(f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit={limit}&q={q}")
    return pd.json_normalize(data)

def player_landing(player_id: int) -> dict:
    return get_json(f"https://api-web.nhle.com/v1/player/{player_id}/landing")

def player_game_log(player_id: int, season: int, game_type: int = 2) -> pd.DataFrame:
    data = get_json(f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/{game_type}")
    df = pd.json_normalize(data.get("gameLog", []))
    if not df.empty and "gameDate" in df.columns:
        df["gameDate"] = pd.to_datetime(df["gameDate"])
        df = df.sort_values("gameDate", ascending=False).reset_index(drop=True)
    return df

def fetch_club_stats_goalies_safe(team_abbr: str, season: int, game_type: int = 2) -> pd.DataFrame:
    """
    Always use club-stats for team-level baselines.
    If current season 404s for a team, auto-fallback to previous season for that team.
    """
    try:
        data = get_json(f"https://api-web.nhle.com/v1/club-stats/{team_abbr}/{season}/{game_type}")
        return pd.json_normalize(data.get("goalies", []))
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            # fallback to previous season for this team
            start = int(str(season)[:4])
            prev = int(f"{start-1}{start}")
            try:
                st.info(f"{team_abbr} club-stats not live for {str(season)[:4]}-{str(season)[4:]}. Using {str(prev)[:4]}-{str(prev)[4:]} for baselines.")
                data = get_json(f"https://api-web.nhle.com/v1/club-stats/{team_abbr}/{prev}/{game_type}")
                return pd.json_normalize(data.get("goalies", []))
            except requests.exceptions.HTTPError:
                return pd.DataFrame()
        return pd.DataFrame()

# ----------------------------- League & Opponent baselines (club-stats only) -----------------------------
def clubstats_sa_ga_pg_for_team(team_abbr: str, season: int, game_type: int = 2):
    """
    From goalies table:
      SA_sum = sum(shotsAgainst), SV_sum = sum(saves), GP_sum = sum(gamesPlayed)
      SA/GP = SA_sum / GP_sum; GA/GP = (SA_sum - SV_sum) / GP_sum
    Uses per-team season fallback (previous season) if needed.
    """
    gdf = fetch_club_stats_goalies_safe(team_abbr, season, game_type)
    if gdf.empty:
        return (np.nan, np.nan, 0.0)  # SA/GP, GA/GP, GP_sum
    sa = pd.to_numeric(gdf.get("shotsAgainst", 0), errors="coerce").fillna(0).sum()
    sv = pd.to_numeric(gdf.get("saves", 0), errors="coerce").fillna(0).sum()
    gp = pd.to_numeric(gdf.get("gamesPlayed", 0), errors="coerce").fillna(0).sum()
    sa_pg = safe_div(sa, gp)
    ga_pg = safe_div(sa - sv, gp)
    return (sa_pg, ga_pg, gp)

def league_baselines_from_clubstats(season: int, game_type: int = 2):
    """
    Sweep all teams via club-stats with per-team season fallback.
    Compute league SA/GP and GA/GP by averaging teams with finite values (coverage-insensitive).
    """
    recs = []
    for abbr in TEAM_ABBRS:
        sa_pg, ga_pg, gp = clubstats_sa_ga_pg_for_team(abbr, season, game_type)
        recs.append({"teamAbbrev": abbr, "SA_per_GP": sa_pg, "GA_per_GP": ga_pg, "GP_sum": gp})
    df = pd.DataFrame(recs)
    league_sa_pg = float(np.nanmean(pd.to_numeric(df["SA_per_GP"], errors="coerce")))
    league_ga_pg = float(np.nanmean(pd.to_numeric(df["GA_per_GP"], errors="coerce")))
    return df, league_sa_pg, league_ga_pg

def opponent_metrics(opponent: str, season: int, game_type: int, league_sa_pg: float, league_ga_pg: float):
    opp_sa_pg, opp_ga_pg, _ = clubstats_sa_ga_pg_for_team(opponent, season, game_type)
    # Adjusters
    shots_adj = 1.0 if not np.isfinite(opp_sa_pg) or not np.isfinite(league_sa_pg) else float(opp_sa_pg / max(league_sa_pg, 1e-9))
    score_adj = 1.0 if not np.isfinite(opp_ga_pg) or not np.isfinite(league_ga_pg) else float(league_ga_pg / max(opp_ga_pg, 1e-9))
    return opp_ga_pg, opp_sa_pg, score_adj, shots_adj

# ----------------------------- Player & projections -----------------------------
def resolve_player_team(bio: dict, gl_df: pd.DataFrame, players_df: pd.DataFrame, landing: dict):
    t = (bio or {}).get("currentTeamAbbrev")
    if t: return str(t)
    if isinstance(gl_df, pd.DataFrame) and "teamAbbrev" in gl_df.columns and not gl_df.empty:
        v = gl_df["teamAbbrev"].dropna().astype(str)
        if len(v): return v.iloc[0]
    row = players_df.head(1)
    if not row.empty and pd.notna(row.iloc[0].get("teamAbbrev")):
        return str(row.iloc[0]["teamAbbrev"])
    stot = landing.get("seasonTotals")
    if isinstance(stot, list) and len(stot):
        st_df = pd.json_normalize(stot)
        for cand in ["teamAbbrev","teamTricode","teamCommonName.default"]:
            if cand in st_df.columns:
                vals = st_df[cand].dropna().astype(str).tolist()
                if vals:
                    return vals[-1]
    return None

def series_mean(s, k=None):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return np.nan
    if k: s = s.head(k)
    return float(s.mean())

def career_pg_from_season_totals(stot_df: pd.DataFrame, num_col, gp_col="gamesPlayed"):
    if stot_df.empty or num_col not in stot_df.columns or gp_col not in stot_df.columns:
        return np.nan
    num = pd.to_numeric(stot_df[num_col], errors="coerce").fillna(0).sum()
    gp  = pd.to_numeric(stot_df[gp_col],  errors="coerce").fillna(0).sum()
    return safe_div(num, gp)

def skater_projection(gl_df: pd.DataFrame, stot_df: pd.DataFrame, w_recent, w_season, w_career, shots_adj, score_adj, K=5):
    # recent
    sog_r = series_mean(gl_df["shots"], K)
    g_r   = series_mean(gl_df["goals"], K)
    a_r   = series_mean(gl_df["assists"], K)
    sh_r  = safe_div(g_r, sog_r)

    # season
    sog_s = series_mean(gl_df["shots"])
    g_s   = series_mean(gl_df["goals"])
    a_s   = series_mean(gl_df["assists"])
    sh_s  = safe_div(g_s, sog_s)

    # career (from season totals)
    sog_c = career_pg_from_season_totals(stot_df, "shots")
    g_c   = career_pg_from_season_totals(stot_df, "goals")
    a_c   = career_pg_from_season_totals(stot_df, "assists")
    sh_c  = safe_div(g_c, sog_c)

    sog_bl = blend_vals(sog_r, sog_s, sog_c, w_recent, w_season, w_career)
    sh_bl  = blend_vals(sh_r,  sh_s,  sh_c,  w_recent, w_season, w_career)
    a_bl   = blend_vals(a_r,   a_s,   a_c,   w_recent, w_season, w_career)

    SOG = max(0.0, (sog_bl if np.isfinite(sog_bl) else 0.0) * shots_adj)
    G   = max(0.0, SOG * float(np.clip(sh_bl if np.isfinite(sh_bl) else 0.10, 0, 1))) * score_adj
    A   = max(0.0, (a_bl if np.isfinite(a_bl) else 0.0) * score_adj)
    PTS = G + A

    return pd.DataFrame({"Stat":["PTS","G","A","SOG"], "Proj":[PTS, G, A, SOG]}).round(2)

def goalie_projection(gl_df: pd.DataFrame, stot_df: pd.DataFrame, w_recent, w_season, w_career, shots_adj, K=5):
    def mean_col(col, k=None):
        s = pd.to_numeric(gl_df[col], errors="coerce").dropna()
        if s.empty: return np.nan
        if k: s = s.head(k)
        return float(s.mean())

    SA_r = mean_col("shotsAgainst", K)
    SA_s = mean_col("shotsAgainst")
    SA_c = career_pg_from_season_totals(stot_df, "shotsAgainst")

    if "savePct" in gl_df.columns and gl_df["savePct"].notna().any():
        SVP_r = series_mean(gl_df["savePct"], K)
        SVP_s = series_mean(gl_df["savePct"])
    else:
        SVP_r = safe_div(series_mean(gl_df["saves"], K), series_mean(gl_df["shotsAgainst"], K))
        SVP_s = safe_div(series_mean(gl_df["saves"]), series_mean(gl_df["shotsAgainst"]))

    if not stot_df.empty and {"saves","shotsAgainst"}.issubset(stot_df.columns):
        SVP_c = safe_div(pd.to_numeric(stot_df["saves"], errors="coerce").sum(),
                         pd.to_numeric(stot_df["shotsAgainst"], errors="coerce").sum())
    else:
        SVP_c = np.nan

    SA_bl  = blend_vals(SA_r, SA_s, SA_c, w_recent, w_season, w_career)
    SVP_bl = blend_vals(SVP_r, SVP_s, SVP_c, w_recent, w_season, w_career)

    SA = max(0.0, (SA_bl if np.isfinite(SA_bl) else 0.0) * shots_adj)
    SV = max(0.0, SA * float(np.clip(SVP_bl if np.isfinite(SVP_bl) else 0.900, 0, 1)))
    SVP = safe_div(SV, SA)

    return pd.DataFrame({"Stat":["SA","SV","SV%"], "Proj":[SA, SV, SVP]}).round(2)

def last_n_vs_opponent(player_id: int, opponent_abbr: str, seasons: list, n=5, game_type=2):
    frames = []
    for s in seasons:
        df = player_game_log(player_id, s, game_type)
        if df.empty: continue
        if "opponentAbbrev" in df.columns:
            m = df[df["opponentAbbrev"] == opponent_abbr].copy()
            if not m.empty:
                frames.append(m)
        if sum(len(x) for x in frames) >= n:
            break
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values("gameDate", ascending=False)
    return out.head(n)

# ----------------------------- UI -----------------------------
st.title("NHL Player Scouting Dashboard")

with st.sidebar:
    st.subheader("Filters")
    SEASONS = season_options(8)
    # Default to 2025-26 (index 0)
    SEASON = st.selectbox("Season", SEASONS, index=0, format_func=lambda x: f"{str(x)[:4]}-{str(x)[4:]}")
    GAME_TYPE = st.selectbox("Game Type", {2:"Regular Season", 3:"Playoffs"}, index=0, format_func=lambda x: {2:"Regular",3:"Playoffs"}[x])
    query = st.text_input("Player search", "McDavid")
    recent_window = st.select_slider("Recent window (games)", options=[5,10,15,20], value=5)
    st.markdown("---")
    st.caption("Blend Weights")
    w_recent = st.slider("Recent", 0.0, 1.0, 0.55, 0.05)
    w_season = st.slider("Season", 0.0, 1.0, 0.30, 0.05)
    w_career = st.slider("Career", 0.0, 1.0, 0.15, 0.05)

# Search & select player
players_df = search_players(query)
if players_df.empty:
    st.warning("No players found. Try another name.")
    st.stop()

players_df = players_df.sort_values(["active"], ascending=False).reset_index(drop=True)
player_choices = players_df["name"].tolist()
player_name = st.selectbox("Choose player", player_choices, index=0)
sel = players_df.loc[players_df["name"] == player_name].iloc[0]
player_id = int(sel["playerId"])
position_code = str(sel.get("positionCode", ""))

# Player data
landing = player_landing(player_id)
bio = landing.get("playerBio", {}) or {}
season_totals = pd.json_normalize(landing.get("seasonTotals", []))
gl = player_game_log(player_id, SEASON, GAME_TYPE)
# If no logs this season, fallback to earlier seasons for charting
if gl.empty:
    for s in SEASONS[1:]:
        gl = player_game_log(player_id, s, GAME_TYPE)
        if not gl.empty:
            st.info(f"No logs for {str(SEASON)[:4]}-{str(SEASON)[4:]}. Showing {str(s)[:4]}-{str(s)[4:]} instead for charts.")
            break

# Resolve team (best-effort)
player_team = resolve_player_team(bio, gl, players_df, landing)

# League baselines (club-stats sweep only; with per-team season fallback)
league_df, league_sa_pg, league_ga_pg = league_baselines_from_clubstats(SEASON, GAME_TYPE)

# Opponent picker (static list; default to most recent opponent if available)
team_options = TEAM_ABBRS
opp_default = gl["opponentAbbrev"].dropna().astype(str).iloc[0] if ("opponentAbbrev" in gl.columns and not gl.empty and pd.notna(gl.loc[0,"opponentAbbrev"])) else ("TOR" if (player_team == "MTL") else "MTL")
if opp_default not in team_options:
    opp_default = team_options[0]
opponent = st.selectbox("Opponent", team_options, index=team_options.index(opp_default))

# Opponent metrics & adjusters (from club-stats only)
opp_ga_pg, opp_sa_pg, score_adj, shots_adj = opponent_metrics(opponent, SEASON, GAME_TYPE, league_sa_pg, league_ga_pg)

# Header
col1, col2 = st.columns([2,1])
with col1:
    full_name = bio.get("fullName") or player_name
    age = bio.get("age", "—")
    pos = bio.get("positionCode", position_code)
    seasons_played = len(season_totals["seasonId"].unique()) if not season_totals.empty and "seasonId" in season_totals.columns else "—"
    # current season GP (from seasonTotals if present)
    gp_cur = "—"
    if not season_totals.empty and {"seasonId","gamesPlayed"}.issubset(season_totals.columns):
        row = season_totals.loc[season_totals["seasonId"] == SEASON]
        if not row.empty:
            gp_cur = int(pd.to_numeric(row["gamesPlayed"], errors="coerce").fillna(0).iloc[0])
    st.markdown(f"### {full_name}")
    st.markdown(f"**Pos**: {pos} &nbsp;&nbsp; **Age**: {age} &nbsp;&nbsp; **Seasons**: {seasons_played} &nbsp;&nbsp; **{str(SEASON)[:4]}-{str(SEASON)[4:]} GP**: {gp_cur}")
with col2:
    st.markdown(f"**Opponent:** {opponent}")
    if np.isfinite(opp_ga_pg) and np.isfinite(opp_sa_pg):
        st.markdown(f"GA/GP: **{opp_ga_pg:.2f}**  •  SA/GP: **{opp_sa_pg:.2f}**")
        st.caption(f"League baselines — GA/GP: {league_ga_pg:.2f} • SA/GP: {league_sa_pg:.2f}")
    else:
        st.markdown("Opponent metrics unavailable (using neutral adjusters).")
        st.caption(f"League baselines — GA/GP: {league_ga_pg:.2f} • SA/GP: {league_sa_pg:.2f}")

st.markdown("---")

# Projections
is_goalie = (pos == "G")
if is_goalie:
    proj = goalie_projection(gl, season_totals, w_recent, w_season, w_career, shots_adj, K=recent_window)
else:
    proj = skater_projection(gl, season_totals, w_recent, w_season, w_career, shots_adj, score_adj, K=recent_window)

st.subheader("Projection Summary")
st.table(proj.set_index("Stat"))

# Recent trends
st.subheader("Recent Trends")
if gl.empty:
    st.info("No game logs available to chart.")
else:
    plot_cols = []
    if is_goalie:
        if "shotsAgainst" in gl.columns: plot_cols.append(("shotsAgainst","Shots Against"))
        if "saves" in gl.columns:        plot_cols.append(("saves","Saves"))
    else:
        tmp = gl.copy()
        tmp["PTS"] = pd.to_numeric(tmp.get("goals",0), errors="coerce").fillna(0) + pd.to_numeric(tmp.get("assists",0), errors="coerce").fillna(0)
        gl = tmp
        for c, label in [("PTS","Points"),("goals","Goals"),("assists","Assists"),("shots","Shots on Goal")]:
            if c in gl.columns: plot_cols.append((c,label))

    plot_df = gl.head(min(len(gl), recent_window*2)).iloc[::-1]
    for c, label in plot_cols:
        if c not in plot_df.columns: continue
        fig = px.line(plot_df, x="gameDate", y=c, markers=True, title=label)
        fig.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

# Last 5 games (whole numbers for single-game stats)
st.subheader(f"Last 5 Games — {str(SEASON)[:4]}-{str(SEASON)[4:]}")
if gl.empty:
    st.info("No games this season/selection.")
else:
    last5 = gl.sort_values("gameDate", ascending=False).head(5).copy()
    if not is_goalie:
        last5["PTS"] = pd.to_numeric(last5.get("goals",0), errors="coerce").fillna(0) + pd.to_numeric(last5.get("assists",0), errors="coerce").fillna(0)
        meta = ["gameDate","teamAbbrev","opponentAbbrev","homeRoad"]
        stats = ["PTS","goals","assists","shots"]
        cols  = [c for c in meta + stats if c in last5.columns]
        ints  = [c for c in stats if c in cols]
        st.dataframe(last5[cols].style.format({c:"{:.0f}" for c in ints}), use_container_width=True, height=auto_height(last5[cols]))
        avg = last5[stats].mean(numeric_only=True).to_frame().T.round(2); avg.index=["Average (Last 5)"]
        st.table(avg)
    else:
        meta = ["gameDate","teamAbbrev","opponentAbbrev","homeRoad"]
        stats = ["shotsAgainst","saves"]
        cols  = [c for c in meta + stats if c in last5.columns]
        ints  = [c for c in stats if c in cols]
        st.dataframe(last5[cols].style.format({c:"{:.0f}" for c in ints}), use_container_width=True, height=auto_height(last5[cols]))
        avg = last5[stats].mean(numeric_only=True).to_frame().T.round(2); avg.index=["Average (Last 5)"]
        st.table(avg)

# Last 5 vs Opponent (cross-season)
st.subheader(f"Last 5 vs {opponent} — Most Recent Seasons")
vs = last_n_vs_opponent(player_id, opponent, SEASONS, n=5, game_type=GAME_TYPE)
if vs.empty:
    st.info(f"No head-to-head games vs {opponent} found in recent seasons.")
else:
    if not is_goalie:
        vs["PTS"] = pd.to_numeric(vs.get("goals",0), errors="coerce").fillna(0) + pd.to_numeric(vs.get("assists",0), errors="coerce").fillna(0)
        meta = ["gameDate","teamAbbrev","opponentAbbrev","homeRoad"]
        stats = ["PTS","goals","assists","shots"]
        cols  = [c for c in meta + stats if c in vs.columns]
        vs = vs.sort_values("gameDate", ascending=False)
        st.dataframe(vs[cols].style.format({c:"{:.0f}" for c in stats if c in cols}), use_container_width=True, height=auto_height(vs[cols]))
        avg_vs = vs[stats].mean(numeric_only=True).to_frame().T.round(2); avg_vs.index=["Average (vs Opp)"]
        st.table(avg_vs)
    else:
        meta = ["gameDate","teamAbbrev","opponentAbbrev","homeRoad"]
        stats = ["shotsAgainst","saves"]
        cols  = [c for c in meta + stats if c in vs.columns]
        vs = vs.sort_values("gameDate", ascending=False)
        st.dataframe(vs[cols].style.format({c:"{:.0f}" for c in stats if c in cols}), use_container_width=True, height=auto_height(vs[cols]))
        avg_vs = vs[stats].mean(numeric_only=True).to_frame().T.round(2); avg_vs.index=["Average (vs Opp)"]
        st.table(avg_vs)
