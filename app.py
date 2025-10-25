# app.py — NHL Player Scouting Dashboard
# --------------------------------------
# Run:  streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import math
import plotly.express as px

# ----------------------------- Utilities -----------------------------
st.set_page_config(page_title="NHL Player Scouting Dashboard", layout="wide")

pd.set_option("display.max_columns", 0)

TEAM_ABBRS = [
    "ANA","ARI","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET","EDM","FLA","LAK",
    "MIN","MTL","NSH","NJD","NYI","NYR","OTT","PHI","PIT","SEA","SJS","STL","TBL","TOR",
    "VAN","VGK","WPG","WSH"
]

def get_json(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def nhl_season_label_for_date(d=None) -> int:
    d = d or dt.date.today()
    start = d.year if d.month >= 7 else d.year - 1
    return int(f"{start}{start+1}")

def season_options(num=6):
    """Return [current, prev1, prev2, ...] as ints like 20252026."""
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

def standings_payload(season: int):
    """Try standings for season; fallback to previous season if 404; else None."""
    try:
        return get_json(f"https://api-web.nhle.com/v1/standings/{season}")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            start = int(str(season)[:4])
            prev = int(f"{start-1}{start}")
            try:
                st.info(f"Standings {season} not available yet. Using previous season {prev}.")
                return get_json(f"https://api-web.nhle.com/v1/standings/{prev}")
            except requests.exceptions.HTTPError:
                return None
        return None

def teams_from_standings(payload) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()
    blocks = payload if isinstance(payload, list) else [payload]
    rows = []
    for blk in blocks:
        for r in blk.get("standings", []):
            rows.append({
                "teamAbbrev": r.get("teamAbbrev"),
                "teamName": (r.get("teamCommonName") or {}).get("default") if isinstance(r.get("teamCommonName"), dict) else r.get("teamName"),
                "gamesPlayed": r.get("gamesPlayed"),
                "goalsAgainst": r.get("goalsAgainst"),
            })
    df = pd.DataFrame(rows).dropna(subset=["teamAbbrev"]).drop_duplicates("teamAbbrev").reset_index(drop=True)
    if df.empty:
        return df
    df["gamesPlayed"] = pd.to_numeric(df["gamesPlayed"], errors="coerce")
    df["goalsAgainst"] = pd.to_numeric(df["goalsAgainst"], errors="coerce")
    df["GA_per_GP"] = df.apply(lambda r: safe_div(r["goalsAgainst"], r["gamesPlayed"]), axis=1)
    return df

def club_stats_goalies(team_abbr: str, season: int, game_type: int=2) -> pd.DataFrame:
    data = get_json(f"https://api-web.nhle.com/v1/club-stats/{team_abbr}/{season}/{game_type}")
    return pd.json_normalize(data.get("goalies", []))

def teams_from_club_stats(season: int, game_type: int=2) -> pd.DataFrame:
    """League sweep (when standings down): compute GA/GP and SA/GP from goalies."""
    rows = []
    for abbr in TEAM_ABBRS:
        gdf = club_stats_goalies(abbr, season, game_type)
        if gdf.empty:
            rows.append({"teamAbbrev": abbr, "GA_per_GP": np.nan, "SA_per_GP": np.nan})
            continue
        sa = pd.to_numeric(gdf.get("shotsAgainst", 0), errors="coerce").fillna(0).sum()
        sv = pd.to_numeric(gdf.get("saves", 0), errors="coerce").fillna(0).sum()
        gp = pd.to_numeric(gdf.get("gamesPlayed", 0), errors="coerce").fillna(0).sum()
        ga = sa - sv
        rows.append({
            "teamAbbrev": abbr,
            "GA_per_GP": safe_div(ga, gp),
            "SA_per_GP": safe_div(sa, gp),
        })
    return pd.DataFrame(rows)

# ----------------------------- Opponent adjusters -----------------------------
def build_league_context(season: int, game_type: int=2):
    stand = standings_payload(season)
    teams_df = teams_from_standings(stand)
    sa_map = {}
    if teams_df.empty:
        st.info("Standings unavailable. Using club-stats sweep for league baselines.")
        league = teams_from_club_stats(season, game_type)
        teams_df = league[["teamAbbrev","GA_per_GP"]].copy()
        sa_map = {r["teamAbbrev"]: r["SA_per_GP"] for _, r in league.iterrows()}
    team_options = sorted([t for t in teams_df["teamAbbrev"].dropna().astype(str).unique().tolist() if t in TEAM_ABBRS] or TEAM_ABBRS)
    # League baselines
    league_ga_pg = float(pd.to_numeric(teams_df["GA_per_GP"], errors="coerce").mean())
    if sa_map:
        league_sa_pg = float(np.nanmean(list(sa_map.values())))
    else:
        # compute SA/GP per team on-demand (slower once)
        league_sa_pg = float(np.nanmean([
            safe_div(pd.to_numeric(club_stats_goalies(t, season, game_type).get("shotsAgainst", 0), errors="coerce").sum(),
                     pd.to_numeric(club_stats_goalies(t, season, game_type).get("gamesPlayed", 0), errors="coerce").sum())
            for t in team_options
        ]))
    return teams_df, team_options, league_ga_pg, league_sa_pg, sa_map

def opp_metrics(opponent: str, season: int, game_type: int, teams_df: pd.DataFrame, sa_map: dict, league_sa_pg: float):
    row = teams_df.loc[teams_df["teamAbbrev"] == opponent]
    opp_ga_pg = float(row["GA_per_GP"].iloc[0]) if not row.empty else np.nan
    if opponent in sa_map:
        opp_sa_pg = float(sa_map[opponent])
    else:
        gdf = club_stats_goalies(opponent, season, game_type)
        sa = pd.to_numeric(gdf.get("shotsAgainst", 0), errors="coerce").sum()
        gp = pd.to_numeric(gdf.get("gamesPlayed", 0), errors="coerce").sum()
        opp_sa_pg = safe_div(sa, gp)
    return opp_ga_pg, opp_sa_pg

def scoring_adjuster(opp_ga_pg, league_ga_pg):
    if not np.isfinite(opp_ga_pg) or not np.isfinite(league_ga_pg):
        return 1.0
    return float(league_ga_pg / max(opp_ga_pg, 1e-9))

def shots_adjuster(opp_sa_pg, league_sa_pg):
    if not np.isfinite(opp_sa_pg) or not np.isfinite(league_sa_pg):
        return 1.0
    return float(opp_sa_pg / max(league_sa_pg, 1e-9))

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

    # career
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

    # career SA/GP
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
    SEASON = st.selectbox("Season", SEASONS, index=0, format_func=lambda x: f"{str(x)[:4]}-{str(x)[4:]}")
    GAME_TYPE = st.selectbox("Game Type", {2:"Regular Season", 3:"Playoffs"}, index=0, format_func=lambda x: {2:"Regular",3:"Playoffs"}[x])
    query = st.text_input("Player search", "McDavid")
    recent_window = st.select_slider("Recent window (games)", options=[5,10,15,20], value=5)
    st.markdown("---")
    st.caption("Blend Weights")
    w_recent = st.slider("Recent", 0.0, 1.0, 0.55, 0.05)
    w_season = st.slider("Season", 0.0, 1.0, 0.30, 0.05)
    w_career = st.slider("Career", 0.0, 1.0, 0.15, 0.05)

# Search players
players_df = search_players(query)
if players_df.empty:
    st.warning("No players found. Try another name.")
    st.stop()

# List players (active first)
players_df = players_df.sort_values(["active"], ascending=False).reset_index(drop=True)
player_choices = players_df["name"].tolist()
player_name = st.selectbox("Choose player", player_choices, index=0)
sel = players_df.loc[players_df["name"] == player_name].iloc[0]
player_id = int(sel["playerId"])
position_code = str(sel.get("positionCode", ""))

# Data fetch
landing = player_landing(player_id)
bio = landing.get("playerBio", {}) or {}
season_totals = pd.json_normalize(landing.get("seasonTotals", []))
gl = player_game_log(player_id, SEASON, GAME_TYPE)
# If no logs this season, fallback to previous seasons until found
if gl.empty:
    for s in SEASONS[1:]:
        gl = player_game_log(player_id, s, GAME_TYPE)
        if not gl.empty:
            st.info(f"No logs for {str(SEASON)[:4]}-{str(SEASON)[4:]}. Showing {str(s)[:4]}-{str(s)[4:]} instead.")
            SEASON = s
            break

# Resolve player's team
player_team = resolve_player_team(bio, gl, players_df, landing)

# League context & opponent
teams_df, team_options, league_ga_pg, league_sa_pg, sa_map = build_league_context(SEASON, GAME_TYPE)
opp_default = gl["opponentAbbrev"].dropna().astype(str).iloc[0] if ("opponentAbbrev" in gl.columns and not gl.empty and pd.notna(gl.loc[0,"opponentAbbrev"])) else (team_options[0] if team_options else "EDM")
opponent = st.selectbox("Opponent", team_options, index=max(0, team_options.index(opp_default) if opp_default in team_options else 0))
opp_ga_pg, opp_sa_pg = opp_metrics(opponent, SEASON, GAME_TYPE, teams_df, sa_map, league_sa_pg)
shots_adj = shots_adjuster(opp_sa_pg, league_sa_pg)
score_adj = scoring_adjuster(opp_ga_pg, league_ga_pg)

# Header info block
col1, col2 = st.columns([2,1])
with col1:
    full_name = bio.get("fullName") or player_name
    age = bio.get("age", "—")
    pos = bio.get("positionCode", position_code)
    seasons_played = len(season_totals["seasonId"].unique()) if not season_totals.empty and "seasonId" in season_totals.columns else "—"
    # current season GP:
    gp_cur = "—"
    if not season_totals.empty and {"seasonId","gamesPlayed"}.issubset(season_totals.columns):
        row = season_totals.loc[season_totals["seasonId"] == SEASON]
        if not row.empty:
            gp_cur = int(pd.to_numeric(row["gamesPlayed"], errors="coerce").fillna(0).iloc[0])
    st.markdown(f"### {full_name}")
    st.markdown(f"**Pos**: {pos} &nbsp;&nbsp; **Age**: {age} &nbsp;&nbsp; **Seasons**: {seasons_played} &nbsp;&nbsp; **{str(SEASON)[:4]}-{str(SEASON)[4:]} GP**: {gp_cur}")
with col2:
    st.markdown(f"**Opponent:** {opponent}")
    st.markdown(f"GA/GP: **{opp_ga_pg:.2f}**  •  SA/GP: **{opp_sa_pg:.2f}**")
    st.caption(f"League baselines — GA/GP: {league_ga_pg:.2f} • SA/GP: {league_sa_pg:.2f}")

st.markdown("---")

# Projections
is_goalie = (pos == "G")
if is_goalie:
    proj = goalie_projection(gl, season_totals, w_recent, w_season, w_career, shots_adj, K=recent_window)
else:
    proj = skater_projection(gl, season_totals, w_recent, w_season, w_career, shots_adj, score_adj, K=recent_window)

st.subheader("Projection Summary")
st.table(proj.set_index("Stat"))  # static table, no scroll

# Recent trends charts
st.subheader("Recent Trends")
if gl.empty:
    st.info("No game logs available to chart.")
else:
    plot_cols = []
    if is_goalie:
        # SA / SV
        if "shotsAgainst" in gl.columns: plot_cols.append(("shotsAgainst","Shots Against"))
        if "saves" in gl.columns:        plot_cols.append(("saves","Saves"))
    else:
        # PTS = G+A, Goals, Assists, Shots
        tmp = gl.copy()
        tmp["PTS"] = pd.to_numeric(tmp.get("goals",0), errors="coerce").fillna(0) + pd.to_numeric(tmp.get("assists",0), errors="coerce").fillna(0)
        gl = tmp
        for c, label in [("PTS","Points"),("goals","Goals"),("assists","Assists"),("shots","Shots on Goal")]:
            if c in gl.columns: plot_cols.append((c,label))

    # limit to recent_window*2 for a nicer view
    plot_df = gl.head(min(len(gl), recent_window*2)).iloc[::-1]  # chronological
    for c, label in plot_cols:
        if c not in plot_df.columns: continue
        fig = px.line(plot_df, x="gameDate", y=c, markers=True, title=label)
        fig.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

# Last 5 games (single-game rows = whole numbers)
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
