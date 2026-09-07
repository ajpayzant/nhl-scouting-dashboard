"""NHL season projections — the app.

Run it with `streamlit run app/streamlit_app.py` from the repo root, or double-click
`run_app.bat`. Streamlit puts this file's directory on the import path, which is why the
views import `core` directly and `core` puts the repo root on the path for the engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit puts the script's directory on the path itself, but say so explicitly: it is
# the difference between the app running and a bare ImportError.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

st.set_page_config(page_title="NHL Projections", page_icon="🏒", layout="wide",
                   initial_sidebar_state="expanded")

# A little restraint on the default look: tighter top padding so a table starts near the
# top of the window, and metrics that read as figures rather than headlines.
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
  [data-testid="stMetricValue"] {font-size: 1.5rem;}
  [data-testid="stMetricLabel"] {font-size: 0.8rem; opacity: 0.75;}
  h3 {margin-bottom: 0.2rem;}
</style>
""", unsafe_allow_html=True)

from views import (edits, export, games, goalies, home, model,  # noqa: E402
                   performance, player, skaters, teams)

# Every view exposes a callable called `page`, so the URL path has to be given
# explicitly -- Streamlit would otherwise infer all ten as "page".
PAGES = [
    st.Page(home.page, title="Overview", icon=":material/home:", url_path="overview",
            default=True),
    st.Page(player.page, title="Player dashboard", icon=":material/person:",
            url_path="player"),
    st.Page(skaters.page, title="Skaters", icon=":material/ice_skating:",
            url_path="skaters"),
    st.Page(goalies.page, title="Goalies", icon=":material/shield:", url_path="goalies"),
    st.Page(teams.page, title="Teams", icon=":material/groups:", url_path="teams"),
    st.Page(games.page, title="Game by game", icon=":material/calendar_month:",
            url_path="games"),
    st.Page(edits.page, title="Scenario", icon=":material/edit_note:",
            url_path="scenario"),
    st.Page(model.page, title="Model check", icon=":material/fact_check:",
            url_path="model"),
    st.Page(performance.page, title="Model performance", icon=":material/timeline:",
            url_path="performance"),
    st.Page(export.page, title="Export", icon=":material/download:", url_path="export"),
]

st.navigation(PAGES).run()
