"""Central configuration for the NHL season-long projection system."""
from pathlib import Path

# --- Directories ---
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"
for _d in (DATA_RAW, DATA_PROCESSED, OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)

# --- Seasons ---
# MoneyPuck files are named by the starting year of the season (e.g. 2024 = 2024-25).
# 2025 = the just-completed 2025-26 season. The upcoming season to project is 2026-27.
FIRST_SEASON = 2008           # earliest MoneyPuck season summary
LAST_COMPLETED_SEASON = 2025  # 2025-26, most recent finished season (history through here)
TARGET_SEASON = 2026          # 2026-27, the season we project

HISTORY_SEASONS = list(range(FIRST_SEASON, LAST_COMPLETED_SEASON + 1))

# --- Data sources ---
MONEYPUCK_SKATERS = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/skaters.csv"
MONEYPUCK_GOALIES = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/goalies.csv"
# NHL stats REST API (aggregate leaderboards). seasonId is e.g. 20242025, gameTypeId 2 = regular season.
NHL_SKATER_BIOS = "https://api.nhle.com/stats/rest/en/skater/bios"
NHL_GOALIE_SUMMARY = "https://api.nhle.com/stats/rest/en/goalie/summary"
NHL_SKATER_SUMMARY = "https://api.nhle.com/stats/rest/en/skater/summary"

# api-web endpoints for current-season structural data (rosters, schedule).
NHL_ROSTER = "https://api-web.nhle.com/v1/roster/{team}/{season_id}"
NHL_CLUB_SCHEDULE = "https://api-web.nhle.com/v1/club-schedule-season/{team}/{season_id}"
NHL_DAY_SCHEDULE = "https://api-web.nhle.com/v1/schedule/{date}"

# MoneyPuck line combinations (5on5), for line-chemistry context.
MONEYPUCK_LINES = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/lines.csv"
# MoneyPuck team summaries (offense/defense), for strength-of-schedule context.
MONEYPUCK_TEAMS = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/teams.csv"

# A date that falls inside the target regular season, used to enumerate the 32 active teams.
TARGET_SEASON_SAMPLE_DATE = "2026-10-15"

USER_AGENT = "Mozilla/5.0 (NHL-Projection research; contact via project owner)"
REQUEST_TIMEOUT = 30
REQUEST_PAUSE = 0.4  # polite pause between NHL API calls

def season_id(year: int) -> int:
    """MoneyPuck start-year -> NHL API seasonId (e.g. 2024 -> 20242025)."""
    return int(f"{year}{year + 1}")

# --- Projection parameters ---
# Marcel-style recency weights, most-recent season first (season t-1, t-2, t-3).
RECENCY_WEIGHTS = [5.0, 4.0, 3.0]

# Regression-to-mean: number of "league-average" games/60-min chunks added as a prior.
# Larger => thin-history players pulled harder toward positional mean.
SKATER_REGRESS_TOI_MIN = 900.0    # ~ regress rates as if adding this many league-avg minutes
GOALIE_REGRESS_SHOTS = 750.0      # regress SV% as if adding this many league-avg shots faced

# Games-played model: blend recent GP with an age/durability prior.
GP_RECENCY_WEIGHTS = [3.0, 2.0, 1.0]
# NHL moved to an 84-game regular season starting 2026-27 (new CBA). Historical
# durability priors are still expressed as a fraction of the season length.
SEASON_GAMES = 84
MAX_GP = SEASON_GAMES

# Minimum icetime (seconds) in a season for that season to count as a real sample.
MIN_ICETIME_SEC = 60 * 20  # 20 minutes

# --- Schedule / context adjustments (per-game decomposition) ---
# How many recent seasons to average for team offense/defense strength ratings.
TEAM_STRENGTH_SEASONS = 2
# Skater opponent effect is deliberately gentle: SOS moves skater point totals only
# ~1-3 pts (see research). This caps how far a single game's opponent multiplier can
# stray from 1.0 before normalization. Goalies use the full (uncapped) shot-context effect.
SKATER_OPP_STRENGTH = 0.5   # 0 = ignore opponent, 1 = full team-defense delta
HOME_ICE_BOOST = 0.015      # ~1.5% offensive bump at home (mean-0 across balanced schedule)
B2B_PENALTY = 0.04          # 4% production penalty on the 2nd night of a back-to-back
