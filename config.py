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
NHL_GOALIE_BIOS = "https://api.nhle.com/stats/rest/en/goalie/bios"
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
# Marcel-style recency weights, most-recent season first (season t-1 .. t-5). A FIVE-
# season window with a steep decay beats the old flat 3-season [5,4,3]: extending the
# window improved every stat monotonically in backtest (points MAE 9.60->9.51->9.49),
# and a front-loaded profile beat flatter ones (best of 10 profiles swept). The oldest
# seasons carry a small but real signal, chiefly for players with injury-shortened
# recent years. SKATER_HISTORY_SEASONS derives the window length from this list.
RECENCY_WEIGHTS = [9.0, 4.0, 1.8, 0.8, 0.4]
SKATER_HISTORY_SEASONS = len(RECENCY_WEIGHTS)

# Regression-to-mean: number of "league-average" minutes added as a prior. Larger =>
# thin-history players pulled harder toward the (usage-tier) mean. 900->450->300 over
# successive backtests: 900 over-regressed established players (compressed the rate
# scale, under-projecting stars); on the 5-season window k=300 further eases elite
# under-projection (70+ bias -2.6->-2.0) with no cost to other bands.
SKATER_REGRESS_TOI_MIN = 300.0    # ~ regress rates as if adding this many league-avg minutes
GOALIE_REGRESS_SHOTS = 1100.0     # regress SV% as if adding this many league-avg shots faced

# Strength of the TOI/GAME regression toward the usage-tier prior, in league-avg minutes.
# Set to 0 after directly measuring projected-vs-actual TOI/GP error: a player's OWN
# ice-time history (+ the TOI age curve) beats dragging them toward a tier average in
# every slice, including thin histories (TOI/GP MAE 1.89@600 -> 1.53@0). A tiny floor is
# kept only to avoid divide-by-zero for a zero-TOI sample.
SKATER_TOI_PRIOR_MIN = 0.0

# Games-played model: blend recent GP with an age/durability prior. Five-season,
# front-loaded window beat the old 3-season [3,2,1] in backtest (points MAE 9.57->9.45).
GP_RECENCY_WEIGHTS = [4.0, 2.0, 1.0, 0.5, 0.3]
# GP-reliability (interval width) is measured over only the most RECENT seasons: a
# player's availability is a recent signal, and a floor/spread over the whole 5-year
# window would wrongly punish a now-healthy player for an injury years ago.
GP_RELIABILITY_SEASONS = 3

# Goalie recency blend (SV%, workload, win rate). Kept SEPARATE from the skater GP
# weights so skater-side tuning can't silently reshape goalie projections. Goalies are
# lower-signal year-to-year, so a 3-season, less front-loaded profile is appropriate;
# the GP model inside project_goalies uses its own [0.65,0.22,0.13] blend (last season
# dominates workload). Backtest-confirmed on held-out goalies.
GOALIE_RECENCY_WEIGHTS = [3.0, 2.0, 1.0]
# How much of the measured goalie ageing curve to apply (see age_curves.
# build_goalie_age_curves for why it is damped rather than believed in full). Swept in
# backtest_goalies.py; 0.0 reproduces the old age-blind behaviour.
GOALIE_AGE_STRENGTH = 0.5
# Workload blend for goalie STARTS, most recent season first. Last season dominates
# (corr .56 with next season's workload, vs .53 for a flat recency blend) because a
# depth chart is a recent fact. Only the SHAPE matters now: the level is set by the
# 84-start team budget, so this decides how a team's starts split, not how many exist.
GOALIE_WORKLOAD_WEIGHTS = [0.65, 0.22, 0.13]
# The busiest goalie in the league started 63-67 games of 82 in each of the last five
# seasons, so 68 of 84 is a ceiling nobody has approached, not a working constraint.
MAX_GOALIE_STARTS = 68
# How a team's goaltending divides by DEPTH-CHART RANK, and the rank that matters is the
# rank the MODEL can see -- goalies ordered by what their own histories claim, not by who
# turned out to play the most. Measured over 269 clean team-seasons (2012-2025) as the
# share of team goalie minutes taken by the 1st, 2nd, ... claimant:
#
#   by claim rank (usable):    .561  .318  .087  .025  .007
#   by actual rank (hindsight): .601 .324  .064  .009  .002
#
# The difference between those two lines is the model's own error, and using the second
# one as a prior -- which is what this constant held first -- built that error in as if it
# were knowledge: it promised the projected starter 60% of a season when a projected
# starter historically gets 56%, and it promised the third goalie almost nothing when in
# fact third claimants take 8.7% because depth charts do not survive contact with a
# season. Switching to the claim-rank line cut the allocator's error (mse .0245 -> .0236)
# and its top-goalie bias (+.052 -> +.035 of a season) at the same time.
#
# The cumulative sums are also the coverage of a roster that lists m goalies -- .561,
# .879, .966, .991 -- because the goalies not yet named end up BELOW the named ones on
# the chart. A team uses 3.08 distinct goalies in a season and the creases the model
# projects hold 2.94 (see ROSTER_GOALIE_CAP below), so a listed crease is about 96% of the
# goaltending, not all of it; an in-season page listing a tandem is 88%.
GOALIE_RANK_SHARES = [0.561, 0.318, 0.087, 0.025, 0.007]
# Within a team, the claims are turned into shares as claim**exp, i.e. deliberately
# flatter than proportional: summed claims over-count (each goalie's recent starts were
# earned in a role that was partly someone else's injury), and the over-count is worst for
# the depth goalie who filled in. Swept jointly with the quality tilt below and confirmed
# out of sample.
GOALIE_CLAIM_SHARE_EXP = 0.60
# A coach gives the net to whoever stops the puck, and the workload history alone does not
# know that. Measured as a residual: after allocating a team's starts from workload claims
# exactly as this model does, what is left over correlates -0.35 with the goalie's prior
# goals-allowed-per-shot, and WITHIN a team a goalie 5% better than his partner takes 5.4
# more starts of 82. So a goalie's share is shifted by this coefficient times his log
# goals-per-shot relative to his own team's mean -- team-centred, so the shifts cancel and
# the team's 84 starts are still exactly accounted for.
#
# This was the largest single improvement to the start allocation: error on a team's share
# split fell 7% and the mean error on the busiest goalie's start count fell from 11.4 to
# 10.5 games, while the goalie named as the busiest was right 75% of the time instead of
# 73%. Chosen on 2012-20 and confirmed on 2021-25 (mse .0221 vs .0244 without it).
GOALIE_QUALITY_START_TILT = 1.0
# How much the rank prior counts is per goalie and set by EVIDENCE, not taste:
# weight = GOALIE_RANK_PRIOR_STARTS / (that + his starts over the recency window). A
# goalie with three 60-start seasons behind him barely moves (0.12), one with 20 career
# starts moves halfway, and a debutant IS the prior -- which is the honest position, since
# there is nothing else to say about him. This beat every fixed weight tried.
GOALIE_RANK_PRIOR_STARTS = 20.0
# Global dial on that: 1.0 = the evidence-weighted blend above, 0.0 = split a team's starts
# purely by what its goalies' own histories claim (and lose the depth-chart shape).
GOALIE_RANK_PRIOR_WEIGHT = 1.0
# The p10/p90 band on STARTS, measured directly rather than assumed. Knots are (projected
# starts, 10th percentile of what actually happened, 90th percentile), from 828 historical
# goalie-team-seasons on an 82-game season; the model interpolates between them and scales
# them to the season length. Coverage 82% against an 80% target, 76-88% in every workload
# band, with 8% of outcomes below p10 and 10% above p90.
#
# There is no parametric band that fits this, which is why it is a table. A depth goalie's
# season is BIMODAL -- he starts nothing, or the man ahead of him gets hurt and he starts
# thirty -- so his honest p10 is zero and his p90 is a third of a season, and no lognormal
# can say both at once (the count-style band this borrowed from wins covered 62% of
# outcomes, and widening it to reach 80% overall made a starter's band absurd while still
# covering barely half of the backups).
GOALIE_START_BAND = {
    "proj": [1.5, 4.5, 8.0, 12.5, 17.5, 23.0, 29.0, 35.0, 41.0, 47.0, 55.0],
    "p10":  [0.0, 0.0, 0.0, 1.0, 2.0, 2.5, 8.0, 16.0, 27.0, 33.0, 50.0],
    "p90":  [19.0, 22.0, 24.0, 30.0, 33.0, 34.0, 49.0, 55.0, 60.0, 63.0, 66.0],
    "measured_season": 82,
}

# Within a team, the better goalie wins a larger share of his starts than the worse one.
# The exponent is on the ratio of league to own goals-allowed-per-shot; team quality
# itself is NOT in the claim, it is in the budget, which is what stops the old model's
# double-count (a goalie's own win rate already contained his old team, and it was then
# multiplied by his new team's quality as well).
GOALIE_WIN_QUALITY_EXP = 1.0
# NHL moved to an 84-game regular season starting 2026-27 (new CBA). Historical
# durability priors are still expressed as a fraction of the season length.
SEASON_GAMES = 84
MAX_GP = SEASON_GAMES

# Optional user override of projected games played. If this CSV exists, any playerId
# (or exact name) listed sets that player's proj_gp directly (e.g. injury news, a known
# return date). Columns: playerId,name,games_played  (playerId OR name may be blank).
# Games played is the single largest source of season-total error and is near-random
# to predict, so a human who KNOWS a player's status can sharpen the projection here.
GP_OVERRIDE_FILE = ROOT / "gp_overrides.csv"

# Minimum icetime (seconds) in a season for that season to count as a real sample.
MIN_ICETIME_SEC = 60 * 20  # 20 minutes

# --- Team accounting budgets -------------------------------------------------
# A season is a closed system and the old model ignored that: every player was
# projected in isolation, so nothing stopped the sum of a team's projections from
# exceeding what a team HAS to give. Measured on the 2026-27 output before this was
# added: teams were projected 354 skater-minutes per game against a budget of 297
# (+19%, and unevenly -- 278 for one team, 415 for another), 3.70 goals per team-game
# against 3.08 actual, 35.1 shots against 27.8, and goalies 114 appearances per team
# against a mandatory ~88. That error is invisible to the reader and impossible to
# override away one player at a time, because it is a property of the team, not the
# player. So: project a CLAIM per player, then settle the claims against the team's
# budget (budgets.py / allocate.py).
ENFORCE_BUDGETS = True

# How the overshoot is taken back. take_i ~ claim_i**tilt, so tilt=1.0 is a flat
# proportional rescale (everyone loses the same PERCENTAGE) and tilt<1.0 protects the
# high-usage players: at 0.6 a first-line centre gives back a smaller share of his
# claim than a 13th forward does. That is the right shape -- the surplus on a deep
# roster is bottom-of-the-lineup ice time that will be scratched, not top-line ice
# time -- and it is the same mechanism as the NFL tool's pool tilt.
BUDGET_TILT = 0.6

# League rate LEVELS (goals, shots, hits ... per team-game) are trending, not flat:
# shots on goal have fallen 31.6 -> 27.8 per team-game since 2021 and league save
# percentage from .9042 to .8955. A flat multi-season mean therefore projects a level
# the league left behind, so the level is a front-loaded blend of the last 3 seasons.
LEAGUE_RATE_WEIGHTS = [0.55, 0.28, 0.17]

# How much of a team's measured rating to believe for next season. The rest is shrunk
# to league average, because a 2-season rating is part talent and part luck. These are
# MEASURED, not chosen: budgets.rating_persistence() regresses each team's actual rating
# in a season on the rating a projection would have had going in, over 2008-2025
# (n=367 team-seasons per stat), and the slope of that regression is the fraction of a
# deviation that survives. Re-run it after a refresh to update these.
TEAM_RATING_PERSISTENCE = {
    "goals": 0.696, "ixg": 0.724, "shots": 0.682, "blocks": 0.651,
    "hits": 0.673, "pim": 0.678, "faceoffs_won": 0.644,
}
TEAM_RATING_PERSISTENCE_DEFAULT = 0.68

# Global dial on the measured persistence above: 1.0 = believe it as measured, 0.0 =
# every team gets the league-average budget. Exposed as a scenario knob so the effect
# of team strength on a projection can be turned off and looked at.
TEAM_RATING_SHRINK = 1.0

# How much of a team's offensive budget comes from what THIS season's projected roster
# claims, versus the team's own recent rating. The rating knows about last year's team;
# the claims know about the players actually on the roster now, so a team that traded
# for a scorer needs the claims to move its budget. Held at a half-and-half blend: the
# league total is conserved either way, so this only shifts budget BETWEEN teams, and
# any team total can be edited directly in the app, which is the honest answer for a
# roster change too recent for either source to know about.
TEAM_CLAIM_WEIGHT = 0.5

# --- Rookies / no-history players ---------------------------------------------
# A rostered player with no NHL history used to be dropped entirely (17 skaters and 5
# goalies on the 2026-27 rosters), which silently handed their ice time and their
# points to the veterans on the same team. They are now projected from the prior
# alone: bottom-usage-tier rates for their position, and this games-played prior.
ROOKIE_GP = 45.0
ROOKIE_GP_RELIABILITY = 0.15
ROOKIE_TOI_TIER = 0          # lowest TOI/game quartile within position
# A goalie with no NHL history claims a third-stringer's share of his team's 84 starts.
# It only has to be the right ORDER of magnitude: the claim is settled against the team's
# starts, so if he is behind two established goalies he stays behind them, and if he is
# in fact the plan the app can say so directly.
ROOKIE_GOALIE_START_SHARE = 0.15
# What a goalie's FIRST NHL season is worth, measured directly rather than extrapolated
# off the age curve: goals allowed per shot in a debut season runs 1.021x the league (89
# debuts, 43,178 shots, first year of the data excluded as censored), and from the second
# season on the effect is gone (1.001, 1.002, 1.006). Small, real, and the honest prior for
# a goalie with no history -- the age curve, stretched seven years back from its anchor,
# said such a goalie would be 10% BETTER than league.
GOALIE_DEBUT_PENALTY = 1.021

# --- Schedule / context adjustments (per-game decomposition) ---
# How many recent seasons to average for team offense/defense strength ratings.
TEAM_STRENGTH_SEASONS = 2
# Skater opponent effect is deliberately gentle: SOS moves skater point totals only
# ~1-3 pts (see research). This caps how far a single game's opponent multiplier can
# stray from 1.0 before normalization. Goalies use the full (uncapped) shot-context effect.
SKATER_OPP_STRENGTH = 0.5   # 0 = ignore opponent, 1 = full team-defense delta
HOME_ICE_BOOST = 0.015      # ~1.5% offensive bump at home (mean-0 across balanced schedule)
B2B_PENALTY = 0.04          # 4% production penalty on the 2nd night of a back-to-back

# --- Scenarios / overrides ----------------------------------------------------
# The workbook could only be read; the Streamlit app can be argued with. A scenario
# is a JSON file of disagreements with the model -- per player, per team, or a league
# knob -- and nothing else, so it stays small, readable and diffable, and a player
# with no entry is by definition the model's own opinion. `gp_overrides.csv` is still
# honoured (it was the only editable input the workbook had) and is folded in as
# games-played edits.
SCENARIOS = ROOT / "scenarios"
SCENARIOS.mkdir(parents=True, exist_ok=True)
LIVE_SCENARIO = SCENARIOS / "working.json"

# League knobs a scenario is allowed to set, with their defaults. Anything not in
# here is a code constant, not a setting, and the app will refuse to patch it.
def league_defaults() -> dict:
    return {
        "enforce_budgets": ENFORCE_BUDGETS,
        "budget_tilt": BUDGET_TILT,
        "team_rating_shrink": TEAM_RATING_SHRINK,
        "goals_xg_weight": GOALS_XG_WEIGHT,
        "skater_regress_toi_min": SKATER_REGRESS_TOI_MIN,
        "goalie_regress_shots": GOALIE_REGRESS_SHOTS,
        "season_games": SEASON_GAMES,
    }

# --- Counting stats -----------------------------------------------------------
# Projected on a per-60 rate basis and settled against a team budget. The first five
# were all the model had; blocks / hits / PIM / faceoffs carry their own prop markets
# and describe how a player is used, and they were simply missing. All of them are
# already columns on the MoneyPuck all-situations file, so they cost no extra download.
SKATER_STATS = ["goals", "primaryAssists", "secondaryAssists", "shots", "pp_points",
                "sh_points", "blocks", "hits", "pim", "faceoffs_won", "ixg"]

# Physical ceilings on ice time per game, by position group. Settling can scale a claim
# UP as well as down (a team has 297 skater-minutes a night to give whether or not its
# published roster claims them all), so without these a thin roster would be handed
# 30-minute nights. Set just above the observed maxima: the busiest forwards live around
# 22-23 minutes and the busiest defencemen around 26.
MAX_TOI_PER_GP = {"F": 24.0, "D": 27.0}

# How many players a team's roster page is allowed to put on the ice, by position group.
#
# The NHL publishes one roster endpoint and it does not say who is on the active roster. In
# February it returns about 22 skaters and 2.6 goalies, which is the shape every coverage
# constant in this file was measured against. In September it returns the training camp --
# measured 2026-09-06: 39.3 players a team, 22.5 F / 12.5 D / 4.3 G, one team listing 50 --
# because camp invitees and the whole AHL affiliate are on it.
#
# Those extras are not free. Every listed player claims games and minutes from the same
# 18-skaters x 84-games pool, so a camp roster makes every team look massively
# oversubscribed and settlement takes the overshoot off everybody: claimed skater-games per
# team ran 1856 against a 1512 budget, and Connor McDavid came out at 60 games instead of
# 76. The regulars were paying for the invitees.
#
# So the listed roster is cut to a plausible active roster before any budget is built, by
# the model's own claim ranking (projected minutes, which is what a camp battle is decided
# on anyway). The players cut are not deleted -- they keep a projection, they are marked
# `status="camp"`, and naming a team for one in the app puts him straight back on it. The
# caps are the observed in-season shape, 14 forwards and 8 defencemen dressed-or-scratched,
# so a mid-season run and a September run are settled on the same terms.
ROSTER_SKATER_CAP = {"F": 14, "D": 8}
# A team carries two goalies and sometimes a third; the rank-share table above only reaches
# a fifth goalie because seasons happen, not because anyone lists five.
ROSTER_GOALIE_CAP = 3
# A first power-play unit plays about 3.5-4.5 minutes a night; nobody plays six.
MAX_PP_TOI_PER_GP = 6.0
MAX_SH_TOI_PER_GP = 5.0

# Individual expected goals is carried as a rate for two reasons: it is the honest
# base for a goal projection (a shooting percentage regresses far harder than the
# shot quality behind it), and it is worth showing next to goals so a reader can see
# which projections are carried by finishing. Weight swept in the backtest.
GOALS_XG_WEIGHT = 0.35

# There is deliberately no scoring conversion here. This system projects hockey -- goals,
# ice time, starts, save percentage -- and a single points formula sitting in the middle of
# it only ever answered one league's question while implying it answered everyone's. Every
# stat column is exported, so any scoring can be applied downstream.
