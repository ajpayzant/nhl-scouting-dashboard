# NHL Season-Long Projections

Projects season-long stat lines for NHL skaters and goalies for the upcoming season
(currently **2026-27**), using multi-season history, empirical age curves, and a
games-played model.

## Quick start

```bash
pip install -r requirements.txt
python run.py --refresh     # first run: download + cache all source data
python run.py               # subsequent runs: use cached data
python run.py --games       # also write game-by-game projections
python run.py --backtest    # also print the accuracy backtest
```

Outputs land in `output/`:
- `skater_projections_2026.csv` — G / A / points / shots / PP points + projected GP & TOI/GP
- `goalie_projections_2026.csv` — W / SV% / GAA / saves / shutouts + projected GP
- `skater_game_projections_2026.csv` — one row per player per scheduled game, with
  matchup/home-away/rest-adjusted expected stats that sum back to the season total

## Data sources (all free, no API key)

| Source | Role | Notes |
|---|---|---|
| **MoneyPuck** season summaries | primary skater/goalie stats, xG, TOI, situation splits | `playerId` IS the NHL id → clean join |
| **MoneyPuck** teams.csv / lines.csv | team offense/defense (SOS), line combinations | `lineId` = concatenated player ids |
| **NHL stats REST API** (`api.nhle.com/stats/rest`) | birthdates (age curves), goalie W/SV%/GAA/SO, clean player names | bulk, paged |
| **NHL api-web** (`/roster`, `/club-schedule-season`) | **current 2026-27 rosters** (players' new teams) + **actual 84-game schedule** | published; joins on player id |

Source data is cached to `data/raw/*.parquet`; re-runs are offline unless `--refresh`.

### Team context, schedule & new teams

- **Current team** — each player's upcoming-season team comes from the *published roster*,
  so trades and free-agent signings are reflected (season-summary stats are tied to the
  old team). An `on_roster` flag marks confirmed roster spots vs. projected-but-unsigned.
- **Strength of schedule** — a mean-1.0-normalized opponent multiplier from team
  defense ratings. For **skaters this is intentionally tiny** (the data shows the toughest
  vs. easiest 2026-27 schedule differs by <1%, i.e. <1 pt on a star). For **goalies it is
  first-order**: the new team's defense drives shots-against (saves/GAA) and win rate.
- **Note:** the 2026-27 season is **84 games** (new CBA), not 82.

### Why no game-by-game Monte Carlo simulation?

By linearity of expectation, `E[Σ games] = Σ E[games]` — summing simulated games gives
the *same* season-total mean as projecting the total directly (it only adds a
distribution). This matches how Marcel/ZiPS/Steamer work: season totals are projected
directly; Monte Carlo is reserved for team playoff-odds, not player point totals. And
preseason Bayesian prior-updating is circular (no new data until games are played). So
game-by-game lines are produced **top-down** — the accurate season total is the anchor,
distributed across the real schedule with normalized weights, guaranteeing
`Σ game_i = season_total` (verified to <0.02 pts).

## Method

**Skaters** — projected on a per-60 *rate* basis so injury-shortened seasons don't
distort talent, then scaled back up by projected volume:
1. **Rate base** — blend last ≤3 seasons of per-60 rates, weighted by recency
   (`5/4/3`) × that season's TOI.
2. **Regression** — pull toward the positional (F/D) league-average rate, weighted by
   sampled TOI. Thin history regresses hard; established stars barely move.
3. **Age curve** — empirical, built via the **delta method** (paired same-player
   year-over-year changes → chained multiplicative curve, peak-normalised, with the
   survivor-biased tail regularised to be non-increasing after the peak).
4. **Volume** — project TOI/GP and games played (recency-weighted, durability prior),
   then `rate × TOI` → counting totals.

**Goalies** — deliberately conservative (SV% is low-signal year-over-year): recency +
shots-weighted SV% regressed hard to league mean, projected workload (GP, shots-against/GP),
win rate anchored to recent form. Derives saves / GAA / shutouts.

## Validation

`backtest.py` projects each past season using only prior data and scores vs actuals.
The model beats "last season = next season" and "3-year average" baselines on RMSE
every season tested (2022-2025), and cuts per-60 rate error ~13% vs last-season-only.

## Files

- `config.py` — seasons, source URLs, projection parameters (weights, regression, context)
- `data_layer.py` — download + cache MoneyPuck / NHL API data (stats, bios, rosters, schedule, teams, lines)
- `age_curves.py` — delta-method empirical age curves
- `context.py` — team strength ratings + per-game schedule context (opponent, home/away, rest)
- `project_skaters.py` / `project_goalies.py` — the season projection models
- `project_games.py` — top-down game-by-game decomposition (+ optional prop probabilities)
- `backtest.py` — accuracy validation
- `run.py` — one-shot runner (`--refresh`, `--games`, `--backtest`)

## Not yet modelled (future work)

- **Rookies with no NHL history** — currently projectable only with ≥1 prior season
  (~15 of 718 rostered skaters). Would need a draft-pedigree / junior-league prior.
- **Line/PP-unit chemistry** — `lines.csv` is downloaded (`load_moneypuck_lines`,
  `split_line_ids`) but not yet fed into rates; a linemate-quality adjustment (who a
  player is projected to skate with on their new team) is the natural next lever.
- **Explicit role/usage projection** — TOI is projected from the player's own history,
  not from a depth-chart model of their new team (a 3rd-liner traded into a top-6 role
  won't see the usage bump until it shows in the data).
