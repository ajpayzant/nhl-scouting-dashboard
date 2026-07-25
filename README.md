# NHL Season-Long Projections

Projects season-long stat lines for NHL skaters and goalies for the upcoming season
(currently **2026-27**), using multi-season history, empirical age curves, and a
games-played model.

## Quick start

```bash
pip install -r requirements.txt
python run.py --refresh     # first run: download + cache all source data
python run.py               # subsequent runs: use cached data
python run.py --backtest    # also print the accuracy backtest
```

Outputs land in `output/`:
- `skater_projections_2026.csv` — G / A / points / shots / PP points + projected GP & TOI/GP
- `goalie_projections_2026.csv` — W / SV% / GAA / saves / shutouts + projected GP

## Data sources (all free, no API key)

| Source | Role | Notes |
|---|---|---|
| **MoneyPuck** season summaries | primary skater/goalie stats, xG, TOI, situation splits | `playerId` IS the NHL id → clean join |
| **NHL stats REST API** (`api.nhle.com/stats/rest`) | birthdates (age curves), goalie W/SV%/GAA/SO, clean player names | bulk, paged |

Source data is cached to `data/raw/*.parquet`; re-runs are offline unless `--refresh`.

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

- `config.py` — seasons, source URLs, projection parameters (weights, regression strength)
- `data_layer.py` — download + cache MoneyPuck / NHL API data
- `age_curves.py` — delta-method empirical age curves
- `project_skaters.py` / `project_goalies.py` — the projection models
- `backtest.py` — accuracy validation
- `run.py` — one-shot runner

## Not yet modelled (future work)

Line/PP-unit chemistry, team-context changes (trades, new linemates), schedule
strength for goalie wins, and rookies with no NHL history (currently only projectable
once they have ≥1 season).
