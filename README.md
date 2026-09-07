# NHL Season-Long Projections

Season stat lines for every NHL skater and goalie for the upcoming season (currently
**2026-27**, which is **84 games** under the new CBA), built from multi-season per-60
rates, empirical age curves and a games-played model, then settled against what each team
can actually produce.

The interface is a Streamlit app. The Excel workbook is still produced, as an export.

## Quick start

```bash
pip install -r requirements.txt
python run.py --refresh     # first run: download + cache all source data
run_app.bat                 # open the app (or: python -m streamlit run app/streamlit_app.py)
```

```bash
python run.py               # write the projection CSVs to output/
python run.py --games       # also write game-by-game projections
python run.py --backtest    # also print the accuracy backtest
python build_workbook.py    # the standalone Google-Sheets workbook, from the CSVs
```

## Deploying it for other people

The app runs on Streamlit Community Cloud with no changes: main file path
`app/streamlit_app.py`, Python **3.13**, and `requirements.txt` is already pinned below the
next majors of pandas and numpy so a reboot cannot silently upgrade under the app.

The cached source data (`data/raw/*.parquet`, ~19 MB) **is committed on purpose**. A fresh
container therefore boots in seconds instead of making several hundred API calls, and
everyone reading the app is reading the same reviewable snapshot. `.github/workflows/refresh-data.yml`
re-downloads it weekly and commits the diff, which Streamlit Cloud picks up as a redeploy;
you can also run it from the Actions tab whenever a trade happens, or refresh locally and
push.

**One person versus many.** The scenario is the app's only mutable state, and deployed it
must not be shared: Streamlit gives each visitor a session but only one container and one
filesystem, so a single `working.json` would mean one person's lock rewriting everybody
else's view. `core.MULTIUSER` (on automatically under Streamlit Cloud, or forced either way
with a `multiuser` secret) moves the working scenario into session memory. Each visitor
starts from the model, edits privately, and publishes to the shared library when they want
someone else to see it.

**The shared library** (Scenario → Save & share) needs somewhere durable, because a
deployed container's disk is wiped on redeploy. Point it at a GitHub gist and it survives:

```toml
# Streamlit Cloud → your app → Settings → Secrets. Never in the repo.
gist_id = "…"        # create an empty secret gist; the id is the last part of its URL
github_token = "…"   # a fine-grained token with read+write on Gists and nothing else
```

Without those two the library falls back to the container's disk and says so on the page —
publishing still works, it just does not survive a restart.

## The app

`app/streamlit_app.py` — nine pages, all reading the same cached projection:

- **Overview** — where the season stands, the leaders, how many edits you have made
- **Player dashboard** — one skater or goalie at a time: every prior season as totals *and* as
  per-60 rates, the model's own rate for the upcoming season sitting in the same table, and a
  form for editing those rates directly. This is the deepest edit in the app — a rate is what
  the model measured about the player, so changing one is an argument about him rather than
  about his totals, and it still settles against his team's budget
- **Skaters** — filterable board (scoring / per 60 / physical / usage / shooting columns);
  click a player for his history, his floors and ceilings, and what his own rates claimed
  before the team budget settled them. Edit one player carefully, or a run of them in a grid
- **Goalies** — starts first, because a goalie season is a depth-chart decision before it is
  a talent question. Shows the whole crease and the measured floor-to-ceiling band on starts
- **Teams** — six tabs: the **budget** and how much of it the published roster covers, the
  **roster**, eighteen seasons of **team history** per game (with a check on whether the budget
  is plausible against the last one, three and five years), a **roster review** of where the ice
  time and the power play go and how old the team is by minute, the **schedule** it faces, and
  the budget edit form. This is where a projection that looks low explains itself
- **Game by game** — the season total spread over the real schedule
- **Scenario** — every edit in one list, each one removable, plus the league-wide settings
- **Model check** — the accounting identities, the measured constants, and how old the data is
- **Export** — the workbook and the CSVs, with your edits in them

Nothing in the app is destructive: edits live in `scenarios/working.json` as a list of only
the disagreements, so clearing one restores the model's own number exactly. Scenarios can be
saved under a name, downloaded, and mailed to someone.

### Overriding a projection

Three kinds of edit, all reversible:

- **Inputs** — games played, ice time, power-play ice time, a per-60 rate. These flow through
  the model: raise a player's ice time and his goals, shots and blocks all move, and his
  teammates give up the minutes he gained.
- **Locks** — a season total stated outright ("he scores 50"). Honoured exactly, and taken
  out of the team's budget *first*, so the rest of the roster resettles around it instead of
  being quietly rescaled away.
- **Structural** — his team, or whether he is on a roster at all. This is how a signing or a
  trade gets in before the published roster catches up.

Team budgets and the league knobs are editable on the same terms. The legacy
`gp_overrides.csv` is still read, so anything already written in it keeps working.

## Method

Every projection is a **claim** settled against a **budget**.

A player's own rates say what he asks for; his team's expected totals say what there is to
give. Overshoot is taken back in proportion to `claim**tilt` with `tilt < 1`, which protects
high-usage players — they are the least likely to be the reason a team is oversubscribed.

**Skaters** — projected on a per-60 rate basis so injury-shortened seasons do not look like
lost talent, then scaled back up by projected volume:

1. **Rate base** — recency-weighted blend of the last ≤3 seasons of per-60 rates, weighted
   by each season's ice time.
2. **Regression** — pulled toward the positional (F/D) league rate, weighted by sampled ice
   time. Thin history regresses hard; established stars barely move.
3. **Age curve** — empirical, built by the delta method (paired same-player year-over-year
   changes, chained multiplicatively, peak-normalised, with the survivor-biased tail
   regularised to be non-increasing after the peak).
4. **Volume** — project TOI/GP and games played, then settle both against the team's ice
   time and dressed skater-games.

Counting stats: goals, primary/secondary assists, shots, PP points, SH points, blocks, hits,
PIM, faceoffs won, individual expected goals. Every one gets a p10/p90 band.

There is deliberately no points-per-stat conversion anywhere in the system. It projects
hockey, and a single scoring formula in the middle of it only ever answers one league's
question while implying it answers everyone's. Every stat column is exported, so any scoring
can be applied downstream.

**Goalies** — the same two steps, with the starts split done first because everything else
is downstream of it:

- **Depth chart** — goalies ranked by claim take measured shares of the team's minutes
  (`.561 / .318 / .087 / .025 / .007`), measured by the rank the *model* can see rather than
  by hindsight; using hindsight ranks would build the model's own error in as knowledge.
- **Own share** — flatter than proportional (`claim**0.60`), because summed claims over-count:
  each goalie's recent starts were partly someone else's injury.
- **Quality tilt** — a coach gives the net to whoever stops the puck. Within a team, a goalie
  5% better than his partner takes about 5 more starts of 82. Applied team-centred, so the
  shifts cancel and the team's 84 starts stay exact.
- **Starts band** — a measured quantile table, not a parametric interval. A backup's season is
  bimodal (eight starts, or forty because the man ahead of him got hurt), so no smooth
  distribution fits: a sqrt-scaled band covered 62% of real outcomes against an 80% target,
  the table covers 82%.

Save percentage is regressed hard to the league (it is genuinely low-signal year to year) and
then settled against the team's expected goals against, which moves it a point or two from
the goalie's own rate — that gap is the shot quality he actually faces.

### Roster coverage: why the totals look conservative

A published roster is not a season. A real team-season uses about 28 skaters and 3.1 goalies;
the rosters the model projects hold 21.9 and 2.9, which is **92% of the ice time and 96% of
the goaltending**. The shortfall is held back rather than handed to the listed players. That is
why a projected starter takes about 43 of 84 starts and not 60: the number is an expectation
that already prices in the injury and the bad November that cost him the net. State a depth
chart on his page if you want a specific plan instead.

**Camp rosters.** "The rosters the model projects" is not quite the same as the roster page,
and only in September. The NHL's endpoint does not mark who is on the active roster, so in
the weeks before a season it returns the whole camp — measured 39.3 players a team, one team
listing 50. Every listed player claims from the same fixed pool of games and minutes, so left
alone that made every team look 23% oversubscribed and settlement took the overshoot off
everyone: McDavid came out at 60 games instead of 76. The regulars were paying for the
invitees. So the roster is cut to `ROSTER_SKATER_CAP` (14 F, 8 D) and `ROSTER_GOALIE_CAP` (3)
by claimed minutes before any budget is built, which is the in-season shape every coverage
constant here was measured against. Nobody is deleted: 419 skaters and 45 goalies are marked
**camp**, keep a full projection, are listed in the app under the camp they are in, and go
straight back on the roster the moment you give one a team on his page.

### Why no game-by-game Monte Carlo

By linearity of expectation, `E[Σ games] = Σ E[games]` — summing simulated games gives the
same season mean as projecting the total directly, only slower. So game-by-game lines are
top-down: the season total is the anchor, distributed over the real schedule with weights
that average 1.0, so `Σ game_i = season_total` exactly (verified to <0.02 points).

## Data sources (all free, no API key)

| Source | Role | Notes |
|---|---|---|
| **MoneyPuck** season summaries | skater/goalie stats, xG, TOI, situation splits | `playerId` IS the NHL id → clean join |
| **MoneyPuck** teams.csv / lines.csv | team offence/defence ratings, line combinations | `lineId` = concatenated player ids |
| **NHL stats REST API** | birthdates (age curves), goalie W/SV%/GAA/SO, clean names | bulk, paged |
| **NHL api-web** (`/roster`, `/club-schedule-season`) | current rosters and the real 84-game schedule | published; joins on player id |

Cached to `data/raw/*.parquet`; re-runs are offline unless `--refresh`. Note that
`timeOnIce` in the NHL goalie summary is **seconds**, and that a traded goalie's row lists
every team he played for — the split is not in the source data.

The app's **Model check** page shows how old each cached file is and can refresh the rosters
and schedule on its own, which are the parts that go stale in the weeks before a season.

## Validation

`backtest.py` projects each past season using only prior data and scores against actuals. The
model beats "last season = next season" and "3-year average" on RMSE every season tested
(2022-2025) and cuts per-60 rate error ~13% versus last-season-only.

The goalie start allocation was calibrated on 269 clean team-seasons (2012-2025): share MSE
0.02454 → 0.02068, busiest-goalie start error 11.3 → 10.6 games, top-goalie identification
73% → 75%.

## Files

- `config.py` — seasons, source URLs, and every projection constant, each with the
  measurement behind it in a comment
- `data_layer.py` — download and cache MoneyPuck / NHL API data
- `age_curves.py` — delta-method empirical age curves
- `budgets.py` — team and goalie budgets, rating persistence, league levels
- `allocate.py` — the settle step: claims against a budget
- `overrides.py` — scenarios (the edits, and the rules for what is editable)
- `context.py` — team strength ratings, per-game schedule context
- `project_skaters.py` / `project_goalies.py` — the season models
- `project_games.py` — top-down game-by-game decomposition (+ optional prop probabilities)
- `backtest.py` — accuracy validation
- `build_workbook.py` — the standalone Google-Sheets workbook
- `run.py` — one-shot runner (`--refresh`, `--games`, `--backtest`)
- `app/` — the Streamlit app (`core.py` is the shared plumbing; `views/` is one file per page)

## Not yet modelled

- **Line and PP-unit chemistry** — `lines.csv` is downloaded but not fed into rates. A
  linemate-quality adjustment is the natural next lever.
- **Explicit role projection** — ice time comes from a player's own history, not from a
  depth-chart model of his new team, so a third-liner traded into a top-six role does not
  get the usage bump until it shows in the data. Editing his ice time in the app is the
  intended workaround.
- **Backtest coverage** — the backtest scores points, not the full stat set, the goalie chain
  or the team-level budgets, and the prediction intervals for blocks / hits / PIM / faceoffs
  still borrow their coefficient from the scoring stats.
