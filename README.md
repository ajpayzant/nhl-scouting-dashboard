# NHL Season-Long Projections

Season stat lines for every NHL skater and goalie for the upcoming season (currently
**2026-27**, which is **84 games** under the new CBA), built from multi-season per-60
rates, empirical age curves and a games-played model, then settled against what each team
can actually produce.

Once the season starts it keeps up with it: the projection becomes what a player has already
banked plus what he does with the games his team has left, refreshed every morning, and a
**Model performance** page scores the projections filed earlier in the year against what
actually happened.

The interface is a Streamlit app. The Excel workbook is still produced, as an export.

## Quick start

```bash
pip install -r requirements.txt
python run.py --refresh     # first run: download + cache all source data
run_app.bat                 # open the app (or: python -m streamlit run app/streamlit_app.py)
```

```bash
python run.py                 # write the projection CSVs to output/
python run.py --refresh-live  # in season: re-download last night (3 requests) and reproject
python run.py --snapshot      # also file a dated snapshot for the Model performance page
python run.py --games         # also write game-by-game projections
python run.py --backtest      # also print the accuracy backtest
python build_workbook.py      # the standalone Google-Sheets workbook, from the CSVs
```

Every run prints the window it projected — `preseason — the full 84-game season` in
September, `31% of the season played (26.0 games a team) — banked totals plus 58.0 games to
come` in December.

## Deploying it for other people

The app runs on Streamlit Community Cloud with no changes: main file path
`app/streamlit_app.py`, Python **3.13**, and `requirements.txt` is already pinned below the
next majors of pandas and numpy so a reboot cannot silently upgrade under the app.

The cached source data (`data/raw/*.parquet`, ~19 MB) **is committed on purpose**. A fresh
container therefore boots in seconds instead of making several hundred API calls, and
everyone reading the app is reading the same reviewable snapshot. `.github/workflows/refresh-data.yml`
re-downloads it and commits the diff, which Streamlit Cloud picks up as a redeploy; you can
also run it from the Actions tab whenever a trade happens, or refresh locally and push.

It runs on **two cadences in one file**. Every morning it pulls the season in progress and
the rosters — cheap, and necessary, because once games are being played last night's results
are already part of the projection. On Tuesdays it re-downloads all five seasons of history
as well, which is worth doing weekly and no more, since completed seasons do not change. Both
live in the same workflow so they share a concurrency group and can never fight over a
commit. The deployed app also refreshes the live season itself, hourly (`core.LIVE_TTL`), so
a long-running container does not serve last week's numbers between pushes.

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

`app/streamlit_app.py` — ten pages, all reading the same cached projection:

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
- **Model performance** — how the projections filed earlier this season have actually done:
  which stats are most and least accurate, who was under- and over-projected, whether the
  mid-season projection is beating the preseason one, and each player's chance of still
  reaching the total he was given in September
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

### In season: banked totals plus the games that are left

A preseason projection is a statement about 84 unplayed games. On opening night it stops
being that, and a tool that keeps serving it is wrong in a way that gets worse every night.
So from the moment three team-games have been played (`IN_SEASON_MIN_TEAM_GAMES`) every
projection in the app becomes:

```
projection = what he has already done  +  what he does with the games that are left
```

The first half is not a projection at all, it is a fact. The second is the ordinary model run
against a shorter season: a team's budget is its per-game budget times the games it has
**left**, which is the same arithmetic with a smaller number in it. Nothing about the claim,
the settlement or the budget tilt changes, and `proj_*` still means a full-season total — the
banked half is added back at the end, with `act_*` and `ros_*` kept alongside it. That is
deliberate: no page in the app needed rewriting to become in-season aware.

**The bands narrow on their own.** The p10/p90 interval is drawn on the rest-of-season half
and then shifted up by what is banked, so a player's remaining 20 games in February can move
his total far less than his remaining 84 could in September. No separate rule makes that
happen; it falls out of where the band is drawn.

**Three signals, three speeds**, because they carry different amounts of information per game:

| Signal | Constant | Half-weight at | Why |
|---|---|---|---|
| Ice time per game | `IN_SEASON_TOI_K` | 8 player-games | The fastest signal in the sport, and the one a reader most wants noticed: a promotion to the first line is visible in three games and the model's history cannot see it at all. |
| Availability | `IN_SEASON_GP_K` | 20 team-games | A player who has missed 12 of his team's 20 games is a different bet for the rest of the year than his history says. |
| Team budgets | `IN_SEASON_TEAM_K` | 39 team-games | Half a season before this year's team outscores its rating. Team results are mostly goaltending and schedule early on. |
| Production rates | — | ~half a season | No dial at all: the live season enters through the ordinary ice-time-weighted rate blend, so it moves a player's per-60 by about 5% after five games and owns it by March. |

That last row is the important restraint. Scoring rates are the part of a season most
contaminated by luck, and chasing them is the single easiest way to make an in-season
projection *worse* than the preseason one it replaced — which is precisely the claim the
**Model performance** page exists to check rather than assert.

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

The season in progress is the same MoneyPuck files for the target year, cached separately as
`data/raw/live_{skaters,goalies,teams}_2026.parquet`. Before opening night they 404, and every
loader degrades to empty and keeps whatever is already cached — a failed pull must never delete
last night's data.

The app's **Model check** page shows how old each cached file is, states the window it is
projecting, and has two buttons: refresh the rosters and schedule (the parts that go stale in
the weeks before a season), and refresh last night's stats.

## Validation

`backtest.py` projects each past season using only prior data and scores against actuals. The
model beats "last season = next season" and "3-year average" on RMSE every season tested
(2022-2025) and cuts per-60 rate error ~13% versus last-season-only.

The goalie start allocation was calibrated on 269 clean team-seasons (2012-2025): share MSE
0.02454 → 0.02068, busiest-goalie start error 11.3 → 10.6 games, top-goalie identification
73% → 75%.

### Scoring the live season: snapshots

The backtest grades the model on seasons that are over. `snapshots.py` grades it on the one
being played. Each refresh files a dated, baseline-only copy of the projection — three numbers
per stat (`act_` banked, `ros_` still to come, `proj_` the total) — and a copy that cannot be
edited afterwards is the only honest basis for a scorecard. The cadence is measured in hockey
rather than days: a new snapshot once three more team-games have been played, which lands near
weekly and keeps September from filling with identical copies of the preseason projection.

Each snapshot is scored **only on the games played after it was filed**, never against the
season total, so no vintage is credited with knowing something it could not have known, and
there is something to say from the second week of October. Two consequences worth knowing:

- **Total error is split from rate error.** A projection can be right about a player and wrong
  about how often he plays. The rate-only column charges the model just for the games he
  really played, so a wide gap between the two columns says the model understands the player
  and not his health — a different fix.
- **Vintages are compared per game.** The preseason snapshot is judged over a longer window
  than a January one, so only per-game error can sit in the same column. If that number does
  not fall as the season runs, the in-season signal is not paying for itself, and the page
  says so rather than letting anyone assume it is.

The chance a player still reaches his September total comes from the model's own p10-p90 band
on the games that are left, which needs no new assumption: an 80% band implies a standard
deviation, and the question is whether banked plus rest clears the target.

`data/snapshots/` is committed for the same reason `data/raw/` is — the deployed app has to be
able to read a record it did not create.

## Files

- `config.py` — seasons, source URLs, and every projection constant, each with the
  measurement behind it in a comment
- `data_layer.py` — download and cache MoneyPuck / NHL API data
- `age_curves.py` — delta-method empirical age curves
- `budgets.py` — team and goalie budgets, rating persistence, league levels
- `allocate.py` — the settle step: claims against a budget
- `overrides.py` — scenarios (the edits, and the rules for what is editable)
- `context.py` — team strength ratings, per-game schedule context
- `live.py` — the season in progress: where it stands, what is banked, how fast to believe it
- `snapshots.py` — dated projections and the scoring behind the Model performance page
- `project_skaters.py` / `project_goalies.py` — the season models
- `project_games.py` — top-down game-by-game decomposition (+ optional prop probabilities)
- `backtest.py` — accuracy validation
- `build_workbook.py` — the standalone Google-Sheets workbook
- `run.py` — one-shot runner (`--refresh`, `--refresh-live`, `--snapshot`, `--games`, `--backtest`)
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
