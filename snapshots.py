"""Timestamped projections, and how well they turned out.

A projection that is silently overwritten every night can never be wrong, which is the
same as saying it can never be judged. So every time the model runs it can file a dated
copy of itself here, and this module scores those copies against what actually happened.

WHAT A SNAPSHOT IS. One file per date per kind (`skaters_2026-11-14.parquet`), holding the
BASELINE projection only -- never a user's scenario, because the question being asked is
"how good is the model", and a snapshot of somebody's opinion about McDavid cannot answer
it. Each row keeps three numbers per stat, which is what makes scoring possible at all:

    act_x   what he had already done when the snapshot was taken
    ros_x   what the model said he would do in the games that were LEFT
    proj_x  the sum: the season total the snapshot was claiming

plus the p10/p90 band on the rest-of-season half, the games his team had played, and the
games it had left. A preseason snapshot is just the case where `act_x` is zero.

HOW A SNAPSHOT IS SCORED. Not against the season total -- that is unknowable until April.
Against the window that followed it:

    actual   = act_x now  -  act_x then
    expected = ros_x  x  (the fraction of that window which has since been played)

which is a fair question the day after the snapshot and stays fair all season. Two errors
are separated because they are different failures and a reader needs to know which one he
is looking at:

    TOTAL error       expected above, availability included -- the number he cares about
    RATE error        (ros_x / ros_gp) x games he ACTUALLY played -- was the per-game call
                      right, given that we now know how often he played?

A model can be excellent at rates and poor at health, and one aggregate number hides it.

WHY VINTAGES ARE COMPARABLE. Each snapshot is scored on its own subsequent window, and
every error is reported per game as well as in total, so a preseason projection judged over
80 games and a February one judged over 20 can be put in the same table honestly.
"""
from __future__ import annotations

import math
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import live

DIR = C.SNAPSHOTS
INDEX = DIR / "index.csv"
_Z = 1.2816                      # the z the p10/p90 bands are drawn at (80% central)

# The stats a snapshot keeps and the performance page ranks. Counts only: a rate is scored
# from the counts it is made of, so storing it as well would let the two disagree.
SKATER_STATS = ["gp", "toi", "points", "goals", "assists", "shots", "pp_points",
                "sh_points", "blocks", "hits", "pim", "ixg", "faceoffs_won"]
GOALIE_STATS = ["starts", "gp", "minutes", "wins", "losses", "saves", "shots_against",
                "goals_against", "shutouts"]

# Which stats a reader recognises as the model's headline claims, in the order a table
# should show them.
HEADLINE = {"skaters": ["points", "goals", "assists", "shots", "gp", "toi"],
            "goalies": ["wins", "starts", "saves", "goals_against", "shutouts"]}

_META = ["playerId", "name", "team", "status", "on_roster", "snap_date",
         "snap_team_gp", "snap_team_left", "snap_frac_played"]

# Team-games that must have been played since the last snapshot before another is filed.
# Teams play about 3.5 games a week, so this is a weekly record kept by a daily job.
MIN_GAP_GAMES = 3.0


# --------------------------------------------------------------------------- #
# filing a snapshot                                                           #
# --------------------------------------------------------------------------- #
def _path(kind: str, day: str) -> Path:
    return DIR / f"{kind}_{day}.parquet"


def _slim(df: pd.DataFrame, stats: list[str], state, day: str,
          extra: list[str] | None = None) -> pd.DataFrame:
    """The columns worth keeping, with the three-number structure forced into place.

    A preseason projection has no `act_`/`ros_` columns because there is nothing banked and
    nothing left over -- it is all one window. Rather than special-case that everywhere
    downstream, it is written here as `act_x = 0, ros_x = proj_x`, which is exactly what it
    means.
    """
    out = pd.DataFrame({"playerId": df["playerId"].astype(int).to_numpy()})
    for col in ["name", "team", "status", "position"] + list(extra or []):
        if col in df:
            out[col] = df[col].to_numpy()
    out["on_roster"] = df["on_roster"].to_numpy() if "on_roster" in df else True
    out["snap_date"] = day
    out["snap_frac_played"] = float(state.frac_played)
    team = out["team"].astype(str)
    played = state.played if len(state.played) else pd.Series(dtype=float)
    left = state.remaining if len(state.played) else pd.Series(dtype=float)
    out["snap_team_gp"] = team.map(played).fillna(0.0).to_numpy(dtype=float)
    out["snap_team_left"] = team.map(left).fillna(float(C.SEASON_GAMES)).to_numpy(dtype=float)

    for stat in stats:
        proj = f"proj_{stat}"
        if proj not in df:
            continue
        out[proj] = df[proj].to_numpy(dtype=float)
        out[f"act_{stat}"] = (df[f"act_{stat}"].to_numpy(dtype=float)
                             if f"act_{stat}" in df else 0.0)
        out[f"ros_{stat}"] = (df[f"ros_{stat}"].to_numpy(dtype=float)
                             if f"ros_{stat}" in df else out[proj].to_numpy())
        # The band on the REST-OF-SEASON half. Mid-season the projection carries both; the
        # `ros_` copies are the ones a score can be computed from.
        for tail in ("p10", "p90"):
            src = f"ros_{stat}_{tail}" if f"ros_{stat}_{tail}" in df else f"{stat}_{tail}"
            if src in df:
                out[f"ros_{stat}_{tail}"] = df[src].to_numpy(dtype=float)
    return out


def _too_soon(state, verbose: bool) -> bool:
    """Has enough hockey been played since the last snapshot to be worth filing another?

    The refresh runs every morning, but a snapshot is only interesting if the season moved
    under it: filing an identical copy of the preseason projection thirty times in September
    would put 10MB of nothing in the repo and add thirty rows a reader has to skip past. So
    the cadence is measured in games rather than days -- one preseason snapshot, then one
    every `MIN_GAP_GAMES` team-games, which lands near weekly once teams are playing.
    """
    idx = index()
    if idx.empty:
        return False
    last = float(idx.iloc[-1].get("team_games_played", 0.0))
    gap = float(state.team_games_played) - last
    if gap >= MIN_GAP_GAMES:
        return False
    if verbose:
        print(f"  nothing new to file: {gap:.1f} team-games since "
              f"{idx.iloc[-1]['snap_date']}, and {MIN_GAP_GAMES} is the threshold")
    return True


def take(day: str | None = None, force: bool = False, verbose: bool = True) -> dict:
    """File today's baseline projection. Idempotent: one snapshot per day per kind.

    Called from `run.py --snapshot`, which the daily refresh workflow runs, so the record
    builds itself without anybody remembering to do it. `force` overrides both guards --
    the one-per-day file check and the games-played cadence.
    """
    import project_goalies as pg
    import project_skaters as ps

    day = day or _date.today().isoformat()
    state = live.season_state()
    if not force and _too_soon(state, verbose):
        return {"skaters": 0, "goalies": 0}
    wrote = {}
    for kind, stats, project, extra in (
            ("skaters", SKATER_STATS, ps.project_skaters, ["position"]),
            ("goalies", GOALIE_STATS, pg.project_goalies, None)):
        path = _path(kind, day)
        if path.exists() and not force:
            if verbose:
                print(f"  {path.name} already filed")
            wrote[kind] = 0
            continue
        df = project()
        slim = _slim(df, stats, state, day, extra)
        if kind == "goalies":
            # Rates are stored for display only; the scoring rebuilds them from the counts.
            for col in ("proj_save_pct", "proj_gaa"):
                if col in df:
                    slim[col] = df[col].to_numpy(dtype=float)
        slim.to_parquet(path, index=False)
        wrote[kind] = len(slim)
        if verbose:
            print(f"  {path.name}: {len(slim)} rows")
    _write_index(day, state, wrote)
    return wrote


def _write_index(day: str, state, wrote: dict) -> None:
    """One line per snapshot date, so the app never has to open every file to list them."""
    row = {"snap_date": day, "as_of": str(state.as_of.date()),
           "in_season": bool(state.live), "frac_played": round(state.frac_played, 4),
           "team_games_played": round(state.team_games_played, 2),
           "team_games_left": round(float(state.remaining.mean()) if len(state.played)
                                    else float(C.SEASON_GAMES), 2),
           "skaters": wrote.get("skaters", 0), "goalies": wrote.get("goalies", 0)}
    idx = pd.read_csv(INDEX) if INDEX.exists() else pd.DataFrame()
    if not idx.empty:
        idx = idx[idx["snap_date"] != day]
    idx = pd.concat([idx, pd.DataFrame([row])], ignore_index=True)
    idx = idx.sort_values("snap_date").reset_index(drop=True)
    idx.to_csv(INDEX, index=False)


# --------------------------------------------------------------------------- #
# reading them back                                                           #
# --------------------------------------------------------------------------- #
def index() -> pd.DataFrame:
    """Every snapshot on file, oldest first. Rebuilt from the directory if it is missing."""
    if INDEX.exists():
        idx = pd.read_csv(INDEX)
        if not idx.empty:
            return idx.sort_values("snap_date").reset_index(drop=True)
    days = sorted({p.stem.split("_", 1)[1] for p in DIR.glob("*_*.parquet")})
    return pd.DataFrame({"snap_date": days, "in_season": [None] * len(days)})


def vintages(kind: str = "skaters") -> list[str]:
    return sorted(p.stem.split("_", 1)[1] for p in DIR.glob(f"{kind}_*.parquet"))


def load(kind: str = "skaters", day: str | None = None) -> pd.DataFrame:
    """One snapshot. `day=None` is the most recent one on file."""
    days = vintages(kind)
    if not days:
        return pd.DataFrame()
    day = day or days[-1]
    path = _path(kind, day)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def latest_scorable(kind: str = "skaters", min_team_games: float = 5.0) -> str | None:
    """The newest snapshot with enough hockey played since it to be worth scoring.

    Tonight's snapshot is the one a page would naturally default to and the one thing it
    cannot say anything about: no games have been played since it was written. So a
    performance page defaults to the newest snapshot that has a real window behind it.
    """
    idx = index()
    if idx.empty:
        return None
    state = live.season_state()
    now = state.team_games_played
    if "team_games_played" not in idx:
        return vintages(kind)[0] if vintages(kind) else None
    ok = idx[(now - idx["team_games_played"].astype(float)) >= min_team_games]
    days = set(vintages(kind))
    ok = ok[ok["snap_date"].isin(days)]
    return str(ok["snap_date"].iloc[-1]) if not ok.empty else None


def _actuals(kind: str) -> pd.DataFrame:
    return live.skater_actuals() if kind == "skaters" else live.goalie_actuals()


# --------------------------------------------------------------------------- #
# scoring one snapshot                                                        #
# --------------------------------------------------------------------------- #
def score_players(kind: str = "skaters", day: str | None = None,
                  stats: list[str] | None = None, min_team_games: float = 2.0
                  ) -> pd.DataFrame:
    """Per-player error on the window that followed a snapshot.

    Returns one row per player with, for every stat: what the snapshot expected over the
    window since (`exp_x`), what he actually did (`obs_x`), the error, and the same pair
    with availability taken out (`rate_exp_x`, on the games he really played). Empty until
    a couple of games have been played since the snapshot -- before that every number is
    noise and a table of them invites the wrong conclusion.
    """
    snap = load(kind, day)
    if snap.empty:
        return pd.DataFrame()
    now = _actuals(kind)
    if now.empty:
        return pd.DataFrame()
    state = live.season_state()
    stats = stats or [s for s in (SKATER_STATS if kind == "skaters" else GOALIE_STATS)
                      if f"proj_{s}" in snap]

    df = snap.merge(now, on="playerId", how="inner", suffixes=("", "_now"))
    # How much of the window the snapshot was projecting has since been played. Per team,
    # because teams are routinely five games apart; a traded player is credited to the team
    # he was on when the snapshot was taken, which is where his projected window came from.
    team_now = df["team"].astype(str).map(state.played).fillna(0.0).astype(float)
    df["team_games_since"] = (team_now - df["snap_team_gp"]).clip(lower=0.0)
    df["window_share"] = np.where(df["snap_team_left"] > 0,
                                  (df["team_games_since"] / df["snap_team_left"]).clip(0, 1),
                                  0.0)
    df = df[df["team_games_since"] >= min_team_games].copy()
    if df.empty:
        return df

    share = df["window_share"].to_numpy(dtype=float)
    ros_gp = df["ros_gp"].to_numpy(dtype=float) if "ros_gp" in df else np.zeros(len(df))
    obs_gp = (df["act_gp_now"] - df["act_gp"]).to_numpy(dtype=float)
    df["obs_gp_since"] = np.round(obs_gp, 1)
    df["exp_gp_since"] = np.round(ros_gp * share, 1)

    for stat in stats:
        ros = df[f"ros_{stat}"].to_numpy(dtype=float)
        obs = (df[f"act_{stat}_now"] - df[f"act_{stat}"]).to_numpy(dtype=float) \
            if f"act_{stat}_now" in df else np.full(len(df), np.nan)
        exp = ros * share
        df[f"obs_{stat}"] = np.round(obs, 2)
        df[f"exp_{stat}"] = np.round(exp, 2)
        df[f"err_{stat}"] = np.round(obs - exp, 2)
        # Rate error: the model's per-game call, charged only for the games he played. This
        # is the half of the error that is about hockey rather than about health.
        per_gp = np.where(ros_gp > 0, ros / np.maximum(ros_gp, 1e-9), 0.0)
        df[f"rate_exp_{stat}"] = np.round(per_gp * obs_gp, 2)
        df[f"rate_err_{stat}"] = np.round(obs - per_gp * obs_gp, 2)
        # The band, honestly rescaled onto a shorter window: a count's spread grows with
        # the square root of the games it covers, not linearly.
        lo, hi = f"ros_{stat}_p10", f"ros_{stat}_p90"
        if lo in df and hi in df:
            sigma = (df[hi].to_numpy(dtype=float) - df[lo].to_numpy(dtype=float)) / (2 * _Z)
            half = _Z * sigma * np.sqrt(np.clip(share, 0, 1))
            df[f"lo_{stat}"] = np.round(exp - half, 2)
            df[f"hi_{stat}"] = np.round(exp + half, 2)
            df[f"in_band_{stat}"] = (obs >= exp - half) & (obs <= exp + half)
    keep_meta = [c for c in _META + ["position", "team_games_since", "window_share",
                                     "obs_gp_since", "exp_gp_since"] if c in df]
    scored = [c for c in df.columns
              if c.startswith(("obs_", "exp_", "err_", "rate_", "lo_", "hi_", "in_band_"))
              and c not in keep_meta]
    return df[keep_meta + scored].reset_index(drop=True)


def score_stats(kind: str = "skaters", day: str | None = None,
                stats: list[str] | None = None, players: pd.DataFrame | None = None,
                min_window_gp: float = 5.0) -> pd.DataFrame:
    """Which stats the model is getting right, ranked worst to best.

    The ranking column is NORMALISED mean absolute error -- error as a share of what
    actually happened -- because a 6-point miss on points and a 6-hit miss on hits are not
    comparable errors, and a table sorted on raw MAE just lists the big stats first.

    `bias` is the signed mean: positive means the players did MORE than projected, i.e. the
    model was under. That sign convention is the one a reader assumes, so it is the one
    used everywhere on the performance page.
    """
    pl = score_players(kind, day, stats) if players is None else players
    if pl is None or pl.empty:
        return pd.DataFrame()
    pl = pl[pl["obs_gp_since"] >= min_window_gp] if "obs_gp_since" in pl else pl
    stats = stats or [c[4:] for c in pl.columns if c.startswith("obs_")
                      and c != "obs_gp_since"]
    rows = []
    for stat in stats:
        obs = pl[f"obs_{stat}"].to_numpy(dtype=float)
        exp = pl[f"exp_{stat}"].to_numpy(dtype=float)
        ok = np.isfinite(obs) & np.isfinite(exp)
        if ok.sum() < 5:
            continue
        obs, exp = obs[ok], exp[ok]
        err = obs - exp
        rate_err = pl[f"rate_err_{stat}"].to_numpy(dtype=float)[ok]
        mean_obs = float(np.mean(obs))
        row = {
            "stat": stat, "n": int(ok.sum()),
            "observed_per_player": round(mean_obs, 2),
            "projected_per_player": round(float(np.mean(exp)), 2),
            "mae": round(float(np.mean(np.abs(err))), 3),
            "rmse": round(float(np.sqrt(np.mean(err ** 2))), 3),
            "bias": round(float(np.mean(err)), 3),
            "nmae": round(float(np.mean(np.abs(err)) / mean_obs), 4) if mean_obs > 0 else np.nan,
            "rate_mae": round(float(np.mean(np.abs(rate_err))), 3),
            "corr": round(float(np.corrcoef(obs, exp)[0, 1]), 3)
            if obs.std() > 0 and exp.std() > 0 else np.nan,
        }
        band = f"in_band_{stat}"
        row["band_coverage"] = round(float(pl.loc[ok, band].mean()), 3) \
            if band in pl else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("nmae", ascending=False).reset_index(drop=True) \
        if not out.empty else out


def misses(kind: str = "skaters", day: str | None = None, stat: str = "points",
           n: int = 15, players: pd.DataFrame | None = None,
           min_window_gp: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(most under-projected, most over-projected) on one stat, by the size of the miss.

    Sorted on the raw miss rather than a ratio: a player who was given four points and
    scored eleven is a more interesting failure than one given 0.2 and getting 0.6, and the
    ratio version of this table is always a list of fourth-liners.
    """
    pl = score_players(kind, day, [stat]) if players is None else players
    if pl is None or pl.empty or f"err_{stat}" not in pl:
        return pd.DataFrame(), pd.DataFrame()
    pl = pl[pl["obs_gp_since"] >= min_window_gp]
    cols = [c for c in ("name", "team", "position", "obs_gp_since", "exp_gp_since",
                        f"obs_{stat}", f"exp_{stat}", f"err_{stat}",
                        f"rate_err_{stat}") if c in pl]
    return (pl.nlargest(n, f"err_{stat}")[cols].reset_index(drop=True),
            pl.nsmallest(n, f"err_{stat}")[cols].reset_index(drop=True))


# --------------------------------------------------------------------------- #
# comparing vintages                                                          #
# --------------------------------------------------------------------------- #
def by_vintage(kind: str = "skaters", stat: str = "points",
               min_team_games: float = 5.0) -> pd.DataFrame:
    """One row per snapshot: how the projection made that day did on the games since.

    This is the table that answers "is the mid-season projection actually better than the
    preseason one". It has to be read per game, and that is why `mae_per_game` is here: the
    preseason snapshot is judged on everything played so far and a snapshot from last week
    on one week, so their totals are not comparable and their per-game errors are.
    """
    rows = []
    for day in vintages(kind):
        pl = score_players(kind, day, [stat], min_team_games=min_team_games)
        if pl.empty or f"err_{stat}" not in pl:
            continue
        pl = pl[pl["obs_gp_since"] >= 5.0]
        if len(pl) < 10:
            continue
        err = pl[f"err_{stat}"].to_numpy(dtype=float)
        gp = pl["obs_gp_since"].to_numpy(dtype=float)
        obs = pl[f"obs_{stat}"].to_numpy(dtype=float)
        band = f"in_band_{stat}"
        rows.append({
            "snap_date": day,
            "frac_played_then": round(float(pl["snap_frac_played"].iloc[0]), 3),
            "window_games": round(float(pl["team_games_since"].mean()), 1),
            "players": len(pl),
            "mae": round(float(np.mean(np.abs(err))), 3),
            "mae_per_game": round(float(np.mean(np.abs(err) / np.maximum(gp, 1e-9))), 4),
            "nmae": round(float(np.mean(np.abs(err)) / max(float(np.mean(obs)), 1e-9)), 4),
            "bias": round(float(np.mean(err)), 3),
            "band_coverage": round(float(pl[band].mean()), 3) if band in pl else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# will he get there?                                                          #
# --------------------------------------------------------------------------- #
def _norm_sf(z: np.ndarray) -> np.ndarray:
    """P(Z > z) without scipy: the app's requirements do not need to grow for one function."""
    return 0.5 * np.array([math.erfc(v / math.sqrt(2.0)) for v in np.asarray(z, dtype=float)])


def reach(kind: str = "skaters", stat: str = "points", target_day: str | None = None,
          current: pd.DataFrame | None = None, min_target: float = 1.0) -> pd.DataFrame:
    """Chance each player still reaches the total a chosen snapshot projected for him.

    The target is the season total from `target_day` (the preseason snapshot by default,
    which is the projection anybody actually remembers being given). The probability comes
    from the model's own rest-of-season band, so it needs no new assumption: the band is an
    80% interval, so the standard deviation it implies is `(p90 - p10) / 2 / 1.2816`, and

        P(reach) = P(banked + rest >= target)

    A player already past his target is at 1.0 by arithmetic rather than by estimate, and
    one who is out for the year with the target ahead of him is at 0.0 for the same reason.
    """
    days = vintages(kind)
    if not days:
        return pd.DataFrame()
    target = load(kind, target_day or days[0])
    if target.empty or f"proj_{stat}" not in target:
        return pd.DataFrame()
    if current is None:
        import project_goalies as pg
        import project_skaters as ps
        current = ps.project_skaters() if kind == "skaters" else pg.project_goalies()
    if f"proj_{stat}" not in current:
        return pd.DataFrame()

    cur = current.copy()
    if f"act_{stat}" not in cur:          # still preseason: nothing is banked
        cur[f"act_{stat}"] = 0.0
        cur[f"ros_{stat}"] = cur[f"proj_{stat}"]
    lo = cur[f"ros_{stat}_p10"] if f"ros_{stat}_p10" in cur else cur.get(f"{stat}_p10")
    hi = cur[f"ros_{stat}_p90"] if f"ros_{stat}_p90" in cur else cur.get(f"{stat}_p90")
    keep = ["playerId", "name", "team", "on_roster", f"act_{stat}", f"ros_{stat}",
            f"proj_{stat}"]
    if "position" in cur:
        keep.append("position")
    df = cur[[c for c in keep if c in cur]].copy()
    df["_lo"] = np.asarray(lo, dtype=float) if lo is not None else np.nan
    df["_hi"] = np.asarray(hi, dtype=float) if hi is not None else np.nan
    df = df.merge(target[["playerId", f"proj_{stat}"]].rename(
        columns={f"proj_{stat}": "target"}), on="playerId", how="inner")
    df = df[df["target"] >= min_target].copy()
    if df.empty:
        return df

    banked = df[f"act_{stat}"].to_numpy(dtype=float)
    rest = df[f"ros_{stat}"].to_numpy(dtype=float)
    sigma = (df["_hi"].to_numpy(dtype=float) - df["_lo"].to_numpy(dtype=float)) / (2 * _Z)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    need = df["target"].to_numpy(dtype=float) - banked
    p = np.where(need <= 0, 1.0,
                 np.where(np.isfinite(sigma) & (sigma > 0),
                          _norm_sf((need - rest) / np.where(sigma > 0, sigma, 1.0)),
                          (rest >= need).astype(float)))
    df["p_reach"] = np.round(np.clip(p, 0.0, 1.0), 3)
    df["pace"] = np.round(banked + rest, 1)
    df["on_pace"] = df["pace"] >= df["target"]
    df["target"] = df["target"].round(1)
    return df.drop(columns=["_lo", "_hi"]).sort_values("target", ascending=False
                                                       ).reset_index(drop=True)


if __name__ == "__main__":
    st = live.season_state()
    print(f"{st.label()}\n")
    print(f"snapshots on file: {vintages('skaters')}")
    idx = index()
    if not idx.empty:
        print(idx.to_string(index=False))
    for kind in ("skaters", "goalies"):
        day = latest_scorable(kind)
        pl = score_players(kind, day)
        print(f"\n=== {kind}: {len(pl)} scored against the {day} snapshot ===")
        if pl.empty:
            continue
        print(score_stats(kind, day, players=pl).to_string(index=False))
        stat = HEADLINE[kind][0]
        under, over = misses(kind, day, stat=stat, n=8, players=pl)
        print(f"\nMost under-projected on {stat}:")
        print(under.to_string(index=False))
        print(f"\nMost over-projected on {stat}:")
        print(over.to_string(index=False))
    vt = by_vintage("skaters")
    if not vt.empty:
        print("\n=== Preseason vs mid-season vintages (points) ===")
        print(vt.to_string(index=False))
    r = reach("skaters", "points")
    if not r.empty:
        print("\n=== Reaching the preseason projection (points) ===")
        print(r.head(12).to_string(index=False))
