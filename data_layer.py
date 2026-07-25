"""Data acquisition + caching for NHL projections.

Sources (all free, verified live 2026-07):
  - MoneyPuck season summaries (skaters, goalies) -> rich stats, xG, TOI, situation splits.
  - NHL stats REST API -> bulk skater bios (birthdate/age), goalie season summaries (W/SV%/GAA/SO).

MoneyPuck `playerId` IS the NHL player id, so the two sources join cleanly with no name matching.
Everything is cached to data/raw as parquet; re-runs are offline unless `refresh=True`.
"""
from __future__ import annotations

import time
import io
import requests
import pandas as pd

import config as C


def _get(url: str, params: dict | None = None) -> requests.Response:
    r = requests.get(
        url,
        params=params,
        headers={"User-Agent": C.USER_AGENT},
        timeout=C.REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r


# --------------------------------------------------------------------------- #
# MoneyPuck                                                                    #
# --------------------------------------------------------------------------- #
def _moneypuck_csv(url_tmpl: str, year: int) -> pd.DataFrame:
    resp = _get(url_tmpl.format(year=year))
    df = pd.read_csv(io.StringIO(resp.text))
    df = df.copy()  # de-fragment before adding the season column
    df["mp_season_year"] = year  # start-year of the season (2024 = 2024-25)
    return df


def load_moneypuck_skaters(refresh: bool = False) -> pd.DataFrame:
    """All-situation skater rows for every history season, one row per player-season."""
    cache = C.DATA_RAW / "mp_skaters_all.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    frames = []
    for yr in C.HISTORY_SEASONS:
        try:
            df = _moneypuck_csv(C.MONEYPUCK_SKATERS, yr)
        except requests.HTTPError as e:
            print(f"  [skip] skaters {yr}: {e}")
            continue
        df = df[df["situation"] == "all"].copy()
        frames.append(df)
        print(f"  skaters {yr}: {len(df)} players")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


def load_moneypuck_skaters_pp(refresh: bool = False) -> pd.DataFrame:
    """Power-play (5on4) skater rows, for PP-point context. One row per player-season."""
    cache = C.DATA_RAW / "mp_skaters_pp.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    frames = []
    for yr in C.HISTORY_SEASONS:
        try:
            df = _moneypuck_csv(C.MONEYPUCK_SKATERS, yr)
        except requests.HTTPError:
            continue
        df = df[df["situation"] == "5on4"].copy()
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


def load_moneypuck_goalies(refresh: bool = False) -> pd.DataFrame:
    """All-situation goalie rows (xG, danger splits, GP, icetime) for every history season."""
    cache = C.DATA_RAW / "mp_goalies_all.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    frames = []
    for yr in C.HISTORY_SEASONS:
        try:
            df = _moneypuck_csv(C.MONEYPUCK_GOALIES, yr)
        except requests.HTTPError as e:
            print(f"  [skip] goalies {yr}: {e}")
            continue
        df = df[df["situation"] == "all"].copy()
        frames.append(df)
        print(f"  goalies {yr}: {len(df)} goalies")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


# --------------------------------------------------------------------------- #
# NHL stats REST API                                                          #
# --------------------------------------------------------------------------- #
def _nhl_paged(url: str, season: int, extra_expr: str = "") -> list[dict]:
    """Page through an NHL stats REST endpoint for one season (gameTypeId=2 regular)."""
    sid = C.season_id(season)
    expr = f"seasonId={sid} and gameTypeId=2"
    if extra_expr:
        expr += f" and {extra_expr}"
    out: list[dict] = []
    start = 0
    limit = 100
    while True:
        resp = _get(url, params={"limit": limit, "start": start, "cayenneExp": expr})
        payload = resp.json()
        rows = payload.get("data", [])
        out.extend(rows)
        total = payload.get("total", len(out))
        start += limit
        if start >= total or not rows:
            break
        time.sleep(C.REQUEST_PAUSE)
    return out


def load_nhl_skater_bios(refresh: bool = False) -> pd.DataFrame:
    """Bulk skater bios (playerId, birthDate, position, draft) across history seasons.

    A player appears once per season played; we later keep the latest birthDate/position.
    """
    cache = C.DATA_RAW / "nhl_skater_bios.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    frames = []
    for yr in C.HISTORY_SEASONS:
        try:
            rows = _nhl_paged(C.NHL_SKATER_BIOS, yr)
        except requests.HTTPError as e:
            print(f"  [skip] bios {yr}: {e}")
            continue
        df = pd.DataFrame(rows)
        df["mp_season_year"] = yr
        frames.append(df)
        print(f"  bios {yr}: {len(df)} skaters")
        time.sleep(C.REQUEST_PAUSE)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


def load_nhl_goalie_summary(refresh: bool = False) -> pd.DataFrame:
    """Goalie season summaries (W/L/SV/SV%/GAA/SO/GS) across history seasons."""
    cache = C.DATA_RAW / "nhl_goalie_summary.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    frames = []
    for yr in C.HISTORY_SEASONS:
        try:
            rows = _nhl_paged(C.NHL_GOALIE_SUMMARY, yr)
        except requests.HTTPError as e:
            print(f"  [skip] goalie summary {yr}: {e}")
            continue
        df = pd.DataFrame(rows)
        df["mp_season_year"] = yr
        frames.append(df)
        print(f"  goalie summary {yr}: {len(df)} goalies")
        time.sleep(C.REQUEST_PAUSE)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


def player_birthdates(bios: pd.DataFrame) -> pd.DataFrame:
    """Collapse bios to one birthDate + latest position per playerId."""
    b = bios.dropna(subset=["birthDate"]).copy()
    b = b.sort_values("mp_season_year")
    latest = b.groupby("playerId").agg(
        birthDate=("birthDate", "last"),
        positionCode=("positionCode", "last"),
        fullName=("skaterFullName", "last"),
        shootsCatches=("shootsCatches", "last"),
    ).reset_index()
    return latest


if __name__ == "__main__":
    print("Downloading MoneyPuck skaters ...")
    sk = load_moneypuck_skaters()
    print(f"skaters total rows: {len(sk)}\n")
    print("Downloading MoneyPuck goalies ...")
    g = load_moneypuck_goalies()
    print(f"goalies total rows: {len(g)}\n")
    print("Downloading NHL skater bios ...")
    bios = load_nhl_skater_bios()
    print(f"bios total rows: {len(bios)}\n")
    print("Downloading NHL goalie summaries ...")
    gs = load_nhl_goalie_summary()
    print(f"goalie summary total rows: {len(gs)}")
