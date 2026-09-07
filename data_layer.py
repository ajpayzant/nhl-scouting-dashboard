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


def load_moneypuck_situation(situation: str, refresh: bool = False) -> pd.DataFrame:
    """Skater rows for one on-ice situation, every history season.

    MoneyPuck splits each player-season into 'all', '5on5', '5on4' (power play),
    '4on5' (short handed) and 'other'. The situation matters for a RATE: a player's
    power-play points per 60 minutes of POWER PLAY is a real skill that survives a
    change of team, whereas his PP points per 60 minutes of total ice time is mostly
    a statement about how much power-play time his last coach gave him.
    """
    cache = C.DATA_RAW / f"mp_skaters_{situation}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    frames = []
    for yr in C.HISTORY_SEASONS:
        try:
            df = _moneypuck_csv(C.MONEYPUCK_SKATERS, yr)
        except requests.HTTPError:
            continue
        frames.append(df[df["situation"] == situation].copy())
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


def load_moneypuck_skaters_pp(refresh: bool = False) -> pd.DataFrame:
    """Power-play (5on4) skater rows, for PP-point context. One row per player-season."""
    legacy = C.DATA_RAW / "mp_skaters_pp.parquet"
    if legacy.exists() and not refresh and not (C.DATA_RAW / "mp_skaters_5on4.parquet").exists():
        return pd.read_parquet(legacy)
    return load_moneypuck_situation("5on4", refresh=refresh)


def load_moneypuck_skaters_sh(refresh: bool = False) -> pd.DataFrame:
    """Short-handed (4on5) skater rows, for short-handed points."""
    return load_moneypuck_situation("4on5", refresh=refresh)


SITUATIONS = ("all", "5on5", "5on4", "4on5")


def refresh_skater_situations(situations=SITUATIONS) -> None:
    """Download each season's skater file ONCE and split every situation out of it.

    One MoneyPuck season file carries all five situations, so fetching it per situation
    downloads the same bytes four times. This is the same work in a quarter of the
    requests, which matters when it is 18 seasons.
    """
    frames = {s: [] for s in situations}
    for yr in C.HISTORY_SEASONS:
        try:
            df = _moneypuck_csv(C.MONEYPUCK_SKATERS, yr)
        except requests.HTTPError as e:
            print(f"  [skip] skaters {yr}: {e}")
            continue
        for s in situations:
            frames[s].append(df[df["situation"] == s].copy())
        print(f"  skaters {yr}: {len(df)} rows -> "
              + " ".join(f"{s}={len(frames[s][-1])}" for s in situations))
    for s, parts in frames.items():
        if not parts:
            continue
        out = pd.concat(parts, ignore_index=True)
        name = "mp_skaters_all.parquet" if s == "all" else f"mp_skaters_{s}.parquet"
        out.to_parquet(C.DATA_RAW / name, index=False)
    # Keep the pre-split name working for anything that still asks for it by hand.
    if "5on4" in frames and frames["5on4"]:
        pd.concat(frames["5on4"], ignore_index=True).to_parquet(
            C.DATA_RAW / "mp_skaters_pp.parquet", index=False)


def load_moneypuck_teams_situation(situation: str, refresh: bool = False) -> pd.DataFrame:
    """Team summaries for one on-ice situation (team-level power-play / short-handed)."""
    cache = C.DATA_RAW / f"mp_teams_{situation}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    frames = []
    for yr in C.HISTORY_SEASONS:
        try:
            df = _moneypuck_csv(C.MONEYPUCK_TEAMS, yr)
        except requests.HTTPError:
            continue
        frames.append(df[df["situation"] == situation].copy())
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


def load_nhl_goalie_bios(refresh: bool = False) -> pd.DataFrame:
    """Goalie bios (birthDate, draft position) across history seasons.

    The goalie model had no age term at all, so a 38-year-old was projected exactly
    like a 28-year-old with the same recent numbers. This is the birthdate feed that
    fixes it (age_curves.build_goalie_age_curves).
    """
    cache = C.DATA_RAW / "nhl_goalie_bios.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    frames = []
    for yr in C.HISTORY_SEASONS:
        try:
            rows = _nhl_paged(C.NHL_GOALIE_BIOS, yr)
        except requests.HTTPError as e:
            print(f"  [skip] goalie bios {yr}: {e}")
            continue
        df = pd.DataFrame(rows)
        df["mp_season_year"] = yr
        frames.append(df)
        print(f"  goalie bios {yr}: {len(df)} goalies")
        time.sleep(C.REQUEST_PAUSE)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


def goalie_birthdates(bios: pd.DataFrame) -> pd.DataFrame:
    """Collapse goalie bios to one birthDate + draft position per playerId."""
    b = bios.dropna(subset=["birthDate"]).sort_values("mp_season_year")
    return b.groupby("playerId").agg(
        birthDate=("birthDate", "last"),
        fullName=("goalieFullName", "last"),
        draftOverall=("draftOverall", "last"),
    ).reset_index()


# --------------------------------------------------------------------------- #
# Current-season structural data: teams, rosters, schedule (api-web)          #
# --------------------------------------------------------------------------- #
def active_teams(season: int = C.TARGET_SEASON, refresh: bool = False) -> list[str]:
    """The 32 teams active in the target season (derived from a mid-season game day)."""
    cache = C.DATA_RAW / f"active_teams_{season}.json"
    if cache.exists() and not refresh:
        import json
        return json.loads(cache.read_text())
    resp = _get(C.NHL_DAY_SCHEDULE.format(date=C.TARGET_SEASON_SAMPLE_DATE))
    teams = set()
    for wk in resp.json().get("gameWeek", []):
        for g in wk.get("games", []):
            teams.add(g["awayTeam"]["abbrev"])
            teams.add(g["homeTeam"]["abbrev"])
    out = sorted(teams)
    import json
    cache.write_text(json.dumps(out))
    return out


def load_rosters(season: int = C.TARGET_SEASON, refresh: bool = False) -> pd.DataFrame:
    """Current published roster for every active team in `season`.

    This is how we know a player's team for the UPCOMING season (captures trades /
    free-agent moves that season-summary stats, tied to the prior team, cannot).
    One row per (playerId, team). Columns: playerId, team, positionCode, fullName, ...
    """
    cache = C.DATA_RAW / f"rosters_{season}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    sid = C.season_id(season)
    rows = []
    for team in active_teams(season, refresh=refresh):
        try:
            resp = _get(C.NHL_ROSTER.format(team=team, season_id=sid))
        except requests.HTTPError as e:
            print(f"  [skip] roster {team}: {e}")
            continue
        data = resp.json()
        for grp, grp_pos in (("forwards", "F"), ("defensemen", "D"), ("goalies", "G")):
            for p in data.get(grp, []):
                rows.append({
                    "playerId": p["id"],
                    "team": team,
                    "roster_group": grp_pos,
                    "positionCode": p.get("positionCode"),
                    "fullName": f"{p.get('firstName', {}).get('default', '')} "
                                f"{p.get('lastName', {}).get('default', '')}".strip(),
                    "birthDate": p.get("birthDate"),
                    "shootsCatches": p.get("shootsCatches"),
                })
        print(f"  roster {team}: {len(data.get('forwards', []))}F "
              f"{len(data.get('defensemen', []))}D {len(data.get('goalies', []))}G")
        time.sleep(C.REQUEST_PAUSE)
    out = pd.DataFrame(rows)
    out.to_parquet(cache, index=False)
    return out


def load_schedule(season: int = C.TARGET_SEASON, refresh: bool = False) -> pd.DataFrame:
    """Full regular-season schedule for `season`: one row per team per game.

    Long format (each game appears twice, once per team) so per-team opponent /
    home-away / rest can be computed directly. gameType 2 = regular season.
    """
    cache = C.DATA_RAW / f"schedule_{season}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    sid = C.season_id(season)
    seen_games = set()
    rows = []
    for team in active_teams(season, refresh=refresh):
        try:
            resp = _get(C.NHL_CLUB_SCHEDULE.format(team=team, season_id=sid))
        except requests.HTTPError as e:
            print(f"  [skip] schedule {team}: {e}")
            continue
        for g in resp.json().get("games", []):
            if g.get("gameType") != 2:
                continue
            gid = g["id"]
            home = g["homeTeam"]["abbrev"]
            away = g["awayTeam"]["abbrev"]
            for side, opp, is_home in ((home, away, True), (away, home, False)):
                rows.append({
                    "gameId": gid, "gameDate": g["gameDate"], "season": season,
                    "team": side, "opponent": opp, "is_home": is_home,
                    "neutralSite": g.get("neutralSite", False),
                })
            seen_games.add(gid)
        time.sleep(C.REQUEST_PAUSE)
    out = pd.DataFrame(rows).drop_duplicates(["gameId", "team"])
    out = out.sort_values(["team", "gameDate"]).reset_index(drop=True)
    print(f"  schedule {season}: {len(seen_games)} unique games, "
          f"{out.groupby('team').size().median():.0f} games/team (median)")
    out.to_parquet(cache, index=False)
    return out


def load_moneypuck_lines(refresh: bool = False) -> pd.DataFrame:
    """5on5 line/pairing combinations per history season (line-chemistry context).

    `lineId` is the players' NHL ids concatenated (7 digits each): a 14-digit id is a
    defense pairing, a 21-digit id is a forward line. Carries xG%/Corsi%/TOI per unit.
    """
    cache = C.DATA_RAW / "mp_lines.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    frames = []
    for yr in C.HISTORY_SEASONS:
        try:
            df = _moneypuck_csv(C.MONEYPUCK_LINES, yr)
        except requests.HTTPError:
            continue
        df = df[df["situation"] == "5on5"].copy()
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


def load_moneypuck_teams(refresh: bool = False) -> pd.DataFrame:
    """All-situation team offense/defense summaries per history season (SOS ratings)."""
    cache = C.DATA_RAW / "mp_teams.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    frames = []
    for yr in C.HISTORY_SEASONS:
        try:
            df = _moneypuck_csv(C.MONEYPUCK_TEAMS, yr)
        except requests.HTTPError:
            continue
        df = df[df["situation"] == "all"].copy()
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


def split_line_ids(line_id) -> list[int]:
    """Decompose a MoneyPuck lineId into its 7-digit NHL player ids."""
    s = str(int(line_id))
    return [int(s[i:i + 7]) for i in range(0, len(s), 7)]


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
