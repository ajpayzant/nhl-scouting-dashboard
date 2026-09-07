"""One-shot runner: refresh data (optional), then produce skater + goalie projections.

Usage:
    python run.py                 # use cached data, write projection CSVs
    python run.py --refresh       # re-download all source data first
    python run.py --refresh-live  # re-download only the season in progress (3 requests)
    python run.py --snapshot      # also file a dated snapshot for the performance page
    python run.py --backtest      # also run the accuracy backtest
"""
from __future__ import annotations

import argparse

import config as C
import data_layer as dl
import project_skaters as ps
import project_goalies as pg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download all source data")
    ap.add_argument("--refresh-live", action="store_true",
                    help="re-download only the season in progress")
    ap.add_argument("--snapshot", action="store_true",
                    help="file a dated snapshot of the baseline projection")
    ap.add_argument("--backtest", action="store_true", help="run accuracy backtest")
    ap.add_argument("--games", action="store_true", help="also write game-by-game projections")
    args = ap.parse_args()

    if args.refresh:
        print("Refreshing source data ...")
        dl.load_moneypuck_skaters(refresh=True)
        dl.load_moneypuck_skaters_pp(refresh=True)
        dl.load_moneypuck_goalies(refresh=True)
        dl.load_moneypuck_teams(refresh=True)
        dl.load_nhl_skater_bios(refresh=True)
        dl.load_nhl_goalie_summary(refresh=True)
        dl.load_rosters(refresh=True)
        dl.load_schedule(refresh=True)

    if args.refresh_live or args.refresh:
        print("Refreshing the season in progress ...")
        print(f"  {dl.refresh_live(schedule=not args.refresh)}")

    import live
    print(f"\nWindow: {live.season_state().label()}")

    print(f"\n=== Skater projections for {C.TARGET_SEASON}-{C.TARGET_SEASON+1} ===")
    sk = ps.project_skaters()
    sk_path = C.OUTPUT / f"skater_projections_{C.TARGET_SEASON}.csv"
    sk.to_csv(sk_path, index=False, encoding="utf-8-sig")
    print(f"{len(sk)} skaters -> {sk_path}")
    print(sk[["name", "team", "position", "proj_gp", "proj_points",
              "points_p10", "points_p90", "proj_goals", "proj_assists"]].head(10).to_string(index=False))
    n_ovr = int(sk["gp_override"].sum()) if "gp_override" in sk else 0
    if n_ovr:
        print(f"({n_ovr} projected with a manual games-played override)")

    print(f"\n=== Goalie projections for {C.TARGET_SEASON}-{C.TARGET_SEASON+1} ===")
    g = pg.project_goalies()
    g_path = C.OUTPUT / f"goalie_projections_{C.TARGET_SEASON}.csv"
    g.to_csv(g_path, index=False, encoding="utf-8-sig")
    print(f"{len(g)} goalies -> {g_path}")
    print(g[["name", "team", "proj_gp", "proj_wins", "proj_save_pct",
             "proj_gaa"]].head(10).to_string(index=False))

    if args.snapshot:
        # Dated, baseline-only, one per day: the record the performance page scores against.
        import snapshots
        print("\n=== Filing a projection snapshot ===")
        snapshots.take()

    if args.games:
        import project_games as pgm
        print(f"\n=== Game-by-game skater projections ===")
        games = pgm.project_games(season_proj=sk)
        gpath = C.OUTPUT / f"skater_game_projections_{C.TARGET_SEASON}.csv"
        games.to_csv(gpath, index=False, encoding="utf-8-sig")
        print(f"{len(games)} game rows ({games['playerId'].nunique()} players) -> {gpath}")

    if args.backtest:
        print("\n=== Backtest ===")
        import backtest  # noqa: F401  (module runs its report on import? no — call it)
        import runpy
        runpy.run_module("backtest", run_name="__main__")


if __name__ == "__main__":
    main()
