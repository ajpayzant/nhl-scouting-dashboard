"""Settling player claims against a team budget.

A season is a closed system. A team plays 84 games, and across those games it has
exactly ~297 skater-minutes per night to give, it will score about as many goals as
its offence has historically been worth, and exactly one of its goalies is in net at
a time. Projecting each player in isolation respects none of that, and the sums come
out wrong in a way no per-player override can fix (measured on the pre-budget 2026-27
output: +19% on team ice time, +20% on league goals, +36% on goalie appearances).

So each player states a CLAIM -- what his own history, age curve and role say he
would do if nothing constrained him -- and the claims are then settled against the
budget. `settle()` is the whole of that arithmetic, and the shape of it is the part
worth arguing about:

    a flat rescale is wrong.

If a team's forwards claim 20% more ice time than exists, taking 20% off everyone
says the first-line centre and the 13th forward are equally likely to be the source
of the error. They are not. The surplus on a deep roster is bottom-of-the-lineup ice
time that gets scratched; the top line's minutes are the most certain thing about a
hockey team. So the overshoot is taken back in proportion to `claim**tilt` with
tilt < 1, which means the small claims give back a larger FRACTION of themselves than
the big ones do. At tilt = 1.0 this reduces exactly to the flat rescale, which makes
the knob honest: the caller can always ask for the naive behaviour and see it.

An override is a `lock`: it is taken out of the budget first at face value and the
remaining players settle around what is left. That is what makes editing one player
in the app meaningful -- his number stays put, and his teammates absorb the change.
"""
from __future__ import annotations

import numpy as np


def _settle_free(c: np.ndarray, budget: float, tilt: float) -> np.ndarray:
    """The core rescale for one set of unconstrained claims. Sums to `budget`."""
    total = c.sum()
    if total <= 0:
        return c.copy()
    over = total - budget
    if over <= 0:
        # Under budget: nothing to protect anyone from, so fill the gap proportionally.
        return c * (budget / total)
    # Take the overshoot back in proportion to claim**tilt. `min(c, ...)` stops a
    # player being taken below zero; the final rescale then restores the exact sum,
    # so clipping redistributes rather than leaking budget.
    w = np.power(c, tilt)
    ws = w.sum()
    take = np.minimum(c, over * w / ws) if ws > 0 else np.zeros_like(c)
    kept = c - take
    ks = kept.sum()
    return kept * (budget / ks) if ks > 0 else c * (budget / total)


def settle(claim, want: float, tilt: float = 0.6, locked=None, cap=None,
           only_reduce: bool = False) -> np.ndarray:
    """Scale `claim` so it sums to `want`, taking any overshoot from small claims first.

    claim  : per-player unconstrained projection (non-negative).
    want   : the team's budget for this quantity.
    tilt   : 1.0 = flat proportional rescale (everyone loses the same percentage);
             below 1.0 protects large claims; above 1.0 punishes them.
    locked : optional boolean mask of claims to honour exactly (user overrides). They
             come out of `want` first; the rest settle against what remains.
    cap    : optional per-player ceiling (scalar or array). Anyone who reaches it is
             pinned there and the remainder is re-settled among the rest, so the sum
             still holds.
    only_reduce : if the claims already fit inside the budget, leave them alone instead
             of scaling them up to fill it.

    `only_reduce` exists because of a real and measurable fact about rosters. A team's
    24,978 skater-minutes are played by about 28 different skaters over a season (min 22,
    max 35, measured 2025-26), but a roster published before opening night lists closer
    to 22 -- the rest are call-ups, waiver claims and players signed later. So the listed
    players genuinely account for only ~91% of the ice time, and scaling them up to fill
    the whole budget would hand the missing 9% to the stars, which is precisely wrong:
    those are fourth-line and seventh-defenceman minutes. Left as a shortfall it can be
    reported honestly instead ("35 minutes a game not yet accounted for").

    Returns an array summing to `want` where that is arithmetically possible and the
    claims reach it; otherwise the shortfall is reported rather than invented away.
    """
    claim = np.asarray(claim, dtype=float)
    out = np.where(np.isfinite(claim) & (claim > 0), claim, 0.0)
    if out.size == 0:
        return out

    free = np.ones(out.size, dtype=bool)
    if locked is not None:
        free = ~np.asarray(locked, dtype=bool)

    # Locked claims are honoured at face value and spend the budget first. If they
    # spend all of it there is nothing to settle and the free players go to zero --
    # a real answer to an over-committed set of overrides, not a silent rescale of
    # the numbers the user explicitly asked for.
    budget = float(want) - float(out[~free].sum())
    if budget <= 0:
        out[free] = 0.0
        return out
    if not free.any():
        return out

    caps = None
    if cap is not None:
        caps = np.broadcast_to(np.asarray(cap, dtype=float), out.shape).astype(float)
        # A claim above its own physical ceiling is wrong whether or not the budget
        # binds, so the cap is applied before anything else.
        out[free] = np.minimum(out[free], caps[free])

    base = out.copy()
    if only_reduce and float(base[free].sum()) <= budget:
        return out

    pinned = np.zeros(out.size, dtype=bool)
    for _ in range(out.size + 1):
        active = free & ~pinned
        left = budget - float(out[pinned].sum())
        if not active.any():
            break
        if left <= 0:
            out[active] = 0.0
            break
        out[active] = _settle_free(base[active], left, tilt)
        if caps is None:
            break
        overshot = active & (out > caps)
        if not overshot.any():
            break
        out[overshot] = caps[overshot]
        pinned |= overshot
    return out


def settle_frame(df, claim_col: str, want_by_team: dict, tilt: float = 0.6,
                 team_col: str = "team", lock_col: str | None = None,
                 cap=None, only_reduce: bool = False) -> np.ndarray:
    """`settle` applied per team over a DataFrame; returns the allocation column.

    Teams with no budget entry are passed through unchanged, which is what should
    happen to a player who is not on an NHL roster for the season being projected
    (the old output still had 15 players on the relocated Arizona Coyotes).
    """
    out = df[claim_col].to_numpy(dtype=float).copy()
    caps = None
    if cap is not None:
        caps = np.broadcast_to(np.asarray(cap, dtype=float), out.shape).astype(float)
    locks = df[lock_col].to_numpy() if lock_col and lock_col in df else None
    for team, idx in df.groupby(team_col, sort=False).indices.items():
        want = want_by_team.get(team)
        if want is None or not np.isfinite(want):
            continue
        out[idx] = settle(out[idx], float(want), tilt=tilt,
                          locked=None if locks is None else locks[idx],
                          cap=None if caps is None else caps[idx],
                          only_reduce=only_reduce)
    return out


def budget_report(df, claim_col: str, alloc_col: str, want_by_team: dict,
                  team_col: str = "team"):
    """Per-team claimed vs budgeted vs allocated -- the check that the sums now hold.

    This is the report that would have caught the original error, so it is part of the
    model rather than a script someone remembers to run.
    """
    import pandas as pd

    g = df.groupby(team_col).agg(
        players=(claim_col, "size"), claimed=(claim_col, "sum"), allocated=(alloc_col, "sum"),
    )
    g["budget"] = g.index.map(want_by_team)
    g["claim_vs_budget"] = (g["claimed"] / g["budget"] - 1.0)
    g["error"] = (g["allocated"] - g["budget"]).abs()
    return g.sort_values("claim_vs_budget", ascending=False)


if __name__ == "__main__":
    claims = np.array([100.0, 80.0, 60.0, 40.0, 20.0, 5.0])

    flat = settle(claims, 200.0, tilt=1.0)
    assert np.isclose(flat.sum(), 200.0)
    frac = flat / claims
    assert np.allclose(frac, frac[0]), "tilt=1.0 must be a flat proportional rescale"

    tilted = settle(claims, 200.0, tilt=0.6)
    kept = tilted / claims
    assert np.isclose(tilted.sum(), 200.0)
    assert np.all(np.diff(kept) < 0), "small claims must give back a larger fraction"
    print("claim     ", " ".join(f"{c:7.1f}" for c in claims))
    print("flat  200 ", " ".join(f"{c:7.1f}" for c in flat))
    print("tilt  200 ", " ".join(f"{c:7.1f}" for c in tilted))
    print("kept frac ", " ".join(f"{c:7.3f}" for c in kept))

    # A lock is honoured exactly and the rest settle around it.
    lock = np.array([True, False, False, False, False, False])
    out = settle(claims, 200.0, tilt=0.6, locked=lock)
    assert np.isclose(out[0], 100.0) and np.isclose(out.sum(), 200.0)
    print("locked[0] ", " ".join(f"{c:7.1f}" for c in out))

    # Scaling up respects the cap, and the sum still holds.
    out = settle(claims, 400.0, tilt=0.6, cap=120.0)
    assert np.isclose(out.sum(), 400.0) and out.max() <= 120.0 + 1e-9
    print("cap 120   ", " ".join(f"{c:7.1f}" for c in out))

    # Infeasible: every cap binding, so the shortfall is reported, not invented.
    out = settle(claims, 1000.0, tilt=0.6, cap=50.0)
    assert np.allclose(out, 50.0)
    print("infeasible", " ".join(f"{c:7.1f}" for c in out), f" sum={out.sum():.0f} of 1000")
    print("\nallocate: all checks passed")
