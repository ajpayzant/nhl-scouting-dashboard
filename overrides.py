"""Scenarios: the disagreements a reader is allowed to have with the model.

The workbook could be read and not argued with. Its one editable input was a CSV of
games played, which is a fair choice -- games played is the largest single source of
season-total error -- but it left every other number unarguable, including the ones a
person watching training camp knows better than a five-year rate blend does: who is on
the first power-play unit, who is playing 21 minutes instead of 15, which prospect made
the team.

A scenario is a JSON file of ONLY the disagreements. A player with no entry is the
model's own opinion, so the file stays small enough to read, diff and mail to someone,
and `is_baseline` is a real question with a real answer. Three kinds of edit:

  inputs      -- games, ice time, power-play ice time, a per-60 rate. These flow through
                 the model: raise a player's ice time and his goals, shots, blocks and
                 points all move, and his teammates give up the minutes he gained.
  locks       -- a season total stated outright ("he scores 50"). The number is honoured
                 exactly and comes out of the team's budget FIRST; everyone else settles
                 around what is left. This is the edit that makes the budget useful
                 rather than an obstacle: stating one player's total redistributes the
                 rest instead of being quietly rescaled away.
  structural  -- his team, whether he is on a roster at all. This is how a signing or a
                 trade gets in before the published roster catches up.

Team budgets and the league knobs are editable on the same terms, and clearing an edit
is always possible and always exact, because the baseline was never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config as C

# Per-player fields that feed the model. Editing one of these re-runs the arithmetic
# downstream of it, which is the point: they are causes, not results.
INPUT_FIELDS = ("gp", "toi_per_gp", "pp_toi_per_gp")
RATE_PREFIX = "rate_"

# Season totals that can be stated outright. Stating one locks it: it is taken out of
# the team budget at face value before anyone else is allocated anything.
LOCKABLE = ("goals", "primaryAssists", "secondaryAssists", "assists", "points", "shots",
            "pp_points", "sh_points", "blocks", "hits", "pim", "faceoffs_won")
STRUCTURAL = ("team", "on_roster")

# Goalies. `starts` is the input side (a depth-chart decision, which is what a user
# actually knows better than the model); `save_pct` and the three `rate_*` fields are
# ratings and flow through to goals against, saves, shutouts and GAA together; the rest
# are season totals honoured exactly out of the team budget. Structural edits work the
# same as for skaters.
GOALIE_RATE_FIELDS = ("rate_sa_per_60", "rate_so_per_start", "rate_relief_per_start")
GOALIE_INPUT_FIELDS = ("gp", "starts", "minutes", "save_pct", "wins", "shutouts",
                       "shots_against", "goals_against", "team", "on_roster",
                       *GOALIE_RATE_FIELDS)


def _clean(value: Any) -> Any:
    """JSON-safe, and numpy scalars are not."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


@dataclass
class Scenario:
    """A named set of edits. Immutable-ish: the patch helpers return a new Scenario."""

    name: str = "baseline"
    players: dict[str, dict[str, Any]] = field(default_factory=dict)
    goalies: dict[str, dict[str, Any]] = field(default_factory=dict)
    teams: dict[str, dict[str, Any]] = field(default_factory=dict)
    league: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ state
    @property
    def is_baseline(self) -> bool:
        return not (self.players or self.goalies or self.teams or self.league)

    @property
    def digest(self) -> str:
        """Stable hash of the edits, so a cache can be keyed on "which scenario"."""
        blob = json.dumps(
            {"p": self.players, "g": self.goalies, "t": self.teams, "l": self.league},
            sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def count(self) -> dict[str, int]:
        return {
            "players": len(self.players), "goalies": len(self.goalies),
            "teams": len(self.teams), "league": len(self.league),
            "edits": sum(len(v) for v in self.players.values())
                     + sum(len(v) for v in self.goalies.values())
                     + sum(len(v) for v in self.teams.values()) + len(self.league),
        }

    # ------------------------------------------------------------------ reads
    def player(self, pid) -> dict[str, Any]:
        return dict(self.players.get(str(pid), {}))

    def goalie(self, pid) -> dict[str, Any]:
        return dict(self.goalies.get(str(pid), {}))

    def team(self, team: str) -> dict[str, Any]:
        return dict(self.teams.get(str(team), {}))

    def knob(self, key: str, default=None):
        d = C.league_defaults()
        return self.league.get(key, d.get(key, default))

    def settings(self) -> dict[str, Any]:
        return {**C.league_defaults(), **self.league}

    def edited_ids(self) -> set[str]:
        return set(self.players) | set(self.goalies)

    # ------------------------------------------------------------------ writes
    def set_player(self, pid, **fields) -> "Scenario":
        return self._set("players", pid, fields, allowed=self._player_fields())

    def set_goalie(self, pid, **fields) -> "Scenario":
        return self._set("goalies", pid, fields, allowed=set(GOALIE_INPUT_FIELDS))

    def set_team(self, team, **fields) -> "Scenario":
        return self._set("teams", team, fields, allowed=None)

    def _player_fields(self) -> set[str]:
        return set(INPUT_FIELDS) | set(LOCKABLE) | set(STRUCTURAL) | {
            RATE_PREFIX + s for s in C.SKATER_STATS}

    def _set(self, bucket: str, key, fields: dict, allowed: set[str] | None) -> "Scenario":
        key = str(key)
        store = {b: {k: dict(v) for k, v in getattr(self, b).items()}
                 for b in ("players", "goalies", "teams")}
        entry = store[bucket].setdefault(key, {})
        for k, v in fields.items():
            if allowed is not None and k not in allowed:
                raise ValueError(f"{k!r} is not an editable field ({sorted(allowed)})")
            # None is how the app says "forget this edit", which has to be exact --
            # a scenario that cannot be undone is a scenario nobody will experiment in.
            if v is None:
                entry.pop(k, None)
            else:
                entry[k] = _clean(v)
        if not entry:
            store[bucket].pop(key, None)
        return Scenario(name=self.name, league=dict(self.league), notes=dict(self.notes),
                        **store)

    def clear_player(self, *pids) -> "Scenario":
        return self._clear("players", pids)

    def clear_goalie(self, *pids) -> "Scenario":
        return self._clear("goalies", pids)

    def clear_team(self, *teams) -> "Scenario":
        return self._clear("teams", teams)

    def _clear(self, bucket: str, keys) -> "Scenario":
        store = {b: {k: dict(v) for k, v in getattr(self, b).items()}
                 for b in ("players", "goalies", "teams")}
        for k in keys:
            store[bucket].pop(str(k), None)
        return Scenario(name=self.name, league=dict(self.league), notes=dict(self.notes),
                        **store)

    def clear_all(self) -> "Scenario":
        return Scenario(name=self.name)

    def patch_league(self, **kw) -> "Scenario":
        d = C.league_defaults()
        league = dict(self.league)
        for k, v in kw.items():
            if k not in d:
                raise ValueError(f"{k!r} is not a league setting ({sorted(d)})")
            # Store only the disagreement, so a knob put back to its default leaves no
            # trace and the scenario reads as baseline again.
            if v is None or v == d[k]:
                league.pop(k, None)
            else:
                league[k] = _clean(v)
        return Scenario(name=self.name, players={k: dict(v) for k, v in self.players.items()},
                        goalies={k: dict(v) for k, v in self.goalies.items()},
                        teams={k: dict(v) for k, v in self.teams.items()},
                        league=league, notes=dict(self.notes))

    # ------------------------------------------------------------------ io
    def to_json(self) -> str:
        return json.dumps({"name": self.name, "players": self.players,
                           "goalies": self.goalies, "teams": self.teams,
                           "league": self.league, "notes": self.notes},
                          indent=1, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "Scenario":
        d = json.loads(text) if text.strip() else {}
        return cls(name=d.get("name", "baseline"), players=d.get("players", {}),
                   goalies=d.get("goalies", {}), teams=d.get("teams", {}),
                   league=d.get("league", {}), notes=d.get("notes", {}))

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path else (C.SCENARIOS / f"{self.name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Scenario":
        path = Path(path)
        if not path.exists():
            return cls(name=path.stem)
        sc = cls.from_json(path.read_text(encoding="utf-8"))
        sc.name = sc.name or path.stem
        return sc


def live() -> Scenario:
    """The scenario the app is editing right now."""
    sc = Scenario.load(C.LIVE_SCENARIO)
    sc.name = "working"
    return sc


def save_live(sc: Scenario) -> Path:
    return sc.save(C.LIVE_SCENARIO)


def saved_scenarios() -> list[str]:
    return sorted(p.stem for p in C.SCENARIOS.glob("*.json") if p.stem != "working")


# --------------------------------------------------------------------------- #
# legacy games-played CSV                                                      #
# --------------------------------------------------------------------------- #
def gp_override_csv() -> dict:
    """The old `gp_overrides.csv`, keyed by playerId and by lowercased name.

    Still read, because it was the only editable input the workbook had and anything
    already written in it should keep working. The app writes scenarios instead.
    """
    path = C.GP_OVERRIDE_FILE
    if not path.exists():
        return {}
    try:
        ov = pd.read_csv(path, comment="#", skip_blank_lines=True)
    except Exception:                                          # noqa: BLE001
        return {}
    if ov.empty or "games_played" not in ov:
        return {}
    out: dict = {}
    for _, r in ov.iterrows():
        val = r.get("games_played")
        if pd.isna(val):
            continue
        val = float(np.clip(val, 0, C.MAX_GP))
        if "playerId" in ov.columns and pd.notna(r.get("playerId")):
            out[int(r["playerId"])] = val
        if "name" in ov.columns and pd.notna(r.get("name")):
            out[str(r["name"]).strip().lower()] = val
    return out


if __name__ == "__main__":
    sc = Scenario(name="demo")
    sc = sc.set_player(8478402, gp=84, toi_per_gp=22.5)
    sc = sc.set_player(8477492, goals=55)
    sc = sc.set_team("EDM", goals=290)
    sc = sc.patch_league(budget_tilt=0.5)
    print(sc.to_json())
    print("counts:", sc.count(), "digest:", sc.digest)
    back = Scenario.from_json(sc.to_json())
    assert back.digest == sc.digest, "round trip changed the scenario"
    assert sc.clear_all().is_baseline
    assert sc.patch_league(budget_tilt=C.BUDGET_TILT).league == {}, "default should not persist"
    print("round trip ok")
