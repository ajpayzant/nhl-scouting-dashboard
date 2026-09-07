"""The shared scenario library: saved projections other people can open.

Locally there is nothing to solve -- a scenario is a small JSON file in `scenarios/` and
the filesystem is yours. Deployed, two things change and both matter.

First, every visitor shares one container, so a single `working.json` on disk would mean
one person's edits silently rewriting everybody else's view. The working scenario is
therefore held in session memory when the app is shared (see `core.MULTIUSER`), and each
visitor starts from the baseline model.

Second, a deployed container's disk is temporary. It is wiped on every redeploy and after
the app sleeps, so "Save as" writing to `scenarios/` would look like it worked and then
lose the file a day later. That is worse than not offering it. So the library has two
backends and says which one it is using:

  gist  -- a GitHub gist, addressed by `gist_id` + `github_token` in Streamlit secrets.
           Durable, versioned by GitHub, shared by every visitor, and the token never
           enters the repository. This is the one that makes "review someone else's
           projections" true.
  disk  -- `scenarios/*.json`. The local default, and the fallback when no gist is
           configured, in which case the app says out loud that saves are temporary.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import requests
import streamlit as st

import config as C
import overrides as ov

GIST_API = "https://api.github.com/gists"
PREFIX = "scenario__"          # so the gist can hold other files without confusing us
TIMEOUT = 20


# --------------------------------------------------------------------------- #
# which backend                                                               #
# --------------------------------------------------------------------------- #
def secret(key: str, default=None):
    """A secret, or the default. Absent secrets are normal, not an error."""
    try:
        return st.secrets.get(key, default)
    except Exception:                                          # noqa: BLE001
        return default


def _gist() -> tuple[str, str] | None:
    gid, tok = secret("gist_id"), secret("github_token")
    return (str(gid), str(tok)) if gid and tok else None


def backend() -> str:
    return "gist" if _gist() else "disk"


def durable() -> bool:
    """Do saves survive a restart? False means the app should say so."""
    return backend() == "gist"


# --------------------------------------------------------------------------- #
# gist backend                                                                #
# --------------------------------------------------------------------------- #
def _headers(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


@st.cache_data(ttl=20, show_spinner=False)
def _gist_files(gid: str, tok: str, bust: int = 0) -> dict[str, str]:
    """{name: json text} for every scenario in the gist. Cached briefly.

    `bust` is bumped after a write so the next read cannot serve a stale list -- the
    alternative, a 20-second window in which your own save is invisible, reads as a bug.
    """
    r = requests.get(f"{GIST_API}/{gid}", headers=_headers(tok), timeout=TIMEOUT)
    r.raise_for_status()
    out = {}
    for fname, meta in (r.json().get("files") or {}).items():
        if not fname.startswith(PREFIX) or not fname.endswith(".json"):
            continue
        text = meta.get("content")
        if meta.get("truncated") and meta.get("raw_url"):
            text = requests.get(meta["raw_url"], timeout=TIMEOUT).text
        out[fname[len(PREFIX):-5]] = text or ""
    return out


def _gist_write(gid: str, tok: str, fname: str, content: str | None) -> None:
    body = {"files": {fname: (None if content is None else {"content": content})}}
    r = requests.patch(f"{GIST_API}/{gid}", headers=_headers(tok), json=body,
                       timeout=TIMEOUT)
    r.raise_for_status()
    st.session_state["lib_bust"] = st.session_state.get("lib_bust", 0) + 1


# --------------------------------------------------------------------------- #
# the library                                                                 #
# --------------------------------------------------------------------------- #
def _meta(name: str, text: str) -> dict:
    """Name, author and size of one saved scenario, for the list a reader picks from."""
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return {"name": name, "author": "?", "saved": "?", "edits": 0, "broken": True}
    notes = d.get("notes") or {}
    edits = sum(len(v) for k in ("players", "goalies", "teams")
                for v in (d.get(k) or {}).values()) + len(d.get("league") or {})
    return {"name": name, "author": notes.get("author") or "-",
            "saved": (notes.get("saved_at") or "-")[:16].replace("T", " "),
            "edits": edits, "broken": False}


def entries() -> list[dict]:
    """Every saved scenario, newest first."""
    g = _gist()
    if g:
        try:
            files = _gist_files(*g, bust=st.session_state.get("lib_bust", 0))
        except requests.RequestException as exc:
            st.warning(f"The scenario library is unreachable ({exc}). "
                       "Your own edits are unaffected.")
            return []
        rows = [_meta(n, t) for n, t in files.items()]
    else:
        rows = []
        for p in sorted(C.SCENARIOS.glob("*.json")):
            if p.stem == "working":
                continue
            rows.append(_meta(p.stem, p.read_text(encoding="utf-8")))
    return sorted(rows, key=lambda r: r["saved"], reverse=True)


def names() -> list[str]:
    return [r["name"] for r in entries()]


def save(sc: ov.Scenario, name: str, author: str = "") -> None:
    """Publish a copy of `sc` under `name`. Overwrites a scenario of the same name."""
    name = "".join(ch for ch in name.strip() if ch.isalnum() or ch in " -_").strip()
    if not name:
        raise ValueError("A scenario needs a name.")
    notes = dict(sc.notes)
    notes["author"] = (author or "anonymous").strip()[:60]
    notes["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    copy = ov.Scenario(name=name, players=sc.players, goalies=sc.goalies, teams=sc.teams,
                       league=sc.league, notes=notes)
    g = _gist()
    if g:
        _gist_write(*g, f"{PREFIX}{name}.json", copy.to_json())
    else:
        copy.save(C.SCENARIOS / f"{name}.json")


def load(name: str) -> ov.Scenario:
    g = _gist()
    if g:
        files = _gist_files(*g, bust=st.session_state.get("lib_bust", 0))
        if name not in files:
            raise FileNotFoundError(name)
        sc = ov.Scenario.from_json(files[name])
    else:
        path = C.SCENARIOS / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(name)
        sc = ov.Scenario.load(path)
    sc.name = "working"
    return sc


def delete(name: str) -> None:
    g = _gist()
    if g:
        _gist_write(*g, f"{PREFIX}{name}.json", None)
    else:
        (C.SCENARIOS / f"{name}.json").unlink(missing_ok=True)
