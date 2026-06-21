"""Sidecar — persistent per-target state (mirrors PDFDRILL/CHATDRILL sidecar.py).

An HTML document's natural subject-of-analysis is a URL (or a local .html file),
not a stable filesystem path you can drop a sidecar next to. So — like CHATDRILL
keys by chat id — htmldrill keys artifacts by a *local id* derived from the URL,
under a work root (``HTMLDRILL_WORK``, default ``./drills``):

    <work>/<id>.htmldrill.json   state: facts, evidence, layers, transitions
    <work>/<id>.htmldrill/       heavy blobs: raw.html, headers.json, model, ...

The sidecar is the single source of truth. Each command reads it on entry, does
its work, appends to it, writes on exit. Facts are cumulative — milestones that
accumulate, not a linear sequence.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

VERSION = "0.1.0"


def work_root(work: Optional[str] = None) -> Path:
    """Resolve the artifact root: explicit arg > $HTMLDRILL_WORK > ./drills."""
    p = work or os.environ.get("HTMLDRILL_WORK") or "drills"
    return Path(p).expanduser()


def resolve_local_id(local_id: str, work: Optional[str] = None) -> str:
    """Canonical id from an existing sidecar, by exact match or unique prefix —
    globs the work dir, so status/steps need no network. Returns local_id
    unchanged when nothing local matches."""
    root = work_root(work)
    if (root / f"{local_id}.htmldrill.json").exists():
        return local_id
    # A URL or an absolute path is NOT a bare id token — it yields a non-relative
    # glob pattern that Path.glob rejects (and a '/'-bearing pattern would never
    # match a flat work dir anyway). Treat those as "no prefix match" and return
    # them unchanged; callers map them through local_id_for separately.
    if "/" in local_id or "\\" in local_id or "://" in local_id:
        return local_id
    matches = sorted(root.glob(f"{local_id}*.htmldrill.json"))
    if len(matches) == 1:
        return matches[0].name[: -len(".htmldrill.json")]
    if len(matches) > 1:
        raise ValueError(f"id prefix {local_id!r} matches {len(matches)} "
                         f"local sidecars — give more characters.")
    return local_id


class Sidecar:
    """Read/write the per-target ``.htmldrill.json`` state file."""

    def __init__(self, local_id: str, work: Optional[str] = None):
        self.local_id = local_id
        root = work_root(work)
        self.json_path = root / f"{local_id}.htmldrill.json"
        self.blob_dir = root / f"{local_id}.htmldrill"
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.json_path.exists():
            self._data = json.loads(self.json_path.read_text(encoding="utf-8"))
        else:
            self._data = {
                "local_id": self.local_id,
                "htmldrill_version": VERSION,
                "facts": [],
                "evidence": {},
                "layers": {},
                "transitions": [],
            }

    def save(self) -> None:
        self._data["htmldrill_version"] = VERSION
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    # -- Facts (cumulative state) --
    @property
    def facts(self) -> set[str]:
        return set(self._data.get("facts", []))

    def add_fact(self, fact: str) -> None:
        facts = self._data.setdefault("facts", [])
        if fact not in facts:
            facts.append(fact)

    def remove_fact(self, fact: str) -> None:
        facts = self._data.get("facts", [])
        if fact in facts:
            facts.remove(fact)

    def has(self, fact: str) -> bool:
        return fact in self.facts

    # -- Evidence --
    @property
    def evidence(self) -> dict:
        return self._data.setdefault("evidence", {})

    def set_evidence(self, key: str, value: Any) -> None:
        self._data.setdefault("evidence", {})[key] = value

    def get_evidence(self, key: str, default: Any = None) -> Any:
        return self._data.get("evidence", {}).get(key, default)

    # -- Layers (references to blobs) --
    @property
    def layers(self) -> dict:
        return self._data.setdefault("layers", {})

    def set_layer(self, name: str, meta: dict) -> None:
        self._data.setdefault("layers", {})[name] = meta

    def get_layer(self, name: str) -> dict | None:
        return self._data.get("layers", {}).get(name)

    # -- Blob storage --
    def write_blob(self, name: str, content: str) -> str:
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        path = self.blob_dir / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    def write_blob_bytes(self, name: str, content: bytes) -> str:
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        path = self.blob_dir / name
        path.write_bytes(content)
        return str(path)

    def read_blob(self, name: str) -> str | None:
        path = self.blob_dir / name
        return path.read_text(encoding="utf-8") if path.exists() else None

    def blob_path(self, name: str) -> Path:
        return self.blob_dir / name

    def has_blob(self, name: str) -> bool:
        return (self.blob_dir / name).exists()

    # -- Transition log --
    def log_transition(self, node: str, from_facts: str, to_fact: str,
                       cost_ms: float = 0, detail: str = "") -> None:
        self._data.setdefault("transitions", []).append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "node": node,
            "from": from_facts,
            "to": to_fact,
            "cost_ms": round(cost_ms, 1),
            "detail": detail,
        })

    @property
    def transitions(self) -> list[dict]:
        return self._data.get("transitions", [])
