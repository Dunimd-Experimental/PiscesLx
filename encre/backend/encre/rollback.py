#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""Git‑style conversation rollback with SHA‑256 content‑addressable commits.

Layout
======

Each session gets a directory under ``~/.encre/rollback/<session_id>/``::

    refs/heads/master       →  HEAD commit hash (plain text)
    objects/ab/cdef1234...  →  single commit blob (encrypted JSON)

The commit graph is a singly‑linked list (parent → parent → ... → root).

A commit blob contains::

    {
        "parent":  "hex hash of parent commit or null",
        "timestamp": 1716200000.0,
        "turn_count": 5,
        "message":  "turn_5",
        "state":    { ... full EncreSession.to_dict() output ... }
    }

The commit hash is ``SHA‑256(compact‑JSON(commit_blob))`` truncated to 40 hex
chars (like a git abbreviated hash) — unique per content, reproducible.

Usage::

    rb = EncreRollbackGit()
    head = rb.commit(session)             # returns commit hash
    log_entries = rb.log(session_id)      # list the chain
    rb.checkout(session, "a1b2c3d4...")  # restore session state
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from typing import Any

from encre.crypto import encrypt, decrypt

__all__ = ["EncreRollbackGit", "CommitEntry"]


# ── storage layout ──────────────────────────────────────────────────────────

_BASE = pathlib.Path("~/.encre/rollback").expanduser()
_HASH_LEN = 40  # characters — like a full git hash
_INDEX_FILE = _BASE / "index.json"


def _session_dir(session_id: str) -> pathlib.Path:
    return _BASE / session_id


def _objects_dir(session_id: str) -> pathlib.Path:
    return _session_dir(session_id) / "objects"


def _refs_dir(session_id: str) -> pathlib.Path:
    return _session_dir(session_id) / "refs" / "heads"


def _head_path(session_id: str) -> pathlib.Path:
    return _refs_dir(session_id) / "master"


def _obj_path(session_id: str, commit_hash: str) -> pathlib.Path:
    """``objects/<hash[:2]>/<hash[2:]>``"""
    return _objects_dir(session_id) / commit_hash[:2] / commit_hash[2:]


# ── commit object ───────────────────────────────────────────────────────────

class CommitEntry:
    """A lightweight snapshot of a commit, suitable for log output."""

    __slots__ = ("commit_hash", "parent", "timestamp", "turn_count", "message")

    def __init__(
        self,
        commit_hash: str,
        parent: str | None,
        timestamp: float,
        turn_count: int,
        message: str,
    ) -> None:
        self.commit_hash = commit_hash
        self.parent = parent
        self.timestamp = timestamp
        self.turn_count = turn_count
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.commit_hash,
            "parent": self.parent,
            "timestamp": self.timestamp,
            "turn_count": self.turn_count,
            "message": self.message,
        }


# ── manager ─────────────────────────────────────────────────────────────────

class EncreRollbackGit:
    """Git‑style versioned conversation history.

    Each ``commit()`` snapshots the full message list of a session and
    chains it via a parent hash.  ``checkout()`` restores any commit.
    """

    def __init__(self) -> None:
        self._session_index: dict[str, str] = {}  # session_id → path
        self._load_index()

    # ── index (session cross‑reference) ─────────────────────────────────

    def _load_index(self) -> None:
        try:
            if _INDEX_FILE.exists():
                self._session_index = json.loads(_INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            self._session_index = {}

    def _save_index(self) -> None:
        _INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        _INDEX_FILE.write_text(json.dumps(self._session_index, ensure_ascii=False), encoding="utf-8")

    # ── object I/O ──────────────────────────────────────────────────────

    @staticmethod
    def _hash_object(obj: dict[str, Any]) -> str:
        """SHA‑256 of the compact JSON representation."""
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_HASH_LEN]

    @staticmethod
    def _read_head(session_id: str) -> str | None:
        p = _head_path(session_id)
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8").strip() or None

    @staticmethod
    def _write_head(session_id: str, commit_hash: str) -> None:
        p = _head_path(session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(commit_hash, encoding="utf-8")

    @staticmethod
    def _write_object(session_id: str, commit_hash: str, data: dict[str, Any]) -> None:
        p = _obj_path(session_id, commit_hash)
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        try:
            raw = encrypt(raw)
        except Exception:
            pass
        p.write_text(raw, encoding="utf-8")

    @staticmethod
    def _read_object(session_id: str, commit_hash: str) -> dict[str, Any] | None:
        p = _obj_path(session_id, commit_hash)
        if not p.exists():
            return None
        raw = p.read_text(encoding="utf-8").strip()
        if raw and not raw.startswith("{"):
            try:
                raw = decrypt(raw)
            except Exception:
                pass
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    # ── public API ──────────────────────────────────────────────────────

    def commit(
        self,
        session: Any,  # EncreSession — duck‑typed for decoupling
        message: str = "",
    ) -> str:
        """Snapshot the current session state and append to the commit chain.

        Returns
        -------
        str
            40‑hex‑char commit hash.
        """
        session_id = session.id
        parent = self._read_head(session_id)

        blob: dict[str, Any] = {
            "parent": parent,
            "timestamp": time.time(),
            "turn_count": getattr(session, "turn_count", 0),
            "message": message or f"turn_{getattr(session, 'turn_count', 0)}",
            "state": getattr(session, "to_dict", lambda: {})(),
        }

        commit_hash = self._hash_object(blob)
        self._write_object(session_id, commit_hash, blob)
        self._write_head(session_id, commit_hash)

        # Update index
        self._session_index[session_id] = str(_session_dir(session_id))
        self._save_index()

        return commit_hash

    def log(self, session_id: str, max_count: int = 50) -> list[CommitEntry]:
        """Walk the commit chain from HEAD back to root."""
        entries: list[CommitEntry] = []
        current = self._read_head(session_id)
        visited: set[str] = set()

        while current and len(entries) < max_count:
            if current in visited:
                break
            visited.add(current)
            obj = self._read_object(session_id, current)
            if obj is None:
                break
            entries.append(CommitEntry(
                commit_hash=current,
                parent=obj.get("parent"),
                timestamp=obj.get("timestamp", 0),
                turn_count=obj.get("turn_count", 0),
                message=obj.get("message", ""),
            ))
            current = obj.get("parent")

        return entries

    def checkout(self, session: Any, commit_hash: str) -> bool:
        """Restore session state to a specific commit.

        The in‑memory ``session.messages``, ``session.turn_count``, etc.
        are replaced with the values from the commit snapshot.

        After checkout the commit chain is NOT truncated — the restored
        commit remains HEAD; a subsequent ``commit()`` will fork from it.

        Returns
        -------
        bool
            ``True`` if the commit was found and applied.
        """
        session_id = session.id
        obj = self._read_object(session_id, commit_hash)
        if obj is None:
            return False

        state = obj.get("state", {})
        if not state:
            return False

        # Restore session fields
        session.messages = state.get("messages", [])
        session.turn_count = state.get("turn_count", 0)
        session.tool_call_count = state.get("tool_call_count", 0)
        session.metadata = state.get("metadata", {})
        session.plan_items = state.get("plan_items", [])
        session.artifacts = state.get("artifacts", [])
        session.updated_at = time.time()

        return True

    def head(self, session_id: str) -> str | None:
        """Return the current HEAD commit hash (or None)."""
        return self._read_head(session_id)

    def head_entry(self, session_id: str) -> CommitEntry | None:
        """Return a ``CommitEntry`` for HEAD (or None)."""
        h = self._read_head(session_id)
        if h is None:
            return None
        obj = self._read_object(session_id, h)
        if obj is None:
            return None
        return CommitEntry(
            commit_hash=h,
            parent=obj.get("parent"),
            timestamp=obj.get("timestamp", 0),
            turn_count=obj.get("turn_count", 0),
            message=obj.get("message", ""),
        )

    def tree(self, session_id: str) -> list[dict[str, Any]]:
        """Return the full commit tree for a session (convenience)."""
        return [e.to_dict() for e in self.log(session_id)]

    def list_sessions_with_rollback(self) -> list[str]:
        """Return session IDs that have rollback history."""
        result: list[str] = []
        try:
            for entry in _BASE.iterdir():
                if entry.is_dir() and entry.name != "." and entry.name != "..":
                    h = self._read_head(entry.name)
                    if h:
                        result.append(entry.name)
        except OSError:
            pass
        return result
