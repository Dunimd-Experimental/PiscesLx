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

"""Background index manager — owns workspace code indices independently of
any WebSocket connection.  Lives at the EncreServer level and survives
window open/close cycles.

Indexing runs in a **subprocess** to avoid blocking the server's event loop.
"""

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from .indexer import EncreCodeIndex

logger = logging.getLogger("encre.codebase.index_manager")

_INDEXER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_data_dir() -> str:
    return os.environ.get("ENCRE_DATA_DIR", os.path.join(os.path.expanduser("~"), ".dunimd", "encre"))


def _progress_path(ws_id: str) -> str:
    return os.path.join(_get_data_dir(), "iwork", ws_id, "index_progress.json")


def _metadata_path(ws_id: str) -> str:
    return os.path.join(_get_data_dir(), "iwork", ws_id, "index_metadata.json")


class IndexManager:
    """Manages code indices per workspace, with background building and
    progress subscribers.

    Thread safety: all access to ``self._indices`` is protected by
    ``self._lock``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._indices: dict[str, dict[str, Any]] = {}
        self._poll_tasks: dict[str, asyncio.Task] = {}
        self._failed: set[str] = set()  # ws_ids whose last index attempt permanently failed

    # ── Public API ───────────────────────────────────────────────────────

    def start_index(self, ws_id: str, ws_path: str, force: bool = False) -> None:
        """Start building index for *ws_path* in a background subprocess.

        If the workspace already has a built index in memory and no index
        task is running, this is a no-op (the index is already ready).

        If *force* is ``True``, the existing index is discarded and rebuilt
        from scratch (used by reindex).  Failed workspaces are only retried
        when *force* is ``True``.

        If the workspace is already being indexed, this is a no-op.
        """
        with self._lock:
            # Never auto-retry a workspace whose subprocess already failed
            if not force and ws_id in self._failed:
                logger.debug("[index_manager] ws=%s is in failed state, skipping auto-retry", ws_id)
                return
            entry = self._indices.get(ws_id)
            if entry and entry.get("subprocess") is not None:
                logger.debug("[index_manager] already indexing ws=%s", ws_id)
                return
            # Already has a built index — skip rebuild unless forced
            if not force and entry and entry.get("index") is not None:
                logger.debug("[index_manager] already indexed ws=%s, skipping", ws_id)
                return
            # Check on-disk metadata — if a successful index exists from a
            # previous server lifetime, skip spawning the subprocess entirely.
            if not force and os.path.isfile(_metadata_path(ws_id)):
                logger.info("[index_manager] on-disk cache hit ws=%s, skipping subprocess", ws_id)
                self._indices[ws_id] = {
                    "index": "__loading__",
                    "subprocess": None,
                    "progress": {"progress": 100, "status": "loading", "files": 0},
                    "subscribers": [],
                }
                # Eagerly load from disk in background — no subprocess window
                loop = asyncio.get_running_loop()
                task = asyncio.ensure_future(self._load_cached_index(ws_id, ws_path))
                with self._lock:
                    self._indices[ws_id]["task"] = task
                return
            self._indices[ws_id] = {
                "index": None,
                "subprocess": None,
                "progress": {"progress": 0, "status": "idle", "files": 0},
                "subscribers": [],
            }
        loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(self._spawn_indexer(ws_id, ws_path))
        with self._lock:
            entry = self._indices.get(ws_id)
            if entry:
                entry["task"] = task

    def reindex(self, ws_id: str, ws_path: str) -> None:
        """Force a full re-index: cancel any in-flight task, start fresh."""
        self.cancel_index(ws_id)
        with self._lock:
            self._failed.discard(ws_id)
            entry = self._indices.get(ws_id)
            if entry:
                entry["index"] = None
        self.start_index(ws_id, ws_path, force=True)

    def delete_index(self, ws_id: str, ws_path: str) -> None:
        """Delete on-disk index files and metadata, clear in-memory state."""
        self.cancel_index(ws_id)
        self._remove_index_files(ws_id, ws_path)
        with self._lock:
            self._indices.pop(ws_id, None)

    def cancel_index(self, ws_id: str) -> None:
        """Cancel any running index subprocess for *ws_id*."""
        with self._lock:
            entry = self._indices.get(ws_id)
            if entry:
                proc = entry.get("subprocess")
                if proc and proc.returncode is None:
                    proc.kill()
                entry["subprocess"] = None
        # Cancel the asyncio task that monitors the subprocess
        task = self._poll_tasks.pop(ws_id, None)
        if task and not task.done():
            task.cancel()

    def shutdown(self) -> None:
        """Cancel all running index tasks (called during server shutdown)."""
        with self._lock:
            for ws_id in list(self._indices.keys()):
                entry = self._indices[ws_id]
                proc = entry.get("subprocess")
                if proc and proc.returncode is None:
                    proc.kill()
        for task in self._poll_tasks.values():
            if not task.done():
                task.cancel()
        self._poll_tasks.clear()

    # ── Subscriber API ───────────────────────────────────────────────────

    def subscribe(self, ws_id: str, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a progress callback for *ws_id*.

        The callback receives a dict with keys ``progress``, ``status``,
        ``files``.  The current state is sent immediately on subscribe.
        """
        with self._lock:
            entry = self._indices.get(ws_id)
            if entry:
                entry["subscribers"].append(callback)
                try:
                    data = dict(entry["progress"])
                    proc = entry.get("subprocess")
                    is_running = proc is not None and proc.returncode is None
                    has_index = entry["index"] is not None
                    if is_running:
                        data["status"] = "indexing"
                    elif has_index:
                        data["status"] = "ready"
                    else:
                        data["status"] = "idle"
                    callback(data)
                except Exception:
                    pass

    def unsubscribe(self, ws_id: str, callback: Callable[[dict[str, Any]], None]) -> None:
        """Remove a previously registered callback."""
        with self._lock:
            entry = self._indices.get(ws_id)
            if entry and callback in entry["subscribers"]:
                entry["subscribers"].remove(callback)

    # ── Query API ─────────────────────────────────────────────────────────

    def get_index(self, ws_id: str) -> EncreCodeIndex | None:
        """Return the built ``EncreCodeIndex`` for *ws_id*, or ``None``."""
        with self._lock:
            entry = self._indices.get(ws_id)
            return entry["index"] if entry else None

    def get_task(self, ws_id: str) -> asyncio.Task | None:
        """Return the active build task for *ws_id*, or ``None``."""
        with self._lock:
            entry = self._indices.get(ws_id)
            return entry.get("task") if entry else None

    def get_status(self, ws_id: str) -> dict[str, Any]:
        """Return current indexing status dict."""
        with self._lock:
            entry = self._indices.get(ws_id)
            if entry:
                proc = entry.get("subprocess")
                is_running = proc is not None and proc.returncode is None
                has_index = entry["index"] is not None
                if is_running:
                    status = "indexing"
                elif has_index:
                    status = "ready"
                elif ws_id in self._failed:
                    status = "error"
                else:
                    status = "idle"
                prog = entry["progress"]
                return {
                    "status": status,
                    "progress": prog.get("progress", 0),
                    "files": prog.get("files", 0),
                }
            return {"status": "no_workspace", "progress": 0, "files": 0}

    # ── Internal ─────────────────────────────────────────────────────────

    async def _spawn_indexer(self, ws_id: str, ws_path: str) -> None:
        """Spawn ``index_worker.py`` as a subprocess and poll its progress."""
        loop = asyncio.get_running_loop()
        data_dir = _get_data_dir()

        prog_path = _progress_path(ws_id)
        os.makedirs(os.path.dirname(prog_path), exist_ok=True)
        # Clear old progress
        try:
            os.remove(prog_path)
        except OSError:
            pass

        # Check for cached metadata (skip re-index if already done)
        meta_path = _metadata_path(ws_id)
        if os.path.isfile(meta_path):
            logger.info("[index_manager] cache hit ws=%s", ws_id)
            try:
                idx = await loop.run_in_executor(None, lambda: EncreCodeIndex(ws_path))
            except Exception:
                idx = None
            if idx and not idx._need_reindex:
                self._notify(ws_id, {"progress": 100, "status": "ready", "files": len(idx._modules)})
                with self._lock:
                    entry = self._indices.get(ws_id)
                    if entry:
                        entry["index"] = idx
                return

        logger.info("[index_manager] cache miss ws=%s, spawning subprocess", ws_id)

        # Spawn the worker subprocess
        python = sys.executable
        worker_script = os.path.join(_INDEXER_ROOT, "codebase", "index_worker.py")
        env = os.environ.copy()
        if _INDEXER_ROOT not in env.get("PYTHONPATH", ""):
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{_INDEXER_ROOT}{os.pathsep}{existing}" if existing else _INDEXER_ROOT
        env["ENCRE_DATA_DIR"] = data_dir

        # Redirect stderr to a log file so crashes are debuggable
        stderr_path = os.path.join(os.path.dirname(prog_path), "index_stderr.log")
        try:
            os.makedirs(os.path.dirname(stderr_path), exist_ok=True)
            stderr_file = open(stderr_path, "wb")
        except Exception:
            stderr_file = None

        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        proc = await asyncio.create_subprocess_exec(
            python, worker_script,
            "--ws-id", ws_id,
            "--ws-path", ws_path,
            "--data-dir", data_dir,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=stderr_file or asyncio.subprocess.DEVNULL,
            **kwargs,
        )

        with self._lock:
            entry = self._indices.get(ws_id)
            if entry:
                entry["subprocess"] = proc

        self._notify(ws_id, {"progress": 0, "status": "indexing", "files": 0})

        # Poll the progress file while the subprocess runs
        async def _poll_progress():
            while True:
                await asyncio.sleep(2)
                try:
                    with open(prog_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._notify(ws_id, data)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                # Check if subprocess exited
                if proc.returncode is not None:
                    break

        poller = asyncio.create_task(_poll_progress())
        self._poll_tasks[ws_id] = poller

        try:
            await asyncio.wait_for(proc.wait(), timeout=600)
        except asyncio.TimeoutError:
            logger.warning("[index_manager] subprocess TIMEOUT after 600s ws=%s", ws_id)
            proc.kill()
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller
            self._poll_tasks.pop(ws_id, None)
            if stderr_file:
                try:
                    stderr_file.close()
                except Exception:
                    pass

        # Read final result or error
        try:
            with open(prog_path, "r", encoding="utf-8") as f:
                final = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            final = {}

        status = final.get("status", "idle")
        if status == "ready":
            # Load the index from disk
            try:
                idx = await loop.run_in_executor(None, lambda: EncreCodeIndex(ws_path))
                with self._lock:
                    entry = self._indices.get(ws_id)
                    if entry:
                        entry["index"] = idx
            except Exception as e:
                logger.warning("[index_manager] failed to load index after build: %s", e)
                status = f"error: {e}"

        is_ready = status == "ready"
        self._notify(ws_id, {
            "progress": final.get("progress", 0),
            "status": status,
            "files": final.get("files", 0),
        })

        with self._lock:
            entry = self._indices.get(ws_id)
            if entry:
                entry["subprocess"] = None
            if not is_ready:
                logger.warning("[index_manager] subprocess FAILED ws=%s status=%s", ws_id, status)
                self._failed.add(ws_id)
            else:
                self._failed.discard(ws_id)

    def _remove_index_files(self, ws_id: str, ws_path: str) -> None:
        """Remove on-disk index files for a workspace."""
        code_index_path = os.path.join(ws_path, ".encre", "code_index.json")
        try:
            if os.path.isfile(code_index_path):
                os.remove(code_index_path)
        except Exception:
            pass

        meta_path = _metadata_path(ws_id)
        try:
            if os.path.isfile(meta_path):
                os.remove(meta_path)
        except Exception:
            pass

        prog_path = _progress_path(ws_id)
        try:
            if os.path.isfile(prog_path):
                os.remove(prog_path)
        except Exception:
            pass

    def _notify(self, ws_id: str, data: dict[str, Any]) -> None:
        """Notify all subscribers of *ws_id* with *data*."""
        with self._lock:
            entry = self._indices.get(ws_id)
            if entry:
                entry["progress"] = {
                    "progress": data.get("progress", 0),
                    "status": data.get("status", "indexing"),
                    "files": data.get("files", 0),
                }
                for cb in list(entry["subscribers"]):
                    try:
                        cb(data)
                    except Exception:
                        pass

    async def _load_cached_index(self, ws_id: str, ws_path: str) -> None:
        """Load a previously-built index from disk without spawning a subprocess."""
        loop = asyncio.get_running_loop()
        try:
            idx = await loop.run_in_executor(None, lambda: EncreCodeIndex(ws_path))
        except Exception as e:
            logger.warning("[index_manager] failed to load cached index ws=%s: %s", ws_id, e)
            idx = None
        with self._lock:
            entry = self._indices.get(ws_id)
            if entry:
                entry["index"] = idx
                if idx is not None:
                    entry["progress"] = {"progress": 100, "status": "ready", "files": len(idx._modules)}
                else:
                    entry["progress"] = {"progress": 0, "status": "error", "files": 0}
            if idx:
                self._notify(ws_id, {"progress": 100, "status": "ready", "files": len(idx._modules)})
            else:
                self._notify(ws_id, {"progress": 0, "status": "error", "files": 0})
