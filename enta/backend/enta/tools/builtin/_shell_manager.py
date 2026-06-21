#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of EnTA.
# The EnTA project belongs to the Dunimd Team.
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


"""Background shell manager used by the bash_io (bash_output / bash_kill /
bash_list) and the ``bash`` tool when ``run_in_background=True``.

The manager owns a small registry of long-lived subprocesses spawned via
:func:`asyncio.create_subprocess_shell`.  Each shell record tracks the
process handle, the running command, the working directory, the start
timestamp, the accumulated stdout / stderr buffers and the read cursor so
that ``read_new_output`` returns only bytes the caller has not seen yet.

The implementation is intentionally pure-Python and dependency-free so it
works both on Linux (where the foreground bash path is served by the
Rust ``sandbox_execute``) and on Kaggle / Windows dev environments where
the Rust extension is unavailable.  A single global singleton is exposed
via :meth:`BackgroundShellManager.instance` so the manager survives
across the lifetime of the training session.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class _ShellRecord:
    """In-memory state of a single backgrounded shell."""

    id: str
    command: str
    cwd: Optional[str]
    started_at: float
    proc: asyncio.subprocess.Process
    _stdout_buffer: bytearray = field(default_factory=bytearray)
    _stderr_buffer: bytearray = field(default_factory=bytearray)
    _stdout_cursor: int = 0
    _stderr_cursor: int = 0
    exit_code: Optional[int] = None
    finished: bool = False

    def append_stdout(self, data: bytes) -> None:
        self._stdout_buffer.extend(data)

    def append_stderr(self, data: bytes) -> None:
        self._stderr_buffer.extend(data)

    def drain_new(self) -> Dict[str, Any]:
        out = bytes(self._stdout_buffer[self._stdout_cursor :])
        err = bytes(self._stderr_buffer[self._stderr_cursor :])
        self._stdout_cursor = len(self._stdout_buffer)
        self._stderr_cursor = len(self._stderr_buffer)
        return {
            "id": self.id,
            "running": not self.finished,
            "exit_code": self.exit_code,
            "stdout": out.decode("utf-8", errors="replace"),
            "stderr": err.decode("utf-8", errors="replace"),
            "command": self.command,
            "cwd": self.cwd or "",
            "started_at": self.started_at,
        }


class BackgroundShellManager:
    """Track and drive backgrounded shells spawned by the bash tool."""

    _instance: Optional["BackgroundShellManager"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._shells: Dict[str, _ShellRecord] = {}
        self._shells_lock: asyncio.Lock = asyncio.Lock()
        self._reader_tasks: Dict[str, asyncio.Task] = {}

    @classmethod
    def instance(cls) -> "BackgroundShellManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def spawn(self, command: str, cwd: Optional[str] = None) -> _ShellRecord:
        """Start ``command`` as a backgrounded shell and return its record."""
        shell_id = uuid.uuid4().hex[:8]
        effective_cwd = cwd if cwd and os.path.isdir(cwd) else os.getcwd()

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=effective_cwd,
        )

        rec = _ShellRecord(
            id=shell_id,
            command=command,
            cwd=effective_cwd,
            started_at=time.time(),
            proc=proc,
        )

        async with self._shells_lock:
            self._shells[shell_id] = rec

        self._reader_tasks[shell_id] = asyncio.create_task(
            self._pump(rec), name=f"bg-shell-{shell_id}"
        )
        return rec

    async def _pump(self, rec: _ShellRecord) -> None:
        """Continuously drain a shell's pipes until it exits."""
        proc = rec.proc
        try:
            assert proc.stdout is not None and proc.stderr is not None
            stdout_task = asyncio.create_task(proc.stdout.read())
            stderr_task = asyncio.create_task(proc.stderr.read())
            while True:
                done, _pending = await asyncio.wait(
                    {stdout_task, stderr_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stdout_task in done:
                    chunk = stdout_task.result()
                    if chunk:
                        rec.append_stdout(chunk)
                    else:
                        stdout_task = asyncio.create_task(proc.stdout.read())
                        done.discard(stdout_task)
                if stderr_task in done:
                    chunk = stderr_task.result()
                    if chunk:
                        rec.append_stderr(chunk)
                    else:
                        stderr_task = asyncio.create_task(proc.stderr.read())
                        done.discard(stderr_task)
                if not done:
                    break
                if proc.returncode is not None and stdout_task.done() and stderr_task.done():
                    tail_out = stdout_task.result()
                    tail_err = stderr_task.result()
                    if tail_out:
                        rec.append_stdout(tail_out)
                    if tail_err:
                        rec.append_stderr(tail_err)
                    break
        except Exception:
            pass
        finally:
            try:
                await proc.wait()
            except Exception:
                pass
            rec.exit_code = proc.returncode
            rec.finished = True

    def read_new_output(self, shell_id: str) -> Dict[str, Any]:
        rec = self._shells.get(shell_id)
        if rec is None:
            return {"error": f"unknown shell id: {shell_id}"}
        return rec.drain_new()

    async def kill(self, shell_id: str, force: bool = False) -> Dict[str, Any]:
        rec = self._shells.get(shell_id)
        if rec is None:
            return {"error": f"unknown shell id: {shell_id}"}

        proc = rec.proc
        try:
            if rec.finished:
                return {
                    "id": shell_id,
                    "running": False,
                    "exit_code": rec.exit_code,
                    "summary": "shell already exited",
                }
            if force:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            else:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                if not force:
                    try:
                        proc.kill()
                        await proc.wait()
                    except ProcessLookupError:
                        pass
        except Exception as exc:
            return {"error": f"kill failed: {exc}"}
        return {
            "id": shell_id,
            "running": False,
            "exit_code": rec.exit_code,
            "summary": "shell terminated",
        }

    def list_shells(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for rec in self._shells.values():
            result.append(
                {
                    "id": rec.id,
                    "command": rec.command,
                    "cwd": rec.cwd or "",
                    "started_at": rec.started_at,
                    "running": not rec.finished,
                    "exit_code": rec.exit_code,
                }
            )
        return result
