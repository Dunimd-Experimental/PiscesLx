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

"""Background shell manager.

Maintains running shell processes started via the bash tool with
``run_in_background=true``. Output is accumulated into per-shell buffers,
exposing an offset-based read protocol so the bash_output tool can stream
new bytes since the last read without re-sending what the model already saw.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _ShellRecord:
    id: str
    command: str
    cwd: str
    started_at: float
    process: asyncio.subprocess.Process
    stdout_buf: bytearray = field(default_factory=bytearray)
    stderr_buf: bytearray = field(default_factory=bytearray)
    stdout_offset: int = 0
    stderr_offset: int = 0
    exit_code: int | None = None
    reader_tasks: list[asyncio.Task] = field(default_factory=list)
    finished_at: float | None = None

    @property
    def running(self) -> bool:
        return self.exit_code is None and self.process.returncode is None


class BackgroundShellManager:
    """Process-wide registry of backgrounded shells."""

    _instance: "BackgroundShellManager | None" = None
    _MAX_BUFFER_BYTES = 4 * 1024 * 1024  # 4 MiB ring per stream

    def __init__(self) -> None:
        self._shells: dict[str, _ShellRecord] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def instance(cls) -> "BackgroundShellManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    async def spawn(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _ShellRecord:
        """Spawn a shell command in the background and return its record."""
        shell_id = "shell_" + secrets.token_hex(4)

        if sys.platform == "win32":
            # cmd.exe on Windows for shell semantics
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd or None,
                env=({**os.environ, **env} if env else None),
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            # /bin/bash -lc for POSIX. Use a process group so we can kill children.
            proc = await asyncio.create_subprocess_exec(
                "/bin/bash", "-lc", command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd or None,
                env=({**os.environ, **env} if env else None),
                start_new_session=True,
            )

        rec = _ShellRecord(
            id=shell_id,
            command=command,
            cwd=cwd or os.getcwd(),
            started_at=time.time(),
            process=proc,
        )

        rec.reader_tasks.append(asyncio.create_task(self._drain(rec, "stdout")))
        rec.reader_tasks.append(asyncio.create_task(self._drain(rec, "stderr")))
        rec.reader_tasks.append(asyncio.create_task(self._wait(rec)))

        async with self._lock:
            self._shells[shell_id] = rec

        return rec

    async def _drain(self, rec: _ShellRecord, stream: str) -> None:
        pipe = rec.process.stdout if stream == "stdout" else rec.process.stderr
        if pipe is None:
            return
        buf = rec.stdout_buf if stream == "stdout" else rec.stderr_buf
        try:
            while True:
                chunk = await pipe.read(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) > self._MAX_BUFFER_BYTES:
                    # Drop the oldest half, but advance the offset cursor so
                    # consumers don't see a phantom rewind.
                    drop = len(buf) - self._MAX_BUFFER_BYTES // 2
                    del buf[:drop]
                    if stream == "stdout":
                        rec.stdout_offset = max(0, rec.stdout_offset - drop)
                    else:
                        rec.stderr_offset = max(0, rec.stderr_offset - drop)
        except Exception:
            pass

    async def _wait(self, rec: _ShellRecord) -> None:
        try:
            rec.exit_code = await rec.process.wait()
            rec.finished_at = time.time()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_new_output(self, shell_id: str) -> dict[str, Any]:
        rec = self._shells.get(shell_id)
        if rec is None:
            return {"error": f"Unknown shell id: {shell_id}"}

        out_bytes = bytes(rec.stdout_buf[rec.stdout_offset:])
        err_bytes = bytes(rec.stderr_buf[rec.stderr_offset:])
        rec.stdout_offset = len(rec.stdout_buf)
        rec.stderr_offset = len(rec.stderr_buf)

        return {
            "id": rec.id,
            "command": rec.command,
            "running": rec.running,
            "exit_code": rec.exit_code,
            "stdout": out_bytes.decode("utf-8", errors="replace"),
            "stderr": err_bytes.decode("utf-8", errors="replace"),
            "stdout_total_bytes": len(rec.stdout_buf),
            "stderr_total_bytes": len(rec.stderr_buf),
            "started_at": rec.started_at,
            "finished_at": rec.finished_at,
        }

    # ------------------------------------------------------------------
    # Kill
    # ------------------------------------------------------------------

    async def kill(self, shell_id: str, force: bool = False) -> dict[str, Any]:
        rec = self._shells.get(shell_id)
        if rec is None:
            return {"error": f"Unknown shell id: {shell_id}"}

        if not rec.running:
            return {"id": rec.id, "already_exited": True, "exit_code": rec.exit_code}

        try:
            if sys.platform == "win32":
                rec.process.kill() if force else rec.process.terminate()
            else:
                pgid = os.getpgid(rec.process.pid)
                sig = signal.SIGKILL if force else signal.SIGTERM
                os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except Exception as exc:
            return {"error": f"Failed to signal shell: {exc}"}

        try:
            await asyncio.wait_for(rec.process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            if not force:
                return await self.kill(shell_id, force=True)
            return {"error": "Shell did not die after SIGKILL"}

        return {
            "id": rec.id,
            "killed": True,
            "exit_code": rec.exit_code,
        }

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_shells(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in self._shells.values():
            out.append({
                "id": rec.id,
                "command": rec.command,
                "cwd": rec.cwd,
                "started_at": rec.started_at,
                "finished_at": rec.finished_at,
                "running": rec.running,
                "exit_code": rec.exit_code,
                "stdout_total_bytes": len(rec.stdout_buf),
                "stderr_total_bytes": len(rec.stderr_buf),
            })
        return out
