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


"""Bash execution — ALL commands go through Rust ``sandbox_execute``.

There is **one** execution path for every bash command:

    Python  bash._bash_execute()
        └── Rust  sandbox::sandbox_execute(command, timeout, workspace)
                ├── Linux (Landlock): fork → landlock_restrict_self → exec
                │     ↳ read/write workspace ONLY, no network, no exec outside
                ├── Windows:         cmd.exe /C with CREATE_NO_WINDOW
                └── macOS:           sh -c with process-group isolation

No Python-level container sandbox, no fallback chain, no duplicate logic.
The Rust function handles platform differences, timeout, and encoding.

When the user configures ``sandbox_enabled=true`` + a workspace, the
loop injects the workspace path into the contextvar below.  The Rust
layer picks it up automatically and applies Landlock when available.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import os
import sys
from typing import Any

from enta.tools.base import build_tool
from enta.tools.builtin._shell_manager import BackgroundShellManager

# ── Workspace injection (set by the loop per turn) ────────────────
# The active loop injects its workspace path here before each turn.
# The Rust sandbox_execute receives it and applies Landlock when
# available (Linux 5.13+).  On non-Linux this is a no-op.
_current_workspace: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "bash_workspace", default=None,
)


def set_workspace(ws: str | None) -> contextvars.Token:
    """Set the sandbox workspace path for the current turn.

    Called by the loop before each ``_run_impl`` turn.  The returned
    token restores the previous value via ``reset_workspace()``.
    """
    return _current_workspace.set(ws)


def reset_workspace(token: contextvars.Token) -> None:
    _current_workspace.reset(token)


def _get_workspace() -> str | None:
    return _current_workspace.get()


# ── Constants ─────────────────────────────────────────────────────

DEFAULT_MAX_OUTPUT_CHARS = 30_000
_BINARY_PROBE_BYTES = 1024
_BINARY_THRESHOLD = 0.30


# ── Encoding helpers ──────────────────────────────────────────────

def _decode_for_model(value: Any) -> tuple[str, dict[str, Any]]:
    """Best-effort decode of a shell output stream to str + metadata."""
    if value is None:
        return "", {"encoding": "utf-8", "binary": False, "output_bytes": 0}
    if isinstance(value, str):
        raw_bytes = value.encode("utf-8", errors="replace")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(value)
    else:
        return str(value), {"encoding": "utf-8", "binary": False, "output_bytes": 0}
    n = len(raw_bytes)
    if n == 0:
        return "", {"encoding": "utf-8", "binary": False, "output_bytes": 0}
    sample = raw_bytes[:_BINARY_PROBE_BYTES]
    non_printable = sum(
        1 for b in sample if b < 0x09 or (0x0E <= b <= 0x1F) or b == 0x7F
    )
    binary = (non_printable / max(1, len(sample))) > _BINARY_THRESHOLD
    return _decode_bytes(raw_bytes), {"encoding": "utf-8", "binary": binary, "output_bytes": n}


def _decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8", "gbk", "gb18030", "big5", "shift_jis", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode(sys.getdefaultencoding() or "utf-8", errors="replace")


def _truncate(text: str, limit: int) -> tuple[str, bool, int]:
    if limit <= 0 or len(text) <= limit:
        return text, False, 0
    return (
        text[:limit] + f"\n...(truncated, {len(text) - limit} bytes omitted)",
        True,
        len(text) - limit,
    )


# ── Result envelope ───────────────────────────────────────────────

def _envelope(
    command: str,
    stdout: str,
    stderr: str,
    exit_code: int,
    cwd: str | None,
    elapsed_ms: int,
    stdout_meta: dict[str, Any],
    stderr_meta: dict[str, Any],
    max_chars: int,
) -> str:
    """Build the model-facing JSON envelope (same shape as Claude Code / Codex)."""
    stdout_clean, stdout_truncated, stdout_saved = _truncate(stdout, max_chars)
    stderr_clean, stderr_truncated, stderr_saved = _truncate(stderr, max_chars)
    success = exit_code == 0
    summary = (
        f"command exited with code {exit_code}"
        if not success
        else "command succeeded"
    )
    if stdout_truncated or stderr_truncated:
        summary += (
            f" (output truncated: {stdout_saved + stderr_saved} bytes omitted; "
            "raise max_output_chars to see more)"
        )
    if stdout_meta.get("binary") or stderr_meta.get("binary"):
        summary += " [binary stream detected -- decoded with errors=replace]"
    envelope = {
        "success": success,
        "exit_code": exit_code,
        "command": command,
        "cwd": cwd or "",
        "elapsed_ms": elapsed_ms,
        "stdout": stdout_clean,
        "stderr": stderr_clean,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "stdout_bytes": stdout_meta.get("output_bytes", 0),
        "stderr_bytes": stderr_meta.get("output_bytes", 0),
        "stdout_binary": stdout_meta.get("binary", False),
        "stderr_binary": stderr_meta.get("binary", False),
        "summary": summary,
    }
    return json.dumps(envelope, ensure_ascii=False)


# ── Main execute function ─────────────────────────────────────────

async def _bash_execute(**kwargs: Any) -> str:
    command = kwargs.get("command", "")
    if not command:
        return json.dumps({
            "success": False,
            "error": "command is required",
            "summary": "no command provided",
        }, ensure_ascii=False)

    # Background shells still use BackgroundShellManager (Python async)
    if bool(kwargs.get("run_in_background", False)):
        cwd = kwargs.get("cwd") or None
        mgr = BackgroundShellManager.instance()
        try:
            rec = await mgr.spawn(command, cwd=cwd)
        except Exception as exc:
            return json.dumps({
                "success": False,
                "error": f"spawn failed: {exc}",
                "summary": "background spawn failed",
            }, ensure_ascii=False)
        return json.dumps({
            "success": True,
            "id": rec.id,
            "running": True,
            "command": rec.command,
            "cwd": rec.cwd,
            "started_at": rec.started_at,
            "summary": f"background shell started as {rec.id}",
            "hint": "Use bash_output with this id to read output, bash_kill to stop.",
        }, ensure_ascii=False)

    timeout = int(kwargs.get("timeout", 120))
    cwd = kwargs.get("cwd") or None
    max_chars = _resolve_max_chars(kwargs)

    # ── UNIFIED: Rust sandbox_execute for ALL platforms ──────────
    # One function.  One call.  Landlock on Linux, clean subprocess
    # everywhere else.  No Python fallback chain.
    from enta import native as _native

    started = asyncio.get_running_loop().time()
    workspace = _get_workspace()
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                _native.sandbox_execute,
                command,
                timeout,
                workspace,
            ),
        )
    except Exception as exc:
        return json.dumps({
            "success": False,
            "error": f"execution failed: {exc}",
            "command": command,
            "cwd": cwd or "",
            "summary": "execution error",
        }, ensure_ascii=False)

    elapsed_ms = int((asyncio.get_running_loop().time() - started) * 1000)
    stdout_text, stdout_meta = _decode_for_model(result.get("stdout", ""))
    stderr_text, stderr_meta = _decode_for_model(result.get("stderr", ""))
    exit_code = int(result.get("exit_code", -1))

    return _envelope(
        command=command,
        stdout=stdout_text,
        stderr=stderr_text,
        exit_code=exit_code,
        cwd=cwd,
        elapsed_ms=elapsed_ms,
        stdout_meta=stdout_meta,
        stderr_meta=stderr_meta,
        max_chars=max_chars,
    )


def _resolve_max_chars(kwargs: dict[str, Any]) -> int:
    raw = kwargs.get("max_output_chars")
    if raw is None:
        return DEFAULT_MAX_OUTPUT_CHARS
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_OUTPUT_CHARS
    return v if v >= 0 else 0


# ── Tool definition ───────────────────────────────────────────────

EncreBashTool = build_tool(
    name="bash",
    description=(
        "Execute a shell command. **LAST RESORT ONLY** -- use a dedicated tool "
        "if one exists for your task.\n\n"
        "| Instead of bash | Use this dedicated tool |\n"
        "|-----------------|-------------------------|\n"
        "| cat, head, tail, read file content | file_read |\n"
        "| write file, redirect >, tee | file_write, file_edit |\n"
        "| grep, rg, find, locate | grep, glob |\n"
        "| ls, dir, tree, stat | glob |\n"
        "| curl, wget (fetch URL) | web_fetch |\n"
        "| web search queries | web_search |\n"
        "| git commands | git |\n"
        "| npm install, pip, cargo | use their native args; bash for build scripts only |\n"
        "| docker commands | docker |\n"
        "| python scripts, pytest | test_runner |\n"
        "| lint, format, fmt | lint_format |\n"
        "| database queries | database |\n"
        "| spawning sub-agents | agent |\n"
        "| multi-step workflows | workflow |\n"
        "| cron / scheduled tasks | cron_create |\n"
        "| memory management | memory_* |\n"
        "| PDF processing | pdf |\n"
        "| spreadsheet, CSV | spreadsheet |\n"
        "| image processing | image |\n"
        "| browser automation | browser |\n"
        "| Jupyter notebooks | notebook |\n"
        "\n"
        "Only use bash for: running build tools (npm build, cargo build), "
        "dev servers (npm run dev), custom scripts, or operations with "
        "NO dedicated tool available. By default runs synchronously. "
        "Set run_in_background=true for dev servers or watchers.\n\n"
        "Returns a JSON envelope: {success, exit_code, stdout, stderr, "
        "stdout_truncated, stderr_truncated, stdout_bytes, stderr_bytes, "
        "stdout_binary, stderr_binary, elapsed_ms, summary}.  Output is "
        "UTF-8 decoded with errors=replace (handles Chinese / cp1252 / "
        "mixed encodings from cmd.exe).  Each stream is capped at "
        "max_output_chars (default 30000); set to 0 for unlimited.  "
        "Binary streams are detected and flagged so you don't see a "
        "wall of garbled glyphs."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds for foreground execution (default: 120). Ignored in background mode.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the command",
            },
            "run_in_background": {
                "type": "boolean",
                "description": (
                    "If true, spawn the command as a backgrounded shell and "
                    "return a shell id. Use bash_output to read its output "
                    "and bash_kill to stop it."
                ),
            },
            "dangerous": {
                "type": "boolean",
                "description": "Explicitly mark as dangerous to bypass safety checks",
            },
            "max_output_chars": {
                "type": "integer",
                "description": (
                    "Per-stream (stdout / stderr) truncation cap.  "
                    "Output longer than this is truncated with a "
                    "'...(truncated, N bytes omitted)' marker and "
                    "stdout_truncated / stderr_truncated set to true.  "
                    "Default 30000 (~7500 tokens).  Use 0 to disable "
                    "truncation."
                ),
            },
        },
        "required": ["command"],
    },
    execute=_bash_execute,
    intents=["general", "coding", "data"],
)
