#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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

from __future__ import annotations

"""
Real linter / formatter that drives ruff, eslint, prettier and cargo fmt.

The tool is intentionally framework-aware: it inspects the workspace
to pick the right toolchain, invokes the linter / formatter as a real
subprocess, and parses the diagnostic stream into a structured JSON
document that the agent can consume and that the desktop UI can render
as a yellow / red panel.

Supported linters / formatters
------------------------------
- **Python -- ruff** (linter + auto-fixer + formatter).  One tool handles
  ``--check``, ``--fix`` and ``format`` modes, which is why this is
  the single entry point for Python projects.
- **JavaScript / TypeScript -- eslint** (linter + auto-fixer) and
  **prettier** (formatter).  ``auto`` mode runs eslint ``--fix`` then
  prettier ``--write`` on the targeted paths.
- **Rust -- cargo fmt** (formatter) and ``cargo clippy`` (linter).
  ``auto`` mode runs ``cargo fmt`` plus ``cargo clippy --fix``.

All subprocesses are launched with the same window-hiding flags as
the rest of the framework (Windows: ``CREATE_NO_WINDOW``,
Unix: ``start_new_session=True``) so background linting never flashes
a console window in the desktop session.

Output schema
-------------
The tool returns a JSON string with the following fields::

    {
      "linter": "ruff",
      "mode": "check" | "fix" | "format",
      "ok": true | false,
      "duration_s": 1.42,
      "summary": "5 errors, 12 warnings, 34 fixed",
      "diagnostics": [
        {
          "file": "src/example.py",
          "line": 42,
          "column": 5,
          "code": "E501",
          "severity": "error" | "warning" | "info",
          "message": "line too long (110 > 100)",
          "fixed": false
        }
      ],
      "fixed_files": ["src/example.py"],
      "raw_output": "..."
    }
"""

import asyncio
import contextlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from enta.tools.base import build_tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _popen_kwargs() -> dict[str, Any]:
    from enta.tools.builtin._suppress_window import hidden_subprocess_kwargs
    return hidden_subprocess_kwargs()


async def _exec(cmd: list[str], cwd: str, timeout: float) -> tuple[int, str, str]:
    """Run *cmd* in *cwd* and capture (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **_popen_kwargs(),
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return 124, "", f"timeout after {timeout}s"
    out = stdout_b.decode("utf-8", errors="replace")
    err = stderr_b.decode("utf-8", errors="replace")
    return proc.returncode or 0, out, err


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------


def _detect_toolchain(workspace: str, hint: str | None) -> str:
    """Return one of ``"ruff"``, ``"eslint"``, ``"cargo"``."""
    if hint and hint in ("ruff", "eslint", "cargo"):
        return hint
    ws = Path(workspace)
    if (ws / "Cargo.toml").exists():
        return "cargo"
    pkg = ws / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        deps = {
            *(data.get("dependencies") or {}).keys(),
            *(data.get("devDependencies") or {}).keys(),
        }
        if "eslint" in deps or "prettier" in deps:
            return "eslint"
    if (ws / "pyproject.toml").exists() or (ws / "setup.cfg").exists() or (ws / "requirements.txt").exists():  # noqa: E501
        return "ruff"
    # Fallback based on file extensions
    py_files = list(ws.rglob("*.py"))
    if py_files:
        return "ruff"
    js_files = list(ws.rglob("*.js")) + list(ws.rglob("*.ts"))
    if js_files:
        return "eslint"
    return "ruff"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LintDiagnostic:
    file: str
    line: int = 0
    column: int = 0
    code: str = ""
    severity: str = "info"  # error | warning | info
    message: str = ""
    fixed: bool = False


@dataclass
class LintReport:
    linter: str
    mode: str
    ok: bool
    duration_s: float = 0.0
    summary: str = ""
    diagnostics: list[LintDiagnostic] = field(default_factory=list)
    fixed_files: list[str] = field(default_factory=list)
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-toolchain runners
# ---------------------------------------------------------------------------


async def _run_ruff(workspace: str, paths: list[str], mode: str, timeout: float) -> LintReport:
    """Run ruff in ``check`` / ``fix`` / ``format`` mode.

    ``paths`` is the set of relative paths to operate on; an empty
    list means "the whole workspace".
    """
    if mode == "format":
        cmd: list[str] = [sys.executable, "-m", "ruff", "format"]
    elif mode == "fix":
        cmd = [sys.executable, "-m", "ruff", "check", "--fix", "--output-format=concise"]
    else:
        cmd = [sys.executable, "-m", "ruff", "check", "--output-format=concise"]
    if paths:
        cmd.extend(paths)
    t0 = time.time()
    code, out, err = await _exec(cmd, workspace, timeout)
    duration = round(time.time() - t0, 3)
    report = LintReport(linter="ruff", mode=mode, ok=code == 0, duration_s=duration,
                        raw_output=(out + "\n" + err).strip())
    diagnostics: list[LintDiagnostic] = []
    fixed_files: set[str] = set()
    # ruff output formats we handle:
    # - JSON: --output-format=json
    # - Concise: ``path:line:col: CODE message``
    if mode in ("fix", "check"):
        json_payload = await _try_ruff_json(workspace, paths, timeout)
        if json_payload:
            for entry in json_payload:
                loc = entry.get("location") or {}
                file = str(entry.get("filename") or loc.get("row") or "")
                if file and not os.path.isabs(file):
                    file = str(Path(workspace) / file)
                diagnostics.append(
                    LintDiagnostic(
                        file=file,
                        line=int(loc.get("row") or 0),
                        column=int(loc.get("column") or 0),
                        code=str(entry.get("code") or ""),
                        severity=("error" if (entry.get("code") or "").startswith("E") else "warning"),  # noqa: E501
                        message=str(entry.get("message") or "").strip(),
                        fixed=bool(entry.get("fix")),
                    )
                )
                if entry.get("fix") and entry.get("filename"):
                    fixed_files.add(str(entry["filename"]))
    # Fall back to regex parsing of the concise output
    if not diagnostics:
        for line in (out + "\n" + err).splitlines():
            m = re.match(r"^(.+?):(\d+):(\d+):\s+([A-Z]+\d+)\s+(.*)$", line.strip())
            if not m:
                continue
            file, ln, col, code_, message = m.groups()
            diagnostics.append(
                LintDiagnostic(
                    file=file,
                    line=int(ln),
                    column=int(col),
                    code=code_,
                    severity=("error" if code_.startswith("E") else "warning"),
                    message=message.strip(),
                )
            )
    report.diagnostics = diagnostics
    report.fixed_files = sorted(fixed_files)
    report.summary = _summarise_diagnostics(diagnostics, fixed_count=len(fixed_files))
    return report


async def _try_ruff_json(workspace: str, paths: list[str], timeout: float) -> list[dict[str, Any]]:
    """Try to obtain a JSON payload from ``ruff check --output-format=json``."""
    cmd: list[str] = [sys.executable, "-m", "ruff", "check", "--output-format=json"]
    if paths:
        cmd.extend(paths)
    try:
        code, out, _err = await _exec(cmd, workspace, timeout)
    except Exception:
        return []
    if code > 1:
        return []
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


async def _run_eslint(workspace: str, paths: list[str], mode: str, timeout: float) -> LintReport:
    """Run eslint + prettier on JavaScript / TypeScript projects."""
    npx = shutil.which("npx") or shutil.which("npm")
    if npx is None:
        return LintReport(
            linter="eslint", mode=mode, ok=False, summary="npx/npm not found in PATH"
        )
    diagnostics: list[LintDiagnostic] = []
    fixed_files: set[str] = set()
    raw_lines: list[str] = []
    t0 = time.time()
    if mode in ("check", "fix"):
        cmd = [npx, "--no-install", "eslint", "--format=json"]
        if mode == "fix":
            cmd.append("--fix")
        if paths:
            cmd.extend(paths)
        code, out, err = await _exec(cmd, workspace, timeout)
        raw_lines.append(out + "\n" + err)
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            payload = []
        for file_entry in payload:
            file_path = str(file_entry.get("filePath") or file_entry.get("filePath") or "")
            for msg in file_entry.get("messages", []) or []:
                rule = msg.get("ruleId") or ""
                severity_raw = msg.get("severity")
                if severity_raw == 2:
                    severity = "error"
                elif severity_raw == 1:
                    severity = "warning"
                else:
                    severity = "info"
                diagnostics.append(
                    LintDiagnostic(
                        file=file_path,
                        line=int(msg.get("line") or 0),
                        column=int(msg.get("column") or 0),
                        code=rule,
                        severity=severity,
                        message=str(msg.get("message") or "").strip(),
                        fixed=bool(msg.get("fix")),
                    )
                )
                if msg.get("fix") and file_path:
                    fixed_files.add(file_path)
    if mode in ("fix", "format"):
        cmd = [npx, "--no-install", "prettier", "--write"]
        if paths:
            cmd.extend(paths)
        else:
            cmd.append(".")
        code, out, err = await _exec(cmd, workspace, timeout)
        raw_lines.append(out + "\n" + err)
        # Prettier writes a list of rewritten files to stdout
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("Checking"):
                continue
            # Prettier output: ``src/foo.ts 45ms (1)``
            m = re.match(r"^(\S+)\s+\d+(?:\.\d+)?(?:ms|s)\b", line)
            if m:
                fixed_files.add(m.group(1))
            elif line and not line.startswith("npm") and not line.startswith("="):
                fixed_files.add(line.split()[0])
    elif mode == "check":
        cmd = [npx, "--no-install", "prettier", "--check"]
        if paths:
            cmd.extend(paths)
        else:
            cmd.append(".")
        code, out, err = await _exec(cmd, workspace, timeout)
        raw_lines.append(out + "\n" + err)
    duration = round(time.time() - t0, 3)
    ok = not any(
        d.severity == "error"
        for d in diagnostics
    ) and (mode != "check" or code == 0)
    report = LintReport(
        linter="eslint", mode=mode, ok=ok, duration_s=duration,
        raw_output="\n".join(raw_lines).strip(),
        diagnostics=diagnostics,
        fixed_files=sorted(fixed_files),
    )
    report.summary = _summarise_diagnostics(diagnostics, fixed_count=len(fixed_files))
    return report


async def _run_cargo(workspace: str, paths: list[str], mode: str, timeout: float) -> LintReport:
    """Run ``cargo fmt`` / ``cargo clippy`` on Rust projects."""
    diagnostics: list[LintDiagnostic] = []
    fixed_files: set[str] = set()
    raw_lines: list[str] = []
    t0 = time.time()
    if mode in ("check", "format"):
        cmd = ["cargo", "fmt", "--check"]
        code, out, err = await _exec(cmd, workspace, timeout)
        raw_lines.append(out + "\n" + err)
        if code != 0 and not paths:
            # diff format: not a structured report -- record a single
            # summary diagnostic so the agent knows fmt is unhappy.
            diagnostics.append(
                LintDiagnostic(
                    file="<workspace>",
                    severity="warning",
                    message="cargo fmt --check found formatting differences (run with mode='fix' to apply)",  # noqa: E501
                )
            )
    if mode in ("fix", "format"):
        code, out, err = await _exec(["cargo", "fmt"], workspace, timeout)
        raw_lines.append(out + "\n" + err)
        if code == 0:
            for entry in os.scandir(workspace):
                if entry.is_file() and entry.name.endswith(".rs"):
                    fixed_files.add(entry.name)
    if mode in ("check", "fix"):
        clippy_cmd: list[str] = ["cargo", "clippy", "--message-format=json", "--quiet"]
        if mode == "fix":
            clippy_cmd.append("--fix")
        code, out, err = await _exec(clippy_cmd, workspace, timeout)
        raw_lines.append(out + "\n" + err)
        for line in out.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = obj.get("message") or {}
            if not isinstance(message, dict):
                continue
            if obj.get("reason") != "compiler-message":
                continue
            spans = message.get("spans") or []
            primary = next(
                (s for s in spans if (s.get("is_primary") if isinstance(s, dict) else False)),
                spans[0] if spans else None,
            )
            if not isinstance(primary, dict):
                continue
            file_name = primary.get("file_name") or "<unknown>"
            severity_kind = (message.get("level") or "warning").lower()
            if severity_kind not in ("error", "warning", "note", "help"):
                severity_kind = "warning"
            diagnostics.append(
                LintDiagnostic(
                    file=str(file_name),
                    line=int(primary.get("line_start") or 0),
                    column=int(primary.get("column_start") or 0),
                    code=str(message.get("code") or {}).get("code", "") if isinstance(message.get("code"), dict) else "",  # noqa: E501
                    severity=severity_kind,
                    message=str(message.get("message") or "").strip(),
                    fixed=False,
                )
            )
    duration = round(time.time() - t0, 3)
    ok = not any(d.severity == "error" for d in diagnostics) and code == 0
    report = LintReport(
        linter="cargo", mode=mode, ok=ok, duration_s=duration,
        raw_output="\n".join(raw_lines).strip(),
        diagnostics=diagnostics,
        fixed_files=sorted(fixed_files),
    )
    report.summary = _summarise_diagnostics(diagnostics, fixed_count=len(fixed_files))
    return report


def _summarise_diagnostics(
    diagnostics: list[LintDiagnostic],
    fixed_count: int = 0,
) -> str:
    """Build a short human-readable summary from a list of diagnostics."""
    if not diagnostics and fixed_count == 0:
        return "clean"
    errors = sum(1 for d in diagnostics if d.severity == "error")
    warnings = sum(1 for d in diagnostics if d.severity == "warning")
    infos = sum(1 for d in diagnostics if d.severity == "info")
    parts: list[str] = []
    if errors:
        parts.append(f"{errors} error{'s' if errors != 1 else ''}")
    if warnings:
        parts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
    if infos:
        parts.append(f"{infos} info")
    if fixed_count:
        parts.append(f"{fixed_count} fixed")
    return ", ".join(parts) if parts else "clean"


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------


async def _lint_format_execute(**kwargs: Any) -> str:
    workspace = str(kwargs.get("workspace") or kwargs.get("path") or "").strip()
    if not workspace:
        return "Error: workspace is required"
    ws = Path(workspace)
    if not ws.is_dir():
        return f"Error: workspace does not exist or is not a directory: {workspace}"

    linter = _detect_toolchain(workspace, kwargs.get("linter"))
    mode = str(kwargs.get("mode") or "check").lower()
    if mode not in ("check", "fix", "format"):
        return f"Error: mode must be one of check|fix|format, got {mode!r}"

    raw_paths = kwargs.get("paths") or kwargs.get("files")
    if isinstance(raw_paths, str):
        raw_paths = [p.strip() for p in raw_paths.split(",") if p.strip()]
    elif raw_paths is None:
        raw_paths = []
    if not isinstance(raw_paths, list):
        raw_paths = []

    max_duration = float(kwargs.get("max_duration") or kwargs.get("timeout") or 120.0)
    if max_duration <= 0:
        max_duration = 120.0

    if linter == "ruff":
        report = await _run_ruff(workspace, raw_paths, mode, max_duration)
    elif linter == "eslint":
        report = await _run_eslint(workspace, raw_paths, mode, max_duration)
    elif linter == "cargo":
        report = await _run_cargo(workspace, raw_paths, mode, max_duration)
    else:
        return f"Error: unsupported linter: {linter}"

    payload = report.to_dict()
    payload["workspace"] = workspace
    payload["paths"] = raw_paths
    return json.dumps(payload, ensure_ascii=False, indent=2)


EncreLintFormatTool = build_tool(
    name="lint_format",
    description=(
        "Run a linter and / or formatter on the workspace and return a "
        "structured JSON report.  Supports ``ruff`` (Python), "
        "``eslint`` + ``prettier`` (JavaScript / TypeScript) and "
        "``cargo fmt`` + ``cargo clippy`` (Rust).  Mode ``check`` "
        "reports diagnostics without modifying files; ``fix`` runs "
        "linter auto-fixers (ruff check --fix, eslint --fix, cargo "
        "clippy --fix, cargo fmt); ``format`` runs the formatter "
        "(ruff format / prettier --write / cargo fmt).  Returns "
        "per-diagnostic file / line / column / code / severity / "
        "message and a list of files that were modified.  Use the "
        "``paths`` argument to scope to specific files or "
        "directories."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "workspace": {
                "type": "string",
                "description": (
                    "Absolute path to the project root.  Defaults to the "
                    "agent's configured workspace when omitted."
                ),
            },
            "linter": {
                "type": "string",
                "enum": ["ruff", "eslint", "cargo"],
                "description": (
                    "Force a specific toolchain.  When omitted the tool "
                    "auto-detects from project files."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["check", "fix", "format"],
                "description": (
                    "Operation mode.  ``check`` reports diagnostics "
                    "without writing to disk; ``fix`` applies "
                    "linter auto-fixes; ``format`` runs the formatter."
                ),
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of file or directory paths to "
                    "operate on (relative to the workspace).  When "
                    "omitted the entire workspace is processed."
                ),
            },
            "max_duration": {
                "type": "number",
                "description": (
                    "Maximum wall-clock duration in seconds.  Defaults "
                    "to 120.  The subprocess is killed if it overruns."
                ),
            },
        },
    },
    execute=_lint_format_execute,
    intents=["coding", "data", "general"],
)
