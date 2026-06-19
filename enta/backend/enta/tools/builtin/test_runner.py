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



"""
Real test runner that drives pytest, vitest, jest and cargo test.

The tool is intentionally framework-aware: it inspects the workspace
to pick the right test runner, invokes it as a real subprocess, and
parses the resulting report into a structured JSON document that the
agent can consume (and that the desktop UI can render as a red/green
test panel).

Supported frameworks
--------------------
- **pytest** -- discovered from ``pyproject.toml`` (``[tool.pytest]``),
  ``pytest.ini``, ``setup.cfg`` or ``tox.ini``.  Output is parsed from
  JUnit XML (``--junit-xml``) which is built into pytest.
- **vitest** -- discovered from ``package.json`` (dependency
  ``vitest`` or config file ``vitest.config.*``).  Output is parsed
  from JSON mode (``--reporter=json``).  # noqa: E402
- **jest** -- discovered from ``package.json`` (dependency ``jest``).
  Output is parsed from JSON mode (``--json``).
- **cargo** -- discovered from ``Cargo.toml``.  Output is parsed from
  Cargo's JSON messages (``--message-format=json``).

All subprocesses are launched with the same window-hiding flags as
the rest of the framework (Windows: ``CREATE_NO_WINDOW``,
Unix: ``start_new_session=True``) so running tests in the background
does not flash a console window in the desktop session.

Output schema
-------------
The tool returns a JSON string with the following fields:

::

    {
      "framework": "pytest",
      "passed": 12,
      "failed": 2,
      "skipped": 1,
      "errors": 0,
      "duration_s": 4.18,
      "tests": [
        {
          "name": "tests/test_foo.py::test_bar",
          "status": "failed" | "passed" | "skipped" | "error",
          "duration_s": 0.21,
          "file": "tests/test_foo.py",
          "line": 42,
          "message": "AssertionError: ...",
          "stdout": ""
        },
        ...
      ],
      "raw_output": "..."
    }
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
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


def _detect_framework(workspace: str, hint: str | None) -> str:
    """Return one of ``"pytest"``, ``"vitest"``, ``"jest"``, ``"cargo"``."""
    if hint and hint in ("pytest", "vitest", "jest", "cargo"):
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
        if "vitest" in deps:
            return "vitest"
        if "jest" in deps:
            return "jest"
    if (ws / "pyproject.toml").exists() or (ws / "pytest.ini").exists() or (ws / "setup.cfg").exists():  # noqa: E501
        return "pytest"
    # Fallback: if any *.py exists, use pytest; else if any *.js/*.ts, jest.
    py_files = list(ws.rglob("*.py"))
    if py_files:
        return "pytest"
    js_files = list(ws.rglob("*.js")) + list(ws.rglob("*.ts"))
    if js_files:
        return "jest"
    return "pytest"


# ---------------------------------------------------------------------------
# Per-framework runners
# ---------------------------------------------------------------------------


@dataclass
class TestRecord:
    name: str
    status: str
    duration_s: float
    file: str = ""
    line: int = 0
    message: str = ""
    stdout: str = ""


@dataclass
class TestReport:
    framework: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_s: float = 0.0
    tests: list[TestRecord] = field(default_factory=list)
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


async def _run_pytest(workspace: str, test_filter: str | None, timeout: float) -> TestReport:
    report = TestReport(framework="pytest")
    junit_path = Path(workspace) / ".enta" / "pytest-junit.xml"
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "--tb=short",
        "--junit-xml",
        str(junit_path),
        "--color=no",
    ]
    if test_filter:
        cmd.extend(["-k", test_filter])
    t0 = time.time()
    _code, out, err = await _exec(cmd, workspace, timeout)
    report.duration_s = round(time.time() - t0, 3)
    report.raw_output = (out + "\n" + err).strip()
    if not junit_path.exists():
        # pytest failed before producing a JUnit report; record raw text.
        report.errors = 1
        return report
    try:
        tree = ET.parse(junit_path)
    except ET.ParseError:
        return report
    root = tree.getroot()
    for tc in root.iter("testcase"):
        classname = tc.get("classname", "")
        name = tc.get("name", "")
        file_path = tc.get("file", "")
        line_no = int(tc.get("line") or 0)
        time_s = float(tc.get("time") or 0.0)
        record = TestRecord(
            name=f"{classname}::{name}" if classname else name,
            status="passed",
            duration_s=round(time_s, 4),
            file=file_path,
            line=line_no,
        )
        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")
        if failure is not None:
            record.status = "failed"
            record.message = (failure.get("message") or "") + "\n" + (failure.text or "")
            record.message = record.message.strip()[:2000]
            report.failed += 1
        elif error is not None:
            record.status = "error"
            record.message = (error.get("message") or "") + "\n" + (error.text or "")
            record.message = record.message.strip()[:2000]
            report.errors += 1
        elif skipped is not None:
            record.status = "skipped"
            record.message = (skipped.get("message") or "").strip()
            report.skipped += 1
        else:
            report.passed += 1
        report.tests.append(record)
    return report


async def _run_npm_test(
    workspace: str,
    framework: str,
    test_filter: str | None,
    timeout: float,
) -> TestReport:
    """Shared implementation for jest and vitest (both speak JSON)."""
    report = TestReport(framework=framework)
    npx = shutil.which("npx") or shutil.which("npm")
    if npx is None:
        report.errors = 1
        report.raw_output = "npx/npm not found in PATH"
        return report
    bin_name = "vitest" if framework == "vitest" else "jest"
    cmd: list[str] = [npx, "--no-install", bin_name, "run", "--reporter=json", "--outputFile=/dev/null"]  # noqa: E501
    # vitest honours ``--reporter=json`` and writes JSON to stdout
    # (no outputFile); jest writes via ``--json``.
    if framework == "jest":
        cmd = [npx, "--no-install", "jest", "--json"]
    if test_filter:
        if framework == "vitest":
            cmd.extend(["-t", test_filter])
        else:
            cmd.extend(["-t", test_filter])
    t0 = time.time()
    _code, out, err = await _exec(cmd, workspace, timeout)
    report.duration_s = round(time.time() - t0, 3)
    report.raw_output = (out + "\n" + err).strip()
    # The JSON document may be the entire stdout; the last JSON
    # object in the output is the actual report (some runners prefix
    # warnings).  Try strict parse, then fall back to a brace scan.
    payload: Any = None
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(out):
            brace = out.find("{", idx)
            if brace == -1:
                break
            try:
                obj, end = decoder.raw_decode(out[brace:])
                payload = obj
                idx = brace + end
            except json.JSONDecodeError:
                idx = brace + 1
    if not isinstance(payload, dict):
        # The runner may have written JSON to a file; nothing to do.
        return report
    num_total = int(payload.get("numTotalTests") or payload.get("numTotal") or 0)
    num_passed = int(payload.get("numPassedTests") or payload.get("numPassed") or 0)
    num_failed = int(payload.get("numFailedTests") or payload.get("numFailed") or 0)
    num_skipped = int(
        payload.get("numPendingTests")
        or payload.get("numPending")
        or payload.get("numSkipped")
        or 0
    )
    num_todo = int(payload.get("numTodoTests") or payload.get("numTodo") or 0)
    report.passed = num_passed
    report.failed = num_failed
    report.skipped = num_skipped + num_todo
    report.errors = 0
    # Build per-test records if available
    test_results = payload.get("testResults") or []
    for tr in test_results:
        file_path = tr.get("name") or tr.get("file") or ""
        for inner in tr.get("testResults", []) or tr.get("assertionResults", []) or []:
            full_name = inner.get("fullName") or inner.get("title") or inner.get("name") or ""
            status_raw = (inner.get("status") or "").lower()
            duration = float(inner.get("duration") or 0.0) / 1000.0
            message = ""
            if status_raw == "passed":
                status = "passed"
            elif status_raw in ("failed", "fail"):
                status = "failed"
                message_lines = inner.get("failureMessages") or []
                message = "\n".join(message_lines)[:2000]
            elif status_raw in ("skipped", "pending", "todo", "disabled"):
                status = "skipped"
            else:
                status = status_raw or "unknown"
            rec = TestRecord(
                name=full_name,
                status=status,
                duration_s=round(duration, 4),
                file=file_path,
                message=message,
            )
            report.tests.append(rec)
    if not test_results:
        # We still have the aggregate counts; emit a single roll-up.
        report.tests.append(
            TestRecord(
                name=f"<{framework} aggregate>",
                status="passed" if report.failed == 0 and report.errors == 0 else "failed",
                duration_s=report.duration_s,
            )
        )
    # ``numTotalTests`` is the most reliable total; if it is
    # inconsistent with the per-test results we still trust the
    # aggregate counts reported by the runner.
    if num_total and len(report.tests) == 0:
        report.tests.append(
            TestRecord(
                name=f"<{framework} aggregate>",
                status="passed" if report.failed == 0 else "failed",
                duration_s=report.duration_s,
            )
        )
    return report


async def _run_cargo(workspace: str, test_filter: str | None, timeout: float) -> TestReport:
    report = TestReport(framework="cargo")
    cmd = ["cargo", "test", "--no-fail-fast", "--message-format=json"]
    if test_filter:
        cmd.append(test_filter)
    t0 = time.time()
    _code, out, err = await _exec(cmd, workspace, timeout)
    report.duration_s = round(time.time() - t0, 3)
    report.raw_output = (out + "\n" + err).strip()
    # Cargo emits one JSON object per line.  Collect ``test`` events
    # which carry a ``test`` name and a status (``ok``, ``failed``,
    # ``ignored``).  Errors are also surfaced as ``exec`` events.
    suite: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        msg = obj.get("message") or {}
        kind = obj.get("reason") or msg.get("reason") or ""
        if kind == "compiler-message" and msg.get("message"):
            failures.append(
                {
                    "name": "compile",
                    "status": "error",
                    "message": str(msg.get("message"))[:2000],
                }
            )
        if "test" in msg and isinstance(msg["test"], str):
            name = msg["test"]
            event = msg.get("event") or ("ok" if "passed" not in obj else "ok")
            suite.setdefault(name, []).append({"event": event, "exec": obj})
    # Build per-test records from the latest event per name.
    for name, events in suite.items():
        latest = events[-1]
        event = latest["event"]
        status = "passed" if event == "ok" else ("skipped" if event == "ignored" else "failed")
        record = TestRecord(
            name=name,
            status=status,
            duration_s=0.0,
        )
        if status == "failed":
            # Find the matching failure message (the next ``test``
            # event with a ``stdout`` containing ``FAILED`` is rare;
            # fall back to the ``exec.stderr`` if present).
            exec_obj = latest.get("exec") or {}
            rendered = exec_obj.get("message", {}).get("rendered")
            if rendered:
                record.message = str(rendered)[:2000]
            report.failed += 1
        elif status == "skipped":
            report.skipped += 1
        else:
            report.passed += 1
        report.tests.append(record)
    # Attach any compile errors as test records too.
    for f in failures:
        report.tests.append(
            TestRecord(
                name=f.get("name", "compile"),
                status="error",
                duration_s=0.0,
                message=f.get("message", ""),
            )
        )
        report.errors += 1
    return report


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------


async def _test_run_execute(**kwargs: Any) -> str:
    workspace = str(kwargs.get("workspace") or kwargs.get("path") or "").strip()
    if not workspace:
        return "Error: workspace is required"
    ws = Path(workspace)
    if not ws.is_dir():
        return f"Error: workspace does not exist or is not a directory: {workspace}"
    framework = _detect_framework(workspace, kwargs.get("framework"))
    test_filter = kwargs.get("filter") or kwargs.get("test_filter")
    max_duration = float(kwargs.get("max_duration") or kwargs.get("timeout") or 120.0)
    if max_duration <= 0:
        max_duration = 120.0

    if framework == "pytest":
        report = await _run_pytest(workspace, test_filter, max_duration)
    elif framework in ("vitest", "jest"):
        report = await _run_npm_test(workspace, framework, test_filter, max_duration)
    elif framework == "cargo":
        report = await _run_cargo(workspace, test_filter, max_duration)
    else:
        return f"Error: unsupported framework: {framework}"

    payload = report.to_dict()
    payload["workspace"] = workspace
    payload["filter"] = test_filter
    return json.dumps(payload, ensure_ascii=False, indent=2)


EncreTestRunTool = build_tool(
    name="test_run",
    description=(
        "Run the project's test suite and return a structured JSON "
        "report.  Auto-detects pytest, vitest, jest and cargo test.  "
        "Each result includes per-test status (passed / failed / "
        "skipped / error), duration, file location, and a truncated "
        "failure message -- enough for the agent to decide which tests "
        "to fix and where.  Use the 'filter' argument to scope to a "
        "single test (pytest ``-k``, jest/vitest ``-t``, cargo test "
        "name)."
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
            "framework": {
                "type": "string",
                "enum": ["pytest", "vitest", "jest", "cargo"],
                "description": (
                    "Force a specific test framework.  When omitted the "
                    "tool auto-detects from project files."
                ),
            },
            "filter": {
                "type": "string",
                "description": (
                    "Optional test filter -- pytest ``-k`` expression, "
                    "jest/vitest ``-t`` substring, or a cargo test name."
                ),
            },
            "max_duration": {
                "type": "number",
                "description": (
                    "Maximum wall-clock duration in seconds.  Defaults to "
                    "120.  The subprocess is killed if it overruns."
                ),
            },
        },
    },
    execute=_test_run_execute,
    intents=["coding", "data", "general"],
)
