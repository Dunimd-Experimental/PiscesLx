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



"""High-fidelity grep that uses ripgrep when available, with a real Python
fallback covering the same flag set. The flags mirror the ripgrep CLI so the
model can rely on familiar semantics regardless of which backend runs."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
from typing import Any

from enta.tools.base import build_tool

_TYPE_GLOBS: dict[str, list[str]] = {
    "py": ["*.py"],
    "python": ["*.py"],
    "js": ["*.js", "*.jsx", "*.mjs", "*.cjs"],
    "ts": ["*.ts", "*.tsx", "*.mts", "*.cts"],
    "json": ["*.json"],
    "md": ["*.md", "*.markdown"],
    "css": ["*.css"],
    "html": ["*.html", "*.htm"],
    "yaml": ["*.yml", "*.yaml"],
    "rust": ["*.rs"],
    "go": ["*.go"],
    "java": ["*.java"],
    "c": ["*.c", "*.h"],
    "cpp": ["*.cc", "*.cpp", "*.cxx", "*.hpp", "*.hh"],
    "sql": ["*.sql"],
}


async def _grep_execute(**kwargs: Any) -> str:
    pattern = str(kwargs.get("pattern", ""))
    path = str(kwargs.get("path") or ".")
    glob_filter = str(kwargs.get("glob") or "")
    type_filter = str(kwargs.get("type") or "")
    output_mode = str(kwargs.get("output_mode") or "content")
    case_insensitive = bool(kwargs.get("-i") or kwargs.get("case_insensitive"))
    show_numbers = bool(kwargs.get("-n", True))
    context_after = int(kwargs.get("-A") or kwargs.get("after_context") or 0)
    context_before = int(kwargs.get("-B") or kwargs.get("before_context") or 0)
    context = kwargs.get("-C") or kwargs.get("context")
    if context is not None:
        context_after = context_before = int(context)
    head_limit_raw = kwargs.get("head_limit")
    head_limit = int(head_limit_raw) if head_limit_raw not in (None, "") else None
    multiline = bool(kwargs.get("multiline"))

    if not pattern:
        return "Error: pattern is required"

    rg = shutil.which("rg")
    if rg and not multiline:
        try:
            return await _run_rg(
                rg, pattern, path, output_mode,
                case_insensitive, show_numbers,
                context_after, context_before,
                glob_filter, type_filter, head_limit,
            )
        except Exception:
            pass

    return _run_python(
        pattern, path, output_mode,
        case_insensitive, show_numbers,
        context_after, context_before,
        glob_filter, type_filter, head_limit,
        multiline,
    )


# ------------------------------------------------------------------
# ripgrep backend
# ------------------------------------------------------------------

async def _run_rg(
    rg: str,
    pattern: str,
    path: str,
    output_mode: str,
    case_insensitive: bool,
    show_numbers: bool,
    context_after: int,
    context_before: int,
    glob_filter: str,
    type_filter: str,
    head_limit: int | None,
) -> str:
    args: list[str] = [rg, "--color=never"]
    if case_insensitive:
        args.append("-i")
    if output_mode == "files_with_matches":
        args.append("-l")
    elif output_mode == "count":
        args.append("-c")
    else:
        if show_numbers:
            args.append("-n")
        if context_after:
            args.extend(["-A", str(context_after)])
        if context_before:
            args.extend(["-B", str(context_before)])
    if glob_filter:
        args.extend(["-g", glob_filter])
    if type_filter and type_filter in _TYPE_GLOBS:
        # Use --type-add to ensure exact alias coverage even when rg
        # doesn't ship that alias.
        globs = ",".join(_TYPE_GLOBS[type_filter])
        args.extend(["--type-add", f"enta:{globs}", "--type", "enta"])
    args.extend(["--", pattern, path])

    from enta.tools.builtin._suppress_window import hidden_subprocess_kwargs
    kwargs = hidden_subprocess_kwargs()
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode not in (0, 1):  # 1 = no matches in rg
        err = stderr.decode("utf-8", errors="replace").strip()
        if err:
            raise RuntimeError(f"ripgrep failed: {err}")
    text = stdout.decode("utf-8", errors="replace")
    if head_limit is not None and head_limit > 0:
        lines = text.splitlines()
        if len(lines) > head_limit:
            lines = [*lines[:head_limit], f"... ({len(lines) - head_limit} more line(s) truncated)"]
            text = "\n".join(lines)
    return text if text else "(no matches)"


# ------------------------------------------------------------------
# Python backend
# ------------------------------------------------------------------

def _run_python(
    pattern: str,
    path: str,
    output_mode: str,
    case_insensitive: bool,
    show_numbers: bool,
    context_after: int,
    context_before: int,
    glob_filter: str,
    type_filter: str,
    head_limit: int | None,
    multiline: bool,
) -> str:
    flags = re.MULTILINE
    if case_insensitive:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.DOTALL
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return f"Error: invalid regex: {exc}"

    files = list(_iter_files(path, glob_filter, type_filter))

    if output_mode == "files_with_matches":
        return _py_files_with_matches(regex, files, multiline, head_limit)
    if output_mode == "count":
        return _py_count(regex, files, multiline, head_limit)
    return _py_content(
        regex, files, show_numbers,
        context_after, context_before, multiline, head_limit,
    )


def _iter_files(root: str, glob_filter: str, type_filter: str):
    if os.path.isfile(root):
        if _file_matches(root, glob_filter, type_filter):
            yield root
        return
    if not os.path.isdir(root):
        return
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                 "dist", "build", "target", ".mypy_cache", ".pytest_cache",
                 ".idea", ".vscode"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for f in filenames:
            full = os.path.join(dirpath, f)
            if _file_matches(full, glob_filter, type_filter):
                yield full


def _file_matches(path: str, glob_filter: str, type_filter: str) -> bool:
    name = os.path.basename(path)
    if glob_filter:
        # Support both "*.py" and "**/*.py" style globs
        if "/" in glob_filter or "\\" in glob_filter:
            norm_path = path.replace("\\", "/")
            if not (fnmatch.fnmatch(norm_path, glob_filter)
                    or fnmatch.fnmatch(name, glob_filter)):
                return False
        else:
            if not fnmatch.fnmatch(name, glob_filter):
                return False
    if type_filter:
        globs = _TYPE_GLOBS.get(type_filter)
        if globs is None:
            return False
        if not any(fnmatch.fnmatch(name, g) for g in globs):
            return False
    return True


def _read_text(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8 * 1024 * 1024)  # cap at 8 MiB per file
    except (OSError, PermissionError):
        return None
    # Quick binary sniff
    if b"\x00" in raw[:
        8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return None


def _py_files_with_matches(
    regex: re.Pattern[str],
    files: list[str],
    multiline: bool,
    head_limit: int | None,
) -> str:
    hits: list[str] = []
    for f in files:
        text = _read_text(f)
        if text is None:
            continue
        if multiline:
            if regex.search(text):
                hits.append(f)
        else:
            if any(regex.search(line) for line in text.splitlines()):
                hits.append(f)
        if head_limit is not None and len(hits) >= head_limit:
            break
    return "\n".join(hits) if hits else "(no matches)"


def _py_count(
    regex: re.Pattern[str],
    files: list[str],
    multiline: bool,
    head_limit: int | None,
) -> str:
    rows: list[str] = []
    for f in files:
        text = _read_text(f)
        if text is None:
            continue
        if multiline:
            count = len(regex.findall(text))
        else:
            count = sum(1 for line in text.splitlines() if regex.search(line))
        if count:
            rows.append(f"{f}:{count}")
        if head_limit is not None and len(rows) >= head_limit:
            break
    return "\n".join(rows) if rows else "(no matches)"


def _py_content(
    regex: re.Pattern[str],
    files: list[str],
    show_numbers: bool,
    context_after: int,
    context_before: int,
    multiline: bool,
    head_limit: int | None,
) -> str:
    out_lines: list[str] = []
    for f in files:
        text = _read_text(f)
        if text is None:
            continue
        if multiline:
            # In multiline mode we emit each match with its byte/char position.
            for m in regex.finditer(text):
                start_line = text.count("\n", 0, m.start()) + 1
                snippet = m.group(0)
                if show_numbers:
                    out_lines.append(f"{f}:{start_line}:{snippet}")
                else:
                    out_lines.append(f"{f}:{snippet}")
                if head_limit is not None and len(out_lines) >= head_limit:
                    break
        else:
            lines = text.splitlines()
            emitted_idxs: set[int] = set()
            for i, line in enumerate(lines):
                if not regex.search(line):
                    continue
                lo = max(0, i - context_before)
                hi = min(len(lines), i + context_after + 1)
                for j in range(lo, hi):
                    if j in emitted_idxs:
                        continue
                    emitted_idxs.add(j)
                    sep = "-" if j != i else ":"
                    if show_numbers:
                        out_lines.append(f"{f}{sep}{j + 1}{sep}{lines[j]}")
                    else:
                        out_lines.append(f"{f}{sep}{lines[j]}")
                if head_limit is not None and len(out_lines) >= head_limit:
                    break
        if head_limit is not None and len(out_lines) >= head_limit:
            break

    if not out_lines:
        return "(no matches)"
    if head_limit is not None and len(out_lines) > head_limit:
        out_lines = [*out_lines[:head_limit], f"... ({len(out_lines) - head_limit} more line(s) truncated)"]  # noqa: E501
    return "\n".join(out_lines)


EncreGrepTool = build_tool(
    name="grep",
    description=(
        "Search files for a regex pattern. Wraps ripgrep when available, "
        "with a full Python fallback. Supports context lines (-A/-B/-C), "
        "line numbers, multiline patterns, file-type filtering, glob filters, "
        "case-insensitive matching, head_limit, and three output modes.\n\n"
        "DO NOT use this tool for searching persistent memory -- use "
        "memory_search instead. DO NOT use this tool for searching the web "
        "-- use web_search instead."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regular expression pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search (default: current dir)",
            },
            "glob": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g. *.py, **/*.ts)",
            },
            "type": {
                "type": "string",
                "description": (
                    "File type alias (py, rust, go, js, ts, java, c, cpp, "
                    "html, css, json, yaml, md, sql, ...). Filters files like "
                    "ripgrep --type does."
                ),
            },
            "-i": {
                "type": "boolean",
                "description": "Case insensitive search",
            },
            "-n": {
                "type": "boolean",
                "description": "Include line numbers (default true for content mode)",
            },
            "-A": {
                "type": "integer",
                "description": "Lines of context to show after each match",
            },
            "-B": {
                "type": "integer",
                "description": "Lines of context to show before each match",
            },
            "-C": {
                "type": "integer",
                "description": "Lines of context to show before and after each match",
            },
            "multiline": {
                "type": "boolean",
                "description": "Enable multiline mode (. matches \\n, patterns can span lines)",
            },
            "head_limit": {
                "type": "integer",
                "description": "Cap the number of output lines (or files / counts)",
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": "Output mode (default: files_with_matches)",
            },
        },
        "required": ["pattern"],
    },
    execute=_grep_execute,
    intents=["general", "coding", "data"],
    is_concurrency_safe=lambda _: True,
)
