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

"""Apply a unified diff across multiple files.

Supports:
- ``diff --git a/path b/path`` headers
- ``--- a/path`` / ``+++ b/path`` filename markers (including ``/dev/null``
  for create/delete operations)
- ``rename from`` / ``rename to`` headers
- ``new file mode`` / ``deleted file mode``
- ``@@ -l,c +l,c @@`` hunk headers
- ``\\ No newline at end of file`` markers
- Fuzzy hunk placement: if the recorded line number doesn't match, search
  ±200 lines for the context.

Returns a JSON-encoded report listing each file touched and the result.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from encre.tools.base import build_tool


# ──────────────────────────────────────────────────────────────────────
# Diff parsing
# ──────────────────────────────────────────────────────────────────────


_HUNK_RE = re.compile(
    r"^@@\s+-(?P<old_start>\d+)(?:,(?P<old_count>\d+))?"
    r"\s+\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))?\s+@@"
)


@dataclass
class _Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)  # raw lines with prefix
    no_newline_at_eof: bool = False


@dataclass
class _FileDiff:
    old_path: str
    new_path: str
    is_new: bool = False
    is_deleted: bool = False
    is_rename: bool = False
    hunks: list[_Hunk] = field(default_factory=list)


def _strip_ab_prefix(p: str) -> str:
    if p.startswith("a/") or p.startswith("b/"):
        return p[2:]
    return p


def _parse_patch(patch: str) -> list[_FileDiff]:
    lines = patch.splitlines(keepends=False)
    files: list[_FileDiff] = []
    i = 0
    n = len(lines)
    cur: _FileDiff | None = None

    while i < n:
        line = lines[i]

        if line.startswith("diff --git "):
            if cur is not None:
                files.append(cur)
            # Extract paths from the header just to seed something; the real
            # paths come from the --- / +++ markers below.
            parts = line.split()
            old_p = _strip_ab_prefix(parts[2]) if len(parts) > 2 else ""
            new_p = _strip_ab_prefix(parts[3]) if len(parts) > 3 else old_p
            cur = _FileDiff(old_path=old_p, new_path=new_p)
            i += 1
            continue

        if line.startswith("rename from "):
            if cur is None:
                cur = _FileDiff(old_path="", new_path="")
            cur.is_rename = True
            cur.old_path = line[len("rename from "):].strip()
            i += 1
            continue

        if line.startswith("rename to "):
            if cur is None:
                cur = _FileDiff(old_path="", new_path="")
            cur.is_rename = True
            cur.new_path = line[len("rename to "):].strip()
            i += 1
            continue

        if line.startswith("new file mode"):
            if cur is not None:
                cur.is_new = True
            i += 1
            continue

        if line.startswith("deleted file mode"):
            if cur is not None:
                cur.is_deleted = True
            i += 1
            continue

        if line.startswith("--- "):
            old_p = line[4:].strip()
            # Some patches use --- /dev/null for new files
            if cur is None:
                cur = _FileDiff(old_path="", new_path="")
            cur.old_path = "/dev/null" if old_p == "/dev/null" else _strip_ab_prefix(old_p)
            if cur.old_path == "/dev/null":
                cur.is_new = True
            i += 1
            continue

        if line.startswith("+++ "):
            new_p = line[4:].strip()
            if cur is None:
                cur = _FileDiff(old_path="", new_path="")
            cur.new_path = "/dev/null" if new_p == "/dev/null" else _strip_ab_prefix(new_p)
            if cur.new_path == "/dev/null":
                cur.is_deleted = True
            i += 1
            continue

        if line.startswith("@@"):
            m = _HUNK_RE.match(line)
            if m is None or cur is None:
                i += 1
                continue
            hunk = _Hunk(
                old_start=int(m.group("old_start")),
                old_count=int(m.group("old_count") or "1"),
                new_start=int(m.group("new_start")),
                new_count=int(m.group("new_count") or "1"),
            )
            i += 1
            while i < n:
                hl = lines[i]
                if hl.startswith("@@") or hl.startswith("diff --git ") or hl.startswith("--- "):
                    break
                if hl.startswith("\\ No newline at end of file"):
                    hunk.no_newline_at_eof = True
                    i += 1
                    continue
                # A hunk line is one of " ", "+", "-", or empty (which counts
                # as a context-empty-line in some emitters).
                if hl == "":
                    hunk.lines.append(" ")
                elif hl[0] in (" ", "+", "-"):
                    hunk.lines.append(hl)
                else:
                    # Junk between hunks (e.g. binary patch markers) — stop.
                    break
                i += 1
            cur.hunks.append(hunk)
            continue

        # Anything else (index lines, similarity indices, binary markers) skip.
        i += 1

    if cur is not None:
        files.append(cur)
    return files


def _count_patch_add_del(fd: _FileDiff) -> tuple[int, int]:
    add_count = 0
    del_count = 0
    for h in fd.hunks:
        for raw in h.lines:
            if not raw:
                continue
            if raw.startswith("+++") or raw.startswith("---"):
                continue
            if raw[0] == "+":
                add_count += 1
            elif raw[0] == "-":
                del_count += 1
    return add_count, del_count


# ──────────────────────────────────────────────────────────────────────
# Hunk application
# ──────────────────────────────────────────────────────────────────────


def _apply_hunks(original: str, hunks: list[_Hunk]) -> tuple[str, list[str]]:
    """Apply hunks to ``original`` text. Returns (new_text, hunk_notes)."""
    src_lines = original.splitlines(keepends=False)
    src_has_final_newline = original.endswith("\n")
    out_lines: list[str] = []
    src_cursor = 0  # next index in src_lines we haven't copied yet
    notes: list[str] = []

    for h_idx, h in enumerate(hunks):
        target = max(0, h.old_start - 1)  # convert to 0-based
        # Build expected pre-image (context + minus) and post-image (context + plus)
        pre: list[str] = []
        post: list[str] = []
        for raw in h.lines:
            tag = raw[0] if raw else " "
            body = raw[1:] if raw else ""
            if tag == " ":
                pre.append(body)
                post.append(body)
            elif tag == "-":
                pre.append(body)
            elif tag == "+":
                post.append(body)

        # Locate the actual position
        pos = _locate(src_lines, pre, target, src_cursor)
        if pos is None:
            raise RuntimeError(
                f"hunk #{h_idx + 1} could not be applied (context not found near line {h.old_start})"
            )

        if pos < src_cursor:
            raise RuntimeError(
                f"hunk #{h_idx + 1} is out of order (would rewind from line {src_cursor} to {pos})"
            )

        # Copy unchanged lines between src_cursor and pos
        out_lines.extend(src_lines[src_cursor:pos])
        # Splice in post-image
        out_lines.extend(post)
        # Advance src_cursor past the pre-image
        src_cursor = pos + len(pre)

        offset = pos - target
        if offset == 0:
            notes.append(f"hunk #{h_idx + 1}: applied at line {pos + 1}")
        else:
            notes.append(
                f"hunk #{h_idx + 1}: applied at line {pos + 1} (offset {offset:+d} from header)"
            )

    # Tail
    out_lines.extend(src_lines[src_cursor:])

    # Reconstruct newline policy
    result = "\n".join(out_lines)
    if src_has_final_newline and out_lines:
        result += "\n"
    elif out_lines and not src_has_final_newline:
        # If the original didn't end with newline and last hunk says it still doesn't, keep.
        # If any hunk explicitly added a final newline, joining lines doesn't add it back —
        # diff format only signals removal of final newline via the marker. Be conservative.
        pass
    return result, notes


def _locate(
    src: list[str],
    pre: list[str],
    target: int,
    min_pos: int,
) -> int | None:
    """Find ``pre`` in ``src`` starting somewhere reasonable around ``target``.

    Strategy: prefer the exact target. If that doesn't match, search ±N lines.
    """
    if not pre:
        # Pure-addition hunk: any position works in principle, but stick to
        # the target unless it's out of bounds.
        if target > len(src):
            return len(src)
        return max(target, min_pos)

    candidates = [target]
    for delta in range(1, 201):
        candidates.append(target + delta)
        candidates.append(target - delta)
    for pos in candidates:
        if pos < min_pos:
            continue
        if pos + len(pre) > len(src):
            continue
        if src[pos:pos + len(pre)] == pre:
            return pos
    # Last-ditch: tolerate trailing-whitespace differences
    for pos in candidates:
        if pos < min_pos:
            continue
        if pos + len(pre) > len(src):
            continue
        if all(src[pos + i].rstrip() == pre[i].rstrip() for i in range(len(pre))):
            return pos
    return None


def _resolve(root: str, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.normpath(os.path.join(root, rel))


# ──────────────────────────────────────────────────────────────────────
# Tool
# ──────────────────────────────────────────────────────────────────────


async def _apply_patch_execute(**kwargs: Any) -> str:
    patch = kwargs.get("patch", "")
    if not patch:
        return "Error: patch is required"
    root = kwargs.get("root") or os.getcwd()
    dry_run = bool(kwargs.get("dry_run", False))

    try:
        files = _parse_patch(patch)
    except Exception as exc:
        return f"Error parsing patch: {exc}"

    if not files:
        return "Error: no file diffs found in patch"

    report: list[dict[str, Any]] = []
    any_failure = False
    total_add = 0
    total_del = 0

    for fd in files:
        entry: dict[str, Any] = {
            "old_path": fd.old_path,
            "new_path": fd.new_path,
        }
        file_add, file_del = _count_patch_add_del(fd)
        entry["insertions"] = file_add
        entry["deletions"] = file_del
        try:
            if fd.is_deleted or fd.new_path == "/dev/null":
                abs_path = _resolve(root, fd.old_path)
                if not dry_run:
                    if os.path.exists(abs_path):
                        os.remove(abs_path)
                entry["action"] = "delete"
                entry["status"] = "ok"

            elif fd.is_new or fd.old_path == "/dev/null":
                abs_path = _resolve(root, fd.new_path)
                new_text, notes = _apply_hunks("", fd.hunks)
                if not dry_run:
                    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
                    with open(abs_path, "w", encoding="utf-8", newline="") as fh:
                        fh.write(new_text)
                entry["action"] = "create"
                entry["status"] = "ok"
                entry["hunks"] = notes

            elif fd.is_rename and not fd.hunks:
                src_path = _resolve(root, fd.old_path)
                dst_path = _resolve(root, fd.new_path)
                if not dry_run:
                    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
                    os.replace(src_path, dst_path)
                entry["action"] = "rename"
                entry["status"] = "ok"

            else:
                src_path = _resolve(root, fd.old_path)
                dst_path = _resolve(root, fd.new_path)
                with open(src_path, "r", encoding="utf-8", newline="") as fh:
                    original = fh.read()
                new_text, notes = _apply_hunks(original, fd.hunks)
                if not dry_run:
                    if fd.is_rename and src_path != dst_path:
                        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
                        with open(dst_path, "w", encoding="utf-8", newline="") as fh:
                            fh.write(new_text)
                        if os.path.exists(src_path) and os.path.abspath(src_path) != os.path.abspath(dst_path):
                            os.remove(src_path)
                    else:
                        with open(dst_path, "w", encoding="utf-8", newline="") as fh:
                            fh.write(new_text)
                entry["action"] = "modify" if not fd.is_rename else "rename+modify"
                entry["status"] = "ok"
                entry["hunks"] = notes
        except Exception as exc:
            any_failure = True
            entry["status"] = "error"
            entry["error"] = str(exc)

        if entry.get("status") == "ok":
            total_add += file_add
            total_del += file_del
        report.append(entry)

    result = {
        "applied": not any_failure,
        "dry_run": dry_run,
        "files": report,
    }
    summary = f"{total_add} insertions(+), {total_del} deletions(-)"
    return f"{summary}\n{json.dumps(result, ensure_ascii=False, indent=2)}"


EncreApplyPatchTool = build_tool(
    name="apply_patch",
    description=(
        "Apply a unified diff (git-style) to the working tree. Supports "
        "multi-file patches, new file creation, file deletion, renames, "
        "and fuzzy hunk placement when line numbers drift by up to 200 "
        "lines. Returns a JSON report listing each file outcome."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": "The unified diff text to apply.",
            },
            "root": {
                "type": "string",
                "description": (
                    "Root directory the patch paths are relative to "
                    "(default: current working directory)."
                ),
            },
            "dry_run": {
                "type": "boolean",
                "description": "Parse and check the patch without writing any files.",
            },
        },
        "required": ["patch"],
    },
    execute=_apply_patch_execute,
    intents=["general", "coding"],
)
