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



import re
from typing import Any

from enta.native import compute_diff as _native_diff
from enta.native import read_file as _native_read
from enta.native import write_file as _native_write
from enta.tools.base import build_tool
from enta.tools.builtin._sandbox import remap_tool_path

_WS_RE = re.compile(r"[ \t]+")
_DIFF_ADD_RE = re.compile(r"^\+(?!\+\+)", re.MULTILINE)
_DIFF_DEL_RE = re.compile(r"^-(?!--)", re.MULTILINE)


def _normalize_ws(text: str) -> str:
    """Collapse runs of horizontal whitespace, keeping line structure."""
    return "\n".join(_WS_RE.sub(" ", line).rstrip() for line in text.split("\n"))


def _find_with_fallback(haystack: str, needle: str) -> tuple[int, int, str] | None:
    """Locate ``needle`` in ``haystack``.

    Returns ``(start, end, matched_substring)`` of the first match, or None.

    Strategy:
    1. Exact match.
    2. Whitespace-normalized match -- the indices and the *actual* matched
       substring in the original haystack are recovered so the caller can
       splice with original indentation preserved.
    """
    idx = haystack.find(needle)
    if idx != -1:
        return idx, idx + len(needle), needle

    norm_hay = _normalize_ws(haystack)
    norm_needle = _normalize_ws(needle)
    if not norm_needle:
        return None

    n_idx = norm_hay.find(norm_needle)
    if n_idx == -1:
        return None

    # Map normalized index back to original. Walk through original, comparing
    # against normalized cursor.
    orig_start = _map_norm_to_orig(haystack, n_idx)
    orig_end = _map_norm_to_orig(haystack, n_idx + len(norm_needle))
    if orig_start is None or orig_end is None or orig_end <= orig_start:
        return None
    return orig_start, orig_end, haystack[orig_start:orig_end]


def _map_norm_to_orig(orig: str, norm_pos: int) -> int | None:
    """Given a normalized-string offset, find the corresponding offset in
    the original. ``norm_pos`` counts post-normalization characters."""
    cur_norm = 0
    i = 0
    while i < len(orig):
        if cur_norm == norm_pos:
            return i
        ch = orig[i]
        if ch in (" ", "\t"):
            j = i
            while j < len(orig) and orig[j] in (" ", "\t"):
                j += 1
            cur_norm += 1  # collapsed run -> single space
            i = j
            continue
        if ch == "\n":
            # rstrip means trailing spaces before \n don't appear; \n itself stays
            cur_norm += 1
            i += 1
            continue
        cur_norm += 1
        i += 1
    if cur_norm == norm_pos:
        return len(orig)
    return None


def _apply_one_edit(
    content: str,
    old_str: str,
    new_str: str,
    replace_all: bool,
) -> tuple[str, int, str]:
    """Apply one edit. Returns ``(new_content, count, note)``.

    ``count`` is the number of replacements performed.
    Raises ``ValueError`` on failure cases the caller cannot recover from.
    """
    if not old_str:
        raise ValueError("old_str must not be empty")

    if replace_all:
        if old_str not in content:
            # try whitespace fallback for replace_all by repeatedly locating
            count = 0
            cursor = 0
            buf: list[str] = []
            while cursor < len(content):
                hit = _find_with_fallback(content[cursor:], old_str)
                if hit is None:
                    buf.append(content[cursor:])
                    break
                s, e, _ = hit
                buf.append(content[cursor:cursor + s])
                buf.append(new_str)
                cursor += e
                count += 1
            if count == 0:
                raise ValueError("old_str not found (replace_all, even with whitespace tolerance)")
            return "".join(buf), count, "replace_all+ws_fallback"
        # fast path
        count = content.count(old_str)
        return content.replace(old_str, new_str), count, "replace_all"

    # single-replacement path
    exact_count = content.count(old_str)
    if exact_count == 1:
        return content.replace(old_str, new_str, 1), 1, "exact"
    if exact_count > 1:
        raise ValueError(
            f"old_str matched {exact_count} times. Pass replace_all=true to apply "
            "everywhere, or add more surrounding context to make the match unique."
        )

    # exact_count == 0 -> try whitespace fallback
    hit = _find_with_fallback(content, old_str)
    if hit is None:
        raise ValueError(
            "old_str not found in file (also tried whitespace-normalized match)"
        )
    s, e, _matched = hit
    # Ensure the ws-tolerant match is unique too -- search again past `e`
    second = _find_with_fallback(content[e:], old_str)
    if second is not None:
        raise ValueError(
            "Whitespace-tolerant match is ambiguous. Add more surrounding context "
            "or set replace_all=true."
        )
    return content[:s] + new_str + content[e:], 1, "ws_fallback"


async def _file_edit_execute(**kwargs: Any) -> str:
    file_path = kwargs.get("file_path", "")
    if not file_path:
        return "Error: 缺少 file_path 参数 (file_path is required)"
    file_path = remap_tool_path(file_path)

    dry_run = bool(kwargs.get("dry_run", False))
    tool_call_id = str(kwargs.get("tool_call_id", ""))

    edits_param = kwargs.get("edits")
    if edits_param:
        if not isinstance(edits_param, list):
            return "Error: 'edits' 必须是数组 ('edits' must be an array)"
        edits = []
        for i, e in enumerate(edits_param):
            if not isinstance(e, dict) or "old_str" not in e or "new_str" not in e:
                return f"Error: edits[{i}] 必须包含 'old_str' 和 'new_str' (must have 'old_str' and 'new_str')"  # noqa: E501
            edits.append(
                (
                    str(e["old_str"]),
                    str(e["new_str"]),
                    bool(e.get("replace_all", False)),
                )
            )
    else:
        old_str = kwargs.get("old_str")
        new_str = kwargs.get("new_str")
        if old_str is None or new_str is None:
            return "Error: 请提供 'edits' 或同时提供 'old_str' 和 'new_str' (provide either 'edits' or both 'old_str' and 'new_str')"  # noqa: E501
        edits = [(str(old_str), str(new_str), bool(kwargs.get("replace_all", False)))]

    try:
        original = _native_read(file_path, 0, 0)
    except FileNotFoundError:
        return f"Error: 文件未找到 (File not found): {file_path}"
    except PermissionError:
        return f"Error: 权限不足，无法读取 (Permission denied): {file_path}"
    except Exception as e:
        return f"Error: 读取文件出错 (Error reading file): {e}"

    content = original
    report: list[str] = []
    for i, (old_str, new_str, replace_all) in enumerate(edits):
        try:
            content, count, note = _apply_one_edit(content, old_str, new_str, replace_all)
        except ValueError as exc:
            return f"Error: 编辑 #{i + 1} 失败 (Edit #{i + 1} failed): {exc}"
        report.append(f"edit #{i + 1}: {count} replacement(s) ({note})")

    if content == original:
        return "No-op: edits produced no change."

    diff_text = _native_diff(original, content)
    add_count = len(_DIFF_ADD_RE.findall(diff_text))
    del_count = len(_DIFF_DEL_RE.findall(diff_text))

    if dry_run:
        # When ``dry_run`` is true the tool computes the diff but does
        # **not** write the file.  Instead it returns a JSON-encoded
        # ``EditProposal`` that the loop and the desktop UI use to
        # render an inline diff and let the user accept or reject the
        # change.  ``apply_edit`` is exposed by the desktop IPC layer
        # so a UI button can commit the change without re-running the
        # tool.
        import json

        proposal = {
            "kind": "edit_proposal",
            "tool_call_id": tool_call_id,
            "file_path": file_path,
            "diff_text": diff_text,
            "original": original,
            "proposed": content,
            "added": add_count,
            "removed": del_count,
            "summary": "; ".join(report),
        }
        return "```json\n" + json.dumps(proposal, ensure_ascii=False) + "\n```"

    try:
        _native_write(file_path, content)
    except PermissionError:
        return f"Error: 权限不足，无法写入 (Permission denied): {file_path}"
    except Exception as e:
        return f"Error: 写入文件出错 (Error writing file): {e}"

    return (
        f"Applied {len(edits)} edit(s) to {file_path}.\n"
        + f"{add_count} insertions(+), {del_count} deletions(-)\n"
        + "\n".join(report)
        + f"\n```diff\n{diff_text}\n```"
    )


EncreFileEditTool = build_tool(
    name="file_edit",
    description=(
        "Apply one or more search-and-replace edits to an existing file. "
        "Supports unique single-match edits (default), replace_all mode, "
        "multi-hunk edits in one call via the 'edits' array, and a "
        "whitespace-normalized fallback when the exact text isn't found. "
        "Each edit is applied sequentially in the order provided."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit (filename, relative path, or absolute path within the workspace)",  # noqa: E501
            },
            "old_str": {
                "type": "string",
                "description": (
                    "The exact text to search for. Required for single-edit "
                    "mode. Ignored if 'edits' is provided."
                ),
            },
            "new_str": {
                "type": "string",
                "description": (
                    "The replacement text. Required for single-edit mode. "
                    "Ignored if 'edits' is provided."
                ),
            },
            "replace_all": {
                "type": "boolean",
                "description": (
                    "When true, replace every occurrence of old_str. When "
                    "false (default), the match must be unique."
                ),
            },
            "edits": {
                "type": "array",
                "description": (
                    "List of edit hunks to apply sequentially. Each item is "
                    "an object with 'old_str', 'new_str', and optional "
                    "'replace_all'. When provided, 'old_str'/'new_str' at "
                    "the top level are ignored."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["old_str", "new_str"],
                },
            },
            "dry_run": {
                "type": "boolean",
                "default": False,
                "description": (
                    "When true, compute the diff but do NOT write the "
                    "file.  The tool returns a JSON-encoded EditProposal "
                    "that the desktop UI renders as an inline diff with "
                    "Accept / Reject buttons.  This is what gives Codex-"
                    "class review-before-apply editing."
                ),
            },
            "tool_call_id": {
                "type": "string",
                "description": (
                    "Optional originating tool-call id.  When provided "
                    "together with dry_run=true, the returned proposal "
                    "carries this id so the UI can route the user's "
                    "accept/reject decision back to the right pending "
                    "request."
                ),
            },
        },
        "required": ["file_path"],
    },
    execute=_file_edit_execute,
    intents=["general", "coding", "data"],
)
