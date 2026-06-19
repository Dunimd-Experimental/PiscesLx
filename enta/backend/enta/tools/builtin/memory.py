#!/usr/bin/env python3

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



"""Memory tools -- persistent, file-based memory ledger for the model.

This is a self-contained replacement for the legacy ``EncreMemorySystem``
+ ``enta.crypto`` stack that lived under ``enta.memdir`` (removed during
the EnTA slim-down).  The implementation is a plain markdown-on-disk
store with a very small in-process search index.  It is intentionally
simple: it is the tool the model uses to remember things across turns
during adversarial training, not an encrypted persistence engine.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from enta.tools.base import build_tool

logger = logging.getLogger(__name__)

# The store lives in a single per-process directory; the directory itself
# is created on first write.  No external service is required.
_DEFAULT_MEMORY_DIR = os.path.join(
    os.path.expanduser("~"), ".piscesl1", "enta_memory"
)


def _memory_dir() -> str:
    return os.environ.get("PISCESL1_MEMORY_DIR", _DEFAULT_MEMORY_DIR)


def _sanitize_filename(name: str) -> str:
    slug = re.sub(r"[^\w\-]", "_", name.strip()).strip("_").lower()
    return slug if slug else "memory"


def _safe_path(filename: str) -> str:
    base = os.path.abspath(_memory_dir())
    target = os.path.abspath(os.path.join(base, filename))
    if not target.startswith(base + os.sep) and target != base:
        raise ValueError(f"unsafe memory path: {filename!r}")
    return target


def _list_memory_files() -> list[str]:
    base = _memory_dir()
    if not os.path.isdir(base):
        return []
    return sorted(
        f for f in os.listdir(base)
        if f.endswith(".md") and os.path.isfile(os.path.join(base, f))
    )


def _search_files(query: str, top_k: int) -> list[tuple[str, float, str]]:
    """Naive keyword scoring for memory_search.

    Score = number of query tokens that appear in the file.  This is not a
    semantic index, but it is honest, deterministic, and works without
    any extra dependency.
    """
    tokens = [t for t in re.split(r"\W+", query.lower()) if t]
    if not tokens:
        return []

    scored: list[tuple[str, float, str]] = []
    for fname in _list_memory_files():
        path = os.path.join(_memory_dir(), fname)
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read().lower()
        except OSError:
            continue
        hits = sum(1 for t in tokens if t in content)
        if hits > 0:
            snippet = ""
            for line in content.splitlines():
                if any(t in line for t in tokens):
                    snippet = line.strip()[:200]
                    break
            scored.append((fname, float(hits), snippet))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


async def _memory_create_execute(**kwargs: Any) -> str:
    filename = kwargs.get("filename", "")
    content = kwargs.get("content", "")
    if not filename:
        return "Error: 'filename' is required."
    if not filename.endswith(".md"):
        filename += ".md"
    filename = _sanitize_filename(os.path.splitext(filename)[0]) + ".md"
    path = _safe_path(filename)

    if os.path.exists(path):
        return (
            f"Memory file '{filename}' already exists. "
            f"Use memory_update to modify it, or choose a different filename."
        )
    os.makedirs(_memory_dir(), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Memory '{filename}' created."


async def _memory_read_execute(**kwargs: Any) -> str:
    filename = kwargs.get("filename", "")
    if not filename:
        return "Error: 'filename' is required."
    if not filename.endswith(".md"):
        filename += ".md"
    filename = _sanitize_filename(os.path.splitext(filename)[0]) + ".md"
    path = _safe_path(filename)
    if not os.path.isfile(path):
        return f"Memory file '{filename}' not found."
    with open(path, encoding="utf-8") as f:
        return f.read()


async def _memory_update_execute(**kwargs: Any) -> str:
    filename = kwargs.get("filename", "")
    content = kwargs.get("content", "")
    if not filename:
        return "Error: 'filename' is required."
    if not filename.endswith(".md"):
        filename += ".md"
    filename = _sanitize_filename(os.path.splitext(filename)[0]) + ".md"
    path = _safe_path(filename)
    if not os.path.isfile(path):
        return f"Memory file '{filename}' does not exist. Use memory_create to create it."
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Memory '{filename}' updated."


async def _memory_delete_execute(**kwargs: Any) -> str:
    filename = kwargs.get("filename", "")
    if not filename:
        return "Error: 'filename' is required."
    if not filename.endswith(".md"):
        filename += ".md"
    filename = _sanitize_filename(os.path.splitext(filename)[0]) + ".md"
    path = _safe_path(filename)
    if not os.path.isfile(path):
        return f"Memory file '{filename}' not found."
    os.remove(path)
    return f"Memory '{filename}' deleted."


async def _memory_search_execute(**kwargs: Any) -> str:
    query = kwargs.get("query", "")
    top_k = int(kwargs.get("top_k", 5))
    if not query:
        return "Error: 'query' is required."

    hits = _search_files(query, top_k)
    if not hits:
        return f"No memories found matching '{query}'."

    lines: list[str] = [f"Memory search results for '{query}':", ""]
    for i, (fname, score, snippet) in enumerate(hits, 1):
        lines.append(f"{i}. **{fname}** (score: {score:.0f})")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines)


EncreMemoryCreateTool = build_tool(
    name="memory_create",
    description=(
        "Create a new persistent memory file.  Memories are stored as "
        "plain markdown files in the per-process memory directory and "
        "are searchable by memory_search."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The .md filename for the memory (e.g. 'user_preferences.md')",
            },
            "content": {
                "type": "string",
                "description": "Full markdown content with optional YAML frontmatter",
            },
        },
        "required": ["filename", "content"],
    },
    execute=_memory_create_execute,
    intents=["general", "coding"],
)

EncreMemoryReadTool = build_tool(
    name="memory_read",
    description="Read a memory file by filename. Returns the full content.",
    input_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The .md filename to read (e.g. 'user_preferences.md')",
            },
        },
        "required": ["filename"],
    },
    execute=_memory_read_execute,
    intents=["general", "coding", "research"],
    is_concurrency_safe=lambda _: True,
)

EncreMemoryUpdateTool = build_tool(
    name="memory_update",
    description="Update an existing memory file. Content is replaced entirely.",
    input_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The .md filename to update",
            },
            "content": {
                "type": "string",
                "description": "The new full content for the memory file",
            },
        },
        "required": ["filename", "content"],
    },
    execute=_memory_update_execute,
    intents=["general", "coding"],
)

EncreMemoryDeleteTool = build_tool(
    name="memory_delete",
    description="Delete a memory file permanently. This cannot be undone.",
    input_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The .md filename to delete",
            },
        },
        "required": ["filename"],
    },
    execute=_memory_delete_execute,
    intents=["general", "coding"],
)

EncreMemorySearchTool = build_tool(
    name="memory_search",
    description=(
        "Search memory files by keyword overlap. Returns the most relevant "
        "memory files matching the query, ranked by token overlap score."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant memories",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5)",
            },
        },
        "required": ["query"],
    },
    execute=_memory_search_execute,
    intents=["general", "coding", "research"],
    is_concurrency_safe=lambda _: True,
)
