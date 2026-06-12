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

"""Memory tools — create, read, update, delete persistent encrypted memories."""

import os
import re
from typing import Any

from encre.tools.base import build_tool
from encre.memdir.system import EncreMemorySystem


def _get_memory_dir() -> str:
    from encre.config import get_data_dir
    return str(get_data_dir() / "memory")


def _sanitize_filename(name: str) -> str:
    """Normalize a memory name into a safe .md filename."""
    slug = re.sub(r"[^\w\-]", "_", name.strip()).strip("_").lower()
    return slug if slug else "memory"


def _write_encrypted(filepath: str, content: str) -> None:
    from encre.crypto import encrypt
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(encrypt(content))


def _read_encrypted(filepath: str) -> str | None:
    from encre.crypto import decrypt
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return ""
        if raw.startswith("---"):
            return raw  # legacy plaintext
        return decrypt(raw)
    except Exception:
        return None


# ── Tools ────────────────────────────────────────────────────────────────────


async def _memory_create_execute(**kwargs: Any) -> str:
    filename = kwargs.get("filename", "")
    content = kwargs.get("content", "")
    if not filename.endswith(".md"):
        filename += ".md"
    slug = _sanitize_filename(os.path.splitext(filename)[0])
    filename = f"{slug}.md"

    mem_dir = _get_memory_dir()
    filepath = os.path.join(mem_dir, filename)

    if os.path.exists(filepath):
        return (
            f"Memory file '{filename}' already exists. "
            f"Use memory_update to modify it, or choose a different filename."
        )

    try:
        _write_encrypted(filepath, content)
        ms = EncreMemorySystem(mem_dir)
        ms.refresh_index()
        return f"Memory '{filename}' created and encrypted successfully."
    except Exception as e:
        return f"Error creating memory: {e}"


async def _memory_read_execute(**kwargs: Any) -> str:
    filename = kwargs.get("filename", "")
    if not filename.endswith(".md"):
        filename += ".md"

    filepath = os.path.join(_get_memory_dir(), filename)
    content = _read_encrypted(filepath)
    if content is None:
        return f"Memory file '{filename}' not found."

    if not content:
        return f"Memory file '{filename}' is empty."

    return content


async def _memory_update_execute(**kwargs: Any) -> str:
    filename = kwargs.get("filename", "")
    content = kwargs.get("content", "")
    if not filename.endswith(".md"):
        filename += ".md"

    filepath = os.path.join(_get_memory_dir(), filename)
    if not os.path.isfile(filepath):
        return f"Memory file '{filename}' does not exist. Use memory_create to create it."

    try:
        _write_encrypted(filepath, content)
        ms = EncreMemorySystem(_get_memory_dir())
        ms.refresh_index()
        return f"Memory '{filename}' updated and encrypted successfully."
    except Exception as e:
        return f"Error updating memory: {e}"


async def _memory_delete_execute(**kwargs: Any) -> str:
    filename = kwargs.get("filename", "")
    if not filename.endswith(".md"):
        filename += ".md"

    filepath = os.path.join(_get_memory_dir(), filename)
    if not os.path.isfile(filepath):
        return f"Memory file '{filename}' does not exist."

    try:
        os.remove(filepath)
        ms = EncreMemorySystem(_get_memory_dir())
        ms.refresh_index()
        return f"Memory '{filename}' deleted."
    except Exception as e:
        return f"Error deleting memory: {e}"


async def _memory_search_execute(**kwargs: Any) -> str:
    query = kwargs.get("query", "")
    top_k = kwargs.get("top_k", 5)

    ms = EncreMemorySystem(_get_memory_dir())
    results = ms.search(query, top_k=top_k)

    if not results:
        return f"No memories found matching '{query}'."

    lines: list[str] = [f"Memory search results for '{query}':", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r.file_name}** (score: {r.score:.2f})")
        snippet = r.snippet
        if snippet:
            lines.append(f"   {snippet[:200]}")
        lines.append("")
    return "\n".join(lines)


async def _memory_profile_execute(**kwargs: Any) -> str:
    field = kwargs.get("field", "")
    value = kwargs.get("value")
    confidence = float(kwargs.get("confidence", 0.7))

    from encre.profile.system import EncreProfileSystem
    from encre.config import get_data_dir
    mem_dir = str(get_data_dir() / "memory")
    ps = EncreProfileSystem(mem_dir)
    ps.load()

    if value is not None:
        # ── Update mode ──────────────────────────────────────────
        if not field:
            return "Error: field is required when updating."
        valid_fields = {
            "expertise_level", "domain", "formality", "detail_preference",
            "tone", "response_style", "testing_preference", "learning_style",
            "error_tolerance", "os", "editor", "name",
            "language_preference", "timezone",
            "preferred_languages", "preferred_frameworks",
            "skill_levels", "common_goals",
        }
        if field not in valid_fields:
            return (
                f"Unknown field '{field}'. Valid fields: "
                f"{', '.join(sorted(valid_fields))}"
            )
        try:
            ps.update_field(field, value, confidence=confidence)
            return (
                f"Profile field '{field}' updated to '{value}' "
                f"(confidence: {confidence:.2f})."
            )
        except Exception as e:
            return f"Error updating profile: {e}"

    # ── Query mode ───────────────────────────────────────────────
    data = ps.get_data()
    if field:
        if field not in data or not data[field]:
            return f"Profile field '{field}' is not set."
        conf = data.get("confidence", {}).get(field, 0)
        val = data[field]
        if isinstance(val, list):
            val = ", ".join(val)
        elif isinstance(val, dict):
            val = ", ".join(f"{k}: {v}" for k, v in val.items())
        return f"{field}: {val} (confidence: {conf:.2f})"

    # Full profile dump
    lines: list[str] = ["## User Profile", ""]
    sections = {
        "Basic": ["name", "language_preference", "timezone", "expertise_level", "domain"],
        "Communication": ["formality", "detail_preference", "tone", "response_style"],
        "Technical": ["preferred_languages", "preferred_frameworks", "skill_levels", "os", "editor"],
        "Behavioral": ["testing_preference", "learning_style", "error_tolerance", "common_goals"],
    }
    has_any = False
    for section_name, fields in sections.items():
        section_parts: list[str] = []
        for f in fields:
            val = data.get(f)
            if val and val != "" and val != [] and val != {}:
                conf = data.get("confidence", {}).get(f, 0)
                display = val
                if isinstance(val, list):
                    display = ", ".join(val)
                elif isinstance(val, dict):
                    display = ", ".join(f"{k}: {v}" for k, v in val.items())
                section_parts.append(f"  - {f}: {display} (conf: {conf:.2f})")
                has_any = True
        if section_parts:
            lines.append(f"### {section_name}")
            lines.extend(section_parts)
            lines.append("")
    if not has_any:
        lines.append("No profile data yet. Profile is built over time as you interact.")
    lines.append(f"Total updates: {data.get('update_count', 0)}")
    return "\n".join(lines)


EncreMemoryCreateTool = build_tool(
    name="memory_create",
    description=(
        "Create a new persistent memory file. Memories are encrypted, "
        "persist across sessions, and are automatically loaded into the "
        "agent's context on future runs. Use frontmatter (YAML between --- "
        "lines) to set metadata: name, description, type (user/feedback/"
        "project/reference), and tags."
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
    description=(
        "Read a memory file by filename. Returns the full decrypted content "
        "including frontmatter."
    ),
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
    description=(
        "Update an existing memory file. The existing file is read, and the "
        "new content replaces it entirely. Content is encrypted on save."
    ),
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
    description=(
        "Delete a memory file permanently. This cannot be undone."
    ),
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
        "Search memory files semantically. Returns the most relevant memory "
        "files matching the query."
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

EncreMemoryProfileTool = build_tool(
    name="memory_profile",
    description=(
        "Read or update the user profile — structured observations about the "
        "user (expertise, communication style, preferences, OS, editor, etc.). "
        "This is stored as `_profile.md` in the memory directory and is part "
        "of the unified memory system.\n\n"
        "Query (no value): returns all known profile fields with confidence "
        "levels so you can tailor your responses.\n"
        "Update (field + value): records an observation about the user."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": (
                    "Profile field to query or update. Valid fields: "
                    "expertise_level, domain, formality, detail_preference, "
                    "tone, response_style, testing_preference, learning_style, "
                    "error_tolerance, os, editor, name, language_preference, "
                    "timezone, preferred_languages, preferred_frameworks, "
                    "skill_levels, common_goals"
                ),
            },
            "value": {
                "type": "string",
                "description": (
                    "If set, updates the field with this value. "
                    "If omitted, queries the field (or all fields if "
                    "field is also omitted)."
                ),
            },
            "confidence": {
                "type": "number",
                "description": "How confident you are (0.0-1.0, default 0.7). Only used when updating.",
            },
        },
    },
    execute=_memory_profile_execute,
    intents=["general", "coding", "research"],
    is_concurrency_safe=lambda _: True,
)
