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

import json
from contextvars import ContextVar
from typing import Any

from enta.tools.base import build_tool

_current_loop: ContextVar[Any] = ContextVar("yim_current_loop", default=None)

_parent_loop: Any = None


def set_parent_loop(loop: Any) -> None:
    global _parent_loop
    _parent_loop = loop


def set_active_loop(loop: Any) -> Any:
    """Set the loop that `find_tool` should consult during this turn.

    Returns the previous value so callers can restore via reset_active_loop().
    """
    token = _current_loop.set(loop)
    return token


def reset_active_loop(token: Any) -> None:
    _current_loop.reset(token)


def _resolve_loop() -> Any:
    # Prefer a contextvar-set loop (e.g. sub-agent in progress); fall back to class-level.
    ctx_loop = _current_loop.get()
    if ctx_loop is not None:
        return ctx_loop
    return _parent_loop


async def _find_tool_execute(**kwargs: Any) -> str:
    """Discover and unlock tools by natural-language query.

    Beyond the always-on base set (file_read/write/edit, bash, grep, glob, todo),
    the model does not see every tool's schema in the system prompt. Instead,
    it calls find_tool with a short description of what it wants to do; the
    matching tools are unlocked for the rest of the session and become callable
    natively on the next turn.
    """
    query = (kwargs.get("query") or "").strip()
    if not query:
        return "Error: 'query' is required. Describe the capability you need in natural language."

    top_k = kwargs.get("top_k", 5)
    try:
        top_k = max(1, min(int(top_k), 15))
    except (TypeError, ValueError):
        top_k = 5

    category = (kwargs.get("category") or "").strip() or None

    loop = _resolve_loop()
    if loop is None:
        return "Error: find_tool requires a parent loop reference."

    discovery = getattr(loop, "discovery", None)
    if discovery is None:
        return "Error: tool discovery is not initialised for this loop."

    session_id = getattr(loop.session, "id", "default")

    results = discovery.search(query, top_k=top_k, category=category)
    if not results:
        cats = discovery.list_by_category()
        cat_summary = ", ".join(f"{k} ({len(v)})" for k, v in sorted(cats.items()))
        return (
            f"No tools matched '{query}'. "
            f"Try a broader query or pick a category to browse. "
            f"Available categories: {cat_summary}."
        )

    newly_unlocked = discovery.unlock(session_id, [r["name"] for r in results])

    cards = []
    for r in results:
        cards.append({
            "name": r["name"],
            "category": r["category"],
            "description": r["description"],
            "parameters": r["parameters"],
        })

    payload = {
        "matches": cards,
        "unlocked_now": newly_unlocked,
        "note": (
            "These tools are now available -- call them directly on your next turn. "
            "Already-unlocked tools persist for the rest of this session."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


EncreFindToolTool = build_tool(
    name="find_tool",
    description=(
        "Discover and unlock specialized tools for your task. "
        "Only basic tools (file_read/write/edit, bash, grep, glob, "
        "web_search, todo) are always available.  Everything else must "
        "be unlocked first via find_tool. "
        "Call this proactively at the START of any task that might need "
        "non-basic capabilities. "
        "Examples: 'fetch a URL and extract text content', "
        "'run git log and show a diff', 'take a screenshot of the screen', "
        "'execute SQL queries on a database', 'open a headless browser and "
        "navigate to a page', 'spawn a sub-agent to do research', "
        "'schedule a cron job', 'use Docker to run a container', "
        "'search my memory for notes about X'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description of the capability you need.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results to return (default 5, max 15).",
            },
            "category": {
                "type": "string",
                "description": (
                    "Optional category filter: filesystem, search, shell, web, "
                    "infra, data, code_intel, gui, docs, media, memory, task, "
                    "delegation. Omit for global search."
                ),
            },
        },
        "required": ["query"],
    },
    execute=_find_tool_execute,
    intents=["general", "coding", "research", "data", "system"],
    category="meta",
    triggers=["find tool", "search tools", "discover tool"],
    always_available=True,
    is_concurrency_safe=lambda _: True,
)
