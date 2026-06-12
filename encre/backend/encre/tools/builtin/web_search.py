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

"""Web search via encre's built-in search engine — no API keys, no Docker.

Uses ``EncreSearchManager`` (DuckDuckGo Lite by default, with optional
external SearXNG support).  Zero configuration required.
"""

from __future__ import annotations

from typing import Any

from encre.search.manager import EncreSearchManager
from encre.tools.base import build_tool

# Global manager singleton — shared across all tool instances.
_manager: EncreSearchManager | None = None


def _get_manager() -> EncreSearchManager:
    global _manager
    if _manager is None:
        _manager = EncreSearchManager()
    return _manager


def set_search_mode(mode: str, searxng_url: str = "") -> None:
    """Switch the global search backend at runtime.

    Args:
        mode: ``"builtin"`` or ``"searxng"``.
        searxng_url: Base URL of external SearXNG instance (only for ``"searxng"`` mode).
    """
    mgr = _get_manager()
    mgr.mode = mode
    if searxng_url:
        mgr.searxng_url = searxng_url
        mgr._resolved_searxng_url = searxng_url


async def _web_search_execute(**kwargs: Any) -> str:
    query = kwargs.get("query", "")
    if not query:
        return "Error: No search query provided."

    num = min(int(kwargs.get("num", 5)), 5)
    language = kwargs.get("language", "")
    categories = kwargs.get("categories", "general")

    manager = _get_manager()
    result = await manager.search(
        query,
        num=num,
        language=language,
        categories=categories,
    )

    error = result.get("_error", "")
    if error and not result.get("results"):
        return f"Error: {error}"

    results = result.get("results", [])
    suggestions = result.get("suggestions", [])

    if not results:
        if suggestions:
            return f"No results found. Did you mean: {' | '.join(suggestions)}?"
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results[:num], 1):
        title = r.get("title", "").strip()
        url = r.get("url", "")
        content = r.get("content", "").strip()
        entry = f"{i}. [{title}]({url})"
        if content:
            entry += f"\n   {content}"
        lines.append(entry)

    output = "\n\n".join(lines)
    if suggestions:
        output += f"\n\nSuggestions: {' | '.join(suggestions[:5])}"

    return output


EncreWebSearchTool = build_tool(
    name="web_search",
    description=(
        "Search the internet for up-to-date information, news, or answers. "
        "Powered by DuckDuckGo Lite — zero setup, zero API keys."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "num": {
                "type": "integer",
                "description": "Maximum number of results (default: 5, max: 10)",
            },
            "language": {
                "type": "string",
                "description": "Search language code, e.g. zh-CN, en-US (default: all)",
            },
            "categories": {
                "type": "string",
                "description": "Search category (ignored in builtin mode, used with external SearXNG): general, news",
            },
        },
        "required": ["query"],
    },
    execute=_web_search_execute,
    intents=["general", "research"],
    is_concurrency_safe=lambda _: True,
)


__all__ = ["EncreWebSearchTool", "set_search_mode", "_get_manager"]
