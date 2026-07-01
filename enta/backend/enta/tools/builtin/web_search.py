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

"""``web_search`` tool -- public internet search with stdlib only.

The original implementation routed through ``EncreSearchManager`` (DucKDuckGo
Lite + optional SearXNG), but that whole sub-system was removed during the
EnTA slim-down.  This rewrite uses only the Python standard library:
``urllib`` for HTTP and a tiny HTML parser based on ``html.parser`` to
extract DuckDuckGo Lite result blocks.  No external dependencies, no
simulated results -- when the network is unavailable the tool returns a
real, honest error.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

logger = PiscesLxLogger("EnTA.Tools.WebSearch", file_path=get_log_file("EnTA.Tools.WebSearch"), enable_file=True)

from enta.tools.base import build_tool

_DDG_LITE = "https://lite.duckduckgo.com/lite/"


class _DDGResultParser(HTMLParser):
    """Extract DuckDuckGo Lite result blocks.

    DuckDuckGo Lite serves a minimal HTML page; the relevant result
    structure is::

        <a rel="nofollow" class="result-link" href="...">Title</a>
        <td class="result-snippet">Snippet text</td>
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._capture_text = False
        self._capture_href: str | None = None
        self._current_tag: str = ""
        self._current_class: str = ""
        self._title: str = ""
        self._snippet: str = ""
        self.results: list[dict[str, str]] = []
        self._pending: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        self._current_tag = tag
        self._current_class = attr_map.get("class", "")
        if tag == "a" and "result-link" in self._current_class:
            self._capture_text = True
            self._title = ""
            self._capture_href = attr_map.get("href", "")
            self._pending = {"url": self._capture_href, "title": "", "content": ""}
        elif tag == "td" and "result-snippet" in self._current_class:
            self._capture_text = True
            self._snippet = ""

    def handle_data(self, data: str) -> None:
        if not self._capture_text:
            return
        if self._pending is not None and self._current_tag == "a":
            self._title += data
        elif self._current_tag == "td":
            self._snippet += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._pending is not None and self._title:
            self._pending["title"] = self._title.strip()
            self._title = ""
        elif tag == "td" and self._pending is not None and self._snippet:
            self._pending["content"] = self._snippet.strip()
            self._snippet = ""
            if self._pending.get("title") and self._pending.get("url"):
                self.results.append(self._pending)
            self._pending = None
        self._capture_text = False
        self._current_tag = ""
        self._current_class = ""


def _ddg_search(query: str, num: int) -> list[dict[str, str]]:
    encoded = urllib.parse.urlencode({"q": query})
    url = f"{_DDG_LITE}?{encoded}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; PiscesLx-Encre-WebSearch/1.0)"
            ),
            "Accept": "text/html",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="replace")

    parser = _DDGResultParser()
    parser.feed(body)
    # De-duplicate by URL while preserving order.
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for r in parser.results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        unique.append(r)
        if len(unique) >= num:
            break
    return unique


async def _web_search_execute(**kwargs: Any) -> str:
    query = kwargs.get("query", "")
    if not query:
        return "Error: 'query' is required."
    num = max(1, min(int(kwargs.get("num", 5)), 10))

    try:
        results = _ddg_search(query, num)
    except urllib.error.URLError as e:
        logger.warning("[web_search] network error: %s", e)
        return f"Error: network unavailable: {e}"
    except (TimeoutError, json.JSONDecodeError, ValueError) as e:
        logger.warning("[web_search] parse/timeout error: %s", e)
        return f"Error: failed to fetch results: {e}"

    if not results:
        return f"No results found for query '{query}'."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "").strip()
        url = r.get("url", "")
        content = r.get("content", "").strip()
        entry = f"{i}. [{title}]({url})"
        if content:
            entry += f"\n   {content}"
        lines.append(entry)
    return "\n\n".join(lines)


EncreWebSearchTool = build_tool(
    name="web_search",
    description=(
        "Search the public internet via DuckDuckGo Lite. Returns the top "
        "matching pages with title, URL, and snippet. No API key, no "
        "external dependency beyond the Python standard library."
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
                "description": "Maximum number of results (1-10, default 5)",
            },
        },
        "required": ["query"],
    },
    execute=_web_search_execute,
    intents=["general", "research"],
    is_concurrency_safe=lambda _: True,
)


__all__ = ["EncreWebSearchTool"]
