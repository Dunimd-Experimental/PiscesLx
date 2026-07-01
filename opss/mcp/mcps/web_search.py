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
WebSearchTool - Public internet search via DuckDuckGo Lite.

Uses only the Python standard library (urllib + html.parser).
No external dependencies, no API key required.
"""

import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from .base import POPSSMCPToolBase, POPSSMCPToolResult

_DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"


class _DDGResultParser(HTMLParser):
    """Extract DuckDuckGo Lite result blocks from the minimal HTML response.

    DuckDuckGo Lite serves a stripped-down HTML page; the relevant structure::

        <a rel="nofollow" class="result-link" href="...">Title</a>
        <td class="result-snippet">Snippet text</td>
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._capture_text = False
        self._capture_href: Optional[str] = None
        self._current_tag: str = ""
        self._current_class: str = ""
        self._title: str = ""
        self._snippet: str = ""
        self.results: List[Dict[str, str]] = []
        self._pending: Optional[Dict[str, str]] = None

    def handle_starttag(
        self, tag: str, attrs: List[tuple[str, Optional[str]]]
    ) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        self._current_tag = tag
        self._current_class = attr_map.get("class", "")
        if tag == "a" and "result-link" in self._current_class:
            self._capture_text = True
            self._title = ""
            self._capture_href = attr_map.get("href", "")
            self._pending = {
                "url": self._capture_href or "",
                "title": "",
                "snippet": "",
            }
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
            self._pending["snippet"] = self._snippet.strip()
            self._snippet = ""
            if self._pending.get("title") and self._pending.get("url"):
                self.results.append(self._pending)
            self._pending = None
        self._capture_text = False
        self._current_tag = ""
        self._current_class = ""


def _search_ddg(query: str, num: int = 5) -> List[Dict[str, str]]:
    """Perform a DuckDuckGo Lite search and return structured results."""
    encoded = urllib.parse.urlencode({"q": query})
    url = f"{_DDG_LITE_URL}?{encoded}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; PiscesLx-WebSearch/1.0)"
            ),
            "Accept": "text/html",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="replace")

    parser = _DDGResultParser()
    parser.feed(body)

    seen: set[str] = set()
    unique: List[Dict[str, str]] = []
    for r in parser.results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        unique.append(r)
        if len(unique) >= num:
            break
    return unique


class WebSearchTool(POPSSMCPToolBase):
    name = "web_search"
    description = (
        "Search the public internet via DuckDuckGo Lite. "
        "Returns top matching pages with title, URL, and snippet. "
        "No API key or external dependencies required."
    )
    category = "search"
    tags = ["web", "search", "internet", "ddg"]

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-10)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        query = arguments.get("query", "")
        max_results = min(max(int(arguments.get("max_results", 5)), 1), 10)

        if not query:
            return self._create_error_result("query is required", "ValidationError")

        try:
            results = _search_ddg(query, num=max_results)
        except urllib.error.URLError as e:
            return self._create_error_result(
                f"Network error: {e.reason}", "NetworkError"
            )
        except (TimeoutError, OSError) as e:
            return self._create_error_result(
                f"Request failed: {e}", "RequestError"
            )

        output = {
            "query": query,
            "count": len(results),
            "results": results,
        }
        return self._create_success_result(output)


__all__ = ["WebSearchTool"]
