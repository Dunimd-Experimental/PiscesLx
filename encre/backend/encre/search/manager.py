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

"""Hidden MCP-powered search engine.

Replaces the original DuckDuckGo backend with an MCP-based search service.
The MCP server URL is embedded in this module and auto-encrypted to disk on
first use, so users — and the model — cannot discover the endpoint address.

Encrypted config on disk: ``<data_dir>/dsp_cache.bin``
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import re
from typing import Any

from encre.crypto import decrypt, encrypt
from encre.tools.mcp import HttpTransport, MCPClient

logger = logging.getLogger("encre.search")

# ── MCP search server identity ─────────────────────────────────────────
# Hardcoded here; auto-encrypted to a machine-bound file on first use so
# the endpoint never appears on disk in plaintext.

_MCP_SEARCH_URL = "https://mcp.exa.ai/mcp"
_MCP_SEARCH_HEADERS: dict[str, str] = {}
_MCP_SEARCH_TIMEOUT: float = 60.0

# Obfuscated filename — looks like a generic DSP/embedding cache
_MCP_SEARCH_CONFIG_FILE = "dsp_cache.bin"


def _config_path() -> pathlib.Path:
    from encre.config import get_data_dir
    return get_data_dir() / _MCP_SEARCH_CONFIG_FILE


class EncreSearchManager:
    """Search engine that routes queries through an MCP service.

    The MCP server address is loaded from an encrypted config file on first
    use.  The connection is established lazily so callers can construct this
    object synchronously.
    """

    def __init__(self) -> None:
        self._client: MCPClient | None = None
        self._search_tool: str = ""
        self._tool_schema: dict[str, Any] | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config() -> dict[str, Any]:
        path = _config_path()
        if path.is_file():
            try:
                raw = path.read_text(encoding="utf-8")
                return json.loads(decrypt(raw))
            except Exception as exc:
                logger.warning("Failed to decrypt MCP search config, rebuilding: %s", exc)

        # First use — encrypt hardcoded config to disk
        cfg: dict[str, Any] = {
            "url": _MCP_SEARCH_URL,
            "timeout": _MCP_SEARCH_TIMEOUT,
        }
        if _MCP_SEARCH_HEADERS:
            cfg["headers"] = dict(_MCP_SEARCH_HEADERS)

        plain = json.dumps(cfg, ensure_ascii=False)
        encrypted = encrypt(plain)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encrypted, encoding="utf-8")
        path.chmod(0o600)
        logger.info("MCP search config auto-created at %s", path)
        return cfg

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> None:
        if self._client is not None and self._client.is_initialized:
            return
        async with self._lock:
            if self._client is not None and self._client.is_initialized:
                return

            config = self._load_config()
            url = config.get("url", "")
            if not url:
                raise RuntimeError("MCP search config missing 'url'")

            headers = dict(config.get("headers", {}))
            timeout = float(config.get("timeout", 60.0))

            transport = HttpTransport(url, timeout=timeout, headers=headers)
            client = MCPClient(transport)
            try:
                await client.initialize()
                tools = await client.list_tools()
            except Exception:
                await client.close()
                raise

            if not tools:
                await client.close()
                raise RuntimeError("MCP search server exposes no tools")

            self._search_tool = _pick_search_tool(tools)
            # Store the schema for dynamic parameter mapping
            for t in tools:
                if t.get("name") == self._search_tool:
                    self._tool_schema = t.get("inputSchema", {})
                    break
            logger.info(
                "MCP search connected, tool=%s, server=%s",
                self._search_tool, url,
            )
            self._client = client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("MCP search disconnected")

    # ------------------------------------------------------------------
    # Public API — matches the original DuckDuckGo interface exactly
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        num: int = 10,
        language: str = "",
        categories: str = "general",
    ) -> dict[str, Any]:
        """Execute a search query via the MCP service.

        Returns the same dict shape as the original DuckDuckGo backend::

            {"results": [{title, url, content}], "suggestions": []}
        """
        if not query:
            return {"results": [], "suggestions": []}

        await self._ensure_connected()
        assert self._client is not None

        # Build arguments dynamically from the tool's input schema
        args: dict[str, Any] = _map_search_args(
            query=query, num=num, language=language,
            categories=categories, schema=self._tool_schema,
        )

        try:
            content = await self._client.call_tool(self._search_tool, args)
        except Exception as exc:
            logger.warning("MCP search failed: %s", exc)
            return {"results": [], "suggestions": [], "_error": f"Search failed: {exc}"}

        return _normalize_mcp_response(content)

    async def search_batch(
        self,
        queries: list[str],
        *,
        num: int = 5,
        language: str = "",
    ) -> list[dict[str, Any]]:
        tasks = [self.search(q, num=num, language=language) for q in queries]
        return await asyncio.gather(*tasks, return_exceptions=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _map_search_args(
    query: str,
    num: int = 10,
    language: str = "",
    categories: str = "general",
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map standard search parameters to whatever the MCP tool's schema expects.

    Different MCP search servers use different parameter names:
    - ``open-webSearch`` uses ``query``, ``max_results``
    - ``web_search_prime`` (智谱) uses ``search_query``, ``content_size``

    This function inspects the schema and maps accordingly.
    """
    if not schema:
        return {"query": query}

    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    args: dict[str, Any] = {}

    for prop_name, prop_def in props.items():
        ptype = prop_def.get("type", "string")
        pdesc = (prop_def.get("description", "") + " " + prop_name).lower()

        if "query" in pdesc or "search" in pdesc or "keyword" in pdesc:
            # This is likely the search query parameter
            if prop_name in required:
                args[prop_name] = query

    # If no schema match found, fall back to our standard parameter
    if not args:
        args["query"] = query

    # Pass optional params if the tool supports them
    for pname in props:
        low = pname.lower()
        if "max" in low and ("result" in low or "limit" in low or "count" in low):
            args[pname] = num
        if "language" in low or "locale" in low or "region" in low:
            if language:
                args[pname] = language
        if "category" in low or "source" in low or "engine" in low:
            if categories != "general":
                args[pname] = categories

    return args


def _pick_search_tool(tools: list[dict[str, Any]]) -> str:
    """Pick the best tool from the MCP server's tool list.

    Prefers tools whose name contains "search" or "web_search";
    falls back to the first tool.
    """
    candidates: list[str] = []
    for t in tools:
        name: str = t.get("name", "")
        low = name.lower()
        if "search" in low or "web" in low:
            candidates.append(name)
    if candidates:
        return candidates[0]
    return tools[0].get("name", "")


def _normalize_mcp_response(content: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse MCP tool response into ``{results, suggestions}`` format."""
    results: list[dict[str, Any]] = []

    # Concatenate all text content blocks
    raw_text = ""
    for item in content:
        t = item.get("type", "text")
        if t == "text":
            raw_text += item.get("text", "")

    if not raw_text.strip():
        return {"results": [], "suggestions": []}

    stripped = raw_text.strip()

    # Try top-level JSON array of result objects
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, dict):
                        results.append(_normalize_result_entry(entry))
                return {"results": results, "suggestions": []}
        except json.JSONDecodeError:
            pass

    # Try JSON object with "results" key
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                items = parsed.get("results", parsed.get("items", []))
                suggestions = parsed.get("suggestions", parsed.get("related", []))
                if isinstance(items, list):
                    for entry in items:
                        if isinstance(entry, dict):
                            results.append(_normalize_result_entry(entry))
                    return {"results": results, "suggestions": list(suggestions)}
        except json.JSONDecodeError:
            pass

    # Try structured plaintext: blocks separated by "---" with "Title:" / "URL:" lines
    blocks = re.split(r"\n---+\n", stripped)
    if len(blocks) > 1 or "\nTitle:" in stripped:
        for block in blocks:
            entry = _parse_text_block(block.strip())
            if entry:
                results.append(entry)
        if results:
            return {"results": results, "suggestions": []}

    # Fallback: wrap the raw text as a single result
    results.append({
        "title": "Search Result",
        "url": "",
        "content": stripped,
    })
    return {"results": results, "suggestions": []}


_TEXT_BLOCK_RE = re.compile(
    r"^Title:\s*(?P<title>.+)$",
    re.MULTILINE,
)


def _parse_text_block(block: str) -> dict[str, Any] | None:
    """Parse a single text block like::

        Title: asyncio — Asynchronous I/O
        URL: https://docs.python.org/3/library/asyncio.html
        Published: N/A
        Author: N/A
        Highlights:
        asyncio is a library to write concurrent code...
    """
    lines = block.split("\n")
    title = ""
    url = ""
    content_lines: list[str] = []
    in_highlights = False
    for line in lines:
        if line.startswith("Title:"):
            title = line[6:].strip()
        elif line.startswith("URL:"):
            url = line[4:].strip()
        elif line.startswith("Highlights:"):
            in_highlights = True
        elif in_highlights:
            cleaned = line.strip()
            if cleaned == "[...]":
                cleaned = ""
            content_lines.append(cleaned)
    if title and url:
        return {
            "title": title,
            "url": url,
            "content": "\n".join(c for c in content_lines if c).strip(),
        }
    return None


def _normalize_result_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Map any common result key name to our canonical ``{title, url, content}``."""
    return {
        "title": str(
            entry.get("title")
            or entry.get("name")
            or entry.get("heading")
            or ""
        ),
        "url": str(
            entry.get("url")
            or entry.get("link")
            or entry.get("href")
            or entry.get("source")
            or ""
        ),
        "content": str(
            entry.get("content")
            or entry.get("snippet")
            or entry.get("description")
            or entry.get("text")
            or entry.get("body")
            or ""
        ),
    }


__all__ = ["EncreSearchManager"]
