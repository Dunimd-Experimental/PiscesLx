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

from __future__ import annotations
"""Multi-language LSP manager with auto-discovery of installed servers.

Scans PATH for known LSP servers and activates only those found.
Provides configuration hooks for custom server commands.
"""

import asyncio
import logging
import os
import shutil
from typing import Any

from encre.lsp.protocol import Diagnostic, HoverResult, LSPState, Location, Position, Range
from encre.lsp.client import EncreLSPClient

logger = logging.getLogger("encre.lsp.manager")

# ── Known LSP servers registry ───────────────────────────────────────────
# Each entry: (server_command, [default_args], display_name)
# Arranged by language for readability — auto-discovery picks them up
# by checking if `command` is on PATH.

LANGUAGE_SERVER_REGISTRY: dict[str, list[tuple[str, list[str], str]]] = {
    "python": [
        ("pyright-langserver", ["--stdio"], "Pyright"),
        ("basedpyright-langserver", ["--stdio"], "BasedPyright"),
        ("pylsp", [], "python-lsp-server"),
        ("jedi-language-server", [], "Jedi"),
    ],
    "typescript": [
        ("typescript-language-server", ["--stdio"], "TypeScript"),
    ],
    "javascript": [
        ("typescript-language-server", ["--stdio"], "TypeScript"),
    ],
    "rust": [
        ("rust-analyzer", [], "rust-analyzer"),
    ],
    "go": [
        ("gopls", [], "gopls"),
    ],
    "java": [
        ("java-language-server", [], "Java"),
        ("eclipse.jdt.ls", [], "Eclipse JDT LS"),
    ],
    "csharp": [
        ("omnisharp", [], "OmniSharp"),
        ("csharp-ls", [], "csharp-ls"),
    ],
    "cpp": [
        ("clangd", [], "clangd"),
    ],
    "php": [
        ("phpactor", ["language-server"], "Phpactor"),
        ("intelephense", ["--stdio"], "Intelephense"),
    ],
    "ruby": [
        ("solargraph", ["socket", "--port", "7658"], "Solargraph"),
    ],
    "swift": [
        ("sourcekit-lsp", [], "SourceKit"),
    ],
    "kotlin": [
        ("kotlin-language-server", [], "Kotlin LS"),
    ],
    "css": [
        ("vscode-css-language-server", ["--stdio"], "CSS LS"),
    ],
    "html": [
        ("vscode-html-language-server", ["--stdio"], "HTML LS"),
    ],
    "json": [
        ("vscode-json-language-server", ["--stdio"], "JSON LS"),
    ],
    "yaml": [
        ("yaml-language-server", ["--stdio"], "YAML LS"),
    ],
    "dockerfile": [
        ("docker-langserver", ["--stdio"], "Docker LS"),
    ],
    "lua": [
        ("lua-language-server", [], "Lua LS"),
    ],
    "sql": [
        ("sql-language-server", [], "SQL LS"),
    ],
}

# File extension → language mapping
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".pyx": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".c": "cpp",
    ".cpp": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".css": "css",
    ".scss": "css",
    ".less": "css",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".dockerfile": "dockerfile",
    ".lua": "lua",
    ".sql": "sql",
    ".md": "markdown",
}


class EncreLSPManager:
    """Multi-language LSP manager with auto-discovery.

    On initialization, scans PATH for known LSP servers and activates
    those that are installed.  Unavailable servers are silently skipped.
    """

    def __init__(self) -> None:
        self._clients: dict[str, EncreLSPClient] = {}
        self._status = LSPState(status="not_started")
        self._workspace: str = ""
        self._open_documents: dict[str, int] = {}

    async def initialize_for_workspace(self, workspace: str) -> None:
        """Detect workspace languages and start available LSP servers."""
        self._workspace = workspace
        self._status = LSPState(status="pending")

        detected = self._detect_languages(workspace)
        if not detected:
            self._status = LSPState(status="success")
            return

        tasks: list[tuple[str, asyncio.Task]] = []
        for lang in detected:
            candidates = LANGUAGE_SERVER_REGISTRY.get(lang, [])
            if not candidates:
                continue
            task = asyncio.create_task(self._try_start_servers(lang, candidates, workspace))
            tasks.append((lang, task))

        if not tasks:
            self._status = LSPState(status="success")
            return

        started = []
        for lang, task in tasks:
            client, err = await task
            if client is not None:
                self._clients[lang] = client
                started.append(lang)
                logger.info("[lsp] %s: started %s", lang, client._server_name)
            else:
                logger.info("[lsp] %s: no server available (%s)", lang, err)

        self._status = LSPState(status="success")
        if started:
            logger.info("[lsp] active languages: %s", ", ".join(started))

    async def _try_start_servers(
        self, lang: str, candidates: list[tuple[str, list[str], str]], workspace: str
    ) -> tuple[EncreLSPClient | None, str]:
        """Try each candidate server for a language, return the first that works."""
        last_error = ""
        for command, args, display_name in candidates:
            found = shutil.which(command)
            if not found:
                last_error = f"{command} not found"
                continue
            client = EncreLSPClient(display_name)
            try:
                await client.start(found, args, workspace)
                root_uri = self._path_to_uri(workspace)
                await client.initialize(root_uri)
                logger.info("[lsp] started %s for %s", display_name, lang)
                return client, ""
            except Exception as e:
                last_error = str(e)
                try:
                    await client.close()
                except Exception:
                    pass
        return None, last_error

    # ── LSP query methods ──────────────────────────────────────────────

    async def get_diagnostics(self, file_path: str) -> list[Diagnostic]:
        client = self._get_client(file_path)
        if client is None:
            return []
        file_uri = self._path_to_uri(file_path)
        await self._ensure_document_opened(client, file_uri, file_path)
        try:
            raw = await client.send_request(
                "textDocument/diagnostic",
                {"textDocument": {"uri": file_uri}},
            )
        except Exception:
            return []
        return self._parse_diagnostics(raw)

    async def go_to_definition(
        self, file_path: str, line: int, char: int
    ) -> list[Location]:
        return await self._send_position_request(file_path, line, char, "textDocument/definition")

    async def find_references(
        self, file_path: str, line: int, char: int
    ) -> list[Location]:
        return await self._send_position_request(file_path, line, char, "textDocument/references")

    async def hover(self, file_path: str, line: int, char: int) -> HoverResult | None:
        result = await self._send_position_request_raw(file_path, line, char, "textDocument/hover")
        return self._parse_hover(result) if result else None

    async def document_symbols(self, file_path: str) -> list[dict[str, Any]]:
        client = self._get_client(file_path)
        if client is None:
            return []
        file_uri = self._path_to_uri(file_path)
        await self._ensure_document_opened(client, file_uri, file_path)
        try:
            result = await client.send_request(
                "textDocument/documentSymbol",
                {"textDocument": {"uri": file_uri}},
            )
        except Exception:
            return []
        return result if isinstance(result, list) else []

    async def shutdown(self) -> None:
        await self.close()

    async def close(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
        self._status = LSPState(status="not_started")

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_client(self, file_path: str) -> EncreLSPClient | None:
        ext = os.path.splitext(file_path)[1].lower()
        lang = EXTENSION_MAP.get(ext)
        if lang is None or lang not in self._clients:
            return None
        return self._clients[lang]

    async def _send_position_request(
        self, file_path: str, line: int, char: int, method: str
    ) -> list[Location]:
        client = self._get_client(file_path)
        if client is None:
            return []
        file_uri = self._path_to_uri(file_path)
        await self._ensure_document_opened(client, file_uri, file_path)
        try:
            result = await client.send_request(method, {
                "textDocument": {"uri": file_uri},
                "position": {"line": line, "character": char},
            })
        except Exception:
            return []
        return self._parse_locations(result)

    async def _send_position_request_raw(
        self, file_path: str, line: int, char: int, method: str
    ) -> Any:
        client = self._get_client(file_path)
        if client is None:
            return None
        file_uri = self._path_to_uri(file_path)
        await self._ensure_document_opened(client, file_uri, file_path)
        try:
            return await client.send_request(method, {
                "textDocument": {"uri": file_uri},
                "position": {"line": line, "character": char},
            })
        except Exception:
            return None

    async def _ensure_document_opened(
        self, client: EncreLSPClient, file_uri: str, file_path: str
    ) -> None:
        # Track open count to send didOpen only once per file
        prev = self._open_documents.get(file_uri, 0)
        self._open_documents[file_uri] = prev + 1
        if prev > 0:
            return
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception:
            text = ""
        await client.send_notification("textDocument/didOpen", {
            "textDocument": {"uri": file_uri, "languageId": "", "version": 1, "text": text},
        })

    def _detect_languages(self, workspace: str) -> list[str]:
        languages: set[str] = set()
        known_exts = set(EXTENSION_MAP.keys())
        try:
            for root, dirs, files in os.walk(workspace):
                # Skip common non-source directories
                base = os.path.basename(root)
                if base.startswith(".") or base in (
                    "node_modules", "__pycache__", "target", "build", "dist",
                    "venv", ".venv", ".git",
                ):
                    continue
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in known_exts:
                        languages.add(EXTENSION_MAP[ext])
        except Exception:
            pass
        return sorted(languages)

    @staticmethod
    def _path_to_uri(path: str) -> str:
        abs_path = os.path.abspath(path).replace("\\", "/")
        return f"file:///{abs_path.lstrip('/')}"

    @staticmethod
    def _parse_locations(data: Any) -> list[Location]:
        if not data:
            return []
        if isinstance(data, list):
            return [Location(
                uri=item.get("uri", ""),
                range=Range(
                    start=Position(
                        line=item["range"]["start"]["line"],
                        character=item["range"]["start"]["character"],
                    ),
                    end=Position(
                        line=item["range"]["end"]["line"],
                        character=item["range"]["end"]["character"],
                    ),
                ),
            ) for item in data if "range" in item]
        # Single location
        if "range" in data:
            return [Location(
                uri=data.get("uri", ""),
                range=Range(
                    start=Position(
                        line=data["range"]["start"]["line"],
                        character=data["range"]["start"]["character"],
                    ),
                    end=Position(
                        line=data["range"]["end"]["line"],
                        character=data["range"]["end"]["character"],
                    ),
                ),
            )]
        return []

    @staticmethod
    def _parse_hover(data: Any) -> HoverResult | None:
        if not data:
            return None
        contents = data.get("contents", {})
        if isinstance(contents, str):
            return HoverResult(content=contents)
        if isinstance(contents, dict):
            return HoverResult(content=contents.get("value", ""))
        if isinstance(contents, list):
            parts = []
            for item in contents:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("value", ""))
            return HoverResult(content="\n".join(parts))
        return None

    @staticmethod
    def _parse_diagnostics(data: Any) -> list[Diagnostic]:
        if not data:
            return []
        items = data if isinstance(data, list) else data.get("diagnostics", [])
        return [Diagnostic(
            range=Range(
                start=Position(
                    line=item["range"]["start"]["line"],
                    character=item["range"]["start"]["character"],
                ),
                end=Position(
                    line=item["range"]["end"]["line"],
                    character=item["range"]["end"]["character"],
                ),
            ),
            severity=item.get("severity", 0),
            message=item.get("message", ""),
            source=item.get("source", ""),
        ) for item in items if "range" in item]
