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

"""Tool discovery — replace "list every tool in the system prompt" with a search interface.

Architecture
============

Default tool set exposed to the model:
  - file_read, file_write, file_edit, bash, grep, glob, todo (the 7 base tools)
  - find_tool (the dispatcher) — always present

Everything else (browser, docker, deploy, database, lsp, notebook, pdf,
spreadsheet, image, web_fetch, web_search, desktop, rest_client, agent,
git, apply_patch, bash_output/kill/list, memory_*, task_*, cron_*, MCP-discovered
tools, plugin tools) lives only in the discovery index. They are *unlocked*
into the session's active tool set once the model calls
  find_tool(query="...")
and a match comes back. Subsequent backend.chat() calls include them in the
`tools` array, so the model can call them natively with full schema.

The discovery index is built lazily from the global ToolRegistry on first
search. It refreshes when the registry changes (e.g. MCP servers connect).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from encre.tools.base import EncreTool
    from encre.tools.registry import ToolRegistry


# ── Per-tool discovery metadata (overrides class-level defaults) ─────────
# Keyed by tool name. Each entry contributes searchable text and a coarse
# category to help the model judge relevance.

_TOOL_HINTS: dict[str, dict[str, Any]] = {
    # File I/O — base (also indexed for queries that mention them by name)
    "file_read": {
        "category": "filesystem",
        "triggers": ["read", "open file", "cat", "view", "load file", "look at", "inspect"],
    },
    "file_write": {
        "category": "filesystem",
        "triggers": ["write", "create file", "save", "new file", "overwrite"],
    },
    "file_edit": {
        "category": "filesystem",
        "triggers": ["edit", "modify", "change line", "patch", "replace text", "rewrite"],
    },
    "apply_patch": {
        "category": "filesystem",
        "triggers": ["apply patch", "diff", "unified diff", "multi-file edit", "git patch"],
    },
    "grep": {
        "category": "search",
        "triggers": ["search", "find text", "regex", "ripgrep", "rg", "match pattern"],
    },
    "glob": {
        "category": "search",
        "triggers": ["find files", "glob", "by pattern", "list files", "*.py", "**/*.ts"],
    },
    # Shell
    "bash": {
        "category": "shell",
        "triggers": ["shell", "command", "execute", "run script", "terminal"],
    },
    "bash_output": {
        "category": "shell",
        "triggers": ["background", "background shell output", "poll process", "shell tail"],
    },
    "bash_kill": {
        "category": "shell",
        "triggers": ["kill process", "terminate shell", "stop bash"],
    },
    "bash_list": {
        "category": "shell",
        "triggers": ["list shells", "background processes", "running shells"],
    },
    # Web / network
    "web_fetch": {
        "category": "web",
        "triggers": ["fetch url", "download", "http get", "scrape page", "read web page"],
    },
    "web_search": {
        "category": "web",
        "triggers": ["search web", "google", "duckduckgo", "internet search", "lookup online"],
    },
    "rest_client": {
        "category": "web",
        "triggers": ["rest api", "http request", "api call", "curl", "post json"],
    },
    # Containers / deploy / data infra
    "docker": {
        "category": "infra",
        "triggers": ["docker", "container", "image", "compose"],
    },
    "deploy": {
        "category": "infra",
        "triggers": ["deploy", "release", "publish", "ship"],
    },
    "database": {
        "category": "data",
        "triggers": ["database", "sql", "query", "postgres", "mysql", "sqlite"],
    },
    # Code intelligence
    "lsp": {
        "category": "code_intel",
        "triggers": ["lsp", "language server", "hover", "go to definition", "diagnostics", "rename symbol"],
    },
    "git": {
        "category": "code_intel",
        "triggers": ["git", "commit", "diff", "branch", "log", "blame", "status"],
    },
    # Browser / GUI
    "browser": {
        "category": "gui",
        "triggers": ["browser", "headless", "playwright", "chromium", "click button", "navigate url", "screenshot page"],
    },
    "desktop": {
        "category": "gui",
        "triggers": ["desktop", "windows", "ui automation", "click ui", "type", "screenshot", "screen", "mouse", "keyboard"],
    },
    # Notebook / docs
    "notebook": {
        "category": "data",
        "triggers": ["jupyter", "ipynb", "cells", "kernel", "execute notebook"],
    },
    "pdf": {
        "category": "docs",
        "triggers": ["pdf", "extract text", "pdf pages"],
    },
    "spreadsheet": {
        "category": "data",
        "triggers": ["excel", "xlsx", "csv", "sheet", "rows", "columns"],
    },
    "image": {
        "category": "media",
        "triggers": ["image", "resize", "crop", "ocr", "convert image"],
    },
    # Memory (shared, both modes)
    "memory_create": {
        "category": "memory",
        "triggers": ["remember", "save note", "memory create"],
    },
    "memory_read": {
        "category": "memory",
        "triggers": ["recall", "load memory", "memory read"],
    },
    "memory_update": {
        "category": "memory",
        "triggers": ["update memory", "edit note"],
    },
    "memory_delete": {
        "category": "memory",
        "triggers": ["forget", "delete memory"],
    },
    "memory_search": {
        "category": "memory",
        "triggers": ["search memory", "find note", "memory semantic"],
    },
    "memory_profile": {
        "category": "memory",
        "triggers": [
            "user profile", "user preferences", "about the user",
            "expertise", "communication style", "persona",
            "profile query", "profile update",
        ],
    },
    # Task / scheduling
    "task_create": {"category": "task", "triggers": ["task", "schedule task", "background task"]},
    "task_list": {"category": "task", "triggers": ["list tasks"]},
    "task_get": {"category": "task", "triggers": ["task details"]},
    "task_update": {"category": "task", "triggers": ["update task"]},
    "task_stop": {"category": "task", "triggers": ["stop task"]},
    "task_output": {"category": "task", "triggers": ["task output", "task result"]},
    "cron_create": {"category": "task", "triggers": ["cron", "schedule", "recurring", "every minute", "every hour"]},
    "cron_list": {"category": "task", "triggers": ["list cron"]},
    "cron_delete": {"category": "task", "triggers": ["delete cron", "remove schedule"]},
    # Sub-agent
    "agent": {
        "category": "delegation",
        "triggers": ["sub agent", "delegate", "spawn agent", "team", "parallel work"],
    },
    # Self
    "find_tool": {
        "category": "meta",
        "triggers": ["find tool", "search tools", "discover tool"],
    },
    "question": {
        "category": "communication",
        "triggers": ["ask user", "question", "clarify", "confirm intent", "gather context"],
    },
    "todo": {
        "category": "task",
        "triggers": ["todo", "checklist", "track progress"],
    },
}


# ── Always-available base tool set ───────────────────────────────────────
# These N are *NOT* routed through discovery — they ship in the default
# tools array every turn.  Everything else is behind find_tool.

BASE_TOOLS: frozenset[str] = frozenset({
    "file_read",
    "file_write",
    "file_edit",
    "bash",
    "grep",
    "glob",
    "todo",
    "web_search",
    "agent",
    "question",
})


# ── Search index ─────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class _IndexEntry:
    name: str
    category: str
    description: str
    triggers: list[str]
    tokens: Counter[str]
    length: int


@dataclass
class _SessionUnlockState:
    unlocked: set[str] = field(default_factory=set)


class ToolDiscovery:
    """Lexical search over tools + per-session unlock tracking.

    A single ToolDiscovery instance is shared across sessions (the index is
    pure data derived from the global ToolRegistry). Per-session unlock state
    lives in self._sessions, keyed by session_id.

    The model "unlocks" a tool by calling find_tool(query=...). Unlocked tools
    are returned in get_active_tools(session_id) which the loop sends with
    every subsequent backend.chat() call.
    """

    def __init__(self, registry: "ToolRegistry") -> None:
        self.registry = registry
        self._entries: list[_IndexEntry] = []
        self._avgdl: float = 1.0
        self._df: Counter[str] = Counter()
        self._signature: int = -1  # changes when registry changes; triggers rebuild
        self._sessions: dict[str, _SessionUnlockState] = {}
        self._payload_cache: dict[tuple[int, str, str], list[dict[str, Any]]] = {}

    # ─── Index lifecycle ─────────────────────────────────────────────

    def _registry_signature(self) -> int:
        tools = self.registry.list_tools()
        return hash(tuple(sorted(tools.keys())))

    def _ensure_built(self) -> None:
        sig = self._registry_signature()
        if sig == self._signature and self._entries:
            return
        self._build()
        self._signature = sig
        self._payload_cache.clear()

    def _build(self) -> None:
        entries: list[_IndexEntry] = []
        df: Counter[str] = Counter()
        for name, tool in self.registry.list_tools().items():
            hint = _TOOL_HINTS.get(name, {})
            tool_category = getattr(tool, "category", "general") or "general"
            # Centralised hints take precedence over the class default so we don't
            # have to annotate every tool file; an explicit non-default class value
            # still wins.
            category = hint.get("category") if tool_category == "general" else tool_category
            if not category:
                category = "general"
            triggers = list(getattr(tool, "triggers", []) or []) + list(hint.get("triggers", []))
            desc = (getattr(tool, "description", "") or "").strip()
            corpus = " ".join([name, name.replace("_", " "), category, desc, " ".join(triggers)])
            toks = _tokenize(corpus)
            tok_counter = Counter(toks)
            entries.append(_IndexEntry(
                name=name,
                category=category,
                description=desc,
                triggers=triggers,
                tokens=tok_counter,
                length=max(len(toks), 1),
            ))
            for term in tok_counter:
                df[term] += 1
        self._entries = entries
        self._df = df
        if entries:
            self._avgdl = sum(e.length for e in entries) / len(entries)

    # ─── BM25 scoring ───────────────────────────────────────────────

    def _score(self, query_tokens: list[str], entry: _IndexEntry, k1: float = 1.5, b: float = 0.75) -> float:
        if not query_tokens or not self._entries:
            return 0.0
        n = len(self._entries)
        score = 0.0
        for term in query_tokens:
            tf = entry.tokens.get(term, 0)
            if tf == 0:
                continue
            df_t = self._df.get(term, 0)
            idf = math.log(1.0 + (n - df_t + 0.5) / (df_t + 0.5))
            norm = (1.0 - b) + b * (entry.length / max(self._avgdl, 1.0))
            score += idf * ((tf * (k1 + 1.0)) / (tf + k1 * norm))
        return score

    # ─── Public search API ──────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        category: str | None = None,
        exclude: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_built()
        qtokens = _tokenize(query)
        if not qtokens:
            return []
        exclude = set(exclude or set())
        # find_tool never appears in its own results; base tools are already
        # always-on so we omit them too unless the user explicitly searched a
        # category that includes them.
        exclude.add("find_tool")
        if category is None:
            exclude.update(BASE_TOOLS)
        scored: list[tuple[float, _IndexEntry]] = []
        for entry in self._entries:
            if entry.name in exclude:
                continue
            if category and entry.category != category:
                continue
            s = self._score(qtokens, entry)
            if s > 0:
                scored.append((s, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, entry in scored[:top_k]:
            tool = self.registry.get(entry.name)
            if tool is None:
                continue
            card = {
                "name": entry.name,
                "category": entry.category,
                "description": entry.description,
                "parameters": getattr(tool, "input_schema", {}),
                "score": round(score, 3),
            }
            results.append(card)
        return results

    def list_by_category(self) -> dict[str, list[str]]:
        self._ensure_built()
        out: dict[str, list[str]] = {}
        for entry in self._entries:
            out.setdefault(entry.category, []).append(entry.name)
        for cat in out:
            out[cat].sort()
        return out

    # ─── Per-session unlock tracking ────────────────────────────────

    def _state(self, session_id: str) -> _SessionUnlockState:
        st = self._sessions.get(session_id)
        if st is None:
            st = _SessionUnlockState()
            self._sessions[session_id] = st
        return st

    def unlock(self, session_id: str, tool_names: list[str]) -> list[str]:
        """Mark tools as unlocked for the given session. Returns newly-unlocked names."""
        st = self._state(session_id)
        newly: list[str] = []
        for name in tool_names:
            if name in BASE_TOOLS or name == "find_tool":
                continue  # already always-on
            if name in st.unlocked:
                continue
            if self.registry.get(name) is None:
                continue
            st.unlocked.add(name)
            newly.append(name)
        if newly:
            self._payload_cache = {
                key: value for key, value in self._payload_cache.items()
                if key[1] != session_id
            }
        return newly

    def get_unlocked(self, session_id: str) -> list[str]:
        return sorted(self._state(session_id).unlocked)

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._payload_cache = {
            key: value for key, value in self._payload_cache.items()
            if key[1] != session_id
        }

    def get_active_tool_names(self, session_id: str) -> list[str]:
        """Tool names the model should see this turn.

        Exposes:
        - always-on base tools
        - `find_tool`
        - tools unlocked in this session
        - MCP-discovered tools (always visible once connected)
        """
        self._ensure_built()
        present_names = set(self.registry.list_tools().keys())
        names: set[str] = {"find_tool"}
        names.update(name for name in BASE_TOOLS if name in present_names)
        names.update(name for name in self.get_unlocked(session_id) if name in present_names)
        for n in present_names:
            if n.startswith("mcp__"):
                names.add(n)
        return sorted(names)

    def get_active_tools_payload(self, session_id: str, fmt: str = "openai") -> list[dict[str, Any]]:
        """Materialize the active tool list as backend-format dicts."""
        self._ensure_built()
        cache_key = (self._signature, session_id, fmt)
        cached = self._payload_cache.get(cache_key)
        if cached is not None:
            return cached

        names = self.get_active_tool_names(session_id)
        out: list[dict[str, Any]] = []
        for name in names:
            tool = self.registry.get(name)
            if tool is None:
                continue
            if fmt == "anthropic":
                out.append(tool.to_anthropic_format())
            else:
                out.append(tool.to_openai_format())
        self._payload_cache[cache_key] = out
        return out


__all__ = ["ToolDiscovery", "BASE_TOOLS"]
