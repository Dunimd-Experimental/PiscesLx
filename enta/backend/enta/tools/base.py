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
Tool definitions for enta.

Two ways to create a tool -- both produce objects with the same duck-typed
interface so they are fully interchangeable in the registry, discovery
system, and agent loop.

**Recommended -- ``build_tool()``**::

    EncreGlobTool = build_tool(
        name="glob",
        description="List files matching a glob pattern.",
        input_schema={...},
        execute=_glob_execute,
        intents=["general", "coding"],
        is_concurrency_safe=lambda _: True,
    )

**Legacy -- subclass ``EncreTool``**::

    class EncreGlobTool(EncreTool):
        name = "glob"
        description = "..."
        input_schema = {...}
        async def execute(self, **kwargs): ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# ── Defaults for optional fields ──────────────────────────────────────

_TOOL_DEFAULTS = {
    "intents": ["general"],
    "category": "general",
    "triggers": [],
    "always_available": False,
    "max_result_size_chars": 100_000,
}


# ── EncreTool (legacy base -- NOT abstract) ─────────────────────────────

class EncreTool:
    """Base class for tool definitions.

    Legacy API kept for backward compatibility.  New tools should use the
    ``build_tool()`` factory instead.

    Subclasses override ``name``, ``description``, ``input_schema``, and
    ``execute()``.  All other fields have sensible defaults.
    """

    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}
    intents: list[str] = ["general"]
    category: str = "general"
    triggers: list[str] = []
    always_available: bool = False
    max_result_size_chars: int = 100_000

    async def execute(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def is_concurrency_safe(self, input_data: dict[str, Any]) -> bool:
        return False

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _minify_schema(self.input_schema),
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": _minify_schema(self.input_schema),
        }

    def discovery_card(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "parameters": self.input_schema,
        }


# ── build_tool -- the recommended way ──────────────────────────────────

def build_tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    execute: Callable[..., Any],
    intents: list[str] | None = None,
    category: str | None = None,
    triggers: list[str] | None = None,
    always_available: bool | None = None,
    max_result_size_chars: int | None = None,
    is_concurrency_safe: Callable[[dict[str, Any]], bool] | None = None,
) -> EncreTool:
    """Create a tool object without subclassing ``EncreTool``.

    Parameters
    ----------
    name
        Canonical tool name (must be unique across the registry).
    description
        Prompt description shown to the model.
    input_schema
        JSON Schema dict describing the expected parameters.
    execute
        Async callable implementing the tool logic.
        Signature: ``async def execute(**kwargs) -> str``
    intents
        Intent tags for intent-based tool filtering (default ``["general"]``).
    category
        Category label for the discovery system (default ``"general"``).
    triggers
        Natural-language trigger phrases (default ``[]``).
    always_available
        If True, the tool is always visible without discovery (default ``False``).
    max_result_size_chars
        Maximum character count for tool results (default ``100_000``).
    is_concurrency_safe
        Predicate that receives the input dict and returns True when the
        tool may run in parallel with other concurrency-safe tools.

    Returns
    -------
    A ``EncreTool`` instance (legacy compat).  The returned object also
    supports the ``EncreGlobTool()`` call pattern -- calling it returns
    ``self`` -- so existing ``registry.register(EncreGlobTool())`` call
    sites continue to work unchanged.
    """
    if not name or not callable(execute):
        raise ValueError("build_tool: 'name' and 'execute' callable are required")

    # Build a tool instance from attributes, mimicking a EncreTool subclass.
    attrs = {
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "intents": intents if intents is not None else _TOOL_DEFAULTS["intents"],
        "category": category if category is not None else _TOOL_DEFAULTS["category"],
        "triggers": triggers if triggers is not None else _TOOL_DEFAULTS["triggers"],
        "always_available": always_available if always_available is not None else _TOOL_DEFAULTS["always_available"],  # noqa: E501
        "max_result_size_chars": max_result_size_chars if max_result_size_chars is not None else _TOOL_DEFAULTS["max_result_size_chars"],  # noqa: E501
        "_concurrency_check": is_concurrency_safe or (lambda _: False),
        "_execute_fn": execute,
        "execute": _make_execute(execute),
        "is_concurrency_safe": _make_concurrency_check(is_concurrency_safe),
        "to_openai_format": _make_to_openai_format(name, description, input_schema),
        "to_anthropic_format": _make_to_anthropic_format(name, description, input_schema),
        "discovery_card": _make_discovery_card(name, category or "general", description, input_schema),  # noqa: E501
        "__call__": lambda self: self,
    }
    tool_cls = type(name.title().replace("_", "") + "Tool", (EncreTool,), attrs)
    return tool_cls()


# ── Schema compression helper ────────────────────────────────────────
# Parameter names that are self-explanatory -- their descriptions can be
# omitted from API tool schemas to reduce prompt token usage.
_SELF_EXPLANATORY_PARAMS: frozenset[str] = frozenset({
    "url", "text", "code", "selector", "key", "name", "path", "pattern",
    "port", "host", "timeout", "query", "value", "label", "title",
    "filename", "file_path", "content", "data", "source", "target",
    "x", "y", "x2", "y2", "width", "height", "depth", "count",
    "cwd", "command", "line", "lines", "start", "end", "index",
    "offset", "limit", "glob", "multiline", "output", "result",
    "old_str", "new_str", "replace_all", "max_pages", "max_items",
    "directory", "dir", "token", "tokens", "enabled", "disabled",
    "cell_id", "cell_type", "kernel_name", "node_id", "parent_id",
    "session_id", "task_id", "job_id", "shell_id", "pid",
})


def _minify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove verbose descriptions from self-explanatory parameters."""
    if "properties" not in schema:
        return schema
    out_props: dict[str, Any] = {}
    for pname, pinfo in schema["properties"].items():
        info = dict(pinfo)
        info.get("description", "")
        # Remove description if param name is self-explanatory and
        # the description is longer than ~5 words (just rephrases the name)
        if "description" in info and pname in _SELF_EXPLANATORY_PARAMS:
            del info["description"]
        elif pname == "action" and "description" in info:
            # Keep action descriptions short
            info["description"] = "Action to perform"
        out_props[pname] = info
    result = dict(schema)
    result["properties"] = out_props
    return result


# ── Factory helpers (closures to avoid lambda pitfalls) ───────────────

def _make_execute(fn: Callable[..., Any]) -> Callable[..., Any]:
    async def _execute(self, **kwargs: Any) -> str:
        return await fn(**kwargs)
    return _execute


def _make_concurrency_check(fn: Callable[[dict[str, Any]], bool] | None) -> Callable[..., bool]:
    if fn is None:
        return lambda self, input_data: False
    def _check(self, input_data: dict[str, Any]) -> bool:
        return fn(input_data)
    return _check


def _make_to_openai_format(name: str, description: str, input_schema: dict) -> Callable[..., dict]:
    minified = _minify_schema(input_schema)
    def _fmt(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": minified,
            },
        }
    return _fmt


def _make_to_anthropic_format(name: str, description: str, input_schema: dict) -> Callable[..., dict]:  # noqa: E501
    minified = _minify_schema(input_schema)
    def _fmt(self) -> dict[str, Any]:
        return {
            "name": name,
            "description": description,
            "input_schema": minified,
        }
    return _fmt


def _make_discovery_card(name: str, category: str, description: str, input_schema: dict) -> Callable[..., dict]:  # noqa: E501
    def _card(self) -> dict[str, Any]:
        return {
            "name": name,
            "category": category,
            "description": description,
            "parameters": input_schema,
        }
    return _card
