#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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



from typing import Any

from enta.lsp.manager import EncreLSPManager
from enta.tools.base import build_tool

_manager: EncreLSPManager | None = None


def _get_manager() -> EncreLSPManager:
    global _manager
    if _manager is None:
        _manager = EncreLSPManager()
    return _manager


def _format_symbols(symbols: list[dict[str, Any]], indent: int = 0) -> str:
    lines: list[str] = []
    prefix = "  " * indent
    for sym in symbols:
        name = sym.get("name", "?")
        kind = sym.get("kind", 0)
        kind_name = _symbol_kind_name(kind)
        if "range" in sym and isinstance(sym["range"], dict):
            start = sym["range"].get("start", {})
            lines.append(
                f"{prefix}{kind_name}: {name} "
                f"({start.get('line', 0)}:{start.get('character', 0)})"
            )
        else:
            lines.append(f"{prefix}{kind_name}: {name}")

        children = sym.get("children", [])
        if isinstance(children, list) and children:
            lines.append(_format_symbols(children, indent + 1))
    return "\n".join(lines)


def _symbol_kind_name(kind: int) -> str:
    names: dict[int, str] = {
        1: "File",
        2: "Module",
        3: "Namespace",
        4: "Package",
        5: "Class",
        6: "Method",
        7: "Property",
        8: "Field",
        9: "Constructor",
        10: "Enum",
        11: "Interface",
        12: "Function",
        13: "Variable",
        14: "Constant",
        15: "String",
        16: "Number",
        17: "Boolean",
        18: "Array",
        19: "Object",
        20: "Key",
        21: "Null",
        22: "EnumMember",
        23: "Struct",
        24: "Event",
        25: "Operator",
        26: "TypeParameter",
    }
    return names.get(kind, f"Kind({kind})")


async def _lsp_execute(**kwargs: Any) -> str:
    operation = kwargs.get("operation", "")
    file_path = kwargs.get("file_path", "")
    line = kwargs.get("line", 0)
    character = kwargs.get("character", 0)
    workspace = kwargs.get("workspace", "")

    if operation == "initialize":
        if not workspace:
            return "Error: workspace is required for initialization"
        await _get_manager().initialize_for_workspace(workspace)
        return "LSP servers initialized"

    if operation == "shutdown":
        await _get_manager().shutdown()
        global _manager
        _manager = None
        return "LSP servers shut down"

    manager = _get_manager()

    if operation == "diagnostics":
        if not file_path:
            return "Error: file_path is required"
        diagnostics = await manager.get_diagnostics(file_path)
        if not diagnostics:
            return "No diagnostics found"
        lines: list[str] = []
        for d in diagnostics:
            lines.append(
                f"[{d.severity}] {d.message} "
                f"at {d.range.start.line}:{d.range.start.character}"
            )
        return "\n".join(lines)

    if operation == "go_to_definition":
        if not file_path:
            return "Error: file_path is required"
        locations = await manager.go_to_definition(file_path, line, character)
        if not locations:
            return "No definition found"
        lines = []
        for loc in locations:
            lines.append(
                f"{loc.uri} "
                f"({loc.range.start.line}:{loc.range.start.character})"
            )
        return "\n".join(lines)

    if operation == "find_references":
        if not file_path:
            return "Error: file_path is required"
        locations = await manager.find_references(file_path, line, character)
        if not locations:
            return "No references found"
        lines = []
        for loc in locations:
            lines.append(
                f"{loc.uri} "
                f"({loc.range.start.line}:{loc.range.start.character})"
            )
        return "\n".join(lines)

    if operation == "hover":
        if not file_path:
            return "Error: file_path is required"
        hover_result = await manager.hover(file_path, line, character)
        if hover_result is None:
            return "No hover information available"
        return hover_result.contents

    if operation == "document_symbols":
        if not file_path:
            return "Error: file_path is required"
        symbols = await manager.document_symbols(file_path)
        if not symbols:
            return "No symbols found"
        return _format_symbols(symbols)

    return f"Unknown operation: {operation}"


EncreLSPTool = build_tool(
    name="lsp",
    description=(
        "Query LSP language server for code intelligence: "
        "go to definition, find references, hover info, diagnostics, "
        "and document symbols."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "go_to_definition",
                    "find_references",
                    "hover",
                    "diagnostics",
                    "document_symbols",
                    "initialize",
                    "shutdown",
                ],
                "description": "The LSP operation to perform",
            },
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to query",
            },
            "line": {
                "type": "integer",
                "description": "0-based line number for cursor position",
            },
            "character": {
                "type": "integer",
                "description": "0-based character offset for cursor position",
            },
            "workspace": {
                "type": "string",
                "description": "Workspace root directory for initialization",
            },
        },
        "required": ["operation"],
    },
    execute=_lsp_execute,
    intents=["coding"],
    is_concurrency_safe=lambda _: True,
)
