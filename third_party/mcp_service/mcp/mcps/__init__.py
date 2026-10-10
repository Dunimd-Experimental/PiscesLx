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
MCP Tools Package

Provides a single tool — WebSearchTool — for the PiscesLx inference engine.
All other MCP tools have been removed; the inference engine is a pure
model-serving endpoint, not a development assistant like Claude Code.
"""

from typing import Any, List, Optional

from .base import (
    POPSSMCPToolBase,
    POPSSMCPToolResult,
    POPSSMCPToolRegistry,
)

from .web_search import WebSearchTool


ALL_TOOLS: List[type[POPSSMCPToolBase]] = [
    WebSearchTool,
]


def register_all_tools(registry: Optional[Any] = None) -> List[str]:
    """Register all available tools to the unified tool registry.

    Args:
        registry: POPSSToolRegistry instance. If None, uses POPSSMCPToolRegistry.

    Returns:
        List of registered tool names.
    """
    registered: List[str] = []

    for ToolClass in ALL_TOOLS:
        tool = ToolClass()

        if registry is not None:
            try:
                registry.register_mcp_tool(
                    tool_name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
            except Exception:
                pass

        POPSSMCPToolRegistry.register(tool)
        registered.append(tool.name)

    return registered


def get_tool(name: str) -> Optional[POPSSMCPToolBase]:
    """Get a tool by name."""
    return POPSSMCPToolRegistry.get(name)


def list_tools() -> List[str]:
    """List all available tool names."""
    return POPSSMCPToolRegistry.list()


def list_tools_info() -> List[dict]:
    """List all tools with their info."""
    return POPSSMCPToolRegistry.list_info()


__all__ = [
    "POPSSMCPToolBase",
    "POPSSMCPToolResult",
    "POPSSMCPToolRegistry",
    "ALL_TOOLS",
    "register_all_tools",
    "get_tool",
    "list_tools",
    "list_tools_info",
    "WebSearchTool",
]
