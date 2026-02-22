#!/usr/bin/env/python3
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

"""
MCP Tools Package

All MCP tools for the PiscesL1 system.
"""

from typing import List, Type

from .base import (
    POPSSMCPToolBase,
    POPSSMCPToolResult,
    POPSSMCPToolRegistry,
)

from .web_search import WebSearchTool, ImageSearchTool
from .fetch import FetchTool, URLInfoTool
from .crypto import CryptoPriceTool, CryptoTrendingTool, CryptoSearchTool
from .time_tools import CurrentTimeTool, TimeConvertTool, TimezoneListTool, StopwatchTool
from .document_processor import PDFReaderTool, WordReaderTool, PowerPointReaderTool, TextFileReaderTool
from .sequential_thinking import SequentialThinkingTool, ProblemDecompositionTool


ALL_TOOLS: List[Type[POPSSMCPToolBase]] = [
    WebSearchTool,
    ImageSearchTool,
    FetchTool,
    URLInfoTool,
    CryptoPriceTool,
    CryptoTrendingTool,
    CryptoSearchTool,
    CurrentTimeTool,
    TimeConvertTool,
    TimezoneListTool,
    StopwatchTool,
    PDFReaderTool,
    WordReaderTool,
    PowerPointReaderTool,
    TextFileReaderTool,
    SequentialThinkingTool,
    ProblemDecompositionTool,
]


def register_all_tools(registry: Optional[Any] = None) -> List[str]:
    """Register all MCP tools to the unified tool registry.
    
    Args:
        registry: POPSSToolRegistry instance. If None, uses POPSSMCPToolRegistry.
        
    Returns:
        List of registered tool names
    """
    registered = []
    
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


def get_tool(name: str) -> POPSSMCPToolBase:
    """Get a tool by name.
    
    Args:
        name: Tool name
        
    Returns:
        Tool instance
    """
    return POPSSMCPToolRegistry.get(name)


def list_tools() -> List[str]:
    """List all available tool names.
    
    Returns:
        List of tool names
    """
    return POPSSMCPToolRegistry.list()


def list_tools_info() -> List[dict]:
    """List all tools with their info.
    
    Returns:
        List of tool info dictionaries
    """
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
    "ImageSearchTool",
    "FetchTool",
    "URLInfoTool",
    "CryptoPriceTool",
    "CryptoTrendingTool",
    "CryptoSearchTool",
    "CurrentTimeTool",
    "TimeConvertTool",
    "TimezoneListTool",
    "StopwatchTool",
    "PDFReaderTool",
    "WordReaderTool",
    "PowerPointReaderTool",
    "TextFileReaderTool",
    "SequentialThinkingTool",
    "ProblemDecompositionTool",
]
