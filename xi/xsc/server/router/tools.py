#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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
MCP Tools routes.
"""

from typing import Dict, Any, Optional

from fastapi import FastAPI

from ...core.dc import XiLogger


def setup_tools_routes(app: FastAPI, logger: XiLogger, request_count: Dict[str, int]) -> None:
    """
    Setup MCP tools routes.
    
    Args:
        app: FastAPI application
        logger: XiLogger instance
        request_count: Mutable request count reference
    """
    @app.get("/v1/tools/list")
    async def list_tools(category: Optional[str] = None):
        request_count["value"] = request_count.get("value", 0) + 1
        try:
            from opss.mcp.mcps import POPSSToolRegistry
            registry = POPSSToolRegistry.get_instance()
            tools = registry.list_tools()
            
            if category:
                tools = [t for t in tools if t.get("category") == category]
            
            return {"tools": tools, "total": len(tools)}
        except Exception as e:
            logger.error(f"MCP tools list error: {e}", event="xi.mcp.error")
            return {"tools": [], "total": 0}
    
    @app.post("/v1/tools/execute")
    async def execute_tool(request: dict):
        request_count["value"] = request_count.get("value", 0) + 1
        try:
            from opss.mcp.mcps import POPSSToolRegistry
            registry = POPSSToolRegistry.get_instance()
            
            tool_name = request.get("tool")
            arguments = request.get("arguments", {})
            
            result = await registry.execute_tool(tool_name, arguments)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"MCP tool execute error: {e}", event="xi.mcp.exec_error")
            return {"success": False, "error": str(e)}
