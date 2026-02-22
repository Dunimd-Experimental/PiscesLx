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

"""
MCP Tool Base Classes

Provides the base class for all MCP tools in the PiscesL1 system.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, ClassVar

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from configs.version import VERSION


@dataclass
class POPSSMCPToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class POPSSMCPToolBase(ABC):
    name: ClassVar[str] = "base_tool"
    description: ClassVar[str] = "Base tool class"
    parameters: ClassVar[Dict[str, Any]] = {}
    version: ClassVar[str] = VERSION
    category: ClassVar[str] = "general"
    tags: ClassVar[List[str]] = []
    
    def __init__(self):
        self._LOG = PiscesLxLogger("PiscesLx.Opss.MCP",file_path=get_log_file("PiscesLx.Opss.MCP"), enable_file=True)
    
    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        raise NotImplementedError("Subclasses must implement execute()")
    
    def execute_sync(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.execute(arguments)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.execute(arguments))
        except Exception as e:
            return POPSSMCPToolResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
            )
    
    def to_registry_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "version": self.version,
            "category": self.category,
            "tags": self.tags,
        }
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Optional[str]:
        if not self.parameters:
            return None
        
        properties = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        
        for req in required:
            if req not in arguments:
                return f"Missing required parameter: {req}"
        
        for key, value in arguments.items():
            if key not in properties:
                continue
            
            prop_def = properties[key]
            prop_type = prop_def.get("type", "string")
            
            if prop_type == "string" and not isinstance(value, str):
                return f"Parameter '{key}' must be a string"
            elif prop_type == "integer" and not isinstance(value, int):
                return f"Parameter '{key}' must be an integer"
            elif prop_type == "number" and not isinstance(value, (int, float)):
                return f"Parameter '{key}' must be a number"
            elif prop_type == "boolean" and not isinstance(value, bool):
                return f"Parameter '{key}' must be a boolean"
            elif prop_type == "array" and not isinstance(value, list):
                return f"Parameter '{key}' must be an array"
            elif prop_type == "object" and not isinstance(value, dict):
                return f"Parameter '{key}' must be an object"
        
        return None
    
    def _create_success_result(self, output: Any, metadata: Optional[Dict] = None) -> POPSSMCPToolResult:
        return POPSSMCPToolResult(
            success=True,
            output=output,
            metadata=metadata or {},
        )
    
    def _create_error_result(self, error: str, error_type: Optional[str] = None) -> POPSSMCPToolResult:
        return POPSSMCPToolResult(
            success=False,
            error=error,
            error_type=error_type or "Error",
        )


class POPSSMCPToolRegistry:
    _tools: Dict[str, POPSSMCPToolBase] = {}
    
    @classmethod
    def register(cls, tool: POPSSMCPToolBase) -> None:
        cls._tools[tool.name] = tool
        PiscesLxLogger("PiscesLx.Opss.MCP",file_path=get_log_file("PiscesLx.Opss.MCP"), enable_file=True).info(f"Registered tool: {tool.name}")
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        if name in cls._tools:
            del cls._tools[name]
            return True
        return False
    
    @classmethod
    def get(cls, name: str) -> Optional[POPSSMCPToolBase]:
        return cls._tools.get(name)
    
    @classmethod
    def list(cls) -> List[str]:
        return list(cls._tools.keys())
    
    @classmethod
    def list_info(cls) -> List[Dict[str, Any]]:
        return [tool.to_registry_info() for tool in cls._tools.values()]
    
    @classmethod
    async def execute(cls, name: str, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        tool = cls.get(name)
        if not tool:
            return POPSSMCPToolResult(
                success=False,
                error=f"Tool not found: {name}",
                error_type="ToolNotFoundError",
            )
        
        validation_error = tool.validate_arguments(arguments)
        if validation_error:
            return POPSSMCPToolResult(
                success=False,
                error=validation_error,
                error_type="ValidationError",
            )
        
        start_time = time.time()
        try:
            result = await tool.execute(arguments)
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            return POPSSMCPToolResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                execution_time=time.time() - start_time,
            )
    
    @classmethod
    def clear(cls) -> None:
        cls._tools.clear()
