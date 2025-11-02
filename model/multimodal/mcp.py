#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import uuid
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional, Union
# Import from utils.mcp instead of local types
from utils.mcp import (
    PiscesLxCoreMCPMessageType, PiscesLxCoreMCPMessage, PiscesLxCoreAgenticAction, PiscesLxCoreAgenticObservation,
    PiscesLxCoreMCPProtocol, PiscesLxCoreMCPRegistry, PiscesLxCoreMCPUnifiedToolExecutor, 
    get_unified_tool_executor, PiscesLxCoreMCPTreeSearchReasoner,
    PiscesLxCoreMCPToolMetadata, PiscesLxCoreMCPToolType
)
from utils.mcp.execution import PiscesLxCoreMCPExecutionResult, PiscesLxCoreMCPExecutionManager
# Import types with fallback for standalone testing
try:
    from .reasoner.multipath_core import ArcticMultiPathReasoningEngine
except ImportError:
    # Fallback for standalone testing
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    try:
        from multimodal.reasoner.multipath_core import ArcticMultiPathReasoningEngine
    except ImportError:
        ArcticMultiPathReasoningEngine = None

class ArcticMCPToolRegistry:
    """A registry for managing MCP tools and capabilities.
    
    This class provides functionality to store and manage tools and capabilities,
    as well as handle tool calls. Extends PiscesLxCoreMCPRegistry for better integration.
    """
    
    def __init__(self, agentic_id: str, message_handler: Callable, reasoning_engine: Optional[ArcticMultiPathReasoningEngine] = None):
        """Initialize the MCPToolRegistry instance.

        Args:
            agentic_id (str): The ID of the agentic.
            message_handler (Callable): The message handler callable.
            reasoning_engine (Optional[ArcticMultiPathReasoningEngine]): The 8-path reasoning engine for intelligent tool selection.
        """
        self.agentic_id = agentic_id
        self.message_handler = message_handler
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.capabilities: Dict[str, Dict[str, Any]] = {}
        self.reasoning_engine = reasoning_engine
        self.execution_stats = {
            "native_executions": 0,
            "external_calls": 0,
            "total_executions": 0,
            "average_execution_time": 0.0
        }
        self._native_tools: Dict[str, Callable] = {}  # 原生工具函数直接映射
        
        # Initialize unified tool executor for better integration
        self.unified_executor = get_unified_tool_executor()
        
        # Initialize core registry for base functionality
        self.core_registry = PiscesLxCoreMCPRegistry()
    
    async def register_tool(self, name: str, description: str, parameters: Dict[str, Any], native_handler: Optional[Callable] = None):
        """Register a tool.

        Args:
            name (str): The name of the tool.
            description (str): The description of the tool.
            parameters (Dict[str, Any]): The parameters of the tool.
            native_handler (Optional[Callable]): Optional native function handler for direct execution.
        """
        self.tools[name] = {
            "description": description,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
            "has_native_handler": native_handler is not None,
            "execution_mode": "native" if native_handler else "external"
        }
        
        if native_handler:
            self._native_tools[name] = native_handler
            # Register with unified executor as native tool
            tool_metadata = PiscesLxCoreMCPToolMetadata(
                name=name,
                description=description,
                tool_type=PiscesLxCoreMCPToolType.NATIVE,
                parameters=parameters,
                function=native_handler,
                priority=10  # High priority for native tools
            )
            self.unified_executor.register_tool(tool_metadata)
            print(f"[ArcticMCPToolRegistry] Registered native tool: {name}")
        else:
            # Register with unified executor as external tool
            tool_metadata = PiscesLxCoreMCPToolMetadata(
                name=name,
                description=description,
                tool_type=PiscesLxCoreMCPToolType.EXTERNAL,
                parameters=parameters,
                priority=5  # Medium priority for external tools
            )
            self.unified_executor.register_tool(tool_metadata)
            print(f"[ArcticMCPToolRegistry] Registered external tool: {name}")
        
        # Also register with core registry for base functionality
        await self.core_registry.register_tool(name, description, parameters, native_handler)
    
    async def handle_tool_call(self, tool_name: str, **kwargs) -> Any:
        """Handle tool calls with unified execution support.

        Args:
            tool_name (str): The name of the tool to call.
            **kwargs: The arguments to pass to the tool.

        Returns:
            Any: The result of the tool execution.
        """
        # 统一使用核心注册表的执行器，支持原生和外部执行模式
        return await self.core_registry.execute_tool(tool_name, kwargs)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics from core registry.

        Returns:
            Dict[str, Any]: The execution statistics from core registry.
        """
        return self.core_registry.get_stats()
    
    def reset_execution_stats(self):
        """Reset execution statistics."""
        self.execution_stats = {
            "native_executions": 0,
            "external_calls": 0,
            "total_executions": 0,
            "average_execution_time": 0.0
        }
        print("[ArcticMCPToolRegistry] Execution statistics reset")
    
    async def register_native_tool(self, name: str, description: str, 
                                   parameters: Dict[str, Any], handler: Callable):
        """Convenience method to register a native tool with direct execution capability.
        
        Args:
            name (str): The name of the tool.
            description (str): The description of the tool.
            parameters (Dict[str, Any]): The parameters of the tool.
            handler (Callable): The native function handler.
        """
        await self.register_tool(name, description, parameters, native_handler=handler)

class PiscesLxCoreMCPProtocol:
    """Protocol for handling MCP messages and interactions with dual-track execution support.
    
    Extends the utils.mcp.PiscesLxCoreMCPProtocol with Arctic-specific functionality.
    """
    
    def __init__(self, agent_id: str, tool_registry: Optional[ArcticMCPToolRegistry] = None):
        """Initialize the PiscesLxCoreMCPProtocol instance.

        Args:
            agent_id (str): The ID of the agent.
            tool_registry (Optional[ArcticMCPToolRegistry]): The tool registry for dual-track execution.
        """
        # Initialize base protocol from utils.mcp
        self.base_protocol = PiscesLxCoreMCPProtocol(agent_id)
        self.agent_id = agent_id
        self.tool_registry = tool_registry
        self.execution_modes = {
            "native": "Direct native execution for zero-latency performance",
            "external": "External MCP server execution for extended capabilities",
            "auto": "Intelligent routing based on tool availability and performance"
        }
    
    @staticmethod
    def create_message(
        message_type: PiscesLxCoreMCPMessageType,
        agent_id: str,
        payload: Dict[str, Any],
        correlation_id: str = ""
    ) -> PiscesLxCoreMCPMessage:
        """Create a new MCP message.

        Args:
            message_type (PiscesLxCoreMCPMessageType): The type of the message.
            agent_id (str): The ID of the agent sending the message.
            payload (Dict[str, Any]): The payload of the message.
            correlation_id (str, optional): The correlation ID of the message. 
                If not provided, a new UUID will be generated. Defaults to "".

        Returns:
            PiscesLxCoreMCPMessage: A new PiscesLxCoreMCPMessage instance.
        """
        # Delegate to base protocol
        return PiscesLxCoreMCPProtocol.create_message(message_type, agent_id, payload, correlation_id)

    async def create_tool_call_message(self, tool_name: str, arguments: Dict[str, Any], 
                                       receiver_id: str, execution_mode: str = "auto") -> PiscesLxCoreMCPMessage:
        """Create a tool call message with dual-track execution support.

        Args:
            tool_name (str): The name of the tool to call.
            arguments (Dict[str, Any]): The arguments to pass to the tool.
            receiver_id (str): The ID of the receiver.
            execution_mode (str): The execution mode ("native", "external", or "auto").

        Returns:
            PiscesLxCoreMCPMessage: The created tool call message.
        """
        # 智能路由决策
        if execution_mode == "auto" and self.tool_registry:
            execution_mode = await self._determine_execution_mode(tool_name, arguments)
        
        # Use base protocol to create the message
        message = await self.base_protocol.create_tool_call_message(tool_name, arguments, execution_mode)
        
        # Add receiver-specific information
        message.payload["receiver_id"] = receiver_id
        message.payload["dual_track_enabled"] = True
        
        return message
    
    async def _determine_execution_mode(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Intelligently determine the execution mode based on tool availability and performance."""
        if not self.tool_registry:
            return "external"
        
        # 检查工具是否注册为原生工具
        tool_info = self.tool_registry.tools.get(tool_name, {})
        if tool_info.get("has_native_handler", False):
            return "native"
        
        # 使用推理引擎优化选择
        if self.tool_registry.reasoning_engine:
            reasoning_result = await self.tool_registry.reasoning_engine.analyze_execution_mode(
                tool_name, arguments, self.tool_registry.tools
            )
            return reasoning_result.get("recommended_mode", "external")
        
        return "external"
    
    async def execute_tool_with_fallback(self, tool_name: str, arguments: Dict[str, Any], 
                                       receiver_id: str) -> Dict[str, Any]:
        """Execute tool with intelligent fallback between native and external modes."""
        try:
            # 首先尝试原生执行
            if self.tool_registry and tool_name in self.tool_registry._native_tools:
                result = await self.tool_registry.handle_tool_call(tool_name, **arguments)
                return {
                    "success": True,
                    "result": result,
                    "execution_mode": "native",
                    "execution_time": 0.0  # 原生执行接近零延迟
                }
            
            # 原生执行不可用，创建外部调用消息
            message = await self.create_tool_call_message(tool_name, arguments, receiver_id, "external")
            return {
                "success": True,
                "message": message,
                "execution_mode": "external",
                "execution_time": None  # 外部执行时间由接收方记录
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_mode": "failed",
                "execution_time": 0.0
            }
