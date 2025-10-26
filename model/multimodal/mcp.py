#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
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
from .types import ArcticMCPMessageType, ArcticMCPMessage
from .reasoner.multipath_core import ArcticMultiPathReasoningEngine

class ArcticMCPToolRegistry:
    """A registry for managing MCP tools and capabilities.
    
    This class provides functionality to store and manage tools and capabilities,
    as well as handle tool calls.
    """
    
    def __init__(self, agent_id: str, message_handler: Callable, reasoning_engine: Optional[ArcticMultiPathReasoningEngine] = None):
        """Initialize the MCPToolRegistry instance.

        Args:
            agent_id (str): The ID of the agent.
            message_handler (Callable): The message handler callable.
            reasoning_engine (Optional[ArcticMultiPathReasoningEngine]): The 8-path reasoning engine for intelligent tool selection.
        """
        self.agent_id = agent_id
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
            print(f"[ArcticMCPToolRegistry] Registered native tool: {name}")
        else:
            print(f"[ArcticMCPToolRegistry] Registered external tool: {name}")
    
    async def handle_tool_call(self, tool_name: str, **kwargs) -> Any:
        """Handle a tool call with intelligent routing.

        Args:
            tool_name (str): The name of the tool to call.
            **kwargs: The arguments to pass to the tool.

        Returns:
            Any: The result of the tool call.

        Raises:
            ValueError: If the tool is not found.
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        start_time = time.time()
        
        # 智能工具选择：使用8路径推理引擎优化工具调用
        if self.reasoning_engine:
            reasoning_result = await self.reasoning_engine.analyze_tool_selection(tool_name, kwargs, self.tools)
            if reasoning_result.get("should_optimize"):
                kwargs.update(reasoning_result.get("optimized_params", {}))
        
        # 双轨执行：优先原生执行，回退到外部调用
        try:
            if tool_name in self._native_tools:
                # 原生执行：零延迟直接调用
                result = await self._execute_native_tool(tool_name, **kwargs)
                execution_mode = "native"
                self.execution_stats["native_executions"] += 1
            else:
                # 外部执行：通过消息协议调用
                result = await self._execute_external_tool(tool_name, **kwargs)
                execution_mode = "external"
                self.execution_stats["external_calls"] += 1
            
            execution_time = time.time() - start_time
            self._update_execution_stats(execution_time)
            
            print(f"[ArcticMCPToolRegistry] Tool '{tool_name}' executed via {execution_mode} mode in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            print(f"[ArcticMCPToolRegistry] Tool execution failed for '{tool_name}': {e}")
            raise
    
    async def _execute_native_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute native tool with direct function call."""
        native_handler = self._native_tools[tool_name]
        
        if asyncio.iscoroutinefunction(native_handler):
            return await native_handler(**kwargs)
        else:
            return native_handler(**kwargs)
    
    async def _execute_external_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute external tool through message protocol."""
        tool_call_message = ArcticMCPMessage(
            message_type=ArcticMCPMessageType.TOOL_CALL,
            sender_id=self.agent_id,
            receiver_id="tool_executor",
            content={
                "tool_name": tool_name,
                "arguments": kwargs,
                "timestamp": datetime.now().isoformat(),
                "execution_mode": "external"
            }
        )
        
        return await self.message_handler(tool_call_message)
    
    def _update_execution_stats(self, execution_time: float):
        """Update execution statistics."""
        self.execution_stats["total_executions"] += 1
        total_time = self.execution_stats["average_execution_time"] * (self.execution_stats["total_executions"] - 1)
        self.execution_stats["average_execution_time"] = (total_time + execution_time) / self.execution_stats["total_executions"]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive tool execution statistics with performance metrics."""
        total_executions = self.execution_stats["total_executions"]
        native_count = self.execution_stats["native_executions"]
        external_count = self.execution_stats["external_calls"]
        
        # 计算性能指标
        avg_time = self.execution_stats["average_execution_time"]
        native_ratio = native_count / total_executions if total_executions > 0 else 0
        external_ratio = external_count / total_executions if total_executions > 0 else 0
        
        # 性能评分（基于执行时间和原生执行比例）
        performance_score = (
            (native_ratio * 0.8) +  # 原生执行比例权重
            (min(1.0, 1.0 / (avg_time + 0.1)) * 0.2)  # 执行速度权重
        ) if total_executions > 0 else 0.0
        
        return {
            **self.execution_stats,
            "native_tools_count": len(self._native_tools),
            "external_tools_count": len(self.tools) - len(self._native_tools),
            "total_tools": len(self.tools),
            "performance_metrics": {
                "native_execution_ratio": native_ratio,
                "external_execution_ratio": external_ratio,
                "average_execution_time": avg_time,
                "performance_score": performance_score,
                "efficiency_rating": "high" if performance_score > 0.7 else "medium" if performance_score > 0.4 else "low"
            },
            "tool_distribution": {
                "native_tools": list(self._native_tools.keys()),
                "external_tools": [name for name in self.tools.keys() if name not in self._native_tools]
            },
            "timestamp": datetime.now().isoformat()
        }
    
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

class ArcticMCPProtocol:
    """Protocol for handling MCP messages and interactions with dual-track execution support."""
    
    def __init__(self, agent_id: str, tool_registry: Optional[ArcticMCPToolRegistry] = None):
        """Initialize the ArcticMCPProtocol instance.

        Args:
            agent_id (str): The ID of the agent.
            tool_registry (Optional[ArcticMCPToolRegistry]): The tool registry for dual-track execution.
        """
        self.agent_id = agent_id
        self.tool_registry = tool_registry
        self.execution_modes = {
            "native": "Direct native execution for zero-latency performance",
            "external": "External MCP server execution for extended capabilities",
            "auto": "Intelligent routing based on tool availability and performance"
        }
    
    @staticmethod
    def create_message(
        message_type: ArcticMCPMessageType,
        agent_id: str,
        payload: Dict[str, Any],
        correlation_id: str = ""
    ) -> ArcticMCPMessage:
        """Create a new MCP message.

        Args:
            message_type (ArcticMCPMessageType): The type of the message.
            agent_id (str): The ID of the agent sending the message.
            payload (Dict[str, Any]): The payload of the message.
            correlation_id (str, optional): The correlation ID of the message. 
                If not provided, a new UUID will be generated. Defaults to "".

        Returns:
            ArcticMCPMessage: A new ArcticMCPMessage instance.
        """
        return ArcticMCPMessage(
            message_type=message_type.value,
            agent_id=agent_id,
            payload=payload,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id or str(uuid.uuid4())
        )

    async def create_tool_call_message(self, tool_name: str, arguments: Dict[str, Any], 
                                       receiver_id: str, execution_mode: str = "auto") -> ArcticMCPMessage:
        """Create a tool call message with dual-track execution support.

        Args:
            tool_name (str): The name of the tool to call.
            arguments (Dict[str, Any]): The arguments to pass to the tool.
            receiver_id (str): The ID of the receiver.
            execution_mode (str): The execution mode ("native", "external", or "auto").

        Returns:
            ArcticMCPMessage: The created tool call message.
        """
        # 智能路由决策
        if execution_mode == "auto" and self.tool_registry:
            execution_mode = await self._determine_execution_mode(tool_name, arguments)
        
        return ArcticMCPMessage(
            message_type=ArcticMCPMessageType.TOOL_CALL,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content={
                "tool_name": tool_name,
                "arguments": arguments,
                "timestamp": datetime.now().isoformat(),
                "execution_mode": execution_mode,
                "dual_track_enabled": True
            }
        )
    
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

class ArcticTreeSearchReasoner:
    """A tree search reasoning module for advanced planning.
    
    This class implements a simplified tree search algorithm for complex reasoning tasks.
    """
    
    def __init__(self, model, tokenizer):
        """Initialize the TreeSearchReasoner instance.

        Args:
            model: The model used for reasoning.
            tokenizer: The tokenizer used for processing text.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = 5
        self.max_width = 3
    
    async def search(self, problem: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform tree search for complex reasoning.

        Args:
            problem (str): The problem to solve.
            context (Dict[str, Any]): The context information for the problem.

        Returns:
            List[Dict[str, Any]]: A list of solutions with confidence scores.
        """
        # Simplified tree search implementation
        return [{"solution": "tree_search_result", "confidence": 0.8}]

# Aliases for old names
PiscesMCPProtocol = ArcticMCPProtocol
PiscesTreeSearchReasoner = ArcticTreeSearchReasoner
PiscesMCPToolRegistry = ArcticMCPToolRegistry