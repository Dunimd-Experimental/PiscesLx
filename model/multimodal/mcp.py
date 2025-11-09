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

"""Model Context Protocol integration helpers for Arctic multimodal agents.

The module defines registries and protocol adapters that extend the shared
Pisces MCP infrastructure with Arctic-specific routing, reasoning hooks, and
compatibility shims. The helpers coordinate tool registration, execution
statistics, and message construction while preserving backward compatibility
with utils.mcp implementations.
"""

import uuid
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional, Union
# Import from utils.mcp instead of local types
from utils.mcp import (
    PiscesLxCoreMCPMessageType, PiscesLxCoreMCPMessage, PiscesLxCoreAgenticAction, PiscesLxCoreAgenticObservation,
    PiscesLxCoreMCPRegistry, PiscesLxCoreMCPUnifiedToolExecutor,
    get_unified_tool_executor, PiscesLxCoreMCPTreeSearchReasoner,
    PiscesLxCoreMCPToolMetadata, PiscesLxCoreMCPToolType
)
import utils.mcp
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
    """Registry bridging Arctic tools with the shared Pisces MCP ecosystem.

    The registry augments :class:`PiscesLxCoreMCPRegistry` with Arctic-specific
    tracking such as native/external execution statistics, dual registration with
    the unified executor, and optional integration with the multipath reasoning
    engine to inform execution decisions.

    Attributes:
        agentic_id (str): Identifier for the owning agent instance.
        message_handler (Callable): Callback used to dispatch inbound MCP messages.
        tools (Dict[str, Dict[str, Any]]): Metadata for registered tools.
        capabilities (Dict[str, Dict[str, Any]]): Registered capability descriptors.
        reasoning_engine (Optional[ArcticMultiPathReasoningEngine]): Optional reasoning backend.
        execution_stats (Dict[str, Union[int, float]]): Local execution counters retained for compatibility.
        _native_tools (Dict[str, Callable]): Mapping of native tool names to their handlers.
        unified_executor (PiscesLxCoreMCPUnifiedToolExecutor): Shared executor used for dual registration.
        core_registry (PiscesLxCoreMCPRegistry): Underlying registry from ``utils.mcp``.
    """

    def __init__(self, agentic_id: str, message_handler: Callable, reasoning_engine: Optional[ArcticMultiPathReasoningEngine] = None):
        """Initialize the registry and attach optional reasoning capabilities.

        Args:
            agentic_id (str): Identifier for the owning agent.
            message_handler (Callable): Coroutine invoked for inbound MCP messages.
            reasoning_engine (Optional[ArcticMultiPathReasoningEngine]): Optional reasoning helper
                used to recommend execution modes.
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
        self._native_tools: Dict[str, Callable] = {}
        
        # Initialize unified tool executor for better integration
        self.unified_executor = get_unified_tool_executor()
        
        # Initialize core registry for base functionality
        self.core_registry = PiscesLxCoreMCPRegistry()
    
    async def register_tool(self, name: str, description: str, parameters: Dict[str, Any], native_handler: Optional[Callable] = None):
        """Register a tool with both the unified executor and core registry.

        Args:
            name (str): Tool identifier.
            description (str): Human-readable description of the tool.
            parameters (Dict[str, Any]): Parameter schema for the tool.
            native_handler (Optional[Callable]): Optional coroutine invoked for native execution.
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
        """Execute a tool invocation using the core registry executor.

        Args:
            tool_name (str): Name of the tool to invoke.
            **kwargs: Arguments forwarded to the tool handler.

        Returns:
            Any: Execution result returned by the core registry.
        """
        # Delegate execution to the core registry executor supporting native and external modes.
        return await self.core_registry.execute_tool(tool_name, kwargs)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Return execution statistics collected by the core registry."""
        return self.core_registry.get_stats()
    
    def reset_execution_stats(self):
        """Reset locally tracked execution counters and print a status message."""
        self.execution_stats = {
            "native_executions": 0,
            "external_calls": 0,
            "total_executions": 0,
            "average_execution_time": 0.0
        }
        print("[ArcticMCPToolRegistry] Execution statistics reset")
    
    async def register_native_tool(self, name: str, description: str, 
                                   parameters: Dict[str, Any], handler: Callable):
        """Convenience wrapper for registering a native tool.

        Args:
            name (str): Tool identifier.
            description (str): Short description of the tool.
            parameters (Dict[str, Any]): Tool parameter schema.
            handler (Callable): Native handler invoked for execution.
        """
        await self.register_tool(name, description, parameters, native_handler=handler)

class PiscesLxCoreMCPProtocol:
    """Protocol adapter enabling dual-track MCP execution for Arctic agents.

    The adapter wraps :class:`utils.mcp.PiscesLxCoreMCPProtocol` to add metadata
    for dual-track routing and to integrate with the Arctic tool registry when
    available.
    """

    def __init__(self, agent_id: str, tool_registry: Optional[ArcticMCPToolRegistry] = None):
        """Initialize the protocol adapter and link the optional tool registry.

        Args:
            agent_id (str): Identifier of the agent using the protocol.
            tool_registry (Optional[ArcticMCPToolRegistry]): Registry providing
                tool metadata and reasoning hooks.
        """
        # Initialize base protocol from utils.mcp using fully qualified name to avoid shadowing
        self.base_protocol = utils.mcp.PiscesLxCoreMCPProtocol(agent_id)
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
        """Create an MCP message via the base protocol helper.

        Args:
            message_type (PiscesLxCoreMCPMessageType): Message type identifier.
            agent_id (str): Sender agent identifier.
            payload (Dict[str, Any]): Message payload.
            correlation_id (str): Optional correlation identifier.

        Returns:
            PiscesLxCoreMCPMessage: Constructed message instance.
        """
        # Delegate to base protocol via fully qualified name
        return utils.mcp.PiscesLxCoreMCPProtocol.create_message(message_type, agent_id, payload, correlation_id)

    async def create_tool_call_message(self, tool_name: str, arguments: Dict[str, Any], 
                                       receiver_id: str, execution_mode: str = "auto") -> PiscesLxCoreMCPMessage:
        """Construct a tool call message, optionally leveraging dual-track routing.

        Args:
            tool_name (str): Tool identifier.
            arguments (Dict[str, Any]): Tool invocation arguments.
            receiver_id (str): Message recipient identifier.
            execution_mode (str): Preferred execution mode (``"native"``, ``"external"``, ``"auto"``).

        Returns:
            PiscesLxCoreMCPMessage: Tool call message populated with routing metadata.
        """
        # Evaluate routing strategy when automatic mode is requested.
        if execution_mode == "auto" and self.tool_registry:
            execution_mode = await self._determine_execution_mode(tool_name, arguments)
        
        # Use base protocol to create the message
        message = await self.base_protocol.create_tool_call_message(tool_name, arguments, execution_mode)
        
        # Add receiver-specific information
        message.payload["receiver_id"] = receiver_id
        message.payload["dual_track_enabled"] = True
        
        return message
    
    async def _determine_execution_mode(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Determine execution mode using tool metadata and optional reasoning."""
        if not self.tool_registry:
            return "external"

        # Check whether the tool has a registered native handler.
        tool_info = self.tool_registry.tools.get(tool_name, {})
        if tool_info.get("has_native_handler", False):
            return "native"
        
        # Leverage the reasoning engine to select an execution mode when available.
        if self.tool_registry.reasoning_engine:
            reasoning_result = await self.tool_registry.reasoning_engine.analyze_execution_mode(
                tool_name, arguments, self.tool_registry.tools
            )
            return reasoning_result.get("recommended_mode", "external")
        
        return "external"
    
    async def execute_tool_with_fallback(self, tool_name: str, arguments: Dict[str, Any], 
                                       receiver_id: str) -> Dict[str, Any]:
        """Execute a tool, falling back to external routing when native execution fails."""
        try:
            # Attempt native execution first when a handler is available.
            if self.tool_registry and tool_name in self.tool_registry._native_tools:
                result = await self.tool_registry.handle_tool_call(tool_name, **arguments)
                return {
                    "success": True,
                    "result": result,
                    "execution_mode": "native",
                    "execution_time": 0.0
                }
            
            # Fallback: create an external execution message when native execution is unavailable.
            message = await self.create_tool_call_message(tool_name, arguments, receiver_id, "external")
            return {
                "success": True,
                "message": message,
                "execution_mode": "external",
                "execution_time": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_mode": "failed",
                "execution_time": 0.0
            }
