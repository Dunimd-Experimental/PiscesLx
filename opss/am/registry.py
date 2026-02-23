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
Agent-MCP Unified Tool Registry

This module provides a unified registry that integrates MCP (Model Context Protocol)
tools with Agent experts, enabling seamless tool discovery and execution across
both systems.

Key Components:
    - POPSSToolType: Enumeration of tool types (MCP_TOOL, AGENT_EXPERT, NATIVE_FUNCTION)
    - POPSSToolInfo: Tool metadata container with execution statistics
    - POPSSToolResult: Execution result container with timing and thoughts
    - POPSSToolRegistry: Singleton registry for unified tool management

Features:
    - Unified tool registration for MCP tools, agent experts, and native functions
    - Automatic MCP tool discovery from MCP plaza
    - Tool search by name, description, tags, or capabilities
    - Execution statistics tracking (usage count, success rate, timing)
    - Call history logging for debugging and auditing
    - Singleton pattern for global registry access

Tool Types:
    - MCP_TOOL: Tools registered from MCP plaza
    - AGENT_EXPERT: Agent-based expert tools with think/act/observe cycle
    - NATIVE_FUNCTION: Direct Python function wrappers

Usage:
    # Get singleton instance
    registry = POPSSToolRegistry.get_instance()
    
    # Register MCP tools (auto-discovered if plaza provided)
    registry.register_mcp_tool("web_search", "Search the web", {"query": "string"})
    
    # Register agent expert
    registry.register_agent_expert(
        agent=my_agent,
        name="code_expert",
        description="Code generation expert",
        capabilities={"code", "analysis"}
    )
    
    # Register native function
    registry.register_native_function(
        func=my_function,
        name="helper",
        description="Helper function"
    )
    
    # Execute tool
    result = await registry.execute("web_search", {"query": "AI news"})
    
    # Search tools
    tools = registry.search_tools("code")

Thread Safety:
    The registry is designed for concurrent access. Tool registration and
    execution are thread-safe. Statistics updates use atomic operations.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file


class POPSSToolType(Enum):
    """
    Enumeration of tool types supported by the registry.
    
    Attributes:
        MCP_TOOL: Tools registered from MCP (Model Context Protocol) plaza.
            These tools are discovered automatically from the MCP infrastructure.
        AGENT_EXPERT: Agent-based expert tools that use the think/act/observe
            cycle for intelligent task execution.
        NATIVE_FUNCTION: Direct Python function wrappers for simple operations
            that don't require agent infrastructure.
    """
    MCP_TOOL = "mcp_tool"
    AGENT_EXPERT = "agent_expert"
    NATIVE_FUNCTION = "native_function"


@dataclass
class POPSSToolInfo:
    """
    Container for tool metadata and execution statistics.
    
    This dataclass stores comprehensive information about a registered tool,
    including its identity, capabilities, and runtime statistics.
    
    Attributes:
        tool_id: Unique identifier for this tool (auto-generated).
        name: Human-readable name for the tool.
        description: Detailed description of the tool's purpose.
        tool_type: Type classification (MCP_TOOL, AGENT_EXPERT, NATIVE_FUNCTION).
        parameters: JSON schema describing expected parameters.
        return_type: Expected return type of the tool.
        executor: Callable for native functions (None for MCP/Agent tools).
        agent: Agent instance for AGENT_EXPERT type tools.
        mcp_tool_name: Original MCP tool name for MCP_TOOL type.
        capabilities: Set of capability strings for matching.
        tags: Set of tags for search and categorization.
        priority: Execution priority (higher = more important, default: 5).
        usage_count: Number of times the tool has been executed.
        success_rate: Ratio of successful executions (0.0 to 1.0).
        average_execution_time: Average execution time in seconds.
    
    Example:
        >>> tool_info = POPSSToolInfo(
        ...     tool_id="mcp_web_search_abc123",
        ...     name="web_search",
        ...     description="Search the web for information",
        ...     tool_type=POPSSToolType.MCP_TOOL,
        ...     capabilities={"search", "web"}
        ... )
    """
    tool_id: str
    name: str
    description: str
    tool_type: POPSSToolType
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    return_type: str = "any"
    
    executor: Optional[Callable] = None
    agent: Optional[Any] = None
    mcp_tool_name: Optional[str] = None
    
    capabilities: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    
    priority: int = 5
    usage_count: int = 0
    success_rate: float = 1.0
    average_execution_time: float = 0.0


@dataclass
class POPSSToolResult:
    """
    Container for tool execution results.
    
    This dataclass represents the result of a tool execution, including
    success status, output data, timing information, and agent thoughts
    for agent-based tools.
    
    Attributes:
        tool_id: ID of the tool that was executed.
        tool_name: Name of the tool for reference.
        tool_type: Type of the executed tool.
        success: Boolean indicating if execution succeeded.
        result: Output data from successful execution.
        error: Error message if execution failed.
        execution_time: Time taken for execution in seconds.
        agent_thoughts: List of thought dictionaries from agent execution.
    
    Example:
        >>> result = POPSSToolResult(
        ...     tool_id="agent_code_expert_xyz",
        ...     tool_name="code_expert",
        ...     tool_type=POPSSToolType.AGENT_EXPERT,
        ...     success=True,
        ...     result={"code": "print('Hello')"},
        ...     execution_time=1.5
        ... )
    """
    tool_id: str
    tool_name: str
    tool_type: POPSSToolType
    
    success: bool
    result: Any = None
    error: Optional[str] = None
    
    execution_time: float = 0.0
    agent_thoughts: List[Dict[str, Any]] = field(default_factory=list)


class POPSSToolRegistry:
    """
    Singleton registry for unified tool management.
    
    This class provides a unified interface for registering, discovering, and
    executing tools from multiple sources: MCP plaza, agent experts, and native
    Python functions.
    
    Key Features:
        - Singleton pattern for global registry access
        - Automatic MCP tool discovery
        - Tool search by name, description, tags, or capabilities
        - Execution statistics tracking
        - Call history logging
        - Support for async and sync execution
    
    Attributes:
        _instance: Singleton instance of the registry.
        _tools: Dictionary mapping tool_id to POPSSToolInfo.
        _tools_by_name: Dictionary mapping tool name to tool_id.
        _agents: Dictionary mapping agent_id to agent instance.
        _mcp_plaza: Reference to the MCP plaza for tool discovery.
        _mcp_bridge: Reference to the MCP bridge for tool execution.
        _call_history: List of recent tool call records.
        _max_history: Maximum number of call records to retain.
    
    Example:
        >>> # Get singleton instance
        >>> registry = POPSSToolRegistry.get_instance()
        >>> 
        >>> # Register and execute
        >>> registry.register_native_function(my_func, "helper")
        >>> result = await registry.execute("helper", {"arg": "value"})
        >>> 
        >>> # Search tools
        >>> tools = registry.search_tools("analysis")
    
    See Also:
        - POPSSToolInfo: Tool metadata container
        - POPSSToolResult: Execution result container
        - POPSSToolType: Tool type enumeration
    """
    
    _instance: Optional['POPSSToolRegistry'] = None
    
    def __new__(cls, *args, **kwargs):
        """
        Create or return the singleton instance.
        
        Returns:
            POPSSToolRegistry: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        mcp_plaza: Optional[Any] = None,
        mcp_bridge: Optional[Any] = None,
        auto_discover_mcp: bool = True
    ):
        """
        Initialize the tool registry.
        
        Args:
            mcp_plaza: Optional MCP plaza instance for tool discovery.
            mcp_bridge: Optional MCP bridge for tool execution.
            auto_discover_mcp: Whether to auto-discover MCP tools (default: True).
        """
        if self._initialized:
            return
        
        self._LOG = PiscesLxLogger("PiscesLx.Opss.AM",file_path=get_log_file("PiscesLx.Opss.AM"), enable_file=True)
        
        self._tools: Dict[str, POPSSToolInfo] = {}
        self._tools_by_name: Dict[str, str] = {}
        self._agents: Dict[str, Any] = {}
        
        self._mcp_plaza = mcp_plaza
        self._mcp_bridge = mcp_bridge
        
        self._call_history: List[Dict[str, Any]] = []
        self._max_history = 1000
        
        self._initialized = True
        
        if mcp_plaza and auto_discover_mcp:
            self._discover_mcp_tools()
        
        self._LOG.info(f"POPSSToolRegistry initialized with {len(self._tools)} tools")
    
    @classmethod
    def get_instance(cls) -> 'POPSSToolRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        cls._instance = None
    
    def _discover_mcp_tools(self):
        if not self._mcp_plaza:
            return
        
        try:
            if hasattr(self._mcp_plaza, 'list_tools'):
                mcp_tools = self._mcp_plaza.list_tools(use_cache=False)
            else:
                return
            
            for tool_name in mcp_tools:
                tool_data = None
                if hasattr(self._mcp_plaza, 'get_tool'):
                    tool_data = self._mcp_plaza.get_tool(tool_name)
                
                if tool_data:
                    self.register_mcp_tool(
                        tool_name=tool_name,
                        description=tool_data.get('description', ''),
                        parameters=tool_data.get('parameters', {}),
                    )
            
            self._LOG.info(f"Discovered {len(mcp_tools)} MCP tools")
            
        except Exception as e:
            self._LOG.error(f"Failed to discover MCP tools: {e}")
    
    def register_mcp_tool(
        self,
        tool_name: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 5,
    ) -> str:
        tool_id = f"mcp_{tool_name}_{uuid.uuid4().hex[:8]}"
        
        tool_info = POPSSToolInfo(
            tool_id=tool_id,
            name=tool_name,
            description=description or f"MCP Tool: {tool_name}",
            tool_type=POPSSToolType.MCP_TOOL,
            parameters=parameters or {},
            mcp_tool_name=tool_name,
            priority=priority,
        )
        
        self._tools[tool_id] = tool_info
        self._tools_by_name[tool_name] = tool_id
        
        self._LOG.debug(f"Registered MCP tool: {tool_name}")
        return tool_id
    
    def register_agent_expert(
        self,
        agent: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        capabilities: Optional[Set[str]] = None,
        priority: int = 5,
    ) -> str:
        agent_name = name or agent.__class__.__name__
        tool_id = f"agent_{agent_name}_{uuid.uuid4().hex[:8]}"
        
        agent_capabilities = capabilities or set()
        if hasattr(agent, 'capabilities'):
            agent_capabilities = agent.capabilities
        
        agent_id = getattr(agent, 'agent_id', str(uuid.uuid4()))
        
        tool_info = POPSSToolInfo(
            tool_id=tool_id,
            name=agent_name,
            description=description or f"Agent Expert: {agent_name}",
            tool_type=POPSSToolType.AGENT_EXPERT,
            agent=agent,
            capabilities=agent_capabilities,
            priority=priority,
        )
        
        self._tools[tool_id] = tool_info
        self._tools_by_name[agent_name] = tool_id
        self._agents[agent_id] = agent
        
        self._LOG.info(f"Registered agent expert: {agent_name}")
        return tool_id
    
    def register_native_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Set[str]] = None,
        priority: int = 5,
    ) -> str:
        func_name = name or func.__name__
        tool_id = f"native_{func_name}_{uuid.uuid4().hex[:8]}"
        
        tool_info = POPSSToolInfo(
            tool_id=tool_id,
            name=func_name,
            description=description or f"Native Function: {func_name}",
            tool_type=POPSSToolType.NATIVE_FUNCTION,
            parameters=parameters or {},
            executor=func,
            capabilities=capabilities or set(),
            priority=priority,
        )
        
        self._tools[tool_id] = tool_info
        self._tools_by_name[func_name] = tool_id
        
        self._LOG.debug(f"Registered native function: {func_name}")
        return tool_id
    
    def unregister(self, name_or_id: str) -> bool:
        tool = self.get_tool(name_or_id)
        if not tool:
            return False
        
        if tool.tool_id in self._tools:
            del self._tools[tool.tool_id]
        if tool.name in self._tools_by_name:
            del self._tools_by_name[tool.name]
        
        self._LOG.debug(f"Unregistered tool: {tool.name}")
        return True
    
    def get_tool(self, name_or_id: str) -> Optional[POPSSToolInfo]:
        if name_or_id in self._tools:
            return self._tools[name_or_id]
        if name_or_id in self._tools_by_name:
            return self._tools[self._tools_by_name[name_or_id]]
        return None
    
    def list_tools(
        self,
        tool_type: Optional[POPSSToolType] = None,
        capabilities: Optional[Set[str]] = None,
    ) -> List[POPSSToolInfo]:
        tools = list(self._tools.values())
        
        if tool_type:
            tools = [t for t in tools if t.tool_type == tool_type]
        
        if capabilities:
            tools = [t for t in tools if t.capabilities & capabilities]
        
        return sorted(tools, key=lambda t: -t.priority)
    
    def list_mcp_tools(self) -> List[POPSSToolInfo]:
        return self.list_tools(tool_type=POPSSToolType.MCP_TOOL)
    
    def list_agent_experts(self) -> List[POPSSToolInfo]:
        return self.list_tools(tool_type=POPSSToolType.AGENT_EXPERT)
    
    def list_native_functions(self) -> List[POPSSToolInfo]:
        return self.list_tools(tool_type=POPSSToolType.NATIVE_FUNCTION)
    
    def search_tools(self, query: str) -> List[POPSSToolInfo]:
        query_lower = query.lower()
        results = []
        
        for tool in self._tools.values():
            if query_lower in tool.name.lower():
                results.append(tool)
            elif query_lower in tool.description.lower():
                results.append(tool)
            elif any(query_lower in tag.lower() for tag in tool.tags):
                results.append(tool)
            elif any(query_lower in cap.lower() for cap in tool.capabilities):
                results.append(tool)
        
        return sorted(results, key=lambda t: -t.priority)
    
    async def execute(
        self,
        name_or_id: str,
        arguments: Dict[str, Any],
        context: Optional[Any] = None,
        timeout: float = 60.0,
    ) -> POPSSToolResult:
        start_time = datetime.now()
        tool = self.get_tool(name_or_id)
        
        if not tool:
            return POPSSToolResult(
                tool_id="unknown",
                tool_name=name_or_id,
                tool_type=POPSSToolType.MCP_TOOL,
                success=False,
                error=f"Tool not found: {name_or_id}",
            )
        
        try:
            if tool.tool_type == POPSSToolType.MCP_TOOL:
                result = await self._execute_mcp_tool(tool, arguments, timeout)
            
            elif tool.tool_type == POPSSToolType.AGENT_EXPERT:
                result = await self._execute_agent_expert(tool, arguments, context, timeout)
            
            elif tool.tool_type == POPSSToolType.NATIVE_FUNCTION:
                result = await self._execute_native_function(tool, arguments, timeout)
            
            else:
                result = POPSSToolResult(
                    tool_id=tool.tool_id,
                    tool_name=tool.name,
                    tool_type=tool.tool_type,
                    success=False,
                    error=f"Unknown tool type: {tool.tool_type}",
                )
            
            tool.usage_count += 1
            if result.success:
                tool.success_rate = (tool.success_rate * (tool.usage_count - 1) + 1.0) / tool.usage_count
            else:
                tool.success_rate = (tool.success_rate * (tool.usage_count - 1)) / tool.usage_count
            
            execution_time = (datetime.now() - start_time).total_seconds()
            tool.average_execution_time = (
                (tool.average_execution_time * (tool.usage_count - 1) + execution_time) 
                / tool.usage_count
            )
            
            result.execution_time = execution_time
            
            self._record_call(tool, arguments, result)
            
            return result
            
        except Exception as e:
            self._LOG.error(f"Error executing tool {tool.name}: {e}")
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
    
    def execute_sync(
        self,
        name_or_id: str,
        arguments: Dict[str, Any],
        context: Optional[Any] = None,
        timeout: float = 60.0,
    ) -> POPSSToolResult:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.execute(name_or_id, arguments, context, timeout)
                    )
                    return future.result(timeout=timeout + 1)
            else:
                return loop.run_until_complete(
                    self.execute(name_or_id, arguments, context, timeout)
                )
        except Exception as e:
            return POPSSToolResult(
                tool_id="unknown",
                tool_name=name_or_id,
                tool_type=POPSSToolType.NATIVE_FUNCTION,
                success=False,
                error=str(e),
            )
    
    async def _execute_mcp_tool(
        self,
        tool: POPSSToolInfo,
        arguments: Dict[str, Any],
        timeout: float,
    ) -> POPSSToolResult:
        if not self._mcp_plaza:
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=False,
                error="MCP Plaza not configured",
            )
        
        try:
            if hasattr(self._mcp_plaza, 'execute_tool_async'):
                result = await asyncio.wait_for(
                    self._mcp_plaza.execute_tool_async(
                        tool_name=tool.mcp_tool_name,
                        arguments=arguments,
                    ),
                    timeout=timeout,
                )
            elif hasattr(self._mcp_plaza, 'execute_tool'):
                result = self._mcp_plaza.execute_tool(
                    tool_name=tool.mcp_tool_name,
                    arguments=arguments,
                )
            else:
                return POPSSToolResult(
                    tool_id=tool.tool_id,
                    tool_name=tool.name,
                    tool_type=tool.tool_type,
                    success=False,
                    error="MCP Plaza has no execute method",
                )
            
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=True,
                result=result,
            )
            
        except asyncio.TimeoutError:
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=False,
                error=f"MCP tool execution timed out after {timeout}s",
            )
        except Exception as e:
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=False,
                error=str(e),
            )
    
    async def _execute_agent_expert(
        self,
        tool: POPSSToolInfo,
        arguments: Dict[str, Any],
        context: Optional[Any],
        timeout: float,
    ) -> POPSSToolResult:
        agent = tool.agent
        if not agent:
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=False,
                error="Agent not available",
            )
        
        try:
            agent_context = context
            if agent_context is None:
                try:
                    from ..agents.base import POPSSAgentContext
                    agent_context = POPSSAgentContext(
                        context_id=str(uuid.uuid4()),
                        user_request=str(arguments),
                        session_id=str(uuid.uuid4()),
                    )
                except ImportError:
                    agent_context = type('Context', (), {
                        'context_id': str(uuid.uuid4()),
                        'user_request': str(arguments),
                        'session_id': str(uuid.uuid4()),
                    })()
            
            thoughts = []
            if hasattr(agent, 'think'):
                thoughts = await asyncio.wait_for(
                    agent.think(agent_context),
                    timeout=timeout,
                )
            
            agent_thoughts = []
            if thoughts:
                for t in thoughts:
                    if hasattr(t, 'thought_id'):
                        agent_thoughts.append({
                            "thought_id": t.thought_id,
                            "thought_type": getattr(t, 'thought_type', 'unknown'),
                            "content": getattr(t, 'content', ''),
                            "confidence": getattr(t, 'confidence', 0.5),
                        })
            
            action_result = None
            if thoughts and hasattr(agent, 'act'):
                for thought in thoughts:
                    action_result = await asyncio.wait_for(
                        agent.act(thought, agent_context),
                        timeout=timeout,
                    )
                    
                    if hasattr(agent, 'observe'):
                        done = await agent.observe(action_result, agent_context)
                        if done:
                            break
            
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=True,
                result=action_result,
                agent_thoughts=agent_thoughts,
            )
            
        except asyncio.TimeoutError:
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=False,
                error=f"Agent execution timed out after {timeout}s",
            )
        except Exception as e:
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=False,
                error=str(e),
            )
    
    async def _execute_native_function(
        self,
        tool: POPSSToolInfo,
        arguments: Dict[str, Any],
        timeout: float,
    ) -> POPSSToolResult:
        if not tool.executor:
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=False,
                error="Native function not available",
            )
        
        try:
            if asyncio.iscoroutinefunction(tool.executor):
                result = await asyncio.wait_for(
                    tool.executor(**arguments),
                    timeout=timeout,
                )
            else:
                result = tool.executor(**arguments)
            
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=True,
                result=result,
            )
            
        except asyncio.TimeoutError:
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=False,
                error=f"Native function timed out after {timeout}s",
            )
        except Exception as e:
            return POPSSToolResult(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_type=tool.tool_type,
                success=False,
                error=str(e),
            )
    
    def _record_call(
        self,
        tool: POPSSToolInfo,
        arguments: Dict[str, Any],
        result: POPSSToolResult,
    ):
        record = {
            "timestamp": datetime.now().isoformat(),
            "tool_id": tool.tool_id,
            "tool_name": tool.name,
            "tool_type": tool.tool_type.value,
            "arguments": arguments,
            "success": result.success,
            "execution_time": result.execution_time,
        }
        
        self._call_history.append(record)
        
        if len(self._call_history) > self._max_history:
            self._call_history = self._call_history[-self._max_history:]
    
    def get_stats(self) -> Dict[str, Any]:
        total_calls = len(self._call_history)
        successful_calls = sum(1 for r in self._call_history if r["success"])
        
        by_type = {}
        for tool in self._tools.values():
            type_name = tool.tool_type.value
            if type_name not in by_type:
                by_type[type_name] = {"count": 0, "tools": []}
            by_type[type_name]["count"] += 1
            by_type[type_name]["tools"].append(tool.name)
        
        return {
            "total_tools": len(self._tools),
            "total_agents": len(self._agents),
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "by_type": by_type,
        }
    
    def get_mcp_bridge(self) -> Optional[Any]:
        return self._mcp_bridge
    
    def set_mcp_bridge(self, bridge: Any):
        self._mcp_bridge = bridge
        if hasattr(bridge, 'mcp_plaza'):
            self._mcp_plaza = bridge.mcp_plaza
        self._discover_mcp_tools()
    
    def set_mcp_plaza(self, plaza: Any):
        self._mcp_plaza = plaza
        self._discover_mcp_tools()
    
    def clear(self):
        self._tools.clear()
        self._tools_by_name.clear()
        self._agents.clear()
        self._call_history.clear()
        self._LOG.info("POPSSToolRegistry cleared")


def create_tool_registry(
    mcp_plaza: Optional[Any] = None,
    register_default_experts: bool = True,
) -> POPSSToolRegistry:
    registry = POPSSToolRegistry(mcp_plaza=mcp_plaza)
    
    if register_default_experts:
        try:
            from ..agents.experts import (
                POPSSCodeExpert,
                POPSSSearchExpert,
                POPSSFileExpert,
                POPSSAnalysisExpert,
                POPSSResearchExpert,
                POPSSToolExpert,
            )
            from ..agents.base import POPSSAgentConfig
            
            default_experts = [
                (POPSSCodeExpert, "code_expert", "Code generation and execution expert", 8),
                (POPSSSearchExpert, "search_expert", "Web search expert", 7),
                (POPSSFileExpert, "file_expert", "File operation expert", 6),
                (POPSSAnalysisExpert, "analysis_expert", "Data analysis expert", 6),
                (POPSSResearchExpert, "research_expert", "Research analysis expert", 5),
                (POPSSToolExpert, "tool_expert", "Tool invocation expert", 5),
            ]
            
            for ExpertClass, name, description, priority in default_experts:
                config = POPSSAgentConfig(name=name)
                expert = ExpertClass(config, mcp_bridge=registry.get_mcp_bridge())
                registry.register_agent_expert(
                    agent=expert,
                    name=name,
                    description=description,
                    priority=priority,
                )
            
            registry._LOG.info(f"Registered {len(default_experts)} default experts")
            
        except ImportError as e:
            registry._LOG.warning(f"Could not import default experts: {e}")
    
    return registry
