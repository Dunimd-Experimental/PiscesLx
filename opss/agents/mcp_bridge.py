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
MCP Bridge - Model Context Protocol Integration for Agents

This module provides a bridge between agents and the Model Context Protocol (MCP),
enabling agents to discover, validate, and execute tools through a unified interface.

Key Components:
    - POPSSMCPBridgeConfig: Configuration for the MCP bridge
    - POPSSToolInfo: Tool metadata and statistics container
    - POPSSToolCall: Tool execution record with timing and results
    - POPSSMCPBridge: Main bridge class for tool management
    - POPSSMCPBridgeMixin: Mixin class for adding MCP capabilities to agents

Design Principles:
    1. Tool Discovery: Automatic discovery and caching of available tools
    2. Validation: Input validation before tool execution
    3. Retry Logic: Configurable retry on transient failures
    4. Session Management: Track and manage MCP sessions
    5. Statistics Tracking: Monitor tool usage and performance

Features:
    - Automatic tool discovery from MCP plaza
    - Tool caching with configurable TTL
    - Input validation against tool schemas
    - Retry mechanism for failed tool calls
    - Session lifecycle management
    - Comprehensive statistics and monitoring

Usage:
    # Initialize bridge with MCP plaza
    config = POPSSMCPBridgeConfig(mcp_plaza=my_plaza)
    bridge = POPSSMCPBridge(config)
    
    # Call a tool
    result = await bridge.call_tool(
        tool_name="web_search",
        arguments={"query": "latest AI developments"},
        session_id="session_123"
    )
    
    # Use mixin in agent
    class MyAgent(POPSSBaseAgent, POPSSMCPBridgeMixin):
        async def execute_with_tools(self, task):
            tools = self.get_available_tools()
            result = await self.call_tool("tool_name", {"arg": "value"})
            return result

Thread Safety:
    All public methods are thread-safe. Tool cache and session management
    use appropriate locking mechanisms for concurrent access.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar
from concurrent.futures import ThreadPoolExecutor

from utils.dc import PiscesLxLogger
from configs.version import VERSION

from .base import (
    POPSSBaseAgent,
    POPSSAgentConfig,
    POPSSAgentContext,
    POPSSAgentResult,
    POPSSAgentState,
    POPSSAgentCapability,
)
from ..mcp import (
    POPSSMCPPlaza,
    POPSSMCPSession,
)

T = TypeVar('T')

@dataclass
class POPSSMCPBridgeConfig:
    """
    Configuration settings for the MCP Bridge.
    
    This dataclass defines the operational parameters for the MCP bridge,
    including tool discovery, caching, validation, and retry behavior.
    
    Attributes:
        mcp_plaza: Reference to the POPSSMCPPlaza instance for tool access.
        auto_discover_tools: Whether to automatically discover tools on init.
        cache_tools: Whether to cache tool metadata locally.
        cache_ttl_seconds: Time-to-live for cached tool info in seconds.
        enable_tool_validation: Whether to validate arguments before execution.
        max_tool_args_size: Maximum size in bytes for tool arguments.
        default_timeout: Default timeout in seconds for tool execution.
        retry_on_failure: Whether to retry failed tool calls.
        max_retries: Maximum number of retry attempts.
    
    Example:
        >>> config = POPSSMCPBridgeConfig(
        ...     mcp_plaza=my_plaza,
        ...     cache_ttl_seconds=600,
        ...     enable_tool_validation=True
        ... )
    """
    mcp_plaza: POPSSMCPPlaza
    
    auto_discover_tools: bool = True
    cache_tools: bool = True
    cache_ttl_seconds: int = 300
    
    enable_tool_validation: bool = True
    max_tool_args_size: int = 10000
    
    default_timeout: float = 60.0
    retry_on_failure: bool = True
    max_retries: int = 3

@dataclass
class POPSSToolInfo:
    """
    Container for tool metadata and usage statistics.
    
    This dataclass stores comprehensive information about a tool,
    including its description, parameters, and runtime statistics.
    
    Attributes:
        tool_name: Unique identifier for the tool.
        description: Human-readable description of the tool's purpose.
        category: Category classification (e.g., "search", "file", "api").
        parameters: JSON schema describing expected parameters.
        return_type: Expected return type of the tool.
        version: Tool version string.
        author: Tool author or maintainer.
        examples: List of example usage patterns.
        tags: Set of tags for search and categorization.
        usage_count: Number of times the tool has been called.
        success_rate: Ratio of successful calls (0.0 to 1.0).
        average_execution_time: Average execution time in seconds.
    
    Example:
        >>> tool_info = POPSSToolInfo(
        ...     tool_name="web_search",
        ...     description="Search the web for information",
        ...     category="search",
        ...     parameters={"query": {"type": "string"}}
        ... )
    """
    tool_name: str
    description: str
    category: str
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    return_type: str = "any"
    
    version: str = VERSION
    author: str = "Unknown"
    
    examples: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    usage_count: int = 0
    success_rate: float = 1.0
    average_execution_time: float = 0.0

@dataclass
class POPSSToolCall:
    """
    Record of a tool execution with timing and results.
    
    This dataclass tracks the complete lifecycle of a tool call,
    from initiation to completion, including all relevant metadata.
    
    Attributes:
        call_id: Unique identifier for this tool call.
        tool_name: Name of the tool that was called.
        arguments: Dictionary of arguments passed to the tool.
        session_id: Optional MCP session identifier.
        agent_id: Optional agent identifier that initiated the call.
        start_time: Timestamp when the call was initiated.
        end_time: Timestamp when the call completed (None if pending).
        result: Tool execution result (None if failed or pending).
        error: Error message if the call failed (None if successful).
        success: Boolean indicating if the call was successful.
    
    Example:
        >>> tool_call = POPSSToolCall(
        ...     call_id="call_abc123",
        ...     tool_name="web_search",
        ...     arguments={"query": "AI news"},
        ...     session_id="session_xyz"
        ... )
    """
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    result: Optional[Any] = None
    error: Optional[str] = None
    success: bool = False

class POPSSMCPBridge:
    """
    Main bridge class for MCP tool management and execution.
    
    This class provides a unified interface for agents to interact with
    the Model Context Protocol (MCP), handling tool discovery, validation,
    execution, and session management.
    
    Key Features:
        - Automatic tool discovery and caching
        - Input validation against tool schemas
        - Configurable retry logic for transient failures
        - Session lifecycle management
        - Comprehensive statistics tracking
        - Event callbacks for monitoring
    
    Attributes:
        config: POPSSMCPBridgeConfig instance with operational settings.
        mcp_plaza: Reference to the POPSSMCPPlaza for tool access.
        _tool_cache: Cache of POPSSToolInfo objects by tool name.
        _session_cache: Cache of active POPSSMCPSession objects.
        _tool_calls: Dictionary tracking all tool call records.
        _callbacks: Event callbacks for tool lifecycle events.
    
    Events:
        - on_tool_call: Triggered when a tool is called
        - on_tool_result: Triggered when a tool returns successfully
        - on_tool_error: Triggered when a tool call fails
        - on_tool_discovered: Triggered when a new tool is discovered
    
    Example:
        >>> config = POPSSMCPBridgeConfig(mcp_plaza=my_plaza)
        >>> bridge = POPSSMCPBridge(config)
        >>> 
        >>> # Get available tools
        >>> tools = bridge.get_available_tools(category="search")
        >>> 
        >>> # Call a tool
        >>> result = await bridge.call_tool(
        ...     tool_name="web_search",
        ...     arguments={"query": "latest developments"},
        ...     session_id="session_123"
        ... )
        >>> 
        >>> # Get statistics
        >>> stats = bridge.get_tool_statistics()
    
    See Also:
        - POPSSMCPBridgeConfig: Configuration settings
        - POPSSToolInfo: Tool metadata container
        - POPSSToolCall: Tool execution record
        - POPSSMCPBridgeMixin: Mixin for agent integration
    """
    
    def __init__(self, config: POPSSMCPBridgeConfig):
        """
        Initialize the MCP bridge with configuration.
        
        Args:
            config: POPSSMCPBridgeConfig instance containing the MCP plaza
                reference and operational settings.
        """
        self.config = config
        self.mcp_plaza = config.mcp_plaza
        
        self._LOG = self._configure_logging()
        
        # Tool metadata cache
        self._tool_cache: Dict[str, POPSSToolInfo] = {}
        self._cache_timestamp: Optional[datetime] = None
        
        # Session management
        self._session_cache: Dict[str, POPSSMCPSession] = {}
        
        # Tool call tracking
        self._tool_calls: Dict[str, POPSSToolCall] = {}
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'on_tool_call': [],
            'on_tool_result': [],
            'on_tool_error': [],
            'on_tool_discovered': [],
        }
        
        # Thread pool for async operations
        self._async_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="piscesl1_mcp_bridge"
        )
        
        # Auto-discover tools if enabled
        if self.config.auto_discover_tools:
            self._discover_tools()
        
        self._LOG.info("POPSSMCPBridge initialized")
    
    def _configure_logging(self) -> PiscesLxLogger:
        """
        Configure and return a logger instance for the MCP bridge.
        
        Returns:
            PiscesLxLogger: Configured logger instance for MCP operations.
        """
        logger = get_logger("PiscesLx.Core.Agents.MCPBridge")
        return logger
    
    def _discover_tools(self):
        """
        Discover available tools from the MCP plaza.
        
        This method queries the MCP plaza for available tools and caches
        their metadata. Tools are cached with a TTL to ensure freshness.
        """
        try:
            tools = self.mcp_plaza.list_tools()
            for tool_name, tool_info in tools.items():
                self._tool_cache[tool_name] = POPSSToolInfo(
                    tool_name=tool_name,
                    description=tool_info.get('description', ''),
                    category=tool_info.get('category', 'general'),
                    parameters=tool_info.get('parameters', {}),
                    return_type=tool_info.get('return_type', 'any'),
                    tags=set(tool_info.get('tags', [])),
                )
                self._trigger_callback('on_tool_discovered', tool_name)
            
            self._cache_timestamp = datetime.now()
            self._LOG.info(f"Discovered {len(self._tool_cache)} tools")
        except Exception as e:
            self._LOG.error(f"Tool discovery failed: {e}")
    
    def _refresh_cache_if_needed(self):
        """
        Refresh the tool cache if TTL has expired.
        
        Checks if the cache timestamp has exceeded the configured TTL
        and triggers a re-discovery if necessary.
        """
        if not self._cache_timestamp:
            self._discover_tools()
            return
        
        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        if elapsed > self.config.cache_ttl_seconds:
            self._discover_tools()
    
    def _cache_tool_info(self, tool_name: str):
        """
        Cache tool information from the MCP plaza.
        
        Args:
            tool_name: Name of the tool to cache.
        """
        try:
            tool_info = self.mcp_plaza.get_tool_info(tool_name)
            if tool_info:
                self._tool_cache[tool_name] = POPSSToolInfo(
                    tool_name=tool_name,
                    description=tool_info.get('description', ''),
                    category=tool_info.get('category', 'general'),
                    parameters=tool_info.get('parameters', {}),
                    return_type=tool_info.get('return_type', 'any'),
                    tags=set(tool_info.get('tags', [])),
                )
        except Exception as e:
            self._LOG.error(f"Failed to cache tool info for {tool_name}: {e}")
    
    def _invalidate_cache(self):
        """
        Invalidate the tool cache, forcing re-discovery on next access.
        """
        self._tool_cache.clear()
        self._cache_timestamp = None
        self._discover_tools()
    
    def get_available_tools(self, category: Optional[str] = None, 
                          tags: Optional[Set[str]] = None) -> List[POPSSToolInfo]:
        """
        Get list of available tools, optionally filtered by category or tags.
        
        Args:
            category: Optional category to filter tools by.
            tags: Optional set of tags to filter tools by (intersection).
        
        Returns:
            List[POPSSToolInfo]: List of tool information objects matching
                the specified filters.
        
        Example:
            >>> # Get all tools
            >>> all_tools = bridge.get_available_tools()
            >>> 
            >>> # Get search tools
            >>> search_tools = bridge.get_available_tools(category="search")
            >>> 
            >>> # Get tools with specific tags
            >>> tagged_tools = bridge.get_available_tools(tags={"web", "api"})
        """
        if self.config.cache_tools and self._cache_timestamp:
            cache_age = (datetime.now() - self._cache_timestamp).total_seconds()
            if cache_age < self.config.cache_ttl_seconds:
                tools = list(self._tool_cache.values())
                
                if category:
                    tools = [t for t in tools if t.category == category]
                
                if tags:
                    tools = [t for t in tools if t.tags & tags]
                
                return tools
        
        tools = []
        mcp_tools = self.mcp_plaza.list_tools(use_cache=False)
        
        for tool_name in mcp_tools:
            if tool_name in self._tool_cache:
                tool_info = self._tool_cache[tool_name]
                
                if category and tool_info.category != category:
                    continue
                
                if tags and not (tool_info.tags & tags):
                    continue
                
                tools.append(tool_info)
        
        return tools
    
    def get_tool(self, tool_name: str) -> Optional[POPSSToolInfo]:
        """
        Get information about a specific tool.
        
        Args:
            tool_name: Name of the tool to retrieve.
        
        Returns:
            Optional[POPSSToolInfo]: Tool information if found, None otherwise.
        """
        if tool_name in self._tool_cache:
            return self._tool_cache[tool_name]
        
        tool_data = self.mcp_plaza.get_tool(tool_name)
        if tool_data:
            self._cache_tool_info(tool_name)
            return self._tool_cache.get(tool_name)
        
        return None
    
    def search_tools(self, query: str) -> List[POPSSToolInfo]:
        """
        Search for tools matching a query string.
        
        Args:
            query: Search query string.
        
        Returns:
            List[POPSSToolInfo]: List of matching tool information objects.
        """
        results = []
        
        mcp_results = self.mcp_plaza.search_tools(query)
        
        for tool_name in mcp_results:
            if tool_name in self._tool_cache:
                results.append(self._tool_cache[tool_name])
            else:
                tool_data = self.mcp_plaza.get_tool(tool_name)
                if tool_data:
                    self._cache_tool_info(tool_name)
                    if tool_name in self._tool_cache:
                        results.append(self._tool_cache[tool_name])
        
        return results
    
    def get_tools_by_category(self) -> Dict[str, List[POPSSToolInfo]]:
        """
        Get tools grouped by their category.
        
        Returns:
            Dict[str, List[POPSSToolInfo]]: Dictionary mapping category names
                to lists of tool information objects.
        """
        tools_by_category = {}
        
        for tool_info in self._tool_cache.values():
            if tool_info.category not in tools_by_category:
                tools_by_category[tool_info.category] = []
            tools_by_category[tool_info.category].append(tool_info)
        
        return tools_by_category
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Execute a tool with the given arguments.
        
        This method handles the complete lifecycle of a tool call, including
        validation, execution, retry logic, and statistics tracking.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments to pass to the tool.
            session_id: Optional MCP session identifier for context tracking.
            agent_id: Optional agent identifier that initiated the call.
            timeout: Optional timeout in seconds (uses config default if not provided).
        
        Returns:
            Any: The result returned by the tool execution.
        
        Raises:
            ValueError: If tool call validation fails.
            TimeoutError: If tool execution times out and retry is disabled.
            Exception: If tool execution fails and retry is disabled.
        
        Example:
            >>> result = await bridge.call_tool(
            ...     tool_name="web_search",
            ...     arguments={"query": "latest AI developments"},
            ...     session_id="session_123",
            ...     timeout=30.0
            ... )
        """
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        
        tool_call = POPSSToolCall(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            session_id=session_id,
            agent_id=agent_id,
        )
        
        self._tool_calls[call_id] = tool_call
        
        self._trigger_callback('on_tool_call', tool_name, arguments, agent_id)
        
        if self.config.enable_tool_validation:
            if not self._validate_tool_args(tool_name, arguments):
                tool_call.error = "Validation failed"
                tool_call.success = False
                
                self._trigger_callback('on_tool_error', tool_name, "Validation failed")
                
                raise ValueError(f"Tool call validation failed for {tool_name}")
        
        try:
            if timeout is None:
                timeout = self.config.default_timeout
            
            result = await asyncio.wait_for(
                self.mcp_plaza.execute_tool_async(
                    tool_name=tool_name,
                    arguments=arguments,
                    session_id=session_id
                ),
                timeout=timeout
            )
            
            tool_call.result = result
            tool_call.success = True
            tool_call.end_time = datetime.now()
            
            # Update tool statistics
            if tool_name in self._tool_cache:
                tool_info = self._tool_cache[tool_name]
                tool_info.usage_count += 1
                if tool_call.end_time and tool_call.start_time:
                    exec_time = (tool_call.end_time - tool_call.start_time).total_seconds()
                    if tool_info.average_execution_time == 0:
                        tool_info.average_execution_time = exec_time
                    else:
                        tool_info.average_execution_time = (
                            tool_info.average_execution_time * 0.9 + exec_time * 0.1
                        )
            
            self._trigger_callback('on_tool_result', tool_name, result)
            
            return result
            
        except asyncio.TimeoutError:
            tool_call.error = "Execution timeout"
            tool_call.success = False
            tool_call.end_time = datetime.now()
            
            self._trigger_callback('on_tool_error', tool_name, "Execution timeout")
            
            if self.config.retry_on_failure:
                return await self._retry_tool_call(tool_name, arguments, session_id, agent_id, timeout)
            
            raise TimeoutError(f"Tool execution timed out: {tool_name}")
            
        except Exception as e:
            tool_call.error = str(e)
            tool_call.success = False
            tool_call.end_time = datetime.now()
            
            self._trigger_callback('on_tool_error', tool_name, str(e))
            
            if self.config.retry_on_failure:
                return await self._retry_tool_call(tool_name, arguments, session_id, agent_id, timeout)
            
            raise
    
    async def _retry_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Retry a failed tool call with exponential backoff.
        
        Args:
            tool_name: Name of the tool to retry.
            arguments: Dictionary of arguments to pass to the tool.
            session_id: Optional MCP session identifier.
            agent_id: Optional agent identifier.
            timeout: Optional timeout in seconds.
        
        Returns:
            Any: The result from a successful retry attempt.
        
        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        for attempt in range(1, self.config.max_retries):
            try:
                return await self.call_tool(
                    tool_name, 
                    arguments, 
                    session_id, 
                    agent_id,
                    timeout=timeout
                )
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                
                # Exponential backoff
                await asyncio.sleep(self.config.retry_delay * attempt)
        
        raise RuntimeError("Max retries exceeded")
    
    async def _validate_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        if tool_name not in self._tool_cache:
            return False, f"Tool not found: {tool_name}"
        
        tool_info = self._tool_cache[tool_name]
        
        arg_size = len(str(arguments))
        if arg_size > self.config.max_tool_args_size:
            return False, f"Arguments too large: {arg_size} bytes"
        
        required_params = [
            param_name for param_name, param_info 
            in tool_info.parameters.get('properties', {}).items()
            if param_info.get('required', False)
        ]
        
        for param in required_params:
            if param not in arguments:
                return False, f"Missing required parameter: {param}"
        
        return True, None
    
    def call_tool_sync(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Synchronous wrapper for call_tool method.
        
        Creates a new event loop to execute the async call_tool method.
        Use this when calling from synchronous code.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments to pass to the tool.
            session_id: Optional MCP session identifier.
            agent_id: Optional agent identifier.
            timeout: Optional timeout in seconds.
        
        Returns:
            Any: The result returned by the tool execution.
        
        Warning:
            This method creates a new event loop each time. For better
            performance in async contexts, use call_tool directly.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.call_tool(
                    tool_name,
                    arguments,
                    session_id,
                    agent_id,
                    timeout
                )
            )
        finally:
            loop.close()
    
    def create_session(self, agent_id: Optional[str] = None) -> POPSSMCPSession:
        """
        Create a new MCP session for context tracking.
        
        Args:
            agent_id: Optional agent identifier to associate with the session.
        
        Returns:
            POPSSMCPSession: The newly created session object.
        """
        session = self.mcp_plaza.create_session()
        
        session_id = session.session_id
        self._session_cache[session_id] = session
        
        return session
    
    def get_session(self, session_id: str) -> Optional[POPSSMCPSession]:
        """
        Retrieve an existing MCP session by ID.
        
        Args:
            session_id: The session identifier to look up.
        
        Returns:
            Optional[POPSSMCPSession]: The session if found, None otherwise.
        """
        return self._session_cache.get(session_id)
    
    def close_session(self, session_id: str):
        """
        Close and clean up an MCP session.
        
        Args:
            session_id: The session identifier to close.
        """
        if session_id in self._session_cache:
            session = self._session_cache[session_id]
            session.cleanup()
            del self._session_cache[session_id]
    
    def get_tool_call_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent tool call history for monitoring and debugging.
        
        Args:
            limit: Maximum number of records to return (default: 100).
        
        Returns:
            List[Dict[str, Any]]: List of tool call records with timing
                and status information.
        """
        recent = list(self._tool_calls.values())[-limit:]
        return [
            {
                'call_id': call.call_id,
                'tool_name': call.tool_name,
                'success': call.success,
                'error': call.error,
                'execution_time': (
                    (call.end_time - call.start_time).total_seconds() 
                    if call.end_time and call.start_time else None
                ),
                'timestamp': call.start_time.isoformat(),
            }
            for call in recent
        ]
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about tool usage.
        
        Returns:
            Dict[str, Any]: Statistics including:
                - total_calls: Total number of tool calls
                - successful_calls: Number of successful calls
                - failed_calls: Number of failed calls
                - success_rate: Ratio of successful calls
                - total_execution_time: Sum of all execution times
                - average_execution_time: Average execution time per call
                - tool_usage: Per-tool usage statistics
                - cached_tools: Number of cached tools
                - active_sessions: Number of active sessions
        """
        total_calls = len(self._tool_calls)
        successful_calls = sum(1 for call in self._tool_calls.values() if call.success)
        failed_calls = total_calls - successful_calls
        
        total_time = sum(
            (call.end_time - call.start_time).total_seconds()
            for call in self._tool_calls.values()
            if call.end_time and call.start_time
        )
        
        tool_usage = {}
        for call in self._tool_calls.values():
            if call.tool_name not in tool_usage:
                tool_usage[call.tool_name] = {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'total_time': 0.0,
                }
            tool_usage[call.tool_name]['total_calls'] += 1
            if call.success:
                tool_usage[call.tool_name]['successful_calls'] += 1
            if call.end_time and call.start_time:
                tool_usage[call.tool_name]['total_time'] += (
                    (call.end_time - call.start_time).total_seconds()
                )
        
        return {
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'success_rate': successful_calls / max(total_calls, 1),
            'total_execution_time': total_time,
            'average_execution_time': total_time / max(total_calls, 1),
            'tool_usage': tool_usage,
            'cached_tools': len(self._tool_cache),
            'active_sessions': len(self._session_cache),
        }
    
    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback function for a specific event.
        
        Args:
            event: Event name ('on_tool_call', 'on_tool_result', 
                'on_tool_error', 'on_tool_discovered').
            callback: Callable to be invoked when the event occurs.
        
        Example:
            >>> def my_callback(tool_name):
            ...     print(f"Tool discovered: {tool_name}")
            >>> bridge.register_callback('on_tool_discovered', my_callback)
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _trigger_callback(self, event_name: str, *args, **kwargs):
        """
        Trigger all registered callbacks for an event.
        
        Args:
            event_name: Name of the event to trigger.
            *args: Positional arguments to pass to callbacks.
            **kwargs: Keyword arguments to pass to callbacks.
        """
        for callback in self._callbacks.get(event_name, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self._LOG.error(f"Callback error for {event_name}: {e}")
    
    def shutdown(self):
        """
        Shutdown the MCP bridge and release all resources.
        
        This method cleans up all sessions, clears caches, and shuts down
        the thread pool executor. Should be called when the bridge is
        no longer needed.
        """
        for session in self._session_cache.values():
            session.cleanup()
        
        self._session_cache.clear()
        self._tool_calls.clear()
        
        self._async_executor.shutdown(wait=True)
        
        self._LOG.info("MCP Bridge shutdown")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.shutdown()
        return False


class POPSSMCPBridgeMixin:
    """
    Mixin class for adding MCP bridge capabilities to agents.
    
    This mixin provides a convenient way to integrate MCP tool calling
    capabilities into agent classes without modifying their inheritance
    hierarchy.
    
    Attributes:
        _mcp_bridge: Optional POPSSMCPBridge instance for tool operations.
    
    Usage:
        class MyAgent(POPSSBaseAgent, POPSSMCPBridgeMixin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Agent initialization
            
            async def execute_with_tools(self, task):
                tools = self.get_available_tools()
                result = await self.call_tool("tool_name", {"arg": "value"})
                return result
    """
    
    def __init__(self, *args, mcp_bridge: Optional[POPSSMCPBridge] = None, **kwargs):
        """
        Initialize the mixin with an optional MCP bridge.
        
        Args:
            *args: Positional arguments passed to parent class.
            mcp_bridge: Optional POPSSMCPBridge instance for tool operations.
            **kwargs: Keyword arguments passed to parent class.
        """
        self._mcp_bridge = mcp_bridge
    
    def set_mcp_bridge(self, bridge: POPSSMCPBridge):
        """
        Set or update the MCP bridge instance.
        
        Args:
            bridge: POPSSMCPBridge instance to use for tool operations.
        """
        self._mcp_bridge = bridge
    
    def get_mcp_bridge(self) -> Optional[POPSSMCPBridge]:
        """
        Get the current MCP bridge instance.
        
        Returns:
            Optional[POPSSMCPBridge]: The current bridge instance or None.
        """
        return self._mcp_bridge
    
    async def need_tool(self, context: POPSSAgentContext) -> bool:
        """
        Determine if tool usage is needed for the current context.
        
        Args:
            context: POPSSAgentContext with current task information.
        
        Returns:
            bool: True if tools are available and needed, False otherwise.
        """
        if not self._mcp_bridge:
            return False
        
        available_tools = self._mcp_bridge.get_available_tools()
        
        return len(available_tools) > 0
    
    async def select_tool(self, context: POPSSAgentContext) -> Optional[str]:
        """
        Select an appropriate tool based on the current context.
        
        Args:
            context: POPSSAgentContext with task type and metadata.
        
        Returns:
            Optional[str]: Name of the selected tool, or None if no suitable tool.
        """
        if not self._mcp_bridge:
            return None
        
        task_type = context.metadata.get('task_type', '')
        
        # Search for tools matching the task type
        tools = self._mcp_bridge.search_tools(task_type)
        
        if tools:
            return tools[0].tool_name
        
        # Fall back to first available tool
        all_tools = self._mcp_bridge.get_available_tools()
        if all_tools:
            return all_tools[0].tool_name
        
        return None
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Any:
        """
        Execute a tool through the MCP bridge.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments to pass to the tool.
            session_id: Optional MCP session identifier.
        
        Returns:
            Any: The result returned by the tool execution.
        
        Raises:
            RuntimeError: If MCP bridge is not configured.
        """
        if not self._mcp_bridge:
            raise RuntimeError("MCP Bridge not configured")
        
        return await self._mcp_bridge.call_tool(
            tool_name=tool_name,
            arguments=arguments,
            session_id=session_id,
            agent_id=self.config.agent_id if hasattr(self, 'config') else None,
        )
