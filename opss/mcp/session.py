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
MCP Session - Model Context Protocol Session Management

This module provides session management for the Model Context Protocol (MCP),
handling execution contexts, memory, caching, and statistics for tool usage.

Key Components:
    - POPSSMCPSession: Main session class for MCP operations

Features:
    - Execution context tracking with timing and results
    - Session memory with tool-specific histories
    - Global variable storage for cross-tool state
    - Tool result caching with access tracking
    - Usage statistics per tool
    - Session timeout and cleanup
    - Thread-safe operations with RLock

Usage:
    # Create a new session
    session = POPSSMCPSession(config={'session_timeout': 3600})
    
    # Create execution context for a tool call
    exec_id = session.create_execution_context("web_search", {"query": "AI news"})
    
    # Update with results
    session.update_execution_context(exec_id, result={"results": [...]})
    
    # Get session info
    info = session.get_session_info()
    
    # Cleanup when done
    session.cleanup_session()

Thread Safety:
    All public methods are thread-safe using RLock for internal state protection.
"""

import uuid
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Callable
from collections import defaultdict

from utils.dc import PiscesLxLogger

from .types import (
    POPSSMCPSessionMemory,
    POPSSMCPExecutionContext,
    POPSSMCPModuleStats,
    POPSSMCPModuleStatus
)

class POPSSMCPSession:
    """
    MCP Session for managing tool execution contexts and state.
    
    This class provides comprehensive session management for MCP operations,
    including execution tracking, memory management, caching, and statistics.
    
    Attributes:
        session_id: Unique identifier for this session.
        config: Configuration dictionary for session behavior.
        created_at: Timestamp when the session was created.
        last_activity: Timestamp of the most recent activity.
        is_active: Boolean indicating if the session is active.
        memory: POPSSMCPSessionMemory for storing execution history.
        execution_contexts: Dictionary of execution contexts by ID.
        tool_usage_stats: Dictionary of statistics per tool.
        global_variables: Dictionary for cross-tool state storage.
        tool_cache: Dictionary for caching tool results.
        active_tools: Set of currently executing tool names.
    
    Configuration Options:
        - max_executions: Maximum execution contexts (default: 1000)
        - session_timeout: Session timeout in seconds (default: 3600)
        - memory_limit: Maximum memory entries (default: 100)
        - allow_concurrent_tools: Allow same tool concurrent execution (default: True)
        - max_memory_executions: Maximum recent executions in memory (default: 100)
        - max_tool_memory: Maximum entries per tool memory (default: 20)
    
    Example:
        >>> session = POPSSMCPSession()
        >>> exec_id = session.create_execution_context("tool_name", {"arg": "value"})
        >>> # ... execute tool ...
        >>> session.update_execution_context(exec_id, result={"output": "data"})
        >>> stats = session.get_tool_usage_stats()
    """
    
    def __init__(self, session_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new MCP session.
        
        Args:
            session_id: Optional session ID. If not provided, a UUID will be generated.
            config: Optional configuration dictionary for session behavior.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.config = config or {}
        self._LOG = self._configure_logging()
        
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.is_active = True
        
        # Session memory and tracking
        self.memory = POPSSMCPSessionMemory(session_id=self.session_id)
        self.execution_contexts: Dict[str, POPSSMCPExecutionContext] = {}
        self.tool_usage_stats: Dict[str, POPSSMCPModuleStats] = {}
        
        # State storage
        self.global_variables: Dict[str, Any] = {}
        self.tool_cache: Dict[str, Any] = {}
        self.active_tools: Set[str] = set()
        
        # Thread safety
        self._lock = threading.RLock()
        self._session_lock = threading.Lock()
        
        # Configuration with defaults
        self.max_executions = getattr(self.config, 'max_executions', 1000)
        self.session_timeout = getattr(self.config, 'session_timeout', 3600)
        self.memory_limit = getattr(self.config, 'memory_limit', 100)
        
        self._LOG.info(f"Session initialized: {self.session_id}")
    
    def _configure_logging(self) -> PiscesLxLogger:
        """
        Configure and return a logger instance for this session.
        
        Returns:
            PiscesLxLogger: Configured logger instance.
        """
        logger = get_logger(f"PiscesLx.Core.MCP.Session.{self.session_id[:8]}")
        return logger
    
    def create_execution_context(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Create a new execution context for a tool call.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments to pass to the tool.
        
        Returns:
            str: Unique execution ID for tracking this execution.
        
        Raises:
            RuntimeError: If maximum executions reached or tool already executing
                (when concurrent execution is disabled).
        """
        with self._lock:
            execution_id = f"exec_{int(time.time() * 1000000)}_{tool_name}"
            
            if len(self.execution_contexts) >= self.max_executions:
                raise RuntimeError(f"Maximum executions ({self.max_executions}) reached for session")
            
            if tool_name in self.active_tools and not getattr(self.config, 'allow_concurrent_tools', True):
                raise RuntimeError(f"Tool {tool_name} is already executing in this session")
            
            context = POPSSMCPExecutionContext(
                session_id=self.session_id,
                tool_name=tool_name,
                arguments=arguments,
                start_time=datetime.now()
            )
            
            self.execution_contexts[execution_id] = context
            self.active_tools.add(tool_name)
            self.last_activity = datetime.now()
            
            # Initialize tool stats if not exists
            if tool_name not in self.tool_usage_stats:
                self.tool_usage_stats[tool_name] = POPSSMCPModuleStats(
                    name=tool_name,
                    load_time=0.0,
                    tool_count=1,
                    status=POPSSMCPModuleStatus.SUCCESS
                )
            
            self._LOG.debug(f"Execution context created: {execution_id} for {tool_name}")
            return execution_id
    
    def update_execution_context(self, execution_id: str, 
                               result: Any = None, 
                               error: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None):
        """
        Update an execution context with results or error.
        
        Args:
            execution_id: ID of the execution context to update.
            result: Optional result from the tool execution.
            error: Optional error message if execution failed.
            metadata: Optional additional metadata about the execution.
        """
        with self._lock:
            if execution_id not in self.execution_contexts:
                self._LOG.warning(f"Execution context not found: {execution_id}")
                return
            
            context = self.execution_contexts[execution_id]
            context.end_time = datetime.now()
            context.result = result
            context.error = error
            
            if metadata:
                context.performance_metrics.update(metadata)
            
            # Update tool statistics
            tool_name = context.tool_name
            if tool_name in self.tool_usage_stats:
                stats = self.tool_usage_stats[tool_name]
                if context.start_time and context.end_time:
                    exec_time = (context.end_time - context.start_time).total_seconds()
                    stats.load_time = exec_time
            
            self.active_tools.discard(tool_name)
            self._store_execution_in_memory(context)
            
            self._LOG.debug(f"Execution context updated: {execution_id}")
    
    def _store_execution_in_memory(self, context: POPSSMCPExecutionContext):
        """
        Store execution context in session memory.
        
        Args:
            context: The POPSSMCPExecutionContext to store.
        """
        if not hasattr(self.memory, 'recent_executions'):
            self.memory.recent_executions = []
        
        # Add to recent executions
        self.memory.recent_executions.append({
            'tool_name': context.tool_name,
            'arguments': context.arguments,
            'result': context.result,
            'error': context.error,
            'execution_time': (context.end_time - context.start_time).total_seconds() if context.end_time else None,
            'timestamp': context.start_time
        })
        
        # Limit recent executions
        max_executions = getattr(self.config, 'max_memory_executions', 100)
        if len(self.memory.recent_executions) > max_executions:
            self.memory.recent_executions = self.memory.recent_executions[-max_executions:]
        
        # Add to tool-specific memory
        if context.tool_name not in self.memory.tool_memories:
            self.memory.tool_memories[context.tool_name] = []
        
        self.memory.tool_memories[context.tool_name].append({
            'arguments': context.arguments,
            'result': context.result,
            'error': context.error,
            'timestamp': context.start_time
        })
        
        # Limit tool memory
        max_tool_memory = getattr(self.config, 'max_tool_memory', 20)
        if len(self.memory.tool_memories[context.tool_name]) > max_tool_memory:
            self.memory.tool_memories[context.tool_name] = self.memory.tool_memories[context.tool_name][-max_tool_memory:]
    
    def get_execution_context(self, execution_id: str) -> Optional[POPSSMCPExecutionContext]:
        """
        Retrieve an execution context by ID.
        
        Args:
            execution_id: The execution ID to look up.
        
        Returns:
            Optional[POPSSMCPExecutionContext]: The context if found, None otherwise.
        """
        with self._lock:
            return self.execution_contexts.get(execution_id)
    
    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent execution records from memory.
        
        Args:
            limit: Maximum number of records to return (default: 10).
        
        Returns:
            List[Dict[str, Any]]: List of recent execution records.
        """
        with self._lock:
            if not hasattr(self.memory, 'recent_executions'):
                return []
            return self.memory.recent_executions[-limit:] if self.memory.recent_executions else []
    
    def get_tool_memory(self, tool_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get memory records for a specific tool.
        
        Args:
            tool_name: Name of the tool to get memory for.
            limit: Maximum number of records to return (default: 10).
        
        Returns:
            List[Dict[str, Any]]: List of tool-specific memory records.
        """
        with self._lock:
            tool_mem = self.memory.tool_memories.get(tool_name, [])
            return tool_mem[-limit:] if tool_mem else []
    
    def set_global_variable(self, key: str, value: Any):
        """
        Set a global variable for cross-tool state sharing.
        
        Args:
            key: Variable name.
            value: Variable value.
        """
        with self._lock:
            self.global_variables[key] = value
            self.last_activity = datetime.now()
            self._LOG.debug(f"Global variable set: {key}")
    
    def get_global_variable(self, key: str, default: Any = None) -> Any:
        """
        Get a global variable value.
        
        Args:
            key: Variable name.
            default: Default value if not found.
        
        Returns:
            Any: The variable value or default.
        """
        with self._lock:
            return self.global_variables.get(key, default)
    
    def set_tool_cache(self, tool_name: str, cache_key: str, value: Any):
        """
        Cache a value for a specific tool.
        
        Args:
            tool_name: Name of the tool.
            cache_key: Cache key for the value.
            value: Value to cache.
        """
        with self._lock:
            if tool_name not in self.tool_cache:
                self.tool_cache[tool_name] = {}
            
            self.tool_cache[tool_name][cache_key] = {
                'value': value,
                'timestamp': datetime.now(),
                'access_count': 0
            }
            
            self.last_activity = datetime.now()
    
    def get_tool_cache(self, tool_name: str, cache_key: str, default: Any = None) -> Any:
        """
        Retrieve a cached value for a tool.
        
        Args:
            tool_name: Name of the tool.
            cache_key: Cache key for the value.
            default: Default value if not found.
        
        Returns:
            Any: The cached value or default.
        """
        with self._lock:
            if tool_name not in self.tool_cache:
                return default
            
            cache_entry = self.tool_cache[tool_name].get(cache_key)
            if cache_entry is None:
                return default
            
            # Update access count
            cache_entry['access_count'] += 1
            self.last_activity = datetime.now()
            
            return cache_entry['value']
    
    def clear_tool_cache(self, tool_name: Optional[str] = None):
        """
        Clear cache for a specific tool or all tools.
        
        Args:
            tool_name: Optional tool name. If None, clears all caches.
        """
        with self._lock:
            if tool_name:
                if tool_name in self.tool_cache:
                    del self.tool_cache[tool_name]
                    self._LOG.debug(f"Cache cleared for tool: {tool_name}")
            else:
                self.tool_cache.clear()
                self._LOG.debug("All tool caches cleared")
            
            self.last_activity = datetime.now()
    
    def get_tool_usage_stats(self, tool_name: Optional[str] = None) -> Dict[str, POPSSMCPModuleStats]:
        """
        Get usage statistics for tools.
        
        Args:
            tool_name: Optional tool name. If None, returns all stats.
        
        Returns:
            Dict[str, POPSSMCPModuleStats]: Dictionary of tool statistics.
        """
        with self._lock:
            if tool_name:
                return {tool_name: self.tool_usage_stats.get(tool_name)} if tool_name in self.tool_usage_stats else {}
            return self.tool_usage_stats.copy()
    
    def is_session_active(self) -> bool:
        """
        Check if the session is still active.
        
        Sessions become inactive when explicitly closed or after timeout.
        
        Returns:
            bool: True if session is active, False otherwise.
        """
        with self._lock:
            if not self.is_active:
                return False
            
            # Check for timeout
            if self.session_timeout > 0:
                time_since_activity = (datetime.now() - self.last_activity).total_seconds()
                if time_since_activity > self.session_timeout:
                    self.is_active = False
                    self._LOG.info(f"Session timed out after {self.session_timeout} seconds")
                    return False
            
            return True
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the session.
        
        Returns:
            Dict[str, Any]: Dictionary containing session metadata and statistics.
        """
        with self._lock:
            return {
                'session_id': self.session_id,
                'created_at': self.created_at,
                'last_activity': self.last_activity,
                'is_active': self.is_active,
                'total_executions': len(self.execution_contexts),
                'active_tools': list(self.active_tools),
                'global_variables': len(self.global_variables),
                'tool_cache_entries': sum(len(cache) for cache in self.tool_cache.values()),
                'tool_usage_stats': {
                    tool: {
                        'load_time': stats.load_time,
                        'tool_count': stats.tool_count,
                        'status': stats.status.value
                    }
                    for tool, stats in self.tool_usage_stats.items()
                }
            }
    
    def cleanup_session(self):
        """
        Clean up all session resources.
        
        This method clears all execution contexts, statistics, variables,
        caches, and marks the session as inactive.
        """
        with self._lock:
            self._LOG.info("Cleaning up session")
            
            self.execution_contexts.clear()
            self.tool_usage_stats.clear()
            self.global_variables.clear()
            self.tool_cache.clear()
            self.active_tools.clear()
            
            if hasattr(self.memory, 'recent_executions'):
                self.memory.recent_executions.clear()
            self.memory.tool_memories.clear()
            
            self.is_active = False
            self._LOG.info("Session cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_session()
