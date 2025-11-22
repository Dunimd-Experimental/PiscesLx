#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import uuid
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Callable
from collections import defaultdict

from .types import (
    PiscesLxCoreMCPSessionMemory,
    PiscesLxCoreMCPExecutionContext,
    PiscesLxCoreMCPModuleStats,
    PiscesLxCoreMCPModuleStatus
)

from utils.log.core import PiscesLxCoreLog

class PiscesLxCoreMCPSession:
    """Session management system for PiscesLxCoreMCP.
    
    Manages tool execution sessions, memory state, context tracking,
    and session lifecycle operations with thread-safe operations.
    """
    
    def __init__(self, session_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize a new MCP session.
        
        Args:
            session_id: Optional custom session ID, auto-generated if None
            config: Optional session configuration
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.config = config or {}
        self.logger = self._configure_logging()
        
        # Session lifecycle tracking
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.is_active = True
        
        # Memory management
        self.memory = PiscesLxCoreMCPSessionMemory(session_id=self.session_id)
        self.execution_contexts: Dict[str, PiscesLxCoreMCPExecutionContext] = {}
        self.tool_usage_stats: Dict[str, PiscesLxCoreMCPModuleStats] = {}
        
        # Session state
        self.global_variables: Dict[str, Any] = {}
        self.tool_cache: Dict[str, Any] = {}
        self.active_tools: Set[str] = set()
        
        # Thread safety
        self._lock = threading.RLock()
        self._session_lock = threading.Lock()
        
        # Configuration defaults
        self.max_executions = getattr(self.config, 'max_executions', 1000)
        self.session_timeout = getattr(self.config, 'session_timeout', 3600)  # 1 hour
        self.memory_limit = getattr(self.config, 'memory_limit', 100)  # MB
        
        self.logger.info(f"Session initialized: {self.session_id}")
    
    def _configure_logging(self) -> PiscesLxCoreLog:
        """Configure structured logging for the session.
        
        Returns:
            Configured logger instance
        """
        logger = PiscesLxCoreLog(f"PiscesLx.Core.MCP.Session.{self.session_id[:8]}")
        return logger
    
    def create_execution_context(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Create a new execution context for a tool.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Input arguments for the tool
            
        Returns:
            Execution context ID
        """
        with self._lock:
            execution_id = f"exec_{int(time.time() * 1000000)}_{tool_name}"
            
            # Check session limits
            if len(self.execution_contexts) >= self.max_executions:
                raise RuntimeError(f"Maximum executions ({self.max_executions}) reached for session")
            
            # Check if tool is already active (if required)
            if tool_name in self.active_tools and not getattr(self.config, 'allow_concurrent_tools', True):
                raise RuntimeError(f"Tool {tool_name} is already executing in this session")
            
            # Create execution context
            context = PiscesLxCoreMCPExecutionContext(
                session_id=self.session_id,
                tool_name=tool_name,
                arguments=arguments,
                start_time=datetime.now()
            )
            
            # Store context
            self.execution_contexts[execution_id] = context
            self.active_tools.add(tool_name)
            
            # Update session activity
            self.last_activity = datetime.now()
            
            # Initialize tool stats if needed
            if tool_name not in self.tool_usage_stats:
                self.tool_usage_stats[tool_name] = PiscesLxCoreMCPModuleStats(
                    module_name=tool_name,
                    total_calls=0,
                    successful_calls=0,
                    failed_calls=0,
                    average_execution_time=0.0,
                    last_used=datetime.now()
                )
            
            self.logger.debug(f"Execution context created: {execution_id} for {tool_name}")
            return execution_id
    
    def update_execution_context(self, execution_id: str, 
                               result: Any = None, 
                               error: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None):
        """Update an execution context with results.
        
        Args:
            execution_id: Execution context ID
            result: Execution result (if successful)
            error: Error message (if failed)
            metadata: Additional execution metadata
        """
        with self._lock:
            if execution_id not in self.execution_contexts:
                self.logger.warning(f"Execution context not found: {execution_id}")
                return
            
            context = self.execution_contexts[execution_id]
            context.end_time = datetime.now()
            context.result = result
            context.error = error
            
            if metadata:
                context.metadata.update(metadata)
            
            # Update tool usage stats
            tool_name = context.tool_name
            stats = self.tool_usage_stats[tool_name]
            stats.total_calls += 1
            stats.last_used = datetime.now()
            
            if error:
                stats.failed_calls += 1
            else:
                stats.successful_calls += 1
            
            # Calculate execution time
            if context.start_time and context.end_time:
                exec_time = (context.end_time - context.start_time).total_seconds()
                
                # Update average execution time
                if stats.total_calls == 1:
                    stats.average_execution_time = exec_time
                else:
                    # Weighted average: 70% old average, 30% new time
                    stats.average_execution_time = (stats.average_execution_time * 0.7) + (exec_time * 0.3)
            
            # Remove from active tools
            self.active_tools.discard(tool_name)
            
            # Store in session memory
            self._store_execution_in_memory(context)
            
            self.logger.debug(f"Execution context updated: {execution_id}")
    
    def _store_execution_in_memory(self, context: PiscesLxCoreMCPExecutionContext):
        """Store execution context in session memory.
        
        Args:
            context: Execution context to store
        """
        # Add to recent executions
        self.memory.recent_executions.append({
            'tool_name': context.tool_name,
            'arguments': context.arguments,
            'result': context.result,
            'error': context.error,
            'execution_time': (context.end_time - context.start_time).total_seconds() if context.end_time else None,
            'timestamp': context.start_time
        })
        
        # Limit memory size
        max_executions = getattr(self.config, 'max_memory_executions', 100)
        if len(self.memory.recent_executions) > max_executions:
            self.memory.recent_executions = self.memory.recent_executions[-max_executions:]
        
        # Update tool-specific memory
        if context.tool_name not in self.memory.tool_memories:
            self.memory.tool_memories[context.tool_name] = []
        
        self.memory.tool_memories[context.tool_name].append({
            'arguments': context.arguments,
            'result': context.result,
            'error': context.error,
            'timestamp': context.start_time
        })
        
        # Limit tool-specific memory
        max_tool_memory = getattr(self.config, 'max_tool_memory', 20)
        if len(self.memory.tool_memories[context.tool_name]) > max_tool_memory:
            self.memory.tool_memories[context.tool_name] = self.memory.tool_memories[context.tool_name][-max_tool_memory:]
    
    def get_execution_context(self, execution_id: str) -> Optional[PiscesLxCoreMCPExecutionContext]:
        """Get an execution context by ID.
        
        Args:
            execution_id: Execution context ID
            
        Returns:
            Execution context or None if not found
        """
        with self._lock:
            return self.execution_contexts.get(execution_id)
    
    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history.
        
        Args:
            limit: Maximum number of executions to return
            
        Returns:
            List of recent executions
        """
        with self._lock:
            return self.memory.recent_executions[-limit:] if self.memory.recent_executions else []
    
    def get_tool_memory(self, tool_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get memory for a specific tool.
        
        Args:
            tool_name: Name of the tool
            limit: Maximum number of memory entries to return
            
        Returns:
            List of tool memory entries
        """
        with self._lock:
            tool_mem = self.memory.tool_memories.get(tool_name, [])
            return tool_mem[-limit:] if tool_mem else []
    
    def set_global_variable(self, key: str, value: Any):
        """Set a global variable in the session.
        
        Args:
            key: Variable name
            value: Variable value
        """
        with self._lock:
            self.global_variables[key] = value
            self.last_activity = datetime.now()
            self.logger.debug(f"Global variable set: {key}")
    
    def get_global_variable(self, key: str, default: Any = None) -> Any:
        """Get a global variable from the session.
        
        Args:
            key: Variable name
            default: Default value if key not found
            
        Returns:
            Variable value or default
        """
        with self._lock:
            return self.global_variables.get(key, default)
    
    def set_tool_cache(self, tool_name: str, cache_key: str, value: Any):
        """Set a cache value for a tool.
        
        Args:
            tool_name: Name of the tool
            cache_key: Cache key
            value: Value to cache
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
        """Get a cached value for a tool.
        
        Args:
            tool_name: Name of the tool
            cache_key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
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
        """Clear tool cache.
        
        Args:
            tool_name: Optional specific tool name, or all tools if None
        """
        with self._lock:
            if tool_name:
                if tool_name in self.tool_cache:
                    del self.tool_cache[tool_name]
                    self.logger.debug(f"Cache cleared for tool: {tool_name}")
            else:
                self.tool_cache.clear()
                self.logger.debug("All tool caches cleared")
            
            self.last_activity = datetime.now()
    
    def get_tool_usage_stats(self, tool_name: Optional[str] = None) -> Dict[str, PiscesLxCoreMCPModuleStats]:
        """Get tool usage statistics.
        
        Args:
            tool_name: Optional specific tool name, or all tools if None
            
        Returns:
            Dictionary of tool statistics
        """
        with self._lock:
            if tool_name:
                return {tool_name: self.tool_usage_stats.get(tool_name)} if tool_name in self.tool_usage_stats else {}
            return self.tool_usage_stats.copy()
    
    def is_session_active(self) -> bool:
        """Check if the session is still active.
        
        Returns:
            True if session is active, False otherwise
        """
        with self._lock:
            if not self.is_active:
                return False
            
            # Check timeout
            if self.session_timeout > 0:
                time_since_activity = (datetime.now() - self.last_activity).total_seconds()
                if time_since_activity > self.session_timeout:
                    self.is_active = False
                    self.logger.info(f"Session timed out after {self.session_timeout} seconds")
                    return False
            
            return True
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get comprehensive session information.
        
        Returns:
            Session information dictionary
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
                        'total_calls': stats.total_calls,
                        'successful_calls': stats.successful_calls,
                        'failed_calls': stats.failed_calls,
                        'error_rate': stats.failed_calls / max(stats.total_calls, 1),
                        'average_execution_time': stats.average_execution_time,
                        'last_used': stats.last_used
                    }
                    for tool, stats in self.tool_usage_stats.items()
                }
            }
    
    def cleanup_session(self):
        """Clean up session resources."""
        with self._lock:
            self.logger.info("Cleaning up session")
            
            # Clear all data structures
            self.execution_contexts.clear()
            self.tool_usage_stats.clear()
            self.global_variables.clear()
            self.tool_cache.clear()
            self.active_tools.clear()
            
            # Update memory
            self.memory.recent_executions.clear()
            self.memory.tool_memories.clear()
            
            self.is_active = False
            self.logger.info("Session cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_session()