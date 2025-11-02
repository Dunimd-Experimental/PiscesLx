#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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

import time
import threading
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union, Set
from collections import defaultdict
from pathlib import Path

from .types import (
    PiscesLxCoreMCPToolMetadata,
    PiscesLxCoreMCPExecutionContext,
    PiscesLxCoreMCPModuleStats,
    PiscesLxCoreMCPModuleStatus
)
from .tools import PiscesLxCoreMCPTools

from utils.log.core import PiscesLxCoreLog

class PiscesLxCoreMCPRegistry:
    """Tool registry system for PiscesLxCoreMCP.
    
    Manages tool registration, discovery, execution routing, and lifecycle
    operations with intelligent tool selection and execution coordination.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the tool registry.
        
        Args:
            config: Optional configuration for registry operations
        """
        self.config = config or {}
        self.logger = self._configure_logging()
        
        # Tool management
        self.tools_manager = PiscesLxCoreMCPTools(config=getattr(self.config, 'tools_config', {}))
        
        # Registration tracking
        self.registered_tools: Dict[str, PiscesLxCoreMCPToolMetadata] = {}
        self.tool_aliases: Dict[str, str] = {}
        self.tool_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.tool_priorities: Dict[str, int] = {}
        
        # Execution tracking
        self.execution_history: List[PiscesLxCoreMCPExecutionContext] = []
        self.execution_stats: Dict[str, PiscesLxCoreMCPModuleStats] = {}
        
        # Registry state
        self.is_initialized = False
        self.last_discovery = None
        self.discovery_interval = getattr(self.config, 'discovery_interval', 300)  # 5 minutes
        
        # Thread safety
        self._lock = threading.RLock()
        self._execution_lock = threading.Lock()
        
        # Configuration
        self.auto_discovery = getattr(self.config, 'auto_discovery', True)
        self.strict_registration = getattr(self.config, 'strict_registration', True)
        self.max_execution_history = getattr(self.config, 'max_execution_history', 1000)
        
        self.logger.info("PiscesLxCoreMCPRegistry initialized")
    
    def _configure_logging(self) -> PiscesLxCoreLog:
        """Configure structured logging for the registry.
        
        Returns:
            Configured logger instance
        """
        logger = PiscesLxCoreLog("PiscesLx.Utils.MCP.Registry")
        return logger
    
    def initialize(self) -> bool:
        """Initialize the registry with tool discovery and loading.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing tool registry")
            
            # Discover available tools
            if self.auto_discovery:
                discovered_tools = self.discover_tools()
                self.logger.info(f"Discovered {len(discovered_tools)} tools")
            
            # Load tools from configuration
            tools_to_load = getattr(self.config, 'load_tools', [])
            for tool_name in tools_to_load:
                self.load_tool(tool_name)
            
            # Load default tools if specified
            if getattr(self.config, 'load_default_tools', True):
                self._load_default_tools()
            
            self.is_initialized = True
            self.logger.info("Tool registry initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Registry initialization failed: {e}")
            return False
    
    def _load_default_tools(self):
        """Load default system tools from MCP.json configuration."""
        try:
            # Load MCP.json configuration
            mcp_config_path = Path("MCP/MCP.json")
            if not mcp_config_path.exists():
                self.logger.warning("MCP.json not found, falling back to basic tool discovery")
                return
            
            with open(mcp_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Get enabled tools from configuration
            tools_config = config.get('tools', {})
            enabled_tools = [
                tool_name for tool_name, tool_info in tools_config.items() 
                if tool_info.get('enabled', False)
            ]
            
            self.logger.info(f"Loading {len(enabled_tools)} enabled tools from MCP.json")
            
            # Load each enabled tool
            for tool_name in enabled_tools:
                try:
                    self.load_tool(tool_name)
                except Exception as e:
                    self.logger.warning(f"Failed to load tool {tool_name} from MCP.json: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to load tools from MCP.json: {e}")
            # Fallback to original method
            self._load_default_tools_fallback()
    
    def _load_default_tools_fallback(self):
        """Fallback method for loading default tools when MCP.json is not available."""
        default_tools = [
            'crypto',
            'time',
            'web_search',
            'fetch',
            'document_processor',
            'sequentialthinking',
            'template'
        ]
        
        for tool_name in default_tools:
            try:
                self.load_tool(tool_name)
            except Exception as e:
                self.logger.warning(f"Failed to load default tool {tool_name}: {e}")
    
    def discover_tools(self, force: bool = False) -> List[str]:
        """Discover available tools.
        
        Args:
            force: Force discovery even if recently performed
            
        Returns:
            List of discovered tool names
        """
        with self._lock:
            # Check if discovery is needed
            if not force and self.last_discovery:
                time_since_discovery = (datetime.now() - self.last_discovery).total_seconds()
                if time_since_discovery < self.discovery_interval:
                    return list(self.registered_tools.keys())
            
            try:
                # Use tools manager for discovery
                discovered_tools = self.tools_manager.discover_tools()
                
                self.last_discovery = datetime.now()
                self.logger.info(f"Tool discovery completed: {len(discovered_tools)} tools found")
                
                return discovered_tools
                
            except Exception as e:
                self.logger.error(f"Tool discovery failed: {e}")
                return []
    
    def register_tool(self, tool_name: str, tool_metadata: PiscesLxCoreMCPToolMetadata, 
                      aliases: Optional[List[str]] = None, priority: int = 0) -> bool:
        """Register a tool with the registry.
        
        Args:
            tool_name: Name of the tool
            tool_metadata: Tool metadata
            aliases: Optional list of aliases for the tool
            priority: Tool priority (higher values = higher priority)
            
        Returns:
            True if registration successful, False otherwise
        """
        with self._lock:
            try:
                # Validate tool metadata
                if self.strict_registration and not self._validate_tool_metadata(tool_metadata):
                    self.logger.error(f"Tool metadata validation failed: {tool_name}")
                    return False
                
                # Check for conflicts
                if tool_name in self.registered_tools:
                    self.logger.warning(f"Tool already registered: {tool_name}")
                    return False
                
                # Register the tool
                self.registered_tools[tool_name] = tool_metadata
                self.tool_priorities[tool_name] = priority
                
                # Register aliases
                if aliases:
                    for alias in aliases:
                        if alias in self.tool_aliases and self.tool_aliases[alias] != tool_name:
                            self.logger.warning(f"Alias conflict: {alias} already points to {self.tool_aliases[alias]}")
                        else:
                            self.tool_aliases[alias] = tool_name
                
                # Initialize statistics
                self.execution_stats[tool_name] = PiscesLxCoreMCPModuleStats(
                    module_name=tool_name,
                    total_calls=0,
                    successful_calls=0,
                    failed_calls=0,
                    average_execution_time=0.0,
                    last_used=None
                )
                
                self.logger.info(f"Tool registered: {tool_name} (priority: {priority})")
                return True
                
            except Exception as e:
                self.logger.error(f"Tool registration failed for {tool_name}: {e}")
                return False
    
    def _validate_tool_metadata(self, metadata: PiscesLxCoreMCPToolMetadata) -> bool:
        """Validate tool metadata.
        
        Args:
            metadata: Tool metadata to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['name', 'description']
        
        for field in required_fields:
            if not getattr(metadata, field, None):
                return False
        
        # Validate name format
        if not metadata.name.replace('_', '').replace('-', '').isalnum():
            return False
        
        return True
    
    def load_tool(self, tool_name: str) -> bool:
        """Load a tool into the registry.
        
        Args:
            tool_name: Name of the tool to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load tool using tools manager
            if not self.tools_manager.load_tool(tool_name):
                self.logger.error(f"Failed to load tool: {tool_name}")
                return False
            
            # Get tool metadata from tools manager
            tool_metadata = self.tools_manager.get_tool(tool_name)
            if not tool_metadata:
                self.logger.error(f"Tool metadata not available: {tool_name}")
                return False
            
            # Register the tool
            return self.register_tool(tool_name, tool_metadata)
            
        except Exception as e:
            self.logger.error(f"Error loading tool {tool_name}: {e}")
            return False
    
    def get_tool(self, tool_name: str) -> Optional[PiscesLxCoreMCPToolMetadata]:
        """Get tool metadata from registry.
        
        Args:
            tool_name: Name or alias of the tool
            
        Returns:
            Tool metadata or None if not found
        """
        with self._lock:
            # Resolve alias
            resolved_name = self.tool_aliases.get(tool_name, tool_name)
            return self.registered_tools.get(resolved_name)
    
    def select_tool(self, tool_name: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Intelligently select the best tool for the task.
        
        Args:
            tool_name: Requested tool name or description
            context: Optional context for tool selection
            
        Returns:
            Selected tool name or None if no suitable tool found
        """
        with self._lock:
            # Direct match
            if tool_name in self.registered_tools:
                return tool_name
            
            # Alias match
            if tool_name in self.tool_aliases:
                return self.tool_aliases[tool_name]
            
            # Fuzzy matching based on description and tags
            candidates = []
            for registered_name, metadata in self.registered_tools.items():
                score = self._calculate_tool_match_score(tool_name, metadata, context)
                if score > 0:
                    candidates.append((registered_name, score))
            
            # Sort by score and priority
            candidates.sort(key=lambda x: (x[1], self.tool_priorities.get(x[0], 0)), reverse=True)
            
            if candidates:
                selected_tool = candidates[0][0]
                self.logger.debug(f"Tool selected: {selected_tool} (score: {candidates[0][1]})")
                return selected_tool
            
            return None
    
    def _calculate_tool_match_score(self, query: str, metadata: PiscesLxCoreMCPToolMetadata, 
                                    context: Optional[Dict[str, Any]]) -> float:
        """Calculate tool match score for selection.
        
        Args:
            query: Search query
            metadata: Tool metadata
            context: Optional context
            
        Returns:
            Match score (0.0 - 1.0)
        """
        score = 0.0
        query_lower = query.lower()
        
        # Name matching
        if query_lower in metadata.name.lower():
            score += 0.4
        
        # Description matching
        if query_lower in metadata.description.lower():
            score += 0.3
        
        # Tags matching
        for tag in metadata.tags:
            if query_lower in tag.lower():
                score += 0.2
                break
        
        # Category matching
        if query_lower in metadata.category.lower():
            score += 0.1
        
        # Context-based matching
        if context:
            # Check if tool requirements match context capabilities
            if 'required_capabilities' in context:
                required_caps = set(context['required_capabilities'])
                tool_caps = set(metadata.tags + [metadata.category])
                if required_caps.intersection(tool_caps):
                    score += 0.2
        
        return min(score, 1.0)
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any], 
                    session_id: Optional[str] = None) -> Any:
        """Execute a tool with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for tool execution
            session_id: Optional session ID for tracking
            
        Returns:
            Tool execution result
            
        Raises:
            RuntimeError: If tool execution fails
        """
        with self._execution_lock:
            try:
                # Select the appropriate tool
                selected_tool = self.select_tool(tool_name)
                if not selected_tool:
                    raise RuntimeError(f"Tool not found: {tool_name}")
                
                # Get tool metadata
                tool_metadata = self.registered_tools[selected_tool]
                if not tool_metadata.is_enabled:
                    raise RuntimeError(f"Tool is disabled: {selected_tool}")
                
                # Create execution context
                execution_context = PiscesLxCoreMCPExecutionContext(
                    session_id=session_id or "registry",
                    tool_name=selected_tool,
                    arguments=arguments,
                    start_time=datetime.now()
                )
                
                # Execute the tool
                result = self._execute_tool_internal(selected_tool, arguments)
                
                # Update execution context
                execution_context.end_time = datetime.now()
                execution_context.result = result
                
                # Record successful execution
                self._record_execution(execution_context, success=True)
                
                self.logger.debug(f"Tool execution completed: {selected_tool}")
                return result
                
            except Exception as e:
                # Record failed execution
                if 'execution_context' in locals():
                    execution_context.end_time = datetime.now()
                    execution_context.error = str(e)
                    self._record_execution(execution_context, success=False)
                
                self.logger.error(f"Tool execution failed for {tool_name}: {e}")
                raise RuntimeError(f"Tool execution failed: {e}")
    
    def _execute_tool_internal(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Internal tool execution.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for tool execution
            
        Returns:
            Tool execution result
        """
        # Use tools manager for execution
        tool_function = self.tools_manager.get_tool_function(tool_name)
        if not tool_function:
            raise RuntimeError(f"Tool function not available: {tool_name}")
        
        return tool_function(arguments)
    
    def _record_execution(self, context: PiscesLxCoreMCPExecutionContext, success: bool):
        """Record tool execution.
        
        Args:
            context: Execution context
            success: Whether execution was successful
        """
        with self._lock:
            # Add to execution history
            self.execution_history.append(context)
            
            # Limit history size
            if len(self.execution_history) > self.max_execution_history:
                self.execution_history = self.execution_history[-self.max_execution_history:]
            
            # Update statistics
            tool_name = context.tool_name
            if tool_name in self.execution_stats:
                stats = self.execution_stats[tool_name]
                stats.total_calls += 1
                stats.last_used = datetime.now()
                
                if success:
                    stats.successful_calls += 1
                else:
                    stats.failed_calls += 1
                
                # Update average execution time
                if context.start_time and context.end_time:
                    exec_time = (context.end_time - context.start_time).total_seconds()
                    if stats.total_calls == 1:
                        stats.average_execution_time = exec_time
                    else:
                        stats.average_execution_time = (stats.average_execution_time * 0.7) + (exec_time * 0.3)
    
    def list_tools(self, category: Optional[str] = None, enabled_only: bool = True) -> List[str]:
        """List registered tools.
        
        Args:
            category: Optional category filter
            enabled_only: Only return enabled tools
            
        Returns:
            List of tool names
        """
        with self._lock:
            tools = []
            for tool_name, metadata in self.registered_tools.items():
                if enabled_only and not metadata.is_enabled:
                    continue
                if category and metadata.category != category:
                    continue
                tools.append(tool_name)
            return tools
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive tool information.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information dictionary or None if not found
        """
        with self._lock:
            resolved_name = self.tool_aliases.get(tool_name, tool_name)
            if resolved_name not in self.registered_tools:
                return None
            
            metadata = self.registered_tools[resolved_name]
            stats = self.execution_stats.get(resolved_name)
            
            return {
                'metadata': metadata,
                'statistics': stats,
                'aliases': [alias for alias, target in self.tool_aliases.items() if target == resolved_name],
                'priority': self.tool_priorities.get(resolved_name, 0)
            }
    
    def enable_tool(self, tool_name: str) -> bool:
        """Enable a tool.
        
        Args:
            tool_name: Name of the tool to enable
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            resolved_name = self.tool_aliases.get(tool_name, tool_name)
            if resolved_name not in self.registered_tools:
                return False
            
            self.registered_tools[resolved_name].is_enabled = True
            self.logger.info(f"Tool enabled: {resolved_name}")
            return True
    
    def disable_tool(self, tool_name: str) -> bool:
        """Disable a tool.
        
        Args:
            tool_name: Name of the tool to disable
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            resolved_name = self.tool_aliases.get(tool_name, tool_name)
            if resolved_name not in self.registered_tools:
                return False
            
            self.registered_tools[resolved_name].is_enabled = False
            self.logger.info(f"Tool disabled: {resolved_name}")
            return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Registry statistics dictionary
        """
        with self._lock:
            total_tools = len(self.registered_tools)
            enabled_tools = sum(1 for tool in self.registered_tools.values() if tool.is_enabled)
            total_executions = sum(stats.total_calls for stats in self.execution_stats.values())
            successful_executions = sum(stats.successful_calls for stats in self.execution_stats.values())
            
            return {
                'total_tools': total_tools,
                'enabled_tools': enabled_tools,
                'disabled_tools': total_tools - enabled_tools,
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'failed_executions': total_executions - successful_executions,
                'execution_history_size': len(self.execution_history),
                'last_discovery': self.last_discovery,
                'is_initialized': self.is_initialized
            }
    
    def cleanup_registry(self):
        """Clean up registry resources."""
        with self._lock:
            self.logger.info("Cleaning up registry")
            
            # Clear all data structures
            self.registered_tools.clear()
            self.tool_aliases.clear()
            self.tool_dependencies.clear()
            self.tool_priorities.clear()
            self.execution_history.clear()
            self.execution_stats.clear()
            
            self.is_initialized = False
            self.logger.info("Registry cleanup completed")