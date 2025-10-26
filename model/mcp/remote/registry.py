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

"""
Tool Registry for Remote MCP Client.

This module provides tool discovery, registration, and management
for the ArcticRemoteMCPClient system.
"""

import json
import os
import fnmatch
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
import asyncio

from .types import RemoteToolCall, RemoteExecutionResult
from .config import get_config_manager, ToolRegistryConfig
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("Arctic.Remote.MCP.Registry")

@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    return_type: str = "any"
    category: str = "utility"
    tags: List[str] = field(default_factory=list)
    author: str = ""
    version: str = "1.0.0"
    requires_user_data: bool = False
    security_level: str = "low"  # low, medium, high
    performance_critical: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "category": self.category,
            "tags": self.tags,
            "author": self.author,
            "version": self.version,
            "requires_user_data": self.requires_user_data,
            "security_level": self.security_level,
            "performance_critical": self.performance_critical
        }

@dataclass
class RegisteredTool:
    """Represents a registered tool."""
    
    metadata: ToolMetadata
    client_id: str
    registration_time: float
    last_used: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    average_execution_time: float = 0.0
    
    def get_success_rate(self) -> float:
        """Get success rate."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

class ToolRegistry:
    """
    Tool registry for Remote MCP system.
    
    Manages tool discovery, registration, and metadata for available tools.
    """
    
    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, RegisteredTool] = {}
        self._client_tools: Dict[str, Set[str]] = {}  # client_id -> tool_names
        self._categories: Dict[str, Set[str]] = {}  # category -> tool_names
        self._tags: Dict[str, Set[str]] = {}  # tag -> tool_names
        self._registry_lock = asyncio.Lock()
        
        logger.info("Initialized Tool Registry")
    
    async def register_tool(self, tool_metadata: ToolMetadata, 
                          client_id: str) -> bool:
        """
        Register a tool with the registry.
        
        Args:
            tool_metadata: Tool metadata
            client_id: Client ID that provides this tool
            
        Returns:
            True if registration successful
        """
        async with self._registry_lock:
            try:
                # Validate tool metadata
                if not self._validate_tool_metadata(tool_metadata):
                    logger.error(f"Invalid tool metadata for {tool_metadata.name}")
                    return False
                
                # Check if tool is allowed
                if not await self._is_tool_allowed(tool_metadata.name):
                    logger.warning(f"Tool {tool_metadata.name} is not allowed")
                    return False
                
                # Create registered tool
                registered_tool = RegisteredTool(
                    metadata=tool_metadata,
                    client_id=client_id,
                    registration_time=asyncio.get_event_loop().time()
                )
                
                # Register tool
                self._tools[tool_metadata.name] = registered_tool
                
                # Update client tools mapping
                if client_id not in self._client_tools:
                    self._client_tools[client_id] = set()
                self._client_tools[client_id].add(tool_metadata.name)
                
                # Update category mapping
                if tool_metadata.category not in self._categories:
                    self._categories[tool_metadata.category] = set()
                self._categories[tool_metadata.category].add(tool_metadata.name)
                
                # Update tag mappings
                for tag in tool_metadata.tags:
                    if tag not in self._tags:
                        self._tags[tag] = set()
                    self._tags[tag].add(tool_metadata.name)
                
                logger.info(f"Registered tool {tool_metadata.name} from client {client_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register tool {tool_metadata.name}: {e}")
                return False
    
    async def unregister_tool(self, tool_name: str, client_id: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            tool_name: Tool name
            client_id: Client ID
            
        Returns:
            True if unregistration successful
        """
        async with self._registry_lock:
            try:
                if tool_name not in self._tools:
                    logger.warning(f"Tool {tool_name} not found in registry")
                    return False
                
                registered_tool = self._tools[tool_name]
                
                # Verify client ownership
                if registered_tool.client_id != client_id:
                    logger.warning(f"Client {client_id} cannot unregister tool {tool_name} (owned by {registered_tool.client_id})")
                    return False
                
                # Remove tool
                del self._tools[tool_name]
                
                # Update mappings
                if client_id in self._client_tools:
                    self._client_tools[client_id].discard(tool_name)
                    if not self._client_tools[client_id]:
                        del self._client_tools[client_id]
                
                # Update category mapping
                category = registered_tool.metadata.category
                if category in self._categories:
                    self._categories[category].discard(tool_name)
                    if not self._categories[category]:
                        del self._categories[category]
                
                # Update tag mappings
                for tag in registered_tool.metadata.tags:
                    if tag in self._tags:
                        self._tags[tag].discard(tool_name)
                        if not self._tags[tag]:
                            del self._tags[tag]
                
                logger.info(f"Unregistered tool {tool_name} from client {client_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unregister tool {tool_name}: {e}")
                return False
    
    async def unregister_all_client_tools(self, client_id: str) -> int:
        """
        Unregister all tools from a specific client.
        
        Args:
            client_id: Client ID
            
        Returns:
            Number of tools unregistered
        """
        async with self._registry_lock:
            if client_id not in self._client_tools:
                return 0
            
            tool_names = list(self._client_tools[client_id])
            unregistered_count = 0
            
            for tool_name in tool_names:
                if await self.unregister_tool(tool_name, client_id):
                    unregistered_count += 1
            
            logger.info(f"Unregistered {unregistered_count} tools from client {client_id}")
            return unregistered_count
    
    async def get_tool(self, tool_name: str) -> Optional[RegisteredTool]:
        """
        Get a registered tool.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Registered tool or None
        """
        async with self._registry_lock:
            return self._tools.get(tool_name)
    
    async def get_tools_by_client(self, client_id: str) -> List[RegisteredTool]:
        """
        Get all tools provided by a specific client.
        
        Args:
            client_id: Client ID
            
        Returns:
            List of registered tools
        """
        async with self._registry_lock:
            if client_id not in self._client_tools:
                return []
            
            tools = []
            for tool_name in self._client_tools[client_id]:
                if tool_name in self._tools:
                    tools.append(self._tools[tool_name])
            
            return tools
    
    async def get_tools_by_category(self, category: str) -> List[RegisteredTool]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Tool category
            
        Returns:
            List of registered tools
        """
        async with self._registry_lock:
            if category not in self._categories:
                return []
            
            tools = []
            for tool_name in self._categories[category]:
                if tool_name in self._tools:
                    tools.append(self._tools[tool_name])
            
            return tools
    
    async def get_tools_by_tag(self, tag: str) -> List[RegisteredTool]:
        """
        Get all tools with a specific tag.
        
        Args:
            tag: Tool tag
            
        Returns:
            List of registered tools
        """
        async with self._registry_lock:
            if tag not in self._tags:
                return []
            
            tools = []
            for tool_name in self._tags[tag]:
                if tool_name in self._tools:
                    tools.append(self._tools[tool_name])
            
            return tools
    
    async def search_tools(self, query: str, 
                         category: Optional[str] = None,
                         tags: Optional[List[str]] = None) -> List[RegisteredTool]:
        """
        Search for tools by name, description, or tags.
        
        Args:
            query: Search query
            category: Optional category filter
            tags: Optional tags filter
            
        Returns:
            List of matching tools
        """
        async with self._registry_lock:
            query_lower = query.lower()
            matching_tools = []
            
            for tool_name, registered_tool in self._tools.items():
                metadata = registered_tool.metadata
                
                # Check category filter
                if category and metadata.category != category:
                    continue
                
                # Check tags filter
                if tags and not all(tag in metadata.tags for tag in tags):
                    continue
                
                # Check name and description
                if (query_lower in tool_name.lower() or 
                    query_lower in metadata.description.lower() or
                    any(query_lower in tag.lower() for tag in metadata.tags)):
                    matching_tools.append(registered_tool)
            
            return matching_tools
    
    async def get_all_tools(self) -> List[RegisteredTool]:
        """
        Get all registered tools.
        
        Returns:
            List of all registered tools
        """
        async with self._registry_lock:
            return list(self._tools.values())
    
    async def record_tool_usage(self, tool_name: str, 
                              success: bool, 
                              execution_time: float) -> None:
        """
        Record tool usage statistics.
        
        Args:
            tool_name: Tool name
            success: Whether execution was successful
            execution_time: Execution time in seconds
        """
        async with self._registry_lock:
            if tool_name not in self._tools:
                return
            
            registered_tool = self._tools[tool_name]
            
            # Update usage statistics
            registered_tool.usage_count += 1
            registered_tool.last_used = asyncio.get_event_loop().time()
            
            if success:
                registered_tool.success_count += 1
            
            # Update average execution time
            if registered_tool.average_execution_time == 0:
                registered_tool.average_execution_time = execution_time
            else:
                # Exponential moving average
                registered_tool.average_execution_time = (
                    0.7 * registered_tool.average_execution_time + 
                    0.3 * execution_time
                )
    
    async def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Registry statistics
        """
        async with self._registry_lock:
            total_tools = len(self._tools)
            
            if total_tools == 0:
                return {
                    "total_tools": 0,
                    "total_clients": 0,
                    "categories": {},
                    "tags": {},
                    "average_success_rate": 0.0,
                    "most_used_tools": []
                }
            
            # Calculate statistics
            categories = {cat: len(tools) for cat, tools in self._categories.items()}
            tags = {tag: len(tools) for tag, tools in self._tags.items()}
            
            # Calculate average success rate
            total_success_rate = sum(
                tool.get_success_rate() for tool in self._tools.values()
            )
            average_success_rate = total_success_rate / total_tools
            
            # Find most used tools
            most_used = sorted(
                self._tools.values(), 
                key=lambda t: t.usage_count, 
                reverse=True
            )[:10]
            
            most_used_tools = [
                {
                    "name": tool.metadata.name,
                    "usage_count": tool.usage_count,
                    "success_rate": tool.get_success_rate(),
                    "client_id": tool.client_id
                }
                for tool in most_used
            ]
            
            return {
                "total_tools": total_tools,
                "total_clients": len(self._client_tools),
                "categories": categories,
                "tags": tags,
                "average_success_rate": average_success_rate,
                "most_used_tools": most_used_tools
            }
    
    def _validate_tool_metadata(self, metadata: ToolMetadata) -> bool:
        """Validate tool metadata."""
        if not metadata.name or not metadata.name.strip():
            return False
        
        if not metadata.description or not metadata.description.strip():
            return False
        
        if not isinstance(metadata.parameters, dict):
            return False
        
        if metadata.security_level not in ["low", "medium", "high"]:
            return False
        
        return True
    
    async def _is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed based on configuration."""
        try:
            config_manager = await get_config_manager()
            registry_config = config_manager.get_tool_registry_config()
            
            # Check whitelist
            if registry_config.enable_whitelist:
                if tool_name not in registry_config.whitelisted_tools:
                    return False
            
            # Check blacklist
            if registry_config.enable_blacklist:
                for pattern in registry_config.blacklisted_tools:
                    if fnmatch.fnmatch(tool_name.lower(), pattern.lower()):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking tool permissions: {e}")
            return False  # Deny by default on error
    
    async def discover_tools_from_directory(self, directory: Path) -> List[ToolMetadata]:
        """
        Discover tools from a directory.
        
        Args:
            directory: Directory to search for tools
            
        Returns:
            List of discovered tool metadata
        """
        discovered_tools = []
        
        try:
            config_manager = await get_config_manager()
            registry_config = config_manager.get_tool_registry_config()
            
            if not registry_config.auto_discovery:
                return discovered_tools
            
            # Search for tool configuration files
            for pattern in registry_config.discovery_patterns:
                for config_file in directory.glob(pattern):
                    try:
                        tools = await self._load_tools_from_file(config_file)
                        discovered_tools.extend(tools)
                    except Exception as e:
                        logger.warning(f"Failed to load tools from {config_file}: {e}")
            
            logger.info(f"Discovered {len(discovered_tools)} tools from {directory}")
            return discovered_tools
            
        except Exception as e:
            logger.error(f"Error discovering tools from directory: {e}")
            return discovered_tools
    
    async def _load_tools_from_file(self, config_file: Path) -> List[ToolMetadata]:
        """Load tools from a configuration file."""
        tools = []
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Support different formats
            if "tools" in config_data:
                tools_data = config_data["tools"]
            elif "mcp_tools" in config_data:
                tools_data = config_data["mcp_tools"]
            else:
                tools_data = [config_data]  # Single tool format
            
            for tool_data in tools_data:
                try:
                    metadata = self._parse_tool_metadata(tool_data)
                    tools.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to parse tool metadata: {e}")
            
            return tools
            
        except Exception as e:
            logger.error(f"Failed to load tools from {config_file}: {e}")
            return tools
    
    def _parse_tool_metadata(self, tool_data: Dict[str, Any]) -> ToolMetadata:
        """Parse tool metadata from data."""
        return ToolMetadata(
            name=tool_data['name'],
            description=tool_data.get('description', ''),
            parameters=tool_data.get('parameters', {}),
            return_type=tool_data.get('return_type', 'any'),
            category=tool_data.get('category', 'utility'),
            tags=tool_data.get('tags', []),
            author=tool_data.get('author', ''),
            version=tool_data.get('version', '1.0.0'),
            requires_user_data=tool_data.get('requires_user_data', False),
            security_level=tool_data.get('security_level', 'low'),
            performance_critical=tool_data.get('performance_critical', False)
        )
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"ToolRegistry(tools={len(self._tools)}, clients={len(self._client_tools)})"

# Global tool registry instance
_registry: Optional[ToolRegistry] = None

async def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.
    
    Returns:
        Tool registry instance
    """
    global _registry
    
    if _registry is None:
        _registry = ToolRegistry()
    
    return _registry

def reset_tool_registry() -> None:
    """Reset the global tool registry instance."""
    global _registry
    _registry = None