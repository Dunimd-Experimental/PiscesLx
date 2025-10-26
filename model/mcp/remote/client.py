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
Remote MCP Client implementation for user-local tool execution.

This module provides the main client interface that integrates with the existing
ArcticMCP ecosystem while supporting user-local execution of tools.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from .types import (
    RemoteClientConfig, RemoteToolCall, RemoteExecutionResult,
    RemoteExecutionMode, RemoteConnectionError, RemoteAuthenticationError
)
from .protocol import ArcticRemoteMCPProtocol
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("Arctic.Remote.MCP.Client")

class ArcticRemoteMCPClient:
    """
    Remote MCP Client for user-local tool execution.
    
    This client provides a unified interface for executing tools on user-local
    MCP servers while maintaining compatibility with the existing ArcticMCP
    ecosystem.
    """
    
    def __init__(self, config: RemoteClientConfig):
        """
        Initialize the remote MCP client.
        
        Args:
            config: Configuration for the remote client
        """
        self.config = config
        self.protocol = ArcticRemoteMCPProtocol(config)
        self._connected = False
        self._connection_lock = asyncio.Lock()
        self._tool_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamp: float = 0.0
        self._cache_ttl = 300.0  # 5 minutes
        
    async def connect(self) -> bool:
        """
        Establish connection to the user-local MCP server.
        
        Returns:
            True if connection successful, False otherwise
        """
        async with self._connection_lock:
            if self._connected:
                return True
            
            try:
                logger.info(f"Connecting to remote MCP client at {self.config.base_url}")
                success = await self.protocol.connect()
                
                if success:
                    self._connected = True
                    logger.info("Successfully connected to remote MCP client")
                    return True
                else:
                    logger.error("Failed to connect to remote MCP client")
                    return False
                    
            except RemoteAuthenticationError as e:
                logger.error(f"Authentication failed: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error during connection: {e}")
                return False
    
    async def disconnect(self) -> None:
        """Disconnect from the user-local MCP server."""
        async with self._connection_lock:
            if not self._connected:
                return
            
            try:
                await self.protocol.disconnect()
                self._connected = False
                self._tool_cache.clear()
                logger.info("Disconnected from remote MCP client")
                
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], 
                          call_id: Optional[str] = None) -> RemoteExecutionResult:
        """
        Execute a tool on the user-local MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            call_id: Optional call ID for tracking
            
        Returns:
            Execution result
            
        Raises:
            RemoteConnectionError: If not connected to server
        """
        if not self._connected:
            raise RemoteConnectionError("Client is not connected")
        
        if call_id is None:
            call_id = f"remote_{int(time.time() * 1000)}"
        
        tool_call = RemoteToolCall(
            tool_name=tool_name,
            parameters=parameters,
            call_id=call_id,
            execution_mode=RemoteExecutionMode.REMOTE
        )
        
        logger.debug(f"Executing remote tool: {tool_name}")
        
        try:
            result = await self.protocol.execute_tool(tool_call)
            
            if result.success:
                logger.info(f"Remote tool execution successful: {tool_name}")
            else:
                logger.warning(f"Remote tool execution failed: {tool_name} - {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Remote tool execution exception: {e}")
            return RemoteExecutionResult(
                success=False,
                result=None,
                tool_name=tool_name,
                call_id=call_id,
                execution_time=0.0,
                execution_mode=RemoteExecutionMode.REMOTE,
                error_message=str(e),
                error_code="CLIENT_ERROR"
            )
    
    async def list_available_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of available tools from the user-local MCP server.
        
        Args:
            force_refresh: Force refresh of tool cache
            
        Returns:
            List of available tools
            
        Raises:
            RemoteConnectionError: If not connected to server
        """
        if not self._connected:
            raise RemoteConnectionError("Client is not connected")
        
        # Check cache validity
        current_time = time.time()
        if (not force_refresh and 
            self._tool_cache and 
            (current_time - self._cache_timestamp) < self._cache_ttl):
            return list(self._tool_cache.values())
        
        try:
            tools = await self.protocol.list_available_tools()
            
            # Update cache
            self._tool_cache = {tool.get("name", ""): tool for tool in tools}
            self._cache_timestamp = current_time
            
            logger.debug(f"Retrieved {len(tools)} tools from remote client")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to list remote tools: {e}")
            return []
    
    async def get_client_info(self) -> Dict[str, Any]:
        """
        Get information about the remote client connection.
        
        Returns:
            Client information
            
        Raises:
            RemoteConnectionError: If not connected to server
        """
        if not self._connected:
            raise RemoteConnectionError("Client is not connected")
        
        try:
            client_info = await self.protocol.get_client_info()
            
            return {
                "client_id": client_info.client_id,
                "connection_state": client_info.connection_state.value,
                "connected_at": client_info.connected_at,
                "last_heartbeat": client_info.last_heartbeat,
                "available_tools": client_info.available_tools,
                "client_version": client_info.client_version,
                "capabilities": client_info.capabilities,
                "performance_metrics": client_info.performance_metrics,
                "config": {
                    "base_url": self.config.base_url,
                    "timeout": self.config.timeout,
                    "retry_attempts": self.config.retry_attempts,
                    "heartbeat_interval": self.config.heartbeat_interval
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get client info: {e}")
            return {
                "client_id": self.config.client_id,
                "connection_state": "error",
                "error": str(e)
            }
    
    async def is_connected(self) -> bool:
        """
        Check if client is connected to the remote server.
        
        Returns:
            True if connected, False otherwise
        """
        return self._connected and self.protocol.connection_state.value == "authenticated"
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """
        Get detailed connection status.
        
        Returns:
            Connection status information
        """
        return {
            "connected": await self.is_connected(),
            "connection_state": self.protocol.connection_state.value,
            "client_id": self.config.client_id,
            "base_url": self.config.base_url,
            "last_connection_attempt": time.time(),
            "tool_cache_size": len(self._tool_cache),
            "cache_age": time.time() - self._cache_timestamp if self._cache_timestamp else None
        }
    
    @asynccontextmanager
    async def connection_context(self):
        """
        Context manager for connection lifecycle management.
        
        Usage:
            async with client.connection_context():
                # Use client here
                result = await client.execute_tool("tool_name", {})
        """
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()
    
    def __repr__(self) -> str:
        """String representation of the client."""
        return f"ArcticRemoteMCPClient(base_url='{self.config.base_url}', connected={self._connected})"