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

"""
Unified remote client for MCP system.

This module provides a unified interface for remote MCP tool execution,
integrating the functionality from model/mcp/remote/client.py.
"""

import asyncio
import json
import time
import socket
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from utils.log.core import PiscesLxCoreLog
    logger = PiscesLxCoreLog("Arctic.Utils.MCP.RemoteClient")
except ImportError:
    # Fallback to simple logger if utils.log.core is not available
    import logging
    logger = logging.getLogger("Arctic.Utils.MCP.RemoteClient")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

# Import execution module with fallback for standalone testing
try:
    from .execution import PiscesLxCoreMCPExecutionResult, PiscesLxCoreMCPExecutionMode, PiscesLxCoreMCPExecutionStatus
except ImportError:
    # Fallback for standalone testing
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from execution import PiscesLxCoreMCPExecutionResult, PiscesLxCoreMCPExecutionMode, PiscesLxCoreMCPExecutionStatus


@dataclass
class _RemoteToolMetadata:
    """Metadata for remote tools."""
    name: str
    description: str
    parameters: Dict[str, Any]
    client_id: str
    category: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass
class _RemoteClientConfig:
    """Configuration for remote MCP clients."""
    host: str = "localhost"
    port: int = 8080
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_pool_size: int = 5
    enable_caching: bool = True
    cache_ttl: float = 300.0  # 5 minutes


class PiscesLxCoreMCPRemoteClient(ABC):
    """Abstract base class for remote MCP clients."""
    
    def __init__(self, client_id: str, config: Optional[_RemoteClientConfig] = None):
        """Initialize remote MCP client."""
        self.client_id = client_id
        self.config = config or _RemoteClientConfig()
        self._connected = False
        self._connection_lock = asyncio.Lock()
        self._tool_cache: Dict[str, tuple] = {}  # tool_name -> (metadata, timestamp)
        
        logger.info(f"RemoteMCPClient initialized for client_id: {client_id}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to remote MCP server."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from remote MCP server."""
        pass
    
    @abstractmethod
    async def list_available_tools(self) -> List[_RemoteToolMetadata]:
        """List available tools from remote server."""
        pass
    
    @abstractmethod
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> PiscesLxCoreMCPExecutionResult:
        """Execute a tool remotely."""
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected
    
    def _get_cached_tools(self) -> Optional[List[_RemoteToolMetadata]]:
        """Get cached tools if available."""
        if not self.config.enable_caching:
            return None
        
        current_time = time.time()
        cached_items = []
        
        for tool_name, (metadata, timestamp) in self._tool_cache.items():
            if current_time - timestamp < self.config.cache_ttl:
                cached_items.append(metadata)
        
        return cached_items if cached_items else None
    
    def _cache_tools(self, tools: List[_RemoteToolMetadata]):
        """Cache tools for future use."""
        if not self.config.enable_caching:
            return
        
        current_time = time.time()
        for tool in tools:
            self._tool_cache[tool.name] = (tool, current_time)


class PiscesLxCoreMCPArcticRemoteClient(PiscesLxCoreMCPRemoteClient):
    """Arctic-specific remote MCP client implementation."""
    
    def __init__(self, client_id: str, config: Optional[_RemoteClientConfig] = None):
        """Initialize Arctic remote MCP client."""
        super().__init__(client_id, config)
        self._socket: Optional[socket.socket] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
    
    async def connect(self) -> bool:
        """Connect to remote MCP server."""
        async with self._connection_lock:
            if self._connected:
                return True
            
            try:
                logger.info(f"Connecting to remote MCP server at {self.config.host}:{self.config.port}")
                
                # Create connection
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(self.config.host, self.config.port),
                    timeout=self.config.timeout
                )
                
                self._connected = True
                logger.info(f"Successfully connected to remote MCP server")
                return True
                
            except Exception as e:
                logger.error(f"Failed to connect to remote MCP server: {e}")
                self._connected = False
                return False
    
    async def disconnect(self) -> bool:
        """Disconnect from remote MCP server."""
        async with self._connection_lock:
            if not self._connected:
                return True
            
            try:
                if self._writer:
                    self._writer.close()
                    await self._writer.wait_closed()
                
                self._connected = False
                self._reader = None
                self._writer = None
                
                logger.info("Disconnected from remote MCP server")
                return True
                
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
                return False
    
    async def list_available_tools(self) -> List[_RemoteToolMetadata]:
        """List available tools from remote server."""
        # Check cache first
        cached_tools = self._get_cached_tools()
        if cached_tools:
            logger.debug(f"Returning {len(cached_tools)} cached tools")
            return cached_tools
        
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            logger.error("Cannot list tools: not connected to remote server")
            return []
        
        try:
            # Send list tools request
            request = {
                "action": "list_tools",
                "client_id": self.client_id,
                "timestamp": time.time()
            }
            
            await self._send_request(request)
            response = await self._receive_response()
            
            if response.get("success"):
                tools_data = response.get("tools", [])
                tools = []
                
                for tool_data in tools_data:
                    tool = _RemoteToolMetadata(
                        name=tool_data["name"],
                        description=tool_data["description"],
                        parameters=tool_data.get("parameters", {}),
                        client_id=self.client_id,
                        category=tool_data.get("category"),
                        tags=tool_data.get("tags", [])
                    )
                    tools.append(tool)
                
                # Cache the results
                self._cache_tools(tools)
                
                logger.info(f"Retrieved {len(tools)} tools from remote server")
                return tools
            else:
                logger.error(f"Failed to list tools: {response.get('error', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> PiscesLxCoreMCPExecutionResult:
        """Execute a tool remotely."""
        start_time = time.time()
        
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            execution_time = time.time() - start_time
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=execution_time,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message="Not connected to remote server",
                error_code="CONNECTION_ERROR",
                mode=PiscesLxCoreMCPExecutionMode.REMOTE
            )
        
        try:
            # Send execute tool request
            request = {
                "action": "execute_tool",
                "tool_name": tool_name,
                "parameters": parameters,
                "client_id": self.client_id,
                "timestamp": time.time()
            }
            
            await self._send_request(request)
            response = await self._receive_response()
            
            execution_time = time.time() - start_time
            
            if response.get("success"):
                return PiscesLxCoreMCPExecutionResult(
                    success=True,
                    result=response.get("result"),
                    execution_time=execution_time,
                    status=PiscesLxCoreMCPExecutionStatus.COMPLETED,
                    mode=PiscesLxCoreMCPExecutionMode.REMOTE
                )
            else:
                return PiscesLxCoreMCPExecutionResult(
                    success=False,
                    result=None,
                    execution_time=execution_time,
                    status=PiscesLxCoreMCPExecutionStatus.FAILED,
                    error_message=response.get("error", "Unknown error"),
                    error_code="REMOTE_EXECUTION_ERROR",
                    mode=PiscesLxCoreMCPExecutionMode.REMOTE
                )
                
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=execution_time,
                status=PiscesLxCoreMCPExecutionStatus.TIMEOUT,
                error_message=f"Remote execution timed out after {self.config.timeout}s",
                error_code="REMOTE_TIMEOUT",
                mode=PiscesLxCoreMCPExecutionMode.REMOTE
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing tool {tool_name}: {e}")
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=execution_time,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=str(e),
                error_code="REMOTE_EXECUTION_ERROR",
                mode=PiscesLxCoreMCPExecutionMode.REMOTE
            )
    
    async def _send_request(self, request: Dict[str, Any]):
        """Send request to remote server."""
        if not self._writer:
            raise RuntimeError("Not connected to remote server")
        
        message = json.dumps(request).encode() + b"\n"
        self._writer.write(message)
        await self._writer.drain()
    
    async def _receive_response(self) -> Dict[str, Any]:
        """Receive response from remote server."""
        if not self._reader:
            raise RuntimeError("Not connected to remote server")
        
        data = await self._reader.readline()
        if not data:
            raise RuntimeError("Connection closed by remote server")
        
        return json.loads(data.decode().strip())


class PiscesLxCoreMCPRemoteClientPool:
    """Pool for managing multiple remote MCP clients."""
    
    def __init__(self, max_clients: int = 10):
        """Initialize client pool."""
        self.max_clients = max_clients
        self.clients: Dict[str, PiscesLxCoreMCPArcticRemoteClient] = {}
        self._lock = asyncio.Lock()
        
        logger.info(f"RemoteMCPClientPool initialized with max {max_clients} clients")
    
    async def get_client(self, client_id: str, config: Optional[_RemoteClientConfig] = None) -> PiscesLxCoreMCPArcticRemoteClient:
        """Get or create a client for the given ID."""
        async with self._lock:
            if client_id not in self.clients:
                if len(self.clients) >= self.max_clients:
                    # Remove oldest client
                    oldest_client_id = min(self.clients.keys(), key=lambda k: id(k))
                    await self.clients[oldest_client_id].disconnect()
                    del self.clients[oldest_client_id]
                
                self.clients[client_id] = PiscesLxCoreMCPArcticRemoteClient(client_id, config)
            
            return self.clients[client_id]
    
    async def remove_client(self, client_id: str) -> bool:
        """Remove a client from the pool."""
        async with self._lock:
            if client_id in self.clients:
                await self.clients[client_id].disconnect()
                del self.clients[client_id]
                return True
            return False
    
    async def shutdown(self):
        """Shutdown all clients in the pool."""
        async with self._lock:
            for client in self.clients.values():
                await client.disconnect()
            self.clients.clear()
        
        logger.info("RemoteMCPClientPool shutdown completed")


# Global client pool instance
_client_pool: Optional[PiscesLxCoreMCPRemoteClientPool] = None


def get_remote_client_pool(max_clients: int = 10) -> PiscesLxCoreMCPRemoteClientPool:
    """Get the global remote client pool instance."""
    global _client_pool
    if _client_pool is None:
        _client_pool = PiscesLxCoreMCPRemoteClientPool(max_clients)
    return _client_pool


async def execute_remote_tool(
    client_id: str,
    tool_name: str,
    parameters: Dict[str, Any],
    config: Optional[_RemoteClientConfig] = None
) -> PiscesLxCoreMCPExecutionResult:
    """
    Convenience function for remote tool execution.
    
    Args:
        client_id: Remote client identifier
        tool_name: Name of the tool to execute
        parameters: Parameters for the tool
        config: Optional client configuration
        
    Returns:
        Execution result
    """
    pool = get_remote_client_pool()
    client = await pool.get_client(client_id, config)
    
    return await client.execute_tool(tool_name, parameters)