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
Remote MCP Client Connection Manager.

This module manages multiple remote MCP client connections, providing connection
pooling, load balancing, and failover capabilities for user-local MCP clients.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator
import asyncio
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from .types import (
    RemoteClientConfig, RemoteConnectionState, RemoteExecutionResult,
    RemoteExecutionMode, RemoteConnectionError, RemoteClientInfo
)
from .client import ArcticRemoteMCPClient
from model.mcp.translator import ArcticMCPTranslationLayer as ArcticRemoteMCPTranslator, ArcticAgentCall
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("Arctic.Remote.MCP.Manager")

@dataclass
class ClientPoolEntry:
    """Represents a client in the connection pool."""
    client: ArcticRemoteMCPClient
    config: RemoteClientConfig
    created_at: float
    last_used: float
    usage_count: int = 0
    consecutive_errors: int = 0
    is_healthy: bool = True
    
    def mark_used(self) -> None:
        """Mark client as used."""
        self.last_used = time.time()
        self.usage_count += 1
        self.consecutive_errors = 0
    
    def mark_error(self) -> None:
        """Mark client as having an error."""
        self.consecutive_errors += 1
        if self.consecutive_errors >= 3:  # Threshold for marking unhealthy
            self.is_healthy = False

class ArcticRemoteMCPManager:
    """
    Connection manager for remote MCP clients.
    
    Manages multiple user-local MCP client connections with connection pooling,
    health monitoring, and intelligent routing capabilities.
    """
    
    def __init__(self, max_connections: int = 10, connection_timeout: float = 30.0):
        """
        Initialize the connection manager.
        
        Args:
            max_connections: Maximum number of concurrent connections
            connection_timeout: Timeout for connection operations
        """
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self._client_pools: Dict[str, ClientPoolEntry] = {}
        self._active_clients: Dict[str, ArcticRemoteMCPClient] = {}
        self._translator = ArcticRemoteMCPTranslator()
        self._manager_lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized Remote MCP Manager with max {max_connections} connections")
    
    async def start(self) -> None:
        """Start the manager services."""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Remote MCP Manager started")
    
    async def stop(self) -> None:
        """Stop the manager services and cleanup connections."""
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        
        # Disconnect all clients
        await self._disconnect_all_clients()
        
        logger.info("Remote MCP Manager stopped")
    
    async def register_client(self, config: RemoteClientConfig) -> str:
        """
        Register a new remote client configuration.
        
        Args:
            config: Client configuration
            
        Returns:
            Client ID for the registered client
            
        Raises:
            RemoteConnectionError: If registration fails
        """
        async with self._manager_lock:
            client_id = config.client_id or str(uuid.uuid4())
            
            if client_id in self._client_pools:
                logger.warning(f"Client {client_id} already registered")
                return client_id
            
            try:
                # Create client
                client = ArcticRemoteMCPClient(config)
                
                # Create pool entry
                pool_entry = ClientPoolEntry(
                    client=client,
                    config=config,
                    created_at=time.time(),
                    last_used=0.0
                )
                
                self._client_pools[client_id] = pool_entry
                
                logger.info(f"Registered remote client: {client_id} -> {config.base_url}")
                return client_id
                
            except Exception as e:
                logger.error(f"Failed to register client {client_id}: {e}")
                raise RemoteConnectionError(f"Registration failed: {e}")
    
    async def unregister_client(self, client_id: str) -> bool:
        """
        Unregister a remote client.
        
        Args:
            client_id: Client ID to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        async with self._manager_lock:
            if client_id not in self._client_pools:
                return False
            
            try:
                pool_entry = self._client_pools[client_id]
                
                # Disconnect if connected
                if pool_entry.client.is_connected():
                    await pool_entry.client.disconnect()
                
                # Remove from pools
                del self._client_pools[client_id]
                
                # Remove from active clients if present
                if client_id in self._active_clients:
                    del self._active_clients[client_id]
                
                logger.info(f"Unregistered remote client: {client_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error unregistering client {client_id}: {e}")
                return False
    
    async def execute_tool(self, client_id: str, tool_name: str, 
                          parameters: Dict[str, Any]) -> RemoteExecutionResult:
        """
        Execute a tool on a specific remote client.
        
        Args:
            client_id: Target client ID
            tool_name: Tool name to execute
            parameters: Tool parameters
            
        Returns:
            Execution result
            
        Raises:
            RemoteConnectionError: If client not found or connection fails
        """
        async with self._manager_lock:
            if client_id not in self._client_pools:
                raise RemoteConnectionError(f"Client {client_id} not found")
            
            pool_entry = self._client_pools[client_id]
            
            if not pool_entry.is_healthy:
                raise RemoteConnectionError(f"Client {client_id} is not healthy")
        
        try:
            # Ensure client is connected
            if not await pool_entry.client.is_connected():
                connected = await pool_entry.client.connect()
                if not connected:
                    raise RemoteConnectionError(f"Failed to connect to client {client_id}")
            
            # Execute tool
            result = await pool_entry.client.execute_tool(tool_name, parameters)
            
            # Update pool entry
            pool_entry.mark_used()
            
            return result
            
        except Exception as e:
            pool_entry.mark_error()
            logger.error(f"Tool execution failed for client {client_id}: {e}")
            raise
    
    async def execute_tool_call(self, tool_call: ArcticAgentCall, client_id: Optional[str] = None) -> RemoteExecutionResult:
        """
        Execute a tool call on a remote client.
        
        Args:
            tool_call: Tool call to execute (ArcticAgentCall)
            client_id: Target client ID (optional, will auto-select if not provided)
            
        Returns:
            Execution result
            
        Raises:
            RemoteConnectionError: If no suitable client found or execution fails
        """
        if client_id is None:
            # Auto-select best client
            client_id = await self._select_best_client(tool_call.tool_name)
            if not client_id:
                raise RemoteConnectionError("No suitable client available for tool execution")
        
        # Execute using regular tool execution
        return await self.execute_tool(client_id, tool_call.tool_name, tool_call.parameters)
    
    async def list_client_tools(self, client_id: str) -> List[Dict[str, Any]]:
        """
        List available tools for a specific client.
        
        Args:
            client_id: Client ID
            
        Returns:
            List of available tools
            
        Raises:
            RemoteConnectionError: If client not found
        """
        async with self._manager_lock:
            if client_id not in self._client_pools:
                raise RemoteConnectionError(f"Client {client_id} not found")
            
            pool_entry = self._client_pools[client_id]
        
        try:
            # Ensure client is connected
            if not await pool_entry.client.is_connected():
                connected = await pool_entry.client.connect()
                if not connected:
                    raise RemoteConnectionError(f"Failed to connect to client {client_id}")
            
            # List tools
            tools = await pool_entry.client.list_available_tools()
            
            # Update pool entry
            pool_entry.mark_used()
            
            return tools
            
        except Exception as e:
            pool_entry.mark_error()
            logger.error(f"Tool listing failed for client {client_id}: {e}")
            raise
    
    async def get_client_info(self, client_id: str) -> RemoteClientInfo:
        """
        Get information about a specific client.
        
        Args:
            client_id: Client ID
            
        Returns:
            Client information
            
        Raises:
            RemoteConnectionError: If client not found
        """
        async with self._manager_lock:
            if client_id not in self._client_pools:
                raise RemoteConnectionError(f"Client {client_id} not found")
            
            pool_entry = self._client_pools[client_id]
        
        try:
            return await pool_entry.client.protocol.get_client_info()
            
        except Exception as e:
            logger.error(f"Failed to get client info for {client_id}: {e}")
            raise
    
    async def get_all_clients_status(self) -> Dict[str, Any]:
        """
        Get status of all registered clients.
        
        Returns:
            Dictionary containing status information for all clients
        """
        async with self._manager_lock:
            status = {
                "total_clients": len(self._client_pools),
                "healthy_clients": 0,
                "connected_clients": 0,
                "clients": {}
            }
            
            for client_id, pool_entry in self._client_pools.items():
                try:
                    is_connected = await pool_entry.client.is_connected()
                    connection_status = await pool_entry.client.get_connection_status()
                    
                    client_info = {
                        "base_url": pool_entry.config.base_url,
                        "is_healthy": pool_entry.is_healthy,
                        "is_connected": is_connected,
                        "usage_count": pool_entry.usage_count,
                        "consecutive_errors": pool_entry.consecutive_errors,
                        "created_at": pool_entry.created_at,
                        "last_used": pool_entry.last_used,
                        "connection_status": connection_status
                    }
                    
                    status["clients"][client_id] = client_info
                    
                    if pool_entry.is_healthy:
                        status["healthy_clients"] += 1
                    
                    if is_connected:
                        status["connected_clients"] += 1
                        
                except Exception as e:
                    logger.error(f"Error getting status for client {client_id}: {e}")
                    status["clients"][client_id] = {
                        "error": str(e),
                        "is_healthy": False,
                        "is_connected": False
                    }
            
            return status
    
    @asynccontextmanager
    async def client_connection(self, client_id: str):
        """
        Context manager for client connection.
        
        Usage:
            async with manager.client_connection("client_1") as client:
                result = await client.execute_tool("tool_name", {})
        """
        async with self._manager_lock:
            if client_id not in self._client_pools:
                raise RemoteConnectionError(f"Client {client_id} not found")
            
            pool_entry = self._client_pools[client_id]
        
        try:
            # Ensure connection
            if not await pool_entry.client.is_connected():
                connected = await pool_entry.client.connect()
                if not connected:
                    raise RemoteConnectionError(f"Failed to connect to client {client_id}")
            
            yield pool_entry.client
            
        finally:
            # Keep connection alive for potential reuse
            pass
    
    async def _health_check_loop(self) -> None:
        """Health check loop for all clients."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                async with self._manager_lock:
                    for client_id, pool_entry in list(self._client_pools.items()):
                        try:
                            # Check if client is still healthy
                            if pool_entry.consecutive_errors >= 3:
                                pool_entry.is_healthy = False
                                logger.warning(f"Client {client_id} marked as unhealthy")
                            
                            # Try to recover unhealthy clients
                            elif not pool_entry.is_healthy:
                                # Attempt reconnection
                                if await pool_entry.client.connect():
                                    pool_entry.is_healthy = True
                                    pool_entry.consecutive_errors = 0
                                    logger.info(f"Client {client_id} recovered")
                                
                        except Exception as e:
                            logger.error(f"Health check failed for client {client_id}: {e}")
                            pool_entry.mark_error()
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _select_best_client(self, tool_name: str) -> Optional[str]:
        """
        Select the best client for executing a specific tool.
        
        Args:
            tool_name: Name of the tool to execute
            
        Returns:
            Best client ID or None if no suitable client found
        """
        async with self._manager_lock:
            best_client_id = None
            best_score = -1
            
            for client_id, pool_entry in self._client_pools.items():
                if not pool_entry.is_healthy:
                    continue
                
                # Score based on health, usage, and connection status
                score = 0
                
                # Health bonus
                if pool_entry.is_healthy:
                    score += 10
                
                # Connection bonus
                try:
                    if await pool_entry.client.is_connected():
                        score += 5
                except:
                    continue
                
                # Usage penalty (prefer less used clients)
                score -= min(pool_entry.usage_count // 10, 5)
                
                # Error penalty
                score -= pool_entry.consecutive_errors * 2
                
                if score > best_score:
                    best_score = score
                    best_client_id = client_id
            
            return best_client_id

    async def _cleanup_loop(self) -> None:
        """Cleanup loop for idle connections."""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                current_time = time.time()
                
                async with self._manager_lock:
                    for client_id, pool_entry in list(self._client_pools.items()):
                        try:
                            # Disconnect idle clients (idle for more than 10 minutes)
                            if (pool_entry.last_used > 0 and 
                                current_time - pool_entry.last_used > 600 and
                                await pool_entry.client.is_connected()):
                                
                                await pool_entry.client.disconnect()
                                logger.info(f"Disconnected idle client {client_id}")
                                
                        except Exception as e:
                            logger.error(f"Cleanup failed for client {client_id}: {e}")
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _disconnect_all_clients(self) -> None:
        """Disconnect all clients."""
        async with self._manager_lock:
            for client_id, pool_entry in list(self._client_pools.items()):
                try:
                    if await pool_entry.client.is_connected():
                        await pool_entry.client.disconnect()
                        logger.debug(f"Disconnected client {client_id}")
                        
                except Exception as e:
                    logger.error(f"Error disconnecting client {client_id}: {e}")
    
    def __repr__(self) -> str:
        """String representation of the manager."""
        return f"ArcticRemoteMCPManager(total_clients={len(self._client_pools)}, max_connections={self.max_connections})"