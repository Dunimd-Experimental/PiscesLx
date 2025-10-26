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
Remote MCP Protocol implementation for user-local client communication.

This module implements the protocol layer that enables communication between
the PiscesL1 model server and user-local MCP clients using official MCP protocol
over HTTP/SSE connections.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, AsyncGenerator
import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError
from urllib.parse import urljoin

from .types import (
    RemoteClientConfig, RemoteToolCall, RemoteExecutionResult, 
    RemoteConnectionState, RemoteExecutionMode, RemoteClientInfo,
    RemoteConnectionError, RemoteExecutionError, RemoteAuthenticationError
)
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("Arctic.Remote.MCP.Protocol")

class ArcticRemoteMCPProtocol:
    """
    Protocol handler for remote MCP client communication.
    
    Implements the client-side protocol for connecting to user-local MCP servers
    and executing tools using official MCP protocol over HTTP/SSE.
    """
    
    def __init__(self, config: RemoteClientConfig):
        """
        Initialize the remote MCP protocol handler.
        
        Args:
            config: Configuration for the remote client connection
        """
        self.config = config
        self.session: Optional[ClientSession] = None
        self.connection_state = RemoteConnectionState.DISCONNECTED
        self._connection_lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_handlers: Dict[str, callable] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
    async def connect(self) -> bool:
        """
        Establish connection to the remote MCP server.
        
        Returns:
            True if connection successful, False otherwise
        """
        async with self._connection_lock:
            if self.connection_state != RemoteConnectionState.DISCONNECTED:
                return True
                
            try:
                self.connection_state = RemoteConnectionState.CONNECTING
                
                # Create session with timeout configuration
                timeout = ClientTimeout(total=self.config.timeout)
                connector = aiohttp.TCPConnector(
                    limit=self.config.connection_pool_size,
                    ssl=self.config.enable_tls
                )
                
                self.session = ClientSession(
                    timeout=timeout,
                    connector=connector,
                    headers={
                        'Authorization': f'Bearer {self.config.auth_token}' if self.config.auth_token else '',
                        'Content-Type': 'application/json',
                        'User-Agent': 'PiscesL1-RemoteMCP-Client/1.0'
                    }
                )
                
                # Test connection and perform handshake
                if await self._perform_handshake():
                    self.connection_state = RemoteConnectionState.AUTHENTICATED
                    self._start_heartbeat()
                    logger.info(f"Connected to remote MCP server: {self.config.base_url}")
                    return True
                else:
                    self.connection_state = RemoteConnectionState.ERROR
                    await self._cleanup()
                    return False
                    
            except Exception as e:
                self.connection_state = RemoteConnectionState.ERROR
                logger.error(f"Failed to connect to remote MCP server: {e}")
                await self._cleanup()
                return False
    
    async def disconnect(self) -> None:
        """Disconnect from the remote MCP server."""
        async with self._connection_lock:
            if self.connection_state == RemoteConnectionState.DISCONNECTED:
                return
                
            self.connection_state = RemoteConnectionState.DISCONNECTED
            
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                self._heartbeat_task = None
                
            await self._cleanup()
            logger.info("Disconnected from remote MCP server")
    
    async def execute_tool(self, tool_call: RemoteToolCall) -> RemoteExecutionResult:
        """
        Execute a tool on the remote MCP server.
        
        Args:
            tool_call: The tool call to execute remotely
            
        Returns:
            Execution result from the remote server
            
        Raises:
            RemoteConnectionError: If not connected to server
            RemoteExecutionError: If tool execution fails
        """
        if self.connection_state != RemoteConnectionState.AUTHENTICATED:
            raise RemoteConnectionError("Not connected to remote MCP server")
        
        start_time = time.time()
        
        try:
            # Convert to official MCP protocol format
            mcp_request = self._convert_to_mcp_request(tool_call)
            
            # Send request to remote server
            response = await self._send_mcp_request("tools/call", mcp_request)
            
            # Convert response to our format
            result = self._convert_from_mcp_response(response, tool_call)
            result.execution_time = time.time() - start_time
            
            logger.debug(f"Remote tool execution completed: {tool_call.tool_name}")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Remote tool execution failed: {e}")
            
            return RemoteExecutionResult(
                success=False,
                result=None,
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                execution_time=execution_time,
                execution_mode=RemoteExecutionMode.REMOTE,
                error_message=str(e),
                error_code="EXECUTION_ERROR"
            )
    
    async def list_available_tools(self) -> list:
        """
        Get list of available tools from the remote MCP server.
        
        Returns:
            List of available tool names
            
        Raises:
            RemoteConnectionError: If not connected to server
        """
        if self.connection_state != RemoteConnectionState.AUTHENTICATED:
            raise RemoteConnectionError("Not connected to remote MCP server")
        
        try:
            response = await self._send_mcp_request("tools/list", {})
            return response.get("tools", [])
            
        except Exception as e:
            logger.error(f"Failed to list remote tools: {e}")
            return []
    
    async def get_client_info(self) -> RemoteClientInfo:
        """
        Get information about the remote client.
        
        Returns:
            Client information
            
        Raises:
            RemoteConnectionError: If not connected to server
        """
        if self.connection_state != RemoteConnectionState.AUTHENTICATED:
            raise RemoteConnectionError("Not connected to remote MCP server")
        
        try:
            response = await self._send_mcp_request("info", {})
            
            return RemoteClientInfo(
                client_id=self.config.client_id,
                connection_state=self.connection_state,
                connected_at=response.get("connected_at", ""),
                last_heartbeat=response.get("last_heartbeat", ""),
                available_tools=response.get("available_tools", []),
                client_version=response.get("version", "unknown"),
                capabilities=response.get("capabilities", {}),
                performance_metrics=response.get("performance_metrics")
            )
            
        except Exception as e:
            logger.error(f"Failed to get client info: {e}")
            raise
    
    async def _perform_handshake(self) -> bool:
        """Perform MCP handshake with the remote server."""
        try:
            handshake_data = {
                "client_id": self.config.client_id,
                "protocol_version": "1.0",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": True},
                    "prompts": {"listChanged": True}
                }
            }
            
            response = await self._send_mcp_request("initialize", handshake_data)
            
            # Check if handshake was successful
            if response.get("result") == "ok":
                return True
            else:
                logger.error(f"Handshake failed: {response.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Handshake exception: {e}")
            return False
    
    def _start_heartbeat(self) -> None:
        """Start the heartbeat task to maintain connection."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop to maintain connection."""
        while self.connection_state == RemoteConnectionState.AUTHENTICATED:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                heartbeat_data = {
                    "timestamp": time.time(),
                    "client_id": self.config.client_id
                }
                
                await self._send_mcp_request("heartbeat", heartbeat_data)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
                # Don't break on heartbeat failure, try to recover
    
    async def _send_mcp_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an MCP request to the remote server.
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response data
            
        Raises:
            RemoteConnectionError: If request fails
        """
        if not self.session:
            raise RemoteConnectionError("No active session")
        
        url = urljoin(self.config.base_url, f"/mcp/{endpoint}")
        
        try:
            async with self.session.post(url, json=data) as response:
                if response.status == 401:
                    raise RemoteAuthenticationError("Authentication failed")
                elif response.status != 200:
                    raise RemoteConnectionError(f"HTTP {response.status}: {await response.text()}")
                
                result = await response.json()
                
                if result.get("error"):
                    raise RemoteExecutionError(result["error"].get("message", "Unknown error"))
                
                return result.get("result", result)
                
        except ClientError as e:
            raise RemoteConnectionError(f"Request failed: {e}")
    
    def _convert_to_mcp_request(self, tool_call: RemoteToolCall) -> Dict[str, Any]:
        """
        Convert our tool call format to official MCP protocol.
        
        Args:
            tool_call: Our internal tool call format
            
        Returns:
            MCP protocol compliant request
        """
        return {
            "jsonrpc": "2.0",
            "id": tool_call.call_id,
            "method": "tools/call",
            "params": {
                "name": tool_call.tool_name,
                "arguments": tool_call.parameters
            }
        }
    
    def _convert_from_mcp_response(self, mcp_response: Dict[str, Any], 
                                  original_call: RemoteToolCall) -> RemoteExecutionResult:
        """
        Convert MCP protocol response to our format.
        
        Args:
            mcp_response: MCP protocol response
            original_call: Original tool call
            
        Returns:
            Our execution result format
        """
        success = mcp_response.get("success", True)
        
        return RemoteExecutionResult(
            success=success,
            result=mcp_response.get("result"),
            tool_name=original_call.tool_name,
            call_id=original_call.call_id,
            execution_time=0.0,  # Will be set by caller
            execution_mode=RemoteExecutionMode.REMOTE,
            error_message=mcp_response.get("error", {}).get("message") if not success else None,
            error_code=mcp_response.get("error", {}).get("code") if not success else None,
            metadata=mcp_response.get("metadata")
        )
    
    async def _cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None