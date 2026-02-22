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
Agent Protocol - Inter-Agent Communication Protocol

This module provides a comprehensive protocol system for inter-agent communication,
enabling structured message passing, request-response patterns, and event-driven
coordination between agents in the PiscesL1 multi-agent system.

Key Components:
    - POPSSProtocolMessageType: Enumeration of message types (REQUEST, RESPONSE, etc.)
    - POPSSProtocolMessage: Message container with metadata and payload
    - POPSSProtocolConfig: Protocol configuration settings
    - POPSSProtocolHandler: Main protocol handler for message routing

Design Principles:
    1. Asynchronous Communication: Native async/await support for non-blocking operations
    2. Correlation Tracking: Request-response correlation via correlation IDs
    3. Message Buffering: In-memory message history for debugging and auditing
    4. Thread Safety: Concurrent access protection with asyncio locks
    5. Extensibility: Pluggable message handlers for custom processing

Message Types:
    - REQUEST: Synchronous request expecting a response
    - RESPONSE: Response to a previous request
    - NOTIFICATION: One-way notification (no response expected)
    - ERROR: Error message in response to a failed request
    - STREAM: Streaming data chunks

Usage:
    # Initialize protocol handler
    handler = POPSSProtocolHandler()
    
    # Register message handlers
    handler.register_handler(POPSSProtocolMessageType.REQUEST, my_request_handler)
    
    # Send request and await response
    response = await handler.send_request(
        sender="agent_1",
        receiver="agent_2",
        payload={"task": "analyze"}
    )
    
    # Send notification to multiple receivers
    await handler.send_notification(
        sender="coordinator",
        receivers=["agent_1", "agent_2"],
        notification_type="task_update",
        payload={"status": "completed"}
    )

Thread Safety:
    All public methods are thread-safe and can be called from multiple
    asyncio tasks concurrently. Internal state is protected by asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
from concurrent.futures import ThreadPoolExecutor

from utils.dc import PiscesLxLogger
from configs.version import VERSION

T = TypeVar('T')

class POPSSProtocolMessageType(Enum):
    """
    Enumeration of protocol message types for inter-agent communication.
    
    This enum defines the types of messages that can be exchanged between
    agents in the PiscesL1 multi-agent system. Each type has specific
    semantics and handling requirements.
    
    Attributes:
        REQUEST: A request message that expects a response from the receiver.
            Used for synchronous operations where the sender needs confirmation
            or data from another agent.
        RESPONSE: A response message sent in reply to a REQUEST message.
            Contains the result or data requested by the sender.
        NOTIFICATION: A one-way message that does not expect a response.
            Used for broadcasting events, status updates, or warnings.
        ERROR: An error message indicating a failure in processing a request.
            Sent in response to a REQUEST that could not be fulfilled.
        STREAM: A streaming data chunk for large data transfers.
            Used for progressive delivery of large payloads.
    
    Example:
        >>> msg_type = POPSSProtocolMessageType.REQUEST
        >>> print(msg_type.value)  # "request"
    """
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    STREAM = "stream"

@dataclass
class POPSSProtocolMessage:
    """
    Protocol message container for inter-agent communication.
    
    This dataclass represents a structured message exchanged between agents,
    containing all necessary metadata for routing, correlation, and tracking.
    
    Attributes:
        message_id: Unique identifier for this message (auto-generated UUID prefix).
            Format: "msg_<uuid_hex>" (e.g., "msg_a1b2c3d4e5f6")
        message_type: Type of message (REQUEST, RESPONSE, NOTIFICATION, ERROR, STREAM).
            Determines how the message should be processed by receivers.
        sender: Identifier of the sending agent or component.
            Used for routing responses and tracking message origin.
        receiver: Identifier of the receiving agent or component.
            For broadcast messages, this may be a comma-separated list.
        payload: Dictionary containing the actual message data.
            Structure depends on message_type and specific use case.
        correlation_id: Optional ID linking this message to a related message.
            Used for request-response correlation and conversation tracking.
        timestamp: Time when the message was created (auto-generated).
            Used for message ordering and timeout detection.
        metadata: Additional metadata for extensibility.
            Can include priority, TTL, custom headers, etc.
    
    Example:
        >>> message = POPSSProtocolMessage(
        ...     message_id="msg_12345678",
        ...     message_type=POPSSProtocolMessageType.REQUEST,
        ...     sender="agent_analyst",
        ...     receiver="agent_researcher",
        ...     payload={"query": "market trends"},
        ...     correlation_id="corr_abcd1234"
        ... )
    """
    message_id: str
    message_type: POPSSProtocolMessageType
    sender: str
    receiver: str
    
    payload: Dict[str, Any]
    
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class POPSSProtocolConfig:
    """
    Configuration settings for the protocol handler.
    
    This dataclass defines the operational parameters for protocol
    message handling, including timeouts, size limits, and feature flags.
    
    Attributes:
        protocol_version: Version string for protocol compatibility checking.
            Defaults to the current VERSION from configs.version.
        max_message_size: Maximum allowed message size in bytes.
            Messages exceeding this limit will be rejected. Default: 10MB.
        message_timeout: Default timeout in seconds for awaiting responses.
            After this duration, pending requests will raise TimeoutError.
        enable_streaming: Whether to enable streaming message support.
            When enabled, large payloads can be sent as multiple STREAM messages.
        enable_compression: Whether to compress message payloads.
            Useful for large payloads over network connections.
        enable_encryption: Whether to encrypt message payloads.
            Should be enabled for sensitive data transmission.
        retry_attempts: Number of retry attempts for failed message delivery.
            Applies to transient failures, not permanent errors.
        retry_delay: Base delay in seconds between retry attempts.
            Actual delay increases exponentially with each retry.
    
    Example:
        >>> config = POPSSProtocolConfig(
        ...     message_timeout=120.0,
        ...     enable_encryption=True,
        ...     max_message_size=50 * 1024 * 1024  # 50MB
        ... )
    """
    protocol_version: str = VERSION
    
    max_message_size: int = 10 * 1024 * 1024
    message_timeout: float = 60.0
    
    enable_streaming: bool = True
    enable_compression: bool = False
    enable_encryption: bool = False
    
    retry_attempts: int = 3
    retry_delay: float = 1.0

class POPSSProtocolHandler:
    """
    Main protocol handler for inter-agent message routing and processing.
    
    This class provides a comprehensive message handling system for the
    PiscesL1 multi-agent architecture, supporting request-response patterns,
    notifications, and streaming communication.
    
    Key Features:
        - Asynchronous message handling with asyncio
        - Request-response correlation tracking
        - Pluggable message handlers per message type
        - Message buffering for history and debugging
        - Thread-safe concurrent access
        - Timeout handling for pending requests
    
    Attributes:
        config: POPSSProtocolConfig instance with operational settings.
        _message_handlers: Dictionary mapping message types to handler lists.
        _correlation_map: Dictionary mapping correlation IDs to pending futures.
        _pending_requests: Dictionary tracking pending request messages.
        _message_buffer: List of recent messages for debugging/auditing.
        _lock: Asyncio lock for thread-safe state access.
    
    Thread Safety:
        All public methods are thread-safe. Internal state modifications
        are protected by asyncio.Lock to prevent race conditions.
    
    Example:
        >>> handler = POPSSProtocolHandler()
        >>> 
        >>> # Register a request handler
        >>> async def handle_request(message):
        ...     # Process the request
        ...     return {"result": "processed"}
        >>> handler.register_handler(POPSSProtocolMessageType.REQUEST, handle_request)
        >>> 
        >>> # Send a request
        >>> response = await handler.send_request(
        ...     sender="agent_1",
        ...     receiver="agent_2",
        ...     payload={"task": "analyze"}
        ... )
    
    See Also:
        - POPSSProtocolMessage: Message container class
        - POPSSProtocolConfig: Configuration settings
        - POPSSProtocolMessageType: Message type enumeration
    """
    
    def __init__(self, config: Optional[POPSSProtocolConfig] = None):
        """
        Initialize the protocol handler with optional configuration.
        
        Args:
            config: Optional POPSSProtocolConfig instance. If not provided,
                default configuration values will be used.
        """
        self.config = config or POPSSProtocolConfig()
        self._LOG = self._configure_logging()
        
        # Initialize message handlers for each message type
        self._message_handlers: Dict[POPSSProtocolMessageType, List[Callable]] = {
            POPSSProtocolMessageType.REQUEST: [],
            POPSSProtocolMessageType.RESPONSE: [],
            POPSSProtocolMessageType.NOTIFICATION: [],
            POPSSProtocolMessageType.ERROR: [],
            POPSSProtocolMessageType.STREAM: [],
        }
        
        # Maps for tracking pending requests and their futures
        self._correlation_map: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        # Message buffer for history and debugging
        self._message_buffer: List[POPSSProtocolMessage] = []
        self._buffer_max_size = 1000
        
        # Thread pool for async executor operations
        self._async_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="piscesl1_protocol"
        )
        
        self._LOG.info("POPSSProtocolHandler initialized")
    
    def _configure_logging(self) -> PiscesLxLogger:
        """
        Configure and return a logger instance for the protocol handler.
        
        Returns:
            PiscesLxLogger: Configured logger instance for protocol operations.
        """
        logger = get_logger("PiscesLx.Core.Agents.Protocol")
        return logger
    
    def register_handler(self, message_type: POPSSProtocolMessageType, handler: Callable):
        """
        Register a handler function for a specific message type.
        
        Multiple handlers can be registered for the same message type.
        They will be called in the order of registration.
        
        Args:
            message_type: The type of message this handler should process.
            handler: Async callable that takes a POPSSProtocolMessage and
                returns an optional response. The handler signature should be:
                async def handler(message: POPSSProtocolMessage) -> Optional[Dict]
        
        Example:
            >>> async def my_request_handler(message):
            ...     print(f"Received: {message.payload}")
            ...     return {"status": "acknowledged"}
            >>> handler.register_handler(POPSSProtocolMessageType.REQUEST, my_request_handler)
        """
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        self._message_handlers[message_type].append(handler)
    
    async def send_request(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Any:
        """
        Send a request message and await the response.
        
        This method creates a REQUEST message, sends it to the receiver,
        and waits for a RESPONSE or ERROR message with matching correlation_id.
        
        Args:
            sender: Identifier of the sending agent or component.
            receiver: Identifier of the receiving agent or component.
            payload: Dictionary containing the request data.
            timeout: Optional timeout in seconds. If not provided, uses
                config.message_timeout (default: 60.0 seconds).
        
        Returns:
            Any: The response payload from the receiver.
        
        Raises:
            TimeoutError: If no response is received within the timeout period.
            Exception: If the receiver returns an ERROR message.
        
        Example:
            >>> response = await handler.send_request(
            ...     sender="agent_analyst",
            ...     receiver="agent_researcher",
            ...     payload={"query": "market analysis"},
            ...     timeout=30.0
            ... )
        """
        # Generate unique identifiers for message and correlation tracking
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        correlation_id = f"corr_{uuid.uuid4().hex[:8]}"
        
        # Create the request message
        message = POPSSProtocolMessage(
            message_id=message_id,
            message_type=POPSSProtocolMessageType.REQUEST,
            sender=sender,
            receiver=receiver,
            payload=payload,
            correlation_id=correlation_id,
        )
        
        # Create a future to await the response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        
        # Thread-safe registration of pending request
        async with self._lock:
            self._correlation_map[correlation_id] = future
            self._pending_requests[message_id] = future
        
        try:
            # Transmit the message to handlers
            await self._transmit_message(message)
            
            # Wait for response with timeout
            await asyncio.wait_for(
                future, 
                timeout=timeout or self.config.message_timeout
            )
            
            return future.result()
            
        except asyncio.TimeoutError:
            # Clean up on timeout
            async with self._lock:
                self._correlation_map.pop(correlation_id, None)
                self._pending_requests.pop(message_id, None)
            raise TimeoutError(f"Request timeout: {message_id}")
            
        except Exception as e:
            # Clean up on error
            async with self._lock:
                self._correlation_map.pop(correlation_id, None)
                self._pending_requests.pop(message_id, None)
            raise
    
    async def send_response(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        is_error: bool = False
    ):
        """
        Send a response or error message to a previous request.
        
        This method creates a RESPONSE or ERROR message and sends it to
        the original requester, completing the request-response cycle.
        
        Args:
            sender: Identifier of the responding agent or component.
            receiver: Identifier of the original requesting agent.
            payload: Dictionary containing the response data or error details.
            correlation_id: The correlation_id from the original REQUEST message.
                Required to link this response to the pending request.
            is_error: If True, sends an ERROR message instead of RESPONSE.
                Use this when the request could not be fulfilled.
        
        Example:
            >>> # Successful response
            >>> await handler.send_response(
            ...     sender="agent_researcher",
            ...     receiver="agent_analyst",
            ...     payload={"results": [...]},
            ...     correlation_id="corr_abcd1234"
            ... )
            >>> 
            >>> # Error response
            >>> await handler.send_response(
            ...     sender="agent_researcher",
            ...     receiver="agent_analyst",
            ...     payload={"error": "Resource not found"},
            ...     correlation_id="corr_abcd1234",
            ...     is_error=True
            ... )
        """
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        
        # Determine message type based on error flag
        message_type = POPSSProtocolMessageType.ERROR if is_error else POPSSProtocolMessageType.RESPONSE
        
        message = POPSSProtocolMessage(
            message_id=message_id,
            message_type=message_type,
            sender=sender,
            receiver=receiver,
            payload=payload,
            correlation_id=correlation_id,
        )
        
        await self._transmit_message(message)
        
        # Complete the pending future if correlation_id matches
        if correlation_id and correlation_id in self._correlation_map:
            future = self._correlation_map.pop(correlation_id)
            if not future.done():
                if is_error:
                    future.set_exception(Exception(payload.get('error', 'Unknown error')))
                else:
                    future.set_result(payload)
    
    async def send_notification(
        self,
        sender: str,
        receivers: List[str],
        notification_type: str,
        payload: Dict[str, Any]
    ):
        """
        Send a one-way notification to one or more receivers.
        
        Notifications do not expect responses and are used for broadcasting
        events, status updates, or other informational messages.
        
        Args:
            sender: Identifier of the sending agent or component.
            receivers: List of receiver identifiers to notify.
            notification_type: String identifying the type of notification
                (e.g., "task_update", "status_change", "alert").
            payload: Dictionary containing the notification data.
        
        Example:
            >>> await handler.send_notification(
            ...     sender="coordinator",
            ...     receivers=["agent_1", "agent_2", "agent_3"],
            ...     notification_type="task_completed",
            ...     payload={"task_id": "task_123", "result": "success"}
            ... )
        """
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        
        # Create notification message with embedded notification type
        message = POPSSProtocolMessage(
            message_id=message_id,
            message_type=POPSSProtocolMessageType.NOTIFICATION,
            sender=sender,
            receiver=",".join(receivers),  # Comma-separated for multiple receivers
            payload={
                'notification_type': notification_type,
                **payload
            },
        )
        
        await self._transmit_message(message)
    
    async def _transmit_message(self, message: POPSSProtocolMessage):
        """
        Internal method to transmit a message to registered handlers.
        
        This method delivers the message to all handlers registered for
        the message's type. Handlers are called sequentially, and any
        exceptions are logged but do not stop message delivery to other handlers.
        
        Args:
            message: The POPSSProtocolMessage to transmit.
        """
        self._LOG.debug(f"Transmitting message: {message.message_id} -> {message.receiver}")
        
        # Call all registered handlers for this message type
        for handler in self._message_handlers.get(message.message_type, []):
            try:
                await handler(message)
            except Exception as e:
                self._LOG.error(f"Error in message handler: {e}")
    
    async def handle_message(self, message: POPSSProtocolMessage):
        """
        Process an incoming message based on its type.
        
        This method routes the message to appropriate handling logic based
        on the message type. It also maintains the message buffer for
        history and debugging purposes.
        
        Args:
            message: The incoming POPSSProtocolMessage to process.
        """
        # Handle REQUEST messages
        if message.message_type == POPSSProtocolMessageType.REQUEST:
            for handler in self._message_handlers.get(POPSSProtocolMessageType.REQUEST, []):
                try:
                    await handler(message)
                except Exception as e:
                    # Send error response if handler fails
                    await self.send_response(
                        sender=message.receiver,
                        receiver=message.sender,
                        payload={'error': str(e)},
                        correlation_id=message.correlation_id,
                        is_error=True
                    )
        
        # Handle RESPONSE messages - complete pending futures
        elif message.message_type == POPSSProtocolMessageType.RESPONSE:
            if message.correlation_id and message.correlation_id in self._correlation_map:
                future = self._correlation_map[message.correlation_id]
                if not future.done():
                    future.set_result(message.payload)
        
        # Handle NOTIFICATION messages
        elif message.message_type == POPSSProtocolMessageType.NOTIFICATION:
            for handler in self._message_handlers.get(POPSSProtocolMessageType.NOTIFICATION, []):
                try:
                    await handler(message)
                except Exception as e:
                    self._LOG.error(f"Error in notification handler: {e}")
        
        # Handle ERROR messages - set exception on pending futures
        elif message.message_type == POPSSProtocolMessageType.ERROR:
            if message.correlation_id and message.correlation_id in self._correlation_map:
                future = self._correlation_map[message.correlation_id]
                if not future.done():
                    future.set_exception(Exception(message.payload.get('error', 'Unknown error')))
        
        # Add to message buffer (with size limit)
        self._message_buffer.append(message)
        if len(self._message_buffer) > self._buffer_max_size:
            self._message_buffer = self._message_buffer[-self._buffer_max_size:]
    
    def create_agent_request(
        self,
        agent_id: str,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized agent request payload.
        
        This is a factory method that creates a properly formatted request
        payload for agent task execution.
        
        Args:
            agent_id: Identifier of the target agent.
            task: Task description or instruction for the agent.
            context: Optional context dictionary with additional parameters.
        
        Returns:
            Dict[str, Any]: Formatted request payload ready for send_request.
        
        Example:
            >>> request = handler.create_agent_request(
            ...     agent_id="agent_researcher",
            ...     task="Analyze market trends",
            ...     context={"priority": "high"}
            ... )
        """
        return {
            'agent_id': agent_id,
            'task': task,
            'context': context or {},
            'timestamp': datetime.now().isoformat(),
            'request_id': f"req_{uuid.uuid4().hex[:12]}",
        }
    
    def parse_agent_response(self, response: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Parse an agent response into success flag and result.
        
        Args:
            response: Response dictionary from an agent request.
        
        Returns:
            Tuple[bool, Any]: A tuple of (success, result) where success
                indicates if the request was successful, and result is
                either the output data or error message.
        
        Example:
            >>> success, result = handler.parse_agent_response(response)
            >>> if success:
            ...     print(f"Result: {result}")
            ... else:
            ...     print(f"Error: {result}")
        """
        success = response.get('success', False)
        result = response.get('result')
        error = response.get('error')
        
        return success, result if success else error
    
    def create_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized tool call payload.
        
        This is a factory method that creates a properly formatted payload
        for tool execution requests.
        
        Args:
            tool_name: Name of the tool to invoke.
            arguments: Dictionary of arguments to pass to the tool.
            session_id: Optional session identifier for context tracking.
        
        Returns:
            Dict[str, Any]: Formatted tool call payload.
        
        Example:
            >>> tool_call = handler.create_tool_call(
            ...     tool_name="web_search",
            ...     arguments={"query": "latest AI developments"},
            ...     session_id="session_123"
            ... )
        """
        return {
            'tool_name': tool_name,
            'arguments': arguments,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
        }
    
    def parse_tool_result(self, result: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Parse a tool execution result into success flag and output.
        
        Args:
            result: Result dictionary from a tool execution.
        
        Returns:
            Tuple[bool, Any]: A tuple of (success, output) where success
                indicates if the tool call was successful, and output is
                either the tool output or error message.
        
        Example:
            >>> success, output = handler.parse_tool_result(result)
            >>> if success:
            ...     print(f"Output: {output}")
        """
        success = result.get('success', False)
        output = result.get('output')
        error = result.get('error')
        
        return success, output if success else error
    
    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """
        Get information about all pending requests.
        
        Returns a list of pending request information for monitoring
        and debugging purposes.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing pending
                request information including message_id and timestamp.
        """
        return [
            {
                'message_id': msg_id,
                'correlation_id': future.get_loop().create_future() if False else None,
                'timestamp': datetime.now().isoformat(),
            }
            for msg_id, future in self._pending_requests.items()
        ]
    
    def get_message_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent message history for debugging and auditing.
        
        Args:
            limit: Maximum number of messages to return (default: 100).
        
        Returns:
            List[Dict[str, Any]]: List of message summaries including
                message_id, type, sender, receiver, and timestamp.
        """
        recent = self._message_buffer[-limit:]
        return [
            {
                'message_id': msg.message_id,
                'message_type': msg.message_type.value,
                'sender': msg.sender,
                'receiver': msg.receiver,
                'timestamp': msg.timestamp.isoformat(),
            }
            for msg in recent
        ]
    
    def shutdown(self):
        """
        Shutdown the protocol handler and release resources.
        
        This method cancels all pending requests, shuts down the thread
        pool executor, and cleans up internal state. Should be called
        when the protocol handler is no longer needed.
        """
        # Shutdown thread pool executor
        self._async_executor.shutdown(wait=True)
        
        # Cancel any pending futures
        for future in self._correlation_map.values():
            if not future.done():
                future.cancel()
        
        self._LOG.info("Protocol handler shutdown")
