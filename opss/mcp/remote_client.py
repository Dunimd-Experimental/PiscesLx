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

import asyncio
import json
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

from utils.dc import PiscesLxLogger

class POPSSMCPMessageType(Enum):
    TOOL_CALL = 1
    TOOL_RESULT = 2
    TOOL_ERROR = 3
    HEARTBEAT = 4
    HEARTBEAT_ACK = 5
    DISCONNECT = 6
    REGISTER = 7
    REGISTER_ACK = 8

@dataclass
class POPSSMCPMessage:
    message_type: POPSSMCPMessageType
    payload: Dict[str, Any]
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

class POPSSMCPClientConfig:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        connection_timeout: float = 10.0,
        heartbeat_interval: float = 30.0,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 5,
        auto_reconnect: bool = True
    ):
        self.host = host
        self.port = port
        self.connection_timeout = connection_timeout
        self.heartbeat_interval = heartbeat_interval
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.auto_reconnect = auto_reconnect

class POPSSMCPClient:
    def __init__(self, config: Optional[POPSSMCPClientConfig] = None):
        self.config = config or POPSSMCPClientConfig()
        self._LOG = self._configure_logging()
        
        self._socket: Optional[socket.socket] = None
        self._connected = False
        self._connection_lock = threading.Lock()
        
        self._message_handlers: Dict[POPSSMCPMessageType, List[Callable]] = {
            POPSSMCPMessageType.TOOL_RESULT: [],
            POPSSMCPMessageType.TOOL_ERROR: [],
            POPSSMCPMessageType.HEARTBEAT_ACK: [],
            POPSSMCPMessageType.REGISTER_ACK: [],
        }
        
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._pending_lock = threading.Lock()
        
        self._receive_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop_event = threading.Event()
        
        self._server_info: Optional[Dict[str, Any]] = None
        self._registered_tools: List[str] = []
        
        self._reconnect_attempts = 0
        
        self._LOG.info("POPSSMCPClient initialized")
    
    def _configure_logging(self) -> PiscesLxLogger:
        _LOG = get_logger("PiscesLx.Core.MCP.Remote")
        return _LOG
    
    def connect(self) -> bool:
        with self._connection_lock:
            if self._connected:
                self._LOG.warning("Already connected")
                return True
            
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self.config.connection_timeout)
                self._socket.connect((self.config.host, self.config.port))
                self._socket.settimeout(None)
                
                self._connected = True
                self._reconnect_attempts = 0
                
                self._start_receive_loop()
                
                if self.config.heartbeat_interval > 0:
                    self._start_heartbeat()
                
                self._LOG.info(f"Connected to {self.config.host}:{self.config.port}")
                return True
                
            except socket.timeout:
                self._LOG.error(f"Connection timed out to {self.config.host}:{self.config.port}")
                return False
            except ConnectionRefusedError:
                self._LOG.error(f"Connection refused by {self.config.host}:{self.config.port}")
                return False
            except Exception as e:
                self._LOG.error(f"Failed to connect: {e}")
                return False
    
    def disconnect(self):
        with self._connection_lock:
            if not self._connected:
                return
            
            self._stop_event.set()
            self._heartbeat_stop_event.set()
            
            if self._heartbeat_thread:
                self._heartbeat_thread.join(timeout=2)
            
            if self._receive_thread:
                self._receive_thread.join(timeout=2)
            
            try:
                if self._socket:
                    self._socket.close()
            except Exception:
                pass
            
            self._connected = False
            self._socket = None
            
            self._LOG.info("Disconnected from MCP server")
    
    def is_connected(self) -> bool:
        return self._connected
    
    def _start_receive_loop(self):
        self._stop_event.clear()
        self._receive_thread = threading.Thread(
            target=self._receive_loop,
            name="piscesl1_mcp_receive",
            daemon=True
        )
        self._receive_thread.start()
    
    def _receive_loop(self):
        buffer = b""
        
        while not self._stop_event.is_set():
            try:
                data = self._socket.recv(4096)
                if not data:
                    self._LOG.warning("Server closed connection")
                    self._handle_disconnect()
                    break
                
                buffer += data
                
                while len(buffer) >= 4:
                    length = struct.unpack('!I', buffer[:4])[0]
                    
                    if len(buffer) < 4 + length:
                        break
                    
                    message_data = buffer[4:4 + length]
                    buffer = buffer[4 + length:]
                    
                    try:
                        message = self._deserialize_message(message_data)
                        self._process_message(message)
                    except Exception as e:
                        self._LOG.error(f"Error processing message: {e}")
                        
            except socket.timeout:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    self._LOG.error(f"Receive error: {e}")
                    self._handle_disconnect()
                break
    
    def _start_heartbeat(self):
        self._heartbeat_stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="piscesl1_mcp_heartbeat",
            daemon=True
        )
        self._heartbeat_thread.start()
    
    def _heartbeat_loop(self):
        while not self._heartbeat_stop_event.is_set():
            try:
                time.sleep(self.config.heartbeat_interval)
                
                if self._heartbeat_stop_event.is_set():
                    break
                
                if self._connected:
                    self._send_message(POPSSMCPMessage(
                        message_type=POPSSMCPMessageType.HEARTBEAT,
                        payload={'timestamp': time.time()}
                    ))
                    
            except Exception as e:
                if not self._heartbeat_stop_event.is_set():
                    self._LOG.error(f"Heartbeat error: {e}")
    
    def _handle_disconnect(self):
        was_connected = self._connected
        self._connected = False
        
        if was_connected:
            self._LOG.warning("Disconnected from server")
            
            for request_id, future in list(self._pending_requests.items()):
                if not future.done():
                    future.set_exception(ConnectionError("Connection lost"))
            
            self._pending_requests.clear()
            
            if self.config.auto_reconnect and self._reconnect_attempts < self.config.max_reconnect_attempts:
                self._reconnect_attempts += 1
                self._LOG.info(f"Attempting to reconnect ({self._reconnect_attempts}/{self.config.max_reconnect_attempts})...")
                
                time.sleep(self.config.reconnect_delay)
                
                if self.connect():
                    self._LOG.info("Reconnected successfully")
                else:
                    self._LOG.error("Reconnect failed")
    
    def _serialize_message(self, message: POPSSMCPMessage) -> bytes:
        data = {
            'type': message.message_type.value,
            'payload': message.payload,
            'correlation_id': message.correlation_id,
            'timestamp': message.timestamp
        }
        
        json_data = json.dumps(data, default=str).encode('utf-8')
        
        length = struct.pack('!I', len(json_data))
        
        return length + json_data
    
    def _deserialize_message(self, data: bytes) -> POPSSMCPMessage:
        message_data = json.loads(data.decode('utf-8'))
        
        message_type = POPSSMCPMessageType(message_data['type'])
        
        return POPSSMCPMessage(
            message_type=message_type,
            payload=message_data.get('payload', {}),
            correlation_id=message_data.get('correlation_id', str(uuid.uuid4())),
            timestamp=message_data.get('timestamp', time.time())
        )
    
    def _send_message(self, message: POPSSMCPMessage) -> bool:
        with self._connection_lock:
            if not self._connected:
                self._LOG.error("Cannot send message: not connected")
                return False
            
            try:
                serialized = self._serialize_message(message)
                self._socket.sendall(serialized)
                return True
                
            except Exception as e:
                self._LOG.error(f"Failed to send message: {e}")
                self._handle_disconnect()
                return False
    
    def _process_message(self, message: POPSSMCPMessage):
        message_type = message.message_type
        
        if message_type in self._message_handlers:
            for handler in self._message_handlers[message_type]:
                try:
                    handler(message)
                except Exception as e:
                    self._LOG.error(f"Error in message handler: {e}")
        
        if message_type in [POPSSMCPMessageType.TOOL_RESULT, POPSSMCPMessageType.TOOL_ERROR]:
            correlation_id = message.payload.get('correlation_id')
            
            if correlation_id and correlation_id in self._pending_requests:
                future = self._pending_requests.pop(correlation_id)
                
                if message_type == POPSSMCPMessageType.TOOL_RESULT:
                    future.set_result(message.payload.get('result'))
                else:
                    future.set_exception(Exception(message.payload.get('error', 'Unknown error')))
    
    def register_message_handler(self, message_type: POPSSMCPMessageType, handler: Callable):
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        
        self._message_handlers[message_type].append(handler)
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: float = 60.0
    ) -> Any:
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        
        correlation_id = str(uuid.uuid4())
        
        future = asyncio.get_event_loop().create_future()
        
        with self._pending_lock:
            self._pending_requests[correlation_id] = future
        
        message = POPSSMCPMessage(
            message_type=POPSSMCPMessageType.TOOL_CALL,
            payload={
                'tool_name': tool_name,
                'arguments': arguments,
                'correlation_id': correlation_id
            },
            correlation_id=correlation_id
        )
        
        if not self._send_message(message):
            with self._pending_lock:
                self._pending_requests.pop(correlation_id, None)
            raise ConnectionError("Failed to send message")
        
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            with self._pending_lock:
                self._pending_requests.pop(correlation_id, None)
            raise TimeoutError(f"Tool call timed out: {tool_name}")
    
    def call_tool_sync(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: float = 60.0
    ) -> Any:
        correlation_id = str(uuid.uuid4())
        
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        
        with self._pending_lock:
            self._pending_requests[correlation_id] = future
        
        message = POPSSMCPMessage(
            message_type=POPSSMCPMessageType.TOOL_CALL,
            payload={
                'tool_name': tool_name,
                'arguments': arguments,
                'correlation_id': correlation_id
            },
            correlation_id=correlation_id
        )
        
        if not self._send_message(message):
            with self._pending_lock:
                self._pending_requests.pop(correlation_id, None)
            raise ConnectionError("Failed to send message")
        
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            with self._pending_lock:
                self._pending_requests.pop(correlation_id, None)
            raise
    
    def register_tools(self, tools: List[Dict[str, Any]]) -> bool:
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        
        correlation_id = str(uuid.uuid4())
        
        message = POPSSMCPMessage(
            message_type=POPSSMCPMessageType.REGISTER,
            payload={
                'tools': tools,
                'correlation_id': correlation_id
            },
            correlation_id=correlation_id
        )
        
        return self._send_message(message)
    
    def get_server_info(self) -> Optional[Dict[str, Any]]:
        return self._server_info
    
    def get_registered_tools(self) -> List[str]:
        return self._registered_tools.copy()
    
    def shutdown(self):
        self.disconnect()
        self._LOG.info("POPSSMCPClient shutdown complete")


class POPSSMCPMessageProtocol:
    @staticmethod
    def create_tool_call(tool_name: str, arguments: Dict[str, Any], correlation_id: Optional[str] = None) -> POPSSMCPMessage:
        return POPSSMCPMessage(
            message_type=POPSSMCPMessageType.TOOL_CALL,
            payload={
                'tool_name': tool_name,
                'arguments': arguments
            },
            correlation_id=correlation_id or str(uuid.uuid4())
        )
    
    @staticmethod
    def create_tool_result(result: Any, correlation_id: str) -> POPSSMCPMessage:
        return POPSSMCPMessage(
            message_type=POPSSMCPMessageType.TOOL_RESULT,
            payload={
                'result': result,
                'correlation_id': correlation_id
            },
            correlation_id=correlation_id
        )
    
    @staticmethod
    def create_tool_error(error: str, correlation_id: str) -> POPSSMCPMessage:
        return POPSSMCPMessage(
            message_type=POPSSMCPMessageType.TOOL_ERROR,
            payload={
                'error': error,
                'correlation_id': correlation_id
            },
            correlation_id=correlation_id
        )
    
    @staticmethod
    def create_heartbeat() -> POPSSMCPMessage:
        return POPSSMCPMessage(
            message_type=POPSSMCPMessageType.HEARTBEAT,
            payload={'timestamp': time.time()}
        )
    
    @staticmethod
    def parse_payload(message: POPSSMCPMessage) -> Dict[str, Any]:
        return message.payload
