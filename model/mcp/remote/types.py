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
Type definitions for remote MCP client support.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from enum import Enum

class RemoteExecutionMode(Enum):
    """Execution modes for remote tool calls."""
    NATIVE = "native"      # Execute on server
    REMOTE = "remote"      # Execute on user client  
    AUTO = "auto"         # Smart routing decision

class RemoteConnectionState(Enum):
    """Connection states for remote clients."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"

@dataclass
class RemoteClientConfig:
    """Configuration for remote MCP client connections."""
    client_id: str
    base_url: str
    auth_token: Optional[str] = None
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_tls: bool = True
    verify_cert: bool = True
    connection_pool_size: int = 10
    heartbeat_interval: float = 60.0
    max_concurrent_calls: int = 5

@dataclass 
class RemoteToolCall:
    """Represents a tool call to be executed remotely."""
    tool_name: str
    parameters: Dict[str, Any]
    execution_mode: RemoteExecutionMode
    session_id: str
    call_id: str
    priority: int = 1
    timeout: Optional[float] = None
    
@dataclass
class RemoteExecutionResult:
    """Result of a remote tool execution."""
    success: bool
    result: Any
    tool_name: str
    call_id: str
    execution_time: float
    execution_mode: RemoteExecutionMode
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RemoteClientInfo:
    """Information about a connected remote client."""
    client_id: str
    connection_state: RemoteConnectionState
    connected_at: str
    last_heartbeat: str
    available_tools: List[str]
    client_version: str
    capabilities: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]] = None

@dataclass
class RemoteRoutingDecision:
    """Decision made by the remote routing engine."""
    tool_name: str
    execution_mode: RemoteExecutionMode
    reason: str
    confidence: float
    alternative_tools: List[str]
    estimated_latency: Optional[float] = None
    
class RemoteMCPError(Exception):
    """Base exception for remote MCP operations."""
    pass

class RemoteConnectionError(RemoteMCPError):
    """Exception raised when remote connection fails."""
    pass

class RemoteExecutionError(RemoteMCPError):
    """Exception raised when remote execution fails."""
    pass

class RemoteAuthenticationError(RemoteMCPError):
    """Exception raised when remote authentication fails."""
    pass