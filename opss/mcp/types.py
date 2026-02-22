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

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set, Union, TypeVar
from pathlib import Path
from enum import Enum
from configs.version import VERSION

T = TypeVar('T')

class POPSSMCPModuleStatus(Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    ERROR = "error"
    INCOMPATIBLE = "incompatible"

class POPSSMCPToolCategory(Enum):
    PISCESL1_EXTENSION = "PiscesL1 Extension"
    FASTMCP_COMPATIBLE = "FastMCP Compatible"
    CUSTOM = "custom"
    UTILITY = "utility"
    ANALYTICS = "analytics"


class POPSSMCPToolType(Enum):
    NATIVE = "native"
    EXTERNAL = "external"
    HYBRID = "hybrid"

@dataclass
class POPSSMCPToolMetadata:
    name: str
    description: str
    category: str
    version: str = VERSION
    author: str = "PiscesL1 Team"
    last_updated: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    performance_score: float = 1.0
    usage_count: int = 0
    error_rate: float = 0.0
    memory_usage: Optional[int] = None
    last_used: Optional[datetime] = None

@dataclass
class POPSSMCPModuleStats:
    name: str
    load_time: float
    tool_count: int
    status: POPSSMCPModuleStatus
    error: Optional[str] = None
    memory_usage: Optional[int] = None
    last_used: Optional[datetime] = None
    file_path: Optional[Path] = None
    last_modified: Optional[datetime] = None

@dataclass
class POPSSMCPExecutionContext:
    session_id: str
    tool_name: str
    arguments: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class POPSSMCPSessionMemory:
    session_id: str
    tool_memories: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    context_history: List[Dict[str, Any]] = field(default_factory=list)
    usage_statistics: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)

@dataclass
class POPSSMCPPerformanceMetrics:
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    total_memory_usage: int = 0
    error_rate: float = 0.0
    uptime: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)

@dataclass
class POPSSMCPHealthStatus:
    status: str = "healthy"
    component_statuses: Dict[str, str] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class POPSSMCPConfiguration:
    discovery_interval: int = 5
    max_workers: int = 4
    cache_timeout: int = 300
    max_session_memory: int = 1000
    blacklisted_tools: Set[str] = field(default_factory=lambda: {
        'calculator',
        'translator',
        'weather',
    })
    enable_hot_reload: bool = True
    enable_performance_monitoring: bool = True
    log_level: str = "INFO"
    max_queue_size: int = 1000
    monitoring_interval: int = 30
    cleanup_interval: int = 300
    error_recovery_interval: int = 60
    max_execution_history: int = 1000
    session_timeout: int = 3600
    memory_limit: int = 100
    max_executions: int = 1000
    max_memory_executions: int = 100
    max_tool_memory: int = 20
    allow_concurrent_tools: bool = True
    load_default_tools: bool = True
    load_tools: List[str] = field(default_factory=list)
    tool_paths: List[str] = field(default_factory=list)
    auto_reload: bool = False
    strict_validation: bool = True
    max_tool_cache_size: int = 100
    verbose_logging: bool = False

@dataclass
class POPSSMCPFileWatcherConfig:
    watch_directory: Path
    file_extensions: List[str] = field(default_factory=lambda: ['.py'])
    check_interval: int = 2
    debounce_delay: int = 1
    enabled: bool = True

@dataclass
class POPSSMCPAgenticCall:
    tool_name: str
    parameters: Dict[str, Any]
    raw_match: str
    start_pos: int
    end_pos: int

class POPSSMCPMessageType(Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE_UPDATE = "state_update"
    CAPABILITY_REGISTER = "capability_register"
    HEARTBEAT = "heartbeat"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"

@dataclass
class POPSSMCPMessage:
    message_type: str
    agentic_id: str
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: str = ""
    priority: str = "normal"

@dataclass
class POPSSAgenticAction:
    action_type: str
    parameters: Dict[str, Any]
    confidence: float = 1.0
    reasoning: str = ""

@dataclass
class POPSSAgenticObservation:
    modality: str
    content: Any
    metadata: Dict[str, Any]

class POPSSMCPProtocol:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.execution_modes = {
            "native": "Direct native execution for zero-latency performance",
            "external": "External MCP server execution for extended capabilities",
            "auto": "Intelligent routing based on tool availability and performance"
        }
    
    @staticmethod
    def create_message(
        message_type: POPSSMCPMessageType,
        agent_id: str,
        payload: Dict[str, Any],
        correlation_id: str = ""
    ) -> POPSSMCPMessage:
        from datetime import datetime
        import uuid
        return POPSSMCPMessage(
            message_type=message_type.value,
            agentic_id=agent_id,
            payload=payload,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id or str(uuid.uuid4())
        )
    
    async def create_tool_call_message(self, tool_name: str, arguments: Dict[str, Any], 
                                       execution_mode: str = "auto") -> POPSSMCPMessage:
        from datetime import datetime
        return POPSSMCPMessage(
            message_type=POPSSMCPMessageType.TOOL_CALL.value,
            agentic_id=self.agent_id,
            payload={
                "tool_name": tool_name,
                "arguments": arguments,
                "timestamp": datetime.now().isoformat(),
                "execution_mode": execution_mode,
                "dual_track_enabled": True
            },
            timestamp=datetime.now().isoformat()
        )
