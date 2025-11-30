#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

T = TypeVar('T')

class PiscesLxCoreMCPModuleStatus(Enum):
    """Status enumeration for MCP module loading states."""
    SUCCESS = "success"
    SKIPPED = "skipped"
    ERROR = "error"
    INCOMPATIBLE = "incompatible"

class PiscesLxCoreMCPToolCategory(Enum):
    """Tool category enumeration for MCP system."""
    PISCESL1_EXTENSION = "PiscesL1 Extension"
    FASTMCP_COMPATIBLE = "FastMCP Compatible"
    CUSTOM = "custom"
    UTILITY = "utility"
    ANALYTICS = "analytics"

@dataclass
class PiscesLxCoreMCPToolMetadata:
    """Metadata structure for tools in the PiscesLxCoreMCP system.
    
    Attributes:
        name: Unique identifier for the tool
        description: Brief explanation of tool functionality
        category: Classification grouping for the tool
        version: Current version string
        author: Creator or maintainer of the tool
        last_updated: Timestamp of last modification
        dependencies: Required external libraries or modules
        performance_score: Runtime efficiency metric (0.0-1.0)
        usage_count: Number of times tool has been executed
        error_rate: Frequency of execution failures (0.0-1.0)
        memory_usage: Memory consumption in bytes
        last_used: Timestamp of last execution
    """
    name: str
    description: str
    category: str
    version: str = "1.0.0"  # Will be updated to use centralized version
    author: str = "PiscesL1 Team"
    last_updated: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    performance_score: float = 1.0
    usage_count: int = 0
    error_rate: float = 0.0
    memory_usage: Optional[int] = None
    last_used: Optional[datetime] = None

@dataclass
class PiscesLxCoreMCPModuleStats:
    """Statistics tracking for MCP module loading and execution.
    
    Attributes:
        name: Name of the module/file being tracked
        load_time: Time taken to load the module in seconds
        tool_count: Number of tools discovered in the module
        status: Current state of the module
        error: Error message if loading failed
        memory_usage: Memory consumption in bytes
        last_used: Timestamp of last execution
        file_path: Path to the module file
        last_modified: File modification timestamp
    """
    name: str
    load_time: float
    tool_count: int
    status: PiscesLxCoreMCPModuleStatus
    error: Optional[str] = None
    memory_usage: Optional[int] = None
    last_used: Optional[datetime] = None
    file_path: Optional[Path] = None
    last_modified: Optional[datetime] = None

@dataclass
class PiscesLxCoreMCPExecutionContext:
    """Execution context for MCP tool operations.
    
    Attributes:
        session_id: Unique identifier for the session
        tool_name: Name of the tool being executed
        arguments: Input arguments for the tool
        start_time: Execution start timestamp
        end_time: Execution end timestamp
        result: Execution result
        error: Error information if execution failed
        performance_metrics: Performance statistics
    """
    session_id: str
    tool_name: str
    arguments: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PiscesLxCoreMCPSessionMemory:
    """Session memory structure for MCP tool operations.
    
    Attributes:
        session_id: Unique identifier for the session
        tool_memories: Dictionary of tool-specific memories
        context_history: Historical context information
        usage_statistics: Tool usage statistics
        created_at: Session creation timestamp
        last_accessed: Last access timestamp
    """
    session_id: str
    tool_memories: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    context_history: List[Dict[str, Any]] = field(default_factory=list)
    usage_statistics: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)

@dataclass
class PiscesLxCoreMCPPerformanceMetrics:
    """Performance metrics for MCP system monitoring.
    
    Attributes:
        total_executions: Total number of tool executions
        successful_executions: Number of successful executions
        failed_executions: Number of failed executions
        average_execution_time: Average execution time in seconds
        total_memory_usage: Total memory consumption
        error_rate: Overall system error rate
        uptime: System uptime in seconds
        last_health_check: Last health check timestamp
    """
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    total_memory_usage: int = 0
    error_rate: float = 0.0
    uptime: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)

@dataclass
class PiscesLxCoreMCPHealthStatus:
    """Health status information for MCP system.
    
    Attributes:
        status: Overall system status (healthy/degraded/unhealthy)
        component_statuses: Individual component health statuses
        alerts: Active alerts and warnings
        recommendations: Performance optimization recommendations
        last_updated: Last status update timestamp
    """
    status: str = "healthy"
    component_statuses: Dict[str, str] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class PiscesLxCoreMCPConfiguration:
    """Configuration settings for PiscesLxCoreMCP system.
    
    Attributes:
        discovery_interval: Tool discovery interval in seconds
        max_workers: Maximum thread pool workers
        cache_timeout: Cache timeout in seconds
        max_session_memory: Maximum session memory size
        blacklisted_tools: Set of blacklisted tool names
        enable_hot_reload: Whether to enable hot reloading
        enable_performance_monitoring: Whether to enable performance monitoring
        log_level: Logging level
        max_queue_size: Maximum queue size for execution queue
        monitoring_interval: Monitoring loop interval in seconds
        cleanup_interval: Cleanup loop interval in seconds
        error_recovery_interval: Error recovery interval in seconds
        max_execution_history: Maximum execution history size
        session_timeout: Session timeout in seconds
        memory_limit: Memory limit per session
        max_executions: Maximum executions per session
        max_memory_executions: Maximum memory executions per session
        max_tool_memory: Maximum tool memory per session
        allow_concurrent_tools: Whether to allow concurrent tool execution
        load_default_tools: Whether to load default tools
        load_tools: List of tools to load
        tool_paths: List of tool paths
        auto_reload: Whether to enable auto reload
        strict_validation: Whether to enable strict validation
        max_tool_cache_size: Maximum tool cache size
        verbose_logging: Whether to enable verbose logging
    """
    discovery_interval: int = 5
    max_workers: int = 4
    cache_timeout: int = 300
    max_session_memory: int = 1000
    blacklisted_tools: Set[str] = field(default_factory=lambda: {
        'calculator',  # Conflicts with PiscesL1's math module
        'translator',  # Conflicts with PiscesL1's multilingual module
        'weather',     # Conflicts with PiscesL1's real-time data module
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
class PiscesLxCoreMCPFileWatcherConfig:
    """Configuration for file watching and hot reload.
    
    Attributes:
        watch_directory: Directory to monitor for changes
        file_extensions: File extensions to watch
        check_interval: Check interval in seconds
        debounce_delay: Debounce delay in seconds
        enabled: Whether file watching is enabled
    """
    watch_directory: Path
    file_extensions: List[str] = field(default_factory=lambda: ['.py'])
    check_interval: int = 2
    debounce_delay: int = 1
    enabled: bool = True


@dataclass
class PiscesLxCoreMCPAgenticCall:
    """Represents a parsed agentic call from XML content."""
    tool_name: str
    parameters: Dict[str, Any]
    raw_match: str
    start_pos: int
    end_pos: int


class PiscesLxCoreMCPMessageType(Enum):
    """Message types for PiscesLxCore MCP protocol."""
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
class PiscesLxCoreMCPMessage:
    """Message structure for PiscesLxCore MCP protocol."""
    message_type: str
    agentic_id: str
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: str = ""
    priority: str = "normal"


@dataclass
class PiscesLxCoreAgenticAction:
    """Action structure for PiscesLxCore agentic system."""
    action_type: str
    parameters: Dict[str, Any]
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class PiscesLxCoreAgenticObservation:
    """Observation structure for PiscesLxCore agentic system."""
    modality: str  # "text", "image", "audio", "tool_result"
    content: Any
    metadata: Dict[str, Any]


class PiscesLxCoreMCPProtocol:
    """Protocol for handling Ruchbah MCP messages and interactions."""
    
    def __init__(self, agent_id: str):
        """Initialize the PiscesLxCoreMCPProtocol instance.

        Args:
            agent_id (str): The ID of the agent.
        """
        self.agent_id = agent_id
        self.execution_modes = {
            "native": "Direct native execution for zero-latency performance",
            "external": "External MCP server execution for extended capabilities",
            "auto": "Intelligent routing based on tool availability and performance"
        }
    
    @staticmethod
    def create_message(
        message_type: PiscesLxCoreMCPMessageType,
        agent_id: str,
        payload: Dict[str, Any],
        correlation_id: str = ""
    ) -> PiscesLxCoreMCPMessage:
        """Create a new MCP message.

        Args:
            message_type (PiscesLxCoreMCPMessageType): The type of the message.
            agent_id (str): The ID of the agent sending the message.
            payload (Dict[str, Any]): The payload of the message.
            correlation_id (str, optional): The correlation ID of the message. 
                If not provided, a new UUID will be generated. Defaults to "".

        Returns:
            PiscesLxCoreMCPMessage: A new PiscesLxCoreMCPMessage instance.
        """
        from datetime import datetime
        import uuid
        return PiscesLxCoreMCPMessage(
            message_type=message_type.value,
            agentic_id=agent_id,
            payload=payload,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id or str(uuid.uuid4())
        )
    
    async def create_tool_call_message(self, tool_name: str, arguments: Dict[str, Any], 
                                       execution_mode: str = "auto") -> PiscesLxCoreMCPMessage:
        """Create a tool call message.

        Args:
            tool_name (str): The name of the tool to call.
            arguments (Dict[str, Any]): The arguments to pass to the tool.
            execution_mode (str): The execution mode ("native", "external", or "auto").

        Returns:
            PiscesLxCoreMCPMessage: The created tool call message.
        """
        from datetime import datetime
        return PiscesLxCoreMCPMessage(
            message_type=PiscesLxCoreMCPMessageType.TOOL_CALL.value,
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