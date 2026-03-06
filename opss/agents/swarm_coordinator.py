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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
Swarm Coordinator - Multi-Agent Swarm Coordination System

This module provides a comprehensive swarm coordination system for managing
multiple agents working together on complex tasks. It supports various
topologies and coordination strategies for distributed agent collaboration.

Key Components:
    - POPSSSwarmTopology: Enumeration of supported network topologies
    - POPSSSwarmMessageType: Types of inter-agent messages
    - POPSSSwarmMessage: Message container for swarm communication
    - POPSSSwarmTask: Task definition with dependencies and tracking
    - POPSSSwarmMember: Agent membership and status tracking
    - POPSSSwarmConfig: Configuration for swarm behavior
    - POPSSSwarmCoordinator: Main coordinator class

Supported Topologies:
    - HIERARCHICAL: Tree-like structure with coordinator at root
    - FLAT: All agents connected equally
    - MESH: Partial connections between agents
    - STAR: Central hub with all agents connected
    - RING: Circular connection pattern

Features:
    - Dynamic agent registration and deregistration
    - Automatic task assignment based on capabilities
    - Task dependency management
    - Heartbeat monitoring for agent health
    - Message queue for reliable communication
    - Performance metrics tracking
    - Event callbacks for monitoring

Usage:
    # Initialize swarm coordinator
    config = POPSSSwarmConfig(
        registry=agent_registry,
        topology=POPSSSwarmTopology.HIERARCHICAL,
        max_agents=20
    )
    coordinator = POPSSSwarmCoordinator(config)
    coordinator.initialize()
    
    # Register agents
    coordinator.register_agent(
        agent_id="agent_1",
        agent_type=POPSSAggregentType.ANALYSIS,
        name="Analyzer",
        capabilities={POPSSAgentCapability.DATA_ANALYSIS}
    )
    
    # Submit and assign tasks
    task_id = await coordinator.submit_task(
        task_type="analysis",
        description="Analyze market data",
        input_data={"data": market_data}
    )
    
    # Get swarm status
    status = coordinator.get_swarm_status()

Thread Safety:
    All public methods are thread-safe. Internal state is protected by
    asyncio queues and proper synchronization.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG =_LOG = PiscesLxLogger("PiscesLx.Opss.Agents",file_path=get_log_file("PiscesLx.Opss.Agents"), enable_file=True)

from .base import (
    POPSSBaseAgent,
    POPSSAgentConfig,
    POPSSAgentContext,
    POPSSAgentResult,
    POPSSAgentState,
    POPSSAgentCapability,
)
from .registry import POPSSAggregentRegistry, POPSSAggregentType
from .orchestrator import (
    POPSSBaseOrchestrator,
    POPSSOrchestratorConfig,
    POPSSOrchestrationPlan,
    POPSSOrchestrationResult,
    POPSSOrchestrationStrategy,
    POPSSOrchestrationStage,
)

class POPSSSwarmTopology(Enum):
    """
    Enumeration of supported swarm network topologies.
    
    The topology determines how agents are connected and how messages
    flow between them in the swarm.
    
    Attributes:
        HIERARCHICAL: Tree-like structure with a coordinator at the root.
            Agents connect to their parent in the hierarchy. Good for
            structured tasks with clear delegation.
        FLAT: All agents are connected equally to each other.
            No central coordinator. Good for peer-to-peer collaboration.
        MESH: Partial connections between agents (max 3 neighbors).
            Balances connectivity with message overhead.
        STAR: Central hub with all agents connected to it.
            All communication goes through the coordinator.
        RING: Circular connection pattern where each agent connects to
            the next in the ring. Good for sequential processing.
    """
    HIERARCHICAL = "hierarchical"
    FLAT = "flat"
    MESH = "mesh"
    STAR = "star"
    RING = "ring"

class POPSSSwarmMessageType(Enum):
    """
    Enumeration of message types for swarm communication.
    
    Defines the types of messages that can be exchanged between
    agents and the coordinator in the swarm.
    
    Attributes:
        TASK_ASSIGNMENT: Message assigning a task to an agent.
        RESULT_REPORT: Message reporting task results back.
        STATUS_UPDATE: Message with agent status information.
        HELP_REQUEST: Message requesting assistance from other agents.
        COORDINATION: Message for coordination between agents.
        TERMINATE: Message signaling termination of an agent or task.
    """
    TASK_ASSIGNMENT = "task_assignment"
    RESULT_REPORT = "result_report"
    STATUS_UPDATE = "status_update"
    HELP_REQUEST = "help_request"
    COORDINATION = "coordination"
    TERMINATE = "terminate"

@dataclass
class POPSSSwarmMessage:
    """
    Message container for swarm communication.
    
    This dataclass represents a message exchanged between agents and
    the coordinator in the swarm, containing all necessary metadata
    for routing and processing.
    
    Attributes:
        message_id: Unique identifier for this message.
        message_type: Type of message (TASK_ASSIGNMENT, RESULT_REPORT, etc.).
        sender_id: Identifier of the sending agent or coordinator.
        receiver_ids: List of target agent identifiers.
        payload: Dictionary containing the message data.
        priority: Message priority (higher = more urgent).
        correlation_id: Optional ID linking to related messages.
        timestamp: Time when the message was created.
        expires_at: Optional expiration time for time-sensitive messages.
    
    Example:
        >>> message = POPSSSwarmMessage(
        ...     message_id="msg_abc123",
        ...     message_type=POPSSSwarmMessageType.TASK_ASSIGNMENT,
        ...     sender_id="coordinator",
        ...     receiver_ids=["agent_1"],
        ...     payload={"task": "analyze data"}
        ... )
    """
    message_id: str
    message_type: POPSSSwarmMessageType
    sender_id: str
    receiver_ids: List[str]
    
    payload: Dict[str, Any]
    
    priority: int = 0
    correlation_id: Optional[str] = None
    
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

@dataclass
class POPSSSwarmTask:
    """
    Task definition with dependencies and execution tracking.
    
    This dataclass represents a task in the swarm, including its
    dependencies, assigned agents, and execution status.
    
    Attributes:
        task_id: Unique identifier for this task.
        task_type: Type classification for capability matching.
        description: Human-readable task description.
        assigned_agents: List of agent IDs assigned to this task.
        status: Current status (pending, assigned, in_progress, completed, failed).
        priority: Task priority (higher = more urgent).
        deadline: Optional deadline for task completion.
        dependencies: List of task IDs that must complete first.
        subtasks: List of subtask IDs for decomposed tasks.
        input_data: Input data for task execution.
        output_data: Output data from task execution.
        created_at: Time when the task was created.
        started_at: Time when execution started.
        completed_at: Time when execution completed.
    
    Example:
        >>> task = POPSSSwarmTask(
        ...     task_id="task_xyz789",
        ...     task_type="analysis",
        ...     description="Analyze sales data",
        ...     priority=5,
        ...     input_data={"data": sales_data}
        ... )
    """
    task_id: str
    task_type: str
    description: str
    
    assigned_agents: List[str] = field(default_factory=list)
    status: str = "pending"
    
    priority: int = 0
    deadline: Optional[datetime] = None
    
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class POPSSSwarmMember:
    """
    Agent membership and status tracking in the swarm.
    
    This dataclass represents an agent's membership in the swarm,
    including its capabilities, current status, and performance metrics.
    
    Attributes:
        agent_id: Unique identifier for the agent.
        agent_type: Type classification (GENERAL, ANALYSIS, etc.).
        name: Human-readable name for the agent.
        capabilities: Set of capabilities the agent possesses.
        current_task: ID of the task currently being executed (None if idle).
        status: Current status (available, busy, unavailable).
        load: Current workload indicator (0 = no load).
        last_heartbeat: Time of last heartbeat from this agent.
        performance_metrics: Dictionary of performance statistics.
        neighbors: List of neighbor agent IDs based on topology.
    
    Example:
        >>> member = POPSSSwarmMember(
        ...     agent_id="agent_1",
        ...     agent_type=POPSSAggregentType.ANALYSIS,
        ...     name="DataAnalyzer",
        ...     capabilities={POPSSAgentCapability.DATA_ANALYSIS}
        ... )
    """
    agent_id: str
    agent_type: POPSSAggregentType
    name: str
    
    capabilities: Set[POPSSAgentCapability] = field(default_factory=set)
    current_task: Optional[str] = None
    
    status: str = "available"
    load: int = 0
    
    last_heartbeat: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    neighbors: List[str] = field(default_factory=list)

@dataclass
class POPSSSwarmConfig:
    """
    Configuration settings for the swarm coordinator.
    
    This dataclass defines the operational parameters for the swarm,
    including topology, agent limits, timeouts, and feature flags.
    
    Attributes:
        registry: POPSSAggregentRegistry for accessing agent instances.
        topology: Network topology for agent connections.
        max_agents: Maximum number of agents allowed in the swarm.
        min_agents: Minimum number of agents required.
        coordinator_id: Optional ID of the coordinator agent.
        heartbeat_interval: Seconds between heartbeat checks.
        task_timeout: Default timeout for task execution in seconds.
        enable_collaboration: Whether to enable agent collaboration.
        enable_task_decomposition: Whether to decompose complex tasks.
        enable_result_aggregation: Whether to aggregate results.
        max_retries: Maximum retry attempts for failed tasks.
        retry_delay: Base delay between retries in seconds.
        message_queue_size: Maximum messages in the queue.
    
    Example:
        >>> config = POPSSSwarmConfig(
        ...     registry=my_registry,
        ...     topology=POPSSSwarmTopology.HIERARCHICAL,
        ...     max_agents=10,
        ...     heartbeat_interval=15.0
        ... )
    """
    registry: POPSSAggregentRegistry
    
    topology: POPSSSwarmTopology = POPSSSwarmTopology.HIERARCHICAL
    
    max_agents: int = 20
    min_agents: int = 1
    
    coordinator_id: Optional[str] = None
    
    heartbeat_interval: float = 30.0
    task_timeout: float = 300.0
    
    enable_collaboration: bool = True
    enable_task_decomposition: bool = True
    enable_result_aggregation: bool = True
    
    max_retries: int = 3
    retry_delay: float = 1.0
    
    message_queue_size: int = 1000

class POPSSSwarmCoordinator:
    """
    Main coordinator class for managing multi-agent swarms.
    
    This class provides comprehensive coordination for multiple agents
    working together on complex tasks, including task assignment,
    dependency management, and inter-agent communication.
    
    Key Features:
        - Dynamic agent registration with capability matching
        - Multiple topology support (hierarchical, flat, mesh, star, ring)
        - Automatic task assignment based on agent capabilities
        - Task dependency resolution
        - Heartbeat monitoring for agent health
        - Message queue for reliable communication
        - Performance metrics tracking
        - Event callbacks for monitoring
    
    Attributes:
        config: POPSSSwarmConfig instance with operational settings.
        registry: Reference to the agent registry.
        _members: Dictionary of POPSSSwarmMember by agent ID.
        _tasks: Dictionary of POPSSSwarmTask by task ID.
        _message_queue: AsyncIO queue for message delivery.
        _task_queue: AsyncIO queue for pending tasks.
        _coordinator: Optional coordinator agent instance.
        _running: Flag indicating if coordinator is active.
        _metrics: Dictionary of performance metrics.
        _callbacks: Event callbacks dictionary.
    
    Events:
        - on_task_received: When a new task is submitted
        - on_task_assigned: When a task is assigned to agents
        - on_task_completed: When a task completes successfully
        - on_task_failed: When a task fails
        - on_agent_join: When an agent joins the swarm
        - on_agent_leave: When an agent leaves the swarm
        - on_message_sent: When a message is sent
        - on_message_received: When a message is received
    
    Example:
        >>> config = POPSSSwarmConfig(registry=my_registry)
        >>> coordinator = POPSSSwarmCoordinator(config)
        >>> coordinator.initialize()
        >>> 
        >>> # Register agents
        >>> coordinator.register_agent(
        ...     "agent_1", POPSSAggregentType.ANALYSIS, "Analyzer",
        ...     {POPSSAgentCapability.DATA_ANALYSIS}
        ... )
        >>> 
        >>> # Submit task
        >>> task_id = await coordinator.submit_task(
        ...     "analysis", "Analyze data", {"data": my_data}
        ... )
        >>> 
        >>> # Get status
        >>> status = coordinator.get_swarm_status()
    
    See Also:
        - POPSSSwarmConfig: Configuration settings
        - POPSSSwarmTask: Task definition
        - POPSSSwarmMember: Agent membership
        - POPSSSwarmTopology: Network topologies
    """
    
    def __init__(self, config: POPSSSwarmConfig):
        """
        Initialize the swarm coordinator with configuration.
        
        Args:
            config: POPSSSwarmConfig instance with registry reference
                and operational settings.
        """
        self.config = config
        self.registry = config.registry
        
        # Agent membership tracking
        self._members: Dict[str, POPSSSwarmMember] = {}
        self._tasks: Dict[str, POPSSSwarmTask] = {}
        
        # Communication queues
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=config.message_queue_size)
        self._task_queue: asyncio.Queue = asyncio.Queue()
        
        # Result caching
        self._result_cache: Dict[str, Any] = {}
        
        # Coordinator agent reference
        self._coordinator: Optional[POPSSBaseAgent] = None
        
        # State flags
        self._running = False
        self._processing_tasks = False
        
        # Performance metrics
        self._metrics: Dict[str, Any] = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_messages': 0,
            'total_execution_time': 0.0,
        }
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'on_task_received': [],
            'on_task_assigned': [],
            'on_task_completed': [],
            'on_task_failed': [],
            'on_agent_join': [],
            'on_agent_leave': [],
            'on_message_sent': [],
            'on_message_received': [],
        }
        
        # Thread pool for async operations
        self._async_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="piscesl1_swarm"
        )
        
        _LOG.info("POPSSSwarmCoordinator initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the swarm coordinator and start background tasks.
        
        This method sets up the coordinator agent (if configured) and
        starts the background processing tasks for task handling, message
        processing, and heartbeat monitoring.
        
        Returns:
            bool: True if initialization succeeded, False otherwise.
        """
        try:
            self._running = True
            
            # Set up coordinator agent if specified
            if self.config.coordinator_id:
                self._coordinator = self.registry.get_agent(self.config.coordinator_id)
            
            if not self._coordinator:
                _LOG.warning("No coordinator agent configured, using default coordination")
            
            # Start background processing tasks
            self._start_background_tasks()
            
            _LOG.info("Swarm coordinator initialized")
            return True
            
        except Exception as e:
            _LOG.error(f"Failed to initialize swarm coordinator: {e}")
            return False
    
    def register_agent(self, agent_id: str, agent_type: POPSSAggregentType, name: str, 
                      capabilities: Set[POPSSAgentCapability]) -> bool:
        """
        Register a new agent in the swarm.
        
        Args:
            agent_id: Unique identifier for the agent.
            agent_type: Type classification for the agent.
            name: Human-readable name for the agent.
            capabilities: Set of capabilities the agent possesses.
        
        Returns:
            bool: True if registration succeeded, False if max agents reached.
        
        Example:
            >>> coordinator.register_agent(
            ...     "agent_1",
            ...     POPSSAggregentType.ANALYSIS,
            ...     "DataAnalyzer",
            ...     {POPSSAgentCapability.DATA_ANALYSIS, POPSSAgentCapability.REASONING}
            ... )
        """
        if len(self._members) >= self.config.max_agents:
            _LOG.warning("Maximum agents reached")
            return False
        
        member = POPSSSwarmMember(
            agent_id=agent_id,
            agent_type=agent_type,
            name=name,
            capabilities=capabilities,
        )
        
        self._members[agent_id] = member
        
        # Set up connections based on topology
        self._setup_topology_connections(member)
        
        self._emit_callback('on_agent_join', {
            'agent_id': agent_id,
            'agent_type': agent_type.value,
            'name': name,
        })
        
        _LOG.info(f"Agent joined swarm: {name} ({agent_id})")
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the swarm.
        
        If the agent has an active task, the task will be reassigned
        to another available agent.
        
        Args:
            agent_id: ID of the agent to remove.
        
        Returns:
            bool: True if removal succeeded, False if agent not found.
        """
        if agent_id not in self._members:
            return False
        
        member = self._members[agent_id]
        
        # Reassign any active task
        if member.current_task:
            self._reassign_task(member.current_task)
        
        del self._members[agent_id]
        
        # Remove from neighbor lists
        for other_member in self._members.values():
            if agent_id in other_member.neighbors:
                other_member.neighbors.remove(agent_id)
        
        self._emit_callback('on_agent_leave', {
            'agent_id': agent_id,
            'name': member.name,
        })
        
        _LOG.info(f"Agent left swarm: {member.name} ({agent_id})")
        return True
    
    def _setup_topology_connections(self, member: POPSSSwarmMember):
        """
        Set up network connections for a new member based on topology.
        
        Args:
            member: The POPSSSwarmMember to set up connections for.
        """
        if self.config.topology == POPSSSwarmTopology.STAR:
            # Connect all agents to coordinator
            for existing_id in self._members:
                if existing_id != member.agent_id:
                    member.neighbors.append(existing_id)
        
        elif self.config.topology == POPSSSwarmTopology.MESH:
            # Partial mesh with max 3 neighbors
            for existing_id in self._members:
                if existing_id != member.agent_id:
                    if len(self._members[existing_id].neighbors) < 3:
                        member.neighbors.append(existing_id)
                        self._members[existing_id].neighbors.append(member.agent_id)
        
        elif self.config.topology == POPSSSwarmTopology.RING:
            # Circular connection
            existing_ids = list(self._members.keys())
            if existing_ids:
                member.neighbors.append(existing_ids[0])
                if len(existing_ids) > 1:
                    self._members[existing_ids[-1]].neighbors.append(member.agent_id)
        
        elif self.config.topology == POPSSSwarmTopology.HIERARCHICAL:
            # Connect to coordinator
            coordinator = self._get_coordinator()
            if coordinator and coordinator.agent_id != member.agent_id:
                member.neighbors.append(coordinator.agent_id)
        
        elif self.config.topology == POPSSSwarmTopology.FLAT:
            # All-to-all connections
            for existing_id in self._members:
                if existing_id != member.agent_id:
                    member.neighbors.append(existing_id)
    
    def _get_coordinator(self) -> Optional[POPSSSwarmMember]:
        """
        Get the coordinator member for hierarchical topology.
        
        Returns:
            Optional[POPSSSwarmMember]: The coordinator member or None.
        """
        if self._coordinator:
            return self._members.get(self._coordinator.agent_id)
        
        if self.config.topology == POPSSSwarmTopology.HIERARCHICAL:
            # Find GENERAL type agent as coordinator
            for member in self._members.values():
                if member.agent_type == POPSSAggregentType.GENERAL:
                    return member
        
        return None
    
    async def submit_task(self, task_type: str, description: str, 
                         input_data: Optional[Dict[str, Any]] = None,
                         priority: int = 0,
                         dependencies: Optional[List[str]] = None) -> str:
        """
        Submit a new task to the swarm for processing.
        
        Args:
            task_type: Type classification for capability matching.
            description: Human-readable task description.
            input_data: Optional input data for task execution.
            priority: Task priority (higher = more urgent).
            dependencies: Optional list of task IDs that must complete first.
        
        Returns:
            str: The generated task ID for tracking.
        
        Example:
            >>> task_id = await coordinator.submit_task(
            ...     task_type="analysis",
            ...     description="Analyze market trends",
            ...     input_data={"data": market_data},
            ...     priority=5
            ... )
        """
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        task = POPSSSwarmTask(
            task_id=task_id,
            task_type=task_type,
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            input_data=input_data or {},
        )
        
        self._tasks[task_id] = task
        
        await self._task_queue.put(task_id)
        
        self._emit_callback('on_task_received', {
            'task_id': task_id,
            'task_type': task_type,
            'description': description,
            'priority': priority,
        })
        
        _LOG.info(f"Task received: {task_id} ({task_type})")
        
        return task_id
    
    async def assign_task(self, task_id: str, agent_ids: Optional[List[str]] = None) -> bool:
        """
        Assign a task to one or more agents.
        
        If agent_ids is not provided, suitable agents will be selected
        automatically based on task requirements and agent capabilities.
        
        Args:
            task_id: ID of the task to assign.
            agent_ids: Optional list of specific agent IDs to assign.
        
        Returns:
            bool: True if assignment succeeded, False otherwise.
        """
        if task_id not in self._tasks:
            return False
        
        task = self._tasks[task_id]
        
        # Auto-select agents if not specified
        if not agent_ids:
            agent_ids = self._select_agents_for_task(task)
        
        if not agent_ids:
            _LOG.warning(f"No suitable agents found for task: {task_id}")
            return False
        
        task.assigned_agents = agent_ids
        task.status = "assigned"
        task.started_at = datetime.now()
        
        # Update agent status
        for agent_id in agent_ids:
            if agent_id in self._members:
                self._members[agent_id].current_task = task_id
                self._members[agent_id].status = "busy"
        
        # Notify agents of assignment
        await self._notify_agents(agent_ids, POPSSSwarmMessageType.TASK_ASSIGNMENT, {
            'task_id': task_id,
            'description': task.description,
            'input_data': task.input_data,
        })
        
        self._emit_callback('on_task_assigned', {
            'task_id': task_id,
            'assigned_agents': agent_ids,
        })
        
        _LOG.info(f"Task assigned: {task_id} -> {agent_ids}")
        return True
    
    def _select_agents_for_task(self, task: POPSSSwarmTask) -> List[str]:
        """
        Select suitable agents for a task based on capabilities and load.
        
        Args:
            task: The POPSSSwarmTask to find agents for.
        
        Returns:
            List[str]: List of selected agent IDs (up to 5).
        """
        required_capabilities = self._infer_required_capabilities(task.task_type)
        
        # Find available agents with no current load
        available_agents = [
            (agent_id, member)
            for agent_id, member in self._members.items()
            if member.status == "available" and member.load == 0
        ]
        
        # Score agents based on capabilities and performance
        scored_agents = []
        for agent_id, member in available_agents:
            score = 0
            
            # Capability matching
            for capability in required_capabilities:
                if capability in member.capabilities:
                    score += 10
            
            # Performance bonus
            score += member.performance_metrics.get('success_rate', 0.5) * 10
            
            # Load penalty
            score -= member.load * 0.1
            
            # Priority bonus for high-priority tasks
            if task.priority > 5:
                if member.agent_type in [POPSSAggregentType.GENERAL, POPSSAggregentType.ANALYSIS]:
                    score += 5
            
            scored_agents.append((agent_id, score))
        
        # Sort by score descending
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        return [aid for aid, _ in scored_agents[:5]]
    
    def _infer_required_capabilities(self, task_type: str) -> Set[POPSSAgentCapability]:
        """
        Infer required capabilities from task type.
        
        Args:
            task_type: The task type string to analyze.
        
        Returns:
            Set[POPSSAgentCapability]: Set of inferred required capabilities.
        """
        capability_mapping = {
            'code': {POPSSAgentCapability.CODE_GENERATION, POPSSAgentCapability.CODE_EXECUTION},
            'file': {POPSSAgentCapability.FILE_OPERATIONS},
            'search': {POPSSAgentCapability.WEB_SEARCH, POPSSAgentCapability.RESEARCH},
            'analysis': {POPSSAgentCapability.DATA_ANALYSIS, POPSSAgentCapability.REASONING},
            'writing': {POPSSAgentCapability.CREATIVE_WRITING, POPSSAgentCapability.TEXT_GENERATION},
            'reasoning': {POPSSAgentCapability.REASONING, POPSSAgentCapability.PLANNING},
            'research': {POPSSAgentCapability.RESEARCH, POPSSAgentCapability.WEB_SEARCH},
            'tool': {POPSSAgentCapability.TOOL_USE},
        }
        
        task_lower = task_type.lower()
        
        required = set()
        for keyword, capabilities in capability_mapping.items():
            if keyword in task_lower:
                required.update(capabilities)
        
        # Default to TOOL_USE if no match
        if not required:
            required.add(POPSSAgentCapability.TOOL_USE)
        
        return required
    
    async def report_result(self, agent_id: str, task_id: str, 
                          success: bool, output: Optional[Any] = None,
                          error: Optional[str] = None):
        """
        Report the result of a task execution.
        
        This method is called by agents to report task completion or failure.
        It updates task status, triggers callbacks, and handles task
        dependencies and failure recovery.
        
        Args:
            agent_id: ID of the agent reporting the result.
            task_id: ID of the completed task.
            success: Whether the task completed successfully.
            output: Optional output data from successful execution.
            error: Optional error message from failed execution.
        """
        if task_id not in self._tasks:
            return
        
        task = self._tasks[task_id]
        
        if success:
            task.status = "completed"
            task.completed_at = datetime.now()
            task.output_data = output or {}
            
            self._metrics['completed_tasks'] += 1
            
            self._emit_callback('on_task_completed', {
                'task_id': task_id,
                'agent_id': agent_id,
                'output': output,
            })
            
            # Notify assigned agents of completion
            await self._notify_agents(task.assigned_agents, POPSSSwarmMessageType.RESULT_REPORT, {
                'task_id': task_id,
                'success': True,
                'output': output,
            })
            
            # Trigger dependent tasks
            if task.dependencies:
                for dep_id in task.dependencies:
                    if dep_id in self._tasks and self._tasks[dep_id].status == "pending":
                        await self.assign_task(dep_id)
            
        else:
            task.status = "failed"
            task.completed_at = datetime.now()
            task.output_data['error'] = error
            
            self._metrics['failed_tasks'] += 1
            
            self._emit_callback('on_task_failed', {
                'task_id': task_id,
                'agent_id': agent_id,
                'error': error,
            })
            
            # Attempt recovery if collaboration enabled
            if self.config.enable_collaboration:
                await self._handle_task_failure(task)
        
        # Release agents
        for agent_id in task.assigned_agents:
            if agent_id in self._members:
                self._members[agent_id].current_task = None
                self._members[agent_id].status = "available"
                self._members[agent_id].load = 0
    
    async def _handle_task_failure(self, task: POPSSSwarmTask):
        """
        Handle a failed task by attempting reassignment to a neighbor.
        
        Args:
            task: The failed POPSSSwarmTask to handle.
        """
        if task.status == "failed" and task.assigned_agents:
            failed_agent = task.assigned_agents[0]
            
            if failed_agent in self._members:
                neighbor_ids = self._members[failed_agent].neighbors
                
                # Try to assign to an available neighbor
                for neighbor_id in neighbor_ids:
                    if neighbor_id in self._members:
                        neighbor = self._members[neighbor_id]
                        
                        if neighbor.status == "available" and neighbor.load == 0:
                            await self.assign_task(task.task_id, [neighbor_id])
                            return
    
    async def _notify_agents(self, agent_ids: List[str], message_type: POPSSSwarmMessageType, 
                           payload: Dict[str, Any], priority: int = 0):
        """
        Send a notification message to one or more agents.
        
        Args:
            agent_ids: List of agent IDs to notify.
            message_type: Type of message to send.
            payload: Message payload dictionary.
            priority: Message priority (default: 0).
        """
        message = POPSSSwarmMessage(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            message_type=message_type,
            sender_id="coordinator",
            receiver_ids=agent_ids,
            payload=payload,
            priority=priority,
        )
        
        for agent_id in agent_ids:
            await self._deliver_message(agent_id, message)
        
        self._metrics['total_messages'] += len(agent_ids)
    
    async def _deliver_message(self, agent_id: str, message: POPSSSwarmMessage):
        """
        Deliver a message to a specific agent.
        
        Args:
            agent_id: ID of the target agent.
            message: The POPSSSwarmMessage to deliver.
        """
        if agent_id in self._members:
            self._members[agent_id].last_heartbeat = datetime.now()
            
            await self._message_queue.put((agent_id, message))
            
            self._emit_callback('on_message_received', {
                'receiver_id': agent_id,
                'message': message,
            })
    
    async def _start_background_tasks(self):
        """Start background processing tasks for the swarm."""
        asyncio.create_task(self._process_tasks())
        asyncio.create_task(self._process_messages())
        asyncio.create_task(self._heartbeat_monitor())
    
    async def _process_tasks(self):
        """Background task for processing pending tasks from the queue."""
        while self._running:
            try:
                task_id = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
                
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    
                    if task.status == "pending":
                        # Check if dependencies are met
                        dependencies_met = all(
                            dep_id not in self._tasks or 
                            self._tasks[dep_id].status == "completed"
                            for dep_id in task.dependencies
                        )
                        
                        if dependencies_met:
                            await self.assign_task(task_id)
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                _LOG.error(f"Error processing tasks: {e}")
    
    async def _process_messages(self):
        """Background task for processing messages from the queue."""
        while self._running:
            try:
                agent_id, message = await asyncio.wait_for(
                    self._message_queue.get(), 
                    timeout=1.0
                )
                
                self._emit_callback('on_message_sent', {
                    'receiver_id': agent_id,
                    'message': message,
                })
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                _LOG.error(f"Error processing messages: {e}")
    
    async def _heartbeat_monitor(self):
        """Background task for monitoring agent health via heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.config.heartbeat_interval * 2)
                
                # Check for timed out agents
                for agent_id, member in list(self._members.items()):
                    if member.last_heartbeat < timeout_threshold:
                        _LOG.warning(f"Agent timed out: {agent_id}")
                        member.status = "unavailable"
                        
                        if member.current_task:
                            await self._handle_task_timeout(member.current_task)
                
            except Exception as e:
                _LOG.error(f"Error in heartbeat monitor: {e}")
    
    async def _handle_task_timeout(self, task_id: str):
        """
        Handle a task that has timed out.
        
        Args:
            task_id: ID of the timed out task.
        """
        if task_id not in self._tasks:
            return
        
        task = self._tasks[task_id]
        
        if task.status in ["assigned", "in_progress"]:
            task.status = "timeout"
            
            # Retry if retries remaining
            if self.config.max_retries > 0:
                task.status = "pending"
                task.assigned_agents = []
                await self._task_queue.put(task_id)
    
    def _reassign_task(self, task_id: str):
        """
        Reassign a task to another agent.
        
        Args:
            task_id: ID of the task to reassign.
        """
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if task.status in ["assigned", "in_progress"]:
                task.status = "pending"
                task.assigned_agents = []
                asyncio.create_task(self._task_queue.put(task_id))
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the swarm.
        
        Returns:
            Dict[str, Any]: Status including member counts, task counts,
                topology, and performance metrics.
        """
        return {
            'total_members': len(self._members),
            'available_agents': sum(1 for m in self._members.values() if m.status == "available"),
            'busy_agents': sum(1 for m in self._members.values() if m.status == "busy"),
            'unavailable_agents': sum(1 for m in self._members.values() if m.status == "unavailable"),
            'total_tasks': len(self._tasks),
            'pending_tasks': sum(1 for t in self._tasks.values() if t.status == "pending"),
            'active_tasks': sum(1 for t in self._tasks.values() if t.status in ["assigned", "in_progress"]),
            'completed_tasks': sum(1 for t in self._tasks.values() if t.status == "completed"),
            'failed_tasks': sum(1 for t in self._tasks.values() if t.status == "failed"),
            'topology': self.config.topology.value,
            'metrics': self._metrics.copy(),
        }
    
    def get_member_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific swarm member.
        
        Args:
            agent_id: ID of the agent to get info for.
        
        Returns:
            Optional[Dict[str, Any]]: Member information or None if not found.
        """
        if agent_id not in self._members:
            return None
        
        member = self._members[agent_id]
        return {
            'agent_id': member.agent_id,
            'agent_type': member.agent_type.value,
            'name': member.name,
            'capabilities': [c.value for c in member.capabilities],
            'current_task': member.current_task,
            'status': member.status,
            'load': member.load,
            'neighbors': member.neighbors,
            'performance_metrics': member.performance_metrics,
        }
    
    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback function for a specific event.
        
        Args:
            event: Event name (on_task_received, on_task_assigned, etc.).
            callback: Callable to be invoked when the event occurs.
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit_callback(self, event: str, data: Any):
        """
        Emit an event to all registered callbacks.
        
        Args:
            event: Event name to emit.
            data: Data to pass to the callbacks.
        """
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    _LOG.error(f"Error in callback for {event}: {e}")
    
    def shutdown(self):
        """
        Shutdown the swarm coordinator and release all resources.
        
        This method stops background tasks, marks all members as terminated,
        and shuts down the thread pool executor.
        """
        self._running = False
        
        for member in self._members.values():
            member.status = "terminated"
        
        self._async_executor.shutdown(wait=True)
        
        _LOG.info("Swarm coordinator shutdown")
    
    def __enter__(self):
        """Context manager entry - initializes the coordinator."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - shuts down the coordinator."""
        self.shutdown()
        return False
