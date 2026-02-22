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

"""
Agent Registry - Operator-based Agent Registration

This module provides agent registry that integrates with the OPSC operator
infrastructure for unified agent management and discovery.

Key Components:
    - POPSSAggregentType: Agent type enumeration
    - POPSSAggregentMetadata: Agent metadata container
    - POPSSAggregentRegistration: Agent registration record
    - POPSSAggregentRegistry: Agent registry (extends PiscesLxOperatorRegistry)

Design Principles:
    1. Operator Registry Integration: Use OPSC registry as base
    2. Agent-specific Metadata: Track capabilities, usage stats
    3. Type-based Discovery: Find agents by type and capability
    4. Factory Pattern: Support custom agent instantiation

Usage:
    registry = POPSSAggregentRegistry()
    registry.register(MyAgentClass, config)

    agent = registry.get_agent("agent_id")
    agents = registry.list_agents(capability=POPSSAgentCapability.CODE_GENERATION)
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from utils.opsc.registry import PiscesLxOperatorRegistry, PiscesLxOperatorRegistryHub
from utils.opsc.interface import PiscesLxOperatorInterface
from configs.version import VERSION

from .base import (
    POPSSBaseAgent,
    POPSSAgentConfig,
    POPSSAgentCapability,
)

T = TypeVar('T', bound=POPSSBaseAgent)


class POPSSAggregentType(Enum):
    """
    Agent type enumeration for categorization.

    Types:
        GENERAL: General-purpose agent
        CODE: Code generation and execution
        RESEARCH: Research and information gathering
        CREATIVE: Creative content generation
        ANALYSIS: Data analysis and insights
        TOOL: Tool-specific agent
    """
    GENERAL = "general"
    CODE = "code"
    RESEARCH = "research"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    TOOL = "tool"


@dataclass
class POPSSAggregentMetadata:
    """
    Agent metadata for registration tracking.

    Attributes:
        agent_id: Unique agent identifier
        agent_type: Type of agent
        name: Agent name
        description: Agent description
        version: Agent version
        author: Agent author
        created_at: Creation timestamp
        updated_at: Last update timestamp
        capabilities: Set of agent capabilities
        tags: Set of tags for search
        usage_count: Number of times used
        success_rate: Success rate (0-1)
        average_execution_time: Average execution time in seconds
    """
    agent_id: str
    agent_type: POPSSAggregentType
    name: str
    description: str
    version: str = VERSION
    author: str = "PiscesL1"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    capabilities: Set[POPSSAgentCapability] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    usage_count: int = 0
    success_rate: float = 1.0
    average_execution_time: float = 0.0


@dataclass
class POPSSAggregentRegistration:
    """
    Agent registration record.

    Attributes:
        agent_class: Agent class type
        config: Agent configuration
        metadata: Agent metadata
        factory: Optional factory function for creating instances
        instance: Optional singleton instance
        enabled: Whether agent is enabled
        singleton: Whether to use singleton pattern
    """
    agent_class: Type[POPSSBaseAgent]
    config: POPSSAgentConfig
    metadata: POPSSAggregentMetadata
    factory: Optional[Callable[..., POPSSBaseAgent]] = None
    instance: Optional[POPSSBaseAgent] = None
    enabled: bool = True
    singleton: bool = True


class POPSSAggregentRegistry:
    """
    Agent registry integrating with OPSC operator infrastructure.

    This class provides agent-specific registration and discovery on top
    of the OPSC operator registry, enabling unified management of agents
    as operators.

    Key Features:
        - OPSC registry integration for execution
        - Agent-specific metadata tracking
        - Capability-based agent discovery
        - Type-based categorization
        - Singleton and factory patterns

    Attributes:
        _agents: Dictionary of agent registrations
        _agent_instances: Dictionary of agent instances
        _agent_types: Type to agent ID mapping
        _capability_agents: Capability to agent ID mapping

    Usage:
        registry = POPSSAggregentRegistry()
        registry.register(MyAgent, config)

        agent = registry.get_agent("my_agent")
        agents = registry.list_agents(capability=POPSSAgentCapability.CODE_GENERATION)
    """

    def __init__(self, opsc_registry: Optional[PiscesLxOperatorRegistry] = None):
        """
        Initialize agent registry.

        Args:
            opsc_registry: Optional OPSC registry to use
        """
        from utils.dc import PiscesLxLogger, PiscesLxMetrics
        from utils.paths import get_log_file
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Agents.Registry", file_path=get_log_file("PiscesLx.Opss.Agents.Registry"), enable_file=True)
        self._metrics = PiscesLxMetrics()

        self._opsc_registry = opsc_registry or PiscesLxOperatorRegistryHub.get_registry()

        self._agents: Dict[str, POPSSAggregentRegistration] = {}
        self._agent_instances: Dict[str, POPSSBaseAgent] = {}

        self._agent_types: Dict[POPSSAggregentType, List[str]] = {}
        self._capability_agents: Dict[POPSSAgentCapability, List[str]] = {}

        self._lock = threading.RLock()

        self._callbacks: Dict[str, List[Callable]] = {
            'on_register': [],
            'on_unregister': [],
            'on_enable': [],
            'on_disable': [],
        }

        self._LOG.info("agent_registry_initialized")

    def register(
        self,
        agent_class: Type[POPSSBaseAgent],
        config: POPSSAgentConfig,
        factory: Optional[Callable[..., POPSSBaseAgent]] = None,
        enabled: bool = True,
        singleton: bool = True,
    ) -> bool:
        """
        Register an agent class with the registry.

        Args:
            agent_class: Agent class to register
            config: Agent configuration
            factory: Optional factory function
            enabled: Whether agent is enabled
            singleton: Whether to use singleton pattern

        Returns:
            True if registration succeeded
        """
        with self._lock:
            try:
                agent_id = config.agent_id

                if agent_id in self._agents:
                    self._LOG.warning("agent_already_registered", agent_id=agent_id)
                    return False

                metadata = POPSSAggregentMetadata(
                    agent_id=agent_id,
                    agent_type=self._infer_agent_type(agent_class, config),
                    name=config.name,
                    description=config.description,
                    capabilities=config.capabilities,
                )

                registration = POPSSAggregentRegistration(
                    agent_class=agent_class,
                    config=config,
                    metadata=metadata,
                    factory=factory or self._default_factory,
                    enabled=enabled,
                    singleton=singleton,
                )

                self._agents[agent_id] = registration

                self._opsc_registry.register(agent_class)

                agent_type = registration.metadata.agent_type
                if agent_type not in self._agent_types:
                    self._agent_types[agent_type] = []
                self._agent_types[agent_type].append(agent_id)

                for capability in config.capabilities:
                    if capability not in self._capability_agents:
                        self._capability_agents[capability] = []
                    self._capability_agents[capability].append(agent_id)

                self._metrics.counter("agent_registrations")
                self._metrics.gauge("agent_registry_total", len(self._agents))

                self._emit_callback('on_register', {
                    'agent_id': agent_id,
                    'agent_type': agent_type.value,
                    'name': config.name,
                })

                self._LOG.info("agent_registered", agent_id=agent_id, name=config.name)
                return True

            except Exception as e:
                self._LOG.error("agent_registration_failed", error=str(e))
                return False

    def _default_factory(self, registration: POPSSAggregentRegistration) -> POPSSBaseAgent:
        """Default factory for creating agent instances."""
        config = registration.config
        agent_class = registration.agent_class

        if registration.factory:
            return registration.factory(config)

        return agent_class(config)

    def _infer_agent_type(
        self,
        agent_class: Type[POPSSBaseAgent],
        config: POPSSAgentConfig
    ) -> POPSSAggregentType:
        """Infer agent type from class and config."""
        name_lower = config.name.lower()

        if 'code' in name_lower or 'programming' in name_lower:
            return POPSSAggregentType.CODE
        elif 'research' in name_lower or 'search' in name_lower:
            return POPSSAggregentType.RESEARCH
        elif 'creative' in name_lower or 'write' in name_lower or 'art' in name_lower:
            return POPSSAggregentType.CREATIVE
        elif 'analysis' in name_lower or 'analyze' in name_lower:
            return POPSSAggregentType.ANALYSIS
        elif 'tool' in name_lower:
            return POPSSAggregentType.TOOL
        else:
            return POPSSAggregentType.GENERAL

    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent from the registry.

        Args:
            agent_id: Agent ID to unregister

        Returns:
            True if unregistration succeeded
        """
        with self._lock:
            if agent_id not in self._agents:
                self._LOG.warning("agent_not_found_for_unregister", agent_id=agent_id)
                return False

            registration = self._agents[agent_id]

            if agent_id in self._agent_instances:
                instance = self._agent_instances[agent_id]
                instance.teardown()
                del self._agent_instances[agent_id]

            agent_type = registration.metadata.agent_type
            if agent_type in self._agent_types and agent_id in self._agent_types[agent_type]:
                self._agent_types[agent_type].remove(agent_id)

            for capability in registration.config.capabilities:
                if capability in self._capability_agents and agent_id in self._capability_agents[capability]:
                    self._capability_agents[capability].remove(agent_id)

            del self._agents[agent_id]

            self._metrics.counter("agent_unregistrations")
            self._metrics.gauge("agent_registry_total", len(self._agents))

            self._emit_callback('on_unregister', {
                'agent_id': agent_id,
                'name': registration.config.name,
            })

            self._LOG.info("agent_unregistered", agent_id=agent_id)
            return True

    def get_agent(self, agent_id: str, create: bool = True) -> Optional[POPSSBaseAgent]:
        """
        Get an agent instance by ID.

        Args:
            agent_id: Agent ID
            create: Whether to create instance if not exists

        Returns:
            Agent instance or None
        """
        with self._lock:
            if agent_id not in self._agents:
                return None

            registration = self._agents[agent_id]

            if not registration.enabled:
                self._LOG.warning("agent_disabled", agent_id=agent_id)
                return None

            if registration.singleton:
                if agent_id not in self._agent_instances:
                    if create:
                        instance = registration.factory(registration)
                        self._agent_instances[agent_id] = instance
                    else:
                        return None
                return self._agent_instances[agent_id]
            else:
                return registration.factory(registration)

    def list_agents(
        self,
        agent_type: Optional[POPSSAggregentType] = None,
        capability: Optional[POPSSAgentCapability] = None,
        enabled_only: bool = True,
    ) -> List[str]:
        """
        List agents matching criteria.

        Args:
            agent_type: Filter by agent type
            capability: Filter by capability
            enabled_only: Only return enabled agents

        Returns:
            List of matching agent IDs
        """
        with self._lock:
            agent_ids = set()

            if agent_type:
                if agent_type in self._agent_types:
                    agent_ids.update(self._agent_types[agent_type])
                else:
                    return []
            elif capability:
                if capability in self._capability_agents:
                    agent_ids.update(self._capability_agents[capability])
                else:
                    return []
            else:
                agent_ids.update(self._agents.keys())

            if enabled_only:
                agent_ids = {
                    aid for aid in agent_ids
                    if aid in self._agents and self._agents[aid].enabled
                }

            return sorted(agent_ids)

    def list_agent_types(self) -> List[POPSSAggregentType]:
        """List all agent types in registry."""
        with self._lock:
            return list(self._agent_types.keys())

    def list_capabilities(self) -> List[POPSSAgentCapability]:
        """List all capabilities in registry."""
        with self._lock:
            return list(self._capability_agents.keys())

    def get_metadata(self, agent_id: str) -> Optional[POPSSAggregentMetadata]:
        """Get metadata for an agent."""
        with self._lock:
            if agent_id not in self._agents:
                return None
            return self._agents[agent_id].metadata

    def get_all_metadata(self) -> List[POPSSAggregentMetadata]:
        """Get metadata for all agents."""
        with self._lock:
            return [reg.metadata for reg in self._agents.values()]

    def enable_agent(self, agent_id: str) -> bool:
        """Enable an agent."""
        with self._lock:
            if agent_id not in self._agents:
                return False

            self._agents[agent_id].enabled = True
            self._agents[agent_id].metadata.updated_at = datetime.now()

            self._metrics.counter("agent_enables")
            self._emit_callback('on_enable', {'agent_id': agent_id})

            self._LOG.info("agent_enabled", agent_id=agent_id)
            return True

    def disable_agent(self, agent_id: str) -> bool:
        """Disable an agent."""
        with self._lock:
            if agent_id not in self._agents:
                return False

            self._agents[agent_id].enabled = False

            if agent_id in self._agent_instances:
                instance = self._agent_instances[agent_id]
                instance.teardown()
                del self._agent_instances[agent_id]

            self._metrics.counter("agent_disables")
            self._emit_callback('on_disable', {'agent_id': agent_id})

            self._LOG.info("agent_disabled", agent_id=agent_id)
            return True

    def update_stats(self, agent_id: str, success: bool, execution_time: float) -> None:
        """Update agent statistics."""
        with self._lock:
            if agent_id not in self._agents:
                return

            metadata = self._agents[agent_id].metadata
            metadata.usage_count += 1

            if success:
                if metadata.average_execution_time == 0:
                    metadata.average_execution_time = execution_time
                else:
                    metadata.average_execution_time = (
                        metadata.average_execution_time * 0.9 + execution_time * 0.1
                    )

            if metadata.usage_count == 1:
                metadata.success_rate = 1.0 if success else 0.0
            else:
                success_count = metadata.success_rate * (metadata.usage_count - 1)
                success_count += 1 if success else 0
                metadata.success_rate = success_count / metadata.usage_count

    def search_agents(self, query: str) -> List[str]:
        """Search agents by query."""
        with self._lock:
            query_lower = query.lower()
            matching = []

            for agent_id, registration in self._agents.items():
                if query_lower in registration.config.name.lower():
                    matching.append(agent_id)
                    continue
                if query_lower in registration.config.description.lower():
                    matching.append(agent_id)
                    continue
                for tag in registration.metadata.tags:
                    if query_lower in tag.lower():
                        matching.append(agent_id)
                        break

            return sorted(set(matching))

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get registry summary."""
        with self._lock:
            total_agents = len(self._agents)
            enabled_agents = sum(1 for reg in self._agents.values() if reg.enabled)

            total_usage = sum(reg.metadata.usage_count for reg in self._agents.values())
            avg_success_rate = sum(reg.metadata.success_rate for reg in self._agents.values()) / max(total_agents, 1)

            type_distribution = {
                agent_type.value: len(agents)
                for agent_type, agents in self._agent_types.items()
            }

            return {
                'total_agents': total_agents,
                'enabled_agents': enabled_agents,
                'disabled_agents': total_agents - enabled_agents,
                'agent_types': list(self._agent_types.keys()),
                'type_distribution': type_distribution,
                'capabilities': list(self._capability_agents.keys()),
                'total_usage': total_usage,
                'average_success_rate': avg_success_rate,
            }

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for an event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit_callback(self, event: str, data: Any) -> None:
        """Emit a callback event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    self._LOG.error("callback_error", event=event, error=str(e))

    def shutdown(self) -> None:
        """Shutdown the registry."""
        with self._lock:
            for agent_id, instance in self._agent_instances.items():
                instance.teardown()
            self._agent_instances.clear()

        self._LOG.info("agent_registry_shutdown")

    def __enter__(self) -> 'POPSSAggregentRegistry':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.shutdown()
        return False
