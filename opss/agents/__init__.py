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
PiscesL1 Agent Module - Operator-based Multi-Agent System

This module provides a comprehensive multi-agent system that integrates with
the OPSC operator infrastructure for unified execution, monitoring, and
resource management.

Key Components:
    - Base Agent Classes: POPSSBaseAgent, POPSSAgentConfig, POPSSAgentResult
    - Agent Registry: POPSSAggregentRegistry for agent management
    - Orchestrator: POPSSDynamicOrchestrator for multi-agent coordination
    - Swarm Coordinator: POPSSSwarmCoordinator for distributed agent swarms
    - Protocol Handler: POPSSProtocolHandler for agent communication
    - MCP Bridge: POPSSMCPBridge for tool integration

Design Principles:
    1. Operator Integration: All agents are operators with reasoning capabilities
    2. Unified Infrastructure: Use OPSC for execution, metrics, tracing
    3. Async Support: Native async execution for concurrent operations
    4. Event-Driven: Callback system for monitoring and extensibility
"""

from .base import (
    POPSSAgentState,
    POPSSAgentCapability,
    POPSSAgentConfig,
    POPSSAgentContext,
    POPSSAgentResult,
    POPSSAgentThought,
    POPSSBaseAgent,
)

from .registry import (
    POPSSAggregentType,
    POPSSAggregentMetadata,
    POPSSAggregentRegistration,
    POPSSAggregentRegistry,
)

from .orchestrator import (
    POPSSOrchestrationStrategy,
    POPSSOrchestrationPlan,
    POPSSOrchestrationStage,
    POPSSOrchestrationResult,
    POPSSOrchestratorConfig,
    POPSSBaseOrchestrator,
    POPSSDynamicOrchestrator,
)

from .swarm_coordinator import (
    POPSSSwarmTopology,
    POPSSSwarmMessageType,
    POPSSSwarmMessage,
    POPSSSwarmTask,
    POPSSSwarmMember,
    POPSSSwarmConfig,
    POPSSSwarmCoordinator,
)

from .protocol import (
    POPSSProtocolMessageType,
    POPSSProtocolMessage,
    POPSSProtocolConfig,
    POPSSProtocolHandler,
)

from .mcp_bridge import (
    POPSSMCPBridgeConfig,
    POPSSToolInfo,
    POPSSToolCall,
    POPSSMCPBridge,
    POPSSMCPBridgeMixin,
)

from .loader import (
    POPSSPromptConfig,
    POPSSPromptLoader,
)

from .factory import (
    POPSSExpertFactory,
)

from .base import (
    POPSSAgentMode,
    POPSSPromptBasedAgent,
)

__all__ = [
    "POPSSAgentState",
    "POPSSAgentCapability",
    "POPSSAgentConfig",
    "POPSSAgentContext",
    "POPSSAgentResult",
    "POPSSAgentThought",
    "POPSSBaseAgent",
    "POPSSAgentMode",
    "POPSSPromptBasedAgent",
    "POPSSAggregentType",
    "POPSSAggregentMetadata",
    "POPSSAggregentRegistration",
    "POPSSAggregentRegistry",
    "POPSSOrchestrationStrategy",
    "POPSSOrchestrationPlan",
    "POPSSOrchestrationStage",
    "POPSSOrchestrationResult",
    "POPSSOrchestratorConfig",
    "POPSSBaseOrchestrator",
    "POPSSDynamicOrchestrator",
    "POPSSSwarmTopology",
    "POPSSSwarmMessageType",
    "POPSSSwarmMessage",
    "POPSSSwarmTask",
    "POPSSSwarmMember",
    "POPSSSwarmConfig",
    "POPSSSwarmCoordinator",
    "POPSSProtocolMessageType",
    "POPSSProtocolMessage",
    "POPSSProtocolConfig",
    "POPSSProtocolHandler",
    "POPSSMCPBridgeConfig",
    "POPSSToolInfo",
    "POPSSToolCall",
    "POPSSMCPBridge",
    "POPSSMCPBridgeMixin",
    "POPSSPromptConfig",
    "POPSSPromptLoader",
    "POPSSExpertFactory",
]
