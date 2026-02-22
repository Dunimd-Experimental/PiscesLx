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
Agentic Module - Unified Entry Point for Autonomous Agent Components.

This module serves as the primary public interface for all agentic (autonomous agent)
functionality within the PiscesL1 framework. It consolidates imports from the underlying
multimodal.agentic subsystem, providing a clean and unified API for external consumers.

Architecture Overview:
    The agentic system implements a sophisticated autonomous agent framework inspired by
    cutting-edge large language models (LLaMA 4, DeepSeek-V3.2, Qwen 3.5, GLM 5, etc.).
    It comprises several interconnected components:

    1. Core Agent (YvAgentic): The foundational agent implementation with reasoning
       and action capabilities, supporting multi-step task execution and tool integration.

    2. Encoder (YvAgenticEncoder): Specialized encoder for processing agent-specific
       inputs, handling prompt engineering, context management, and state serialization.

    3. Enhanced Components: Advanced planning, orchestration, evaluation, and memory
       systems that extend the core agent with long-term reasoning capabilities.

    4. ReAct Framework (YvReActAgentic, YvReActEngine): Implementation of the
       Reasoning + Acting paradigm, enabling iterative thought-action-observation cycles.

    5. State Machine (YvStateMachine, YvAgenticState, YvAgenticEvent):
       Finite state machine for managing agent lifecycle, transitions, and event handling.

    6. Tool Executor (YvToolExecutor, YvToolResult): Robust tool execution
       framework with result handling, error recovery, and execution tracing.

Design Rationale:
    - Separation of Concerns: Each component handles a specific aspect of agent behavior
    - Composability: Components can be combined and extended for different use cases
    - State Management: Explicit state machine ensures predictable agent behavior
    - Tool Integration: Flexible tool execution with comprehensive error handling

Module Organization:
    The actual implementations reside in model.multimodal.agentic* modules. This __init__.py
    provides:
    - Backward compatibility for existing code importing from model.agentic
    - A centralized import point reducing coupling to internal module structure
    - Clear public API definition through __all__ exports

Usage Example:
    >>> from model.agentic import YvAgentic, YvReActAgentic
    >>> agent = YvAgentic(config)
    >>> react_agent = YvReActAgentic(agent, tools)

Dependencies:
    - model.multimodal.agentic: Core agent implementation
    - model.multimodal.agentic_encoder: Agent input/output encoding
    - model.multimodal.enhanced_agentic: Advanced agent capabilities
    - model.multimodal.react_agentic: ReAct framework implementation
    - model.multimodal.state_machine: State management infrastructure
    - model.multimodal.tool_executor: Tool execution framework

Note:
    All classes follow the YvXxx naming convention as per project standards.
    This module does not expose any functions or internal classes directly.
"""

from ..multimodal.agentic import YvAgentic
from ..multimodal.agentic_encoder import YvAgenticEncoder
from ..multimodal.enhanced_agentic import (
    YvLongTermPlanner,
    YvToolOrchestrator,
    YvSelfEvaluator,
    YvPersistentMemory,
)
from ..multimodal.react_agentic import YvReActAgentic, YvReActEngine
from ..multimodal.state_machine import (
    YvStateMachine,
    YvAgenticState,
    YvAgenticEvent,
)
from ..multimodal.tool_executor import YvToolExecutor, YvToolResult

__all__ = [
    "YvAgentic",
    "YvAgenticEncoder",
    "YvLongTermPlanner",
    "YvToolOrchestrator",
    "YvSelfEvaluator",
    "YvPersistentMemory",
    "YvReActAgentic",
    "YvReActEngine",
    "YvStateMachine",
    "YvAgenticState",
    "YvAgenticEvent",
    "YvToolExecutor",
    "YvToolResult",
]
