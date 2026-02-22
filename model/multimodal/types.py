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

"""Type definitions shared across Yv multimodal subsystems.

This module provides comprehensive type definitions for the Yv multimodal
architecture, including enumerations, dataclasses, and type aliases used
throughout the agent lifecycle, MCP communication, and memory management.

Module Components:
    1. Enumerations:
       - YvMCPMessageType: MCP message categories for agent communication
    
    2. Dataclasses:
       - YvGenerationCondition: Conditioning signals for generation
       - YvMCPMessage: MCP transport envelope for agent messages
       - YvAgenticAction: Agent action selection with metadata
       - YvAgenticObservation: External observations for agents
       - YvAgenticMemory: Persisted agent memories

Key Features:
    - Type-safe agent lifecycle states
    - MCP message classification
    - Generation conditioning with emotion/style support
    - Structured action/observation representation
    - Memory persistence for retrieval and reflection

Usage Example:
    >>> from model.multimodal.types import (
    ...     YvMCPMessageType,
    ...     YvGenerationCondition,
    ...     YvAgenticAction
    ... )
    >>> 
    >>> # Create generation condition
    >>> condition = YvGenerationCondition(
    ...     text_prompt="Generate a summary",
    ...     style_params={"formality": 0.8}
    ... )
    >>> 
    >>> # Create agent action
    >>> action = YvAgenticAction(
    ...     action_type="search",
    ...     parameters={"query": "example"},
    ...     confidence=0.95
    ... )

Note:
    All types use dataclass for immutability and serialization.
    MCP messages require correlation_id for request/response matching.
"""

import uuid
import torch
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .state_machine import YvAgenticState

class YvMCPMessageType(Enum):
    """Enumerate MCP message categories exchanged with orchestrators.
    
    Defines the message types used in Model Context Protocol (MCP) 
    communication between agents and orchestration systems. Each type
    represents a distinct communication pattern in the agent lifecycle.
    
    Message Types:
        OBSERVATION: External observation consumed by the agent.
            Used to pass perceptual data from the environment.
        ACTION: Agent action selection for execution.
            Contains action type and parameters for tool invocation.
        TOOL_CALL: Request to invoke a specific tool.
            Structured tool invocation with parameters.
        TOOL_RESULT: Result returned from tool execution.
            Contains execution status and output data.
        STATE_UPDATE: Agent state transition notification.
            Signals changes in agent lifecycle state.
        CAPABILITY_REGISTER: Agent capability registration.
            Used during agent initialization to declare available tools.
        HEARTBEAT: Keep-alive signal for connection management.
            Periodic ping to maintain connection health.
        SYNC_REQUEST: Request for state synchronization.
            Triggers full state sync between components.
        SYNC_RESPONSE: Response to synchronization request.
            Contains current agent state snapshot.
    
    Example:
        >>> msg_type = YvMCPMessageType.ACTION
        >>> if msg_type == YvMCPMessageType.TOOL_CALL:
        ...     execute_tool(message.payload)
    
    Note:
        Message types are used in YvMCPMessage for routing.
        OBSERVATION and ACTION form the primary agent loop.
    """

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
class YvGenerationCondition:
    """Capture conditioning signals that steer multimodal generation.
    
    Provides a unified interface for specifying generation parameters across
    different modalities. Supports text prompts, emotion vectors, style
    parameters, and arbitrary generation configurations.
    
    Attributes:
        text_prompt (str): Natural language prompt provided by the caller.
            Primary conditioning signal for text-based generation.
            Default: "" (empty string).
        emotion_vector (torch.Tensor | None): Learned emotion embedding used when
            no explicit prompt is supplied. Typically a dense vector from an
            emotion encoder. Shape: [embedding_dim]. Default: None.
        style_params (Dict[str, float] | None): Style knobs forwarded to modality
            specific decoders. Examples: {"formality": 0.8, "creativity": 0.5}.
            Default: None.
        generation_params (Dict[str, Any] | None): Arbitrary generator keyword
            arguments such as sampling temperature, top_k, top_p, or seed.
            Default: None.
    
    Example:
        >>> condition = YvGenerationCondition(
        ...     text_prompt="Write a formal email",
        ...     style_params={"formality": 0.9, "conciseness": 0.7},
        ...     generation_params={"temperature": 0.7, "max_tokens": 500}
        ... )
    
    Note:
        emotion_vector takes precedence over text_prompt when both are provided.
        style_params are modality-specific and may be ignored by some generators.
    """

    text_prompt: str = ""
    emotion_vector: Optional[torch.Tensor] = None
    style_params: Optional[Dict[str, float]] = None
    generation_params: Optional[Dict[str, Any]] = None

@dataclass
class YvMCPMessage:
    """Represent an MCP transport envelope carrying agent messages.
    
    Provides a standardized message format for communication between agents
    and orchestration systems following the Model Context Protocol (MCP).
    Supports request/response correlation, priority scheduling, and
    timestamp tracking.
    
    Attributes:
        message_type (str): One of :class:`YvMCPMessageType` describing intent.
            Determines how the message is routed and processed.
        agentic_id (str): Identifier linking the payload to a specific agent run.
            Used for session management and state tracking.
        payload (Dict[str, Any]): Arbitrary serialized data associated with the message.
            Structure depends on message_type.
        timestamp (str): ISO formatted timestamp supplied by the caller.
            Format: "YYYY-MM-DDTHH:MM:SS.ffffff".
        correlation_id (str): Optional identifier used for request/response matching.
            Empty string if not part of a correlated exchange. Default: "".
        priority (str): Scheduling hint such as ``"normal"`` or ``"high"``.
            Affects message queue ordering. Default: "normal".
    
    Example:
        >>> message = YvMCPMessage(
        ...     message_type=YvMCPMessageType.ACTION.value,
        ...     agentic_id="agent-001",
        ...     payload={"action": "search", "query": "example"},
        ...     timestamp=datetime.now().isoformat(),
        ...     correlation_id="req-123"
        ... )
    
    Note:
        correlation_id is essential for async request/response patterns.
        priority should be one of: "low", "normal", "high", "critical".
    """

    message_type: str
    agentic_id: str
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: str = ""
    priority: str = "normal"

@dataclass
class YvAgenticAction:
    """Describe an agent action selection with supporting metadata.
    
    Represents a single action chosen by the agent policy, including
    the action type, parameters, confidence score, and reasoning.
    Used in the agent action-observation loop for tool execution.
    
    Attributes:
        action_type (str): Action label emitted by the policy.
            Examples: "search", "generate", "analyze", "tool_call".
        parameters (Dict[str, Any]): Arguments passed to downstream tools.
            Structure depends on action_type. May contain nested dicts.
        confidence (float): Confidence score between 0 and 1.
            Indicates policy certainty in the action selection. Default: 1.0.
        reasoning (str): Optional natural language rationale.
            Provides explainability for the action choice. Default: "".
    
    Example:
        >>> action = YvAgenticAction(
        ...     action_type="search",
        ...     parameters={"query": "machine learning", "limit": 10},
        ...     confidence=0.92,
        ...     reasoning="User requested information about ML topics"
        ... )
    
    Note:
        confidence should be calibrated across the action space.
        reasoning is optional but recommended for transparency.
    """

    action_type: str
    parameters: Dict[str, Any]
    confidence: float = 1.0
    reasoning: str = ""

@dataclass
class YvAgenticObservation:
    """Capture external observations consumed by the agent.
    
    Represents a single observation from the environment or external systems,
    including the modality, content, and associated metadata. Used as input
    to the agent's perception and reasoning pipeline.
    
    Attributes:
        modality (str): Modality identifier such as ``"text"`` or ``"image"``.
            Supported values: "text", "image", "audio", "video", "document".
        content (Any): Observation payload—may be text, tensors, or tool outputs.
            Type varies based on modality and observation source.
        metadata (Dict[str, Any]): Auxiliary properties like timestamps or source tags.
            May include: "timestamp", "source", "confidence", "language".
    
    Example:
        >>> observation = YvAgenticObservation(
        ...     modality="text",
        ...     content="The user asked about machine learning",
        ...     metadata={"source": "user_input", "timestamp": "2024-01-15T10:30:00"}
        ... )
    
    Note:
        content should be serializable for memory persistence.
        metadata is extensible for domain-specific information.
    """

    modality: str
    content: Any
    metadata: Dict[str, Any]


@dataclass
class YvAgenticMemory:
    """Persisted agent memories used for retrieval and reflection.
    
    Stores the complete memory state of an agent, including observations,
    actions, and reflections. Used for long-term memory persistence,
    experience replay, and agent state serialization.
    
    Attributes:
        observations (List[YvAgenticObservation]): Sequence of observations
            received by the agent. Ordered chronologically.
        actions (List[YvAgenticAction]): Sequence of actions taken by the
            agent. Ordered chronologically, paired with observations.
        reflections (List[str]): Natural language reflections on agent behavior.
            Generated during reflection phases for self-improvement.
    
    Example:
        >>> memory = YvAgenticMemory(
        ...     observations=[observation1, observation2],
        ...     actions=[action1, action2],
        ...     reflections=["Should explore more before committing to action"]
        ... )
    
    Note:
        Observations and actions should maintain temporal alignment.
        Reflections are optional but enhance agent learning.
        Used by YvMemory for persistent storage.
    """

    observations: List[YvAgenticObservation]
    actions: List[YvAgenticAction]
    reflections: List[str]
