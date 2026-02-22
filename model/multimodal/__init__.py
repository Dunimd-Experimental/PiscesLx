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

"""Yv Multimodal Module - Unified Multimodal Processing for PiscesL1.

This module provides comprehensive multimodal processing capabilities for the
Yv model, integrating vision, audio, video, document, and agentic
modalities into a unified architecture matching the latest large language
model designs.

Module Components:
    1. Perception Encoders:
       - YvVisionEncoder: Image encoding with 3D RoPE support
       - YvVideoEncoder: Video encoding with temporal modeling
       - YvAudioEncoder: Multi-task audio feature extraction
       - YvDocEncoder: Document understanding with layout analysis
       - YvAgenticEncoder: Agent-specific encoding
       - YvSemanticEncoder: Semantic text representation
    
    2. Fusion Components:
       - YvDynamicModalFusion: Dynamic cross-modal fusion
       - YvEnhancedModalFusion: Enhanced fusion with gating
       - YvCrossModalAttention: Cross-modal attention mechanism
       - YvModalFusionConfig: Fusion configuration
    
    3. Generation Components:
       - YvGenerator: Multimodal content generation
       - YvGenerationResult: Generation output container
       - YvGenerationBackend: Generation backend enumeration
    
    4. Agentic Components:
       - YvAgentic: Agent orchestration controller
       - YvAgenticState: Agent state enumeration
       - YvAgenticAction: Agent action representation
       - YvAgenticObservation: Agent observation container
    
    5. Memory System:
       - YvMemory: Unified memory management
       - YvMemoryConfig: Memory configuration
    
    6. MCP Integration:
       - YvMCPGenerationServer: MCP generation server
       - YvMCPToolRegistry: MCP tool registry
       - YvMCPMessage: MCP message envelope
       - YvMCPMessageType: MCP message types
    
    7. Reasoning:
       - YvUnifiedReasoner: Unified reasoning engine
    
    8. Hardware Adaptation:
       - YvHardwareAdaptiveConfig: Hardware-aware configuration

Key Features:
    - Native multimodal tokenization and fusion
    - 3D spatio-temporal rotary positional embeddings
    - Cross-modal attention with xFormers support
    - Multi-task audio encoding (emotion, prosody, spectrum)
    - Document understanding with layout and table analysis
    - Agent orchestration with MCP protocol support
    - Unified memory system with semantic retrieval
    - Hardware-adaptive configuration

Performance Characteristics:
    - Vision encoding: O(N^2) attention with SDPA optimization
    - Audio encoding: O(T * mel_bins) spectral processing
    - Cross-modal fusion: O(M * N) where M is modalities
    - Memory retrieval: O(log N) with FAISS indexing

Usage Example:
    >>> from model.multimodal import (
    ...     YvVisionEncoder,
    ...     YvAudioEncoder,
    ...     YvDynamicModalFusion,
    ...     YvAgentic,
    ...     YvMemory
    ... )
    >>> 
    >>> # Initialize encoders
    >>> vision = YvVisionEncoder(config)
    >>> audio = YvAudioEncoder(config)
    >>> 
    >>> # Encode modalities
    >>> vision_features = vision(images)
    >>> audio_features = audio(waveforms)
    >>> 
    >>> # Fuse modalities
    >>> fusion = YvDynamicModalFusion(config)
    >>> fused = fusion({"image": vision_features, "audio": audio_features})
    >>> 
    >>> # Agent orchestration
    >>> agent = YvAgentic(config)
    >>> result = agent.process(observations)

Note:
    All classes follow the YvXxx naming convention.
    MCP integration requires proper tool registration.
    Memory system supports both FAISS and NumPy backends.
"""

from .vision import YvVisionEncoder, YvSpatioTemporalRoPE3D
from .video import YvVideoEncoder
from .audio import YvAudioEncoder
from .doc import YvDocEncoder
from .agentic import YvAgentic
from .agentic_encoder import YvAgenticEncoder
from .attention import YvCrossModalAttention
from .fusion import YvDynamicModalFusion
from .enhanced_fusion import YvEnhancedModalFusion, YvModalFusionConfig
from .generator import YvGenerator, YvGenerationResult, YvGenerationBackend
from .hw import YvHardwareAdaptiveConfig
from ..reasoning import YvUnifiedReasoner
from .server import YvMCPGenerationServer
from .mcp import YvMCPToolRegistry
from .types import (
    YvMCPMessageType,
    YvGenerationCondition,
    YvMCPMessage,
    YvAgenticAction,
    YvAgenticObservation,
)
# YvAgenticState is in state_machine.py
from .state_machine import YvAgenticState
# YvMemory is the unified memory system
from .memory import YvMemory, YvMemoryConfig

__all__ = [
    "YvVisionEncoder",
    "YvSpatioTemporalRoPE3D",
    "YvVideoEncoder",
    "YvAudioEncoder",
    "YvDocEncoder",
    "YvAgentic",
    "YvAgenticEncoder",
    "YvCrossModalAttention",
    "YvDynamicModalFusion",
    "YvEnhancedModalFusion",
    "YvModalFusionConfig",
    "YvGenerator",
    "YvGenerationResult",
    "YvGenerationBackend",
    "YvHardwareAdaptiveConfig",
    "YvUnifiedReasoner",
    "YvMCPGenerationServer",
    "YvMCPToolRegistry",
    "YvAgenticState",
    "YvMCPMessageType",
    "YvGenerationCondition",
    "YvMCPMessage",
    "YvAgenticAction",
    "YvAgenticObservation",
    "YvMemory",
    "YvMemoryConfig",
]
