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
Yv Mixture-of-Experts (MoE) Module

This module provides comprehensive Mixture-of-Experts components for the Yv
model, implementing state-of-the-art MoE architectures matching the latest
large language model designs.

MoE Architecture Overview:
    Mixture-of-Experts replaces dense feed-forward layers with sparse expert
    routing, enabling massive model scaling while maintaining computational
    efficiency. Each token is routed to a subset of experts based on learned
    routing decisions.

Module Components:
    1. Expert Networks (expert.py):
       - YvExpertBase: Abstract base class for all experts
       - YvStandardExpert: Standard FFN expert
       - YvSwiGLUExpert: SwiGLU-based expert (recommended)
       - YvGeGLUExpert: GeGLU-based expert
       - YvMultiLayerExpert: Deep multi-layer expert
       - YvSharedExpert: Shared expert for DeepSeek-V3 style MoE
       - YvExpertFactory: Factory for creating expert instances
       - YvExpertConfig: Configuration dataclass for experts
       - YvExpertType: Enumeration of expert types
    
    2. Routing Components (gate.py):
       - YvMoEGate: Standard top-k routing gate with load balancing
       - YvStableMoEGate: Stable routing with load prediction
       - moe_init_weights: Weight initialization function
    
    3. MoE Layers (layer.py):
       - YvDynamicMoELayer: Dynamic MoE with shared expert support
       - YvExpertChoiceRouter: Expert-choice routing mechanism
       - YvFineGrainedRouter: Fine-grained expert segmentation
       - YvDeepSeekMoELayer: DeepSeek-V3 style complete MoE

Key Features:
    - Expert-choice routing with capacity constraints
    - Shared expert isolation (DeepSeek-V3 style)
    - Fine-grained expert segmentation for flexible routing
    - Auxiliary loss-free load balancing
    - Dynamic device migration for large expert pools
    - UltraMem TDQKR optimization for large expert counts

Performance Characteristics:
    - Routing overhead: O(num_experts) for gate computation
    - Expert computation: O(top_k * expert_size) per token
    - Memory: Expert parameters + routing buffers
    - Typical speedup: 2-4x over dense equivalent

Design Principles:
    - Single implementation per feature (no redundancy)
    - Flagship-level completeness matching latest LLM architectures
    - Efficient memory management with dynamic expert placement
    - Robust load balancing without auxiliary loss degradation

Usage Example:
    >>> from model.moe import (
    ...     YvDeepSeekMoELayer,
    ...     YvExpertFactory,
    ...     YvExpertConfig,
    ...     YvExpertType
    ... )
    >>> 
    >>> # Create expert configuration
    >>> config = YvExpertConfig(
    ...     hidden_size=4096,
    ...     intermediate_size=11008,
    ...     expert_type=YvExpertType.SWIGLU
    ... )
    >>> 
    >>> # Create single expert
    >>> expert = YvExpertFactory.create(config)
    >>> output = expert(hidden_states)
    >>> 
    >>> # Create full MoE layer (typically via model config)
    >>> # moe_layer = YvDeepSeekMoELayer(model_config)

Note:
    All classes follow the YvXxx naming convention.
    MoE layers are typically instantiated through model configuration.
    Expert count and top-k should be tuned based on computational budget.
"""

from .gate import YvMoEGate, moe_init_weights
from .layer import (
    YvDynamicMoELayer,
    YvExpertChoiceRouter,
    YvFineGrainedRouter,
    YvDeepSeekMoELayer,
)
from .expert import (
    YvExpertBase,
    YvStandardExpert,
    YvSwiGLUExpert,
    YvGeGLUExpert,
    YvMultiLayerExpert,
    YvSharedExpert,
    YvExpertFactory,
    YvExpertConfig,
    YvExpertType,
    create_expert_module,
)

__all__ = [
    "YvMoEGate",
    "moe_init_weights",
    "YvDynamicMoELayer",
    "YvExpertChoiceRouter",
    "YvFineGrainedRouter",
    "YvDeepSeekMoELayer",
    "YvExpertBase",
    "YvStandardExpert",
    "YvSwiGLUExpert",
    "YvGeGLUExpert",
    "YvMultiLayerExpert",
    "YvSharedExpert",
    "YvExpertFactory",
    "YvExpertConfig",
    "YvExpertType",
    "create_expert_module",
]
