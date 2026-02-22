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
Yv Core Modeling Module - Unified Entry Point for Model Components.

This module serves as the primary public interface for all core modeling components
within the PiscesL1 framework. It consolidates imports from various submodules,
providing a clean and unified API for external consumers while maintaining
strict adherence to the YvXxx naming convention.

Architecture Overview:
    The core modeling system implements a sophisticated transformer-based architecture
    with the following major subsystems:

    1. Model Classes (YvModel, YvModelForCausalLM, etc.):
       - Main model implementations supporting various downstream tasks
       - Causal language modeling, sequence classification, token classification
       - Question answering and masked language modeling variants
       - Layer routing and configuration management

    2. Attention Mechanisms (YvAttention, YvFlashAttention, etc.):
       - Standard multi-head attention with Flash Attention 2/3 support
       - Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)
       - Linear attention for efficient O(n) long-context processing
       - Sliding window, sparse, paged, and ring attention variants
       - ALiBi position encoding and attention sinks for streaming

    3. Transformer Blocks (YvTransformerBlock, YvParallelBlock, etc.):
       - Standard sequential attention-MLP blocks
       - Parallel attention-MLP architecture
       - DeepNorm for training stability in deep networks
       - LayerScale, SwiGLU, GeGLU activations
       - LoRA/DoRA parameter-efficient fine-tuning
       - Mixture-of-Depths for dynamic computation

    4. Cache Management (YvUnifiedCacheManager, YvPagedCacheManager, etc.):
       - Unified KV cache management for transformer layers
       - Paged attention cache for efficient memory allocation
       - SSM cache for state space models
       - Speculative decoding cache
       - Multimodal generation cache with compression

    5. Normalization (YvRMSNorm, YvLayerNorm, etc.):
       - RMS Normalization (computationally efficient)
       - Layer Normalization with optional RMS mode
       - Adaptive normalization with conditioning
       - Group normalization for vision transformers
       - DeepNorm for deep network stability

    6. Position Embeddings (YvRotaryEmbedding, YvYaRNRotaryEmbedding, etc.):
       - Rotary Position Embeddings (RoPE) with YaRN scaling
       - Dynamic position embeddings for long contexts
       - Sinusoidal and learned positional embeddings

    7. Embeddings (YvTokenEmbedding, YvMultimodalEmbeddingProjector, etc.):
       - Token embeddings with adaptive scaling
       - Modality-aware embeddings for multimodal inputs
       - Unified embedding interface

    8. Hybrid Architecture (YvHybridBlock, YvSelectiveSSM, etc.):
       - Hybrid attention-SSM blocks
       - Progressive gating for stable training
       - Dynamic routing based on sequence characteristics
       - Hierarchical fusion strategies

    9. State Space Models (YvMamba3Block, YvSelectiveScan, etc.):
       - Mamba-3 implementation with trapezoidal discretization
       - Complex state space and MIMO structure
       - Parallel and chunked scan algorithms
       - Bidirectional SSM for encoder tasks

Design Rationale:
    - Separation of Concerns: Each component handles a specific aspect of model behavior
    - Composability: Components can be combined and extended for different use cases
    - Memory Efficiency: Multiple caching strategies for different scenarios
    - Long Context Support: Architecture designed for ultra-long sequences (1M+ tokens)
    - Multimodal Ready: Native support for text, image, audio, video, document modalities

Module Organization:
    The actual implementations reside in individual submodule files:
    - model.py: Main model classes and layer routing
    - attention.py: All attention mechanism implementations
    - blocks.py: Transformer block variants
    - cache.py: Cache management systems
    - norms.py: Normalization layers and position embeddings
    - embedding.py: Token and position embedding layers
    - hybrid.py: Hybrid attention-SSM blocks
    - mamba3.py: State space model implementations

Usage Example:
    >>> from model.core import YvModel, YvAttention, YvTransformerBlock
    >>> from model.core import YvRMSNorm, YvRotaryEmbedding
    >>> 
    >>> # Initialize model
    >>> model = YvModel(config)
    >>> 
    >>> # Use individual components
    >>> attention = YvAttention(config)
    >>> norm = YvRMSNorm(hidden_size)

Dependencies:
    - torch: PyTorch deep learning framework
    - model.config: Configuration classes
    - model.moe: Mixture of Experts components
    - model.reasoning: Reasoning enhancement modules
    - model.multimodal: Multimodal encoder components
    - model.generation: Generation and decoding utilities
    - utils.dc: Logging and utility functions

Note:
    All classes follow the YvXxx naming convention as per project standards.
    This module does not expose any functions or internal classes directly.
    For configuration options, see model.config module.
"""

from .model import (
    YvModel,
    YvModelForCausalLM,
    YvModelForSequenceClassification,
    YvModelForTokenClassification,
    YvModelForQuestionAnswering,
    YvModelForMaskedLM,
    YvModelType,
    YvLayerType,
    YvLayerConfig,
    YvLayerRouter,
)

from .attention import (
    YvAttention,
    YvAttentionConfig,
    YvAttentionBackend,
    YvFlashAttention,
    YvMultiQueryAttention,
    YvLinearAttention,
    YvSlidingWindowAttention,
    YvSparseAttention,
    YvPagedAttention,
    YvLocalGlobalAttention,
    YvRingAttention,
    YvALiBi,
    YvAttentionSink,
    YvQKNormalizer,
    YvH2OAttention,
    YvDynamicH2OAttention,
)

from .blocks import (
    YvTransformerBlock,
    YvBlockConfig,
    YvBlockType,
    YvLayerScale,
    YvSwiGLU,
    YvGeGLU,
    YvLoRA,
    YvDoRA,
    YvAdaptiveComputationTime,
    YvMixtureOfDepths,
    YvCrossAttention,
    YvCrossAttentionBlock,
    YvExpertChoiceMLP,
    YvParallelBlock,
    YvDeepNormBlock,
    YvManifoldConstraint,
    YvHyperConnection,
    YvMHCBlock,
    YvMHCTransformer,
    YvMHCLayerReplacement,
    YvMHCLoss,
    create_mhc_transformer,
)

from .cache import (
    YvUnifiedCacheManager,
    YvCacheConfig,
    YvCacheType,
    YvEvictionPolicy,
    YvCacheBlock,
    YvPagedCacheManager,
    YvSSMCacheManager,
    YvSpeculativeCacheManager,
    YvMultimodalCacheManager,
    YvCacheCompressor,
)

from .norms import (
    YvRMSNorm,
    YvRotaryEmbedding,
    YvLayerNorm,
    YvGroupNorm,
    YvAdaptiveLayerNorm,
    YvDeepNorm,
    YvParallelResidualNorm,
    YvYaRNRotaryEmbedding,
    YvDynamicYaRNRotaryEmbedding,
)

from .embedding import (
    YvEmbeddingConfig,
    YvAdaptiveEmbeddingScaling,
    YvSinusoidalPositionEmbedding,
    YvLearnedPositionEmbedding,
    YvModalityEmbedding,
    YvTokenEmbedding,
    YvUnifiedEmbedding,
    YvMultimodalEmbeddingProjector,
)

from .hybrid import (
    YvHybridBlock,
    YvHybridConfig,
    YvHybridMode,
    YvSelectiveSSM,
    YvProgressiveHybridGate,
    YvAdaptiveRouter,
    YvHierarchicalFusion,
    YvJambaBlock,
)

from .mamba3 import (
    YvMamba3Block,
    YvMamba3Config,
    YvMamba3Stack,
    YvMamba3Integration,
    YvMamba3Encoder,
    YvMamba3Decoder,
    YvSSMMode,
    YvSSMType,
    YvTrapezoidalDiscretization,
    YvComplexStateSpace,
    YvMIMOStateSpace,
    YvSelectiveScan,
    YvParallelScan,
    YvChunkedParallelScan,
    YvBidirectionalSSM,
    YvGatedSSM,
    YvMultiHeadSSM,
    YvAdaptiveDiscretization,
    YvVKernel,
    YvSSDuality,
    YvHierarchicalSSM,
    YvStateCache,
    YvFlashSSM,
)

__all__ = [
    "YvModel",
    "YvModelForCausalLM",
    "YvModelForSequenceClassification",
    "YvModelForTokenClassification",
    "YvModelForQuestionAnswering",
    "YvModelForMaskedLM",
    "YvModelType",
    "YvLayerType",
    "YvLayerConfig",
    "YvLayerRouter",

    "YvAttention",
    "YvAttentionConfig",
    "YvAttentionBackend",
    "YvFlashAttention",
    "YvMultiQueryAttention",
    "YvLinearAttention",
    "YvSlidingWindowAttention",
    "YvSparseAttention",
    "YvPagedAttention",
    "YvLocalGlobalAttention",
    "YvRingAttention",
    "YvALiBi",
    "YvAttentionSink",
    "YvQKNormalizer",
    "YvH2OAttention",
    "YvDynamicH2OAttention",

    "YvTransformerBlock",
    "YvBlockConfig",
    "YvBlockType",
    "YvLayerScale",
    "YvSwiGLU",
    "YvGeGLU",
    "YvLoRA",
    "YvDoRA",
    "YvAdaptiveComputationTime",
    "YvMixtureOfDepths",
    "YvCrossAttention",
    "YvCrossAttentionBlock",
    "YvExpertChoiceMLP",
    "YvParallelBlock",
    "YvDeepNormBlock",
    "YvManifoldConstraint",
    "YvHyperConnection",
    "YvMHCBlock",
    "YvMHCTransformer",
    "YvMHCLayerReplacement",
    "YvMHCLoss",
    "create_mhc_transformer",

    "YvUnifiedCacheManager",
    "YvCacheConfig",
    "YvCacheType",
    "YvEvictionPolicy",
    "YvCacheBlock",
    "YvPagedCacheManager",
    "YvSSMCacheManager",
    "YvSpeculativeCacheManager",
    "YvMultimodalCacheManager",
    "YvCacheCompressor",

    "YvRMSNorm",
    "YvRotaryEmbedding",
    "YvLayerNorm",
    "YvGroupNorm",
    "YvAdaptiveLayerNorm",
    "YvDeepNorm",
    "YvParallelResidualNorm",
    "YvYaRNRotaryEmbedding",
    "YvDynamicYaRNRotaryEmbedding",

    "YvEmbeddingConfig",
    "YvAdaptiveEmbeddingScaling",
    "YvSinusoidalPositionEmbedding",
    "YvLearnedPositionEmbedding",
    "YvModalityEmbedding",
    "YvTokenEmbedding",
    "YvUnifiedEmbedding",
    "YvMultimodalEmbeddingProjector",

    "YvHybridBlock",
    "YvHybridConfig",
    "YvHybridMode",
    "YvSelectiveSSM",
    "YvProgressiveHybridGate",
    "YvAdaptiveRouter",
    "YvHierarchicalFusion",
    "YvJambaBlock",

    "YvMamba3Block",
    "YvMamba3Config",
    "YvMamba3Stack",
    "YvMamba3Integration",
    "YvMamba3Encoder",
    "YvMamba3Decoder",
    "YvSSMMode",
    "YvSSMType",
    "YvTrapezoidalDiscretization",
    "YvComplexStateSpace",
    "YvMIMOStateSpace",
    "YvSelectiveScan",
    "YvParallelScan",
    "YvChunkedParallelScan",
    "YvBidirectionalSSM",
    "YvGatedSSM",
    "YvMultiHeadSSM",
    "YvAdaptiveDiscretization",
    "YvVKernel",
    "YvSSDuality",
    "YvHierarchicalSSM",
    "YvStateCache",
    "YvFlashSSM",
]
