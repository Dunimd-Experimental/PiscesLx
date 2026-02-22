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

"""Yv Model - Unified Flagship Architecture for Next-Generation AI.

This package provides the complete Yv model implementation, a flagship-level
architecture designed to match and exceed the capabilities of the latest large
language models (LLaMA 4, Kimi 2.5, DeepSeek-V3.2, Qwen 3.5, etc.).

Architecture Overview:
    The Yv model is organized into several interconnected modules:
    
    1. **Core Components** (model.core):
       - Transformer blocks with multiple attention variants
       - Hybrid Attention-Mamba architectures
       - Advanced normalization (RMSNorm, DeepNorm, Adaptive LayerNorm)
       - Rotary position embeddings (RoPE, YaRN, Dynamic YaRN)
       - KV cache management with paged and compressed variants
    
    2. **Mixture-of-Experts** (model.moe):
       - Fine-grained expert segmentation (64+ experts)
       - Expert-choice routing with load balancing
       - Shared expert isolation for stability
       - Dynamic capacity scaling
    
    3. **Multimodal Processing** (model.multimodal):
       - Vision encoder with native resolution support
       - Audio encoder with mel-spectrogram processing
       - Video encoder with 3D spatio-temporal RoPE
       - Document encoder with layout analysis
       - Cross-modal fusion with semantic alignment
    
    4. **Reasoning Components** (model.reasoning):
       - Chain-of-Thought with memory augmentation
       - Multi-path parallel reasoning
       - Recursive depth reasoning with verification
       - Thought tree search with MCTS
       - Meta-learning for strategy adaptation
    
    5. **Generation Utilities** (model.generation):
       - Speculative decoding with parallel verification
       - Multiple sampling strategies (nucleus, top-k, contrastive)
       - Adaptive parameter adjustment
    
    6. **Agent Capabilities** (model.agentic):
       - Tool registration and execution
       - Memory management with retrieval
       - State tracking and planning
    
    7. **Tokenization** (model.tokenizer):
       - Multi-backend support (Qwen3, H-Network visual)
       - BPE training from corpus
       - Special token management

Key Features:
    - **Flagship Completeness**: Single implementation per feature, no redundancy
    - **Hybrid Architecture**: Seamless integration of attention and Mamba-3 SSM
    - **Multimodal Native**: Built-in support for vision, audio, video, documents
    - **Efficient Inference**: Paged KV cache, speculative decoding, quantization
    - **Advanced Reasoning**: Multi-path, recursive, and verification-based reasoning
    - **Agent Integration**: Tool use, memory, and planning capabilities

Design Principles:
    - Unified entry point for training and inference
    - Single implementation per feature (no redundancy)
    - Flagship-level completeness matching latest LLM architectures
    - Support for hybrid attention-Mamba architectures
    - Exposed classes follow naming conventions:
      * model/ classes: YvXxxXxx
      * opss/ classes: POPSSXxxXxx
      * Other directories: PiscesLxXxxXx

Example:
    >>> from model import YvConfig, YvModelForCausalLM
    >>> 
    >>> # Load configuration
    >>> config = YvConfig.from_json("config.json")
    >>> 
    >>> # Initialize model
    >>> model = YvModelForCausalLM(config)
    >>> 
    >>> # Forward pass
    >>> outputs = model(input_ids, attention_mask=attention_mask)
    >>> logits = outputs.logits

Modules:
    - config: Configuration dataclasses and enums
    - core: Core transformer components
    - tokenizer: Multi-backend tokenization
    - moe: Mixture-of-Experts implementation
    - multimodal: Multimodal encoders and fusion
    - reasoning: Reasoning engines and meta-learning
    - generation: Sampling and speculative decoding
    - utils: RoPE and quantization utilities

Note:
    This module exposes only classes (not functions) following the naming
    convention YvXxxXxx for model/ directory components.
"""

from .config import (
    YvConfig,
    YvAttentionType,
    YvPositionEmbeddingType,
    YvActivationType,
)

from .core import (
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
    YvRMSNorm,
    YvRotaryEmbedding,
    YvLayerNorm,
    YvGroupNorm,
    YvAdaptiveLayerNorm,
    YvDeepNorm,
    YvParallelResidualNorm,
    YvYaRNRotaryEmbedding,
    YvDynamicYaRNRotaryEmbedding,
    YvEmbeddingConfig,
    YvAdaptiveEmbeddingScaling,
    YvSinusoidalPositionEmbedding,
    YvLearnedPositionEmbedding,
    YvModalityEmbedding,
    YvTokenEmbedding,
    YvUnifiedEmbedding,
    YvMultimodalEmbeddingProjector,
    YvHybridBlock,
    YvHybridConfig,
    YvHybridMode,
    YvSelectiveSSM,
    YvProgressiveHybridGate,
    YvAdaptiveRouter,
    YvHierarchicalFusion,
    YvJambaBlock,
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

from .tokenizer import (
    YvTokenizer,
    YvTokenizerBuilder,
    YvTokenizerConfig,
    YvSpecialTokens,
    YvSpecialTokenType,
)

from .moe import (
    YvMoEGate,
    YvDynamicMoELayer,
    YvExpertChoiceRouter,
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

from .multimodal import (
    YvUnifiedReasoner,
    YvVisionEncoder,
    YvAudioEncoder,
    YvDocEncoder,
    YvVideoEncoder,
    YvAgenticEncoder,
    YvDynamicModalFusion,
)

from .reasoning import (
    YvCoTMemoryReasoner,
    YvMultiPathReasoningEngine,
    YvMultiPathMetaLearner,
    YvMultiPathInferenceEngine,
    YvUnifiedMultiPathReasoningSystem,
    YvRecursiveDepthReasoner,
    YvRecursiveReasoningConfig,
    YvThoughtTreeReasoner,
    YvEnhancedReasoningSystem,
    YvEnhancedReasoningConfig,
    YvMultiModalReasoningEnhancer,
)

from .generation import (
    YvAdaptiveSpeculativeDecoder,
    YvSpeculativeConfig,
    YvSampler,
    YvSamplingConfig,
    YvSamplingStrategy,
)

from .utils import (
    YvYaRNRotaryEmbedding,
    YvDynamicYaRNRotaryEmbedding,
    YvLinearScalingRoPE,
    YvQuantizer,
    YvQuantizationConfig,
    YvQuantizationType,
)

__all__ = [
    "YvConfig",
    "YvAttentionType",
    "YvPositionEmbeddingType",
    "YvActivationType",

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

    "YvTokenizer",
    "YvTokenizerBuilder",
    "YvTokenizerConfig",
    "YvSpecialTokens",
    "YvSpecialTokenType",

    "YvMoEGate",
    "YvDynamicMoELayer",
    "YvExpertChoiceRouter",
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

    "YvUnifiedReasoner",
    "YvVisionEncoder",
    "YvAudioEncoder",
    "YvDocEncoder",
    "YvVideoEncoder",
    "YvAgenticEncoder",
    "YvDynamicModalFusion",

    "YvCoTMemoryReasoner",
    "YvMultiPathReasoningEngine",
    "YvMultiPathMetaLearner",
    "YvMultiPathInferenceEngine",
    "YvUnifiedMultiPathReasoningSystem",
    "YvRecursiveDepthReasoner",
    "YvRecursiveReasoningConfig",
    "YvThoughtTreeReasoner",
    "YvEnhancedReasoningSystem",
    "YvEnhancedReasoningConfig",
    "YvMultiModalReasoningEnhancer",

    "YvAdaptiveSpeculativeDecoder",
    "YvSpeculativeConfig",
    "YvSampler",
    "YvSamplingConfig",
    "YvSamplingStrategy",

    "YvYaRNRotaryEmbedding",
    "YvDynamicYaRNRotaryEmbedding",
    "YvLinearScalingRoPE",
    "YvQuantizer",
    "YvQuantizationConfig",
    "YvQuantizationType",
]
