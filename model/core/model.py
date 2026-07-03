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

from __future__ import annotations

"""
Main Model Implementation for Yv Architecture.

This module implements the complete Yv model architecture, serving as the
primary entry point for model instantiation and inference. It integrates all
core components including transformer layers, multimodal encoders, reasoning
enhancers, and generation capabilities into a unified model class.

Architecture Overview:
    The Yv model implements a sophisticated multi-component architecture:

    1. Core Model Classes:
       - YvModel: Base model class
         * Embedding layer with multimodal support
         * Stack of transformer/hybrid blocks
         * Final normalization layer
         * Supports both training and inference modes
       
       - YvModelForCausalLM: Causal language modeling
         * Inherits from YvModel
         * Language modeling head for next-token prediction
         * Supports generation with various decoding strategies
         * Integrated with speculative decoding
       
       - YvModelForSequenceClassification: Sequence classification
         * Classification head on top of pooled output
         * Supports multi-class and multi-label tasks
         * Optional pooling strategies (CLS, mean, max)
       
       - YvModelForTokenClassification: Token classification
         * Per-token classification head
         * Supports NER, POS tagging, etc.
         * CRF layer option for structured prediction
       
       - YvModelForQuestionAnswering: Question answering
         * Span extraction head
         * Start and end position predictions
         * Supports extractive QA tasks
       
       - YvModelForMaskedLM: Masked language modeling
         * MLM head for BERT-style pretraining
         * Supports bidirectional attention
         * Useful for encoder-only variants

    2. Layer Architecture:
       - YvLayerRouter: Dynamic layer routing
         * Routes inputs through different layer configurations
         * Supports conditional computation
         * Layer skipping for efficiency
         * Adaptive depth based on input complexity
       
       - YvLayerStack: Layer stack management
         * Manages collection of transformer blocks
         * Supports heterogeneous layer types
         * Handles forward and backward passes
         * Gradient checkpointing integration

    3. Multimodal Integration:
       - Vision Encoder: YvVisionEncoder
         * Processes image inputs
         * Vision transformer or CNN backbone
         * Projects to model dimension
       
       - Audio Encoder: YvAudioEncoder
         * Processes audio spectrograms
         * Audio transformer architecture
         * Supports streaming audio
       
       - Document Encoder: YvDocEncoder
         * Processes document images
         * Layout-aware encoding
         * OCR integration support
       
       - Video Encoder: YvVideoEncoder
         * Processes video frames
         * Temporal modeling
         * Frame sampling strategies
       
       - Agentic Encoder: YvAgenticEncoder
         * Processes agentic/action inputs
         * Action space encoding
         * State representation
       
       - Dynamic Modal Fusion: YvDynamicModalFusion
         * Fuses multiple modalities
         * Cross-modal attention
         * Modality-aware gating

    4. Reasoning Enhancement:
       - YvMultiModalReasoningEnhancer: Reasoning module
         * Chain-of-thought generation
         * Multi-step reasoning
         * Self-consistency verification
         * Tool use integration

    5. Generation Capabilities:
       - YvAdaptiveSpeculativeDecoder: Speculative decoding
         * Draft-then-verify paradigm
         * Adaptive speculation length
         * Multiple draft candidates
         * Efficient batch verification
       
       - Generation utilities:
         * Beam search with diverse decoding
         * Nucleus (top-p) sampling
         * Temperature scaling
         * Repetition penalty

    6. Cache and Memory Management:
       - YvUnifiedCacheManager: Cache integration
         * KV cache for attention layers
         * SSM cache for Mamba layers
         * Hybrid cache management
         * Memory-efficient generation

Design Rationale:
    - Modularity: Clean separation between model components
    - Flexibility: Multiple task-specific model variants
    - Multimodal: Native support for multiple input modalities
    - Efficiency: Speculative decoding and cache management
    - Reasoning: Enhanced reasoning capabilities for complex tasks

Model Configuration:
    The model is configured via YvConfig which includes:
    - Architecture: hidden_size, num_layers, num_heads
    - Attention: attention_type, sliding_window, sparse_pattern
    - Hybrid: attention_ratio, ssm_state_dim
    - Multimodal: vision_config, audio_config, etc.
    - Generation: max_position_embeddings, rope_scaling

Performance Considerations:
    - Hybrid attention-SSM reduces memory by 40-60%
    - Speculative decoding provides 2-3x speedup
    - Gradient checkpointing reduces memory by 50%
    - Flash Attention provides 2-4x speedup
    - Paged cache enables efficient batched serving

Dependencies:
    - torch: PyTorch deep learning framework
    - .norms: Normalization layers
    - .blocks: Transformer block implementations
    - .cache: Cache management system
    - .hybrid: Hybrid attention-SSM blocks
    - ..config: Configuration classes
    - ..reasoning: Reasoning enhancement modules
    - ..generation: Generation utilities
    - ..multimodal: Multimodal encoders
    - utils.dc: Logging utilities

Usage Example:
    >>> from model.core.model import YvModelForCausalLM
    >>> from model.config import YvConfig
    >>> 
    >>> # Load configuration
    >>> config = YvConfig.from_pretrained("path/to/config")
    >>> 
    >>> # Initialize model
    >>> model = YvModelForCausalLM(config)
    >>> 
    >>> # Forward pass
    >>> outputs = model(
    ...     input_ids=input_ids,
    ...     attention_mask=attention_mask,
    ...     labels=labels  # For training
    ... )
    >>> 
    >>> # Generation
    >>> generated = model.generate(
    ...     input_ids=input_ids,
    ...     max_new_tokens=100,
    ...     temperature=0.7
    ... )

Note:
    All classes follow the YvXxx naming convention.
    Model weights should be loaded via from_pretrained() method.
    For multimodal inputs, use appropriate encoder methods.
    Generation uses speculative decoding by default for efficiency.
"""

import torch
from torch import nn
import torch.nn.functional as F
from .norms import YvRMSNorm
from ..config import YvConfig
from .hybrid import YvHybridBlock
from utils.dc import PiscesLxLogger
from .blocks import YvTransformerBlock
from .cache import YvUnifiedCacheManager
from .dual_injector import YvDualInjector
from typing import Optional, Tuple, Dict, Any, List, Union
from ..generation.speculative import YvAdaptiveSpeculativeDecoder, YvSpeculativeConfig
from ..multimodal import (
    YvUnifiedReasoner,
    YvVisionEncoder,
    YvMoVEVisionEncoder,
    YvSparseCutRouter,
    YvAudioEncoder,
    YvDocEncoder,
    YvVideoEncoder,
    YvAgenticEncoder,
    YvDynamicModalFusion
)
# RCA: Recursive Cross-Modal Attention (ACM TOMM 2026)
from ..multimodal.rca_fusion import YvDeepCrossLayerInjector
# SEER: Self-Guided Experience-Enhanced Reasoning (arXiv:2508.15214)
from ..multimodal.seer_executor import YvSEERExecutor
# VeriCoT: Neuro-Symbolic CoT Validation (arXiv:2511.04662)
from ..reasoning.vericot import YvVeriCoTVerifier, YvVeriCoTReflector
# CRV: Circuit-based Reasoning Verification (Zhao et al., ICLR 2026 Oral, arXiv:2510.09312)
from ..reasoning.verification import YvCRVIntegration
# CoMeT: Collaborative Memory Transformer (arXiv:2602.01766, ACL 2026)
from .comet import YvCoMeTMemory
# Token Sparse Attention / Tactic (Kan Zhu et al., ICLR 2026, arXiv:2502.12216)
from .token_sparse_attn import YvTokenSparseAttention
# mHC-lite (arXiv:2601.05732)
from .mhc_lite import YvMHCLiteHyperConnection
# OOMB: Million-token chunked processing (Yv Architecture)
from .long_context import YvOOMBContext, YvREFORM
from dataclasses import dataclass
from enum import Enum
import math

from utils.paths import get_log_file
_LOG = PiscesLxLogger("Yv.Core", file_path=get_log_file("Yv.Core"), enable_file=True)

class YvModelType(Enum):
    """Enumeration of model architecture types for different task configurations.
    
    Defines the overall architecture pattern of the model, determining
    how layers are connected and what tasks the model supports.
    
    Attributes:
        DECODER_ONLY: Decoder-only architecture for autoregressive generation.
            Standard GPT-style architecture with causal attention.
            Suitable for text generation, completion, and dialogue.
            All layers attend only to previous positions.
        ENCODER_DECODER: Encoder-decoder architecture for seq2seq tasks.
            BART/T5-style architecture with separate encoder and decoder.
            Suitable for translation, summarization, and rewriting.
            Encoder uses bidirectional attention, decoder uses causal.
        ENCODER_ONLY: Encoder-only architecture for understanding tasks.
            BERT-style architecture with bidirectional attention.
            Suitable for classification, extraction, and embedding.
            No generation capability.
        HYBRID: Hybrid architecture combining multiple patterns.
            Custom architecture with mixed attention patterns.
            May combine decoder and encoder components.
            Flexible for complex multi-task scenarios.
    
    Example:
        >>> model_type = YvModelType.DECODER_ONLY
        >>> if model_type == YvModelType.ENCODER_DECODER:
        ...     print("Using encoder-decoder architecture")
    
    Note:
        Architecture choice affects:
        - Attention mask patterns (causal vs bidirectional)
        - Layer connectivity (cross-attention presence)
        - Supported tasks (generation vs understanding)
    """
    DECODER_ONLY = "decoder_only"
    ENCODER_DECODER = "encoder_decoder"
    ENCODER_ONLY = "encoder_only"
    HYBRID = "hybrid"


class YvLayerType(Enum):
    """Enumeration of layer types for heterogeneous model architectures.
    
    Defines the computational pattern of individual layers, enabling
    mixed architectures with different layer types at different depths.
    
    Attributes:
        ATTENTION: Pure attention layer with standard transformer block.
            Full O(n^2) attention for maximum quality.
            Best for tasks requiring global context.
            Higher memory and compute cost.
        SSM: Pure state space model layer (Mamba-style).
            O(n) complexity for efficient long-context.
            Best for tasks with very long sequences.
            Lower memory footprint.
        HYBRID: Hybrid attention-SSM layer.
            Combines attention and SSM for balanced quality/efficiency.
            Adaptive routing based on sequence characteristics.
            Optimal for mixed-length inputs.
        MOE: Mixture-of-Experts layer.
            Multiple expert networks with routing.
            Increased capacity with efficient computation.
            Best for scaling model capacity.
    
    Example:
        >>> layer_type = YvLayerType.HYBRID
        >>> if layer_type == YvLayerType.MOE:
        ...     print("Using MoE layer with expert routing")
    
    Note:
        Layer types can be mixed within a single model:
        - Early layers: ATTENTION for local patterns
        - Middle layers: HYBRID for balanced processing
        - Late layers: SSM for long-range dependencies
    """
    ATTENTION = "attention"
    SSM = "ssm"
    HYBRID = "hybrid"
    MOE = "moe"
    RECURRENT = "recurrent"


@dataclass
class YvLayerConfig:
    """Configuration dataclass for individual layer specification.
    
    Encapsulates all parameters needed to configure a single layer,
    enabling heterogeneous architectures with different layer types
    and configurations at different depths.
    
    Attributes:
        layer_idx (int): Index of this layer in the model stack.
            Used for position-dependent configurations.
        layer_type (YvLayerType): Type of computational layer.
            Determines attention/SSM/MoE pattern.
        use_checkpoint (bool): Whether to use gradient checkpointing.
            Reduces memory at cost of recomputation. Default: False.
        use_mamba3 (bool): Whether to use Mamba-3 SSM variant.
            Enables advanced SSM features. Default: False.
        use_moe (bool): Whether to use Mixture-of-Experts.
            Enables expert routing. Default: False.
        num_experts (int): Number of experts for MoE layers.
            Only used when use_moe=True. Default: 8.
        expert_capacity (float): Capacity factor for expert routing.
            Values > 1.0 allow token dropping. Default: 1.25.
    
    Example:
        >>> config = YvLayerConfig(
        ...     layer_idx=0,
        ...     layer_type=YvLayerType.ATTENTION,
        ...     use_checkpoint=True
        ... )
    
    Note:
        Layer configurations are typically generated by YvLayerRouter
        based on the overall model configuration.
    """
    layer_idx: int
    layer_type: YvLayerType
    use_checkpoint: bool = False
    use_mamba3: bool = False
    use_moe: bool = False
    num_experts: int = 8
    expert_capacity: float = 1.25


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class YvLayerRouter:
    """Layer router for dynamic layer type assignment.
    
    Routes inputs to appropriate layer type based on configuration,
    determining which layer type (attention, SSM, hybrid, MoE) to use
    for each position in the model stack.
    
    Routing Strategy:
        - Mamba3 layers: Configured via mamba3_layers list
        - MoE layers: Configured via moe_layers list
        - Hybrid layers: Based on sequence threshold
        - Default: Standard attention layers
    
    Key Features:
        - Supports heterogeneous layer architectures
        - Configurable layer type assignments
        - Sequence-length aware routing for hybrid layers
        - Integration with model configuration
    
    Attributes:
        config (YvConfig): Model configuration object.
        n_layer (int): Total number of layers.
        layer_configs (List[YvLayerConfig]): Per-layer configurations.
    
    Example:
        >>> router = YvLayerRouter(config)
        >>> layer_config = router.get_layer_config(0)
        >>> if layer_config.use_mamba3:
        ...     print("Layer 0 uses Mamba-3 SSM")
    
    Note:
        Layer routing is determined at model initialization time
        and remains fixed during training and inference.
    """

    def __init__(self, config: YvConfig):
        """Initialize layer router with model configuration.
        
        Args:
            config: Model configuration containing layer specifications.
                Relevant fields: n_layer, mamba3_layers, moe_layers,
                use_mamba3, mamba3_sequence_threshold.
        """
        self.config = config
        self.n_layer = getattr(config, 'n_layer', 32)
        self.layer_configs: List[YvLayerConfig] = []

        self._build_layer_configs()

    def _build_layer_configs(self):
        """Build layer configurations based on model settings.
        
        Constructs the layer_configs list by determining the type
        of each layer based on configuration parameters.
        
        Priority order:
            1. MoE layers (if in moe_layers list)
            2. SSM layers (if in mamba3_layers list)
            3. Recurrent layers (if in rdt_layer_indices list)
            4. Hybrid layers (if use_mamba3 and sequence threshold met)
            5. Default attention layers
        """
        mamba3_layers = getattr(self.config, 'mamba3_layers', [])
        moe_layers = getattr(self.config, 'moe_layers', [])
        rdt_layer_indices = getattr(self.config, 'rdt_layer_indices', [])
        use_rdt_layers = getattr(self.config, 'use_rdt_layers', True)

        if not rdt_layer_indices and use_rdt_layers:
            n = self.n_layer
            if n >= 16:
                rdt_start = max(0, n - max(2, n // 8))
                rdt_layer_indices = list(range(rdt_start, n - max(1, n // 16)))
            elif n >= 8:
                rdt_layer_indices = [n - 2, n - 3]

        for i in range(self.n_layer):
            layer_type = YvLayerType.ATTENTION

            if i in mamba3_layers:
                layer_type = YvLayerType.SSM
            elif i in moe_layers:
                layer_type = YvLayerType.MOE
            elif i in rdt_layer_indices:
                layer_type = YvLayerType.RECURRENT
            elif getattr(self.config, 'use_mamba3', False):
                threshold = getattr(self.config, 'mamba3_sequence_threshold', 8192)
                if not mamba3_layers or i in mamba3_layers:
                    layer_type = YvLayerType.HYBRID

            self.layer_configs.append(YvLayerConfig(
                layer_idx=i,
                layer_type=layer_type,
                use_mamba3=layer_type in [YvLayerType.SSM, YvLayerType.HYBRID],
                use_moe=layer_type == YvLayerType.MOE,
                num_experts=getattr(self.config, 'num_experts', 8),
                expert_capacity=getattr(self.config, 'expert_capacity', 1.25)
            ))

    def get_layer_config(self, layer_idx: int) -> YvLayerConfig:
        """Get configuration for a specific layer.
        
        Args:
            layer_idx: Index of the layer to get configuration for.
            
        Returns:
            YvLayerConfig for the specified layer.
            Falls back to last layer config if index out of range.
        """
        if layer_idx < len(self.layer_configs):
            return self.layer_configs[layer_idx]
        return self.layer_configs[-1]

class YvModelOutput(dict):
    """Model output supporting both dict-style and attribute-style access.

    Wraps forward return dicts so training engine code (GRPO, DPO, etc.)
    can use either result["logits"] or result.logits seamlessly.
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"YvModelOutput has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"YvModelOutput has no attribute '{name}'")


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class YvModel(nn.Module):
    """Main Yv model implementation with multimodal and reasoning capabilities.
    
    Implements a comprehensive transformer-based language model with support for
    multimodal inputs, hybrid attention-Mamba blocks, reasoning capabilities,
    and speculative decoding. This is the core model class that integrates all
    components into a unified architecture.
    
    Architecture Components:
        1. Embedding Layer:
           - Token embeddings with configurable vocabulary size
           - Optional rotary position embeddings (RoPE)
           - Support for multimodal token embeddings
        
        2. Transformer Layers:
           - Heterogeneous layer architecture (attention, SSM, hybrid, MoE)
           - Configurable layer types per depth
           - Gradient checkpointing support
           - Cache management integration
        
        3. Multimodal Encoders:
           - Vision encoder for image inputs
           - Audio encoder for audio inputs
           - Video encoder for video inputs
           - Document encoder for document images
           - Agentic encoder for action/state inputs
        
        4. Modal Fusion:
           - Dynamic fusion of multiple modalities
           - Cross-modal attention mechanisms
           - Modality-aware gating
        
        5. Output Heads:
           - Language modeling head for generation
           - Task head for classification
           - Evaluation head for scoring
           - MTP (Multi-Token Prediction) heads
        
        6. Reasoning Integration:
           - Unified reasoner for chain-of-thought
           - Multi-modal reasoning enhancer
           - Tool use triggering
    
    Key Features:
        - Multimodal: Native support for text, image, audio, video, documents
        - Hybrid: Combines attention and SSM for efficiency
        - Reasoning: Built-in chain-of-thought and tool use
        - Efficient: Speculative decoding and cache management
        - Flexible: Configurable layer types and architectures
    
    Generation Modes:
        - fast: Standard generation for quick responses
        - thinking: Enhanced reasoning with chain-of-thought
        - auto: Automatic mode selection based on input
    
    Attributes:
        cfg: Configuration object with model hyperparameters.
        config: Alias for cfg for compatibility.
        cache_manager (YvUnifiedCacheManager): Cache management system.
        layer_router (YvLayerRouter): Layer type routing system.
        embed (nn.Embedding): Token embedding layer.
        rotary_emb: Rotary position embedding (optional).
        layers (nn.ModuleList): Stack of transformer/hybrid blocks.
        norm (YvRMSNorm): Final normalization layer.
        vision (YvVisionEncoder): Vision encoder.
        video (YvVideoEncoder): Video encoder.
        audio (YvAudioEncoder): Audio encoder.
        doc (YvDocEncoder): Document encoder.
        modal_fusion: Modal fusion module.
        lm_head (nn.Linear): Language modeling head.
        task_head (nn.Linear): Task classification head.
        eval_head (nn.Linear): Evaluation scoring head.
        reasoner (YvUnifiedReasoner): Reasoning module.
        speculative_decoder: Speculative decoding module.
    
    Example:
        >>> from model.config import YvConfig
        >>> config = YvConfig(
        ...     hidden_size=4096,
        ...     n_layer=32,
        ...     vocab_size=128000
        ... )
        >>> model = YvModel(config)
        >>> 
        >>> # Text-only forward pass
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs["loss"]
        >>> logits = outputs["logits"]
        >>> 
        >>> # Multimodal forward pass
        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     images=images,
        ...     audio=audio
        ... )
        >>> 
        >>> # Generation
        >>> generated, stats = model.generate(
        ...     input_ids=input_ids,
        ...     max_length=100,
        ...     temperature=0.7
        ... )
    
    Note:
        All classes follow the YvXxx naming convention.
        Model weights should be loaded via from_pretrained() method.
        For multimodal inputs, use appropriate encoder methods.
        Generation uses speculative decoding by default for efficiency.
    """

    def named_children(self):
        """Override to exclude certain modules from named_children.

        Excludes agentic module from standard module enumeration
        to prevent it from being included in state_dict operations.

        Yields:
            Tuple[str, nn.Module]: Name and module pairs excluding agentic.
        """
        for name, module in super().named_children():
            if name == "agentic":
                continue
            yield name, module

    def parameters(self, recurse: bool = True):
        """Override to de-duplicate parameters that share the same storage.

        When weight tying (``tie_word_embeddings``) or MTP head sharing
        (``mtp_share_embeddings``) is enabled, multiple ``nn.Parameter``
        objects share the same underlying ``Storage``.  A normal
        ``parameters()`` call yields every one of them, causing optimisers
        such as AdamW to allocate separate momentum/variance entries for
        identical data — inflating optimiser memory by up to 5×.

        This override yields each unique storage exactly once, so the
        optimiser builds exactly one momentum/variance entry per unique
        ``Storage``.  The model's forward pass and gradients are completely
        unaffected because the storage union ensures that a gradient written
        through one ``Parameter`` is immediately visible through all aliases.

        Note:
            This is safe for every major optimiser (AdamW, Adam, SGD, Muon,
            GaLore, INK).  The only observable effect is that
            ``len(list(model.parameters()))`` may be smaller than
            ``sum(1 for _ in model.named_parameters())`` when sharing is
            active.
        """
        seen_data_ptrs: set = set()
        for p in super().parameters(recurse=recurse):
            ptr = p.data_ptr()
            if ptr not in seen_data_ptrs:
                seen_data_ptrs.add(ptr)
                yield p

    def __init__(self, cfg, device=None, dtype=None, quantization_config=None, lora_config=None, modalities=None):
        super().__init__()
        _LOG.debug("YvModel: __init__ start")
        self.cfg = cfg
        self.config = cfg

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype is None:
            dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

        self._modalities = modalities or {'text'}
        self._device = device
        self._dtype = dtype
        self._causal_mask_views: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}

        # Automatic VRAM optimization: selects optimal settings without conflicts
        from utils.vram_controller import auto_optimize, get_vram_monitor
        auto_optimize(cfg)
        self._vram_monitor = get_vram_monitor(cfg)

        if getattr(cfg, 'use_quartet', False):
            cfg.use_fp4 = True
            cfg.fp4_block_size = getattr(cfg, 'fp4_block_size', 16)
            cfg.fp4_stochastic_rounding = True
            cfg.fp4_master_weights_dtype = 'fp32'
            cfg.coat_enabled = True

        # Resolve conflicting VRAM settings across dual flag systems
        if getattr(cfg, 'vram_fp4_training', False):
            cfg.use_fp4 = True
        if getattr(cfg, 'vram_kv_cache_quantization', False):
            cfg.cache_quantization = True
        if getattr(cfg, 'vram_gradient_checkpointing', False):
            cfg.use_checkpoint = True
        if getattr(cfg, 'vram_flash_attention', False):
            cfg.use_flash_attention = True

        if getattr(cfg, 'moe_num_experts', 0) >= 16 and getattr(cfg, 'moe_shared_experts', 0) == 0:
            setattr(self.config, 'moe_shared_experts', max(1, getattr(cfg, 'moe_num_experts', 64) // 32))

        if not hasattr(self.config, 'num_layers'):
            setattr(self.config, 'num_layers', getattr(self.config, 'n_layer', 0))
        if not hasattr(self.config, 'num_heads'):
            setattr(self.config, 'num_heads', getattr(self.config, 'n_head', 0))
        if not hasattr(self.config, 'n_kv_head'):
            setattr(
                self.config,
                'n_kv_head',
                getattr(self.config, 'n_kv_head', getattr(self.config, 'n_head', 0))
            )

        self._apply_backbone_defaults()

        self.quantization_config = quantization_config
        self.lora_config = lora_config

        cache_config = getattr(cfg, 'cache_config', {
            "enabled": True,
            "kv_cache_max_size": 2048,
            "h2o_cache_max_size": 1024,
            "generation_cache_max_size": 512,
            "speculative_cache_max_size": 256,
            "quantization_enabled": True,
            "dynamic_quantization": True,
            "cache_eviction_policy": "lru",
            "cache_type": getattr(cfg, 'cache_type', 'hybrid'),
            "use_h2o_attention": getattr(cfg, 'use_h2o_attention', True),
            "enable_cache_compression": getattr(cfg, 'enable_cache_compression', True),
            "cache_compression_ratio": getattr(cfg, 'cache_compression_ratio', 0.5),
        })
        self.cache_manager = YvUnifiedCacheManager(cache_config)

        self.layer_router = YvLayerRouter(cfg)

        _LOG.debug("YvModel: initializing embedding...")
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, device=device, dtype=dtype)

        if getattr(cfg, 'use_rotary_pos_emb', True):
            self.rotary_emb = self._create_rotary_embedding(cfg, device, dtype)
        else:
            self.rotary_emb = None

        _LOG.debug(f"YvModel: initializing {cfg.n_layer} transformer layers...")

        # PathMoE: shared gate across layer stages
        use_path_moe = getattr(cfg, 'use_path_moe', False)
        path_moe_stage_size = getattr(cfg, 'path_moe_stage_size', 4)
        path_moe_stage_gate = None
        setattr(cfg, '_path_moe_model_id', id(self))

        self.layers = nn.ModuleList([])
        # Announce layer construction only at the first, the midpoint,
        # and the last layer. Logging once per layer is noise at any
        # non-trivial depth.
        n_layer = cfg.n_layer
        mid_layer = max(1, n_layer // 2) if n_layer > 1 else n_layer
        for i in range(n_layer):
            if i == 0 or i == mid_layer or i == n_layer - 1:
                _LOG.debug(f"YvModel: initializing TransformerBlock {i+1}/{n_layer}")

            # PathMoE: (re)create shared gate at stage boundaries
            if use_path_moe:
                if i % path_moe_stage_size == 0:
                    from ..moe.gate import YvStableMoEGate
                    path_moe_stage_gate = YvStableMoEGate(
                        cfg.hidden_size, getattr(cfg, 'moe_num_experts', 8),
                        top_k=getattr(cfg, 'moe_top_k', 2),
                        device=device, dtype=dtype,
                        capacity_factor=getattr(cfg, 'moe_capacity_factor', 1.0),
                        min_capacity=getattr(cfg, 'moe_min_capacity', 4),
                        prediction_horizon=getattr(cfg, 'moe_prediction_horizon', 10),
                        enable_dynamic_capacity=getattr(cfg, 'enable_dynamic_capacity', True),
                        enable_cognitive_density=getattr(cfg, 'enable_cognitive_density', False)
                    )
                setattr(cfg, '_path_moe_stage_idx', i // path_moe_stage_size)

            _ = self.layer_router.get_layer_config(i)
            block = self._build_backbone_block(
                cfg,
                i,
                device,
                dtype,
                path_moe_stage_gate if use_path_moe else None,
            )

            block.cache_manager = self.cache_manager
            block.layer_idx = i
            self.layers.append(block)

        _LOG.debug("YvModel: initializing norm...")
        self.norm = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)

        # Dual-path knowledge injection (FiLM + KV). Always active.
        _LOG.debug("YvModel: initializing dual injector...")
        self.dual_injector = YvDualInjector(cfg, device=device, dtype=dtype)
        self.subconscious = self.dual_injector.subconscious
        self._comet_write_interval = max(1, int(getattr(cfg, 'comet_write_interval', 1)))
        self._comet_write_step = 0

        _LOG.debug("YvModel: initializing multimodal encoders...")
        self._lazy_initialized: Dict[str, bool] = {}
        def _lazy_flag(key: str, expr: bool) -> bool:
            flag = bool(getattr(cfg, 'lazy_init_enabled', True)) and expr
            self._lazy_initialized[key] = not flag
            return flag

        _needs_vision = 'image' in self._modalities
        self._lazy_vision_encoder = None
        if _needs_vision and _lazy_flag('vision', getattr(cfg, 'lazy_init_vision_encoder', True)):
            self.vision = None
        elif _needs_vision:
            base_vision = YvVisionEncoder(cfg, device=device, dtype=dtype)
            self.vision = YvMoVEVisionEncoder(cfg, base_encoder=base_vision, device=device, dtype=dtype) if getattr(cfg, 'use_move_encoder', False) else base_vision
        else:
            self.vision = None
        self._lazy_initialized['vision'] = self.vision is not None

        self.sparse_cut_router = YvSparseCutRouter(cfg) if (getattr(cfg, 'use_sparse_cut', False) and _needs_vision) else None

        _needs_video = 'video' in self._modalities
        self._lazy_video_encoder = None
        if _needs_video and _lazy_flag('video', getattr(cfg, 'lazy_init_video_encoder', True)):
            self.video = None
        elif _needs_video:
            self.video = YvVideoEncoder(cfg, device=device, dtype=dtype)
        else:
            self.video = None
        self._lazy_initialized['video'] = self.video is not None

        _needs_audio = 'audio' in self._modalities
        self._lazy_audio_encoder = None
        if _needs_audio and _lazy_flag('audio', getattr(cfg, 'lazy_init_audio_encoder', True)):
            self.audio = None
        elif _needs_audio:
            self.audio = YvAudioEncoder(cfg, device=device, dtype=dtype)
        else:
            self.audio = None
        self._lazy_initialized['audio'] = self.audio is not None

        _needs_doc = 'doc' in self._modalities
        self._lazy_doc_encoder = None
        if _needs_doc and _lazy_flag('doc', getattr(cfg, 'lazy_init_doc_encoder', True)):
            self.doc = None
        elif _needs_doc:
            self.doc = YvDocEncoder(cfg, device=device, dtype=dtype)
        else:
            self.doc = None
        self._lazy_initialized['doc'] = self.doc is not None

        _needs_multimodal = _needs_vision or _needs_video or _needs_audio or _needs_doc
        if getattr(cfg, 'use_agentic', False) and _needs_multimodal:
            self.agent_encoder = YvAgenticEncoder(cfg, device=device, dtype=dtype)
        else:
            self.agent_encoder = None

        # Unified multimodal fusion — absorbs Dynamic/Enhanced/RecurrentRefiner/SyncFusion/RCA
        if _needs_multimodal:
            self.modal_fusion = YvDynamicModalFusion(cfg, device=device, dtype=dtype)
        else:
            self.modal_fusion = None

        # === 2026 flagship feature init (all lazy / conditional) ===
        self.deep_cross_layer_injector = None  # lazy: _lazy_get_rca
        self.seer_executor = None              # lazy: _lazy_get_seer
        self.vericot_verifier = None           # lazy: _lazy_get_vericot
        self.vericot_reflector = None
        self.crv_integration = None            # lazy: _lazy_get_crv
        self.comet_memory = None               # lazy: _lazy_get_comet
        self.token_sparse_attn = None          # lazy: _lazy_get_long_context
        self.mhc_lite = None
        self.reform_processor = None
        self.oomb_processor = None
        self.reasoner = None                   # lazy: _lazy_get_reasoner
        self.agentic = None                    # lazy: _lazy_get_agentic
        self.speculative_decoder = None        # lazy: _lazy_get_speculative
        self.speculative_config = None
        self._lazy_initialized['rca'] = False
        self._lazy_initialized['seer'] = False
        self._lazy_initialized['vericot'] = False
        self._lazy_initialized['crv'] = False
        self._lazy_initialized['comet'] = False
        self._lazy_initialized['long_context'] = False
        self._lazy_initialized['reasoner'] = False
        self._lazy_initialized['agentic'] = False
        self._lazy_initialized['speculative'] = False

        _LOG.debug("YvModel: initializing output heads...")
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device=device, dtype=dtype)
        self.task_head = None     # lazy: _lazy_get_task_head
        self.eval_head = None     # lazy: _lazy_get_eval_head
        self._lazy_initialized['task_head'] = False
        self._lazy_initialized['eval_head'] = False

        self.num_mtp_heads = int(getattr(cfg, 'num_mtp_heads', 4))
        self.mtp_loss_weight = float(getattr(cfg, 'mtp_loss_weight', 0.5))
        self.mtp_share_embeddings = bool(getattr(cfg, 'mtp_share_embeddings', True))
        self.mtp_heads = None     # lazy: _lazy_get_mtp_heads
        self._lazy_initialized['mtp_heads'] = False

        self.modal_token_count = getattr(cfg, 'modal_token_count', 8)
        self.fusion_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False, device=device, dtype=dtype)

        if lora_config is not None:
            raise RuntimeError(
                "LoRA injection via lora_config is no longer supported internally. "
                "Wrap the model externally with get_peft_model(): "
                "model = get_peft_model(model, lora_config)"
            )

        if getattr(cfg, 'depth_aware_init', True):
            from ..core.norms import _depth_aware_init_weights
            self.apply(lambda m: _depth_aware_init_weights(m, cfg.n_layer, cfg.hidden_size))

        initializer_range = getattr(cfg, 'initializer_range', 0.02)
        use_scaled_init = getattr(cfg, 'use_scaled_init', True)
        if use_scaled_init and initializer_range != 0.02:
            scaled_std = initializer_range / math.sqrt(cfg.n_layer)
            for module in self.modules():
                if isinstance(module, (nn.LayerNorm, nn.Embedding)):
                    continue
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dim() >= 2:
                        nn.init.normal_(module.weight, mean=0, std=scaled_std)

        if getattr(cfg, 'tie_word_embeddings', False):
            if hasattr(self, 'lm_head') and hasattr(self, 'embed'):
                self.lm_head.weight = self.embed.weight

        # Causal mask cache (persistent=False — not saved in state_dict)
        self.register_buffer(
            '_causal_mask_cache',
            torch.zeros(0, 0, device=device, dtype=dtype),
            persistent=False,
        )

        total_params = sum(p.numel() for p in self.parameters())
        _LOG.debug(f"YvModel: total parameters = {total_params/1e6:.2f}M")
        _LOG.debug("YvModel: __init__ end")

    def _create_rotary_embedding(self, cfg, device, dtype):
        """Create rotary position embedding (RoPE) buffers.
        
        Initializes the inverse frequency buffer used for computing
        rotary position embeddings. RoPE enables relative position
        encoding through rotation matrices.
        
        Mathematical Formulation:
            inv_freq[i] = 1 / (base^(2i/d))
            where d is the head dimension and base is the rope_theta.
        
        Args:
            cfg: Configuration containing rope_theta and max_position_embeddings.
            device: Device to place the buffer on.
            dtype: Data type for the buffer.
            
        Returns:
            None (buffer is registered directly to the module).
        
        Note:
            The actual rotation is computed in the attention layer.
            This only creates the frequency basis.
        """
        dim = cfg.hidden_size // getattr(cfg, 'n_head', 1)
        max_seq_len = getattr(cfg, 'max_position_embeddings', 4096)
        base = getattr(cfg, 'rope_theta', 10000.0)

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def _get_causal_mask(
        self,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a cached causal (upper-triangular) mask, extending on demand.

        Each sequence length is cached in ``_causal_mask_cache`` so that
        repeated forward passes with the same length avoid re-allocating
        and re-filling the :math:`T \\times T` matrix.
        """
        cache_key = (str(device), dtype)
        cached = self._causal_mask_cache
        cached_view = self._causal_mask_views.get(cache_key)

        if cached_view is not None and cached_view.shape[-1] >= seq_len:
            return cached_view[:seq_len, :seq_len]

        if cached.shape[-1] >= seq_len:
            if cached.device == device and cached.dtype == dtype:
                self._causal_mask_views[cache_key] = cached
                return cached[:seq_len, :seq_len]
            converted = cached.to(dtype=dtype, device=device)
            self._causal_mask_views[cache_key] = converted
            return converted[:seq_len, :seq_len]
        # Extend cache to the next power-of-two to amortise growth
        new_len = 1
        while new_len < seq_len:
            new_len <<= 1
        new_mask = torch.triu(
            torch.full((new_len, new_len), float('-inf'), device=cached.device, dtype=cached.dtype),
            diagonal=1,
        )
        self._causal_mask_cache = new_mask
        self._causal_mask_views.clear()
        if new_mask.device == device and new_mask.dtype == dtype:
            self._causal_mask_views[cache_key] = new_mask
            return new_mask[:seq_len, :seq_len]
        converted = new_mask.to(dtype=dtype, device=device)
        self._causal_mask_views[cache_key] = converted
        return converted[:seq_len, :seq_len]

    def _apply_backbone_defaults(self) -> None:
        """Harden the flagship backbone configuration onto one main sequence lane."""
        if getattr(self.config, 'max_position_embeddings', 0) >= 1_048_576:
            setattr(self.config, 'use_h2o_attention', True)

        if not getattr(self.config, 'backbone_allow_legacy_blocks', False):
            setattr(self.config, 'use_mamba3', True)
            # Only auto-populate mamba3_layers if user hasn't explicitly set them.
            # __post_init__ already sets mamba3_layers = list(range(n_layer)) when
            # use_mamba3=True and mamba3_layers is empty, so by the time we reach
            # here the list is already populated unless the user explicitly emptied it.
            current = getattr(self.config, 'mamba3_layers', [])
            if not current:
                setattr(self.config, 'mamba3_layers', list(range(getattr(self.config, 'n_layer', 0))))

        if getattr(self.config, 'use_oomb_context', False):
            setattr(self.config, 'use_h2o_attention', True)
            setattr(self.config, 'cache_type', getattr(self.config, 'cache_type', 'hybrid') or 'hybrid')

    def _convert_cache_precision(
        self,
        cache_pair: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
        cache_dtype: torch.dtype,
        use_mixed_precision_cache: bool,
    ) -> Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Convert KV cache tensors to the target cache dtype with optional rope preservation."""
        if cache_pair is None:
            return None

        key_states, value_states = cache_pair
        if not use_mixed_precision_cache:
            return tuple(
                tensor.to(cache_dtype) if tensor is not None else None
                for tensor in (key_states, value_states)
            )

        converted = []
        for tensor in (key_states, value_states):
            if tensor is None:
                converted.append(None)
                continue
            head_dim = tensor.shape[-1]
            partial_rope_dim = min(128, head_dim // 4)
            rope_part = tensor[..., :partial_rope_dim].to(self._dtype)
            non_rope_part = tensor[..., partial_rope_dim:].to(cache_dtype)
            converted.append(torch.cat([rope_part, non_rope_part], dim=-1))
        return tuple(converted)

    def _build_backbone_block(
        self,
        cfg,
        layer_idx: int,
        device,
        dtype,
        path_moe_stage_gate,
    ):
        """Build one backbone block while keeping the flagship path as default."""
        allow_legacy_blocks = bool(getattr(cfg, 'backbone_allow_legacy_blocks', False))
        use_hybrid = bool(getattr(cfg, 'use_mamba3', False))

        if not allow_legacy_blocks:
            # The block type is constant across the whole backbone, so
            # logging it once at the first layer is enough. Subsequent
            # calls are silent.
            if layer_idx == 0:
                _LOG.debug(
                    f"YvModel: backbone uses YvHybridBlock for all "
                    f"{getattr(cfg, 'n_layer', '?')} layers"
                )
            return YvHybridBlock(
                cfg,
                device=device,
                dtype=dtype,
                quantization_config=self.quantization_config,
            )

        if use_hybrid:
            mamba3_layers = getattr(cfg, 'mamba3_layers', [])
            if not mamba3_layers or layer_idx in mamba3_layers:
                if layer_idx == 0:
                    _LOG.debug(
                        f"YvModel: backbone uses YvHybridBlock for hybrid "
                        f"layers (mamba3_layers={mamba3_layers})"
                    )
                return YvHybridBlock(
                    cfg,
                    device=device,
                    dtype=dtype,
                    quantization_config=self.quantization_config,
                )

        return YvTransformerBlock(
            cfg,
            device=device,
            dtype=dtype,
            quantization_config=self.quantization_config,
            gate=path_moe_stage_gate,
        )

    def _init_long_context_stack(self, cfg, device, dtype) -> None:
        """Initialize the long-context refinement stack on one shared post-backbone lane."""
        self.reform_processor = None
        self.oomb_processor = None
        self.token_sparse_attn = None
        self.mhc_lite = None

        if getattr(cfg, 'use_reform', False):
            self.reform_processor = YvREFORM(
                compression_ratio=getattr(cfg, 'reform_compression_ratio', 4),
                importance_threshold=getattr(cfg, 'reform_importance_threshold', 0.1),
            )

        if getattr(cfg, 'use_oomb_context', True):
            _LOG.debug("YvModel: initializing OOMB processor...")
            self.oomb_processor = YvOOMBContext(
                chunk_size=getattr(cfg, 'oomb_chunk_size', 32768),
                max_context_length=getattr(cfg, 'max_position_embeddings', 4194304)
            )

        if getattr(cfg, 'use_token_sparse_attn', False) or getattr(cfg, 'use_tactic', False):
            _LOG.debug("YvModel: initializing Token Sparse Attention / Tactic...")
            self.token_sparse_attn = YvTokenSparseAttention(cfg, device=device, dtype=dtype)

        if getattr(cfg, 'use_mhc_lite', False):
            _LOG.debug("YvModel: initializing mHC-lite...")
            self.mhc_lite = YvMHCLiteHyperConnection(
                num_streams=getattr(cfg, 'mhc_streams', 4),
                num_permutations=getattr(cfg, 'mhc_permutations', 8),
                device=device, dtype=dtype,
            )

    def _apply_long_context_refinement(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the shared long-context refinement stack on the main sequence path.

        This keeps token sparsification, hyper-connections, and REFORM-style
        importance refinement on one post-backbone lane instead of scattering
        the logic across multiple forward branches.
        """
        refined = hidden_states

        if self.token_sparse_attn is not None:
            refined = refined + self.token_sparse_attn(refined, attention_mask)

        if self.mhc_lite is not None:
            streams = refined.unsqueeze(1).expand(-1, self.mhc_lite.num_streams, -1)
            refined = refined + self.mhc_lite(streams, refined.mean(dim=1)).mean(dim=1)

        if self.reform_processor is not None:
            importance = refined.norm(dim=-1)
            important_mask = importance > importance.mean(dim=-1, keepdim=True)
            if important_mask.any():
                refined = refined + 0.01 * refined * important_mask.unsqueeze(-1).float()

        return refined

    def _should_use_long_context_path(self, sequence_length: int) -> bool:
        """Return whether the current sequence should use the long-context lane."""
        return (
            self.oomb_processor is not None
            and sequence_length > self.oomb_processor.chunk_size
        )

    def save_pretrained(self, save_directory: str):
        """Save model in HF-compatible format for training engine checkpointing."""
        import os, json
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        cfg_dict = {
            "architectures": ["YvModelForCausalLM"],
            "model_type": "yv_model",
            "hidden_size": self.cfg.hidden_size,
            "num_hidden_layers": self.cfg.n_layer,
            "vocab_size": self.cfg.vocab_size,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(cfg_dict, f)

    def set_gradient_checkpointing(self, enabled: bool = True):
        """Enable or disable gradient checkpointing for memory efficiency.
        
        Gradient checkpointing trades compute for memory by not storing
        intermediate activations during forward pass, recomputing them
        during backward pass.
        
        Args:
            enabled: Whether to enable gradient checkpointing. Default: True.
        
        Memory Impact:
            - Enabled: ~30-50% memory reduction
            - Disabled: Faster training, more memory
        
        Note:
            This affects all layers uniformly. For selective checkpointing,
            modify individual layer.use_checkpoint attributes.
        """
        for layer in self.layers:
            layer.use_checkpoint = enabled
    
    def _lazy_get_vision_encoder(self):
        if self.vision is None:
            base_vision = YvVisionEncoder(self.cfg, device=self._device, dtype=self._dtype)
            self.vision = YvMoVEVisionEncoder(self.cfg, base_encoder=base_vision, device=self._device, dtype=self._dtype) if getattr(self.cfg, 'use_move_encoder', False) else base_vision
            self._lazy_initialized['vision'] = True
        return self.vision

    def _lazy_get_audio_encoder(self):
        if self.audio is None:
            self.audio = YvAudioEncoder(self.cfg, device=self._device, dtype=self._dtype)
            self._lazy_initialized['audio'] = True
        return self.audio

    def _lazy_get_video_encoder(self):
        if self.video is None:
            self.video = YvVideoEncoder(self.cfg, device=self._device, dtype=self._dtype)
            self._lazy_initialized['video'] = True
        return self.video

    def _lazy_get_doc_encoder(self):
        if self.doc is None:
            self.doc = YvDocEncoder(self.cfg, device=self._device, dtype=self._dtype)
            self._lazy_initialized['doc'] = True
        return self.doc
    
    def is_lazy_initialized(self, component: str) -> bool:
        """Check if a component has been lazy initialized.

        Args:
            component: Component name ('vision', 'audio', 'video', 'doc', 'rca',
                      'seer', 'vericot', 'crv', 'comet', 'long_context',
                      'reasoner', 'agentic', 'speculative', 'task_head',
                      'eval_head', 'mtp_heads')

        Returns:
            True if the component has been initialized, False otherwise.
        """
        return self._lazy_initialized.get(component, True)

    def _lazy_get_reasoner(self) -> None:
        if self.reasoner is None:
            from ..reasoning import YvUnifiedReasoner
            _LOG.debug("YvModel: lazy-init reasoner...")
            self.reasoner = YvUnifiedReasoner(self.cfg, device=self._device, dtype=self._dtype)
            self.reasoner.initialize_reasoning_tokens(None)
            self._lazy_initialized['reasoner'] = True

    def _lazy_get_agentic(self) -> None:
        if self.agentic is None:
            from ..multimodal import YvAgentic
            _LOG.debug("YvModel: lazy-init agentic...")
            self.agentic = YvAgentic(self.cfg, model=self)
            self._lazy_initialized['agentic'] = True

    def _lazy_get_speculative(self) -> None:
        if self.speculative_decoder is None:
            if bool(getattr(self.cfg, 'use_speculative_decoder', False)) and bool(getattr(self.cfg, 'enable_speculative_decoding', True)):
                _LOG.debug("YvModel: lazy-init speculative decoder...")
                from ..generation.speculative import YvAdaptiveSpeculativeDecoder, YvSpeculativeConfig
                self.speculative_config = YvSpeculativeConfig(
                    num_candidates=getattr(self.cfg, 'speculative_candidates', 4),
                    draft_length=getattr(self.cfg, 'speculative_draft_length', 5),
                    acceptance_threshold=getattr(self.cfg, 'speculative_acceptance_threshold', 0.8),
                    temperature=getattr(self.cfg, 'speculative_temperature', 0.7),
                    top_k=getattr(self.cfg, 'speculative_top_k', 50),
                    top_p=getattr(self.cfg, 'speculative_top_p', 0.9),
                    tree_width=getattr(self.cfg, 'speculative_tree_width', 4),
                    tree_depth=getattr(self.cfg, 'speculative_tree_depth', 5)
                )
                self.speculative_decoder = YvAdaptiveSpeculativeDecoder(self.speculative_config, self, None)
                self._lazy_initialized['speculative'] = True

    def _lazy_get_rca(self) -> None:
        if self.deep_cross_layer_injector is None and getattr(self.cfg, 'use_rca_fusion', True):
            from ..multimodal.rca_fusion import YvDeepCrossLayerInjector
            _LOG.debug("YvModel: lazy-init RCA cross-layer injector...")
            self.deep_cross_layer_injector = YvDeepCrossLayerInjector(
                self.cfg, num_layers=self.cfg.n_layer, device=self._device, dtype=self._dtype
            )
            self._lazy_initialized['rca'] = True

    def _lazy_get_seer(self) -> None:
        if self.seer_executor is None and bool(getattr(self.cfg, 'use_seer_executor', False)):
            from ..multimodal.seer_executor import YvSEERExecutor
            _LOG.debug("YvModel: lazy-init SEER executor...")
            self.seer_executor = YvSEERExecutor(self.cfg)
            self._lazy_initialized['seer'] = True

    def _lazy_get_vericot(self) -> None:
        if self.vericot_verifier is None:
            if getattr(self.cfg, 'use_vericot', False) or getattr(self.cfg, 'use_spell', False):
                from ..reasoning.vericot import YvVeriCoTVerifier, YvVeriCoTReflector
                _LOG.debug("YvModel: lazy-init VeriCoT/SPELL...")
                self.vericot_verifier = YvVeriCoTVerifier(self.cfg, device=self._device, dtype=self._dtype)
                self.vericot_reflector = YvVeriCoTReflector(self.vericot_verifier)
                self._lazy_initialized['vericot'] = True

    def _lazy_get_crv(self) -> None:
        if self.crv_integration is None and getattr(self.cfg, 'use_crv_verification', True):
            from ..reasoning.verification import YvCRVIntegration
            _LOG.debug("YvModel: lazy-init CRV...")
            self.crv_integration = YvCRVIntegration(hidden_size=self.cfg.hidden_size)
            self._lazy_initialized['crv'] = True

    def _lazy_get_comet(self) -> None:
        if self.comet_memory is None:
            if getattr(self.cfg, 'use_comet_memory', False) or getattr(self.cfg, 'use_seirenes', False):
                from .comet import YvCoMeTMemory
                _LOG.debug("YvModel: lazy-init CoMeT/Seirênes...")
                self.comet_memory = YvCoMeTMemory(self.cfg, device=self._device, dtype=self._dtype)
                self._lazy_initialized['comet'] = True

    def _lazy_get_long_context(self) -> None:
        if not self._lazy_initialized.get('long_context', True):
            from .long_context import YvOOMBContext, YvREFORM
            from .token_sparse_attn import YvTokenSparseAttention
            from .mhc_lite import YvMHCLiteHyperConnection
            if getattr(self.cfg, 'use_reform', False):
                self.reform_processor = YvREFORM(
                    compression_ratio=getattr(self.cfg, 'reform_compression_ratio', 4),
                    importance_threshold=getattr(self.cfg, 'reform_importance_threshold', 0.1),
                )
            if getattr(self.cfg, 'use_oomb_context', True):
                self.oomb_processor = YvOOMBContext(
                    chunk_size=getattr(self.cfg, 'oomb_chunk_size', 32768),
                    max_context_length=getattr(self.cfg, 'max_position_embeddings', 4194304)
                )
            if getattr(self.cfg, 'use_token_sparse_attn', False) or getattr(self.cfg, 'use_tactic', False):
                self.token_sparse_attn = YvTokenSparseAttention(self.cfg, device=self._device, dtype=self._dtype)
            if getattr(self.cfg, 'use_mhc_lite', False):
                self.mhc_lite = YvMHCLiteHyperConnection(
                    num_streams=getattr(self.cfg, 'mhc_streams', 4),
                    num_permutations=getattr(self.cfg, 'mhc_permutations', 8),
                    device=self._device, dtype=self._dtype,
                )
            self._lazy_initialized['long_context'] = True

    def _lazy_get_task_head(self) -> None:
        if self.task_head is None:
            self.task_head = nn.Linear(
                self.cfg.hidden_size, self.cfg.task_classes,
                device=self._device, dtype=self._dtype
            )
            self._lazy_initialized['task_head'] = True

    def _lazy_get_eval_head(self) -> None:
        if self.eval_head is None:
            self.eval_head = nn.Linear(
                self.cfg.hidden_size, self.cfg.eval_dims,
                device=self._device, dtype=self._dtype
            )
            self._lazy_initialized['eval_head'] = True

    def _lazy_get_mtp_heads(self) -> None:
        if self.mtp_heads is None and self.num_mtp_heads > 0:
            self.mtp_heads = nn.ModuleList([
                nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False,
                          device=self._device, dtype=self._dtype)
                for _ in range(self.num_mtp_heads)
            ])
            if self.mtp_share_embeddings:
                for mtp_head in self.mtp_heads:
                    mtp_head.weight = self.lm_head.weight
            self._lazy_initialized['mtp_heads'] = True

    def ensure_all_modules(self) -> None:
        """Force-initialise all lazy modules so ``parameters()`` is complete.

        Call **once** after creating the optimiser so that every module's
        parameters are visible and the optimiser's param groups are fully
        populated.  After this call the model is fully materialised.
        """
        self._lazy_get_vision_encoder()
        self._lazy_get_audio_encoder()
        self._lazy_get_video_encoder()
        self._lazy_get_doc_encoder()
        self._lazy_get_reasoner()
        self._lazy_get_agentic()
        self._lazy_get_speculative()
        self._lazy_get_rca()
        self._lazy_get_seer()
        self._lazy_get_vericot()
        self._lazy_get_crv()
        self._lazy_get_comet()
        self._lazy_get_long_context()
        self._lazy_get_task_head()
        self._lazy_get_eval_head()
        self._lazy_get_mtp_heads()

    def to(self, device=None, dtype=None, non_blocking=False):
        if device is None and dtype is None:
            return super().to(device, dtype, non_blocking)

        modules = [
            self.embed,
            self.layers,
            self.norm,
            self.lm_head,
            self.task_head,
            self.eval_head,
            self.fusion_proj,
            getattr(self, 'modal_fusion', None),
            getattr(self, 'reasoner', None),
            getattr(self, 'mm_reasoning_enhancer', None),
            getattr(self, 'cache_manager', None),
            getattr(self, 'sparse_cut_router', None),
        ]

        for m in modules:
            if m is None:
                continue
            if isinstance(m, nn.ModuleList):
                for sub in m:
                    sub.to(device=device, dtype=dtype, non_blocking=non_blocking)
            elif isinstance(m, nn.Module):
                m.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return self

    def resize_token_embeddings(self, new_num_tokens):
        """Resize the token embedding matrix to a new vocabulary size.
        
        Useful when adding new tokens to the vocabulary (e.g., special tokens,
        domain-specific tokens). Handles both embedding and LM head resizing.
        
        Args:
            new_num_tokens: New vocabulary size.
            
        Side Effects:
            - Replaces self.embed with new embedding layer
            - Replaces self.lm_head with new output layer
            - Updates cfg.vocab_size
            - Reinitializes reasoner tokens
        
        Note:
            Existing embeddings are copied to the new layers.
            New tokens are initialized randomly.
            Remember to update special token IDs after resizing.
        """
        old_embed = self.embed
        new_embed = nn.Embedding(
            new_num_tokens,
            self.cfg.hidden_size,
            device=old_embed.weight.device,
            dtype=old_embed.weight.dtype
        )
        num_to_copy = min(old_embed.num_embeddings, new_num_tokens)
        new_embed.weight.data[:num_to_copy, :] = old_embed.weight.data[:num_to_copy, :]
        self.embed = new_embed

        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(
            self.cfg.hidden_size,
            new_num_tokens,
            bias=False,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype
        )
        new_lm_head.weight.data[:num_to_copy, :] = old_lm_head.weight.data[:num_to_copy, :]
        self.lm_head = new_lm_head

        if getattr(self.cfg, 'tie_word_embeddings', False):
            self.lm_head.weight = self.embed.weight

        if hasattr(self.reasoner, 'resize_vocab'):
            self.reasoner.resize_vocab(new_num_tokens)
        self.cfg.vocab_size = new_num_tokens

        self.reasoner.initialize_reasoning_tokens(None)
        _LOG.info(
            f"Resized token embeddings to {new_num_tokens}. "
            f"Remember to update special token IDs in the reasoner."
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        **kwargs
    ):
        """Prepare inputs for autoregressive generation.
        
        Constructs the input dictionary for the forward pass during generation,
        handling attention masks, position IDs, and KV cache.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len]. If None, creates
                an all-ones mask.
            position_ids: Position IDs [batch, seq_len]. If None, computed
                from attention mask cumsum.
            past_key_values: Cached key-value states from previous steps.
            use_cache: Whether to use KV caching. Default: True.
            **kwargs: Additional arguments to pass to forward.
            
        Returns:
            Dict containing:
                - input_ids: Input token IDs
                - attention_mask: Attention mask
                - position_ids: Position IDs
                - past_key_values: KV cache (if provided)
                - use_cache: Cache flag
        
        Note:
            Position IDs are computed as cumulative sum of attention mask,
            ensuring correct positions for padding-aware generation.
        """
        model_inputs = {"input_ids": input_ids}

        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape,
                dtype=torch.long,
                device=input_ids.device
            )
        model_inputs["attention_mask"] = attention_mask

        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        model_inputs["position_ids"] = position_ids

        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

        model_inputs["use_cache"] = use_cache
        model_inputs.update(kwargs)
        return model_inputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        use_speculative: bool = True,
        mode: str = 'auto',
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate text autoregressively with optional speculative decoding.
        
        Main entry point for text generation. Supports multiple generation
        modes and speculative decoding for improved efficiency.
        
        Generation Modes:
            - fast: Standard generation without reasoning enhancement.
                Quick responses for simple queries.
            - thinking: Enhanced generation with chain-of-thought.
                Better for complex reasoning tasks.
            - auto: Automatic mode selection based on input characteristics.
                Uses thinking mode for long sequences or high diversity.
        
        Speculative Decoding:
            When enabled, uses a draft-then-verify paradigm:
            1. Draft model generates candidate tokens
            2. Target model verifies candidates in parallel
            3. Accept valid tokens, reject and resample invalid ones
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            max_length: Maximum total sequence length. Default: from config.
            temperature: Sampling temperature. Higher = more random. Default: from config.
            top_k: Top-k sampling vocabulary size. Default: from config.
            top_p: Nucleus sampling cumulative probability. Default: from config.
            use_speculative: Whether to use speculative decoding. Default: True.
            mode: Generation mode ('fast', 'thinking', 'auto'). Default: 'auto'.
            **kwargs: Additional generation parameters.
            
        Returns:
            Tuple of:
                - generated_ids: Generated token IDs [batch, new_seq_len]
                - stats: Dictionary with generation statistics
                    - routing: Generation mode used
                    - total_draft_tokens: Draft tokens generated (speculative)
                    - accepted_tokens: Accepted draft tokens (speculative)
                    - rejected_tokens: Rejected draft tokens (speculative)
                    - draft_acceptance_rate: Acceptance rate (speculative)
                    - speedup: Speedup factor (speculative)
        
        Example:
            >>> generated, stats = model.generate(
            ...     input_ids=prompt_ids,
            ...     max_length=256,
            ...     temperature=0.8,
            ...     mode='thinking'
            ... )
            >>> print(f"Generated {generated.shape[1]} tokens")
            >>> print(f"Speedup: {stats['speedup']:.2f}x")
        """
        max_length = max_length or getattr(self.cfg, 'generation_max_tokens', 100)
        temperature = temperature if temperature is not None else getattr(self.cfg, 'generation_temperature', 0.7)
        top_k = top_k if top_k is not None else getattr(self.cfg, 'generation_top_k', 50)
        top_p = top_p if top_p is not None else getattr(self.cfg, 'generation_top_p', 0.9)
        
        routing = 'fast'
        if mode == 'thinking':
            routing = 'thinking'
        elif mode == 'auto':
            seq_len = input_ids.shape[1]
            if seq_len > 256 or top_k >= 50 or top_p >= 0.9:
                routing = 'thinking'
        else:
            routing = 'fast'

        use_speculative_final = use_speculative
        temperature_final = temperature
        top_k_final = top_k
        top_p_final = top_p

        if routing == 'thinking':
            use_speculative_final = True
            temperature_final = max(0.6, temperature * 0.9)
            top_k_final = max(50, top_k)
            top_p_final = max(0.9, top_p)

        if use_speculative_final and hasattr(self, 'speculative_decoder'):
            self._lazy_get_speculative()
            if self.speculative_decoder is not None:
                self.speculative_config.temperature = temperature_final
                self.speculative_config.top_k = top_k_final
                self.speculative_config.top_p = top_p_final
                out_ids, stats = self.speculative_decoder.speculative_generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    cache_manager=self.cache_manager if hasattr(self, 'cache_manager') else None,
                    **kwargs
                )
                stats['routing'] = routing
                return out_ids, stats

        out_ids, stats = self._standard_generate(
                input_ids,
                attention_mask,
                max_length,
                temperature_final,
                top_k_final,
                top_p_final,
                **kwargs
            )
        stats['routing'] = routing
        return out_ids, stats

    def _standard_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Standard autoregressive generation without speculative decoding.
        
        Implements the basic generation loop with top-k and top-p sampling.
        Used as fallback when speculative decoding is disabled or unavailable.
        
        Sampling Process:
            1. Forward pass to get logits
            2. Apply temperature scaling
            3. Apply top-k filtering (keep top k logits)
            4. Apply top-p filtering (nucleus sampling)
            5. Sample from the filtered distribution
            6. Append sampled token to sequence
            7. Repeat until max_length or EOS
        
        Adaptive MoE:
            Supports adaptive temperature adjustment for MoE layers
            to encourage exploration during generation.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            max_length: Maximum total sequence length. Default: 100.
            temperature: Sampling temperature. Default: 0.7.
            top_k: Top-k sampling vocabulary size. Default: 50.
            top_p: Nucleus sampling cumulative probability. Default: 0.9.
            **kwargs: Additional generation parameters.
                - adaptive_moe: Dict with 'enabled', 'temp_step', 'interval', 'temp_cap'
                - min_new_tokens: Minimum new tokens before allowing EOS
                
        Returns:
            Tuple of:
                - generated_ids: Generated token IDs [batch, new_seq_len]
                - stats: Dictionary with generation statistics
        
        Note:
            This method runs in torch.no_grad() context for efficiency.
            EOS token ID is taken from cfg.eos_token_id (default: 2).
        """
        device = input_ids.device
        generated_ids = input_ids.clone()
        stats = {
            'total_draft_tokens': 0,
            'accepted_tokens': 0,
            'rejected_tokens': 0,
            'draft_acceptance_rate': 0.0,
            'speedup': 1.0,
            'method': 'standard'
        }

        adaptive_cfg = kwargs.pop('adaptive_moe', None)
        adaptive_enabled = bool(adaptive_cfg and adaptive_cfg.get('enabled', False))
        adaptive_step = float(adaptive_cfg.get('temp_step', 0.03)) if adaptive_enabled else 0.0
        adaptive_interval = int(adaptive_cfg.get('interval', 16)) if adaptive_enabled else 0
        adaptive_cap = float(adaptive_cfg.get('temp_cap', 1.30)) if adaptive_enabled else 0.0

        min_new_tokens = int(kwargs.pop('min_new_tokens', 0)) if 'min_new_tokens' in kwargs else 0
        new_tokens_generated = 0
        self._finished = None

        def _bump_gate_temperature(_model, delta: float, cap: float):
            """Adjust gate temperature for MoE layers during generation.
            
            Increases the temperature of gating networks to encourage
            expert diversity during generation.
            
            Args:
                _model: The model to adjust.
                delta: Temperature increment.
                cap: Maximum temperature cap.
            """
            for m in _model.modules():
                if hasattr(m, 'temperature'):
                    if isinstance(m.temperature, torch.Tensor):
                        cur = m.temperature.detach().float().mean().item()
                        newv = min(cap, cur + delta)
                        m.temperature.fill_(newv)
                    else:
                        cur = float(getattr(m, 'temperature', 1.0))
                        newv = min(cap, cur + delta)
                        setattr(m, 'temperature', newv)

        eos_token_id = getattr(self.cfg, 'eos_token_id', 2)

        with torch.no_grad():
            remaining_steps = max_length - input_ids.shape[1]
            if remaining_steps <= 0:
                return input_ids
            for step_idx in range(remaining_steps):
                model_inputs = self.prepare_inputs_for_generation(
                    generated_ids,
                    attention_mask,
                    **kwargs
                )

                model_inputs.pop("attention_mask", None)
                model_inputs.pop('adaptive_moe', None)

                outputs = self(**model_inputs)
                logits = outputs.get('logits', outputs) if isinstance(outputs, dict) else outputs

                next_token_logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(
                        next_token_logits,
                        min(top_k, next_token_logits.size(-1))
                    )
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_logits)

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1,
                        sorted_indices,
                        sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                new_tokens_generated += 1

                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            device=device,
                            dtype=attention_mask.dtype
                        )
                    ], dim=-1)

                eos_mask = (next_token == eos_token_id).squeeze(-1)
                if eos_mask.any() and new_tokens_generated >= min_new_tokens:
                    if not hasattr(self, '_finished') or self._finished is None:
                        self._finished = eos_mask.clone()
                    else:
                        self._finished = self._finished | eos_mask
                    if self._finished.all():
                        break

                if adaptive_enabled and adaptive_interval > 0 and ((step_idx + 1) % adaptive_interval == 0):
                    _bump_gate_temperature(self, adaptive_step, adaptive_cap)

        return generated_ids, stats

    def forward(
        self,
        input_ids,
        images=None,
        audio=None,
        video=None,
        docs=None,
        labels=None,
        agent_mode=False,
        task=None,
        max_steps=None,
        agent_obs=None,
        agent_embed=None,
        past_key_values=None,
        use_cache=False,
        attention_mask=None,
        position_ids=None,
        **kwargs
    ):
        """Forward pass through the Yv model.
        
        Processes inputs through the model, supporting multimodal inputs,
        agent mode, and various output configurations.
        
        Processing Pipeline:
            1. Agent Mode Check: If agent_mode=True, delegate to agentic module
            2. Text Embedding: Convert input_ids to embeddings
            3. Multimodal Encoding: Process images, audio, video, docs
            4. Modal Fusion: Combine multimodal features with text
            5. Layer Processing: Pass through transformer/hybrid layers
            6. Output Generation: Compute logits, losses, and auxiliary outputs
        
        Multimodal Support:
            - images: Vision encoder processes image tensors
            - audio: Audio encoder processes spectrograms
            - video: Video encoder processes video frames
            - docs: Document encoder processes document images
            - agent_embed/agent_obs: Agentic encoder for RL inputs
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            images: Image inputs [batch, channels, height, width].
            audio: Audio inputs [batch, channels, time] or spectrograms.
            video: Video inputs [batch, frames, channels, height, width].
            docs: Document images [batch, channels, height, width].
            labels: Target labels for loss computation [batch, seq_len].
            agent_mode: Whether to run in agent mode. Default: False.
            task: Task specification for agent mode.
            max_steps: Maximum steps for agent mode.
            agent_obs: Agent observations for agentic encoder.
            agent_embed: Pre-computed agent embeddings.
            past_key_values: Cached key-value states for incremental decoding.
            use_cache: Whether to use KV caching. Default: False.
            attention_mask: Attention mask [batch, seq_len].
            position_ids: Position IDs [batch, seq_len].
            **kwargs: Additional arguments.
            
        Returns:
            Dict containing:
                - logits: Language model logits [batch, seq_len, vocab_size]
                - loss: Total loss (if labels provided)
                - mtp_logits: Multi-token prediction logits
                - mtp_loss: MTP auxiliary loss
                - task_logits: Task classification logits
                - eval_score: Evaluation scores
                - aux_loss: Auxiliary losses (MoE routing, etc.)
                - reasoner_out: Reasoning module outputs
                - tool_output: External tool execution result (if used)
                - tool_experience: Retrieved execution experience/memory (if used)
                - vericot_out: Verification output from reasoning checker (if enabled)
                - past_key_values: Updated KV cache (if use_cache=True)
                - cache_stats: Cache statistics (if available)
        
        Example:
            >>> # Text-only forward pass
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs["loss"]
            >>> logits = outputs["logits"]
            >>> 
            >>> # Multimodal forward pass
            >>> outputs = model(
            ...     input_ids=input_ids,
            ...     images=images,
            ...     audio=audio
            ... )
            >>> 
            >>> # Agent mode
            >>> outputs = model(
            ...     input_ids=input_ids,
            ...     agent_mode=True,
            ...     task="summarize"
            ... )
        
        Note:
            For generation, use the generate() method instead.
            Agent mode bypasses standard forward pass.
            Cache is automatically quantized for long sequences.
        """
        import torch.utils.checkpoint as cp

        return_task_logits = bool(kwargs.pop("return_task_logits", True))
        return_eval_score = bool(kwargs.pop("return_eval_score", True))
        return_tool_outputs = bool(kwargs.pop("return_tool_outputs", True))
        return_reasoner_outputs = bool(kwargs.pop("return_reasoner_outputs", True))
        return_verifier_outputs = bool(kwargs.pop("return_verifier_outputs", return_reasoner_outputs))
        return_mtp_logits = bool(kwargs.pop("return_mtp_logits", True))

        if agent_mode:
            self._lazy_get_agentic()
            return self.agentic.run(
                input_ids=input_ids,
                images=images,
                audio=audio,
                video=video,
                task=task,
                max_steps=max_steps
            )

        b, t = input_ids.shape

        # Runtime VRAM check — auto-adjust if memory pressure detected
        if hasattr(self, '_vram_monitor'):
            self._vram_monitor.check_and_adjust()

        text_emb = self.embed(input_ids)
        modal_features = {'text': text_emb}

        if images is not None and getattr(self, 'vision', None) is not None:
            img_out = self.vision(images)
            if self.sparse_cut_router is not None:
                img_out = self.sparse_cut_router(img_out)
            modal_features['image'] = (
                img_out['features'] if isinstance(img_out, dict) and 'features' in img_out else img_out
            )

        if audio is not None and getattr(self, 'audio', None) is not None:
            aud_out = self.audio(audio)
            modal_features['audio'] = (
                aud_out['features'] if isinstance(aud_out, dict) and 'features' in aud_out else aud_out
            )

        if video is not None and getattr(self, 'video', None) is not None:
            vid_out = self.video(video)
            modal_features['video'] = (
                vid_out['features'] if isinstance(vid_out, dict) and 'features' in vid_out else vid_out
            )

        if docs is not None and getattr(self, 'doc', None) is not None:
            doc_out = self.doc(docs)
            doc_features = (
                doc_out['features'] if isinstance(doc_out, dict) and 'features' in doc_out else doc_out
            )
            modal_features['doc'] = doc_features
            modal_features['document'] = doc_features

        if agent_embed is not None:
            agent_input = {
                'observations': agent_embed.get('observations', []),
                'actions': agent_embed.get('actions', []),
                'reflections': agent_embed.get('reflections', []),
                'current_state': agent_embed.get('current_state', None),
                'task_context': agent_embed.get('task_context', None)
            }
            modal_features['agentic'] = self.agent_encoder(agent_input)

        if agent_obs is not None:
            agent_obs_input = {
                'observations': agent_obs.get('observations', []),
                'actions': agent_obs.get('actions', []),
                'reflections': agent_obs.get('reflections', []),
                'current_state': agent_obs.get('current_state', None),
                'task_context': agent_obs.get('task_context', None)
            }
            agent_feat = self.agent_encoder(agent_obs_input)
            modal_features['agentic'] = agent_feat

        # === Unified Multimodal Fusion ===
        fused_features = None
        rca_output = None
        if len(modal_features) > 1:
            fused_features = self.modal_fusion(modal_features)
            if fused_features is None:
                raise ValueError("Multimodal fusion returned None while non-text modalities were present.")
            if fused_features.dim() == 3:
                if fused_features.dtype != text_emb.dtype:
                    fused_features = fused_features.to(text_emb.dtype)
                if fused_features.device != text_emb.device:
                    fused_features = fused_features.to(text_emb.device)
                x = torch.cat([fused_features, text_emb], dim=1)
            elif fused_features.dim() == 2:
                B, H = fused_features.shape
                ff = fused_features.to(device=text_emb.device, dtype=text_emb.dtype)
                proj = self.fusion_proj(ff)
                tokens = proj.unsqueeze(1).expand(B, self.modal_token_count, H).contiguous()
                x = torch.cat([tokens, text_emb], dim=1)
            else:
                x = text_emb
            # Store aligned per-modality features for deep layer injection
            rca_output = getattr(self.modal_fusion, '_last_modality_features', None) or None
        else:
            x = text_emb

        t = x.shape[1]
        lm_seq_len = x.shape[1]

        modal_id = None
        if getattr(self.cfg, 'modal_aware_routing', True) or getattr(self.cfg, 'modal_protection_mod', True):
            n_modalities = getattr(self.cfg, 'n_modalities', 7)
            modal_id = torch.zeros(b, t, dtype=torch.long, device=x.device)
            if len(modal_features) > 1:
                modal_token_count = getattr(self, 'modal_token_count', 8)
                if fused_features is not None:
                    actual_modal_tokens = min(modal_token_count, t)
                    modal_id[:, :actual_modal_tokens] = n_modalities - 1
        
        causal_mask = self._get_causal_mask(t, x.dtype, x.device)
        mask = causal_mask
        if attention_mask is not None:
            attention_mask_bool = attention_mask.to(device=x.device, dtype=torch.bool)
            if not bool(attention_mask_bool.all()):
                ext_mask = attention_mask_bool[:, None, :]  # [B, 1, T]
                mask = causal_mask.unsqueeze(0).expand(b, -1, -1)
                mask = mask.masked_fill(~ext_mask, float('-inf'))

        total_aux_loss = x.new_zeros(())
        chunk_size = min(getattr(self.cfg, 'max_position_embeddings', 2048), 8192)
        outputs = []

        # OOMB: Out-of-Order Memory Banking for million-token context
        if hasattr(self, 'oomb_processor') and self.oomb_processor is not None:
            chunk_size = min(chunk_size, self.oomb_processor.chunk_size)

        if use_cache:
            seq_len = x.shape[1]
            if seq_len > 4096:
                cache_dtype = torch.float16
                cache_quant_bits = 8
            elif seq_len > 1024:
                cache_dtype = torch.float16
                cache_quant_bits = 8
            else:
                cache_dtype = torch.bfloat16
                cache_quant_bits = 16
        else:
            cache_dtype = torch.bfloat16
            cache_quant_bits = 16

        use_mixed_precision_cache = getattr(self.cfg, 'use_mixed_precision_cache', True)

        next_cache = [] if use_cache else None
        rca_fused = None
        if isinstance(rca_output, dict) and rca_output:
            rca_fused = torch.cat(list(rca_output.values()), dim=1)
        elif fused_features is not None:
            rca_fused = fused_features
        use_rca_path = rca_fused is not None and getattr(self.cfg, 'use_rca_fusion', True)
        use_crv_path = return_reasoner_outputs and getattr(self.cfg, 'use_crv_verification', True)

        if not use_cache or past_key_values is None:
            seq_is_long = self._should_use_long_context_path(x.shape[1])

            # Lazy-init modules used below
            if use_rca_path:
                self._lazy_get_rca()
            if use_crv_path:
                self._lazy_get_crv()
            self._lazy_get_comet()
            self._lazy_get_long_context()

            crv_active = self.crv_integration is not None
            if crv_active:
                n_layers = len(self.layers)
                crv_checkpoints = {n_layers // 4, n_layers // 2, 3 * n_layers // 4}

            if seq_is_long:
                aux_loss_bucket = [x.new_zeros(())]

                def _oomb_chunk(chunk, chunk_mask):
                    h_chunk = chunk
                    for layer_idx, layer in enumerate(self.layers):
                        past_kv = None
                        h_chunk, extra_kv = self.dual_injector.inject(h_chunk, layer_idx)
                        film_params = None

                        if hasattr(layer, 'set_sequence_length'):
                            layer.set_sequence_length(h_chunk.shape[1])

                        if self.deep_cross_layer_injector is not None and rca_fused is not None:
                            h_chunk = self.deep_cross_layer_injector(h_chunk, rca_fused, layer_idx)

                        if self.comet_memory is not None and layer_idx % 4 == 0:
                            h_chunk = self.comet_memory(h_chunk, update_memory=False)

                        h_chunk, aux_loss = layer(
                            h_chunk, chunk_mask, past_key_values=past_kv, use_cache=False,
                            subconscious_kv=extra_kv, film_params=film_params,
                            modal_id=modal_id,
                        )
                        aux_loss_bucket[0] = aux_loss_bucket[0] + (aux_loss if aux_loss is not None else 0.0)

                        if crv_active and layer_idx in crv_checkpoints:
                            self.crv_integration.record(h_chunk)

                    h_chunk = self._apply_long_context_refinement(h_chunk, chunk_mask)

                    return h_chunk

                h = self.oomb_processor.process(x, _oomb_chunk, mask)
                total_aux_loss = aux_loss_bucket[0]
            else:
                h = x
                for layer_idx, layer in enumerate(self.layers):
                    past_kv = None
                    h, extra_kv = self.dual_injector.inject(h, layer_idx)
                    film_params = None

                    if hasattr(layer, 'set_sequence_length'):
                        layer.set_sequence_length(h.shape[1])

                    if self.deep_cross_layer_injector is not None and rca_fused is not None:
                        h = self.deep_cross_layer_injector(h, rca_fused, layer_idx)

                    if self.comet_memory is not None and layer_idx % 4 == 0:
                        h = self.comet_memory(h, update_memory=False)

                    h, aux_loss = layer(
                        h, mask, past_key_values=past_kv, use_cache=False,
                        subconscious_kv=extra_kv, film_params=film_params,
                        modal_id=modal_id,
                    )
                    total_aux_loss = total_aux_loss + (aux_loss if aux_loss is not None else 0.0)

                    if crv_active and layer_idx in crv_checkpoints:
                        self.crv_integration.record(h)

                h = self._apply_long_context_refinement(h, mask)

            outputs = [h]
        else:
            # Chunked processing for incremental inference with KV cache
            for i in range(0, x.shape[1], chunk_size):
                x_chunk = x[:, i:i+chunk_size, ...]
                if mask.dim() == 3:
                    mask_chunk = mask[:, i:i+chunk_size, :i+chunk_size]
                else:
                    mask_chunk = mask[i:i+chunk_size, :i+chunk_size]

                def block_fn(xc, msk, layer_past_key_values=None):
                    h = xc
                    aux = 0.0
                    new_caches = []
                    seq_len = xc.shape[1]

                    for layer_idx, layer in enumerate(self.layers):
                        if self.comet_memory is not None:
                            if layer_idx % max(1, len(self.layers) // 2) == 0:
                                comet_ctx = self.comet_memory.read(h)
                                if comet_ctx is not None and hasattr(layer, '_set_memory_context'):
                                    layer._set_memory_context(comet_ctx)

                        if self.deep_cross_layer_injector is not None and rca_fused is not None:
                            h = self.deep_cross_layer_injector(h, rca_fused, layer_idx)

                        past_kv = self.cache_manager.get_kv_cache(
                            layer_idx,
                            layer_past_key_values[layer_idx] if layer_past_key_values is not None else None
                        )

                        if past_kv is not None and cache_quant_bits < 16:
                            past_kv = self._convert_cache_precision(
                                past_kv,
                                cache_dtype=cache_dtype,
                                use_mixed_precision_cache=use_mixed_precision_cache,
                            )

                        h, extra_kv = self.dual_injector.inject(h, layer_idx)
                        film_params = None

                        if hasattr(layer, 'set_sequence_length'):
                            layer.set_sequence_length(seq_len)

                        h, aux_loss, cache = layer(
                            h, msk, past_key_values=past_kv, use_cache=True,
                            subconscious_kv=extra_kv, film_params=film_params,
                            modal_id=modal_id,
                        )

                        if cache is not None:
                            key_states, value_states = cache
                            updated = self.cache_manager.update_kv_cache(
                                layer_idx,
                                key_states,
                                value_states,
                                i + xc.shape[1],
                            )
                            cache = updated

                            if cache_quant_bits < 16:
                                cache = self._convert_cache_precision(
                                    cache,
                                    cache_dtype=cache_dtype,
                                    use_mixed_precision_cache=False,
                                )

                        if hasattr(self, 'cache_manager') and self.cache_manager is not None:
                            self.cache_manager.compute_pending_prediction(layer_idx, h)

                        new_caches.append(cache)
                        aux = aux + (aux_loss if aux_loss is not None else 0.0)

                    h = self._apply_long_context_refinement(h, None)

                    return h, aux, new_caches

                with torch.amp.autocast("cuda", dtype=cache_dtype, enabled=(x.device.type == 'cuda')):
                    h_chunk, aux_chunk, cache_chunk = block_fn(x_chunk, mask_chunk, past_key_values)
                if next_cache is not None and cache_chunk is not None:
                    next_cache.extend(cache_chunk)

                outputs.append(h_chunk)
                total_aux_loss = total_aux_loss + aux_chunk

        # Clear dual-injector caches after layer processing
        self.dual_injector.clear_cache()

        # Concatenate all chunks at once after the loop (more efficient than per-chunk)
        if outputs:
            x = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)

            if x.shape[1] == 0:
                task_logits = (
                    torch.zeros(x.shape[0], self.cfg.task_classes, device=x.device)
                    if return_task_logits
                    else None
                )
                eval_score = (
                    torch.zeros(x.shape[0], self.cfg.eval_dims, device=x.device)
                    if return_eval_score
                    else None
                )
                return YvModelOutput({
                    "logits": self.lm_head(x),
                    "loss": torch.tensor(0.0, device=x.device, requires_grad=True),
                    "task_logits": task_logits,
                    "eval_score": eval_score,
                    "aux_loss": total_aux_loss,
                    "reasoner_out": None,
                    "tool_output": None,
                    "tool_experience": None,
                    "vericot_out": None,
                })

            x = self.norm(x)
            logits = self.lm_head(x)

            # Lazy-init reasoner and downstream modules
            if return_reasoner_outputs:
                self._lazy_get_reasoner()
            if return_verifier_outputs:
                self._lazy_get_vericot()
            if return_tool_outputs:
                self._lazy_get_seer()

            reasoner_out = None
            if return_reasoner_outputs and self.reasoner is not None:
                reasoner_input_ids = (
                    input_ids[:, :x.shape[1]] if input_ids.shape[1] > x.shape[1] else input_ids
                )
                reasoner_labels = (
                    labels[:, :x.shape[1]]
                    if labels is not None and labels.shape[1] > x.shape[1]
                    else labels
                )

                reasoner_out = self.reasoner(
                    input_ids=reasoner_input_ids,
                    attention_mask=attention_mask,
                    labels=reasoner_labels,
                    hidden_states=x,
                )

            # CRV: circuit-based contradiction detection (ICLR 2026 Oral)
            if return_reasoner_outputs and self.crv_integration is not None and reasoner_out is not None:
                crv_out = self.crv_integration(hidden_states=x)
                reasoner_out['crv_verified'] = crv_out.get('verified', torch.tensor(True, device=x.device))
                reasoner_out['crv_confidence'] = crv_out.get('confidence', torch.tensor(1.0, device=x.device))

            # VeriCoT: neuro-symbolic verification of reasoning chain
            vericot_out = None
            if return_verifier_outputs and self.vericot_verifier is not None and reasoner_out is not None:
                vericot_out = self.vericot_verifier.verify_batch(
                    hidden_states=x,
                    logits=logits,
                    input_ids=input_ids,
                    reasoner_out=reasoner_out,
                )
                reasoner_out['vericot_verified'] = vericot_out.get('verified', False)
                reasoner_out['vericot_confidence'] = vericot_out.get('confidence', torch.tensor(1.0, device=x.device))
                reasoner_out['vericot_correction'] = vericot_out.get('correction_logits', None)
                if vericot_out.get('correction_logits') is not None:
                    logits = logits + 0.1 * vericot_out['correction_logits']
                verifier_loss = vericot_out.get('verifier_loss')
                if verifier_loss is not None:
                    total_aux_loss = total_aux_loss + verifier_loss

            loss = None
            mtp_loss = logits.new_zeros(())
            mtp_logits_list = []
            
            if labels is not None and self.comet_memory is not None:
                self._comet_write_step += 1
                if self._comet_write_step % self._comet_write_interval == 0:
                    self.comet_memory.write(x.detach(), input_ids=input_ids)

            if labels is not None:
                text_seq_len = labels.shape[1]
                lm_loss = F.cross_entropy(
                    logits[:, -text_seq_len:, :].reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100
                )
                reasoner_loss = (
                    reasoner_out.get("loss")
                    if reasoner_out is not None and reasoner_out.get("loss") is not None
                    else logits.new_zeros(())
                )
                loss = lm_loss + reasoner_loss

                if self.num_mtp_heads > 0 and hasattr(self, 'mtp_heads'):
                    self._lazy_get_mtp_heads()
                    for i, mtp_head in enumerate(self.mtp_heads):
                        offset = i + 1
                        if x.shape[1] > offset and labels.shape[1] > offset:
                            mtp_logits = mtp_head(x[:, :-offset])
                            mtp_labels = labels[:, offset:]
                            if mtp_logits.shape[1] >= mtp_labels.shape[1]:
                                mtp_logits = mtp_logits[:, :mtp_labels.shape[1]]
                            mtp_loss_i = F.cross_entropy(
                                mtp_logits.reshape(-1, mtp_logits.size(-1)),
                                mtp_labels.reshape(-1),
                                ignore_index=-100
                            )
                            mtp_loss = mtp_loss + mtp_loss_i
                            if return_mtp_logits:
                                mtp_logits_list.append(mtp_logits)
                    
                    mtp_loss = mtp_loss / max(1, self.num_mtp_heads)
                    loss = loss + self.mtp_loss_weight * mtp_loss

            task_logits = None
            if return_task_logits:
                self._lazy_get_task_head()
                task_logits = self.task_head(x[:, 0])

            eval_score = None
            if return_eval_score:
                self._lazy_get_eval_head()
                eval_score = self.eval_head(x.mean(1))

        tool_output = None
        tool_experience = None
        if return_tool_outputs and self.seer_executor is not None and outputs:
            seer_result = self.seer_executor(
                query_hidden=x,
                reasoner_out=reasoner_out,
                input_ids=input_ids,
            )
            tool_output = seer_result.get('tool_result')
            tool_experience = seer_result.get('experience_recalled')
            if tool_output is not None:
                seer_loss = seer_result.get('seer_loss')
                if seer_loss is not None:
                    total_aux_loss = total_aux_loss + seer_loss

        result = YvModelOutput({
            "logits": logits,
            "mtp_logits": mtp_logits_list if (self.num_mtp_heads > 0 and return_mtp_logits) else [],
            "mtp_loss": mtp_loss if self.num_mtp_heads > 0 else logits.new_zeros(()),
            "loss": loss,
            "task_logits": task_logits,
            "eval_score": eval_score,
            "aux_loss": total_aux_loss,
            "reasoner_out": reasoner_out,
            "tool_output": tool_output if tool_output is not None else None,
            "tool_experience": tool_experience if tool_experience is not None else None,
            "vericot_out": vericot_out if vericot_out is not None else None,
        })

        if use_cache:
            result["past_key_values"] = next_cache

        if kwargs.get("output_hidden_states"):
            result["hidden_states"] = (x,)
        if kwargs.get("output_attentions"):
            result["attentions"] = ()

        if hasattr(self, 'cache_manager') and self.cache_manager is not None:
            result["cache_stats"] = self.cache_manager.get_cache_stats()

        return result

class YvModelForCausalLM(YvModel):
    """Yv model specialized for causal language modeling.
    
    Extends YvModel with specific functionality for autoregressive
    text generation tasks. This is the primary model class for text
    generation, dialogue, and completion tasks.
    
    Key Features:
        - Autoregressive generation with various decoding strategies
        - Support for speculative decoding for efficiency
        - Integration with reasoning and tool use
        - Multimodal input support for vision-language tasks
    
    Generation Strategies:
        - Greedy: Deterministic, highest probability tokens
        - Sampling: Stochastic with temperature control
        - Top-k: Limited vocabulary sampling
        - Top-p (nucleus): Cumulative probability threshold
        - Beam search: Multiple hypothesis tracking
    
    Attributes:
        Inherits all attributes from YvModel.
    
    Example:
        >>> model = YvModelForCausalLM(config)
        >>> 
        >>> # Training
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs["loss"]
        >>> 
        >>> # Generation
        >>> generated, stats = model.generate(
        ...     input_ids=prompt_ids,
        ...     max_length=100,
        ...     temperature=0.7,
        ...     top_p=0.9
        ... )
    
    Note:
        This class is the recommended entry point for text generation tasks.
        Use YvModel directly for more control over the forward pass.
    """

    def __init__(self, cfg, device=None, dtype=None, quantization_config=None, lora_config=None):
        """Initialize causal language model.
        
        Args:
            cfg: Configuration object containing model hyperparameters.
            device: Device to place model parameters on.
            dtype: Data type for model parameters.
            quantization_config: Configuration for model quantization.
            lora_config: Configuration for LoRA adapters.
        """
        super().__init__(cfg, device, dtype, quantization_config, lora_config)

        # SOLAR: parameter-level meta-learning self-optimization
        self.solar = None
        if getattr(cfg, 'use_solar', False):
            from ..reasoning.self_evolution import YvSOLAR
            self.solar = YvSOLAR(self, meta_lr=getattr(cfg, 'solar_meta_lr', 1e-4))

        # Self-Play: self-generation → self-critique → self-training
        self.self_play_trainer = None
        if getattr(cfg, 'use_self_play', False):
            from opss.train.self_play import POPSSSelfPlayTrainer
            self.self_play_trainer = POPSSSelfPlayTrainer(
                model=self,
                num_rounds=getattr(cfg, 'self_play_num_rounds', 3),
                num_samples=getattr(cfg, 'self_play_num_samples', 4),
                temperature=getattr(cfg, 'self_play_temperature', 0.8),
                dpo_beta=getattr(cfg, 'self_play_dpo_beta', 0.1),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass for causal language modeling.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            labels: Target labels for loss computation [batch, seq_len].
            **kwargs: Additional arguments passed to parent forward.
            
        Returns:
            Dictionary containing:
                - logits: Output logits [batch, seq_len, vocab_size]
                - loss: Cross-entropy loss (if labels provided)
                - aux_loss: Auxiliary losses from MoE routing
                - reasoner_out: Reasoning module outputs
                - Other outputs from parent model
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        # SOLAR: meta-update after loss computation during training
        if self.solar is not None and self.training and outputs.get('loss') is not None:
            self.solar.meta_update(outputs['loss'])
        return outputs


class YvModelForSequenceClassification(nn.Module):
    """Yv model for sequence classification tasks.
    
    Adds a classification head on top of the base model for tasks like
    sentiment analysis, topic classification, and natural language inference.
    
    Architecture:
        - Base YvModel for encoding
        - Pooling layer (CLS token or mean pooling)
        - Dropout layer for regularization
        - Linear classification head
    
    Pooling Strategies:
        - CLS: Use first token representation
        - Mean: Average all token representations
        - Max: Maximum pooling across tokens
    
    Attributes:
        model (YvModel): Base model for encoding.
        num_labels (int): Number of classification classes.
        classifier (nn.Linear): Classification head.
        dropout (nn.Dropout): Dropout layer.
    
    Example:
        >>> model = YvModelForSequenceClassification(config, num_labels=3)
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> logits = outputs["logits"]
        >>> loss = outputs["loss"]
    
    Note:
        For multi-label classification, use sigmoid activation and
        binary cross-entropy loss instead of softmax.
    """

    def __init__(self, cfg, num_labels: int, device=None, dtype=None):
        """Initialize sequence classification model.
        
        Args:
            cfg: Configuration object containing model hyperparameters.
            num_labels: Number of classification classes.
            device: Device to place model parameters on.
            dtype: Data type for model parameters.
        """
        super().__init__()
        self.model = YvModel(cfg, device, dtype)
        self.num_labels = num_labels

        self.classifier = nn.Linear(cfg.hidden_size, num_labels, device=device, dtype=dtype)
        self.dropout = nn.Dropout(getattr(cfg, 'classifier_dropout', 0.1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass for sequence classification.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            labels: Target class labels [batch].
            **kwargs: Additional arguments passed to base model.
            
        Returns:
            Dictionary containing:
                - logits: Classification logits [batch, num_labels]
                - loss: Cross-entropy loss (if labels provided)
                - hidden_states: Hidden states from base model
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        hidden_states = outputs['logits'] if isinstance(outputs, dict) else outputs
        if isinstance(hidden_states, torch.Tensor):
            pooled = hidden_states[:, 0]
        else:
            pooled = hidden_states.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }


class YvModelForTokenClassification(nn.Module):
    """Yv model for token classification tasks.
    
    Adds a token-level classification head for tasks like named entity
    recognition (NER), part-of-speech (POS) tagging, and chunking.
    
    Architecture:
        - Base YvModel for encoding
        - Dropout layer for regularization
        - Linear classification head (per-token)
    
    Key Features:
        - Per-token predictions
        - Supports BIO/BIOES tagging schemes
        - Optional CRF layer for structured prediction
    
    Attributes:
        model (YvModel): Base model for encoding.
        num_labels (int): Number of token classes.
        classifier (nn.Linear): Token classification head.
        dropout (nn.Dropout): Dropout layer.
    
    Example:
        >>> model = YvModelForTokenClassification(config, num_labels=9)
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> logits = outputs["logits"]  # [batch, seq_len, num_labels]
    
    Note:
        For structured prediction, consider adding a CRF layer on top
        of the classification head to enforce valid tag sequences.
    """

    def __init__(self, cfg, num_labels: int, device=None, dtype=None):
        """Initialize token classification model.
        
        Args:
            cfg: Configuration object containing model hyperparameters.
            num_labels: Number of token classes (e.g., 9 for BIO NER).
            device: Device to place model parameters on.
            dtype: Data type for model parameters.
        """
        super().__init__()
        self.model = YvModel(cfg, device, dtype)
        self.num_labels = num_labels

        self.classifier = nn.Linear(cfg.hidden_size, num_labels, device=device, dtype=dtype)
        self.dropout = nn.Dropout(getattr(cfg, 'classifier_dropout', 0.1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass for token classification.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            labels: Target token labels [batch, seq_len].
            **kwargs: Additional arguments passed to base model.
            
        Returns:
            Dictionary containing:
                - logits: Token classification logits [batch, seq_len, num_labels]
                - loss: Cross-entropy loss (if labels provided)
                - hidden_states: Hidden states from base model
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        hidden_states = outputs.get('logits', outputs)
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states['logits']

        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }

class YvModelForQuestionAnswering(nn.Module):
    """Yv model for extractive question answering tasks.
    
    Adds span prediction heads for extracting answer spans from context
    passages. Suitable for tasks like SQuAD-style question answering.
    
    Architecture:
        - Base YvModel for encoding
        - Linear head predicting start and end positions
        - Optional answer type classification
    
    Key Features:
        - Span extraction from context
        - Start and end position prediction
        - Support for unanswerable questions
        - Multi-span answer support
    
    Attributes:
        model (YvModel): Base model for encoding.
        qa_outputs (nn.Linear): QA output head (2 outputs: start, end).
    
    Example:
        >>> model = YvModelForQuestionAnswering(config)
        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     start_positions=start_labels,
        ...     end_positions=end_labels
        ... )
        >>> start_logits = outputs["start_logits"]
        >>> end_logits = outputs["end_logits"]
    
    Note:
        For answerability prediction, add a third output for "no answer"
        classification and use a threshold to determine if the question
        is answerable.
    """

    def __init__(self, cfg, device=None, dtype=None):
        """Initialize question answering model.
        
        Args:
            cfg: Configuration object containing model hyperparameters.
            device: Device to place model parameters on.
            dtype: Data type for model parameters.
        """
        super().__init__()
        self.model = YvModel(cfg, device, dtype)

        self.qa_outputs = nn.Linear(cfg.hidden_size, 2, device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass for question answering.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            start_positions: Start position labels [batch].
            end_positions: End position labels [batch].
            **kwargs: Additional arguments passed to base model.
            
        Returns:
            Dictionary containing:
                - start_logits: Start position logits [batch, seq_len]
                - end_logits: End position logits [batch, seq_len]
                - loss: Combined start/end loss (if labels provided)
                - hidden_states: Hidden states from base model
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        hidden_states = outputs.get('logits', outputs)
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states['logits']

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            start_loss = F.cross_entropy(start_logits, start_positions)
            end_loss = F.cross_entropy(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'loss': total_loss,
            'hidden_states': hidden_states
        }


class YvModelForMaskedLM(nn.Module):
    """Yv model for masked language modeling (MLM).
    
    Adds an MLM head for BERT-style pretraining with masked token
    prediction. Suitable for encoder-only pretraining and fine-tuning.
    
    Architecture:
        - Base YvModel for encoding
        - Linear head for vocabulary prediction
        - Optional layer normalization before head
    
    Key Features:
        - Masked token prediction
        - Support for bidirectional attention
        - Useful for encoder pretraining
        - Transfer learning to downstream tasks
    
    Pretraining Strategy:
        - Mask 15% of tokens randomly
        - 80% replaced with [MASK]
        - 10% replaced with random token
        - 10% kept unchanged
    
    Attributes:
        model (YvModel): Base model for encoding.
        lm_head (nn.Linear): Language modeling head.
    
    Example:
        >>> model = YvModelForMaskedLM(config)
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> logits = outputs["logits"]
        >>> loss = outputs["loss"]
    
    Note:
        For bidirectional attention, ensure the base model is configured
        with appropriate attention mask patterns. The standard causal
        mask should be disabled for MLM tasks.
    """

    def __init__(self, cfg, device=None, dtype=None):
        """Initialize masked language model.
        
        Args:
            cfg: Configuration object containing model hyperparameters.
            device: Device to place model parameters on.
            dtype: Data type for model parameters.
        """
        super().__init__()
        self.model = YvModel(cfg, device, dtype)

        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass for masked language modeling.
        
        Args:
            input_ids: Input token IDs [batch, seq_len] with masked tokens.
            attention_mask: Attention mask [batch, seq_len].
            labels: Target labels for masked positions [batch, seq_len].
                Use -100 for non-masked positions to ignore in loss.
            **kwargs: Additional arguments passed to base model.
            
        Returns:
            Dictionary containing:
                - logits: Vocabulary logits [batch, seq_len, vocab_size]
                - loss: Cross-entropy loss (if labels provided)
                - hidden_states: Hidden states from base model
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        hidden_states = outputs.get('logits', outputs)
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states['logits']

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }
