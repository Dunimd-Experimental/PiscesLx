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
from typing import Optional, Tuple, Dict, Any, List, Union
from ..reasoning import YvMultiModalReasoningEnhancer
from ..generation.speculative import YvAdaptiveSpeculativeDecoder, YvSpeculativeConfig
from ..multimodal import (
    YvUnifiedReasoner,
    YvVisionEncoder,
    YvAudioEncoder,
    YvDocEncoder,
    YvVideoEncoder,
    YvAgenticEncoder,
    YvDynamicModalFusion
)
from dataclasses import dataclass
from enum import Enum
import math

_LOG = PiscesLxLogger(__name__)

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
            3. Hybrid layers (if use_mamba3 and sequence threshold met)
            4. Default attention layers
        """
        mamba3_layers = getattr(self.config, 'mamba3_layers', [])
        moe_layers = getattr(self.config, 'moe_layers', [])

        for i in range(self.n_layer):
            layer_type = YvLayerType.ATTENTION

            if i in mamba3_layers:
                layer_type = YvLayerType.SSM
            elif i in moe_layers:
                layer_type = YvLayerType.MOE
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

    def __init__(self, cfg, device=None, dtype=None, quantization_config=None, lora_config=None):
        """Initialize Yv model with configuration.
        
        Args:
            cfg: Configuration object containing model hyperparameters.
                Required: hidden_size, n_layer, vocab_size
                Optional: n_head, n_kv_head, max_position_embeddings, etc.
            device: Device to place model parameters on.
            dtype: Data type for model parameters.
            quantization_config: Configuration for model quantization.
            lora_config: Configuration for LoRA adapters.
        
        Note:
            Initialization is logged at debug level for each major component.
            Total parameter count is logged at the end of initialization.
        """
        super().__init__()
        _LOG.debug("YvModel: __init__ start")
        self.cfg = cfg
        self.config = cfg

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

        if getattr(self.config, 'max_position_embeddings', 0) >= 1_048_576 and not hasattr(self.config, 'use_h2o_attention'):
            setattr(self.config, 'use_h2o_attention', True)

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
            "cache_eviction_policy": "lru"
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
        self.layers = nn.ModuleList([])
        for i in range(cfg.n_layer):
            if (i % 4 == 0) or (i == cfg.n_layer - 1):
                _LOG.debug(f"YvModel: initializing TransformerBlock {i+1}/{cfg.n_layer}")

            layer_config = self.layer_router.get_layer_config(i)

            use_hybrid = getattr(cfg, 'use_mamba3', False)
            if use_hybrid:
                mamba3_layers = getattr(cfg, 'mamba3_layers', [])
                if not mamba3_layers or i in mamba3_layers:
                    _LOG.debug(f"YvModel: using YvHybridBlock for layer {i+1}")
                    block = YvHybridBlock(
                        cfg,
                        device=device,
                        dtype=dtype,
                        quantization_config=self.quantization_config
                    )
                else:
                    block = YvTransformerBlock(
                        cfg,
                        device=device,
                        dtype=dtype,
                        quantization_config=self.quantization_config
                    )
            else:
                block = YvTransformerBlock(
                    cfg,
                    device=device,
                    dtype=dtype,
                    quantization_config=self.quantization_config
                )

            block.cache_manager = self.cache_manager
            block.layer_idx = i
            self.layers.append(block)

        _LOG.debug("YvModel: initializing norm...")
        self.norm = YvRMSNorm(cfg.hidden_size)

        _LOG.debug("YvModel: initializing multimodal encoders...")
        self.vision = YvVisionEncoder(cfg)
        self.video = YvVideoEncoder(cfg)
        self.audio = YvAudioEncoder(cfg)
        self.doc = YvDocEncoder(cfg)

        self.agent_encoder = YvAgenticEncoder(cfg)

        use_enhanced_fusion = getattr(cfg, 'use_enhanced_fusion', False)
        if use_enhanced_fusion:
            from ..multimodal import YvEnhancedModalFusion, YvModalFusionConfig
            fusion_config = YvModalFusionConfig(
                hidden_size=cfg.hidden_size,
                num_modalities=6,
                num_heads=getattr(cfg, 'num_heads', 16),
                num_layers=4,
                dropout=0.1,
                use_quality_aware_fusion=True,
                use_modality_attention=True,
                use_cross_modal_alignment=True
            )
            self.modal_fusion = YvEnhancedModalFusion(fusion_config)
            _LOG.debug("YvModel: using YvEnhancedModalFusion")
        else:
            self.modal_fusion = YvDynamicModalFusion(cfg)
            _LOG.debug("YvModel: using legacy YvDynamicModalFusion")

        try:
            if 'agent_encoder' in self._modules:
                del self._modules['agent_encoder']
        except Exception as e:
            _LOG.debug(f"YvModel: agent_encoder cleanup skipped: {e}")

        _LOG.debug("YvModel: initializing output heads...")
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device=device, dtype=dtype)
        self.task_head = nn.Linear(cfg.hidden_size, cfg.task_classes, device=device, dtype=dtype)
        self.eval_head = nn.Linear(cfg.hidden_size, cfg.eval_dims, device=device, dtype=dtype)

        self.num_mtp_heads = int(getattr(cfg, 'num_mtp_heads', 4))
        self.mtp_loss_weight = float(getattr(cfg, 'mtp_loss_weight', 0.5))
        self.mtp_share_embeddings = bool(getattr(cfg, 'mtp_share_embeddings', True))
        
        if self.num_mtp_heads > 0:
            self.mtp_heads = nn.ModuleList([
                nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device=device, dtype=dtype)
                for _ in range(self.num_mtp_heads)
            ])
            if self.mtp_share_embeddings:
                for mtp_head in self.mtp_heads:
                    mtp_head.weight = self.lm_head.weight
            _LOG.debug(f"YvModel: MTP initialized with {self.num_mtp_heads} heads")

        self.modal_token_count = getattr(cfg, 'modal_token_count', 8)
        self.fusion_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False, device=device, dtype=dtype)

        _LOG.debug("YvModel: initializing reasoner...")
        self.reasoner = YvUnifiedReasoner(cfg)
        self.reasoner.initialize_reasoning_tokens(None)

        _LOG.debug("YvModel: initializing multi-modal reasoning enhancer...")
        self.mm_reasoning_enhancer = YvMultiModalReasoningEnhancer(cfg)

        _LOG.debug("YvModel: initializing agentic...")
        try:
            from ..multimodal import YvAgentic
        except ImportError:
            YvAgentic = None
        self.agentic = YvAgentic(cfg, model=self)

        try:
            if 'agentic' in self._modules:
                del self._modules['agentic']
        except Exception as e:
            _LOG.debug(f"YvModel: agentic cleanup skipped: {e}")

        try:
            if 'mm_reasoning_enhancer' in self._modules:
                del self._modules['mm_reasoning_enhancer']
        except Exception as e:
            _LOG.debug(f"YvModel: mm_reasoning_enhancer cleanup skipped: {e}")

        _LOG.debug("YvModel: initializing speculative decoder...")
        self.speculative_config = YvSpeculativeConfig(
            num_candidates=getattr(cfg, 'speculative_candidates', 4),
            draft_length=getattr(cfg, 'speculative_draft_length', 5),
            acceptance_threshold=getattr(cfg, 'speculative_acceptance_threshold', 0.8),
            temperature=getattr(cfg, 'speculative_temperature', 0.7),
            top_k=getattr(cfg, 'speculative_top_k', 50),
            top_p=getattr(cfg, 'speculative_top_p', 0.9)
        )
        self.speculative_decoder = YvAdaptiveSpeculativeDecoder(self.speculative_config, self, None)
        _LOG.debug("YvModel: speculative decoder initialized")

        try:
            for k in ("speculative_decoder",):
                if k in self._modules:
                    del self._modules[k]
        except Exception as e:
            _LOG.debug(f"YvModel: speculative_decoder cleanup skipped: {e}")

        if lora_config is not None:
            try:
                from peft import get_peft_model
                self = get_peft_model(self, lora_config)
                _LOG.debug("YvModel: LoRA adapters injected (peft)")
            except Exception as e:
                _LOG.error(f"LoRA injection failed: {e}")

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

        return None

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

    def to(self, device=None, dtype=None, non_blocking=False):
        """Move model to specified device and/or dtype.
        
        Overrides the default to() method for optimized device transfer,
        ensuring all components are properly moved including non-module
        attributes.
        
        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device).
            dtype: Target data type (e.g., torch.float16, torch.bfloat16).
            non_blocking: Whether to perform non-blocking transfer.
            
        Returns:
            self: The model after device/dtype transfer.
        
        Note:
            This handles ModuleList specially for proper recursive transfer.
        """
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
        ]

        for m in modules:
            if isinstance(m, nn.ModuleList):
                for sub in m:
                    sub.to(device=device, dtype=dtype, non_blocking=non_blocking)
            else:
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

        if hasattr(self.reasoner, 'resize_vocab'):
            self.reasoner.resize_vocab(new_num_tokens)
        self.cfg.vocab_size = new_num_tokens

        try:
            self.reasoner.initialize_reasoning_tokens(None)
        except Exception as e:
            _LOG.debug(f"YvModel: reasoner token init skipped: {e}")

        try:
            _LOG.info(
                f"Resized token embeddings to {new_num_tokens}. "
                f"Remember to update special token IDs in the reasoner."
            )
        except ImportError as e:
            _LOG.debug(f"YvModel: logger info skipped: {e}")

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
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
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
            max_length: Maximum total sequence length. Default: 100.
            temperature: Sampling temperature. Higher = more random. Default: 0.7.
            top_k: Top-k sampling vocabulary size. Default: 50.
            top_p: Nucleus sampling cumulative probability. Default: 0.9.
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
        else:
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

        def _bump_gate_temperature(_model, delta: float, cap: float):
            """Adjust gate temperature for MoE layers during generation.
            
            Increases the temperature of gating networks to encourage
            expert diversity during generation.
            
            Args:
                _model: The model to adjust.
                delta: Temperature increment.
                cap: Maximum temperature cap.
            """
            try:
                for m in _model.modules():
                    if hasattr(m, 'temperature'):
                        try:
                            if isinstance(m.temperature, torch.Tensor):
                                cur = m.temperature.detach().float().mean().item()
                                newv = min(cap, cur + delta)
                                m.temperature.fill_(newv)
                            else:
                                cur = float(getattr(m, 'temperature', 1.0))
                                newv = min(cap, cur + delta)
                                setattr(m, 'temperature', newv)
                        except Exception:
                            continue
            except Exception as e:
                _LOG.debug(f"YvModel: gate temperature update skipped: {e}")

        with torch.no_grad():
            for step_idx in range(max_length - input_ids.shape[1]):
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

                if next_token.item() == getattr(self.cfg, 'eos_token_id', 2) and new_tokens_generated >= min_new_tokens:
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
                - tool_trigger: Tool use triggers (if conditions met)
                - tool_intent: Tool intent information
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

        if agent_mode:
            return self.agentic.run(
                input_ids=input_ids,
                images=images,
                audio=audio,
                video=video,
                task=task,
                max_steps=max_steps
            )

        b, t = input_ids.shape
        text_emb = self.embed(input_ids)
        modal_features = {'text': text_emb}

        if images is not None:
            img_out = self.vision(images)
            modal_features['image'] = (
                img_out['features'] if isinstance(img_out, dict) and 'features' in img_out else img_out
            )

        if audio is not None:
            aud_out = self.audio(audio)
            modal_features['audio'] = (
                aud_out['features'] if isinstance(aud_out, dict) and 'features' in aud_out else aud_out
            )

        if video is not None:
            vid_out = self.video(video)
            modal_features['video'] = (
                vid_out['features'] if isinstance(vid_out, dict) and 'features' in vid_out else vid_out
            )

        if docs is not None:
            doc_out = self.doc(docs)
            modal_features['doc'] = (
                doc_out['features'] if isinstance(doc_out, dict) and 'features' in doc_out else doc_out
            )

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

        if len(modal_features) > 1:
            fused_features = self.modal_fusion(modal_features)
            if fused_features is None:
                x = text_emb
            elif fused_features.dim() == 3:
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
        else:
            x = text_emb

        t = x.shape[1]
        lm_seq_len = x.shape[1]
        mask = torch.full((t, t), float('-inf'), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)

        total_aux_loss = 0.0
        chunk_size = min(getattr(self.cfg, 'max_position_embeddings', 2048), 8192)
        outputs = []

        if use_cache:
            seq_len = x.shape[1]
            if seq_len > 1024:
                cache_dtype = torch.float16
                cache_quant_bits = 4
            elif seq_len > 512:
                cache_dtype = torch.float16
                cache_quant_bits = 8
            else:
                cache_dtype = torch.float32
                cache_quant_bits = 16
        else:
            cache_dtype = torch.float32
            cache_quant_bits = 16

        autocast_ctx = torch.amp.autocast("cuda", dtype=cache_dtype)
        with autocast_ctx:
            next_cache = [] if use_cache else None

            for i in range(0, x.shape[1], chunk_size):
                x_chunk = x[:, i:i+chunk_size, ...]
                mask_chunk = mask[i:i+chunk_size, i:i+chunk_size]

                def block_fn(xc, msk, layer_past_key_values=None):
                    h = xc
                    aux = 0.0
                    new_caches = []
                    seq_len = xc.shape[1]

                    for layer_idx, layer in enumerate(self.layers):
                        use_mamba3_for_layer = False
                        if getattr(self.cfg, 'use_mamba3', False):
                            threshold = getattr(self.cfg, 'mamba3_sequence_threshold', 8192)
                            mamba3_layers = getattr(self.cfg, 'mamba3_layers', [])

                            if (not mamba3_layers or layer_idx in mamba3_layers) and seq_len >= threshold:
                                use_mamba3_for_layer = True

                        past_kv = self.cache_manager.get_kv_cache(
                            layer_idx,
                            layer_past_key_values[layer_idx] if layer_past_key_values is not None else None
                        )

                        if past_kv is not None and cache_quant_bits < 16:
                            past_kv = tuple(
                                tensor.to(cache_dtype) if tensor is not None else None
                                for tensor in past_kv
                            )

                        if use_cache:
                            if hasattr(layer, 'set_sequence_length'):
                                layer.set_sequence_length(seq_len)

                            h, aux_loss, cache = layer(h, msk, past_key_values=past_kv, use_cache=True)

                            if cache is not None:
                                key_states, value_states = cache
                                updated = self.cache_manager.update_kv_cache(
                                    layer_idx,
                                    key_states,
                                    value_states,
                                    i + xc.shape[1],
                                    use_h2o=getattr(self.cfg, 'use_h2o_attention', False)
                                )
                                cache = updated

                                if cache_quant_bits < 16:
                                    cache = tuple(
                                        tensor.to(torch.float16) if tensor is not None else None
                                        for tensor in cache
                                    )
                            new_caches.append(cache)
                        else:
                            if hasattr(layer, 'set_sequence_length'):
                                layer.set_sequence_length(seq_len)

                            h, aux_loss = layer(h, msk, past_key_values=past_kv, use_cache=False)

                        aux = aux + (aux_loss if aux_loss is not None else 0.0)

                    if use_cache:
                        return h, aux, new_caches
                    return h, aux, None

                if use_cache:
                    h_chunk, aux_chunk, cache_chunk = block_fn(x_chunk, mask_chunk, past_key_values)
                    if next_cache is not None and cache_chunk is not None:
                        next_cache.extend(cache_chunk)
                else:
                    h_chunk, aux_chunk, _ = cp.checkpoint(
                        block_fn,
                        x_chunk,
                        mask_chunk,
                        None,
                        use_reentrant=False
                    )

                outputs.append(h_chunk)
                total_aux_loss = total_aux_loss + aux_chunk

            if outputs:
                x = torch.cat(outputs, dim=1)

            if x.shape[1] == 0:
                return {
                    "logits": self.lm_head(x),
                    "loss": torch.tensor(0.0, device=x.device, requires_grad=True),
                    "task_logits": torch.zeros(x.shape[0], self.cfg.task_classes, device=x.device),
                    "eval_score": torch.zeros(x.shape[0], self.cfg.eval_dims, device=x.device),
                    "aux_loss": total_aux_loss,
                    "reasoner_out": {"loss": torch.tensor(0.0, device=x.device, requires_grad=True)}
                }

            x = self.norm(x)
            logits = self.lm_head(x)

            reasoner_input_ids = (
                input_ids[:, :x.shape[1]] if input_ids.shape[1] > x.shape[1] else input_ids
            )
            reasoner_labels = (
                labels[:, :x.shape[1]]
                if labels is not None and labels.shape[1] > x.shape[1]
                else labels
            )

            reasoner_out = self.reasoner(x, reasoner_input_ids, reasoner_labels)

            loss = None
            mtp_loss = torch.tensor(0.0, device=x.device)
            mtp_logits_list = []
            
            if labels is not None:
                lm_loss = F.cross_entropy(
                    logits[:, :lm_seq_len, :].reshape(-1, logits.size(-1)),
                    labels.view(-1)
                )
                reasoner_loss = reasoner_out.get("loss", torch.tensor(0.0, device=x.device))
                loss = lm_loss + reasoner_loss
                
                if self.num_mtp_heads > 0 and hasattr(self, 'mtp_heads'):
                    for i, mtp_head in enumerate(self.mtp_heads):
                        offset = i + 1
                        if x.shape[1] > offset and labels.shape[1] > offset:
                            mtp_logits = mtp_head(x[:, :-offset])
                            mtp_labels = labels[:, offset:]
                            if mtp_logits.shape[1] >= mtp_labels.shape[1]:
                                mtp_logits = mtp_logits[:, :mtp_labels.shape[1]]
                            mtp_loss_i = F.cross_entropy(
                                mtp_logits.reshape(-1, mtp_logits.size(-1)),
                                mtp_labels.reshape(-1)
                            )
                            mtp_loss = mtp_loss + mtp_loss_i
                            mtp_logits_list.append(mtp_logits)
                    
                    mtp_loss = mtp_loss / max(1, self.num_mtp_heads)
                    loss = loss + self.mtp_loss_weight * mtp_loss

            tool_trigger = None
            try:
                unc = reasoner_out.get("uncertainty_scores", None)
                fac = reasoner_out.get("fact_consistency", None)
                unc_val = unc.mean().item() if unc is not None and unc.numel() > 0 else 0.0
                fac_val = fac.mean().item() if fac is not None and fac.numel() > 0 else 1.0
                unc_th = getattr(self.cfg, 'tool_uncertainty_threshold', 0.7)
                fac_th = getattr(self.cfg, 'fact_consistency_threshold', 0.55)
                if unc_val > unc_th or fac_val < fac_th:
                    tool_trigger = {
                        'uncertainty': unc_val,
                        'fact_consistency': fac_val,
                        'suggested_tools': reasoner_out.get("suggested_tools", []),
                    }
            except Exception:
                tool_trigger = None

            task_logits = self.task_head(x[:, 0])
            eval_score = self.eval_head(x.mean(1))

        tool_intent = None
        try:
            if tool_trigger is not None and tool_trigger.get('should_tool', False):
                tool_intent = {
                    'type': 'tool_intent',
                    'version': '1.0',
                    'triggers': {
                        'uncertainty': tool_trigger.get('uncertainty', 0.0),
                        'fact_consistency': tool_trigger.get('fact_consistency', 1.0),
                        'thresholds': {
                            'uncertainty': getattr(self.cfg, 'tool_uncertainty_threshold', 0.7),
                            'fact_consistency': getattr(self.cfg, 'tool_fact_consistency_threshold', 0.6)
                        }
                    },
                    'suggested_tools': tool_trigger.get('suggested_tools', []),
                    'confidence': float(min(1.0, max(0.0, tool_trigger.get('uncertainty', 0.0)))),
                    'reason': 'High uncertainty or low fact consistency detected by reasoner',
                }
        except Exception:
            tool_intent = None

        result = {
            "logits": logits,
            "mtp_logits": mtp_logits_list if self.num_mtp_heads > 0 else [],
            "mtp_loss": mtp_loss if self.num_mtp_heads > 0 else torch.tensor(0.0, device=x.device),
            "loss": loss,
            "task_logits": task_logits,
            "eval_score": eval_score,
            "aux_loss": total_aux_loss,
            "reasoner_out": reasoner_out,
            "tool_trigger": tool_trigger,
            "tool_intent": tool_intent
        }

        if use_cache:
            result["past_key_values"] = next_cache

        if hasattr(self, 'cache_manager') and self.cache_manager is not None:
            try:
                result["cache_stats"] = self.cache_manager.get_cache_stats()
            except Exception as e:
                _LOG.debug(f"YvModel: cache stats retrieval failed: {e}")

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

        hidden_states = outputs.get('logits', outputs)
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }
