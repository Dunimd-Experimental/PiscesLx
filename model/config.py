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

"""Configuration utilities for Yv flagship multimodal models.

This module provides comprehensive configuration management for the Yv
architecture, including model hyperparameters, MoE settings, multimodal
configurations, and inference optimizations.

Architecture Overview:
    The configuration system is designed around a single dataclass
    (YvConfig) that encapsulates all model parameters, with supporting
    enumerations for categorical options.
    
    Key Configuration Categories:
    
    1. **Model Architecture**:
       - hidden_size, n_layer, n_head: Core transformer dimensions
       - vocab_size: Token vocabulary (default: 151646 for Qwen3)
       - intermediate_size: FFN hidden dimension
    
    2. **Mixture-of-Experts**:
       - moe_num_experts: Total expert count (default: 64)
       - moe_top_k: Activated experts per token (default: 2)
       - moe_capacity_factor: Routing capacity multiplier
       - Load balancing and noise parameters
    
    3. **Multimodal Processing**:
       - image_res, max_image_res: Vision input resolutions
       - mm_tokens, audio_tokens: Multimodal token counts
       - Fusion quality thresholds
    
    4. **Attention and Cache**:
       - attention_type: Standard, Flash, H2O, etc.
       - max_cache_size, kv_cache_block_size: KV cache settings
       - Sliding window and streaming configurations
    
    5. **Speculative Decoding**:
       - speculative_candidates, draft_length: Decoding parameters
       - Acceptance thresholds and sampling settings
    
    6. **Mamba-3 SSM**:
       - mamba3_d_state, mamba3_d_conv: SSM dimensions
       - Hybrid layer configuration
       - Complex state and trapezoidal discretization

Configuration Presets:
    - get_small_config(): 768 hidden, 12 layers, 8 experts
    - get_base_config(): 2048 hidden, 24 layers, 64 experts
    - get_large_config(): 4096 hidden, 32 layers, 128 experts
    - get_xl_config(): 6144 hidden, 48 layers, 256 experts
    - get_hybrid_config(): Attention-Mamba hybrid
    - get_jamba_style_config(): Jamba-style MoE-Mamba

Example:
    >>> from model.config import YvConfig
    >>> 
    >>> # Load from JSON file
    >>> config = YvConfig.from_json("config.json")
    >>> 
    >>> # Create from dictionary
    >>> config = YvConfig.from_dict({"hidden_size": 4096, "n_layer": 32})
    >>> 
    >>> # Use preset
    >>> config = YvConfig.get_large_config()
    >>> 
    >>> # Validate configuration
    >>> config.validate()  # Raises ValueError if invalid
    >>> 
    >>> # Save configuration
    >>> config.to_json("output_config.json")

Dependencies:
    - dataclasses: For configuration dataclass
    - enum: For enumeration types
    - json: For serialization
"""

import json
import yaml
import copy
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, Optional, List, Union
from enum import Enum

from configs.version import VERSION


class YvAttentionType(Enum):
    """Enumeration of attention mechanism types.
    
    Defines the available attention implementations for the Yv model,
    each optimized for different use cases and sequence lengths.
    
    Attributes:
        STANDARD: Standard scaled dot-product attention with O(n²) complexity.
        FLASH: FlashAttention v1 for memory-efficient attention.
        FLASH2: FlashAttention v2 with improved parallelism.
        FLASH3: FlashAttention v3 with FP8 support.
        STREAMING_LLM: Streaming attention for infinite-length generation.
        H2O: Heavy-Hitter Oracle attention with KV compression.
        SLIDING_WINDOW: Local attention with fixed window size.
        LINEAR: Linear attention with kernel feature maps.
        SPARSE: Sparse attention patterns for long sequences.
        RING: Ring attention for distributed long-context processing.
    
    Example:
        >>> config.attention_type = YvAttentionType.FLASH2
    """
    STANDARD = "standard"
    FLASH = "flash"
    FLASH2 = "flash2"
    FLASH3 = "flash3"
    STREAMING_LLM = "streaming_llm"
    H2O = "h2o_attention"
    SLIDING_WINDOW = "sliding_window"
    LINEAR = "linear"
    SPARSE = "sparse"
    RING = "ring"


class YvPositionEmbeddingType(Enum):
    """Enumeration of position embedding types.
    
    Defines the available position encoding strategies for the model.
    
    Attributes:
        ROPE: Rotary Position Embedding, applies rotation to query/key.
        ALIBI: Attention with Linear Biases, no positional embeddings needed.
        YARN: Yet another RoPE extensioN method for long contexts.
        LERPE: Learnable Rotary Position Embedding.
        LEARNED: Standard learned absolute position embeddings.
        NONE: No position encoding (for autoregressive models).
    
    Example:
        >>> config.rope_type = YvPositionEmbeddingType.YARN
    """
    ROPE = "rope"
    ALIBI = "alibi"
    YARN = "yarn"
    LERPE = "lerpe"
    LEARNED = "learned"
    NONE = "none"


class YvActivationType(Enum):
    """Enumeration of activation function types.
    
    Defines the available activation functions for the feed-forward networks.
    
    Attributes:
        GELU: Gaussian Error Linear Unit.
        SILU: Sigmoid Linear Unit (Swish).
        SWIGLU: SwiGLU gated activation with Swish gate.
        GEGGLU: GeGLU gated activation with GELU gate.
        REGLU: ReGLU gated activation with ReLU gate.
        SOFTMAX: Softmax activation (for attention scores).
    
    Example:
        >>> config.activation_type = YvActivationType.SWIGLU
    """
    GELU = "gelu"
    SILU = "silu"
    SWIGLU = "swiglu"
    GEGGLU = "gegglu"
    REGLU = "reglu"
    SOFTMAX = "softmax"

@dataclass
class YvConfig:
    """Dataclass encapsulating PiscesL1 model configuration parameters.

    Attributes:
        model_type (str): Human-readable model identifier. Defaults to ``"piscesl1"``.
        vocab_size (int): Token vocabulary size. Defaults to ``151646`` (Qwen3).
        hidden_size (int): Transformer hidden dimension. Defaults to ``2048``.
        n_layer (int): Number of transformer layers. Defaults to ``24``.
        n_head (int): Number of attention heads. Defaults to ``16``.
        n_kv_head (int): Number of key-value heads for grouped attention. Defaults to ``4``.
        moe_num_experts (int): Total experts for Mixture-of-Experts blocks. Defaults to ``64``.
        moe_top_k (int): Number of activated experts per token. Defaults to ``2``.
        moe_capacity_factor (float): Routing capacity multiplier. Defaults to ``1.0``.
        moe_load_balance_alpha (float): Coefficient for load-balancing loss. Defaults to ``0.01``.
        moe_noise_std (float): Standard deviation of routing noise. Defaults to ``0.1``.
        moe_use_stable_gate (bool): Whether to use a stabilized MoE gate. Defaults to ``True``.
        moe_min_capacity (int): Minimum routing capacity per expert. Defaults to ``4``.
        moe_prediction_horizon (int): Horizon length for predictive capacity tuning. Defaults to ``8``.
        intermediate_size (int): Transformer feed-forward hidden size. Defaults to ``5632``.
        max_position_embeddings (int): Maximum positional embeddings. Defaults to ``8192``.
        rope_theta (float): Base theta parameter for RoPE. Defaults to ``1e6``.
        dropout (float): Dropout probability applied throughout the model. Defaults to ``0.0``.
        image_res (int): Default input image resolution. Defaults to ``224``.
        max_image_res (int): Maximum supported image resolution. Defaults to ``1024``.
        image_patch (int): Image patch size for vision encoder. Defaults to ``14``.
        use_native_resolution (bool): Whether to keep original image resolution. Defaults to ``True``.
        enable_patch_pack (bool): Whether to enable patch packing. Defaults to ``True``.
        mm_tokens (int): Number of multimodal tokens. Defaults to ``256``.
        audio_tokens (int): Number of audio tokens. Defaults to ``512``.
        task_classes (int): Number of classification tasks. Defaults to ``256``.
        eval_dims (int): Evaluation dimension cardinality. Defaults to ``7``.
        rope_scaling (Dict[str, Any]): RoPE scaling configuration; defaults to YaRN scaling.
        fusion_quality_threshold (float): Quality threshold for modality inclusion. Defaults to 0.3.
        fusion_dropout (float): Dropout for fusion layers. Defaults to 0.1.
        modal_token_count (int): Number of fused multimodal tokens to prepend when fusion returns [B, H]. Defaults to 8.
        enable_cognitive_density (bool): Whether to enable cognitive density optimization for all 64 experts. Defaults to True.
        enable_dynamic_capacity (bool): Whether to enable dynamic capacity scaling for balanced loading. Defaults to True.
        cognitive_enhancement_scale (float): Scale factor for cognitive enhancement. Defaults to 0.1.
        expert_temperature_max (float): Maximum routing temperature for exploration. Defaults to 5.0.
        expert_load_balance_threshold (float): Threshold for load imbalance warnings. Defaults to 0.15.
        expert_init_method (str): Expert initialization method ('hybrid', 'random', 'cluster'). Defaults to 'hybrid'.
        diversity_weight (float): Weight for diversity loss in knowledge density optimization. Defaults to 0.01.
        mi_weight (float): Weight for mutual information loss in knowledge density optimization. Defaults to 0.1.
        online_clustering (bool): Whether to use online clustering for routing. Defaults to False.
        orthogonality_weight (float): Weight for orthogonality loss in expert specialization. Defaults to 0.01.
        routing_entropy_weight (float): Weight for routing entropy loss. Defaults to 0.001.
        activation_variance_weight (float): Weight for activation variance loss. Defaults to 0.01.
        expert_warmup_steps (int): Number of warmup steps for expert training. Defaults to 100.
        auto_detect_clusters (bool): Whether to automatically detect the number of clusters. Defaults to True.
        min_clusters (int): Minimum number of clusters for knowledge clustering. Defaults to 4.
        max_clusters (int): Maximum number of clusters for knowledge clustering. Defaults to 16.
        use_3d_spatio_temporal_rope (bool): Whether to enable 3D spatio-temporal RoPE for video frames. Defaults to False.
        max_temporal_frames (int): Maximum number of temporal frames for 3D RoPE. Defaults to 64.
        attention_type (str): Type of attention, options: "standard", "streaming_llm", "h2o_attention". Defaults to "standard".
        use_h2o_attention (bool): Whether to enable H2O attention. Defaults to True.
        streaming_window (int): Window size for streaming attention. Defaults to 16384.
        compression_ratio (int): Compression ratio for H2O attention. Defaults to 8.
        use_sliding_window (bool): Whether to enable sliding window attention for long contexts. Defaults to False.
        long_factor (int): Long context scaling factor. Defaults to 32.
        max_cache_size (int): Maximum number of tokens kept in the KV cache (paged). Defaults to 8192.
        cache_quantization (bool): Whether to enable KV cache quantization. Defaults to True.
        kv_cache_block_size (int): Paged KV block size. Defaults to 512.
        sdpa_prefer_flash (bool): Whether to prefer FlashAttention backend for Scaled Dot-Product Attention (SDPA) when available. Defaults to True.
        speculative_candidates (int): Number of candidate tokens for speculative decoding. Defaults to 4.
        speculative_draft_length (int): Length of the draft sequence. Defaults to 5.
        speculative_acceptance_threshold (float): Threshold for accepting draft tokens. Defaults to 0.8.
        speculative_temperature (float): Temperature for speculative sampling. Defaults to 0.7.
        speculative_top_k (int): Top-k for speculative sampling. Defaults to 50.
        speculative_top_p (float): Top-p for speculative sampling. Defaults to 0.9.
        enable_speculative_decoding (bool): Whether to enable speculative decoding. Defaults to True.
        tool_uncertainty_threshold (float): Trigger tools when uncertainty exceeds this value. Defaults to 0.7.
        tool_fact_consistency_threshold (float): Trigger tools when fact consistency is below this value. Defaults to 0.6.
        enable_debug_outputs (bool): If True, model.forward returns a 'debug' section with shapes and data types. Defaults to False.
        debug_verbose (bool): If True, include extra debug information like modality presence and fusion shapes. Defaults to False.
    """
    model_type: str = "piscesl1"
    vocab_size: int = 151646
    hidden_size: int = 2048
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 4
    head_dim: Optional[int] = None

    moe_num_experts: int = 64
    moe_top_k: int = 2
    moe_capacity_factor: float = 1.0
    moe_load_balance_alpha: float = 0.01
    moe_noise_std: float = 0.1
    moe_use_stable_gate: bool = True
    moe_min_capacity: int = 4
    moe_prediction_horizon: int = 8
    moe_expert_grad_clip: float = 0.1
    moe_z_loss_alpha: float = 1e-4
    moe_random_to_gradient_steps: int = 500
    moe_gate_warmup_alpha: float = 0.05
    moe_attention_mamba_temp: float = 0.3
    moe_l2_smooth_8k: float = 0.01
    moe_layers: List[int] = field(default_factory=list)
    moe_shared_experts: int = 0
    moe_expert_parallel: bool = False
    moe_token_dispatcher: str = "allgather"

    intermediate_size: int = 5632
    max_position_embeddings: int = 8192
    rope_theta: float = 1e6
    dropout: float = 0.0
    image_res: int = 224
    max_image_res: int = 1024
    image_patch: int = 14
    use_native_resolution: bool = True
    enable_patch_pack: bool = True
    mm_tokens: int = 256
    audio_tokens: int = 512
    task_classes: int = 256
    eval_dims: int = 7
    rope_scaling: dict = field(default_factory=lambda: {"type": "yarn", "factor": 32, "original_max_position_embeddings": 32768})

    residual_dropout_p: float = 0.1
    use_gradient_checkpointing: bool = True
    use_pre_norm: bool = True
    attention_dropout: float = 0.0
    fused_qkv: bool = False
    use_parallel_attention_mlp: bool = False
    use_deepnorm: bool = False
    use_layerscale: bool = False
    layerscale_init: float = 1e-5

    # ========================================
    # Algorithm Optimization Configuration
    # ========================================
    use_qk_norm: bool = True
    label_smoothing: float = 0.1
    learnable_attention_scale: bool = True
    layer_scale_init: float = 1e-5
    depth_aware_init: bool = True
    residual_alpha: float = 2.0
    embedding_norm_weight: float = 0.01

    use_attn_res: bool = False
    attn_res_block_size: int = 8
    attn_res_use_two_phase: bool = True
    attn_res_use_online_softmax: bool = True
    attn_res_cache_pipeline: bool = True
    attn_res_max_blocks: int = 32
    attn_res_learnable_query: bool = True
    attn_res_use_rmsnorm: bool = True

    enable_dynamic_fusion: bool = True
    fusion_quality_threshold: float = 0.3
    fusion_dropout: float = 0.1
    modal_token_count: int = 8
    use_enhanced_fusion: bool = False

    enable_cognitive_density: bool = True
    enable_dynamic_capacity: bool = True
    cognitive_enhancement_scale: float = 0.1
    expert_temperature_max: float = 5.0
    expert_load_balance_threshold: float = 0.15

    expert_init_method: str = "hybrid"
    diversity_weight: float = 0.01
    mi_weight: float = 0.1
    online_clustering: bool = False
    orthogonality_weight: float = 0.01
    routing_entropy_weight: float = 0.001
    activation_variance_weight: float = 0.01
    expert_warmup_steps: int = 100
    auto_detect_clusters: bool = True
    min_clusters: int = 4
    max_clusters: int = 16

    use_3d_spatio_temporal_rope: bool = False
    max_temporal_frames: int = 64

    attention_type: str = "standard"
    use_h2o_attention: bool = True
    streaming_window: int = 16384
    compression_ratio: int = 8
    use_sliding_window: bool = False
    sliding_window_size: int = 4096
    long_factor: int = 32
    max_cache_size: int = 8192
    cache_quantization: bool = True
    kv_cache_block_size: int = 512
    sdpa_prefer_flash: bool = True
    use_flash_attention: bool = True
    flash_attention_version: int = 2

    speculative_candidates: int = 4
    speculative_draft_length: int = 5
    speculative_acceptance_threshold: float = 0.8
    speculative_temperature: float = 0.7
    speculative_top_k: int = 50
    speculative_top_p: float = 0.9
    enable_speculative_decoding: bool = True
    speculative_tree_width: int = 4
    speculative_tree_depth: int = 5

    tool_uncertainty_threshold: float = 0.7
    tool_fact_consistency_threshold: float = 0.6

    enable_debug_outputs: bool = False
    debug_verbose: bool = False

    use_mamba3: bool = False
    mamba3_layers: List[int] = field(default_factory=list)
    mamba3_d_state: int = 128
    mamba3_d_conv: int = 4
    mamba3_expand: int = 2
    mamba3_dt_rank: str = "auto"
    mamba3_conv_bias: bool = True
    mamba3_proj_bias: bool = False
    mamba3_use_fast_path: bool = True
    mamba3_layer_norm_eps: float = 1e-4
    mamba3_sequence_threshold: int = 8192
    mamba3_gate_mode: str = "adaptive"
    mamba3_gate_init: float = 0.5
    mamba3_gate_temperature: float = 1.0
    mamba3_complex_state: bool = True
    mamba3_trapezoidal: bool = True
    mamba3_mimo: bool = True
    mamba3_dropout: float = 0.0
    mamba3_chunk_size: int = 256
    mamba3_use_v_kernel: bool = True
    mamba3_use_ss_duality: bool = True
    mamba3_use_adaptive_dt: bool = True
    mamba3_use_bidirectional: bool = False
    mamba3_use_gated: bool = True
    mamba3_n_heads: int = 8
    mamba3_use_flash_ssm: bool = True
    mamba3_use_rms_norm: bool = True

    dsa_sparse_ratio: float = 0.3
    dsa_importance_threshold: float = 0.1
    dsa_use_dynamic: bool = True

    use_hisa_attention: bool = False
    hisa_block_size: int = 64
    hisa_superblock_size: int = 512
    hisa_local_ratio: float = 0.4
    hisa_block_ratio: float = 0.3

    thinking_intensity: float = 0.5
    complexity_threshold_low: float = 0.3
    complexity_threshold_high: float = 0.7

    swarm_intensity: float = 0.5
    num_swarm_agents: int = 4

    flagship_level: float = 0.5

    galore_enabled: bool = False
    galore_rank: int = 128
    galore_update_interval: int = 200
    galore_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    galore_lr_ratio: float = 1.0
    galore_min_rank: int = 32
    galore_max_rank: int = 512
    galore_rank_adapt_interval: int = 1000
    galore_rank_adapt_threshold: float = 0.1
    galore_quantization_bits: int = 0
    galore_memory_efficient: bool = True
    galore_moe_expert_only: bool = False
    galore_multimodal_modules: List[str] = field(default_factory=lambda: ["vision_encoder", "audio_encoder", "multimodal_fusion"])
    galore_sequence_threshold: int = 4096
    galore_gradient_accumulation_sync: bool = True

    # FP4 Training Configuration (75% memory savings vs BF16)
    use_fp4: bool = False
    fp4_block_size: int = 16
    fp4_stochastic_rounding: bool = True
    fp4_master_weights_dtype: str = "fp32"
    
    # COAT FP8 Enhancement (1.54x memory savings, +43% speed)
    coat_enabled: bool = True
    coat_amax_epsilon: float = 1e-3
    coat_scale_factor: float = 1.0
    mixed_grain_activation: bool = True
    per_tensor_threshold: int = 1024
    per_group_size: int = 128
    
    # Q-GaLore (89.5% optimizer memory reduction)
    use_int4_projection: bool = True
    use_int8_weights: bool = True
    adaptive_rank_update: bool = True
    convergence_threshold: float = 0.01
    
    # GRASS Structured Sparsity (supports large models, +100% throughput)
    structured_sparsity: bool = True
    grass_block_size: int = 32
    gradient_compression_ratio: float = 0.1
    
    # Adacc Adaptive Recomputation (60-80% activation memory savings)
    adaptive_recomputation: bool = True
    compute_cost_threshold: float = 0.5
    activation_size_threshold: int = 1048576
    
    # TERAIO Offloading (supports ultra-large models)
    enable_teraio: bool = False
    gpu_memory_budget: int = 42949672960
    cpu_memory_budget: int = 137438953472
    enable_gds: bool = True

    chinchilla_optimal: bool = False
    chinchilla_c_budget: float = 0.0
    chinchilla_d_ratio: float = 1.0

    use_mla: bool = True
    kv_lora_rank: int = 512
    mla_q_lora_rank: Optional[int] = None
    
    # ========================================
    # Lazy Initialization Configuration
    # ========================================
    lazy_init_enabled: bool = False
    lazy_init_vision_encoder: bool = True
    lazy_init_audio_encoder: bool = True
    lazy_init_video_encoder: bool = True
    lazy_init_doc_encoder: bool = True
    lazy_init_modal_fusion: bool = False
    lazy_init_reasoner: bool = False
    lazy_init_speculative_decoder: bool = True

    # ========================================
    # Flagship Algorithm Integration (2025-2026)
    # ========================================

    # EG-MLA: Embedding-Gated Multi-Head Latent Attention
    use_eg_mla: bool = False
    eg_mla_compression_ratio: float = 0.916

    # DuoAttention: Retrieval vs Streaming Heads
    use_duo_attention: bool = False
    duo_attention_retrieval_ratio: float = 0.2
    duo_attention_buffer_size: int = 1024

    # Test-Time Training (TTT-E2E)
    use_ttt_e2e: bool = False
    ttt_update_layers: int = 2
    ttt_learning_rate: float = 1e-5
    ttt_max_steps: int = 5
    ttt_confidence_threshold: float = 0.6
    ttt_complexity_threshold: float = 0.7

    # Expert Evolution
    use_expert_evolution: bool = False
    expert_evolution_base_lr: float = 1e-5
    expert_evolution_decay: float = 0.99

    # EWC: Elastic Weight Consolidation
    use_ewc: bool = False
    ewc_lambda: float = 1000.0

    # SEAL: Self-Adapting LLMs
    use_seal: bool = False
    seal_confidence_threshold: float = 0.85
    seal_max_synthetic_samples: int = 100

    # DAPO: Decoupled Clipping Policy Optimization
    use_dapo: bool = False
    dapo_epsilon_low: float = 0.2
    dapo_epsilon_high: float = 0.4
    dapo_diversity_threshold: float = 0.3

    # Verification: CRV + OTV + ARES
    use_crv_verification: bool = False
    use_otv_verification: bool = False
    use_ares_verification: bool = False
    otv_quality_threshold: float = 0.6

    # SyncFusion: Audio-Video Synchronous Understanding
    use_sync_fusion: bool = False
    sync_fusion_temporal_bins: int = 16

    # Coupled Mamba Fusion
    use_coupled_mamba_fusion: bool = False
    coupled_mamba_coupling_strength: float = 0.3

    # SparseSSM: Training-Free Mamba Pruning
    use_sparse_ssm: bool = False
    sparse_ssm_ratio: float = 0.5

    # Gated Delta Networks
    use_gated_delta: bool = False

    # OOMB: Million-Token Context
    use_oomb_context: bool = False
    oomb_chunk_size: int = 32768
    oomb_max_context: int = 4194304

    # REFORM: Compress-Gather-Recompute
    use_reform: bool = False
    reform_compression_ratio: int = 4
    reform_importance_threshold: float = 0.1

    # Quartet: End-to-End FP4 Training
    use_quartet: bool = False

    # Long Context Extensions
    max_context_length: int = 1048576  # 1M default, extensible to 4M+
    mla_rope_scaling_factor: float = 1.0

    num_mtp_heads: int = 4
    mtp_loss_weight: float = 0.5
    mtp_share_embeddings: bool = True

    use_rotary_pos_emb: bool = True
    use_alibi: bool = False
    alibi_num_heads: int = 16

    activation_type: str = "silu"
    use_swiglu: bool = True
    use_geglu: bool = False

    initializer_range: float = 0.02
    use_scaled_init: bool = True

    layer_norm_eps: float = 1e-6
    use_rms_norm: bool = True

    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    unk_token_id: int = 3

    tie_word_embeddings: bool = False

    use_cache: bool = True

    quantization_config: Optional[Dict[str, Any]] = None
    lora_config: Optional[Dict[str, Any]] = None

    image_gen_resolution: int = 256
    image_gen_num_steps: int = 50
    image_gen_guidance_scale: float = 7.5
    image_gen_initial_size: int = 8
    image_gen_use_attention: bool = True
    image_gen_use_residual: bool = True
    image_gen_use_diffusion: bool = False

    audio_gen_sample_rate: int = 16000
    audio_gen_n_mels: int = 128
    audio_gen_duration: float = 5.0
    audio_gen_num_steps: int = 100
    audio_gen_streaming: bool = True
    audio_gen_num_codebooks: int = 2
    audio_gen_codebook_size: int = 4096

    video_gen_fps: int = 24
    video_gen_num_frames: int = 16
    video_gen_resolution: int = 256

    generation_max_tokens: int = 1024
    generation_temperature: float = 1.0
    generation_top_p: float = 0.95
    generation_top_k: int = 50

    ink_optimizer_enabled: bool = True
    ink_momentum_bits: int = 8
    ink_variance_bits: int = 4
    ink_sparse_ratio: float = 0.01
    ink_gradient_bits: int = 8
    ink_kv_cache_bits: int = 8
    ink_max_experts_on_gpu: int = 4
    ink_checkpoint_ratio: float = 0.5
    ink_momentum_block_size: int = 128
    ink_variance_block_size: int = 256
    ink_sparse_warmup_steps: int = 1000
    ink_sparse_adaptive: bool = True

    cpu_offload_optimizer: bool = False
    cpu_offload_weights: bool = False
    cpu_offload_gradients: bool = False
    activation_quantization: bool = False
    activation_quant_bits: int = 8
    activation_quant_block_size: int = 128

    extreme_memory_mode: bool = False
    ultra_low_memory: bool = False
    memory_efficient_attention: bool = True
    gradient_compression_ratio: float = 0.1

    vram_offload_optimizer: bool = False
    vram_offload_weights: bool = False
    vram_offload_gradients: bool = False
    vram_offload_activations: bool = False
    vram_offload_kv_cache: bool = False
    vram_max_experts_on_gpu: int = 4
    vram_dynamic_expert_loading: bool = True
    vram_expert_lru_cache_size: int = 8
    vram_activation_checkpointing: bool = True
    vram_selective_checkpointing: bool = True
    vram_flash_attention: bool = True
    vram_gradient_checkpointing: bool = True
    vram_kv_cache_quantization: bool = True
    vram_weight_quantization: bool = False
    vram_weight_quant_bits: int = 4
    vram_optimizer_state_quantization: bool = True
    vram_optimizer_state_bits: int = 8
    vram_mixed_precision: str = "bf16"
    vram_fp4_training: bool = False
    vram_fp8_attention: bool = False
    vram_sequence_parallel: bool = False
    vram_tensor_parallel: int = 1
    vram_pipeline_parallel: int = 1
    vram_zero_stage: int = 3
    vram_cpu_pin_memory: bool = True
    vram_cpu_prefetch: bool = True
    vram_async_transfer: bool = True
    vram_peak_memory_limit: int = 0
    extreme_vram_mode: bool = False
    ultra_low_vram: bool = False

    modal_aware_routing: bool = True
    n_cross_modal_experts: int = 0
    modal_affinity_alpha: float = 1.0
    n_modalities: int = 7

    use_recurrent_modal_refiner: bool = True
    rdt_max_loops: int = 3
    rdt_spectral_radius: float = 0.95
    rdt_convergence_threshold: float = 0.99
    rdt_refine_heads: int = 2
    rdt_refine_ffn_ratio: float = 1.0

    modal_protection_mod: bool = True

    use_ultra_sparse_gate: bool = False
    ultra_sparse_tier1_threshold: float = 0.3
    ultra_sparse_tier2_threshold: float = 0.8
    ultra_sparse_tier1_topk: int = 1
    ultra_sparse_tier2_topk: int = 2
    ultra_sparse_tier3_topk: int = 4

    use_circulant_attention: bool = True
    circulant_threshold: int = 4096

    use_rdt_layers: bool = True
    rdt_layer_indices: List[int] = field(default_factory=list)
    rdt_loops_per_layer: int = 2

    def __post_init__(self):
        """Initialize computed fields after dataclass construction.
        
        This method is automatically called after the dataclass is initialized.
        It computes derived values and converts enum types to their string values.
        
        Side Effects:
            - Sets head_dim to hidden_size // n_head if not specified
            - Converts YvAttentionType enum to string value
            - Converts YvActivationType enum to string value
        """
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.n_head

        if isinstance(self.attention_type, YvAttentionType):
            self.attention_type = self.attention_type.value

        if isinstance(self.activation_type, YvActivationType):
            self.activation_type = self.activation_type.value

        if hasattr(self, 'flagship_level') and self.flagship_level != 0.5:
            level = self.flagship_level
            
            self.dsa_sparse_ratio = 0.3 * (0.5 + level)
            self.thinking_intensity = 0.5 * (0.5 + level)
            self.swarm_intensity = 0.5 * (0.5 + level)

    @classmethod
    def from_json(cls, path: str) -> 'YvConfig':
        """Load configuration from a JSON file.
        
        Reads a JSON configuration file and creates a YvConfig instance.
        Unknown fields in the JSON are silently ignored.
        
        Args:
            path (str): Path to the JSON configuration file.
        
        Returns:
            YvConfig: Configuration instance loaded from file.
        
        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        
        Example:
            >>> config = YvConfig.from_json("model_config.json")
        """
        with open(path, 'r') as f:
            config_data = json.load(f)

        model_fields = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config_data.items() if k in model_fields}

        return cls(**filtered_config)

    @classmethod
    def from_yaml(cls, path: str) -> 'YvConfig':
        """Load configuration from a YAML file.
        
        Reads a YAML configuration file and creates a YvConfig instance.
        Unknown fields in the YAML are silently ignored.
        
        Args:
            path (str): Path to the YAML configuration file.
        
        Returns:
            YvConfig: Configuration instance loaded from file.
        
        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the file contains invalid YAML.
        
        Example:
            >>> config = YvConfig.from_yaml("model_config.yaml")
        """
        with open(path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}

        # Replace {{VERSION}} placeholder with actual version
        if "version" in config_data and config_data["version"] == "{{VERSION}}":
            config_data["version"] = VERSION

        model_fields = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config_data.items() if k in model_fields}

        return cls(**filtered_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'YvConfig':
        """Create configuration from a dictionary.
        
        Creates a YvConfig instance from a dictionary, filtering out
        unknown fields.
        
        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration
                parameters. Keys should match dataclass field names.
        
        Returns:
            YvConfig: Configuration instance from dictionary.
        
        Example:
            >>> config = YvConfig.from_dict({"hidden_size": 4096, "n_layer": 32})
        """
        model_fields = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in model_fields}

        return cls(**filtered_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of all configuration
                parameters, suitable for serialization.
        
        Example:
            >>> config_dict = config.to_dict()
            >>> print(config_dict['hidden_size'])
        """
        return asdict(self)

    def to_json(self, path: str):
        """Save configuration to a JSON file.
        
        Serializes the configuration to a JSON file with pretty formatting.
        
        Args:
            path (str): Output file path for the JSON configuration.
        
        Example:
            >>> config.to_json("output_config.json")
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_yaml(self, path: str):
        """Save configuration to a YAML file.
        
        Serializes the configuration to a YAML file.
        
        Args:
            path (str): Output file path for the YAML configuration.
        
        Example:
            >>> config.to_yaml("output_config.yaml")
        """
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def update(self, **kwargs):
        """Update configuration parameters.
        
        Modifies configuration parameters in-place. Raises an error if
        attempting to set an unknown parameter.
        
        Args:
            **kwargs: Keyword arguments where keys are parameter names
                and values are new values.
        
        Returns:
            YvConfig: Self for method chaining.
        
        Raises:
            ValueError: If an unknown parameter name is provided.
        
        Example:
            >>> config.update(hidden_size=4096, n_layer=32)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self

    def copy(self) -> 'YvConfig':
        """Create a deep copy of the configuration.
        
        Returns:
            YvConfig: A new configuration instance with identical values.
        
        Example:
            >>> new_config = config.copy()
            >>> new_config.hidden_size = 8192  # Doesn't affect original
        """
        return copy.deepcopy(self)

    def validate(self) -> bool:
        """Validate configuration parameters.
        
        Checks that configuration parameters satisfy all constraints required
        for the model to function correctly.
        
        Returns:
            bool: True if validation passes.
        
        Raises:
            ValueError: If any constraint is violated:
                - hidden_size must be divisible by n_head
                - n_head must be divisible by n_kv_head
                - moe_top_k cannot exceed moe_num_experts
                - intermediate_size must be even for SwiGLU/GeGLU
        
        Example:
            >>> try:
            ...     config.validate()
            ...     print("Configuration is valid")
            ... except ValueError as e:
            ...     print(f"Invalid configuration: {e}")
        """
        if self.hidden_size % self.n_head != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by n_head ({self.n_head})")

        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")

        if self.moe_top_k > self.moe_num_experts:
            raise ValueError(f"moe_top_k ({self.moe_top_k}) cannot exceed moe_num_experts ({self.moe_num_experts})")

        if self.intermediate_size % 2 != 0 and (self.use_swiglu or self.use_geglu):
            raise ValueError(f"intermediate_size ({self.intermediate_size}) must be even for SwiGLU/GeGLU")

        return True

    def get_head_dim(self) -> int:
        """Get the dimension of each attention head.
        
        Returns:
            int: Head dimension, either the explicitly set value or
                hidden_size // n_head.
        
        Example:
            >>> head_dim = config.get_head_dim()
            >>> print(f"Each head has dimension {head_dim}")
        """
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_size // self.n_head

    def get_num_kv_heads(self) -> int:
        """Get the number of key-value heads for grouped-query attention.
        
        Returns:
            int: Number of KV heads, either n_kv_head or n_head if not set.
        
        Example:
            >>> kv_heads = config.get_num_kv_heads()
            >>> print(f"Using {kv_heads} key-value heads")
        """
        return self.n_kv_head if self.n_kv_head is not None else self.n_head

    def get_num_groups(self) -> int:
        """Get the number of attention groups for grouped-query attention.
        
        The number of groups determines how many query heads share the same
        key-value head.
        
        Returns:
            int: Number of groups (n_head // n_kv_head).
        
        Example:
            >>> groups = config.get_num_groups()
            >>> print(f"Each KV head serves {groups} query heads")
        """
        return self.n_head // self.get_num_kv_heads()

    def get_intermediate_size(self) -> int:
        """Get the effective intermediate size for FFN.
        
        For gated activations (SwiGLU, GeGLU), the intermediate size is
        doubled to account for the gating mechanism.
        
        Returns:
            int: Effective intermediate size.
        
        Example:
            >>> ffn_size = config.get_intermediate_size()
            >>> print(f"FFN hidden size: {ffn_size}")
        """
        if self.use_swiglu or self.use_geglu:
            return self.intermediate_size * 2
        return self.intermediate_size

    @classmethod
    def get_small_config(cls) -> 'YvConfig':
        """Get a small model configuration preset.
        
        Suitable for experimentation and debugging. Uses smaller dimensions
        and fewer experts for faster training and inference.
        
        Returns:
            YvConfig: Small configuration with:
                - 768 hidden size
                - 12 layers
                - 12 attention heads
                - 8 experts
                - 2048 max positions
        
        Example:
            >>> config = YvConfig.get_small_config()
        """
        return cls(
            hidden_size=768,
            n_layer=12,
            n_head=12,
            n_kv_head=12,
            intermediate_size=2048,
            moe_num_experts=8,
            moe_top_k=2,
            max_position_embeddings=2048
        )

    @classmethod
    def get_base_config(cls) -> 'YvConfig':
        """Get the base model configuration preset.
        
        The standard configuration for most use cases, balancing model
        capacity with computational efficiency.
        
        Returns:
            YvConfig: Base configuration with:
                - 2048 hidden size
                - 24 layers
                - 16 attention heads (4 KV heads)
                - 64 experts
                - 8192 max positions
        
        Example:
            >>> config = YvConfig.get_base_config()
        """
        return cls(
            hidden_size=2048,
            n_layer=24,
            n_head=16,
            n_kv_head=4,
            intermediate_size=5632,
            moe_num_experts=64,
            moe_top_k=2,
            max_position_embeddings=8192
        )

    @classmethod
    def get_large_config(cls) -> 'YvConfig':
        """Get a large model configuration preset.
        
        Suitable for production deployments requiring higher model capacity
        and better performance on complex tasks.
        
        Returns:
            YvConfig: Large configuration with:
                - 4096 hidden size
                - 32 layers
                - 32 attention heads (8 KV heads)
                - 128 experts
                - 16384 max positions
        
        Example:
            >>> config = YvConfig.get_large_config()
        """
        return cls(
            hidden_size=4096,
            n_layer=32,
            n_head=32,
            n_kv_head=8,
            intermediate_size=11008,
            moe_num_experts=128,
            moe_top_k=4,
            max_position_embeddings=16384
        )

    @classmethod
    def get_xl_config(cls) -> 'YvConfig':
        """Get an extra-large model configuration preset.
        
        Maximum capacity configuration for demanding applications requiring
        the highest model performance.
        
        Returns:
            YvConfig: XL configuration with:
                - 6144 hidden size
                - 48 layers
                - 48 attention heads (8 KV heads)
                - 256 experts
                - 32768 max positions
        
        Example:
            >>> config = YvConfig.get_xl_config()
        """
        return cls(
            hidden_size=6144,
            n_layer=48,
            n_head=48,
            n_kv_head=8,
            intermediate_size=16384,
            moe_num_experts=256,
            moe_top_k=8,
            max_position_embeddings=32768
        )

    @classmethod
    def get_hybrid_config(cls) -> 'YvConfig':
        """Get a hybrid Attention-Mamba configuration preset.
        
        Combines attention layers with Mamba-3 SSM layers for efficient
        long-context processing with linear complexity.
        
        Returns:
            YvConfig: Hybrid configuration with:
                - 4096 hidden size
                - 32 layers (every 4th layer is Mamba-3)
                - 64 experts
                - 16384 max positions
                - Mamba-3 SSM enabled
        
        Example:
            >>> config = YvConfig.get_hybrid_config()
        """
        config = cls(
            hidden_size=4096,
            n_layer=32,
            n_head=32,
            n_kv_head=8,
            intermediate_size=11008,
            moe_num_experts=64,
            moe_top_k=4,
            max_position_embeddings=16384,
            use_mamba3=True,
            mamba3_layers=[i for i in range(32) if i % 4 == 0]
        )
        return config

    @classmethod
    def get_jamba_style_config(cls) -> 'YvConfig':
        """Get a Jamba-style MoE-Mamba configuration preset.
        
        Inspired by the Jamba architecture, this configuration combines
        Mixture-of-Experts with Mamba-3 SSM layers for extremely long
        context processing (up to 262K tokens).
        
        Returns:
            YvConfig: Jamba-style configuration with:
                - 4096 hidden size
                - 32 layers (layers 1,2,5,6,9,10,... are Mamba-3)
                - 16 experts (smaller for efficiency)
                - 262144 max positions (256K context)
                - Mamba-3 SSM enabled
        
        Example:
            >>> config = YvConfig.get_jamba_style_config()
        """
        config = cls(
            hidden_size=4096,
            n_layer=32,
            n_head=32,
            n_kv_head=8,
            intermediate_size=11008,
            moe_num_experts=16,
            moe_top_k=2,
            max_position_embeddings=262144,
            use_mamba3=True,
            mamba3_layers=[i for i in range(32) if i % 4 in [1, 2]]
        )
        return config

    @classmethod
    def get_extreme_memory_config(cls, base_config: Optional['YvConfig'] = None) -> 'YvConfig':
        """Get extreme VRAM optimization configuration preset.
        
        Applies maximum VRAM optimization settings for training large models
        on limited GPU memory. This configuration enables all available VRAM
        saving techniques including INT8/INT4 compression, sparse gradients,
        CPU offloading, and activation quantization.
        
        VRAM Savings:
            - FP4 weights: 75% vs BF16
            - INT8 momentum: 4x compression
            - INT4 variance: 8x compression
            - Sparse gradients: 100x (top 1%)
            - CPU offload: Offloads optimizer states to CPU RAM
            - Activation quantization: 4x compression
            - MLA KV cache: 80%+ compression
            - Total: 70-85% VRAM reduction
        
        Args:
            base_config: Optional base configuration to apply VRAM optimizations to.
                        If None, uses get_base_config() as the starting point.
        
        Returns:
            YvConfig: Configuration with extreme VRAM optimizations enabled.
        
        Example:
            >>> # Apply to default base config
            >>> config = YvConfig.get_extreme_memory_config()
            >>> 
            >>> # Apply to specific model size
            >>> base = YvConfig.get_large_config()
            >>> config = YvConfig.get_extreme_memory_config(base)
        """
        if base_config is None:
            config = cls.get_base_config()
        else:
            config = base_config.copy()
        
        config.use_fp4 = True
        config.fp4_block_size = 16
        config.fp4_stochastic_rounding = True
        
        config.galore_enabled = True
        config.galore_memory_efficient = True
        config.galore_quantization_bits = 8
        
        config.ink_optimizer_enabled = True
        config.ink_momentum_bits = 8
        config.ink_variance_bits = 4
        config.ink_sparse_ratio = 0.01
        config.ink_gradient_bits = 8
        config.ink_kv_cache_bits = 8
        config.ink_checkpoint_ratio = 0.5
        
        config.cpu_offload_optimizer = True
        config.cpu_offload_weights = False
        config.activation_quantization = True
        config.activation_quant_bits = 8
        
        config.extreme_memory_mode = True
        config.memory_efficient_attention = True
        config.gradient_compression_ratio = 0.1
        
        config.use_mla = True
        config.cache_quantization = True
        config.use_gradient_checkpointing = True
        
        config.vram_offload_optimizer = True
        config.vram_offload_weights = False
        config.vram_offload_gradients = False
        config.vram_offload_activations = False
        config.vram_offload_kv_cache = False
        config.vram_max_experts_on_gpu = 4
        config.vram_dynamic_expert_loading = True
        config.vram_activation_checkpointing = True
        config.vram_flash_attention = True
        config.vram_gradient_checkpointing = True
        config.vram_kv_cache_quantization = True
        config.vram_optimizer_state_quantization = True
        config.vram_optimizer_state_bits = 8
        config.vram_mixed_precision = "bf16"
        config.vram_zero_stage = 3
        config.extreme_vram_mode = True
        
        return config

    @classmethod
    def get_ultra_low_memory_config(cls, base_config: Optional['YvConfig'] = None) -> 'YvConfig':
        """Get ultra-low VRAM configuration for extreme constraints.
        
        Maximum VRAM optimization for training on very limited hardware.
        This configuration sacrifices some performance for maximum VRAM savings.
        
        VRAM Savings:
            - All optimizations from get_extreme_memory_config()
            - CPU weight offloading: Additional 50-70% GPU VRAM savings
            - INT4 activation quantization: 8x compression
            - Ultra-sparse gradients: top 0.5%
            - Minimum experts on GPU: 2
            - KV cache offload to CPU
        
        Args:
            base_config: Optional base configuration to apply optimizations to.
        
        Returns:
            YvConfig: Configuration with ultra-low VRAM settings.
        
        Example:
            >>> config = YvConfig.get_ultra_low_memory_config()
        """
        config = cls.get_extreme_memory_config(base_config)
        
        config.ink_variance_bits = 4
        config.ink_sparse_ratio = 0.005
        config.ink_max_experts_on_gpu = 2
        
        config.cpu_offload_weights = True
        config.cpu_offload_gradients = True
        config.activation_quant_bits = 4
        
        config.ultra_low_memory = True
        config.gradient_compression_ratio = 0.05
        
        config.vram_offload_weights = True
        config.vram_offload_gradients = True
        config.vram_offload_activations = True
        config.vram_offload_kv_cache = True
        config.vram_max_experts_on_gpu = 2
        config.vram_weight_quantization = True
        config.vram_weight_quant_bits = 4
        config.vram_fp4_training = True
        config.ultra_low_vram = True
        
        return config

    @classmethod
    def get_extreme_vram_config(cls, base_config: Optional['YvConfig'] = None) -> 'YvConfig':
        """Get extreme VRAM optimization configuration for GPU memory constraints.
        
        This is the primary method for VRAM optimization, specifically designed
        for training large models on consumer GPUs with limited VRAM.
        
        VRAM Optimization Techniques:
            1. Weight Quantization: FP4/INT4 reduces weight VRAM by 75-87.5%
            2. Optimizer State Quantization: INT8 reduces optimizer VRAM by 75%
            3. Gradient Checkpointing: Recompute activations, saves 50-70%
            4. KV Cache Quantization: INT8 reduces cache VRAM by 75%
            5. MLA (Multi-Head Latent Attention): 80%+ KV compression
            6. Flash Attention: Memory-efficient attention computation
            7. Dynamic Expert Loading: Only load active experts to GPU
            8. ZeRO-3: Shard optimizer states across devices
            9. CPU Offloading: Move non-active data to CPU RAM
        
        Args:
            base_config: Optional base configuration to apply optimizations to.
        
        Returns:
            YvConfig: Configuration optimized for minimal VRAM usage.
        
        Example:
            >>> # 7B model on 24GB GPU
            >>> config = YvConfig.from_yaml("configs/model/7B.yaml")
            >>> config = YvConfig.get_extreme_vram_config(config)
            >>> vram = config.estimate_vram_usage()
            >>> print(f"Estimated VRAM: {vram['total']:.1f} GB")
        """
        config = cls.get_extreme_memory_config(base_config)
        
        config.vram_offload_optimizer = True
        config.vram_offload_weights = False
        config.vram_max_experts_on_gpu = 4
        config.vram_dynamic_expert_loading = True
        config.vram_expert_lru_cache_size = 8
        config.vram_activation_checkpointing = True
        config.vram_selective_checkpointing = True
        config.vram_flash_attention = True
        config.vram_gradient_checkpointing = True
        config.vram_kv_cache_quantization = True
        config.vram_optimizer_state_quantization = True
        config.vram_optimizer_state_bits = 8
        config.vram_mixed_precision = "bf16"
        config.vram_zero_stage = 3
        config.vram_cpu_pin_memory = True
        config.vram_cpu_prefetch = True
        config.vram_async_transfer = True
        config.extreme_vram_mode = True
        
        return config

    @classmethod
    def get_ultra_low_vram_config(cls, base_config: Optional['YvConfig'] = None) -> 'YvConfig':
        """Get ultra-low VRAM configuration for extreme GPU memory constraints.
        
        Maximum VRAM optimization for training on consumer GPUs (8-16GB VRAM).
        This enables training models that would normally require 100GB+ VRAM.
        
        VRAM Savings:
            - 7B model: 140GB -> 12-16GB (fits on RTX 4090)
            - 32B model: 640GB -> 50-80GB (fits on A100 80GB)
            - 70B model: 1.4TB -> 100-160GB (fits on 2x A100 80GB)
        
        Args:
            base_config: Optional base configuration to apply optimizations to.
        
        Returns:
            YvConfig: Configuration with maximum VRAM savings.
        """
        config = cls.get_extreme_vram_config(base_config)
        
        config.vram_offload_weights = True
        config.vram_offload_gradients = True
        config.vram_offload_activations = True
        config.vram_offload_kv_cache = True
        config.vram_max_experts_on_gpu = 2
        config.vram_weight_quantization = True
        config.vram_weight_quant_bits = 4
        config.vram_fp4_training = True
        config.vram_fp8_attention = True
        config.vram_optimizer_state_bits = 4
        config.ultra_low_vram = True
        
        return config

    def apply_memory_optimization(self, level: str = "extreme") -> 'YvConfig':
        """Apply VRAM optimization to current configuration.
        
        Modifies the current configuration in-place with VRAM optimization
        settings appropriate for the specified level.
        
        Args:
            level: Optimization level, one of:
                - "none": No additional optimization
                - "moderate": Basic optimizations (checkpointing, MLA)
                - "aggressive": INT8 compression, sparse gradients
                - "extreme": All optimizations enabled
                - "ultra": Maximum savings, some performance impact
        
        Returns:
            YvConfig: Self for method chaining.
        
        Raises:
            ValueError: If level is not a valid optimization level.
        
        Example:
            >>> config = YvConfig(hidden_size=4096, n_layer=32)
            >>> config.apply_memory_optimization("extreme")
        """
        valid_levels = ["none", "moderate", "aggressive", "extreme", "ultra"]
        if level not in valid_levels:
            raise ValueError(f"Invalid optimization level: {level}. Must be one of {valid_levels}")
        
        if level == "none":
            return self
        
        self.use_gradient_checkpointing = True
        self.use_mla = True
        self.cache_quantization = True
        
        self.vram_gradient_checkpointing = True
        self.vram_flash_attention = True
        self.vram_kv_cache_quantization = True
        
        if level in ["aggressive", "extreme", "ultra"]:
            self.use_fp4 = True
            self.galore_enabled = True
            self.galore_memory_efficient = True
            self.ink_optimizer_enabled = True
            self.ink_momentum_bits = 8
            self.ink_variance_bits = 4
            self.ink_sparse_ratio = 0.01
            self.activation_quantization = True
            self.extreme_memory_mode = True
            
            self.vram_offload_optimizer = True
            self.vram_optimizer_state_quantization = True
            self.vram_optimizer_state_bits = 8
            self.vram_zero_stage = 3
            self.extreme_vram_mode = True
        
        if level == "extreme":
            self.cpu_offload_optimizer = True
            self.ink_max_experts_on_gpu = 4
            
            self.vram_max_experts_on_gpu = 4
            self.vram_dynamic_expert_loading = True
            self.vram_activation_checkpointing = True
        
        if level == "ultra":
            self.cpu_offload_weights = True
            self.cpu_offload_gradients = True
            self.ink_sparse_ratio = 0.005
            self.ink_max_experts_on_gpu = 2
            self.activation_quant_bits = 4
            self.ultra_low_memory = True
            
            self.vram_offload_weights = True
            self.vram_offload_gradients = True
            self.vram_offload_activations = True
            self.vram_offload_kv_cache = True
            self.vram_max_experts_on_gpu = 2
            self.vram_weight_quantization = True
            self.vram_weight_quant_bits = 4
            self.vram_fp4_training = True
            self.ultra_low_vram = True
        
        return self

    def estimate_memory_usage(self, batch_size: int = 1, seq_length: int = 2048) -> Dict[str, float]:
        """Estimate VRAM usage for current configuration.
        
        Provides a detailed estimate of GPU VRAM requirements for training
        with the current configuration settings.
        
        Args:
            batch_size: Training batch size.
            seq_length: Sequence length.
        
        Returns:
            Dict containing VRAM estimates in GB:
                - weights: Model weight VRAM
                - optimizer: Optimizer state VRAM
                - gradients: Gradient VRAM
                - activations: Activation VRAM
                - kv_cache: KV cache VRAM
                - total: Total estimated VRAM
        
        Example:
            >>> config = YvConfig.get_extreme_vram_config()
            >>> vram = config.estimate_vram_usage(batch_size=1, seq_length=4096)
            >>> print(f"Total VRAM: {vram['total']:.1f} GB")
        """
        return self.estimate_vram_usage(batch_size, seq_length)

    def estimate_vram_usage(self, batch_size: int = 1, seq_length: int = 2048) -> Dict[str, float]:
        """Estimate VRAM usage for current configuration.
        
        Provides a detailed estimate of GPU VRAM requirements for training
        with the current configuration settings. This is the primary method
        for VRAM estimation.
        
        VRAM Components:
            1. Model Weights: Parameters stored on GPU
            2. Optimizer States: Momentum and variance
            3. Gradients: Computed gradients
            4. Activations: Intermediate layer outputs
            5. KV Cache: Key-value cache for attention
        
        Args:
            batch_size: Training batch size.
            seq_length: Sequence length.
        
        Returns:
            Dict containing VRAM estimates in GB.
        """
        params = (
            self.vocab_size * self.hidden_size +
            self.n_layer * (
                self.hidden_size * self.hidden_size * 4 +
                self.hidden_size * self.intermediate_size * 3 +
                self.hidden_size * self.n_head * 2
            )
        )
        
        if self.moe_num_experts > 0:
            expert_params = self.n_layer * (
                self.hidden_size * self.intermediate_size * 3 * self.moe_num_experts
            )
            active_ratio = self.moe_top_k / self.moe_num_experts
            if self.vram_dynamic_expert_loading or self.vram_max_experts_on_gpu < self.moe_num_experts:
                active_ratio = min(active_ratio, self.vram_max_experts_on_gpu / self.moe_num_experts)
            params += expert_params * active_ratio
        
        bytes_per_param = 2
        if self.use_fp4 or self.vram_fp4_training:
            bytes_per_param = 0.5
        elif self.vram_weight_quantization:
            bytes_per_param = self.vram_weight_quant_bits / 8
        elif self.vram_mixed_precision == "fp16":
            bytes_per_param = 2
        elif self.vram_mixed_precision == "bf16":
            bytes_per_param = 2
        
        weight_vram = params * bytes_per_param / 1e9
        
        optimizer_multiplier = 2
        if self.ink_optimizer_enabled:
            optimizer_multiplier = self.ink_momentum_bits / 32 + self.ink_variance_bits / 32
        elif self.vram_optimizer_state_quantization:
            optimizer_multiplier = self.vram_optimizer_state_bits / 32
        optimizer_vram = weight_vram * optimizer_multiplier
        
        gradient_vram = weight_vram
        if self.ink_optimizer_enabled and self.ink_sparse_ratio < 1.0:
            gradient_vram *= self.ink_sparse_ratio
        elif self.gradient_compression_ratio < 1.0:
            gradient_vram *= self.gradient_compression_ratio
        
        activation_vram = batch_size * seq_length * self.hidden_size * self.n_layer * 4 / 1e9
        if self.use_gradient_checkpointing or self.vram_gradient_checkpointing:
            activation_vram *= self.ink_checkpoint_ratio if self.ink_optimizer_enabled else 0.5
        if self.activation_quantization:
            activation_vram *= 32 / self.activation_quant_bits
        
        kv_vram = batch_size * seq_length * self.hidden_size * 2 / 1e9
        if self.use_mla:
            kv_vram *= 0.2
        if self.cache_quantization or self.vram_kv_cache_quantization:
            kv_vram *= 0.25
        
        if self.vram_offload_optimizer or self.cpu_offload_optimizer:
            optimizer_vram *= 0.1
        if self.vram_offload_weights or self.cpu_offload_weights:
            weight_vram *= 0.1
        if self.vram_offload_gradients or self.cpu_offload_gradients:
            gradient_vram *= 0.1
        if self.vram_offload_activations:
            activation_vram *= 0.1
        if self.vram_offload_kv_cache:
            kv_vram *= 0.1
        
        if self.vram_tensor_parallel > 1:
            weight_vram /= self.vram_tensor_parallel
            optimizer_vram /= self.vram_tensor_parallel
            gradient_vram /= self.vram_tensor_parallel
        if self.vram_pipeline_parallel > 1:
            weight_vram /= self.vram_pipeline_parallel
            optimizer_vram /= self.vram_pipeline_parallel
            gradient_vram /= self.vram_pipeline_parallel
        
        total = weight_vram + optimizer_vram + gradient_vram + activation_vram + kv_vram
        
        return {
            "weights": round(weight_vram, 2),
            "optimizer": round(optimizer_vram, 2),
            "gradients": round(gradient_vram, 2),
            "activations": round(activation_vram, 2),
            "kv_cache": round(kv_vram, 2),
            "total": round(total, 2)
        }
