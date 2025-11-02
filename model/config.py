#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ArcticConfig:
    """Dataclass for storing configuration parameters of the PiscesL1 model.

    Attributes:
        model_type (str): The type of the model. Defaults to "pisces_l1".
        vocab_size (int): The size of the vocabulary. Defaults to 71164.
        hidden_size (int): The size of the hidden layer. Defaults to 2048.
        n_layer (int): The number of layers in the model. Defaults to 24.
        n_head (int): The number of attention heads. Defaults to 16.
        n_kv_head (int): The number of key-value attention heads. Defaults to 4.
        moe_num_experts (int): The number of experts in the Mixture of Experts (MoE). Defaults to 64.
        moe_top_k (int): The number of top-k experts to use in MoE. Defaults to 2.
        moe_capacity_factor (float): Capacity factor for expert routing. Defaults to 1.0.
        moe_load_balance_alpha (float): Load balancing loss coefficient. Defaults to 0.01.
        moe_noise_std (float): Routing noise standard deviation. Defaults to 0.1.
        moe_use_stable_gate (bool): Whether to use a stable routing gate. Defaults to True.
        moe_min_capacity (int): Minimum capacity for expert routing. Defaults to 4.
        moe_prediction_horizon (int): Prediction horizon for dynamic capacity. Defaults to 8.
        intermediate_size (int): The size of the intermediate layer. Defaults to 5632.
        max_position_embeddings (int): The maximum number of position embeddings. Defaults to 8192.
        rope_theta (float): The theta value for Rotary Position Embedding (RoPE). Defaults to 1e6.
        dropout (float): The dropout rate. Defaults to 0.0.
        image_res (int): The resolution of the image. Defaults to 224.
        max_image_res (int): The maximum resolution of the image. Defaults to 1024.
        image_patch (int): The size of the image patch. Defaults to 14.
        use_native_resolution (bool): Whether to use native resolution. Defaults to True.
        enable_patch_pack (bool): Whether to enable patch packing. Defaults to True.
        mm_tokens (int): The number of multimodal tokens. Defaults to 256.
        audio_tokens (int): The number of audio tokens. Defaults to 512.
        task_classes (int): The number of task classes. Defaults to 256.
        eval_dims (int): The number of evaluation dimensions. Defaults to 7.
        rope_scaling (dict): The configuration for RoPE scaling. Defaults to a dict specifying "yarn" type with factor 32 and original max position 32768.
        residual_dropout_p (float): Dropout rate for residual connections. Defaults to 0.1.
        use_gradient_checkpointing (bool): Whether to enable gradient checkpointing for memory efficiency. Defaults to True.
        use_pre_norm (bool): Whether to use Pre-Norm architecture for stability. Defaults to True.
        attention_dropout (float): Dropout rate for attention layers. Defaults to 0.0.
        fused_qkv (bool): Whether to use fused QKV projection in attention for better efficiency when supported. Defaults to False.
        enable_dynamic_fusion (bool): Whether to enable native token-level multimodal fusion. Defaults to True.
        fusion_quality_threshold (float): Quality threshold for modality inclusion. Defaults to 0.3.
        fusion_dropout (float): Dropout for fusion layers. Defaults to 0.1.
        modal_token_count (int): Number of fused multimodal tokens to prepend when fusion returns [B, H]. Defaults to 8.
        enable_cognitive_density (bool): Whether to enable cognitive density optimization for all 64 experts. Defaults to True.
        enable_dynamic_capacity (bool): Whether to enable dynamic capacity scaling for balanced loading. Defaults to True.
        cognitive_enhancement_scale (float): Scale factor for cognitive enhancement. Defaults to 0.1.
        expert_temperature_max (float): Maximum routing temperature for exploration. Defaults to 5.0.
        expert_load_balance_threshold (float): Threshold for load imbalance warnings. Defaults to 0.15.
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
    model_type: str = "pisces_l1"
    vocab_size: int = 71164
    hidden_size: int = 2048
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 4
    
    # MoE-related configurations
    moe_num_experts: int = 64  # Number of experts in the Mixture of Experts
    moe_top_k: int = 2  # Top-k experts to use in Mixture of Experts
    moe_capacity_factor: float = 1.0  # Capacity factor for expert routing
    moe_load_balance_alpha: float = 0.01  # Load balancing loss coefficient
    moe_noise_std: float = 0.1  # Routing noise standard deviation
    moe_use_stable_gate: bool = True  # Whether to use stable routing gate
    moe_min_capacity: int = 4  # Minimum capacity for expert routing
    moe_prediction_horizon: int = 8  # Prediction horizon for dynamic capacity
    
    # Expert-level gradient clipping for 3rd-order mixture stability
    moe_expert_grad_clip: float = 0.1  # Expert-level gradient clipping threshold
    moe_z_loss_alpha: float = 1e-4  # Z-loss coefficient for routing stability
    moe_random_to_gradient_steps: int = 500  # Steps to switch from random to gradient routing
    moe_gate_warmup_alpha: float = 0.05  # Gate warmup coefficient for cold start
    moe_attention_mamba_temp: float = 0.3  # Temperature for attention-mamba at 8k length
    moe_l2_smooth_8k: float = 0.01  # L2 smoothing for 8k sequence length
    
    intermediate_size: int = 5632
    max_position_embeddings: int = 8192
    rope_theta: float = 1e6
    dropout: float = 0.0
    image_res: int = 224
    max_image_res: int = 1024
    image_patch: int = 14
    use_native_resolution: bool = True  # Whether to use native resolution
    enable_patch_pack: bool = True  # Whether to enable patch packing
    mm_tokens: int = 256
    audio_tokens: int = 512
    task_classes: int = 256
    eval_dims: int = 7
    rope_scaling: dict = field(default_factory=lambda: {"type": "yarn", "factor": 32, "original_max_position_embeddings": 32768})
    
    # Stability configurations
    residual_dropout_p: float = 0.1  # Dropout rate for residual connections
    use_gradient_checkpointing: bool = True  # Whether to enable gradient checkpointing for memory efficiency
    use_pre_norm: bool = True  # Whether to use Pre-Norm architecture for stability
    attention_dropout: float = 0.0  # Dropout rate for attention layers
    # Kernel optimizations
    fused_qkv: bool = False  # Whether to use fused QKV projection in Attention for better efficiency when supported
    
    # Dynamic multimodal fusion configurations
    enable_dynamic_fusion: bool = True  # Whether to enable native token-level multimodal fusion
    fusion_quality_threshold: float = 0.3  # Quality threshold for modality inclusion
    fusion_dropout: float = 0.1  # Dropout for fusion layers
    modal_token_count: int = 8  # Number of fused multimodal tokens to prepend when fusion returns [B, H]
    
    # Advanced optimization configurations
    enable_cognitive_density: bool = True  # Whether to enable cognitive density optimization for all 64 experts
    enable_dynamic_capacity: bool = True  # Whether to enable dynamic capacity scaling for balanced loading
    cognitive_enhancement_scale: float = 0.1  # Scale factor for cognitive enhancement
    expert_temperature_max: float = 5.0  # Maximum routing temperature for exploration
    expert_load_balance_threshold: float = 0.15  # Threshold for load imbalance warnings
    
    # 3D Spatio-Temporal RoPE configurations for video understanding
    use_3d_spatio_temporal_rope: bool = False  # Whether to enable 3D spatio-temporal RoPE for video frames
    max_temporal_frames: int = 64  # Maximum number of temporal frames for 3D RoPE
    
    # Long context configurations
    attention_type: str = "standard"  # Type of attention: standard, streaming_llm, h2o_attention
    use_h2o_attention: bool = True  # Whether to enable H2O attention
    streaming_window: int = 16384  # Window size for streaming attention
    compression_ratio: int = 8  # Compression ratio for H2O attention
    use_sliding_window: bool = False  # Whether to enable sliding window attention for long contexts
    long_factor: int = 32  # Long context scaling factor
    # KV cache and attention kernel knobs
    max_cache_size: int = 8192  # Maximum tokens kept in KV cache (paged)
    cache_quantization: bool = True  # Whether to enable KV cache quantization
    kv_cache_block_size: int = 512  # Paged KV block size
    sdpa_prefer_flash: bool = True  # Whether to prefer FlashAttention backend for SDPA when available
    
    # Speculative decoding configurations
    speculative_candidates: int = 4  # Number of candidate tokens for speculative decoding
    speculative_draft_length: int = 5  # Length of draft sequence
    speculative_acceptance_threshold: float = 0.8  # Threshold for accepting draft tokens
    speculative_temperature: float = 0.7  # Temperature for speculative sampling
    speculative_top_k: int = 50  # Top-k for speculative sampling
    speculative_top_p: float = 0.9  # Top-p for speculative sampling
    enable_speculative_decoding: bool = True  # Whether to enable speculative decoding
    
    # MCP tool-use routing thresholds
    tool_uncertainty_threshold: float = 0.7  # Trigger tools when uncertainty exceeds this value
    tool_fact_consistency_threshold: float = 0.6  # Trigger tools when fact consistency is below this value
    
    # Debug output controls
    enable_debug_outputs: bool = False  # If True, model.forward returns a 'debug' section with shapes/dtypes
    debug_verbose: bool = False  # If True, include extra debug like modality presence and fusion shapes
    
    # Mamba-3 integration configurations
    use_mamba3: bool = False  # Whether to enable Mamba-3 state space model integration
    mamba3_layers: list = field(default_factory=list)  # List of layer indices to use Mamba-3, empty means use in all layers
    mamba3_d_state: int = 128  # State dimension for Mamba-3 SSM
    mamba3_d_conv: int = 4  # Convolution kernel size for Mamba-3
    mamba3_expand: int = 2  # Expansion factor for Mamba-3
    mamba3_dt_rank: str = "auto"  # Time step rank, "auto" means ceil(hidden_size/16)
    mamba3_conv_bias: bool = True  # Whether to use bias in convolution
    mamba3_proj_bias: bool = False  # Whether to use bias in projections
    mamba3_use_fast_path: bool = True  # Whether to use fast path optimization
    mamba3_layer_norm_eps: float = 1e-4  # Layer normalization epsilon for Mamba-3 (increased for mixed precision stability)
    mamba3_sequence_threshold: int = 8192  # Sequence length threshold to switch between attention and Mamba-3
    mamba3_gate_mode: str = "adaptive"  # Gate mode: "learned", "adaptive", "fixed"
    mamba3_gate_init: float = 0.5  # Initial gate value for learned mode
    mamba3_gate_temperature: float = 1.0  # Temperature for adaptive gate mode
    mamba3_complex_state: bool = True  # Whether to enable complex state space
    mamba3_trapezoidal: bool = True  # Whether to enable trapezoidal discretization
    mamba3_mimo: bool = True  # Whether to enable MIMO architecture
    
    # GaLore training optimization configurations
    galore_enabled: bool = False  # Whether to enable GaLore gradient low-rank projection
    galore_rank: int = 128  # Low-rank dimension for gradient projection (typically 64-256)
    galore_update_interval: int = 200  # Steps between low-rank projection updates
    galore_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])  # Target modules for GaLore optimization
    galore_lr_ratio: float = 1.0  # Learning rate ratio for GaLore vs full parameters
    galore_min_rank: int = 32  # Minimum rank for adaptive rank adjustment
    galore_max_rank: int = 512  # Maximum rank for adaptive rank adjustment
    galore_rank_adapt_interval: int = 1000  # Steps between rank adaptation
    galore_rank_adapt_threshold: float = 0.1  # Gradient norm threshold for rank adaptation
    galore_quantization_bits: int = 0  # Quantization bits for GaLore states (0=disable, 8=8-bit)
    galore_memory_efficient: bool = True  # Enable memory-efficient implementation
    galore_moe_expert_only: bool = False  # Apply GaLore only to MoE experts
    galore_multimodal_modules: list = field(default_factory=lambda: ["vision_encoder", "audio_encoder", "multimodal_fusion"])  # Multimodal modules for GaLore
    galore_sequence_threshold: int = 4096  # Sequence length threshold to enable GaLore
    galore_gradient_accumulation_sync: bool = True  # Sync GaLore projections across gradient accumulation steps
    mamba3_dropout: float = 0.0  # Dropout rate for Mamba-3 layers
    
    # Chinchilla scaling law configurations
    chinchilla_optimal: bool = False  # Whether to enable Chinchilla scaling law optimization
    chinchilla_c_budget: float = 0.0  # Training compute budget in FLOPs or GPU hours
    chinchilla_d_ratio: float = 1.0  # Internal cache for optimal D/N ratio
    
    @classmethod
    def from_json(cls, path: str) -> 'ArcticConfig':
        """Load the model configuration from a JSON file.

        Args:
            path (str): The path to the JSON file containing the configuration.

        Returns:
            ArcticConfig: An instance of the ArcticConfig class initialized with the loaded configuration.
        """
        with open(path, 'r') as f:
            config_data = json.load(f)
        
        model_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_data.items() if k in model_fields}
        
        return cls(**filtered_config)
