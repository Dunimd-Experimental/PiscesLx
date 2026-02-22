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

    chinchilla_optimal: bool = False
    chinchilla_c_budget: float = 0.0
    chinchilla_d_ratio: float = 1.0

    use_mla: bool = True
    kv_lora_rank: int = 512
    mla_q_lora_rank: Optional[int] = None
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
