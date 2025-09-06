#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
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
class PiscesConfig:
    """Configuration for the Pisces L1 model with MoE load balancing improvements.
    
    Attributes:
        model_type (str): The type of the model, default is "pisces_l1".
        vocab_size (int): The size of the vocabulary, default is 100,352.
        hidden_size (int): The size of the hidden layer, default is 2048.
        n_layer (int): The number of layers in the model, default is 24.
        n_head (int): The number of attention heads, default is 16.
        n_kv_head (int): The number of key-value attention heads, default is 4.
        moe_num_experts (int): The number of experts in the Mixture of Experts, default is 64.
        moe_top_k (int): The top-k experts to use in Mixture of Experts, default is 2.
        moe_capacity_factor (float): Capacity factor for expert routing, default is 1.25.
        moe_load_balance_alpha (float): Load balancing loss coefficient, default is 0.01.
        moe_noise_std (float): Routing noise standard deviation, default is 0.1.
        moe_use_stable_gate (bool): Whether to use stable routing gate, default is True.
        intermediate_size (int): The size of the intermediate layer, default is 5632.
        max_position_embeddings (int): The maximum number of position embeddings, default is 8192.
        rope_theta (float): The theta value for RoPE, default is 1e6.
        dropout (float): The dropout rate, default is 0.0.
        image_res (int): The resolution of the image, default is 224.
        image_patch (int): The size of the image patch, default is 14.
        mm_tokens (int): The number of multimodal tokens, default is 256.
        audio_tokens (int): The number of audio tokens, default is 512.
        task_classes (int): The number of task classes, default is 256.
        eval_dims (int): The number of evaluation dimensions, default is 7.
        rope_scaling (dict): The configuration for RoPE scaling, default is a dict specifying "yarn" type with factor 32 and original max position 32768.
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
    moe_capacity_factor: float = 1.25  # Capacity factor for expert routing
    moe_load_balance_alpha: float = 0.01  # Load balancing loss coefficient
    moe_noise_std: float = 0.1  # Routing noise standard deviation
    moe_use_stable_gate: bool = True  # Whether to use stable routing gate
    moe_min_capacity: int = 4  # Minimum capacity
    
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
    
    # Stability configurations
    residual_dropout_p: float = 0.1  # Dropout rate for residual connections
    use_gradient_checkpointing: bool = True  # Enable gradient checkpointing for memory efficiency
    use_pre_norm: bool = True  # Use Pre-Norm architecture for stability
    attention_dropout: float = 0.0  # Dropout rate for attention layers
    
    # Dynamic multimodal fusion configurations
    enable_dynamic_fusion: bool = True  # Enable dynamic content-based modality weighting
    fusion_quality_threshold: float = 0.3  # Quality threshold for modality inclusion
    fusion_dropout: float = 0.1  # Dropout for fusion layers
    
    # 3D Spatio-Temporal RoPE configurations for video understanding
    use_3d_spatio_temporal_rope: bool = False  # Enable 3D spatio-temporal RoPE for video frames
    max_temporal_frames: int = 64  # Maximum number of temporal frames for 3D RoPE
    
    # Long context configurations
    attention_type: str = "standard"  # Type of attention: standard, streaming_llm, h2o_attention
    streaming_window: int = 4096  # Window size for streaming attention
    compression_ratio: int = 4  # Compression ratio for H2O attention
    use_sliding_window: bool = False  # Enable sliding window attention for long contexts
    long_factor: int = 32  # Long context scaling factor
    
    @classmethod
    def from_json(cls, path):
        """Load the model configuration from a JSON file.

        Args:
            path (str): The path to the JSON file containing the configuration.

        Returns:
            PiscesConfig: An instance of the PiscesConfig class initialized with the loaded configuration.
        """
        with open(path, 'r') as f:
            config_data = json.load(f)
        
        model_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_data.items() if k in model_fields}
        
        return cls(**filtered_config)