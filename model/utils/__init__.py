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

"""Yv Utilities Module - Position embeddings and quantization utilities.

This module provides essential utility components for the Yv model,
including advanced rotary position embeddings and model quantization support.

Architecture Overview:
    The utilities module is organized into two main components:
    
    1. **Rotary Position Embeddings** (rope.py):
       - YvYaRNRotaryEmbedding: YaRN-scaled RoPE for long contexts
       - YvDynamicYaRNRotaryEmbedding: Dynamic NTK scaling
       - YvLinearScalingRoPE: Simple linear position scaling
       - precompute_freqs_cis: Frequency precomputation for efficiency
       - apply_rotary_emb: Efficient rotary embedding application
    
    2. **Quantization** (quantization.py):
       - YvQuantizer: Unified quantization interface
       - YvQuantizationConfig: Quantization settings
       - YvQuantizationType: Supported quantization types
       - KV cache quantization functions

Key Features:
    - **Long Context Support**: YaRN and dynamic scaling for extended sequences
    - **Memory Efficiency**: INT8/INT4 quantization for inference
    - **KV Cache Compression**: Quantized cache for memory savings
    - **Flexible Configuration**: Per-channel, symmetric, and asymmetric options

Position Embedding Types:
    - **YaRN (Yet another RoPE extensioN)**: Extends context length by
      combining NTK-aware scaling with temperature adjustment
    - **Dynamic YaRN**: Automatically adjusts scaling based on sequence length
    - **Linear Scaling**: Simple division of position indices for extension

Quantization Types:
    - **INT8**: 8-bit integer quantization for weights
    - **INT4**: 4-bit quantization for extreme compression
    - **FP8**: 8-bit floating point (E4M3/E5M2)
    - **Dynamic INT8**: Runtime quantization of activations
    - **Static INT8**: Calibration-based quantization
    - **QAT**: Quantization-aware training support

Example:
    >>> from model.utils import (
    ...     YvYaRNRotaryEmbedding,
    ...     YvQuantizer,
    ...     YvQuantizationConfig,
    ... )
    >>> 
    >>> # Create YaRN position embedding
    >>> rope = YvYaRNRotaryEmbedding(
    ...     dim=128,
    ...     max_position_embeddings=32768,
    ...     scale=32
    ... )
    >>> 
    >>> # Quantize model weights
    >>> quant_config = YvQuantizationConfig(
    ...     quantization_type=YvQuantizationType.INT8,
    ...     per_channel=True
    ... )
    >>> quantizer = YvQuantizer(quant_config)
    >>> quantized_model = quantizer.quantize_model(model)

Dependencies:
    - torch: For tensor operations and neural network modules
    - dataclasses: For configuration dataclasses
    - enum: For enumeration types
"""

from .rope import (
    YvYaRNRotaryEmbedding,
    YvDynamicYaRNRotaryEmbedding,
    YvLinearScalingRoPE,
    precompute_freqs_cis,
    apply_rotary_emb,
)
from .quantization import (
    YvQuantizer,
    YvQuantizationConfig,
    YvQuantizationType,
)

__all__ = [
    "YvYaRNRotaryEmbedding",
    "YvDynamicYaRNRotaryEmbedding",
    "YvLinearScalingRoPE",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "YvQuantizer",
    "YvQuantizationConfig",
    "YvQuantizationType",
]
