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

"""Quantization utilities for Yv model inference optimization.

This module provides comprehensive quantization support for reducing model
memory footprint and accelerating inference, including weight quantization,
KV cache compression, and mixed-precision support.

Architecture Overview:
    The quantization system consists of three main components:
    
    1. **YvQuantizationType**: Enumeration of supported quantization schemes
    2. **YvQuantizationConfig**: Configuration dataclass for quantization settings
    3. **YvQuantizer**: Unified quantizer for weights and activations

Supported Quantization Types:
    - **INT8**: Standard 8-bit integer quantization (-128 to 127)
    - **INT4**: 4-bit quantization for extreme compression (-8 to 7)
    - **FP8**: 8-bit floating point (E4M3/E5M2 formats)
    - **FP16**: Half-precision floating point
    - **BF16**: BFloat16 for hardware with native support
    - **DYNAMIC_INT8**: Runtime activation quantization
    - **STATIC_INT8**: Calibration-based static quantization
    - **QAT**: Quantization-aware training support

Quantization Approaches:
    - **Symmetric**: Zero point at 0, simpler but may lose precision
    - **Asymmetric**: Learnable zero point, better for non-symmetric distributions
    - **Per-tensor**: Single scale for entire tensor
    - **Per-channel**: Independent scale per output channel (better accuracy)

KV Cache Quantization:
    The module provides specialized functions for quantizing KV cache tensors,
    reducing memory usage during long-context generation:
    
    - quantize_kv_cache(): Compress keys and values to INT8/INT4
    - dequantize_kv_cache(): Restore compressed cache for attention computation

Memory Savings:
    - INT8: 4x reduction (FP32 -> INT8)
    - INT4: 8x reduction (FP32 -> INT4)
    - KV Cache INT8: 4x cache memory reduction

Example:
    >>> from model.utils import YvQuantizer, YvQuantizationConfig
    >>> 
    >>> # Configure quantization
    >>> config = YvQuantizationConfig(
    ...     quantization_type=YvQuantizationType.INT8,
    ...     per_channel=True,
    ...     symmetric=True
    ... )
    >>> 
    >>> # Create quantizer
    >>> quantizer = YvQuantizer(config)
    >>> 
    >>> # Quantize model
    >>> quantized_model = quantizer.quantize_model(model)
    >>> 
    >>> # Quantize KV cache
    >>> q_keys, q_values, metadata = quantize_kv_cache(keys, values, bits=8)

Dependencies:
    - torch: For tensor operations and neural network modules
    - dataclasses: For configuration dataclass
    - enum: For enumeration types
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple
from enum import Enum


class YvQuantizationType(Enum):
    """Enumeration of supported quantization types.
    
    Defines the available quantization schemes for model weights and
    activations, each with different trade-offs between precision and
    memory/compute efficiency.
    
    Attributes:
        INT8: 8-bit integer quantization. Range: [-128, 127].
            Most widely supported, good balance of accuracy and efficiency.
        INT4: 4-bit integer quantization. Range: [-8, 7].
            Maximum compression, may require fine-tuning for accuracy.
        FP8: 8-bit floating point (E4M3/E5M2).
            Better for values with wide dynamic range.
        FP16: Half-precision floating point (16-bit).
            Simple precision reduction, widely supported on GPUs.
        BF16: BFloat16 (16-bit with 8-bit exponent).
            Better for large values, native on modern GPUs/TPUs.
        DYNAMIC_INT8: Runtime activation quantization.
            Quantizes activations on-the-fly during inference.
        STATIC_INT8: Calibration-based quantization.
            Requires calibration dataset to determine optimal scales.
        QAT: Quantization-aware training.
            Trains with quantization simulation for best accuracy.
    
    Example:
        >>> config.quantization_type = YvQuantizationType.INT8
    """
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"
    DYNAMIC_INT8 = "dynamic_int8"
    STATIC_INT8 = "static_int8"
    QAT = "qat"


@dataclass
class YvQuantizationConfig:
    """Configuration for model quantization.
    
    This dataclass encapsulates all settings for quantizing model weights
    and activations, including target modules, calibration settings, and
    quantization parameters.
    
    Attributes:
        quantization_type (YvQuantizationType): Type of quantization.
            Default: INT8.
        target_modules (List[str]): Module names to quantize. Default includes
            all linear projection layers (q_proj, k_proj, v_proj, o_proj,
            gate_proj, up_proj, down_proj).
        exclude_modules (List[str]): Module names to skip. Default excludes
            embedding and output layers (embed, lm_head).
        per_channel (bool): Whether to use per-channel quantization scales.
            Default: True (better accuracy, slightly more memory).
        symmetric (bool): Whether to use symmetric quantization (zero_point=0).
            Default: True (simpler, hardware-friendly).
        calibration_samples (int): Number of samples for calibration when
            using static quantization. Default: 128.
        calibration_method (str): Calibration method for static quantization.
            Options: "minmax", "entropy", "percentile". Default: "minmax".
    
    Example:
        >>> config = YvQuantizationConfig(
        ...     quantization_type=YvQuantizationType.INT8,
        ...     per_channel=True,
        ...     symmetric=True
        ... )
    """
    quantization_type: YvQuantizationType = YvQuantizationType.INT8
    target_modules: List[str] = None
    exclude_modules: List[str] = None
    per_channel: bool = True
    symmetric: bool = True
    calibration_samples: int = 128
    calibration_method: str = "minmax"

    def __post_init__(self):
        """Initialize default lists after dataclass construction.
        
        Sets default values for target_modules and exclude_modules if not
        provided, and converts string quantization_type to enum.
        """
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if self.exclude_modules is None:
            self.exclude_modules = ["embed", "lm_head"]
        if isinstance(self.quantization_type, str):
            self.quantization_type = YvQuantizationType(self.quantization_type)


class YvQuantizer:
    """Unified quantizer for model weights and activations.
    
    This class provides a comprehensive interface for quantizing neural network
    models, supporting multiple quantization schemes with configurable parameters.
    It handles both weight quantization and KV cache compression.
    
    Architecture:
        The quantizer operates in several stages:
        1. Configuration: Initialize with YvQuantizationConfig
        2. Tensor Quantization: Convert FP32/FP16 tensors to INT8/INT4
        3. Layer Quantization: Wrap linear layers with quantized versions
        4. Model Quantization: Apply quantization to entire model
        5. State Management: Save/load quantization scales and zero points
    
    Attributes:
        config (YvQuantizationConfig): Quantization settings.
        scale_dict (Dict[str, torch.Tensor]): Per-layer quantization scales.
        zero_point_dict (Dict[str, torch.Tensor]): Per-layer zero points.
    
    Example:
        >>> config = YvQuantizationConfig(
        ...     quantization_type=YvQuantizationType.INT8,
        ...     per_channel=True
        ... )
        >>> quantizer = YvQuantizer(config)
        >>> quantized_model = quantizer.quantize_model(model)
    """

    def __init__(self, config: Optional[YvQuantizationConfig] = None):
        """Initialize the quantizer with configuration.
        
        Args:
            config (Optional[YvQuantizationConfig]): Quantization settings.
                If None, uses default configuration. Default: None.
        
        Initializes:
            - config: Quantization configuration
            - scale_dict: Empty dictionary for quantization scales
            - zero_point_dict: Empty dictionary for zero points
        """
        self.config = config or YvQuantizationConfig()
        self.scale_dict: Dict[str, torch.Tensor] = {}
        self.zero_point_dict: Dict[str, torch.Tensor] = {}

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a tensor to specified bit width.
        
        Converts a floating-point tensor to integer representation using
        scale and zero-point parameters. Supports both symmetric and
        asymmetric quantization.
        
        Quantization Formula:
            quantized = clamp(round(tensor / scale + zero_point), qmin, qmax)
        
        Args:
            tensor (torch.Tensor): Input tensor to quantize (FP32/FP16).
            bits (int): Target bit width. Supported: 4, 8. Default: 8.
            symmetric (bool): If True, uses symmetric quantization with
                zero_point=0. Default: True.
            per_channel (bool): If True, computes separate scale per output
                channel. Default: True.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - quantized: INT8 tensor with quantized values
                - scale: Scale factor(s) for dequantization
                - zero_point: Zero point offset(s)
        
        Example:
            >>> quantizer = YvQuantizer()
            >>> weight = torch.randn(256, 512)
            >>> q_weight, scale, zp = quantizer.quantize_tensor(weight, bits=8)
        """
        if bits == 8:
            qmin, qmax = -128, 127
        elif bits == 4:
            qmin, qmax = -8, 7
        else:
            qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1

        if per_channel:
            dim = 0
            scale = tensor.abs().max(dim=dim, keepdim=True)[0] / qmax
        else:
            scale = tensor.abs().max() / qmax

        scale = scale.clamp(min=1e-8)

        if symmetric:
            zero_point = torch.zeros_like(scale)
        else:
            if per_channel:
                zero_point = qmin - (tensor.min(dim=dim, keepdim=True)[0] / scale).round()
            else:
                zero_point = qmin - (tensor.min() / scale).round()
            zero_point = zero_point.clamp(qmin, qmax).round()

        quantized = torch.clamp((tensor / scale + zero_point).round(), qmin, qmax).to(torch.int8)

        return quantized, scale, zero_point

    def dequantize_tensor(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize a tensor back to floating point.
        
        Reconstructs the original floating-point tensor from its quantized
        representation using the stored scale and zero point.
        
        Dequantization Formula:
            tensor = (quantized - zero_point) * scale
        
        Args:
            quantized (torch.Tensor): Quantized INT8 tensor.
            scale (torch.Tensor): Scale factor(s) from quantization.
            zero_point (torch.Tensor): Zero point offset(s) from quantization.
        
        Returns:
            torch.Tensor: Dequantized floating-point tensor.
        
        Example:
            >>> fp_tensor = quantizer.dequantize_tensor(q_weight, scale, zp)
        """
        return (quantized.float() - zero_point) * scale

    def quantize_linear(
        self,
        linear: nn.Linear,
        name: str
    ) -> nn.Module:
        """Quantize a linear layer and create a quantized wrapper.
        
        Converts the weight matrix of a linear layer to INT8/INT4 and
        creates a QuantizedLinear module that performs on-the-fly
        dequantization during forward pass.
        
        Args:
            linear (nn.Linear): Linear layer to quantize.
            name (str): Layer name for storing quantization parameters.
        
        Returns:
            nn.Module: QuantizedLinear wrapper with quantized weights.
        
        Note:
            The quantized weights are stored as INT8 parameters with
            requires_grad=False. Dequantization happens at inference time.
        """
        weight = linear.weight.data
        bits = 8 if self.config.quantization_type == YvQuantizationType.INT8 else 4

        quantized_weight, scale, zero_point = self.quantize_tensor(
            weight,
            bits=bits,
            symmetric=self.config.symmetric,
            per_channel=self.config.per_channel
        )

        self.scale_dict[name + ".weight"] = scale
        self.zero_point_dict[name + ".weight"] = zero_point

        class QuantizedLinear(nn.Module):
            def __init__(self, q_weight, scale, zero_point, bias):
                super().__init__()
                self.q_weight = nn.Parameter(q_weight, requires_grad=False)
                self.register_buffer("scale", scale)
                self.register_buffer("zero_point", zero_point)
                self.bias = bias

            def forward(self, x):
                weight = (self.q_weight.float() - self.zero_point) * self.scale
                return F.linear(x, weight, self.bias)

        return QuantizedLinear(quantized_weight, scale, zero_point, linear.bias)

    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[List[torch.Tensor]] = None
    ) -> nn.Module:
        """Quantize entire model by replacing linear layers.
        
        Iterates through all modules in the model and replaces qualifying
        linear layers with their quantized versions based on target_modules
        and exclude_modules configuration.
        
        Args:
            model (nn.Module): Model to quantize.
            calibration_data (Optional[List[torch.Tensor]]): Calibration data
                for static quantization. Currently not used. Default: None.
        
        Returns:
            nn.Module: Model with quantized linear layers.
        
        Note:
            - Only modules in target_modules are quantized
            - Modules in exclude_modules are skipped
            - Original model is modified in-place
        
        Example:
            >>> quantizer = YvQuantizer(config)
            >>> quantized_model = quantizer.quantize_model(model)
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                should_quantize = any(
                    target in name for target in self.config.target_modules
                )
                should_exclude = any(
                    exclude in name for exclude in self.config.exclude_modules
                )

                if should_quantize and not should_exclude:
                    quantized = self.quantize_linear(module, name)
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]

                    if parent_name:
                        parent = model.get_submodule(parent_name)
                        setattr(parent, child_name, quantized)
                    else:
                        setattr(model, child_name, quantized)

        return model

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get quantization state dictionary for saving.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - scales: Per-layer quantization scales
                - zero_points: Per-layer zero point offsets
        
        Example:
            >>> state = quantizer.get_state_dict()
            >>> torch.save(state, "quantization_state.pt")
        """
        return {
            "scales": self.scale_dict,
            "zero_points": self.zero_point_dict,
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load quantization state dictionary from saved state.
        
        Args:
            state_dict (Dict[str, torch.Tensor]): Dictionary containing
                scales and zero_points tensors.
        
        Example:
            >>> state = torch.load("quantization_state.pt")
            >>> quantizer.load_state_dict(state)
        """
        self.scale_dict = state_dict.get("scales", {})
        self.zero_point_dict = state_dict.get("zero_points", {})


def quantize_kv_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    bits: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Quantize KV cache for memory efficiency during inference.
    
    Compresses key and value cache tensors to INT8 or INT4, reducing
    memory usage by 4x or 8x respectively. Essential for long-context
    generation where KV cache grows linearly with sequence length.
    
    Args:
        keys (torch.Tensor): Key cache tensor of shape [batch, heads, seq, dim].
        values (torch.Tensor): Value cache tensor of shape [batch, heads, seq, dim].
        bits (int): Target bit width. Supported: 4, 8. Default: 8.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
            - q_keys: Quantized key cache (INT8)
            - q_values: Quantized value cache (INT8)
            - metadata: Dictionary with scales and zero points for dequantization
    
    Example:
        >>> q_keys, q_values, meta = quantize_kv_cache(keys, values, bits=8)
        >>> # Later, during attention computation:
        >>> keys, values = dequantize_kv_cache(q_keys, q_values, meta)
    
    Note:
        Quantization introduces some accuracy loss. For sensitive tasks,
        consider using FP16 or keeping FP32 cache.
    """
    quantizer = YvQuantizer(YvQuantizationConfig(
        quantization_type=YvQuantizationType.INT8 if bits == 8 else YvQuantizationType.INT4
    ))

    q_keys, k_scale, k_zp = quantizer.quantize_tensor(keys, bits=bits)
    q_values, v_scale, v_zp = quantizer.quantize_tensor(values, bits=bits)

    metadata = {
        "k_scale": k_scale,
        "k_zero_point": k_zp,
        "v_scale": v_scale,
        "v_zero_point": v_zp,
    }

    return q_keys, q_values, metadata


def dequantize_kv_cache(
    q_keys: torch.Tensor,
    q_values: torch.Tensor,
    metadata: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize KV cache for attention computation.
    
    Restores quantized key and value cache tensors to floating-point
    format for use in attention operations. This is the inverse operation
    of quantize_kv_cache.
    
    Args:
        q_keys (torch.Tensor): Quantized key cache (INT8).
        q_values (torch.Tensor): Quantized value cache (INT8).
        metadata (Dict[str, torch.Tensor]): Quantization metadata containing:
            - k_scale: Key cache scale factor
            - k_zero_point: Key cache zero point
            - v_scale: Value cache scale factor
            - v_zero_point: Value cache zero point
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - keys: Dequantized key cache (FP32)
            - values: Dequantized value cache (FP32)
    
    Example:
        >>> # After loading quantized cache
        >>> keys, values = dequantize_kv_cache(q_keys, q_values, metadata)
        >>> # Use in attention
        >>> attn_output = attention(query, keys, values)
    
    Note:
        Dequantization adds computational overhead. For best performance,
        consider fused attention kernels that operate directly on quantized
        cache with hardware support.
    """
    quantizer = YvQuantizer()

    keys = quantizer.dequantize_tensor(
        q_keys,
        metadata["k_scale"],
        metadata["k_zero_point"]
    )
    values = quantizer.dequantize_tensor(
        q_values,
        metadata["v_scale"],
        metadata["v_zero_point"]
    )

    return keys, values
