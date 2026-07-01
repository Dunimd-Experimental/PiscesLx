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
Ink KV Cache Quantization Module

This module provides INT8/INT4 quantization for KV Cache during inference,
dramatically reducing memory usage for long context generation.

Key Features:
    - Block-wise INT8/INT4 quantization for KV Cache
    - Dynamic scaling for each block
    - Minimal quality loss through optimal quantization strategy
    - Support for both key and value caches

Memory Savings:
    - FP16 KV Cache: 2 * seq_len * 2 * head_dim * batch * 2 bytes
    - INT8 KV Cache: 4x reduction
    - INT4 KV Cache: 8x reduction
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from configs.version import VERSION


class POPSSInkKVCacheQuantizer:
    """
    INT8/INT4 KV Cache Quantizer.
    
    Provides block-wise quantization for Key and Value caches during inference,
    reducing memory footprint significantly while maintaining generation quality.
    
    Attributes:
        kv_bits: Quantization bits (8 for INT8, 4 for INT4)
        block_size: Block size for per-block quantization
        device: Device for tensor operations
    
    Example:
        >>> quantizer = POPSSInkKVCacheQuantizer(kv_bits=8, block_size=64)
        >>> k_quantized, k_scales = quantizer.quantize_k(k_cache)
        >>> k_dequantized = quantizer.dequantize_k(k_quantized, k_scales, k_shape)
    """
    
    def __init__(
        self,
        kv_bits: int = 8,
        block_size: int = 64,
        device: Optional[torch.device] = None,
    ):
        self.kv_bits = kv_bits
        self.block_size = block_size
        self.device = device
        
        self._k_cache: Dict[str, torch.Tensor] = {}
        self._v_cache: Dict[str, torch.Tensor] = {}
        self._k_scales: Dict[str, torch.Tensor] = {}
        self._v_scales: Dict[str, torch.Tensor] = {}
        self._cache_stats: Dict[str, int] = {}
    
    def quantize_k(
        self,
        k_tensor: torch.Tensor,
        layer_name: str = "default",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize key cache.
        
        Args:
            k_tensor: Key tensor [batch, heads, seq_len, head_dim]
            layer_name: Name of the layer for caching
        
        Returns:
            Tuple of (quantized_tensor, scales)
        """
        if self.kv_bits == 8:
            return self._quantize_int8(k_tensor, layer_name, "k")
        else:
            return self._quantize_int4(k_tensor, layer_name, "k")
    
    def quantize_v(
        self,
        v_tensor: torch.Tensor,
        layer_name: str = "default",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize value cache.
        
        Args:
            v_tensor: Value tensor [batch, heads, seq_len, head_dim]
            layer_name: Name of the layer for caching
        
        Returns:
            Tuple of (quantized_tensor, scales)
        """
        if self.kv_bits == 8:
            return self._quantize_int8(v_tensor, layer_name, "v")
        else:
            return self._quantize_int4(v_tensor, layer_name, "v")
    
    def dequantize_k(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        shape: Tuple[int, ...],
        layer_name: str = "default",
    ) -> torch.Tensor:
        """
        Dequantize key cache.
        
        Args:
            quantized: Quantized tensor
            scales: Per-block scales
            shape: Original tensor shape
            layer_name: Name of the layer
        
        Returns:
            Dequantized tensor
        """
        if self.kv_bits == 8:
            return self._dequantize_int8(quantized, scales, shape, layer_name, "k")
        else:
            return self._dequantize_int4(quantized, scales, shape, layer_name, "k")
    
    def dequantize_v(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        shape: Tuple[int, ...],
        layer_name: str = "default",
    ) -> torch.Tensor:
        """
        Dequantize value cache.
        
        Args:
            quantized: Quantized tensor
            scales: Per-block scales
            shape: Original tensor shape
            layer_name: Name of the layer
        
        Returns:
            Dequantized tensor
        """
        if self.kv_bits == 8:
            return self._dequantize_int8(quantized, scales, shape, layer_name, "v")
        else:
            return self._dequantize_int4(quantized, scales, shape, layer_name, "v")
    
    def _quantize_int8(
        self,
        tensor: torch.Tensor,
        layer_name: str,
        cache_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to INT8 with per-block scaling."""
        orig_shape = tensor.shape
        tensor_flat = tensor.flatten()
        numel = tensor_flat.numel()
        num_blocks = (numel + self.block_size - 1) // self.block_size
        
        padded_size = num_blocks * self.block_size
        if padded_size > numel:
            padding = torch.zeros(
                padded_size - numel,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor_flat = torch.cat([tensor_flat, padding])
        
        tensor_blocks = tensor_flat.view(num_blocks, self.block_size)
        
        block_max = tensor_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        scales = block_max / 127.0
        
        scaled = tensor_blocks / scales
        quantized = torch.clamp(torch.round(scaled), -128, 127).to(torch.int8)
        
        cache_key = f"{layer_name}_{cache_type}"
        self._cache_stats[cache_key] = numel
        
        if cache_type == "k":
            self._k_cache[cache_key] = quantized
            self._k_scales[cache_key] = scales.squeeze(-1)
        else:
            self._v_cache[cache_key] = quantized
            self._v_scales[cache_key] = scales.squeeze(-1)
        
        return quantized, scales.squeeze(-1)
    
    def _quantize_int4(
        self,
        tensor: torch.Tensor,
        layer_name: str,
        cache_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to INT4 with per-block scaling."""
        orig_shape = tensor.shape
        tensor_flat = tensor.flatten()
        numel = tensor_flat.numel()
        
        padded_numel = ((numel + 1) // 2) * 2
        if padded_numel > numel:
            padding = torch.zeros(
                padded_numel - numel,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor_flat = torch.cat([tensor_flat, padding])
        
        num_blocks = padded_numel // self.block_size
        tensor_blocks = tensor_flat[:num_blocks * self.block_size].view(num_blocks, self.block_size)
        
        block_max = tensor_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        scales = block_max / 7.0
        
        scaled = tensor_blocks / scales
        quantized = torch.clamp(torch.round(scaled), -8, 7).to(torch.int8)
        
        pairs = quantized.view(-1, 2)
        packed = (pairs[:, 0].char() & 0x0F) | ((pairs[:, 1].char() & 0x0F) << 4)
        
        cache_key = f"{layer_name}_{cache_type}"
        self._cache_stats[cache_key] = numel
        
        if cache_type == "k":
            self._k_cache[cache_key] = packed
            self._k_scales[cache_key] = scales.squeeze(-1)
        else:
            self._v_cache[cache_key] = packed
            self._v_scales[cache_key] = scales.squeeze(-1)
        
        return packed, scales.squeeze(-1)
    
    def _dequantize_int8(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        shape: Tuple[int, ...],
        layer_name: str,
        cache_type: str,
    ) -> torch.Tensor:
        """Dequantize INT8."""
        numel = shape.numel()
        num_blocks = scales.numel()
        
        dequant_flat = quantized.flatten().float() * scales.unsqueeze(-1)
        dequant_flat = dequant_flat.flatten()[:numel]
        
        return dequant_flat.view(shape)
    
    def _dequantize_int4(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        shape: Tuple[int, ...],
        layer_name: str,
        cache_type: str,
    ) -> torch.Tensor:
        """Dequantize INT4."""
        numel = shape.numel()
        num_blocks = scales.numel()
        
        packed = quantized.flatten()
        num_pairs = packed.numel()
        
        low_nibbles = (packed & 0x0F).char()
        high_nibbles = ((packed >> 4) & 0x0F).char()
        
        dequant_pairs = torch.zeros(num_pairs * 2, dtype=torch.float32, device=quantized.device)
        dequant_pairs[0::2] = low_nibbles.float()
        dequant_pairs[1::2] = high_nibbles.float()
        
        block_size = self.block_size
        dequant_blocks = dequant_pairs[:num_blocks * block_size].view(num_blocks, block_size)
        dequant_blocks = dequant_blocks * scales.unsqueeze(-1)
        
        dequant_flat = dequant_blocks.flatten()[:numel]
        
        return dequant_flat.view(shape)
    
    def get_cache_size(self, layer_name: str = "default") -> Dict[str, int]:
        """Get cache size in bytes for a layer."""
        k_key = f"{layer_name}_k"
        v_key = f"{layer_name}_v"
        
        k_size = self._k_cache[k_key].numel() if k_key in self._k_cache else 0
        v_size = self._v_cache[v_key].numel() if v_key in self._v_cache else 0
        
        if self.kv_bits == 8:
            bytes_per_elem = 1
        else:
            bytes_per_elem = 0.5
            k_size *= 2
            v_size *= 2
        
        return {
            "k_bytes": k_size * bytes_per_elem,
            "v_bytes": v_size * bytes_per_elem,
            "total_bytes": (k_size + v_size) * bytes_per_elem,
        }
    
    def clear_cache(self, layer_name: Optional[str] = None):
        """Clear cached tensors."""
        if layer_name is None:
            self._k_cache.clear()
            self._v_cache.clear()
            self._k_scales.clear()
            self._v_scales.clear()
            self._cache_stats.clear()
        else:
            for key in list(self._k_cache.keys()):
                if key.startswith(layer_name):
                    del self._k_cache[key]
                    del self._k_scales[key]
            for key in list(self._v_cache.keys()):
                if key.startswith(layer_name):
                    del self._v_cache[key]
                    del self._v_scales[key]
            for key in list(self._cache_stats.keys()):
                if key.startswith(layer_name):
                    del self._cache_stats[key]
    
    def get_memory_savings(self, original_bytes: int) -> Dict[str, Any]:
        """Calculate memory savings."""
        if self.kv_bits == 8:
            ratio = 4.0
        else:
            ratio = 8.0
        
        return {
            "original_bytes": original_bytes,
            "compressed_bytes": int(original_bytes / ratio),
            "compression_ratio": ratio,
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state for serialization."""
        return {
            "kv_bits": self.kv_bits,
            "block_size": self.block_size,
            "device": str(self.device) if self.device else None,
            "k_cache": {k: v.clone() for k, v in self._k_cache.items()},
            "v_cache": {k: v.clone() for k, v in self._v_cache.items()},
            "k_scales": {k: v.clone() for k, v in self._k_scales.items()},
            "v_scales": {k: v.clone() for k, v in self._v_scales.items()},
            "cache_stats": self._cache_stats.copy(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from dictionary."""
        self.kv_bits = state_dict["kv_bits"]
        self.block_size = state_dict["block_size"]
        self._k_cache = {k: v.clone() for k, v in state_dict["k_cache"].items()}
        self._v_cache = {k: v.clone() for k, v in state_dict["v_cache"].items()}
        self._k_scales = {k: v.clone() for k, v in state_dict["k_scales"].items()}
        self._v_scales = {k: v.clone() for k, v in state_dict["v_scales"].items()}
        self._cache_stats = state_dict["cache_stats"].copy()