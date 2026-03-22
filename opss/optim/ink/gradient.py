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

"""
Ink Gradient Compression Module

This module provides INT8 gradient compression for memory-efficient training,
complementing the sparse gradient selection in the main optimizer.

Key Features:
    - INT8 block-wise gradient quantization
    - Stochastic rounding for statistical accuracy
    - Adaptive compression based on gradient magnitude
    - Integration with sparse selection

Memory Savings:
    - FP32 gradient: 4 bytes per element
    - INT8 gradient: 1 byte per element (4x reduction)
    - Combined with sparse selection: up to 100x total reduction
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from configs.version import VERSION


class POPSSInkGradientCompressor:
    """
    INT8 Gradient Compressor.
    
    Compresses gradients to INT8 format with block-wise scaling for memory
    efficiency during training. Works with sparse gradient selection to
    maximize memory savings.
    
    Attributes:
        gradient_bits: Quantization bits (8 for INT8)
        block_size: Block size for per-block quantization
        stochastic_rounding: Whether to use stochastic rounding
    
    Example:
        >>> compressor = POPSSInkGradientCompressor(gradient_bits=8, block_size=128)
        >>> compressed_grad, scales = compressor.compress(gradient)
        >>> decompressed = compressor.decompress(compressed_grad, scales, shape)
    """
    
    def __init__(
        self,
        gradient_bits: int = 8,
        block_size: int = 128,
        stochastic_rounding: bool = True,
    ):
        self.gradient_bits = gradient_bits
        self.block_size = block_size
        self.stochastic_rounding = stochastic_rounding
        
        self._compression_stats: Dict[str, Any] = {
            "total_compressions": 0,
            "total_decompressions": 0,
            "avg_compression_ratio": 0.0,
        }
    
    def compress(
        self,
        gradient: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress gradient to INT8.
        
        Args:
            gradient: Input gradient tensor
        
        Returns:
            Tuple of (compressed_gradient, scales)
        """
        orig_shape = gradient.shape
        grad_flat = gradient.flatten()
        numel = grad_flat.numel()
        num_blocks = (numel + self.block_size - 1) // self.block_size
        
        padded_size = num_blocks * self.block_size
        if padded_size > numel:
            padding = torch.zeros(
                padded_size - numel,
                dtype=grad_flat.dtype,
                device=grad_flat.device,
            )
            grad_flat = torch.cat([grad_flat, padding])
        
        grad_blocks = grad_flat.view(num_blocks, self.block_size)
        
        block_max = grad_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        
        if self.gradient_bits == 8:
            max_val = 127.0
            dtype = torch.int8
        else:
            max_val = 7.0
            dtype = torch.uint8
        
        scales = block_max / max_val
        
        scaled = grad_blocks / scales
        scaled = torch.clamp(scaled, -max_val, max_val)
        
        if self.stochastic_rounding:
            compressed = self._stochastic_round(scaled, dtype)
        else:
            compressed = torch.round(scaled).to(dtype)
        
        self._compression_stats["total_compressions"] += 1
        
        return compressed, scales.squeeze(-1)
    
    def decompress(
        self,
        compressed: torch.Tensor,
        scales: torch.Tensor,
        shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Decompress gradient from INT8.
        
        Args:
            compressed: Compressed gradient
            scales: Per-block scales
            shape: Original tensor shape
        
        Returns:
            Decompressed gradient tensor
        """
        numel = shape.numel()
        num_blocks = scales.numel()
        
        dequant_blocks = compressed.float() * scales.unsqueeze(-1)
        dequant_flat = dequant_blocks.flatten()[:numel]
        
        self._compression_stats["total_decompressions"] += 1
        
        return dequant_flat.view(shape)
    
    def _stochastic_round(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Stochastic rounding for better statistical properties."""
        if dtype == torch.int8:
            low = -128.0
            high = 127.0
        else:
            low = 0.0
            high = 255.0
        
        noise = torch.rand_like(tensor)
        rounded = torch.floor(tensor + noise)
        clamped = torch.clamp(rounded, low, high)
        
        return clamped.to(dtype)
    
    def compress_sparse(
        self,
        gradient: torch.Tensor,
        sparse_ratio: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress only top-K% of gradient elements (sparse compression).
        
        Args:
            gradient: Input gradient tensor
            sparse_ratio: Fraction of elements to keep (0.01 = top 1%)
        
        Returns:
            Tuple of (compressed_values, scales, mask)
        """
        orig_shape = gradient.shape
        grad_flat = gradient.flatten()
        numel = grad_flat.numel()
        
        k = max(1, int(numel * sparse_ratio))
        
        abs_grad = grad_flat.abs()
        threshold = torch.kthvalue(abs_grad, numel - k).values
        
        mask = abs_grad >= threshold
        sparse_grad = grad_flat * mask.float()
        
        masked_blocks = sparse_grad.masked_fill(~mask, 0).view(-1, self.block_size)
        
        block_max = masked_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        
        if self.gradient_bits == 8:
            max_val = 127.0
            dtype = torch.int8
        else:
            max_val = 7.0
            dtype = torch.uint8
        
        scales = block_max / max_val
        scaled = masked_blocks / scales
        scaled = torch.clamp(scaled, -max_val, max_val)
        
        if self.stochastic_rounding:
            compressed = self._stochastic_round(scaled, dtype)
        else:
            compressed = torch.round(scaled).to(dtype)
        
        return compressed, scales.squeeze(-1), mask
    
    def decompress_sparse(
        self,
        compressed: torch.Tensor,
        scales: torch.Tensor,
        mask: torch.Tensor,
        shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Decompress sparse gradient.
        
        Args:
            compressed: Compressed values
            scales: Per-block scales
            mask: Boolean mask for sparse positions
            shape: Original tensor shape
        
        Returns:
            Decompressed gradient tensor
        """
        num_blocks = scales.numel()
        block_size = self.block_size
        
        dequant_blocks = compressed.float() * scales.unsqueeze(-1)
        dequant_flat = dequant_blocks.flatten()
        
        if dequant_flat.numel() < mask.numel():
            padding = torch.zeros(
                mask.numel() - dequant_flat.numel(),
                dtype=dequant_flat.dtype,
                device=dequant_flat.device,
            )
            dequant_flat = torch.cat([dequant_flat, padding])
        
        dequant_flat = dequant_flat[:mask.numel()]
        dequant = dequant_flat.view(shape)
        
        sparse_result = dequant * mask.float()
        
        return sparse_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compression statistics."""
        stats = self._compression_stats.copy()
        total_ops = stats["total_compressions"] + stats["total_decompressions"]
        if total_ops > 0:
            stats["avg_compression_ratio"] = 4.0 if self.gradient_bits == 8 else 8.0
        return stats
    
    def reset_statistics(self):
        """Reset compression statistics."""
        self._compression_stats = {
            "total_compressions": 0,
            "total_decompressions": 0,
            "avg_compression_ratio": 0.0,
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state for serialization."""
        return {
            "gradient_bits": self.gradient_bits,
            "block_size": self.block_size,
            "stochastic_rounding": self.stochastic_rounding,
            "compression_stats": self._compression_stats.copy(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from dictionary."""
        self.gradient_bits = state_dict["gradient_bits"]
        self.block_size = state_dict["block_size"]
        self.stochastic_rounding = state_dict["stochastic_rounding"]
        self._compression_stats = state_dict["compression_stats"].copy()