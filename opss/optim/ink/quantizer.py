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
Ink Block Quantizer - INT8/INT4 State Compression

This module implements block-wise quantization for optimizer state compression,
enabling significant memory savings while maintaining training stability.

Key Features:
    - INT8 momentum compression (4x memory savings)
    - INT8 variance compression (4x memory savings)
    - Per-block scaling for dynamic range preservation
    - Per-channel quantization for weight quantization
    - Mixed precision mapping for layer types
    - Quantization-Aware Training (QAT) support

Algorithm:
    Block-wise quantization preserves local dynamic range by computing
    per-block scale factors. This is critical for optimizer states which
    have highly non-uniform distributions.

    Per-channel quantization computes scale per output channel, providing
    finer granularity for weight quantization.

    Mixed precision assigns different bit-widths based on layer sensitivity:
    - attention.qkv, attention.proj: FP8 (higher precision)
    - mlp layers: INT4 (lower precision)

    QAT simulates quantization during training via Straight-Through Estimator.

Memory Savings:
    - FP32: 32 bits per value
    - INT8: 8 bits per value + 32 bits per block scale
    - INT4: 4 bits per value + 32 bits per block scale
    - Per-channel: Additional granularity for better accuracy

    For block_size=128:
    - INT8: ~4x compression (32 → 8.25 bits effective)
    - INT4: ~8x compression (32 → 4.25 bits effective)
"""

import torch
from typing import Tuple, Optional, Union
from configs.version import VERSION


class PiscesLxInkBlockQuantizer:
    """
    Block-wise Quantizer for Optimizer State Compression.
    
    This class provides efficient INT8 and INT4 quantization with per-block
    scaling, enabling significant memory savings for optimizer states while
    maintaining training stability.
    
    The block-wise approach is critical for optimizer states because:
    1. Momentum and variance have highly non-uniform distributions
    2. Different parameters have vastly different magnitude ranges
    3. Per-block scaling preserves local dynamic range
    
    Attributes:
        momentum_bits: Number of bits for momentum quantization (4 or 8)
        variance_bits: Number of bits for variance quantization (4 or 8)
        momentum_block_size: Block size for momentum quantization
        variance_block_size: Block size for variance quantization
    
    Example:
        >>> quantizer = PiscesLxInkBlockQuantizer(
        ...     momentum_bits=8,
        ...     variance_bits=4,
        ...     momentum_block_size=128,
        ...     variance_block_size=256
        ... )
        >>> 
        >>> # Quantize momentum to INT8
        >>> momentum = torch.randn(1024, 1024)
        >>> quantized, scales = quantizer.quantize_int8(momentum, block_size=128)
        >>> 
        >>> # Dequantize back
        >>> reconstructed = quantizer.dequantize_int8(quantized, scales, momentum.shape)
        >>> 
        >>> # Quantize variance to INT4
        >>> variance = torch.rand(1024, 1024)
        >>> packed, scales = quantizer.quantize_int4(variance, block_size=256)
        >>> reconstructed = quantizer.dequantize_int4(packed, scales, variance.shape)
    """
    
    def __init__(
        self,
        momentum_bits: int = 8,
        variance_bits: int = 4,
        momentum_block_size: int = 128,
        variance_block_size: int = 256,
    ):
        """
        Initialize the block quantizer.
        
        Args:
            momentum_bits: Number of bits for momentum (4 or 8)
            variance_bits: Number of bits for variance (4 or 8)
            momentum_block_size: Block size for momentum quantization
            variance_block_size: Block size for variance quantization
        """
        self.momentum_bits = momentum_bits
        self.variance_bits = variance_bits
        self.momentum_block_size = momentum_block_size
        self.variance_block_size = variance_block_size
        
        self._int8_max = 127.0
        self._int4_max = 7.0
    
    @staticmethod
    def _quantise_block_rows(rows, divisor, clip_lo, clip_hi):
        """Quantise aligned block rows; rows is [blocks, width] float32."""
        block_max = rows.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
        scales = block_max / divisor
        quantised = torch.clamp(torch.round(rows / scales), min=clip_lo, max=clip_hi).to(torch.int8)
        return quantised, scales

    def _quantise_flat_blocks(self, flat, block_size, divisor, clip_lo, clip_hi):
        """Quantise a flat tensor as an aligned head plus one ragged tail block.

        Padding the tensor up to a whole number of blocks allocated a second full-size
        fp32 copy of every parameter, twice per step (momentum and variance). Splitting the
        aligned head from the tail performs the same arithmetic per element with no extra
        copy, which is what made the host's commit charge fail during training.
        """
        num_elements = flat.numel()
        full_blocks = num_elements // block_size
        quantised_parts = []
        scale_parts = []
        if full_blocks:
            rows = flat[:full_blocks * block_size].view(full_blocks, block_size)
            rows_q, rows_s = self._quantise_block_rows(rows.float(), divisor, clip_lo, clip_hi)
            quantised_parts.append(rows_q.reshape(-1))
            scale_parts.append(rows_s.reshape(-1))
        tail = flat[full_blocks * block_size:]
        if tail.numel():
            rows_q, rows_s = self._quantise_block_rows(tail.float().unsqueeze(0), divisor, clip_lo, clip_hi)
            quantised_parts.append(rows_q.reshape(-1))
            scale_parts.append(rows_s.reshape(-1))
        return torch.cat(quantised_parts), torch.cat(scale_parts)

    def _dequantise_flat_blocks(self, quantised, scales, block_size, num_elements, dtype):
        """Reconstruct a flat tensor written by ``_quantise_flat_blocks``."""
        out = torch.empty(num_elements, dtype=dtype, device=quantised.device)
        full_blocks = num_elements // block_size
        position = 0
        block_index = 0
        if full_blocks:
            rows = quantised[:full_blocks * block_size].view(full_blocks, block_size)
            out[:full_blocks * block_size] = (rows.to(dtype)
                                              * scales[block_index:block_index + full_blocks].unsqueeze(-1)
                                              ).reshape(-1)
            position = full_blocks * block_size
            block_index = full_blocks
        if position < num_elements:
            rows = quantised[position:num_elements].unsqueeze(0)
            out[position:num_elements] = (rows.to(dtype) * scales[block_index].reshape(1, 1)).reshape(-1)
        return out

    def quantize_int8(
        self,
        tensor: torch.Tensor,
        block_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to INT8 with per-block scaling.
        
        This method divides the tensor into blocks and quantizes each block
        independently with its own scale factor, preserving local dynamic range.
        
        Args:
            tensor: Input tensor to quantize (any shape, will be flattened)
            block_size: Block size for quantization (default: momentum_block_size)
        
        Returns:
            Tuple of:
                - quantized: INT8 quantized tensor (torch.int8)
                - scales: Per-block scale factors (torch.float32)
        
        Memory Savings:
            Original: 32 bits per element
            Quantized: 8 bits per element + 32 bits per block
            For block_size=128: ~4x compression
        """
        block_size = block_size or self.momentum_block_size
        original_shape = tensor.shape
        flat = tensor.reshape(-1)
        quantized_flat, scales = self._quantise_flat_blocks(
            flat, block_size, self._int8_max, -128.0, 127.0)
        return quantized_flat.view(original_shape), scales
    
    def dequantize_int8(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        original_shape: Tuple[int, ...],
        block_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Dequantize INT8 tensor back to floating point.

        Reconstructs the original tensor from quantized values and scale factors.

        Args:
            quantized: INT8 quantized tensor
            scales: Per-block scale factors (one scalar per block)
            original_shape: Original tensor shape for reconstruction
            block_size: The block size used during quantization.
                        Required for correct reconstruction when block_size
                        differs between momentum (128) and variance (256).

        Returns:
            Reconstructed floating point tensor
        """
        device = quantized.device

        flat_quantized = quantized.flatten().float()
        num_elements = flat_quantized.numel()
        num_blocks = scales.numel()

        if block_size is None:
            block_size = (num_elements + num_blocks - 1) // num_blocks
        padded_size = num_blocks * block_size

        if padded_size > num_elements:
            padding = torch.zeros(
                padded_size - num_elements,
                dtype=torch.float32,
                device=device
            )
            flat_quantized = torch.cat([flat_quantized, padding])

        # Reconstruct straight from the aligned-head-plus-tail layout written by
        # _quantise_flat_blocks: padding the stored buffer first cost a full-size
        # temporary per parameter per step.
        return self._dequantise_flat_blocks(
            flat_quantized, scales.reshape(-1), block_size, num_elements, scales.dtype
        ).view(original_shape)
    
    def quantize_int4(
        self,
        tensor: torch.Tensor,
        block_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to INT4 with per-block scaling and packing.
        
        This method quantizes to INT4 range [-7, 7] and packs two values
        into one INT8 for efficient storage.
        
        Args:
            tensor: Input tensor to quantize
            block_size: Block size for quantization (default: variance_block_size)
        
        Returns:
            Tuple of:
                - packed: Packed INT4 tensor (two values per INT8)
                - scales: Per-block scale factors
        
        Memory Savings:
            Original: 32 bits per element
            Quantized: 4 bits per element + 32 bits per block
            For block_size=256: ~8x compression
        """
        block_size = block_size or self.variance_block_size
        original_shape = tensor.shape
        device = tensor.device
        
        flat_tensor = tensor.flatten().float()
        num_elements = flat_tensor.numel()
        
        num_blocks = (num_elements + block_size - 1) // block_size
        padded_size = num_blocks * block_size
        
        if padded_size > num_elements:
            padding = torch.zeros(
                padded_size - num_elements,
                dtype=torch.float32,
                device=device
            )
            flat_tensor = torch.cat([flat_tensor, padding])
        
        blocks = flat_tensor.view(num_blocks, block_size)
        
        block_max = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
        scales = block_max / self._int4_max
        
        scaled_blocks = blocks / scales
        
        quantized_blocks = torch.clamp(
            torch.round(scaled_blocks),
            min=-8.0,
            max=7.0
        )
        
        packed = self._pack_int4(quantized_blocks)
        
        return packed, scales.squeeze(-1)
    
    def dequantize_int4(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        original_shape: Tuple[int, ...],
        block_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Dequantize packed INT4 tensor back to floating point.

        Unpacks the INT4 values and reconstructs the original tensor.

        Args:
            packed: Packed INT4 tensor (two values per INT8)
            scales: Per-block scale factors
            original_shape: Original tensor shape for reconstruction
            block_size: The block size used during quantization.

        Returns:
            Reconstructed floating point tensor
        """
        device = packed.device

        unpacked = self._unpack_int4(packed)

        num_elements = original_shape.numel() if isinstance(original_shape, tuple) else original_shape

        if unpacked.numel() > num_elements:
            unpacked = unpacked[:num_elements]

        num_blocks = scales.numel()
        if block_size is None:
            block_size = (num_elements + num_blocks - 1) // num_blocks
        padded_size = num_blocks * block_size
        
        if unpacked.numel() < padded_size:
            padding = torch.zeros(
                padded_size - unpacked.numel(),
                dtype=torch.float32,
                device=device
            )
            unpacked = torch.cat([unpacked, padding])
        
        blocks = unpacked.view(num_blocks, block_size)
        
        scaled_blocks = blocks * scales.unsqueeze(-1)
        
        result = scaled_blocks.flatten()[:num_elements]
        
        return result.view(original_shape)
    
    def _pack_int4(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pack two INT4 values into one INT8.
        
        Packing format: high_nibble | low_nibble
        Each nibble stores one INT4 value in range [-8, 7].
        
        Args:
            tensor: Tensor with INT4 values (any shape)
        
        Returns:
            Packed tensor with half the elements
        """
        flat = tensor.flatten().to(torch.int8)
        
        num_elements = flat.numel()
        padded_num = (num_elements + 1) // 2 * 2
        
        if padded_num > num_elements:
            padding = torch.zeros(padded_num - num_elements, dtype=torch.int8, device=flat.device)
            flat = torch.cat([flat, padding])
        
        flat = flat.view(-1, 2)
        
        high = (flat[:, 0] & 0x0F).to(torch.int8)
        low = (flat[:, 1] & 0x0F).to(torch.int8)
        
        packed = (high << 4) | low
        
        return packed
    
    def _unpack_int4(self, packed: torch.Tensor) -> torch.Tensor:
        """
        Unpack INT8 to two INT4 values.
        
        Args:
            packed: Packed tensor with INT8 values
        
        Returns:
            Unpacked tensor with twice the elements
        """
        packed = packed.to(torch.int8)
        
        low = (packed & 0x0F).to(torch.float32)
        high = ((packed >> 4) & 0x0F).to(torch.float32)
        
        low = torch.where(low > 7, low - 16, low)
        high = torch.where(high > 7, high - 16, high)
        
        unpacked = torch.stack([high, low], dim=-1).flatten()
        
        return unpacked
    
    def _variance_log_grid(self, rows: torch.Tensor, levels: int):
        """Log-uniform grid for non-negative block rows; returns (codes, log_lo, log_step).

        Variance is a running sum of squares: non-negative and spanning several orders of
        magnitude inside one block, which is exactly the distribution a symmetric absmax codec
        destroys (an entry two orders below its block maximum rounds to 0, and 0 in the
        denominator turns Adam's bounded sign step into an unbounded one). Coding log(v)
        instead makes the error multiplicative and uniform: with step = (log vmax - log
        vmin)/(levels-2) every positive entry satisfies exp(-step/2) <= v_hat/v <= exp(step/2).
        Code 0 is reserved for a true zero, so an entry that has never seen a gradient
        reconstructs as exactly 0 and cannot inflate the denominator either way.
        """
        tiny = torch.finfo(torch.float32).tiny
        row_max = rows.max(dim=1, keepdim=True).values
        row_min = torch.where(rows > 0, rows, torch.full_like(rows, float("inf"))
                              ).min(dim=1, keepdim=True).values
        has_positive = row_max > 0
        log_hi = torch.where(has_positive, row_max.clamp_min(tiny).log(),
                             torch.zeros_like(row_max))
        log_lo = torch.where(has_positive & (row_min > 0) & (row_min < float("inf")),
                             row_min.clamp_min(tiny).log(), log_hi)
        intervals = max(levels - 2, 1)
        log_step = ((log_hi - log_lo) / intervals).clamp(min=0.0)
        positive = rows > 0
        log_rows = rows.clamp_min(tiny).log()
        codes = torch.where(
            positive,
            (1.0 + (log_rows - log_lo) / log_step.clamp_min(tiny)).round().clamp(
                min=1.0, max=float(levels - 1)),
            torch.zeros_like(rows),
        )
        return codes.to(torch.uint8), log_lo.squeeze(-1), log_step.squeeze(-1)

    def _pack_unsigned_nibbles(self, codes: torch.Tensor) -> torch.Tensor:
        """Pack two 4-bit unsigned codes per byte (low nibble first)."""
        flat = codes.reshape(-1).to(torch.uint8)
        num_elements = flat.numel()
        if num_elements % 2:
            flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
        pairs = flat.view(-1, 2)
        return (pairs[:, 1] << 4) | pairs[:, 0]

    def _unpack_unsigned_nibbles(self, packed: torch.Tensor, num_elements: int) -> torch.Tensor:
        """Inverse of ``_pack_unsigned_nibbles`` truncated to ``num_elements`` codes."""
        bytes_ = packed.reshape(-1).to(torch.uint8)
        low = bytes_ & 0x0F
        high = (bytes_ >> 4) & 0x0F
        return torch.stack([low, high], dim=-1).reshape(-1)[:num_elements]

    def quantize_variance_log(
        self,
        variance: torch.Tensor,
        block_size: Optional[int] = None,
        bits: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise non-negative state on a per-block log grid.

        Args:
            variance: Non-negative tensor (any shape, flattened into blocks)
            block_size: Elements per block (default: variance_block_size)
            bits: 8 or 4, i.e. 256 or 16 codes per block

        Returns:
            (codes, log_lo, log_step) -- codes are packed two-per-byte when bits == 4,
            one per element otherwise; log_lo/log_step hold one float32 per block.
        """
        block_size = block_size or self.variance_block_size
        bits = bits or self.variance_bits
        levels = 256 if bits == 8 else 16
        flat = variance.reshape(-1).float()
        num_elements = flat.numel()
        full_blocks = num_elements // block_size

        code_parts, lo_parts, step_parts = [], [], []
        if full_blocks:
            rows = flat[:full_blocks * block_size].view(full_blocks, block_size)
            codes, log_lo, log_step = self._variance_log_grid(rows, levels)
            code_parts.append(codes.reshape(-1))
            lo_parts.append(log_lo)
            step_parts.append(log_step)
        tail = flat[full_blocks * block_size:]
        if tail.numel():
            codes, log_lo, log_step = self._variance_log_grid(tail.unsqueeze(0), levels)
            code_parts.append(codes.reshape(-1))
            lo_parts.append(log_lo)
            step_parts.append(log_step)
        codes = torch.cat(code_parts)
        if bits == 4:
            codes = self._pack_unsigned_nibbles(codes)
        return codes, torch.cat(lo_parts), torch.cat(step_parts)

    def dequantize_variance_log(
        self,
        codes: torch.Tensor,
        log_lo: torch.Tensor,
        log_step: torch.Tensor,
        num_elements: int,
        block_size: Optional[int] = None,
        bits: Optional[int] = None,
    ) -> torch.Tensor:
        """Reconstruct variance written by ``quantize_variance_log`` as a flat tensor."""
        block_size = block_size or self.variance_block_size
        bits = bits or self.variance_bits
        device = codes.device
        if bits == 4:
            codes = self._unpack_unsigned_nibbles(codes, num_elements)
        else:
            codes = codes.reshape(-1).to(torch.uint8)[:num_elements]
        out = torch.zeros(num_elements, dtype=torch.float32, device=device)
        position = 0
        block_index = 0
        full_blocks = num_elements // block_size
        if full_blocks:
            width = full_blocks * block_size
            rows = codes[:width].view(full_blocks, block_size).to(torch.float32)
            exponent = log_lo[block_index:block_index + full_blocks].unsqueeze(-1) \
                + (rows - 1.0) * log_step[block_index:block_index + full_blocks].unsqueeze(-1)
            out[:width] = torch.where(rows > 0, exponent.exp(), torch.zeros_like(exponent)
                                      ).reshape(-1)
            position = width
            block_index = full_blocks
        if position < num_elements:
            rows = codes[position:num_elements].to(torch.float32)
            exponent = log_lo[block_index] + (rows - 1.0) * log_step[block_index]
            out[position:num_elements] = torch.where(
                rows > 0, exponent.exp(), torch.zeros_like(exponent))
        return out

    def quantize_momentum(
        self,
        momentum: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize momentum tensor using configured settings.
        
        Args:
            momentum: Momentum tensor to quantize
        
        Returns:
            Tuple of (quantized tensor, scales)
        """
        if self.momentum_bits == 8:
            return self.quantize_int8(momentum, self.momentum_block_size)
        else:
            return self.quantize_int4(momentum, self.momentum_block_size)
    
    def dequantize_momentum(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Dequantize momentum tensor using configured settings.
        
        Args:
            quantized: Quantized momentum tensor
            scales: Per-block scale factors
            original_shape: Original tensor shape
        
        Returns:
            Reconstructed momentum tensor
        """
        if self.momentum_bits == 8:
            return self.dequantize_int8(quantized, scales, original_shape, block_size=self.momentum_block_size)
        else:
            return self.dequantize_int4(quantized, scales, original_shape, block_size=self.momentum_block_size)

    def quantize_variance(
        self,
        variance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize variance tensor using configured settings.

        Variance is stored on the log grid rather than with the symmetric absmax codec:
        the state is non-negative and heavy-tailed, and only the log grid keeps every
        positive entry strictly positive after reconstruction (docs/proofs/02).

        Args:
            variance: Variance tensor to quantize

        Returns:
            Tuple of (codes, log_lo, log_step)
        """
        return self.quantize_variance_log(variance, self.variance_block_size)
    
    def dequantize_variance(
        self,
        quantized: torch.Tensor,
        log_lo: torch.Tensor,
        log_step: torch.Tensor,
        num_elements: int,
    ) -> torch.Tensor:
        """Dequantize variance tensor using configured settings.

        Args:
            quantized: Codes written by ``quantize_variance``
            log_lo: Per-block lower end of the log grid
            log_step: Per-block log spacing
            num_elements: Number of elements to reconstruct

        Returns:
            Reconstructed variance tensor (flat)
        """
        return self.dequantize_variance_log(
            quantized, log_lo, log_step, num_elements,
            block_size=self.variance_block_size)

    def compute_compression_ratio(
        self,
        num_elements: int,
        bits: int,
        block_size: int,
    ) -> float:
        """
        Compute compression ratio for given parameters.
        
        Args:
            num_elements: Number of tensor elements
            bits: Bits per quantized value (4 or 8)
            block_size: Block size for quantization
        
        Returns:
            Compression ratio (original size / compressed size)
        """
        original_bits = num_elements * 32
        
        num_blocks = (num_elements + block_size - 1) // block_size
        quantized_bits = num_elements * bits
        scale_bits = num_blocks * 32
        
        compressed_bits = quantized_bits + scale_bits
        
        return original_bits / compressed_bits
    
    def get_memory_stats(self) -> dict:
        """
        Get memory statistics for the quantizer.

        Returns:
            Dictionary with memory statistics
        """
        return {
            "momentum_bits": self.momentum_bits,
            "variance_bits": self.variance_bits,
            "momentum_block_size": self.momentum_block_size,
            "variance_block_size": self.variance_block_size,
            "momentum_compression": self.compute_compression_ratio(
                1000000, self.momentum_bits, self.momentum_block_size
            ),
            "variance_compression": self.compute_compression_ratio(
                1000000, self.variance_bits, self.variance_block_size
            ),
        }

    def quantize_per_channel(
        self,
        tensor: torch.Tensor,
        bits: int = 8,
        channel_dim: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor per output channel.

        Each output channel has its own scale factor, providing finer
        granularity and reducing quantization error.

        Args:
            tensor: Input tensor [out_channels, in_channels, ...]
            bits: Quantization bits (8 or 4)
            channel_dim: Dimension representing output channels (default: 0)

        Returns:
            Tuple of (quantized tensor, per-channel scales)
        """
        original_shape = tensor.shape
        device = tensor.device

        if bits == 8:
            max_val = 127.0
            dtype = torch.int8
        else:
            max_val = 7.0
            dtype = torch.int8

        channel_size = original_shape[channel_dim]
        other_dims = [original_shape[i] for i in range(len(original_shape)) if i != channel_dim]
        num_elements_per_channel = 1
        for dim_size in other_dims:
            num_elements_per_channel *= dim_size

        flat_tensor = tensor.flatten(channel_dim + 1 if channel_dim == 0 else 0)
        if channel_dim != 0:
            flat_tensor = flat_tensor.transpose(0, channel_dim)
        flat_tensor = flat_tensor.flatten(1)

        channel_max = flat_tensor.abs().max(dim=1).values.clamp(min=1e-12)
        scales = channel_max / max_val

        scaled = flat_tensor / scales.unsqueeze(-1)
        quantized = torch.clamp(torch.round(scaled), -max_val, max_val).to(dtype)

        quantized_per_channel = quantized.view(channel_size, -1)
        scales_per_channel = scales

        result_shape = list(original_shape)
        result_shape[channel_dim] = channel_size
        quantized_result = quantized_per_channel.view(result_shape)

        return quantized_result, scales_per_channel

    def dequantize_per_channel(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        original_shape: Tuple[int, ...],
        channel_dim: int = 0,
    ) -> torch.Tensor:
        """
        Dequantize per-channel quantized tensor.

        Args:
            quantized: Quantized tensor
            scales: Per-channel scale factors
            original_shape: Original tensor shape
            channel_dim: Dimension representing output channels

        Returns:
            Dequantized tensor
        """
        if channel_dim != 0:
            quantized = quantized.transpose(channel_dim, 0)
        dequant_flat = quantized.flatten(1) * scales.unsqueeze(-1)
        dequant = dequant_flat.reshape(quantized.shape[0], *original_shape[1:])
        if channel_dim != 0:
            dequant = dequant.transpose(0, channel_dim)
        return dequant.reshape(original_shape)

    def quantize_with_mapping(
        self,
        tensor: torch.Tensor,
        layer_name: str,
        default_bits: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Quantize tensor with mixed precision based on layer type.

        Sensitive layers (attention, proj) use higher precision.
        Other layers (mlp, fc) use lower precision.

        Args:
            tensor: Input tensor to quantize
            layer_name: Name of the layer for type detection
            default_bits: Default bits if no mapping matches

        Returns:
            Tuple of (quantized tensor, scales, bits used)
        """
        layer_name_lower = layer_name.lower()

        if "attn" in layer_name_lower or "qkv" in layer_name_lower or "proj" in layer_name_lower:
            bits = 8
        elif "mlp" in layer_name_lower or "fc" in layer_name_lower or "ffn" in layer_name_lower:
            bits = 4
        elif "embed" in layer_name_lower or "pos" in layer_name_lower:
            bits = 8
        else:
            bits = default_bits

        if bits == 8:
            return self.quantize_int8(tensor, self.momentum_block_size), self._compute_scales_int8(tensor), bits
        else:
            return self.quantize_int4(tensor, self.momentum_block_size), self._compute_scales_int4(tensor), bits

    def _compute_scales_int8(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute per-block scales for INT8."""
        block_size = self.momentum_block_size
        flat = tensor.flatten().float()
        num_blocks = (flat.numel() + block_size - 1) // block_size
        padded = torch.zeros(num_blocks * block_size, device=flat.device, dtype=flat.dtype)
        padded[:flat.numel()] = flat
        blocks = padded.view(num_blocks, block_size)
        block_max = blocks.abs().max(dim=1).values.clamp(min=1e-12)
        return (block_max / 127.0).squeeze(-1)

    def _compute_scales_int4(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute per-block scales for INT4."""
        block_size = self.momentum_block_size
        flat = tensor.flatten().float()
        num_blocks = (flat.numel() + block_size - 1) // block_size
        padded = torch.zeros(num_blocks * block_size, device=flat.device, dtype=flat.dtype)
        padded[:flat.numel()] = flat
        blocks = padded.view(num_blocks, block_size)
        block_max = blocks.abs().max(dim=1).values.clamp(min=1e-12)
        return (block_max / 7.0).squeeze(-1)

    def qat_forward(self, tensor: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantization-Aware Training forward pass with Straight-Through Estimator.

        Forward: Quantize tensor (round to discrete levels)
        Backward: Pass gradient through unchanged (STE)

        Args:
            tensor: Input tensor (float32)
            bits: Quantization bits (8 or 4)

        Returns:
            Tuple of (quantized value for forward, scale for dequantization)
        """
        if bits == 8:
            max_val = 127.0
            block_size = self.momentum_block_size
        else:
            max_val = 7.0
            block_size = self.momentum_block_size

        flat = tensor.flatten().float()
        num_blocks = (flat.numel() + block_size - 1) // block_size
        padded = torch.zeros(num_blocks * block_size, device=flat.device, dtype=flat.dtype)
        padded[:flat.numel()] = flat
        blocks = padded.view(num_blocks, block_size)

        block_max = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
        scales = (block_max / max_val).squeeze(-1)

        scaled = blocks / block_max * max_val
        quantized = torch.clamp(torch.round(scaled), -max_val, max_val)

        fake_quantized = quantized / max_val * block_max.squeeze(-1)
        fake_quantized = fake_quantized.flatten()[:flat.numel()].view_as(tensor)

        return fake_quantized, scales

    def qat_backward(self, grad_output: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        QAT backward pass - Straight-Through Estimator.

        Gradient passes through unchanged during STE.

        Args:
            grad_output: Gradient from next layer
            scales: Scale factors (used for shape info if needed)

        Returns:
            Gradient for previous layer
        """
        return grad_output

    def qat_dequantize(self, fake_quantized: torch.Tensor, scales: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Dequantize QAT output.

        Args:
            fake_quantized: Quantized tensor from qat_forward
            scales: Scale factors
            shape: Original shape

        Returns:
            Dequantized tensor
        """
        return fake_quantized.view(shape)
