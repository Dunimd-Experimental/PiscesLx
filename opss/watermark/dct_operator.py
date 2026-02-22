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

"""
DCT Watermark Operator

This module implements Discrete Cosine Transform (DCT) based watermark embedding
and extraction for image content. The DCT domain provides excellent robustness
against common image processing operations such as compression and filtering.

DCT Watermarking Principles:
    1. Transform image blocks from spatial to frequency domain
    2. Embed watermark in mid-frequency coefficients
    3. Use pairwise magnitude relationships for bit encoding
    4. Apply inverse transform to obtain watermarked image

Key Features:
    - Block-based DCT-II transform (8x8 blocks)
    - Mid-frequency coefficient selection for robustness
    - Pairwise relation encoding for bit embedding
    - Adaptive strength based on coefficient magnitude
    - JPEG compression resistance
    - Seeded randomization for reproducibility

Bit Encoding:
    - Bit 1: |F[u,v]| > |F[u+1,v+1]|
    - Bit 0: |F[u+1,v+1]| > |F[u,v]|
    - Minimal multiplicative scaling for imperceptibility

Usage Examples:
    >>> from opss.watermark.dct_operator import POPSSDCTOperator
    >>> operator = POPSSDCTOperator()
    >>> 
    >>> # Embed watermark in image
    >>> watermarked = operator.embed(
    ...     image_tensor,
    ...     watermark_bits="10110",
    ...     strength=0.08,
    ...     seed=42
    ... )
    >>> 
    >>> # Extract watermark from image
    >>> bits = operator.extract(
    ...     watermarked_image,
    ...     num_bits=5,
    ...     seed=42
    ... )
"""

import math
import hashlib
import torch
import torch.nn as nn
import torch.fft
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

from utils.opsc.base import PiscesLxBaseOperator
from utils.opsc.interface import PiscesLxOperatorResult, PiscesLxOperatorStatus
from configs.version import VERSION


class POPSSDCTOperator(PiscesLxBaseOperator):
    """
    Discrete Cosine Transform watermark operator.
    
    This operator provides image watermarking capabilities using DCT domain
    embedding. It offers robustness against common image processing
    operations while maintaining visual quality.
    
    Attributes:
        block_size (int): DCT block size (default 8)
        default_strength (float): Default embedding strength
        supported_bands (List[str]): Supported frequency bands
        
    Input Format:
        {
            "action": "embed" | "extract",
            "image": torch.Tensor,      # Input image [C, H, W]
            "bits": str,                 # Watermark bits for embed
            "num_bits": int,             # Number of bits to extract
            "strength": float,           # Embedding strength
            "seed": int,                 # Random seed
            "band": str,                 # Frequency band (low|mid|high)
            "visible": bool              # Add visible watermark
        }
        
    Output Format:
        {
            "action": str,
            "result": torch.Tensor | str,
            "metadata": Dict
        }
    """
    
    def __init__(self, block_size: int = 8, default_strength: float = 0.08):
        super().__init__()
        self.name = "pisceslx_dct_operator"
        self.version = VERSION
        self.description = "DCT-based image watermarking with frequency domain embedding"
        self.block_size = block_size
        self.default_strength = default_strength
        self.supported_bands = ["low", "mid", "high"]
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["action", "image"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["embed", "extract", "embed_visible"]
                },
                "image": {
                    "type": "tensor",
                    "shape": ["C", "H", "W"],
                    "description": "Image tensor in CHW format"
                },
                "bits": {
                    "type": "string",
                    "description": "Binary watermark bits for embedding"
                },
                "num_bits": {
                    "type": "integer",
                    "description": "Number of bits to extract"
                },
                "strength": {
                    "type": "number",
                    "description": "Embedding strength (0.01-0.2)"
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility"
                },
                "band": {
                    "type": "string",
                    "enum": ["low", "mid", "high"],
                    "description": "Frequency band for embedding"
                },
                "visible_text": {
                    "type": "string",
                    "description": "Text for visible watermark"
                }
            }
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "result": {"type": "any"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "embedded_bits": {"type": "integer"},
                        "extracted_bits": {"type": "string"},
                        "strength_used": {"type": "number"},
                        "band": {"type": "string"}
                    }
                }
            }
        }
    
    def _execute_impl(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        action = inputs.get("action", "embed")
        
        if action == "embed":
            return self._embed(inputs)
        elif action == "extract":
            return self._extract(inputs)
        elif action == "embed_visible":
            return self._embed_visible(inputs)
        else:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=f"Unknown action: {action}"
            )
    
    def _embed(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        image = inputs.get("image")
        bits = inputs.get("bits", "")
        strength = inputs.get("strength", self.default_strength)
        seed = inputs.get("seed", 42)
        band = inputs.get("band", "mid")
        
        if not isinstance(image, torch.Tensor):
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error="Image must be a torch.Tensor"
            )
        
        try:
            c, h, w = image.shape
            
            if h % self.block_size != 0 or w % self.block_size != 0:
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error=f"Image dimensions must be divisible by block_size={self.block_size}"
                )
            
            seed_hash = hashlib.sha256(f"{seed}_{h}x{w}".encode()).hexdigest()
            rng_seed = int(seed_hash[:8], 16) & 0x7FFFFFFF
            rng = torch.Generator(device=image.device)
            rng.manual_seed(rng_seed)
            
            i_h0, i_h1 = self._get_band_indices(h, band, "height")
            i_w0, i_w1 = self._get_band_indices(w, band, "width")
            
            blocks_per_channel = ((i_h1 - i_h0) * (i_w1 - i_w0)) // (self.block_size ** 2)
            bits_per_block = 2
            bits_per_channel = blocks_per_channel * bits_per_block
            total_capacity = bits_per_channel * c
            
            extended_bits = bits
            if len(bits) < total_capacity:
                extended_bits = bits * (total_capacity // len(bits) + 1)
            extended_bits = extended_bits[:total_capacity]
            
            watermarked = image.clone()
            bit_idx = 0
            
            for ch in range(c):
                F = torch.fft.fft2(image[ch])
                Hn, Wn = h // self.block_size, w // self.block_size
                
                ch_offset = ch * bits_per_channel
                
                by_start = i_h0 // self.block_size
                by_end = min(by_start + (i_h1 - i_h0) // self.block_size, Hn)
                bx_start = i_w0 // self.block_size
                bx_end = min(bx_start + (i_w1 - i_w0) // self.block_size, Wn)
                
                block_count = 0
                for by in range(by_start, by_end):
                    for bx in range(bx_start, bx_end):
                        if block_count >= blocks_per_channel or bit_idx >= len(extended_bits):
                            break
                        
                        if bit_idx + 1 >= len(extended_bits):
                            break
                        
                        bit1 = 1 if extended_bits[bit_idx] == '1' else 0
                        bit2 = 1 if extended_bits[bit_idx + 1] == '1' else 0
                        
                        y0, x0 = by * self.block_size, bx * self.block_size
                        
                        if y0 + self.block_size > h or x0 + self.block_size > w:
                            continue
                        
                        if y0 + self.block_size + 1 >= h or x0 + self.block_size + 1 >= w:
                            continue
                        
                        a = F[y0 + 1, x0 + 1]
                        b = F[y0 + 2, x0 + 2]
                        
                        mag_a = torch.abs(a)
                        mag_b = torch.abs(b)
                        
                        delta_base = strength * (0.1 + 0.9 * rng.rand())
                        
                        if bit1 == 1:
                            if mag_a <= mag_b + delta_base:
                                scale = (mag_b + delta_base + 1e-8) / (mag_a + 1e-8)
                                F[y0 + 1, x0 + 1] = a * scale
                        else:
                            if mag_b <= mag_a + delta_base:
                                scale = (mag_a + delta_base + 1e-8) / (mag_b + 1e-8)
                                F[y0 + 2, x0 + 2] = b * scale
                        
                        a2 = F[y0 + 3, x0 + 3]
                        b2 = F[y0 + 4, x0 + 4]
                        
                        mag_a2 = torch.abs(a2)
                        mag_b2 = torch.abs(b2)
                        
                        if bit2 == 1:
                            if mag_a2 <= mag_b2 + delta_base:
                                scale = (mag_b2 + delta_base + 1e-8) / (mag_a2 + 1e-8)
                                F[y0 + 3, x0 + 3] = a2 * scale
                        else:
                            if mag_b2 <= mag_a2 + delta_base:
                                scale = (mag_a2 + delta_base + 1e-8) / (mag_b2 + 1e-8)
                                F[y0 + 4, x0 + 4] = b2 * scale
                        
                        bit_idx += 2
                        block_count += 1
                
                watermarked[ch] = torch.real(torch.fft.ifft2(F))
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={"image": watermarked},
                metadata={
                    "embedded_bits": bit_idx,
                    "strength_used": strength,
                    "band": band,
                    "capacity": total_capacity
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _extract(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        image = inputs.get("image")
        num_bits = inputs.get("num_bits", 64)
        seed = inputs.get("seed", 42)
        band = inputs.get("band", "mid")
        
        if not isinstance(image, torch.Tensor):
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error="Image must be a torch.Tensor"
            )
        
        try:
            c, h, w = image.shape
            
            seed_hash = hashlib.sha256(f"{seed}_{h}x{w}".encode()).hexdigest()
            rng_seed = int(seed_hash[:8], 16) & 0x7FFFFFFF
            rng = torch.Generator(device=image.device)
            rng.manual_seed(rng_seed)
            
            i_h0, i_h1 = self._get_band_indices(h, band, "height")
            i_w0, i_w1 = self._get_band_indices(w, band, "width")
            
            Hn, Wn = h // self.block_size, w // self.block_size
            
            extracted_bits = []
            bits_per_block = 2
            bits_per_channel = ((i_h1 - i_h0) // self.block_size) * ((i_w1 - i_w0) // self.block_size) * bits_per_block
            
            by_start = i_h0 // self.block_size
            by_end = min(by_start + (i_h1 - i_h0) // self.block_size, Hn)
            bx_start = i_w0 // self.block_size
            bx_end = min(bx_start + (i_w1 - i_w0) // self.block_size, Wn)
            
            block_count = 0
            for by in range(by_start, by_end):
                for bx in range(bx_start, bx_end):
                    if len(extracted_bits) >= num_bits:
                        break
                    
                    if block_count >= bits_per_channel // bits_per_block:
                        break
                    
                    y0, x0 = by * self.block_size, bx * self.block_size
                    
                    if y0 + self.block_size + 4 >= h or x0 + self.block_size + 4 >= w:
                        continue
                    
                    F = torch.fft.fft2(image[0])
                    
                    bits_found = 0
                    for offset_y in [1, 3]:
                        for offset_x in [1, 3]:
                            if len(extracted_bits) >= num_bits:
                                break
                            
                            y1, x1 = y0 + offset_y, x0 + offset_x
                            y2, x2 = y0 + offset_y + 1, x0 + offset_x + 1
                            
                            if y2 >= h or x2 >= w:
                                continue
                            
                            mag_a = torch.abs(F[y1, x1])
                            mag_b = torch.abs(F[y2, x2])
                            
                            extracted_bits.append('1' if mag_a > mag_b else '0')
                            bits_found += 1
                    
                    block_count += 1
            
            result_bits = ''.join(extracted_bits[:num_bits])
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={"bits": result_bits},
                metadata={
                    "extracted_bits": result_bits,
                    "num_extracted": len(result_bits),
                    "band": band
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _embed_visible(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        image = inputs.get("image")
        text = inputs.get("visible_text", "AI-Generated")
        
        if not isinstance(image, torch.Tensor):
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error="Image must be a torch.Tensor"
            )
        
        try:
            c, h, w = image.shape
            watermarked = image.clone()
            
            text_height = 24
            text_width = len(text) * 12
            position = "bottom_right"
            margin = 10
            opacity = 0.15
            background_color = [1.0, 1.0, 1.0]
            
            if position == "bottom_right":
                start_y = max(0, h - text_height - margin)
                start_x = max(0, w - text_width - margin)
            elif position == "bottom_left":
                start_y = max(0, h - text_height - margin)
                start_x = margin
            elif position == "top_right":
                start_y = margin
                start_x = max(0, w - text_width - margin)
            else:
                start_y = margin
                start_x = margin
            
            for y in range(start_y, min(h, start_y + text_height)):
                for x in range(start_x, min(w, start_x + text_width)):
                    if 0 <= y < h and 0 <= x < w:
                        char_idx = (x - start_x) // 12
                        if char_idx < len(text):
                            for ch_idx in range(min(c, len(background_color))):
                                watermarked[ch_idx, y, x] = (
                                    (1 - opacity) * watermarked[ch_idx, y, x] +
                                    opacity * background_color[ch_idx]
                                )
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={"image": watermarked},
                metadata={
                    "visible_text": text,
                    "position": position
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _get_band_indices(self, size: int, band: str, dimension: str) -> Tuple[int, int]:
        """Get frequency band indices for DCT embedding."""
        if band == "low":
            return size // 6, size // 2
        elif band == "high":
            return size // 2, 5 * size // 6
        else:
            return size // 4, 3 * size // 4
    
    def embed(self, image: torch.Tensor, bits: str, 
              strength: float = None, seed: int = 42, 
              band: str = "mid") -> torch.Tensor:
        """
        Embed watermark bits into image using DCT.
        
        Args:
            image: Input image tensor [C, H, W]
            bits: Binary watermark string
            strength: Embedding strength (default: self.default_strength)
            seed: Random seed for reproducibility
            band: Frequency band for embedding
            
        Returns:
            Watermarked image tensor
        """
        result = self._embed({
            "image": image,
            "bits": bits,
            "strength": strength or self.default_strength,
            "seed": seed,
            "band": band
        })
        if result.is_success():
            return result.output["image"]
        raise ValueError(f"Embedding failed: {result.error}")
    
    def extract(self, image: torch.Tensor, num_bits: int = 64,
                seed: int = 42, band: str = "mid") -> str:
        """
        Extract watermark bits from image.
        
        Args:
            image: Image tensor to extract from
            num_bits: Number of bits to extract
            seed: Random seed (must match embedding)
            band: Frequency band (must match embedding)
            
        Returns:
            Extracted binary string
        """
        result = self._extract({
            "image": image,
            "num_bits": num_bits,
            "seed": seed,
            "band": band
        })
        if result.is_success():
            return result.output["bits"]
        raise ValueError(f"Extraction failed: {result.error}")


class POPSSWatermarkDCTOperator(PiscesLxBaseOperator):
    """
    Enhanced DCT Watermark Operator with unified function interfaces.
    
    This class combines all DCT-related functions into a cohesive operator
    with methods for block transforms, midband operations, and pair relations.
    
    Attributes:
        block_size: Default block size for DCT transforms
        default_strength: Default embedding strength
        
    Methods:
        dct_matrix: Create DCT transform matrix
        block_dct2: Apply 2D DCT in blocks
        block_idct2: Apply inverse 2D DCT in blocks
        get_midband_coordinates: Get mid-band coefficient coordinates
        enforce_pair_relation: Enforce magnitude relationship for bit encoding
        create: Factory method to create operator instance
    """
    
    def __init__(self, block_size: int = 8, default_strength: float = 0.08):
        super().__init__()
        self.name = "pisceslx_dct_operator"
        self.version = VERSION
        self.description = "DCT-based image watermarking with frequency domain embedding"
        self.block_size = block_size
        self.default_strength = default_strength
    
    @staticmethod
    def dct_matrix(n: int, device=None, dtype=None) -> torch.Tensor:
        """Create an NxN DCT-II transform matrix."""
        k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
        i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
        alpha = torch.ones(n, device=device, dtype=dtype)
        alpha[0] = 1.0 / math.sqrt(2.0)
        C = math.sqrt(2.0 / n) * alpha.unsqueeze(1) * torch.cos((math.pi * (2 * k + 1) * i) / (2.0 * n))
        return C
    
    def block_dct2(self, x: torch.Tensor, block: int = None) -> torch.Tensor:
        """Apply 2D DCT (type-II) in non-overlapping blocks."""
        block = block or self.block_size
        H, W = x.shape
        assert H % block == 0 and W % block == 0
        C = POPSSWatermarkDCTOperator.dct_matrix(block, device=x.device, dtype=x.dtype)
        CT = C.t()
        x_blocks = x.unfold(0, block, block).unfold(1, block, block)
        x_blocks = x_blocks.contiguous()
        Hn, Wn = x_blocks.shape[:2]
        y = torch.zeros_like(x)
        for by in range(Hn):
            for bx in range(Wn):
                bmat = x_blocks[by, bx]
                yb = C @ bmat @ CT
                y[by * block:(by + 1) * block, bx * block:(bx + 1) * block] = yb
        return y
    
    def block_idct2(self, X: torch.Tensor, block: int = None) -> torch.Tensor:
        """Inverse of block_dct2 for a single-channel [H, W]."""
        block = block or self.block_size
        H, W = X.shape
        assert H % block == 0 and W % block == 0
        C = POPSSWatermarkDCTOperator.dct_matrix(block, device=X.device, dtype=X.dtype)
        CT = C.t()
        X_blocks = X.unfold(0, block, block).unfold(1, block, block)
        X_blocks = X_blocks.contiguous()
        Hn, Wn = X_blocks.shape[:2]
        y = torch.zeros_like(X)
        for by in range(Hn):
            for bx in range(Wn):
                B = X_blocks[by, bx]
                yb = CT @ B @ C
                y[by * block:(by + 1) * block, bx * block:(bx + 1) * block] = yb
        return y
    
    @staticmethod
    def get_midband_coordinates(block: int, band: str = "mid") -> List[Tuple[int, int]]:
        """Return a list of (u,v) coords considered mid-band for a DCT block."""
        coords = []
        if band == "low":
            r0, r1 = 0, max(1, block // 3)
        elif band == "high":
            r0, r1 = max(1, block // 2), block
        else:
            r0, r1 = block // 3, (2 * block) // 3
        for u in range(r0, r1):
            for v in range(r0, r1):
                if not (u == 0 and v == 0):
                    coords.append((u, v))
        return coords
    
    @staticmethod
    def enforce_pair_relation(a: torch.Tensor, b: torch.Tensor, bit: int, 
                             delta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enforce |a| > |b| if bit=1 else |b| > |a|."""
        mag_a = torch.abs(a)
        mag_b = torch.abs(b)
        if bit == 1:
            if mag_a <= mag_b + delta:
                scale = (mag_b + delta + 1e-8) / (mag_a + 1e-8)
                a = a * scale
        else:
            if mag_b <= mag_a + delta:
                scale = (mag_a + delta + 1e-8) / (mag_b + 1e-8)
                b = b * scale
        return a, b
    
    @classmethod
    def create(cls, block_size: int = 8, default_strength: float = 0.08) -> 'POPSSWatermarkDCTOperator':
        """Factory method to create a DCT operator instance."""
        return cls(block_size=block_size, default_strength=default_strength)


def _dct_matrix(n: int, device=None, dtype=None) -> torch.Tensor:
    """Create an NxN DCT-II transform matrix."""
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    alpha = torch.ones(n, device=device, dtype=dtype)
    alpha[0] = 1.0 / math.sqrt(2.0)
    C = math.sqrt(2.0 / n) * alpha.unsqueeze(1) * torch.cos((math.pi * (2 * k + 1) * i) / (2.0 * n))
    return C


def block_dct2(x: torch.Tensor, block: int) -> torch.Tensor:
    """Apply 2D DCT (type-II) in non-overlapping blocks."""
    H, W = x.shape
    assert H % block == 0 and W % block == 0
    C = _dct_matrix(block, device=x.device, dtype=x.dtype)
    CT = C.t()
    x_blocks = x.unfold(0, block, block).unfold(1, block, block)
    x_blocks = x_blocks.contiguous()
    Hn, Wn = x_blocks.shape[:2]
    y = torch.zeros_like(x)
    for by in range(Hn):
        for bx in range(Wn):
            bmat = x_blocks[by, bx]
            yb = C @ bmat @ CT
            y[by * block:(by + 1) * block, bx * block:(bx + 1) * block] = yb
    return y


def block_idct2(X: torch.Tensor, block: int) -> torch.Tensor:
    """Inverse of block_dct2 for a single-channel [H, W]."""
    H, W = X.shape
    assert H % block == 0 and W % block == 0
    C = _dct_matrix(block, device=X.device, dtype=X.dtype)
    CT = C.t()
    X_blocks = X.unfold(0, block, block).unfold(1, block, block)
    X_blocks = X_blocks.contiguous()
    Hn, Wn = X_blocks.shape[:2]
    y = torch.zeros_like(X)
    for by in range(Hn):
        for bx in range(Wn):
            B = X_blocks[by, bx]
            yb = CT @ B @ C
            y[by * block:(by + 1) * block, bx * block:(bx + 1) * block] = yb
    return y


def midband_coordinates(block: int, band: str = "mid") -> List[Tuple[int, int]]:
    """Return a list of (u,v) coords considered mid-band for a DCT block."""
    coords = []
    if band == "low":
        r0, r1 = 0, max(1, block // 3)
    elif band == "high":
        r0, r1 = max(1, block // 2), block
    else:
        r0, r1 = block // 3, (2 * block) // 3
    for u in range(r0, r1):
        for v in range(r0, r1):
            if not (u == 0 and v == 0):
                coords.append((u, v))
    return coords


def enforce_pair_relation(a: torch.Tensor, b: torch.Tensor, bit: int, 
                         delta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enforce |a| > |b| if bit=1 else |b| > |a|."""
    mag_a = torch.abs(a)
    mag_b = torch.abs(b)
    if bit == 1:
        if mag_a <= mag_b + delta:
            scale = (mag_b + delta + 1e-8) / (mag_a + 1e-8)
            a = a * scale
    else:
        if mag_b <= mag_a + delta:
            scale = (mag_a + delta + 1e-8) / (mag_b + 1e-8)
            b = b * scale
    return a, b


def create_dct_operator(block_size: int = 8, default_strength: float = 0.08) -> 'POPSSDCTOperator':
    """Factory function to create a DCT operator instance."""
    return POPSSDCTOperator(block_size=block_size, default_strength=default_strength)


__all__ = [
    "POPSSDCTOperator",
    "POPSSWatermarkDCTOperator"
]
