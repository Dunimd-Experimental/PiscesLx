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

"""Rotary Position Embedding (RoPE) utilities for position-aware attention.

This module provides implementations of Rotary Position Embeddings and their
extensions for handling long-context scenarios in transformer models.

Architecture Overview:
    Rotary Position Embeddings encode position information by rotating the
    query and key vectors in the attention mechanism. This approach has
    several advantages over absolute or relative position embeddings:
    
    - **Position Invariance**: Attention scores depend only on relative positions
    - **Extrapolation**: Can handle sequences longer than training length
    - **Efficiency**: No additional parameters, simple computation
    
    The module provides three scaling strategies for extending context length:
    
    1. **Linear Scaling**: Simple division of position indices
    2. **YaRN (Yet another RoPE extensioN)**: NTK-aware scaling + temperature
    3. **Dynamic YaRN**: Automatic scaling based on sequence length

Mathematical Foundation:
    RoPE applies rotation to pairs of dimensions:
    
        rotate(x, θ) = [x₁cos(θ) - x₂sin(θ), x₁sin(θ) + x₂cos(θ)]
    
    Where θ is determined by position and dimension:
    
        θ(p, d) = p × base^(-2d/D)
    
    For long contexts, the base frequency is adjusted:
    
    - **Linear**: θ' = θ / scale
    - **YaRN**: base' = base × α^(D/(D-2))
    - **Dynamic YaRN**: α adapts to sequence length

Components:
    - YvLinearScalingRoPE: Simple linear position scaling
    - YvYaRNRotaryEmbedding: YaRN-scaled RoPE (imported from core.norms)
    - YvDynamicYaRNRotaryEmbedding: Dynamic NTK scaling (imported from core.norms)
    - precompute_freqs_cis: Efficient frequency precomputation
    - apply_rotary_emb: Apply rotation to tensors

Scaling Strategies Comparison:
    +------------------+-------------------+-------------------+
    | Strategy         | Accuracy          | Max Extension     |
    +------------------+-------------------+-------------------+
    | Linear Scaling   | Good for 2-4x     | ~4x training len  |
    | YaRN             | Excellent 8-16x   | ~16x training len |
    | Dynamic YaRN     | Excellent 16-32x  | ~32x training len |
    +------------------+-------------------+-------------------+

Example:
    >>> from model.utils import YvYaRNRotaryEmbedding, apply_rotary_emb
    >>> 
    >>> # Create YaRN embedding for 32K context
    >>> rope = YvYaRNRotaryEmbedding(
    ...     dim=128,
    ...     max_position_embeddings=32768,
    ...     scale=32
    ... )
    >>> 
    >>> # Precompute frequencies
    >>> cos, sin = precompute_freqs_cis(128, 4096)
    >>> 
    >>> # Apply to queries and keys
    >>> q_rotated = apply_rotary_emb(queries, cos, sin)
    >>> k_rotated = apply_rotary_emb(keys, cos, sin)

Dependencies:
    - torch: For tensor operations and neural network modules
    - math: For mathematical constants
    - model.core.norms: For YaRN implementations

Note:
    The YaRN and DynamicYaRN classes are implemented in model.core.norms
    and re-exported here for convenience.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..core.norms import (
    YvYaRNRotaryEmbedding,
    YvDynamicYaRNRotaryEmbedding,
)


class YvLinearScalingRoPE(nn.Module):
    """Linear Scaling Rotary Position Embedding.
    
    Implements the simplest approach to extending RoPE to longer sequences
    by linearly scaling position indices. This method divides position
    indices by a scale factor, effectively compressing positions.
    
    Mathematical Formulation:
        For position p and scale s:
            p_scaled = p / s
            θ_scaled = p_scaled × base^(-2d/D)
    
    When to Use:
        - Extending context by 2-4x training length
        - When simplicity and speed are priorities
        - As a baseline for comparison with advanced methods
    
    Limitations:
        - Accuracy degrades beyond 4x extension
        - May lose fine-grained position information
        - Not optimal for very long contexts (16x+)
    
    Attributes:
        dim (int): Dimension of the embedding (head_dim).
        max_position_embeddings (int): Maximum sequence length during training.
        base (int): Base frequency for RoPE. Default: 10000.
        scale (float): Scaling factor for position indices.
        inv_freq (torch.Tensor): Inverse frequency buffer.
    
    Example:
        >>> rope = YvLinearScalingRoPE(
        ...     dim=128,
        ...     max_position_embeddings=8192,
        ...     scale=4.0  # Extends to 32K
        ... )
        >>> rotated = rope(queries)  # Apply rotation
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: int = 10000,
        scale: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """Initialize Linear Scaling RoPE.
        
        Args:
            dim (int): Dimension of the embedding (typically head_dim).
            max_position_embeddings (int): Maximum sequence length during training.
                Default: 8192.
            base (int): Base frequency for computing rotation angles.
                Default: 10000 (standard RoPE value).
            scale (float): Scaling factor for position indices. Values > 1
                extend context length. Default: 1.0 (no scaling).
            device (Optional[torch.device]): Target device for tensors.
                Default: None (uses default device).
        
        Note:
            For extending 8K context to 32K, use scale=4.0.
            For extending 8K context to 64K, use scale=8.0.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """Apply rotary position embedding to input tensor.
        
        Computes rotation angles based on scaled positions and applies
        rotation to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape:
                - [batch, n_head, seq_len, head_dim] (4D)
                - [batch, seq_len, dim] (3D)
            seq_len (Optional[int]): Sequence length override. If None,
                inferred from input shape. Default: None.
        
        Returns:
            torch.Tensor: Rotated tensor with same shape as input.
        
        Raises:
            ValueError: If input tensor is not 3D or 4D.
        """
        device = x.device
        
        if x.dim() == 4:
            actual_seq_len = seq_len or x.shape[2]
            embedding_dim = x.shape[3]
        elif x.dim() == 3:
            actual_seq_len = seq_len or x.shape[1]
            embedding_dim = x.shape[2]
        else:
            raise ValueError(f"Input tensor must be 3D or 4D, got {x.dim()}D")
        
        t = torch.arange(actual_seq_len, device=device, dtype=torch.float32) / self.scale
        freqs = torch.outer(t, self.inv_freq.to(device))
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        if x.dim() == 4:
            cos = cos[:x.shape[2], :]
            sin = sin[:x.shape[2], :]
        else:
            cos = cos[:x.shape[1], :]
            sin = sin[:x.shape[1], :]
        
        return self._rotate_half(x, cos, sin)
    
    @staticmethod
    def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotation to pairs of dimensions.
        
        Implements the core rotation operation:
            rotate([x₁, x₂], θ) = [x₁cos(θ) - x₂sin(θ), x₁sin(θ) + x₂cos(θ)]
        
        Args:
            x (torch.Tensor): Input tensor to rotate.
            cos (torch.Tensor): Cosine values for rotation.
            sin (torch.Tensor): Sine values for rotation.
        
        Returns:
            torch.Tensor: Rotated tensor.
        """
        if x.dim() == 4:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        
        return torch.cat([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: int = 10000,
    scale: float = 1.0,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine values for rotary position embeddings.
    
    Efficiently precomputes the rotation angles for all positions up to
    max_seq_len. This avoids repeated computation during forward passes
    and improves performance.
    
    Mathematical Formulation:
        For each position p and dimension pair d:
            θ(p, d) = p/scale × base^(-2d/D)
            cos(p, d) = cos(θ(p, d))
            sin(p, d) = sin(θ(p, d))
    
    Args:
        dim (int): Embedding dimension (head_dim). Must be even.
        max_seq_len (int): Maximum sequence length to precompute.
        base (int): Base frequency for RoPE. Default: 10000.
        scale (float): Scaling factor for positions. Default: 1.0.
        device (Optional[torch.device]): Target device. Default: None.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - cos: Cosine values of shape [max_seq_len, dim//2]
            - sin: Sine values of shape [max_seq_len, dim//2]
    
    Example:
        >>> cos, sin = precompute_freqs_cis(128, 4096)
        >>> # Use in attention
        >>> q_rot = apply_rotary_emb(queries, cos, sin)
    
    Note:
        Precomputed tensors can be cached and reused across forward passes
        for the same sequence length, significantly improving efficiency.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32) / scale
    freqs = torch.outer(t, inv_freq)
    
    cos = freqs.cos()
    sin = freqs.sin()
    
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embedding to input tensor.
    
    Applies the rotation operation to the input tensor using precomputed
    cosine and sine values. This is the core operation for RoPE.
    
    Rotation Formula:
        For each pair of dimensions (x₁, x₂) at position p:
            x₁' = x₁ × cos(θ_p) - x₂ × sin(θ_p)
            x₂' = x₁ × sin(θ_p) + x₂ × cos(θ_p)
    
    Args:
        x (torch.Tensor): Input tensor of shape:
            - [batch, n_head, seq_len, head_dim] (4D attention)
            - [batch, seq_len, dim] (3D general)
        cos (torch.Tensor): Cosine values of shape [seq_len, dim//2].
        sin (torch.Tensor): Sine values of shape [seq_len, dim//2].
    
    Returns:
        torch.Tensor: Rotated tensor with same shape as input.
    
    Example:
        >>> # Precompute frequencies
        >>> cos, sin = precompute_freqs_cis(128, 4096)
        >>> 
        >>> # Apply to queries and keys
        >>> q_rotated = apply_rotary_emb(queries, cos, sin)
        >>> k_rotated = apply_rotary_emb(keys, cos, sin)
        >>> 
        >>> # Compute attention with position-aware scores
        >>> attn_scores = torch.matmul(q_rotated, k_rotated.transpose(-2, -1))
    
    Note:
        The input dimension must be even, as rotation operates on pairs
        of dimensions. The cos/sin tensors should have half the dimension.
    """
    if x.dim() == 4:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    
    return torch.cat([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1)


__all__ = [
    "YvYaRNRotaryEmbedding",
    "YvDynamicYaRNRotaryEmbedding",
    "YvLinearScalingRoPE",
    "precompute_freqs_cis",
    "apply_rotary_emb",
]
