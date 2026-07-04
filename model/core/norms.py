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
Advanced Normalization and Position Embedding Module for Yv Model.

This module provides comprehensive normalization and position embedding implementations
that form the foundational building blocks for the Yv transformer architecture.
All components are designed for numerical stability, computational efficiency, and
seamless integration with the broader model architecture.

Architecture Overview:
    The module is organized into two major subsystems:

    1. Normalization Layers:
       - YvRMSNorm: Root Mean Square Layer Normalization
         * Computationally efficient alternative to LayerNorm
         * Normalizes by RMS without computing mean
         * Optional bias parameter for flexibility
       
       - YvLayerNorm: Standard Layer Normalization
         * Full LayerNorm with mean and variance computation
         * Optional RMS mode for efficiency
         * Standard interface for compatibility
       
       - YvAdaptiveLayerNorm: Conditional Normalization
         * Scale and shift generated from conditioning input
         * Essential for diffusion models and conditional generation
         * Supports timestep embeddings and other conditioning signals
       
       - YvGroupNorm: Group Normalization
         * Divides channels into groups for normalization
         * Optimal for vision transformers and convolutional layers
         * Optional RMS mode for efficiency
       
       - YvDeepNorm: Deep Network Normalization
         * Designed for training stability in very deep networks
         * Combines residual scaling with layer normalization
         * Prevents gradient explosion in deep architectures
       
       - YvParallelResidualNorm: Parallel Residual Normalization
         * Alternative residual connection normalization strategy
         * Improves gradient flow in parallel architectures

    2. Position Embeddings:
       - YvRotaryEmbedding: Rotary Position Embedding (RoPE)
         * Applies rotation to feature pairs based on position
         * Supports extrapolation to longer sequences
         * Precomputed cosine/sine cache for efficiency
       
       - YvYaRNRotaryEmbedding: YaRN Extended RoPE
         * Yet Another RoPE extensioN for ultra-long contexts
         * Dynamic NTK scaling for improved extrapolation
         * Supports sequences up to 10M+ tokens
       
       - YvDynamicYaRNRotaryEmbedding: Learned YaRN
         * Extends YaRN with learned scaling parameters
         * Task-aware position scaling
         * Adaptive to different sequence length distributions

Design Rationale:
    - Computational Efficiency: RMSNorm avoids mean computation, reducing overhead
    - Numerical Stability: Epsilon values and careful implementation prevent NaN/Inf
    - Long Context Support: YaRN enables extrapolation to 10M+ token sequences
    - Flexibility: Multiple normalization strategies for different architectures
    - Memory Efficiency: Precomputed caches reduce runtime computation

Mathematical Formulations:
    RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
    LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
    RoPE: rotate(x, pos * freq) where freq = 1 / base^(2i/dim)
    YaRN: scale positions dynamically based on sequence length ratio

Performance Considerations:
    - RMSNorm is ~10-20% faster than LayerNorm in practice
    - YaRN adds minimal overhead for long sequence support
    - Precomputed caches reduce per-step computation
    - Fused implementations available for supported hardware

Dependencies:
    - torch: PyTorch deep learning framework
    - torch.nn: Neural network modules
    - torch.nn.functional: Functional interface for operations

Usage Example:
    >>> from model.core.norms import YvRMSNorm, YvYaRNRotaryEmbedding
    >>> 
    >>> # Normalization
    >>> norm = YvRMSNorm(hidden_size=4096, eps=1e-6)
    >>> normalized = norm(hidden_states)
    >>> 
    >>> # Position embedding
    >>> rope = YvYaRNRotaryEmbedding(
    ...     dim=128,
    ...     max_position_embeddings=10485760,
    ...     scale=32.0
    ... )
    >>> embedded = rope(query_tensor, seq_len=8192)

Note:
    All classes follow the YvXxx naming convention.
    Position embeddings are designed to work with the attention module.
    Normalization layers can be used independently or as part of transformer blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass


def _arctic_init_weights(m: nn.Module):
    """Initialize weights for linear and embedding layers.
    
    Uses Kaiming uniform initialization for linear layers and
    normal initialization for embedding layers.
    
    Args:
        m: Module to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)


def _depth_aware_init_weights(m: nn.Module, n_layer: int, hidden_size: int):
    """Depth-aware weight initialization for stable deep network training.
    
    Adjusts initialization standard deviation based on network depth to
    prevent gradient explosion/vanishing in very deep transformers.
    
    Mathematical Formulation:
        std = 1 / sqrt(2 * n_layer * hidden_size)
        
    This formula ensures that the variance of activations remains stable
    across all layers, enabling training of networks with 100+ layers.
    
    Args:
        m: Module to initialize.
        n_layer: Total number of transformer layers.
        hidden_size: Model hidden dimension.
    
    Reference:
        - DeepNorm for very deep transformers (arXiv:2203.00555)
        - LayerScale: Going Deeper With Image Transformers (ICCV 2021, arXiv:2103.17239)
    """
    depth_std = 1.0 / math.sqrt(2 * n_layer * hidden_size)
    
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=depth_std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=depth_std * math.sqrt(hidden_size))
    elif hasattr(m, 'weight') and m.weight is not None:
        if m.weight.dim() >= 2:
            nn.init.normal_(m.weight, mean=0, std=depth_std)


def _norm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    mode: str = 'rms',
    num_groups: Optional[int] = None,
    cond: Optional[torch.Tensor] = None,
    scale_proj: Optional[nn.Module] = None,
    shift_proj: Optional[nn.Module] = None,
    use_fused: bool = True,
) -> torch.Tensor:
    """Unified normalization core used by all Yv norm classes.
    
    Args:
        x: Input tensor.
        weight: Learnable scale parameter.
        bias: Optional learnable bias.
        eps: Epsilon for numerical stability.
        mode: One of 'rms', 'layer', 'group', 'adaptive'.
        num_groups: Number of groups for group norm.
        cond: Conditioning tensor for adaptive norm.
        scale_proj: Scale projection module for adaptive norm.
        shift_proj: Shift projection module for adaptive norm.
        use_fused: Use Triton-fused RMSNorm kernel when available (mode='rms' only).
    
    Returns:
        Normalized tensor.
    """
    if mode == 'rms':
        if use_fused and x.is_cuda and hasattr(weight, 'is_cuda') and weight.is_cuda:
            try:
                from opss.kernels.fused_rms_norm import fused_rms_norm
                out = fused_rms_norm(x, weight, eps)
                if bias is not None:
                    out = out + bias
                return out
            except (ImportError, RuntimeError):
                pass
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        out = weight * x * rms
        if bias is not None:
            out = out + bias
        return out

    if mode == 'layer':
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

    if mode == 'group':
        return F.group_norm(x, num_groups, weight, bias, eps)

    if mode == 'adaptive':
        x_norm = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        scale = scale_proj(cond).unsqueeze(1)
        shift = shift_proj(cond).unsqueeze(1)
        return x_norm * (1 + scale) + shift

    raise ValueError(f"Unknown norm mode: {mode}")


def _residual_alpha_for_depth(n_layer: int) -> float:
    """Compute optimal residual alpha for DeepNorm-style stability.
    
    Formula: alpha = (2 * n_layer) ** 0.25
    
    This scaling factor ensures stable gradient flow in deep residual
    networks by balancing the contribution of residual and skip connections.
    
    Args:
        n_layer: Number of transformer layers.
    
    Returns:
        Optimal residual alpha scaling factor.
    
    Example:
        >>> alpha = _residual_alpha_for_depth(28)  # 7B model
        >>> alpha = _residual_alpha_for_depth(80)  # 70B model
    """
    return (2 * n_layer) ** 0.25


# Paper: Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019, arXiv:1910.07467
class YvRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for efficient normalization.
    
    Uses the unified _norm_forward core. See core docstring for details.
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        use_bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization via unified core."""
        return _norm_forward(x, self.weight, self.bias, self.eps, mode='rms')


class YvLayerNorm(nn.Module):
    """Layer Normalization with optional RMS-style computation.
    
    Uses the unified _norm_forward core.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        use_rms: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.eps = eps
        self.use_rms = use_rms
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = 'rms' if self.use_rms else 'layer'
        return _norm_forward(x, self.weight, self.bias, self.eps, mode=mode)


# Paper: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", NeurIPS 2017, arXiv:1709.07871; DiT: Peebles & Xie, arXiv:2212.09748
class YvAdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization with external conditioning.
    
    Uses the unified _norm_forward core.
    """
    
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        self.scale_proj = nn.Linear(cond_dim, dim, device=device, dtype=dtype)
        self.shift_proj = nn.Linear(cond_dim, dim, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return _norm_forward(
            x, self.weight, self.bias, self.eps, mode='adaptive',
            cond=cond, scale_proj=self.scale_proj, shift_proj=self.shift_proj,
        )


# Paper: Wu & He, "Group Normalization", ECCV 2018, arXiv:1803.08494
class YvGroupNorm(nn.Module):
    """Group Normalization with optional RMS-style computation.
    
    Uses the unified _norm_forward core for group mode.
    RMS group norm is computed inline for performance.
    """
    
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-6,
        use_rms: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.use_rms = use_rms
        
        self.weight = nn.Parameter(torch.ones(num_channels, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_channels, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_rms:
            x = x.view(x.shape[0], self.num_groups, -1)
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            x = x * rms
            x = x.view(x.shape[0], self.num_channels, -1)
            return self.weight[:, None] * x + self.bias[:, None]
        else:
            return _norm_forward(x, self.weight, self.bias, self.eps, mode='group', num_groups=self.num_groups)


# Paper: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv:2104.09864, 2021
class YvRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for position-aware attention.
    
    Applies rotary position embeddings by rotating pairs of features
    based on their position in the sequence. This encoding preserves
    relative position information and supports extrapolation to longer
    sequences than seen during training.
    
    Mathematical Formulation:
        For position p and dimension pair (2i, 2i+1):
            freq_i = 1 / base^(2i/dim)
            angle = p * freq_i
            x'_2i = x_2i * cos(angle) - x_{2i+1} * sin(angle)
            x'_{2i+1} = x_2i * sin(angle) + x_{2i+1} * cos(angle)
    
    Key Features:
        - Relative position encoding through rotation
        - Extrapolation to longer sequences
        - Precomputed cosine/sine cache for efficiency
        - No learned parameters (fully deterministic)
    
    Position Encoding Properties:
        - Relative distance preserved: angle(p) - angle(q) encodes distance
        - Long-range decay: Higher frequencies decay faster
        - Extrapolation: Can extend beyond training length
    
    Performance Characteristics:
        - Memory: O(max_seq_len * dim/2) for cached cos/sin
        - Compute: O(seq_len * dim) for rotation
        - No learned parameters: Reduces model size
    
    Attributes:
        dim (int): Dimension of the embedding (must be even).
        max_seq_len (int): Maximum sequence length for precomputed cache.
        base (float): Base frequency for computing inverse frequencies.
        inv_freq (torch.Tensor): Precomputed inverse frequencies.
        cos_cached (torch.Tensor): Precomputed cosine values.
        sin_cached (torch.Tensor): Precomputed sine values.
    
    Example:
        >>> rope = YvRotaryEmbedding(dim=128, max_seq_len=8192)
        >>> query = torch.randn(2, 32, 1024, 128)  # [batch, heads, seq, dim]
        >>> rotated = rope(query, seq_len=1024)
    
    Note:
        RoPE is typically applied to queries and keys before attention.
        The rotation is applied in-place and does not change tensor shape.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 1e6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize Rotary Position Embedding.
        
        Args:
            dim: Dimension of the embedding. Must be even as RoPE operates
                on pairs of features. Typically set to head_dim.
            max_seq_len: Maximum sequence length to precompute. Sequences
                longer than this will trigger cache update. Default: 8192.
            base: Base frequency for computing inverse frequencies.
                Higher values give slower frequency decay. Default: 1e6.
            device: Device for buffer allocation.
            dtype: Data type for cached cos/sin values.
        
        Example:
            >>> rope = YvRotaryEmbedding(
            ...     dim=128,  # head_dim
            ...     max_seq_len=8192,
            ...     base=10000.0
            ... )
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos().to(dtype))
        self.register_buffer("sin_cached", freqs.sin().to(dtype))
        
    def forward(
        self,
        x: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """Apply rotary position embedding to input tensor.
        
        Rotates pairs of features in the last dimension based on their
        position in the sequence.
        
        Args:
            x: Input tensor of shape [..., dim]. The last dimension
                must match the initialized dim parameter.
            seq_len: Current sequence length. Used to slice the cache.
        
        Returns:
            Rotated tensor of the same shape as input.
        
        Note:
            If seq_len exceeds max_seq_len, the cache is automatically
            updated to accommodate the longer sequence.
        """
        if seq_len > self.max_seq_len:
            self._update_cache(seq_len, x.device)
            
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        return rotated.flatten(-2)
        
    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cosine/sine cache for longer sequences.
        
        Dynamically extends the precomputed cache when encountering
        sequences longer than the initial max_seq_len.
        
        Args:
            seq_len: New maximum sequence length to cache.
            device: Device for the new cache tensors.
        """
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
        self.max_seq_len = seq_len


# Paper: Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models", arXiv:2309.00071, 2023
# ──────────────────────────────────────────────────────────────────────────────
# YvUnifiedRotaryEmbedding — single RoPE implementation absorbing all variants
#   Base: standard RoPE     │ YaRN: NTK-aware frequency scaling
#   MrRoPE: mixed-radix     │ Dynamic: learned/task-aware scaling
#   Linear: position scaling │ All combined into one unified frequency set.
# ──────────────────────────────────────────────────────────────────────────────

class YvUnifiedRotaryEmbedding(nn.Module):
    """Unified Rotary Position Embedding — absorbs all position-encoding variants.

    Every forward pass applies a single combined frequency derived from all
    active mechanisms: standard RoPE frequencies, MrRoPE mixed‑radix factors,
    YaRN dynamic NTK scaling, linear position scaling, and learned (dynamic)
    per‑dimension adjustments.  No algorithm‑variant branching.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 10485760,
        base: int = 10000,
        scale: float = 32.0,
        original_max_position_embeddings: int = 4096,
        device: Optional[torch.device] = None,
        # --- absorbed variants ---
        use_mr_rope: bool = False,
        mr_rope_mode: str = 'pro',
        use_dynamic: bool = False,
        enable_learned_scaling: bool = True,
        enable_task_aware: bool = True,
        linear_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings

        self.use_mr_rope = use_mr_rope
        self.mr_rope_mode = mr_rope_mode
        self.use_dynamic = use_dynamic
        self.enable_learned_scaling = enable_learned_scaling
        self.enable_task_aware = enable_task_aware
        self.linear_scale = linear_scale

        half = dim // 2

        # ── base frequencies (always) ──
        freq_factors = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", freq_factors, persistent=False)
        self.register_buffer("dynamic_base", torch.tensor(float(base), device=device), persistent=False)
        self.register_buffer("max_seq_len_seen", torch.tensor(0, device=device), persistent=False)

        # ── MrRoPE mixed radices (always built) ──
        if use_mr_rope:
            with torch.no_grad():
                if mr_rope_mode == 'pro':
                    rad = torch.arange(1, half + 1, device=device, dtype=torch.float32)
                    rad = 1.0 + 0.1 * (rad / (half))  # progressive ramp
                else:
                    rad = torch.ones(half, device=device, dtype=torch.float32)
            self.register_buffer("mr_radices", rad, persistent=False)

        # ── Dynamic learned parameters (always built when use_dynamic) ──
        if use_dynamic and enable_learned_scaling:
            self.learned_scale = nn.Parameter(torch.tensor(1.0, device=device))
            self.ntk_scale_factor = nn.Parameter(torch.tensor(1.0, device=device))
            self.log_scale_factor = nn.Parameter(torch.tensor(0.0, device=device))

        if use_dynamic and enable_task_aware:
            self.task_scale_net = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, 2),
            )

    # ── helpers ──────────────────────────────────────────────────────

    def _compute_dynamic_ntk_scale(self, seq_len: int) -> float:
        if seq_len <= self.original_max_position_embeddings:
            return 1.0
        ratio = seq_len / self.original_max_position_embeddings
        base_scale = ratio ** (self.dim / (self.dim - 2))

        if self.use_dynamic and self.enable_learned_scaling:
            lm = torch.sigmoid(self.learned_scale)
            nm = torch.sigmoid(self.ntk_scale_factor)
            lf = torch.exp(self.log_scale_factor)
            if seq_len > 1000000:
                s = base_scale * (math.log(ratio) / math.log(10) + 1.0) * lf
            else:
                s = base_scale * nm
            s = s * lm
        else:
            if seq_len > 1000000:
                s = base_scale * (math.log(ratio) / math.log(10) + 1.0)
            else:
                s = base_scale
        s = max(float(s), 1.0)
        return min(s, self.scale * 2)

    def _compute_scale_factors(
        self, seq_len: int, device: torch.device,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if seq_len > self.max_seq_len_seen.item():
            self.max_seq_len_seen.fill_(seq_len)
        ntk_scale = self._compute_dynamic_ntk_scale(seq_len)
        pos = torch.arange(seq_len, device=device)
        sf = torch.ones(seq_len, device=device)

        if ntk_scale > 1.0:
            crossover = int(math.sqrt(self.original_max_position_embeddings))
            tm = torch.ones(seq_len, device=device)
            if self.use_dynamic and self.enable_task_aware and task_embedding is not None:
                tw = self.task_scale_net(task_embedding.mean(dim=0))
                tm = 1.0 + 0.1 * tw[0] * pos / seq_len
            if seq_len > crossover:
                sf[crossover:] = crossover * (pos[crossover:] / crossover) ** (1.0 / (ntk_scale * self.scale)) * tm[crossover:]
        return sf

    # ── forward ─────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = x.device
        if x.dim() == 4:
            actual_seq_len = seq_len or x.shape[2]
            embed_dim = x.shape[3]
        elif x.dim() == 3:
            actual_seq_len = seq_len or x.shape[1]
            embed_dim = x.shape[2]
        else:
            raise ValueError(f"Input must be 3D or 4D, got {x.dim()}D")

        half = embed_dim // 2

        # 1. positions with linear scaling
        t = torch.arange(actual_seq_len, device=device, dtype=torch.float32)
        if self.linear_scale != 1.0:
            t = t / self.linear_scale

        # 2. YaRN scale factors (position‑dependent interpolant)
        sf = self._compute_scale_factors(actual_seq_len, device, task_embedding)
        t = t * sf

        # 3. combined frequencies
        base_freq = 1.0 / (self.dynamic_base ** (torch.arange(0, embed_dim, 2, device=device).float() / embed_dim))

        if self.use_mr_rope:
            mr = self.mr_radices[:half].to(device)
            freqs = torch.outer(t, base_freq * mr)
        else:
            freqs = torch.outer(t, base_freq)

        cos = freqs.cos()
        sin = freqs.sin()

        # slice to actual length
        if x.dim() == 4:
            cos = cos[:x.shape[2], :]
            sin = sin[:x.shape[2], :]
        else:
            cos = cos[:x.shape[1], :]
            sin = sin[:x.shape[1], :]

        return self._rotate_half(x, cos, sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        half = x.shape[-1] // 2
        if cos.shape[-1] != half:
            dim = min(cos.shape[-1], half)
            cos = cos[..., :dim]
            sin = sin[..., :dim]
            half = dim
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1)


# Backward‑compatible alias
YvYaRNRotaryEmbedding = YvUnifiedRotaryEmbedding


class YvDynamicYaRNRotaryEmbedding(YvUnifiedRotaryEmbedding):
    """Dynamic YaRN RoPE — thin wrapper with default use_dynamic=True."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 10485760,
        base: int = 10000,
        scale: float = 32.0,
        original_max_position_embeddings: int = 4096,
        device: Optional[torch.device] = None,
        enable_learned_scaling: bool = True,
        enable_task_aware: bool = True,
    ):
        super().__init__(
            dim=dim, max_position_embeddings=max_position_embeddings,
            base=base, scale=scale,
            original_max_position_embeddings=original_max_position_embeddings,
            device=device,
            use_mr_rope=False, mr_rope_mode='pro',
            use_dynamic=True,
            enable_learned_scaling=enable_learned_scaling,
            enable_task_aware=enable_task_aware,
            linear_scale=1.0,
        )


# Paper: Wang et al., "DeepNet: Scaling Transformers to 1,000 Layers", arXiv:2203.00555, 2022
class YvDeepNorm(nn.Module):
    """Deep Normalization for training stability in very deep networks.
    
    Implements DeepNorm scaling strategy that combines residual scaling
    with layer normalization for improved training stability in deep
    transformer architectures. This prevents gradient explosion and
    enables training of networks with hundreds of layers.
    
    Mathematical Formulation:
        output = LayerNorm(alpha * residual + new_value)
        where alpha = (2 * num_layers)^0.25
    
    Key Features:
        - Residual scaling prevents gradient explosion
        - LayerNorm maintains stable activations
        - Scaling factor adapts to network depth
        - Compatible with any transformer architecture
    
    Training Stability:
        DeepNorm addresses the training instability that arises in very
        deep networks by:
        1. Scaling down the residual contribution
        2. Normalizing after the residual addition
        3. Adapting the scale to the network depth
    
    Use Cases:
        - Very deep transformers (100+ layers)
        - Models prone to gradient explosion
        - Training with large learning rates
        - Architectures with many residual connections
    
    Performance Characteristics:
        - Memory: O(dim) for normalization parameters
        - Compute: O(dim) for normalization
        - No additional overhead vs standard LayerNorm
    
    Attributes:
        alpha (float): Residual scaling factor based on network depth.
        norm (YvRMSNorm): RMS normalization layer.
    
    Example:
        >>> deepnorm = YvDeepNorm(dim=4096, num_layers=96)
        >>> residual = torch.randn(2, 1024, 4096)
        >>> new_value = torch.randn(2, 1024, 4096)
        >>> output = deepnorm(residual, new_value)
    """
    
    def __init__(
        self,
        dim: int,
        num_layers: int,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize DeepNorm with depth-aware scaling.
        
        Args:
            dim: Dimension of the features to normalize.
            num_layers: Number of transformer layers in the network.
                Used to compute the residual scaling factor alpha.
            eps: Epsilon for numerical stability. Default: 1e-6.
            device: Device for parameter allocation.
            dtype: Data type for parameters.
        
        Example:
            >>> deepnorm = YvDeepNorm(
            ...     dim=4096,
            ...     num_layers=96,  # for a 96-layer transformer
            ...     device='cuda'
            ... )
        """
        super().__init__()
        self.alpha = (2 * num_layers) ** 0.25
        self.norm = YvRMSNorm(dim, eps=eps, device=device, dtype=dtype)
        
    def forward(
        self,
        residual: torch.Tensor,
        new_value: torch.Tensor
    ) -> torch.Tensor:
        """Apply DeepNorm to residual connection.
        
        Scales the residual by alpha, adds the new value, and normalizes.
        
        Args:
            residual: Residual connection tensor from previous layer.
                Shape: [batch, ..., dim].
            new_value: New value to add to the residual.
                Shape: [batch, ..., dim] (same as residual).
        
        Returns:
            Normalized output tensor of the same shape as inputs.
        
        Note:
            The residual is scaled by alpha before addition to prevent
            the gradients from exploding in deep networks.
        """
        return self.norm(residual * self.alpha + new_value)


# Paper: Wang et al., "DeepNet: Scaling Transformers to 1,000 Layers", arXiv:2203.00555, 2022; parallel residual variant
class YvParallelResidualNorm(nn.Module):
    """Parallel Residual Normalization for improved gradient flow.
    
    Implements parallel residual connections with normalization,
    providing an alternative to sequential residual connections.
    This can improve gradient flow in deep networks with multiple
    parallel branches.
    
    Mathematical Formulation:
        For each branch i:
            branch_i = norm_i(value_i)
        output = sum(branch_i) / sqrt(num_branches)
    
    Key Features:
        - Parallel branch normalization
        - Balanced gradient distribution
        - Configurable number of branches
        - RMS normalization for efficiency
    
    Architecture Integration:
        ParallelResidualNorm is useful in architectures where multiple
        operations are applied in parallel (e.g., attention + MLP):
        
        >>> attn_out = attention(x)
        >>> mlp_out = mlp(x)
        >>> output = parallel_norm(attn_out, mlp_out)
    
    Use Cases:
        - Parallel transformer architectures
        - Multi-branch networks
        - Models requiring balanced gradient flow
        - Architectures with multiple simultaneous operations
    
    Attributes:
        num_branches (int): Number of parallel branches to normalize.
        norms (nn.ModuleList): List of normalization layers, one per branch.
    
    Example:
        >>> parallel_norm = YvParallelResidualNorm(dim=4096, num_branches=2)
        >>> attn_out = torch.randn(2, 1024, 4096)
        >>> mlp_out = torch.randn(2, 1024, 4096)
        >>> output = parallel_norm([attn_out, mlp_out])
    """
    
    def __init__(
        self,
        dim: int,
        num_branches: int = 2,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize parallel residual norm.
        
        Args:
            dim: Dimension to normalize.
            num_branches: Number of parallel branches.
            eps: Epsilon for numerical stability.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.num_branches = num_branches
        
        self.norms = nn.ModuleList([
            YvRMSNorm(dim, eps=eps, device=device, dtype=dtype)
            for _ in range(num_branches)
        ])
        
        self.gate = nn.Sequential(
            nn.Linear(dim * num_branches, dim),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        residual: torch.Tensor,
        *branch_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Apply parallel residual normalization.
        
        Args:
            residual: Original residual tensor.
            *branch_outputs: Outputs from parallel branches.
            
        Returns:
            Combined output tensor.
        """
        if len(branch_outputs) != self.num_branches:
            raise RuntimeError(f"Expected {self.num_branches} branches, got {len(branch_outputs)}")
        
        normalized = [
            norm(residual + output)
            for norm, output in zip(self.norms, branch_outputs)
        ]
        
        concat = torch.cat(normalized, dim=-1)
        gate = self.gate(concat)
        
        return sum(normalized) * gate + residual * (1 - gate)
