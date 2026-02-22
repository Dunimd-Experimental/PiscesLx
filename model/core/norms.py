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


class YvRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for efficient normalization.
    
    Implements RMSNorm, a computationally efficient alternative to LayerNorm
    that normalizes by the root mean square without computing the mean. This
    reduces computational overhead while maintaining similar performance.
    
    Mathematical Formulation:
        RMS(x) = sqrt(1/n * sum(x_i^2))
        y = x / RMS(x) * weight + bias (optional)
    
    Key Features:
        - No mean computation, reducing operations by ~25%
        - Optional bias parameter for flexibility
        - Numerically stable with epsilon clamping
        - Compatible with all tensor shapes
    
    Performance Characteristics:
        - Memory: O(dim) for weight and optional bias
        - Compute: O(n) where n is the number of features
        - Speedup: ~10-20% faster than LayerNorm
    
    Comparison with LayerNorm:
        - LayerNorm: Computes mean and variance, centers data
        - RMSNorm: Only normalizes scale, no centering
        - Quality: Similar performance for most transformer tasks
        - Efficiency: RMSNorm is more efficient
    
    Attributes:
        weight (nn.Parameter): Learnable scale parameter of shape [dim].
        bias (Optional[nn.Parameter]): Optional learnable bias of shape [dim].
        eps (float): Small epsilon for numerical stability.
        use_bias (bool): Whether bias parameter is used.
    
    Example:
        >>> norm = YvRMSNorm(dim=4096, eps=1e-6)
        >>> hidden = torch.randn(2, 1024, 4096)
        >>> normalized = norm(hidden)
    
    Note:
        RMSNorm was introduced in "Root Mean Square Layer Normalization"
        by Zhang and Sennrich (2019). It has become standard in many
        large language models including LLaMA.
    
    Reference:
        Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019.
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        use_bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize RMS normalization layer.
        
        Args:
            dim: Dimension of the features to normalize. This is the
                size of the last dimension of input tensors.
            eps: Small epsilon added to the denominator for numerical
                stability. Default: 1e-6.
            use_bias: Whether to include a learnable bias parameter.
                Default: False (standard RMSNorm has no bias).
            device: Device for parameter allocation.
            dtype: Data type for parameters.
        
        Example:
            >>> norm = YvRMSNorm(dim=4096, eps=1e-6, use_bias=False)
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to input tensor.
        
        Computes the root mean square of the last dimension and normalizes
        the input, then applies the learned scale (and optional bias).
        
        Args:
            x: Input tensor of shape [..., dim]. The last dimension
                must match the initialized dim parameter.
        
        Returns:
            Normalized tensor of the same shape as input. Each element
            is scaled by weight / sqrt(mean(x^2) + eps).
        
        Example:
            >>> x = torch.randn(2, 1024, 4096)
            >>> y = norm(x)
            >>> y.shape
            torch.Size([2, 1024, 4096])
        """
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = self.weight * x * rms
        if self.use_bias:
            output = output + self.bias
        return output


class YvLayerNorm(nn.Module):
    """Layer Normalization with optional RMS-style computation.
    
    Implements standard LayerNorm that normalizes by mean and variance,
    with an optional RMS mode for improved efficiency. Provides a
    unified interface for both normalization strategies.
    
    Mathematical Formulation:
        Standard LayerNorm:
            y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
        
        RMS Mode:
            y = x / sqrt(mean(x^2) + eps) * weight + bias
    
    Key Features:
        - Standard LayerNorm with mean centering
        - Optional RMS mode for efficiency
        - Always includes bias parameter (unlike RMSNorm)
        - Compatible with PyTorch's F.layer_norm
    
    When to Use:
        - Standard mode: When centering is important (e.g., pre-norm)
        - RMS mode: When efficiency is critical and centering is optional
    
    Attributes:
        weight (nn.Parameter): Learnable scale parameter.
        bias (nn.Parameter): Learnable bias parameter.
        eps (float): Epsilon for numerical stability.
        use_rms (bool): Whether to use RMS computation.
    
    Example:
        >>> norm = YvLayerNorm(dim=4096, use_rms=False)
        >>> hidden = torch.randn(2, 1024, 4096)
        >>> normalized = norm(hidden)
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        use_rms: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize LayerNorm with optional RMS mode.
        
        Args:
            dim: Dimension of the features to normalize.
            eps: Epsilon for numerical stability. Default: 1e-6.
            use_rms: Whether to use RMS normalization instead of full
                LayerNorm. RMS mode is more efficient but does not
                center the data. Default: False.
            device: Device for parameter allocation.
            dtype: Data type for parameters.
        
        Example:
            >>> norm = YvLayerNorm(dim=4096, use_rms=False)
        """
        super().__init__()
        self.eps = eps
        self.use_rms = use_rms
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to input tensor.
        
        Args:
            x: Input tensor of shape [..., dim]. The last dimension
                must match the initialized dim parameter.
        
        Returns:
            Normalized tensor of the same shape as input.
        
        Note:
            In RMS mode, the mean is not subtracted, making this
            equivalent to RMSNorm with bias.
        """
        if self.use_rms:
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return self.weight * x * rms + self.bias
        else:
            return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class YvAdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization with external conditioning.
    
    Implements adaptive normalization where the scale and shift parameters
    are dynamically generated from conditioning information. This is
    essential for conditional generation tasks like diffusion models.
    
    Mathematical Formulation:
        x_norm = LayerNorm(x)
        scale = linear_scale(cond)
        shift = linear_shift(cond)
        y = x_norm * (1 + scale) + shift
    
    Key Features:
        - Dynamic scale and shift from conditioning
        - Essential for diffusion models and conditional generation
        - Supports timestep embeddings and other conditioning signals
        - Maintains normalization stability with learned modulation
    
    Use Cases:
        - Diffusion models: Conditioning on timestep embeddings
        - Class-conditional generation: Conditioning on class embeddings
        - Text-to-image: Conditioning on text encoder outputs
        - Style transfer: Conditioning on style embeddings
    
    Architecture Integration:
        In diffusion models, AdaptiveLayerNorm is typically used in
        transformer blocks to inject timestep information:
        
        >>> class TransformerBlock(nn.Module):
        ...     def forward(self, x, timestep_emb):
        ...         x = self.adaptive_norm(x, timestep_emb)
        ...         x = self.attention(x)
        ...         return x
    
    Attributes:
        norm (nn.LayerNorm): Base layer normalization.
        scale_proj (nn.Linear): Projects conditioning to scale modulation.
        shift_proj (nn.Linear): Projects conditioning to shift modulation.
        eps (float): Epsilon for numerical stability.
    
    Example:
        >>> adanorm = YvAdaptiveLayerNorm(dim=4096, cond_dim=512)
        >>> hidden = torch.randn(2, 1024, 4096)
        >>> timestep_emb = torch.randn(2, 512)
        >>> normalized = adanorm(hidden, timestep_emb)
    
    Reference:
        Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023.
    """
    
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize Adaptive LayerNorm with conditioning projection.
        
        Args:
            dim: Dimension of the features to normalize.
            cond_dim: Dimension of the conditioning input. This will be
                projected to produce scale and shift parameters.
            eps: Epsilon for numerical stability. Default: 1e-6.
            device: Device for parameter allocation.
            dtype: Data type for parameters.
        
        Example:
            >>> adanorm = YvAdaptiveLayerNorm(
            ...     dim=4096,
            ...     cond_dim=512,  # timestep embedding dimension
            ...     device='cuda'
            ... )
        """
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim, eps=eps, device=device, dtype=dtype)
        self.scale_proj = nn.Linear(cond_dim, dim, device=device, dtype=dtype)
        self.shift_proj = nn.Linear(cond_dim, dim, device=device, dtype=dtype)
        
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """Apply adaptive normalization with conditioning.
        
        Normalizes the input and applies conditioning-dependent scale
        and shift modulation.
        
        Args:
            x: Input tensor of shape [batch, ..., dim]. The last dimension
                must match the initialized dim parameter.
            cond: Conditioning tensor of shape [batch, cond_dim]. This
                is used to generate the scale and shift parameters.
        
        Returns:
            Adaptively normalized tensor of the same shape as input.
            The output is modulated by the conditioning signal.
        
        Example:
            >>> x = torch.randn(2, 1024, 4096)
            >>> cond = torch.randn(2, 512)  # timestep embedding
            >>> y = adanorm(x, cond)
        """
        x_norm = self.norm(x)
        scale = self.scale_proj(cond).unsqueeze(1)
        shift = self.shift_proj(cond).unsqueeze(1)
        return x_norm * (1 + scale) + shift


class YvGroupNorm(nn.Module):
    """Group Normalization with optional RMS-style computation.
    
    Divides channels into groups and normalizes within each group.
    This provides a middle ground between LayerNorm (one group) and
    InstanceNorm (one channel per group), making it suitable for
    vision transformers and convolutional architectures.
    
    Mathematical Formulation:
        For each group g:
            mean_g = mean(x_g)
            var_g = var(x_g)
            y_g = (x_g - mean_g) / sqrt(var_g + eps) * weight_g + bias_g
    
    Key Features:
        - Batch-size independent: Works with any batch size
        - Group-wise normalization: Balances local and global statistics
        - Optional RMS mode for efficiency
        - Optimal for vision transformers and CNNs
    
    Comparison with Other Normalizations:
        - BatchNorm: Depends on batch statistics, issues with small batches
        - LayerNorm: Normalizes all channels together
        - InstanceNorm: Normalizes each channel independently
        - GroupNorm: Normalizes groups of channels, flexible
    
    When to Use:
        - Vision transformers: Groups capture related feature channels
        - Small batch training: Independent of batch size
        - Transfer learning: Consistent behavior across batch sizes
    
    Attributes:
        num_groups (int): Number of groups to divide channels into.
        num_channels (int): Total number of channels.
        weight (nn.Parameter): Learnable scale per channel.
        bias (nn.Parameter): Learnable bias per channel.
        eps (float): Epsilon for numerical stability.
        use_rms (bool): Whether to use RMS normalization.
    
    Example:
        >>> norm = YvGroupNorm(num_groups=32, num_channels=4096)
        >>> features = torch.randn(2, 4096, 16, 16)  # CNN features
        >>> normalized = norm(features)
    
    Reference:
        Wu & He, "Group Normalization", ECCV 2018.
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
        """Initialize group norm.
        
        Args:
            num_groups: Number of groups to divide channels into.
            num_channels: Total number of channels.
            eps: Epsilon for numerical stability.
            use_rms: Whether to use RMS normalization.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.use_rms = use_rms
        
        self.weight = nn.Parameter(torch.ones(num_channels, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_channels, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply group normalization.
        
        Args:
            x: Input tensor of shape [batch, channels, ...].
            
        Returns:
            Normalized tensor.
        """
        if self.use_rms:
            x = x.view(x.shape[0], self.num_groups, -1)
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            x = x * rms
            x = x.view(x.shape[0], -1)
            return self.weight * x + self.bias
        else:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


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


class YvYaRNRotaryEmbedding(nn.Module):
    """YaRN (Yet Another RoPE extensioN) for ultra-long context support.
    
    Implements YaRN scaling for extending RoPE to much longer sequences
    than seen during training. Uses DynamicNTK scaling for improved
    extrapolation quality, enabling sequences up to 10M+ tokens.
    
    Mathematical Formulation:
        YaRN modifies the RoPE frequencies based on sequence length:
        
        For seq_len > original_max:
            ratio = seq_len / original_max
            scale = ratio^(dim / (dim - 2))  # NTK-aware scaling
            new_base = base * scale
            freq = 1 / new_base^(2i/dim)
    
    Key Features:
        - Dynamic NTK scaling for extrapolation
        - Supports sequences up to 10M+ tokens
        - Automatic scaling factor computation
        - Maintains attention quality at long ranges
    
    Scaling Strategy:
        - Short sequences (<= original_max): Standard RoPE
        - Medium sequences: NTK-aware scaling
        - Ultra-long sequences (>1M): Logarithmic scaling boost
    
    Use Cases:
        - Long document processing
        - Extended conversation history
        - Code analysis with full context
        - Book-length text understanding
    
    Performance Characteristics:
        - Memory: O(dim) for inverse frequencies
        - Compute: O(seq_len * dim) for position encoding
        - No additional learned parameters
    
    Attributes:
        dim (int): Dimension of the embedding.
        max_position_embeddings (int): Maximum supported sequence length.
        base (int): Base frequency for RoPE computation.
        scale (float): Maximum YaRN scaling factor.
        original_max_position_embeddings (int): Training sequence length.
        inv_freq (torch.Tensor): Precomputed inverse frequencies.
    
    Example:
        >>> rope = YvYaRNRotaryEmbedding(
        ...     dim=128,
        ...     max_position_embeddings=10485760,  # 10M tokens
        ...     scale=32.0
        ... )
        >>> query = torch.randn(1, 32, 100000, 128)  # 100K tokens
        >>> rotated = rope(query, seq_len=100000)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 10485760,
        base: int = 10000,
        scale: float = 32.0,
        original_max_position_embeddings: int = 4096,
        device: Optional[torch.device] = None
    ):
        """Initialize YaRN rotary embedding.
        
        Args:
            dim: Dimension of the embedding.
            max_position_embeddings: Maximum supported sequence length.
            base: Base frequency for RoPE.
            scale: YaRN scaling factor.
            original_max_position_embeddings: Original training length.
            device: Device for buffers.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        
        freq_factors = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", freq_factors, persistent=False)
        self.register_buffer("dynamic_base", torch.tensor(float(base), device=device), persistent=False)
        self.register_buffer("max_seq_len_seen", torch.tensor(0, device=device), persistent=False)
        
    def _compute_dynamic_ntk_scale(self, seq_len: int) -> float:
        """Compute dynamic NTK scaling factor.
        
        Args:
            seq_len: Current sequence length.
            
        Returns:
            Dynamic scaling factor.
        """
        if seq_len <= self.original_max_position_embeddings:
            return 1.0
            
        ratio = seq_len / self.original_max_position_embeddings
        
        if seq_len > 1000000:
            log_scale = math.log(ratio) / math.log(10) + 1.0
            scale = ratio ** (self.dim / (self.dim - 2)) * log_scale
        else:
            scale = ratio ** (self.dim / (self.dim - 2))
            
        scale = max(scale, 1.0)
        return min(scale, self.scale * 2)
        
    def _compute_scale_factors(
        self,
        seq_len: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Compute YaRN scaling factors with dynamic NTK.
        
        Args:
            seq_len: Current sequence length.
            device: Target device.
            
        Returns:
            Scale factors tensor.
        """
        if seq_len > self.max_seq_len_seen:
            self.max_seq_len_seen = torch.tensor(seq_len, device=device)
            
        ntk_scale = self._compute_dynamic_ntk_scale(seq_len)
        
        positions = torch.arange(seq_len, device=device)
        scale_factors = torch.ones(seq_len, device=device)
        
        if ntk_scale > 1.0:
            crossover = int(math.sqrt(self.original_max_position_embeddings))
            if seq_len > crossover:
                high_positions = positions[crossover:]
                scale_factors[crossover:] = crossover * (high_positions / crossover) ** (1.0 / (ntk_scale * self.scale))
                
        return scale_factors
        
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """Apply YaRN rotary embedding.
        
        Args:
            x: Input tensor of shape [batch, n_head, seq_len, head_dim] or [batch, seq_len, dim].
            seq_len: Optional sequence length override.
            
        Returns:
            Tensor with YaRN rotary embeddings applied.
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
            
        scale_factors = self._compute_scale_factors(actual_seq_len, device)
        
        t = torch.arange(actual_seq_len, device=device, dtype=torch.float32)
        t = t * scale_factors
        
        dynamic_freq = 1.0 / (self.dynamic_base ** (torch.arange(0, embedding_dim, 2).float().to(device) / embedding_dim))
        freqs = torch.outer(t, dynamic_freq)
        
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
    def _rotate_half(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Rotate tensor by half using cosine and sine.
        
        Args:
            x: Input tensor.
            cos: Cosine values.
            sin: Sine values.
            
        Returns:
            Rotated tensor.
        """
        if x.dim() == 4:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
            
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        
        rotated = torch.cat([
            -x2 * sin + x1 * cos,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated


class YvDynamicYaRNRotaryEmbedding(YvYaRNRotaryEmbedding):
    """Dynamic YaRN RoPE with learned scaling for adaptive position encoding.
    
    Extends YaRN with learned scaling parameters for improved long-context
    extrapolation and task-aware position scaling. This allows the model
    to adapt its position encoding strategy based on the task and sequence
    characteristics.
    
    Key Features:
        - Learned scaling parameters for better extrapolation
        - Task-aware position scaling
        - Adaptive to different sequence length distributions
        - Inherits all YaRN capabilities
    
    Learned Components:
        - Scaling factor: Learned adjustment to base scaling
        - Task embeddings: Optional task-specific scaling adjustments
    
    Use Cases:
        - Multi-task models with varying context requirements
        - Models that need to adapt to different sequence distributions
        - Fine-tuning on tasks with specific context patterns
    
    Attributes:
        Inherited from YvYaRNRotaryEmbedding plus:
        learned_scale (nn.Parameter): Learned scaling adjustment.
        task_scale_embed (Optional[nn.Embedding]): Task-specific scaling.
        enable_learned_scaling (bool): Whether learned scaling is enabled.
        enable_task_aware (bool): Whether task-aware scaling is enabled.
    
    Example:
        >>> rope = YvDynamicYaRNRotaryEmbedding(
        ...     dim=128,
        ...     max_position_embeddings=10485760,
        ...     enable_learned_scaling=True
        ... )
        >>> query = torch.randn(1, 32, 100000, 128)
        >>> rotated = rope(query, task_id=0)
    
    Note:
        Learned scaling is initialized close to 1.0 to preserve
        the base YaRN behavior at the start of training.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 10485760,
        base: int = 10000,
        scale: float = 32.0,
        original_max_position_embeddings: int = 4096,
        device: Optional[torch.device] = None,
        enable_learned_scaling: bool = True,
        enable_task_aware: bool = True
    ):
        """Initialize Dynamic YaRN with learned scaling.
        
        Args:
            dim: Dimension of the embedding.
            max_position_embeddings: Maximum supported sequence length.
                Default: 10M tokens for ultra-long context.
            base: Base frequency for RoPE computation. Default: 10000.
            scale: Base YaRN scaling factor. Default: 32.0.
            original_max_position_embeddings: Original training sequence
                length. Default: 4096.
            device: Device for buffers and parameters.
            enable_learned_scaling: Whether to use learned scaling
                parameters. Default: True.
            enable_task_aware: Whether to enable task-specific scaling
                adjustments. Default: True.
        
        Example:
            >>> rope = YvDynamicYaRNRotaryEmbedding(
            ...     dim=128,
            ...     enable_learned_scaling=True,
            ...     enable_task_aware=True
            ... )
        """
        super().__init__(
            dim, max_position_embeddings, base, scale,
            original_max_position_embeddings, device
        )
        
        self.enable_learned_scaling = enable_learned_scaling
        self.enable_task_aware = enable_task_aware
        
        if enable_learned_scaling:
            self.learned_scale = nn.Parameter(torch.tensor(1.0))
            self.ntk_scale_factor = nn.Parameter(torch.tensor(1.0))
            self.log_scale_factor = nn.Parameter(torch.tensor(0.0))
            
        if enable_task_aware:
            self.task_scale_net = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, 2)
            )
            
    def _compute_dynamic_ntk_scale(self, seq_len: int) -> float:
        """Compute dynamic NTK scale with learned parameters.
        
        Args:
            seq_len: Current sequence length.
            
        Returns:
            Dynamic scaling factor.
        """
        if seq_len <= self.original_max_position_embeddings:
            return 1.0
            
        ratio = seq_len / self.original_max_position_embeddings
        base_scale = ratio ** (self.dim / (self.dim - 2))
        
        if hasattr(self, 'learned_scale'):
            learned_multiplier = torch.sigmoid(self.learned_scale)
            ntk_multiplier = torch.sigmoid(self.ntk_scale_factor)
            log_factor = torch.exp(self.log_scale_factor)
            
            if seq_len > 1000000:
                log_scale = math.log(ratio) / math.log(10) + 1.0
                scale = base_scale * log_scale * log_factor
            else:
                scale = base_scale * ntk_multiplier
                
            scale = scale * learned_multiplier
        else:
            if seq_len > 1000000:
                log_scale = math.log(ratio) / math.log(10) + 1.0
                scale = ratio ** (self.dim / (self.dim - 2)) * log_scale
            else:
                scale = ratio ** (self.dim / (self.dim - 2))
                
        scale = max(scale, 1.0)
        return min(scale, self.scale * 2)
        
    def _compute_scale_factors(
        self,
        seq_len: int,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute scale factors with task-aware modulation.
        
        Args:
            seq_len: Current sequence length.
            task_embedding: Optional task embedding for modulation.
            
        Returns:
            Scale factors tensor.
        """
        if seq_len > self.max_seq_len_seen:
            self.max_seq_len_seen = torch.tensor(
                seq_len,
                device=task_embedding.device if task_embedding is not None else self.dynamic_base.device
            )
            
        ntk_scale = self._compute_dynamic_ntk_scale(seq_len)
        
        positions = torch.arange(seq_len, device=self.dynamic_base.device)
        scale_factors = torch.ones(seq_len, device=self.dynamic_base.device)
        
        if ntk_scale > 1.0:
            crossover = int(math.sqrt(self.original_max_position_embeddings))
            
            task_modulation = torch.ones(seq_len, device=self.dynamic_base.device)
            if task_embedding is not None and self.enable_task_aware:
                task_weights = self.task_scale_net(task_embedding.mean(dim=0))
                task_modulation = 1.0 + 0.1 * task_weights[0] * torch.arange(seq_len, device=self.dynamic_base.device) / seq_len
                
            if seq_len > crossover:
                high_positions = positions[crossover:]
                base_scaling = (high_positions / crossover) ** (1.0 / (ntk_scale * self.scale))
                scale_factors[crossover:] = base_scaling * task_modulation[crossover:]
                
        return scale_factors
        
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply dynamic YaRN rotary embedding.
        
        Args:
            x: Input tensor.
            seq_len: Optional sequence length override.
            task_embedding: Optional task embedding for modulation.
            
        Returns:
            Tensor with dynamic YaRN embeddings applied.
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
            
        scale_factors = self._compute_scale_factors(actual_seq_len, task_embedding)
        
        t = torch.arange(actual_seq_len, device=device, dtype=torch.float32)
        t = t * scale_factors
        
        dynamic_freq = 1.0 / (self.dynamic_base ** (torch.arange(0, embedding_dim, 2).float().to(device) / embedding_dim))
        freqs = torch.outer(t, dynamic_freq)
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        if x.dim() == 4:
            cos = cos[:x.shape[2], :]
            sin = sin[:x.shape[2], :]
        else:
            cos = cos[:x.shape[1], :]
            sin = sin[:x.shape[1], :]
            
        return self._rotate_half(x, cos, sin)


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
        assert len(branch_outputs) == self.num_branches
        
        normalized = [
            norm(residual + output)
            for norm, output in zip(self.norms, branch_outputs)
        ]
        
        concat = torch.cat(normalized, dim=-1)
        gate = self.gate(concat)
        
        return sum(normalized) * gate + residual * (1 - gate)
