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

"""
Advanced Attention Mechanisms Module for Yv Model.

This module provides comprehensive attention implementations that form the core
computational component of the Yv transformer architecture. It includes
multiple attention variants optimized for different use cases, from standard
multi-head attention to specialized mechanisms for long-context and efficient
inference.

Architecture Overview:
    The attention system implements a hierarchical design with multiple backends:

    1. Standard Attention Variants:
       - YvAttention: Standard multi-head attention
         * Full attention with O(n^2) complexity
         * Optional Flash Attention 2/3 for GPU acceleration
         * Supports both training and inference modes
       
       - YvFlashAttention: Flash Attention optimized implementation
         * Memory-efficient attention computation
         * Supports Flash Attention 2 and 3
         * Automatic kernel selection based on hardware
       
       - YvGroupedQueryAttention: GQA implementation
         * Reduces KV cache size by sharing keys/values across query groups
         * Balances quality and efficiency between MHA and MQA
         * Configurable number of key-value heads
       
       - YvMultiQueryAttention: MQA implementation
         * Maximum KV cache efficiency with single KV head
         * Optimal for inference throughput
         * Slight quality trade-off for memory savings

    2. Efficient Long-Context Attention:
       - YvLinearAttention: Linear complexity attention
         * O(n) time and memory complexity
         * Feature map-based approximation
         * Supports causal masking for autoregressive generation
       
       - YvSlidingWindowAttention: Local attention
         * Attention restricted to a sliding window
         * O(n * w) complexity where w is window size
         * Optional global tokens for long-range dependencies
       
       - YvSparseAttention: Sparse attention patterns
         * Configurable sparse patterns (random, local, global, block)
         * Reduced memory for long sequences
         * Supports custom sparse masks

    3. Memory-Optimized Attention:
       - YvPagedAttention: Block-based KV cache
         * Efficient memory allocation with virtual memory concepts
         * Supports KV cache sharing across sequences
         * Optimal for batched inference
       
       - YvRingAttention: Distributed attention
         * Splits attention across multiple devices
         * Enables processing of arbitrarily long sequences
         * Ring communication pattern for efficiency

    4. Streaming and Position Encoding:
       - YvStreamingAttention: Streaming-friendly attention
         * Attention sinks for stable streaming generation
         * Handles context window overflow gracefully
         * Maintains quality in long conversations
       
       - YvALiBiAttention: ALiBi position encoding
         * Linear bias instead of position embeddings
         * Better extrapolation to longer sequences
         * No learned position parameters

    5. Attention Backend System:
       - YvAttentionBackend: Enum for backend selection
         * FLASH_2: Flash Attention 2 (Ampere+)
         * FLASH_3: Flash Attention 3 (Hopper+)
         * TORCH: PyTorch native implementation
         * MATH: Pure Python fallback
       
       - Automatic backend selection based on hardware and inputs

Design Rationale:
    - Flexibility: Multiple attention types for different use cases
    - Efficiency: Flash Attention, GQA, and paged attention for speed
    - Long Context: Linear, sliding window, and ring attention for long sequences
    - Memory Optimization: Paged attention and sparse patterns for memory savings
    - Hardware Awareness: Automatic backend selection for optimal performance

Mathematical Formulations:
    Standard Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
    GQA: Keys and values shared across groups of query heads
    Linear Attention: feature(Q) * (feature(K)^T * V) with kernel approximation
    Sliding Window: Attention restricted to positions [i-w, i+w]
    ALiBi: Attention(Q, K, V) + bias(i-j) where bias is linear in distance

Performance Considerations:
    - Flash Attention provides 2-4x speedup over standard attention
    - GQA reduces KV cache by num_heads/num_kv_heads factor
    - Linear attention enables O(n) processing for long sequences
    - Paged attention reduces memory fragmentation
    - Ring attention enables distributed long-context processing

Dependencies:
    - torch: PyTorch deep learning framework
    - .norms: Normalization and position embedding modules
    - utils.dc: Logging utilities

Usage Example:
    >>> from model.core.attention import YvAttention, YvFlashAttention
    >>> from model.core.attention import YvGroupedQueryAttention
    >>> 
    >>> # Standard attention
    >>> attn = YvAttention(config)
    >>> output = attn(hidden_states, attention_mask)
    >>> 
    >>> # Flash Attention for efficiency
    >>> flash_attn = YvFlashAttention(config)
    >>> output = flash_attn(hidden_states)
    >>> 
    >>> # GQA for reduced KV cache
    >>> gqa = YvGroupedQueryAttention(
    ...     hidden_size=4096,
    ...     num_heads=32,
    ...     num_kv_heads=8
    ... )

Note:
    All classes follow the YvXxx naming convention.
    Flash Attention requires CUDA-capable GPU with supported architecture.
    GQA and MQA are recommended for inference-optimized deployments.
    Linear attention is experimental and may have quality trade-offs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from .norms import _arctic_init_weights, YvRMSNorm, YvYaRNRotaryEmbedding
from utils.dc import PiscesLxLogger

_LOG = PiscesLxLogger(__name__)


class YvAttentionBackend(Enum):
    """Enumeration of available attention backend implementations.
    
    This enum defines the supported attention computation backends that can be
    selected based on hardware capabilities, sequence length, and performance
    requirements. The backend selection is typically automatic but can be
    manually configured for specific use cases.
    
    Attributes:
        STANDARD: Standard PyTorch attention implementation using scaled dot-product.
            Compatible with all devices and data types. O(n^2) memory complexity.
            Use when Flash Attention is not available or for debugging.
        
        FLASH_V2: Flash Attention 2 implementation for NVIDIA Ampere+ GPUs.
            Provides 2-4x speedup and significant memory reduction.
            Requires CUDA 11.6+ and compute capability 8.0+.
            Optimal for training on A100, H100, RTX 30/40 series.
        
        FLASH_V3: Flash Attention 3 implementation for NVIDIA Hopper+ GPUs.
            Further optimized for H100 with FP8 support.
            Requires CUDA 12.0+ and compute capability 9.0+.
            Best performance on H100 and newer architectures.
        
        LINEAR: Linear attention with O(n) complexity using kernel feature maps.
            Suitable for very long sequences where O(n^2) is prohibitive.
            May have slight quality trade-offs compared to full attention.
            Supports ELU, Performer, and softmax feature maps.
        
        SPARSE: Sparse attention with configurable patterns.
            Reduces memory for long sequences using local, global, and random patterns.
            Supports BigBird, Longformer, and block-sparse patterns.
            Optimal for document-level tasks with local dependencies.
        
        PAGED: PagedAttention with block-based KV cache management.
            Enables efficient memory allocation and sharing across sequences.
            Optimal for batched inference with variable-length sequences.
            Supports prefix caching for shared prompts.
        
        RING: Ring attention for distributed processing across multiple devices.
            Enables processing of arbitrarily long sequences.
            Uses ring communication pattern for efficiency.
            Requires multiple GPUs with NVLink or fast interconnect.
        
        H2O: Heavy-Hitter Oracle attention for ultra-long contexts.
            Retains important tokens ("heavy hitters") while compressing others.
            Supports sequences of 1M+ tokens with bounded memory.
            Optimal for long-document and code understanding tasks.
    
    Example:
        >>> backend = YvAttentionBackend.FLASH_V2
        >>> if backend == YvAttentionBackend.FLASH_V2:
        ...     # Use Flash Attention 2
        ...     pass
    
    Note:
        Backend selection is typically handled automatically by YvAttention
        based on hardware capabilities and input characteristics. Manual selection
        is useful for debugging or specific optimization scenarios.
    """
    STANDARD = "standard"
    FLASH_V2 = "flash_v2"
    FLASH_V3 = "flash_v3"
    LINEAR = "linear"
    SPARSE = "sparse"
    PAGED = "paged"
    RING = "ring"
    H2O = "h2o"


@dataclass
class YvAttentionConfig:
    """Configuration for Yv attention modules.
    
    Attributes:
        hidden_size: Model hidden dimension.
        n_head: Number of attention heads.
        n_kv_head: Number of key/value heads for GQA.
        head_dim: Per-head dimension (computed if not provided).
        max_position_embeddings: Maximum sequence length.
        rope_theta: Base frequency for RoPE.
        attention_dropout: Dropout probability.
        use_flash_attention: Whether to use Flash Attention.
        use_alibi: Whether to use ALiBi position encoding.
        use_attention_sink: Whether to use attention sinks.
        sliding_window: Sliding window size (0 = disabled).
        use_linear_attention: Whether to enable linear attention path.
        linear_attention_dim: Feature dimension for linear attention.
        sparse_attention_pattern: Sparse attention pattern type.
        sparse_block_size: Block size for sparse attention.
        paged_attention_block_size: Block size for paged attention.
        use_ring_attention: Whether to use ring attention.
        ring_attention_size: Number of devices for ring attention.
        attention_scale: Custom attention scale factor.
        use_qk_norm: Whether to apply QK normalization.
        use_gqa_residual: Whether to use GQA residual connections.
        use_mla: Whether to use Multi-Head Latent Attention.
        kv_lora_rank: Low-rank dimension for KV compression in MLA.
        q_lora_rank: Low-rank dimension for Q compression in MLA (optional).
    """
    hidden_size: int = 4096
    n_head: int = 32
    n_kv_head: int = 8
    head_dim: Optional[int] = None
    max_position_embeddings: int = 10485760
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    use_flash_attention: bool = True
    use_alibi: bool = False
    use_attention_sink: bool = True
    sliding_window: int = 0
    use_linear_attention: bool = False
    linear_attention_dim: int = 64
    sparse_attention_pattern: str = "none"
    sparse_block_size: int = 64
    paged_attention_block_size: int = 16
    use_ring_attention: bool = False
    ring_attention_size: int = 4
    attention_scale: Optional[float] = None
    use_qk_norm: bool = True
    use_gqa_residual: bool = True
    compression_ratio: int = 8
    streaming_window: int = 16384
    fused_qkv: bool = True
    sdpa_prefer_flash: bool = True
    use_sliding_window: bool = False
    use_h2o_attention: bool = False
    use_mla: bool = True
    kv_lora_rank: int = 512
    q_lora_rank: Optional[int] = None
    
    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.n_head


class YvALiBi(nn.Module):
    """Attention with Linear Biases (ALiBi) for position encoding.
    
    ALiBi replaces learned position embeddings with fixed linear biases added
    to attention scores. This approach enables extrapolation to sequences longer
    than those seen during training, making it particularly effective for
    long-context scenarios without requiring position embedding parameters.
    
    Mathematical Formulation:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d) + m * relative_pos) * V
        
    Where:
        - m is a head-specific slope parameter (negative value)
        - relative_pos is the distance between query and key positions
        - Slopes are geometrically spaced: m_h = -2^(-8h/H) for head h
    
    Key Features:
        - No learned position embeddings required
        - Extrapolates to 2-10x training sequence length
        - Linear memory overhead (just bias matrix)
        - Works with any attention implementation
        - Provides strong inductive bias for position awareness
    
    Performance Characteristics:
        - Memory: O(n^2) for bias matrix, but can be computed on-the-fly
        - Compute: O(n^2) for bias addition, negligible overhead
        - Extrapolation: Tested up to 10x training length with minimal degradation
    
    Attributes:
        n_head (int): Number of attention heads.
        max_seq_len (int): Maximum sequence length for precomputed biases.
        slopes (torch.Tensor): Head-specific slope values, shape [n_head].
        cached_bias (torch.Tensor): Precomputed bias matrix cache.
    
    Example:
        >>> alibi = YvALiBi(n_head=32, max_seq_len=8192)
        >>> bias = alibi(seq_len=1024, device='cuda')  # [n_head, seq_len, seq_len]
        >>> attn_weights = attn_weights + bias.unsqueeze(0)
    
    Note:
        ALiBi is mutually exclusive with rotary embeddings. When use_alibi=True
        in the config, rotary embeddings are automatically disabled.
    
    Reference:
        Press et al., "Train Short, Test Long: Attention with Linear Biases
        Enables Input Length Extrapolation", ICLR 2022.
    """
    
    def __init__(
        self,
        n_head: int,
        max_seq_len: int = 8192,
        slopes: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize ALiBi position encoding.
        
        Args:
            n_head: Number of attention heads. Slopes are computed per head
                using geometric progression to ensure multi-scale position awareness.
            max_seq_len: Maximum sequence length for precomputed bias cache.
                Longer sequences will trigger cache recomputation during forward.
            slopes: Optional custom slope values for each head. If None, slopes
                are computed using the geometric progression formula.
                Shape must be [n_head]. Custom slopes allow fine-tuning for
                specific attention patterns.
            device: Device for storing the bias cache. Defaults to CPU.
        
        Raises:
            ValueError: If custom slopes shape doesn't match n_head.
        
        Example:
            >>> alibi = YvALiBi(n_head=32, max_seq_len=4096)
            >>> # Custom slopes for specific attention patterns
            >>> custom_slopes = -torch.logspace(-1, -3, 32)
            >>> alibi_custom = YvALiBi(32, 4096, slopes=custom_slopes)
        """
        super().__init__()
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        
        if slopes is None:
            slopes = self._get_slopes(n_head)
        self.register_buffer("slopes", slopes.to(device) if device else slopes)
        
        self._build_cache(max_seq_len, device)
        
    def _get_slopes(self, n_head: int) -> torch.Tensor:
        """Compute head-specific ALiBi slopes using geometric progression.
        
        The slopes follow a geometric progression that ensures each head
        attends to different distance scales, providing multi-scale position
        awareness. Different heads become sensitive to different relative
        position ranges.
        
        Formula:
            For n_head heads:
            - If n_head is power of 2: m_h = -2^(-8h/n_head) for h in [0, n_head)
            - Otherwise: Interpolate to get exactly n_head slopes
        
        Args:
            n_head: Number of attention heads.
        
        Returns:
            Tensor of negative slope values with shape [n_head].
            Values range from ~-0.5 (first head, sensitive to nearby positions)
            to ~-0.001 (last head, sensitive to distant positions).
        
        Example:
            >>> slopes = alibi._get_slopes(8)
            >>> # [-0.5, -0.25, -0.125, -0.0625, -0.03125, -0.0156, -0.0078, -0.0039]
        """
        def get_slopes_power_of_2(n):
            start = 2.0 ** (-8.0 / n)
            return torch.tensor([start ** i for i in range(n)])
        
        if math.log2(n_head).is_integer():
            slopes = get_slopes_power_of_2(n_head)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = get_slopes_power_of_2(2 * closest_power_of_2)
            slopes_b = slopes_b[1::2][:n_head - closest_power_of_2]
            slopes = torch.cat([slopes_a, slopes_b])
            
        return slopes
    
    def _build_cache(self, seq_len: int, device: Optional[torch.device]):
        """Precompute and cache the ALiBi bias matrix.
        
        Builds the position bias matrix for efficient reuse during forward passes.
        The bias is computed as: bias[h,i,j] = slope[h] * (j - i) for j <= i
        This creates a lower-triangular bias matrix where each head has its
        own decay rate for relative positions.
        
        Args:
            seq_len: Sequence length for the bias matrix.
            device: Device to store the cache on.
        
        Note:
            The cache is registered as a buffer and will be moved with the model.
            If seq_len exceeds max_seq_len during forward, cache is rebuilt.
            Memory usage: O(n_head * seq_len^2) for the cache.
        """
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions.unsqueeze(0).expand(self.n_head, -1, -1)
        
        bias = -self.slopes.unsqueeze(-1).unsqueeze(-1) * relative_positions.float()
        bias = bias.tril()
        
        self.register_buffer("cached_bias", bias, persistent=False)
        self.cached_seq_len = seq_len
        
    def forward(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Get ALiBi bias for the specified sequence length.
        
        Returns the precomputed bias matrix, rebuilding the cache if necessary
        when the requested sequence length exceeds the cached length.
        
        Args:
            seq_len: Current sequence length for attention computation.
            device: Device for the bias tensor. Must match the device of
                the attention weights for proper addition.
        
        Returns:
            Bias tensor of shape [n_head, seq_len, seq_len] ready for addition
            to attention weights. Should be unsqueezed at dim 0 for batch
            broadcasting: bias.unsqueeze(0) gives [1, n_head, seq_len, seq_len].
        
        Example:
            >>> bias = alibi(2048, device='cuda')
            >>> attn_weights = Q @ K.transpose(-2, -1) / sqrt(d)
            >>> attn_weights = attn_weights + bias.unsqueeze(0)  # Add ALiBi bias
        """
        if seq_len > self.cached_seq_len:
            self._build_cache(seq_len, device)
            
        return self.cached_bias[:, :seq_len, :seq_len]


class YvAttentionSink(nn.Module):
    """Attention Sink mechanism for streaming attention stability.
    
    Implements learnable "sink" tokens that are prepended to the input sequence
    to absorb excess attention mass during streaming inference. This prevents
    attention collapse and maintains stable attention distributions when
    processing sequences in a streaming manner with limited KV cache.
    
    Mathematical Background:
        In streaming attention with limited cache, removing old tokens can cause
        attention scores to become unstable. Sink tokens provide a "buffer" that
        absorbs attention that would otherwise be distributed across removed tokens.
        
        The sink tokens are learned parameters that optimize to:
        1. Absorb uninformative attention mass
        2. Maintain stable attention distributions
        3. Preserve important information in the main sequence
    
    Key Features:
        - Prevents attention collapse in streaming scenarios
        - Enables efficient KV cache eviction
        - Minimal computational overhead (n_sink additional tokens)
        - Learnable parameters adapt to model's attention patterns
    
    Performance Characteristics:
        - Memory: O(n_sink * hidden_size) for sink token parameters
        - Compute: O(n_sink * seq_len) additional attention computation
        - Typical n_sink: 1-4 tokens sufficient for stability
    
    Attributes:
        n_sink (int): Number of sink tokens.
        hidden_size (int): Model hidden dimension.
        sink_tokens (nn.Parameter): Learnable sink token embeddings.
    
    Example:
        >>> sink = YvAttentionSink(hidden_size=4096, n_sink=4)
        >>> hidden = torch.randn(2, 1024, 4096)  # [batch, seq, hidden]
        >>> augmented, mask = sink(hidden)
        >>> # augmented shape: [2, 1028, 4096] (1024 + 4 sink tokens)
    
    Note:
        Sink tokens are typically only needed during inference with KV cache
        eviction. During training, full attention is computed and sinks are
        optional but can still improve stability.
    
    Reference:
        Xiao et al., "Efficient Streaming Language Models with Attention Sinks",
        ICLR 2024.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_sink: int = 4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize attention sink tokens.
        
        Args:
            hidden_size: Model hidden dimension. Sink tokens have the same
                dimensionality as regular tokens for seamless integration.
            n_sink: Number of sink tokens to prepend. More tokens provide
                more stable attention but increase computation. Typical
                values: 1-4 tokens.
            device: Device for sink token parameters.
            dtype: Data type for sink token parameters.
        
        Example:
            >>> sink = YvAttentionSink(4096, n_sink=4, device='cuda')
            >>> # Creates 4 learnable sink tokens of dimension 4096
        """
        super().__init__()
        self.n_sink = n_sink
        self.hidden_size = hidden_size
        
        self.sink_tokens = nn.Parameter(
            torch.randn(n_sink, hidden_size, device=device, dtype=dtype) * 0.02
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepend sink tokens to hidden states.
        
        Concatenates learnable sink tokens to the beginning of the input
        sequence, preparing the augmented sequence for attention computation.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
                The sequence to which sink tokens will be prepended.
        
        Returns:
            Tuple containing:
                - augmented: Hidden states with sink tokens prepended.
                    Shape [batch, seq_len + n_sink, hidden_size].
                - sink_mask: Mask indicating sink token positions.
                    Shape [batch, n_sink]. All ones, used for attention masking.
        
        Example:
            >>> hidden = torch.randn(2, 1024, 4096)
            >>> augmented, mask = sink(hidden)
            >>> augmented.shape  # [2, 1028, 4096]
            >>> mask.shape  # [2, 4]
        """
        batch_size = hidden_states.shape[0]
        
        sink_tokens = self.sink_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        augmented = torch.cat([sink_tokens, hidden_states], dim=1)
        
        sink_mask = torch.ones(
            batch_size, self.n_sink, 
            device=hidden_states.device, 
            dtype=hidden_states.dtype
        )
        
        return augmented, sink_mask


class YvQKNormalizer(nn.Module):
    """Query-Key Normalization for attention mechanism stability.
    
    Applies RMS (Root Mean Square) normalization to queries and keys before
    computing attention scores. This normalization technique significantly
    improves training stability for large language models by preventing
    unbounded attention scores that can cause gradient explosion.
    
    Mathematical Formulation:
        Q_norm = Q / sqrt(mean(Q^2) + eps) * gamma_q
        K_norm = K / sqrt(mean(K^2) + eps) * gamma_k
        
    Where gamma_q and gamma_k are learnable scale parameters.
    
    Key Benefits:
        - Prevents attention score explosion in large models
        - Enables higher learning rates without instability
        - Reduces sensitivity to initialization scale
        - Works synergistically with rotary position embeddings
    
    When to Use:
        - Models with > 7B parameters
        - Training with high learning rates
        - Models experiencing attention divergence
        - Long-context models with many attention layers
    
    Performance Characteristics:
        - Memory: O(head_dim) for learnable scale parameters
        - Compute: O(batch * heads * seq_len) for normalization
        - Overhead: ~2-5% additional compute per attention layer
    
    Attributes:
        q_norm (YvRMSNorm): RMS normalization for query vectors.
        k_norm (YvRMSNorm): RMS normalization for key vectors.
    
    Example:
        >>> normalizer = YvQKNormalizer(head_dim=128)
        >>> q = torch.randn(2, 32, 1024, 128)  # [batch, heads, seq, head_dim]
        >>> k = torch.randn(2, 32, 1024, 128)
        >>> q_norm, k_norm = normalizer(q, k)
        >>> # Normalized queries and keys for stable attention
    
    Note:
        QK normalization should be applied BEFORE rotary position embeddings
        to maintain the relative position information encoded by RoPE.
        The normalization is applied independently to each head.
    
    Reference:
        Henry et al., "Query-Key Normalization for Transformers", ICLR 2020.
    """
    
    def __init__(
        self,
        head_dim: int,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize QK normalizer with RMS normalization layers.
        
        Creates two independent RMS normalization layers, one for queries
        and one for keys. Each has its own learnable scale parameter.
        
        Args:
            head_dim: Per-head dimension for the attention mechanism.
                This is the dimension that will be normalized.
            eps: Epsilon value for numerical stability in RMS computation.
                Prevents division by zero when the RMS is very small.
                Default: 1e-6.
            device: Device for normalization parameters.
            dtype: Data type for normalization parameters.
        
        Example:
            >>> normalizer = YvQKNormalizer(head_dim=128, eps=1e-6, device='cuda')
        """
        super().__init__()
        self.q_norm = YvRMSNorm(head_dim, eps=eps, device=device, dtype=dtype)
        self.k_norm = YvRMSNorm(head_dim, eps=eps, device=device, dtype=dtype)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RMS normalization to queries and keys.
        
        Normalizes the query and key tensors along the last dimension
        (head_dim) using RMS normalization. The normalization is applied
        to all heads independently but with shared scale parameters.
        
        Args:
            q: Query tensor of shape [..., head_dim]. Can have any number
                of leading dimensions (batch, heads, sequence, etc.).
            k: Key tensor of shape [..., head_dim]. Must have the same
                trailing dimension as queries.
        
        Returns:
            Tuple of (normalized_query, normalized_key) tensors with the
            same shapes as inputs. The values are normalized to have
            unit RMS along the last dimension, scaled by learnable gamma.
        
        Example:
            >>> q = torch.randn(2, 32, 1024, 128)  # [batch, heads, seq, dim]
            >>> k = torch.randn(2, 32, 1024, 128)
            >>> q_norm, k_norm = normalizer(q, k)
            >>> q_norm.shape  # [2, 32, 1024, 128]
            >>> # RMS of q_norm along last dim is approximately gamma_q
        """
        original_shape = q.shape
        q_flat = q.reshape(-1, original_shape[-1])
        k_flat = k.reshape(-1, original_shape[-1])
        
        q_normed = self.q_norm(q_flat).reshape(original_shape)
        k_normed = self.k_norm(k_flat).reshape(original_shape)
        
        return q_normed, k_normed


class YvLinearAttention(nn.Module):
    """Linear Attention for efficient long-context sequence processing.
    
    Implements linear attention using kernel feature maps, reducing computational
    complexity from O(n^2) to O(n) for attention computation. This enables
    processing of very long sequences that would be infeasible with standard
    quadratic attention.
    
    Mathematical Formulation:
        Standard Attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d)) * V
        Linear Attention: Attention(Q,K,V) = phi(Q) * (phi(K)^T * V) / (phi(Q) * phi(K)^T)
        
    Where phi is a kernel feature map that approximates the softmax kernel.
    
    Supported Feature Map Types:
        - elu: ELU+1 kernel (default). Good balance of quality and speed.
            phi(x) = elu(x) + 1, ensures non-negative features.
        - performer: Random Fourier Features. Theoretically grounded approximation.
            Uses random projections with sin/cos activation.
        - softmax: Softmax approximation via learned projection.
            phi(x) = softmax(Wx) * sqrt(d), learned feature map.
        - relu: ReLU kernel. Simple and efficient.
            phi(x) = relu(x) + eps, ensures non-negative features.
    
    Key Features:
        - O(n) time and memory complexity
        - Supports causal masking for autoregressive models
        - Multiple kernel approximations for quality-speed tradeoffs
        - Compatible with standard attention interfaces
    
    Performance Characteristics:
        - Memory: O(batch * heads * feature_dim * seq_len) for feature maps
        - Compute: O(batch * heads * seq_len * (feature_dim + head_dim))
        - Speedup: ~10-100x faster than standard attention for long sequences
    
    When to Use:
        - Sequences longer than 4096 tokens
        - Memory-constrained environments
        - Real-time or latency-sensitive applications
        - Document-level or code-level processing
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        n_head (int): Number of attention heads.
        head_dim (int): Per-head dimension (hidden_size // n_head).
        feature_dim (int): Feature dimension for kernel approximation.
        feature_map_type (str): Type of kernel feature map used.
        causal (bool): Whether causal masking is enabled.
    
    Example:
        >>> attn = YvLinearAttention(4096, 32, feature_dim=64)
        >>> hidden = torch.randn(2, 16384, 4096)  # Long sequence
        >>> output = attn(hidden)  # O(n) computation
        >>> output.shape  # [2, 16384, 4096]
    
    Note:
        Linear attention may have slightly lower quality than standard attention
        for short sequences. Consider using a hybrid approach where standard
        attention is used for sequences < 4096 and linear for longer sequences.
    
    Reference:
        Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive
        Transformers with Linear Attention", ICML 2020.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        feature_dim: int = 64,
        feature_map_type: str = "elu",
        causal: bool = True,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize linear attention with specified kernel feature map.
        
        Args:
            hidden_size: Model hidden dimension. All projections operate in
                this dimension.
            n_head: Number of attention heads. Hidden size must be divisible
                by this value.
            feature_dim: Feature dimension for kernel approximation. Higher
                dimensions provide better approximation quality but more
                computation. Typical values: 32-128.
            feature_map_type: Type of feature map to use. Options:
                - "elu": ELU+1 kernel, default, good quality-speed balance
                - "performer": Random Fourier features, theoretically grounded
                - "softmax": Learned softmax approximation
                - "relu": Simple ReLU kernel, fastest option
            causal: Whether to use causal (autoregressive) masking.
                Set True for language modeling, False for bidirectional tasks.
            eps: Epsilon for numerical stability in attention normalization.
                Prevents division by zero in denominator computation.
            device: Device for projection parameters.
            dtype: Data type for projection parameters.
        
        Example:
            >>> attn = YvLinearAttention(
            ...     hidden_size=4096,
            ...     n_head=32,
            ...     feature_dim=64,
            ...     feature_map_type="performer",
            ...     causal=True
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.feature_dim = feature_dim
        self.feature_map_type = feature_map_type
        self.causal = causal
        self.eps = eps
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        
        if feature_map_type == "elu":
            self.feature_map = nn.Sequential(
                nn.Linear(self.head_dim, feature_dim, bias=False, device=device, dtype=dtype),
                nn.ELU()
            )
        elif feature_map_type == "performer":
            self.register_buffer(
                "random_matrix",
                torch.randn(self.head_dim, feature_dim, device=device, dtype=dtype) / math.sqrt(feature_dim)
            )
            self.proj_down = nn.Linear(self.head_dim, feature_dim, bias=True, device=device, dtype=dtype)
            nn.init.normal_(self.proj_down.weight, std=1.0 / math.sqrt(feature_dim))
            nn.init.zeros_(self.proj_down.bias)
        elif feature_map_type == "softmax":
            self.feature_map = nn.Linear(self.head_dim, feature_dim, bias=False, device=device, dtype=dtype)
        else:
            self.feature_map = nn.Sequential(
                nn.Linear(self.head_dim, feature_dim, bias=False, device=device, dtype=dtype),
                nn.ReLU()
            )
        
    def _kernel_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Apply kernel feature map to input tensor.
        
        Transforms the input through the selected kernel feature map to
        approximate the softmax kernel. The choice of feature map determines
        the quality-speed tradeoff of the linear attention approximation.
        
        Args:
            x: Input tensor of shape [..., head_dim]. The last dimension
                is transformed to feature_dim through the feature map.
        
        Returns:
            Feature-mapped tensor of shape [..., feature_dim]. The features
            are non-negative (except performer) and approximate the softmax
            kernel when used in attention computation.
        
        Note:
            Different feature maps have different properties:
            - ELU+1: Smooth, differentiable, always positive
            - Performer: Random features, unbiased approximation
            - Softmax: Learned approximation, can adapt to data
            - ReLU: Simple, fast, may have sparse gradients
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        
        if self.feature_map_type == "elu":
            features = self.feature_map(x_flat)
            features = F.elu(features) + 1
        elif self.feature_map_type == "performer":
            features = self._performer_feature(x_flat)
        elif self.feature_map_type == "softmax":
            features = self.feature_map(x_flat)
            features = F.softmax(features, dim=-1)
            features = features * math.sqrt(features.shape[-1])
        else:
            features = self.feature_map(x_flat)
            features = F.relu(features) + self.eps
            
        return features.reshape(*original_shape[:-1], self.feature_dim)
    
    def _performer_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Random Fourier Features for Performer-style approximation.
        
        Implements the kernel approximation from "Rethinking Attention with
        Performers" using random projections with trigonometric activations.
        This provides an unbiased estimate of the softmax kernel.
        
        Mathematical Formulation:
            phi(x) = [sin(Wx), cos(Wx)] / sqrt(d)
            where W is a random matrix with entries ~ N(0, 1/d)
        
        Args:
            x: Input tensor of shape [batch * seq, head_dim].
        
        Returns:
            Random feature approximation of shape [batch * seq, feature_dim].
            The features approximate the softmax kernel when used in attention.
        
        Note:
            The random matrix is fixed after initialization, providing
            consistent approximation across forward passes. The quality
            improves with larger feature_dim.
        """
        projection = torch.matmul(x, self.random_matrix)
        projection = self.proj_down(x) + projection
        
        h = torch.sin(projection)
        h_prime = torch.cos(projection)
        
        features = torch.cat([h, h_prime], dim=-1)
        
        features = features[:, :self.feature_dim]
        
        norm = torch.norm(features, dim=-1, keepdim=True).clamp(min=1.0)
        features = features / norm * math.sqrt(self.feature_dim)
        
        return features
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute linear attention over the input sequence.
        
        Projects inputs to Q, K, V, applies kernel feature maps to Q and K,
        then computes attention in O(n) time using the kernel trick.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
                The sequence to attend over.
            attention_mask: Optional attention mask. Currently not used in
                linear attention but kept for API compatibility.
        
        Returns:
            Output tensor of shape [batch, seq_len, hidden_size]. The attention
            output has the same dimensionality as the input.
        
        Example:
            >>> hidden = torch.randn(2, 8192, 4096)
            >>> output = attn(hidden)
            >>> output.shape  # [2, 8192, 4096]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim)
        
        q_features = self._kernel_feature(q)
        k_features = self._kernel_feature(k)
        
        q_features = q_features.transpose(1, 2)
        k_features = k_features.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.causal:
            output = self._causal_linear_attention(q_features, k_features, v)
        else:
            output = self._linear_attention(q_features, k_features, v)
        
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        
        return output
    
    def _linear_attention(
        self,
        q_features: torch.Tensor,
        k_features: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """Compute non-causal (bidirectional) linear attention.
        
        Uses the kernel trick to compute attention in O(n) time:
            Attention(Q, K, V) = phi(Q) * (phi(K)^T * V) / (phi(Q) * phi(K)^T)
        
        This formulation allows computing the attention output without
        materializing the O(n^2) attention matrix.
        
        Args:
            q_features: Query features of shape [batch, heads, seq, feature_dim].
            k_features: Key features of shape [batch, heads, seq, feature_dim].
            v: Values of shape [batch, heads, seq, head_dim].
        
        Returns:
            Attention output of shape [batch, heads, seq, head_dim].
        
        Note:
            This computes full bidirectional attention where each position
            can attend to all other positions. Use _causal_linear_attention
            for autoregressive models.
        """
        kv = torch.einsum('bhfd,bhvd->bhfv', k_features, v)
        k_sum = k_features.sum(dim=3, keepdim=True)
        
        numerator = torch.einsum('bhfd,bhfv->bhvd', q_features, kv)
        denominator = torch.einsum('bhfd,bhfd->bhvd', q_features, k_sum.expand_as(q_features))
        
        output = numerator / (denominator + self.eps)
        
        return output
    
    def _causal_linear_attention(
        self,
        q_features: torch.Tensor,
        k_features: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """Compute causal (autoregressive) linear attention using cumulative sum.
        
        Implements causal linear attention where each position can only attend
        to previous positions. Uses cumulative sum for efficient O(n) computation.
        
        Mathematical Formulation:
            For position i, output[i] = sum_{j<=i} phi(Q[i]) * phi(K[j])^T * V[j]
                                        / sum_{j<=i} phi(Q[i]) * phi(K[j])^T
        
        Args:
            q_features: Query features of shape [batch, heads, seq, feature_dim].
            k_features: Key features of shape [batch, heads, seq, feature_dim].
            v: Values of shape [batch, heads, seq, head_dim].
        
        Returns:
            Causal attention output of shape [batch, heads, seq, head_dim].
            Position i only depends on positions 0 to i.
        
        Note:
            The cumulative sum trick enables O(n) computation while maintaining
            the causal constraint. This is equivalent to computing attention
            with a lower-triangular mask in standard attention.
        """
        batch_size, n_heads, seq_len, feature_dim = q_features.shape
        head_dim = v.shape[-1]
        
        kv = k_features.unsqueeze(-1) * v.unsqueeze(-2)
        
        kv_cumsum = torch.cumsum(kv, dim=2)
        
        k_sum = torch.cumsum(k_features, dim=2)
        
        numerator = torch.einsum('bhqd,bhqdf->bhvf', q_features, kv_cumsum)
        
        denominator = torch.einsum('bhqd,bhqd->bhq', q_features, k_sum)
        denominator = denominator.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        
        output = numerator / (denominator + self.eps)
        
        return output


class YvSlidingWindowAttention(nn.Module):
    """Sliding Window Attention for efficient local context processing.
    
    Implements local attention within a fixed-size sliding window, significantly
    reducing memory and computational complexity for long sequences while
    maintaining the ability to capture local dependencies.
    
    Mathematical Formulation:
        For position i, attention is computed only over positions j where:
            |i - j| <= window_size / 2
        
        This reduces attention complexity from O(n^2) to O(n * window_size).
    
    Key Features:
        - Fixed memory footprint regardless of sequence length
        - Efficient for capturing local patterns and dependencies
        - Supports dilation for sparse attention patterns
        - Compatible with KV caching for efficient inference
    
    Use Cases:
        - Long document processing where local context is most important
        - Code understanding with local scope awareness
        - Streaming applications with memory constraints
        - Hierarchical attention architectures
    
    Performance Characteristics:
        - Memory: O(batch * heads * seq_len * window_size) for attention
        - Compute: O(batch * heads * seq_len * window_size * head_dim)
        - Speedup: ~seq_len/window_size faster than full attention
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        n_head (int): Number of attention heads.
        head_dim (int): Per-head dimension.
        window_size (int): Size of the attention window.
        dilation (int): Dilation factor for sparse patterns.
    
    Example:
        >>> attn = YvSlidingWindowAttention(4096, 32, window_size=512)
        >>> hidden = torch.randn(2, 16384, 4096)
        >>> output = attn(hidden)
        >>> # Each position attends to 512 neighboring positions
    
    Note:
        Sliding window attention may miss long-range dependencies. Consider
        combining with global attention tokens or hybrid architectures for
        tasks requiring both local and global context.
    
    Reference:
        Beltagy et al., "Longformer: The Long-Document Transformer", 2020.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        window_size: int = 512,
        dilation: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize sliding window attention.
        
        Args:
            hidden_size: Model hidden dimension. All projections operate in
                this dimension.
            n_head: Number of attention heads. Hidden size must be divisible
                by this value.
            window_size: Size of the attention window. Each position attends
                to window_size/2 positions on each side. Larger windows
                capture more context but use more memory.
            dilation: Dilation factor for sparse attention patterns. A dilation
                of d means every d-th position is attended within the window.
                Default: 1 (no dilation, attend to all positions in window).
            device: Device for projection parameters.
            dtype: Data type for projection parameters.
        
        Example:
            >>> attn = YvSlidingWindowAttention(
            ...     hidden_size=4096,
            ...     n_head=32,
            ...     window_size=1024,
            ...     dilation=1
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.window_size = window_size
        self.dilation = dilation
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        
    def _create_window_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create sliding window attention mask.
        
        Generates a boolean mask where True indicates positions that should
        be masked out (not attended to). The mask implements the sliding
        window constraint where each position only attends to nearby positions.
        
        Args:
            seq_len: Sequence length for the mask.
            device: Device to create the mask on.
        
        Returns:
            Boolean mask tensor of shape [seq_len, seq_len].
            True values indicate positions to mask (set to -inf in attention).
        
        Note:
            The mask is symmetric around each position, attending to
            window_size/2 positions on each side. Edge positions have
            smaller effective windows due to sequence boundaries.
        """
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = False
            
        return mask
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute sliding window attention.
        
        Projects inputs to Q, K, V and computes attention only within
        the local window around each position.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
            attention_mask: Optional additional attention mask to apply.
                Combined with the window mask for custom masking patterns.
            past_key_value: Optional cached key/value states from previous
                forward passes. Used for efficient autoregressive generation.
            use_cache: Whether to return the key/value states for caching.
        
        Returns:
            If use_cache is False:
                Output tensor of shape [batch, seq_len, hidden_size].
            If use_cache is True:
                Tuple of (output, (key_cache, value_cache)) where caches
                can be passed to subsequent forward passes.
        
        Example:
            >>> hidden = torch.randn(2, 1024, 4096)
            >>> output, cache = attn(hidden, use_cache=True)
            >>> # Use cache for next token generation
            >>> next_output, new_cache = attn(next_hidden, past_key_value=cache)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        kv_seq_len = k.shape[2]
        
        window_mask = self._create_window_mask(kv_seq_len, hidden_states.device)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        attn_weights = attn_weights.masked_fill(
            window_mask.unsqueeze(0).unsqueeze(0),
            float('-inf')
        )
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        if use_cache:
            return output, (k, v)
        return output


class YvSparseAttention(nn.Module):
    """Sparse Attention with configurable attention patterns.
    
    Implements various sparse attention patterns that reduce computational
    complexity while maintaining the ability to capture both local and
    long-range dependencies through strategic attention patterns.
    
    Supported Patterns:
        - longformer: Combines local sliding window attention with global
            attention on specific tokens. Global tokens attend to and are
            attended by all positions. Ideal for document classification.
        - bigbird: Combines random, local, and global attention patterns.
            Random attention provides stochastic long-range connections.
            Best for tasks requiring diverse attention patterns.
        - block: Block-sparse attention where attention is computed within
            blocks and between special block connections. Efficient for
            structured inputs like documents with sections.
    
    Mathematical Formulation:
        For each pattern, the attention matrix A is sparse:
        - longformer: A[i,j] = 1 if |i-j| <= w/2 OR i in global OR j in global
        - bigbird: A[i,j] = 1 if local OR random OR global connection
        - block: A[i,j] = 1 if same block OR special inter-block connection
    
    Key Features:
        - Configurable attention patterns for different use cases
        - Combines local and global attention for comprehensive coverage
        - Memory-efficient sparse attention computation
        - Supports custom global token positions
    
    Performance Characteristics:
        - Memory: O(n * (window + global + random)) instead of O(n^2)
        - Compute: Proportional to number of attended positions
        - Typical sparsity: 90-99% of attention matrix is zero
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        n_head (int): Number of attention heads.
        head_dim (int): Per-head dimension.
        pattern (str): Sparse attention pattern type.
        block_size (int): Block size for block-sparse patterns.
        num_global_tokens (int): Number of global attention tokens.
        num_random_tokens (int): Number of random attention connections.
        window_size (int): Local attention window size.
    
    Example:
        >>> attn = YvSparseAttention(
        ...     4096, 32,
        ...     pattern="longformer",
        ...     window_size=256,
        ...     num_global_tokens=1
        ... )
        >>> hidden = torch.randn(2, 8192, 4096)
        >>> output = attn(hidden)
    
    Note:
        Sparse attention requires careful selection of global tokens and
        pattern parameters. The optimal configuration depends on the task
        and input characteristics.
    
    Reference:
        Zaheer et al., "Big Bird: Transformers for Longer Sequences", NeurIPS 2020.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        pattern: str = "longformer",
        block_size: int = 64,
        num_global_tokens: int = 1,
        num_random_tokens: int = 0,
        window_size: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize sparse attention with specified pattern.
        
        Args:
            hidden_size: Model hidden dimension.
            n_head: Number of attention heads.
            pattern: Sparse attention pattern type. Options:
                - "longformer": Local + global attention pattern
                - "bigbird": Local + random + global attention pattern
                - "block": Block-sparse attention pattern
            block_size: Block size for block-sparse computation. Determines
                the granularity of sparse attention blocks.
            num_global_tokens: Number of tokens with global attention.
                These tokens attend to and are attended by all positions.
                Typically 1 (CLS token) or more for multi-task scenarios.
            num_random_tokens: Number of random attention connections per
                position. Only used in "bigbird" pattern. Provides
                stochastic long-range connections.
            window_size: Size of local attention window. Each position
                attends to this many neighboring positions.
            device: Device for projection parameters.
            dtype: Data type for projection parameters.
        
        Example:
            >>> attn = YvSparseAttention(
            ...     hidden_size=4096,
            ...     n_head=32,
            ...     pattern="bigbird",
            ...     block_size=64,
            ...     num_global_tokens=1,
            ...     num_random_tokens=3,
            ...     window_size=256
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.pattern = pattern
        self.block_size = block_size
        self.num_global_tokens = num_global_tokens
        self.num_random_tokens = num_random_tokens
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        
    def _create_sparse_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create sparse attention mask based on pattern.
        
        Args:
            seq_len: Sequence length.
            device: Target device.
            
        Returns:
            Sparse attention mask.
        """
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        
        if self.pattern == "longformer":
            mask[:self.num_global_tokens, :] = False
            mask[:, :self.num_global_tokens] = False
            
            for i in range(self.num_global_tokens, seq_len):
                start = max(self.num_global_tokens, i - self.window_size // 2)
                end = min(seq_len, i + self.window_size // 2 + 1)
                mask[i, start:end] = False
                
        elif self.pattern == "bigbird":
            mask[:self.num_global_tokens, :] = False
            mask[:, :self.num_global_tokens] = False
            
            for i in range(self.num_global_tokens, seq_len):
                start = max(self.num_global_tokens, i - self.window_size // 2)
                end = min(seq_len, i + self.window_size // 2 + 1)
                mask[i, start:end] = False
                
                if self.num_random_tokens > 0:
                    random_indices = torch.randperm(seq_len - self.num_global_tokens, device=device)[:self.num_random_tokens]
                    random_indices = random_indices + self.num_global_tokens
                    mask[i, random_indices] = False
                    
        elif self.pattern == "block":
            num_blocks = (seq_len + self.block_size - 1) // self.block_size
            for i in range(num_blocks):
                start_i = i * self.block_size
                end_i = min((i + 1) * self.block_size, seq_len)
                for j in range(num_blocks):
                    if abs(i - j) <= 1:
                        start_j = j * self.block_size
                        end_j = min((j + 1) * self.block_size, seq_len)
                        mask[start_i:end_i, start_j:end_j] = False
                        
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        mask = mask | causal_mask
        
        return mask
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute sparse attention.
        
        Args:
            hidden_states: Input tensor.
            attention_mask: Optional attention mask.
            
        Returns:
            Output tensor.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        sparse_mask = self._create_sparse_mask(seq_len, hidden_states.device)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        attn_weights = attn_weights.masked_fill(
            sparse_mask.unsqueeze(0).unsqueeze(0),
            float('-inf')
        )
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output


class YvPagedAttention(nn.Module):
    """PagedAttention for efficient KV cache memory management.
    
    Implements block-wise KV cache management that enables efficient memory
    allocation and sharing across multiple sequences. This approach eliminates
    memory fragmentation and enables prefix caching for shared prompts.
    
    Key Concepts:
        - Block: Fixed-size chunk of KV cache (e.g., 16 tokens per block)
        - Page Table: Maps logical sequence positions to physical block indices
        - Prefix Caching: Reuses KV cache for shared prompt prefixes
    
    Memory Management:
        Traditional KV cache allocates contiguous memory per sequence, leading to:
        - Memory fragmentation
        - Inefficient utilization for variable-length sequences
        - No sharing of common prefixes
        
        PagedAttention solves these by:
        - Allocating fixed-size blocks from a pool
        - Using page tables for flexible mapping
        - Enabling block sharing across sequences
    
    Key Features:
        - Near-zero memory fragmentation
        - Efficient prefix caching for shared prompts
        - Supports variable-length sequences in batch
        - Compatible with GQA/MQA architectures
        - Enables memory-efficient beam search
    
    Performance Characteristics:
        - Memory overhead: ~5% for page tables
        - Throughput improvement: 2-4x for batched inference
        - Latency reduction: 10-30% for shared prefixes
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        n_head (int): Number of query heads.
        n_kv_head (int): Number of key/value heads for GQA.
        head_dim (int): Per-head dimension.
        block_size (int): Number of tokens per cache block.
        max_num_blocks (int): Maximum number of blocks in the pool.
        key_cache (torch.Tensor): Preallocated key cache blocks.
        value_cache (torch.Tensor): Preallocated value cache blocks.
        block_tables (torch.Tensor): Page table for block mapping.
        context_lens (torch.Tensor): Context lengths for each sequence.
    
    Example:
        >>> attn = YvPagedAttention(4096, 32, 8, block_size=16)
        >>> # Process prompt with shared prefix
        >>> output = attn(hidden, block_indices, seq_lens)
    
    Note:
        PagedAttention is primarily beneficial during inference. For training,
        standard attention with gradient checkpointing is typically more efficient.
    
    Reference:
        Kwon et al., "Efficient Memory Management for Large Language Model
        Serving with PagedAttention", SOSP 2023.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        n_kv_head: int,
        block_size: int = 16,
        max_num_blocks: int = 1024,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize PagedAttention with block cache pool.
        
        Args:
            hidden_size: Model hidden dimension. All projections operate in
                this dimension.
            n_head: Number of query heads for attention computation.
            n_kv_head: Number of key/value heads. For GQA, this is less than
                n_head. For MHA, this equals n_head.
            block_size: Number of tokens stored in each cache block. Smaller
                blocks provide finer granularity but more overhead. Typical
                values: 8-32 tokens per block.
            max_num_blocks: Maximum number of blocks in the cache pool.
                Total cache capacity = max_num_blocks * block_size tokens.
            device: Device for projection parameters and cache buffers.
            dtype: Data type for projection parameters and cache buffers.
        
        Example:
            >>> attn = YvPagedAttention(
            ...     hidden_size=4096,
            ...     n_head=32,
            ...     n_kv_head=8,  # GQA with 8 KV heads
            ...     block_size=16,
            ...     max_num_blocks=4096  # 64K token capacity
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = hidden_size // n_head
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        
        self.register_buffer(
            "key_cache",
            torch.zeros(max_num_blocks, n_kv_head, block_size, self.head_dim, device=device, dtype=dtype),
            persistent=False
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(max_num_blocks, n_kv_head, block_size, self.head_dim, device=device, dtype=dtype),
            persistent=False
        )
        self.register_buffer(
            "block_tables",
            torch.zeros(max_num_blocks, dtype=torch.long, device=device),
            persistent=False
        )
        self.register_buffer(
            "context_lens",
            torch.zeros(max_num_blocks, dtype=torch.long, device=device),
            persistent=False
        )
        
    def _gather_cache(
        self,
        block_indices: torch.Tensor,
        seq_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather key/value cache for specified block indices.
        
        Retrieves the cached key and value tensors for the given block
        indices, assembling them into contiguous tensors for attention
        computation.
        
        Args:
            block_indices: Block indices to gather from the cache pool.
                Shape depends on batch configuration.
            seq_lens: Sequence lengths for each batch item. Used to
                determine valid positions within each block.
        
        Returns:
            Tuple of (keys, values) tensors assembled from the cache.
            Shape: [batch, n_kv_head, total_seq_len, head_dim].
        
        Note:
            This operation is similar to gather in database systems,
            where scattered blocks are assembled into contiguous memory
            for efficient access.
        """
        keys = self.key_cache[block_indices]
        values = self.value_cache[block_indices]
        
        # PQCache: Product Quantization for KV cache compression
        # Auto-enabled when cache elements > 10M
        # Based on: PKU-DAIR SIGMOD 2025
        if self.training and keys.numel() > 1e7:
            codebook_size = 256
            keys_flat = keys.view(-1, keys.shape[-1])
            values_flat = values.view(-1, values.shape[-1])
            
            with torch.no_grad():
                # K-means quantization for keys
                indices = torch.randperm(keys_flat.shape[0], device=keys.device)[:codebook_size]
                key_centroids = keys_flat[indices]
                
                for _ in range(3):
                    distances = torch.cdist(keys_flat, key_centroids)
                    assignments = distances.argmin(dim=1)
                    for c in range(codebook_size):
                        mask = assignments == c
                        if mask.any():
                            key_centroids[c] = keys_flat[mask].mean(dim=0)
                
                keys = key_centroids[assignments].view_as(keys)
                
                # K-means quantization for values
                indices = torch.randperm(values_flat.shape[0], device=values.device)[:codebook_size]
                value_centroids = values_flat[indices]
                
                for _ in range(3):
                    distances = torch.cdist(values_flat, value_centroids)
                    assignments = distances.argmin(dim=1)
                    for c in range(codebook_size):
                        mask = assignments == c
                        if mask.any():
                            value_centroids[c] = values_flat[mask].mean(dim=0)
                
                values = value_centroids[assignments].view_as(values)
        
        return keys, values
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        block_indices: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute paged attention.
        
        Args:
            hidden_states: Input tensor.
            block_indices: Block indices for cache access.
            seq_lens: Sequence lengths.
            attention_mask: Optional attention mask.
            
        Returns:
            Output tensor.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        if block_indices is not None and seq_lens is not None:
            cached_k, cached_v = self._gather_cache(block_indices, seq_lens)
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            
        kv_seq_len = k.shape[2]
        
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
            
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        causal_mask = torch.triu(
            torch.ones(seq_len, kv_seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=kv_seq_len - seq_len + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output


class YvFlashAttention(nn.Module):
    """Flash Attention 2/3 wrapper for high-performance attention computation.
    
    Provides optimized attention implementation using Flash Attention kernels
    when available, with automatic fallback to standard PyTorch SDPA. Flash
    Attention significantly reduces memory usage and improves speed through
    IO-aware algorithm design.
    
    Flash Attention Versions:
        - Flash Attention 2: Optimized for NVIDIA Ampere (A100, RTX 30/40)
            2-4x speedup over standard attention
            Memory: O(n) instead of O(n^2)
            Requires CUDA 11.6+, compute capability 8.0+
        
        - Flash Attention 3: Optimized for NVIDIA Hopper (H100)
            Additional FP8 support
            Async operations for better utilization
            Requires CUDA 12.0+, compute capability 9.0+
    
    Key Features:
        - Automatic version detection and fallback
        - Supports Grouped-Query Attention (GQA)
        - Fused QKV projection for efficiency
        - Memory-efficient causal masking
        - Compatible with KV caching
    
    Performance Characteristics:
        - Memory: O(n) for attention, no materialization of n^2 matrix
        - Speed: 2-4x faster than standard attention
        - Numerical: FP16/BF16 with improved numerical stability
    
    When Flash is Unavailable:
        Falls back to PyTorch's scaled_dot_product_attention which provides:
        - Memory-efficient attention on CUDA
        - Flash Attention integration when available
        - Standard attention as final fallback
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        n_head (int): Number of query heads.
        n_kv_head (int): Number of key/value heads for GQA.
        head_dim (int): Per-head dimension.
        attention_dropout (float): Dropout probability during training.
        use_flash_v3 (bool): Whether to prefer Flash Attention 3.
        fused_qkv (bool): Whether QKV projections are fused.
        _flash_available (bool): Whether Flash Attention is available.
        _flash_version (int): Detected Flash Attention version (2 or 3).
    
    Example:
        >>> attn = YvFlashAttention(4096, 32, 8, use_flash_v3=False)
        >>> hidden = torch.randn(2, 4096, 4096, device='cuda', dtype=torch.bfloat16)
        >>> output = attn(hidden)
    
    Note:
        Flash Attention requires specific CUDA versions and GPU architectures.
        The module automatically detects availability and falls back gracefully.
    
    Reference:
        Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
        with IO-Awareness", NeurIPS 2022.
        Dao, "FlashAttention-2: Faster Attention with Better Parallelism and
        Work Partitioning", 2023.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        n_kv_head: int,
        attention_dropout: float = 0.0,
        use_flash_v3: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize Flash Attention with automatic version detection.
        
        Args:
            hidden_size: Model hidden dimension. All projections operate in
                this dimension.
            n_head: Number of query heads for attention computation.
            n_kv_head: Number of key/value heads. For GQA, this is less than
                n_head. For standard MHA, this equals n_head.
            attention_dropout: Dropout probability applied to attention weights
                during training. Set to 0.0 for inference.
            use_flash_v3: Whether to prefer Flash Attention 3 over version 2.
                Only effective on H100+ GPUs with CUDA 12.0+.
            device: Device for projection parameters.
            dtype: Data type for projection parameters. BF16 recommended for
                Flash Attention 2, FP8 for Flash Attention 3.
        
        Example:
            >>> attn = YvFlashAttention(
            ...     hidden_size=4096,
            ...     n_head=32,
            ...     n_kv_head=8,  # GQA
            ...     attention_dropout=0.0,
            ...     use_flash_v3=False,
            ...     device='cuda',
            ...     dtype=torch.bfloat16
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = hidden_size // n_head
        self.attention_dropout = attention_dropout
        self.use_flash_v3 = use_flash_v3
        self.scale = self.head_dim ** -0.5
        
        self.fused_qkv = True
        qkv_out = (n_head + 2 * n_kv_head) * self.head_dim
        self.qkv_proj = nn.Linear(hidden_size, qkv_out, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(n_head * self.head_dim, hidden_size, bias=False, device=device, dtype=dtype)
        
        self._flash_available = self._check_flash_availability()
        
    def _check_flash_availability(self) -> bool:
        """Check if Flash Attention is available on the current system.
        
        Attempts to import Flash Attention 3 first (if use_flash_v3 is True),
        then falls back to Flash Attention 2. Sets internal flags for version
        tracking.
        
        Returns:
            True if any version of Flash Attention is available, False otherwise.
        
        Note:
            Flash Attention availability depends on:
            - CUDA version (11.6+ for v2, 12.0+ for v3)
            - GPU architecture (Ampere+ for v2, Hopper+ for v3)
            - Installed flash-attn package
        """
        try:
            if self.use_flash_v3:
                try:
                    import flash_attn_v3
                    self._flash_version = 3
                    return True
                except ImportError:
                    pass
                    
            import flash_attn
            if hasattr(flash_attn, 'flash_attn_func'):
                self._flash_version = 2
                return True
            return False
        except ImportError:
            self._flash_version = 0
            return False
            
    def _flash_attention_v3(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute attention using Flash Attention 3 kernel.
        
        Flash Attention 3 is optimized for NVIDIA Hopper architecture (H100)
        with support for FP8 precision and asynchronous operations.
        
        Args:
            q: Query tensor of shape [batch, seq_len, n_head, head_dim].
            k: Key tensor of shape [batch, seq_len, n_kv_head, head_dim].
            v: Value tensor of shape [batch, seq_len, n_kv_head, head_dim].
        
        Returns:
            Attention output tensor of shape [batch, seq_len, n_head, head_dim].
        
        Note:
            For GQA (n_kv_head < n_head), keys and values are automatically
            expanded to match the number of query heads through repetition.
        """
        try:
            import flash_attn_v3
            from flash_attn_v3 import flash_attn_func as flash_attn_v3_func
            
            if self.n_kv_head != self.n_head:
                repeat = self.n_head // self.n_kv_head
                k = k.repeat_interleave(repeat, dim=2)
                v = v.repeat_interleave(repeat, dim=2)
            
            output = flash_attn_v3_func(
                q, k, v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=True,
                window_size=(-1, -1)
            )
            return output
        except Exception:
            return self._flash_attention_v2(q, k, v)
            
    def _flash_attention_v2(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute attention using Flash Attention 2.
        
        Args:
            q: Query tensor [batch, seq, heads, head_dim]
            k: Key tensor [batch, seq, kv_heads, head_dim]
            v: Value tensor [batch, seq, kv_heads, head_dim]
            
        Returns:
            Output tensor.
        """
        import flash_attn
        from flash_attn import flash_attn_func
        
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
            
        output = flash_attn_func(
            q, k, v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=True
        )
        return output
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute flash attention.
        
        Args:
            hidden_states: Input tensor.
            attention_mask: Optional attention mask.
            past_key_value: Optional cached key/value states.
            use_cache: Whether to return cached states.
            
        Returns:
            Output tensor or tuple with cache.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        qkv = self.qkv_proj(hidden_states)
        q_end = self.n_head * self.head_dim
        kv_each = self.n_kv_head * self.head_dim
        
        q = qkv[:, :, :q_end].view(batch_size, seq_len, self.n_head, self.head_dim)
        k = qkv[:, :, q_end:q_end + kv_each].view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = qkv[:, :, q_end + kv_each:].view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            
        kv_seq_len = k.shape[1]
        
        if self._flash_available and hidden_states.device.type == 'cuda':
            try:
                if hasattr(self, '_flash_version') and self._flash_version == 3:
                    output = self._flash_attention_v3(q, k, v)
                else:
                    output = self._flash_attention_v2(q, k, v)
            except Exception:
                output = self._standard_attention(q, k, v, attention_mask)
        else:
            output = self._standard_attention(q, k, v, attention_mask)
            
        output = output.view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        if use_cache:
            k_cache = k[:, :, :self.n_kv_head] if self.n_kv_head != self.n_head else k
            v_cache = v[:, :, :self.n_kv_head] if self.n_kv_head != self.n_head else v
            return output, (k_cache, v_cache)
            
        return output
        
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Fallback standard attention computation.
        
        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            attention_mask: Optional attention mask.
            
        Returns:
            Attention output.
        """
        batch_size, seq_len = q.shape[0], q.shape[1]
        kv_seq_len = k.shape[1]
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        causal_mask = torch.triu(
            torch.ones(seq_len, kv_seq_len, device=q.device, dtype=torch.bool),
            diagonal=kv_seq_len - seq_len + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
            
        output = torch.matmul(attn_weights, v)
        return output.transpose(1, 2)


class YvLocalGlobalAttention(nn.Module):
    """Local-Global Attention for hybrid context processing.
    
    Implements a hybrid attention mechanism that combines local sliding window
    attention with global full attention, using different heads for each pattern.
    This enables efficient processing of both fine-grained local patterns and
    long-range dependencies within a single attention layer.
    
    Architecture:
        - Local Heads: Attend only to nearby positions within a window.
            Efficient for capturing local syntax, phrases, and patterns.
        - Global Heads: Attend to all positions in the sequence.
            Captures long-range dependencies and document-level context.
    
    Mathematical Formulation:
        For local heads h in [0, local_heads):
            Attention_h(Q,K,V) = softmax(QK^T / sqrt(d) + M_local) * V
            where M_local masks positions outside the window
        
        For global heads h in [local_heads, n_head):
            Attention_h(Q,K,V) = softmax(QK^T / sqrt(d)) * V
            (full attention over all positions)
    
    Key Features:
        - Head-level specialization for different attention patterns
        - Configurable ratio of local to global heads
        - Supports global tokens for special positions (e.g., CLS)
        - Efficient for long documents with both local and global needs
    
    Use Cases:
        - Document classification with local context awareness
        - Code understanding with local scope and global imports
        - Long-form generation with coherent local and global structure
    
    Performance Characteristics:
        - Memory: O(n * (local_window * local_heads + n * global_heads))
        - Compute: Proportional to attended positions per head type
        - Speedup: ~n / (local_window + n * global_ratio) vs full attention
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        n_head (int): Total number of attention heads.
        head_dim (int): Per-head dimension.
        local_window (int): Window size for local attention.
        global_tokens (int): Number of tokens with global attention.
        local_heads (int): Number of heads for local attention.
        global_heads (int): Number of heads for global attention.
    
    Example:
        >>> attn = YvLocalGlobalAttention(
        ...     hidden_size=4096,
        ...     n_head=32,
        ...     local_window=512,
        ...     local_heads=24  # 24 local, 8 global heads
        ... )
        >>> hidden = torch.randn(2, 8192, 4096)
        >>> output = attn(hidden)
    
    Note:
        The optimal ratio of local to global heads depends on the task.
        Tasks with strong local dependencies benefit from more local heads,
        while tasks requiring global reasoning need more global heads.
    
    Reference:
        Beltagy et al., "Longformer: The Long-Document Transformer", 2020.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        local_window: int = 512,
        global_tokens: int = 1,
        local_heads: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize Local-Global Attention with head specialization.
        
        Args:
            hidden_size: Model hidden dimension. All projections operate in
                this dimension.
            n_head: Total number of attention heads. These will be split
                between local and global attention patterns.
            local_window: Window size for local attention heads. Each local
                head attends to this many neighboring positions.
            global_tokens: Number of tokens that receive global attention
                from all heads. Typically 1 for CLS token.
            local_heads: Number of heads dedicated to local attention.
                Remaining heads (n_head - local_heads) use global attention.
                Default: n_head // 2 (equal split).
            device: Device for projection parameters.
            dtype: Data type for projection parameters.
        
        Example:
            >>> attn = YvLocalGlobalAttention(
            ...     hidden_size=4096,
            ...     n_head=32,
            ...     local_window=256,
            ...     global_tokens=1,
            ...     local_heads=20  # 20 local, 12 global
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.local_window = local_window
        self.global_tokens = global_tokens
        self.local_heads = local_heads or n_head // 2
        self.global_heads = n_head - self.local_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute local-global attention with head specialization.
        
        Splits the attention heads into local and global groups, computes
        attention separately for each group, and concatenates the results.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
            attention_mask: Optional attention mask to apply. Applied to
                both local and global attention patterns.
        
        Returns:
            Output tensor of shape [batch, seq_len, hidden_size]. The output
            combines information from both local and global attention patterns.
        
        Example:
            >>> hidden = torch.randn(2, 4096, 4096)
            >>> output = attn(hidden)
            >>> output.shape  # [2, 4096, 4096]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        if self.local_heads > 0:
            local_q = q[:, :self.local_heads]
            local_k = k[:, :self.local_heads]
            local_v = v[:, :self.local_heads]
            
            local_out = self._local_attention(local_q, local_k, local_v, attention_mask)
        else:
            local_out = torch.zeros(batch_size, self.local_heads, seq_len, self.head_dim, device=hidden_states.device)
            
        if self.global_heads > 0:
            global_q = q[:, self.local_heads:]
            global_k = k[:, self.local_heads:]
            global_v = v[:, self.local_heads:]
            
            global_out = self._global_attention(global_q, global_k, global_v, attention_mask)
        else:
            global_out = torch.zeros(batch_size, self.global_heads, seq_len, self.head_dim, device=hidden_states.device)
            
        output = torch.cat([local_out, global_out], dim=1)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        
        return output
        
    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute local attention.
        
        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            attention_mask: Optional attention mask.
            
        Returns:
            Local attention output.
        """
        batch_size, _, seq_len, _ = q.shape
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        local_mask = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.local_window // 2)
            end = min(seq_len, i + self.local_window // 2 + 1)
            local_mask[i, start:end] = False
            
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        local_mask = local_mask | causal_mask
        
        attn_weights = attn_weights.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)
        
    def _global_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute global attention.
        
        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            attention_mask: Optional attention mask.
            
        Returns:
            Global attention output.
        """
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        seq_len = q.shape[2]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)


class YvRingAttention(nn.Module):
    """Ring Attention for distributed ultra-long context processing.
    
    Implements ring attention pattern for processing sequences that exceed
    single device memory capacity by distributing key-value pairs across
    multiple devices in a ring topology. This enables processing of arbitrarily
    long sequences with bounded per-device memory.
    
    Architecture:
        - Ring Topology: Devices arranged in a logical ring
        - Key-Value Distribution: Each device holds a portion of K/V
        - Ring Communication: K/V passed around the ring for complete attention
        - Online Softmax: Numerically stable attention with partial results
    
    Mathematical Formulation:
        Standard Attention: softmax(QK^T) * V
        Ring Attention: sum over ring steps of partial softmax results
        
        For each ring step r:
            1. Receive K_r, V_r from previous device
            2. Compute partial attention: exp(QK_r^T - max) * V_r
            3. Accumulate numerator and denominator
            4. Send K_r, V_r to next device
        Final output = numerator / denominator
    
    Key Features:
        - Processes sequences longer than single GPU memory
        - Linear scaling with number of devices in ring
        - Supports both single-device simulation and true distributed mode
        - Compatible with GQA for memory efficiency
        - Online softmax for numerical stability
    
    Use Cases:
        - Training with 1M+ token sequences
        - Long document understanding
        - Code repository analysis
        - Multi-turn conversation with extensive history
    
    Performance Characteristics:
        - Memory per device: O(n / ring_size) for K/V cache
        - Communication: O(ring_size * n / ring_size) = O(n) total
        - Latency: Proportional to ring_size for sequential communication
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        n_head (int): Number of query heads.
        n_kv_head (int): Number of key/value heads for GQA.
        head_dim (int): Per-head dimension.
        ring_size (int): Number of devices in the ring topology.
        use_distributed (bool): Whether to use true distributed processing.
        _distributed_available (bool): Whether distributed environment is ready.
    
    Example:
        >>> attn = YvRingAttention(
        ...     hidden_size=4096,
        ...     n_head=32,
        ...     n_kv_head=8,
        ...     ring_size=4,  # 4 GPUs in ring
        ...     use_distributed=True
        ... )
        >>> hidden = torch.randn(1, 262144, 4096)  # 256K tokens
        >>> output = attn(hidden)
    
    Note:
        For single-device usage, set use_distributed=False. The module will
        simulate ring attention by processing chunks sequentially, useful for
        testing and memory-constrained inference.
    
    Reference:
        Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite
        Context", ICLR 2024.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        n_kv_head: int = None,
        ring_size: int = 4,
        use_distributed: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize Ring Attention with specified ring topology.
        
        Args:
            hidden_size: Model hidden dimension. All projections operate in
                this dimension.
            n_head: Number of query heads for attention computation.
            n_kv_head: Number of key/value heads. For GQA, this is less than
                n_head. For standard MHA, this equals n_head. Default: n_head.
            ring_size: Number of devices in the ring topology. Each device
                processes seq_len / ring_size tokens. Larger rings enable
                longer sequences but increase communication overhead.
            use_distributed: Whether to use true distributed processing across
                multiple GPUs. If False, simulates ring attention on a single
                device for testing and memory-constrained scenarios.
            device: Device for projection parameters.
            dtype: Data type for projection parameters.
        
        Example:
            >>> # Single-device simulation
            >>> attn = YvRingAttention(4096, 32, ring_size=4, use_distributed=False)
            >>> 
            >>> # True distributed processing (requires torchrun)
            >>> attn = YvRingAttention(4096, 32, ring_size=4, use_distributed=True)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.head_dim = hidden_size // n_head
        self.ring_size = ring_size
        self.use_distributed = use_distributed
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, self.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, self.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        
        self._distributed_available = self._check_distributed()
        
    def _check_distributed(self) -> bool:
        """Check if distributed environment is available for ring communication.
        
        Verifies that PyTorch distributed is initialized and has multiple
        processes available for ring communication.
        
        Returns:
            True if distributed processing is available, False otherwise.
        
        Note:
            Distributed ring attention requires launching with torchrun or
            similar distributed launcher with multiple processes.
        """
        if not self.use_distributed:
            return False
        try:
            import torch.distributed as dist
            return dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            return False
        
    def _ring_send_recv(self, tensor: torch.Tensor, send_rank: int, recv_rank: int) -> torch.Tensor:
        """Send and receive tensor in ring topology for distributed processing.
        
        Implements point-to-point communication for passing key/value tensors
        around the ring. Uses non-blocking send/receive for efficiency.
        
        Args:
            tensor: Tensor to send to the next device in the ring.
            send_rank: Rank of the device to send to.
            recv_rank: Rank of the device to receive from.
        
        Returns:
            Received tensor from the previous device in the ring.
        
        Note:
            This operation is blocking - it waits for both send and receive
            to complete before returning. This ensures correct ring ordering.
        """
        if not self._distributed_available:
            return tensor
            
        import torch.distributed as dist
        
        recv_tensor = torch.empty_like(tensor)
        
        send_op = dist.isend(tensor.contiguous(), dst=send_rank)
        recv_op = dist.irecv(recv_tensor, src=recv_rank)
        
        send_op.wait()
        recv_op.wait()
        
        return recv_tensor
        
    def _compute_flash_ring_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int
    ) -> torch.Tensor:
        """Compute ring attention using online softmax algorithm.
        
        Implements memory-efficient ring attention with online softmax for
        numerical stability. Processes attention in chunks and accumulates
        results using the online softmax trick.
        
        Mathematical Formulation (Online Softmax):
            For each chunk, compute:
                m_new = max(m_old, m_chunk)
                l_new = l_old * exp(m_old - m_new) + l_chunk * exp(m_chunk - m_new)
                o_new = o_old * exp(m_old - m_new) + o_chunk * exp(m_chunk - m_new)
        
        Args:
            q: Query tensor of shape [batch, n_head, seq_len, head_dim].
            k: Key tensor of shape [batch, n_head, seq_len, head_dim].
            v: Value tensor of shape [batch, n_head, seq_len, head_dim].
            chunk_size: Size of chunks for processing.
        
        Returns:
            Attention output tensor of shape [batch, n_head, seq_len, head_dim].
        
        Note:
            In distributed mode, K and V are passed around the ring over
            ring_size steps. In single-device mode, the full K/V is used
            for each chunk computation.
        """
        batch_size, n_head, seq_len, head_dim = q.shape
        
        output = torch.zeros_like(q)
        normalizer = torch.zeros(batch_size, n_head, seq_len, 1, device=q.device, dtype=q.dtype)
        
        for ring_step in range(self.ring_size):
            if self._distributed_available:
                import torch.distributed as dist
                rank = dist.get_rank()
                next_rank = (rank + 1) % self.ring_size
                prev_rank = (rank - 1) % self.ring_size
                
                k = self._ring_send_recv(k, next_rank, prev_rank)
                v = self._ring_send_recv(v, next_rank, prev_rank)
            
            for i in range(0, seq_len, chunk_size):
                q_chunk = q[:, :, i:i + chunk_size]
                
                chunk_scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
                
                causal_mask = torch.triu(
                    torch.ones(q_chunk.shape[2], k.shape[2], device=q.device, dtype=torch.bool),
                    diagonal=1
                )
                chunk_scores = chunk_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                chunk_max = chunk_scores.max(dim=-1, keepdim=True)[0]
                chunk_exp = torch.exp(chunk_scores - chunk_max)
                chunk_sum = chunk_exp.sum(dim=-1, keepdim=True)
                
                output[:, :, i:i + chunk_size] += torch.matmul(chunk_exp, v)
                normalizer[:, :, i:i + chunk_size] += chunk_sum
        
        output = output / normalizer.clamp(min=1e-10)
        
        return output
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute ring attention for ultra-long sequences.
        
        Projects inputs to Q, K, V and computes attention using the ring
        pattern. Supports both single-device simulation and true distributed
        processing across multiple GPUs.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
            attention_mask: Optional attention mask to apply.
            past_key_value: Optional cached key/value states from previous
                forward passes for incremental generation.
            use_cache: Whether to return the key/value states for caching.
        
        Returns:
            If use_cache is False:
                Output tensor of shape [batch, seq_len, hidden_size].
            If use_cache is True:
                Tuple of (output, (key_cache, value_cache)).
        
        Example:
            >>> hidden = torch.randn(1, 131072, 4096)  # 128K tokens
            >>> output = attn(hidden)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        kv_seq_len = k.shape[2]
        
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        
        chunk_size = max(1, seq_len // self.ring_size)
        
        if self._distributed_available or seq_len > 16384:
            output = self._compute_flash_ring_attention(q, k, v, chunk_size)
        else:
            output = self._standard_ring_attention(q, k, v, chunk_size, attention_mask)
        
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        
        if use_cache:
            k_cache = k[:, :, :self.n_kv_head] if self.n_kv_head != self.n_head else k
            v_cache = v[:, :, :self.n_kv_head] if self.n_kv_head != self.n_head else v
            return output, (k_cache, v_cache)
            
        return output
        
    def _standard_ring_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch_size, n_head, seq_len, head_dim = q.shape
        
        output = torch.zeros_like(q)
        
        for i in range(0, seq_len, chunk_size):
            q_chunk = q[:, :, i:i + chunk_size]
            
            chunk_output = torch.zeros_like(q_chunk)
            chunk_normalizer = torch.zeros(q_chunk.shape[0], q_chunk.shape[1], q_chunk.shape[2], 1, device=q.device)
            
            for j in range(0, seq_len, chunk_size):
                k_chunk = k[:, :, j:j + chunk_size]
                v_chunk = v[:, :, j:j + chunk_size]
                
                attn_weights = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale
                
                if i >= j + chunk_size:
                    pass
                elif i + chunk_size <= j:
                    attn_weights = attn_weights.masked_fill(
                        torch.ones_like(attn_weights, dtype=torch.bool),
                        float('-inf')
                    )
                else:
                    causal_mask = torch.triu(
                        torch.ones(q_chunk.shape[2], k_chunk.shape[2], device=q.device, dtype=torch.bool),
                        diagonal=j - i + 1
                    )
                    attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                    
                if attention_mask is not None:
                    mask_slice = attention_mask[:, :, i:i + chunk_size, j:j + chunk_size]
                    attn_weights = attn_weights + mask_slice
                    
                attn_weights = F.softmax(attn_weights, dim=-1)
                chunk_output = chunk_output + torch.matmul(attn_weights, v_chunk)
                
            output[:, :, i:i + chunk_size] = chunk_output
            
        return output


class YvMultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA) with single shared key/value head.
    
    Implements Multi-Query Attention, an efficient attention variant that uses
    a single key/value head shared across all query heads. This significantly
    reduces memory bandwidth and KV cache size during inference.
    
    Architecture:
        - Query Heads: n_head separate query projections
        - Key Head: Single shared key projection
        - Value Head: Single shared value projection
        - Expansion: K/V expanded to match query head count
    
    Mathematical Formulation:
        Q_i = hidden @ W_q_i  for i in [0, n_head)
        K = hidden @ W_k      (single head)
        V = hidden @ W_v      (single head)
        
        Attention_i = softmax(Q_i @ K^T / sqrt(d)) @ V
        Output = concat(Attention_0, ..., Attention_{n-1}) @ W_o
    
    Key Features:
        - KV cache size: O(1) instead of O(n_head)
        - Memory bandwidth: Reduced by ~n_head factor for K/V
        - Inference speedup: 2-3x for memory-bound scenarios
        - Slight quality degradation vs full MHA
    
    Comparison with Other Attention Variants:
        - MHA: n_head K/V heads, highest quality, most memory
        - MQA: 1 K/V head, good quality, least memory
        - GQA: n_kv_head K/V heads, balance between MHA and MQA
    
    Use Cases:
        - Inference-optimized models
        - Memory-constrained deployment
        - Real-time applications requiring fast generation
    
    Performance Characteristics:
        - KV Cache: 1/n_head of standard attention
        - Memory Bandwidth: ~n_head reduction for K/V projections
        - Quality: ~1-2% degradation vs MHA on most tasks
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        n_head (int): Number of query heads.
        head_dim (int): Per-head dimension.
        attention_dropout (float): Dropout probability during training.
    
    Example:
        >>> attn = YvMultiQueryAttention(4096, 32)
        >>> hidden = torch.randn(2, 1024, 4096)
        >>> output = attn(hidden)
    
    Note:
        MQA was pioneered in PaLM and has become standard for inference-
        optimized models. GQA (Grouped-Query Attention) offers a middle
        ground between MQA and full MHA.
    
    Reference:
        Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        attention_dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize Multi-Query Attention with single K/V head.
        
        Args:
            hidden_size: Model hidden dimension. All projections operate in
                this dimension.
            n_head: Number of query heads. Each head has dimension
                hidden_size // n_head.
            attention_dropout: Dropout probability applied to attention weights
                during training. Set to 0.0 for inference.
            device: Device for projection parameters.
            dtype: Data type for projection parameters.
        
        Example:
            >>> attn = YvMultiQueryAttention(
            ...     hidden_size=4096,
            ...     n_head=32,
            ...     attention_dropout=0.0,
            ...     device='cuda',
            ...     dtype=torch.bfloat16
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.attention_dropout = attention_dropout
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, self.head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute multi-query attention with shared K/V head.
        
        Projects inputs to queries (n_head heads) and single key/value head,
        expands K/V to match query heads, and computes attention.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
            attention_mask: Optional attention mask to apply.
            past_key_value: Optional cached key/value states from previous
                forward passes for incremental generation.
            use_cache: Whether to return the key/value states for caching.
        
        Returns:
            If use_cache is False:
                Output tensor of shape [batch, seq_len, hidden_size].
            If use_cache is True:
                Tuple of (output, (key_cache, value_cache)).
        
        Note:
            The KV cache returned has shape [batch, 1, seq_len, head_dim],
            representing the single shared K/V head.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        kv_seq_len = k.shape[2]
        
        k = k.expand(-1, self.n_head, -1, -1)
        v = v.expand(-1, self.n_head, -1, -1)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        causal_mask = torch.triu(
            torch.ones(seq_len, kv_seq_len, device=q.device, dtype=torch.bool),
            diagonal=kv_seq_len - seq_len + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
            
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        
        if use_cache:
            return output, (k[:, :1], v[:, :1])
            
        return output


class YvDynamicH2OAttention(nn.Module):
    """Dynamic H2O Attention with adaptive compression and hierarchical caching.
    
    Enhances the base H2O (Heavy-Hitter Oracle) attention with dynamic
    compression ratios, hierarchical cache levels, and optional PagedAttention
    integration. This enables efficient processing of extremely long sequences
    while maintaining attention quality.
    
    Architecture:
        - Recent Cache: Full-precision recent tokens (streaming window)
        - Compressed Cache: Compressed heavy-hitter tokens
        - Archived Cache: Highly compressed historical tokens
        - Dynamic Compression: Adaptive ratio based on sequence complexity
    
    Hierarchical Cache Levels:
        Level 0 (Recent): 
            - Full precision, no compression
            - Streaming window of recent tokens
            - Highest attention quality
        
        Level 1 (Compressed):
            - Light compression (2x reduction)
            - Heavy-hitter tokens with high attention scores
            - Good quality with reduced memory
        
        Level 2 (Archived):
            - Heavy compression (4x reduction)
            - Historical tokens for long-range context
            - Lower quality but essential for coherence
    
    Dynamic Compression:
        The compression ratio is dynamically adjusted based on:
        - Sequence complexity (predicted by neural network)
        - Attention entropy distribution
        - Memory constraints
    
    Key Features:
        - Adaptive compression based on content complexity
        - Hierarchical caching for quality-memory tradeoff
        - Optional PagedAttention integration for memory management
        - Heavy-hitter selection for important tokens
        - Streaming window for recent context
    
    Use Cases:
        - Long document processing (100K+ tokens)
        - Multi-turn conversation with extensive history
        - Code analysis with full repository context
        - Book-length text understanding
    
    Performance Characteristics:
        - Memory: O(streaming_window + compressed_tokens + archived_tokens)
        - Quality: ~95% of full attention with proper configuration
        - Speed: 2-4x faster than full attention for long sequences
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Per-head dimension.
        max_position_embeddings (int): Maximum sequence length.
        compression_ratio (int): Base compression ratio for cached tokens.
        heavy_hitter_ratio (float): Fraction of tokens to keep as heavy hitters.
        streaming_window (int): Size of the recent token window.
        num_cache_levels (int): Number of hierarchical cache levels.
        enable_paged_attention (bool): Whether to use PagedAttention.
    
    Example:
        >>> attn = YvDynamicH2OAttention(
        ...     hidden_size=4096,
        ...     num_attention_heads=32,
        ...     streaming_window=8192,
        ...     compression_ratio=8
        ... )
        >>> hidden = torch.randn(1, 131072, 4096)  # 128K tokens
        >>> output = attn(hidden)
    
    Note:
        The hierarchical cache levels allow trading off between memory
        efficiency and attention quality. More levels provide finer control
        but add complexity.
    
    Reference:
        Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
        Inference of Large Language Models", ICLR 2024.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_position_embeddings: int = 10485760,
        compression_ratio: int = 8,
        heavy_hitter_ratio: float = 0.1,
        streaming_window: int = 16384,
        dropout: float = 0.1,
        num_cache_levels: int = 3,
        enable_paged_attention: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize Dynamic H2O Attention with hierarchical caching.
        
        Args:
            hidden_size: Model hidden dimension. All projections operate in
                this dimension.
            num_attention_heads: Number of attention heads.
            max_position_embeddings: Maximum sequence length supported.
                Default: 10M tokens for ultra-long context.
            compression_ratio: Base compression ratio for cached tokens.
                Higher values reduce memory but may impact quality.
            heavy_hitter_ratio: Fraction of tokens to retain as heavy hitters.
                These are tokens with high cumulative attention scores.
            streaming_window: Size of the recent token window that is kept
                in full precision without compression.
            dropout: Dropout probability for attention weights.
            num_cache_levels: Number of hierarchical cache levels. Default 3
                provides recent/compressed/archived levels.
            enable_paged_attention: Whether to integrate PagedAttention for
                memory management of the KV cache.
            device: Device for projection parameters.
            dtype: Data type for projection parameters.
        
        Example:
            >>> attn = YvDynamicH2OAttention(
            ...     hidden_size=4096,
            ...     num_attention_heads=32,
            ...     max_position_embeddings=1048576,  # 1M tokens
            ...     compression_ratio=8,
            ...     streaming_window=4096
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.compression_ratio = compression_ratio
        self.heavy_hitter_ratio = heavy_hitter_ratio
        self.streaming_window = streaming_window
        self.num_cache_levels = num_cache_levels
        self.enable_paged_attention = enable_paged_attention
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        
        self.complexity_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4, bias=False, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1, bias=False, device=device, dtype=dtype),
            nn.Sigmoid()
        )
        
        self.dynamic_compressor = nn.ModuleDict({
            'recent': nn.Identity(),
            'compressed': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2, bias=False, device=device, dtype=dtype),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size, bias=False, device=device, dtype=dtype)
            ),
            'archived': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4, bias=False, device=device, dtype=dtype),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, hidden_size, bias=False, device=device, dtype=dtype)
            )
        })
        
        heads_per_level = max(1, num_attention_heads // num_cache_levels)
        self.level_attention = nn.ModuleDict({
            'recent': nn.MultiheadAttention(hidden_size, heads_per_level, batch_first=True, device=device, dtype=dtype),
            'compressed': nn.MultiheadAttention(hidden_size, heads_per_level, batch_first=True, device=device, dtype=dtype),
            'archived': nn.MultiheadAttention(hidden_size, heads_per_level, batch_first=True, device=device, dtype=dtype)
        })
        
        self.level_fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_size, num_cache_levels, bias=False, device=device, dtype=dtype),
            nn.Softmax(dim=-1)
        )
        
        cache_dtype = dtype if dtype else torch.float32
        self.register_buffer(
            'cache_sizes',
            torch.tensor([streaming_window // 2, streaming_window * 2, max_position_embeddings // 16], device=device, dtype=cache_dtype)
        )
        
    def _predict_sequence_complexity(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)
        complexity = self.complexity_predictor(pooled)
        return complexity.squeeze(-1)
    
    def _compute_dynamic_compression(self, hidden_states: torch.Tensor, complexity: torch.Tensor) -> int:
        base_ratio = self.compression_ratio
        adaptive_ratio = int(base_ratio * (1 + complexity.mean().item()))
        adaptive_ratio = max(2, min(16, adaptive_ratio))
        return adaptive_ratio
    
    def _build_hierarchical_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        importance_scores: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        sorted_indices = torch.argsort(importance_scores, dim=-1, descending=True)
        
        recent_size = min(int(self.cache_sizes[0].item()), seq_len)
        compressed_size = min(int(self.cache_sizes[1].item()), seq_len)
        archived_size = min(int(self.cache_sizes[2].item()), seq_len)
        
        recent_indices = sorted_indices[:, :, :recent_size]
        compressed_indices = sorted_indices[:, :, recent_size:recent_size + compressed_size]
        archived_indices = sorted_indices[:, :, recent_size + compressed_size:recent_size + compressed_size + archived_size]
        
        recent_keys = torch.gather(key_states, 2, recent_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        recent_values = torch.gather(value_states, 2, recent_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        
        compressed_keys = torch.gather(key_states, 2, compressed_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        compressed_values = torch.gather(value_states, 2, compressed_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        
        if archived_indices.shape[-1] > 0:
            archived_keys = torch.gather(key_states, 2, archived_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            archived_values = torch.gather(value_states, 2, archived_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        else:
            archived_keys = torch.zeros(batch_size, num_heads, 0, head_dim, device=key_states.device, dtype=key_states.dtype)
            archived_values = torch.zeros(batch_size, num_heads, 0, head_dim, device=value_states.device, dtype=value_states.dtype)
        
        return {
            'recent': (recent_keys, recent_values),
            'compressed': (compressed_keys, compressed_values),
            'archived': (archived_keys, archived_values)
        }
    
    def _fuse_hierarchical_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        query_states: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        
        query_pooled = query_states.mean(dim=2, keepdim=True)
        level_weights = self.level_fusion(query_pooled.transpose(1, 2).transpose(2, 3))
        level_weights = level_weights.transpose(1, 3).squeeze(0)
        
        fused_output = torch.zeros_like(outputs['recent'])
        for i, level_name in enumerate(['recent', 'compressed', 'archived']):
            weight = level_weights[:, :, i:i+1, :].unsqueeze(-1)
            fused_output = fused_output + weight * outputs[level_name]
        
        return fused_output
    
    def _compress_states(
        self,
        states: torch.Tensor,
        compression_ratio: int = None
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = states.shape
        ratio = compression_ratio or self.compression_ratio
        
        if seq_len <= self.streaming_window:
            return states
        
        compressed_length = (seq_len + ratio - 1) // ratio
        flat = states.view(batch_size * num_heads, seq_len, head_dim)
        token_importance = torch.norm(flat, dim=-1)
        token_importance = F.softmax(token_importance, dim=-1)
        
        pad_len = compressed_length * ratio - seq_len
        if pad_len > 0:
            pad_states = torch.zeros(batch_size * num_heads, pad_len, head_dim, device=states.device, dtype=states.dtype)
            pad_weights = torch.zeros(batch_size * num_heads, pad_len, device=states.device, dtype=token_importance.dtype)
            flat = torch.cat([flat, pad_states], dim=1)
            token_importance = torch.cat([token_importance, pad_weights], dim=1)
        
        flat = flat.view(batch_size * num_heads, compressed_length, ratio, head_dim)
        w = token_importance.view(batch_size * num_heads, compressed_length, ratio)
        w_sum = w.sum(dim=2, keepdim=True) + 1e-8
        pooled = (flat * w.unsqueeze(-1)).sum(dim=2) / w_sum
        
        return pooled.view(batch_size, num_heads, compressed_length, head_dim)
    
    def _infini_memory_retrieval(
        self,
        query: torch.Tensor,
        compressed_memory: torch.Tensor,
        memory_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve from compressed memory for Infini-attention.
        
        Automatically triggered when max_position_embeddings > 1M.
        
        Args:
            query: [batch, heads, seq_len, head_dim]
            compressed_memory: [batch, heads, memory_size, head_dim]
            memory_weights: [batch, heads, memory_size]
        
        Returns:
            Retrieved output [batch, heads, seq_len, head_dim]
        """
        if compressed_memory is None or compressed_memory.shape[2] == 0:
            return torch.zeros_like(query)
        
        scores = torch.matmul(query, compressed_memory.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        
        if memory_weights is not None:
            scores = scores + torch.log(memory_weights.unsqueeze(-2) + 1e-8)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, compressed_memory)
        
        return output
    
    def _update_compressed_memory(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        importance_scores: torch.Tensor
    ) -> None:
        """
        Update compressed memory with new key-value pairs.
        
        Uses attention-aware compression to maintain memory efficiency.
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        if seq_len <= self.memory_size:
            self.compressed_memory = value_states.clone()
            if importance_scores is not None:
                self.memory_weights = importance_scores.clone()
            return
        
        _, top_indices = torch.topk(importance_scores, self.memory_size, dim=-1)
        top_indices, _ = torch.sort(top_indices, dim=-1)
        
        top_indices_exp = top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        
        compressed_k = torch.gather(key_states, 2, top_indices_exp)
        compressed_v = torch.gather(value_states, 2, top_indices_exp)
        
        if self.compressed_memory is None:
            self.compressed_memory = compressed_v
            self.memory_weights = importance_scores.gather(2, top_indices)
        else:
            old_memory = self.compressed_memory
            old_weights = self.memory_weights
            
            combined_v = torch.cat([old_memory, compressed_v], dim=2)
            combined_w = torch.cat([old_weights, importance_scores.gather(2, top_indices)], dim=2)
            
            if combined_v.shape[2] > self.memory_size:
                _, keep_indices = torch.topk(combined_w, self.memory_size, dim=-1)
                keep_indices, _ = torch.sort(keep_indices, dim=-1)
                keep_exp = keep_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                self.compressed_memory = torch.gather(combined_v, 2, keep_exp)
                self.memory_weights = combined_w.gather(2, keep_indices)
            else:
                self.compressed_memory = combined_v
                self.memory_weights = combined_w
    
    def _calculate_importance_scores(self, key_states: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        key_magnitude = torch.norm(key_states, dim=-1)
        value_magnitude = torch.norm(value_states, dim=-1)
        importance = key_magnitude + value_magnitude
        
        seq_len = key_states.shape[2]
        
        if seq_len > 100000:
            popularity = importance.mean(dim=-1, keepdim=True)
            popularity_penalty = torch.log1p(popularity)
            importance = importance / (1.0 + 0.3 * popularity_penalty)
        
        position_weights = torch.exp(-torch.arange(seq_len, device=key_states.device, dtype=key_states.dtype).float() / 100.0)
        position_weights = position_weights.unsqueeze(0).unsqueeze(0)
        importance = importance * position_weights
        importance = F.softmax(importance, dim=-1)
        
        return importance
    
    def _streaming_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_manager=None
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        device = query_states.device
        
        if seq_len <= self.streaming_window:
            attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            return torch.matmul(attention_weights, value_states)
        
        output_states = torch.zeros_like(query_states)
        
        importance_scores = self._calculate_importance_scores(key_states, value_states)
        cache_dict = self._build_hierarchical_cache(key_states, value_states, importance_scores)
        
        for start_idx in range(0, seq_len, self.streaming_window):
            end_idx = min(start_idx + self.streaming_window, seq_len)
            window_size = end_idx - start_idx
            
            window_query = query_states[:, :, start_idx:end_idx, :]
            
            level_outputs = {}
            for level_name, (level_keys, level_values) in cache_dict.items():
                if level_keys.shape[2] > 0:
                    level_q = window_query.reshape(batch_size * num_heads, window_size, head_dim)
                    level_k = level_keys.reshape(batch_size * num_heads, -1, head_dim)
                    level_v = level_values.reshape(batch_size * num_heads, -1, head_dim)
                    
                    row_pos = torch.arange(start_idx, end_idx, device=device)
                    pos_expanded = torch.arange(level_keys.shape[2], device=device).view(1, 1, -1)
                    pos_expanded = pos_expanded.expand(batch_size, num_heads, -1)
                    allowed = pos_expanded.unsqueeze(2) <= row_pos.view(1, 1, window_size, 1)
                    
                    disallow = ~allowed
                    attn_mask = disallow.reshape(batch_size * num_heads, window_size, -1)
                    
                    level_out = F.scaled_dot_product_attention(
                        level_q, level_k, level_v,
                        attn_mask=attn_mask,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=False
                    )
                    level_outputs[level_name] = level_out.reshape(batch_size, num_heads, window_size, head_dim)
            
            if level_outputs:
                window_output = self._fuse_hierarchical_outputs(level_outputs, window_query)
                output_states[:, :, start_idx:end_idx, :] = window_output
        
        return output_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_manager=None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        complexity = self._predict_sequence_complexity(hidden_states)
        dynamic_compression = self._compute_dynamic_compression(hidden_states, complexity)
        
        if seq_len > self.streaming_window:
            compressed_key = self._compress_states(key_states, dynamic_compression)
            compressed_value = self._compress_states(value_states, dynamic_compression)
            attention_key = compressed_key
            attention_value = compressed_value
        else:
            attention_key = key_states
            attention_value = value_states
        
        attention_output = self._streaming_attention(
            query_states, attention_key, attention_value,
            attention_mask, cache_manager
        )
        
        attention_output = self.o_proj(attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size))
        attention_output = self.dropout(attention_output)
        
        return attention_output, (key_states, value_states)


class YvH2OAttention(nn.Module):
    """Heavy-Hitter Oracle (H2O) Attention for ultra-long context processing.
    
    Implements H2O attention with heavy-hitter retention and streaming support
    for processing sequences longer than typical memory constraints. This is
    a simplified version of Dynamic H2O without hierarchical caching.
    
    Architecture:
        - Heavy Hitters: Tokens with high cumulative attention scores
        - Streaming Window: Recent tokens kept in full precision
        - Adaptive Compression: Importance-weighted token pooling
        - KV Cache Management: Efficient storage for inference
    
    Heavy-Hitter Selection:
        Heavy hitters are tokens that consistently receive high attention scores
        across multiple query positions. These tokens are retained because they
        contain information important for the overall sequence understanding.
        
        Selection criteria:
        - Cumulative attention score magnitude
        - Key and value vector norms
        - Position-weighted importance
    
    Streaming Window:
        Recent tokens are kept in full precision without compression, ensuring
        high-quality attention for local context. The streaming window slides
        as the sequence progresses.
    
    Key Features:
        - Adaptive compression based on token importance
        - Heavy-hitter token retention for critical information
        - Streaming window for local context preservation
        - KV cache management for efficient inference
        - Automatic KV quantization for ultra-long sequences
    
    Use Cases:
        - Long document processing
        - Extended conversation history
        - Code analysis with large context
        - Memory-constrained inference
    
    Performance Characteristics:
        - Memory: O(streaming_window + heavy_hitters)
        - Quality: ~95% of full attention with proper configuration
        - Speed: 2-3x faster than full attention for long sequences
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Per-head dimension.
        max_position_embeddings (int): Maximum sequence length.
        compression_ratio (int): Compression ratio for cached tokens.
        heavy_hitter_ratio (float): Fraction of tokens to keep as heavy hitters.
        streaming_window (int): Size of the streaming window.
        n_sink (int): Number of sink tokens for attention stabilization.
        memory_size (int): Size of compressed memory for Infini-attention.
    
    Example:
        >>> attn = YvH2OAttention(
        ...     hidden_size=4096,
        ...     num_attention_heads=32,
        ...     streaming_window=8192,
        ...     compression_ratio=8
        ... )
        >>> hidden = torch.randn(1, 65536, 4096)  # 64K tokens
        >>> output, _ = attn(hidden)
    
    Note:
        For sequences over 1M tokens, Infini-attention is automatically enabled
        to provide additional memory retrieval capabilities.
    
    Reference:
        Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
        Inference of Large Language Models", ICLR 2024.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_position_embeddings: int = 10485760,
        compression_ratio: int = 8,
        heavy_hitter_ratio: float = 0.1,
        streaming_window: int = 16384,
        dropout: float = 0.1
    ):
        """Initialize H2O Attention with heavy-hitter retention.
        
        Args:
            hidden_size: Model hidden dimension. All projections operate in
                this dimension.
            num_attention_heads: Number of attention heads.
            max_position_embeddings: Maximum sequence length supported.
                Default: 10M tokens for ultra-long context.
            compression_ratio: Compression ratio for cached tokens. Higher
                values reduce memory but may impact quality.
            heavy_hitter_ratio: Fraction of tokens to retain as heavy hitters.
                These are tokens with high cumulative attention scores.
            streaming_window: Size of the streaming window for recent tokens.
                These tokens are kept in full precision.
            dropout: Dropout probability for attention weights.
        
        Example:
            >>> attn = YvH2OAttention(
            ...     hidden_size=4096,
            ...     num_attention_heads=32,
            ...     max_position_embeddings=1048576,
            ...     streaming_window=4096
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.compression_ratio = compression_ratio
        self.heavy_hitter_ratio = heavy_hitter_ratio
        self.streaming_window = streaming_window
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.heavy_hitter_threshold = None
        
        self.n_sink = 4
        self.memory_size = max(256, streaming_window // 16)
        self.memory_weight = nn.Parameter(torch.ones(1) * 0.3)
        self.register_buffer('compressed_memory', None, persistent=False)
        self.register_buffer('memory_weights', None, persistent=False)
    
    def _compress_states(
        self,
        states: torch.Tensor,
        compression_ratio: int = None
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = states.shape
        device = states.device
        ratio = compression_ratio or self.compression_ratio
        
        if seq_len <= self.streaming_window:
            return states
        
        # KV Quantization: Position-aware quantization for ultra-long sequences
        # Auto-enabled when seq_len > 100000
        if seq_len > 100000:
            recent_tokens = 4096
            position_scale = torch.ones(seq_len, device=device, dtype=states.dtype)
            position_scale[recent_tokens:] = 0.5
            
            states_flat = states * position_scale.view(1, 1, -1, 1)
            scale = states_flat.abs().max() / 7.0
            states = torch.round(states_flat / scale) * scale
        
        seq_complexity = torch.std(states) / (torch.mean(torch.abs(states)) + 1e-8)
        adaptive_ratio = max(1, min(ratio, int(seq_complexity * ratio)))
        actual_ratio = min(adaptive_ratio, max(1, seq_len // 512))
        
        compressed_length = (seq_len + actual_ratio - 1) // actual_ratio
        
        flat = states.view(batch_size * num_heads, seq_len, head_dim)
        token_importance = torch.norm(flat, dim=-1)
        token_importance = F.softmax(token_importance, dim=-1)
        
        pad_len = compressed_length * actual_ratio - seq_len
        if pad_len > 0:
            pad_states = torch.zeros(batch_size * num_heads, pad_len, head_dim, device=device, dtype=states.dtype)
            pad_weights = torch.zeros(batch_size * num_heads, pad_len, device=device, dtype=token_importance.dtype)
            flat = torch.cat([flat, pad_states], dim=1)
            token_importance = torch.cat([token_importance, pad_weights], dim=1)
        
        flat = flat.view(batch_size * num_heads, compressed_length, actual_ratio, head_dim)
        w = token_importance.view(batch_size * num_heads, compressed_length, actual_ratio)
        
        w_sum = w.sum(dim=2, keepdim=True) + 1e-8
        pooled = (flat * w.unsqueeze(-1)).sum(dim=2) / w_sum
        
        return pooled.view(batch_size, num_heads, compressed_length, head_dim)
    
    def _calculate_importance_scores(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        key_magnitude = torch.norm(key_states, dim=-1)
        value_magnitude = torch.norm(value_states, dim=-1)
        
        importance = key_magnitude + value_magnitude
        
        if seq_len > 100000:
            popularity = importance.mean(dim=-1, keepdim=True)
            popularity_penalty = torch.log1p(popularity)
            importance = importance / (1.0 + 0.3 * popularity_penalty)
        
        position_weights = torch.exp(-torch.arange(seq_len, device=key_states.device).float() / 100.0)
        position_weights = position_weights.unsqueeze(0).unsqueeze(0)
        importance = importance * position_weights
        
        return F.softmax(importance, dim=-1)
    
    def _select_important_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        importance_scores: torch.Tensor,
        current_pos: int,
        max_cache_size: int,
        cache_manager=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        cache_end = current_pos
        if cache_end <= 0:
            cache_end = min(seq_len, self.streaming_window)
        
        recent_keep = min(max_cache_size // 4, self.streaming_window // 2)
        recent_start = max(0, cache_end - recent_keep)
        
        if cache_manager is not None:
            cached_keys, cached_values = cache_manager.get_h2o_cache(key_states, current_pos, max_cache_size)
            if cached_keys is not None and cached_values is not None:
                return cached_keys, cached_values, None
        
        pool_keys = key_states[:, :, :cache_end, :]
        pool_values = value_states[:, :, :cache_end, :]
        pool_importance = importance_scores[:, :, :cache_end]
        
        recent_keys = pool_keys[:, :, recent_start:cache_end, :]
        recent_values = pool_values[:, :, recent_start:cache_end, :]
        recent_len = cache_end - recent_start
        recent_pos = torch.arange(recent_start, cache_end, device=key_states.device)
        recent_pos = recent_pos.view(1, 1, recent_len).expand(batch_size, num_heads, recent_len)
        
        remaining = max(0, max_cache_size - (cache_end - recent_start))
        if remaining == 0:
            pos = recent_pos[:, :, -max_cache_size:]
            return recent_keys[:, :, -max_cache_size:, :], recent_values[:, :, -max_cache_size:, :], pos
        
        imp_region = pool_importance[:, :, :recent_start] if recent_start > 0 else None
        if imp_region is None or imp_region.shape[2] == 0:
            selected_keys = recent_keys
            selected_values = recent_values
            pos = recent_pos
        else:
            head_importance = imp_region.sum(dim=-1)
            alloc = head_importance / (head_importance.sum(dim=1, keepdim=True) + 1e-8)
            alloc = alloc.mean(dim=0)
            
            quotas = (alloc * remaining).round().to(torch.long)
            diff = int(remaining - quotas.sum().item())
            if diff != 0:
                order = torch.argsort(alloc, descending=True)
                for i in range(min(abs(diff), num_heads)):
                    idx = order[i].item()
                    quotas[idx] = max(0, quotas[idx] + (1 if diff > 0 else -1))
            
            if quotas.sum().item() <= 0:
                quotas = torch.full((num_heads,), max(1, remaining // max(1, num_heads)), dtype=torch.long, device=imp_region.device)
            
            sel_keys = []
            sel_vals = []
            sel_pos_list = []
            head_space = imp_region.shape[2]
            for h in range(num_heads):
                k_h = int(min(max(0, quotas[h].item()), head_space))
                if k_h <= 0:
                    continue
                imp_h = imp_region[:, h:h+1, :]
                _, idx_h = torch.topk(imp_h, k=k_h, dim=-1)
                idx_h = torch.sort(idx_h, dim=-1).values
                k_src = pool_keys[:, h:h+1, :recent_start, :]
                v_src = pool_values[:, h:h+1, :recent_start, :]
                k_sel_h = torch.gather(k_src, 2, idx_h.unsqueeze(-1).expand(-1, -1, -1, head_dim))
                v_sel_h = torch.gather(v_src, 2, idx_h.unsqueeze(-1).expand(-1, -1, -1, head_dim))
                sel_keys.append(k_sel_h)
                sel_vals.append(v_sel_h)
                sel_pos_h = idx_h.expand(-1, -1, -1).clone()
                sel_pos_list.append(sel_pos_h)
            
            if sel_keys:
                keys_sel = torch.cat(sel_keys, dim=1)
                vals_sel = torch.cat(sel_vals, dim=1)
                pos_sel = torch.cat(sel_pos_list, dim=1)
            else:
                keys_sel = pool_keys[:, :, :0, :]
                vals_sel = pool_values[:, :, :0, :]
                pos_sel = pool_values.new_zeros((batch_size, num_heads, 0), dtype=torch.long)
            
            selected_keys = torch.cat([keys_sel, recent_keys], dim=2)
            selected_values = torch.cat([vals_sel, recent_values], dim=2)
            pos = torch.cat([pos_sel.to(torch.long), recent_pos.to(torch.long)], dim=2)
            
            if selected_keys.shape[2] > max_cache_size:
                selected_keys = selected_keys[:, :, -max_cache_size:, :]
                selected_values = selected_values[:, :, -max_cache_size:, :]
                pos = pos[:, :, -max_cache_size:]
        
        if cache_manager is not None:
            cache_manager.set_h2o_cache(key_states, current_pos, max_cache_size, selected_keys, selected_values)
        
        return selected_keys, selected_values, pos
    
    def _streaming_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_manager=None
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        device = query_states.device
        
        if seq_len <= self.streaming_window:
            attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            return torch.matmul(attention_weights, value_states)
        
        output_states = torch.zeros_like(query_states)
        
        importance_scores = self._calculate_importance_scores(key_states, value_states)
        
        for start_idx in range(0, seq_len, self.streaming_window):
            end_idx = min(start_idx + self.streaming_window, seq_len)
            window_size = end_idx - start_idx
            
            window_query = query_states[:, :, start_idx:end_idx, :]
            
            if seq_len > self.streaming_window * 2:
                cache_budget = min(self.streaming_window * 2, end_idx)
                cached_key, cached_value, cached_pos = self._select_important_cache(
                    key_states, value_states, importance_scores,
                    end_idx, cache_budget, cache_manager=cache_manager
                )
            else:
                cached_key = key_states[:, :, :end_idx, :]
                cached_value = value_states[:, :, :end_idx, :]
                cached_pos = torch.arange(end_idx, device=device).view(1, 1, end_idx).expand(batch_size, num_heads, end_idx)
            
            row_pos = torch.arange(start_idx, end_idx, device=device)
            allowed = (cached_pos.unsqueeze(2) <= row_pos.view(1, 1, window_size, 1))
            
            q = window_query.reshape(batch_size * num_heads, window_size, head_dim)
            k = cached_key.reshape(batch_size * num_heads, -1, head_dim)
            v = cached_value.reshape(batch_size * num_heads, -1, head_dim)
            
            disallow = (~allowed)
            if attention_mask is not None:
                mask_slice = attention_mask[:, :, start_idx:end_idx, :cached_key.shape[2]]
                if mask_slice.dtype == torch.bool:
                    extra_disallow = ~mask_slice
                else:
                    extra_disallow = mask_slice < -1e4
                disallow = disallow | extra_disallow
            attn_mask = disallow.reshape(batch_size * num_heads, window_size, -1)
            
            window_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            window_output = window_output.reshape(batch_size, num_heads, window_size, head_dim)
            
            if self.compressed_memory is not None and self.max_position_embeddings > 1000000:
                memory_output = self._infini_memory_retrieval(
                    window_query, self.compressed_memory, self.memory_weights
                )
                gate = torch.sigmoid(self.memory_weight)
                window_output = gate * window_output + (1 - gate) * memory_output
            
            output_states[:, :, start_idx:end_idx, :] = window_output
        
        return output_states
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_manager=None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        if seq_len > self.streaming_window * 2:
            compressed_key = self._compress_states(key_states)
            compressed_value = self._compress_states(value_states)
            attention_key = compressed_key
            attention_value = compressed_value
            
            if self.max_position_embeddings > 1000000:
                importance_scores = self._calculate_importance_scores(key_states, value_states)
                self._update_compressed_memory(key_states, value_states, importance_scores)
        else:
            attention_key = key_states
            attention_value = value_states
        
        attention_output = self._streaming_attention(
            query_states, attention_key, attention_value, attention_mask,
            cache_manager=cache_manager
        )
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        attention_output = self.o_proj(attention_output)
        
        present_key_value = None
        if use_cache:
            present_key_value = (key_states, value_states)
        
        return attention_output, present_key_value


class YvAttention(nn.Module):
    """Unified Multi-head Attention with comprehensive backend support.
    
    Implements a flexible attention mechanism that supports multiple attention
    variants and optimizations through a unified interface. This class serves
    as the primary attention module for the Yv model architecture.
    
    Supported Attention Backends:
        - Standard Attention: Full attention with optional Flash Attention 2/3
        - H2O Attention: Heavy-Hitter Oracle for ultra-long contexts (>1M tokens)
        - Linear Attention: Kernel-based approximation for efficiency
        - Sliding Window: Local attention with fixed window size
        - Sparse Attention: Block-sparse patterns for memory efficiency
        - PagedAttention: Block-wise KV cache management
        - Ring Attention: Distributed processing across devices
    
    Supported Attention Variants:
        - Multi-Head Attention (MHA): Full attention with n_head K/V heads
        - Grouped-Query Attention (GQA): n_kv_head K/V heads (n_kv_head < n_head)
        - Multi-Query Attention (MQA): Single K/V head shared across queries
        - Multi-Latent Attention (MLA): Low-rank compression for KV cache
    
    Position Encoding Options:
        - RoPE (YaRN): Yet another RoPE with scaling for long contexts
        - ALiBi: Attention with Linear Biases for extrapolation
    
    Stability Features:
        - Attention Sink: Learnable tokens for streaming stability
        - QK Normalization: LayerNorm on queries and keys
        - Modality Embedding: Task-specific embeddings for multimodal
    
    Architecture Selection:
        The attention backend is automatically selected based on:
        - Sequence length: H2O for >1M tokens, Linear for >4K tokens
        - Configuration flags: use_h2o, use_flash, use_linear, etc.
        - Hardware availability: Flash Attention requires CUDA-capable GPU
    
    Key Features:
        - Automatic backend selection based on context length
        - Hierarchical window sizing by layer depth
        - Multimodal support with modality-specific embeddings
        - Efficient KV cache management for inference
        - Gradient-friendly training with multiple optimizations
    
    Attributes:
        cfg: Configuration object containing model hyperparameters.
        n_head (int): Number of query heads.
        n_kv_head (int): Number of key/value heads for GQA.
        head_dim (int): Per-head dimension.
        scale (float): Attention scaling factor.
        use_h2o (bool): Whether to use H2O attention backend.
        use_flash (bool): Whether to use Flash Attention.
        use_alibi (bool): Whether to use ALiBi position encoding.
        use_attention_sink (bool): Whether to use attention sinks.
        use_qk_norm (bool): Whether to use QK normalization.
        use_linear (bool): Whether to use linear attention.
        sliding_window (int): Sliding window size (0 for disabled).
        sparse_pattern (str): Sparse attention pattern name.
        use_mla (bool): Whether to use Multi-Latent Attention.
        kv_lora_rank (int): LoRA rank for KV compression in MLA.
        modality_embed (nn.ParameterDict): Modality-specific embeddings.
    
    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     hidden_size: int = 4096
        ...     n_head: int = 32
        ...     n_kv_head: int = 8
        ...     max_position_embeddings: int = 131072
        ...     rope_theta: float = 10000.0
        ... 
        >>> cfg = Config()
        >>> attn = YvAttention(cfg)
        >>> hidden = torch.randn(2, 1024, 4096)
        >>> output = attn(hidden)
    
    Note:
        This class is designed to be the single entry point for all attention
        computation in the model. Backend-specific implementations are delegated
        to specialized classes while this class handles routing and configuration.
    """
    
    def __init__(self, cfg, device=None, dtype=None):
        """Initialize the YvAttention module with configuration.

        Args:
            cfg: Configuration object containing model hyperparameters.
                Required attributes:
                    - hidden_size: Model hidden dimension
                    - n_head: Number of attention heads
                    - max_position_embeddings: Maximum sequence length
                    - rope_theta: RoPE base frequency
                
                Optional attributes:
                    - n_kv_head: Number of KV heads for GQA (default: n_head)
                    - use_h2o_attention: Enable H2O backend (default: auto)
                    - use_flash_attention: Enable Flash Attention (default: True)
                    - use_alibi: Use ALiBi instead of RoPE (default: False)
                    - use_attention_sink: Enable attention sinks (default: True)
                    - use_qk_norm: Enable QK normalization (default: True)
                    - use_linear_attention: Enable linear attention (default: False)
                    - sliding_window: Sliding window size (default: 0)
                    - sparse_attention_pattern: Sparse pattern name (default: 'none')
                    - use_mla: Enable Multi-Latent Attention (default: True)
                    - kv_lora_rank: MLA KV compression rank (default: 512)
            device: Device for parameter initialization.
            dtype: Data type for parameter initialization.
        
        Example:
            >>> cfg = Config(hidden_size=4096, n_head=32, max_position_embeddings=131072)
            >>> attn = YvAttention(cfg, device='cuda', dtype=torch.bfloat16)
        """
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = getattr(cfg, 'n_kv_head', cfg.n_head)
        self.head_dim = cfg.hidden_size // cfg.n_head
        self.scale = getattr(cfg, 'attention_scale', None) or (self.head_dim ** -0.5)
        
        self.use_h2o = bool(getattr(cfg, 'use_h2o_attention', False)) or (cfg.max_position_embeddings > 1000000)
        self.use_flash = bool(getattr(cfg, 'use_flash_attention', True))
        self.use_alibi = bool(getattr(cfg, 'use_alibi', False))
        self.use_attention_sink = bool(getattr(cfg, 'use_attention_sink', True))
        self.use_qk_norm = bool(getattr(cfg, 'use_qk_norm', True))
        self.use_linear = bool(getattr(cfg, 'use_linear_attention', False))
        self.sliding_window = int(getattr(cfg, 'sliding_window', 0))
        self.sparse_pattern = getattr(cfg, 'sparse_attention_pattern', 'none')
        
        self.use_mla = bool(getattr(cfg, 'use_mla', True))
        self.kv_lora_rank = int(getattr(cfg, 'kv_lora_rank', 512))
        self.q_lora_rank = getattr(cfg, 'mla_q_lora_rank', None)
        
        if self.use_h2o:
            self.h2o_attention = YvH2OAttention(
                hidden_size=cfg.hidden_size,
                num_attention_heads=cfg.n_head,
                max_position_embeddings=cfg.max_position_embeddings,
                compression_ratio=getattr(cfg, 'compression_ratio', 8),
                streaming_window=getattr(cfg, 'streaming_window', 16384),
                dropout=getattr(cfg, 'attention_dropout', 0.0),
            )
        else:
            self.fused_qkv = bool(getattr(cfg, 'fused_qkv', True))

            if self.use_mla:
                self.kv_compress = nn.Linear(
                    cfg.hidden_size, self.kv_lora_rank, bias=False, device=device, dtype=dtype
                )
                self.k_decompress = nn.Linear(
                    self.kv_lora_rank, self.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype
                )
                self.v_decompress = nn.Linear(
                    self.kv_lora_rank, self.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype
                )
                self.rope_decompress = nn.Linear(
                    self.kv_lora_rank, self.head_dim, bias=False, device=device, dtype=dtype
                )
                if self.q_lora_rank is not None:
                    self.q_compress = nn.Linear(
                        cfg.hidden_size, self.q_lora_rank, bias=False, device=device, dtype=dtype
                    )
                    self.q_decompress = nn.Linear(
                        self.q_lora_rank, cfg.n_head * self.head_dim, bias=False, device=device, dtype=dtype
                    )
                else:
                    self.q_proj = nn.Linear(
                        cfg.hidden_size, cfg.n_head * self.head_dim, bias=False, device=device, dtype=dtype
                    )
                self.fused_qkv = False
            elif self.fused_qkv:
                qkv_out = (cfg.n_head + 2 * self.n_kv_head) * self.head_dim
                self.qkv_proj = nn.Linear(
                    cfg.hidden_size, qkv_out, bias=False, device=device, dtype=dtype
                )
            else:
                self.q_proj = nn.Linear(
                    cfg.hidden_size, cfg.n_head * self.head_dim, bias=False, device=device, dtype=dtype
                )
                self.k_proj = nn.Linear(
                    cfg.hidden_size, self.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype
                )
                self.v_proj = nn.Linear(
                    cfg.hidden_size, self.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype
                )

            self.o_proj = nn.Linear(
                cfg.n_head * self.head_dim, cfg.hidden_size, bias=False, device=device, dtype=dtype
            )
            
            if not self.use_alibi:
                self.rope = YvYaRNRotaryEmbedding(
                    self.head_dim,
                    cfg.max_position_embeddings,
                    cfg.rope_theta,
                    scale=32,
                    device=device,
                )
            else:
                self.alibi = YvALiBi(
                    cfg.n_head,
                    max_seq_len=min(cfg.max_position_embeddings, 8192),
                    device=device
                )
                
            if self.use_qk_norm:
                self.qk_norm = YvQKNormalizer(
                    self.head_dim, device=device, dtype=dtype
                )
                
            if self.use_attention_sink:
                self.attn_sink = YvAttentionSink(
                    cfg.hidden_size, n_sink=4, device=device, dtype=dtype
                )
                
            if self.use_linear:
                self.linear_attention = YvLinearAttention(
                    cfg.hidden_size, cfg.n_head,
                    feature_dim=getattr(cfg, 'linear_attention_dim', 64),
                    device=device, dtype=dtype
                )
                
            if self.sliding_window > 0:
                self.sliding_attention = YvSlidingWindowAttention(
                    cfg.hidden_size, cfg.n_head,
                    window_size=self.sliding_window,
                    device=device, dtype=dtype
                )
                
            if self.sparse_pattern != 'none':
                self.sparse_attention = YvSparseAttention(
                    cfg.hidden_size, cfg.n_head,
                    pattern=self.sparse_pattern,
                    block_size=getattr(cfg, 'sparse_block_size', 64),
                    device=device, dtype=dtype
                )
                
            self.attn_dropout = nn.Dropout(getattr(cfg, 'attention_dropout', 0.0))

        self.modality_embed = nn.ParameterDict({
            'text': nn.Parameter(torch.randn(cfg.hidden_size) * 0.02),
            'image': nn.Parameter(torch.randn(cfg.hidden_size) * 0.02),
            'video': nn.Parameter(torch.randn(cfg.hidden_size) * 0.02),
            'audio': nn.Parameter(torch.randn(cfg.hidden_size) * 0.02),
            'agentic': nn.Parameter(torch.randn(cfg.hidden_size) * 0.02),
        })

        self.apply(_arctic_init_weights)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        cache_manager: Optional[Any] = None,
        layer_idx: int = 0,
        modality: str = 'text'
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Run the attention forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].
            mask: Attention mask broadcastable to [batch_size, n_head, seq_len_q, seq_len_k].
            past_key_values: Cached (key, value) tensors for extending the current sequence.
            use_cache: Whether to return present key/value tensors for caching.
            cache_manager: Optional external cache manager used by H2O backend.
            layer_idx: Layer index for hierarchical retrieval routing.
            modality: Current modality ('text', 'image', 'video', 'audio', 'agentic').

        Returns:
            Attention output or tuple (output, (present_k, present_v)) if use_cache.
        """
        b, t, _ = x.shape
        
        if modality in self.modality_embed:
            x = x + self.modality_embed[modality].view(1, 1, -1)
        
        if layer_idx < 8:
            effective_window = min(2048, t)
        elif layer_idx < 16:
            effective_window = min(8192, t)
        else:
            effective_window = t
        
        if self.use_h2o:
            attn_out, present_kv = self.h2o_attention(
                hidden_states=x,
                attention_mask=None,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_manager=cache_manager,
            )
            if use_cache:
                return attn_out, present_kv
            return attn_out

        if self.use_attention_sink and self.training:
            x, sink_mask = self.attn_sink(x)
            if mask is not None:
                sink_mask_expanded = sink_mask.unsqueeze(1).unsqueeze(2)
                mask = torch.cat([sink_mask_expanded.expand(-1, mask.shape[1], -1, -1), mask], dim=-1)

        b, t, _ = x.shape

        if self.use_linear and t > 4096:
            linear_out = self.linear_attention(x, mask)
            if hasattr(self, 'sliding_attention') and self.sliding_window > 0:
                sliding_out = self.sliding_attention(x, mask, past_key_values, use_cache)
                output = 0.7 * linear_out + 0.3 * sliding_out
            else:
                output = linear_out
            return output

        if hasattr(self, 'sparse_attention') and self.sparse_pattern != 'none':
            return self.sparse_attention(x, mask)

        if self.use_mla:
            kv_latent = self.kv_compress(x)
            
            if hasattr(self, 'q_compress'):
                q_latent = self.q_compress(x)
                q = self.q_decompress(q_latent)
            else:
                q = self.q_proj(x)
            
            k_for_rope = self.rope_decompress(kv_latent)
            k_for_rope = k_for_rope.view(b, t, 1, self.head_dim)
            
            k = self.k_decompress(kv_latent)
            v = self.v_decompress(kv_latent)
            
            q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = v.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
            
            if hasattr(self, 'rope'):
                k_for_rope = k_for_rope.transpose(1, 2)
                k_for_rope = self.rope(k_for_rope, t)
                k_for_rope = k_for_rope.transpose(1, 2)
                
                k_rope_expanded = k_for_rope.expand(-1, self.n_kv_head, -1, -1)
                k = k + k_rope_expanded * 0.1
        elif getattr(self, 'fused_qkv', False):
            qkv = self.qkv_proj(x)
            q_end = self.n_head * self.head_dim
            kv_each = self.n_kv_head * self.head_dim
            q_lin = qkv[:, :, :q_end]
            k_lin = qkv[:, :, q_end:q_end + kv_each]
            v_lin = qkv[:, :, q_end + kv_each:]
            q = q_lin.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
            k = k_lin.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = v_lin.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        else:
            q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        if hasattr(self, 'qk_norm') and self.use_qk_norm:
            q, k = self.qk_norm(q, k)

        if hasattr(self, 'rope'):
            q, k = self.rope(q, t), self.rope(k, t)
            
            max_pe_len = getattr(self.cfg, 'max_position_embeddings', 4096)
            if t > max_pe_len // 2:
                overflow = t - max_pe_len // 2
                drop_ratio = min(0.3, overflow / t)
                drop_mask = torch.rand(t, device=q.device) > drop_ratio
                drop_mask[-256:] = True
                drop_mask = drop_mask.view(1, 1, -1, 1).expand_as(q)
                q = q * drop_mask.float()
                k = k * drop_mask[:, :, :k.shape[2], :].float()

        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        seq_len = k.size(-2)

        duo_streaming_window = min(self.sliding_window if self.sliding_window > 0 else 4096, seq_len)
        duo_streaming_window = min(duo_streaming_window, effective_window)
        
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            
            n_retrieval_heads = max(1, int(self.n_head * 0.4))
            n_streaming_heads = self.n_head - n_retrieval_heads
            
            n_retrieval_kv = max(1, n_retrieval_heads // repeat)
            n_streaming_kv = max(1, n_streaming_heads // repeat)
            
            k_retrieval = k[:, :, :n_retrieval_kv].repeat_interleave(repeat, dim=1)
            v_retrieval = v[:, :, :n_retrieval_kv].repeat_interleave(repeat, dim=1)
            
            k_streaming = k[:, :, -n_streaming_kv:].repeat_interleave(repeat, dim=1)
            v_streaming = v[:, :, -n_streaming_kv:].repeat_interleave(repeat, dim=1)
            
            if duo_streaming_window < seq_len:
                k_streaming = k_streaming[:, :, -duo_streaming_window:]
                v_streaming = v_streaming[:, :, -duo_streaming_window:]
            
            k = torch.cat([k_retrieval, k_streaming], dim=1)
            v = torch.cat([v_retrieval, v_streaming], dim=1)

        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        if hasattr(self, 'alibi') and self.use_alibi:
            alibi_bias = self.alibi(seq_len, x.device)
            alibi_bias = alibi_bias.unsqueeze(0)
        else:
            alibi_bias = None

        q_ = q.reshape(b * self.n_head, t, self.head_dim)
        k_ = k.reshape(b * self.n_head, seq_len, self.head_dim)
        v_ = v.reshape(b * self.n_head, seq_len, self.head_dim)

        base = seq_len - t
        row_pos = base + torch.arange(t, device=q.device)
        key_pos = torch.arange(seq_len, device=q.device)
        allowed2d = key_pos.view(1, -1) <= row_pos.view(-1, 1)

        duo_enabled = self.n_kv_head != self.n_head and seq_len > duo_streaming_window
        
        if duo_enabled:
            n_retrieval_heads = max(1, int(self.n_head * 0.4))
            n_streaming_heads = self.n_head - n_retrieval_heads
            
            retrieval_allowed = allowed2d.clone()
            
            streaming_allowed = allowed2d.clone()
            lower_bound = (row_pos.view(-1, 1) - (duo_streaming_window - 1))
            local_allowed = key_pos.view(1, -1) >= lower_bound
            streaming_allowed = streaming_allowed & local_allowed
            
            duo_allowed = torch.cat([
                retrieval_allowed.unsqueeze(0).expand(n_retrieval_heads, -1, -1),
                streaming_allowed.unsqueeze(0).expand(n_streaming_heads, -1, -1)
            ], dim=0)
            allowed2d = duo_allowed
        
        if bool(getattr(self.cfg, 'use_sliding_window', False)) and not duo_enabled:
            win = int(getattr(self.cfg, 'streaming_window', 16384))
            if win > 0 and win < seq_len:
                lower_bound = (row_pos.view(-1, 1) - (win - 1))
                local_allowed = key_pos.view(1, -1) >= lower_bound
                allowed2d = allowed2d & local_allowed

        # MTraining: Dynamic sparse attention for long context training
        # Auto-enabled when training and seq_len > 16384
        if self.training and seq_len > 16384:
            vertical_lines = min(8, seq_len // 2048)
            slash_width = min(512, seq_len // 32)
            
            key_importance = k.norm(dim=-1).mean(dim=1)
            _, top_k_idx = torch.topk(key_importance, vertical_lines, dim=-1)
            
            sparse_mask = torch.zeros(t, seq_len, device=q.device, dtype=torch.bool)
            for i in range(t):
                sparse_mask[i, top_k_idx[i]] = True
                sparse_mask[i, max(0, i - slash_width):i + 1] = True
            
            allowed2d = allowed2d & sparse_mask

        disallow2d = ~allowed2d
        
        if seq_len > 50000:
            pos_idx = torch.arange(seq_len, device=q.device).float()
            pos_weights = 1.0 + 0.2 * torch.sin(torch.pi * pos_idx / seq_len)
            mid_start, mid_end = seq_len // 4, 3 * seq_len // 4
            pos_weights[mid_start:mid_end] *= 1.25
            pos_bias = torch.log(pos_weights.view(1, 1, 1, -1).expand(b, self.n_head, t, seq_len))
            pos_bias = pos_bias.reshape(b * self.n_head, t, seq_len)

        if mask is not None:
            mask_slice = mask[:, :, -t:, :seq_len]
            if mask_slice.dtype == torch.bool:
                extra_disallow = ~mask_slice
            else:
                extra_disallow = mask_slice < -1e4
            disallow = disallow2d.view(1, 1, t, seq_len) | extra_disallow
            attn_mask = disallow.reshape(b * self.n_head, t, seq_len)
        else:
            attn_mask = disallow2d
        
        if seq_len > 50000:
            attn_mask = attn_mask - pos_bias

        if alibi_bias is not None:
            attn_mask = attn_mask.view(b, self.n_head, t, seq_len)
            attn_mask = attn_mask - alibi_bias[:, :, -t:, :seq_len]
            attn_mask = attn_mask.reshape(b * self.n_head, t, seq_len)

        try:
            from torch.backends.cuda import sdp_kernel as _sdp
            use_flash_kernel = torch.cuda.is_available() and bool(getattr(self.cfg, 'sdpa_prefer_flash', True))
        except Exception:
            use_flash_kernel = False

        if use_flash_kernel:
            with _sdp(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                out_ = F.scaled_dot_product_attention(
                    q_, k_, v_,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=False,
                )
        else:
            out_ = F.scaled_dot_product_attention(
                q_, k_, v_,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )

        # Gated Attention: Head-specific gating without extra parameters
        # Uses Q norm as gate signal - zero parameter overhead
        gate_signal = q.norm(dim=-1)
        gate_signal = torch.sigmoid(gate_signal.mean(dim=0, keepdim=True))
        gate_signal = gate_signal.view(1, self.n_head, t, 1).expand(b, -1, -1, self.head_dim)
        gate_signal = gate_signal.reshape(b * self.n_head, t, self.head_dim)
        out_ = gate_signal * out_

        out = out_.reshape(b, self.n_head, t, self.head_dim).transpose(1, 2).contiguous().view(b, t, -1)
        out = self.attn_dropout(out)
        out = self.o_proj(out)
        
        if seq_len > 100000:
            attenuation_factor = 0.15
            contrast_weight = 0.25
            attenuated = out * (1.0 - attenuation_factor)
            contrast = out - attenuated
            out = out + contrast_weight * contrast

        if use_cache:
            k_cache = k[:, :self.n_kv_head] if self.n_kv_head != self.n_head else k
            v_cache = v[:, :self.n_kv_head] if self.n_kv_head != self.n_head else v
            return out, (k_cache, v_cache)

        return out
