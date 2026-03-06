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

"""Cross-modal attention primitives for Yv multimodal encoders.

This module provides cross-modal attention components for the Yv model,
enabling inter-modality reasoning through multi-head attention mechanisms with
optional xFormers memory-efficient kernels.

Module Components:
    1. YvCrossModalAttention:
       - Multi-head attention for heterogeneous modality fusion
       - Optional xFormers memory-efficient attention
       - Layer normalization on query and key inputs
       - Dropout regularization

Key Features:
    - Standard scaled dot-product attention
    - xFormers memory-efficient attention support
    - Pre-attention layer normalization
    - Long sequence handling with attenuation/contrast
    - Configurable number of heads and dimensions

Performance Characteristics:
    - Standard attention: O(T_q * T_k * hidden_size)
    - xFormers attention: Reduced memory footprint
    - Projection: O(T * hidden_size^2)
    - Total complexity: O(T^2 * hidden_size)

Usage Example:
    >>> from model.multimodal.attention import YvCrossModalAttention
    >>> 
    >>> # Initialize attention module
    >>> attn = YvCrossModalAttention(config)
    >>> 
    >>> # Cross-modal attention
    >>> query = vision_features  # [B, T_v, hidden_size]
    >>> key = text_features      # [B, T_t, hidden_size]
    >>> value = text_features    # [B, T_t, hidden_size]
    >>> output = attn(query, key, value)  # [B, T_v, hidden_size]
    >>> 
    >>> # Self-attention
    >>> self_attn = attn(features, features, features)

Note:
    Requires hidden_size to be divisible by n_head.
    xFormers is optional; falls back to PyTorch implementation.
    Applies attenuation/contrast for sequences > 10000 tokens.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

class YvCrossModalAttention(nn.Module):
    """Multi-head attention layer for fusing heterogeneous modality embeddings.
    
    A comprehensive attention module that projects inputs into query/key/value
    spaces, performs scaled dot-product attention, and aggregates the result
    through an output projection. Supports optional xFormers memory-efficient
    kernels for reduced memory footprint on long sequences.
    
    Architecture:
        1. Input Normalization:
           - LayerNorm on query input (norm1)
           - LayerNorm on key input (norm2)
        
        2. Projection:
           - Linear projection for Q, K, V
           - Reshape to [B, num_heads, T, head_dim]
        
        3. Attention:
           - xFormers memory-efficient attention (if available)
           - Standard scaled dot-product attention (fallback)
        
        4. Output:
           - Concatenate heads
           - Linear output projection
           - Attenuation/contrast for long sequences
    
    Key Features:
        - Multi-head attention with configurable heads
        - Optional xFormers memory-efficient kernels
        - Pre-attention layer normalization
        - Dropout regularization
        - Long sequence handling with attenuation/contrast
    
    Attributes:
        cfg: Configuration namespace supplying attention hyperparameters.
        num_heads (int): Number of attention heads.
        hidden_size (int): Size of the model embedding dimension.
        head_dim (int): Per-head dimension derived from ``hidden_size``.
        q_proj (nn.Linear): Projection layer generating queries.
        k_proj (nn.Linear): Projection layer generating keys.
        v_proj (nn.Linear): Projection layer generating values.
        o_proj (nn.Linear): Projection layer applied to the concatenated output.
        norm1 (nn.LayerNorm): Layer normalization applied to the query input.
        norm2 (nn.LayerNorm): Layer normalization applied to the key input.
        dropout (nn.Dropout): Dropout applied to attention weights.
    
    Example:
        >>> attn = YvCrossModalAttention(config)
        >>> output = attn(query, key, value)  # [B, T_q, hidden_size]
    
    Note:
        Requires hidden_size to be divisible by n_head.
        xFormers is optional; falls back to PyTorch implementation.
        Applies attenuation/contrast for sequences > 10000 tokens.
    """

    def __init__(self, cfg):
        """Construct the attention module using the provided configuration.
        
        Args:
            cfg: Configuration object defining:
                - n_head: Number of attention heads
                - hidden_size: Model embedding dimension
        
        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``n_head``.
        """
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.n_head
        self.hidden_size = cfg.hidden_size
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by n_head for uniform head dimensions")
        self.head_dim = self.hidden_size // self.num_heads
        
        # Linear projection layers for query, key, value, and output
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key, value, mask=None):
        """Compute cross-modal attention weights and apply them to values.
        
        Performs multi-head attention by projecting inputs into query/key/value
        spaces, computing attention weights, and aggregating values. Supports
        both xFormers memory-efficient attention and standard PyTorch fallback.
        
        Args:
            query (torch.Tensor): Query tensor shaped ``[B, T_q, hidden_size]``.
                Represents the target modality or sequence to attend from.
            key (torch.Tensor): Key tensor shaped ``[B, T_k, hidden_size]``.
                Represents the source modality or sequence to attend to.
            value (torch.Tensor): Value tensor shaped ``[B, T_k, hidden_size]``.
                Represents the content to be aggregated.
            mask (torch.Tensor, optional): Attention mask shaped ``[B, T_q, T_k]``
                where zeros indicate masked positions. Defaults to ``None``.
                Note: xFormers path does not support custom masks.
        
        Returns:
            torch.Tensor: Context tensor shaped ``[B, T_q, hidden_size]``.
                The attended output combining query and key/value information.
        
        Raises:
            RuntimeError: If input tensors cannot be projected to the expected
            dimensions.
        
        Note:
            Uses xFormers memory-efficient attention when available and no mask.
            Falls back to standard scaled dot-product attention otherwise.
            Applies attenuation/contrast adjustment for T_q > 10000.
        """
        B, T_q, _ = query.shape
        B, T_k, _ = key.shape

        # Normalize and project input tensors to query, key, and value space
        # Then reshape and transpose to [B, num_heads, T, head_dim]
        Q = self.q_proj(self.norm1(query)).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(self.norm2(key)).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Try to import xFormers for memory-efficient attention
        try:
            from xformers.ops import memory_efficient_attention  # type: ignore
            _use_xformers = True
        except ImportError:
            _use_xformers = False

        if _use_xformers and mask is None:
            # Use xFormers memory-efficient attention if available and no mask is provided
            out = memory_efficient_attention(Q, K, V)  # Output shape: [B, num_heads, T_q, head_dim]
            out = out.transpose(1, 2).contiguous().view(B, T_q, self.hidden_size)
        else:
            # Use standard scaled dot-product attention as fallback
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, V)
            out = out.transpose(1, 2).contiguous().view(B, T_q, self.hidden_size)

        output = self.o_proj(out)
        
        if T_q > 10000:
            attenuation = 0.15
            contrast_weight = 0.25
            attenuated = output * (1.0 - attenuation)
            contrast = output - attenuated
            output = output + contrast_weight * contrast
        
        return output
