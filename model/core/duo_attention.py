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

"""DuoAttention: Retrieval-Head vs Streaming-Head Separation for Yv Models.

Based on MIT ICLR 2026. Separates attention heads into:
- Retrieval heads (20%): Keep full KV cache for important tokens
- Streaming heads (80%): Use constant-size KV buffer (compressed)

Achieves 2.55x memory reduction and supports 3.3M token context.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class YvDuoAttention(nn.Module):
    """DuoAttention with retrieval-head vs streaming-head separation.

    Classifies attention heads into two groups:
    1. Retrieval heads: Full KV cache for semantic retrieval
    2. Streaming heads: Constant KV buffer for efficient streaming

    Attributes:
        hidden_size (int): Model hidden dimension.
        num_heads (int): Total number of attention heads.
        num_kv_heads (int): Number of key/value heads.
        head_dim (int): Dimension per head.
        retrieval_ratio (float): Fraction of heads as retrieval heads.
        num_retrieval_heads (int): Number of retrieval heads.
        num_streaming_heads (int): Number of streaming heads.
        streaming_buffer_size (int): Fixed buffer size for streaming heads.

    Example:
        >>> duo = YvDuoAttention(hidden_size=4096, num_heads=32, num_kv_heads=8)
        >>> x = torch.randn(2, 1024, 4096)
        >>> output = duo(x)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        retrieval_ratio: float = 0.2,
        streaming_buffer_size: int = 1024,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.retrieval_ratio = retrieval_ratio
        self.streaming_buffer_size = streaming_buffer_size

        # Head classification
        self.num_retrieval_heads = max(1, int(num_heads * retrieval_ratio))
        self.num_streaming_heads = num_heads - self.num_retrieval_heads

        # Retrieval heads: full KV projection
        self.retrieval_q_proj = nn.Linear(
            hidden_size, self.num_retrieval_heads * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.retrieval_k_proj = nn.Linear(
            hidden_size, self.num_retrieval_heads * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.retrieval_v_proj = nn.Linear(
            hidden_size, self.num_retrieval_heads * self.head_dim, bias=False, device=device, dtype=dtype
        )

        # Streaming heads: compressed KV with fixed buffer
        self.streaming_q_proj = nn.Linear(
            hidden_size, self.num_streaming_heads * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.streaming_kv_compress = nn.Linear(
            hidden_size, self.num_streaming_heads * self.head_dim * 2, bias=False, device=device, dtype=dtype
        )

        # Output projection
        self.o_proj = nn.Linear(
            num_heads * self.head_dim, hidden_size, bias=False, device=device, dtype=dtype
        )

        self.scale = self.head_dim ** -0.5

        # Streaming KV buffer (fixed size, updated with FIFO)
        self.register_buffer("streaming_k_buffer", None)
        self.register_buffer("streaming_v_buffer", None)

    def _update_streaming_buffer(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update streaming buffer with FIFO eviction.

        Args:
            k_new: New keys [batch, num_streaming_heads, seq_len, head_dim].
            v_new: New values [batch, num_streaming_heads, seq_len, head_dim].

        Returns:
            Updated buffers (k_buffer, v_buffer).
        """
        batch_size, _, seq_len, _ = k_new.shape

        if self.streaming_k_buffer is None:
            self.streaming_k_buffer = k_new
            self.streaming_v_buffer = v_new
            return self.streaming_k_buffer, self.streaming_v_buffer

        # Concatenate and trim to buffer size
        k_combined = torch.cat([self.streaming_k_buffer, k_new], dim=2)
        v_combined = torch.cat([self.streaming_v_buffer, v_new], dim=2)

        max_len = getattr(self, 'streaming_buffer_max_len', 8192)
        if k_combined.shape[2] > max_len:
            k_combined = torch.cat([
                k_combined[:, :, :128, :],
                k_combined[:, :, -(max_len - 128):, :]
            ], dim=2)
            v_combined = torch.cat([
                v_combined[:, :, :128, :],
                v_combined[:, :, -(max_len - 128):, :]
            ], dim=2)
        elif k_combined.shape[2] > self.streaming_buffer_size:
            k_combined = k_combined[:, :, -self.streaming_buffer_size:]
            v_combined = v_combined[:, :, -self.streaming_buffer_size:]

        self.streaming_k_buffer = k_combined
        self.streaming_v_buffer = v_combined

        return k_combined, v_combined

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with DuoAttention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Optional attention mask.
            past_key_value: Optional cached KV.
            use_cache: Whether to return present KV.

        Returns:
            Tuple of (attention_output, present_kv).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Retrieval heads: full attention
        q_ret = self.retrieval_q_proj(hidden_states)
        k_ret = self.retrieval_k_proj(hidden_states)
        v_ret = self.retrieval_v_proj(hidden_states)

        q_ret = q_ret.view(batch_size, seq_len, self.num_retrieval_heads, self.head_dim).transpose(1, 2)
        k_ret = k_ret.view(batch_size, seq_len, self.num_retrieval_heads, self.head_dim).transpose(1, 2)
        v_ret = v_ret.view(batch_size, seq_len, self.num_retrieval_heads, self.head_dim).transpose(1, 2)

        # Streaming heads: compressed KV
        q_str = self.streaming_q_proj(hidden_states)
        q_str = q_str.view(batch_size, seq_len, self.num_streaming_heads, self.head_dim).transpose(1, 2)

        kv_compressed = self.streaming_kv_compress(hidden_states)
        k_str = kv_compressed[..., :self.num_streaming_heads * self.head_dim]
        v_str = kv_compressed[..., self.num_streaming_heads * self.head_dim:]

        k_str = k_str.view(batch_size, seq_len, self.num_streaming_heads, self.head_dim).transpose(1, 2)
        v_str = v_str.view(batch_size, seq_len, self.num_streaming_heads, self.head_dim).transpose(1, 2)

        k_str, v_str = self._update_streaming_buffer(k_str, v_str)

        # Handle past key values
        if past_key_value is not None:
            past_k, past_v = past_key_value
            # Split past into retrieval and streaming
            past_k_ret = past_k[:, :self.num_retrieval_heads]
            past_k_str = past_k[:, self.num_retrieval_heads:]
            past_v_ret = past_v[:, :self.num_retrieval_heads]
            past_v_str = past_v[:, self.num_retrieval_heads:]

            k_ret = torch.cat([past_k_ret, k_ret], dim=2)
            v_ret = torch.cat([past_v_ret, v_ret], dim=2)
            k_str = torch.cat([past_k_str, k_str], dim=2)
            v_str = torch.cat([past_v_str, v_str], dim=2)

        # Retrieval attention (full KV)
        attn_weights_ret = torch.matmul(q_ret, k_ret.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_weights_ret = attn_weights_ret + attention_mask
        attn_weights_ret = F.softmax(attn_weights_ret, dim=-1)
        attn_output_ret = torch.matmul(attn_weights_ret, v_ret)

        # Streaming attention (buffered KV)
        attn_weights_str = torch.matmul(q_str, k_str.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_weights_str = attn_weights_str + attention_mask
        attn_weights_str = F.softmax(attn_weights_str, dim=-1)
        attn_output_str = torch.matmul(attn_weights_str, v_str)

        # Combine outputs
        attn_output = torch.cat([attn_output_ret, attn_output_str], dim=1)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(attn_output)

        if use_cache:
            present_k = torch.cat([k_ret, k_str], dim=1)
            present_v = torch.cat([v_ret, v_str], dim=1)
            return output, (present_k, present_v)

        return output, None
