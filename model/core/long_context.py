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

"""Long Context Processing for Yv Models.

Implements:
- OOMB: Million-token context training (ICLR 2026)
- REFORM: Compress-Gather-Recompute (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class YvOOMBContext(nn.Module):
    """OOMB: Million-token context training system.

    Enables single-GPU 4M token context via chunked processing
    and gradient accumulation.

    Attributes:
        chunk_size (int): Tokens per chunk.
        max_context_length (int): Maximum supported context length.

    Example:
        >>> oomb = YvOOMBContext(chunk_size=32768, max_context_length=4194304)
        >>> long_input = torch.randn(1, 1000000, 4096)
        >>> output = oomb.process(long_input, attention_fn)
    """

    def __init__(
        self,
        chunk_size: int = 32768,
        max_context_length: int = 4194304
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_context_length = max_context_length

    def process(
        self,
        hidden_states: torch.Tensor,
        attention_fn,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process long context in chunks.

        Args:
            hidden_states: Input [batch, seq_len, hidden].
            attention_fn: Attention function to apply.
            attention_mask: Optional mask.

        Returns:
            Output [batch, seq_len, hidden].
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        if seq_len <= self.chunk_size:
            return attention_fn(hidden_states, attention_mask)

        # Process in overlapping chunks
        outputs = []
        overlap = self.chunk_size // 8

        for start in range(0, seq_len, self.chunk_size - overlap):
            end = min(start + self.chunk_size, seq_len)
            chunk = hidden_states[:, start:end, :]

            chunk_mask = None
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :, start:end, start:end]

            chunk_output = attention_fn(chunk, chunk_mask)
            outputs.append(chunk_output)

        # Merge chunks (simple average at overlaps)
        merged = torch.zeros_like(hidden_states)
        counts = torch.zeros(seq_len, device=hidden_states.device)

        for i, (start, output) in enumerate(zip(
            range(0, seq_len, self.chunk_size - overlap), outputs
        )):
            end = min(start + output.shape[1], seq_len)
            merged[:, start:end, :] += output[:, :end-start, :]
            counts[start:end] += 1

        merged = merged / counts.unsqueeze(0).unsqueeze(-1).clamp(min=1)

        return merged


class YvREFORM(nn.Module):
    """REFORM: Compress-Gather-Recompute for long context.

    Three-stage processing:
    1. Compress KV cache
    2. Gather important tokens
    3. Recompute attention
    """

    def __init__(
        self,
        compression_ratio: int = 4,
        importance_threshold: float = 0.1
    ):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.importance_threshold = importance_threshold

    def compress_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress KV cache by averaging chunks.

        Args:
            key: Key tensor [batch, heads, seq, head_dim].
            value: Value tensor [batch, heads, seq, head_dim].

        Returns:
            Compressed (key, value).
        """
        batch, heads, seq, head_dim = key.shape
        chunk_size = max(1, seq // self.compression_ratio)

        # Pad to multiple of chunk_size
        pad_len = (chunk_size - seq % chunk_size) % chunk_size
        if pad_len > 0:
            key = F.pad(key, (0, 0, 0, pad_len))
            value = F.pad(value, (0, 0, 0, pad_len))

        new_seq = key.shape[2]
        num_chunks = new_seq // chunk_size

        # Reshape and average
        key_compressed = key.view(batch, heads, num_chunks, chunk_size, head_dim).mean(dim=3)
        value_compressed = value.view(batch, heads, num_chunks, chunk_size, head_dim).mean(dim=3)

        return key_compressed, value_compressed

    def gather_important(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """Gather important tokens based on attention weights.

        Args:
            hidden_states: Hidden states [batch, seq, hidden].
            attention_weights: Attention weights [batch, heads, seq, seq].

        Returns:
            Important token indices.
        """
        # Compute importance as average attention received
        importance = attention_weights.mean(dim=(1, 2))  # [batch, seq]

        # Select tokens above threshold
        important_mask = importance > self.importance_threshold

        return important_mask

    def recompute_attention(
        self,
        query: torch.Tensor,
        key_compressed: torch.Tensor,
        value_compressed: torch.Tensor,
        important_key: torch.Tensor,
        important_value: torch.Tensor
    ) -> torch.Tensor:
        """Recompute attention with compressed + important tokens.

        Args:
            query: Query tensor.
            key_compressed: Compressed keys.
            value_compressed: Compressed values.
            important_key: Important keys.
            important_value: Important values.

        Returns:
            Attention output.
        """
        # Combine compressed and important tokens
        key_full = torch.cat([key_compressed, important_key], dim=2)
        value_full = torch.cat([value_compressed, important_value], dim=2)

        # Compute attention
        scale = query.shape[-1] ** -0.5
        attn_weights = torch.matmul(query, key_full.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, value_full)

        return output
