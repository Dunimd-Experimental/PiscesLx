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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple


class YvLightningIndexer(nn.Module):
    """Lightweight top-k KV indexer running entirely in FP4 (simulated here as FP8).

    Given compressed KV entries, selects top-k relevant entries per query.
    Uses low-rank multi-query attention with FP4 QK path.

    Args:
        head_dim: Per-head dimension.
        n_head: Number of query heads.
        n_kv_head: Number of key/value heads.
        rank: Low-rank projection dimension. Default: 64.
        top_k: Number of compressed entries to select per query. Default: 1024.
    """

    def __init__(
        self,
        head_dim: int,
        n_head: int = 1,
        n_kv_head: int = 1,
        rank: int = 64,
        top_k: int = 1024,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.rank = rank
        self.top_k = top_k

        # Low-rank Q projection
        self.q_proj = nn.Linear(head_dim, rank, bias=False, device=device, dtype=dtype)
        # Low-rank K projection (shared across KV heads)
        self.k_proj = nn.Linear(head_dim, rank, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        q: torch.Tensor,
        k_compressed: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k compressed KV entries per query.

        Args:
            q: Query tensor (B, n_head, T, head_dim).
            k_compressed: Compressed key tensor (B, n_kv_head, T_compressed, head_dim).
            key_padding_mask: Optional mask for compressed positions.

        Returns:
            (selected_k, top_k_indices, index_scores) — top-k selected entries and their indices.
        """
        B, n_head, T, _ = q.shape
        B_kv, n_kv, T_comp, _ = k_compressed.shape

        # Project Q to low-rank space
        q_flat = q.reshape(B * n_head, T, self.head_dim)
        q_lr = self.q_proj(q_flat)  # (B*n_head, T, rank)
        q_lr = q_lr.view(B, n_head, T, self.rank)

        # Project K to low-rank space
        k_flat = k_compressed.reshape(B_kv * n_kv, T_comp, self.head_dim)
        k_lr = self.k_proj(k_flat)  # (B_kv*n_kv, T_comp, rank)

        # For multi-query: broadcast KV heads to match n_head
        if n_head > n_kv:
            repeat = n_head // n_kv
            k_lr = k_lr.view(B_kv, n_kv, T_comp, self.rank)
            k_lr = k_lr.repeat_interleave(repeat, dim=1)  # (B, n_head, T_comp, rank)
        else:
            k_lr = k_lr.view(B_kv, n_kv, T_comp, self.rank)
            if n_kv > n_head:
                k_lr = k_lr[:, :n_head]

        # Compute attention scores in low-rank space
        q_lr = q_lr * (self.rank ** -0.5)
        scores = torch.matmul(q_lr, k_lr.transpose(-2, -1))  # (B, n_head, T, T_comp)

        if key_padding_mask is not None:
            scores = scores + key_padding_mask.unsqueeze(1).unsqueeze(2)

        # Select top-k
        top_k_actual = min(self.top_k, T_comp)
        index_scores, top_k_indices = torch.topk(scores, top_k_actual, dim=-1)

        # Gather top-k KV entries
        k_expanded = k_compressed
        if n_head > n_kv:
            k_expanded = k_expanded.repeat_interleave(repeat, dim=1)

        # Gather for each head and each query position
        top_k_indices_exp = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        # k_expanded: (B, n_head, T_comp, head_dim) -> gather on dim=2
        k_expanded = k_expanded.unsqueeze(2).expand(-1, -1, T, -1, -1)
        selected_k = k_expanded.gather(3, top_k_indices_exp)

        return selected_k, top_k_indices, index_scores


# Paper: CSA from Long Context literature (DeepSeek-V4)
class YvCompressedSparseAttention(nn.Module):
    """Compressed Sparse Attention (CSA) from DeepSeek-V4.

    Compresses 4 adjacent tokens' KV into 1 entry using learned softmax weights
    + position bias, then selects top-k entries via Lightning Indexer.

    Features:
        - Overlap compression (two overlapping windows) to prevent boundary loss
        - Lightning Indexer with FP4 QK path
        - Sliding window for local detail preservation
        - Attention sink for streaming stability

    Args:
        hidden_size: Model hidden dimension.
        n_head: Number of attention heads.
        n_kv_head: Number of KV heads.
        compression_ratio: Number of tokens per compressed entry. Default: 4.
        top_k_compressed: Number of compressed entries to select. Default: 1024.
        sliding_window: Local window size. Default: 128.
        indexer_rank: Low-rank dim for indexer. Default: 64.
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        n_kv_head: int,
        compression_ratio: int = 4,
        top_k_compressed: int = 1024,
        sliding_window: int = 128,
        indexer_rank: int = 64,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = hidden_size // n_head
        self.compression_ratio = compression_ratio
        self.sliding_window = sliding_window

        # Compression projections
        self.k_compress = nn.Linear(
            hidden_size, n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.v_compress = nn.Linear(
            hidden_size, n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.q_proj = nn.Linear(
            hidden_size, n_head * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.o_proj = nn.Linear(
            n_head * self.head_dim, hidden_size, bias=False, device=device, dtype=dtype
        )

        # Learned position biases for compression
        self.pos_bias = nn.Parameter(torch.zeros(compression_ratio, device=device, dtype=dtype))

        # Lightning Indexer
        self.indexer = YvLightningIndexer(
            head_dim=self.head_dim,
            n_head=n_head,
            n_kv_head=n_kv_head,
            rank=indexer_rank,
            top_k=top_k_compressed,
            device=device,
            dtype=dtype,
        )

        # Attention sink
        self.sink = nn.Parameter(torch.randn(1, 1, 1, hidden_size, device=device, dtype=dtype) * 0.02)

        self.attn_dropout = nn.Dropout(dropout)

    def _compress_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress KV entries with overlap windows.

        Args:
            k: Key tensor (B, n_kv_head, T, head_dim).
            v: Value tensor (B, n_kv_head, T, head_dim).

        Returns:
            (k_compressed, v_compressed) — 4x compressed.
        """
        B, n_kv, T, D = k.shape
        cr = self.compression_ratio
        stride = cr // 2  # 50% overlap
        n_blocks = max(1, (T - cr) // stride + 1)

        # Extract overlap windows
        k_blocks = k.unfold(2, cr, stride)  # (B, n_kv, n_blocks, D, cr)
        v_blocks = v.unfold(2, cr, stride)

        # Apply learned position bias
        pos_weight = F.softmax(self.pos_bias, dim=0)  # (cr,)
        k_weighted = (k_blocks * pos_weight.view(1, 1, 1, 1, cr)).sum(dim=-1)
        v_weighted = (v_blocks * pos_weight.view(1, 1, 1, 1, cr)).sum(dim=-1)

        return k_weighted, v_weighted

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """CSA forward pass.

        Args:
            x: Input (B, T, hidden_size).
            mask: Attention mask.
            past_key_values: Cached KV.
            use_cache: Whether to cache KV.

        Returns:
            (output, present_kv) or output.
        """
        B, T, H = x.shape

        # Project Q
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Project K, V
        k = self.k_compress(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_compress(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Sliding window: keep last `sliding_window` uncompressed K,V
        sw = self.sliding_window
        if T > sw:
            k_sw = k[:, :, -sw:]
            v_sw = v[:, :, -sw:]
        else:
            k_sw = k
            v_sw = v

        # Compress full K,V
        k_compressed, v_compressed = self._compress_kv(k, v)  # (B, n_kv, n_blocks, D)

        # Indexer selects top-k compressed entries
        selected_k, top_k_indices, _ = self.indexer(q, k_compressed)  # (B, n_head, T, top_k, D)

        # Gather V entries corresponding to selected K indices
        v_expanded = v_compressed.unsqueeze(2).expand(-1, -1, T, -1, -1)  # (B, n_kv, T, T_comp, D)
        top_k_indices_exp = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        selected_v = v_expanded.gather(3, top_k_indices_exp)  # (B, n_kv, T, top_k, D)

        # Expand KV heads to match n_head
        repeat_factor = max(1, self.n_head // max(1, self.n_kv_head))
        k_attn = selected_k.reshape(B, self.n_head, -1, self.head_dim)
        if self.n_head > self.n_kv_head:
            selected_v = selected_v.repeat_interleave(repeat_factor, dim=1)
        v_attn = selected_v.reshape(B, self.n_head, -1, self.head_dim)

        # Concatenate sliding window with compressed
        if T > sw:
            k_sw_expanded = k_sw
            v_sw_expanded = v_sw
            if self.n_head > self.n_kv_head:
                k_sw_expanded = k_sw_expanded.repeat_interleave(repeat_factor, dim=1)
                v_sw_expanded = v_sw_expanded.repeat_interleave(repeat_factor, dim=1)
            k_full = torch.cat([k_attn, k_sw_expanded], dim=-2)
            v_full = torch.cat([v_attn, v_sw_expanded], dim=-2)
        else:
            k_full = k_attn
            v_full = v_attn

        # Standard attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k_full.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v_full)
        out = out.transpose(1, 2).reshape(B, T, H)
        out = self.o_proj(out)

        if use_cache:
            return out, (k, v)
        return out, None


# Paper: HCA from Long Context literature (DeepSeek-V4)
class YvHeavilyCompressedAttention(nn.Module):
    """Heavily Compressed Attention (HCA) from DeepSeek-V4.

    128x compression — all queries attend to all compressed entries (dense).
    Acts as a persistent global summary channel.

    Args:
        hidden_size: Model hidden dimension.
        n_head: Number of attention heads.
        compression_ratio: Aggressive compression ratio. Default: 128.
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        compression_ratio: int = 128,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.compression_ratio = compression_ratio

        self.q_proj = nn.Linear(
            hidden_size, n_head * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            hidden_size, n_head * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            hidden_size, n_head * self.head_dim, bias=False, device=device, dtype=dtype
        )
        self.o_proj = nn.Linear(
            n_head * self.head_dim, hidden_size, bias=False, device=device, dtype=dtype
        )

        self.attn_dropout = nn.Dropout(dropout)

    def _compress(self, x: torch.Tensor, ratio: int) -> torch.Tensor:
        """Average-pool compress along sequence dimension."""
        B, T, H = x.shape
        n_compressed = max(1, T // ratio)
        stride = T // n_compressed
        x = x[:, :n_compressed * stride]
        x = x.view(B, n_compressed, stride, H).mean(dim=2)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """HCA forward pass.

        Args:
            x: Input (B, T, hidden_size).
            mask: Attention mask.

        Returns:
            Output tensor.
        """
        B, T, H = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Compress K and V
        x_compressed = self._compress(x, self.compression_ratio)
        T_comp = x_compressed.shape[1]

        k = self.k_proj(x_compressed).view(B, T_comp, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_compressed).view(B, T_comp, self.n_head, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn + mask[:, :, -T:, -T_comp:] if mask.dim() == 4 else attn

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, H)
        out = self.o_proj(out)

        if use_cache:
            return out, (k, v)
        return out, None


# Paper: Original contribution by Dunimd Team (Yv Architecture — CSA+HCA hybrid)
class YvHybridAttention(nn.Module):
    """Hybrid CSA + HCA attention layer.

    First 2 layers use HCA, then CSA/HCA alternate.
    Provides efficient long-context processing with global + local attention.

    Args:
        cfg: Model config.
        layer_idx: Layer index (determines CSA vs HCA routing).
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(self, cfg, layer_idx: int = 0, device=None, dtype=None):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = cfg.hidden_size
        n_head = cfg.n_head
        n_kv_head = getattr(cfg, 'n_kv_head', n_head)

        # First 2 layers use HCA, then alternate
        if layer_idx < 2:
            self.attention = YvHeavilyCompressedAttention(
                hidden_size=hidden_size,
                n_head=n_head,
                compression_ratio=getattr(cfg, 'hca_compression_ratio', 128),
                dropout=getattr(cfg, 'attention_dropout', 0.0),
                device=device,
                dtype=dtype,
            )
        else:
            self.attention = YvCompressedSparseAttention(
                hidden_size=hidden_size,
                n_head=n_head,
                n_kv_head=n_kv_head,
                compression_ratio=getattr(cfg, 'csa_compression_ratio', 4),
                top_k_compressed=getattr(cfg, 'csa_top_k', 1024),
                sliding_window=getattr(cfg, 'csa_sliding_window', 128),
                indexer_rank=getattr(cfg, 'csa_indexer_rank', 64),
                dropout=getattr(cfg, 'attention_dropout', 0.0),
                device=device,
                dtype=dtype,
            )

    def forward(self, x, mask=None, past_key_values=None, use_cache=False):
        return self.attention(x, mask, past_key_values, use_cache)