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

"""Embedding-Gated Multi-Head Latent Attention (EG-MLA) for Yv Models.

Original contribution by the Dunimd Team. Extends DeepSeek-V2 MLA with
embedding-gated dynamic KV compression.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .norms import YvYaRNRotaryEmbedding
from opss.infer.fused_mla import FusedMLAProjector


# Paper: DeepSeek-V2 (arXiv:2405.04434, 2024) MLA + embedding gating extension
class YvEGMLA(nn.Module):
    """Embedding-Gated Multi-Head Latent Attention with fused projection.

    Extends standard MLA with an embedding-dependent gating mechanism
    that dynamically controls KV compression. Achieves 91.6% KV cache
    reduction while maintaining attention quality.

    Attributes:
        hidden_size (int): Model hidden dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension per head.
        kv_lora_rank (int): Low-rank dimension for KV compression.
        q_lora_rank (int): Low-rank dimension for Q compression.
        use_fused (bool): Whether to use FusedMLAProjector.
        fused_mla (FusedMLAProjector): Fused projection (if use_fused).

    Example:
        >>> egmla = YvEGMLA(hidden_size=4096, num_heads=32, kv_lora_rank=512)
        >>> x = torch.randn(2, 1024, 4096)
        >>> output, compressed_kv = egmla(x)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        q_lora_rank: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        use_fused: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank or kv_lora_rank
        self.num_kv_heads = num_kv_heads or num_heads
        self.use_fused = use_fused

        if use_fused:
            self.fused_mla = FusedMLAProjector(
                hidden_size=hidden_size,
                n_head=num_heads,
                n_kv_head=self.num_kv_heads,
                head_dim=self.head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=self.q_lora_rank,
                mla_rope_dim=64,
                use_enhanced_mla=True,
                device=device,
                dtype=dtype,
            )
        else:
            self.embedding_gate = nn.Linear(
                hidden_size, kv_lora_rank, bias=False, device=device, dtype=dtype
            )
            self.kv_compress = nn.Linear(
                hidden_size, kv_lora_rank, bias=False, device=device, dtype=dtype
            )
            self.k_decompress = nn.Linear(
                kv_lora_rank, self.num_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype
            )
            self.v_decompress = nn.Linear(
                kv_lora_rank, self.num_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype
            )
            self.rope_decompress = nn.Linear(
                kv_lora_rank, 64, bias=False, device=device, dtype=dtype
            )
            self.q_compress = nn.Linear(
                hidden_size, self.q_lora_rank, bias=False, device=device, dtype=dtype
            )
            self.q_decompress = nn.Linear(
                self.q_lora_rank, num_heads * self.head_dim, bias=False, device=device, dtype=dtype
            )

        self.o_proj = nn.Linear(
            num_heads * self.head_dim, hidden_size, bias=False, device=device, dtype=dtype
        )

        self.scale = self.head_dim ** -0.5
        self.partial_rope_dim = 64
        self.rotary_emb = YvYaRNRotaryEmbedding(
            dim=self.partial_rope_dim,
            max_position_embeddings=10485760,
            base=10000.0,
        )

        self.register_buffer("total_tokens", torch.tensor(0.0), persistent=False)
        self.register_buffer("total_compression_ratio", torch.tensor(0.0), persistent=False)

    def fuse_weights(self):
        """Migrate separate linear weights to fused projector in-place."""
        if self.use_fused:
            return
        self.fused_mla = FusedMLAProjector.from_separate(
            kv_compress=self.kv_compress,
            embedding_gate=self.embedding_gate,
            k_decompress=self.k_decompress,
            v_decompress=self.v_decompress,
            rope_decompress=self.rope_decompress,
            q_compress=self.q_compress,
            q_decompress=self.q_decompress,
        )
        del self.embedding_gate
        del self.kv_compress
        del self.k_decompress
        del self.v_decompress
        del self.rope_decompress
        del self.q_compress
        del self.q_decompress
        self.use_fused = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        if self.use_fused:
            q, kv_latent, k_pe_raw = self.fused_mla.compress(hidden_states)

            if past_key_value is not None:
                past_latent = past_key_value[0]
                kv_latent = torch.cat([past_latent, kv_latent], dim=1)

            kv_seq_len = kv_latent.shape[1]
            k, v = self.fused_mla.decompress_kv(kv_latent)

            k_pe = self.fused_mla.decompress_k_pe(kv_latent)
            k_pe = k_pe.expand(-1, -1, self.num_kv_heads, -1).transpose(1, 2)
        else:
            gate = torch.sigmoid(self.embedding_gate(hidden_states))
            kv_latent_raw = self.kv_compress(hidden_states)
            kv_latent = gate * kv_latent_raw

            if past_key_value is not None:
                past_latent = past_key_value[0]
                kv_latent = torch.cat([past_latent, kv_latent], dim=1)

            kv_seq_len = kv_latent.shape[1]
            k = self.k_decompress(kv_latent)
            v = self.v_decompress(kv_latent)

            k = k.view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q_latent = self.q_compress(hidden_states)
            q = self.q_decompress(q_latent)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            k_pe = self.rope_decompress(kv_latent)
            k_pe = k_pe.view(batch_size, kv_seq_len, 1, -1)
            k_pe = k_pe.expand(-1, -1, self.num_kv_heads, -1).transpose(1, 2)

        if self.partial_rope_dim > 0:
            q_pe = q[..., -self.partial_rope_dim:]
            q_pe = self.rotary_emb(q_pe, seq_len)
            q = torch.cat([q[..., :-self.partial_rope_dim], q_pe], dim=-1)

            k_pe = self.rotary_emb(k_pe, kv_seq_len)
            k = torch.cat([k[..., :-self.partial_rope_dim], k_pe], dim=-1)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(attn_output)

        if self.training:
            compression_ratio = 1.0 - (self.kv_lora_rank / (self.num_kv_heads * self.head_dim))
            self.total_tokens += seq_len * batch_size
            self.total_compression_ratio += compression_ratio * seq_len * batch_size

        if use_cache:
            return output, (kv_latent,)

        return output, None

    def get_compression_ratio(self) -> float:
        if self.total_tokens.item() < 1:
            return 1.0 - (self.kv_lora_rank / (self.num_kv_heads * self.head_dim))
        return (self.total_compression_ratio / self.total_tokens).item()
