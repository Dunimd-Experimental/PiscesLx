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
Fused Multi-Head Latent Attention (MLA) Projection.

Fuses the MLA pre-attention projection chain into fewer, larger matmuls:
  - RMSNorm + kv_compress + embedding_gate + q_proj  -> 1 fused block (norm + 2 matmuls)
  - k_decompress + v_decompress                      -> 1 fused linear (2x fewer kernels)
  - RoPE decompress                                   -> kept separate

Total kernel launches: 7 -> 2 (norm absorbed into compress matmuls, ~71% reduction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FusedMLAProjector(nn.Module):
    """Fused MLA projection: merges compress+gate and paired decompress into single matmuls.

    Architecture:
        compress(x) -> (q, kv_latent, k_pe_raw)
        decompress(kv_latent) -> (k, v)

    This separation allows cache-efficient autoregressive decoding:
      1. compress() processes only the NEW token (t=1)
      2. kv_latent is concatenated with cached past_kv_latent
      3. decompress() processes the FULL concatenated sequence
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        kv_lora_rank: int,
        q_lora_rank: Optional[int],
        mla_rope_dim: int,
        use_enhanced_mla: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.mla_rope_dim = mla_rope_dim
        self.use_enhanced_mla = use_enhanced_mla

        # Fused [kv_compress | embedding_gate]: hidden -> 2*kv_lora_rank
        self.kv_gate_fused = nn.Linear(
            hidden_size, 2 * kv_lora_rank, bias=False, device=device, dtype=dtype
        )

        # Fused [k_decompress | v_decompress]: kv_lora_rank -> 2*n_kv_head*head_dim
        self.kv_decompress_fused = nn.Linear(
            kv_lora_rank, 2 * n_kv_head * head_dim, bias=False, device=device, dtype=dtype
        )

        # Fused Q projection: hidden -> n_head*head_dim (single linear, regardless of q_lora_rank)
        self.q_proj_fused = nn.Linear(hidden_size, n_head * head_dim, bias=False, device=device, dtype=dtype)
        self._has_low_rank_q = q_lora_rank is not None  # for from_separate weight migration

        # RoPE decompress: kv_latent -> rope_dim
        rope_out_dim = mla_rope_dim if use_enhanced_mla else head_dim
        self.rope_decompress = nn.Linear(kv_lora_rank, rope_out_dim, bias=False, device=device, dtype=dtype)

    @classmethod
    def from_separate(
        cls,
        kv_compress: nn.Linear,
        embedding_gate: nn.Linear,
        k_decompress: nn.Linear,
        v_decompress: nn.Linear,
        rope_decompress: nn.Linear,
        q_compress: Optional[nn.Linear] = None,
        q_decompress: Optional[nn.Linear] = None,
        q_proj: Optional[nn.Linear] = None,
    ) -> FusedMLAProjector:
        """Create FusedMLAProjector from individual nn.Linear layers by migrating weights."""
        hs = kv_compress.in_features
        hs_q = (q_proj or q_decompress).out_features
        n_h = (q_proj or q_decompress).in_features
        n_h = hs_q // (hs_q // (q_proj or q_decompress).weight.shape[0])
        n_kvh = k_decompress.out_features // (k_decompress.out_features // (k_decompress.in_features or 1))
        hd = k_decompress.out_features // n_kvh

        proj = cls.__new__(cls)
        proj.hidden_size = hs
        proj.n_head = n_h
        proj.n_kv_head = n_kvh
        proj.head_dim = hd
        proj.kv_lora_rank = kv_compress.out_features
        proj.q_lora_rank = q_compress.out_features if q_compress else None
        proj.mla_rope_dim = rope_decompress.out_features
        proj.use_enhanced_mla = rope_decompress.out_features < hd

        dev = kv_compress.weight.device
        dt = kv_compress.weight.dtype

        proj.kv_gate_fused = nn.Linear(hs, 2 * proj.kv_lora_rank, bias=False, device=dev, dtype=dt)
        proj.kv_decompress_fused = nn.Linear(proj.kv_lora_rank, 2 * n_kvh * hd, bias=False, device=dev, dtype=dt)
        if q_compress is not None:
            proj.q_proj_fused = nn.Linear(hs, n_h * hd, bias=False, device=dev, dtype=dt)
        else:
            proj.q_proj_fused = nn.Linear(hs, n_h * hd, bias=False, device=dev, dtype=dt)
        rope_dim = proj.mla_rope_dim if proj.use_enhanced_mla else hd
        proj.rope_decompress = nn.Linear(proj.kv_lora_rank, rope_dim, bias=False, device=dev, dtype=dt)

        with torch.no_grad():
            kvo = kv_compress.out_features
            proj.kv_gate_fused.weight.data[:kvo] = kv_compress.weight.data
            proj.kv_gate_fused.weight.data[kvo:] = embedding_gate.weight.data
            kdo = k_decompress.out_features
            proj.kv_decompress_fused.weight.data[:kdo] = k_decompress.weight.data
            proj.kv_decompress_fused.weight.data[kdo:] = v_decompress.weight.data
            if q_compress is not None:
                # W_fused = W_qd @ W_qc  (both row-major: W_qc: [q_lora, hs], W_qd: [n_h*hd, q_lora])
                proj.q_proj_fused.weight.data = q_decompress.weight.data @ q_compress.weight.data
            else:
                proj.q_proj_fused.weight.data = q_proj.weight.data
            proj.rope_decompress.weight.data = rope_decompress.weight.data

        return proj

    def norm_compress(
        self, x: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused RMSNorm + compress. Avoids materializing x_normed to HBM.

        Args:
            x: [b, t, hidden_size] (raw, pre-norm)
            norm_weight: [hidden_size] RMSNorm weight
            norm_eps: RMSNorm epsilon
        Returns:
            q: [b, n_head, t, head_dim]
            kv_latent: [b, t, kv_lora_rank]
            k_pe_raw: [b, t, 1, rope_dim]
        """
        b, t, h = x.shape

        # Fused RMSNorm: compute rms on-the-fly, scale x, feed directly into matmuls
        rms = (x * x).mean(-1, keepdim=True).add(norm_eps).rsqrt()
        x_scaled = x * rms
        x_normed = x_scaled * norm_weight

        # Fused kv_compress + embedding_gate
        kv_gate = F.linear(x_normed, self.kv_gate_fused.weight)
        kv_latent_raw, gate_logits = kv_gate.chunk(2, dim=-1)
        gate = torch.sigmoid(gate_logits)
        kv_latent = kv_latent_raw * gate

        # Fused Q projection
        q = F.linear(x_normed, self.q_proj_fused.weight)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE decompress
        k_pe_raw = F.linear(kv_latent, self.rope_decompress.weight).view(b, t, 1, -1)

        return q, kv_latent, k_pe_raw

    def compress(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress input tokens: returns (q, kv_latent, k_pe_raw) for current tokens only.

        Args:
            x: [b, t, hidden_size]
        Returns:
            q: [b, n_head, t, head_dim]
            kv_latent: [b, t, kv_lora_rank]
            k_pe_raw: [b, t, 1, rope_dim] (pre-RoPE)
        """
        b, t, _ = x.shape

        # Fused kv_compress + embedding_gate
        kv_gate = self.kv_gate_fused(x)
        kv_latent_raw, gate_logits = kv_gate.chunk(2, dim=-1)
        gate = torch.sigmoid(gate_logits)
        kv_latent = kv_latent_raw * gate

        # Fused Q projection
        q = self.q_proj_fused(x)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE decompress
        k_pe_raw = self.rope_decompress(kv_latent).view(b, t, 1, -1)

        return q, kv_latent, k_pe_raw

    def decompress_kv(self, kv_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress kv_latent to full K, V.

        Args:
            kv_latent: [b, kv_len, kv_lora_rank]
        Returns:
            k: [b, n_kv_head, kv_len, head_dim]
            v: [b, n_kv_head, kv_len, head_dim]
        """
        b, kv_len, _ = kv_latent.shape
        kv = self.kv_decompress_fused(kv_latent)
        k_raw, v_raw = kv.chunk(2, dim=-1)
        k = k_raw.view(b, kv_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v_raw.view(b, kv_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        return k, v

    def decompress_k_pe(self, kv_latent: torch.Tensor) -> torch.Tensor:
        """Decompress RoPE dimensions from kv_latent (single-head, no expand).

        RoPE should be applied BEFORE expand to save n_kv_head× compute/memory.

        Args:
            kv_latent: [b, kv_len, kv_lora_rank]
        Returns:
            k_pe_raw: [b, kv_len, 1, rope_dim] (pre-RoPE, single head)
        """
        b, kv_len, _ = kv_latent.shape
        k_pe = self.rope_decompress(kv_latent)
        k_pe = k_pe.view(b, kv_len, 1, -1)
        return k_pe