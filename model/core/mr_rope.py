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

# MrRoPE: Mixed-radix Rotary Position Embedding
# arXiv:2601.22181 (Jan 2026)
#
# Unified framework for RoPE extension via radix system conversion.
# Implements MrRoPE-Uni (uniform radix) and MrRoPE-Pro (progressive radix)
# for training-free "train short, test long" generalization.

import math
import torch
import torch.nn as nn
from typing import Optional


# Paper: Multi-resolution RoPE (arXiv:2601.22181, Jan 2026)
class YvMrRoPERotaryEmbedding(nn.Module):
    """Mixed-radix Rotary Position Embedding.

    Encodes positions i using mixed-radix representation:
        For dimension pair d, position i is encoded as:
            base_d = 10000^(2d/D)
            theta_d = i / base_d  (standard RoPE)
        
        MrRoPE variant: radices vary per dimension group:
            - Uni:  uniform interleaving of two base scales
            - Pro:  progressive radix (small radix early, large late)

    Supports training-free length extrapolation via radix conversion.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: int = 10000,
        scale: float = 1.0,
        original_max_position_embeddings: int = 4096,
        mode: str = 'pro',
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.mode = mode

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        if mode == 'pro':
            mix_radix = self._compute_pro_radix(dim, device=device)
        else:
            mix_radix = self._compute_uni_radix(dim, device=device)
        self.register_buffer('mix_radix', mix_radix, persistent=False)

        self.register_buffer(
            'max_seq_len_seen',
            torch.tensor(original_max_position_embeddings, dtype=torch.long),
            persistent=False,
        )

    def _compute_uni_radix(self, dim: int, device=None) -> torch.Tensor:
        half = dim // 2
        radix_a = 1.0 / (self.base ** (torch.arange(0, half, device=device).float() / dim))
        radix_b = 1.0 / ((self.base / math.pi) ** (torch.arange(0, half, device=device).float() / dim))
        return torch.stack([radix_a, radix_b], dim=-1).reshape(-1)[:half]

    def _compute_pro_radix(self, dim: int, device=None) -> torch.Tensor:
        half = dim // 2
        ratio = torch.arange(0, half, device=device).float() / half
        radix = 1.0 / ((self.base ** ratio) ** (torch.arange(0, half, device=device).float() / dim))
        return radix

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        device = x.device
        if seq_len is None:
            seq_len = x.shape[-2] if x.dim() == 4 else x.shape[-3] if x.dim() == 5 else x.shape[1]

        if seq_len > self.max_seq_len_seen.item():
            self.max_seq_len_seen.fill_(seq_len)

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        radix = self._apply_radix_extrapolation(self.mix_radix, seq_len, device)
        freqs = torch.outer(t, self.inv_freq * radix) * self.scale
        cos = freqs.cos()
        sin = freqs.sin()

        return self._apply_rope(x, cos, sin)

    def _apply_radix_extrapolation(self, base_radix: torch.Tensor, seq_len: int, device) -> torch.Tensor:
        if seq_len <= self.original_max_position_embeddings:
            return base_radix
        ratio = seq_len / self.original_max_position_embeddings
        ext = torch.logspace(0, math.log10(ratio), base_radix.shape[0], device=device)
        return base_radix * ext

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            batch, n_head, t, head_dim = x.shape
            cos = cos[:t, :head_dim // 2].unsqueeze(0).unsqueeze(0)
            sin = sin[:t, :head_dim // 2].unsqueeze(0).unsqueeze(0)
            x1 = x[..., :head_dim // 2]
            x2 = x[..., head_dim // 2:]
            rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        elif x.dim() == 3:
            t = x.shape[1]
            head_dim = x.shape[-1]
            cos = cos[:t, :head_dim // 2].unsqueeze(0)
            sin = sin[:t, :head_dim // 2].unsqueeze(0)
            x1 = x[..., :head_dim // 2]
            x2 = x[..., head_dim // 2:]
            rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        else:
            raise ValueError(f"Input must be 3D or 4D, got {x.dim()}D")
        return rotated

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_pos={self.max_position_embeddings}, base={self.base}, mode={self.mode}"
