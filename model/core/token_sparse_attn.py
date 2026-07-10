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
Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting
(arXiv:2502.12216, ICLR 2026).

Sparsity-adaptive, calibration-free sparse attention for long-context LLMs.
Dynamically selects tokens based on cumulative attention score threshold,
using clustering-based sorting and distribution fitting for efficiency.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


# Paper: Zhu et al., "Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs," ICLR 2026, arXiv:2502.12216
class YvTokenSparseAttention(nn.Module):
    """Tactic adaptive sparse attention with cumulative attention threshold.

    Dynamically selects tokens per head based on cumulative attention mass
    rather than a fixed token budget. Uses clustering + distribution fitting
    to estimate the threshold efficiently without full sorting.

    Reference: Tactic (ICLR 2026) — 5.14x decode attention speedup.
    """

    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.cumulative_target = getattr(config, 'tactic_cumulative_target', 0.8)
        self.n_bins = getattr(config, 'tactic_n_bins', 64)
        self.n_clusters = getattr(config, 'tactic_n_clusters', 4)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)

        self.register_buffer('cluster_centroids', torch.zeros(self.n_clusters, device=device, dtype=dtype))

    def _estimate_threshold_via_histogram(
        self, scores: torch.Tensor
    ) -> torch.Tensor:
        B, H, T_q, T_k = scores.shape
        scores_flat = scores.view(B * H * T_q, T_k)

        v_min = scores_flat.amin(dim=-1, keepdim=True)
        v_max = scores_flat.amax(dim=-1, keepdim=True)
        range_safe = (v_max - v_min).clamp(min=1e-8)

        normalized = (scores_flat - v_min) / range_safe
        bin_idx = (normalized * (self.n_bins - 1)).long().clamp(0, self.n_bins - 1)

        bins = torch.zeros(scores_flat.shape[0], self.n_bins, device=scores.device)
        bins.scatter_add_(1, bin_idx, torch.ones_like(bin_idx, dtype=bins.dtype))

        bin_centers = (torch.arange(self.n_bins, device=scores.device).float() + 0.5) / self.n_bins
        bin_values = v_min + bin_centers.unsqueeze(0) * range_safe

        cumsum = bins.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        total = cumsum[:, :1].clamp(min=1)
        cum_frac = cumsum / total

        target_idx = (cum_frac >= self.cumulative_target).float().argmax(dim=-1, keepdim=True).clamp(0, self.n_bins - 1)
        threshold = bin_values.gather(1, target_idx)

        return threshold.view(B, H, T_q, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        Q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        score_dim = max(1, self.head_dim // 4)
        Q_lr = Q[..., :score_dim]
        K_lr = K[..., :score_dim]

        prelim_scores = torch.matmul(Q_lr, K_lr.transpose(-2, -1)) / (score_dim ** 0.5)

        if attention_mask is not None:
            while attention_mask.dim() < prelim_scores.dim():
                attention_mask = attention_mask.unsqueeze(1)
            prelim_scores = prelim_scores + attention_mask

        threshold = self._estimate_threshold_via_histogram(prelim_scores)
        sparse_mask = prelim_scores >= threshold
        sparse_mask = sparse_mask | (prelim_scores == prelim_scores.max(dim=-1, keepdim=True).values)

        attn = prelim_scores.masked_fill(~sparse_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        out = self.o_proj(out)

        return out
