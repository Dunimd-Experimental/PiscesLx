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

# HiCL: Hippocampal-Inspired Continual Learning for MoE Routing
# arXiv:2508.16651v3 (AAAI 2026)
#
# Implements DG (Dentate Gyrus) gated routing with top-k sparsity for
# pattern separation and task-specific prototype-based expert selection.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
_LOG = PiscesLxLogger("Yv.HiCl", file_path=get_log_file("Yv.HiCl"), enable_file=True)


# Paper: Kapoor et al., "HiCL: Hippocampal-Inspired Continual Learning", AAAI 2026, arXiv:2508.16651
class YvDGEncoder(nn.Module):
    """Dentate Gyrus sparse pattern separator.

    Projects input to higher dimension, applies top-k sparsity for
    pattern separation (inspired by hippocampal DG sparse coding),
    then normalizes for cosine similarity routing.
    """

    def __init__(
        self,
        hidden_size: int,
        expansion_factor: int = 4,
        sparsity_k: int = 32,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dg_dim = hidden_size * expansion_factor
        self.sparsity_k = sparsity_k

        self.dg_proj = nn.Linear(hidden_size, self.dg_dim, bias=False, device=device, dtype=dtype)
        self.dg_act = nn.ReLU()

        self.register_buffer('dg_bias', torch.zeros(self.dg_dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_size)

        dg_out = self.dg_proj(x_flat) + self.dg_bias
        dg_out = self.dg_act(dg_out)

        k = min(self.sparsity_k, self.dg_dim)
        topk_vals, topk_idx = torch.topk(dg_out, k, dim=-1)
        sparse = torch.zeros_like(dg_out)
        sparse.scatter_(1, topk_idx, topk_vals)
        sparse = F.normalize(sparse, p=2, dim=-1)

        if len(orig_shape) == 3:
            sparse = sparse.view(orig_shape[0], orig_shape[1], self.dg_dim)
        return sparse


# Paper: Kapoor et al., "HiCL: Hippocampal-Inspired Continual Learning", AAAI 2026, arXiv:2508.16651
class YvHiClRouter(nn.Module):
    """HiCL DG-gated MoE router with cosine-similarity prototype matching.

    Replaces standard learned linear gate with:
    1. DG sparse encoding (pattern separation)
    2. Cosine similarity vs task-specific prototypes
    3. Top-k expert selection

    Prototypes are updated via online EMA (no separate gating network needed).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        dg_expansion: int = 4,
        dg_sparsity_k: int = 32,
        prototype_ema_momentum: float = 0.99,
        load_balance_alpha: float = 0.01,
        router_noise_std: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cfg: Optional[object] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.prototype_ema_momentum = prototype_ema_momentum
        self.load_balance_alpha = load_balance_alpha

        self.dg_encoder = YvDGEncoder(
            hidden_size=hidden_size,
            expansion_factor=dg_expansion,
            sparsity_k=dg_sparsity_k,
            device=device, dtype=dtype,
        )

        dg_dim = hidden_size * dg_expansion
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(num_experts, dg_dim, device=device, dtype=dtype) * 0.1, p=2, dim=-1)
        )

        self.register_buffer('step_counter', torch.tensor(0))
        self.register_buffer('expert_usage_count', torch.zeros(num_experts, device=device))
        self.register_buffer('total_routing_count', torch.tensor(0.0, device=device))
        self.register_buffer('expert_bias', torch.zeros(num_experts, device=device))
        self.register_buffer('bias_update_counter', torch.tensor(0, device=device))
        self._is_checkpointing = False

        bias_update_rate = getattr(cfg, 'moe_bias_update_rate', 0.05) if cfg is not None else 0.05
        self.bias_update_rate = bias_update_rate
        z_loss_alpha = getattr(cfg, 'moe_z_loss_alpha', 1e-4) if cfg is not None else 1e-4
        self.z_loss_alpha = z_loss_alpha

    def _normalize_prototypes(self):
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=-1)

    def _ema_update_prototypes(self, dg_repr: torch.Tensor, expert_indices: torch.Tensor):
        if not self.training or self._is_checkpointing:
            return
        momentum = min(self.prototype_ema_momentum, 1.0 - 1.0 / (1.0 + self.step_counter.item() * 0.1))
        for e_idx in range(self.num_experts):
            mask = (expert_indices == e_idx)
            if mask.any():
                assigned = dg_repr[mask]
                centroid = assigned.mean(dim=0)
                self.prototypes.data[e_idx] = F.normalize(
                    momentum * self.prototypes.data[e_idx] + (1 - momentum) * centroid,
                    p=2, dim=-1
                )

    def _compute_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return self.z_loss_alpha * torch.mean(logits ** 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.size(0)

        dg_repr = self.dg_encoder(x_flat)

        sim = F.linear(F.normalize(dg_repr, p=2, dim=-1), F.normalize(self.prototypes, p=2, dim=-1))
        logits = sim * 10.0 + self.expert_bias

        if self.training and not self._is_checkpointing:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise

        scores = F.softmax(logits, dim=-1)
        top_scores, top_idx = torch.topk(scores, self.top_k, dim=-1)
        top_scores = F.softmax(top_scores, dim=-1)

        if self.training and not self._is_checkpointing:
            self.step_counter.add_(1)
            self._ema_update_prototypes(dg_repr, top_idx)
            self.total_routing_count += num_tokens
            flat_idx = top_idx.flatten()
            counts = torch.bincount(flat_idx, minlength=self.num_experts).float()
            self.expert_usage_count += counts
            self._update_expert_bias()

        expert_freq = scores.mean(dim=0)
        ideal_freq = torch.ones_like(expert_freq) / self.num_experts
        load_loss = self.load_balance_alpha * torch.sum((expert_freq - ideal_freq) ** 2)

        z_loss = self._compute_z_loss(logits)
        total_loss = load_loss + z_loss

        return top_scores, top_idx, total_loss

    def _update_expert_bias(self):
        self.bias_update_counter.add_(1)
        if self.bias_update_counter.item() % 10 != 0:
            return
        usage = self.expert_usage_count / (self.total_routing_count + 1e-8)
        target = 1.0 / self.num_experts
        delta = target - usage
        new_bias = self.expert_bias + delta * self.bias_update_rate
        self.expert_bias.copy_(torch.clamp(new_bias, -2.0, 2.0))
