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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class FusedMoEDispatch(nn.Module):
    """Sort-based MoE dispatch with contiguous expert computation.

    Replaces the per-expert mask-scan loop with a single argsort,
    eliminating the O(num_experts * num_tokens * top_k) comparison
    overhead.

    Key optimizations:
    1. Single argsort instead of num_experts full-tensor mask scans
    2. Only iterates over active experts (inference: 128/8 -> ~8)
    3. Contiguous memory access per expert (cache-friendly)
    4. Single bincount for expert boundaries instead of per-expert any()
    5. Single index_add_ scatter instead of per-expert scatter-add

    Zero quality loss: mathematically equivalent to the original loop.
    """

    def __init__(self, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        x_flat: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        experts: nn.ModuleList,
        collect_stats: bool = False,
    ) -> torch.Tensor:
        num_tokens = x_flat.shape[0]
        hidden = x_flat.shape[-1]
        device = x_flat.device
        dtype = x_flat.dtype
        k = self.top_k

        # 1. Flatten indices and weights
        flat_indices = expert_indices.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        num_assignments = flat_indices.numel()

        # 2. Token index map: each token appears k times
        token_ids = torch.arange(num_tokens, device=device)
        token_ids = token_ids.unsqueeze(1).expand(-1, k).reshape(-1)

        # 3. Sort all assignments by expert ID
        sorted_perm = torch.argsort(flat_indices, stable=True)
        sorted_experts = flat_indices[sorted_perm]
        sorted_tokens = token_ids[sorted_perm]
        sorted_weights = flat_weights[sorted_perm]

        # 4. Compute expert boundaries and identify active experts
        expert_counts = torch.bincount(flat_indices, minlength=self.num_experts)
        expert_offsets = torch.zeros(self.num_experts + 1, device=device, dtype=torch.long)
        expert_offsets[1:] = expert_counts.cumsum(0)
        active_ids = torch.where(expert_counts > 0)[0].cpu().tolist()
        offsets = expert_offsets.cpu().tolist()

        # 5. Gather sorted inputs (contiguous per expert)
        sorted_x = x_flat[sorted_tokens]

        # 6. Process each active expert's contiguous chunk
        expert_outputs = torch.zeros(num_assignments, hidden, device=device, dtype=dtype)
        expert_outputs_list = [] if collect_stats else None

        for expert_id in active_ids:
            start = offsets[expert_id]
            end = offsets[expert_id + 1]
            chunk_x = sorted_x[start:end]
            out = experts[expert_id](chunk_x)
            expert_outputs[start:end] = out

            if collect_stats:
                expert_outputs_list.append(out.mean(dim=0))

        # 7. Weight and scatter back via single index_add_
        weighted = sorted_weights.unsqueeze(1) * expert_outputs
        output = torch.zeros_like(x_flat)
        output.index_add_(0, sorted_tokens, weighted)

        if collect_stats:
            return output, expert_outputs_list
        return output


def fused_moe_dispatch(
    x_flat: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    experts: nn.ModuleList,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """Functional interface for sort-based MoE dispatch.

    Convenience wrapper for one-off use without instantiating the module.
    """
    return FusedMoEDispatch(num_experts, top_k)(
        x_flat, routing_weights, expert_indices, experts
    )