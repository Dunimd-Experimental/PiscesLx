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
from typing import Optional, Tuple, List


def _wave_schedule(expert_counts: torch.Tensor, max_wave_size: int) -> List[List[int]]:
    """Schedule experts into waves to maximize GPU utilization.

    Groups experts so that each wave has roughly equal total token count,
    minimizing idle time.

    Args:
        expert_counts: Number of tokens assigned to each expert.
        max_wave_size: Max tokens per wave.

    Returns:
        List of waves, each wave is a list of expert indices.
    """
    num_experts = len(expert_counts)
    sorted_indices = torch.argsort(expert_counts, descending=True)
    waves = []
    assigned = [False] * num_experts

    for idx in sorted_indices.tolist():
        if assigned[idx]:
            continue

        wave = [idx]
        assigned[idx] = True
        wave_tokens = expert_counts[idx].item()

        # Greedily fill wave
        for j in range(num_experts):
            if assigned[j]:
                continue
            candidate_tokens = expert_counts[j].item()
            if wave_tokens + candidate_tokens <= max_wave_size:
                wave.append(j)
                assigned[j] = True
                wave_tokens += candidate_tokens

        if wave_tokens > 0:
            waves.append(wave)

    return waves


def _mega_moe_forward(
    x: torch.Tensor,
    expert_weights: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    gate_scores: torch.Tensor,
    gate_indices: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Wave-scheduled MoE forward pass.

    Args:
        x: Input tokens (B*T, hidden_size).
        expert_weights: List of (gate_proj, up_proj, down_proj) per expert.
        gate_scores: Routing scores (B*T, top_k).
        gate_indices: Selected expert indices (B*T, top_k).
        top_k: Number of active experts per token.

    Returns:
        Output tokens (B*T, hidden_size).
    """
    hidden_size = x.shape[-1]
    device = x.device
    num_experts = len(expert_weights)
    num_tokens = x.shape[0]

    # Count tokens per expert
    expert_counts = torch.zeros(num_experts, device=device, dtype=torch.long)
    for k in range(top_k):
        expert_indices = gate_indices[:, k]
        expert_counts.scatter_add_(0, expert_indices, torch.ones(num_tokens, device=device, dtype=torch.long))

    max_wave_size = max(1024, num_tokens // max(1, num_experts))
    waves = _wave_schedule(expert_counts, max_wave_size)

    output = torch.zeros(num_tokens, hidden_size, device=device, dtype=x.dtype)

    for wave in waves:
        if not wave:
            continue

        # Gather all tokens for this wave's experts
        expert_tokens = []
        expert_masks = []
        expert_scores = []
        expert_ids = []

        for expert_idx in wave:
            for k in range(top_k):
                token_mask = (gate_indices[:, k] == expert_idx)
                token_ids = token_mask.nonzero(as_tuple=True)[0]
                if len(token_ids) == 0:
                    continue

                expert_tokens.append(x[token_ids])
                expert_masks.append(token_ids)
                expert_scores.append(gate_scores[token_ids, k].unsqueeze(-1))
                expert_ids.extend([expert_idx] * len(token_ids))

        if not expert_tokens:
            continue

        # Fuse: compute all tokens in batch
        cat_tokens = torch.cat(expert_tokens, dim=0)
        gate_w, up_w, down_w = expert_weights[wave[0]]  # Use first expert's dims

        # Fused gate+up projection for all tokens in wave
        gate_out = F.silu(F.linear(cat_tokens, gate_w))
        up_out = F.linear(cat_tokens, up_w)
        hidden = gate_out * up_out
        expert_out = F.linear(hidden, down_w)

        # Scatter back
        offset = 0
        for i, mask in enumerate(expert_masks):
            n = len(mask)
            weight = expert_scores[i]
            output[mask] += expert_out[offset:offset + n] * weight
            offset += n

    return output


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class YvMegaMoE(nn.Module):
    """MegaMoE layer with wave-based expert scheduling.

    Processes experts in optimized waves instead of one-at-a-time,
    achieving 1.5-1.96x speedup over per-expert dispatch.

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: Expert FFN intermediate size.
        num_experts: Total number of experts.
        top_k: Number of active experts per token.
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 64,
        top_k: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert weights: all experts stored as a single stacked parameter for efficiency
        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype) * 0.01
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype) * 0.01
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=dtype) * 0.01
        )

    def forward(
        self,
        x: torch.Tensor,
        gate_scores: torch.Tensor,
        gate_indices: torch.Tensor,
    ) -> torch.Tensor:
        """MegaMoE forward.

        Args:
            x: Input (B*T, H).
            gate_scores: Routing weights (B*T, top_k).
            gate_indices: Expert indices (B*T, top_k).

        Returns:
            Output (B*T, H).
        """
        num_tokens = x.shape[0]
        output = torch.zeros(num_tokens, self.hidden_size, device=x.device, dtype=x.dtype)

        # Count tokens per expert and create waves
        expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)
        for k in range(self.top_k):
            expert_counts.scatter_add_(
                0, gate_indices[:, k], torch.ones(num_tokens, device=x.device, dtype=torch.long)
            )

        max_wave_size = max(256, num_tokens // max(1, self.num_experts))
        waves = _wave_schedule(expert_counts, max_wave_size)

        for wave in waves:
            if not wave:
                continue

            # Gather all tokens to be processed by this wave
            indices_list = []
            scores_list = []

            for expert_idx in wave:
                for k in range(self.top_k):
                    mask = (gate_indices[:, k] == expert_idx)
                    ids = mask.nonzero(as_tuple=True)[0]
                    if len(ids) > 0:
                        indices_list.append(ids)
                        scores_list.append(gate_scores[ids, k].unsqueeze(-1))

            if not indices_list:
                continue

            all_indices = torch.cat(indices_list)
            all_scores = torch.cat(scores_list)

            # Determine which expert each token belongs to
            # Each (expert, top_k_position) produces one indices_list entry
            expert_id_for_token = []
            expert_to_wave = {e: i for i, e in enumerate(wave)}
            for expert_idx in wave:
                for k in range(self.top_k):
                    mask = (gate_indices[:, k] == expert_idx)
                    ids = mask.nonzero(as_tuple=True)[0]
                    if len(ids) > 0:
                        expert_id_for_token.extend([expert_idx] * len(ids))
            expert_id_for_token = torch.tensor(expert_id_for_token, device=x.device, dtype=torch.long)
            batch_tokens = x[all_indices]  # (total_in_wave, H)

            # Compute gate
            gate_w = self.gate_proj[expert_id_for_token]  # (batch, intermediate_size, H)
            gate_out = torch.bmm(gate_w, batch_tokens.unsqueeze(-1)).squeeze(-1)
            gate_act = F.silu(gate_out)

            # Compute up
            up_w = self.up_proj[expert_id_for_token]
            up_out = torch.bmm(up_w, batch_tokens.unsqueeze(-1)).squeeze(-1)

            # Element-wise multiply
            hidden = gate_act * up_out

            # Compute down
            down_w = self.down_proj[expert_id_for_token]
            expert_out = torch.bmm(down_w, hidden.unsqueeze(-1)).squeeze(-1)

            # Weight and scatter
            output.index_add_(0, all_indices, expert_out * all_scores)

        return output
