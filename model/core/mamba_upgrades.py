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

"""Mamba SSM Upgrades for Yv Models.

Implements:
- SparseSSM: Training-free Mamba pruning
- Gated Delta Networks: Improved Delta rule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class YvSparseSSM(nn.Module):
    """Training-free Mamba pruning.

    Prunes 50% of Mamba weights via structured magnitude-based pruning
    without retraining.

    Attributes:
        mamba_block: Mamba block to prune.
        sparsity_ratio: Fraction of weights to prune.
        pruning_mask: Mask for pruned weights.
    """

    def __init__(
        self,
        mamba_block: nn.Module,
        sparsity_ratio: float = 0.5
    ):
        super().__init__()
        self.mamba_block = mamba_block
        self.sparsity_ratio = sparsity_ratio
        self.pruning_masks = {}

        self._prune_weights()

    def _prune_weights(self) -> None:
        """Apply structured magnitude-based pruning."""
        for name, param in self.mamba_block.named_parameters():
            if param.dim() >= 2:
                # Compute magnitude
                magnitude = param.abs().mean(dim=-1)

                # Determine threshold for pruning
                threshold = torch.quantile(magnitude, self.sparsity_ratio)

                # Create mask (keep weights above threshold)
                mask = (magnitude.unsqueeze(-1) > threshold).float()
                self.pruning_masks[name] = mask

                # Apply mask
                param.data = param.data * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse weights.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        # Apply masks before forward
        for name, param in self.mamba_block.named_parameters():
            if name in self.pruning_masks:
                param.data = param.data * self.pruning_masks[name]

        return self.mamba_block(x)

    def get_sparsity(self) -> float:
        """Get actual sparsity ratio."""
        total_params = 0
        zero_params = 0

        for name, param in self.mamba_block.named_parameters():
            if name in self.pruning_masks:
                total_params += param.numel()
                zero_params += (param.data == 0).sum().item()

        if total_params == 0:
            return 0.0

        return zero_params / total_params


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class YvGatedDeltaNetwork(nn.Module):
    """Gated Delta Network for improved Mamba state updates.

    Uses gating mechanism for selective state retention:
    gate = sigmoid(W_gate @ x)
    new_state = gate * state + (1 - gate) * delta_update
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Standard Mamba projections
        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2, bias=False, device=device, dtype=dtype)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True, device=device, dtype=dtype)
        self.A_log = nn.Parameter(torch.randn(d_state, d_model))
        self.D = nn.Parameter(torch.ones(d_model))

        # Gating mechanism
        self.gate_proj = nn.Linear(d_model, d_state, bias=False, device=device, dtype=dtype)

        # State transition
        self.B_proj = nn.Linear(d_model, d_state, bias=False, device=device, dtype=dtype)
        self.C_proj = nn.Linear(d_model, d_state, bias=False, device=device, dtype=dtype)

        nn.init.xavier_uniform_(self.x_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.dt_proj.weight, gain=0.1)
        nn.init.uniform_(self.A_log, -3.0, -1.0)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with gated delta updates.

        Args:
            x: Input [batch, seq, d_model].
            state: Previous state [batch, d_state, d_model] or None.

        Returns:
            Tuple of (output, new_state).
        """
        batch, seq_len, _ = x.shape

        # Project input
        x_and_res = self.x_proj(x)
        delta, B, C = torch.split(
            x_and_res,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )

        # Compute delta
        delta = F.softplus(self.dt_proj(delta))

        # Compute A matrix
        A = -torch.exp(self.A_log)

        # Compute gate
        gate = torch.sigmoid(self.gate_proj(x.mean(dim=1)))  # [batch, d_state]

        # Initialize state if None
        if state is None:
            state = torch.zeros(batch, self.d_state, self.d_model, device=x.device, dtype=x.dtype)

        # Discretize
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(-1)

        # Gated state update
        outputs = []
        for t in range(seq_len):
            # Compute delta update
            delta_update = deltaB[:, t:t+1] * x[:, t:t+1].unsqueeze(2)

            # Apply gate: retain old state or accept new update
            g = gate.unsqueeze(1).unsqueeze(-1)  # [batch, 1, d_state, 1]
            state = g * (deltaA[:, t:t+1] * state) + (1 - g) * delta_update

            # Compute output
            y = torch.einsum('bsdm,bsm->bdm', state, C[:, t:t+1]).squeeze(1)
            y = y + self.D * x[:, t, :]
            outputs.append(y)

        output = torch.stack(outputs, dim=1)

        return output, state
