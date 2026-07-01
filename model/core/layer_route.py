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

# LayerRoute: Input-Conditioned Adaptive Layer Skipping via LoRA Fine-Tuning
# arXiv:2606.01838 (June 2026)
#
# Per-layer router with straight-through estimator for hard binary gates
# and LoRA adapters on QKV attention projections.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class YvLoRALayer(nn.Module):
    """Low-Rank Adaptation (LoRA) for a single linear layer.

    Applied in parallel to frozen base weight:
        output = base(x) + lora_A(lora_B(x)) * scale
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8, scale: float = 1.0,
                 device=None, dtype=None):
        super().__init__()
        self.lora_A = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.scale = scale
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scale


class YvLayerRouteAdapter(nn.Module):
    """LayerRoute: per-layer binary skip router + LoRA attention adapters.

    Architecture (per transformer block):
      1. Router: Linear(hidden, 1) -> sigmoid -> straight-through binary gate
      2. LoRA: rank-8 adapters on Q, K, V, O projections

    Gate regularization loss encourages high-magnitude decisions.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 lora_rank: int = 8, lora_scale: float = 1.0,
                 gate_reg_lambda: float = 0.01, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_reg_lambda = gate_reg_lambda

        self.router = nn.Linear(hidden_size, 1, bias=True, device=device, dtype=dtype)
        nn.init.xavier_uniform_(self.router.weight, gain=0.01)
        nn.init.zeros_(self.router.bias)

        qkv_dim = hidden_size
        out_dim = num_heads * head_dim

        self.lora_q = YvLoRALayer(qkv_dim, out_dim, rank=lora_rank, scale=lora_scale, device=device, dtype=dtype)
        self.lora_k = YvLoRALayer(qkv_dim, out_dim, rank=lora_rank, scale=lora_scale, device=device, dtype=dtype)
        self.lora_v = YvLoRALayer(qkv_dim, out_dim, rank=lora_rank, scale=lora_scale, device=device, dtype=dtype)
        self.lora_o = YvLoRALayer(out_dim, hidden_size, rank=lora_rank, scale=lora_scale, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        router_logit = self.router(x).squeeze(-1)
        gate_prob = torch.sigmoid(router_logit)
        gate = (gate_prob > 0.5).float()
        gate = gate + (gate_prob - gate_prob.detach())

        reg_loss = self.gate_reg_lambda * (gate_prob * (1 - gate_prob)).mean()

        return gate, reg_loss

    def lora_adapt(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            q + self.lora_q(o_input.transpose(1, 2).reshape(-1, self.hidden_size)).view(q.shape) if o_input is not None else q,
            k + self.lora_k(o_input.transpose(1, 2).reshape(-1, self.hidden_size)).view(k.shape) if o_input is not None else k,
            v + self.lora_v(o_input.transpose(1, 2).reshape(-1, self.hidden_size)).view(v.shape) if o_input is not None else v,
            None,
        )
