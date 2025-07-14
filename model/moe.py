#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import torch
from torch import nn
import torch.nn.functional as F
import math

def moe_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MoEGate(nn.Module):
    """Expert routing gate for MoE"""
    def __init__(self, hidden_size, num_experts, device=None, dtype=None):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
    
    def forward(self, x):
        logits = self.gate(x)
        scores, idx = torch.topk(logits, 2, dim=-1)
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)
        return scores, idx

class MoELayer(nn.Module):
    """Mixture of Experts layer (Kimi-K2/Qwen style optimized init)"""
    _layer_count = 0
    def __init__(self, cfg, device=None, dtype=None, print_every=8):
        super().__init__()
        MoELayer._layer_count += 1
        self.cfg = cfg
        self.gate = MoEGate(cfg.hidden_size, cfg.moe_num_experts, device=device, dtype=dtype)
        self.experts = nn.ModuleList()
        n = cfg.moe_num_experts
        print_detail = (MoELayer._layer_count == 1)
        if print_detail:
            print(f"[DEBUG] MoELayer: initializing {n} experts...")
        for i in range(n):
            if print_detail and ((i % print_every == 0) or (i == n-1)):
                print(f"[DEBUG] MoELayer: initializing expert {i+1}/{n}")
            expert = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False, device=device, dtype=dtype)
            )
            expert.apply(moe_init_weights)
            self.experts.append(expert)
        if print_detail:
            print(f"[DEBUG] MoELayer: all {n} experts initialized.")
            total_params = sum(p.numel() for p in self.parameters())
            print(f"[DEBUG] MoELayer: total parameters = {total_params/1e6:.2f}M")
        else:
            print(f"[DEBUG] MoELayer: (layer {MoELayer._layer_count}) {n} experts initialized (log suppressed)")
    
    def forward(self, x):
        b, t, d = x.shape
        h = x.view(-1, d)
        scores, idx = self.gate(h)  # [B*T, 2], [B*T, 2]
        y = torch.zeros_like(h)
        
        for i in range(2):
            expert_idx = idx[:, i]  # [B*T]
            expert_scores = scores[:, i]  # [B*T]
            for e in range(self.cfg.moe_num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    y[mask] += expert_scores[mask, None] * self.experts[e](h[mask])
        return y.view(b, t, d)