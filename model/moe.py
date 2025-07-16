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
    """Expert routing gate for MoE (top-k configurable)"""
    def __init__(self, hidden_size, num_experts, top_k=2, device=None, dtype=None):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.top_k = top_k
    
    def forward(self, x):
        logits = self.gate(x)  # [N, num_experts]
        scores, idx = torch.topk(logits, self.top_k, dim=-1)  # [N, top_k]
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)  # [N, top_k]
        return scores, idx

class MoELayer(nn.Module):
    """Mixture of Experts layer (高效路由+负载均衡损失)"""
    _layer_count = 0
    def __init__(self, cfg, device=None, dtype=None, print_every=8):
        super().__init__()
        MoELayer._layer_count += 1
        self.cfg = cfg
        self.top_k = getattr(cfg, 'moe_top_k', 2)
        self.num_experts = getattr(cfg, 'moe_num_experts', 8)
        self.gate = MoEGate(cfg.hidden_size, self.num_experts, top_k=self.top_k, device=device, dtype=dtype)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False, device=device, dtype=dtype)
            ) for _ in range(self.num_experts)
        ])
        for expert in self.experts:
            expert.apply(moe_init_weights)
        if MoELayer._layer_count == 1:
            print(f"✅\tMoELayer: {self.num_experts} experts, top-{self.top_k} routing, efficient implementation.")
    
    def forward(self, x):
        b, t, d = x.shape
        h = x.view(-1, d)  # [B*T, d]
        scores, idx = self.gate(h)  # [N, top_k], [N, top_k]
        N = h.size(0)
        # Load balancing loss (aux_loss)
        # Count the probability of each expert being assigned
        mask = torch.zeros(N, self.num_experts, device=h.device)
        mask.scatter_add_(1, idx, scores)
        load = mask.sum(0) / mask.sum()
        aux_loss = (load * load.log()).sum()
        # Efficient expert assignment: batch grouping
        y = torch.zeros_like(h)
        for expert_id in range(self.num_experts):
            # Find all tokens assigned to this expert
            for k in range(self.top_k):
                sel = (idx[:, k] == expert_id)
                if sel.any():
                    h_sel = h[sel]
                    s_sel = scores[sel, k]
                    y[sel] += s_sel.unsqueeze(1) * self.experts[expert_id](h_sel)
        return y.view(b, t, d), aux_loss