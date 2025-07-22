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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def moe_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ExpertChoiceRouter(nn.Module):
    """动态Expert Choice路由（无外部依赖）
    参考SigLIP 2的路由优化策略，实现更高效的专家选择机制
    """
    def __init__(self, hidden_size, num_experts, capacity_factor=1.25, top_k=2):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
        self.top_k = top_k
        
    def forward(self, x):
        # x: (batch*seq, hidden_size)
        tokens_per_expert = int(x.shape[0] * self.capacity_factor / self.num_experts)
        
        # 计算路由分数
        logits = self.gate(x)  # (tokens, experts)
        
        # Expert Choice: 每个专家选择top-k tokens
        expert_indices = torch.topk(logits.T, tokens_per_expert, dim=1).indices  # (experts, capacity)
        
        # 构建路由矩阵
        dispatch_mask = torch.zeros_like(logits)
        dispatch_mask[expert_indices.T.flatten(), torch.arange(self.num_experts).repeat(tokens_per_expert)] = 1
        
        # 计算专家负载
        expert_load = dispatch_mask.sum(dim=0)
        
        # 计算负载均衡损失
        load_balancing_loss = (expert_load * torch.log(expert_load / expert_load.mean())).mean()
        
        return expert_indices, dispatch_mask, load_balancing_loss


class DynamicMoELayer(nn.Module):
    _layer_count = 0
    def __init__(self, cfg, device=None, dtype=None):
        super().__init__()
        DynamicMoELayer._layer_count += 1
        self.cfg = cfg
        self.top_k = getattr(cfg, 'moe_top_k', 2)
        self.num_experts = getattr(cfg, 'moe_num_experts', 8)
        self.router = ExpertChoiceRouter(
            cfg.hidden_size, 
            self.num_experts, 
            capacity_factor=getattr(cfg, 'moe_capacity_factor', 1.25),
            top_k=self.top_k
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False, device=device, dtype=dtype)
            ) for _ in range(self.num_experts)
        ])
        for expert in self.experts:
            expert.apply(moe_init_weights)
        
        # 专家设备管理
        self.max_gpu_experts = getattr(cfg, 'max_gpu_experts', 4)
        self._active_experts = OrderedDict()  # expert_id: last_used_step
        self._step = 0
        
        if DynamicMoELayer._layer_count == 1:
            print(f"✅	DynamicMoELayer: {self.num_experts} experts, top-{self.top_k} routing, capacity_factor={self.router.capacity_factor}")
    
    def _move_expert_to_gpu(self, expert_id):
        expert = self.experts[expert_id]
        if next(expert.parameters()).device.type != 'cuda':
            expert.to('cuda')
        self._active_experts[expert_id] = self._step
        
        if len(self._active_experts) > self.max_gpu_experts:
            lru_expert_id, _ = self._active_experts.popitem(last=False)
            self._move_expert_to_cpu(lru_expert_id)
    
    def _move_expert_to_cpu(self, expert_id):
        expert = self.experts[expert_id]
        if next(expert.parameters()).device.type != 'cpu':
            expert.to('cpu')
    
    def forward(self, x):
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        
        expert_indices, dispatch_mask, load_balancing_loss = self.router(x_flat)
        
        # 动态路由计算
        outputs = torch.zeros_like(x_flat)
        
        # 专家设备管理
        if self.num_experts > 8 and x.device.type == 'cuda':
            needed_experts = set()
            for expert_id in range(self.num_experts):
                if expert_indices[expert_id].numel() > 0:
                    needed_experts.add(expert_id)
            for expert_id in needed_experts:
                self._move_expert_to_gpu(expert_id)
        
        # 专家计算
        for expert_id, expert in enumerate(self.experts):
            tokens = x_flat[expert_indices[expert_id]]
            if tokens.shape[0] > 0:
                expert_out = expert(tokens)
                outputs[expert_indices[expert_id]] += expert_out
        
        self._step += 1
        return outputs.view(batch_size, seq_len, hidden), load_balancing_loss