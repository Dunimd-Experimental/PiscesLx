#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
from torch import nn
from utils.log import RIGHT
import torch.nn.functional as F
from collections import OrderedDict

# Initialize weights for MoE (Mixture of Experts) related linear layers
def moe_init_weights(m):
    """
    Initialize weights for linear layers in MoE.

    Args:
        m (torch.nn.Module): A PyTorch module.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MoEGate(nn.Module):
    """Expert routing gate for MoE (top-k configurable)"""
    def __init__(self, hidden_size, num_experts, top_k=2, device=None, dtype=None):
        """
        Initialize the MoE gate module.

        Args:
            hidden_size (int): Size of the hidden layer.
            num_experts (int): Number of experts.
            top_k (int, optional): Number of top experts to select. Defaults to 2.
            device (torch.device, optional): Device to place the module on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the module. Defaults to None.
        """
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.top_k = top_k
    
    def forward(self, x):
        """
        Forward pass of the MoE gate.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing scores and indices of the top-k experts.
        """
        logits = self.gate(x)  # [N, num_experts]
        scores, idx = torch.topk(logits, self.top_k, dim=-1)  # [N, top_k]
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)  # [N, top_k]
        return scores, idx

class MoELayer(nn.Module):
    """Mixture of Experts layer (Efficient routing+load balancing loss)"""
    _layer_count = 0
    def __init__(self, cfg, device=None, dtype=None, print_every=8, max_gpu_experts=4):
        """
        Initialize the MoE layer.

        Args:
            cfg: Configuration object.
            device (torch.device, optional): Device to place the module on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the module. Defaults to None.
            print_every (int, optional): Print interval. Defaults to 8.
            max_gpu_experts (int, optional): Maximum number of experts on GPU. Defaults to 4.
        """
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
            RIGHT(f"MoELayer: {self.num_experts} experts, top-{self.top_k} routing, efficient implementation.")

        self.max_gpu_experts = max_gpu_experts
        self._active_experts = OrderedDict()  # expert_id: last_used_step
        self._step = 0

    def _move_expert_to_gpu(self, expert_id):
        """
        Move an expert to GPU and manage the LRU cache of active experts.

        Args:
            expert_id (int): ID of the expert to move to GPU.
        """
        expert = self.experts[expert_id]
        if next(expert.parameters()).device.type != 'cuda':
            expert.to('cuda')
        self._active_experts[expert_id] = self._step
        
        if len(self._active_experts) > self.max_gpu_experts:
            lru_expert_id, _ = self._active_experts.popitem(last=False)
            self._move_expert_to_cpu(lru_expert_id)

    def _move_expert_to_cpu(self, expert_id):
        """
        Move an expert to CPU.

        Args:
            expert_id (int): ID of the expert to move to CPU.
        """
        expert = self.experts[expert_id]
        if next(expert.parameters()).device.type != 'cpu':
            expert.to('cpu')

    def forward(self, x):
        """
        Forward pass of the MoE layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the output tensor and the auxiliary loss.
        """
        b, t, d = x.shape
        h = x.view(-1, d)  # [B*T, d]
        scores, idx = self.gate(h)  # [N, top_k], [N, top_k]
        N = h.size(0)
        # Load balancing loss (aux_loss)
        mask = torch.zeros(N, self.num_experts, device=h.device)
        mask.scatter_add_(1, idx, scores)
        load = mask.sum(0) / mask.sum()
        aux_loss = (load * (load + 1e-9).log()).sum()
        
        if self.num_experts > 8 and h.device.type == 'cuda':
            needed_experts = set(idx.cpu().numpy().flatten().tolist())
            for expert_id in needed_experts:
                self._move_expert_to_gpu(expert_id)
        # Efficient expert assignment: batch grouping
        y = torch.zeros_like(h)
        for expert_id in range(self.num_experts):
            # Find all tokens assigned to this expert
            for k in range(self.top_k):
                sel = (idx[:, k] == expert_id)
                if sel.any():
                    expert = self.experts[expert_id]
                    if next(expert.parameters()).device.type != h.device.type:
                        expert.to(h.device)
                    h_sel = h[sel]
                    s_sel = scores[sel, k]
                    y[sel] += s_sel.unsqueeze(1) * expert(h_sel)
        self._step += 1
        return y.view(b, t, d), aux_loss