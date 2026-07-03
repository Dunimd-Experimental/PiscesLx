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
from typing import Dict, List, Optional, Tuple


class YvKnowledgeExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, device=None, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(hidden_dim, input_dim, bias=False, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()


class YvKnowledgeExpertPool(nn.Module):
    def __init__(self, cfg, device=None, dtype=None):
        super().__init__()
        self.num_experts = cfg.knowledge_num_experts
        self.expert_input_dim = cfg.knowledge_expert_input_dim
        self.expert_hidden_dim = cfg.knowledge_expert_hidden_dim
        self.expert_path = getattr(cfg, 'knowledge_expert_path', 'knowledge_experts')
        self._cache: Dict[int, YvKnowledgeExpert] = {}
        self._cache_max = getattr(cfg, 'knowledge_expert_cache_size', 8)
        self._device = device
        self._dtype = dtype

    def load_expert(self, idx: int) -> YvKnowledgeExpert:
        if idx not in self._cache:
            expert = self._load_from_disk(idx)
            expert.freeze()
            expert.to(device=self._device, dtype=self._dtype)
            self._cache[idx] = expert
            if len(self._cache) > self._cache_max:
                self._evict_oldest()
        return self._cache[idx]

    def _load_from_disk(self, idx: int) -> YvKnowledgeExpert:
        expert = YvKnowledgeExpert(self.expert_input_dim, self.expert_hidden_dim)
        state_path = f"{self.expert_path}/expert_{idx}.pt"
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        expert.load_state_dict(state)
        return expert

    def _evict_oldest(self):
        key = next(iter(self._cache))
        self._cache[key].to('cpu')
        del self._cache[key]

    def forward(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        batch, seq_len, top_k = indices.shape
        output = torch.zeros_like(x)
        device = x.device
        unique_indices = torch.unique(indices).tolist()
        loaded = {idx: self.load_expert(idx) for idx in unique_indices}
        with torch.no_grad():
            for idx in unique_indices:
                mask = (indices == idx)
                if not mask.any():
                    continue
                positions = torch.nonzero(mask)
                rows = x[positions[:, 0], positions[:, 1]]
                expert_out = loaded[idx](rows)
                for p in range(positions.shape[0]):
                    b, s, k = positions[p].tolist()
                    output[b, s] += weights[b, s, k] * expert_out[p]
        return output

    def clear_cache(self):
        for expert in self._cache.values():
            expert.to('cpu')
        self._cache.clear()
