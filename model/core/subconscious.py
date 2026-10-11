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
from typing import Dict, Optional, Tuple


class YvKnowledgeRouter(nn.Module):
    def __init__(self, cfg, device=None, dtype=None):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        # Width of the router's expert-input projection. It used to be sized to match the
        # disk-backed expert pool's input; with that pool removed (docs/audit A26) the
        # projection only feeds `output_proj`, so it collapses to the knowledge width.
        self.expert_input_dim = cfg.subconscious_knowledge_dim
        self.knowledge_dim = cfg.subconscious_knowledge_dim
        self.num_experts = cfg.knowledge_num_experts
        self.top_k = cfg.subconscious_expert_top_k

        self.input_proj = nn.Linear(self.hidden_size, self.expert_input_dim, bias=False, device=device, dtype=dtype)
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False, device=device, dtype=dtype)
        self.output_proj = nn.Linear(self.expert_input_dim, self.knowledge_dim, bias=False, device=device, dtype=dtype)
        # The RL phase scores a state with this head. It is part of the router's declared
        # architecture: attaching it later, from outside, would make the parameter set (and
        # therefore the saved checkpoint's key set) depend on which code path ran first.
        value_width = int(cfg.subconscious_value_head_width)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, value_width, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(value_width, 1, device=device, dtype=dtype),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.router(h)
        top_k = min(self.top_k, self.num_experts)
        top_k_logits, top_k_indices = torch.topk(router_logits, top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits.float(), dim=-1).to(dtype=router_logits.dtype)
        expert_input = self.input_proj(h)
        return expert_input, top_k_weights, top_k_indices, router_logits

    def compute_knowledge_vector(self, h: torch.Tensor) -> torch.Tensor:
        expert_input = self.input_proj(h)
        return self.output_proj(expert_input)


class YvSubconsciousState(nn.Module):
    def __init__(self, cfg, device=None, dtype=None):
        super().__init__()
        self.state_dim = cfg.subconscious_state_dim
        self.knowledge_dim = cfg.subconscious_knowledge_dim
        self.hidden_size = cfg.hidden_size

        combined_dim = self.state_dim * 3
        self.gru = nn.Linear(combined_dim, self.state_dim, bias=False, device=device, dtype=dtype)
        self.hidden_proj = nn.Linear(self.hidden_size, self.state_dim, bias=False, device=device, dtype=dtype)
        self.knowledge_proj = nn.Linear(self.knowledge_dim, self.state_dim, bias=False, device=device, dtype=dtype)

    @staticmethod
    def _masked_mean(x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Mean over the real (unpadded) positions of ``x [B, T, d]``.

        Both the FiLM summary and the recurrent state used to average over *every* position,
        padding included, and the result then modulated every position — so the content of
        masked-out slots changed the logits of real tokens (measured leak 0.43 at real
        positions, positive control 0.42; docs/audit A54).
        """
        if attention_mask is None:
            return x.mean(dim=1, keepdim=True)
        weights = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * weights).sum(dim=1, keepdim=True) / denominator

    def forward(self, s_prev: Optional[torch.Tensor], h: torch.Tensor, k_t: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_pooled = self._masked_mean(h, attention_mask)
        h_proj = self.hidden_proj(h_pooled)
        k_proj = self.knowledge_proj(k_t.mean(dim=1, keepdim=True))
        if s_prev is None:
            s_prev = torch.zeros(h_pooled.size(0), 1, self.state_dim, device=h.device, dtype=h.dtype)
        gate_input = torch.cat([h_proj, k_proj, s_prev], dim=-1)
        s_new = torch.tanh(self.gru(gate_input))
        return s_new


class YvFiLMGenerator(nn.Module):
    def __init__(self, cfg, layer_idx: int, device=None, dtype=None):
        super().__init__()
        self.state_dim = cfg.subconscious_state_dim
        self.hidden_size = cfg.hidden_size
        self.gamma = nn.Linear(self.state_dim, self.hidden_size, bias=False, device=device, dtype=dtype)
        self.beta = nn.Linear(self.state_dim, self.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, s_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gamma(s_t), self.beta(s_t)


class YvSubconsciousSystem(nn.Module):
    def __init__(self, cfg, device=None, dtype=None):
        super().__init__()
        self.router = YvKnowledgeRouter(cfg, device, dtype)
        self.state_evolver = YvSubconsciousState(cfg, device, dtype)
        self.film_generators = nn.ModuleList([
            YvFiLMGenerator(cfg, i, device, dtype) for i in range(cfg.n_layer)
        ])
        self._current_s: Optional[torch.Tensor] = None
        self._current_k_t: Optional[torch.Tensor] = None

    def get_film_params(self, h: torch.Tensor, layer_idx: int,
                        attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if layer_idx == 0:
            expert_input, _top_k_weights, _top_k_indices, _ = self.router(h)
            # The knowledge pool is the model's own `YvSubconsciousBlockField`, applied once per
            # forward in the trunk — not a second, disk-loaded pool here. A file-backed frozen
            # (no_grad) pool is a vector store, which the P0 rule forbids: the activated blocks
            # must be model weights that train. That parallel implementation was removed
            # (docs/audit A26).
            # Pooled over real positions only: the summary used to include masked slots, and
            # the FiLM it produces modulates every position (docs/audit A54).
            expert_summary = YvSubconsciousState._masked_mean(expert_input, attention_mask)
            k_t = self.router.output_proj(expert_summary)
            self._current_k_t = k_t
            # The state is carried across samples, so the incoming copy is detached:
            # the previous step's graph is freed rather than chained into this one
            # (truncated backprop through the recurrent state). Inside this forward the
            # layers below still read the live tensor, so the state evolver keeps its
            # gradient.
            carried = None if self._current_s is None else self._current_s.detach()
            s_t = self.state_evolver(carried, h, k_t, attention_mask)
            self._current_s = s_t
        gamma, beta = self.film_generators[layer_idx](self._current_s)
        return {"scale": gamma, "shift": beta}

    def forward(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        film_params = self.get_film_params(h, layer_idx)
        return h * (1.0 + film_params["scale"]) + film_params["shift"]

    def get_router_aux_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self._current_s.device if self._current_s is not None else 'cpu')

    def clear_cache(self):
        self._current_s = None
        self._current_k_t = None
