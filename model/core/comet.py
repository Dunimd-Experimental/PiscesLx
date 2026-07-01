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

"""
Seirênes: Adversarial Self-Play with Evolving Distractions for LLM Reasoning
(arXiv:2605.11636, May 2026).

Transforms contextual interference from a failure mode into an internal
training signal. Single parameter-shared model acts as both adversary
(constructs distracting contexts) and solver (disambiguates core task).
Co-evolving curriculum drives robust reasoning. +7-10 points on math
reasoning benchmarks across 4B-30B scales.

Reference: Zhang et al. "Seirênes: Adversarial Self-Play with Evolving
Distractions for LLM Reasoning." arXiv:2605.11636, 2026.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


# Paper: Zhang et al. "Seirênes: Adversarial Self-Play with Evolving Distractions for LLM Reasoning." arXiv:2605.11636, 2026.
class YvCoMeTMemory(nn.Module):
    """
    Seirênes adversarial memory: maintains a pool of distracting contexts
    that target the model's current reasoning blind spots. As the model
    learns to solve harder problems, the distractions co-evolve.
    """

    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_adversarial_slots = getattr(config, 'comet_global_slots', 512)
        self.window_size = getattr(config, 'comet_window', 4096)
        self.num_heads = max(1, getattr(config, 'n_head', 16) // 2)

        self.adversarial_pool = nn.Parameter(
            torch.randn(1, self.num_adversarial_slots, self.hidden_size, device=device, dtype=dtype) * 0.02
        )

        self.distraction_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size, device=device, dtype=dtype),
            nn.Tanh(),
        )

        self.solver_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

        self.distraction_filter = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 4, 1, device=device, dtype=dtype),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)

        self.register_buffer('_adversarial_state', self.adversarial_pool.data.clone())
        self.register_buffer('_solver_progress', torch.zeros(1))
        self.register_buffer('_distraction_level', torch.zeros(1))

    def _generate_distraction(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        pool_size = min(self.num_adversarial_slots, T)
        selected = x[:, :pool_size]
        distraction = self.distraction_generator(selected)
        difficulty = torch.sigmoid(self._distraction_level)
        return difficulty * distraction + (1 - difficulty) * selected

    def _filter_distraction(self, x: torch.Tensor, memory_out: torch.Tensor) -> torch.Tensor:
        filter_gate = self.distraction_filter(memory_out)
        return filter_gate * memory_out + (1 - filter_gate) * x

    def forward(
        self,
        x: torch.Tensor,
        write_gate: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        B, T, H = x.shape

        adv_state = self._adversarial_state[:B].expand(B, -1, -1)
        attended, _ = self.solver_attention(x, adv_state, adv_state)

        filtered = self._filter_distraction(x, attended)
        out = self.output_proj(filtered)

        if update_memory and self.training:
            with torch.no_grad():
                distraction = self._generate_distraction(x)
                mix = torch.sigmoid(self._solver_progress)
                new_state = mix * distraction[:, :self.num_adversarial_slots].detach() + (1 - mix) * adv_state
                self._adversarial_state[:B] = new_state
                self._solver_progress = torch.clamp(self._solver_progress + 1e-5, 0, 1)
                self._distraction_level = torch.clamp(self._distraction_level + 5e-6, 0, 0.9)

        return out

    def read(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, update_memory=False)

    def write(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        if self.training:
            with torch.no_grad():
                distraction = self._generate_distraction(x)
                B = x.shape[0]
                adv_state = self._adversarial_state[:B]
                mix = torch.sigmoid(self._solver_progress)
                new_state = mix * distraction[:, :self.num_adversarial_slots].detach() + (1 - mix) * adv_state
                self._adversarial_state[:B] = new_state
                self._solver_progress = torch.clamp(self._solver_progress + 1e-5, 0, 1)

    def reset_memory(self):
        self._adversarial_state = self.adversarial_pool.data.clone()
        self._solver_progress.zero_()
        self._distraction_level.zero_()


# Paper: Zhang et al., "Seirênes: Adversarial Self-Play with Evolving Distractions for LLM Reasoning," arXiv:2605.11636, 2026.
class YvCoMeTLayer(nn.Module):
    """Seirênes layer with adversarial distraction filtering."""

    def __init__(self, config, layer_idx: int, device=None, dtype=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.use_memory = getattr(config, 'comet_use_memory', layer_idx % 2 == 0)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.n_head,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.attn_norm = nn.LayerNorm(self.hidden_size, device=device, dtype=dtype)

        if self.use_memory:
            self.memory = YvCoMeTMemory(config, device=device, dtype=dtype)
            self.memory_norm = nn.LayerNorm(self.hidden_size, device=device, dtype=dtype)

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size, device=device, dtype=dtype),
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.attn_norm(x)
        h, _ = self.self_attn(h, h, h, attn_mask=mask)
        x = x + h

        if self.use_memory:
            h = self.memory_norm(x)
            h = self.memory(h, update_memory=self.training)
            x = x + h

        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h
        return x
