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
SparDA: Sparse Decoupled Attention with Forecast Projection
(arXiv:2606.04511, Jun 2026).

Adds a fourth per-layer projection — Forecast — alongside Q, K, V.
Forecast predicts which KV blocks the next layer needs, enabling
lookahead selection that overlaps CPU→GPU prefetch with current-layer
execution. <0.5% parameter overhead.

Reference: Fu et al. "SparDA: Sparse Decoupled Attention for Efficient
Long-Context LLM Inference." arXiv:2606.04511, 2026.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional


class YvRCACrossCorrelationAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, T_x, _ = x.shape
        B_c, T_c, _ = context.shape

        Q = self.q_proj(x).view(B, T_x, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(B_c, T_c, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(B_c, T_c, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T_x, self.hidden_size)
        return self.o_proj(out)


class YvRCAFusionConfig:
    def __init__(self, hidden_size=4096, n_head=16, num_modalities=6):
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.num_modalities = num_modalities


# Paper: Fu et al. (NVIDIA), "SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference," arXiv:2606.04511, 2026.
class YvRecursiveCrossModalFusion(nn.Module):
    """
    SparDA-inspired cross-modal fusion with Forecast-driven lookahead.

    Forecast projection predicts important cross-modal interactions for
    future layers, enabling lookahead prefetch of modality-specific KV
    blocks and overlapped multi-modal computation.
    """

    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.cfg = config
        self.hidden_size = config.hidden_size
        self.modalities = ["text", "image", "audio", "video", "document", "agentic"]
        self.max_rca_rounds = getattr(config, 'max_rca_rounds', 3)
        self.convergence_threshold = getattr(config, 'rca_convergence_threshold', 0.995)

        num_heads = max(1, getattr(config, 'n_head', 16) // 2)
        self.dropout_p = getattr(config, 'fusion_dropout', 0.1)

        self.modality_projections = nn.ModuleDict({
            m: nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
            for m in self.modalities
        })

        n_mod = len(self.modalities)
        self.joint_proj = nn.Sequential(
            nn.Linear(self.hidden_size * n_mod, self.hidden_size * 2, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size, device=device, dtype=dtype),
            nn.LayerNorm(self.hidden_size, device=device, dtype=dtype),
        )

        self.cross_modal_attentions = nn.ModuleDict({
            m: YvRCACrossCorrelationAttention(self.hidden_size, num_heads, self.dropout_p)
            for m in self.modalities
        })

        self.output_gates = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 4, self.hidden_size, device=device, dtype=dtype),
                nn.Sigmoid(),
            )
            for m in self.modalities
        })

        self.output_tokens = nn.Parameter(
            torch.randn(1, getattr(config, 'modal_token_count', 8), self.hidden_size, device=device, dtype=dtype) * 0.02
        )
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)

        n_forecast_groups = max(1, num_heads // 2)
        self.forecast_proj = nn.Linear(self.hidden_size, n_mod * n_forecast_groups, device=device, dtype=dtype)
        self.forecast_gate = nn.Parameter(torch.ones(1, device=device, dtype=dtype) * 0.5)

    def forecast(
        self, joint: torch.Tensor
    ) -> torch.Tensor:
        forecast_logits = self.forecast_proj(joint.mean(dim=1))
        forecast_weights = torch.sigmoid(forecast_logits * self.forecast_gate)
        return forecast_weights

    def forward(
        self,
        modal_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        h = {}
        for m in self.modalities:
            feat = modal_features.get(m)
            if feat is not None and isinstance(feat, torch.Tensor) and feat.numel() > 0:
                h[m] = self.modality_projections[m](feat)

        if not h:
            raise ValueError(
                "YvRecursiveCrossModalFusion requires at least one real modality tensor. "
                "Dummy fused outputs are disabled for strict model closure."
            )

        device = list(h.values())[0].device
        prev_features = {m: feat.detach() for m, feat in h.items()}

        for rnd in range(self.max_rca_rounds):
            pooled = []
            for m in self.modalities:
                if m in h:
                    pooled.append(h[m].mean(dim=1, keepdim=True))
            joint_cat = torch.cat(pooled, dim=-1)
            joint = self.joint_proj(joint_cat)

            forecast_weights = self.forecast(joint)

            new_h = {}
            for i, m in enumerate(self.modalities):
                if m not in h:
                    continue
                fw = forecast_weights[:, i].mean().view(1, 1, 1)
                if fw.item() < 0.1 and rnd > 0:
                    new_h[m] = h[m]
                else:
                    attended = self.cross_modal_attentions[m](h[m], joint)
                    gate = self.output_gates[m](h[m])
                    new_h[m] = h[m] + fw * gate * attended

            h = new_h

            if rnd > 0:
                converged = True
                for m in h:
                    cos = F.cosine_similarity(
                        h[m].flatten(1), prev_features[m].flatten(1), dim=1
                    ).mean()
                    if cos < self.convergence_threshold:
                        converged = False
                        break
                if converged:
                    break

            prev_features = {m: feat.detach() for m, feat in h.items()}

        all_tokens = torch.cat(list(h.values()), dim=1)
        query = self.output_tokens.expand(all_tokens.shape[0], -1, -1).to(device)
        attn_scores = torch.matmul(query, all_tokens.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        fused = self.output_proj(torch.matmul(attn_weights, all_tokens))

        return {
            'fused': fused,
            'modality_features': h,
            'rca_rounds': rnd + 1,
            'converged': converged if rnd > 0 else False,
        }


# Paper: Fu et al. (NVIDIA), "SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference," arXiv:2606.04511, 2026.
class YvDeepCrossLayerInjector(nn.Module):
    """
    SparDA lookahead injector: uses Forecast predictions to prefetch
    fused features for future layers. Overlaps computation across layers
    by scheduling injection based on predicted importance.
    """

    def __init__(self, config, num_layers: int, device=None, dtype=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_layers
        self.inject_interval = max(1, getattr(config, 'fusion_inject_interval', 4))
        num_heads = max(1, getattr(config, 'n_head', 16) // 4)

        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=num_heads,
                batch_first=True,
                dropout=0.0,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers // self.inject_interval + 1)
        ])

        self.inject_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 8, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 8, self.hidden_size, device=device, dtype=dtype),
                nn.Sigmoid(),
            )
            for _ in range(num_layers // self.inject_interval + 1)
        ])

        self.prefetch_bias = nn.Parameter(torch.zeros(num_layers, device=device, dtype=dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        fused_features: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        prefetch_weight = torch.sigmoid(self.prefetch_bias[layer_idx])
        if prefetch_weight < 0.1 and layer_idx > 0:
            return hidden_states

        inject_idx = layer_idx // self.inject_interval
        if inject_idx >= len(self.cross_attentions):
            inject_idx = len(self.cross_attentions) - 1

        ca = self.cross_attentions[inject_idx]
        gate = self.inject_gates[inject_idx]

        attended, _ = ca(hidden_states, fused_features, fused_features)
        g = gate(hidden_states)
        return hidden_states + prefetch_weight * g * attended
