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

"""Dual-path knowledge injector for the Yv compute stream.

This module implements the single top-level class that wires the subconscious
FiLM path and the memory-separation KV path into the 7B transformer layers.
The injector is always instantiated when ``use_dual_inject=True`` and its
output is consumed by every transformer block.

Architecture:
    - FiLM path (soft injection): ``h_new = h * (1 + scale) + shift``
      produced by :class:`YvSubconsciousSystem`.
    - KV path (hard injection): extra key/value tensors produced by
      :class:`YvMemorySeparationLayer` and concatenated with the standard
      attention KV tensors.
    - Dual mode: a routing gate outputs ``alpha`` and ``beta = 1 - alpha``
      that blend the FiLM-modulated stream with the raw stream.

The injector never emits extra tokens and never writes the extra KV pairs
into the rolling KV cache.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .subconscious import YvSubconsciousSystem
from .memory_attention import YvMemorySeparationLayer


# Paper: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", NeurIPS 2017 (FiLM conditioning)
class YvDualInjector(nn.Module):
    """Top-level dual injector combining FiLM and KV knowledge paths.

    Args:
        cfg: Model configuration. Expected fields include ``hidden_size``,
            ``n_layer``, ``subconscious_*`` and ``dual_inject_mode``.
        device: Device for parameter initialization.
        dtype: Data type for parameter initialization.

    Attributes:
        mode: One of ``"film"``, ``"kv"``, ``"dual"`` or ``"none"``.
        subconscious: The 0.5B dynamic head + 314B implicit knowledge field.
        memory_sep: Per-layer projection producing extra KV pairs.
        film_gate: Optional gating network used in ``"dual"`` mode.
    """

    def __init__(
        self,
        cfg,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.cfg = cfg

        _sc_kw = lambda key, default: getattr(cfg, f"subconscious_{key}", default)

        self.subconscious = YvSubconsciousSystem(
            hidden_size=cfg.hidden_size,
            num_layers=cfg.n_layer,
            knowledge_dim=_sc_kw("knowledge_dim", 256),
            num_codebooks=_sc_kw("num_codebooks", 16),
            codebook_size=_sc_kw("codebook_size", 131072),
            codebook_dim=_sc_kw("codebook_dim", 128),
            num_field_heads=_sc_kw("num_field_heads", 8),
            head_dim=_sc_kw("head_dim", 1024),
            head_num_layers=_sc_kw("head_num_layers", 2),
            head_num_attn_heads=_sc_kw("head_num_attn_heads", 4),
            device=device,
            dtype=dtype,
        )

        self.memory_sep = YvMemorySeparationLayer(
            cfg,
            num_layers=cfg.n_layer,
            device=device,
            dtype=dtype,
        )

        hidden_size = cfg.hidden_size
        self.film_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1, device=device, dtype=dtype),
            nn.Sigmoid(),
        )

    def inject(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Dual-path knowledge injection: FiLM + KV separation.

        Always blends the subconscious FiLM-modulated stream with the
        raw hidden stream (dual mode), and always produces extra KV pairs
        from the memory separation layer.

        Args:
            hidden_states: [batch, seq, hidden_size] from the 7B core.
            layer_idx: Index of the transformer layer consuming the output.

        Returns:
            ``(h_out, extra_kv)`` where ``extra_kv`` is a pair of tensors
            of shape ``[batch, n_kv_head, seq, head_dim]``.
        """
        extra_kv = self.memory_sep(hidden_states, layer_idx)
        film_params = self.subconscious.get_film_params(hidden_states, layer_idx)
        h_film = hidden_states * (1.0 + film_params["scale"]) + film_params["shift"]
        alpha = self.film_gate(hidden_states)
        h_dual = alpha * h_film + (1.0 - alpha) * hidden_states
        return h_dual, extra_kv

    def extra_repr(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"params={total / 1e9:.3f}B"
