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
        self.mode = getattr(cfg, "dual_inject_mode", "dual")
        if self.mode not in ("film", "kv", "dual", "none"):
            raise ValueError(
                f"dual_inject_mode must be one of film/kv/dual/none, got {self.mode}"
            )

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

        if self.mode == "dual":
            hidden_size = cfg.hidden_size
            self.film_gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_size // 4, 1, device=device, dtype=dtype),
                nn.Sigmoid(),
            )
        else:
            self.film_gate = None

    def _compute_alpha(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute FiLM blending weight for dual mode.

        Args:
            hidden_states: [batch, seq, hidden_size].

        Returns:
            alpha: [batch, seq, 1] with values in (0, 1).
        """
        if self.film_gate is None:
            return 0.5  # default balance when not in dual mode
        return self.film_gate(hidden_states)

    def inject(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Apply the configured knowledge injection to ``hidden_states``.

        Args:
            hidden_states: [batch, seq, hidden_size] from the 7B core.
            layer_idx: Index of the transformer layer consuming the output.

        Returns:
            A tuple ``(h_out, extra_kv)`` where ``extra_kv`` is ``None``
            unless the KV path is active. When present, ``extra_kv`` is a
            pair of tensors of shape ``[batch, n_kv_head, seq, head_dim]``.
        """
        if self.mode == "none":
            return hidden_states, None

        h = hidden_states

        if self.mode == "kv":
            # KV-only mode asymmetry: returns extra KV pairs but does NOT
            # modulate hidden_states. The extra KV pairs are consumed by the
            # attention layer via concatenation; the hidden stream passes
            # through unchanged. This is by design — in KV-only mode the
            # knowledge is injected through the attention values rather than
            # through FiLM-style activation modulation.
            extra_kv = self.memory_sep(h, layer_idx)
            return h, extra_kv

        film_params = self.subconscious.get_film_params(h, layer_idx)
        h_film = h * (1.0 + film_params["scale"]) + film_params["shift"]

        if self.mode == "film":
            return h_film, None

        # dual mode: blend FiLM-modulated and raw streams
        extra_kv = self.memory_sep(h, layer_idx)
        alpha = self._compute_alpha(h)
        beta = 1.0 - alpha
        h_dual = alpha * h_film + beta * h
        return h_dual, extra_kv

    def extra_repr(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"mode={self.mode}, params={total / 1e9:.3f}B"
