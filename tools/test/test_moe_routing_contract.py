#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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

"""Regression checks for the MoE routing contract (§6.4).

Every gate hands the layers ``(weights, indices)`` shaped ``[tokens, top_k]``: the
fused dispatch builds its token table from that shape and the non-fused path
reconstructs token positions from it. A gate that compacts its assignments into a
variable-length 1-D list makes the dispatch index past the token table (IndexError)
and, on the non-fused path, pairs tokens with the wrong expert *silently*.
"""

import torch
import torch.nn as nn

from model.config import YvConfig
from model.moe.gate import YvStableMoEGate
from model.moe.layer import YvDeepSeekMoELayer, YvDynamicMoELayer
from opss.infer.fused_moe import PiscesLxFusedMoEDispatch

HIDDEN = 32


class PiscesLxMoeRoutingContractTest:
    """Assert the routing shapes, the capacity semantics and the dispatch agreement."""

    TOKENS = 24
    RESULTS = []

    @classmethod
    def _gate(cls) -> YvStableMoEGate:
        gate = YvStableMoEGate(hidden_size=HIDDEN, num_experts=4, top_k=2,
                             capacity_factor=1.0, min_capacity=1)
        gate.train()
        return gate

    @classmethod
    def _tokens(cls) -> torch.Tensor:
        """Gate input as [batch, seq, hidden]: the shape the layers hand over."""
        return torch.randn(1, cls.TOKENS, HIDDEN)

    @classmethod
    def check_shapes_are_the_contract(cls) -> None:
        gate = cls._gate()
        x = cls._tokens()
        weights, indices, _ = gate(x)
        cls._expect(weights.dim() == 2 and weights.shape[0] == cls.TOKENS,
                    "gate returns one row per token")
        cls._expect(indices.shape == weights.shape,
                    "indices and weights share the [tokens, top_k] shape")

    @classmethod
    def check_weights_are_a_distribution(cls) -> None:
        gate = cls._gate()
        weights, _, _ = gate(cls._tokens())
        totals = weights.sum(dim=-1)
        cls._expect(bool(torch.all((totals <= 1.0 + 1e-5) | (totals == 0))),
                    "each token's routing weights sum to at most one")
        cls._expect(bool(torch.all(weights >= 0)), "routing weights are non-negative")

    @classmethod
    def check_capacity_drops_by_zeroing(cls) -> None:
        gate = cls._gate()
        x = cls._tokens()
        with torch.no_grad():
            # Drive the routing projection so every token prefers expert 0: only the
            # capacity rule can keep that expert from being swamped.
            gate.gate.weight.zero_()
            gate.gate.weight[0] = 1.0
        weights, indices, _ = gate(x)
        # The gate sizes capacity from its own predicted load, not the configured
        # factor alone, so the bound under test comes from the gate itself.
        factor = gate._calculate_dynamic_capacity(x.reshape(-1, HIDDEN))
        capacity = max(int(cls.TOKENS * float(factor) / gate.num_experts), gate.min_capacity)
        served = ((indices == 0) & (weights > 0)).sum().item()
        cls._expect(served <= capacity,
                    f"expert 0 served {served} tokens but capacity is {capacity}")
        collapsed = (indices == 0).any(dim=-1).sum().item()
        cls._expect(collapsed > served,
                    "over-capacity slots were zeroed rather than dropped from the table")
        totals = weights.sum(dim=-1)
        kept = totals > 0
        cls._expect(bool(torch.allclose(totals[kept], torch.ones_like(totals[kept]),
                                        atol=1e-5)),
                    "a served token's weights still form a distribution")

    @classmethod
    def check_dispatch_accepts_the_gate(cls) -> None:
        gate = cls._gate()
        x = cls._tokens().view(cls.TOKENS, HIDDEN)
        weights, indices, _ = gate(cls._tokens())
        experts = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(4)])
        dispatch = PiscesLxFusedMoEDispatch(num_experts=4, top_k=2)
        out = dispatch(x, weights, indices, experts)
        cls._expect(out.shape == x.shape, "fused dispatch returns one row per token")
        cls._expect(bool(torch.isfinite(out).all()), "dispatch output is finite")

    @classmethod
    def check_layers_run_fused_and_plain(cls) -> None:
        cfg = YvConfig()
        cfg.hidden_size = HIDDEN
        cfg.intermediate_size = 64
        cfg.moe_num_experts = 4
        cfg.moe_top_k = 2
        cfg.use_fused_moe = True
        cfg.diversity_weight = 0.0
        x = torch.randn(2, cls.TOKENS // 2, HIDDEN)
        for cls_name, layer_cls in (("dynamic", YvDynamicMoELayer), ("deepseek", YvDeepSeekMoELayer)):
            layer = layer_cls(cfg, device=torch.device("cpu"), dtype=torch.float32)
            layer.use_fused_moe = True
            from opss.infer.fused_moe import PiscesLxFusedMoEDispatch as Dispatch
            layer.fused_dispatch = Dispatch(num_experts=layer.num_experts, top_k=layer.top_k)
            out, _ = layer(x)
            cls._expect(out.shape == x.shape, f"{cls_name} MoE layer keeps the input shape")
            layer.fused_dispatch = None
            layer.use_fused_moe = False
            out_plain, _ = layer(x)
            cls._expect(bool(torch.isfinite(out_plain).all()),
                        f"{cls_name} MoE non-fused path stays finite")

    @classmethod
    def check_path_gate_recomputes_when_shape_changes(cls) -> None:
        """PathMoE's shared assignment must never be replayed onto another batch."""
        from model.moe.gate import YvPathMoEGate

        inner = cls._gate()
        shared = YvPathMoEGate(inner, stage_size=4)
        shared.train()
        small = torch.randn(1, 5, HIDDEN)
        big = torch.randn(1, cls.TOKENS, HIDDEN)
        _, idx_small, _ = shared(small)
        _, idx_big, _ = shared(big)
        cls._expect(int(idx_big.shape[0]) == cls.TOKENS,
                    "the stage re-routes when the token count changes instead of "
                    f"handing {idx_small.shape[0]} cached rows to {cls.TOKENS} tokens")
        # Within one batch the stage still shares a single assignment.
        first, _, _ = shared(big)
        second, _, _ = shared(big)
        cls._expect(bool(torch.equal(first, second)),
                    "layers inside a stage keep sharing one routing decision")

    @classmethod
    def _expect(cls, condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    @classmethod
    def run(cls) -> int:
        checks = (
            cls.check_shapes_are_the_contract,
            cls.check_weights_are_a_distribution,
            cls.check_capacity_drops_by_zeroing,
            cls.check_dispatch_accepts_the_gate,
            cls.check_layers_run_fused_and_plain,
            cls.check_path_gate_recomputes_when_shape_changes,
        )
        failures = 0
        for check in checks:
            try:
                check()
                print(f"[PASS] {check.__name__}")
            except Exception as exc:
                failures += 1
                print(f"[FAIL] {check.__name__}: {type(exc).__name__}: {exc}")
        print(f"{len(checks) - failures}/{len(checks)} checks passed")
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(PiscesLxMoeRoutingContractTest.run())
