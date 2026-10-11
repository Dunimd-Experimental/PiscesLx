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

"""
Guard for the sparsity regularisers' route into the objective (docs/audit/00 A72).

Measured background: the sparse-gate regularisers used to ride inside the block's auxiliary lump,
and the declared balancing mode (`moe_aux_loss_free=True`) kept that whole lump out of the
objective, so the gates were never trained - 80 parameters unchanged after 8 steps, led by
``layers.0.attention.sparse_gate_lca``. The fix routes them through
``YvModel.sparsity_objective()`` and folds them unconditionally.

Negative control (run while writing this): re-gating that fold on ``moe_aux_loss_free`` again
left the whole stage-10 suite green, i.e. nothing guarded the fix. These two checks close that
hole - one at runtime over the walk, one over the fold's own source, because the fold is a single
expression inside ``forward`` that no behavioural test can reach without building a full model.
"""

import inspect
from typing import List, Tuple

import torch
from torch import nn

from model.core.model import YvModel

__all__ = ["PiscesLxSparsityObjectiveContract"]


class PiscesLxSparsityObjectiveContract:
    """The walk sums disjoint contributors, and the fold is not gated away."""

    WEIGHT_A = 0.25
    WEIGHT_B = 0.5
    GATE_A = 1.0
    GATE_B = 2.0
    NEGATIVE_CONTROL_MARK = "TEMPORARY NEGATIVE CONTROL"

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    def record(self, name: str, passed: bool, detail: str) -> None:
        self.results.append((name, passed, detail))

    def _stub_model(self) -> nn.Module:
        """A hierarchy with two disjoint contributors, one of them nested."""
        outer = nn.Module()
        outer.w = nn.Parameter(torch.tensor(self.GATE_A))
        outer.attn = nn.Module()
        outer.attn.w = nn.Parameter(torch.tensor(self.GATE_B))
        outer.w._sparsity_weight = self.WEIGHT_A
        outer.attn.w._sparsity_weight = self.WEIGHT_B

        def outer_loss(_self):
            return _self.w * _self.w._sparsity_weight

        def inner_loss(_self):
            return _self.w * _self.w._sparsity_weight

        outer.sparsity_loss = outer_loss.__get__(outer, nn.Module)
        outer.attn.sparsity_loss = inner_loss.__get__(outer.attn, nn.Module)
        return outer

    def check_walk_sums_disjoint_contributors(self) -> None:
        """`sparsity_objective` must total every contributor once."""
        model = self._stub_model()
        total = YvModel.sparsity_objective(model)
        expected = self.WEIGHT_A * self.GATE_A + self.WEIGHT_B * self.GATE_B
        got = None if total is None else float(total)
        self.record("walk::sums every contributor", got is not None and abs(got - expected) < 1e-6,
                    f"total={got} expected={expected}")

    def check_walk_reaches_every_gate(self) -> None:
        """Gradients must reach both contributors, not just the shallow one."""
        model = self._stub_model()
        total = YvModel.sparsity_objective(model)
        self.record("walk::returns a differentiable term",
                    isinstance(total, torch.Tensor) and total.requires_grad,
                    f"requires_grad={getattr(total, 'requires_grad', None)}")
        if isinstance(total, torch.Tensor) and total.requires_grad:
            total.backward()
        grads = {name: (p.grad is not None and float(p.grad) != 0.0)
                 for name, p in [("outer", model.w), ("nested", model.attn.w)]}
        self.record("walk::gradient reaches both", all(grads.values()), str(grads))

    def check_fold_is_not_gated_away(self) -> None:
        """The fold must not sit behind the balancing mode again.

        The fold is one expression inside ``forward``; reaching it behaviourally needs a full
        model, so the guard reads the source and asserts the two lines are adjacent and that the
        balancing flag does not appear between them.
        """
        source = inspect.getsource(YvModel.forward)
        lines = source.splitlines()
        call_at = next((i for i, l in enumerate(lines) if "self.sparsity_objective()" in l), None)
        self.record("fold::sparsity_objective is called", call_at is not None,
                    "call site found" if call_at is not None else "call site MISSING")
        if call_at is None:
            return
        window = "\n".join(lines[call_at:call_at + 3])
        folded = "loss = loss + sparsity_loss" in window
        self.record("fold::term is added to loss", folded, window.replace("\n", " | ")[:120])
        gated = "moe_aux_loss_free" in window
        self.record("fold::not gated on the balancing mode", not gated,
                    "gated again - the gates lose their gradient" if gated else
                    "unconditional, as the fix requires")

    def run(self) -> int:
        """Execute every check; non-zero exit on any failure."""
        self.check_walk_sums_disjoint_contributors()
        self.check_walk_reaches_every_gate()
        self.check_fold_is_not_gated_away()
        failed = [name for name, ok, _ in self.results if not ok]
        for name, ok, detail in self.results:
            print(f"[{'PASS' if ok else 'FAIL'}] {name:<40} {detail}")
        print(f"\n{len(self.results) - len(failed)}/{len(self.results)} checks passed")
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(PiscesLxSparsityObjectiveContract().run())
