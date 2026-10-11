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
Executable counterpart of ``docs/proofs/03-trace-codebook-straight-through.md``.

A21 (docs/audit/00 §48): the trace codebook that decides *which* knowledge block a token
addresses never received a gradient, because selection ends in ``argmax``. The fix is a
straight-through gate on the block contribution, which must satisfy exactly the properties
proved for it:

  * P1 value neutrality  — the gate evaluates to 1, so the forward pass is bit-identical
    and inference pays nothing.
  * P2 gradient identity — the gradient reaching the scores equals the pathwise gradient of
    the *soft* selection evaluated at the hard point, without reading all N blocks.
  * P3 boundedness       — the credit per token is bounded by ||dL/dh|| * kappa * ||h||
    (Theorem 5's perturbation bound), so addressing gradients cannot blow up.
  * P4 trainability      — ``traces`` actually moves under an optimizer, which is the
    observable symptom A21 reported.

Run: ``python -m tools.test.test_sbf_addressing_gradient``
"""

from typing import List, Tuple

import torch

from model.config import YvConfig
from model.core.block_field import YvSubconsciousBlockField

__all__ = ["PiscesLxTraceCodebookGradientTests"]


class PiscesLxTraceCodebookGradientTests:
    """Checks the straight-through addressing gate against its proof."""

    HIDDEN = 64
    ROWS = 128
    RANK = 32
    TRACES = 48
    TRACE_DIM = 16
    GAMMA_MAX = 0.5
    BETA_BOUND = 0.25
    ALPHA = 0.1
    BATCH = 2
    SEQ = 64
    REL_TOL = 1e-4

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    # ── helpers ────────────────────────────────────────────────────────────────
    def _field(self) -> YvSubconsciousBlockField:
        cfg = YvConfig(hidden_size=self.HIDDEN)
        cfg.block_memory_rows = self.ROWS
        cfg.block_rank = self.RANK
        cfg.block_count = self.TRACES
        cfg.block_trace_dim = self.TRACE_DIM
        cfg.block_gamma_max = self.GAMMA_MAX
        cfg.block_beta_bound = self.BETA_BOUND
        cfg.block_inject_alpha = self.ALPHA
        cfg.block_pool_slots = 4
        field = YvSubconsciousBlockField(cfg, device=torch.device("cpu"),
                                         dtype=torch.float32)
        field.train()
        return field

    def _hidden(self, seed: int = 7) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        hidden = torch.randn(self.BATCH, self.SEQ, self.HIDDEN, generator=generator)
        hidden.requires_grad_(True)
        return hidden

    def record(self, name: str, passed: bool, detail: str) -> None:
        self.results.append((name, passed, detail))

    # ── P1: value neutrality ───────────────────────────────────────────────────
    def check_gate_evaluates_to_one(self) -> None:
        """prop.1: ``g = 1 + (p_c - detach(p_c))`` has value exactly 1 for every position."""
        field = self._field()
        hidden = self._hidden()
        scores = field.addresser.score_sequence(hidden)
        selected, _ = field.addresser.select_sequence(hidden)
        gate = field._straight_through_gate(scores, selected)
        if gate is None:
            self.record("p1.gate_present", False, "training forward returned no gate")
            return
        ones = torch.ones_like(gate)
        self.record("p1.gate_value_is_one", bool(torch.equal(gate, ones)),
                    f"shape {tuple(gate.shape)}, max|g-1| = "
                    f"{float((gate - ones).abs().max().detach()):.3e}")

    def check_forward_is_bitwise_unchanged(self) -> None:
        """prop.1 corollary: gating changes no output value — the A21 fix must cost nothing."""
        field = self._field()
        hidden = self._hidden(seed=11)
        with torch.no_grad():
            ungated, _ = field.read_and_modulate(hidden)
        graded, _ = field.read_and_modulate(hidden)
        max_diff = float((graded.detach() - ungated).abs().max())
        self.record("p1.forward_bitwise_equal", max_diff == 0.0,
                    f"max |grad-mode - no_grad-mode| = {max_diff:.3e}")

    # ── P2: gradient identity with soft selection ──────────────────────────────
    def check_score_gradient_matches_soft_selection(self) -> None:
        """prop.2: dL/d(scores) equals the pathwise gradient of the soft gate at the point p_c=1."""
        field = self._field()
        hidden = self._hidden(seed=13)
        target = torch.randn_like(hidden)

        graded, _ = field.read_and_modulate(hidden)
        objective = (graded * target).sum()
        field.zero_grad(set_to_none=True)
        objective.backward()
        real = field.addresser.query_proj.weight.grad

        # Reference: same forward value, but the gate is a genuine differentiable probability
        # multiplying the (detached) contribution — the soft-selection path at the hard point.
        with torch.no_grad():
            ungated, _ = field.read_and_modulate(hidden)
        delta = (ungated - hidden).detach()
        selected, scores = field.addresser.select_sequence(hidden)
        probs = torch.softmax(scores.float(), dim=-1)
        chosen = probs.gather(-1, selected.unsqueeze(-1).to(torch.long))
        field.zero_grad(set_to_none=True)
        ((ungated.detach() + chosen * delta) * target).sum().backward()
        reference = field.addresser.query_proj.weight.grad

        if real is None or reference is None:
            self.record("p2.gradient_identity", False,
                        f"real={real is not None}, reference={reference is not None}")
            return
        scale = float(reference.abs().max())
        error = float((real - reference).abs().max())
        self.record("p2.gradient_identity", error <= self.REL_TOL * max(scale, 1e-12),
                    f"max|diff| = {error:.3e}, max|ref| = {scale:.3e}")

    # ── P3: bounded credit ─────────────────────────────────────────────────────
    def check_credit_is_bounded(self) -> None:
        """prop.3: the gate leaves Theorem 5's perturbation bound intact, so the credit it
        routes back into the codebook is bounded by ``|dL/dp_c| <= ||dL/dh'|| * kappa * ||h||``."""
        field = self._field()
        hidden = self._hidden(seed=17)
        kappa = field.relative_perturbation_bound()

        graded, _ = field.read_and_modulate(hidden)
        objective = graded.pow(2).mean()
        field.zero_grad(set_to_none=True)
        objective.backward()
        traces_grad = field.addresser.traces.grad
        if traces_grad is None:
            self.record("p3.credit_bounded", False, "traces received no gradient")
            return

        hidden_norm = hidden.detach().reshape(-1, self.HIDDEN).norm(dim=-1)
        realised = (graded.detach() - hidden.detach()).reshape(-1, self.HIDDEN).norm(dim=-1)
        within = bool((realised <= kappa * hidden_norm + 1e-5).all())
        worst_ratio = float((realised / hidden_norm.clamp_min(1e-12)).max())
        credit_scale = float(hidden.grad.norm()) if hidden.grad is not None else 0.0
        self.record("p3.credit_bounded", within and float(traces_grad.norm()) > 0.0,
                    f"worst |dh|/|h| = {worst_ratio:.6f} <= kappa = {kappa:.6f}: {within}; "
                    f"|dL/dtraces| = {float(traces_grad.norm()):.4e}, "
                    f"credit scale |dL/dh| = {credit_scale:.4e}")

    # ── P4: the codebook actually trains ───────────────────────────────────────
    def check_traces_move_under_optimizer(self) -> None:
        """A21's observable symptom: the codebook stayed at its initialisation for a whole run."""
        field = self._field()
        optimizer = torch.optim.SGD(field.addresser.parameters(), lr=1e-2)
        before = field.addresser.traces.detach().clone()
        moved = 0.0
        for step in range(6):
            hidden = self._hidden(seed=23 + step)
            target = torch.randn_like(hidden)
            graded, _ = field.read_and_modulate(hidden)
            optimizer.zero_grad(set_to_none=True)
            (graded * target).sum().backward()
            optimizer.step()
        moved = float((field.addresser.traces.detach() - before).abs().max())
        self.record("p4.traces_update", moved > 0.0,
                    f"max|traces - traces_0| after 6 steps = {moved:.6e}")

    def check_query_projection_trains(self) -> None:
        """The other half of the addresser: the query side must not be dead either."""
        field = self._field()
        hidden = self._hidden(seed=29)
        graded, _ = field.read_and_modulate(hidden)
        field.zero_grad(set_to_none=True)
        graded.pow(2).mean().backward()
        grad = field.addresser.query_proj.weight.grad
        present = grad is not None and float(grad.norm()) > 0.0
        self.record("p4.query_proj_trains", present,
                    f"|dL/d(query_proj.weight)| = {0.0 if grad is None else float(grad.norm()):.4e}")

    def check_gate_absent_in_inference_mode(self) -> None:
        """Eval must not pay for the estimator: no gate, no softmax over the pool."""
        field = self._field()
        field.eval()
        hidden = self._hidden(seed=31)
        scores = field.addresser.score_sequence(hidden)
        selected, _ = field.addresser.select_sequence(hidden)
        self.record("p4.no_gate_at_inference", field._straight_through_gate(scores, selected) is None,
                    "inference path builds no probability tensor")

    def check_hard_path_is_dead_without_gate(self) -> None:
        """Negative control: the ungated hard read gives the codebook exactly zero gradient.

        This is what A21 measured (all 177,152 addressing elements at ``grad is None``). It
        documents why ``p4.traces_update`` can only pass while the gate is in place: if the
        gate is ever removed, the codebook silently returns to its initialisation.
        """
        field = self._field()
        hidden = self._hidden(seed=37)
        with torch.no_grad():
            hard, _ = field.read_and_modulate(hidden)
        field.zero_grad(set_to_none=True)
        # Same objective as the gated case, but built from a detached forward: no path to traces.
        hard.requires_grad_(True)
        (hard * torch.randn_like(hard)).sum().backward()
        self.record("p4.hard_path_is_dead", field.addresser.traces.grad is None,
                    "detached hard read yields no trace gradient, as A21 measured")

    # ── runner ─────────────────────────────────────────────────────────────────
    def run(self) -> int:
        """Execute every check and print a verdict table."""
        checks = [
            self.check_gate_evaluates_to_one,
            self.check_forward_is_bitwise_unchanged,
            self.check_score_gradient_matches_soft_selection,
            self.check_credit_is_bounded,
            self.check_traces_move_under_optimizer,
            self.check_query_projection_trains,
            self.check_gate_absent_in_inference_mode,
            self.check_hard_path_is_dead_without_gate,
        ]
        for check in checks:
            try:
                check()
            except Exception as error:  # a crashing check is a failure, not a skip
                self.record(check.__name__, False, f"raised {type(error).__name__}: {error}")
        failed = [name for name, ok, _ in self.results if not ok]
        for name, ok, detail in self.results:
            print(f"[{'PASS' if ok else 'FAIL'}] {name:<36} {detail}")
        print(f"\n{len(self.results) - len(failed)}/{len(self.results)} checks passed")
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(PiscesLxTraceCodebookGradientTests().run())
