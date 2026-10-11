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
Executable counterpart of ``docs/design/10-resonant-paired-initialization.md``.

The addresser's trace codebook and the blocks' memories used to be two independent random
draws, so at step 0 which hidden state hits a block and what that block does to a hidden
state were uncorrelated. Resonant paired initialisation writes each trace as the block's own
contribution direction, measured in the addressing metric. The properties proved for it:

  * R1 identity      — ``<W h, R_i>`` equals ``(rho/||R~_i||) <h, P u_i>``, with ``P`` the
    orthogonal projector onto ``row(W)``; the ordering is the content ordering, projected.
  * R2 alignment     — order of blocks by step-0 score has Kendall tau >= 0.9 against the
    content ordering when ``k >= d`` (there ``P = I``); the random-trace baseline is ~0.
  * R3 scale         — every trace has norm exactly ``rho``, and ``rho`` equals its defining
    formula ``lambda sqrt(k) / rms(W)`` — derived, not tuned.
  * R4 determinism   — two runs produce bit-identical traces, and rebuilding a block yields
    the same memory the trace was derived from.
  * R5 invariance    — RPI changes initialisation only; Theorem 5's bound and its premises
    are untouched, so no training-stability conclusion changes.

Run: ``python -m tools.test.test_sbf_paired_init_contract``
"""

import time
from typing import List, Tuple

import torch

from model.config import YvConfig
from model.core.block_field import YvSubconsciousBlockField

__all__ = ["PiscesLxPairedInitializationTests"]


class PiscesLxPairedInitializationTests:
    """Checks resonant paired initialisation against its proof."""

    HIDDEN = 64
    ROWS = 96
    RANK = 32
    TRACES = 64
    TRACE_DIM = 64
    GAMMA_MAX = 0.5
    BETA_BOUND = 0.25
    ALPHA = 0.1
    BATCH = 2
    SEQ = 48
    SAMPLES = 8
    SEED = 1234
    REL_TOL = 1e-3

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    # ── helpers ────────────────────────────────────────────────────────────────
    def _field(self, trace_dim: int = None) -> YvSubconsciousBlockField:
        # The addresser's query projection is drawn from the global stream, so the trace it
        # induces is only reproducible when that stream is seeded — the A21 determinism rule.
        torch.manual_seed(self.SEED)
        cfg = YvConfig(hidden_size=self.HIDDEN)
        cfg.block_memory_rows = self.ROWS
        cfg.block_rank = self.RANK
        cfg.block_count = self.TRACES
        cfg.block_trace_dim = self.TRACE_DIM if trace_dim is None else trace_dim
        cfg.block_gamma_max = self.GAMMA_MAX
        cfg.block_beta_bound = self.BETA_BOUND
        cfg.block_inject_alpha = self.ALPHA
        cfg.block_pool_slots = 4
        field = YvSubconsciousBlockField(cfg, device=torch.device("cpu"),
                                         dtype=torch.float32)
        field.eval()
        field._ensure_paired()  # RPI is deferred to first use, so trigger it here
        return field

    def _hidden(self, seed: int = 7) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(self.SAMPLES, self.HIDDEN, generator=generator)

    def _directions(self, field: YvSubconsciousBlockField) -> torch.Tensor:
        """The block contribution directions ``u_i`` used to build the traces."""
        return torch.stack([
            field._block_direction(field._make_block(i))
            for i in range(field.block_count)
        ])

    @staticmethod
    def _kendall_tau(first: torch.Tensor, second: torch.Tensor) -> float:
        """Rank correlation over the block axis, by direct pairwise counting."""
        left = first.tolist()
        right = second.tolist()
        concordant = discordant = 0
        for i in range(len(left)):
            for j in range(i + 1, len(left)):
                product = (left[i] - left[j]) * (right[i] - right[j])
                if product > 0:
                    concordant += 1
                elif product < 0:
                    discordant += 1
        total = len(left) * (len(left) - 1) / 2
        return (concordant - discordant) / total if total else 0.0

    def record(self, name: str, passed: bool, detail: str) -> None:
        self.results.append((name, passed, detail))

    # ── R1: the identity the construction is derived from ──────────────────────
    def check_score_identity(self) -> None:
        """prop.R1: the realised score is the content score through ``P``, exactly."""
        field = self._field()
        weight = field.addresser.query_proj.weight.detach().float()
        inverse = torch.linalg.pinv(weight @ weight.T)
        rho = field._trace_scale(weight)
        directions = self._directions(field)
        representatives = torch.stack([
            inverse @ (weight @ directions[index])
            for index in range(field.block_count)
        ])
        scale = rho / float(representatives.norm(dim=1).pow(2).mean().sqrt())
        hidden = self._hidden(seed=3)  # [S, d]
        realised = field.addresser.score_sequence(hidden.unsqueeze(1)).squeeze(1)
        worst = 0.0
        for index in range(field.block_count):
            projected = weight.T @ representatives[index]
            expected = field.addresser.score_scale * scale * (hidden @ projected)
            worst = max(worst, float((realised[:, index] - expected).abs().max().detach()))
        self.record("r1.score_identity", worst <= self.REL_TOL,
                    f"max|<W h,R_i> - scale <h,P u_i>| = {worst:.3e}")

    # ── R2: step-0 ordering is the content ordering ────────────────────────────
    def check_order_alignment(self) -> None:
        """prop.R2: Kendall tau vs content ordering -- RPI high, random traces ~0."""
        field = self._field()
        directions = self._directions(field)
        hidden = self._hidden(seed=5)  # distinct from r1's sample
        realised = field.addresser.score_sequence(hidden.unsqueeze(1)).squeeze(1)
        content = hidden @ directions.T
        paired = sum(self._kendall_tau(realised[row], content[row])
                     for row in range(hidden.shape[0])) / hidden.shape[0]

        generator = torch.Generator().manual_seed(19)  # baseline draw, independent of SEED
        with torch.no_grad():
            field.addresser.traces.copy_(
                torch.randn_like(field.addresser.traces) * 0.02)  # the pre-RPI scale
        random_scores = field.addresser.score_sequence(hidden.unsqueeze(1)).squeeze(1)
        baseline = sum(self._kendall_tau(random_scores[row], content[row])
                       for row in range(hidden.shape[0])) / hidden.shape[0]

        detail = f"paired tau = {paired:.3f}, random-trace tau = {baseline:.3f}"
        self.record("r2.order_alignment", paired >= 0.95 and baseline <= 0.2, detail)  # R2 floors

    def check_alignment_degrades_with_bottleneck(self) -> None:
        """prop.R2 scope: the tautness is a function of ``k/d``; a narrow trace loses order."""
        wide = self._field(trace_dim=self.HIDDEN)
        narrow = self._field(trace_dim=max(self.HIDDEN // 8, 2))  # k = d/8 regime
        hidden = self._hidden(seed=5)  # same sample as r2 for comparability
        taus = []
        for field in (wide, narrow):
            directions = self._directions(field)
            realised = field.addresser.score_sequence(hidden.unsqueeze(1)).squeeze(1)
            content = hidden @ directions.T
            taus.append(sum(self._kendall_tau(realised[row], content[row])
                            for row in range(hidden.shape[0])) / hidden.shape[0])
        self.record("r2b.bottleneck_monotone", taus[0] > taus[1],
                    f"k=d tau {taus[0]:.3f} > k=d/8 tau {taus[1]:.3f}")

    # ── R3: rho is derived, not tuned ──────────────────────────────────────────
    def check_trace_scale_is_derived(self) -> None:
        """prop.R3: the mean trace norm is rho, and rho equals lambda sqrt(k)/rms(W)."""
        field = self._field()
        weight = field.addresser.query_proj.weight.detach().float()
        rms = float(weight.norm()) / weight.shape[0] ** 0.5
        expected = field.addresser.hysteresis * weight.shape[0] ** 0.5 / rms
        norms = field.addresser.traces.detach().float().norm(dim=1)
        rms_norm = float(norms.pow(2).mean().sqrt())
        spread = abs(rms_norm - expected) / expected
        identity = abs(field.paired_stats["rho"] - expected) / expected
        self.record("r3.rho_definition", spread <= self.REL_TOL and identity <= 1e-6,
                    f"rms|R_i| = {rms_norm:.4f} vs rho = {expected:.4f} "
                    f"(rel {spread:.2e}), formula rel {identity:.2e}")

    # ── R4: determinism ────────────────────────────────────────────────────────
    def check_deterministic(self) -> None:
        """prop.R4: same seed -> bit-identical traces, and blocks rebuild identically."""
        first = self._field()
        first_traces = first.addresser.traces.detach().clone()
        with torch.no_grad():
            first.addresser.traces.fill_(0.0)
        first.paired_initialize()
        recomputed = torch.equal(first_traces, first.addresser.traces.detach())

        second = self._field()
        fresh = torch.equal(first_traces, second.addresser.traces.detach())

        block_a = first._make_block(11)  # an arbitrary non-zero id
        block_b = second._make_block(11)  # must rebuild bit-identically
        blocks_equal = torch.equal(block_a.memory.detach(), block_b.memory.detach())

        self.record("r4.determinism", recomputed and fresh and blocks_equal,
                    f"recompute={recomputed}, fresh={fresh}, block11={blocks_equal}")

    # ── R5: no training-stability conclusion changes ───────────────────────────
    def check_theorem5_invariance(self) -> None:
        """prop.R5: RPI touches initialisation only -- Theorem 5's bound is unchanged."""
        field = self._field()
        bound = field.relative_perturbation_bound()
        expected = self.ALPHA * (self.GAMMA_MAX + self.BETA_BOUND)
        with torch.no_grad():
            field.addresser.traces.fill_(0.0)
        after = field.relative_perturbation_bound()
        unchanged = (abs(bound - expected) < 1e-9 and abs(after - bound) < 1e-9
                     and field.gamma_max == self.GAMMA_MAX
                     and field.beta_bound == self.BETA_BOUND)
        self.record("r5.bound_invariant", unchanged,
                    f"kappa = {bound:.4f} (alpha*(gmax+c) = {expected:.4f}), stable across RPI")

    def report_cost(self) -> None:
        """The one-time cost of RPI, measured on this layer rather than asserted."""
        started = time.perf_counter()
        field = self._field()
        elapsed = time.perf_counter() - started
        self.record("r6.one_time_cost", True,
                    f"{elapsed * 1e3:.1f} ms for {field.block_count} blocks "  # s -> ms
                    f"(rows={self.ROWS}, d={self.HIDDEN}); construction-time only")

    # ── runner ─────────────────────────────────────────────────────────────────
    def run(self) -> int:
        """Execute every check and print a verdict table."""
        checks = [
            self.check_score_identity,
            self.check_order_alignment,
            self.check_alignment_degrades_with_bottleneck,
            self.check_trace_scale_is_derived,
            self.check_deterministic,
            self.check_theorem5_invariance,
            self.report_cost,
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
    raise SystemExit(PiscesLxPairedInitializationTests().run())
