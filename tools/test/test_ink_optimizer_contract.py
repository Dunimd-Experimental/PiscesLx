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

"""Step-size, sparsity and descent contracts for the Ink optimiser (A13/A15/A16).

With bias-corrected estimators m̂ = m/(1-β1ᵗ) and v̂ = v/(1-β2ᵗ), Adam's first step is
m̂/(√v̂ + eps) = g/(|g| + eps) ≈ sign(g): every coordinate it updates must move by ~lr.
Three defects kept that from happening, and each is pinned here by the quantity that
exposes it rather than by a loss curve:

1. an extra (1-β2ᵗ) factor in the numerator made steps ~1000x too small;
2. variance went through the signed symmetric absmax codec, which maps a positive entry
   two orders of magnitude below its block maximum onto exactly 0, so the denominator
   fell to eps and the step became unbounded (the observed NaN divergence);
3. every parameter above block_size*10 elements was trained at a hard-wired 10% density
   from an attribute no configuration set, so `sparse_ratio`, `sparse_warmup_steps` and
   `sparse_adaptive` had no effect -- and the displacement measured across *all*
   coordinates was weight decay, not the Adam term (median|Δ|/lr = 0.0078, agreement
   0.553, which is what made a non-learning optimiser look like a learning one).
"""

import torch

from opss.optim.ink import PiscesLxInkOptimizer, PiscesLxInkConfig

PARAM_ELEMENTS = 4096
LEARNING_RATE = 1e-2
SIGN_STEP_TOLERANCE = 0.02
#: Structured selection works in whole blocks, so the updated count can only land on a
#: multiple of the block width; one block of slack is the exactness this check can ask for.
STRUCTURED_BLOCK = 32
DESCENT_IMPROVEMENT_MIN = 0.5
LEARNING_STEPS = 200
FEATURES = 64
REGRESSION_ROWS = 256


class PiscesLxInkOptimizerContractTest:
    """Assert Ink's steps, declared sparsity and real descent on a solved objective."""

    def __init__(self) -> None:
        self.results = []

    def record(self, name, ok, detail):
        self.results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<40} {detail}")

    def _build(self, parameter, **overrides):
        settings = {"lr": LEARNING_RATE, "weight_decay": 0.0, "sparse_warmup_steps": 0}
        settings.update(overrides)
        return PiscesLxInkOptimizer([parameter], config=PiscesLxInkConfig(**settings))

    def check_first_step_is_adam_sign_step(self) -> None:
        """One step must displace each updated coordinate by exactly lr, downhill."""
        torch.manual_seed(0)
        gradient = torch.randn(PARAM_ELEMENTS)
        start = torch.randn(PARAM_ELEMENTS)
        parameter = start.clone().requires_grad_(True)
        with torch.no_grad():
            parameter.grad = gradient.clone()
        self._build(parameter).step()
        delta = parameter.detach() - start
        updated = delta.abs() > 0
        if int(updated.sum()) == 0:
            self.record("ink.first_step_is_adam_sign_step", False, "no coordinate moved")
            return
        ratios = (delta.abs()[updated] / LEARNING_RATE)
        median_ratio = float(ratios.median())
        agreement = float((delta.sign() == -gradient.sign())[updated].float().mean())
        self.record("ink.first_step_is_adam_sign_step",
                    abs(median_ratio - 1.0) <= SIGN_STEP_TOLERANCE and agreement == 1.0,
                    f"median|delta|/lr={median_ratio:.4f} (Adam=1.0) "
                    f"descent_agreement={agreement:.4f} updated={int(updated.sum())}")

    def check_trust_region_bounds_every_step(self) -> None:
        """No single step may move a coordinate further than lr, whatever the states do."""
        torch.manual_seed(0)
        gradient = torch.randn(PARAM_ELEMENTS)
        parameter = torch.randn(PARAM_ELEMENTS).requires_grad_(True)
        previous = parameter.detach().clone()
        worst = 0.0
        optimizer = self._build(parameter, sparse_ratio=1.0)
        for _ in range(LEARNING_STEPS):
            with torch.no_grad():
                parameter.grad = torch.randn(PARAM_ELEMENTS) * (10.0 ** (worst % 3))
            optimizer.step()
            step = (parameter.detach() - previous).abs()
            worst = max(worst, float((step / LEARNING_RATE).max()))
            previous = parameter.detach().clone()
        self.record("ink.trust_region_bounds_every_step", worst <= 1.0 + SIGN_STEP_TOLERANCE,
                    f"largest single-step displacement={worst:.4f} x lr over {LEARNING_STEPS} steps")

    def check_declared_sparsity_is_authoritative(self) -> None:
        """sparse_ratio must set the updated share; warmup must start dense."""
        updated_by_ratio = {}
        for ratio in (0.01, 0.1):
            torch.manual_seed(1)
            start = torch.randn(PARAM_ELEMENTS)
            parameter = start.clone().requires_grad_(True)
            with torch.no_grad():
                parameter.grad = torch.randn(PARAM_ELEMENTS)
            self._build(parameter, sparse_ratio=ratio).step()
            updated_by_ratio[ratio] = int(((parameter.detach() - start).abs() > 0).sum())
        expected_01 = int(round(PARAM_ELEMENTS * 0.01))
        expected_10 = int(round(PARAM_ELEMENTS * 0.1))
        ratio_ok = (abs(updated_by_ratio[0.01] - expected_01) <= STRUCTURED_BLOCK
                    and abs(updated_by_ratio[0.1] - expected_10) <= STRUCTURED_BLOCK)
        torch.manual_seed(2)
        start = torch.randn(PARAM_ELEMENTS)
        parameter = start.clone().requires_grad_(True)
        with torch.no_grad():
            parameter.grad = torch.randn(PARAM_ELEMENTS)
        config = PiscesLxInkConfig(lr=LEARNING_RATE, weight_decay=0.0, sparse_ratio=0.01,
                                   sparse_warmup_steps=100)
        PiscesLxInkOptimizer([parameter], config=config).step()
        warmup_updated = int(((parameter.detach() - start).abs() > 0).sum())
        self.record("ink.declared_sparsity_is_authoritative", ratio_ok and warmup_updated == PARAM_ELEMENTS,
                    f"updated@0.01={updated_by_ratio[0.01]} (block-rounded target {expected_01}), "
                    f"updated@0.1={updated_by_ratio[0.1]} (target {expected_10}), "
                    f"warmup step 0 updated={warmup_updated}/{PARAM_ELEMENTS}")

    def check_descent_on_a_solved_objective(self) -> None:
        """The optimiser must reduce one fixed objective, not just move numbers.

        Comparing the loss before and after a step only means something if both are the same
        function of the parameters: an earlier version of this check fed a different row per
        step and "measured descent" on numbers that were never comparable.
        """
        torch.manual_seed(0)
        rows = REGRESSION_ROWS
        target = torch.randn(FEATURES)
        design = torch.randn(rows, FEATURES)
        observations = design @ target
        weights = torch.zeros(FEATURES).requires_grad_(True)

        def objective() -> float:
            return float(((design @ weights.detach() - observations) ** 2).mean())

        optimizer = self._build(weights, sparse_ratio=0.5, lr=5e-2)
        first = objective()
        for _ in range(LEARNING_STEPS):
            residual = design @ weights - observations
            with torch.no_grad():
                weights.grad = 2.0 * (design.transpose(0, 1) @ residual) / rows
            optimizer.step()
        last = objective()
        improvement = (first - last) / max(first, 1e-12)
        self.record("ink.reduces_a_real_loss", improvement >= DESCENT_IMPROVEMENT_MIN
                    and bool(torch.isfinite(weights.detach()).all()),
                    f"fixed objective {first:.4e} -> {last:.4e} ({100.0 * improvement:.1f}% down) "
                    f"over {LEARNING_STEPS} steps at sparse_ratio=0.5")

    def run(self) -> int:
        for check in (self.check_first_step_is_adam_sign_step,
                      self.check_trust_region_bounds_every_step,
                      self.check_declared_sparsity_is_authoritative,
                      self.check_descent_on_a_solved_objective):
            try:
                check()
            except Exception as exc:
                self.record(check.__name__, False, f"{type(exc).__name__}: {exc}")
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"{passed}/{len(self.results)} checks passed")
        return 0 if passed == len(self.results) else 1


if __name__ == "__main__":
    raise SystemExit(PiscesLxInkOptimizerContractTest().run())
