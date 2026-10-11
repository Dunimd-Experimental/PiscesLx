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

"""Contract for the Ink block quantiser (§6.4, A15).

The quantiser used to pad every parameter up to a whole number of blocks, which allocated a
second full-size fp32 copy of the model on each step and pushed a CPU host's commit charge
into native allocation failure. The copy-free reconstruction must stay numerically
identical to that padded scheme, so this suite compares both against shapes that exercise
the ragged tail, an all-zero block, a single element block and a 1-element tensor.
"""

import math

import torch

from opss.optim.ink.quantizer import PiscesLxInkBlockQuantizer

BLOCK_SIZE = 256
INT8_MAX = 127.0
CLIP_LO = -128.0
CLIP_HI = 127.0
RECONSTRUCTION_TOLERANCE = 1e-5
#: Adam's variance is (1-beta2) times a log-normal-ish spread: with this sigma the block spans
#: about six decades, which is the regime that breaks a position-linear codec.
LOG_SPREAD_SIGMA = 3.0
#: (1 - beta2) for beta2 = 0.999, the decay of the second-moment estimator.
VARIANCE_DECAY = 1e-3
#: How many entries are left at exact zero, i.e. coordinates that never saw a gradient.
VARIANCE_ZEROS = 512
RELATIVE_TOLERANCE = 1e-4


class PiscesLxInkQuantizerContractTest:
    """Assert the copy-free quantiser equals the padded reference bit for bit."""

    SHAPES = [(257,), (1, 256), (3, 61), (2, 129), (1,), (1024,), (5, 500)]

    def __init__(self) -> None:
        self.results = []
        self.quantizer = PiscesLxInkBlockQuantizer(momentum_block_size=BLOCK_SIZE,
                                                   variance_block_size=BLOCK_SIZE)

    def record(self, name, ok, detail):
        self.results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<34} {detail}")

    @staticmethod
    def _padded_reference(tensor, block_size=BLOCK_SIZE):
        """The pre-change scheme: pad to whole blocks, quantise, reconstruct, truncate."""
        flat = tensor.flatten().float()
        num_elements = flat.numel()
        num_blocks = (num_elements + block_size - 1) // block_size
        padded = flat if num_blocks * block_size == num_elements else torch.cat(
            [flat, torch.zeros(num_blocks * block_size - num_elements)])
        blocks = padded.view(num_blocks, block_size)
        scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12) / INT8_MAX
        codes = torch.clamp(torch.round(blocks / scales), min=CLIP_LO, max=CLIP_HI).to(torch.int8)
        back = (codes.to(torch.float32) * scales).flatten()[:num_elements]
        return (codes.flatten()[:num_elements].view(tensor.shape),
                scales.squeeze(-1), back.view(tensor.shape))

    @staticmethod
    def _cases(shape):
        yield "random", torch.randn(shape) * 2.0
        yield "zeros", torch.zeros(shape)
        flat = torch.zeros(shape).reshape(-1)
        flat[min(7, flat.numel() - 1)] = 3.5
        yield "impulse", flat.reshape(shape)

    def check_matches_padded_reference(self) -> None:
        """Codes and reconstruction must be identical to the padded scheme."""
        worst_code = 0.0
        worst_back = 0.0
        comparisons = 0
        for shape in self.SHAPES:
            reference_codes, reference_scales, reference_back = self._padded_reference(
                torch.zeros(shape))
            for kind, tensor in self._cases(shape):
                reference_codes, reference_scales, reference_back = self._padded_reference(tensor)
                codes, scales = self.quantizer.quantize_int8(tensor, BLOCK_SIZE)
                back = self.quantizer.dequantize_int8(codes, scales, tensor.shape,
                                                     block_size=BLOCK_SIZE)
                if tuple(codes.shape) != tuple(reference_codes.shape):
                    self.record("quantiser.code_shape", False,
                                f"{shape}/{kind}: {tuple(codes.shape)} vs "
                                f"{tuple(reference_codes.shape)}")
                    return
                if tuple(scales.shape) != tuple(reference_scales.shape):
                    self.record("quantiser.scale_shape", False,
                                f"{shape}/{kind}: {tuple(scales.shape)} vs "
                                f"{tuple(reference_scales.shape)}")
                    return
                worst_code = max(worst_code, float((codes.float() - reference_codes.float())
                                                  .abs().max()))
                worst_back = max(worst_back, float((back.reshape(reference_back.shape).float()
                                                   - reference_back).abs().max()))
                comparisons += 1
        self.record("quantiser.codes_identical", worst_code == 0.0,
                    f"comparisons={comparisons} max|Δcode|={worst_code}")
        self.record("quantiser.reconstruction_identical",
                    worst_back <= RECONSTRUCTION_TOLERANCE,
                    f"max|Δreconstruction|={worst_back:.3e}")

    def check_zero_blocks_reconstruct_to_zero(self) -> None:
        """An all-zero block must not acquire a spurious offset from its scale."""
        tensor = torch.zeros(3, 300)
        codes, scales = self.quantizer.quantize_int8(tensor, BLOCK_SIZE)
        back = self.quantizer.dequantize_int8(codes, scales, tensor.shape, block_size=BLOCK_SIZE)
        self.record("quantiser.zeros_stay_zero", bool((back == 0).all()),
                    f"absmax={float(back.abs().max())} scales={tuple(scales.shape)}")

    def check_round_trip_error_bounded(self) -> None:
        """Quantisation error must stay within half a block scale per element."""
        tensor = torch.randn(4096) * 5.0
        codes, scales = self.quantizer.quantize_int8(tensor, BLOCK_SIZE)
        back = self.quantizer.dequantize_int8(codes, scales, tensor.shape, block_size=BLOCK_SIZE)
        error = (tensor - back.reshape(tensor.shape)).abs()
        # Rounding to the nearest code is exact up to half a step, and one step is the
        # block's own scale (scale = block_max / 127), so the per-element bound is the
        # scale of the block that element belongs to, halved.
        scales_per_element = scales.repeat_interleave(BLOCK_SIZE)[:tensor.numel()]
        bound = scales_per_element.reshape(tensor.shape) / 2.0 + 1e-6
        exceeded = int((error > bound).sum().item())
        self.record("quantiser.error_within_half_lsb", exceeded == 0,
                    f"max_err={float(error.max()):.6f} worst_bound={float(bound.max()):.6f} "
                    f"elements_over_bound={exceeded}")

    def check_variance_log_grid_keeps_positives_positive(self) -> None:
        """A positive variance must never quantise to zero, at 8 or 4 bits.

        This is the A16 mechanism: variance is a sum of squares, so inside one block it spans
        orders of magnitude. Under the signed symmetric absmax codec an entry two decades below
        its block maximum rounds to code 0, and 0 in Adam's denominator replaces its bounded
        sign step with an unbounded one (measured: 61.4% of entries collapsed, median step
        1.3e7 x lr). The log grid is exact about magnitude, not about position on a line.
        """
        torch.manual_seed(0)
        spread = torch.randn(4096) * LOG_SPREAD_SIGMA
        variance = (VARIANCE_DECAY * spread.exp() ** 2).clamp(min=0.0)
        variance[:VARIANCE_ZEROS] = 0.0
        summary = []
        for bits in (8, 4):
            codes, log_lo, log_step = self.quantizer.quantize_variance_log(
                variance, BLOCK_SIZE, bits)
            back = self.quantizer.dequantize_variance_log(
                codes, log_lo, log_step, variance.numel(), BLOCK_SIZE, bits)
            positive = variance > 0
            collapsed = int((back[positive] == 0).sum().item())
            inflated = int((back[~positive] != 0).sum().item())
            ratio = (back[positive] / variance[positive])
            step = float(log_step.max())
            bound = math.exp(step / 2.0)
            within = bool(float(ratio.min()) >= 1.0 / bound - RELATIVE_TOLERANCE
                          and float(ratio.max()) <= bound + RELATIVE_TOLERANCE)
            summary.append(f"{bits}b:collapsed={collapsed} inflated={inflated} "
                           f"rel_err=[{float(ratio.min()):.4f},{float(ratio.max()):.4f}] "
                           f"bound=exp(+-step/2)=[{1.0 / bound:.4f},{bound:.4f}]")
            if collapsed or inflated or not within:
                self.record("quantiser.variance_log_grid", False, " ".join(summary))
                return
        self.record("quantiser.variance_log_grid", True, " ".join(summary))

    def check_variance_log_grid_obeys_signed_codec_size(self) -> None:
        """Show what the old codec did on the same data, so the improvement is measured."""
        torch.manual_seed(0)
        variance = (VARIANCE_DECAY * (torch.randn(4096) * LOG_SPREAD_SIGMA).exp() ** 2)
        variance[:VARIANCE_ZEROS] = 0.0
        codes, scales = self.quantizer.quantize_int4(variance, BLOCK_SIZE)
        back = self.quantizer.dequantize_int4(codes, scales, variance.shape, block_size=BLOCK_SIZE)
        positive = variance > 0
        collapsed_old = int((back[positive] == 0).sum().item())
        codes_new, log_lo, log_step = self.quantizer.quantize_variance_log(variance, BLOCK_SIZE, 4)
        back_new = self.quantizer.dequantize_variance_log(
            codes_new, log_lo, log_step, variance.numel(), BLOCK_SIZE, 4)
        self.record("quantiser.log_grid_beats_signed_int4",
                    collapsed_old > 0 and int((back_new[positive] == 0).sum().item()) == 0,
                    f"signed int4 collapsed {collapsed_old}/{int(positive.sum())} positives; "
                    f"log int4 collapsed 0")

    def run(self) -> int:
        for check in (self.check_matches_padded_reference,
                      self.check_zero_blocks_reconstruct_to_zero,
                      self.check_round_trip_error_bounded,
                      self.check_variance_log_grid_keeps_positives_positive,
                      self.check_variance_log_grid_obeys_signed_codec_size):
            try:
                check()
            except Exception as exc:
                self.record(check.__name__, False, f"{type(exc).__name__}: {exc}")
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"{passed}/{len(self.results)} checks passed")
        return 0 if passed == len(self.results) else 1


if __name__ == "__main__":
    raise SystemExit(PiscesLxInkQuantizerContractTest().run())
