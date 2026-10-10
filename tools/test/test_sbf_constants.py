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
Numerical regression for the Subconscious Block Field proofs.

Every constant asserted in ``docs/proofs/00-sbf-theorems.md`` is recomputed here from
first principles, so a proof can never silently drift away from the arithmetic it
claims. Run: ``python -m tools.test.test_sbf_constants``.
"""

import math

__all__ = ["PiscesLxSbfProofConstants", "PiscesLxSbfProofRegression"]


class PiscesLxSbfProofConstants:
    """The quantities fixed by the directive, never by a code literal."""

    BLOCK_COUNT = 628            # 314 B equivalent capacity / 0.5 B per block
    BLOCK_PARAMS = 500_000_000   # 0.5 B parameters per knowledge block
    BYTES_PER_PARAM_FP16 = 2     # FP16 storage
    TRUNK_PARAMS = 7_000_000_000 # 7B main parameters
    T4_VRAM_GB = 16.0            # UPGRADE_07_T4.md
    HOST_LINK_GBPS = 12.0        # PCIe Gen3 x16 practical order
    RETRIEVAL_BUDGET_MS = 5.0    # UPGRADE_09_METRICS.md table 9.3
    TRACE_DIM = 256              # k, addresser trace width
    BITS_PER_BYTE = 8


class PiscesLxSbfProofRegression:
    """Recomputes Lemma 0 and Theorems 1-4 constants and checks them."""

    TOLERANCE = 1e-3

    def __init__(self) -> None:
        self.results = []

    def block_bytes_fp16(self) -> float:
        """Single block footprint in FP16, in GB."""
        c = PiscesLxSbfProofConstants
        return c.BLOCK_PARAMS * c.BYTES_PER_PARAM_FP16 / 1e9

    def transfer_ms(self) -> float:
        """Lemma 0: time to move one whole block over the host link."""
        c = PiscesLxSbfProofConstants
        return self.block_bytes_fp16() / c.HOST_LINK_GBPS * 1000.0

    def max_switch_rate(self) -> float:
        """Theorem 2: p <= budget / transfer_time."""
        c = PiscesLxSbfProofConstants
        return c.RETRIEVAL_BUDGET_MS / self.transfer_ms()

    def min_dwell(self) -> float:
        """Theorem 2 corollary: D = 1/p."""
        return 1.0 / self.max_switch_rate()

    def addressing_bits(self) -> float:
        """Theorem 3: entropy upper bound of the block index, in bits."""
        return math.log2(PiscesLxSbfProofConstants.BLOCK_COUNT)

    def addressing_flops(self) -> float:
        """Theorem 1: score cost N*k."""
        c = PiscesLxSbfProofConstants
        return c.BLOCK_COUNT * c.TRACE_DIM

    def addressing_to_block_ratio(self) -> float:
        """Theorem 1: (N*k)/P."""
        c = PiscesLxSbfProofConstants
        return self.addressing_flops() / c.BLOCK_PARAMS

    def capacity_over_addressing(self, bits_per_param: int = 16) -> float:
        """Theorem 3: addressable bits per bit of addressing overhead.

        FP16 storage is 16 bits/parameter (5.4e11); halving to an 8 bits/parameter
        equivalent gives the conservative figure 2.7e11 quoted in the proof.
        """
        c = PiscesLxSbfProofConstants
        capacity_bits = c.BLOCK_COUNT * c.BLOCK_PARAMS * bits_per_param
        return capacity_bits / self.addressing_bits()

    def max_bits_per_param_for_budget(self) -> float:
        """Lemma 0 corollary: storage width that would fit the budget synchronously."""
        c = PiscesLxSbfProofConstants
        # T_x(b) = (P*b/8)/W  <= budget  =>  b <= budget*W*8/P
        return c.RETRIEVAL_BUDGET_MS / 1000.0 * c.HOST_LINK_GBPS * 8e9 / c.BLOCK_PARAMS

    def assertions(self) -> list:
        """(name, expected, actual) triples taken from the proof document."""
        return [
            ("block FP16 GB", 1.0, self.block_bytes_fp16()),
            ("Lemma 0 transfer ms", 83.3, self.transfer_ms()),
            ("Lemma 0 ratio vs budget", 16.7, self.transfer_ms() / 5.0),
            ("Theorem 2 p_max", 0.060, self.max_switch_rate()),
            ("Theorem 2 D_min", 16.7, self.min_dwell()),
            ("Theorem 1 addressing flops", 160_768.0, self.addressing_flops()),
            ("Theorem 1 addr/block ratio", 3.2e-4, self.addressing_to_block_ratio()),
            ("Theorem 3 log2(628) bits", 9.295, self.addressing_bits()),
            ("Theorem 3 capacity/addressing", 5.4e11, self.capacity_over_addressing()),
            ("Theorem 3 conservative ratio", 2.7e11, self.capacity_over_addressing(8)),
            ("Lemma 0 max bits/param", 0.96, self.max_bits_per_param_for_budget()),
        ]

    def run(self) -> int:
        """Print the comparison table; return the count of drifted constants."""
        drifted = 0
        width = max(len(name) for name, _, _ in self.assertions())
        print(f"{'quantity':<{width}}  {'proof doc':>14}  {'recomputed':>14}  ok")
        for name, expected, actual in self.assertions():
            rel = abs(actual - expected) / max(abs(expected), 1e-12)
            ok = rel <= 0.01  # 1% tolerance: proofs quote rounded values
            if not ok:
                drifted += 1
            print(f"{name:<{width}}  {expected:>14.6g}  {actual:>14.6g}  {'OK' if ok else 'DRIFT'}")
        return drifted


if __name__ == "__main__":
    raise SystemExit(0 if PiscesLxSbfProofRegression().run() == 0 else 1)
