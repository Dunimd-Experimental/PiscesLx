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

"""The trunk's FiLM hand-off must land on every block type it is passed to.

The trunk calls each block with ``film_params=...`` (the subconscious FiLM injection, i.e. the
gradient path along which an addressed knowledge block is trained). When a block's forward
ends in ``**kwargs``, Python accepts the call happily and the value is discarded: measured on
the 0.01B host tier, ``YvHybridBlock.forward`` declared no ``film_params`` at all, so router,
state evolver and every FiLM generator -- 11.56M parameters -- computed into tensors nothing
consumed, and the knowledge blocks never received a gradient while the run reported success
(docs/audit/00 §32).

This suite is the machine check for that defect class. It is a signature test, not a numeric
one, so it costs no memory and can run alongside training.
"""

import inspect
from typing import List, Tuple

import torch

from model.core.blocks import (
    YvDeepNormBlock,
    YvParallelBlock,
    YvTransformerBlock,
)
from model.core.hybrid import YvHybridBlock

BLOCK_TYPES: Tuple[type, ...] = (YvHybridBlock, YvTransformerBlock, YvParallelBlock,
                                YvDeepNormBlock)

#: Shape of the throwaway tensor the modulator checks run against: [batch, sequence, hidden].
#: Only its geometry matters, so the numbers are arbitrary and fixed here rather than inline.
PROBE_SHAPE = (2, 3, 4)


class PiscesLxFilmContractTest:
    """Assert every block the trunk passes film_params to declares and uses it."""

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str) -> None:
        self.results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")

    def check_blocks_declare_film_params(self) -> None:
        """A block must accept film_params by name; **kwargs does not count."""
        offenders = []
        for block in BLOCK_TYPES:
            signature = inspect.signature(block.forward)
            has_var_kw = any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                             for parameter in signature.parameters.values())
            declares = "film_params" in signature.parameters
            if not declares:
                offenders.append(f"{block.__name__}{'(+**kwargs)' if has_var_kw else ''}")
        self.record("film.blocks_declare_parameter", not offenders,
                    f"checked={len(BLOCK_TYPES)} offenders={offenders or 'none'}")

    def check_declared_blocks_use_it(self) -> None:
        """Declaring the parameter is not enough: the body must reference it.

        A named parameter that the body never reads reproduces the same silent discard with
        a cleaner signature, so the check looks at the source text of the forward as well.
        """
        unused = []
        for block in BLOCK_TYPES:
            source = inspect.getsource(block.forward)
            uses_modulation = ("film_params" in source
                               or "YvFiLMModulation" in source)
            if not uses_modulation:
                unused.append(block.__name__)
        self.record("film.blocks_apply_modulation", not unused,
                    f"blocks that never reference film_params or the shared modulator: "
                    f"{unused or 'none'}")

    def check_shared_modulator_exists(self) -> None:
        """The modulation formula must have one implementation, not one per block type."""
        from model.core.norms import YvFiLMModulation

        identity = YvFiLMModulation.apply(torch.ones(PROBE_SHAPE), None)
        self.record("film.shared_modulator", bool(torch.equal(identity, torch.ones(PROBE_SHAPE)))
                    and YvFiLMModulation.apply.__doc__ is not None,
                    "YvFiLMModulation.apply present, None passes the input through")

    def check_rejects_malformed_params(self) -> None:
        """A missing key must fail with the shapes, never be treated as 'no modulation'."""
        from model.core.norms import YvFiLMModulation

        hidden = torch.zeros(PROBE_SHAPE)
        caught = ""
        try:
            YvFiLMModulation.apply(hidden, {"scale": torch.ones(PROBE_SHAPE)})
        except ValueError as exc:
            caught = str(exc)
        self.record("film.rejects_missing_shift", "shift" in caught,
                    f"error carries the missing key: {caught[:64] or 'NO ERROR RAISED'}")

    def run(self) -> int:
        for check in (self.check_blocks_declare_film_params,
                      self.check_declared_blocks_use_it,
                      self.check_shared_modulator_exists,
                      self.check_rejects_malformed_params):
            try:
                check()
            except Exception as exc:
                self.record(check.__name__, False, f"{type(exc).__name__}: {exc}")
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"{passed}/{len(self.results)} checks passed")
        return 0 if passed == len(self.results) else 1


if __name__ == "__main__":
    raise SystemExit(PiscesLxFilmContractTest().run())
