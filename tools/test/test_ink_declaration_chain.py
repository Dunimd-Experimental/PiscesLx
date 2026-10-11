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

"""A tier's declared Ink settings must reach the optimiser that gets built.

Twelve `configs/model/*.yaml` files carry an ``ink_optimizer`` block. None of it was read:
``YvConfig`` exposes flat ``ink_*`` fields, the trainer looked for a section on the *train*
configuration (which had none), and ``getattr`` on a dict returns the fallback silently -- so
every tier ran the dataclass defaults while the logs quoted the declared numbers
(docs/audit/00 §47). This suite pins each link of the chain separately, because the failure
mode of every previous link was silence.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml

from tools.train.config import PiscesLxTrainingConfig
from tools.train.core import PiscesLxTrainingOperator

MODEL_YAML = Path(__file__).resolve().parents[2] / "configs" / "model" / "0.01B.yaml"
#: Two bit-widths that differ from each other, so a reader that ignores the declared value
#: and returns the fallback cannot pass.
DECLARED_BITS = 8
FALLBACK_BITS = 4


class PiscesLxInkDeclarationChainTest:
    """Assert YAML -> train config -> optimiser config carries the declared values."""

    def __init__(self) -> None:
        self.results = []

    def record(self, name: str, ok: bool, detail: str) -> None:
        self.results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<40} {detail}")

    @staticmethod
    def _declared_block() -> dict:
        loaded = yaml.safe_load(MODEL_YAML.read_text(encoding="utf-8")) or {}
        section = loaded.get("ink_optimizer")
        return section if isinstance(section, dict) else {}

    def check_the_model_file_actually_declares(self) -> None:
        """The chain is meaningless if the source file declares nothing."""
        block = self._declared_block()
        self.record("chain.model_declares_ink",
                    {"momentum_bits", "variance_bits", "sparse_ratio"} <= set(block),
                    f"{MODEL_YAML.name} declares {sorted(block)[:5]}")

    def check_reader_prefers_declared_values(self) -> None:
        """``_declared`` must work for mapping sections, not only object sections."""
        mapping = {"variance_bits": DECLARED_BITS}
        from_dict = PiscesLxTrainingOperator._declared(mapping, "variance_bits", 4)
        from_obj = PiscesLxTrainingOperator._declared(
            SimpleNamespace(variance_bits=FALLBACK_BITS), "variance_bits", DECLARED_BITS)
        missing = PiscesLxTrainingOperator._declared({}, "num_update_blocks", 1)
        self.record("chain.reader_handles_both_shapes",
                    from_dict == DECLARED_BITS and from_obj == FALLBACK_BITS and missing == 1,
                    f"dict->{from_dict} object->{from_obj} absent->{missing}")

    def check_train_config_carries_the_section(self) -> None:
        """The train configuration must have a field to hold it, or the copy has nowhere to go."""
        config = PiscesLxTrainingConfig()
        config.ink_optimizer = self._declared_block()
        self.record("chain.train_config_field_exists",
                    isinstance(config.ink_optimizer, dict)
                    and config.ink_optimizer.get("variance_bits") is not None,
                    f"carried variance_bits={config.ink_optimizer.get('variance_bits')} "
                    f"sparse_ratio={config.ink_optimizer.get('sparse_ratio')}")

    def check_declared_values_beat_dataclass_defaults(self) -> None:
        """End to end: the values a tier declares are the values the optimiser is built with."""
        from opss.optim.ink import PiscesLxInkConfig
        block = self._declared_block()
        defaults = PiscesLxInkConfig()
        resolved = {key: PiscesLxTrainingOperator._declared(block, key, getattr(defaults, key))
                    for key in ("momentum_bits", "variance_bits", "sparse_ratio", "galore_rank")}
        self.record("chain.declared_beats_default",
                    resolved["momentum_bits"] == block["momentum_bits"]
                    and resolved["variance_bits"] == block["variance_bits"]
                    and resolved["sparse_ratio"] == block["sparse_ratio"],
                    f"resolved={resolved}")

    def run(self) -> int:
        for check in (self.check_the_model_file_actually_declares,
                      self.check_reader_prefers_declared_values,
                      self.check_train_config_carries_the_section,
                      self.check_declared_values_beat_dataclass_defaults):
            try:
                check()
            except Exception as exc:
                self.record(check.__name__, False, f"{type(exc).__name__}: {exc}")
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"{passed}/{len(self.results)} checks passed")
        return 0 if passed == len(self.results) else 1


if __name__ == "__main__":
    raise SystemExit(PiscesLxInkDeclarationChainTest().run())
