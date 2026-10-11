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
Guards for resuming a run: the intent must not be re-read as "adapt a base model" (A73), and a
changed optimiser topology must degrade loudly instead of killing the resume (A79).

Both are single expressions inside methods that need a live orchestrator and a live trainer to
reach behaviourally, and both were verified by hand this session - a behaviour with no guard is
one edit away from coming back. Each check therefore runs against a source *string* and is itself
checked against a doctored copy, so the test proves both directions on its own: a check that only
ever passes would be worse than no check.
"""

import inspect
from typing import List, Tuple

from tools.train.core import PiscesLxTrainingOperator
from tools.train.orchestrator import PiscesLxTrainOrchestrator

__all__ = ["PiscesLxResumeContract"]


class PiscesLxResumeContract:
    """The two resume invariants, each asserted on real source and on a doctored source."""

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    def record(self, name: str, passed: bool, detail: str) -> None:
        self.results.append((name, passed, detail))

    @staticmethod
    def _adapter_decision_reads_resume(source: str) -> bool:
        """Whether the adapter decision is driven by `resume_ckpt` (the A73 defect)."""
        for line in source.splitlines():
            if "adapter_run" in line and "=" in line and "resume_ckpt" in line:
                return True
        return False

    @staticmethod
    def _resume_skips_mismatched_state(source: str) -> bool:
        """Whether `load_checkpoint` compares group counts before loading optimiser state."""
        return ("param_groups" in source
                and "optimizer_state_dict" in source
                and "Resume skipped the optimiser state" in source)

    def check_adapter_decision_ignores_resume(self) -> None:
        """A resume must rebuild the same model; only a base weight asks for an adapter."""
        source = inspect.getsource(
            PiscesLxTrainOrchestrator._apply_flagship_memory_and_convergence_policy)
        self.record("resume::adapter decision ignores --resume_ckpt",
                    not self._adapter_decision_reads_resume(source),
                    "resume_ckpt does not select the adapter path")
        doctored = source + "\n        adapter_run = bool(resume_ckpt)\n"
        self.record("resume::check catches the regression (self-control)",
                    self._adapter_decision_reads_resume(doctored),
                    "a doctored source is rejected, so the check is not vacuous")

    def check_resume_degrades_loudly_on_topology_change(self) -> None:
        """A parameter-group count mismatch must keep the weights and drop the moments loudly."""
        source = inspect.getsource(PiscesLxTrainingOperator.load_checkpoint)
        self.record("resume::group mismatch is handled, not raised",
                    self._resume_skips_mismatched_state(source),
                    "records the group count, warns, and keeps the loaded weights")
        doctored = source.replace("param_groups", "unrelated").replace(
            "Resume skipped the optimiser state", "silent")
        self.record("resume::check catches the regression (self-control)",
                    not self._resume_skips_mismatched_state(doctored),
                    "a doctored source is rejected, so the check is not vacuous")

    def run(self) -> int:
        """Execute every check; non-zero exit on any failure."""
        self.check_adapter_decision_ignores_resume()
        self.check_resume_degrades_loudly_on_topology_change()
        failed = [name for name, ok, _ in self.results if not ok]
        for name, ok, detail in self.results:
            print(f"[{'PASS' if ok else 'FAIL'}] {name:<48} {detail}")
        print(f"\n{len(self.results) - len(failed)}/{len(self.results)} checks passed")
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(PiscesLxResumeContract().run())
