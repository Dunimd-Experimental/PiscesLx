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
Stage 10 — mechanism regression suites (§6.4).

``tools/test/test_*.py`` hold the executable proofs for the architecture: the
subconscious block field constants and mechanisms, the attention-mask broadcast
contract, and the MoE routing contract.  They are standalone programs, so this
stage runs each in its own interpreter and reports what it printed — a suite that
only a human invoking it by hand can verify is not a check.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

__all__ = ["PiscesLxRegressionSuiteChecker"]


class PiscesLxRegressionSuiteChecker:
    """Run every mechanism regression suite and report its outcome."""

    SUITE_GLOB = "test_*.py"
    TIMEOUT_S = 900
    MESSAGE_CHARS = 120
    SUMMARY_CHARS = 200
    FALLBACK_LINES = 1
    SUMMARY_MARKERS = ("checks passed", "checks failed", "FAIL")

    def __init__(self, root_path: str = ".", verbose: bool = False) -> None:
        self.root_path = Path(root_path).resolve()
        self.verbose = verbose
        self.results: List[Tuple[str, str, str, float]] = []

    def run(self) -> List[Tuple[str, str, str, float]]:
        """Execute each suite as a subprocess; one row per suite."""
        self.results = []
        for suite in sorted((self.root_path / "tools" / "test").glob(self.SUITE_GLOB)):
            self._run_suite(suite)
        if not self.results:
            self.results.append(("regression suites", "WARN",
                                 f"no {self.SUITE_GLOB} found under tools/test", 0.0))
        return self.results

    def _run_suite(self, suite: Path) -> None:
        start = time.time()
        try:
            completed = subprocess.run(
                [sys.executable, str(suite)],
                cwd=str(self.root_path), capture_output=True, text=True,
                timeout=self.TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            self._add(suite.stem, "FAIL", f"timed out after {self.TIMEOUT_S}s", time.time() - start)
            return
        except OSError as exc:
            self._add(suite.stem, "FAIL", f"could not start: {exc}", time.time() - start)
            return
        lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
        summary = next((line for line in reversed(lines)
                        if any(marker in line for marker in self.SUMMARY_MARKERS)), "")
        if completed.returncode == 0:
            self._add(suite.stem, "PASS", summary or "exit 0", time.time() - start)
            return
        failure = next((line for line in lines if line.startswith("[FAIL]")), summary)
        if not failure:
            failure = (completed.stderr or "").strip().splitlines()[-self.FALLBACK_LINES:] or [""]
            failure = failure[0]
        self._add(suite.stem, "FAIL",
                  f"{failure} | {summary}" if summary else failure, time.time() - start)

    def _add(self, name: str, status: str, message: str, duration: float) -> None:
        if self.verbose or status != "PASS":
            print(f"  [{status}] {name}: {message[: self.MESSAGE_CHARS]}")
        self.results.append((name, status, message[: self.SUMMARY_CHARS], duration))
