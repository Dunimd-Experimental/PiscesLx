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

"""Iron rule 3 splits the channels: weights to HF + ModelScope, source to git without them.

Each assertion here is the executable form of one half of that sentence, because the failure
mode is silent: a weight pushed to a source remote is not reverted by a later commit, and a
token written into a config file is published to the world by the next push.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
#: Git executable; overridable so the suite runs where git is not on PATH.
GIT = os.environ.get("PISCESLX_GIT", "git")

from tools.publish.hub import (HF_TOKEN_ENV, MODELSCOPE_TOKEN_ENV, PiscesLxHubConfig,
                               PiscesLxModelHubPublisher)


class PiscesLxHubPublishContractTest:
    """Assert the publisher plans correctly and cannot leak weights or credentials."""

    def __init__(self) -> None:
        self.results = []

    def record(self, name: str, ok: bool, detail: str) -> None:
        self.results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")

    #: Size of the stand-in weight blob; only its bytes-on-disk matters to the plan.
    FIXTURE_WEIGHT_BYTES = 512

    @classmethod
    def _fixture(cls, tmp: Path) -> Path:
        """A tiny stand-in for a real weight file: the plan only stats and fingerprints."""
        checkpoint = tmp / "final_model.pt"
        checkpoint.write_bytes(b"\x00" * cls.FIXTURE_WEIGHT_BYTES)
        config = tmp / "tier.yaml"
        config.write_text("hidden_size: 64\n", encoding="utf-8")
        return checkpoint

    def check_plan_lists_every_artifact(self) -> None:
        """A publication must carry the weight, the config to load it, and the tokenizer."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            checkpoint = self._fixture(tmp)
            plan = PiscesLxModelHubPublisher(PiscesLxHubConfig()).plan(
                str(checkpoint), str(tmp / "tier.yaml"))
            names = [entry["name"] for entry in plan["files"]]
            self.record("hub.plan_lists_artifacts",
                        "final_model.pt" in names and "tier.yaml" in names
                        and plan["total_bytes"] > 0
                        and any(name.startswith("tokenizer/") for name in names),
                        f"files={names[:4]} total_bytes={plan['total_bytes']}")

    def check_never_returns_credential_values(self) -> None:
        """The publisher may report that a token exists, never what it is."""
        os.environ[HF_TOKEN_ENV] = "hf_secretlookingstring_should_not_appear"
        try:
            report = PiscesLxModelHubPublisher().credentials()
            serialised = repr(report)
            ok = report[HF_TOKEN_ENV] is True and "should_not_appear" not in serialised
        finally:
            del os.environ[HF_TOKEN_ENV]
        self.record("hub.credentials_are_masked", ok, f"report keys={sorted(report)}")

    def check_dry_run_touches_nothing(self) -> None:
        """Default must be dry-run: an accidental first call must not publish anything."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            checkpoint = self._fixture(tmp)
            report = PiscesLxModelHubPublisher(PiscesLxHubConfig()).publish(
                str(checkpoint), str(tmp / "tier.yaml"))
            self.record("hub.dry_run_by_default",
                        report["uploaded"]["huggingface"].startswith("skipped")
                        and report["uploaded"]["modelscope"].startswith("skipped"),
                        f"uploaded={report['uploaded']}")

    def check_publish_refuses_without_credentials(self) -> None:
        """A real publish with no token must fail, not half-upload."""
        for name in (HF_TOKEN_ENV, MODELSCOPE_TOKEN_ENV):
            os.environ.pop(name, None)
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            checkpoint = self._fixture(tmp)
            message = ""
            try:
                PiscesLxModelHubPublisher().publish(str(checkpoint), str(tmp / "tier.yaml"),
                                                    dry_run=False)
            except RuntimeError as exc:
                message = str(exc)
            self.record("hub.publish_requires_tokens", "HUGGINGFACE_TOKEN" in message,
                        f"{message[:70] or 'NO ERROR RAISED'}")

    def check_refuses_leaking_local_paths(self) -> None:
        """A config or card carrying an absolute host path must block the upload."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            checkpoint = self._fixture(tmp)
            leaking = tmp / "tier.yaml"
            leaking.write_text("tokenizer_path: C:\\Users\\someone\\tok\n", encoding="utf-8")
            message = ""
            try:
                PiscesLxModelHubPublisher().plan(str(checkpoint), str(leaking))
            except ValueError as exc:
                message = str(exc)
            self.record("hub.blocks_local_path_leak", "C:\\Users\\" in message,
                        f"{message[:70] or 'NO ERROR RAISED'}")

    def check_no_weight_is_tracked_by_git(self) -> None:
        """Source backups must not contain weights: no .pt/.safetensors may be tracked."""
        listing = subprocess.run([GIT, "ls-files"], capture_output=True, text=True,
                                 cwd=REPO_ROOT, check=False)
        tracked = [line for line in listing.stdout.splitlines()
                   if line.endswith((".pt", ".safetensors", ".bin"))]
        self.record("hub.no_weight_in_source_tree", listing.returncode == 0 and not tracked,
                    f"tracked weight files={tracked[:3] or 'none'}")

    def run(self) -> int:
        for check in (self.check_plan_lists_every_artifact,
                      self.check_never_returns_credential_values,
                      self.check_dry_run_touches_nothing,
                      self.check_publish_refuses_without_credentials,
                      self.check_refuses_leaking_local_paths,
                      self.check_no_weight_is_tracked_by_git):
            try:
                check()
            except Exception as exc:
                self.record(check.__name__, False, f"{type(exc).__name__}: {exc}")
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"{passed}/{len(self.results)} checks passed")
        return 0 if passed == len(self.results) else 1


if __name__ == "__main__":
    raise SystemExit(PiscesLxHubPublishContractTest().run())
