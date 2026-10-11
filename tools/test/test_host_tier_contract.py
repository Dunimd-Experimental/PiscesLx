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

"""The host tier contract must refuse configurations the device cannot honour.

Why this is asserted rather than trusted: this project's accelerator is Tesla T4 (sm_75),
which has FP16 tensor cores but **no native BF16 acceleration**, while the repository's
memory presets set ``vram_mixed_precision = "bf16"`` for every device. A run that measures
"FP16 throughput" while asking for BF16, or claims FlashAttention on a CPU host, produces a
number that describes neither tier -- and P1 is entirely throughput metrics.
"""

from types import SimpleNamespace

from model.host_tier import (TURING_COMPUTE_CAPABILITY, YvHostTier, YvHostTierError,
                        YvHostTierResolver)

#: Reference accelerator used only to exercise the pass-through branch of the contract.
MODERN_COMPUTE_CAPABILITY = (9, 0)

CPU_TIER = YvHostTier(name="cpu", device="cpu")
TURING_TIER = YvHostTier(name="turing", device="cuda", compute_capability=TURING_COMPUTE_CAPABILITY,
                         supports_fp16_storage=True, supports_bf16_accelerated=False,
                         supports_flash_attention=False)
MODERN_TIER = YvHostTier(name="accelerated", device="cuda", compute_capability=MODERN_COMPUTE_CAPABILITY,
                         supports_fp16_storage=True, supports_bf16_accelerated=True,
                         supports_flash_attention=True, enforce_host_memory_ceiling=False)


class PiscesLxHostTierContractTest:
    """Assert the tier measures, reports, and refuses the right things."""

    def __init__(self) -> None:
        self.results = []

    def record(self, name: str, ok: bool, detail: str) -> None:
        self.results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<40} {detail}")

    def check_resolves_the_running_device(self) -> None:
        """The tier comes from the device, not from a config string."""
        tier = YvHostTierResolver.resolve("cpu")
        self.record("tier.resolves_from_device",
                    tier.name == "cpu" and not tier.supports_flash_attention
                    and not tier.supports_bf16_accelerated and tier.enforce_host_memory_ceiling,
                    f"{tier.describe()}")

    def check_bf16_is_refused_on_turing(self) -> None:
        """T4 has no BF16 tensor cores; asking for it must fail, not silently fall back."""
        config = SimpleNamespace(vram_mixed_precision="bf16", vram_flash_attention=False)
        findings = YvHostTierResolver.contradictions(TURING_TIER, config)
        bf16_caught = any("bf16" in f.lower() for f in findings)
        self.record("tier.bf16_refused_on_turing", bf16_caught,
                    f"findings={len(findings)}: {(findings[0][:70] if findings else 'none')}")

    def check_flash_attention_is_refused_where_absent(self) -> None:
        """FlashAttention on CPU or Turing-limited paths must not be reported as used."""
        cpu_config = SimpleNamespace(vram_mixed_precision="none", vram_flash_attention=True)
        on_cpu = YvHostTierResolver.contradictions(CPU_TIER, cpu_config)
        on_turing = YvHostTierResolver.contradictions(TURING_TIER, cpu_config)
        allowed = YvHostTierResolver.contradictions(MODERN_TIER, cpu_config)
        self.record("tier.flash_requires_support",
                    any("flash" in f.lower() for f in on_cpu)
                    and any("flash" in f.lower() for f in on_turing)
                    and not allowed,
                    f"cpu_findings={len(on_cpu)} turing_findings={len(on_turing)} "
                    f"sm90_findings={len(allowed)}")

    def check_train_config_field_names_are_honoured(self) -> None:
        """The training schema spells these ``mixed_precision``/``flash_attention``.

        Reading only the model-config aliases would let a contradictory training
        configuration through untouched, which is how the tier would become decoration.
        """
        train_shape = SimpleNamespace(mixed_precision="none", flash_attention=True)
        findings = YvHostTierResolver.contradictions(CPU_TIER, train_shape)
        self.record("tier.train_config_field_names", any("flash" in f.lower() for f in findings),
                    f"findings={findings[:1] or 'none'}")

    def check_a_consistent_configuration_passes(self) -> None:
        """The contract must not invent findings when the configuration agrees."""
        fp16_on_turing = SimpleNamespace(vram_mixed_precision="fp16", vram_flash_attention=False)
        bf16_on_sm90 = SimpleNamespace(vram_mixed_precision="bf16", vram_flash_attention=True)
        none_on_cpu = SimpleNamespace(vram_mixed_precision="none", vram_flash_attention=False)
        self.record("tier.consistent_configuration_passes",
                    not YvHostTierResolver.contradictions(TURING_TIER, fp16_on_turing)
                    and not YvHostTierResolver.contradictions(MODERN_TIER, bf16_on_sm90)
                    and not YvHostTierResolver.contradictions(CPU_TIER, none_on_cpu),
                    "fp16@turing / bf16@sm90 / none@cpu all accepted")

    def check_enforce_raises_before_the_first_step(self) -> None:
        """A contradiction is an error, never a warning the run continues past."""
        config = SimpleNamespace(vram_mixed_precision="bf16", vram_flash_attention=False)
        raised = ""
        try:
            YvHostTierResolver.enforce(config, device="cpu")
        except YvHostTierError as exc:
            raised = str(exc)
        self.record("tier.enforce_raises", "bf16" in raised.lower() and "cpu" in raised.lower(),
                    f"raised message: {raised[:76] or 'NO ERROR'}")

    def run(self) -> int:
        for check in (self.check_resolves_the_running_device,
                      self.check_bf16_is_refused_on_turing,
                      self.check_flash_attention_is_refused_where_absent,
                      self.check_train_config_field_names_are_honoured,
                      self.check_a_consistent_configuration_passes,
                      self.check_enforce_raises_before_the_first_step):
            try:
                check()
            except Exception as exc:
                self.record(check.__name__, False, f"{type(exc).__name__}: {exc}")
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"{passed}/{len(self.results)} checks passed")
        return 0 if passed == len(self.results) else 1


if __name__ == "__main__":
    raise SystemExit(PiscesLxHostTierContractTest().run())
