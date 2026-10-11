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

"""Host tier contract: what the device the run lands on actually permits.

The repository already carries memory-pressure presets (`get_extreme_memory_config` and the
`vram_*` fields), but those decide by *how tight memory is*, not by *what the hardware can
do*, and they set `vram_mixed_precision = "bf16"` regardless of device -- which on the
project's mandated accelerator (Tesla T4, sm_75) has no native BF16 acceleration at all.

This module adds the missing layer as a **validator over the existing fields** rather than a
parallel set of knobs, so no second track of switches appears. It answers three questions and
refuses to guess:

1. which device is this run on (measured, not assumed);
2. what does that device forbid;
3. does the configuration contradict it -- and if so, fail before the first step.

The contradiction rule matters: during this project's host work, four defects of the shape
"declared in configuration, never applied in code" were found (optimizer sparsity ratio,
variance bit-width, the host memory ceiling, the random seed). Each cost a run. A tier that
only *sets* values would add a fifth; a tier that *checks* removes the class.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from utils.device import PiscesLxDeviceResolver

#: Turing (sm_75) has FP16 tensor cores but no native BF16 acceleration.
TURING_COMPUTE_CAPABILITY: Tuple[int, int] = (7, 5)
#: Configurations reporting a capability this low are treated as CPU-only.
MIN_GPU_COMPUTE_CAPABILITY: Tuple[int, int] = (7, 0)
#: First architecture generation with native BF16 tensor cores (Ampere), and the first with a
#: usable FlashAttention kernel. Both live above Turing, which the project's target accelerator is not.
BF16_MIN_COMPUTE_CAPABILITY: Tuple[int, int] = (8, 0)
FLASH_ATTENTION_MIN_COMPUTE_CAPABILITY: Tuple[int, int] = (8, 0)
#: A CPU host cannot satisfy the accelerator throughput target; the tier says so once.
CPU_TOKS_PER_SECOND_TARGET = 1500.0


class YvHostTierError(RuntimeError):
    """Raised when a configuration contradicts the device it is about to run on."""


@dataclass
class YvHostTier:
    """The device-resolved capability contract for one run.

    Attributes:
        name: ``cpu``, ``turing`` or ``accelerated``.
        device: Resolved torch device type.
        compute_capability: CUDA compute capability, or None off-device.
        supports_fp16_storage: Whether FP16 tensors can be stored and computed.
        supports_bf16_accelerated: Whether BF16 has native tensor-core support.
        supports_flash_attention: Whether a FlashAttention-family kernel is available.
        enforce_host_memory_ceiling: Whether the host memory guard is expected on this tier.
    """

    name: str = "cpu"
    device: str = "cpu"
    compute_capability: Optional[Tuple[int, int]] = None
    supports_fp16_storage: bool = False
    supports_bf16_accelerated: bool = False
    supports_flash_attention: bool = False
    enforce_host_memory_ceiling: bool = True
    notes: List[str] = field(default_factory=list)

    def describe(self) -> Dict[str, Any]:
        """A flat, loggable report; the run must print this to prove the tier took effect."""
        return {
            "tier": self.name,
            "device": self.device,
            "compute_capability": self.compute_capability,
            "fp16": self.supports_fp16_storage,
            "bf16_accelerated": self.supports_bf16_accelerated,
            "flash_attention": self.supports_flash_attention,
            "host_ceiling": self.enforce_host_memory_ceiling,
        }


class YvHostTierResolver:
    """Measures the device and turns the tier's requirements into hard failures."""

    @classmethod
    def resolve(cls, device: Any = None) -> YvHostTier:
        """Build the tier from the *measured* device.

        Args:
            device: Requested device, or None to resolve automatically.

        Returns:
            YvHostTier: The contract this run must satisfy.
        """
        resolved = PiscesLxDeviceResolver.resolve(device) if device is None else device
        resolved_type = getattr(resolved, "type", str(resolved))
        if resolved_type != "cuda":
            return YvHostTier(
                name="cpu", device=resolved_type,
                notes=["throughput targets (P1) are not measurable on this tier"])

        capability = torch.cuda.get_device_capability(resolved)
        is_turing = capability[0] == TURING_COMPUTE_CAPABILITY[0] and capability[1] == TURING_COMPUTE_CAPABILITY[1]
        return YvHostTier(
            name="turing" if is_turing else "accelerated",
            device="cuda",
            compute_capability=capability,
            supports_fp16_storage=True,
            # Ampere (sm_80) and newer carry BF16 tensor cores; Turing does not.
            supports_bf16_accelerated=capability >= BF16_MIN_COMPUTE_CAPABILITY,
            supports_flash_attention=capability >= FLASH_ATTENTION_MIN_COMPUTE_CAPABILITY,
            enforce_host_memory_ceiling=False,
            notes=(["Turing (sm_75): FP16 tensor cores, no native BF16 acceleration"]
                   if is_turing else []),
        )

    @classmethod
    def contradictions(cls, tier: YvHostTier, config: Any) -> List[str]:
        """List every way ``config`` disagrees with the measured device.

        Args:
            tier: The resolved contract.
            config: A model configuration exposing the relevant fields.

        Returns:
            Human-readable findings; empty means the configuration is consistent.
        """
        findings: List[str] = []
        mixed = str(getattr(config, "vram_mixed_precision",
                            getattr(config, "mixed_precision", "none")) or "none").lower()
        if mixed == "bf16" and not tier.supports_bf16_accelerated:
            findings.append(
                f"mixed_precision=bf16 on {tier.name}: BF16 has no native tensor-core support "
                "here, so the run silently loses the acceleration it is claiming to measure")
        flash_requested = bool(getattr(config, "vram_flash_attention",
                                       getattr(config, "use_flash_attention",
                                               getattr(config, "flash_attention", False))))
        if flash_requested and not tier.supports_flash_attention:
            findings.append(
                f"flash_attention requested on {tier.name}: no FlashAttention-family kernel is "
                "available, so the setting is decorative")
        if tier.device == "cpu" and mixed in ("fp16", "float16"):
            findings.append(
                "mixed_precision=fp16 on CPU: storage works but there is no FP16 compute path, "
                "so any throughput number measured this way is not the accelerator's")
        return findings

    @classmethod
    def enforce(cls, config: Any, device: Any = None) -> YvHostTier:
        """Resolve the tier, print it, and refuse a contradictory configuration.

        Raises:
            YvHostTierError: when the configuration disagrees with the device.
        """
        tier = cls.resolve(device)
        findings = cls.contradictions(tier, config)
        if findings:
            raise YvHostTierError(
                f"host tier {tier.name} ({tier.describe()}) rejects this configuration: "
                + "; ".join(findings))
        return tier
