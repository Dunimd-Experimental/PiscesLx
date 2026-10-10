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
Single source of truth for accelerator selection, project-wide.

Device choice used to be re-implemented in ``model/core/model.py``,
``model/multimodal/generator.py``, ``model/core/cache.py`` and twice more in
``tools/train/core.py`` — and ``tools/train/core.py`` pinned a literal accelerator
index, which breaks any host whose rank differs from that constant. Selection now happens here once; the model layer
exposes ``YvDeviceResolver`` as a naming-boundary facade over this implementation
(coding standard §6.1 forbids a ``PiscesLx`` name inside ``model/``).
"""

import os
from typing import Optional

import torch

__all__ = ["PiscesLxDeviceResolver"]


class PiscesLxDeviceResolver:
    """Resolves a device request into a concrete ``torch.device``."""

    AUTO = "auto"
    FALLBACK = "cpu"
    #: env var that may pin the accelerator index from the outside (Kaggle, SLURM, …)
    INDEX_ENV = "PISCESLX_DEVICE_INDEX"
    RANK_ENV = "LOCAL_RANK"

    @classmethod
    def resolve(cls, device=None, tensor_hint=None, local_rank: Optional[int] = None) -> torch.device:
        """Return a usable device without ever hardcoding an accelerator index.

        Args:
            device: ``None`` / ``"auto"`` / ``""`` choose the best available backend;
                anything else is honoured as an explicit torch device spec.
            tensor_hint: an existing tensor whose device should be inherited.
            local_rank: distributed rank, when known; falls back to the environment.
        """
        if tensor_hint is not None and hasattr(tensor_hint, "device"):
            return cls._coerce(tensor_hint.device)
        if device is not None and not (isinstance(device, str) and device.strip().lower() in ("", cls.AUTO)):
            return cls._coerce(device)
        return cls.default(local_rank)

    @classmethod
    def default(cls, local_rank: Optional[int] = None) -> torch.device:
        """Best available backend, honouring the configured rank rather than pinning 0."""
        if torch.cuda.is_available():
            index = cls.rank(local_rank, torch.cuda.device_count())
            return torch.device(f"cuda:{index}")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and getattr(mps, "is_available", lambda: False)():
            return torch.device("mps")
        return torch.device(cls.FALLBACK)

    @classmethod
    def rank(cls, local_rank: Optional[int], device_count: int) -> int:
        """Pick an accelerator index from rank/explicit env, clamped to what exists."""
        candidates = [
            local_rank,
            cls._env_int(cls.RANK_ENV),
            cls._env_int(cls.INDEX_ENV),
        ]
        for value in candidates:
            if value is not None and value >= 0:
                return value % max(device_count, 1)
        return 0

    @classmethod
    def as_string(cls, device=None, tensor_hint=None,
                  local_rank: Optional[int] = None) -> str:
        """String form, for configuration that still carries a device as text."""
        return str(cls.resolve(device, tensor_hint, local_rank))

    @classmethod
    def probe(cls, device: torch.device) -> bool:
        """Whether allocating one scalar on ``device`` actually succeeds."""
        if device.type == "cpu":
            return True
        try:
            torch.zeros(1, device=device)
        except Exception:
            return False
        return True

    @staticmethod
    def _env_int(name: str) -> Optional[int]:
        raw = os.environ.get(name)
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce(value) -> torch.device:
        try:
            return torch.device(str(value))
        except (RuntimeError, TypeError, ValueError):
            return torch.device(PiscesLxDeviceResolver.FALLBACK)
