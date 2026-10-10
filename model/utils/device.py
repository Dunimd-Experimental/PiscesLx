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
Single source of truth for device selection across the Yv architecture.

Device choice was previously duplicated in ``model/core/model.py``,
``model/multimodal/generator.py`` and ``model/core/cache.py``, and the cache
defaulted to the literal string ``"cuda"`` — which made a CPU-only host raise
``AssertionError: Torch not compiled with CUDA enabled`` deep inside a ``.to()``
call. Centralising it removes both the duplication (coding standard §4.6) and
the hardcoded accelerator (standard §10.2-13, "禁止硬编码设备/平台").
"""

import torch

__all__ = ["YvDeviceResolver"]


class YvDeviceResolver:
    """Resolves an optional device request into a concrete ``torch.device``."""

    AUTO = "auto"
    FALLBACK = "cpu"

    @classmethod
    def resolve(cls, device=None, tensor_hint=None) -> torch.device:
        """Return a usable device without ever hardcoding an accelerator index.

        Args:
            device: ``None``/``"auto"``/``""`` pick the best available backend;
                anything else is honoured as an explicit torch device spec.
            tensor_hint: an existing tensor whose device should be inherited.
        """
        if tensor_hint is not None and hasattr(tensor_hint, "device"):
            return cls._coerce(tensor_hint.device)
        if device is None:
            return cls.default()
        if isinstance(device, str) and device.strip().lower() in ("", cls.AUTO):
            return cls.default()
        return cls._coerce(device)

    @classmethod
    def default(cls) -> torch.device:
        """Best available backend: CUDA, then MPS, then CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and getattr(mps, "is_available", lambda: False)():
            return torch.device("mps")
        return torch.device(cls.FALLBACK)

    @classmethod
    def as_string(cls, device=None, tensor_hint=None) -> str:
        """String form, for configs that still carry a device as text."""
        return str(cls.resolve(device, tensor_hint))

    @staticmethod
    def _coerce(value) -> torch.device:
        if isinstance(value, torch.device):
            return value
        try:
            return torch.device(str(value))
        except (RuntimeError, TypeError, ValueError):
            return torch.device("cpu") if not str(value) else torch.device("cpu")
