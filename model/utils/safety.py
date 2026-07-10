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

import torch


class YvEPS:
    DEFAULT = 1e-8
    LN = 1e-6
    LOG = 1e-10


class YvNumericalGuard:
    @staticmethod
    def get_eps(dtype: torch.dtype) -> float:
        if dtype in (torch.float16, torch.bfloat16):
            return max(YvEPS.DEFAULT, torch.finfo(dtype).eps * 10)
        return YvEPS.DEFAULT

    @staticmethod
    def safe_div(numerator: torch.Tensor, denominator: torch.Tensor, eps: float | None = None) -> torch.Tensor:
        if eps is None:
            eps = YvNumericalGuard.get_eps(denominator.dtype)
        return numerator / denominator.clamp(min=eps)

    @staticmethod
    def safe_log(x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
        if eps is None:
            eps = YvNumericalGuard.get_eps(x.dtype)
        return torch.log(x.clamp(min=eps))

    @staticmethod
    def safe_clamp(x: torch.Tensor, low: float, high: float | torch.Tensor) -> torch.Tensor:
        if isinstance(high, torch.Tensor):
            high = high.to(dtype=x.dtype, device=x.device)
            return torch.max(torch.min(x, high), torch.tensor(low, dtype=x.dtype, device=x.device))
        return torch.clamp(x, min=low, max=high)

    @staticmethod
    def nan_to_num(x: torch.Tensor, nan: float = 0.0, posinf: float | None = None, neginf: float | None = None) -> torch.Tensor:
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


class YvShapeGuard:
    @staticmethod
    def check_matmul(a: torch.Tensor, b: torch.Tensor, msg: str = "") -> None:
        if a.size(-1) != b.size(-2):
            raise RuntimeError(
                f"matmul shape mismatch: {a.shape} @ {b.shape}. "
                f"Last dim of first ({a.size(-1)}) != second-to-last of second ({b.size(-2)})."
                + (f" Context: {msg}" if msg else "")
            )

    @staticmethod
    def check_cat(tensors: list[torch.Tensor], dim: int, msg: str = "") -> None:
        if len(tensors) < 2:
            return
        ref_shape = list(tensors[0].shape)
        for i, t in enumerate(tensors[1:], 1):
            t_shape = list(t.shape)
            if len(t_shape) != len(ref_shape):
                raise RuntimeError(
                    f"cat shape mismatch at index {i}: ndim {len(t_shape)} != {len(ref_shape)}. "
                    f"Context: {msg}"
                )
            for d in range(len(ref_shape)):
                if d != dim and t_shape[d] != ref_shape[d]:
                    raise RuntimeError(
                        f"cat shape mismatch at index {i}, dim {d}: {t_shape[d]} != {ref_shape[d]}. "
                        f"Context: {msg}"
                    )


class YvDeviceGuard:
    @staticmethod
    def align(
        tensors: list[torch.Tensor],
        reference: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if reference is None:
            reference = tensors[0]
        return [
            t.to(device=reference.device, dtype=reference.dtype) if t is not None else t
            for t in tensors
        ]

    @staticmethod
    def assert_same_device(*tensors: torch.Tensor | None, msg: str = "") -> None:
        devices = {t.device for t in tensors if t is not None}
        if len(devices) > 1:
            raise RuntimeError(f"Device mismatch: {devices}. Context: {msg}")

    @staticmethod
    def assert_same_dtype(*tensors: torch.Tensor | None, msg: str = "") -> None:
        dtypes = {t.dtype for t in tensors if t is not None}
        if len(dtypes) > 1:
            raise RuntimeError(f"dtype mismatch: {dtypes}. Context: {msg}")