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
Numerical routines that are safe under low-precision dtypes (fp16/bf16).

`torch.linalg.qr`, `torch.linalg.svd`, and the rest of the LAPACK-backed
factorizations dispatch on CUDA to kernels (`geqrf_cuda`, `gesvd_cuda`,
etc.) that historically have not supported half precision. Several
parameter-initialization paths in this project need to perform such
factorizations on tensors whose dtype follows the model dtype, so they
break the moment the model is moved to fp16/bf16.

The two helpers below always compute the decomposition in fp32 and cast
the result back to the caller's dtype. They are only meant to be used on
parameter-initialization paths, where the extra upcast is a one-time cost
and never a hot-path concern.
"""

import torch


def qr_safe(matrix: torch.Tensor) -> torch.Tensor:
    """
    QR decomposition that is safe under low-precision dtypes (fp16/bf16).

    Args:
        matrix: Input matrix of any floating dtype.

    Returns:
        Orthogonal factor ``Q`` with the same dtype as ``matrix``.
    """
    original_dtype = matrix.dtype
    if original_dtype not in (torch.float16, torch.bfloat16):
        q, _ = torch.linalg.qr(matrix)
        return q
    q, _ = torch.linalg.qr(matrix.to(torch.float32))
    return q.to(original_dtype)


def svd_safe(matrix: torch.Tensor, full_matrices: bool = False):
    """
    SVD that is safe under low-precision dtypes (fp16/bf16).

    Args:
        matrix: Input matrix of any floating dtype.
        full_matrices: Forwarded to ``torch.linalg.svd``.

    Returns:
        Tuple ``(U, S, V)`` where each tensor keeps the original dtype.
    """
    original_dtype = matrix.dtype
    if original_dtype not in (torch.float16, torch.bfloat16):
        return torch.linalg.svd(matrix, full_matrices=full_matrices)
    u, s, v = torch.linalg.svd(matrix.to(torch.float32), full_matrices=full_matrices)
    return u.to(original_dtype), s.to(original_dtype), v.to(original_dtype)


__all__ = ["qr_safe", "svd_safe"]
