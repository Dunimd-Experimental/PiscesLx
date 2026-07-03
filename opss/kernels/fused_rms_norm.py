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
import torch.nn as nn
from typing import Optional

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except (ImportError, RuntimeError):
    pass


if _HAS_TRITON:

    @triton.jit
    def _rms_norm_fwd_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        stride_x_row: int,
        stride_y_row: int,
        N: int,
        eps: float,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_x_row
        y_row = y_ptr + row * stride_y_row

        acc = tl.zeros([], dtype=tl.float32)
        for start in range(0, N, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x = tl.load(x_row + offsets, mask=mask, other=0.0)
            acc += tl.sum(x * x.to(tl.float32))

        rms = tl.sqrt(acc / N + eps)

        for start in range(0, N, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x = tl.load(x_row + offsets, mask=mask, other=0.0)
            w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
            tl.store(y_row + offsets, (x / rms) * w, mask=mask)


    class _FusedRMSNormFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, weight, eps):
            shape = x.shape
            x_2d = x.reshape(-1, shape[-1])
            N = x_2d.shape[-1]
            M = x_2d.shape[0]
            output = torch.empty_like(x_2d)

            BLOCK_SIZE = min(4096, triton.next_power_of_2(N))

            grid = (M,)
            _rms_norm_fwd_kernel[grid](
                x_2d,
                weight,
                output,
                x_2d.stride(0),
                output.stride(0),
                N,
                eps,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            ctx.save_for_backward(x_2d, weight)
            ctx.eps = eps
            return output.reshape(shape)

        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            eps = ctx.eps
            N = x.shape[-1]
            s = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            n = x / s
            d_y = grad_output.reshape(x.shape)
            d_x = (d_y * weight) / s - (d_y * weight * n.pow(2)).sum(
                -1, keepdim=True
            ) / (s * N)
            d_weight = (d_y * n).sum(dim=list(range(d_y.ndim - 1)))
            return d_x.reshape(grad_output.shape), d_weight, None


    def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Triton-fused RMSNorm: x -> rms * x * weight in a single kernel pass.

        Requires CUDA + Triton. Falls back to PyTorch if Triton is unavailable
        or if the input is not on CUDA.
        """
        if not x.is_cuda or not weight.is_cuda:
            return _pytorch_rms_norm(x, weight, eps)
        return _FusedRMSNormFn.apply(x, weight, eps)


    def _pytorch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return weight * x * rms


else:

    def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return weight * x * rms