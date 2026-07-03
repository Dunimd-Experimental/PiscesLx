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
Fused AdamW Optimizer — Triton kernel fusing the entire AdamW update
into a single kernel launch per parameter.

Replaces 6+ PyTorch kernel launches (mul, sqrt, div, add, copy_ etc.)
with one Triton kernel that reads (p, g, m, v), computes the full update,
and writes (p, m, v) in a single pass.

Usage:
    >>> from opss.kernels.fused_adamw import FusedAdamW
    >>> optimizer = FusedAdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    >>> # Fully compatible with optimizer.state_dict(), checkpointing, etc.
    >>> # Falls back to torch.optim.AdamW when Triton is unavailable.
"""

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.optim import Optimizer

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except (ImportError, RuntimeError):
    pass


if _HAS_TRITON:

    @triton.jit
    def _adamw_kernel(
        p_ptr, g_ptr, m_ptr, v_ptr,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        bias_correction1: float,
        bias_correction2: float,
        maximize: float,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

        g = tl.where(maximize > 0.5, -g, g)

        m_new = beta1 * m + (1 - beta1) * g
        v_new = beta2 * v + (1 - beta2) * g * g
        m_hat = m_new / bias_correction1
        v_hat = v_new / bias_correction2

        denom = tl.sqrt(v_hat) + eps
        p_new = p - lr * m_hat / denom - lr * weight_decay * p

        tl.store(p_ptr + offsets, p_new, mask=mask)
        tl.store(m_ptr + offsets, m_new, mask=mask)
        tl.store(v_ptr + offsets, v_new, mask=mask)

    @triton.jit
    def _adamw_amsgrad_kernel(
        p_ptr, g_ptr, m_ptr, v_ptr, max_v_ptr,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        bias_correction1: float,
        bias_correction2: float,
        maximize: float,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
        max_v = tl.load(max_v_ptr + offsets, mask=mask, other=0.0)

        g = tl.where(maximize > 0.5, -g, g)

        m_new = beta1 * m + (1 - beta1) * g
        v_new = beta2 * v + (1 - beta2) * g * g
        max_v_new = tl.maximum(max_v, v_new)
        m_hat = m_new / bias_correction1
        v_hat = max_v_new / bias_correction2

        denom = tl.sqrt(v_hat) + eps
        p_new = p - lr * m_hat / denom - lr * weight_decay * p

        tl.store(p_ptr + offsets, p_new, mask=mask)
        tl.store(m_ptr + offsets, m_new, mask=mask)
        tl.store(v_ptr + offsets, v_new, mask=mask)
        tl.store(max_v_ptr + offsets, max_v_new, mask=mask)

    _BLOCK_SIZE = 1024

    def _launch_adamw(
        p: torch.Tensor,
        g: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        max_v: Optional[torch.Tensor],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        step: int,
        maximize: bool,
    ):
        n = p.numel()
        grid = (triton.cdiv(n, _BLOCK_SIZE),)

        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        maximize_f = 1.0 if maximize else 0.0

        if max_v is not None:
            _adamw_amsgrad_kernel[grid](
                p, g, m, v, max_v,
                lr, beta1, beta2, eps, weight_decay,
                bc1, bc2, maximize_f, n,
                BLOCK_SIZE=_BLOCK_SIZE,
            )
        else:
            _adamw_kernel[grid](
                p, g, m, v,
                lr, beta1, beta2, eps, weight_decay,
                bc1, bc2, maximize_f, n,
                BLOCK_SIZE=_BLOCK_SIZE,
            )


    class FusedAdamW(Optimizer):
        """Triton-fused AdamW optimizer.

        Drop-in replacement for torch.optim.AdamW. The entire parameter update
        is computed in a single Triton kernel per parameter, reducing kernel
        launch overhead from 6+ to 1 per parameter.

        Falls back to torch.optim.AdamW when Triton is unavailable or the
        parameter is not on CUDA (via the `_pytorch_adamw` helper).
        """

        def __init__(
            self,
            params: Iterable[Union[Dict[str, Any], torch.Tensor]],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01,
            amsgrad: bool = False,
            maximize: bool = False,
        ):
            if not 0.0 <= lr:
                raise ValueError(f"Invalid lr: {lr}")
            if not 0.0 <= eps:
                raise ValueError(f"Invalid eps: {eps}")
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            if not 0.0 <= weight_decay:
                raise ValueError(f"Invalid weight_decay: {weight_decay}")

            defaults = {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": amsgrad,
                "maximize": maximize,
            }
            super().__init__(params, defaults)

        def __setstate__(self, state):
            super().__setstate__(state)
            for group in self.param_groups:
                group.setdefault("amsgrad", False)
                group.setdefault("maximize", False)

        @torch.no_grad()
        def step(self, closure: Optional[Callable] = None) -> Optional[float]:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("FusedAdamW does not support sparse gradients")

                    state = self.state[p]

                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state["step"] += 1

                    m = state["exp_avg"]
                    v = state["exp_avg_sq"]
                    max_v = state.get("max_exp_avg_sq", None)

                    if p.is_cuda and _HAS_TRITON:
                        _launch_adamw(
                            p.data, grad.data, m, v, max_v,
                            lr=group["lr"],
                            beta1=beta1,
                            beta2=beta2,
                            eps=group["eps"],
                            weight_decay=group["weight_decay"],
                            step=state["step"],
                            maximize=group["maximize"],
                        )
                    else:
                        _pytorch_adamw(
                            p.data, grad.data, m, v, max_v,
                            lr=group["lr"],
                            beta1=beta1,
                            beta2=beta2,
                            eps=group["eps"],
                            weight_decay=group["weight_decay"],
                            step=state["step"],
                            maximize=group["maximize"],
                        )

            return loss

else:

    class FusedAdamW(Optimizer):
        """Fallback FusedAdamW — delegates to torch.optim.AdamW when Triton is unavailable."""

        def __init__(
            self,
            params: Iterable[Union[Dict[str, Any], torch.Tensor]],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01,
            amsgrad: bool = False,
            maximize: bool = False,
        ):
            import warnings
            warnings.warn("Triton not available — FusedAdamW falling back to torch.optim.AdamW")
            self._impl = torch.optim.AdamW(
                params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize,
            )

        def __getattr__(self, name):
            if name == "_impl":
                raise AttributeError(name)
            return getattr(self._impl, name)

        def step(self, closure=None):
            return self._impl.step(closure)


def _pytorch_adamw(
    p: torch.Tensor,
    g: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    max_v: Optional[torch.Tensor],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    maximize: bool,
):
    if maximize:
        g = -g

    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

    if max_v is not None:
        torch.maximum(max_v, v, out=max_v)
        v_hat = max_v
    else:
        v_hat = v

    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    step_size = lr / bias_correction1

    denom = v_hat.sqrt().add_(eps)
    p.addcdiv_(m, denom, value=-step_size)
    p.add_(p, alpha=-lr * weight_decay)