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
import math


class MuonParameterError(ValueError):
    """Exception raised for invalid parameters in Muon optimizer.

    This exception is used when input parameters do not meet the requirements
    for Muon optimization operations, such as incorrect tensor dimensions or
    invalid numerical values. It inherits from ValueError to maintain
    compatibility with existing exception handling patterns that catch
    standard Python exceptions.
    """
    pass


def _hybrid_newton_schulz(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Hybrid Newton-Schulz matrix orthogonalization.

    Args:
        G: Input matrix of shape (n, m)
        steps: Total iterations (default: 10, first 8 aggressive, last 2 conservative)

    Returns:
        Orthogonalized matrix O ~= (G G^T)^{-1/2} G

    Raises:
        MuonParameterError: If input tensor does not have exactly 2 dimensions.
    """
    # Structured parameter validation
    if G.ndim != 2:
        raise MuonParameterError(
            f"Muon requires 2D parameter tensor, but got {G.ndim}D tensor with shape {G.shape}. "
            f"Expected shape: (n, m) where n and m are positive integers. "
            f"This error typically occurs when the optimizer is applied to incompatible "
            f"parameter types (e.g., 1D biases, 3D convolution weights). "
            f"Please check your parameter filtering logic in create_muon_optimizer()."
        )
    n, m = G.shape

    if min(n, m) <= 1:
        return G.sign() * math.sqrt(max(n, m))

    # Normalize for numerical stability
    scale = G.norm().item()
    G = G / (scale + 1e-20)

    # Hybrid coefficients
    agg_coeffs = (3.4445, -4.7750, 2.0315)
    cons_coeffs = (2.0, -1.5, 0.5)

    if n >= m:
        X = G.T @ G
        is_transposed = False
    else:
        X = G @ G.T
        is_transposed = True

    # Newton-Schulz iterations
    for i in range(steps):
        a, b, c = agg_coeffs if i < 8 else cons_coeffs
        X = a * X + b * (X @ X) + c * (X @ (X @ X))

    if is_transposed:
        O = (X @ G.T).T
    else:
        O = G @ X

    return O * scale


class YvMuon(torch.optim.Optimizer):
    """Muon optimizer — orthogonalized momentum for MoE training.

    Features:
        - Newton-Schulz momentum orthogonalization
        - Hybrid aggressive/conservative iterations
        - RMS rescaling (gamma=0.18 for AdamW LR reuse)
        - ZeRO-3 compatible bucket allocation
        - Per-parameter group configuration

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate. Default: 2e-4 (Flash), 2.0e-4 (Pro).
        momentum: Momentum coefficient. Default: 0.95.
        nesterov: Use Nesterov momentum. Default: True.
        ns_steps: Newton-Schulz iterations. Default: 10.
        gamma: RMS rescale factor. Default: 0.18.
        weight_decay: Weight decay coefficient. Default: 0.1.
        adamw_params: Subset of params to use AdamW instead (embeddings, norms).
        adamw_lr: Learning rate for AdamW params. Default: same as lr.

    Usage:
        >>> muon_params = [p for n, p in model.named_parameters()
        ...                if p.ndim >= 2 and 'embed' not in n and 'norm' not in n]
        >>> adamw_params = [p for n, p in model.named_parameters()
        ...                 if p.ndim < 2 or 'embed' in n or 'norm' in n]
        >>> optimizer = YvMuon([
        ...     {'params': muon_params},
        ...     {'params': adamw_params, 'use_muon': False, 'lr': 1e-5},
        ... ], lr=2e-4)
    """

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 10,
        gamma: float = 0.18,
        weight_decay: float = 0.1,
        adamw_params: list = None,
        adamw_lr: float = None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            gamma=gamma,
            weight_decay=weight_decay,
            use_muon=True,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            if not group.get('use_muon', True):
                group['use_muon'] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            gamma = group['gamma']
            wd = group['weight_decay']
            use_muon = group.get('use_muon', True)

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]

                if use_muon and p.ndim >= 2:
                    # Muon update
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)

                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)

                    if nesterov:
                        update = g.add(buf, alpha=momentum)
                    else:
                        update = buf.clone()

                    # Orthogonalize via hybrid Newton-Schulz
                    O = _hybrid_newton_schulz(update, steps=ns_steps)

                    # RMS rescale
                    target_norm = math.sqrt(max(p.shape[0], p.shape[1])) * gamma
                    current_norm = O.norm()
                    if current_norm > 0:
                        O.mul_(target_norm / current_norm)

                    # Weight decay + update
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(O, alpha=-lr)

                else:
                    # AdamW for non-Muon params (embeddings, biases, norms)
                    if 'adam_step' not in state:
                        state['adam_step'] = 0
                        state['adam_exp_avg'] = torch.zeros_like(g)
                        state['adam_exp_avg_sq'] = torch.zeros_like(g)

                    betas = (0.9, 0.95)
                    eps = 1e-8

                    exp_avg = state['adam_exp_avg']
                    exp_avg_sq = state['adam_exp_avg_sq']
                    state['adam_step'] += 1

                    exp_avg.mul_(betas[0]).add_(g, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(g.square(), alpha=1 - betas[1])

                    bias_corr1 = 1 - betas[0] ** state['adam_step']
                    bias_corr2 = 1 - betas[1] ** state['adam_step']

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)

                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(exp_avg / bias_corr1, denom, value=-lr)

        return loss


def create_muon_optimizer(
    model,
    lr: float = 2e-4,
    weight_decay: float = 0.1,
    muon_exclude_patterns: tuple = ('embed', 'norm', 'bias', 'lm_head'),
    muon_lr: float = None,
    adamw_lr: float = 1e-5,
):
    """Convenience function to create a Muon optimizer for a Yv model.

    Automatically splits parameters into Muon-eligible (2D weights, no norms/embeds)
    and AdamW (everything else).

    Args:
        model: The YvModel instance.
        lr: Base learning rate for Muon params. Default: 2e-4.
        weight_decay: Weight decay. Default: 0.1.
        muon_exclude_patterns: Parameter name patterns to exclude from Muon.
        muon_lr: LR for Muon groups (defaults to lr).
        adamw_lr: LR for AdamW groups. Default: 1e-5.

    Returns:
        YvMuon optimizer instance.
    """
    muon_lr = muon_lr or lr
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_excluded = any(pat in name.lower() for pat in muon_exclude_patterns)
        if is_excluded or p.ndim < 2:
            adamw_params.append(p)
        else:
            muon_params.append(p)

    param_groups = [
        {'params': muon_params, 'use_muon': True, 'lr': muon_lr},
        {'params': adamw_params, 'use_muon': False, 'lr': adamw_lr},
    ]

    return YvMuon(param_groups, lr=lr, weight_decay=weight_decay)
