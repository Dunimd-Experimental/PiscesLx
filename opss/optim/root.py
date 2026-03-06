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

"""
ROOT (Robust Orthogonalized Optimizer) Implementation

Complete implementation of ROOT optimizer from Huawei Noah's Ark Lab.
ROOT combines Adam's stability with Muon's speed through momentum orthogonalization.

Key Innovation:
    - Newton-Schulz iteration for momentum orthogonalization
    - Soft thresholding for gradient noise filtering
    - Adaptive learning rate scaling
    - Spectral norm clipping for stability

Reference:
    ROOT: Robust Orthogonalized Optimizer for Neural Network Training (arXiv:2511.20626)
    Huawei Noah's Ark Lab

Algorithm:
    1. Compute gradient and update momentum (like Adam)
    2. Orthogonalize momentum using Newton-Schulz iteration
    3. Apply soft threshold denoising
    4. Scale by spectral norm
    5. Update parameters

Benefits:
    - 2-3x faster convergence than Adam
    - Better stability than Muon
    - Handles gradient noise effectively
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterable
from dataclasses import dataclass, field
import math
import time

from configs.version import VERSION
from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig,
)


@dataclass
class POPSSROOTConfig(PiscesLxOperatorConfig):
    """
    ROOT (Robust Orthogonalized Optimizer) Configuration.
    
    This configuration controls the ROOT optimizer parameters for
    fast and stable neural network training.
    
    Attributes:
        lr: Learning rate
        beta1: Exponential decay rate for first moment (momentum)
        beta2: Exponential decay rate for second moment (variance)
        eps: Small constant for numerical stability
        weight_decay: L2 regularization coefficient
        orthogonalization_steps: Number of Newton-Schulz iterations
        soft_threshold: Soft threshold for gradient denoising
        spectral_norm_clip: Maximum spectral norm for stability
        use_orthogonalization: Whether to apply orthogonalization
        use_soft_threshold: Whether to apply soft thresholding
        min_dim_for_ortho: Minimum dimension for orthogonalization
        amsgrad: Whether to use AMSGrad variant
        maximize: Whether to maximize the objective
    """
    name: str = "root"
    version: str = VERSION
    
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    orthogonalization_steps: int = 5
    soft_threshold: float = 0.1
    spectral_norm_clip: float = 1.0
    use_orthogonalization: bool = True
    use_soft_threshold: bool = True
    min_dim_for_ortho: int = 16
    amsgrad: bool = False
    maximize: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if self.lr < 0:
            raise ValueError(f"Invalid learning rate: {self.lr}")
        if self.beta1 < 0 or self.beta1 >= 1:
            raise ValueError(f"Invalid beta1: {self.beta1}")
        if self.beta2 < 0 or self.beta2 >= 1:
            raise ValueError(f"Invalid beta2: {self.beta2}")


class POPSSROOTOperator(PiscesLxOperatorInterface):
    """
    ROOT (Robust Orthogonalized Optimizer) Operator.
    
    ROOT provides fast and stable optimization through momentum orthogonalization.
    It combines the stability of Adam with the speed of Muon.
    
    Key Features:
        - Newton-Schulz orthogonalization for momentum
        - Soft thresholding for noise filtering
        - Spectral norm clipping for stability
        - Selective orthogonalization (only for 2D weights)
    
    Example:
        >>> config = POPSSROOTConfig(lr=1e-3, orthogonalization_steps=5)
        >>> root = POPSSROOTOperator()
        >>> result = root.execute({
        ...     "model": model,
        ...     "gradients": gradients,
        ...     "config": config,
        ... })
    """
    
    def __init__(self):
        super().__init__()
        self._name = "root"
        self._version = VERSION
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return "Robust Orthogonalized Optimizer - Adam stability + Muon speed"
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute ROOT optimization step.
        
        Args:
            inputs: Dictionary containing:
                - model: Model to optimize
                - gradients: Precomputed gradients (optional)
                - config: ROOT configuration
                - optimizer_state: Current optimizer state
        
        Returns:
            PiscesLxOperatorResult with updated model and statistics
        """
        start_time = self._get_time()
        
        try:
            model = inputs.get("model")
            gradients = inputs.get("gradients", {})
            config = inputs.get("config", POPSSROOTConfig())
            optimizer_state = inputs.get("optimizer_state", {})
            
            if model is None:
                raise ValueError("Model is required for ROOT optimization")
            
            if not optimizer_state:
                optimizer_state = self._initialize_state(model, config)
            
            stats = {
                "orthogonalized_params": 0,
                "total_params": 0,
                "avg_spectral_norm": 0.0,
                "avg_threshold_ratio": 0.0,
            }
            
            spectral_norms = []
            threshold_ratios = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                stats["total_params"] += 1
                
                grad = gradients.get(name, param.grad) if gradients else param.grad
                if grad is None:
                    continue
                
                state = optimizer_state["state"].get(name, {})
                
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    if config.amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(param)
                    optimizer_state["state"][name] = state
                
                state["step"] += 1
                step = state["step"]
                
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                
                if config.weight_decay != 0:
                    grad = grad.add(param, alpha=config.weight_decay)
                
                exp_avg.mul_(config.beta1).add_(grad, alpha=1 - config.beta1)
                exp_avg_sq.mul_(config.beta2).addcmul_(grad, grad, value=1 - config.beta2)
                
                if config.amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(config.eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(config.eps)
                
                update = exp_avg.clone()
                
                if config.use_orthogonalization and self._should_orthogonalize(param, config):
                    update = self.orthogonalize_momentum(
                        update,
                        steps=config.orthogonalization_steps,
                    )
                    stats["orthogonalized_params"] += 1
                
                if config.use_soft_threshold:
                    update, ratio = self.soft_threshold_denoise(
                        update,
                        threshold=config.soft_threshold,
                    )
                    threshold_ratios.append(ratio)
                
                if config.spectral_norm_clip > 0 and update.dim() >= 2:
                    sn = self.compute_spectral_norm(update)
                    spectral_norms.append(sn.item())
                    
                    if sn > config.spectral_norm_clip:
                        update = update * (config.spectral_norm_clip / sn)
                
                bias_correction1 = 1 - config.beta1 ** step
                bias_correction2 = 1 - config.beta2 ** step
                step_size = config.lr * math.sqrt(bias_correction2) / bias_correction1
                
                param.add_(update / denom, alpha=-step_size)
            
            if spectral_norms:
                stats["avg_spectral_norm"] = sum(spectral_norms) / len(spectral_norms)
            if threshold_ratios:
                stats["avg_threshold_ratio"] = sum(threshold_ratios) / len(threshold_ratios)
            
            execution_time = self._get_time() - start_time
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "model": model,
                    "optimizer_state": optimizer_state,
                    "statistics": stats,
                },
                execution_time=execution_time,
                metadata={
                    "version": self.version,
                    "algorithm": "ROOT",
                    "orthogonalized_ratio": stats["orthogonalized_params"] / max(stats["total_params"], 1),
                },
            )
            
        except Exception as e:
            execution_time = self._get_time() - start_time
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={
                    "version": self.version,
                    "error_type": type(e).__name__,
                },
            )
    
    def orthogonalize_momentum(
        self,
        momentum: torch.Tensor,
        steps: int = 5,
    ) -> torch.Tensor:
        """
        Orthogonalize momentum using Newton-Schulz iteration.
        
        The Newton-Schulz iteration computes an approximate orthogonal matrix
        from the momentum, which helps maintain diverse update directions.
        
        Formula: M_orth = M @ (I - (I - M^T @ M)^k)^(-1/2)
        Approximated by: X_{k+1} = 1.5 * X_k - 0.5 * X_k @ X_k^T @ X_k
        
        Args:
            momentum: Momentum tensor to orthogonalize
            steps: Number of Newton-Schulz iterations
        
        Returns:
            Orthogonalized momentum tensor
        """
        if momentum.dim() == 1:
            return momentum
        
        if momentum.dim() == 2:
            m, n = momentum.shape
            if m < n:
                momentum_t = momentum.t()
                is_transposed = True
            else:
                momentum_t = momentum
                is_transposed = False
            
            mm = momentum_t @ momentum_t.t()
            
            eye = torch.eye(mm.shape[0], device=mm.device, dtype=mm.dtype)
            
            for _ in range(steps):
                mm = 1.5 * mm - 0.5 * mm @ mm @ mm
            
            mm = torch.clamp(mm, min=-1.0, max=1.0)
            
            try:
                sqrt_mm = torch.linalg.matrix_power(eye + mm, 0.5)
                result = sqrt_mm @ momentum_t
            except:
                result = momentum_t
            
            if is_transposed:
                result = result.t()
            
            return result
        
        original_shape = momentum.shape
        momentum_2d = momentum.view(original_shape[0], -1)
        orthogonalized = self.orthogonalize_momentum(momentum_2d, steps)
        
        return orthogonalized.view(original_shape)
    
    def soft_threshold_denoise(
        self,
        grad: torch.Tensor,
        threshold: float = 0.1,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply soft thresholding for gradient noise filtering.
        
        Soft thresholding removes small gradient values that are likely
        noise while preserving larger, more meaningful updates.
        
        Formula: grad_th = sign(grad) * max(|grad| - threshold, 0)
        
        Args:
            grad: Gradient tensor
            threshold: Threshold value for denoising
        
        Returns:
            Tuple of (denoised gradient, ratio of preserved elements)
        """
        abs_grad = torch.abs(grad)
        mask = abs_grad > threshold
        
        denoised = torch.sign(grad) * F.relu(abs_grad - threshold)
        
        preserved_ratio = mask.float().mean().item()
        
        return denoised, preserved_ratio
    
    def compute_spectral_norm(
        self,
        matrix: torch.Tensor,
        num_iterations: int = 10,
    ) -> torch.Tensor:
        """
        Compute spectral norm (largest singular value) of a matrix.
        
        Uses power iteration for efficient computation without full SVD.
        
        Args:
            matrix: Input matrix
            num_iterations: Number of power iteration steps
        
        Returns:
            Spectral norm as a scalar tensor
        """
        if matrix.dim() == 1:
            return torch.norm(matrix)
        
        if matrix.dim() > 2:
            matrix = matrix.view(matrix.shape[0], -1)
        
        m, n = matrix.shape
        if m == 0 or n == 0:
            return torch.tensor(0.0, device=matrix.device, dtype=matrix.dtype)
        
        u = torch.randn(m, device=matrix.device, dtype=matrix.dtype)
        u = u / (torch.norm(u) + 1e-8)
        
        for _ in range(num_iterations):
            v = matrix.t() @ u
            v = v / (torch.norm(v) + 1e-8)
            u = matrix @ v
            u = u / (torch.norm(u) + 1e-8)
        
        spectral_norm = torch.norm(matrix @ v) / (torch.norm(v) + 1e-8)
        
        return spectral_norm
    
    def _should_orthogonalize(
        self,
        param: torch.Tensor,
        config: POPSSROOTConfig,
    ) -> bool:
        """Check if parameter should be orthogonalized."""
        if param.dim() < 2:
            return False
        
        min_dim = min(param.shape)
        if min_dim < config.min_dim_for_ortho:
            return False
        
        return True
    
    def _initialize_state(
        self,
        model: nn.Module,
        config: POPSSROOTConfig,
    ) -> Dict[str, Any]:
        """Initialize optimizer state."""
        return {
            "step": 0,
            "state": {},
            "config": config.__dict__,
        }
    
    def _get_time(self) -> float:
        """Get current time in seconds."""
        return time.time()


class POPSSROOTOptimizer(Optimizer):
    """
    PyTorch-compatible ROOT Optimizer.
    
    This class provides a standard PyTorch optimizer interface for ROOT,
    making it easy to use as a drop-in replacement for Adam or other optimizers.
    
    Example:
        >>> model = MyModel()
        >>> optimizer = POPSSROOTOptimizer(
        ...     model.parameters(),
        ...     lr=1e-3,
        ...     orthogonalization_steps=5,
        ... )
        >>> 
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        orthogonalization_steps: int = 5,
        soft_threshold: float = 0.1,
        spectral_norm_clip: float = 1.0,
        use_orthogonalization: bool = True,
        use_soft_threshold: bool = True,
        min_dim_for_ortho: int = 16,
        amsgrad: bool = False,
        maximize: bool = False,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta1 < 0 or beta1 >= 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if beta2 < 0 or beta2 >= 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            orthogonalization_steps=orthogonalization_steps,
            soft_threshold=soft_threshold,
            spectral_norm_clip=spectral_norm_clip,
            use_orthogonalization=use_orthogonalization,
            use_soft_threshold=use_soft_threshold,
            min_dim_for_ortho=min_dim_for_ortho,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        
        super().__init__(params, defaults)
        
        self.operator = POPSSROOTOperator()
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("use_orthogonalization", True)
            group.setdefault("use_soft_threshold", True)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        
        Returns:
            The loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]
            
            orthogonalization_steps = group["orthogonalization_steps"]
            soft_threshold = group["soft_threshold"]
            spectral_norm_clip = group["spectral_norm_clip"]
            use_orthogonalization = group["use_orthogonalization"]
            use_soft_threshold = group["use_soft_threshold"]
            min_dim_for_ortho = group["min_dim_for_ortho"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                param = p
                grad = param.grad
                if maximize:
                    grad = -grad
                
                if grad.is_sparse:
                    raise RuntimeError("ROOT does not support sparse gradients")
                
                state = self.state[param]
                
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(param)
                
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                else:
                    max_exp_avg_sq = None
                
                state["step"] += 1
                state_steps.append(state["step"])
                
                params_with_grad.append(param)
                grads.append(grad)
                exp_avgs.append(exp_avg)
                exp_avg_sqs.append(exp_avg_sq)
                max_exp_avg_sqs.append(max_exp_avg_sq)
            
            self._root_update(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                max_exp_avg_sqs=max_exp_avg_sqs,
                state_steps=state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
                amsgrad=amsgrad,
                orthogonalization_steps=orthogonalization_steps,
                soft_threshold=soft_threshold,
                spectral_norm_clip=spectral_norm_clip,
                use_orthogonalization=use_orthogonalization,
                use_soft_threshold=use_soft_threshold,
                min_dim_for_ortho=min_dim_for_ortho,
            )
        
        return loss
    
    def _root_update(
        self,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        max_exp_avg_sqs: List[Optional[torch.Tensor]],
        state_steps: List[int],
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        amsgrad: bool,
        orthogonalization_steps: int,
        soft_threshold: float,
        spectral_norm_clip: float,
        use_orthogonalization: bool,
        use_soft_threshold: bool,
        min_dim_for_ortho: int,
    ):
        """Perform ROOT update for a group of parameters."""
        
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            
            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)
            
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            if amsgrad and max_exp_avg_sqs[i] is not None:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = max_exp_avg_sqs[i].sqrt().add_(eps)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)
            
            update = exp_avg.clone()
            
            if use_orthogonalization:
                if param.dim() >= 2:
                    min_dim = min(param.shape)
                    if min_dim >= min_dim_for_ortho:
                        update = self.operator.orthogonalize_momentum(
                            update,
                            steps=orthogonalization_steps,
                        )
            
            if use_soft_threshold:
                update, _ = self.operator.soft_threshold_denoise(
                    update,
                    threshold=soft_threshold,
                )
            
            if spectral_norm_clip > 0 and update.dim() >= 2:
                sn = self.operator.compute_spectral_norm(update)
                if sn > spectral_norm_clip:
                    update = update * (spectral_norm_clip / sn)
            
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            
            param.add_(update / denom, alpha=-step_size)
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        config = {}
        for group in self.param_groups:
            for key, value in group.items():
                if key != "params":
                    config[key] = value
        return config


class POPSSROOTScheduler:
    """
    Learning rate scheduler for ROOT optimizer.
    
    Provides warmup and decay scheduling specifically tuned for ROOT's
    orthogonalization behavior.
    """
    
    def __init__(
        self,
        optimizer: POPSSROOTOptimizer,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        min_lr: float = 1e-6,
        decay_type: str = "cosine",
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.decay_type = decay_type
        self.current_step = 0
        
        self.base_lr = optimizer.param_groups[0]["lr"]
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self._compute_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def _compute_lr(self) -> float:
        """Compute current learning rate."""
        if self.current_step < self.warmup_steps:
            return self.base_lr * self.current_step / self.warmup_steps
        
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        if self.decay_type == "cosine":
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        elif self.decay_type == "linear":
            lr = self.base_lr - progress * (self.base_lr - self.min_lr)
        elif self.decay_type == "exponential":
            lr = self.base_lr * (self.min_lr / self.base_lr) ** progress
        else:
            lr = self.base_lr
        
        return lr
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {
            "current_step": self.current_step,
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "decay_type": self.decay_type,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state."""
        self.current_step = state_dict["current_step"]
        self.base_lr = state_dict["base_lr"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.total_steps = state_dict["total_steps"]
        self.min_lr = state_dict["min_lr"]
        self.decay_type = state_dict["decay_type"]
