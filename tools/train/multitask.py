#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import torch
import torch.nn as nn
import math


class MultiTaskLossBalancer(nn.Module):
    """
    Multi-task loss balancer with dynamic weight adjustment strategies.

    This module implements several advanced multi-task learning weight balancing strategies:
    1. Uncertainty weighting (Kendall et al., 2018)
    2. GradNorm (Chen et al., 2018)
    3. Dynamic Task Prioritization (DTP)
    4. Loss magnitude balancing

    Args:
        num_tasks (int): Number of tasks
        strategy (str): Weight balancing strategy ('uncertainty', 'gradnorm', 'dtp', 'magnitude', 'adaptive')
        init_weights (list, optional): Initial task weights. Defaults to None (uniform).
        task_importance (list, optional): Static task importance weights. Defaults to None.
        alpha (float, optional): Weight smoothing coefficient. Defaults to 0.9.
        update_freq (int, optional): Weight update frequency in steps. Defaults to 100.
        gradnorm_alpha (float, optional): GradNorm restoration force. Defaults to 1.5.
        device (torch.device, optional): Device to place parameters. Defaults to None.
    """

    def __init__(self, num_tasks, strategy='adaptive', init_weights=None, task_importance=None,
                 alpha=0.9, update_freq=100, gradnorm_alpha=1.5, device=None):
        super(MultiTaskLossBalancer, self).__init__()
        self.num_tasks = num_tasks
        self.strategy = strategy
        self.alpha = alpha
        self.update_freq = update_freq
        self.gradnorm_alpha = gradnorm_alpha
        self.step_count = 0

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if init_weights is not None:
            assert len(init_weights) == num_tasks, "init_weights length must match num_tasks"
            self.task_weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32, device=device))
        else:
            self.task_weights = nn.Parameter(torch.ones(num_tasks, dtype=torch.float32, device=device))

        if task_importance is not None:
            assert len(task_importance) == num_tasks, "task_importance length must match num_tasks"
            self.register_buffer('task_importance', torch.tensor(task_importance, dtype=torch.float32, device=device))
        else:
            self.register_buffer('task_importance', torch.ones(num_tasks, dtype=torch.float32, device=device))

        self.register_buffer('ema_weights', self.task_weights.data.clone())
        self.register_buffer('loss_history', torch.zeros(num_tasks, dtype=torch.float32, device=device))
        self.register_buffer('grad_norm_history', torch.zeros(num_tasks, dtype=torch.float32, device=device))

        if strategy == 'uncertainty':
            self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32, device=device))

    def forward(self, task_losses, shared_params=None, return_weights=False):
        """
        Compute weighted multi-task loss.

        Args:
            task_losses (torch.Tensor): Tensor of shape (num_tasks,) containing individual task losses.
            shared_params (list, optional): List of shared parameters for GradNorm. Defaults to None.
            return_weights (bool, optional): Whether to return current weights. Defaults to False.

        Returns:
            torch.Tensor: Weighted total loss
            torch.Tensor (optional): Current task weights if return_weights=True
        """
        assert len(task_losses) == self.num_tasks, "task_losses length must match num_tasks"

        if self.strategy == 'uncertainty':
            total_loss = self._uncertainty_weighting(task_losses)
        elif self.strategy == 'gradnorm':
            total_loss = self._gradnorm_weighting(task_losses, shared_params)
        elif self.strategy == 'dtp':
            total_loss = self._dynamic_task_prioritization(task_losses)
        elif self.strategy == 'magnitude':
            total_loss = self._magnitude_balancing(task_losses)
        elif self.strategy == 'adaptive':
            total_loss = self._adaptive_weighting(task_losses, shared_params)
        else:
            total_loss = self._uniform_weighting(task_losses)

        if return_weights:
            return total_loss, self.get_effective_weights()
        return total_loss

    def _uniform_weighting(self, task_losses):
        """Uniform weighting baseline."""
        weights = torch.ones_like(task_losses)
        weighted_losses = weights * task_losses
        return weighted_losses.sum()

    def _uncertainty_weighting(self, task_losses):
        """
        Uncertainty weighting (Kendall et al., 2018).
        Learns task-dependent uncertainty parameters.
        """
        precision = torch.exp(-self.log_vars)
        weighted_losses = precision * task_losses + 0.5 * self.log_vars
        return weighted_losses.sum()

    def _gradnorm_weighting(self, task_losses, shared_params):
        """
        GradNorm (Chen et al., 2018).
        Balances gradient magnitudes across tasks.
        """
        if shared_params is None:
            return self._uniform_weighting(task_losses)

        weights = torch.clamp(self.task_weights, min=0.0)
        weighted_losses = weights * task_losses
        total_loss = weighted_losses.sum()

        if self.step_count % self.update_freq == 0 and shared_params:
            grad_norms = self._compute_grad_norms(task_losses, shared_params)
            avg_grad_norm = grad_norms.mean()
            loss_ratios = task_losses / (self.loss_history.mean(dim=1, keepdim=True).clamp(min=1e-6) + 1e-6)
            target_grad_norms = avg_grad_norm * (loss_ratios ** self.gradnorm_alpha)
            gradnorm_loss = torch.abs(grad_norms - target_grad_norms).sum()

            self.grad_norm_history = self.alpha * self.grad_norm_history + (1 - self.alpha) * grad_norms

        return total_loss

    def _dynamic_task_prioritization(self, task_losses):
        """
        Dynamic Task Prioritization (DTP).
        Adjusts weights based on task learning progress.
        """
        self.loss_history = self.alpha * self.loss_history + (1 - self.alpha) * task_losses.detach()

        loss_ratios = task_losses / (self.loss_history + 1e-6)
        progress_scores = 1.0 / (loss_ratios + 1e-6)

        weights = progress_scores * self.task_importance
        weights = weights / (weights.sum() + 1e-6) * self.num_tasks

        weighted_losses = weights * task_losses
        return weighted_losses.sum()

    def _magnitude_balancing(self, task_losses):
        """
        Loss magnitude balancing.
        Inversely proportional to loss magnitudes.
        """
        self.loss_history = self.alpha * self.loss_history + (1 - self.alpha) * task_losses.detach()

        inv_magnitude = 1.0 / (self.loss_history + 1e-6)
        weights = inv_magnitude / (inv_magnitude.sum() + 1e-6) * self.num_tasks
        weights = weights * self.task_importance

        weighted_losses = weights * task_losses
        return weighted_losses.sum()

    def _adaptive_weighting(self, task_losses, shared_params):
        """
        Adaptive weighting combining multiple strategies.
        """
        self.loss_history = self.alpha * self.loss_history + (1 - self.alpha) * task_losses.detach()

        loss_var = torch.var(self.loss_history, dim=0)
        stability_scores = 1.0 / (loss_var + 1e-6)

        inv_magnitude = 1.0 / (self.loss_history + 1e-6)
        magnitude_weights = inv_magnitude / (inv_magnitude.sum() + 1e-6)

        combined_weights = stability_scores * magnitude_weights * self.task_importance
        combined_weights = combined_weights / (combined_weights.sum() + 1e-6) * self.num_tasks

        self.ema_weights = self.alpha * self.ema_weights + (1 - self.alpha) * combined_weights.detach()
        effective_weights = self.ema_weights

        weighted_losses = effective_weights * task_losses
        return weighted_losses.sum()

    def _compute_grad_norms(self, task_losses, shared_params):
        """Compute gradient norms for each task w.r.t shared parameters."""
        grad_norms = []
        for i, loss in enumerate(task_losses):
            grads = torch.autograd.grad(loss, shared_params, retain_graph=True, allow_unused=True)
            grad_norm = 0.0
            for grad in grads:
                if grad is not None:
                    grad_norm += grad.norm(2).item() ** 2
            grad_norms.append(math.sqrt(grad_norm) + 1e-6)
        return torch.tensor(grad_norms, dtype=torch.float32, device=task_losses.device)

    def get_effective_weights(self):
        """Get current effective task weights."""
        if self.strategy == 'uncertainty':
            return torch.exp(-self.log_vars).detach()
        elif self.strategy == 'adaptive':
            return self.ema_weights.clone()
        else:
            return torch.clamp(self.task_weights, min=0.0).detach()

    def get_task_uncertainties(self):
        """Get task uncertainties (for uncertainty weighting)."""
        if self.strategy == 'uncertainty':
            return torch.exp(self.log_vars).detach()
        return None

    def update_step(self):
        """Increment step counter."""
        self.step_count += 1


class AdaptiveMultiTaskScheduler:
    """
    Adaptive scheduler for multi-task learning.

    Dynamically adjusts task learning rates and weights based on training progress.

    Args:
        num_tasks (int): Number of tasks
        base_lr (float): Base learning rate
        task_lr_factors (list, optional): Per-task learning rate factors. Defaults to None.
        weight_update_strategy (str): Strategy for updating task weights ('gradnorm', 'loss_ratio', 'none').
        warmup_steps (int, optional): Number of warmup steps. Defaults to 100.
        min_lr_factor (float, optional): Minimum learning rate factor. Defaults to 0.1.
        max_lr_factor (float, optional): Maximum learning rate factor. Defaults to 10.0.
    """

    def __init__(self, num_tasks, base_lr, task_lr_factors=None, weight_update_strategy='loss_ratio',
                 warmup_steps=100, min_lr_factor=0.1, max_lr_factor=10.0):
        self.num_tasks = num_tasks
        self.base_lr = base_lr
        self.weight_update_strategy = weight_update_strategy
        self.warmup_steps = warmup_steps
        self.min_lr_factor = min_lr_factor
        self.max_lr_factor = max_lr_factor
        self.step_count = 0

        if task_lr_factors is not None:
            assert len(task_lr_factors) == num_tasks, "task_lr_factors length must match num_tasks"
            self.task_lr_factors = torch.tensor(task_lr_factors, dtype=torch.float32)
        else:
            self.task_lr_factors = torch.ones(num_tasks, dtype=torch.float32)

        self.loss_history = []
        self.weight_history = []

    def get_task_lrs(self):
        """Get task-specific learning rates."""
        warmup_factor = min(1.0, self.step_count / self.warmup_steps)
        current_lr_factors = self.task_lr_factors * warmup_factor
        current_lr_factors = torch.clamp(current_lr_factors, self.min_lr_factor, self.max_lr_factor)
        return self.base_lr * current_lr_factors

    def update(self, task_losses, task_weights=None):
        """
        Update scheduler state based on task losses.

        Args:
            task_losses (list): List of task losses.
            task_weights (list, optional): Current task weights. Defaults to None.
        """
        self.step_count += 1
        self.loss_history.append(task_losses)

        if self.weight_update_strategy == 'loss_ratio' and len(self.loss_history) > 1:
            self._update_by_loss_ratio()

        if task_weights is not None:
            self.weight_history.append(task_weights)

    def _update_by_loss_ratio(self):
        """Update task learning rates based on loss ratios."""
        if len(self.loss_history) < 2:
            return

        recent_losses = torch.tensor(self.loss_history[-1])
        prev_losses = torch.tensor(self.loss_history[-2])

        loss_ratios = recent_losses / (prev_losses + 1e-6)

        improvement_factors = 1.0 / (loss_ratios + 1e-6)
        improvement_factors = torch.clamp(improvement_factors, 0.5, 2.0)

        self.task_lr_factors = self.task_lr_factors * improvement_factors
        self.task_lr_factors = torch.clamp(self.task_lr_factors, self.min_lr_factor, self.max_lr_factor)

    def get_state(self):
        """Get scheduler state."""
        return {
            'step_count': self.step_count,
            'task_lr_factors': self.task_lr_factors.tolist(),
            'loss_history': self.loss_history[-100:] if self.loss_history else [],
            'weight_history': self.weight_history[-100:] if self.weight_history else []
        }
