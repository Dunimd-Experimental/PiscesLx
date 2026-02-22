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

"""
Task Uncertainty Multi-Task Learning Operator

This operator implements uncertainty-based multi-task learning, where task-specific
losses are weighted by their learned uncertainty parameters. This approach,
known as Homoscedastic Uncertainty Weighting, allows automatic balancing of
multiple task losses without manual tuning of loss weights.

Key Features:
    - Log-parameterized uncertainty for numerical stability
    - Automatic task loss weighting based on uncertainty
    - Gradient-aware uncertainty optimization
    - Support for heterogeneous task loss scales
    - Per-task uncertainty tracking and reporting

Background:
    In multi-task learning, tasks often have different loss scales and learning
    dynamics. Manually tuning task weights is challenging. This operator uses
    the principle that tasks with higher uncertainty should receive lower
    weight in the combined loss.
    
    The uncertainty-weighted loss is:
    
    L_total = Σ_i (1/2σ_i²) * L_i + Σ_i log(σ_i)
    
    Where:
    - L_i: Task-specific loss
    - σ_i: Learned uncertainty for task i
    - The log term prevents σ_i from collapsing to zero

Reference:
    - Kendall et al. (2018). Multi-Task Learning Using Uncertainty to
      Weigh Losses for Scene Geometry and Semantics. CVPR 2018.
    - Chen et al. (2020). Gradient-Based Learning Rates for Multi-Task
      Training.

Dependencies:
    - torch >= 2.0.0
    - numpy (optional, for statistical computations)

Usage Examples:
    >>> from opss.train.multitask_uncertainty import (
    ...     POPSSMultiTaskConfig,
    ...     POPSSMultiTaskOperator
    ... )
    
    >>> config = POPSSMultiTaskConfig(
    ...     num_tasks=3,
    ...     initial_uncertainty=1.0,
    ...     learnable_uncertainty=True
    ... )
    
    >>> operator = POPSSMultiTaskOperator(config)
    >>> result = operator.execute({
    ...     "task_losses": [loss1, loss2, loss3]
    ... })

Integration:
    This operator integrates with PiscesLxTrainingEngine through the
    AdvancedTrainingCoordinator in opss/train/impl.py. It should be
    executed during loss computation, after individual task losses
    are computed but before backward pass.

See Also:
    - opss/train/regularization.py: Generic regularization operators
    - opss/train/gradient_optimization.py: Gradient processing
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import math

from configs.version import VERSION
from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig
)


class POPSSTaskType(Enum):
    """
    Classification of task types for specialized handling.
    
    Different task types may have different loss characteristics
    and uncertainty dynamics.
    
    Types:
        REGRESSION: Continuous value prediction tasks
        CLASSIFICATION: Discrete class prediction tasks
        SEGMENTATION: Pixel-wise prediction tasks
        DETECTION: Object detection tasks
        RANKING: Ranking/retrieval tasks
        GENERATION: Text/image generation tasks
        MULTIMODAL: Cross-modal understanding tasks
    """
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    RANKING = "ranking"
    GENERATION = "generation"
    MULTIMODAL = "multimodal"


@dataclass
class POPSSMultiTaskConfig(PiscesLxOperatorConfig):
    """
    Configuration for task uncertainty-based multi-task learning.
    
    This configuration controls how task uncertainties are learned and
    applied to balance multi-task training losses.
    
    Attributes:
        num_tasks: Number of tasks in the multi-task setup
        initial_uncertainty: Initial uncertainty value for all tasks
                            (default: 1.0, corresponding to σ = exp(0) = 1)
        learnable_uncertainty: Whether uncertainty is learnable (vs fixed)
        uncertainty_lr: Learning rate multiplier for uncertainty parameters
                       (relative to model learning rate)
        task_names: Optional list of task names for logging
        task_types: Optional list of task types for specialized handling
        reduction: Reduction method for task losses ('mean', 'sum', 'none')
        clamp_uncertainty: Whether to clamp uncertainty values
        uncertainty_min: Minimum uncertainty value (if clamping)
        uncertainty_max: Maximum uncertainty value (if clamping)
        separate_uncertainty: Use separate uncertainty per task (vs shared)
        uncertainty_init_strategy: Strategy for initial uncertainty
                                 ('uniform', 'empirical', 'task_specific')
        warmup_steps: Steps before uncertainty learning activates
        log_uncertainty_interval: Steps between uncertainty logging
        verbose_logging: Enable detailed uncertainty statistics logging
        
    Default Values:
        num_tasks: 1
        initial_uncertainty: 1.0
        learnable_uncertainty: True
        uncertainty_lr: 1.0
        warmup_steps: 0
        clamp_uncertainty: False
        
    Example:
        >>> config = POPSSMultiTaskConfig(
        ...     num_tasks=3,
        ...     initial_uncertainty=1.0,
        ...     learnable_uncertainty=True,
        ...     task_names=['detection', 'segmentation', 'depth']
        ... )
    """
    num_tasks: int = 1
    initial_uncertainty: float = 1.0
    learnable_uncertainty: bool = True
    uncertainty_lr: float = 1.0
    task_names: List[str] = field(default_factory=list)
    task_types: List[POPSSTaskType] = field(default_factory=list)
    reduction: str = "mean"
    clamp_uncertainty: bool = False
    uncertainty_min: float = 0.01
    uncertainty_max: float = 10.0
    separate_uncertainty: bool = True
    uncertainty_init_strategy: str = "uniform"
    warmup_steps: int = 0
    log_uncertainty_interval: int = 100
    verbose_logging: bool = False
    
    def __post_init__(self):
        self.name = "multitask_uncertainty"
        self.version = VERSION
        
        if len(self.task_names) != self.num_tasks:
            self.task_names = [f"task_{i}" for i in range(self.num_tasks)]
        
        if len(self.task_types) != self.num_tasks:
            self.task_types = [POPSSTaskType.REGRESSION] * self.num_tasks


class POPSSMultiTaskOperator(PiscesLxOperatorInterface):
    """
    Task uncertainty-based multi-task learning operator.
    
    This operator provides uncertainty-weighted multi-task loss computation,
    automatically balancing heterogeneous task losses without manual tuning.
    
    The operator implements:
    
    1. Uncertainty Parameter Management:
       - Learnable log(sigma) parameters per task
       - Optional shared vs separate uncertainty
       - Uncertainty learning rate scheduling
    
    2. Loss Combination:
       - Uncertainty-weighted task loss computation
       - Log-uncertainty regularization
       - Gradient-aware uncertainty updates
    
    3. Statistics and Logging:
       - Per-task uncertainty tracking
       - Loss component analysis
       - Training dynamics reporting
    
    Attributes:
        config: POPSSMultiTaskConfig instance
        log_uncertainty: nn.Parameter tensor for log(sigma) values
        step_count: Number of calls since initialization
        uncertainty_history: List of uncertainty values over time
        loss_history: List of combined loss values over time
        
    Thread Safety:
        This operator maintains per-instance state. Each training process
        should use its own operator instance.
        
    Memory Usage:
        O(num_tasks) for uncertainty parameters and statistics.
    """
    
    def __init__(self, config: Optional[POPSSMultiTaskConfig] = None):
        """
        Initialize the multi-task uncertainty operator.
        
        Args:
            config: Optional configuration instance. If None, default config
                   with standard multi-task parameters is used.
        """
        super().__init__(config)
        self.config = config or POPSSMultiTaskConfig()
        self.log_uncertainty: Optional[torch.nn.Parameter] = None
        self.step_count: int = 0
        self.uncertainty_history: List[torch.Tensor] = []
        self.loss_history: List[torch.Tensor] = []
        self._is_warmed_up: bool = False
        
        self._logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup operator logger."""
        return PiscesLxLogger("PiscesLx.Opss.Train",file_path=get_log_file("PiscesLx.Opss.Train"), enable_file=True)
    
    @property
    def name(self) -> str:
        """Get operator name identifier."""
        return "multitask_uncertainty"
    
    @property
    def version(self) -> str:
        """Get operator semantic version."""
        return VERSION
    
    @property
    def description(self) -> str:
        """Get operator description."""
        return (
            "Task uncertainty-based multi-task learning with automatic "
            "loss balancing and gradient-aware uncertainty optimization"
        )
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        Define expected input format for this operator.
        
        Returns:
            Dictionary describing required and optional input fields.
        """
        return {
            "type": "object",
            "required": ["task_losses"],
            "properties": {
                "task_losses": {
                    "type": "list",
                    "description": "List of task-specific loss tensors"
                },
                "task_weights": {
                    "type": "list",
                    "description": "Optional manual weights for each task"
                },
                "model": {
                    "type": "torch.nn.Module",
                    "description": "Model for uncertainty parameter registration"
                },
                "step": {
                    "type": "int",
                    "description": "Current training step"
                },
                "reduction": {
                    "type": "str",
                    "description": "Override reduction method"
                }
            }
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        """
        Define output format from this operator.
        
        Returns:
            Dictionary describing output fields and their types.
        """
        return {
            "type": "object",
            "properties": {
                "combined_loss": {
                    "type": "torch.Tensor",
                    "description": "Combined multi-task loss with uncertainty weighting"
                },
                "task_losses": {
                    "type": "list",
                    "description": "Individual task losses (unweighted)"
                },
                "weighted_losses": {
                    "type": "list",
                    "description": "Uncertainty-weighted task losses"
                },
                "uncertainties": {
                    "type": "list",
                    "description": "Current uncertainty values (sigma) per task"
                },
                "log_uncertainties": {
                    "type": "list",
                    "description": "Log-uncertainty parameters per task"
                },
                "uncertainty_regularization": {
                    "type": "float",
                    "description": "Log-uncertainty regularization term"
                },
                "total_loss_components": {
                    "type": "int",
                    "description": "Number of task losses combined"
                },
                "reduction_method": {
                    "type": "str",
                    "description": "Reduction method used"
                }
            }
        }
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate input parameters before execution.
        
        Args:
            inputs: Dictionary of input parameters from caller.
            
        Returns:
            True if all required inputs are valid, False otherwise.
        """
        if "task_losses" not in inputs:
            self._logger.error("Missing required input: task_losses")
            return False
        
        task_losses = inputs["task_losses"]
        if not isinstance(task_losses, (list, tuple)):
            self._logger.error(
                f"Invalid task_losses type: {type(task_losses)}. "
                "Expected list or tuple of loss tensors."
            )
            return False
        
        if len(task_losses) == 0:
            self._logger.error("task_losses is empty")
            return False
        
        if self.config.num_tasks != len(task_losses):
            self._logger.warning(
                f"Task count mismatch: config={self.config.num_tasks}, "
                f"provided={len(task_losses)}. Using provided count."
            )
        
        if "model" in inputs:
            model = inputs["model"]
            if not isinstance(model, nn.Module):
                self._logger.error(
                    f"Invalid model type: {type(model)}. "
                    "Expected torch.nn.Module instance."
                )
                return False
        
        self._logger.debug("Input validation passed")
        return True
    
    def _initialize_uncertainty(self, num_tasks: int, device: torch.device) -> None:
        """
        Initialize learnable log-uncertainty parameters.
        
        Args:
            num_tasks: Number of tasks for uncertainty parameters.
            device: Device to place parameters on.
        """
        initial_log_sigma = torch.log(torch.tensor(
            self.config.initial_uncertainty,
            device=device
        ))
        
        if self.config.separate_uncertainty:
            self.log_uncertainty = torch.nn.Parameter(
                initial_log_sigma.expand(num_tasks).clone().detach().requires_grad_(True)
            )
        else:
            self.log_uncertainty = torch.nn.Parameter(
                initial_log_sigma.clone().detach().requires_grad_(True)
            )
    
    def _compute_uncertainty_weighted_loss(
        self,
        task_losses: List[torch.Tensor],
        log_uncertainty: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Compute uncertainty-weighted combined loss.
        
        The combined loss is:
        L_total = Σ_i (1/2σ_i²) * L_i + Σ_i log(σ_i)
        
        Args:
            task_losses: List of task-specific losses.
            log_uncertainty: Log-uncertainty parameters.
            
        Returns:
            Tuple of (combined_loss, weighted_losses, log_uncertainty_reg).
        """
        if self.config.separate_uncertainty:
            uncertainty = torch.exp(log_uncertainty)
        else:
            uncertainty = torch.exp(log_uncertainty)
        
        uncertainty_sq = uncertainty ** 2
        
        weighted_losses = []
        for i, loss in enumerate(task_losses):
            if self.config.separate_uncertainty:
                task_uncertainty_sq = uncertainty_sq[i]
            else:
                task_uncertainty_sq = uncertainty_sq
            
            weighted_loss = 0.5 * loss / task_uncertainty_sq
            weighted_losses.append(weighted_loss)
        
        if self.config.reduction == "sum":
            combined_loss = sum(weighted_losses)
        elif self.config.reduction == "mean":
            combined_loss = torch.stack(weighted_losses).mean()
        else:
            combined_loss = torch.stack(weighted_losses).sum()
        
        log_uncertainty_reg = torch.sum(log_uncertainty)
        
        total_loss = combined_loss + log_uncertainty_reg
        
        return total_loss, weighted_losses, log_uncertainty_reg
    
    def _get_uncertainties(self) -> List[float]:
        """
        Get current uncertainty values (sigma) from log-uncertainty.
        
        Returns:
            List of sigma values.
        """
        if self.log_uncertainty is None:
            return [self.config.initial_uncertainty] * self.config.num_tasks
        
        if self.config.separate_uncertainty:
            return [float(torch.exp(uv).item()) for uv in self.log_uncertainty]
        else:
            return [float(torch.exp(self.log_uncertainty).item())] * self.config.num_tasks
    
    def execute(
        self,
        inputs: Dict[str, Any],
        **kwargs
    ) -> PiscesLxOperatorResult:
        """
        Execute uncertainty-weighted multi-task loss computation.
        
        This method combines task-specific losses using learned uncertainty
        weighting, enabling automatic balance of heterogeneous tasks.
        
        Args:
            inputs: Dictionary containing:
                - task_losses: List of task-specific loss tensors
                - task_weights: Optional manual weights per task
                - model: Optional model for uncertainty parameter registration
                - step: Current training step (optional)
                - reduction: Override reduction method (optional)
            **kwargs: Additional keyword arguments for flexibility.
            
        Returns:
            PiscesLxOperatorResult containing:
                - Combined loss tensor
                - Individual (weighted) task losses
                - Uncertainty statistics
                
        Raises:
            ValueError: If task_losses is invalid.
            RuntimeError: If loss computation fails unexpectedly.
        """
        start_time = time.time()
        
        try:
            if not self.validate_inputs(inputs):
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Input validation failed",
                    execution_time=time.time() - start_time
                )
            
            task_losses = inputs["task_losses"]
            task_weights = inputs.get("task_weights")
            model = inputs.get("model")
            current_step = inputs.get("step", self.step_count)
            reduction = inputs.get("reduction", self.config.reduction)
            
            self.step_count += 1
            
            num_tasks = len(task_losses)
            
            device = task_losses[0].device if isinstance(task_losses[0], torch.Tensor) else torch.device("cpu")
            
            if self.log_uncertainty is None:
                self._initialize_uncertainty(num_tasks, device)
            
            if task_weights is not None:
                weighted_losses = []
                for i, loss in enumerate(task_losses):
                    weighted_losses.append(loss * task_weights[i])
            else:
                weighted_losses = list(task_losses)
            
            combined_loss, weighted_task_losses, log_uncertainty_reg = (
                self._compute_uncertainty_weighted_loss(
                    weighted_losses,
                    self.log_uncertainty
                )
            )
            
            uncertainties = self._get_uncertainties()
            log_uncertainties = self.log_uncertainty.detach().cpu().tolist()
            
            if self.config.clamp_uncertainty:
                with torch.no_grad():
                    if self.config.separate_uncertainty:
                        self.log_uncertainty.data.clamp_(
                            min=math.log(self.config.uncertainty_min),
                            max=math.log(self.config.uncertainty_max)
                        )
                    else:
                        self.log_uncertainty.data.clamp_(
                            min=math.log(self.config.uncertainty_min),
                            max=math.log(self.config.uncertainty_max)
                        )
            
            self.uncertainty_history.append(torch.tensor(uncertainties))
            self.loss_history.append(combined_loss.detach())
            
            output = {
                "combined_loss": combined_loss,
                "task_losses": [l.detach() for l in task_losses],
                "weighted_losses": [l.detach() for l in weighted_task_losses],
                "uncertainties": uncertainties,
                "log_uncertainties": log_uncertainties,
                "uncertainty_regularization": log_uncertainty_reg.item(),
                "total_loss_components": num_tasks,
                "reduction_method": reduction,
                "num_tasks": num_tasks
            }
            
            if self.config.verbose_logging and (
                self.step_count % self.config.log_uncertainty_interval == 0
            ):
                uncertainty_str = ", ".join([
                    f"{name}: {u:.4f}" 
                    for name, u in zip(self.config.task_names, uncertainties)
                ])
                self._logger.info(
                    f"Step {self.step_count} - Uncertainties: [{uncertainty_str}]"
                )
            
            self._logger.debug(
                f"Multi-task loss computed: {combined_loss.item():.6f}, "
                f"tasks={num_tasks}"
            )
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=output,
                execution_time=time.time() - start_time,
                metadata={
                    "step": current_step,
                    "num_tasks": num_tasks,
                    "uncertainties": uncertainties
                }
            )
            
        except Exception as e:
            error_msg = f"Multi-task uncertainty execution failed: {str(e)}"
            self._logger.error(error_msg, exc_info=True)
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=error_msg,
                execution_time=time.time() - start_time,
                metadata={
                    "exception_type": type(e).__name__
                }
            )
    
    def get_uncertainty_parameters(self) -> torch.nn.Parameter:
        """
        Get the log-uncertainty parameter tensor.
        
        Returns:
            Log-uncertainty parameter tensor for gradient computation.
        """
        return self.log_uncertainty
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current multi-task learning statistics.
        
        Returns:
            Dictionary containing uncertainty history and loss statistics.
        """
        if self.uncertainty_history:
            uncertainty_stack = torch.stack(self.uncertainty_history)
            mean_uncertainty = uncertainty_stack.mean(dim=0).tolist()
            std_uncertainty = uncertainty_stack.std(dim=0).tolist()
        else:
            mean_uncertainty = self._get_uncertainties()
            std_uncertainty = [0.0] * len(mean_uncertainty)
        
        if self.loss_history:
            loss_stack = torch.stack(self.loss_history)
            mean_loss = loss_stack.mean().item()
            std_loss = loss_stack.std().item()
        else:
            mean_loss = 0.0
            std_loss = 0.0
        
        return {
            "step_count": self.step_count,
            "num_tasks": self.config.num_tasks,
            "current_uncertainties": self._get_uncertainties(),
            "mean_uncertainty": mean_uncertainty,
            "std_uncertainty": std_uncertainty,
            "mean_combined_loss": mean_loss,
            "std_combined_loss": std_loss,
            "is_learnable": self.config.learnable_uncertainty
        }
    
    def reset(self) -> None:
        """
        Reset operator state for a new training run.
        
        Clears history and reinitializes uncertainty parameters.
        """
        self.uncertainty_history.clear()
        self.loss_history.clear()
        self.step_count = 0
        self._is_warmed_up = False
        self.log_uncertainty = None
        self._logger.info("Multi-task uncertainty operator state reset")
    
    def freeze_uncertainty(self) -> None:
        """Freeze uncertainty parameters (stop learning)."""
        if self.log_uncertainty is not None:
            self.log_uncertainty.requires_grad_(False)
        self._logger.info("Uncertainty parameters frozen")
    
    def unfreeze_uncertainty(self) -> None:
        """Unfreeze uncertainty parameters (enable learning)."""
        if self.log_uncertainty is not None:
            self.log_uncertainty.requires_grad_(True)
        self._logger.info("Uncertainty parameters unfrozen")


class POPSSMultiTaskFacade:
    """
    Convenience facade for quick multi-task uncertainty setup.
    
    This facade provides a simplified interface for common use cases,
    automatically handling uncertainty initialization and loss combination.
    
    Usage:
        >>> facade = POPSSMultiTaskFacade(num_tasks=3)
        >>> combined_loss = facade.compute_loss([loss1, loss2, loss3])
    """
    
    def __init__(
        self,
        num_tasks: int = 2,
        initial_uncertainty: float = 1.0,
        task_names: Optional[List[str]] = None,
        reduction: str = "mean"
    ):
        """
        Initialize the facade with multi-task parameters.
        
        Args:
            num_tasks: Number of tasks.
            initial_uncertainty: Starting uncertainty value.
            task_names: Optional task names for logging.
            reduction: Loss reduction method.
        """
        config = POPSSMultiTaskConfig(
            num_tasks=num_tasks,
            initial_uncertainty=initial_uncertainty,
            task_names=task_names or [f"task_{i}" for i in range(num_tasks)],
            reduction=reduction
        )
        self.operator = POPSSMultiTaskOperator(config)
    
    def compute_loss(
        self,
        task_losses: List[torch.Tensor],
        model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Compute uncertainty-weighted combined loss.
        
        Args:
            task_losses: List of task-specific losses.
            model: Optional model for uncertainty parameter registration.
            
        Returns:
            Combined loss tensor.
        """
        result = self.operator.execute({
            "task_losses": task_losses,
            "model": model
        })
        
        if result.is_success():
            return result.output["combined_loss"]
        else:
            raise RuntimeError(f"Loss computation failed: {result.error}")
    
    def get_uncertainties(self) -> List[float]:
        """Get current uncertainty values."""
        return self.operator._get_uncertainties()


class POPSSTaskUncertaintyWeighting:
    def __init__(self, config: Optional[POPSSMultiTaskConfig] = None):
        self.config = config or POPSSMultiTaskConfig()
        self._operator = POPSSMultiTaskOperator(self.config)

    def compute(self, task_losses: List[torch.Tensor], model: Optional[nn.Module] = None) -> PiscesLxOperatorResult:
        return self._operator.execute({"task_losses": task_losses, "model": model})


__all__ = [
    "POPSSTaskType",
    "POPSSMultiTaskConfig",
    "POPSSMultiTaskOperator",
    "POPSSMultiTaskFacade",
    "POPSSTaskUncertaintyWeighting",
]
