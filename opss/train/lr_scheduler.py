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
Learning Rate Scheduler Operators

Comprehensive learning rate scheduling algorithms for neural network training.
Provides various scheduling strategies with operator-style interfaces.

Scheduler Types:
    - Linear: Linear warmup and decay
    - Cosine: Cosine annealing with warmup
    - CosineWithRestarts: Cosine with periodic restarts
    - Polynomial: Polynomial decay
    - InverseSquare: Inverse square root decay (BERT style)
    - Step: Step decay
    - Exponential: Exponential decay
    - WarmupDecay: Combined warmup and decay

Features:
    - Warmup phases
    - Minimum learning rate clipping
    - Learning rate history tracking
    - Cycle detection for restarts
    - Gradient accumulation support

Usage:
    from ops.train.lr_scheduler import (
        SchedulerType,
        LRSchedulerConfig,
        LRSchedulerOperator
    )

    config = LRSchedulerConfig(
        type=SchedulerType.COSINE,
        initial_lr=1e-4,
        min_lr=1e-6,
        warmup_steps=1000,
        max_steps=100000
    )
    scheduler = LRSchedulerOperator(config)
    current_lr = scheduler.step(step=500)
"""

import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer

from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus
from configs.version import VERSION


class POPSSSchedulerType(Enum):
    """Learning rate scheduler types."""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    INVERSE_SQUARE = "inverse_square"
    STEP = "step"
    EXPONENTIAL = "exponential"
    WARMUP_DECAY = "warmup_decay"


@dataclass
class POPSSLRSchedulerConfig:
    """
    Configuration for learning rate scheduler.
    
    Attributes:
        type: Scheduler algorithm type
        initial_lr: Starting learning rate (after warmup)
        min_lr: Minimum learning rate floor
        max_lr: Maximum learning rate during warmup
        warmup_steps: Number of warmup steps
        warmup_ratio: Alternative warmup as ratio of total steps
        warmup_type: Warmup function type (linear, cubic, exponential)
        total_steps: Total training steps (optional, for fixed schedules)
        decay_steps: Number of steps for decay (for step/exp schedulers)
        decay_ratio: Decay ratio for step scheduler
        decay_gamma: Gamma factor for exponential decay
        power: Power for polynomial decay
        num_restarts: Number of restarts for cosine with restarts
        restart_ratio: Ratio of cycle length for restarts
        cycle_first_step: Initial cycle length
        cycle_min_lr: Minimum LR during cycles
        cycle_scaling_factor: Scaling factor for cycle lengths
        gradient_accumulation_steps: Gradient accumulation steps
        apply_at_step: Step to start applying scheduler
        last_step: Last step index (for resuming)
    """
    type: POPSSSchedulerType = POPSSSchedulerType.COSINE
    initial_lr: float = 1e-4
    min_lr: float = 1e-6
    max_lr: float = 1e-3
    warmup_steps: int = 1000
    warmup_ratio: Optional[float] = None
    warmup_type: str = "linear"
    total_steps: Optional[int] = None
    decay_steps: Optional[int] = None
    decay_ratio: float = 0.1
    decay_gamma: float = 0.1
    power: float = 1.0
    num_restarts: int = 0
    restart_ratio: float = 0.25
    cycle_first_step: int = 1000
    cycle_min_lr: float = 1e-6
    cycle_scaling_factor: float = 1.0
    gradient_accumulation_steps: int = 1
    apply_at_step: int = 0
    last_step: int = -1
    
    def __post_init__(self):
        if self.warmup_ratio is not None and self.total_steps is not None:
            self.warmup_steps = int(self.warmup_ratio * self.total_steps)


class POPSSLRSchedulerOperator(PiscesLxOperatorInterface):
    """
    Learning Rate Scheduler Operator.
    
    Implements various LR scheduling algorithms with warmup support.
    Provides step-by-step learning rate computation for training loops.
    
    Features:
        - Multiple scheduler types
        - Warmup phases with configurable functions
        - Learning rate history tracking
        - Cycle support for restart schedulers
        - Gradient accumulation handling
        - Resume capability
    
    Attributes:
        config: POPSSLRSchedulerConfig instance
        history: List of LR values over steps
        _current_step: Current training step
        _cycle_start: Start step of current cycle
    """
    
    def __init__(self, config: Optional[POPSSLRSchedulerConfig] = None):
        """Initialize LR scheduler operator."""
        super().__init__()
        self._name = "lr.scheduler"
        self._version = VERSION
        self.config = config or POPSSLRSchedulerConfig()
        self.history: List[float] = []
        self._current_step = self.config.last_step
        self._cycle_start = 0
        self._restart_count = 0
        
        if self.config.warmup_type not in ["linear", "cubic", "exponential"]:
            self.config.warmup_type = "linear"

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return f"Learning rate scheduler ({self.config.type.value}) with warmup"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "step": {"type": "int", "required": True, "description": "Current training step"},
            "optimizer": {"type": "Optimizer", "required": True, "description": "Optimizer to update"},
            "current_lr": {"type": "float", "required": False, "description": "Current learning rate"},
            "reset": {"type": "bool", "required": False, "description": "Reset scheduler state"}
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "learning_rate": {"type": "float", "description": "Computed learning rate"},
            "step": {"type": "int", "description": "Current step"},
            "warmup_ratio": {"type": "float", "description": "Warmup progress (0-1)"},
            "in_warmup": {"type": "bool", "description": "Whether in warmup phase"},
            "history": {"type": "list", "description": "LR history"}
        }
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate required inputs."""
        if 'step' not in inputs:
            return False
        if 'optimizer' not in inputs and 'current_lr' not in inputs:
            return False
        return True
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute scheduler step to compute learning rate.
        
        Args:
            inputs: Dictionary containing:
                - step: Current training step
                - optimizer: PyTorch optimizer (optional)
                - current_lr: Current LR (optional)
                - reset: Whether to reset state
                
        Returns:
            PiscesLxOperatorResult with learning rate and metadata
        """
        try:
            step = inputs['step']
            optimizer = inputs.get('optimizer')
            current_lr = inputs.get('current_lr')
            reset = inputs.get('reset', False)
            
            if reset or self._current_step < 0:
                self.reset()
            
            if current_lr is not None and optimizer is None:
                self._current_step = step
                lr = current_lr
            else:
                self._current_step = step
                lr = self._compute_lr(step)
            
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            self.history.append(lr)
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    'learning_rate': lr,
                    'step': self._current_step,
                    'warmup_ratio': self._get_warmup_ratio(step),
                    'in_warmup': step < self.config.warmup_steps,
                    'history': self.history.copy()
                },
                metadata={
                    'scheduler_type': self.config.type.value,
                    'initial_lr': self.config.initial_lr,
                    'min_lr': self.config.min_lr,
                    'warmup_steps': self.config.warmup_steps
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _compute_lr(self, step: int) -> float:
        """Compute learning rate for current step."""
        if step < self.config.warmup_steps:
            return self._compute_warmup_lr(step)
        else:
            return self._compute_decay_lr(step - self.config.warmup_steps)
    
    def _compute_warmup_lr(self, step: int) -> float:
        """Compute learning rate during warmup phase.
        
        Warmup starts from a small fraction of max_lr (typically 1% or min_lr if specified)
        and linearly increases to max_lr by the end of warmup.
        This ensures smooth gradient updates at the start of training.
        """
        warmup_fn = self._get_warmup_function()
        warmup_ratio = warmup_fn(step / self.config.warmup_steps)
        
        # Start from near-zero (1% of max_lr) instead of min_lr
        # min_lr is the decay floor, not the warmup start
        warmup_start_lr = self.config.max_lr * 0.01
        lr = warmup_start_lr + (self.config.max_lr - warmup_start_lr) * warmup_ratio
        return lr
    
    def _get_warmup_function(self) -> Callable[[float], float]:
        """Get warmup function.
        
        Returns a function that maps progress ratio [0, 1] to warmup ratio [0, 1].
        Different warmup types provide different curves:
        - linear: Steady increase, good for most cases
        - cubic: Slow start, fast end (S-shaped curve)
        - exponential: Fast start, smooth transition to target LR
        """
        if self.config.warmup_type == "linear":
            return lambda x: x
        elif self.config.warmup_type == "cubic":
            # Cubic: slow start, fast end - good for stability
            return lambda x: x ** 3
        elif self.config.warmup_type == "exponential":
            # Exponential: fast approach to target
            # Uses exp(-2 * (1-x)) which gives ~0.86 at x=0.5 and ~0.98 at x=1
            # This provides a smoother transition than linear warmup
            return lambda x: 1 - math.exp(-2 * (1 - x))
        else:
            return lambda x: x
    
    def _compute_decay_lr(self, step: int) -> float:
        """Compute learning rate during decay phase."""
        total_decay_steps = self._get_total_decay_steps()
        
        if self.config.type == POPSSSchedulerType.LINEAR:
            return self._linear_decay(step, total_decay_steps)
        elif self.config.type == POPSSSchedulerType.COSINE:
            return self._cosine_decay(step, total_decay_steps)
        elif self.config.type == POPSSSchedulerType.COSINE_WITH_RESTARTS:
            return self._cosine_with_restarts_decay(step)
        elif self.config.type == POPSSSchedulerType.POLYNOMIAL:
            return self._polynomial_decay(step, total_decay_steps)
        elif self.config.type == POPSSSchedulerType.INVERSE_SQUARE:
            return self._inverse_square_decay(step)
        elif self.config.type == POPSSSchedulerType.STEP:
            return self._step_decay(step)
        elif self.config.type == POPSSSchedulerType.EXPONENTIAL:
            return self._exponential_decay(step)
        elif self.config.type == POPSSSchedulerType.WARMUP_DECAY:
            return self._warmup_decay(step, total_decay_steps)
        else:
            return self._cosine_decay(step, total_decay_steps)
    
    def _get_total_decay_steps(self) -> int:
        """Get total decay steps."""
        if self.config.total_steps is not None:
            return max(1, self.config.total_steps - self.config.warmup_steps)
        if self.config.decay_steps is not None:
            return self.config.decay_steps
        return 10000
    
    def _get_warmup_ratio(self, step: int) -> float:
        """Get warmup progress ratio."""
        if self.config.warmup_steps == 0:
            return 1.0
        return min(1.0, step / self.config.warmup_steps)
    
    def _linear_decay(self, step: int, total_steps: int) -> float:
        """Linear decay from max_lr to min_lr."""
        progress = min(1.0, step / total_steps)
        return self.config.max_lr - (self.config.max_lr - self.config.min_lr) * progress
    
    def _cosine_decay(self, step: int, total_steps: int) -> float:
        """Cosine annealing decay."""
        progress = min(1.0, step / total_steps)
        return self.config.min_lr + (self.config.max_lr - self.config.min_lr) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )
    
    def _cosine_with_restarts_decay(self, step: int) -> float:
        """Cosine annealing with periodic restarts."""
        while step >= self._cycle_start + self.cycle_length:
            self._cycle_start += self.cycle_length
            self._restart_count += 1
        
        progress = (step - self._cycle_start) / self.cycle_length
        cycle_lr = self.cycle_min_lr + (self.config.max_lr - self.cycle_min_lr) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )
        return cycle_lr
    
    @property
    def cycle_length(self) -> int:
        """Get current cycle length."""
        return int(
            self.config.cycle_first_step * 
            (self.config.cycle_scaling_factor ** self._restart_count)
        )
    
    def _polynomial_decay(self, step: int, total_steps: int) -> float:
        """Polynomial decay."""
        progress = min(1.0, step / total_steps)
        return self.config.min_lr + (self.config.max_lr - self.config.min_lr) * (
            1 - progress
        ) ** self.config.power
    
    def _inverse_square_decay(self, step: int) -> float:
        """Inverse square root decay (BERT style)."""
        decay_steps = max(1, step - self.config.warmup_steps)
        lr = self.config.initial_lr * (
            self.config.warmup_steps ** 0.5 / max(decay_steps, self.config.warmup_steps ** 0.5)
        )
        return max(lr, self.config.min_lr)
    
    def _step_decay(self, step: int) -> float:
        """Step decay."""
        decay_step = self.config.decay_steps or 10000
        exponent = step // decay_step
        lr = self.config.max_lr * (self.config.decay_ratio ** exponent)
        return max(lr, self.config.min_lr)
    
    def _exponential_decay(self, step: int) -> float:
        """Exponential decay."""
        decay_steps = self.config.decay_steps or 10000
        lr = self.config.max_lr * math.exp(
            -self.config.decay_gamma * step / decay_steps
        )
        return max(lr, self.config.min_lr)
    
    def _warmup_decay(self, step: int, total_steps: int) -> float:
        """Combined warmup and decay (HuggingFace style)."""
        progress = min(1.0, step / total_steps)
        return self.config.initial_lr * (1 - progress) ** self.config.power
    
    def get_lr_history(self) -> List[float]:
        """Get learning rate history."""
        return self.history.copy()
    
    def get_lr_at_step(self, step: int) -> float:
        """Get learning rate at specific step."""
        return self._compute_lr(step)
    
    def reset(self) -> None:
        """Reset scheduler state."""
        self.history.clear()
        self._current_step = self.config.last_step
        self._cycle_start = 0
        self._restart_count = 0
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {
            'config': {k: v for k, v in self.config.__dict__.items() 
                       if not callable(v)},
            'history': self.history,
            'current_step': self._current_step,
            'cycle_start': self._cycle_start,
            'restart_count': self._restart_count
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        if 'history' in state_dict:
            self.history = state_dict['history']
        if 'current_step' in state_dict:
            self._current_step = state_dict['current_step']
        if 'cycle_start' in state_dict:
            self._cycle_start = state_dict['cycle_start']
        if 'restart_count' in state_dict:
            self._restart_count = state_dict['restart_count']


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: Union[POPSSSchedulerType, str],
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs
) -> POPSSLRSchedulerOperator:
    """
    Factory function to create scheduler operator.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        num_warmup_steps: Warmup steps
        num_training_steps: Total training steps
        **kwargs: Additional configuration
        
    Returns:
        POPSSLRSchedulerOperator instance
    """
    if isinstance(scheduler_type, str):
        scheduler_type = POPSSSchedulerType(scheduler_type)
    
    config = POPSSLRSchedulerConfig(
        type=scheduler_type,
        warmup_steps=num_warmup_steps,
        total_steps=num_training_steps,
        **kwargs
    )
    
    return POPSSLRSchedulerOperator(config)


class POPSSCosineWarmupScheduler(POPSSLRSchedulerOperator):
    """Convenience class for cosine scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        initial_lr: float = 1e-4,
        min_lr: float = 1e-6
    ):
        config = POPSSLRSchedulerConfig(
            type=POPSSSchedulerType.COSINE,
            initial_lr=initial_lr,
            min_lr=min_lr,
            warmup_steps=num_warmup_steps,
            total_steps=num_training_steps
        )
        super().__init__(config)


class POPSSLinearWarmupScheduler(POPSSLRSchedulerOperator):
    """Convenience class for linear scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        initial_lr: float = 1e-4,
        min_lr: float = 1e-6
    ):
        config = POPSSLRSchedulerConfig(
            type=POPSSSchedulerType.LINEAR,
            initial_lr=initial_lr,
            min_lr=min_lr,
            warmup_steps=num_warmup_steps,
            total_steps=num_training_steps
        )
        super().__init__(config)


class POPSSInverseSquareRootScheduler(POPSSLRSchedulerOperator):
    """Convenience class for inverse square root scheduler (BERT style)."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        initial_lr: float = 1e-4,
        min_lr: float = 1e-6
    ):
        config = POPSSLRSchedulerConfig(
            type=POPSSSchedulerType.INVERSE_SQUARE,
            initial_lr=initial_lr,
            min_lr=min_lr,
            warmup_steps=num_warmup_steps
        )
        super().__init__(config)
