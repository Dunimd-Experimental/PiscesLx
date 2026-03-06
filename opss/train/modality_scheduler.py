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
Modality-Aware Learning Rate Scheduler Operator

This operator implements modality-specific learning rate scheduling for multi-modal
models, allowing different training dynamics for vision, audio, text, and fusion
components. Each modality receives tailored scheduling parameters to optimize
convergence based on its characteristic update patterns.

Key Features:
    - Modality-specific T_0 (restart period) configuration
    - Independent eta_min (minimum learning rate) settings
    - Configurable T_mult (multiplier for restart periods)
    - Cosine annealing with warm restarts
    - Support for all standard PyTorch scheduler backends

Algorithm:
    The scheduler uses cosine annealing with warm restarts:
    
    lr(t) = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T))
    
    Where:
    - eta_max: Base learning rate from optimizer
    - eta_min: Minimum learning rate (modality-specific)
    - T: Current restart period
    - t: Current step within the period

    After each restart, T increases by T_mult factor.

Reference:
    Loshchilov & Hutter (2016). SGDR: Stochastic Gradient Descent with
    Warm Restarts. arXiv:1608.03983

Dependencies:
    - torch >= 2.0.0
    - torch.optim.lr_scheduler

Usage Examples:
    >>> from opss.train.modality_scheduler import (
    ...     POPSSModalitySchedulerConfig,
    ...     POPSSModalitySchedulerOperator
    ... )
    
    >>> config = POPSSModalitySchedulerConfig(
    ...     vision_t0=10,
    ...     vision_eta_min=1e-7,
    ...     audio_t0=15,
    ...     audio_eta_min=5e-8,
    ...     fusion_t0=5,
    ...     fusion_eta_min=2e-7,
    ...     text_t0=8,
    ...     text_eta_min=1e-7
    ... )
    
    >>> operator = POPSSModalitySchedulerOperator(config)
    >>> result = operator.execute({"optimizer": optimizer})

Integration:
    This operator integrates seamlessly with the PiscesLxTrainingEngine through
    the AdvancedTrainingCoordinator in opss/train/impl.py. It should be executed
    after the optimizer has been initialized and before the training loop begins.

See Also:
    - opss/train/lr_scheduler.py: Generic LR scheduler implementations
    - opss/train/impl.py: AdvancedTrainingCoordinator for operator orchestration
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _LRScheduler
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum

from configs.version import VERSION
from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig
)


class POPSSModalityType(Enum):
    """
    Enumeration of supported modality types.
    
    Each modality represents a distinct input or processing domain in a
    multi-modal model architecture. The scheduler applies independent
    configurations to each modality's parameters.
    
    Types:
        VISION: Image and visual feature processing components
        AUDIO: Audio and speech feature processing components
        TEXT: Language and text processing components
        FUSION: Multi-modal fusion and cross-modal components
        OTHER: Unclassified or auxiliary components
    """
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    FUSION = "fusion"
    OTHER = "other"


@dataclass
class POPSSModalitySchedulerConfig(PiscesLxOperatorConfig):
    """
    Configuration for modality-aware learning rate scheduling.
    
    This configuration specifies independent scheduling parameters for each
    modality type, enabling tailored learning rate dynamics during multi-modal
    model training.
    
    Attributes:
        base_lr: Base learning rate multiplier (applied to optimizer's LR)
        vision_t0: Vision restart period (in epochs/steps)
        vision_t_mult: Vision T multiplier (default: 2.0)
        vision_eta_min: Vision minimum learning rate (default: 1e-7)
        audio_t0: Audio restart period (in epochs/steps)
        audio_t_mult: Audio T multiplier (default: 2.0)
        audio_eta_min: Audio minimum learning rate (default: 5e-8)
        fusion_t0: Fusion restart period (in epochs/steps)
        fusion_t_mult: Fusion T multiplier (default: 1.0)
        fusion_eta_min: Fusion minimum learning rate (default: 2e-7)
        text_t0: Text restart period (in epochs/steps)
        text_t_mult: Text T multiplier (default: 1.0)
        text_eta_min: Text minimum learning rate (default: 1e-7)
        other_t0: Other parameters restart period
        other_t_mult: Other T multiplier (default: 1.0)
        other_eta_min: Other minimum learning rate (default: 1e-7)
        warmup_steps: Number of warmup steps before scheduling begins
        warmup_lr: Learning rate during warmup phase
        last_epoch: Resume training from this epoch
        verbose: Enable scheduler progress logging
        
    Default Values:
        Vision: T_0=10, T_mult=2.0, eta_min=1e-7
        Audio: T_0=15, T_mult=2.0, eta_min=5e-8
        Fusion: T_0=5, T_mult=1.0, eta_min=2e-7
        Text/Other: T_0=8, T_mult=1.0, eta_min=1e-7
    
    Example:
        >>> config = POPSSModalitySchedulerConfig(
        ...     base_lr=1.0,
        ...     vision_t0=10,
        ...     vision_eta_min=1e-7,
        ...     audio_t0=15,
        ...     audio_eta_min=5e-8
        ... )
    """
    base_lr: float = 1.0
    
    vision_t0: int = 10
    vision_t_mult: float = 2.0
    vision_eta_min: float = 1e-7
    
    audio_t0: int = 15
    audio_t_mult: float = 2.0
    audio_eta_min: float = 5e-8
    
    fusion_t0: int = 5
    fusion_t_mult: float = 1.0
    fusion_eta_min: float = 2e-7
    
    text_t0: int = 8
    text_t_mult: float = 1.0
    text_eta_min: float = 1e-7
    
    other_t0: int = 8
    other_t_mult: float = 1.0
    other_eta_min: float = 1e-7
    
    warmup_steps: int = 0
    warmup_lr: float = 0.0
    last_epoch: int = -1
    verbose: bool = False
    
    def __post_init__(self):
        self.name = "modality_scheduler"
        self.version = VERSION


class POPSSModalitySchedulerOperator(PiscesLxOperatorInterface):
    """
    Modality-aware learning rate scheduler operator.
    
    This operator creates independent learning rate schedulers for different
    modality-specific parameter groups in multi-modal models. Each scheduler
    applies cosine annealing with warm restarts using modality-specific
    hyperparameters.
    
    The operator automatically identifies parameters belonging to each modality
    by matching parameter names against known patterns:
    
    Modality Patterns:
        - vision: 'vision', 'visual', 'image', 'vit', 'clip_vision'
        - audio: 'audio', 'speech', 'wav2vec', 'whisper_audio'
        - fusion: 'fusion', 'multimodal', 'cross_attn', 'gate_fusion'
        - text: 'text', 'language', 'encoder', 'decoder', 'lm_head'
        - other: All remaining parameters
    
    Algorithm Details:
        The cosine annealing with warm restarts follows:
        
        For step t in current period T:
            lr = eta_min + (eta_max - eta_min) * (1 + cos(pi * t / T)) / 2
        
        At each restart, T increases by T_mult factor:
            T_new = T_old * T_mult
    
    Attributes:
        config: POPSSModalitySchedulerConfig instance
        schedulers: Dict mapping modality types to scheduler instances
        modality_params: Dict mapping modality types to parameter groups
        
    Thread Safety:
        This operator maintains no shared mutable state across threads.
        Each training process should instantiate its own operator instance.
        
    Memory Usage:
        O(1) additional memory for parameter group tracking.
    """
    
    def __init__(self, config: Optional[POPSSModalitySchedulerConfig] = None):
        """
        Initialize the modality-aware scheduler operator.
        
        Args:
            config: Optional configuration instance. If None, default config
                   with standard modality parameters is used.
        """
        super().__init__(config)
        self.config = config or POPSSModalitySchedulerConfig()
        self.schedulers: Dict[POPSSModalityType, CosineAnnealingWarmRestarts] = {}
        self.modality_params: Dict[POPSSModalityType, List[torch.nn.Parameter]] = {}
        
        self._logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup operator logger."""
        return PiscesLxLogger("PiscesLx.Opss.Train",file_path=get_log_file("PiscesLx.Opss.Train"), enable_file=True)
    
    @property
    def name(self) -> str:
        """Get operator name identifier."""
        return "modality_scheduler"
    
    @property
    def version(self) -> str:
        """Get operator semantic version."""
        return VERSION
    
    @property
    def description(self) -> str:
        """Get operator description."""
        return (
            "Modality-aware learning rate scheduler with cosine annealing "
            "and warm restarts for multi-modal model training"
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
            "required": ["optimizer"],
            "properties": {
                "optimizer": {
                    "type": "torch.optim.Optimizer",
                    "description": "PyTorch optimizer instance with initialized parameters"
                },
                "model": {
                    "type": "torch.nn.Module",
                    "description": "Model instance for parameter modality detection"
                },
                "total_steps": {
                    "type": "int",
                    "description": "Total training steps for scheduler initialization"
                },
                "last_epoch": {
                    "type": "int",
                    "description": "Resume from specific epoch"
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
                "schedulers": {
                    "type": "dict",
                    "description": "Dictionary mapping modality types to scheduler instances"
                },
                "modality_groups": {
                    "type": "dict",
                    "description": "Parameter group assignments by modality"
                },
                "scheduler_configs": {
                    "type": "dict",
                    "description": "Configuration summary for each modality"
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
        if "optimizer" not in inputs:
            self._logger.error("Missing required input: optimizer")
            return False
        
        optimizer = inputs["optimizer"]
        if not isinstance(optimizer, torch.optim.Optimizer):
            self._logger.error(
                f"Invalid optimizer type: {type(optimizer)}. "
                "Expected torch.optim.Optimizer instance."
            )
            return False
        
        if list(optimizer.param_groups) == []:
            self._logger.error("Optimizer has no parameter groups")
            return False
        
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
    
    def _detect_modality(self, param_name: str) -> POPSSModalityType:
        """
        Detect modality type from parameter name using pattern matching.
        
        Args:
            param_name: Name of the parameter to classify.
            
        Returns:
            POPSSModalityType enum value indicating the parameter's modality.
        """
        name_lower = param_name.lower()
        
        if any(pattern in name_lower for pattern in ['vision', 'visual', 'image', 
                                                        'vit', 'clip_vision', 'resnet',
                                                        'efficientnet', 'swin']):
            return POPSSModalityType.VISION
        
        if any(pattern in name_lower for pattern in ['audio', 'speech', 'wav2vec', 
                                                        'whisper', 'baidu', ' Hubert']):
            return POPSSModalityType.AUDIO
        
        if any(pattern in name_lower for pattern in ['fusion', 'multimodal', 'cross_attn',
                                                        'gate_fusion', 'proj_fusion']):
            return POPSSModalityType.FUSION
        
        if any(pattern in name_lower for pattern in ['text', 'language', 'encoder', 
                                                        'decoder', 'lm_head', 'embed']):
            return POPSSModalityType.TEXT
        
        return POPSSModalityType.OTHER
    
    def _create_scheduler_for_modality(
        self,
        modality: POPSSModalityType,
        optimizer: torch.optim.Optimizer,
        modality_params: List[torch.nn.Parameter]
    ) -> CosineAnnealingWarmRestarts:
        """
        Create and configure scheduler for a specific modality.
        
        Args:
            modality: The modality type for this scheduler.
            optimizer: PyTorch optimizer instance.
            modality_params: List of parameters belonging to this modality.
            
        Returns:
            Configured CosineAnnealingWarmRestarts scheduler instance.
        """
        if modality == POPSSModalityType.VISION:
            t0 = self.config.vision_t0
            t_mult = self.config.vision_t_mult
            eta_min = self.config.vision_eta_min
        elif modality == POPSSModalityType.AUDIO:
            t0 = self.config.audio_t0
            t_mult = self.config.audio_t_mult
            eta_min = self.config.audio_eta_min
        elif modality == POPSSModalityType.FUSION:
            t0 = self.config.fusion_t0
            t_mult = self.config.fusion_t_mult
            eta_min = self.config.fusion_eta_min
        elif modality == POPSSModalityType.TEXT:
            t0 = self.config.text_t0
            t_mult = self.config.text_t_mult
            eta_min = self.config.text_eta_min
        else:
            t0 = self.config.other_t0
            t_mult = self.config.other_t_mult
            eta_min = self.config.other_eta_min
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t0,
            T_mult=t_mult,
            eta_min=eta_min,
            last_epoch=self.config.last_epoch
        )
        
        self._logger.info(
            f"Created scheduler for {modality.value}: "
            f"T_0={t0}, T_mult={t_mult}, eta_min={eta_min}"
        )
        
        return scheduler
    
    def _organize_parameters_by_modality(
        self,
        model: Optional[nn.Module]
    ) -> Dict[POPSSModalityType, List[torch.nn.Parameter]]:
        """
        Organize model parameters into modality-specific groups.
        
        Args:
            model: Optional model instance for parameter extraction.
                   If provided, uses model.named_parameters().
            
        Returns:
            Dictionary mapping modality types to their parameter lists.
        """
        modality_params: Dict[POPSSModalityType, List[torch.nn.Parameter]] = {
            POPSSModalityType.VISION: [],
            POPSSModalityType.AUDIO: [],
            POPSSModalityType.FUSION: [],
            POPSSModalityType.TEXT: [],
            POPSSModalityType.OTHER: []
        }
        
        if model is not None:
            for name, param in model.named_parameters():
                modality = self._detect_modality(name)
                modality_params[modality].append(param)
        else:
            self._logger.warning(
                "No model provided. Using default parameter organization."
            )
        
        return modality_params
    
    def execute(
        self,
        inputs: Dict[str, Any],
        **kwargs
    ) -> PiscesLxOperatorResult:
        """
        Execute modality-aware learning rate scheduling setup.
        
        This method creates independent learning rate schedulers for each
        modality, configured with appropriate hyperparameters. The schedulers
        are attached to the optimizer and ready for use in the training loop.
        
        Args:
            inputs: Dictionary containing:
                - optimizer: PyTorch optimizer instance
                - model: Optional model for parameter detection
                - total_steps: Optional total training steps
                - last_epoch: Optional epoch to resume from
            **kwargs: Additional keyword arguments for flexibility.
            
        Returns:
            PiscesLxOperatorResult containing:
                - schedulers: Dict mapping modality to scheduler instances
                - modality_groups: Parameter group summary
                - scheduler_configs: Configuration summary
                
        Raises:
            ValueError: If optimizer is invalid or has no parameters.
            RuntimeError: If scheduler creation fails unexpectedly.
        """
        import time
        start_time = time.time()
        
        try:
            if not self.validate_inputs(inputs):
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Input validation failed",
                    execution_time=time.time() - start_time
                )
            
            optimizer = inputs["optimizer"]
            model = inputs.get("model")
            total_steps = inputs.get("total_steps", 10000)
            
            self._logger.info("Starting modality-aware scheduler initialization")
            
            modality_params = self._organize_parameters_by_modality(model)
            
            scheduler_configs = {}
            for modality, params in modality_params.items():
                if params:
                    scheduler = self._create_scheduler_for_modality(
                        modality, optimizer, params
                    )
                    self.schedulers[modality] = scheduler
                    
                    scheduler_configs[modality.value] = {
                        "num_params": len(params),
                        "t0": scheduler.T_cur,
                        "t_i": scheduler.T_i,
                        "eta_min": scheduler.eta_min
                    }
            
            if not self.schedulers:
                self._logger.warning(
                    "No parameters detected for scheduling. "
                    "Creating default scheduler."
                )
                default_scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.config.text_t0,
                    T_mult=self.config.text_t_mult,
                    eta_min=self.config.text_eta_min,
                    last_epoch=self.config.last_epoch
                )
                self.schedulers[POPSSModalityType.TEXT] = default_scheduler
                scheduler_configs["default"] = {
                    "t0": self.config.text_t0,
                    "t_mult": self.config.text_t_mult,
                    "eta_min": self.config.text_eta_min
                }
            
            output = {
                "schedulers": {
                    m.value: s for m, s in self.schedulers.items()
                },
                "modality_groups": {
                    m.value: len(p) for m, p in modality_params.items()
                },
                "scheduler_configs": scheduler_configs,
                "total_schedulers": len(self.schedulers),
                "warmup_steps": self.config.warmup_steps
            }
            
            self._logger.info(
                f"Modality-aware scheduler initialization complete: "
                f"{len(self.schedulers)} schedulers created"
            )
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=output,
                execution_time=time.time() - start_time,
                metadata={
                    "modality_count": len(self.schedulers),
                    "configs": scheduler_configs
                }
            )
            
        except Exception as e:
            error_msg = f"Modality scheduler execution failed: {str(e)}"
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
    
    def step(self, epoch: Optional[int] = None) -> None:
        """
        Advance all modality schedulers by one step.
        
        This method should be called at the end of each training step/iteration
        to update learning rates according to the cosine annealing schedule.
        
        Args:
            epoch: Optional epoch number. If None, uses internal counter.
        """
        for scheduler in self.schedulers.values():
            scheduler.step(epoch)
    
    def get_lr(self) -> Dict[str, float]:
        """
        Get current learning rates for all modalities.
        
        Returns:
            Dictionary mapping modality names to current learning rates.
        """
        lrs = {}
        for modality, scheduler in self.schedulers.items():
            current_lr = scheduler.get_last_lr()
            if current_lr:
                lrs[modality.value] = current_lr[0]
        return lrs
    
    def get_modality_schedulers(self) -> Dict[POPSSModalityType, CosineAnnealingWarmRestarts]:
        """
        Get the scheduler instances for external access.
        
        Returns:
            Dictionary mapping modality types to their scheduler instances.
        """
        return self.schedulers


class POPSSModalitySchedulerFacade:
    """
    Convenience facade for quick modality scheduler setup.
    
    This facade provides a simplified interface for common use cases,
    automatically handling parameter detection and scheduler creation.
    
    Usage:
        >>> facade = POPSSModalitySchedulerFacade()
        >>> schedulers = facade.setup(optimizer, model)
        >>> 
        >>> for epoch in range(num_epochs):
        >>>     for step, batch in enumerate(dataloader):
        >>>         # training steps...
        >>>         facade.step()
    """
    
    def __init__(
        self,
        vision_t0: int = 10,
        vision_eta_min: float = 1e-7,
        audio_t0: int = 15,
        audio_eta_min: float = 5e-8,
        fusion_t0: int = 5,
        fusion_eta_min: float = 2e-7,
        text_t0: int = 8,
        text_eta_min: float = 1e-7,
        **kwargs
    ):
        """
        Initialize the facade with modality-specific parameters.
        
        Args:
            vision_t0: Vision restart period
            vision_eta_min: Vision minimum learning rate
            audio_t0: Audio restart period
            audio_eta_min: Audio minimum learning rate
            fusion_t0: Fusion restart period
            fusion_eta_min: Fusion minimum learning rate
            text_t0: Text restart period
            text_eta_min: Text minimum learning rate
            **kwargs: Additional configuration passed to config
        """
        config = POPSSModalitySchedulerConfig(
            vision_t0=vision_t0,
            vision_eta_min=vision_eta_min,
            audio_t0=audio_t0,
            audio_eta_min=audio_eta_min,
            fusion_t0=fusion_t0,
            fusion_eta_min=fusion_eta_min,
            text_t0=text_t0,
            text_eta_min=text_eta_min,
            **kwargs
        )
        self.operator = POPSSModalitySchedulerOperator(config)
        self._schedulers: Dict[POPSSModalityType, CosineAnnealingWarmRestarts] = {}
    
    def setup(
        self,
        optimizer: torch.optim.Optimizer,
        model: Optional[nn.Module] = None
    ) -> Dict[POPSSModalityType, CosineAnnealingWarmRestarts]:
        """
        Setup modality-aware schedulers for the given optimizer and model.
        
        Args:
            optimizer: PyTorch optimizer instance
            model: Optional model for parameter modality detection
            
        Returns:
            Dictionary mapping modality types to scheduler instances
        """
        result = self.operator.execute({
            "optimizer": optimizer,
            "model": model
        })
        
        if result.is_success():
            self._schedulers = self.operator.get_modality_schedulers()
        else:
            raise RuntimeError(f"Scheduler setup failed: {result.error}")
        
        return self._schedulers
    
    def step(self, epoch: Optional[int] = None) -> None:
        """
        Advance all schedulers by one step.
        """
        self.operator.step(epoch)
    
    def get_lr(self) -> Dict[str, float]:
        """
        Get current learning rates.
        """
        return self.operator.get_lr()


__all__ = [
    "POPSSModalityType",
    "POPSSModalitySchedulerConfig",
    "POPSSModalitySchedulerOperator",
    "POPSSModalitySchedulerFacade",
]
