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
MoE Expert Gradient Optimization Operator

This operator implements specialized gradient optimization for Mixture of Experts (MoE)
models, addressing the unique challenges of training sparse gated models with
multiple expert networks. The key challenge is handling the 10^4 scale difference
between expert gradients and other model parameters, which can cause training
instability if not properly managed.

Key Features:
    - Automatic expert parameter detection via name patterns
    - Three-order gradient scale handling for expert layers
    - Adaptive expert gradient clipping threshold
    - Expert gradient normalization for stable training
    - MoE-specific loss term support

Background:
    MoE models typically consist of:
    1. A gating network that routes inputs to experts
    2. Multiple expert networks (often FFN-based)
    3. Load balancing loss to ensure expert utilization
    
    During training, expert gradients can vary dramatically in scale due to:
    - Varying expert utilization rates
    - Different expert capacities and output magnitudes
    - Sparse activation patterns (only 1-4 experts active per token)
    
    This operator normalizes these gradient differences to enable stable training.

Algorithm:
    1. Expert Detection: Identify parameters belonging to MoE experts
       via naming patterns: 'experts', 'expert', 'moe', 'ffn'
    
    2. Gradient Analysis: Compute gradient norms for all parameters
       and identify expert-specific scale characteristics
    
    3. Adaptive Clipping: Apply modality-specific clipping thresholds
       with smaller values for expert layers to handle 10^4 scale gap
    
    4. Gradient Normalization: Optionally normalize expert gradients
       to match the distribution of non-expert gradients
    
    5. Expert Loss Addition: Compute and return MoE-specific auxiliary losses
       (load balancing, importance, etc.)

Reference:
    - Shazeer et al. (2017). Outrageously Large Neural Networks.
      The Sparsely-Gated Mixture-of-Experts Layer.
    - Fedus et al. (2022). Switch Transformers.
    - Dai et al. (2024). DeepSeekMoE.

Dependencies:
    - torch >= 2.0.0
    - numpy (optional, for statistical computations)

Usage Examples:
    >>> from opss.train.moe_gradient import (
    ...     POPSSMoEGradientConfig,
    ...     POPSSMoEGradientOperator
    ... )
    
    >>> config = POPSSMoEGradientConfig(
    ...     expert_clip_threshold=0.1,
    ...     expert_norm_factor=1e-4,
    ...     enable_adaptive_clip=True,
    ...     enable_gradient_normalization=True
    ... )
    
    >>> operator = POPSSMoEGradientOperator(config)
    >>> result = operator.execute({
    ...     "model": model,
    ...     "gradients": gradients_dict
    ... })

Integration:
    This operator integrates with PiscesLxTrainingEngine through the
    AdvancedTrainingCoordinator in opss/train/impl.py. It should be
    executed after backward pass and before optimizer.step().

See Also:
    - opss/train/gradient_optimization.py: Generic gradient optimization
    - opss/train/kfac.py: K-FAC preconditioning for gradients
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import time

from configs.version import VERSION
from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig
)


class POPSSMoEExpertType(Enum):
    """
    Classification of MoE expert parameter types.
    
    Different expert architectures may require different gradient
    handling strategies. This enumeration helps categorize experts
    for specialized processing.
    
    Types:
        FFN: Feed-forward network based experts (most common)
        ATTENTION: Attention-based expert heads
        EMBEDDING: Expert layers in embedding space
        CROSS_MODAL: Experts for cross-modal processing
        UNIFIED: General purpose unified experts
    """
    FFN = "ffn"
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    CROSS_MODAL = "cross_modal"
    UNIFIED = "unified"


@dataclass
class POPSSMoEGradientConfig(PiscesLxOperatorConfig):
    """
    Configuration for MoE expert gradient optimization.
    
    This configuration controls how expert gradients are processed during
    training, addressing the scale differences and stability challenges
    specific to MoE architectures.
    
    Attributes:
        expert_clip_threshold: Base clipping threshold for expert gradients
                              (default: 0.1, handles 10^4 scale gap)
        expert_norm_factor: Normalization factor for expert gradients
                          (default: 1e-4, scales to match non-expert gradients)
        enable_adaptive_clip: Enable adaptive threshold adjustment based on
                             observed gradient statistics
        enable_gradient_normalization: Normalize expert gradients to match
                                     the statistical distribution of non-expert
        scale_gap_override: Override detected scale gap (None = auto-detect)
        load_balancing_weight: Weight for load balancing auxiliary loss
        importance_weight: Weight for expert importance regularization
        utilization_weight: Weight for expert utilization regularization
        detect_expert_patterns: Patterns for expert parameter detection
        exclude_patterns: Parameter name patterns to exclude from MoE processing
        verbose_logging: Enable detailed gradient statistics logging
        warmup_steps: Steps before MoE gradient optimization activates
        clip_history_window: Window size for adaptive clip threshold history
        clip_percentile: Percentile for adaptive threshold calculation
        
    Default Values:
        expert_clip_threshold: 0.1
        expert_norm_factor: 1e-4
        enable_adaptive_clip: True
        enable_gradient_normalization: True
        scale_gap_override: None (auto-detect)
        load_balancing_weight: 0.01
        importance_weight: 0.01
        utilization_weight: 0.01
        warmup_steps: 100
        
    Example:
        >>> config = POPSSMoEGradientConfig(
        ...     expert_clip_threshold=0.1,
        ...     enable_adaptive_clip=True,
        ...     load_balancing_weight=0.01
        ... )
    """
    expert_clip_threshold: float = 0.1
    expert_norm_factor: float = 1e-4
    enable_adaptive_clip: bool = True
    enable_gradient_normalization: bool = True
    scale_gap_override: Optional[float] = None
    load_balancing_weight: float = 0.01
    importance_weight: float = 0.01
    utilization_weight: float = 0.01
    detect_expert_patterns: List[str] = field(default_factory=lambda: [
        'experts', 'expert', 'moe', 'ffn', 'feed_forward',
        'w1', 'w2', 'w3', 'gate_proj', 'up_proj', 'down_proj'
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        'norm', 'layernorm', 'batchnorm', 'embed', 'head'
    ])
    verbose_logging: bool = False
    warmup_steps: int = 100
    clip_history_window: int = 100
    clip_percentile: float = 95.0
    
    def __post_init__(self):
        self.name = "moe_gradient_optimizer"
        self.version = VERSION


class POPSSMoEGradientOperator(PiscesLxOperatorInterface):
    """
    MoE expert gradient optimization operator.
    
    This operator provides specialized gradient processing for MoE model
    architectures, addressing the scale differences and training stability
    challenges specific to mixture-of-experts models.
    
    The operator implements a comprehensive approach to MoE gradient handling:
    
    1. Parameter Classification:
       - Identifies expert vs non-expert parameters
       - Classifies expert types (FFN, Attention, etc.)
       - Detects gating network parameters
    
    2. Gradient Analysis:
       - Computes gradient norms for all parameters
       - Estimates scale gap between expert/non-expert gradients
       - Tracks gradient statistics over time
    
    3. Adaptive Processing:
       - Applies expert-specific clipping thresholds
       - Optionally normalizes expert gradients
       - Adapts thresholds based on historical statistics
    
    4. Loss Terms:
       - Computes load balancing auxiliary loss
       - Calculates expert importance scores
       - Tracks expert utilization rates
    
    Attributes:
        config: POPSSMoEGradientConfig instance
        clip_history: List of recent gradient norms for adaptive clipping
        expert_params: List of identified expert parameter names
        non_expert_params: List of non-expert parameter names
        gradient_stats: Dictionary tracking gradient statistics
        
    Thread Safety:
        This operator maintains per-instance state. Each training process
        should use its own operator instance to avoid race conditions.
        
    Memory Usage:
        O(num_parameters) for parameter classification tracking.
        O(clip_history_window) for adaptive clipping history.
    """
    
    def __init__(self, config: Optional[POPSSMoEGradientConfig] = None):
        """
        Initialize the MoE gradient optimization operator.
        
        Args:
            config: Optional configuration instance. If None, default config
                   with standard MoE parameters is used.
        """
        super().__init__(config)
        self.config = config or POPSSMoEGradientConfig()
        self.expert_params: List[str] = []
        self.non_expert_params: List[str] = []
        self.gating_params: List[str] = []
        self.clip_history: List[float] = []
        self.gradient_stats: Dict[str, Any] = {}
        self.step_count: int = 0
        self._is_warmed_up: bool = False
        
        self._logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup operator logger."""
        return PiscesLxLogger("PiscesLx.Opss.Train",file_path=get_log_file("PiscesLx.Opss.Train"), enable_file=True)
    
    @property
    def name(self) -> str:
        """Get operator name identifier."""
        return "moe_gradient_optimizer"
    
    @property
    def version(self) -> str:
        """Get operator semantic version."""
        return VERSION
    
    @property
    def description(self) -> str:
        """Get operator description."""
        return (
            "MoE expert gradient optimization with three-order scale handling, "
            "adaptive clipping, and load balancing loss computation"
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
            "required": ["model"],
            "properties": {
                "model": {
                    "type": "torch.nn.Module",
                    "description": "Model containing MoE layers for gradient optimization"
                },
                "gradients": {
                    "type": "dict",
                    "description": "Optional pre-computed gradient dict from model.named_parameters()"
                },
                "gating_outputs": {
                    "type": "dict",
                    "description": "Optional gating network outputs for loss computation"
                },
                "expert_selection": {
                    "type": "tensor",
                    "description": "Optional expert selection indices for load balancing"
                },
                "step": {
                    "type": "int",
                    "description": "Current training step (for warmup handling)"
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
                "clip_applied": {
                    "type": "bool",
                    "description": "Whether gradient clipping was applied"
                },
                "normalization_applied": {
                    "type": "bool",
                    "description": "Whether gradient normalization was applied"
                },
                "expert_gradient_norm": {
                    "type": "float",
                    "description": "Total expert gradient norm before processing"
                },
                "non_expert_gradient_norm": {
                    "type": "float",
                    "description": "Non-expert gradient norm for comparison"
                },
                "scale_gap": {
                    "type": "float",
                    "description": "Detected or estimated scale gap"
                },
                "load_balancing_loss": {
                    "type": "float",
                    "description": "Load balancing auxiliary loss value"
                },
                "importance_loss": {
                    "type": "float",
                    "description": "Expert importance regularization loss"
                },
                "clipping_ratio": {
                    "type": "float",
                    "description": "Ratio of clipped gradients if applicable"
                },
                "num_expert_params": {
                    "type": "int",
                    "description": "Number of identified expert parameters"
                },
                "expert_type_counts": {
                    "type": "dict",
                    "description": "Count of parameters per expert type"
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
        if "model" not in inputs:
            self._logger.error("Missing required input: model")
            return False
        
        model = inputs["model"]
        if not isinstance(model, nn.Module):
            self._logger.error(
                f"Invalid model type: {type(model)}. "
                "Expected torch.nn.Module instance."
            )
            return False
        
        if list(model.parameters()) == []:
            self._logger.error("Model has no parameters")
            return False
        
        self._logger.debug("Input validation passed")
        return True
    
    def _detect_expert_parameter(
        self,
        param_name: str
    ) -> Tuple[bool, POPSSMoEExpertType]:
        """
        Detect if a parameter is an MoE expert parameter.
        
        Args:
            param_name: Name of the parameter to check.
            
        Returns:
            Tuple of (is_expert: bool, expert_type: POPSSMoEExpertType).
        """
        name_lower = param_name.lower()
        
        for pattern in self.config.exclude_patterns:
            if pattern in name_lower:
                return False, POPSSMoEExpertType.UNIFIED
        
        if any(pattern in name_lower for pattern in ['gate', 'routing', 'router']):
            self.gating_params.append(param_name)
            return False, POPSSMoEExpertType.UNIFIED
        
        expert_patterns = self.config.detect_expert_patterns
        if any(pattern in name_lower for pattern in expert_patterns):
            if any(p in name_lower for p in ['w1', 'gate_proj', 'up_proj', 'expert_w1']):
                return True, POPSSMoEExpertType.FFN
            elif any(p in name_lower for p in ['w2', 'down_proj', 'expert_w2']):
                return True, POPSSMoEExpertType.FFN
            elif any(p in name_lower for p in ['w3', 'expert_w3']):
                return True, POPSSMoEExpertType.FFN
            elif any(p in name_lower for p in ['attn', 'attention', 'head']):
                return True, POPSSMoEExpertType.ATTENTION
            elif any(p in name_lower for p in ['embed', 'embedding']):
                return True, POPSSMoEExpertType.EMBEDDING
            elif any(p in name_lower for p in ['cross', 'multimodal']):
                return True, POPSSMoEExpertType.CROSS_MODAL
            else:
                return True, POPSSMoEExpertType.FFN
        
        return False, POPSSMoEExpertType.UNIFIED
    
    def _classify_parameters(self, model: nn.Module) -> Dict[str, List[str]]:
        """
        Classify model parameters into expert and non-expert categories.
        
        Args:
            model: The model to analyze.
            
        Returns:
            Dictionary with 'expert' and 'non_expert' parameter name lists.
        """
        classification = {
            'expert': {'ffn': [], 'attention': [], 'embedding': [], 
                      'cross_modal': [], 'unified': []},
            'non_expert': []
        }
        
        for name, param in model.named_parameters():
            is_expert, expert_type = self._detect_expert_parameter(name)
            
            if is_expert:
                self.expert_params.append(name)
                classification['expert'][expert_type.value].append(name)
            else:
                self.non_expert_params.append(name)
                classification['non_expert'].append(name)
        
        return classification
    
    def _compute_gradient_norm(
        self,
        parameters: List[torch.nn.Parameter]
    ) -> float:
        """
        Compute the total L2 norm of gradients for given parameters.
        
        Args:
            parameters: List of parameters whose gradients to analyze.
            
        Returns:
            Total L2 norm of gradients, or 0.0 if no gradients.
        """
        total_norm = 0.0
        for param in parameters:
            if param.grad is not None:
                grad = param.grad.data
                total_norm += grad.norm(2).item() ** 2
        
        return total_norm ** 0.5
    
    def _get_parameters_by_name(
        self,
        model: nn.Module,
        param_names: List[str]
    ) -> List[torch.nn.Parameter]:
        """
        Retrieve parameter objects by their names.
        
        Args:
            model: Model containing the parameters.
            param_names: Names of parameters to retrieve.
            
        Returns:
            List of parameter objects.
        """
        param_dict = {n: p for n, p in model.named_parameters()}
        return [param_dict[n] for n in param_names if n in param_dict]
    
    def _clip_expert_gradients(
        self,
        model: nn.Module,
        scale_gap: float
    ) -> Tuple[float, int]:
        """
        Apply gradient clipping to expert parameters.
        
        Handles the three-order scale difference (10^4) between expert
        and non-expert gradients by applying adaptive thresholds.
        
        Args:
            model: Model containing expert parameters.
            scale_gap: Detected or estimated scale gap between gradients.
            
        Returns:
            Tuple of (expert_gradient_norm, num_clipped).
        """
        clip_threshold = self.config.expert_clip_threshold
        
        if self.config.enable_adaptive_clip and self.clip_history:
            import numpy as np
            try:
                history_array = np.array(self.clip_history)
                if len(history_array) >= 10:
                    target_percentile = np.percentile(
                        history_array, 
                        self.config.clip_percentile
                    )
                    clip_threshold = min(
                        clip_threshold,
                        target_percentile * 1.2
                    )
            except (ImportError, ValueError):
                pass
        
        total_norm = 0.0
        num_clipped = 0
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            if name in self.expert_params:
                grad = param.grad.data
                grad_norm = grad.norm(2).item()
                total_norm += grad_norm ** 2
                
                if grad_norm > clip_threshold:
                    clip_coef = clip_threshold / (grad_norm + 1e-8)
                    param.grad.data.mul_(clip_coef)
                    num_clipped += 1
        
        return total_norm ** 0.5, num_clipped
    
    def _normalize_expert_gradients(
        self,
        model: nn.Module,
        non_expert_norm: float
    ) -> float:
        """
        Normalize expert gradients to match non-expert distribution.
        
        Args:
            model: Model containing expert parameters.
            non_expert_norm: Reference norm from non-expert parameters.
            
        Returns:
            Normalized expert gradient norm.
        """
        expert_norm = self._compute_gradient_norm(
            self._get_parameters_by_name(model, self.expert_params)
        )
        
        if expert_norm > 0 and non_expert_norm > 0:
            scale_factor = (
                self.config.expert_norm_factor * 
                (non_expert_norm / expert_norm)
            )
            
            for name, param in model.named_parameters():
                if param.grad is not None and name in self.expert_params:
                    param.grad.data.mul_(scale_factor)
        
        return self._compute_gradient_norm(
            self._get_parameters_by_name(model, self.expert_params)
        )
    
    def _compute_load_balancing_loss(
        self,
        gating_outputs: Optional[Dict[str, torch.Tensor]] = None,
        expert_selection: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute load balancing auxiliary loss for MoE training.
        
        This loss encourages equal utilization of all experts, preventing
        the model from relying on a small subset of experts.
        
        Args:
            gating_outputs: Optional dict containing gating network outputs.
            expert_selection: Optional tensor of expert indices for each token.
            
        Returns:
            Load balancing loss value.
        """
        if gating_outputs is None and expert_selection is None:
            return 0.0
        
        try:
            if expert_selection is not None:
                selection_probs = torch.zeros(
                    expert_selection.max() + 1,
                    device=expert_selection.device
                )
                for idx in range(expert_selection.shape[0]):
                    selected = expert_selection[idx]
                    selection_probs[selected] += 1.0
                
                selection_probs = selection_probs / selection_probs.sum()
                uniform_prob = 1.0 / len(selection_probs)
                load_loss = torch.sum(
                    (selection_probs - uniform_prob) ** 2
                ) / len(selection_probs)
                
                return load_loss.item() * self.config.load_balancing_weight
            
            if gating_outputs is not None:
                if 'routing_weights' in gating_outputs:
                    routing_weights = gating_outputs['routing_weights']
                    num_experts = routing_weights.shape[-1]
                    
                    expert_usage = routing_weights.mean(dim=0)
                    uniform_usage = torch.ones_like(expert_usage) / num_experts
                    
                    load_loss = torch.sum(
                        (expert_usage - uniform_usage) ** 2
                    ) / num_experts
                    
                    return load_loss.item() * self.config.load_balancing_weight
            
        except Exception as e:
            self._logger.warning(
                f"Load balancing loss computation failed: {e}"
            )
        
        return 0.0
    
    def _compute_importance_loss(
        self,
        gating_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> float:
        """
        Compute expert importance regularization loss.
        
        This loss encourages experts to have balanced and meaningful
        contributions to the model outputs.
        
        Args:
            gating_outputs: Optional dict containing gating network outputs.
            
        Returns:
            Importance regularization loss value.
        """
        if gating_outputs is None:
            return 0.0
        
        try:
            if 'routing_weights' in gating_outputs:
                routing_weights = gating_outputs['routing_weights']
                expert_importance = routing_weights.sum(dim=0)
                importance_variance = torch.var(expert_importance)
                
                return importance_variance.item() * self.config.importance_weight
            
        except Exception as e:
            self._logger.warning(
                f"Importance loss computation failed: {e}"
            )
        
        return 0.0
    
    def execute(
        self,
        inputs: Dict[str, Any],
        **kwargs
    ) -> PiscesLxOperatorResult:
        """
        Execute MoE gradient optimization.
        
        This method performs comprehensive gradient processing for MoE models,
        including expert detection, gradient analysis, adaptive clipping,
        normalization, and auxiliary loss computation.
        
        Args:
            inputs: Dictionary containing:
                - model: PyTorch model with MoE layers
                - gradients: Optional pre-computed gradient dict
                - gating_outputs: Optional gating network outputs
                - expert_selection: Optional expert selection indices
                - step: Optional current training step
            **kwargs: Additional keyword arguments for flexibility.
            
        Returns:
            PiscesLxOperatorResult containing:
                - Processing results and statistics
                - Auxiliary loss values
                - Gradient norm comparisons
                
        Raises:
            ValueError: If model is invalid or has no parameters.
            RuntimeError: If gradient processing fails unexpectedly.
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
            
            model = inputs["model"]
            gating_outputs = inputs.get("gating_outputs")
            expert_selection = inputs.get("expert_selection")
            current_step = inputs.get("step", self.step_count)
            
            self.step_count += 1
            
            if current_step < self.config.warmup_steps:
                self._logger.debug(
                    f"Warmup: skipping MoE optimization at step {current_step}"
                )
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.SUCCESS,
                    output={
                        "clip_applied": False,
                        "normalization_applied": False,
                        "expert_gradient_norm": 0.0,
                        "non_expert_gradient_norm": 0.0,
                        "scale_gap": 1.0,
                        "load_balancing_loss": 0.0,
                        "importance_loss": 0.0,
                        "clipping_ratio": 0.0,
                        "num_expert_params": len(self.expert_params),
                        "expert_type_counts": {},
                        "message": "Warmup phase - no processing applied"
                    },
                    execution_time=time.time() - start_time
                )
            
            classification = self._classify_parameters(model)
            
            expert_params = self._get_parameters_by_name(
                model, self.expert_params
            )
            non_expert_params = self._get_parameters_by_name(
                model, self.non_expert_params
            )
            
            expert_norm = self._compute_gradient_norm(expert_params)
            non_expert_norm = self._compute_gradient_norm(non_expert_params)
            
            if non_expert_norm > 0 and expert_norm > 0:
                scale_gap = expert_norm / non_expert_norm
            else:
                scale_gap = self.config.scale_gap_override or 1.0
            
            self.gradient_stats = {
                'expert_norm': expert_norm,
                'non_expert_norm': non_expert_norm,
                'scale_gap': scale_gap,
                'num_expert_params': len(self.expert_params),
                'num_non_expert_params': len(self.non_expert_params)
            }
            
            clip_applied = False
            normalization_applied = False
            num_clipped = 0
            
            if self.expert_params and expert_norm > 0:
                if self.config.enable_adaptive_clip or self.config.expert_clip_threshold > 0:
                    clip_applied = True
                    expert_norm, num_clipped = self._clip_expert_gradients(
                        model, scale_gap
                    )
                    self.clip_history.append(expert_norm)
                    
                    if len(self.clip_history) > self.config.clip_history_window:
                        self.clip_history = self.clip_history[-self.config.clip_history_window:]
                
                if self.config.enable_gradient_normalization:
                    normalization_applied = True
                    expert_norm = self._normalize_expert_gradients(
                        model, non_expert_norm
                    )
            
            load_balancing_loss = self._compute_load_balancing_loss(
                gating_outputs, expert_selection
            )
            importance_loss = self._compute_importance_loss(gating_outputs)
            
            expert_type_counts = {
                etype: len(names) 
                for etype, names in classification['expert'].items()
            }
            
            clipping_ratio = (
                num_clipped / len(self.expert_params) 
                if self.expert_params and num_clipped > 0 else 0.0
            )
            
            output = {
                "clip_applied": clip_applied,
                "normalization_applied": normalization_applied,
                "expert_gradient_norm": expert_norm,
                "non_expert_gradient_norm": non_expert_norm,
                "scale_gap": scale_gap,
                "load_balancing_loss": load_balancing_loss,
                "importance_loss": importance_loss,
                "utilization_loss": 0.0,
                "clipping_ratio": clipping_ratio,
                "num_expert_params": len(self.expert_params),
                "num_non_expert_params": len(self.non_expert_params),
                "expert_type_counts": expert_type_counts,
                "total_auxiliary_loss": (
                    load_balancing_loss + importance_loss
                )
            }
            
            if self.config.verbose_logging:
                self._logger.info(
                    f"MoE Gradient Stats: expert_norm={expert_norm:.6f}, "
                    f"non_expert_norm={non_expert_norm:.6f}, "
                    f"scale_gap={scale_gap:.2e}, "
                    f"clipped={num_clipped}/{len(self.expert_params)}"
                )
            
            self._logger.info(
                f"MoE gradient optimization complete: "
                f"expert_params={len(self.expert_params)}, "
                f"scale_gap={scale_gap:.2e}"
            )
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=output,
                execution_time=time.time() - start_time,
                metadata={
                    "step": current_step,
                    "expert_param_count": len(self.expert_params),
                    "expert_types": expert_type_counts
                }
            )
            
        except Exception as e:
            error_msg = f"MoE gradient optimization failed: {str(e)}"
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
    
    def get_expert_parameters(self) -> List[str]:
        """
        Get the list of identified expert parameter names.
        
        Returns:
            List of parameter names classified as experts.
        """
        return self.expert_params.copy()
    
    def get_non_expert_parameters(self) -> List[str]:
        """
        Get the list of identified non-expert parameter names.
        
        Returns:
            List of parameter names classified as non-experts.
        """
        return self.non_expert_params.copy()
    
    def get_gradient_statistics(self) -> Dict[str, Any]:
        """
        Get the latest gradient statistics.
        
        Returns:
            Dictionary containing gradient norm and scale information.
        """
        return self.gradient_stats.copy()
    
    def reset(self) -> None:
        """
        Reset operator state for a new training run.
        
        Clears parameter classifications, history, and statistics.
        """
        self.expert_params.clear()
        self.non_expert_params.clear()
        self.gating_params.clear()
        self.clip_history.clear()
        self.gradient_stats.clear()
        self.step_count = 0
        self._is_warmed_up = False
        self._logger.info("MoE gradient operator state reset")


class POPSSMoEGradientFacade:
    """
    Convenience facade for quick MoE gradient optimization setup.
    
    This facade provides a simplified interface for common use cases,
    automatically handling model analysis and gradient processing.
    
    Usage:
        >>> facade = POPSSMoEGradientFacade()
        >>> result = facade.process(model, step=current_step)
        >>> loss = result.output['total_auxiliary_loss']
    """
    
    def __init__(
        self,
        expert_clip_threshold: float = 0.1,
        enable_adaptive_clip: bool = True,
        load_balancing_weight: float = 0.01,
        **kwargs
    ):
        """
        Initialize the facade with MoE-specific parameters.
        
        Args:
            expert_clip_threshold: Clipping threshold for expert gradients
            enable_adaptive_clip: Enable adaptive threshold adjustment
            load_balancing_weight: Weight for load balancing loss
            **kwargs: Additional configuration passed to config
        """
        config = POPSSMoEGradientConfig(
            expert_clip_threshold=expert_clip_threshold,
            enable_adaptive_clip=enable_adaptive_clip,
            load_balancing_weight=load_balancing_weight,
            **kwargs
        )
        self.operator = POPSSMoEGradientOperator(config)
    
    def process(
        self,
        model: nn.Module,
        step: Optional[int] = None,
        **kwargs
    ) -> PiscesLxOperatorResult:
        """
        Process model gradients with MoE optimization.
        
        Args:
            model: Model containing MoE layers
            step: Current training step (optional)
            **kwargs: Additional arguments passed to execute()
            
        Returns:
            PiscesLxOperatorResult with optimization results
        """
        inputs = {"model": model, **kwargs}
        if step is not None:
            inputs["step"] = step
        
        return self.operator.execute(inputs)
    
    def get_auxiliary_losses(
        self,
        result: PiscesLxOperatorResult
    ) -> Dict[str, float]:
        """
        Extract auxiliary loss values from operator result.
        
        Args:
            result: Result from process() call.
            
        Returns:
            Dictionary of auxiliary loss components.
        """
        if result.is_success() and result.output:
            return {
                "load_balancing": result.output.get("load_balancing_loss", 0.0),
                "importance": result.output.get("importance_loss", 0.0),
                "utilization": result.output.get("utilization_loss", 0.0),
                "total": result.output.get("total_auxiliary_loss", 0.0)
            }
        return {}


class POPSSExpertGradientClipper:
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = float(max_norm)
        self.norm_type = float(norm_type)

    def clip(self, parameters) -> float:
        return float(torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type))


__all__ = [
    "POPSSMoEExpertType",
    "POPSSMoEGradientConfig",
    "POPSSMoEGradientOperator",
    "POPSSMoEGradientFacade",
    "POPSSExpertGradientClipper",
]
