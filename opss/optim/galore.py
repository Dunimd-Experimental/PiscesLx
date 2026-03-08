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
GaLore Optimizer Operator - Gradient Low-Rank Projection for Memory-Efficient Training
Based on GaLore algorithm from tools/train/impl.py and utils/optim/galore.py

This module implements the GaLore (Gradient Low-Rank Projection) optimization technique
for memory-efficient training of large language models. GaLore reduces memory footprint
by projecting gradients onto a low-rank subspace during optimization.

Key Features:
    - Memory-efficient gradient optimization
    - Low-rank projection for gradient compression
    - Configurable rank and update frequency
    - Compatible with standard optimization workflows

Reference:
    GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection
    https://arxiv.org/abs/2403.03507
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn as nn
from configs.version import VERSION
from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorConfig,
    PiscesLxOperatorStatus,
)


@dataclass
class POPSSGaLoreConfig(PiscesLxOperatorConfig):
    """
    Configuration for GaLore optimizer.
    
    Attributes:
        rank: Rank for low-rank projection. Higher rank preserves more gradient
              information but uses more memory. Default: 128
        update_proj_gap: Number of steps between projection matrix updates.
                        More frequent updates improve accuracy but increase computation.
                        Default: 50
        scale: Scaling factor for transformed gradients. Default: 1.0
        proj_type: Type of projection method. Options: "std" (standard SVD).
                  Default: "std"
        quantization_bits: Bits for quantizing projection matrices (0, 4, 8).
                          8 = 8bit quantization, saves 75% memory. Default: 8
        lr_ratio: Learning rate ratio for GaLore parameters. Default: 1.0
        min_rank: Minimum rank for adaptive rank adjustment. Default: 32
        max_rank: Maximum rank for adaptive rank adjustment. Default: 512
        rank_adapt_interval: Steps between rank adaptation. Default: 1000
        rank_adapt_threshold: Threshold for rank adaptation. Default: 0.1
        memory_efficient: Enable memory-efficient mode. Default: False
        moe_expert_only: Apply GaLore only to MoE experts. Default: False
    """
    name: str = "galore"
    version: str = VERSION

    rank: int = 128
    update_proj_gap: int = 50
    scale: float = 1.0
    proj_type: str = "std"
    quantization_bits: int = 8
    lr_ratio: float = 1.0
    min_rank: int = 32
    max_rank: int = 512
    rank_adapt_interval: int = 1000
    rank_adapt_threshold: float = 0.1
    memory_efficient: bool = False
    moe_expert_only: bool = False


class POPSSGaLoreOperator(PiscesLxOperatorInterface):
    """
    GaLore Optimizer Operator - Memory-efficient training via gradient low-rank projection.
    
    This operator implements the GaLore optimization technique which reduces memory
    usage during training by projecting gradients onto a low-rank subspace. This
    allows training larger models with the same memory budget.
    
    The algorithm works by:
        1. Computing the low-rank projection matrix via SVD on gradients
        2. Projecting gradients to the low-rank subspace
        3. Performing optimization in the compressed space
        4. Periodically updating the projection matrix
    
    Example:
        >>> config = POPSSGaLoreConfig(rank=64, update_proj_gap=100)
        >>> operator = POPSSGaLoreOperator()
        >>> result = operator.execute({
        ...     "model": model,
        ...     "gradients": gradients,
        ...     "config": config
        ... })
        >>> if result.success:
        ...     updated_model = result.data["model"]
    """
    
    def __init__(self):
        """Initialize the GaLore optimizer operator."""
        super().__init__()
        self._name = "galore_optimizer"
        self._version = VERSION

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def description(self) -> str:
        return "GaLore optimizer operator (gradient low-rank projection)"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "model": {"type": "nn.Module", "required": True},
            "gradients": {"type": "dict", "required": False},
            "config": {"type": "POPSSGaLoreConfig", "required": False},
            "optimizer_state": {"type": "dict", "required": False},
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "model": {"type": "nn.Module"},
            "optimizer_state": {"type": "dict"},
            "statistics": {"type": "dict"},
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        return isinstance(inputs, dict) and ("model" in inputs)
        
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute GaLore optimization step.
        
        This method performs one step of GaLore optimization, including
        projection matrix updates and gradient transformation.
        
        Args:
            inputs: Dictionary containing optimization inputs
                - model: The model to optimize (nn.Module)
                - gradients: Dictionary mapping parameter names to gradients
                - config: GaLore configuration (POPSSGaLoreConfig)
                - optimizer_state: Optional existing optimizer state
            
        Returns:
            PiscesLxOperatorResult: Result containing
                - model: Updated model with transformed gradients
                - optimizer_state: New optimizer state
                - statistics: Optimization statistics
        
        Raises:
            ValueError: If model is not provided in inputs
        """
        try:
            model = inputs.get("model")
            gradients = inputs.get("gradients", {})
            config = inputs.get("config", POPSSGaLoreConfig())
            optimizer_state = inputs.get("optimizer_state", {})

            if model is None:
                raise ValueError("Model is required for GaLore optimization")

            if not optimizer_state:
                optimizer_state = self._initialize_galore_state(model, config)

            updated_model, new_state, stats = self._perform_galore_step(
                model, gradients, config, optimizer_state
            )

            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "model": updated_model,
                    "optimizer_state": new_state,
                    "statistics": stats,
                },
                metadata={
                    "version": self.version,
                    "algorithm": "GaLore",
                },
            )

        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                output=None,
                error=str(e),
                metadata={
                    "version": self.version,
                    "error_type": type(e).__name__,
                },
            )
    
    def _initialize_galore_state(self, model: nn.Module, config: POPSSGaLoreConfig) -> Dict[str, Any]:
        """
        Initialize GaLore optimizer state.
        
        Creates the initial state dictionary that tracks projection matrices
        and update schedules for each trainable parameter.
        
        Args:
            model: The neural network model
            config: GaLore configuration
        
        Returns:
            Dictionary containing:
                - step: Current optimization step
                - proj_matrices: Projection matrices for each parameter
                - last_update_steps: Last step when projection was updated
                - rank: Configured rank
                - update_proj_gap: Configured update gap
        """
        state = {
            "step": 0,
            "proj_matrices": {},
            "last_update_steps": {},
            "rank": config.rank,
            "update_proj_gap": config.update_proj_gap
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                state["proj_matrices"][name] = None
                state["last_update_steps"][name] = 0
                
        return state
    
    def _perform_galore_step(self, 
                           model: nn.Module, 
                           gradients: Dict[str, torch.Tensor],
                           config: POPSSGaLoreConfig,
                           state: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
        """
        Perform single GaLore optimization step.
        
        Iterates through all parameters, updates projection matrices when
        necessary, and applies GaLore gradient transformation.
        
        Args:
            model: The neural network model
            gradients: Dictionary of parameter gradients
            config: GaLore configuration
            state: Current optimizer state
        
        Returns:
            Tuple of (updated_model, new_state, statistics)
        """
        state["step"] += 1
        stats = {"step": state["step"]}
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if name in gradients:
                grad = gradients[name]
                
                if self._should_update_projection(name, state):
                    proj_matrix = self._compute_projection_matrix(grad, config.rank)
                    state["proj_matrices"][name] = proj_matrix
                    state["last_update_steps"][name] = state["step"]
                    
                transformed_grad = self._apply_galore_transform(
                    grad, state["proj_matrices"][name], config
                )
                
                with torch.no_grad():
                    param.grad = transformed_grad
                    
        stats["active_projections"] = sum(1 for proj in state["proj_matrices"].values() if proj is not None)
        stats["total_parameters"] = len(list(model.parameters()))
        
        return model, state, stats
    
    def _should_update_projection(self, param_name: str, state: Dict[str, Any]) -> bool:
        """
        Determine if projection matrix should be updated.
        
        Checks if enough steps have passed since the last projection update.
        
        Args:
            param_name: Name of the parameter
            state: Current optimizer state
        
        Returns:
            True if projection should be updated, False otherwise
        """
        last_update = state["last_update_steps"][param_name]
        return (state["step"] - last_update) >= state["update_proj_gap"]
    
    def _compute_projection_matrix(self, gradient: torch.Tensor, rank: int) -> torch.Tensor:
        """
        Compute low-rank projection matrix for gradient.
        
        Uses SVD to compute the projection matrix that captures the most
        important directions in the gradient space.
        
        Args:
            gradient: The gradient tensor
            rank: Target rank for projection
        
        Returns:
            Projection matrix of shape (gradient_dim, rank)
        """
        # SVD kernels on CUDA are not consistently implemented for bf16.
        # Compute the projection in fp32 for stability/compatibility, then cast back.
        orig_dtype = gradient.dtype
        work_grad = gradient
        if work_grad.dtype not in (torch.float32, torch.float64):
            work_grad = work_grad.float()

        if work_grad.dim() == 1:
            _, _, V = torch.svd(work_grad.unsqueeze(0))
        else:
            grad_flat = work_grad.view(-1, work_grad.shape[-1])
            try:
                _, _, V = torch.svd_lowrank(grad_flat, q=rank)
            except Exception:
                _, _, V = torch.svd(grad_flat)
        
        if V.shape[1] >= rank:
            projection_matrix = V[:, :rank]
        else:
            projection_matrix = V

        if projection_matrix.dtype != orig_dtype:
            projection_matrix = projection_matrix.to(dtype=orig_dtype)

        return projection_matrix
    
    def _apply_galore_transform(self, 
                              gradient: torch.Tensor, 
                              proj_matrix: Optional[torch.Tensor],
                              config: POPSSGaLoreConfig) -> torch.Tensor:
        """
        Apply GaLore gradient transformation.
        
        Projects the gradient to low-rank space and back, effectively
        compressing the gradient information.
        
        Args:
            gradient: Original gradient tensor
            proj_matrix: Projection matrix (None if not yet computed)
            config: GaLore configuration
        
        Returns:
            Transformed gradient tensor
        """
        if proj_matrix is None:
            return gradient
            
        if gradient.dim() == 1:
            if proj_matrix.shape[0] == 1:
                transformed_grad = gradient * proj_matrix.squeeze()
            else:
                transformed_grad = gradient
        else:
            original_shape = gradient.shape
            grad_flat = gradient.view(-1, original_shape[-1])
            
            low_rank_grad = torch.matmul(grad_flat, proj_matrix)
            transformed_grad = torch.matmul(low_rank_grad, proj_matrix.t())
            
            transformed_grad = transformed_grad.view(original_shape)
            
        transformed_grad = transformed_grad * config.scale
        
        return transformed_grad
    
    def validate_config(self, config: Any) -> bool:
        """
        Validate configuration parameters.
        
        Ensures all configuration values are within valid ranges.
        
        Args:
            config: Configuration to validate
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, POPSSGaLoreConfig):
            return False
            
        return (config.rank > 0 and 
                config.update_proj_gap > 0 and
                config.scale > 0)


class POPSSGaLoreOptimizerAdapter:
    """
    GaLore Optimizer Adapter for integration with existing training pipelines.
    
    This adapter provides a simple interface for using GaLore optimization
    in standard training loops without modifying existing code significantly.
    
    Example:
        >>> config = POPSSGaLoreConfig(rank=64)
        >>> adapter = POPSSGaLoreOptimizerAdapter(config)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     model, stats = adapter.step(model, gradients)
    """
    
    def __init__(self, config: POPSSGaLoreConfig):
        """
        Initialize the GaLore adapter.
        
        Args:
            config: GaLore configuration
        """
        self.operator = POPSSGaLoreOperator()
        self.config = config
        self.state = {}
        
    def step(self, model: nn.Module, gradients: Dict[str, torch.Tensor]) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Execute optimization step.
        
        Performs one step of GaLore optimization and updates internal state.
        
        Args:
            model: The model to optimize
            gradients: Dictionary of parameter gradients
        
        Returns:
            Tuple of (updated_model, statistics)
        
        Raises:
            RuntimeError: If optimization fails
        """
        inputs = {
            "model": model,
            "gradients": gradients,
            "config": self.config,
            "optimizer_state": self.state
        }
        
        result = self.operator.execute(inputs)

        if result is not None and hasattr(result, "is_success") and result.is_success():
            out = result.output or {}
            self.state = out.get("optimizer_state", {}) or {}
            return out.get("model", model), out.get("statistics", {})

        raise RuntimeError(f"GaLore optimization failed: {getattr(result, 'error', None) or 'unknown error'}")
