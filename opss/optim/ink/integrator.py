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
Ink Integrator - Unified Integration of GaLore/FP4/ROOT

This module provides a unified integration layer for combining GaLore, FP4, and ROOT
optimization techniques within the Ink optimizer framework.

Key Features:
    - Lazy loading of optimization components
    - Unified state management across components
    - Seamless integration with existing operators
    - Configuration-driven component activation

Integration Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    POPSSInkIntegrator                        │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
    │  │   GaLore    │ │    FP4      │ │    ROOT     │           │
    │  │ (Optional)  │ │ (Optional)  │ │ (Optional)  │           │
    │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘           │
    │         │               │               │                   │
    │         └───────────────┼───────────────┘                   │
    │                         ▼                                   │
    │              ┌─────────────────────┐                        │
    │              │  Unified Interface  │                        │
    │              └─────────────────────┘                        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Component Roles:
    - GaLore: Gradient low-rank projection for memory efficiency
    - FP4: 4-bit weight quantization for extreme memory savings
    - ROOT: Momentum orthogonalization for faster convergence
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple
from configs.version import VERSION

from .config import POPSSInkConfig


class POPSSInkIntegrator:
    """
    Unified Integrator for GaLore, FP4, and ROOT Components.
    
    This class provides a unified interface for integrating multiple optimization
    techniques within the Ink optimizer. It handles lazy loading, state management,
    and coordinated execution of all enabled components.
    
    The integrator follows a pipeline approach:
        1. Apply sparse gradient selection (from Ink core)
        2. Apply GaLore projection (if enabled)
        3. Apply FP4 quantization (if enabled)
        4. Apply ROOT orthogonalization (if enabled)
    
    Attributes:
        config: POPSSInkConfig instance with all settings
        _galore_operator: Lazy-loaded GaLore operator
        _fp4_quantizer: Lazy-loaded FP4 quantizer
        _root_operator: Lazy-loaded ROOT operator
        _galore_config: GaLore configuration
        _fp4_config: FP4 configuration
        _root_config: ROOT configuration
    
    Example:
        >>> config = POPSSInkConfig(
        ...     use_galore=True,
        ...     use_fp4=True,
        ...     use_root_ortho=True
        ... )
        >>> integrator = POPSSInkIntegrator(config)
        >>> 
        >>> # Apply GaLore projection
        >>> projected_grad = integrator.apply_galore(grad, "layer.weight", state)
        >>> 
        >>> # Apply ROOT orthogonalization
        >>> ortho_momentum = integrator.apply_root_ortho(momentum)
    """
    
    def __init__(self, config: POPSSInkConfig):
        """
        Initialize the integrator with configuration.
        
        Args:
            config: POPSSInkConfig instance with component settings
        """
        self.config = config
        
        self._galore_operator = None
        self._fp4_quantizer = None
        self._root_operator = None
        
        self._galore_config = None
        self._fp4_config = None
        self._root_config = None
        
        self._galore_states: Dict[str, Any] = {}
        self._fp4_states: Dict[str, Any] = {}
        self._root_states: Dict[str, Any] = {}
        
        self._initialized = False
    
    def initialize(self):
        """
        Initialize all enabled components.
        
        This method lazily loads and initializes each component based on
        configuration settings. Components are only loaded when enabled.
        """
        if self._initialized:
            return
        
        if self.config.use_galore:
            self._init_galore()
        
        if self.config.use_fp4:
            self._init_fp4()
        
        if self.config.use_root_ortho:
            self._init_root()
        
        self._initialized = True
    
    def _init_galore(self):
        """Initialize GaLore operator and configuration."""
        try:
            from opss.optim.galore import POPSSGaLoreOperator, POPSSGaLoreConfig
            
            self._galore_operator = POPSSGaLoreOperator()
            self._galore_config = POPSSGaLoreConfig(
                rank=self.config.galore_rank,
                update_proj_gap=self.config.galore_update_proj_gap,
                quantization_bits=self.config.galore_quantization_bits,
                min_rank=self.config.galore_min_rank,
                max_rank=self.config.galore_max_rank,
                rank_adapt_interval=self.config.galore_rank_adapt_interval,
                rank_adapt_threshold=self.config.galore_rank_adapt_threshold,
                memory_efficient=self.config.galore_memory_efficient,
                moe_expert_only=self.config.galore_moe_expert_only,
            )
        except ImportError:
            pass
    
    def _init_fp4(self):
        """Initialize FP4 quantizer and configuration."""
        try:
            from opss.optim.fp4 import POPSSFP4Quantizer, POPSSFP4Config
            
            self._fp4_config = POPSSFP4Config(
                block_size=self.config.fp4_block_size,
                stochastic_rounding=self.config.fp4_stochastic_rounding,
            )
            self._fp4_quantizer = POPSSFP4Quantizer(
                block_size=self._fp4_config.block_size,
                stochastic_rounding=self._fp4_config.stochastic_rounding,
            )
        except ImportError:
            pass
    
    def _init_root(self):
        """Initialize ROOT operator and configuration."""
        try:
            from opss.optim.root import POPSSROOTConfig
            
            self._root_config = POPSSROOTConfig(
                lr=self.config.lr,
                beta1=self.config.betas[0],
                beta2=self.config.betas[1],
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
                orthogonalization_steps=self.config.root_ortho_steps,
                soft_threshold=self.config.root_soft_threshold,
                spectral_norm_clip=self.config.root_spectral_norm_clip,
                use_orthogonalization=True,
                use_soft_threshold=True,
                min_dim_for_ortho=self.config.root_min_dim_for_ortho,
            )
            self._root_available = True
        except ImportError:
            self._root_available = False
    
    def apply_galore(
        self,
        gradient: torch.Tensor,
        param_name: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Apply GaLore gradient projection.
        
        Projects the gradient to a low-rank subspace for memory efficiency.
        
        Args:
            gradient: Input gradient tensor
            param_name: Parameter name for state tracking
            state: Optional state dictionary for GaLore
        
        Returns:
            Projected gradient tensor
        """
        if self._galore_operator is None:
            return gradient
        
        if state is None:
            state = self._galore_states.get(param_name, {})
        
        try:
            result = self._galore_operator.execute({
                "model": None,
                "gradients": {param_name: gradient},
                "config": self._galore_config,
                "optimizer_state": state,
            })
            
            if result.is_success() and result.output:
                new_state = result.output.get("optimizer_state", {})
                self._galore_states[param_name] = new_state
                
                projected = result.output.get("gradients", {}).get(param_name, gradient)
                return projected if projected is not None else gradient
        except Exception:
            pass
        
        return gradient
    
    def apply_fp4_forward(
        self,
        weight: torch.Tensor,
        param_name: str,
    ) -> torch.Tensor:
        """
        Apply FP4 weight quantization for forward pass.
        
        Quantizes weights to FP4 format for memory-efficient forward computation.
        
        Args:
            weight: Weight tensor to quantize
            param_name: Parameter name for state tracking
        
        Returns:
            Quantized weight tensor
        """
        if self._fp4_quantizer is None:
            return weight
        
        try:
            indices, scales = self._fp4_quantizer.quantize(
                weight, self._fp4_config.block_size
            )
            
            dequantized = self._fp4_quantizer.dequantize(
                indices, scales, weight.shape
            )
            
            return dequantized
        except Exception:
            pass
        
        return weight
    
    def apply_fp4_backward(
        self,
        gradient: torch.Tensor,
        param_name: str,
    ) -> torch.Tensor:
        """
        Apply FP4 gradient quantization for backward pass.
        
        Quantizes gradients for memory-efficient backward computation.
        
        Args:
            gradient: Gradient tensor to quantize
            param_name: Parameter name for state tracking
        
        Returns:
            Quantized gradient tensor
        """
        if self._fp4_quantizer is None:
            return gradient
        
        try:
            indices, scales = self._fp4_quantizer.quantize(
                gradient, self._fp4_config.block_size
            )
            
            dequantized = self._fp4_quantizer.dequantize(
                indices, scales, gradient.shape
            )
            
            return dequantized
        except Exception:
            pass
        
        return gradient
    
    def apply_root_ortho(
        self,
        momentum: torch.Tensor,
        param_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Apply ROOT momentum orthogonalization.
        
        Orthogonalizes the momentum tensor for faster convergence.
        
        Args:
            momentum: Momentum tensor to orthogonalize
            param_name: Optional parameter name for state tracking
        
        Returns:
            Orthogonalized momentum tensor
        """
        if self._root_operator is None:
            return momentum
        
        try:
            return self._root_operator.orthogonalize_momentum(
                momentum,
                steps=self.config.root_ortho_steps,
            )
        except Exception:
            pass
        
        return momentum
    
    def apply_root_denoise(
        self,
        gradient: torch.Tensor,
        param_name: Optional[str] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply ROOT soft threshold denoising.
        
        Removes noise from gradients using soft thresholding.
        
        Args:
            gradient: Gradient tensor to denoise
            param_name: Optional parameter name for state tracking
        
        Returns:
            Tuple of (denoised gradient, preservation ratio)
        """
        if self._root_operator is None:
            return gradient, 1.0
        
        try:
            return self._root_operator.soft_threshold_denoise(
                gradient,
                threshold=self.config.root_soft_threshold,
            )
        except Exception:
            pass
        
        return gradient, 1.0
    
    def apply_root_spectral_clip(
        self,
        update: torch.Tensor,
        param_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Apply ROOT spectral norm clipping.
        
        Clips the spectral norm of the update for stability.
        
        Args:
            update: Update tensor to clip
            param_name: Optional parameter name for state tracking
        
        Returns:
            Clipped update tensor
        """
        if self._root_operator is None:
            return update
        
        try:
            sn = self._root_operator.compute_spectral_norm(update)
            
            if sn > self.config.root_spectral_norm_clip:
                return update * (self.config.root_spectral_norm_clip / sn)
        except Exception:
            pass
        
        return update
    
    def should_orthogonalize(self, param: torch.Tensor) -> bool:
        """
        Check if parameter should be orthogonalized.
        
        Only 2D parameters with minimum dimension >= threshold are orthogonalized.
        
        Args:
            param: Parameter tensor to check
        
        Returns:
            True if parameter should be orthogonalized
        """
        if param.dim() < 2:
            return False
        
        min_dim = min(param.shape)
        return min_dim >= self.config.root_min_dim_for_ortho
    
    def get_galore_state(self, param_name: str) -> Dict[str, Any]:
        """Get GaLore state for a parameter."""
        return self._galore_states.get(param_name, {})
    
    def set_galore_state(self, param_name: str, state: Dict[str, Any]):
        """Set GaLore state for a parameter."""
        self._galore_states[param_name] = state
    
    def get_fp4_state(self, param_name: str) -> Dict[str, Any]:
        """Get FP4 state for a parameter."""
        return self._fp4_states.get(param_name, {})
    
    def set_fp4_state(self, param_name: str, state: Dict[str, Any]):
        """Set FP4 state for a parameter."""
        self._fp4_states[param_name] = state
    
    def get_component_status(self) -> Dict[str, bool]:
        """
        Get status of all components.
        
        Returns:
            Dictionary with component availability status
        """
        return {
            "galore_enabled": self.config.use_galore,
            "galore_available": self._galore_operator is not None,
            "fp4_enabled": self.config.use_fp4,
            "fp4_available": self._fp4_quantizer is not None,
            "root_enabled": self.config.use_root_ortho,
            "root_available": getattr(self, '_root_available', False),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all components.
        
        Returns:
            Dictionary with statistics from each component
        """
        stats = {
            "components": self.get_component_status(),
            "galore_states": len(self._galore_states),
            "fp4_states": len(self._fp4_states),
            "root_states": len(self._root_states),
        }
        
        if self._galore_config is not None:
            stats["galore_rank"] = self._galore_config.rank
            stats["galore_update_gap"] = self._galore_config.update_proj_gap
        
        if self._fp4_config is not None:
            stats["fp4_block_size"] = self._fp4_config.block_size
        
        if self._root_config is not None:
            stats["root_ortho_steps"] = self._root_config.orthogonalization_steps
            stats["root_soft_threshold"] = self._root_config.soft_threshold
        
        return stats
    
    def reset_states(self):
        """Reset all component states."""
        self._galore_states.clear()
        self._fp4_states.clear()
        self._root_states.clear()
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for serialization.
        
        Returns:
            State dictionary containing all integrator state
        """
        return {
            "galore_states": self._galore_states.copy(),
            "fp4_states": self._fp4_states.copy(),
            "root_states": self._root_states.copy(),
            "initialized": self._initialized,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state from dictionary.
        
        Args:
            state_dict: State dictionary to load
        """
        self._galore_states = state_dict.get("galore_states", {}).copy()
        self._fp4_states = state_dict.get("fp4_states", {}).copy()
        self._root_states = state_dict.get("root_states", {}).copy()
        self._initialized = state_dict.get("initialized", False)
