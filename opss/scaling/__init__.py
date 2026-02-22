#!/usr/bin/env/python3
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
Scaling Laws Operators Module

Comprehensive implementation of scaling laws for optimal model training.
Based on Chinchilla scaling laws: https://arxiv.org/abs/2203.15556

Features:
    - Chinchilla-optimal parameter and token calculation
    - Memory estimation and correction
    - Config-to-resource mapping
    - Model configuration selection
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

from utils.dc import PiscesLxLogger
from configs.version import VERSION
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus


_LOG = PiscesLxLogger(__name__)


@dataclass
class POPSSScalingConfig:
    """Configuration for scaling law calculations."""
    compute_budget: float = 1e21
    budget_unit: str = "flops"
    target_model_size: Optional[str] = None
    enable_memory_correction: bool = True


@dataclass
class POPSSScalingResult:
    """Result of scaling law calculations."""
    optimal_parameters: float
    optimal_tokens: float
    compute_efficiency: float
    recommended_config: Optional[Dict[str, Any]] = None
    memory_correction: Optional[float] = None


class POPSSChinchillaScaler:
    """
    Chinchilla scaling laws implementation for optimal model parameter and token allocation.
    
    Based on the Chinchilla paper: https://arxiv.org/abs/2203.15556
    
    Key findings:
        - C ≈ 6ND (compute, parameters, data tokens)
        - Optimal allocation: N = D
        - For 1T tokens, optimal is ~4B parameters
    """
    
    A100_FLOPS_PER_SECOND = 3.12e17
    
    RUCHBAH_CONFIGS = [
        (22, 896, "0.5B"),
        (24, 2048, "1.5B"),
        (32, 2560, "7B"),
        (40, 3200, "32B"),
        (48, 4096, "70B"),
        (64, 5120, "314B"),
        (80, 6144, "671B"),
    ]
    
    @staticmethod
    def optimal_nd(c_budget: float, unit: str = "flops") -> Tuple[float, float]:
        """
        Calculate optimal model parameters (N) and training tokens (D) for given compute budget.
        
        Args:
            c_budget: Compute budget (FLOPs or GPU hours)
            unit: Budget unit - "flops" for FLOPs, "gpu_hours" for GPU hours
        
        Returns:
            Tuple[float, float]: (optimal_parameters, optimal_tokens)
        """
        if unit == "gpu_hours":
            flops_budget = c_budget * POPSSChinchillaScaler.A100_FLOPS_PER_SECOND
        else:
            flops_budget = c_budget
        
        optimal_scale = math.sqrt(flops_budget / 6.0)
        
        return optimal_scale, optimal_scale
    
    @staticmethod
    def estimate_params(
        layers: int,
        hidden_size: int,
        vocab_size: int = 151646,
        moe_experts: int = 64,
        moe_top_k: int = 2
    ) -> float:
        """
        Estimate model parameters for Yv architecture.
        
        Args:
            layers: Number of transformer layers
            hidden_size: Hidden dimension size
            vocab_size: Vocabulary size
            moe_experts: Number of MoE experts
            moe_top_k: Top-k experts used
        
        Returns:
            float: Estimated parameter count
        """
        embed_params = vocab_size * hidden_size
        layer_params = 12 * hidden_size * hidden_size
        moe_multiplier = moe_experts / moe_top_k
        layer_params += moe_multiplier * 8 * hidden_size * hidden_size
        
        total_layer_params = layers * layer_params
        output_params = vocab_size * hidden_size
        
        return embed_params + total_layer_params + output_params
    
    @staticmethod
    def get_config_for_params(target_params: float) -> Tuple[int, int, str]:
        """
        Find nearest Yv configuration for target parameter count.
        
        Args:
            target_params: Target parameter count
        
        Returns:
            Tuple[int, int, str]: (layers, hidden_size, name)
        """
        best_config = POPSSChinchillaScaler.RUCHBAH_CONFIGS[0]
        best_diff = float('inf')
        
        for layers, hidden_size, name in POPSSChinchillaScaler.RUCHBAH_CONFIGS:
            params = POPSSChinchillaScaler.estimate_params(layers, hidden_size)
            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_config = (layers, hidden_size, name)
        
        return best_config
    
    @staticmethod
    def chinchilla_memory_correction(
        base_memory: float,
        chinchilla_enabled: bool,
        optimal_params: float,
        original_params: float
    ) -> float:
        """
        Apply memory correction for Chinchilla-optimized models.
        
        Args:
            base_memory: Base memory estimate
            chinchilla_enabled: Whether Chinchilla optimization is enabled
            optimal_params: Optimal parameter count from Chinchilla
            original_params: Original parameter count
        
        Returns:
            float: Corrected memory estimate
        """
        if not chinchilla_enabled or original_params <= 0:
            return base_memory
        
        reduction_factor = (original_params - optimal_params) / original_params
        memory_reduction = 1.0 - (0.05 * reduction_factor)
        memory_reduction = max(0.9, memory_reduction)
        
        return base_memory * memory_reduction


class POPSSScalingOperator(PiscesLxOperatorInterface):
    """
    Scaling Law Operator.
    
    Provides intelligent scaling law calculations for optimal model training.
    
    Features:
        - Compute-optimal parameter and token calculation
        - Memory estimation with Chinchilla correction
        - Automatic model configuration selection
        - Cost analysis and planning
    
    Input:
        config: POPSSScalingConfig with budget and constraints
    
    Output:
        POPSSScalingResult with optimal parameters, tokens, and recommendations
    """
    
    def __init__(self, config: Optional[POPSSScalingConfig] = None):
        super().__init__()
        self.name = "scaling"
        self.version = VERSION
        self.config = config or POPSSScalingConfig()
        self._LOG = get_logger("pisceslx.ops.scaling")
    
    @property
    def description(self) -> str:
        return "Scaling law operator for optimal model training"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "operation": {"type": "str", "required": True, "enum": ["optimal", "estimate", "memory"]},
            "compute_budget": {"type": "float", "required": False},
            "budget_unit": {"type": "str", "required": False, "enum": ["flops", "gpu_hours"]},
            "layers": {"type": "int", "required": False},
            "hidden_size": {"type": "int", "required": False},
            "vocab_size": {"type": "int", "required": False},
            "base_memory": {"type": "float", "required": False},
            "chinchilla_enabled": {"type": "bool", "required": False},
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "success": {"type": "bool"},
            "optimal_parameters": {"type": "float"},
            "optimal_tokens": {"type": "float"},
            "recommended_config": {"type": "dict"},
            "memory_correction": {"type": "float"},
        }
    
    def execute(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Execute scaling operation."""
        operation = inputs.get("operation", "optimal")
        
        try:
            if operation == "optimal":
                return self._optimal_scaling(inputs)
            elif operation == "estimate":
                return self._estimate_params(inputs)
            elif operation == "memory":
                return self._memory_correction(inputs)
            else:
                return PiscesLxOperatorResult(
                    status=PiscesLxOperatorStatus.FAILED,
                    output={},
                    error=f"Unknown operation: {operation}"
                )
        except Exception as e:
            self._LOG.error(f"Scaling operation failed: {e}")
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.FAILED,
                output={},
                error=str(e)
            )
    
    def _optimal_scaling(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Calculate optimal parameters and tokens."""
        budget = inputs.get("compute_budget", self.config.compute_budget)
        unit = inputs.get("budget_unit", self.config.budget_unit)
        
        params, tokens = POPSSChinchillaScaler.optimal_nd(budget, unit)
        
        layers, hidden_size, name = POPSSChinchillaScaler.get_config_for_params(params)
        
        efficiency = min(1.0, params / POPSSChinchillaScaler.estimate_params(layers, hidden_size))
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={
                "optimal_parameters": params,
                "optimal_tokens": tokens,
                "compute_efficiency": efficiency,
                "recommended_config": {
                    "layers": layers,
                    "hidden_size": hidden_size,
                    "config_name": name,
                    "estimated_params": POPSSChinchillaScaler.estimate_params(layers, hidden_size)
                }
            }
        )
    
    def _estimate_params(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Estimate model parameters."""
        layers = inputs.get("layers", 32)
        hidden_size = inputs.get("hidden_size", 2560)
        vocab_size = inputs.get("vocab_size", 151646)
        
        params = POPSSChinchillaScaler.estimate_params(layers, hidden_size, vocab_size)
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"parameters": params}
        )
    
    def _memory_correction(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Apply Chinchilla memory correction."""
        base_memory = inputs.get("base_memory", 0.0)
        chinchilla_enabled = inputs.get("chinchilla_enabled", True)
        optimal_params = inputs.get("optimal_params", 0.0)
        original_params = inputs.get("original_params", 0.0)
        
        corrected = POPSSChinchillaScaler.chinchilla_memory_correction(
            base_memory, chinchilla_enabled, optimal_params, original_params
        )
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={
                "corrected_memory": corrected,
                "correction_factor": corrected / base_memory if base_memory > 0 else 1.0
            }
        )


__all__ = [
    "POPSSScalingConfig",
    "POPSSScalingResult",
    "POPSSChinchillaScaler",
    "POPSSScalingOperator",
]
