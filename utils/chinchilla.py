#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Tuple

class PiscesLxCoreChinchillaScaler:
    """
    Chinchilla scaling laws implementation for optimal model parameter and token allocation.
    
    Based on the Chinchilla paper: https://arxiv.org/abs/2203.15556
    """
    
    @staticmethod
    def optimal_nd(c_budget: float, unit: str = "flops") -> Tuple[float, float]:
        """
        Calculate optimal model parameters (N) and training tokens (D) for given compute budget.
        
        Args:
            c_budget: Compute budget (FLOPs or GPU hours)
            unit: Budget unit - "flops" for FLOPs, "gpu_hours" for GPU hours
        
        Returns:
            Tuple[float, float]: (optimal_parameters, optimal_tokens)
            
        Notes:
            - Based on Chinchilla scaling laws: C ≈ 6ND
            - Optimal allocation: N ∝ C^0.5, D ∝ C^0.5
            - For 6.7B parameters, optimal is ~1.4T tokens (Chinchilla paper)
        """
        # Convert GPU hours to FLOPs if needed
        # Approximate: 1 A100 GPU hour ≈ 3.12e17 FLOPs (FP16)
        if unit == "gpu_hours":
            flops_budget = c_budget * 3.12e17
        else:
            flops_budget = c_budget
        
        # Chinchilla scaling: C ≈ 6ND, optimal N = D
        # So: N = D = sqrt(C / 6)
        optimal_scale = math.sqrt(flops_budget / 6.0)
        
        optimal_parameters = optimal_scale
        optimal_tokens = optimal_scale
        
        return optimal_parameters, optimal_tokens
    
    @staticmethod
    def scale_to_existing_block(parameters: float, existing_blocks: list = None) -> Tuple[int, int]:
        """
        Scale optimal parameters to nearest existing model configuration.
        
        Args:
            parameters: Target parameter count
            existing_blocks: List of existing (layers, hidden_size) configurations
        
        Returns:
            Tuple[int, int]: (layers, hidden_size) for nearest configuration
        """
        if existing_blocks is None:
            # Default Ruchbah configurations (layers, hidden_size)
            existing_blocks = [
                (22, 896),   # ~0.5B
                (24, 2048),  # ~1.5B  
                (32, 2560),  # ~7B
                (40, 3200),  # ~32B
                (48, 4096),  # ~70B
                (64, 5120),  # ~314B
                (80, 6144),  # ~671B
            ]
        
        # Find nearest configuration by parameter count
        best_config = existing_blocks[0]
        min_diff = abs(PiscesLxCoreChinchillaScaler.self_estimate_params(*best_config) - parameters)
        
        for config in existing_blocks[1:]:
            params = PiscesLxCoreChinchillaScaler.self_estimate_params(*config)
            diff = abs(params - parameters)
            if diff < min_diff:
                min_diff = diff
                best_config = config
        
        return best_config
    
    @staticmethod
    def self_estimate_params(layers: int, hidden_size: int, 
                            vocab_size: int = 71164,
                            moe_experts: int = 64,
                            moe_top_k: int = 2) -> float:
        """
        Estimate model parameters for Ruchbah architecture.
        
        Args:
            layers: Number of transformer layers
            hidden_size: Hidden dimension size
            vocab_size: Vocabulary size
            moe_experts: Number of MoE experts
            moe_top_k: Top-k experts used
        
        Returns:
            float: Estimated parameter count
        """
        # Embedding parameters
        embed_params = vocab_size * hidden_size
        
        # Transformer layer parameters (simplified)
        # Attention: 4 * hidden_size^2 (Q,K,V,O projections)
        # FFN: 8 * hidden_size^2 (gate, up, down projections)
        # MoE: (moe_experts / moe_top_k) * 8 * hidden_size^2
        layer_params = 12 * hidden_size * hidden_size  # Base transformer
        moe_multiplier = moe_experts / moe_top_k
        layer_params += moe_multiplier * 8 * hidden_size * hidden_size
        
        total_layer_params = layers * layer_params
        
        # Output head
        output_params = vocab_size * hidden_size
        
        total_params = embed_params + total_layer_params + output_params
        
        return total_params
    
    @staticmethod
    def chinchilla_memory_correction(base_memory: float, 
                                   chinchilla_enabled: bool,
                                   optimal_params: float,
                                   original_params: float) -> float:
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
        
        # Chinchilla models have fewer parameters for same compute
        # Apply 5-10% memory reduction
        memory_reduction = 1.0 - (0.05 * (original_params - optimal_params) / original_params)
        memory_reduction = max(0.9, memory_reduction)  # Cap at 10% minimum reduction
        
        return base_memory * memory_reduction