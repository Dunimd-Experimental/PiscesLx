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
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import math

# Use dms_core logging exclusively
import dms_core
logger = dms_core.log.get_logger(__name__)

@dataclass
class GaLoreConfig:
    """Configuration for GaLore optimizer."""
    enabled: bool = False
    rank: int = 128
    update_interval: int = 200
    target_modules: List[str] = None
    lr_ratio: float = 1.0
    min_rank: int = 32
    max_rank: int = 512
    adaptive_rank: bool = True
    rank_adapt_interval: int = 1000
    rank_adapt_threshold: float = 0.1
    quantization_bits: int = 0  # 0 means no quantization
    memory_efficient: bool = True
    moe_expert_only: bool = False
    multimodal_modules: List[str] = None
    sequence_threshold: int = 4096
    gradient_accumulation_sync: bool = True
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if self.multimodal_modules is None:
            self.multimodal_modules = ["vision_encoder", "audio_encoder", "multimodal_fusion"]


class GaLoreProjection:
    """Gradient Low-Rank Projection for parameter-efficient training."""
    
    def __init__(self, rank: int, update_interval: int = 200, memory_efficient: bool = True):
        self.rank = rank
        self.update_interval = update_interval
        self.memory_efficient = memory_efficient
        self.step_count = 0
        self.projection_matrix = None
        self.momentum = None
        
    def should_update_projection(self) -> bool:
        """Check if projection matrix should be updated."""
        return self.step_count % self.update_interval == 0
    
    def update_projection_matrix(self, grad: torch.Tensor) -> None:
        """Update low-rank projection matrix using SVD."""
        if grad.numel() < self.rank * 2:
            # Skip projection for small gradients
            return
            
        # Reshape gradient to 2D if needed
        original_shape = grad.shape
        if grad.dim() > 2:
            grad_2d = grad.view(-1, original_shape[-1])
        else:
            grad_2d = grad
            
        try:
            # Compute SVD for low-rank approximation
            U, S, Vt = torch.linalg.svd(grad_2d, full_matrices=False)
            
            # Select top-rank components
            rank = min(self.rank, len(S))
            self.projection_matrix = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
            
            if self.memory_efficient:
                # Store only essential components
                self.U = U[:, :rank]
                self.S = S[:rank]
                self.Vt = Vt[:rank, :]
                self.projection_matrix = None  # Clear full matrix
                
        except Exception as e:
            logger.warning(f"SVD computation failed: {e}, skipping projection update")
            self.projection_matrix = None
    
    def project_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Project gradient to low-rank subspace."""
        if self.projection_matrix is None and not hasattr(self, 'U'):
            return grad
            
        original_shape = grad.shape
        
        if hasattr(self, 'U') and hasattr(self, 'S') and hasattr(self, 'Vt'):
            # Use stored components for projection
            if grad.dim() > 2:
                grad_2d = grad.view(-1, original_shape[-1])
            else:
                grad_2d = grad
                
            # Project to low-rank subspace: U * S * Vt * grad
            projected = self.U.T @ grad_2d @ self.Vt.T
            reconstructed = self.U @ torch.diag(self.S) @ projected @ self.Vt
            
            if grad.dim() > 2:
                return reconstructed.view(original_shape)
            else:
                return reconstructed
        else:
            return grad
    
    def step(self) -> None:
        """Increment step counter."""
        self.step_count += 1


class GaLoreOptimizer:
    """GaLore optimizer wrapper for memory-efficient training."""
    
    def __init__(self, base_optimizer: Optimizer, config: GaLoreConfig):
        self.base_optimizer = base_optimizer
        self.config = config
        self.projections = {}
        self.enabled = config.enabled
        self.rank_adapt_counter = 0
        self.gradient_norms = {}
        
        if self.enabled:
            logger.info(f"GaLore optimizer initialized with rank={config.rank}, update_interval={config.update_interval}")
    
    def _should_apply_galore(self, param_name: str, param: torch.Tensor) -> bool:
        """Check if GaLore should be applied to this parameter."""
        # Check if enabled
        if not self.enabled:
            return False
            
        # Check sequence length threshold
        if hasattr(param, 'sequence_length') and param.sequence_length < self.config.sequence_threshold:
            return False
        
        # Check if param_name is valid
        if param_name is None:
            return False
        
        # Check target modules
        for target_module in self.config.target_modules:
            if target_module in param_name:
                return True
                
        # Check MoE expert only mode
        if self.config.moe_expert_only and 'expert' not in param_name:
            return False
            
        # Check multimodal modules
        for mm_module in self.config.multimodal_modules:
            if mm_module in param_name:
                return True
                
        return False
    
    def _get_projection(self, param_name: str, param_shape: tuple) -> GaLoreProjection:
        """Get or create projection for parameter."""
        if param_name not in self.projections:
            self.projections[param_name] = GaLoreProjection(
                rank=self.config.rank,
                update_interval=self.config.update_interval,
                memory_efficient=self.config.memory_efficient
            )
        return self.projections[param_name]
    
    def _adapt_rank(self, param_name: str, grad_norm: float) -> None:
        """Adapt rank based on gradient statistics."""
        if param_name not in self.gradient_norms:
            self.gradient_norms[param_name] = []
        
        self.gradient_norms[param_name].append(grad_norm)
        
        if len(self.gradient_norms[param_name]) > self.config.rank_adapt_interval:
            # Calculate gradient statistics
            recent_norms = self.gradient_norms[param_name][-self.config.rank_adapt_interval:]
            mean_norm = sum(recent_norms) / len(recent_norms)
            max_norm = max(recent_norms)
            
            if max_norm > mean_norm * (1 + self.config.rank_adapt_threshold):
                # Increase rank for high variance gradients
                new_rank = min(self.config.max_rank, int(self.config.rank * 1.5))
                if new_rank != self.config.rank:
                    logger.info(f"Adapting rank for {param_name}: {self.config.rank} -> {new_rank}")
                    self.config.rank = new_rank
                    # Recreate projection with new rank
                    if param_name in self.projections:
                        del self.projections[param_name]
            
            # Clean up old gradient norms
            self.gradient_norms[param_name] = recent_norms[-100:]
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients in base optimizer."""
        self.base_optimizer.zero_grad(set_to_none)
    
    def step(self, closure=None) -> Optional[float]:
        """Perform optimization step with GaLore projection."""
        if not self.enabled:
            return self.base_optimizer.step(closure)
        
        # Apply GaLore projections to gradients
        for group in self.base_optimizer.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                param_name = getattr(param, 'galore_name', None)
                if param_name is None:
                    # Fallback to regular name attribute
                    param_name = getattr(param, 'name', None)
                if param_name is None:
                    # Skip parameters without names
                    continue
                
                if self._should_apply_galore(param_name, param):
                    projection = self._get_projection(param_name, param.shape)
                    
                    # Calculate gradient norm for rank adaptation
                    grad_norm = param.grad.norm().item()
                    self._adapt_rank(param_name, grad_norm)
                    
                    # Update projection matrix if needed
                    if projection.should_update_projection():
                        projection.update_projection(param.grad)
                    
                    # Project gradient
                    param.grad.data = projection.project_gradient(param.grad)
                    
                    projection.step()
        
        # Step base optimizer
        return self.base_optimizer.step(closure)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict."""
        state = self.base_optimizer.state_dict()
        state['galore_config'] = self.config.__dict__
        state['galore_projections'] = {
            name: {
                'step_count': proj.step_count,
                'rank': proj.rank if hasattr(proj, 'rank') else self.config.rank
            }
            for name, proj in self.projections.items()
        }
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state dict."""
        if 'galore_config' in state_dict:
            # Restore GaLore configuration
            config_dict = state_dict.pop('galore_config')
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        if 'galore_projections' in state_dict:
            # Restore projection states
            projections_data = state_dict.pop('galore_projections')
            for name, proj_data in projections_data.items():
                if name in self.projections:
                    self.projections[name].step_count = proj_data.get('step_count', 0)
        
        self.base_optimizer.load_state_dict(state_dict)
    
    @property
    def param_groups(self):
        """Get parameter groups from base optimizer."""
        return self.base_optimizer.param_groups


def create_galore_optimizer(
    model: nn.Module,
    base_optimizer_class: type = torch.optim.AdamW,
    lr: float = 1e-4,
    config: Optional[GaLoreConfig] = None,
    **optimizer_kwargs
) -> GaLoreOptimizer:
    """
    Create GaLore optimizer wrapper.
    
    Args:
        model: Model to optimize
        base_optimizer_class: Base optimizer class (default: AdamW)
        lr: Learning rate
        config: GaLore configuration
        **optimizer_kwargs: Additional arguments for base optimizer
    
    Returns:
        GaLoreOptimizer instance
    """
    if config is None:
        config = GaLoreConfig()
    
    # Create parameter groups
    param_groups = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_groups.append({
                'params': [param],
                'lr': lr * config.lr_ratio,
                'name': name
            })
            # Store parameter name for GaLore filtering
            param.name = name
            param.galore_name = name  # Use a more specific attribute name
    
    # Create base optimizer
    base_optimizer = base_optimizer_class(param_groups, lr=lr, **optimizer_kwargs)
    
    # Wrap with GaLore
    return GaLoreOptimizer(base_optimizer, config)


def get_galore_memory_savings(config: GaLoreConfig, model_size_mb: float) -> float:
    """
    Estimate memory savings from GaLore optimization.
    
    Args:
        config: GaLore configuration
        model_size_mb: Model size in MB
    
    Returns:
        Estimated memory savings percentage
    """
    if not config.enabled:
        return 0.0
    
    # Base savings from reduced optimizer states
    base_savings = 0.65  # 65% reduction in optimizer memory
    
    # Adjust based on rank ratio
    rank_ratio = config.rank / config.max_rank if config.max_rank > 0 else 0.25
    efficiency_multiplier = 1.0 - (rank_ratio * 0.3)  # Higher rank = less savings
    
    # Apply quantization bonus
    quantization_bonus = 0.0
    if config.quantization_bits > 0:
        quantization_bonus = 0.15  # Additional 15% from quantization
    
    total_savings = base_savings * efficiency_multiplier + quantization_bonus
    return min(total_savings, 0.825)  # Cap at 82.5% maximum savings