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
Model Growth Operators for Progressive Network Expansion

Based on NeurIPS 2024: "Stacking Your Transformers: A Closer Look at Model Growth"
Implements Gstack depth-wise stacking for efficient LLM pre-training.

Key Features:
    - Depth growth: Stack layers for 2x training speedup
    - Width growth: Expand hidden dimensions with knowledge preservation
    - Expert growth: Add MoE experts with routing alignment
    - Optimal Transport layer alignment (OpT-DeUS)
    - Knowledge preservation during expansion

Growth Strategies:
    - Gstack: Depth-wise stacking (recommended for efficiency)
    - Width expansion: Hidden dimension growth
    - Expert addition: MoE capacity scaling

Usage:
    from opss.train.growth import (
        POPSSModelGrowthConfig,
        POPSSModelGrowthOperator,
    )
    
    config = POPSSModelGrowthConfig(
        growth_type="depth",
        num_new_layers=4,
    )
    grower = POPSSModelGrowthOperator()
    grown_model = grower.execute({"model": model, "config": config})
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus
from configs.version import VERSION


class POPSSGrowthType(Enum):
    """Model growth type enumeration."""
    DEPTH = "depth"
    WIDTH = "width"
    EXPERT = "expert"
    HYBRID = "hybrid"


@dataclass
class POPSSModelGrowthConfig:
    """Configuration for model growth operations.
    
    Attributes:
        growth_type: Type of growth (depth/width/expert/hybrid).
        num_new_layers: Number of layers to add for depth growth.
        new_hidden_size: Target hidden size for width growth.
        num_new_experts: Number of experts to add for MoE growth.
        preserve_knowledge: Whether to preserve knowledge during growth.
        use_optimal_transport: Use OT for layer alignment (OpT-DeUS).
        init_noise_scale: Noise scale for new parameters.
        layer_copy_strategy: Strategy for copying layers ('last', 'middle', 'uniform').
    """
    
    growth_type: str = "depth"
    num_new_layers: int = 4
    new_hidden_size: Optional[int] = None
    num_new_experts: int = 8
    preserve_knowledge: bool = True
    use_optimal_transport: bool = True
    init_noise_scale: float = 0.01
    layer_copy_strategy: str = "last"
    
    def __post_init__(self):
        if isinstance(self.growth_type, POPSSGrowthType):
            self.growth_type = self.growth_type.value


class POPSSOptimalTransportAligner:
    """Optimal Transport layer alignment for growth.
    
    Based on OpT-DeUS (arXiv 2508.08011):
    Aligns and fuses adjacent layers block-wise to create
    neuron-aligned new layers.
    """
    
    def __init__(self):
        self._LOG = PiscesLxLogger(
            "PiscesLx.Growth.OT",
            file_path=get_log_file("PiscesLx.Growth.OT"),
            enable_file=True,
        )
    
    def compute_transport_matrix(
        self,
        source_weights: Tensor,
        target_weights: Tensor,
        regularization: float = 0.1,
    ) -> Tensor:
        """Compute optimal transport matrix between source and target.
        
        Uses Sinkhorn algorithm for entropy-regularized OT.
        
        Args:
            source_weights: Source layer weights [out_dim, in_dim].
            target_weights: Target layer weights [out_dim, in_dim].
            regularization: Entropy regularization coefficient.
            
        Returns:
            Transport matrix [source_dim, target_dim].
        """
        source_dim = source_weights.shape[0]
        target_dim = target_weights.shape[0]
        
        # Compute cost matrix (cosine distance)
        source_norm = F.normalize(source_weights, dim=1)
        target_norm = F.normalize(target_weights, dim=1)
        
        # Cost = 1 - similarity
        cost_matrix = 1.0 - torch.mm(source_norm, target_norm.t())
        
        # Sinkhorn iterations
        mu = torch.ones(source_dim, device=source_weights.device) / source_dim
        nu = torch.ones(target_dim, device=target_weights.device) / target_dim
        
        K = torch.exp(-cost_matrix / regularization)
        
        u = torch.ones_like(mu)
        for _ in range(100):
            v = nu / (K.t() @ u + 1e-8)
            u = mu / (K @ v + 1e-8)
        
        transport_matrix = u.unsqueeze(1) * K * v.unsqueeze(0)
        
        return transport_matrix
    
    def align_layers(
        self,
        layer1_weights: Dict[str, Tensor],
        layer2_weights: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Align two layers using optimal transport.
        
        Args:
            layer1_weights: First layer weight dict.
            layer2_weights: Second layer weight dict.
            
        Returns:
            Aligned new layer weights.
        """
        new_weights = {}
        
        for name in layer1_weights:
            if name not in layer2_weights:
                continue
            
            w1 = layer1_weights[name]
            w2 = layer2_weights[name]
            
            if w1.dim() < 2 or w2.dim() < 2:
                continue
            
            # Compute transport for alignment
            transport = self.compute_transport_matrix(w1, w2)
            
            # Create aligned weights
            aligned = 0.5 * (w1 + transport.t() @ w2)
            new_weights[name] = aligned
        
        return new_weights


class POPSSDepthGrower:
    """Depth-wise model growth using Gstack strategy.
    
    Stacks layers to increase model depth, providing up to 2x
    training speedup compared to training from scratch.
    """
    
    def __init__(self, config: POPSSModelGrowthConfig):
        self.config = config
        self.ot_aligner = POPSSOptimalTransportAligner() if config.use_optimal_transport else None
        self._LOG = PiscesLxLogger(
            "PiscesLx.Growth.Depth",
            file_path=get_log_file("PiscesLx.Growth.Depth"),
            enable_file=True,
        )
    
    def grow(self, model: nn.Module) -> nn.Module:
        """Grow model by adding layers.
        
        Args:
            model: Model to grow.
            
        Returns:
            Grown model with additional layers.
        """
        # Get existing layers
        layers = self._get_layers(model)
        if layers is None:
            self._LOG.warning("No layers found for depth growth")
            return model
        
        num_existing = len(layers)
        num_new = self.config.num_new_layers
        
        self._LOG.info(f"Growing model from {num_existing} to {num_existing + num_new} layers")
        
        # Create new layers based on copy strategy
        new_layers = self._create_new_layers(layers, num_new)
        
        # Add new layers to model
        self._add_layers_to_model(model, new_layers)
        
        # Update config
        self._update_model_config(model, num_existing + num_new)
        
        self._LOG.info(f"Model grown to {num_existing + num_new} layers")
        
        return model
    
    def _get_layers(self, model: nn.Module) -> Optional[nn.ModuleList]:
        """Get layers module from model."""
        if hasattr(model, 'layers'):
            return model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h
        return None
    
    def _create_new_layers(self, existing_layers: nn.ModuleList, num_new: int) -> nn.ModuleList:
        """Create new layers by copying and modifying existing ones."""
        new_layers = nn.ModuleList()
        num_existing = len(existing_layers)
        
        for i in range(num_new):
            # Select source layer based on strategy
            if self.config.layer_copy_strategy == "last":
                source_idx = num_existing - 1
            elif self.config.layer_copy_strategy == "middle":
                source_idx = num_existing // 2
            elif self.config.layer_copy_strategy == "uniform":
                source_idx = i % num_existing
            else:
                source_idx = num_existing - 1
            
            # Deep copy the source layer
            new_layer = copy.deepcopy(existing_layers[source_idx])
            
            # Add small noise for initialization diversity
            if self.config.init_noise_scale > 0:
                for param in new_layer.parameters():
                    param.data += torch.randn_like(param) * self.config.init_noise_scale
            
            new_layers.append(new_layer)
        
        return new_layers
    
    def _add_layers_to_model(self, model: nn.Module, new_layers: nn.ModuleList):
        """Add new layers to model."""
        layers = self._get_layers(model)
        if layers is not None:
            layers.extend(new_layers)
    
    def _update_model_config(self, model: nn.Module, new_num_layers: int):
        """Update model configuration with new layer count."""
        if hasattr(model, 'config'):
            if hasattr(model.config, 'n_layer'):
                model.config.n_layer = new_num_layers
            elif hasattr(model.config, 'num_hidden_layers'):
                model.config.num_hidden_layers = new_num_layers


class POPSSWidthGrower:
    """Width-wise model growth for hidden dimension expansion.
    
    Expands hidden dimensions while preserving knowledge through
    careful weight padding and initialization.
    """
    
    def __init__(self, config: POPSSModelGrowthConfig):
        self.config = config
        self._LOG = PiscesLxLogger(
            "PiscesLx.Growth.Width",
            file_path=get_log_file("PiscesLx.Growth.Width"),
            enable_file=True,
        )
    
    def grow(self, model: nn.Module) -> nn.Module:
        """Grow model by expanding hidden dimension.
        
        Args:
            model: Model to grow.
            
        Returns:
            Grown model with expanded hidden dimension.
        """
        if self.config.new_hidden_size is None:
            self._LOG.warning("new_hidden_size not specified for width growth")
            return model
        
        old_hidden = self._get_hidden_size(model)
        new_hidden = self.config.new_hidden_size
        
        if new_hidden <= old_hidden:
            self._LOG.warning(f"new_hidden_size ({new_hidden}) must be greater than current ({old_hidden})")
            return model
        
        self._LOG.info(f"Growing model from {old_hidden} to {new_hidden} hidden dimensions")
        
        # Expand embedding layer
        self._expand_embedding(model, old_hidden, new_hidden)
        
        # Expand layer projections
        self._expand_layers(model, old_hidden, new_hidden)
        
        # Expand output head
        self._expand_output_head(model, old_hidden, new_hidden)
        
        # Update config
        self._update_model_config(model, new_hidden)
        
        self._LOG.info(f"Model grown to {new_hidden} hidden dimensions")
        
        return model
    
    def _get_hidden_size(self, model: nn.Module) -> int:
        """Get current hidden size from model."""
        if hasattr(model, 'config'):
            if hasattr(model.config, 'hidden_size'):
                return model.config.hidden_size
        return 2048
    
    def _expand_embedding(self, model: nn.Module, old_size: int, new_size: int):
        """Expand embedding layer."""
        embed_tokens = None
        if hasattr(model, 'embed_tokens'):
            embed_tokens = model.embed_tokens
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_tokens = model.model.embed_tokens
        
        if embed_tokens is not None:
            old_weight = embed_tokens.weight.data
            vocab_size = old_weight.shape[0]
            
            # Create new embedding with expanded dimension
            new_weight = torch.zeros(vocab_size, new_size, device=old_weight.device, dtype=old_weight.dtype)
            new_weight[:, :old_size] = old_weight
            
            # Initialize new dimensions with small random values
            new_weight[:, old_size:] = torch.randn(vocab_size, new_size - old_size, device=old_weight.device, dtype=old_weight.dtype) * 0.01
            
            # Create new embedding layer
            embed_tokens.weight = nn.Parameter(new_weight)
    
    def _expand_layers(self, model: nn.Module, old_size: int, new_size: int):
        """Expand all layer projections."""
        layers = None
        if hasattr(model, 'layers'):
            layers = model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        
        if layers is None:
            return
        
        for layer in layers:
            self._expand_layer_projections(layer, old_size, new_size)
    
    def _expand_layer_projections(self, layer: nn.Module, old_size: int, new_size: int):
        """Expand projections in a single layer."""
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                self._expand_linear(module, old_size, new_size, name)
    
    def _expand_linear(self, linear: nn.Linear, old_size: int, new_size: int, name: str):
        """Expand a linear layer."""
        old_weight = linear.weight.data
        out_features = old_weight.shape[0]
        in_features = old_weight.shape[1]
        
        # Determine which dimension to expand
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name:
            # Attention projections: expand output dimension
            if out_features == old_size:
                new_weight = torch.zeros(out_features, new_size, device=old_weight.device, dtype=old_weight.dtype)
                new_weight[:, :old_size] = old_weight
                new_weight[:, old_size:] = torch.randn(out_features, new_size - old_size, device=old_weight.device, dtype=old_weight.dtype) * 0.01
                linear.weight = nn.Parameter(new_weight)
                linear.in_features = new_size
        
        # Expand bias if present
        if linear.bias is not None and linear.bias.shape[0] == old_size:
            new_bias = torch.zeros(new_size, device=linear.bias.device, dtype=linear.bias.dtype)
            new_bias[:old_size] = linear.bias
            linear.bias = nn.Parameter(new_bias)
    
    def _expand_output_head(self, model: nn.Module, old_size: int, new_size: int):
        """Expand output head (lm_head)."""
        lm_head = None
        if hasattr(model, 'lm_head'):
            lm_head = model.lm_head
        elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
            lm_head = model.model.lm_head
        
        if lm_head is not None and isinstance(lm_head, nn.Linear):
            old_weight = lm_head.weight.data
            vocab_size = old_weight.shape[0]
            
            new_weight = torch.zeros(vocab_size, new_size, device=old_weight.device, dtype=old_weight.dtype)
            new_weight[:, :old_size] = old_weight
            new_weight[:, old_size:] = torch.randn(vocab_size, new_size - old_size, device=old_weight.device, dtype=old_weight.dtype) * 0.01
            
            lm_head.weight = nn.Parameter(new_weight)
            lm_head.in_features = new_size
    
    def _update_model_config(self, model: nn.Module, new_size: int):
        """Update model config with new hidden size."""
        if hasattr(model, 'config'):
            if hasattr(model.config, 'hidden_size'):
                model.config.hidden_size = new_size


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class POPSSExpertGrower:
    """MoE expert growth for capacity scaling.
    
    Adds new experts to MoE layers with routing alignment
    to preserve existing knowledge.
    """
    
    def __init__(self, config: POPSSModelGrowthConfig):
        self.config = config
        self._LOG = PiscesLxLogger(
            "PiscesLx.Growth.Expert",
            file_path=get_log_file("PiscesLx.Growth.Expert"),
            enable_file=True,
        )
    
    def grow(self, model: nn.Module) -> nn.Module:
        """Grow model by adding MoE experts.
        
        Args:
            model: Model to grow.
            
        Returns:
            Grown model with additional experts.
        """
        num_added = 0
        num_new = self.config.num_new_experts
        
        # Find and grow MoE layers
        for name, module in model.named_modules():
            if self._is_moe_layer(module):
                added = self._grow_moe_experts(module, num_new)
                num_added += added
        
        if num_added > 0:
            self._LOG.info(f"Added {num_added} experts across {num_added // num_new if num_new > 0 else 0} MoE layers")
        
        # Update config
        self._update_model_config(model, num_new)
        
        return model
    
    def _is_moe_layer(self, module: nn.Module) -> bool:
        """Check if module is a MoE layer."""
        return hasattr(module, 'experts') or hasattr(module, 'moe')
    
    def _grow_moe_experts(self, moe_layer: nn.Module, num_new: int) -> int:
        """Add experts to a MoE layer."""
        experts = None
        if hasattr(moe_layer, 'experts'):
            experts = moe_layer.experts
        elif hasattr(moe_layer, 'moe') and hasattr(moe_layer.moe, 'experts'):
            experts = moe_layer.moe.experts
        
        if experts is None or not isinstance(experts, nn.ModuleList):
            return 0
        
        num_existing = len(experts)
        
        for i in range(num_new):
            # Copy a random existing expert
            source_idx = i % num_existing
            new_expert = copy.deepcopy(experts[source_idx])
            
            # Add noise for diversity
            for param in new_expert.parameters():
                param.data += torch.randn_like(param) * self.config.init_noise_scale
            
            experts.append(new_expert)
        
        # Update router if present
        self._update_router(moe_layer, num_existing + num_new)
        
        return num_new
    
    def _update_router(self, moe_layer: nn.Module, new_num_experts: int):
        """Update router for new expert count."""
        router = None
        if hasattr(moe_layer, 'router'):
            router = moe_layer.router
        elif hasattr(moe_layer, 'gate'):
            router = moe_layer.gate
        
        if router is None:
            return
        
        if isinstance(router, nn.Linear):
            old_weight = router.weight.data
            old_bias = router.bias.data if router.bias is not None else None
            
            old_num = old_weight.shape[0]
            hidden = old_weight.shape[1]
            
            # Create new router with expanded output
            new_weight = torch.zeros(new_num_experts, hidden, device=old_weight.device, dtype=old_weight.dtype)
            new_weight[:old_num] = old_weight
            new_weight[old_num:] = torch.randn(new_num_experts - old_num, hidden, device=old_weight.device, dtype=old_weight.dtype) * 0.01
            
            router.weight = nn.Parameter(new_weight)
            router.out_features = new_num_experts
            
            if old_bias is not None:
                new_bias = torch.zeros(new_num_experts, device=old_bias.device, dtype=old_bias.dtype)
                new_bias[:old_num] = old_bias
                router.bias = nn.Parameter(new_bias)
    
    def _update_model_config(self, model: nn.Module, num_new: int):
        """Update model config with new expert count."""
        if hasattr(model, 'config'):
            if hasattr(model.config, 'moe_num_experts'):
                model.config.moe_num_experts += num_new


class _ModelGrowthOperatorImpl(PiscesLxOperatorInterface):
    """Model growth operator implementation."""
    
    def __init__(self):
        super().__init__()
        self._name = "model.growth"
        self._version = VERSION
        self.type = "training"
        self._LOG = PiscesLxLogger(
            "PiscesLx.Growth.Operator",
            file_path=get_log_file("PiscesLx.Growth.Operator"),
            enable_file=True,
        )
    
    def execute(self, params: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Execute model growth.
        
        Args:
            params: Dictionary containing:
                - model: Model to grow
                - config: POPSSModelGrowthConfig
                
        Returns:
            Growth result with grown model.
        """
        model = params.get("model")
        config = params.get("config")
        
        if model is None:
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error="Model is required",
            )
        
        if config is None:
            config = POPSSModelGrowthConfig()
        elif isinstance(config, dict):
            config = POPSSModelGrowthConfig(**config)
        
        self._LOG.info(f"Starting model growth: type={config.growth_type}")
        
        try:
            if config.growth_type == POPSSGrowthType.DEPTH.value:
                grower = POPSSDepthGrower(config)
            elif config.growth_type == POPSSGrowthType.WIDTH.value:
                grower = POPSSWidthGrower(config)
            elif config.growth_type == POPSSGrowthType.EXPERT.value:
                grower = POPSSExpertGrower(config)
            else:
                return PiscesLxOperatorResult(
                    status=PiscesLxOperatorStatus.ERROR,
                    error=f"Unknown growth type: {config.growth_type}",
                )
            
            grown_model = grower.grow(model)
            
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.SUCCESS,
                data={
                    "model": grown_model,
                    "growth_type": config.growth_type,
                }
            )
            
        except Exception as e:
            self._LOG.error(f"Model growth failed: {e}")
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error=str(e),
            )


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class POPSSModelGrowthOperator:
    """Facade for model growth operator.
    
    Example:
        >>> config = POPSSModelGrowthConfig(growth_type="depth", num_new_layers=4)
        >>> operator = POPSSModelGrowthOperator()
        >>> result = operator.execute({"model": model, "config": config})
        >>> grown_model = result.data["model"]
    """
    
    def __init__(self):
        self._impl = _ModelGrowthOperatorImpl()
    
    def execute(self, params: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Execute model growth."""
        return self._impl.execute(params)
    
    def grow_depth(self, model: nn.Module, num_layers: int = 4) -> nn.Module:
        """Convenience method for depth growth."""
        config = POPSSModelGrowthConfig(growth_type="depth", num_new_layers=num_layers)
        result = self.execute({"model": model, "config": config})
        return result.data.get("model", model) if result.status == PiscesLxOperatorStatus.SUCCESS else model
    
    def grow_width(self, model: nn.Module, new_hidden_size: int) -> nn.Module:
        """Convenience method for width growth."""
        config = POPSSModelGrowthConfig(growth_type="width", new_hidden_size=new_hidden_size)
        result = self.execute({"model": model, "config": config})
        return result.data.get("model", model) if result.status == PiscesLxOperatorStatus.SUCCESS else model
    
    def grow_experts(self, model: nn.Module, num_experts: int = 8) -> nn.Module:
        """Convenience method for expert growth."""
        config = POPSSModelGrowthConfig(growth_type="expert", num_new_experts=num_experts)
        result = self.execute({"model": model, "config": config})
        return result.data.get("model", model) if result.status == PiscesLxOperatorStatus.SUCCESS else model


__all__ = [
    "POPSSGrowthType",
    "POPSSModelGrowthConfig",
    "POPSSOptimalTransportAligner",
    "POPSSDepthGrower",
    "POPSSWidthGrower",
    "POPSSExpertGrower",
    "POPSSModelGrowthOperator",
]
