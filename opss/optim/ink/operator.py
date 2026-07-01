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
Ink Optimizer Operator - Main Optimization Engine

This module implements the main Ink optimizer operator that orchestrates
all optimization components for unified, memory-efficient training.

Key Features:
    - INT8/INT4 optimizer state compression
    - Sparse gradient selection
    - Integration with GaLore, FP4, ROOT
    - PyTorch-compatible optimizer interface

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    POPSSInkOperator                          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  Pipeline:                                                  │
    │  1. Sparse Gradient Selection (Top-K%)                      │
    │  2. GaLore Projection (if enabled)                          │
    │  3. Dequantize States (INT8 momentum, INT4 variance)        │
    │  4. Adam-style Update with bias correction                  │
    │  5. ROOT Orthogonalization (if enabled)                     │
    │  6. Spectral Norm Clipping (if enabled)                     │
    │  7. Quantize States (back to INT8/INT4)                     │
    │  8. Parameter Update                                        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Memory Efficiency:
    - INT8 momentum: 4x compression
    - INT4 variance: 8x compression
    - Sparse gradients: Up to 100x reduction
    - Total: ~62.5% memory savings for optimizer states
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Iterable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from configs.version import VERSION
from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig,
)
from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from .config import POPSSInkConfig
from .quantizer import POPSSInkBlockQuantizer
from .sparse import POPSSInkSparseSelector
from .integrator import POPSSInkIntegrator
from .kv_cache import POPSSInkKVCacheQuantizer
from .gradient import POPSSInkGradientCompressor
from .moe import POPSSInkMoEManager
from .checkpoint import POPSSInkCheckpointSelector


class POPSSInkOperator(PiscesLxOperatorInterface):
    """
    Ink Optimizer Operator - Unified Memory-Efficient Training.
    
    This operator implements the complete Ink optimization pipeline,
    combining state compression, sparse gradients, and integration
    with existing optimization techniques.
    
    The operator follows the OPSC pattern and can be used either
    through the execute() method or as a PyTorch optimizer.
    
    Attributes:
        name: Operator name identifier
        version: Semantic version string
        config: POPSSInkConfig instance
        quantizer: Block quantizer for state compression
        sparse_selector: Sparse gradient selector
        integrator: Component integrator for GaLore/FP4/ROOT
        
    Example:
        >>> config = POPSSInkConfig(
        ...     lr=1e-4,
        ...     momentum_bits=8,
        ...     variance_bits=4,
        ...     sparse_ratio=0.01
        ... )
        >>> operator = POPSSInkOperator(config)
        >>> operator.initialize(model)
        >>> 
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     operator.step()
    """
    
    def __init__(self, config: Optional[POPSSInkConfig] = None):
        """
        Initialize the Ink operator.
        
        Args:
            config: POPSSInkConfig instance, uses defaults if None
        """
        super().__init__(config or POPSSInkConfig())
        
        self._name = "ink"
        self._version = VERSION
        
        self._quantizer: Optional[POPSSInkBlockQuantizer] = None
        self._sparse_selector: Optional[POPSSInkSparseSelector] = None
        self._integrator: Optional[POPSSInkIntegrator] = None
        self._kv_cache_quantizer: Optional[POPSSInkKVCacheQuantizer] = None
        self._gradient_compressor: Optional[POPSSInkGradientCompressor] = None
        self._moe_manager: Optional[POPSSInkMoEManager] = None
        self._checkpoint_selector: Optional[POPSSInkCheckpointSelector] = None
        
        self._momentum_int8: Dict[str, torch.Tensor] = {}
        self._momentum_scales: Dict[str, torch.Tensor] = {}
        self._variance_int4: Dict[str, torch.Tensor] = {}
        self._variance_scales: Dict[str, torch.Tensor] = {}
        self._step: Dict[str, int] = {}
        
        self._initialized = False
        self._model: Optional[nn.Module] = None
        
        self._stats = {
            "total_steps": 0,
            "total_params": 0,
            "total_elements": 0,
            "memory_saved_bytes": 0,
            "avg_sparsity": 0.0,
            "kv_cache_hits": 0,
            "kv_cache_misses": 0,
            "moe_offloads": 0,
            "moe_loads": 0,
        }
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return "Ink Optimizer - Unified INT8/INT4 State Compression with Sparse Gradients"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "model": {"type": "nn.Module", "required": True},
            "gradients": {"type": "dict", "required": False},
            "config": {"type": "POPSSInkConfig", "required": False},
            "step": {"type": "int", "required": False},
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "model": {"type": "nn.Module"},
            "statistics": {"type": "dict"},
            "memory_info": {"type": "dict"},
        }
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        return isinstance(inputs, dict) and ("model" in inputs or self._initialized)
    
    def initialize(self, model: nn.Module, config: Optional[POPSSInkConfig] = None):
        """
        Initialize the operator with a model.
        
        This method sets up all internal components and initializes
        compressed optimizer states for each trainable parameter.
        
        Args:
            model: Neural network model to optimize
            config: Optional configuration override
        """
        if config is not None:
            self.config = config
        
        self._model = model
        
        self._quantizer = POPSSInkBlockQuantizer(
            momentum_bits=self.config.momentum_bits,
            variance_bits=self.config.variance_bits,
            momentum_block_size=self.config.momentum_block_size,
            variance_block_size=self.config.variance_block_size,
        )
        
        self._sparse_selector = POPSSInkSparseSelector(
            sparse_ratio=self.config.sparse_ratio,
            warmup_steps=self.config.sparse_warmup_steps,
            adaptive=self.config.sparse_adaptive,
        )
        
        self._integrator = POPSSInkIntegrator(self.config)
        self._integrator.initialize()
        
        self._kv_cache_quantizer = POPSSInkKVCacheQuantizer(
            kv_bits=self.config.kv_cache_bits,
            block_size=self.config.kv_cache_block_size,
        )
        
        self._gradient_compressor = POPSSInkGradientCompressor(
            gradient_bits=self.config.gradient_bits,
            block_size=self.config.gradient_block_size,
            stochastic_rounding=True,
        )
        
        self._moe_manager = POPSSInkMoEManager(
            num_experts=64,
            max_experts_on_gpu=self.config.max_experts_on_gpu,
            offload_threshold=self.config.moe_offload_threshold,
            lru_cache_size=self.config.moe_lru_cache_size,
        )
        
        self._checkpoint_selector = POPSSInkCheckpointSelector(
            checkpoint_ratio=self.config.checkpoint_ratio,
            preserve_ratio=self.config.checkpoint_preserve_ratio,
            enable_transformer=self.config.checkpoint_transformer,
        )
        
        self._initialize_compressed_states(model)
        
        self._initialized = True
        
        self.logger.info(
            "Ink operator initialized",
            momentum_bits=self.config.momentum_bits,
            variance_bits=self.config.variance_bits,
            sparse_ratio=self.config.sparse_ratio,
            gradient_bits=self.config.gradient_bits,
            kv_cache_bits=self.config.kv_cache_bits,
            max_experts_on_gpu=self.config.max_experts_on_gpu,
            checkpoint_ratio=self.config.checkpoint_ratio,
            num_params=len(self._momentum_int8),
        )
    
    def _initialize_compressed_states(self, model: nn.Module):
        """
        Initialize compressed optimizer states for all parameters.
        
        Creates INT8 momentum and INT4 variance storage for each
        trainable parameter in the model.
        
        Args:
            model: Neural network model
        """
        total_elements = 0
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            numel = param.numel()
            total_elements += numel
            
            momentum_blocks = (numel + self.config.momentum_block_size - 1) // self.config.momentum_block_size
            variance_blocks = (numel + self.config.variance_block_size - 1) // self.config.variance_block_size
            
            if self.config.momentum_bits == 8:
                self._momentum_int8[name] = torch.zeros(
                    numel, dtype=torch.int8, device=param.device
                )
            else:
                self._momentum_int8[name] = torch.zeros(
                    (numel + 1) // 2, dtype=torch.int8, device=param.device
                )
            
            self._momentum_scales[name] = torch.ones(
                momentum_blocks, dtype=torch.float32, device=param.device
            )
            
            if self.config.variance_bits == 4:
                self._variance_int4[name] = torch.zeros(
                    (numel + 1) // 2, dtype=torch.int8, device=param.device
                )
            else:
                self._variance_int4[name] = torch.zeros(
                    numel, dtype=torch.int8, device=param.device
                )
            
            self._variance_scales[name] = torch.ones(
                variance_blocks, dtype=torch.float32, device=param.device
            )
            
            self._step[name] = 0
        
        self._stats["total_params"] = len(self._momentum_int8)
        self._stats["total_elements"] = total_elements
        
        original_memory = total_elements * 4 * 2
        compressed_memory = self._estimate_compressed_memory(total_elements)
        self._stats["memory_saved_bytes"] = original_memory - compressed_memory
    
    def _estimate_compressed_memory(self, num_elements: int) -> int:
        """Estimate compressed memory usage."""
        momentum_bytes = num_elements * (self.config.momentum_bits / 8)
        variance_bytes = num_elements * (self.config.variance_bits / 8)
        
        num_momentum_blocks = (num_elements + self.config.momentum_block_size - 1) // self.config.momentum_block_size
        num_variance_blocks = (num_elements + self.config.variance_block_size - 1) // self.config.variance_block_size
        
        scale_bytes = (num_momentum_blocks + num_variance_blocks) * 4
        
        return int(momentum_bytes + variance_bytes + scale_bytes)
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute one optimization step.
        
        This method performs a complete optimization step including
        gradient processing, state updates, and parameter updates.
        
        Args:
            inputs: Dictionary containing:
                - model: Model to optimize (optional if already initialized)
                - gradients: Precomputed gradients (optional)
                - config: Configuration override (optional)
                - step: Current step number (optional)
        
        Returns:
            PiscesLxOperatorResult with optimization statistics
        """
        start_time = time.time()
        
        try:
            model = inputs.get("model", self._model)
            gradients = inputs.get("gradients", {})
            
            if model is None:
                raise ValueError("Model is required for Ink optimization")
            
            if not self._initialized:
                self.initialize(model)
            
            stats = self._perform_step(model, gradients)
            
            execution_time = time.time() - start_time
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "model": model,
                    "statistics": stats,
                    "memory_info": self.get_memory_info(),
                },
                execution_time=execution_time,
                metadata={
                    "version": self.version,
                    "algorithm": "Ink",
                    "momentum_bits": self.config.momentum_bits,
                    "variance_bits": self.config.variance_bits,
                },
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={
                    "version": self.version,
                    "error_type": type(e).__name__,
                },
            )
    
    def _perform_step(
        self,
        model: nn.Module,
        gradients: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Perform a single optimization step.
        
        Args:
            model: Model to optimize
            gradients: Precomputed gradients (empty dict uses param.grad)
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "params_updated": 0,
            "total_grad_norm": 0.0,
            "avg_sparsity": 0.0,
        }
        
        total_norm_sq = 0.0
        total_sparsity = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            grad = gradients.get(name, param.grad)
            if grad is None:
                continue
            
            if grad.is_sparse:
                grad = grad.to_dense()
            
            total_norm_sq += grad.norm(2).item() ** 2
            
            sparse_grad, mask = self._sparse_selector.select(grad, name)
            sparsity = 1.0 - mask.float().mean().item()
            total_sparsity += sparsity
            param_count += 1
            
            if self.config.use_galore:
                galore_state = self._integrator.get_galore_state(name)
                sparse_grad = self._integrator.apply_galore(sparse_grad, name, galore_state)
            
            self._step[name] = self._step.get(name, 0) + 1
            step = self._step[name]
            
            momentum = self._dequantize_momentum(name, param.shape)
            variance = self._dequantize_variance(name, param.shape)
            
            momentum = self.config.betas[0] * momentum + (1 - self.config.betas[0]) * sparse_grad
            variance = self.config.betas[1] * variance + (1 - self.config.betas[1]) * sparse_grad ** 2
            
            if self.config.use_root_ortho and self._integrator.should_orthogonalize(param):
                momentum = self._integrator.apply_root_ortho(momentum, name)
            
            if self.config.use_root_ortho:
                momentum = self._integrator.apply_root_spectral_clip(momentum, name)
            
            bias_correction1 = 1 - self.config.betas[0] ** step
            bias_correction2 = 1 - self.config.betas[1] ** step
            
            update = momentum / (variance.sqrt() / math.sqrt(bias_correction2) + self.config.eps)
            update = update * (math.sqrt(bias_correction2) / bias_correction1)
            
            if self.config.weight_decay > 0:
                param.data.add_(param.data, alpha=-self.config.lr * self.config.weight_decay)
            
            param.data.add_(update, alpha=-self.config.lr)
            
            self._quantize_and_store_momentum(name, momentum)
            self._quantize_and_store_variance(name, variance)
            
            stats["params_updated"] += 1
        
        stats["total_grad_norm"] = total_norm_sq ** 0.5
        stats["avg_sparsity"] = total_sparsity / param_count if param_count > 0 else 0.0
        
        self._stats["total_steps"] += 1
        self._stats["avg_sparsity"] = stats["avg_sparsity"]
        
        return stats
    
    def _dequantize_momentum(self, name: str, shape: Tuple[int, ...]) -> torch.Tensor:
        """Dequantize momentum from INT8/INT4 to FP32."""
        if self.config.momentum_bits == 8:
            return self._quantizer.dequantize_int8(
                self._momentum_int8[name],
                self._momentum_scales[name],
                shape,
                block_size=self.config.momentum_block_size,
            )
        else:
            return self._quantizer.dequantize_int4(
                self._momentum_int8[name],
                self._momentum_scales[name],
                shape,
                block_size=self.config.momentum_block_size,
            )

    def _dequantize_variance(self, name: str, shape: Tuple[int, ...]) -> torch.Tensor:
        """Dequantize variance from INT4/INT8 to FP32."""
        if self.config.variance_bits == 4:
            return self._quantizer.dequantize_int4(
                self._variance_int4[name],
                self._variance_scales[name],
                shape,
                block_size=self.config.variance_block_size,
            )
        else:
            return self._quantizer.dequantize_int8(
                self._variance_int4[name],
                self._variance_scales[name],
                shape,
                block_size=self.config.variance_block_size,
            )
    
    def _quantize_and_store_momentum(self, name: str, momentum: torch.Tensor):
        """Quantize momentum and store compressed representation."""
        if self.config.momentum_bits == 8:
            quantized, scales = self._quantizer.quantize_int8(
                momentum, self.config.momentum_block_size
            )
        else:
            quantized, scales = self._quantizer.quantize_int4(
                momentum, self.config.momentum_block_size
            )
        
        self._momentum_int8[name] = quantized
        self._momentum_scales[name] = scales
    
    def _quantize_and_store_variance(self, name: str, variance: torch.Tensor):
        """Quantize variance and store compressed representation."""
        if self.config.variance_bits == 4:
            quantized, scales = self._quantizer.quantize_int4(
                variance, self.config.variance_block_size
            )
        else:
            quantized, scales = self._quantizer.quantize_int8(
                variance, self.config.variance_block_size
            )
        
        self._variance_int4[name] = quantized
        self._variance_scales[name] = scales
    
    def step(self):
        """
        Execute one optimization step using model gradients.
        
        This is a convenience method for use after backward pass.
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        self._perform_step(self._model, {})
    
    def zero_grad(self, set_to_none: bool = True):
        """
        Zero gradients for all parameters.
        
        Args:
            set_to_none: Whether to set gradients to None instead of zero
        """
        if self._model is None:
            return
        
        for param in self._model.parameters():
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory usage information.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            "total_params": self._stats["total_params"],
            "total_elements": self._stats["total_elements"],
            "memory_saved_bytes": self._stats["memory_saved_bytes"],
            "compression_ratio": self.config.get_memory_savings_ratio(),
            "momentum_bits": self.config.momentum_bits,
            "variance_bits": self.config.variance_bits,
            "sparse_ratio": self._sparse_selector.get_effective_ratio() if self._sparse_selector else 1.0,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get optimization statistics.
        
        Returns:
            Dictionary with optimization statistics
        """
        stats = self._stats.copy()
        
        if self._sparse_selector is not None:
            stats["sparse_stats"] = self._sparse_selector.get_statistics()
        
        if self._integrator is not None:
            stats["integrator_stats"] = self._integrator.get_statistics()
        
        return stats
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for serialization.
        
        Returns:
            State dictionary containing all optimizer state
        """
        return {
            "config": self.config.to_dict(),
            "momentum_int8": {k: v.clone() for k, v in self._momentum_int8.items()},
            "momentum_scales": {k: v.clone() for k, v in self._momentum_scales.items()},
            "variance_int4": {k: v.clone() for k, v in self._variance_int4.items()},
            "variance_scales": {k: v.clone() for k, v in self._variance_scales.items()},
            "step": self._step.copy(),
            "stats": self._stats.copy(),
            "sparse_state": self._sparse_selector.state_dict() if self._sparse_selector else {},
            "integrator_state": self._integrator.state_dict() if self._integrator else {},
            "kv_cache_state": self._kv_cache_quantizer.state_dict() if self._kv_cache_quantizer else {},
            "gradient_state": self._gradient_compressor.state_dict() if self._gradient_compressor else {},
            "moe_state": self._moe_manager.state_dict() if self._moe_manager else {},
            "checkpoint_state": self._checkpoint_selector.state_dict() if self._checkpoint_selector else {},
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state from dictionary.
        
        Args:
            state_dict: State dictionary to load
        """
        for key in ["momentum_int8", "momentum_scales", "variance_int4", "variance_scales"]:
            if key in state_dict:
                getattr(self, f"_{key}").update({
                    k: v.clone() for k, v in state_dict[key].items()
                })
        
        self._step = state_dict.get("step", {}).copy()
        self._stats = state_dict.get("stats", {}).copy()
        
        if self._sparse_selector and "sparse_state" in state_dict:
            self._sparse_selector.load_state_dict(state_dict["sparse_state"])
        
        if self._integrator and "integrator_state" in state_dict:
            self._integrator.load_state_dict(state_dict["integrator_state"])
        
        if self._kv_cache_quantizer and "kv_cache_state" in state_dict:
            self._kv_cache_quantizer.load_state_dict(state_dict["kv_cache_state"])
        
        if self._gradient_compressor and "gradient_state" in state_dict:
            self._gradient_compressor.load_state_dict(state_dict["gradient_state"])
        
        if self._moe_manager and "moe_state" in state_dict:
            self._moe_manager.load_state_dict(state_dict["moe_state"])
        
        if self._checkpoint_selector and "checkpoint_state" in state_dict:
            self._checkpoint_selector.load_state_dict(state_dict["checkpoint_state"])
