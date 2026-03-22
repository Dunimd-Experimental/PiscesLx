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
Ink Optimizer - PyTorch-Compatible Wrapper

This module provides a PyTorch-compatible optimizer wrapper for the Ink
optimization system, enabling seamless integration with existing training loops.

Key Features:
    - Drop-in replacement for AdamW
    - Full PyTorch optimizer interface
    - Support for parameter groups
    - Closure pattern support
    - State serialization/deserialization

Usage:
    >>> from opss.optim.ink import POPSSInkOptimizer, POPSSInkConfig
    >>> 
    >>> config = POPSSInkConfig(
    ...     lr=1e-4,
    ...     momentum_bits=8,
    ...     variance_bits=4,
    ...     sparse_ratio=0.01
    ... )
    >>> 
    >>> optimizer = POPSSInkOptimizer(model.parameters(), config)
    >>> 
    >>> for batch in dataloader:
    ...     optimizer.zero_grad()
    ...     loss = model(batch)
    ...     loss.backward()
    ...     optimizer.step()
"""

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from configs.version import VERSION
from .config import POPSSInkConfig
from .operator import POPSSInkOperator
from .quantizer import POPSSInkBlockQuantizer
from .sparse import POPSSInkSparseSelector
from .integrator import POPSSInkIntegrator
from .kv_cache import POPSSInkKVCacheQuantizer
from .gradient import POPSSInkGradientCompressor
from .moe import POPSSInkMoEManager
from .checkpoint import POPSSInkCheckpointSelector


class POPSSInkOptimizer(Optimizer):
    """
    PyTorch-Compatible Ink Optimizer.
    
    This optimizer provides a drop-in replacement for AdamW with
    INT8/INT4 state compression and sparse gradient selection.
    
    The optimizer maintains compressed optimizer states internally
    while providing the standard PyTorch optimizer interface.
    
    Attributes:
        config: POPSSInkConfig instance with all settings
        _operator: Internal POPSSInkOperator instance
        _param_names: Mapping from parameter IDs to names
        _initialized: Whether optimizer has been initialized
    
    Example:
        >>> # Basic usage
        >>> optimizer = POPSSInkOptimizer(
        ...     model.parameters(),
        ...     lr=1e-4,
        ...     momentum_bits=8,
        ...     variance_bits=4
        ... )
        >>> 
        >>> # With configuration object
        >>> config = POPSSInkConfig(
        ...     lr=1e-4,
        ...     momentum_bits=8,
        ...     variance_bits=4,
        ...     sparse_ratio=0.01,
        ...     use_galore=True,
        ...     use_fp4=True,
        ...     use_root_ortho=True
        ... )
        >>> optimizer = POPSSInkOptimizer(model.parameters(), config=config)
        >>> 
        >>> # Training loop
        >>> for epoch in range(epochs):
        ...     for batch in dataloader:
        ...         optimizer.zero_grad()
        ...         loss = model(batch)
        ...         loss.backward()
        ...         optimizer.step()
        >>> 
        >>> # Save/load state
        >>> torch.save(optimizer.state_dict(), "optimizer.pt")
        >>> optimizer.load_state_dict(torch.load("optimizer.pt"))
    """
    
    def __init__(
        self,
        params: Iterable[Union[Dict[str, Any], torch.Tensor]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        config: Optional[POPSSInkConfig] = None,
        momentum_bits: int = 8,
        variance_bits: int = 8,
        momentum_block_size: int = 128,
        variance_block_size: int = 256,
        sparse_ratio: float = 0.01,
        sparse_warmup_steps: int = 1000,
        ortho_momentum: float = 0.9,
        galore_rank: int = 128,
        amsgrad: bool = False,
        maximize: bool = False,
        num_update_blocks: int = 4,
    ):
        """
        Initialize the Ink optimizer.

        Args:
            params: Iterable of parameters or parameter groups
            lr: Learning rate (default: 1e-4)
            betas: Beta coefficients for Adam (default: (0.9, 0.999))
            eps: Epsilon for numerical stability (default: 1e-8)
            weight_decay: Weight decay coefficient (default: 0.01)
            config: Optional POPSSInkConfig instance (overrides other args)
            momentum_bits: Bits for momentum quantization (default: 8)
            variance_bits: Bits for variance quantization (default: 8)
            momentum_block_size: Block size for momentum (default: 128)
            variance_block_size: Block size for variance (default: 256)
            sparse_ratio: Fraction of gradients to keep (default: 0.01)
            sparse_warmup_steps: Warmup steps for sparsity (default: 1000)
            ortho_momentum: Momentum for orthogonal direction tracking (default: 0.9)
            galore_rank: Rank for GaLore projection (default: 128)
            amsgrad: Use AMSGrad variant (default: False)
            maximize: Maximize objective instead of minimize (default: False)
            num_update_blocks: Number of blocks for block-wise updates (default: 4)
        """
        if config is None:
            config = POPSSInkConfig(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                momentum_bits=momentum_bits,
                variance_bits=variance_bits,
                momentum_block_size=momentum_block_size,
                variance_block_size=variance_block_size,
                sparse_ratio=sparse_ratio,
                sparse_warmup_steps=sparse_warmup_steps,
                ortho_momentum=ortho_momentum,
                galore_rank=galore_rank,
                amsgrad=amsgrad,
                maximize=maximize,
            )
        
        defaults = {
            "lr": config.lr,
            "betas": config.betas,
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "amsgrad": config.amsgrad,
            "maximize": config.maximize,
        }
        
        super().__init__(params, defaults)
        
        self.config = config
        
        self._quantizer = POPSSInkBlockQuantizer(
            momentum_bits=config.momentum_bits,
            variance_bits=config.variance_bits,
            momentum_block_size=config.momentum_block_size,
            variance_block_size=config.variance_block_size,
        )
        
        self._sparse_selector = POPSSInkSparseSelector(
            sparse_ratio=config.sparse_ratio,
            warmup_steps=config.sparse_warmup_steps,
            adaptive=config.sparse_adaptive,
        )
        
        self._integrator = POPSSInkIntegrator(config)
        self._integrator.initialize()
        
        self._kv_cache_quantizer = POPSSInkKVCacheQuantizer(
            kv_bits=config.kv_cache_bits,
            block_size=config.kv_cache_block_size,
        )
        
        self._gradient_compressor = POPSSInkGradientCompressor(
            gradient_bits=config.gradient_bits,
            block_size=config.gradient_block_size,
            stochastic_rounding=True,
        )
        
        self._moe_manager = POPSSInkMoEManager(
            num_experts=64,
            max_experts_on_gpu=config.max_experts_on_gpu,
            offload_threshold=config.moe_offload_threshold,
            lru_cache_size=config.moe_lru_cache_size,
        )
        
        self._checkpoint_selector = POPSSInkCheckpointSelector(
            checkpoint_ratio=config.checkpoint_ratio,
            preserve_ratio=config.checkpoint_preserve_ratio,
            enable_transformer=config.checkpoint_transformer,
        )
        
        self._num_blocks = getattr(config, 'num_update_blocks', 4)
        self._current_block_idx = 0
        
        self._param_names: Dict[int, str] = {}
        self._param_shapes: Dict[str, Tuple[int, ...]] = {}
        self._step: Dict[str, int] = {}
        
        self._momentum_int8: Dict[str, torch.Tensor] = {}
        self._momentum_scales: Dict[str, torch.Tensor] = {}
        self._variance_int4: Dict[str, torch.Tensor] = {}
        self._variance_scales: Dict[str, torch.Tensor] = {}
        
        self._initialized = False
        self._global_step = 0
    
    def _lazy_init(self):
        """
        Lazy initialization of optimizer states.
        
        Called on first step() to ensure all parameters are available.
        """
        if self._initialized:
            return
        
        param_idx = 0
        for group in self.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                
                name = f"param_{param_idx}"
                self._param_names[id(param)] = name
                self._param_shapes[name] = param.shape
                self._step[name] = 0
                
                numel = param.numel()
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
                
                param_idx += 1
        
        self._initialized = True
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure that reevaluates the model and returns loss
        
        Returns:
            Optional loss value from closure
        """
        self._lazy_init()
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            maximize = group.get("maximize", False)
            
            params_list = [p for p in group["params"] if p.grad is not None]
            
            block_size = max(1, len(params_list) // self._num_blocks)
            
            start_idx = self._current_block_idx * block_size
            end_idx = min(start_idx + block_size, len(params_list))
            
            block_params = params_list[start_idx:end_idx]
            
            for param in params_list:
                if param.grad is None:
                    continue
                
                grad = param.grad
                if grad.is_sparse:
                    grad = grad.to_dense()
                
                name = self._param_names.get(id(param))
                if name is None:
                    continue
                
                if maximize:
                    grad = -grad
                
                if param in block_params:
                    sparse_grad, mask = self._sparse_selector.select(grad, name)

                    galore_state = self._integrator.get_galore_state(name)
                    sparse_grad = self._integrator.apply_galore(sparse_grad, name, galore_state)

                    self._step[name] = self._step.get(name, 0) + 1
                    step = self._step[name]

                    momentum = self._dequantize_momentum(name, param.shape)
                    variance = self._dequantize_variance(name, param.shape)

                    momentum = beta1 * momentum + (1 - beta1) * sparse_grad
                    variance = beta2 * variance + (1 - beta2) * sparse_grad ** 2

                    if self._integrator.should_orthogonalize(param):
                        momentum = self._integrator.apply_root_ortho(momentum, name)

                    momentum = self._integrator.apply_root_spectral_clip(momentum, name)

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    update = momentum / (variance.sqrt() / math.sqrt(bias_correction2) + eps)
                    update = update * (math.sqrt(bias_correction2) / bias_correction1)

                    if weight_decay > 0:
                        param.add_(param, alpha=-lr * weight_decay)

                    param.add_(update, alpha=-lr)

                    self._quantize_and_store_momentum(name, momentum)
                    self._quantize_and_store_variance(name, variance)
                else:
                    pass
        
        self._current_block_idx = (self._current_block_idx + 1) % self._num_blocks
        
        self._global_step += 1
        
        return loss
    
    def _dequantize_momentum(self, name: str, shape: Tuple[int, ...]) -> torch.Tensor:
        """Dequantize momentum from compressed representation."""
        if self.config.momentum_bits == 8:
            return self._quantizer.dequantize_int8(
                self._momentum_int8[name],
                self._momentum_scales[name],
                shape,
            )
        else:
            return self._quantizer.dequantize_int4(
                self._momentum_int8[name],
                self._momentum_scales[name],
                shape,
            )
    
    def _dequantize_variance(self, name: str, shape: Tuple[int, ...]) -> torch.Tensor:
        """Dequantize variance from compressed representation."""
        if self.config.variance_bits == 4:
            return self._quantizer.dequantize_int4(
                self._variance_int4[name],
                self._variance_scales[name],
                shape,
            )
        else:
            return self._quantizer.dequantize_int8(
                self._variance_int4[name],
                self._variance_scales[name],
                shape,
            )
    
    def _quantize_and_store_momentum(self, name: str, momentum: torch.Tensor):
        """Quantize and store momentum."""
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
        """Quantize and store variance."""
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
    
    def get_compression_ratio(self) -> float:
        """
        Get the current compression ratio.
        
        Returns:
            Ratio of memory saved compared to standard Adam
        """
        return self.config.get_memory_savings_ratio()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        total_elements = sum(
            shape.numel() if hasattr(shape, 'numel') else 1
            for shape in self._param_shapes.values()
        )
        
        original_memory = total_elements * 4 * 2
        
        momentum_memory = total_elements * (self.config.momentum_bits / 8)
        variance_memory = total_elements * (self.config.variance_bits / 8)
        compressed_memory = momentum_memory + variance_memory
        
        return {
            "total_parameters": len(self._param_names),
            "total_elements": total_elements,
            "original_memory_bytes": original_memory,
            "compressed_memory_bytes": int(compressed_memory),
            "compression_ratio": original_memory / compressed_memory if compressed_memory > 0 else 1.0,
            "momentum_bits": self.config.momentum_bits,
            "variance_bits": self.config.variance_bits,
        }
    
    def get_sparsity_stats(self) -> Dict[str, Any]:
        """
        Get gradient sparsity statistics.
        
        Returns:
            Dictionary with sparsity statistics
        """
        return self._sparse_selector.get_statistics()
    
    def get_component_stats(self) -> Dict[str, Any]:
        """
        Get component integration statistics.
        
        Returns:
            Dictionary with component statistics
        """
        return self._integrator.get_statistics()
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for serialization.
        
        Returns:
            State dictionary containing all optimizer state
        """
        state_dict = super().state_dict()
        
        state_dict["ink_config"] = self.config.to_dict()
        state_dict["momentum_int8"] = {k: v.clone() for k, v in self._momentum_int8.items()}
        state_dict["momentum_scales"] = {k: v.clone() for k, v in self._momentum_scales.items()}
        state_dict["variance_int4"] = {k: v.clone() for k, v in self._variance_int4.items()}
        state_dict["variance_scales"] = {k: v.clone() for k, v in self._variance_scales.items()}
        state_dict["step"] = self._step.copy()
        state_dict["global_step"] = self._global_step
        state_dict["param_names"] = self._param_names.copy()
        state_dict["param_shapes"] = self._param_shapes.copy()
        state_dict["sparse_state"] = self._sparse_selector.state_dict()
        state_dict["integrator_state"] = self._integrator.state_dict()
        state_dict["kv_cache_state"] = self._kv_cache_quantizer.state_dict()
        state_dict["gradient_state"] = self._gradient_compressor.state_dict()
        state_dict["moe_state"] = self._moe_manager.state_dict()
        state_dict["checkpoint_state"] = self._checkpoint_selector.state_dict()
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state from dictionary.
        
        Args:
            state_dict: State dictionary to load
        """
        super().load_state_dict(state_dict)
        
        if "ink_config" in state_dict:
            config_dict = state_dict["ink_config"]
            self.config = POPSSInkConfig(**config_dict)
        
        for key in ["momentum_int8", "momentum_scales", "variance_int4", "variance_scales"]:
            if key in state_dict:
                getattr(self, f"_{key}").update({
                    k: v.clone() for k, v in state_dict[key].items()
                })
        
        self._step = state_dict.get("step", {}).copy()
        self._global_step = state_dict.get("global_step", 0)
        self._param_names = state_dict.get("param_names", {}).copy()
        self._param_shapes = state_dict.get("param_shapes", {}).copy()
        
        if "sparse_state" in state_dict:
            self._sparse_selector.load_state_dict(state_dict["sparse_state"])
        
        if "integrator_state" in state_dict:
            self._integrator.load_state_dict(state_dict["integrator_state"])
        
        if "kv_cache_state" in state_dict:
            self._kv_cache_quantizer.load_state_dict(state_dict["kv_cache_state"])
        
        if "gradient_state" in state_dict:
            self._gradient_compressor.load_state_dict(state_dict["gradient_state"])
        
        if "moe_state" in state_dict:
            self._moe_manager.load_state_dict(state_dict["moe_state"])
        
        if "checkpoint_state" in state_dict:
            self._checkpoint_selector.load_state_dict(state_dict["checkpoint_state"])
        
        self._initialized = True
    
    def reset_sparsity(self):
        """Reset sparse selector state."""
        self._sparse_selector.reset_step()
        self._sparse_selector.reset_history()
    
    def set_sparse_ratio(self, ratio: float):
        """
        Set the sparse gradient ratio.
        
        Args:
            ratio: New sparse ratio (0 < ratio <= 1)
        """
        if not 0 < ratio <= 1:
            raise ValueError(f"sparse_ratio must be in (0, 1], got {ratio}")
        self._sparse_selector.sparse_ratio = ratio
        self.config.sparse_ratio = ratio
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lr={self.config.lr}, "
            f"momentum_bits={self.config.momentum_bits}, "
            f"variance_bits={self.config.variance_bits}, "
            f"sparse_ratio={self.config.sparse_ratio}, "
            f"ortho_momentum={self.config.ortho_momentum}, "
            f"galore_rank={self.config.galore_rank}, "
            f"num_blocks={self._num_blocks})"
        )
