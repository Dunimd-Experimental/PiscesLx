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
    >>> from opss.optim.ink import PiscesLxInkOptimizer, PiscesLxInkConfig
    >>> 
    >>> config = PiscesLxInkConfig(
    ...     lr=1e-4,
    ...     momentum_bits=8,
    ...     variance_bits=8,
    ...     sparse_ratio=0.01
    ... )
    >>> 
    >>> optimizer = PiscesLxInkOptimizer(model.parameters(), config)
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
from .config import PiscesLxInkConfig
from .operator import PiscesLxInkOperator
from .quantizer import PiscesLxInkBlockQuantizer
from .sparse import PiscesLxInkSparseSelector
from .integrator import PiscesLxInkIntegrator
from .kv_cache import PiscesLxInkKVCacheQuantizer
from .gradient import PiscesLxInkGradientCompressor
from .moe import PiscesLxInkMoEManager
from .checkpoint import PiscesLxInkCheckpointSelector

#: Once the estimators have caught up with a stationary gradient, Adam's
#: (m̂)/(sqrt(v̂) + eps) has magnitude ~1 per coordinate -- it behaves like sign(g).
#: A ratio above this bound means the coordinate wants to move further than one
#: learning rate in a single step, which is what a collapsed denominator produces.
#: Clamping there is the trust region of docs/proofs/02 Theorem 2.
MAX_COORDINATE_STEP = 1.0


class PiscesLxInkOptimizer(Optimizer):
    """
    PyTorch-Compatible Ink Optimizer.
    
    This optimizer provides a drop-in replacement for AdamW with
    INT8 state compression (variance on a per-block log grid) and sparse
    gradient selection.
    
    The optimizer maintains compressed optimizer states internally
    while providing the standard PyTorch optimizer interface.
    
    Attributes:
        config: PiscesLxInkConfig instance with all settings
        _operator: Internal PiscesLxInkOperator instance
        _param_names: Mapping from parameter IDs to names
        _initialized: Whether optimizer has been initialized
    
    Example:
        >>> # Basic usage
        >>> optimizer = PiscesLxInkOptimizer(
        ...     model.parameters(),
        ...     lr=1e-4,
        ...     momentum_bits=8,
        ...     variance_bits=8
        ... )
        >>> 
        >>> # With configuration object
        >>> config = PiscesLxInkConfig(
        ...     lr=1e-4,
        ...     momentum_bits=8,
        ...     variance_bits=8,
        ...     sparse_ratio=0.01,
        ...     use_galore=True,
        ...     use_fp4=True,
        ...     use_root_ortho=True
        ... )
        >>> optimizer = PiscesLxInkOptimizer(model.parameters(), config=config)
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
        config: Optional[PiscesLxInkConfig] = None,
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
        num_update_blocks: int = 1,
    ):
        """
        Initialize the Ink optimizer.

        Args:
            params: Iterable of parameters or parameter groups
            lr: Learning rate (default: 1e-4)
            betas: Beta coefficients for Adam (default: (0.9, 0.999))
            eps: Epsilon for numerical stability (default: 1e-8)
            weight_decay: Weight decay coefficient (default: 0.01)
            config: Optional PiscesLxInkConfig instance (overrides other args)
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
            num_update_blocks: Parameters per step are 1/num_update_blocks (default: 1, all)
        """
        if config is None:
            config = PiscesLxInkConfig(
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
        
        self._quantizer = PiscesLxInkBlockQuantizer(
            momentum_bits=config.momentum_bits,
            variance_bits=config.variance_bits,
            momentum_block_size=config.momentum_block_size,
            variance_block_size=config.variance_block_size,
        )
        
        self._sparse_selector = PiscesLxInkSparseSelector(
            sparse_ratio=config.sparse_ratio,
            warmup_steps=config.sparse_warmup_steps,
            adaptive=config.sparse_adaptive,
        )
        
        self._integrator = PiscesLxInkIntegrator(config)
        self._integrator.initialize()
        
        self._kv_cache_quantizer = PiscesLxInkKVCacheQuantizer(
            kv_bits=config.kv_cache_bits,
            block_size=config.kv_cache_block_size,
        )
        
        self._gradient_compressor = PiscesLxInkGradientCompressor(
            gradient_bits=config.gradient_bits,
            block_size=config.gradient_block_size,
            stochastic_rounding=True,
        )
        
        self._moe_manager = PiscesLxInkMoEManager(
            num_experts=64,
            max_experts_on_gpu=config.max_experts_on_gpu,
            offload_threshold=config.moe_offload_threshold,
            lru_cache_size=config.moe_lru_cache_size,
        )
        
        self._checkpoint_selector = PiscesLxInkCheckpointSelector(
            checkpoint_ratio=config.checkpoint_ratio,
            preserve_ratio=config.checkpoint_preserve_ratio,
            enable_transformer=config.checkpoint_transformer,
        )
        
        self._num_blocks = max(1, int(getattr(config, 'num_update_blocks', 1)))
        self._current_block_idx = 0
        
        self._param_names: Dict[int, str] = {}
        self._param_shapes: Dict[str, Tuple[int, ...]] = {}
        self._step: Dict[str, int] = {}
        
        self._momentum_codes: Dict[str, torch.Tensor] = {}
        self._momentum_scales: Dict[str, torch.Tensor] = {}
        self._variance_codes: Dict[str, torch.Tensor] = {}
        self._variance_log_lo: Dict[str, torch.Tensor] = {}
        self._variance_log_step: Dict[str, torch.Tensor] = {}
        
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
                    self._momentum_codes[name] = torch.zeros(
                        numel, dtype=torch.int8, device=param.device
                    )
                else:
                    self._momentum_codes[name] = torch.zeros(
                        (numel + 1) // 2, dtype=torch.int8, device=param.device
                    )
                
                self._momentum_scales[name] = torch.ones(
                    momentum_blocks, dtype=torch.float32, device=param.device
                )
                
                variance_code_elements = (numel + 1) // 2 if self.config.variance_bits == 4 else numel
                self._variance_codes[name] = torch.zeros(
                    variance_code_elements, dtype=torch.uint8, device=param.device
                )
                self._variance_log_lo[name] = torch.zeros(
                    variance_blocks, dtype=torch.float32, device=param.device
                )
                self._variance_log_step[name] = torch.zeros(
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
            block_param_ids = {id(p) for p in block_params}

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

                if id(param) in block_param_ids:
                    sparse_grad, _selected_mask = self._sparse_selector.select(grad, name)
                    sparse_grad = sparse_grad.reshape(-1)

                    galore_state = self._integrator.get_galore_state(name)
                    sparse_grad = self._integrator.apply_galore(sparse_grad, name, galore_state)

                    self._step[name] = self._step.get(name, 0) + 1
                    step = self._step[name]

                    momentum = self._dequantize_momentum(name, param.shape).reshape(-1)
                    variance = self._dequantize_variance(name, param.numel())

                    momentum = beta1 * momentum + (1 - beta1) * sparse_grad
                    variance = beta2 * variance + (1 - beta2) * sparse_grad ** 2

                    if self._integrator.should_orthogonalize(param):
                        momentum = self._integrator.apply_root_ortho(momentum, name)

                    momentum = self._integrator.apply_root_spectral_clip(momentum, name)

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    # Adam's bias-corrected estimators are m̂ = m/bc1 and v̂ = v/bc2, so the
                    # step is (m/bc1)/(sqrt(v/bc2) + eps) (Kingma & Ba 2015, sec. 3). The
                    # previous expression divided v by sqrt(bc2) *inside* the denominator and
                    # then multiplied the whole update by sqrt(bc2)/bc1, which leaves an extra
                    # factor bc2 in the numerator: early steps were ~1/bc2 times too small
                    # (0.001× at t=1, 0.1× at t=100 with beta2=0.999), so the optimiser ran
                    # while the parameters barely moved.
                    denominator = (variance / bias_correction2).sqrt() + eps
                    update = (momentum / bias_correction1) / denominator
                    # Trust region. The log-grid variance codec (docs/proofs/02) keeps every
                    # positive entry strictly positive, but a coordinate whose momentum has
                    # not yet caught up with a sudden large gradient can still make
                    # |m̂|/sqrt(v̂) exceed 1, which is a step larger than Adam's own sign step.
                    # Capping the ratio at 1 bounds the per-coordinate displacement at lr, so
                    # no quantisation artefact can move a weight further than the learning
                    # rate intends (Theorem 2, docs/proofs/02).
                    update = update.clamp(-MAX_COORDINATE_STEP, MAX_COORDINATE_STEP)
                    # The update must carry the parameter's own geometry: a 0-dim
                    # parameter (e.g. a learned scalar gate) has a one-element flat
                    # block representation, and the block-wise dequantisation path
                    # returns that as shape [1]. add_() would then refuse to broadcast.
                    update = update.view(param.shape)

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
                self._momentum_codes[name],
                self._momentum_scales[name],
                shape,
                block_size=self.config.momentum_block_size,
            )
        else:
            return self._quantizer.dequantize_int4(
                self._momentum_codes[name],
                self._momentum_scales[name],
                shape,
                block_size=self.config.momentum_block_size,
            )
    
    def _dequantize_variance(self, name: str, num_elements: int) -> torch.Tensor:
        """Dequantize variance from its per-block log grid (flat representation)."""
        return self._quantizer.dequantize_variance_log(
            self._variance_codes[name],
            self._variance_log_lo[name],
            self._variance_log_step[name],
            num_elements,
            block_size=self.config.variance_block_size,
            bits=self.config.variance_bits,
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
        
        self._momentum_codes[name] = quantized
        self._momentum_scales[name] = scales
    
    def _quantize_and_store_variance(self, name: str, variance: torch.Tensor):
        """Quantize and store variance on its per-block log grid."""
        codes, log_lo, log_step = self._quantizer.quantize_variance_log(
            variance, self.config.variance_block_size, self.config.variance_bits
        )
        
        self._variance_codes[name] = codes
        self._variance_log_lo[name] = log_lo
        self._variance_log_step[name] = log_step
    
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
        state_dict["momentum_codes"] = {k: v.clone() for k, v in self._momentum_codes.items()}
        state_dict["momentum_scales"] = {k: v.clone() for k, v in self._momentum_scales.items()}
        state_dict["variance_codes"] = {k: v.clone() for k, v in self._variance_codes.items()}
        state_dict["variance_log_lo"] = {k: v.clone() for k, v in self._variance_log_lo.items()}
        state_dict["variance_log_step"] = {k: v.clone() for k, v in self._variance_log_step.items()}
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
            self.config = PiscesLxInkConfig(**config_dict)
        
        state_keys = ("momentum_codes", "momentum_scales",
                      "variance_codes", "variance_log_lo", "variance_log_step")
        for key in state_keys:
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
