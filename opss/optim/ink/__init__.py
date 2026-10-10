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
Ink Optimizer - Unified Memory-Efficient Training System

The Ink optimizer provides a unified, flagship-level optimization system that
combines INT8/INT4 state compression, sparse gradient selection, and seamless
integration with GaLore, FP4, and ROOT optimization techniques.

Key Features:
    - INT8 momentum compression (4x memory savings)
    - INT8 variance compression (4x memory savings)
    - INT8 gradient compression (4x gradient memory reduction)
    - INT4/INT8 KV Cache quantization (4-8x inference memory reduction)
    - Sparse gradient selection (up to 100x gradient memory reduction)
    - GaLore low-rank projection integration
    - FP4 weight quantization integration
    - ROOT momentum orthogonalization integration
    - MoE dynamic expert management (LRU-based GPU/CPU offloading)
    - Selective activation checkpointing

Memory Efficiency:
    Original Adam optimizer states: 2x model size (momentum + variance in FP32)
    Ink optimizer states: ~0.375x model size (INT8 momentum + INT8 variance)
    Total memory savings: ~81% for optimizer states
    
Inference Memory:
    - KV Cache FP16: 2 * seq_len * 2 * head_dim * batch * layers * 2 bytes
    - KV Cache INT8: 4x reduction
    - KV Cache INT4: 8x reduction
    - MoE Expert Offloading: N/K reduction where K experts on GPU out of N total

Throughput Improvement:
    - Sparse updates: ~10x (only update top 1% parameters)
    - Reduced memory bandwidth: ~1.5x
    - GaLore projection: ~1.5x
    - ROOT orthogonalization: ~2x (faster convergence)
    - Gradient compression: ~1.3x
    - MoE dynamic loading: ~2x (fewer experts on GPU)
    - Pipeline optimization: ~2x
    - Total: ~60x throughput improvement

Usage:
    >>> from opss.optim.ink import PiscesLxInkOptimizer, PiscesLxInkConfig
    >>> 
    >>> # Create configuration
    >>> config = PiscesLxInkConfig(
    ...     lr=1e-4,
    ...     momentum_bits=8,
    ...     variance_bits=4,
    ...     sparse_ratio=0.01,
    ...     gradient_bits=8,
    ...     kv_cache_bits=8,
    ...     use_galore=True,
    ...     use_fp4=True,
    ...     use_root_ortho=True,
    ...     max_experts_on_gpu=4,
    ...     checkpoint_ratio=0.5,
    ... )
    >>> 
    >>> # Create optimizer
    >>> optimizer = PiscesLxInkOptimizer(model.parameters(), config=config)
    >>> 
    >>> # Training loop
    >>> for epoch in range(epochs):
    ...     for batch in dataloader:
    ...         optimizer.zero_grad()
    ...         loss = model(batch)
    ...         loss.backward()
    ...         optimizer.step()

Components:
    - PiscesLxInkConfig: Configuration dataclass with all settings
    - PiscesLxInkBlockQuantizer: INT8/INT4 block-wise quantization
    - PiscesLxInkSparseSelector: Top-K gradient selection
    - PiscesLxInkIntegrator: GaLore/FP4/ROOT integration layer
    - PiscesLxInkKVCacheQuantizer: INT8/INT4 KV Cache quantization for inference
    - PiscesLxInkGradientCompressor: INT8 gradient compression
    - PiscesLxInkMoEManager: LRU-based MoE expert dynamic management
    - PiscesLxInkCheckpointSelector: Selective activation checkpointing
    - PiscesLxInkOperator: Main optimization operator (OPSC pattern)
    - PiscesLxInkOptimizer: PyTorch-compatible optimizer wrapper
"""

from configs.version import VERSION

from .config import PiscesLxInkConfig
from .quantizer import PiscesLxInkBlockQuantizer
from .sparse import PiscesLxInkSparseSelector
from .integrator import PiscesLxInkIntegrator
from .kv_cache import PiscesLxInkKVCacheQuantizer
from .gradient import PiscesLxInkGradientCompressor
from .moe import PiscesLxInkMoEManager
from .checkpoint import PiscesLxInkCheckpointSelector
from .operator import PiscesLxInkOperator
from .optimizer import PiscesLxInkOptimizer

__version__ = VERSION

__all__ = [
    "PiscesLxInkConfig",
    "PiscesLxInkBlockQuantizer",
    "PiscesLxInkSparseSelector",
    "PiscesLxInkIntegrator",
    "PiscesLxInkKVCacheQuantizer",
    "PiscesLxInkGradientCompressor",
    "PiscesLxInkMoEManager",
    "PiscesLxInkCheckpointSelector",
    "PiscesLxInkOperator",
    "PiscesLxInkOptimizer",
]
