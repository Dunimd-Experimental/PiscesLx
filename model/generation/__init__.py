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

"""
Yv Generation Module

This module provides comprehensive generation utilities for the Yv model,
including speculative decoding, sampling strategies, and beam search.

Module Components:
    1. Speculative Decoding (speculative.py):
       - YvAdaptiveSpeculativeDecoder: Main speculative decoding implementation
       - YvSpeculativeConfig: Configuration for speculative decoding parameters
       - YvDraftModel: Lightweight draft model for fast token generation
       - YvVerificationResult: Result container for draft verification
    
    2. Sampling Strategies (sampler.py):
       - YvSampler: Unified sampler supporting multiple strategies
       - YvSamplingConfig: Configuration for sampling parameters
       - YvSamplingStrategy: Enumeration of available sampling strategies

Key Features:
    - Speculative Decoding: Draft-then-verify paradigm for 2-3x speedup
    - Multiple Sampling Strategies: Greedy, top-k, top-p, typical, eta sampling
    - Adaptive Parameters: Dynamic adjustment based on acceptance rates
    - Medusa/EAGLE Support: Multi-head and feature-based speculation

Performance Characteristics:
    - Speculative decoding: 2-3x speedup with high acceptance rates
    - Parallel verification: Single forward pass for all draft tokens
    - Adaptive adjustment: Optimizes draft length based on history

Usage Example:
    >>> from model.generation import YvSampler, YvSamplingConfig
    >>> from model.generation import YvAdaptiveSpeculativeDecoder
    >>> 
    >>> # Standard sampling
    >>> config = YvSamplingConfig(temperature=0.7, top_k=50, top_p=0.9)
    >>> sampler = YvSampler(config)
    >>> next_token = sampler.sample(logits)
    >>> 
    >>> # Speculative decoding
    >>> spec_config = YvSpeculativeConfig(draft_length=5)
    >>> decoder = YvAdaptiveSpeculativeDecoder(spec_config, model)
    >>> generated, stats = decoder.speculative_generate(input_ids)

Note:
    All classes follow the YvXxx naming convention.
    For best performance, use speculative decoding with batch_size=1.
"""

from .speculative import (
    YvAdaptiveSpeculativeDecoder,
    YvSpeculativeConfig,
    YvDraftModel,
    YvVerificationResult,
)
from .sampler import (
    YvSampler,
    YvSamplingConfig,
    YvSamplingStrategy,
)

__all__ = [
    "YvAdaptiveSpeculativeDecoder",
    "YvSpeculativeConfig",
    "YvDraftModel",
    "YvVerificationResult",
    "YvSampler",
    "YvSamplingConfig",
    "YvSamplingStrategy",
]
