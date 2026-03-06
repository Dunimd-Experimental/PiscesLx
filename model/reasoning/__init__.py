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

"""Yv Reasoning Module - Comprehensive reasoning capabilities for Yv model.

This module provides advanced reasoning components for the Yv model,
including chain-of-thought reasoning, multi-path reasoning, recursive depth
reasoning, and multimodal reasoning enhancement.

Module Components:
    1. Core Reasoners:
       - YvUnifiedReasoner: Unified reasoning interface
       - YvCoTMemoryReasoner: Chain-of-thought with memory
       - YvRecursiveDepthReasoner: Recursive depth-limited reasoning

    2. Multi-Path Reasoning:
       - YvMultiPathReasoningEngine: Core multi-path engine
       - YvMultiPathInferenceEngine: Inference optimization
       - YvMultiPathMetaLearner: Meta-learning for path selection
       - YvUnifiedMultiPathReasoningSystem: Unified multi-path system

    3. Enhanced Reasoning:
       - YvEnhancedReasoningSystem: Enhanced reasoning with verification
       - YvMultiModalReasoningEnhancer: Multimodal reasoning enhancement
       - YvThoughtTreeReasoner: Tree-based thought exploration

    4. Configuration:
       - YvRecursiveReasoningConfig: Recursive reasoning settings
       - YvEnhancedReasoningConfig: Enhanced reasoning settings

Key Features:
    - Chain-of-thought (CoT) reasoning with memory persistence
    - Multi-path reasoning with path scoring and selection
    - Recursive depth-limited reasoning for complex problems
    - Tree-based thought exploration with backtracking
    - Multimodal reasoning enhancement for cross-modal tasks
    - Meta-learning for adaptive path selection
    - Fact verification and consistency checking

Performance Characteristics:
    - CoT Reasoning: O(T * L) where T = thought steps, L = sequence length
    - Multi-Path: O(P * T * L) where P = number of paths
    - Recursive: O(D^B) where D = depth, B = branching factor
    - Tree Search: O(N * log N) with pruning

Usage Example:
    >>> from model.reasoning import YvUnifiedReasoner
    >>> 
    >>> # Initialize reasoner
    >>> reasoner = YvUnifiedReasoner(config)
    >>> 
    >>> # Perform reasoning
    >>> result = reasoner.reason(
    ...     query="What is the capital of France?",
    ...     context=context_embeddings
    >>> )
    >>> 
    >>> # Multi-path reasoning
    >>> from model.reasoning import YvMultiPathReasoningEngine
    >>> engine = YvMultiPathReasoningEngine(config)
    >>> paths = engine.explore_paths(query, num_paths=5)

Note:
    All reasoning functionality is implemented in the reasoner/ subdirectory.
    Supports both synchronous and asynchronous reasoning modes.
    Integrates with YvModel for end-to-end reasoning pipelines.
"""

from .reasoner import (
    YvUnifiedReasoner,
    YvCoTMemoryReasoner,
    YvMultiPathReasoningEngine,
    YvMultiPathMetaLearner,
    YvMultiPathInferenceEngine,
    YvUnifiedMultiPathReasoningSystem,
    YvRecursiveDepthReasoner,
    YvRecursiveReasoningConfig,
    YvThoughtTreeReasoner,
    YvEnhancedReasoningSystem,
    YvEnhancedReasoningConfig,
    YvMultiModalReasoningEnhancer,
)

__all__ = [
    "YvUnifiedReasoner",
    "YvCoTMemoryReasoner",
    "YvMultiPathReasoningEngine",
    "YvMultiPathMetaLearner",
    "YvMultiPathInferenceEngine",
    "YvUnifiedMultiPathReasoningSystem",
    "YvRecursiveDepthReasoner",
    "YvRecursiveReasoningConfig",
    "YvThoughtTreeReasoner",
    "YvEnhancedReasoningSystem",
    "YvEnhancedReasoningConfig",
    "YvMultiModalReasoningEnhancer",
]
