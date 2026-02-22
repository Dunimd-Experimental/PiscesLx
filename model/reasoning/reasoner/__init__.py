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

"""Yv Reasoner Submodule - Core reasoning implementations.

This submodule provides the core reasoning implementations for the Yv
model, including unified reasoning, chain-of-thought, multi-path reasoning,
and enhanced reasoning systems.

Exported Components:
    Core Reasoners:
        - YvUnifiedReasoner: Unified interface for all reasoning types
        - YvCoTMemoryReasoner: Chain-of-thought with persistent memory
    
    Multi-Path Reasoning:
        - YvMultiPathReasoningEngine: Core multi-path exploration
        - YvMultiPathInferenceEngine: Optimized inference engine
        - YvMultiPathMetaLearner: Meta-learning for path selection
        - YvUnifiedMultiPathReasoningSystem: Complete multi-path system
    
    Enhanced Reasoning:
        - YvEnhancedReasoningSystem: Enhanced reasoning with verification
        - YvMultiModalReasoningEnhancer: Cross-modal reasoning enhancement
        - YvRecursiveDepthReasoner: Depth-limited recursive reasoning
        - YvThoughtTreeReasoner: Tree-based thought exploration
    
    Configuration:
        - YvRecursiveReasoningConfig: Recursive reasoning parameters
        - YvEnhancedReasoningConfig: Enhanced reasoning parameters
"""

from .unified import YvUnifiedReasoner
from .cot_memory import YvCoTMemoryReasoner
from .multipath_meta import YvMultiPathMetaLearner
from .enhancer import YvMultiModalReasoningEnhancer
from .recursive_depth import YvRecursiveDepthReasoner, YvRecursiveReasoningConfig, YvThoughtTreeReasoner
from .multipath_core import YvMultiPathReasoningEngine
from .multipath_infer import YvMultiPathInferenceEngine
from .multipath_system import YvUnifiedMultiPathReasoningSystem
from .enhanced_system import YvEnhancedReasoningSystem, YvEnhancedReasoningConfig

__all__ = [
    "YvUnifiedReasoner",
    "YvCoTMemoryReasoner",
    "YvMultiModalReasoningEnhancer",
    "YvMultiPathReasoningEngine",
    "YvMultiPathInferenceEngine",
    "YvMultiPathMetaLearner",
    "YvUnifiedMultiPathReasoningSystem",
    "YvRecursiveDepthReasoner",
    "YvRecursiveReasoningConfig",
    "YvThoughtTreeReasoner",
    "YvEnhancedReasoningSystem",
    "YvEnhancedReasoningConfig",
]
