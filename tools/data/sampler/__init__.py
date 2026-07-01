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
PiscesLx Intelligent Sampling Module.

This module provides advanced sampling strategies for efficient training:
- DynamicBatchSampler: Token-based batching to minimize padding waste
- CurriculumSampler: Progressive difficulty training for faster convergence

Key Features:
    - 30%+ reduction in padding tokens (DynamicBatchSampler)
    - Faster model convergence (CurriculumSampler)
    - Full PyTorch DataLoader compatibility
    - Distributed training support

Usage:
    >>> from tools.data.sampler import PiscesLxDynamicBatchSampler, PiscesLxCurriculumSampler
    >>> 
    >>> # Dynamic batching
    >>> sampler = PiscesLxDynamicBatchSampler(dataset, max_tokens=8192)
    >>> loader = DataLoader(dataset, batch_sampler=sampler)
    >>> 
    >>> # Curriculum learning
    >>> sampler = PiscesLxCurriculumSampler(dataset, difficulty_scores=scores)
    >>> loader = DataLoader(dataset, sampler=sampler)
"""

from .dynamic import PiscesLxDynamicBatchSampler
from .curriculum import PiscesLxCurriculumSampler

__all__ = [
    'PiscesLxDynamicBatchSampler',
    'PiscesLxCurriculumSampler',
]
