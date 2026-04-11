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
PiscesL1 Data Augmentation Module.

This module provides data augmentation strategies for training data:
- TextAugmenter: EDA, synonym replacement, back-translation
- ImageAugmenter: Albumentations-based visual augmentations

Key Features:
    - Multiple text augmentation strategies (EDA)
    - Image augmentations with Albumentations fallback
    - Configurable augmentation probability
    - Batch processing support

Usage:
    >>> from tools.data.augment import PiscesLxDataTextAugmenter, PiscesLxDataImageAugmenter
    >>> 
    >>> # Text augmentation
    >>> text_aug = PiscesLxDataTextAugmenter(strategies=['synonym', 'delete'])
    >>> augmented_texts = text_aug.augment("Hello world", num_augmentations=3)
    >>> 
    >>> # Image augmentation
    >>> image_aug = PiscesLxDataImageAugmenter(aug_prob=0.5)
    >>> augmented_image = image_aug.augment(image_array)
"""

from .text_aug import PiscesLxDataTextAugmenter
from .image_aug import PiscesLxDataImageAugmenter

__all__ = [
    'PiscesLxDataTextAugmenter',
    'PiscesLxDataImageAugmenter',
]
