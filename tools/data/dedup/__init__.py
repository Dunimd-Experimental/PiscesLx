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
PiscesL1 Data Deduplication Module.

This module provides deduplication strategies for training data:
- MinHashDeduplicator: Approximate near-duplicate detection using LSH
- ExactDeduplicator: Exact duplicate detection using cryptographic hashes

Key Features:
    - Efficient near-duplicate detection (MinHash LSH)
    - Exact duplicate removal (MD5/SHA256)
    - Streaming/incremental processing
    - Memory-efficient for large datasets

Usage:
    >>> from tools.data.dedup import PiscesLxDataMinHashDeduplicator, PiscesLxDataExactDeduplicator
    >>> 
    >>> # Exact deduplication first
    >>> exact_dedup = PiscesLxDataExactDeduplicator(algorithm='sha256')
    >>> for idx, text in enumerate(texts):
    ...     exact_dedup.add(idx, text)
    >>> unique_indices = exact_dedup.get_unique_indices()
    >>> 
    >>> # Then approximate deduplication
    >>> minhash_dedup = PiscesLxDataMinHashDeduplicator(threshold=0.8)
    >>> for idx in unique_indices:
    ...     minhash_dedup.add(idx, texts[idx])
    >>> final_unique = minhash_dedup.get_unique_indices()
"""

from .minhash import PiscesLxDataMinHashDeduplicator
from .exact import PiscesLxDataExactDeduplicator

__all__ = [
    'PiscesLxDataMinHashDeduplicator',
    'PiscesLxDataExactDeduplicator',
]
