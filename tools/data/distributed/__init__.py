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
PiscesLx Distributed Data Processing Module.

This module provides distributed data processing capabilities:
- RayDistributedCleaner: Parallel data cleaning with Ray
- DaskDistributedLoader: Lazy loading with Dask

Key Features:
    - TB-scale data processing
    - Memory-efficient lazy loading
    - Distributed batch processing
    - Fault tolerance

Usage:
    >>> from tools.data.distributed import PiscesLxDataRayDistributedCleaner, PiscesLxDataDaskDistributedLoader
    >>> 
    >>> # Distributed cleaning
    >>> with PiscesLxDataRayDistributedCleaner(parallelism=8) as cleaner:
    ...     cleaned = cleaner.clean("data.parquet", clean_fn)
    >>> 
    >>> # Lazy loading
    >>> with PiscesLxDataDaskDistributedLoader(chunk_size='256MB') as loader:
    ...     ddf = loader.load("large_dataset.parquet")
    ...     for batch in loader.iter_chunks(ddf, batch_size=10000):
    ...         process(batch)
"""

from .ray_cleaner import PiscesLxDataRayDistributedCleaner
from .dask_loader import PiscesLxDataDaskDistributedLoader

__all__ = [
    'PiscesLxDataRayDistributedCleaner',
    'PiscesLxDataDaskDistributedLoader',
]
