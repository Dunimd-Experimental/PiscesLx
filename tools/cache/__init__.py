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
PiscesLx Cache Management Module.

This module provides cache management functionality for the .pisceslx directory,
allowing users to view cache status and clean cache directories.

The cache manager protects the settings/ directory from any deletion operations,
ensuring user configurations are never accidentally removed.

Architecture:
    The cache system consists of:
    
    1. PiscesLxCacheManager: Main cache management class
       - Scans .pisceslx directory for cache information
       - Provides status display and cleaning functionality
       - Protects settings/ directory permanently
    
    2. Integration with CLI:
       - `python manage.py cache` - Show cache status
       - `python manage.py cache clean` - Clean all cache

Protected Directories:
    - settings/ - User configuration files (never deleted)

Usage:
    >>> from tools.cache import PiscesLxCacheManager
    >>> manager = PiscesLxCacheManager()
    >>> manager.status()  # Show cache status
    >>> manager.clean()   # Clean all cache
"""

from tools.cache.manager import PiscesLxCacheManager

__all__ = ['PiscesLxCacheManager']
