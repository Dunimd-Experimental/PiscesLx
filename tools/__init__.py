#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

# Add project root to Python path for cross-platform compatibility
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure current directory is in path
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Initialize variables with None as fallback
read_config = None
watermark_manager = None
watermark_text = None
get_cache_manager = None
PiscesLxCoreConfigManagerFacade = None

try:
    if 'setup' not in sys.argv:
        # Import local modules first
        try:
            from .mcp import read_config
            from .watermark import watermark_manager, watermark_text
        except ImportError as e:
            print(f"Warning: Could not import local modules: {e}")
        
        # Try multiple import strategies for utils.cache
        import_success = False
        import_attempts = [
            lambda: __import__('utils.cache', fromlist=['get_cache_manager', 'PiscesLxCoreConfigManagerFacade']),
            lambda: __import__('piscesl1.utils.cache', fromlist=['get_cache_manager', 'PiscesLxCoreConfigManagerFacade']),
            lambda: __import__('..utils.cache', fromlist=['get_cache_manager', 'PiscesLxCoreConfigManagerFacade']),
        ]
        
        for attempt in import_attempts:
            try:
                cache_module = attempt()
                get_cache_manager = getattr(cache_module, 'get_cache_manager')
                PiscesLxCoreConfigManagerFacade = getattr(cache_module, 'PiscesLxCoreConfigManagerFacade')
                import_success = True
                break
            except (ImportError, AttributeError):
                continue
        
        if not import_success:
            print(f"Warning: Could not import utils.cache module after multiple attempts")
            
except Exception as e:
    print(f"Warning: Error during module imports: {e}")

# Initialize benchmark module
PiscesL1Benchmark = None
create_benchmark_config = None
run_single_benchmark = None
compare_multiple_models = None

try:
    if 'setup' not in sys.argv:
        from .benchmark import (
            PiscesL1Benchmark,
            create_benchmark_config,
            run_single_benchmark,
            compare_multiple_models
        )
except ImportError as e:
    print(f"Warning: Could not import benchmark module: {e}")

__all__ = [
    'read_config',
    'watermark_manager', 'watermark_text',
    'get_cache_manager', 'PiscesLxCoreConfigManagerFacade',
    'PiscesL1Benchmark', 'create_benchmark_config',
    'run_single_benchmark', 'compare_multiple_models',
]