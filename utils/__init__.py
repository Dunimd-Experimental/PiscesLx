#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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

import sys
from .log import DEBUG, ERROR, RIGHT
from .progress import progress_bar, get_progress_bar
from .ul import (
    display_update_log,
    display_all_versions,
    display_specific_version,
    display_version_changelog,
    get_current_version,
)

try:
    if 'setup' not in sys.argv:
        from .gpu_manager import GPUManager
        from .cache import PiscesCache, get_cache_manager, get_config_manager
    else:
        GPUManager = None
        PiscesCache = None
        get_cache_manager = None
        get_config_manager = None
except ImportError as e:
    # Handle import errors gracefully during setup or when modules are not available
    GPUManager = None
    PiscesCache = None
    get_cache_manager = None
    get_config_manager = None

__all__ = [
    'GPUManager',
    'DEBUG', 'ERROR', 'RIGHT',
    'progress_bar', 'get_progress_bar',
    'display_version_changelog', 'get_current_version',
    'PiscesCache', 'get_cache_manager', 'get_config_manager',
    'display_update_log', 'display_all_versions', 'display_specific_version',
]