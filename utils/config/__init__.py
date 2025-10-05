#!/usr/bin/env python3

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

from .manager import PiscesLxCoreConfigManager, PiscesLxCoreConfigManagerFacade, get_config_manager
from .loader import PiscesLxCoreConfigLoader

__all__ = [
    'PiscesLxCoreConfigManager',
    'PiscesLxCoreConfigManagerFacade',
    'PiscesLxCoreConfigLoader',
    'get_config_manager',  # 保留向后兼容
]