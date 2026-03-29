#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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

"""
Xi Configuration Module

Configuration schema and loaders for Xi Studio.
"""

from .schema import (
    XiConfig,
    XiProjectConfig,
    XiProjectCommandsConfig,
    XiPathsConfig,
    XiApiConfig,
    XiApiHandshakeConfig,
    XiUiConfig,
    XiNotificationConfig,
    XiEnvironmentConfig,
    XiRequirementConfig,
    XiVirtualenvConfig,
    XiCommandConfig,
    XiCommandSchema,
    XiParameterSchema,
    XiTabSchema,
    XiWidgetConfig,
    XiWidgetStyle,
    XiWidgetValidation,
    XiValueMapping,
)
from .loader import XiConfigLoader, get_config_loader, get_xi_config, find_project_root
from .generator import XiConfigGenerator, create_default_config, ensure_config_exists
from .defaults import (
    DEFAULT_XI_TOML,
    DEFAULT_TRAIN_TOML,
    DEFAULT_INFERENCE_TOML,
    DEFAULT_BENCHMARK_TOML,
    DEFAULT_DOWNLOAD_TOML,
    DEFAULT_SCHEMA_JSON,
)

__all__ = [
    "XiConfig",
    "XiProjectConfig",
    "XiProjectCommandsConfig",
    "XiPathsConfig",
    "XiApiConfig",
    "XiApiHandshakeConfig",
    "XiUiConfig",
    "XiNotificationConfig",
    "XiEnvironmentConfig",
    "XiRequirementConfig",
    "XiVirtualenvConfig",
    "XiCommandConfig",
    "XiCommandSchema",
    "XiParameterSchema",
    "XiTabSchema",
    "XiWidgetConfig",
    "XiWidgetStyle",
    "XiWidgetValidation",
    "XiValueMapping",
    "XiConfigLoader",
    "get_config_loader",
    "get_xi_config",
    "find_project_root",
    "XiConfigGenerator",
    "create_default_config",
    "ensure_config_exists",
    "DEFAULT_XI_TOML",
    "DEFAULT_TRAIN_TOML",
    "DEFAULT_INFERENCE_TOML",
    "DEFAULT_BENCHMARK_TOML",
    "DEFAULT_DOWNLOAD_TOML",
    "DEFAULT_SCHEMA_JSON",
]
