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
Xi Configuration Schema Module

Dataclasses for configuration schema definitions.
"""

from .project import XiProjectConfig, XiProjectCommandsConfig
from .paths import XiPathsConfig
from .api import XiApiConfig, XiApiHandshakeConfig
from .ui import XiUiConfig
from .notification import XiNotificationConfig
from .environment import XiEnvironmentConfig, XiRequirementConfig, XiVirtualenvConfig
from .command import XiCommandConfig, XiCommandSchema
from .parameter import XiParameterSchema, XiTabSchema
from .widget import XiWidgetConfig, XiWidgetStyle, XiWidgetValidation, XiValueMapping
from .run_types import XiRunTypeConfig
from .base import XiConfig

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
    "XiRunTypeConfig",
]
