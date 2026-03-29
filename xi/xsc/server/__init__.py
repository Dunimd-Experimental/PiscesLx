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
Server module initialization.
"""

from .app import XiServer, get_app, app
from .middleware import setup_middleware
from .websocket import setup_websockets
from .hardware import collect_system_stats
from .router import (
    setup_health_routes,
    setup_handshake_route,
    setup_runs_routes,
    setup_models_routes,
    setup_inference_routes,
    setup_filesystem_routes,
    setup_xi_routes,
    setup_notification_routes,
    setup_tools_routes,
)

__all__ = [
    "XiServer",
    "get_app",
    "app",
    "setup_middleware",
    "setup_websockets",
    "collect_system_stats",
    "setup_health_routes",
    "setup_handshake_route",
    "setup_runs_routes",
    "setup_models_routes",
    "setup_inference_routes",
    "setup_filesystem_routes",
    "setup_xi_routes",
    "setup_notification_routes",
    "setup_tools_routes",
]
