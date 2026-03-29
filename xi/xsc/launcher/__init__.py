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
Xi Launcher Module

Orchestrates backend and frontend startup.
"""

from .launcher import XiLauncher
from .backend import start_backend, wait_for_backend, check_port_available
from .frontend import start_frontend, ensure_frontend_build
from .process import cleanup_processes, kill_port_process, setup_signal_handlers

__all__ = [
    "XiLauncher",
    "start_backend",
    "wait_for_backend",
    "check_port_available",
    "start_frontend",
    "ensure_frontend_build",
    "cleanup_processes",
    "kill_port_process",
    "setup_signal_handlers",
]
