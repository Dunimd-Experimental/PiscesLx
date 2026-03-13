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
PiscesL1 Developer Mode Module.

This module provides an interactive developer mode for training debugging,
featuring a vim-style command interface at the bottom of the training log display.

Key Components:
    - PiscesLxDevModeManager: Global singleton manager for developer mode
    - PiscesLxDevModeUI: Terminal UI renderer with rich library
    - PiscesLxDevModeCommands: Command registry and executor
    - PiscesLxDevModeOverlay: Temporary overlay display for command results

Usage:
    Enable developer mode:
        $ python manage.py dev enable
    
    During training, commands are available:
        :mem [module]     - Show memory details
        :layer <n>        - Show layer information
        :grad             - Show gradient statistics
        :pause            - Pause training
        :resume           - Resume training
        :save [name]      - Save checkpoint
        :lr <value>       - Adjust learning rate
        :config           - Show configuration
        :watch <var>      - Watch variable
        :inject <target>  - Force injection
        :freeze <layer>   - Freeze layer
        :profile [type]   - Performance profiling
        :help             - Show help
        :q                - Close overlay

Configuration:
    Settings are stored in ~/.pisceslx/settings/settings.yaml:
        dev:
          enabled: false
"""

from .manager import PiscesLxDevModeManager

__all__ = ["PiscesLxDevModeManager"]
