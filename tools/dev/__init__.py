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
PiscesLx Developer Mode Module - Refactored with Rich Live + Layout.

This module provides an interactive developer mode for training debugging,
featuring a vim-style command interface with true split-screen display.

Architecture:
    The UI uses Rich's Live display with Layout for persistent split-screen:
    
    +------------------------------------------+
    |                                          |
    |         Training Logs (scrollable)       |
    |         Layout: logs (ratio=4)           |
    |                                          |
    +------------------------------------------+
    | > _                                      |
    | [Dev Mode] Type :help for commands       |
    | Layout: command (size=3, fixed)          |
    +------------------------------------------+

Key Components:
    - PiscesLxDevModeManager: Global singleton manager with log capture
    - PiscesLxDevModeUI: Terminal UI with Rich Live display
    - PiscesLxDevModeCommands: Command registry and executor
    - PiscesLxDevModeLogCapture: Log handler for capturing training logs

Key Improvements:
    1. Live Display: Persistent UI that won't be overwritten by logs
    2. Split Layout: Logs and command bar in separate regions
    3. Blocking Input: Reliable keyboard capture via queue
    4. Log Handler: Captures training logs for display

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
