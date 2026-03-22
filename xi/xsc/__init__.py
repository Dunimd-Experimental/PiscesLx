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
Xi Studio - Flagship LLM Workstation

This package provides the Xi Studio backend server and launcher.
"""

from .dc import XiLogger, XiLogLevel, XiErrorCode, XiErrorContext, XiError
from .types import (
    XiCommand,
    XiRequest,
    XiResponse,
    XiRunStatus,
    XiRunInfo,
    XiSystemStats,
    XiLogEntry,
    XiControlRequest,
    XiControlResponse,
    XiGpuVendor,
    XiGpuInfo,
)
from .session import XmcSession, XmcSessionManager
from .executor import XiExecutor
from .server import XiServer, app
from .launcher import XiLauncher

__all__ = [
    "XiLogger",
    "XiLogLevel",
    "XiErrorCode",
    "XiErrorContext",
    "XiError",
    "XiCommand",
    "XiRequest",
    "XiResponse",
    "XiRunStatus",
    "XiRunInfo",
    "XiSystemStats",
    "XiLogEntry",
    "XiControlRequest",
    "XiControlResponse",
    "XiGpuVendor",
    "XiGpuInfo",
    "XmcSession",
    "XmcSessionManager",
    "XiExecutor",
    "XiServer",
    "XiLauncher",
    "app",
]
