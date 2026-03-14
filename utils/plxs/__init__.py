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
PLxS (PLx Studio) Backend Server Module.

This module provides the backend API server for PLx Studio, the graphical
workstation for PiscesL1. It exposes REST and WebSocket endpoints for
training management, inference, data operations, and system monitoring.

The server runs on port 3140 and is designed to be started alongside
the PLx Studio frontend via `python manage.py plxs`.

Architecture:
    PLxS Frontend (Next.js) --> PLxS Server (FastAPI:3140) --> manage.py commands

Key Features:
    - Training job management (start, pause, resume, cancel)
    - Real-time log streaming via WebSocket
    - System resource monitoring
    - Model inference endpoints
    - Dataset management operations
"""

from utils.plxs.server import PiscesLxPlxsServer
from utils.plxs.executor import PiscesLxPlxsExecutor
from utils.plxs.launcher import PiscesLxPlxsLauncher
from utils.plxs.types import (
    PiscesLxPlxsCommandRequest,
    PiscesLxPlxsCommandResponse,
    PiscesLxPlxsRunInfo,
    PiscesLxPlxsSystemStats,
    PiscesLxPlxsLogLevel,
    PiscesLxPlxsRunStatus,
    PiscesLxPlxsCommand,
    PiscesLxPlxsLogEntry,
    PiscesLxPlxsControlRequest,
    PiscesLxPlxsControlResponse,
)

__all__ = [
    "PiscesLxPlxsServer",
    "PiscesLxPlxsExecutor",
    "PiscesLxPlxsLauncher",
    "PiscesLxPlxsCommandRequest",
    "PiscesLxPlxsCommandResponse",
    "PiscesLxPlxsRunInfo",
    "PiscesLxPlxsSystemStats",
    "PiscesLxPlxsLogLevel",
    "PiscesLxPlxsRunStatus",
    "PiscesLxPlxsCommand",
    "PiscesLxPlxsLogEntry",
    "PiscesLxPlxsControlRequest",
    "PiscesLxPlxsControlResponse",
]
