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
Type definitions for Xi Studio Backend Server.

This module defines all data types, request/response models, and
enumerations used by the Xi backend API server.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


class XiCommand(str, Enum):
    TRAIN = "train"
    SERVE = "serve"
    BENCHMARK = "benchmark"
    DOWNLOAD = "download"
    MONITOR = "monitor"
    TEST = "test"
    CACHE = "cache"
    DEV = "dev"


class XiRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class XiGpuVendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"


@dataclass
class XiGpuInfo:
    index: int
    vendor: XiGpuVendor
    name: str
    utilization: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    temperature: float = 0.0
    power_draw: float = 0.0
    power_limit: float = 0.0
    driver_version: str = ""


@dataclass
class XiRequest:
    command: XiCommand
    args: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    background: bool = True


@dataclass
class XiResponse:
    success: bool
    run_id: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class XiRunInfo:
    run_id: str
    run_name: str
    command: str
    status: XiRunStatus
    phase: str
    pid: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""
    progress: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class XiSystemStats:
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    gpu_count: int = 0
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_memory_total: List[float] = field(default_factory=list)
    gpu_vendors: List[str] = field(default_factory=list)
    gpu_names: List[str] = field(default_factory=list)
    gpu_temperatures: List[float] = field(default_factory=list)
    gpu_power_draw: List[float] = field(default_factory=list)
    uptime_seconds: float = 0.0
    request_count: int = 0
    qps: float = 0.0


@dataclass
class XiLogEntry:
    timestamp: str
    level: str
    message: str
    source: str = "xi"
    run_id: Optional[str] = None


@dataclass
class XiControlRequest:
    run_id: str
    action: str


@dataclass
class XiControlResponse:
    success: bool
    run_id: str
    action: str
    message: str
    previous_status: Optional[str] = None
    new_status: Optional[str] = None
