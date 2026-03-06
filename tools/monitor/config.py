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

from typing import Any, Optional

from utils.dc import PiscesLxConfiguration


class PiscesLxCoreConfigManager:
    """Simple config manager fallback when utils.dc is not available."""
    def __init__(self):
        self._config: dict = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        self._config[key] = value


class PiscesLxToolsMonitorConfig:
    """Monitor configuration management."""

    DEFAULT_THRESHOLDS = {
        'cpu_percent': 90,
        'memory_percent': 90,
        'gpu_util': 95,
        'gpu_mem_percent': 95,
        'disk_percent': 90,
    }

    def __init__(self, data=None) -> None:
        """Initialize the configuration object."""
        self.data = data or {}
        self.config_manager = PiscesLxCoreConfigManager()
        
        args = data.get('args') if isinstance(data, dict) else None
        
        self.update_interval = getattr(args, 'update_interval', 1) if args else 1
        self.log_interval = getattr(args, 'log_interval', 60) if args else 60
        self.buffer_size = 60
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()

    @classmethod
    def from_args(cls, args: Any) -> "PiscesLxToolsMonitorConfig":
        """Create a config object from CLI args."""
        d: dict = {'args': args}
        if getattr(args, "monitor_mode", None):
            d.setdefault("monitor", {})
            d["monitor"]["mode"] = args.monitor_mode
            
        if getattr(args, "update_interval", None):
            d.setdefault("monitor", {})
            d["monitor"]["update_interval"] = args.update_interval
            
        if getattr(args, "log_interval", None):
            d.setdefault("monitor", {})
            d["monitor"]["log_interval"] = args.log_interval
            
        return cls(d)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Perform dot-path retrieval from the underlying config dict."""
        cur: Any = self.data
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def update_threshold(self, metric: str, value: float) -> None:
        """Update alert threshold for a metric."""
        if metric in self.thresholds:
            self.thresholds[metric] = value

    def dump_effective(self) -> dict:
        """Return a shallow copy of the effective configuration."""
        return dict(self.data)
