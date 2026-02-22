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

import os
from datetime import datetime
from typing import Dict, Any, Optional, List

from utils.dc import PiscesLxFilesystem, PiscesLxLogger
from utils.paths import get_cache_dir

from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Monitor", file_path=get_log_file("PiscesLx.Tools.Monitor"), enable_file=True)


class PiscesLxMonitorCache:
    def __init__(self):
        self._kv: Dict[str, Any] = {}
        self._root = get_cache_dir("monitor")

    def get_cache_dir(self, name: str) -> str:
        return get_cache_dir(os.path.join("monitor", str(name)))

    def get(self, key: str, default: Any = None) -> Any:
        return self._kv.get(key, default)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        self._kv[key] = value


class _SimpleDeviceManager:
    """Simple device manager for monitor."""

    def __init__(self):
        self._devices = []

    def get_gpu_devices(self):
        """Get GPU devices."""
        return []


class PiscesLxMonitorContext:
    """Runtime context management for monitoring operations."""

    def __init__(self):
        """Initialize context manager."""
        self._context = {}
        self._hooks = []

    def set_context(self, key: str, value: Any) -> None:
        """Set context value."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value."""
        return self._context.get(key, default)

    def add_hook(self, hook: callable) -> None:
        """Add hook function."""
        self._hooks.append(hook)

    def emit_event(self, event_type: str, data: Dict[str, Any] = None, **kwargs) -> None:
        """Emit event to all hooks."""
        event_data = data or {}
        event_data.update(kwargs)
        for hook in self._hooks:
            try:
                hook(event_type, event_data)
            except Exception as e:
                _LOG.error("monitor_hook_error", error=str(e), event_type=event_type)


class PiscesLxMonitorUtils:
    """Utility manager for monitoring components."""

    def __init__(self):
        """Initialize utility components."""
        self.fs_manager = PiscesLxFilesystem()
        self.cache_manager = PiscesLxMonitorCache()
        self.device_manager = _SimpleDeviceManager()
        self.logger = PiscesLxLogger("PiscesLx.Tools.Monitor", file_path=get_log_file("PiscesLx.Tools.Monitor"), enable_file=True)

    def get_cache_manager(self):
        """Get cache manager."""
        return self.cache_manager

    def get_device_manager(self):
        """Get device manager."""
        return self.device_manager

    def get_fs_manager(self):
        """Get filesystem manager."""
        return self.fs_manager

    def PiscesLxLogger(self):
        """Get logger."""
        return self.logger


class PiscesLxMonitorContextManager:
    """Global context manager for monitoring operations."""

    def __init__(self):
        """Initialize global context manager."""
        self._context_manager = PiscesLxMonitorContext()
        self._utils_manager = PiscesLxMonitorUtils()

    def set_utils(self, cache_manager=None, device_manager=None, fs_manager=None, logger=None) -> None:
        if cache_manager is not None:
            self._utils_manager.cache_manager = cache_manager
        if device_manager is not None:
            self._utils_manager.device_manager = device_manager
        if fs_manager is not None:
            self._utils_manager.fs_manager = fs_manager
        if logger is not None:
            self._utils_manager.logger = logger

    def set_context(self, key: str, value: Any) -> None:
        """Set global context value."""
        self._context_manager.set_context(key, value)

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get global context value."""
        return self._context_manager.get_context(key, default)

    def add_hook(self, hook: callable) -> None:
        """Add global hook function."""
        self._context_manager.add_hook(hook)

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit global event."""
        self._context_manager.emit_event(event_type, data)

    def get_cache_manager(self):
        """Get global cache manager."""
        return self._utils_manager.get_cache_manager()

    def get_device_manager(self):
        """Get global device manager."""
        return self._utils_manager.get_device_manager()

    def get_fs_manager(self):
        """Get global filesystem manager."""
        return self._utils_manager.get_fs_manager()

    def PiscesLxLogger(self):
        """Get global logger."""
        return self._utils_manager.get_logger()

MONITOR_DATA_CACHE_KEY = "piscesl1_monitor_data"
MONITOR_LAST_IO_CACHE_KEY = "piscesl1_monitor_last_io"
MONITOR_ALERTS_CACHE_KEY = "piscesl1_monitor_alerts"
