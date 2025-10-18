#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from utils import PiscesLxCoreCacheManager
from utils import PiscesLxCoreDeviceFacade
from utils import PiscesLxCoreFS
from utils import PiscesLxCoreDecorators
from utils import PiscesLxCoreLog

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
    
    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event to all hooks."""
        for hook in self._hooks:
            try:
                hook(event_type, data)
            except Exception as e:
                # Use project's log system for error reporting
                from utils import PiscesLxCoreLog
                error_logger = PiscesLxCoreLog("PiscesLx.Tools.Monitor.Context")
                error_logger.error(f"Hook error: {e}")


class PiscesLxMonitorUtils:
    """Utility manager for monitoring components."""
    
    def __init__(self):
        """Initialize utility components."""
        self.fs_manager = PiscesLxCoreFS()
        self.cache_manager = PiscesLxCoreCacheManager()
        self.device_manager = PiscesLxCoreDeviceFacade()
        
        # Build log paths dynamically
        monitor_log_dir = self.fs_manager.logs_dir()
        monitor_log_file = monitor_log_dir / "monitor.log"
        
        # Ensure log directory exists
        self.fs_manager.ensure_dir_exists(monitor_log_dir)
        
        # Initialize logger using project's log system
        self.logger = PiscesLxCoreLog(
            name="pisceslx.monitor.file",
            file_path=str(monitor_log_file),
            console=False,  # Disable console output to avoid polluting app.log
            enable_file=True
        )
    
    def get_cache_manager(self) -> PiscesLxCoreCacheManager:
        """Get cache manager."""
        return self.cache_manager
    
    def get_device_manager(self) -> PiscesLxCoreDeviceFacade:
        """Get device manager."""
        return self.device_manager
    
    def get_fs_manager(self) -> PiscesLxCoreFS:
        """Get filesystem manager."""
        return self.fs_manager
    
    def get_logger(self) -> PiscesLxCoreLog:
        """Get logger."""
        return self.logger


class PiscesLxMonitorContextManager:
    """Global context manager for monitoring operations."""
    
    def __init__(self):
        """Initialize global context manager."""
        self._context_manager = PiscesLxMonitorContext()
        self._utils_manager = PiscesLxMonitorUtils()
    
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
    
    def get_cache_manager(self) -> PiscesLxCoreCacheManager:
        """Get global cache manager."""
        return self._utils_manager.get_cache_manager()
    
    def get_device_manager(self) -> PiscesLxCoreDeviceFacade:
        """Get global device manager."""
        return self._utils_manager.get_device_manager()
    
    def get_fs_manager(self) -> PiscesLxCoreFS:
        """Get global filesystem manager."""
        return self._utils_manager.get_fs_manager()
    
    def get_logger(self) -> PiscesLxCoreLog:
        """Get global logger."""
        return self._utils_manager.get_logger()


# Global instance - only expose class
PiscesLxMonitorGlobalContext = PiscesLxMonitorContextManager()

# Cache keys for monitoring data
MONITOR_DATA_CACHE_KEY = "piscesl1_monitor_data"
MONITOR_LAST_IO_CACHE_KEY = "piscesl1_monitor_last_io"
MONITOR_ALERTS_CACHE_KEY = "piscesl1_monitor_alerts"
MONITOR_ERRORS_CACHE_KEY = "piscesl1_monitor_errors"
MONITOR_SESSION_CACHE_KEY = "piscesl1_monitor_session"