#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import Any, Optional
from utils import RIGHT, ERROR
from utils.hooks import PiscesLxCoreHookBus
from utils.cache.enhanced import PiscesLxCoreEnhancedCacheManager
from utils.concurrency import PiscesLxCoreTimeout, PiscesLxCoreRetry
from utils.device.manager import PiscesLxCoreDeviceManager
from utils.fs.core import PiscesLxCoreFS

# Reuse the training profiler to avoid duplication
try:
    from tools.train.profiler import PiscesLxToolsProfiler
except Exception:
    import time
    from typing import Dict, Optional
    
    class PiscesLxToolsProfiler:  # fallback with basic timing
        """Fallback profiler with basic timing functionality when training profiler is unavailable."""
        
        def __init__(self):
            self._phase_timers: Dict[str, float] = {}
            self._phase_results: Dict[str, float] = {}
            self._active = False
        
        def start(self, phase_name: str = "monitor", **kwargs) -> None:
            """Start profiling for a specific phase with optional metadata."""
            self._active = True
            self._phase_timers[phase_name] = time.perf_counter()
            from utils import DEBUG
            print(f"{DEBUG} Profiler started for phase: {phase_name}")
        
        def stop(self, phase_name: str = "monitor", **kwargs) -> Optional[float]:
            """Stop profiling and return elapsed time for the specified phase."""
            if not self._active or phase_name not in self._phase_timers:
                from utils import DEBUG
                print(f"{DEBUG} Profiler stop called for inactive phase: {phase_name}")
                return None
            
            elapsed = time.perf_counter() - self._phase_timers[phase_name]
            self._phase_results[phase_name] = elapsed
            self._active = False
            
            from utils import DEBUG
            print(f"{DEBUG} Profiler stopped for phase: {phase_name}, elapsed: {elapsed:.3f}s")
            return elapsed

try:
    from .config import PiscesLxToolsMonitorConfig
except Exception:
    class PiscesLxToolsMonitorConfig:  # minimal facade
        def __init__(self, data: dict):
            self.data = data
        @classmethod
        def from_args(cls, args: Any) -> "PiscesLxToolsMonitorConfig":
            d = {}
            if getattr(args, 'monitor_mode', None):
                d.setdefault('monitor', {})
                d['monitor']['mode'] = args.monitor_mode
            return cls(d)
        def get(self, key: str, default: Optional[Any] = None) -> Any:
            cur = self.data
            for part in key.split('.'):
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur

class PiscesLxToolsMonitorOrchestrator:
    """Monitor orchestrator for managing monitoring workflows with enhanced utils integration."""

    def __init__(self, args: Any) -> None:
        """Initialize the monitor orchestrator with utils components."""
        self.args = args
        self.hooks = PiscesLxCoreHookBus()
        self.profiler = PiscesLxToolsProfiler()
        self.cfg = PiscesLxToolsMonitorConfig.from_args(args)
        
        # Initialize utils components
        self._cache_manager = PiscesLxCoreEnhancedCacheManager()
        self._device_manager = PiscesLxCoreDeviceManager()
        self._fs_manager = PiscesLxCoreFS()
        
        # Orchestrator state
        self._start_time = None
        self._total_sessions = 0
        self._error_count = 0
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)

    @PiscesLxCoreTimeout(300.0)
    def run(self, args: Any) -> None:
        """Run the monitoring orchestrator in the configured mode with timeout protection."""
        mode = self.cfg.get('monitor.mode', default=(getattr(args, 'monitor_mode', None) or 'standard'))
        RIGHT(f"Monitor orchestrator mode: {mode}")
        
        try:
            session_start = time.time()
            session_id = f"monitor_session_{int(session_start)}"
            
            # Cache session start
            session_cache = {
                "session_id": session_id,
                "start_time": session_start,
                "mode": mode
            }
            self._cache_manager.set(f"session_{session_id}", session_cache, ttl=3600.0)
            
            self.hooks.trigger("monitor.start", {
                "timestamp": session_start,
                "session_id": session_id,
                "mode": mode
            })
            
            if mode in ('standard',):
                self.run_standard_monitor()
            else:
                ERROR(f"Unknown monitor.mode: {mode}")
                self.logger.error(f"Unknown monitoring mode: {mode}")
                raise SystemExit(1)
                
            # Update session completion
            session_end = time.time()
            session_cache.update({
                "end_time": session_end,
                "duration": session_end - session_start,
                "status": "completed"
            })
            self._cache_manager.set(f"session_{session_id}", session_cache, ttl=3600.0)
            
            self.hooks.trigger("monitor.complete", {
                "timestamp": session_end,
                "session_id": session_id,
                "duration": session_end - session_start
            })
            
        except Exception as e:
            self._error_count += 1
            error_time = time.time()
            
            # Cache error details
            error_cache = {
                "error": str(e),
                "timestamp": error_time,
                "error_count": self._error_count,
                "session_id": session_id if 'session_id' in locals() else "unknown"
            }
            self._cache_manager.set(f"error_{int(error_time)}", error_cache, ttl=1800.0)
            
            self.hooks.trigger("monitor.error", {
                "error": str(e),
                "timestamp": error_time,
                "error_count": self._error_count
            })
            raise

    @PiscesLxCoreRetry(max_attempts=3, delay=1.0, backoff=2.0)
    def run_standard_monitor(self) -> None:
        """Run native monitoring via the class-based runner with enhanced setup and retry logic."""
        try:
            self._start_time = time.time()
            
            # Get device context for setup
            device_info = self._device_manager.get_device_info()
            
            # Trigger setup hook with device context
            self.hooks.trigger("monitor.setup", {
                "config": self.cfg,
                "timestamp": time.time(),
                "device_info": device_info,
                "session_id": f"monitor_{int(time.time())}"
            })
            
            # Cache setup configuration
            setup_cache = {
                "config": self.cfg,
                "device_info": device_info,
                "setup_time": time.time()
            }
            self._cache_manager.set("monitor_setup", setup_cache, ttl=3600.0)
            
            # Initialize runner with enhanced context
            from .runner import PiscesLxToolsMonitorRunner
            runner = PiscesLxToolsMonitorRunner(self.args, hooks=self.hooks, profiler=self.profiler, cfg=self.cfg)
            runner.monitor()
            
        except Exception as e:
            self._error_count += 1
            # Cache error for diagnostics
            error_cache = {
                "error": str(e),
                "timestamp": time.time(),
                "error_count": self._error_count
            }
            self._cache_manager.set("monitor_setup_error", error_cache, ttl=1800.0)
            raise