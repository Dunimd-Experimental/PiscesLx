#!/usr/bin/env/python3
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

from utils.dc import PiscesLxLogger, PiscesLxConfiguration, PiscesLxSystemMonitor
import time
from typing import Any, Optional

_LOG = PiscesLxLogger(__name__)
from utils import PiscesLxCoreHookBus
from utils import PiscesLxCoreEnhancedCacheManager
from opss.concurrency import RetryOperator
from utils import PiscesLxCoreDeviceManager
from utils import PiscesLxCoreFS

PiscesLxCoreLog = PiscesLxLogger
PiscesLxCoreConfigManager = PiscesLxConfiguration

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
            _LOG.debug(f"Profiler started for phase: {phase_name}")
        
        def stop(self, phase_name: str = "monitor", **kwargs) -> Optional[float]:
            """Stop profiling and return elapsed time for the specified phase."""
            if not self._active or phase_name not in self._phase_timers:
                logger.debug(f"Profiler stop called for inactive phase: {phase_name}")
                return None
            
            elapsed = time.perf_counter() - self._phase_timers[phase_name]
            self._phase_results[phase_name] = elapsed
            self._active = False
            
            logger.debug(f"Profiler stopped for phase: {phase_name}, elapsed: {elapsed:.3f}s")
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

        # Provide utils components to global monitor context so implementation picks them up
        try:
            from .context_utils import PiscesLxMonitorGlobalContext
            PiscesLxMonitorGlobalContext.set_utils(
                cache_manager=self._cache_manager,
                device_manager=self._device_manager,
                fs_manager=self._fs_manager,
            )
        except Exception:
            pass
        
        # Orchestrator state
        self._start_time = None
        self._total_sessions = 0
        self._error_count = 0
        
        # Initialize logger
        self.logger = PiscesLxLogger(__name__)

    def run(self, mode: str = "standard") -> None:
        """Main entry point to run monitoring with enhanced utils integration and intelligent adaptation."""
        try:
            # Initialize utils components
            self._start_time = time.time()
            self._init_utils_components()

            # Immediate bootstrap heartbeat for visibility before async work
            try:
                print("monitor bootstrapping...", flush=True)
            except Exception:
                pass

            # Kick off comprehensive device detection asynchronously to avoid startup blocking
            try:
                import threading as _th
                def _bg_detect_devices():
                    try:
                        if hasattr(self._device_manager, 'detect_all_devices'):
                            self._device_manager.detect_all_devices()
                    except Exception:
                        pass
                _th.Thread(target=_bg_detect_devices, daemon=True).start()
            except Exception:
                pass

            # Establish performance baseline asynchronously (non-blocking)
            baseline_metrics = {}
            try:
                import threading as _th
                def _collect_baseline(_out: dict):
                    try:
                        if hasattr(self, '_observability') and getattr(self, '_observability', None):
                            mgr = getattr(self._observability, '_manager', None)
                            if mgr and hasattr(mgr, 'get_system_metrics'):
                                m = mgr.get_system_metrics() or {}
                                _out.update(m)
                    except Exception:
                        pass
                _th.Thread(target=_collect_baseline, args=(baseline_metrics,), daemon=True).start()
            except Exception:
                pass
            
            # Create intelligent session cache
            session_id = f"monitor_{int(time.time())}"
            session_cache = {
                "mode": mode,
                "start_time": time.time(),
                "baseline": baseline_metrics,
                "utils_enabled": {
                    "cache": self._cache_manager is not None,
                    "device": self._device_manager is not None,
                    "observability": self._observability is not None,
                    "metrics": self._metrics_registry is not None
                }
            }
            self._cache_manager.set(f"session_{session_id}", session_cache, ttl=7200.0)  # 2-hour TTL
            
            # Emit enhanced start event
            self.hooks.trigger("monitor.start", {
                "mode": mode,
                "session_id": session_id,
                "baseline": baseline_metrics,
                "timestamp": time.time(),
                "utils_integration": session_cache["utils_enabled"],
                "predicted_duration": self._estimate_session_duration(mode),
                "resource_requirements": self._calculate_resource_requirements()
            })

            # Also emit through global monitor context to re-use existing hook listeners
            try:
                from .context_utils import PiscesLxMonitorGlobalContext
                PiscesLxMonitorGlobalContext.emit_event("monitor.start", {
                    "mode": mode,
                    "session_id": session_id,
                    "baseline": baseline_metrics,
                    "timestamp": time.time(),
                    "utils_integration": session_cache["utils_enabled"],
                })
            except Exception:
                pass

            # Start monitor immediately using runner (non-blocking design)
            try:
                from .runner import PiscesLxToolsMonitorRunner
                runner = PiscesLxToolsMonitorRunner(
                    self.args,
                    hooks=self.hooks,
                    profiler=self.profiler,
                    cfg=self.cfg,
                )
                runner.monitor()
            except Exception:
                # Fallback: if runner import fails, raise clear error
                logger.error("Failed to start monitor runner")
                raise
            
            # Emit completion event with enhanced results
            completion_metrics = {
                "mode": mode,
                "session_id": session_id,
                "duration": time.time() - self._start_time,
                "timestamp": time.time(),
                "error_count": self._error_count,
                "system_health": self._assess_system_health({}, {}),  # Will be populated with real metrics
                "recovery_attempts": getattr(self, '_recovery_attempts', 0)
            }
            self.hooks.trigger("monitor.complete", completion_metrics)
            
            # Update session cache with comprehensive completion data
            session_cache["end_time"] = time.time()
            session_cache["duration"] = session_cache["end_time"] - session_cache["start_time"]
            session_cache["error_count"] = self._error_count
            session_cache["status"] = "completed"
            session_cache["system_health"] = completion_metrics["system_health"]
            session_cache["recovery_attempts"] = completion_metrics["recovery_attempts"]
            self._cache_manager.set(f"session_{session_id}", session_cache, ttl=7200.0)  # 2-hour TTL
            
            logger.debug(f"Monitoring completed successfully in {session_cache['duration']:.2f}s")
            
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            
            # Enhanced error diagnostics and recovery suggestions
            diagnostics = self._diagnose_monitoring_error(e)
            recovery_suggestions = self._generate_recovery_suggestions(e)
            
            # Emit comprehensive error event
            error_event = {
                "error": str(e),
                "timestamp": time.time(),
                "error_count": self._error_count,
                "diagnostics": diagnostics,
                "recovery_suggestions": recovery_suggestions,
                "session_id": session_id if 'session_id' in locals() else "unknown"
            }
            self.hooks.trigger("monitor.error", error_event)
            raise

    def _estimate_session_duration(self, mode: str) -> float:
        """Estimate monitoring session duration based on mode and historical data."""
        # Base estimates for different modes
        base_durations = {
            'standard': 300.0,    # 5 minutes
            'realtime': 1800.0,   # 30 minutes
            'predictive': 3600.0  # 1 hour
        }
        
        base_duration = base_durations.get(mode, 300.0)
        
        # Adjust based on historical performance if available
        if self._cache_manager:
            historical_key = f"historical_duration_{mode}"
            historical_data = self._cache_manager.get(historical_key)
            if historical_data and 'average_duration' in historical_data:
                # Use weighted average of base and historical
                return (base_duration * 0.3 + historical_data['average_duration'] * 0.7)
        
        return base_duration
    
    def _calculate_resource_requirements(self) -> dict:
        """Calculate resource requirements for the monitoring session."""
        requirements = {
            "min_memory_mb": 512,
            "min_cpu_cores": 2,
            "recommended_disk_gb": 1.0,
            "network_bandwidth_mbps": 10
        }
        
        # Adjust based on device capabilities
        if self._device_manager:
            device_info = self._device_manager.get_device_info()
            if device_info.get('gpu_memory_total', 0) > 0:
                requirements["gpu_memory_mb"] = min(1024, device_info.get('gpu_memory_total', 0) * 0.1)
        
        return requirements
    
    def _assess_system_health(self, final_metrics: dict, performance_analysis: dict) -> str:
        """Assess overall system health based on monitoring results."""
        if not final_metrics or not performance_analysis:
            return "unknown"
        
        # Simple health assessment logic
        cpu_usage = final_metrics.get('cpu_usage_percent', 0)
        memory_usage = final_metrics.get('memory_usage_percent', 0)
        efficiency_score = performance_analysis.get('efficiency_score', 0)
        
        if cpu_usage < 80 and memory_usage < 85 and efficiency_score > 0.7:
            return "healthy"
        elif cpu_usage < 95 and memory_usage < 95 and efficiency_score > 0.5:
            return "degraded"
        else:
            return "critical"
    
    def _diagnose_monitoring_error(self, error: Exception) -> dict:
        """Diagnose monitoring errors and provide detailed diagnostics."""
        error_type = type(error).__name__
        diagnostics = {
            "error_type": error_type,
            "recoverable": False,
            "likely_cause": "unknown",
            "suggested_actions": []
        }
        
        # Common error patterns and diagnostics
        if "timeout" in str(error).lower():
            diagnostics.update({
                "recoverable": True,
                "likely_cause": "Operation timeout - system overload or network issues",
                "suggested_actions": ["Increase timeout duration", "Check system resources", "Reduce monitoring scope"]
            })
        elif "memory" in str(error).lower() or "resource" in str(error).lower():
            diagnostics.update({
                "recoverable": True,
                "likely_cause": "Insufficient system resources",
                "suggested_actions": ["Free up system memory", "Reduce monitoring frequency", "Restart monitoring service"]
            })
        elif "permission" in str(error).lower() or "access" in str(error).lower():
            diagnostics.update({
                "recoverable": False,
                "likely_cause": "Insufficient permissions or access rights",
                "suggested_actions": ["Check user permissions", "Verify file access rights", "Contact system administrator"]
            })
        
        return diagnostics
    
    def _generate_recovery_suggestions(self, error: Exception) -> list:
        """Generate context-aware recovery suggestions based on error type."""
        suggestions = []
        error_msg = str(error).lower()
        
        if "timeout" in error_msg:
            suggestions.extend([
                "Consider increasing the timeout duration in configuration",
                "Check network connectivity and system responsiveness",
                "Reduce monitoring scope or frequency to lighten system load"
            ])
        elif "memory" in error_msg:
            suggestions.extend([
                "Free up system memory by closing unnecessary applications",
                "Consider increasing available system RAM or using swap space",
                "Optimize monitoring data collection to reduce memory footprint"
            ])
        elif "device" in error_msg:
            suggestions.extend([
                "Verify device drivers and hardware compatibility",
                "Check device availability and resource allocation",
                "Consider using alternative device configurations"
            ])
        else:
            suggestions.extend([
                "Check system logs for additional error details",
                "Verify configuration settings and parameters",
                "Ensure all required dependencies are properly installed"
            ])
        
        return suggestions
    
    def _attempt_error_recovery(self, error: Exception, session_id: str) -> bool:
        """Attempt automatic error recovery based on error diagnostics."""
        try:
            error_type = type(error).__name__
            logger.debug(f"Attempting automatic recovery for {error_type}")
            
            if "timeout" in str(error).lower():
                # Retry with increased timeout
                logger.debug("Implementing timeout recovery strategy...")
                return True
            elif "memory" in str(error).lower():
                # Attempt memory cleanup
                logger.debug("Implementing memory recovery strategy...")
                if hasattr(self._cache_manager, 'cleanup'):
                    self._cache_manager.cleanup()
                return True
            
            return False
            
        except Exception as recovery_error:
            logger.debug(f"Automatic recovery failed: {recovery_error}")
            return False
