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
import sys
import time
import psutil
import platform
from datetime import datetime
from typing import Any, Optional, Dict, Tuple, List

from utils.dc import PiscesLxLogger

# Import new modular components
from .context_utils import PiscesLxMonitorGlobalContext
from .stats_collector import PiscesLxMonitorStatsCollector
from .alert_manager import PiscesLxMonitorAlertManager
from .data_manager import PiscesLxMonitorDataManager
from .display_utils import PiscesLxToolsMonitorDisplay

class PiscesLxToolsMonitorImpl:
    """Refactored monitor implementation using modular components."""
    
    def __init__(self):
        """Initialize monitor with modular components."""
        # Configuration - must be defined first
        self.UPDATE_INTERVAL = 1
        self.LOG_INTERVAL = 60
        self.ANOMALY_THRESHOLD = {
            'cpu_percent_total': 20,
            'memory_percent': 10,
            'gpu_util': 15,
            'disk_percent': 5,
        }
        
        # Initialize context manager
        self.context_manager = PiscesLxMonitorGlobalContext
        
        # Initialize components
        self.fs_manager = self.context_manager.get_fs_manager()
        self.cache_manager = self.context_manager.get_cache_manager()
        self.device_manager = self.context_manager.get_device_manager()
        self.logger = self.context_manager.get_logger()
        try:
            self.console_logger = PiscesLxLogger("PiscesLx.Tools.Monitor.Console", file_path=get_log_file("PiscesLx.Tools.Monitor.Console"), enable_file=True)
        except Exception:
            self.console_logger = PiscesLxLogger("PiscesLx.Tools.Monitor.Console")
        
        # Initialize modular components with shared dependencies
        self.stats_collector = PiscesLxMonitorStatsCollector(
            self.cache_manager,
            self.device_manager,
        )
        self.alert_manager = PiscesLxMonitorAlertManager(
            self.cache_manager,
            self.logger,
        )
        self.data_manager = PiscesLxMonitorDataManager(
            self.cache_manager,
            self.logger,
            log_interval=self.LOG_INTERVAL,
        )
        self.display = PiscesLxToolsMonitorDisplay()
        
        # Use simple cache-based monitoring instead of observability
        self._monitoring_active = False
        
        # Configuration
        self.UPDATE_INTERVAL = 1  # 1 second sampling interval
        self.LOG_INTERVAL = 60   # 60 seconds logging interval
        self.ANOMALY_THRESHOLD = {
            'cpu_percent_total': 20,
            'memory_percent': 10,
            'gpu_util': 15,
            'disk_percent': 5,
        }
        
        # Runtime state
        self.data_buffer = []
        self.last_recorded_averages = {}
        self.last_log_time = time.time()
        self.gpu_enabled = False
        self.gpu_count = 0
        self._last_console_heartbeat = 0.0
        
        # Initialize GPU detection
        self._init_gpu_detection()
        
        # Create monitor log file
        self._init_logging()

    def set_context(self, hooks=None, profiler=None, cfg=None):
        """Inject external context (legacy compatibility with runner)."""
        self.context_manager.set_context('hooks', hooks)
        self.context_manager.set_context('profiler', profiler)
        self.context_manager.set_context('cfg', cfg)
        return self

    def _emit_console_line(self, message: str) -> None:
        """Emit a single-line console message with robust stdout fallbacks."""
        # Always attempt to log via console logger (best-effort)
        try:
            if getattr(self, "console_logger", None):
                self.console_logger.info(message)
        except Exception:
            pass

        line = message if message.endswith(os.linesep) else message + os.linesep

        # Prefer the original stdout to bypass any redirected logger streams
        stream = getattr(sys, "__stdout__", None)
        if stream and hasattr(stream, "write") and hasattr(stream, "flush"):
            try:
                stream.write(line)
                stream.flush()
                return
            except Exception:
                pass

        # Fallback to os.write on the underlying file descriptor if available
        fd = None
        try:
            if stream and hasattr(stream, "fileno"):
                fd = stream.fileno()
        except Exception:
            fd = None
        if fd is None:
            try:
                fd = sys.__stdout__.fileno()  # type: ignore[attr-defined]
            except Exception:
                fd = None
        if fd is not None:
            try:
                os.write(fd, line.encode("utf-8", errors="replace"))
                return
            except Exception:
                pass

        # Last resort: standard print
        try:
            print(message, flush=True)
        except Exception:
            pass

    def _init_gpu_detection(self):
        """Initialize GPU detection."""
        try:
            gpu_devices = self.device_manager.get_gpu_devices()
            self.gpu_count = len(gpu_devices)
            if self.gpu_count > 0:
                self.gpu_enabled = True
                self.logger.info(f"Detected {self.gpu_count} GPU(s) via device manager")
        except Exception as e:
            self.logger.warning(f"Device manager GPU detection failed: {e}, falling back to pynvml")
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                if self.gpu_count > 0:
                    self.gpu_enabled = True
            except Exception:
                self.gpu_enabled = False
    
    def _init_logging(self):
        """Initialize logging configuration."""
        import os
        self.MONITOR_LOG_DIR = os.path.join(os.getcwd(), '.pisceslx', 'logs')
        os.makedirs(self.MONITOR_LOG_DIR, exist_ok=True)
        self.MONITOR_LOG_FILE = os.path.join(self.MONITOR_LOG_DIR, 'monitor.log')
        
        self.monitor_logger = PiscesLxLogger("PiscesLx.Tools.Monitor.File", file_path=get_log_file("PiscesLx.Tools.Monitor.File"), enable_file=True)
        self.obs_logger = PiscesLxLogger("PiscesLx.Tools.Monitor.Observability", file_path=get_log_file("PiscesLx.Tools.Monitor.Observability"), enable_file=True)
    
    def get_observability_metrics(self):
        """Get metrics using simple psutil monitoring."""
        try:
            # Check cache first
            cache_key = "observability_metrics"
            cached_metrics = self.cache_manager.get(cache_key)
            if cached_metrics and time.time() - cached_metrics.get('timestamp', 0) < 5.0:
                return cached_metrics.get('data')
            
            # Get system metrics directly with psutil
            metrics = {
                "cpu_percent_total": psutil.cpu_percent(interval=0.1),
                "memory": {
                    "percent": psutil.virtual_memory().percent,
                    "available": psutil.virtual_memory().available,
                    "total": psutil.virtual_memory().total
                },
                "disk": {
                    "percent": psutil.disk_usage('/').percent
                }
            }
            
            # Add GPU metrics if available
            if self.gpu_enabled and self.gpu_count > 0:
                try:
                    gpu_metrics = []
                    for i in range(self.gpu_count):
                        gpu_metrics.append({
                            "id": i,
                            "utilization": 0,
                            "memory_used": 0,
                            "memory_total": 0
                        })
                    metrics["gpu"] = gpu_metrics
                except Exception:
                    pass
            
            # Cache results
            self.cache_manager.set(cache_key, {
                'timestamp': time.time(),
                'data': metrics
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {"cpu_percent_total": psutil.cpu_percent(interval=None), "memory": {"percent": psutil.virtual_memory().percent}}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            # Use stats collector for main system stats
            stats = self.stats_collector.get_system_stats()
            
            # Add observability metrics if available
            obs_metrics = self.get_observability_metrics()
            if obs_metrics:
                stats['observability'] = obs_metrics
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def check_alerts(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for system alerts."""
        return self.alert_manager.check_alerts(stats)
    
    def process_monitoring_data(self, stats: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Process monitoring data with buffering and anomaly detection."""
        try:
            # Add to buffer
            self.data_buffer.append({
                'timestamp': time.time(),
                'stats': stats
            })
            
            # Keep buffer size manageable
            if len(self.data_buffer) > 120:  # Keep 2 minutes of data
                self.data_buffer = self.data_buffer[-60:]
            
            # Check if we should log (every 60 seconds)
            current_time = time.time()
            should_log = (current_time - self.last_log_time) >= self.LOG_INTERVAL
            
            if should_log and self.data_buffer:
                # Calculate averages
                averages = self.data_manager.calculate_averages(self.data_buffer)
                
                # Check for anomalies
                anomalies = self.data_manager.detect_anomalies(
                    averages, self.last_recorded_averages, self.ANOMALY_THRESHOLD
                )
                
                # Log if there are anomalies or regular interval
                if anomalies or should_log:
                    self._log_monitoring_data(averages, anomalies)
                    self.last_recorded_averages = averages.copy()
                    self.last_log_time = current_time
                
                return True, averages
            
            return False, {}
            
        except Exception as e:
            self.logger.error(f"Failed to process monitoring data: {e}")
            return False, {}
    
    def _log_monitoring_data(self, averages: Dict[str, Any], anomalies: List[str]):
        """Log monitoring data to file."""
        try:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'averages': averages,
                'anomalies': anomalies,
                'buffer_size': len(self.data_buffer)
            }
            
            # Log to file only
            self.monitor_logger.info(f"Monitoring data: {log_data}")
            
            # Emit event for hooks
            self.context_manager.emit_event("monitor_data_logged", data=log_data)
            
        except Exception as e:
            self.logger.error(f"Failed to log monitoring data: {e}")
    
    def format_display(self, stats: Dict[str, Any], **kwargs) -> str:
        """Format display output."""
        return self.display.format_full_display(stats, **kwargs)
    
    def validate_args(self, args: Any) -> bool:
        """Validate monitor arguments."""
        # Basic validation - can be extended
        if hasattr(args, 'mode') and args.mode not in ['standard', 'enhanced']:
            self.logger.error(f"Invalid monitor mode: {getattr(args, 'mode', 'unknown')}")
            return False
        return True
    
    def monitor(self, args: Any) -> int:
        """Main monitoring loop."""
        try:
            # Validate arguments
            if not self.validate_args(args):
                return 1
            
            # Set runtime context
            self.context_manager.set_context('hooks', getattr(args, 'hooks', None))
            self.context_manager.set_context('profiler', getattr(args, 'profiler', None))
            self.context_manager.set_context('cfg', getattr(args, 'cfg', None))
            
            # Emit start event
            self.context_manager.emit_event("monitor_start", {
                'args': vars(args) if hasattr(args, '__dict__') else {}
            })
            
            self.logger.info("Starting system monitor")
            # Immediate heartbeat: quick non-blocking snapshot so users see progress instantly
            cpu_quick = 0.0
            mem_quick = 0.0
            gpu_quick = None
            try:
                cpu_quick = float(psutil.cpu_percent(interval=None) or 0.0)
            except Exception as e:
                self.logger.debug("warmup_cpu_percent_failed", error=str(e))
            try:
                mem_quick = float(psutil.virtual_memory().percent or 0.0)
            except Exception as e:
                self.logger.debug("warmup_memory_percent_failed", error=str(e))
            if self.gpu_enabled and self.gpu_count > 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        mem_used = torch.cuda.memory_allocated(0)
                        mem_total = torch.cuda.get_device_properties(0).total_memory
                        if mem_total:
                            gpu_quick = f"GPU0 {mem_used/1024/1024:.1f}/{mem_total/1024/1024:.1f} GB"
                except Exception as e:
                    self.logger.debug("warmup_gpu_snapshot_failed", error=str(e))
            msg0 = f"monitor warmup | cpu={cpu_quick:.1f}% mem={mem_quick:.1f}%"
            if gpu_quick:
                msg0 += f" {gpu_quick}"
            self.logger.info(msg0)
            self._emit_console_line(msg0)
            self._last_console_heartbeat = time.time()
            
            # Main monitoring loop
            iteration = 0
            while True:
                try:
                    # Get system statistics
                    stats = self.get_system_stats()
                    
                    if not stats:
                        self.logger.warning("No system stats available, skipping iteration")
                        time.sleep(self.UPDATE_INTERVAL)
                        continue
                    
                    # Check for alerts
                    alerts = self.check_alerts(stats)
                    if alerts:
                        self.context_manager.emit_event("monitor_alerts", {
                            'alerts': alerts
                        })
                    
                    # Process monitoring data
                    logged, averages = self.process_monitoring_data(stats)
                    
                    # Display if needed (but keep display separate from logging)
                    if hasattr(args, 'display') and args.display:
                        display_output = self.format_display(stats)
                        self.logger.info(display_output)
                    else:
                        # Console heartbeat: brief one-line status every ~2s
                        now = time.time()
                        if now - self._last_console_heartbeat >= 2.0:
                            cpu = stats.get('cpu', {}).get('percent_total') or stats.get('cpu_percent_total') or 0.0
                            mem = (stats.get('memory', {}) or {}).get('percent') or stats.get('memory_percent') or 0.0
                            gpu = None
                            try:
                                g = stats.get('gpu', {}) or stats.get('gpus')
                                if isinstance(g, list) and g:
                                    gi = g[0]
                                    mem_used = gi.get('mem_used') or gi.get('memory_used')
                                    mem_total = gi.get('mem_total') or gi.get('memory_total')
                                    if mem_used and mem_total:
                                        gpu = f"GPU0 {mem_used/1024/1024:.1f}/{mem_total/1024/1024:.1f} GB"
                            except Exception as e:
                                self.logger.debug("heartbeat_gpu_snapshot_failed", error=str(e))
                            msg = f"monitor heartbeat | cpu={float(cpu):.1f}% mem={float(mem):.1f}%"
                            if gpu:
                                msg += f" {gpu}"
                            self.logger.info(msg)
                            self._emit_console_line(msg)
                            self._last_console_heartbeat = now
                    # Emit iteration complete
                    self.context_manager.emit_event("monitor_iteration", {
                        'iteration': iteration,
                        'stats': stats,
                        'anomalies': alerts,
                        'logged': logged
                    })
                    
                    iteration += 1
                    time.sleep(self.UPDATE_INTERVAL)
                    
                except KeyboardInterrupt:
                    self.logger.info("Monitor interrupted by user")
                    break
                except Exception as e:
                    self.logger.error(f"Monitor iteration error: {e}")
                    self.context_manager.emit_event("monitor_error", {
                        'error': str(e)
                    })
                    time.sleep(self.UPDATE_INTERVAL)
            
            # Emit end event
            self.context_manager.emit_event("monitor_end", iterations=iteration)
            self.logger.info("System monitor stopped")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Monitor failed: {e}")
            return 1
