#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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

import os
import sys
import time
import psutil
import platform
from datetime import datetime
from typing import Any, Optional, Dict, Tuple, List

from utils import PiscesLxCoreLog
from utils import PiscesLxCoreObservabilityManager
from utils import PiscesLxCoreHookBus
from utils import PiscesLxCoreEnhancedCacheManager
from utils import PiscesLxCoreTimeout, PiscesLxCoreRetry
from utils import PiscesLxCoreFS
from utils import PiscesLxCoreDeviceManager

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
        # Initialize context manager
        self.context_manager = PiscesLxMonitorGlobalContext
        
        # Initialize components
        self.fs_manager = self.context_manager.get_fs_manager()
        self.cache_manager = self.context_manager.get_cache_manager()
        self.device_manager = self.context_manager.get_device_manager()
        self.logger = self.context_manager.get_logger()
        
        # Initialize modular components
        self.stats_collector = PiscesLxMonitorStatsCollector()
        self.alert_manager = PiscesLxMonitorAlertManager()
        self.data_manager = PiscesLxMonitorDataManager()
        self.display = PiscesLxToolsMonitorDisplay()
        
        # Initialize observability manager
        self.observability_manager = PiscesLxCoreObservabilityManager()
        
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
        
        # Initialize GPU detection
        self._init_gpu_detection()
        
        # Create monitor log file
        self._init_logging()
    
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
        self.MONITOR_LOG_DIR = self.fs_manager.path_join(
            self.fs_manager.get_project_root(), '.pisceslx', 'logs'
        )
        self.fs_manager.ensure_dir(self.MONITOR_LOG_DIR)
        self.MONITOR_LOG_FILE = self.fs_manager.path_join(self.MONITOR_LOG_DIR, 'monitor.log')
        
        # Use separate logger for monitor with console disabled
        self.monitor_logger = PiscesLxCoreLog(
            "pisceslx.monitor.file", 
            file_path=self.MONITOR_LOG_FILE, 
            console=False, 
            enable_file=True
        )
        self.obs_logger = PiscesLxCoreLog("pisceslx.monitor.observability")
    
    @PiscesLxCoreRetry(max_attempts=3, delay=1.0)
    def get_observability_metrics(self):
        """Get metrics from observability framework."""
        try:
            # Check cache first
            cache_key = "observability_metrics"
            cached_metrics = self.cache_manager.get(cache_key)
            if cached_metrics and time.time() - cached_metrics.get('timestamp', 0) < 5.0:
                return cached_metrics.get('data')
            
            # Start monitoring session if not active
            if not self.observability_manager.is_monitoring_active():
                session_config = {
                    "session_id": "system_monitor",
                    "interval": 5.0,
                    "enable_gpu": self.gpu_enabled
                }
                self.observability_manager.start_monitoring(session_config)
            
            # Get system metrics
            metrics = self.observability_manager.get_system_metrics()
            
            # Cache results
            if metrics:
                self.cache_manager.set(cache_key, {
                    'timestamp': time.time(),
                    'data': metrics
                }, ttl=5.0)
                
                # Log metrics to file only
                self.obs_logger.info(f"Observability metrics: {metrics}")
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get observability metrics: {e}")
            return None
    
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
            set_context(hooks=getattr(args, 'hooks', None), 
                       profiler=getattr(args, 'profiler', None), 
                       cfg=getattr(args, 'cfg', None))
            
            # Emit start event
            _emit("monitor_start", args=vars(args) if hasattr(args, '__dict__') else {})
            
            self.logger.info("Starting system monitor")
            
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
                        _emit("monitor_alerts", alerts=alerts)
                    
                    # Process monitoring data
                    logged, averages = self.process_monitoring_data(stats)
                    
                    # Display if needed (but keep display separate from logging)
                    if hasattr(args, 'display') and args.display:
                        display_output = self.format_display(stats)
                        self.logger.info(display_output)
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
                    _emit("monitor_error", error=str(e))
                    time.sleep(self.UPDATE_INTERVAL)
            
            # Emit end event
            self.context_manager.emit_event("monitor_end", iterations=iteration)
            self.logger.info("System monitor stopped")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Monitor failed: {e}")
            return 1

# Only expose the main implementation class
# All functionality is accessed through PiscesLxToolsMonitorImpl class

