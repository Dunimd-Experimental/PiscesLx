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

"""
OPSS Run Integration Module - Runtime Management Integration

This module provides integration with OPSS Run components for comprehensive
runtime management of the inference service.

OPSS Run Components:
    - POPSSRunCLI: Command-line interface for run management
    - POPSSRunController: Run lifecycle control
    - POPSSRunStore: Persistent storage for run data
    - POPSSResourceMonitor: CPU/Memory/GPU/IO monitoring
    - POPSSHeartbeatMonitor: Run heartbeat monitoring
    - POPSSRunManager: Run listing and management
    - POPSSRunAttacher: Real-time run attachment

Architecture:
    PiscesLxRunIntegration
    ├── Run Controller Integration
    │   ├── Run initialization
    │   ├── State management
    │   └── Control queue handling
    ├── Run Store Integration
    │   ├── Event logging
    │   ├── Metric recording
    │   └── Artifact management
    ├── Resource Monitor Integration
    │   ├── CPU/Memory monitoring
    │   ├── GPU monitoring
    │   └── IO monitoring
    └── Heartbeat Monitor Integration
        ├── Liveness detection
        └── Timeout handling

Usage:
    >>> from tools.infer.run_integration import PiscesLxRunIntegration
    >>> from tools.infer.config import RunConfig
    >>> 
    >>> config = RunConfig(run_id="infer_001")
    >>> run = PiscesLxRunIntegration(config)
    >>> run.initialize()
    >>> 
    >>> # Log an event
    >>> run.log_event("request_received", {"path": "/v1/chat/completions"})
    >>> 
    >>> # Get resource stats
    >>> stats = run.get_resource_stats()
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import time
import threading

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from .config import RunConfig


@dataclass
class PiscesLxRunInfo:
    """Run information structure."""
    run_id: str
    run_dir: Optional[str] = None
    status: Optional[str] = None
    phase: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    pid: Optional[int] = None


@dataclass
class PiscesLxResourceStats:
    """Resource statistics structure."""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    threads: int = 0
    timestamp: float = 0.0


class PiscesLxRunIntegration:
    """
    OPSS Run Integration for PiscesLx Inference Service.
    
    This class provides comprehensive runtime management including:
    - Run lifecycle control
    - Event and metric logging
    - Resource monitoring
    - Heartbeat monitoring
    - Run listing and management
    
    Features:
        - Automatic run initialization
        - Graceful component degradation
        - Thread-safe operations
        - Comprehensive error handling
    
    Example:
        >>> config = RunConfig(run_id="infer_001", enable_resource_monitor=True)
        >>> run = PiscesLxRunIntegration(config)
        >>> run.initialize()
        >>> 
        >>> # Log events
        >>> run.log_event("request_received", {"path": "/v1/chat"})
        >>> 
        >>> # Record metrics
        >>> run.record_metric({"latency_ms": 150, "tokens": 256})
        >>> 
        >>> # Get resource stats
        >>> stats = run.get_resource_stats()
    """
    
    def __init__(self, config: RunConfig):
        """
        Initialize Run integration.
        
        Args:
            config: Run configuration
        """
        self.config = config
        self._LOG = PiscesLxLogger("PiscesLx.Tools.Infer", file_path=get_log_file("PiscesLx.Tools.Infer"), enable_file=True)
        
        self._run_store = None
        self._controller = None
        self._resource_monitor = None
        self._heartbeat_monitor = None
        self._run_manager = None
        
        self._initialized = False
        self._init_errors: List[str] = []
        
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        self._lock = threading.Lock()
    
    def initialize(self, spec: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize all run components.
        
        Args:
            spec: Optional run specification
        
        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            return True
        
        self._LOG.info("Initializing Run integration...")
        
        try:
            self._init_run_store()
            
            if self._run_store:
                self._init_controller(spec or {})
            
            if self.config.enable_resource_monitor:
                self._init_resource_monitor()
            
            if self.config.enable_heartbeat:
                self._init_heartbeat_monitor()
            
            self._init_run_manager()
            
            self._initialized = True
            self._LOG.info("Run integration initialized successfully")
            
            if self._init_errors:
                self._LOG.warning(f"Initialization warnings: {self._init_errors}")
            
            return True
            
        except Exception as e:
            self._LOG.error(f"Failed to initialize Run integration: {e}")
            return False
    
    def _init_run_store(self):
        """Initialize Run Store."""
        try:
            from opss.run.store import POPSSRunStore
            from opss.run.id_factory import POPSSRunIdFactory
            
            run_id = self.config.run_id
            if not run_id:
                run_id = POPSSRunIdFactory.generate("infer")
            
            self._run_store = POPSSRunStore(run_id)
            self.config.run_id = run_id
            
            self._LOG.info(f"Run Store initialized with run_id: {run_id}")
            
        except ImportError as e:
            self._init_errors.append(f"Run Store import failed: {e}")
            self._LOG.warning(f"Run Store not available: {e}")
        except Exception as e:
            self._init_errors.append(f"Run Store init failed: {e}")
            self._LOG.error(f"Failed to initialize Run Store: {e}")
    
    def _init_controller(self, spec: Dict[str, Any]):
        """Initialize Run Controller."""
        try:
            from opss.run.controller import POPSSRunController
            
            self._controller = POPSSRunController(self._run_store)
            
            full_spec = {
                "type": "infer_service",
                "run_id": self.config.run_id,
                "run_name": self.config.run_name or f"infer_{self.config.run_id}",
                **spec
            }
            
            self._controller.init_run(full_spec)
            self._LOG.info("Run Controller initialized")
            
        except ImportError as e:
            self._init_errors.append(f"Run Controller import failed: {e}")
            self._LOG.warning(f"Run Controller not available: {e}")
        except Exception as e:
            self._init_errors.append(f"Run Controller init failed: {e}")
            self._LOG.error(f"Failed to initialize Run Controller: {e}")
    
    def _init_resource_monitor(self):
        """Initialize Resource Monitor."""
        try:
            from opss.run.monitor import POPSSResourceMonitor
            
            self._resource_monitor = POPSSResourceMonitor(
                store=self._run_store,
                interval_s=self.config.resource_interval_s,
                enable_gpu=self.config.enable_gpu_monitor,
                enable_io=True
            )
            
            self._resource_monitor.start()
            self._LOG.info("Resource Monitor started")
            
        except ImportError as e:
            self._init_errors.append(f"Resource Monitor import failed: {e}")
            self._LOG.warning(f"Resource Monitor not available: {e}")
        except Exception as e:
            self._init_errors.append(f"Resource Monitor init failed: {e}")
            self._LOG.error(f"Failed to initialize Resource Monitor: {e}")
    
    def _init_heartbeat_monitor(self):
        """Initialize Heartbeat Monitor."""
        try:
            from opss.run.monitor import POPSSHeartbeatMonitor
            
            self._heartbeat_monitor = POPSSHeartbeatMonitor(
                store=self._run_store,
                interval_s=10.0,
                timeout_s=60.0
            )
            
            self._heartbeat_monitor.start()
            self._LOG.info("Heartbeat Monitor started")
            
        except ImportError as e:
            self._init_errors.append(f"Heartbeat Monitor import failed: {e}")
            self._LOG.warning(f"Heartbeat Monitor not available: {e}")
        except Exception as e:
            self._init_errors.append(f"Heartbeat Monitor init failed: {e}")
            self._LOG.error(f"Failed to initialize Heartbeat Monitor: {e}")
    
    def _init_run_manager(self):
        """Initialize Run Manager."""
        try:
            from opss.run.monitor import POPSSRunManager
            
            self._run_manager = POPSSRunManager()
            self._LOG.info("Run Manager initialized")
            
        except ImportError as e:
            self._init_errors.append(f"Run Manager import failed: {e}")
            self._LOG.warning(f"Run Manager not available: {e}")
        except Exception as e:
            self._init_errors.append(f"Run Manager init failed: {e}")
            self._LOG.error(f"Failed to initialize Run Manager: {e}")
    
    # === Event and Metric Logging ===
    
    def log_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None):
        """
        Log an event to the run store.
        
        Args:
            event_type: Type of event
            payload: Optional event payload
        """
        if self._controller:
            try:
                self._controller.append_event(event_type, payload=payload)
            except Exception as e:
                self._LOG.error(f"Failed to log event: {e}")
    
    def record_metric(self, metrics: Dict[str, Any]):
        """
        Record metrics to the run store.
        
        Args:
            metrics: Metrics dictionary
        """
        if self._controller:
            try:
                self._controller.append_metric(metrics)
            except Exception as e:
                self._LOG.error(f"Failed to record metric: {e}")
    
    def update_state(self, state: Dict[str, Any]):
        """
        Update run state.
        
        Args:
            state: State updates
        """
        if self._controller:
            try:
                self._controller.update_state(state)
            except Exception as e:
                self._LOG.error(f"Failed to update state: {e}")
    
    def set_phase(self, phase: str):
        """
        Set run phase.
        
        Args:
            phase: Phase name
        """
        self.update_state({"phase": phase})
    
    def set_status(self, status: str):
        """
        Set run status.
        
        Args:
            status: Status name
        """
        self.update_state({"status": status})
    
    # === Request Tracking ===
    
    def increment_request_count(self):
        """Increment request counter."""
        with self._lock:
            self._request_count += 1
    
    def increment_error_count(self):
        """Increment error counter."""
        with self._lock:
            self._error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get run statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            uptime = time.time() - self._start_time
            qps = self._request_count / uptime if uptime > 0 else 0
            
            return {
                "run_id": self.config.run_id,
                "uptime_seconds": uptime,
                "request_count": self._request_count,
                "error_count": self._error_count,
                "qps": qps,
                "start_time": self._start_time,
            }
    
    # === Resource Monitoring ===
    
    def get_resource_stats(self) -> PiscesLxResourceStats:
        """
        Get current resource statistics.
        
        Returns:
            Resource statistics
        """
        stats = PiscesLxResourceStats(timestamp=time.time())
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            stats.cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            stats.memory_mb = memory_info.rss / (1024 * 1024)
            stats.memory_percent = process.memory_percent()
            stats.threads = process.num_threads()
            
            try:
                io_counters = process.io_counters()
                stats.io_read_mb = io_counters.read_bytes / (1024 * 1024)
                stats.io_write_mb = io_counters.write_bytes / (1024 * 1024)
            except (AttributeError, psutil.AccessDenied):
                pass
            
        except ImportError:
            pass
        except Exception as e:
            self._LOG.debug(f"Failed to get resource stats: {e}")
        
        if self.config.enable_gpu_monitor:
            stats.gpu_memory_mb, stats.gpu_utilization = self._get_gpu_stats()
        
        return stats
    
    def _get_gpu_stats(self) -> tuple:
        """
        Get GPU statistics.
        
        Returns:
            Tuple of (gpu_memory_mb, gpu_utilization)
        """
        try:
            import pynvml
            
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_mb = mem_info.used / (1024 * 1024)
            
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = util.gpu
            
            pynvml.nvmlShutdown()
            
            return gpu_memory_mb, gpu_utilization
            
        except ImportError:
            return 0.0, 0.0
        except Exception as e:
            self._LOG.debug(f"Failed to get GPU stats: {e}")
            return 0.0, 0.0
    
    # === Run Management ===
    
    def list_runs(self, status: Optional[str] = None) -> List[PiscesLxRunInfo]:
        """
        List all runs.
        
        Args:
            status: Optional status filter
        
        Returns:
            List of run information
        """
        if not self._run_manager:
            return []
        
        try:
            runs = self._run_manager.list_runs(status=status)
            return [
                PiscesLxRunInfo(
                    run_id=r.get("run_id", ""),
                    run_dir=r.get("run_dir"),
                    status=r.get("status"),
                    phase=r.get("phase"),
                    created_at=r.get("created_at"),
                    updated_at=r.get("updated_at"),
                    pid=r.get("pid"),
                )
                for r in runs
            ]
        except Exception as e:
            self._LOG.error(f"Failed to list runs: {e}")
            return []
    
    def get_run(self, run_id: str) -> Optional[PiscesLxRunInfo]:
        """
        Get run information by ID.
        
        Args:
            run_id: Run identifier
        
        Returns:
            Run information or None
        """
        if not self._run_manager:
            return None
        
        try:
            run = self._run_manager.get_run(run_id)
            if run:
                return PiscesLxRunInfo(
                    run_id=run.get("run_id", ""),
                    run_dir=run.get("run_dir"),
                    status=run.get("status"),
                    phase=run.get("phase"),
                    created_at=run.get("created_at"),
                    updated_at=run.get("updated_at"),
                    pid=run.get("pid"),
                )
            return None
        except Exception as e:
            self._LOG.error(f"Failed to get run {run_id}: {e}")
            return None
    
    def kill_run(self, run_id: str) -> bool:
        """
        Kill a run by ID.
        
        Args:
            run_id: Run identifier
        
        Returns:
            True if kill succeeded
        """
        if not self._run_manager:
            return False
        
        try:
            return self._run_manager.kill_run(run_id)
        except Exception as e:
            self._LOG.error(f"Failed to kill run {run_id}: {e}")
            return False
    
    # === Control Commands ===
    
    def pause(self) -> bool:
        """
        Pause the run.
        
        Returns:
            True if pause succeeded
        """
        if self._controller:
            try:
                self._controller.pause()
                return True
            except Exception as e:
                self._LOG.error(f"Failed to pause: {e}")
        return False
    
    def resume(self) -> bool:
        """
        Resume the run.
        
        Returns:
            True if resume succeeded
        """
        if self._controller:
            try:
                self._controller.resume()
                return True
            except Exception as e:
                self._LOG.error(f"Failed to resume: {e}")
        return False
    
    def cancel(self) -> bool:
        """
        Cancel the run.
        
        Returns:
            True if cancel succeeded
        """
        if self._controller:
            try:
                self._controller.cancel()
                return True
            except Exception as e:
                self._LOG.error(f"Failed to cancel: {e}")
        return False
    
    # === Status ===
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get run integration status.
        
        Returns:
            Status dictionary
        """
        return {
            "initialized": self._initialized,
            "run_id": self.config.run_id,
            "components": {
                "run_store": self._run_store is not None,
                "controller": self._controller is not None,
                "resource_monitor": self._resource_monitor is not None,
                "heartbeat_monitor": self._heartbeat_monitor is not None,
                "run_manager": self._run_manager is not None,
            },
            "stats": self.get_stats(),
            "resource": self._resource_stats_to_dict(self.get_resource_stats()),
            "init_errors": self._init_errors,
        }
    
    def _resource_stats_to_dict(self, stats: PiscesLxResourceStats) -> Dict[str, Any]:
        """Convert resource stats to dictionary."""
        return {
            "cpu_percent": stats.cpu_percent,
            "memory_mb": stats.memory_mb,
            "memory_percent": stats.memory_percent,
            "gpu_memory_mb": stats.gpu_memory_mb,
            "gpu_utilization": stats.gpu_utilization,
            "io_read_mb": stats.io_read_mb,
            "io_write_mb": stats.io_write_mb,
            "threads": stats.threads,
        }
    
    def shutdown(self):
        """Shutdown all run components."""
        self._LOG.info("Shutting down Run integration...")
        
        self.set_status("stopping")
        
        if self._heartbeat_monitor:
            try:
                self._heartbeat_monitor.stop()
            except Exception as e:
                self._LOG.error(f"Error stopping Heartbeat Monitor: {e}")
        
        if self._resource_monitor:
            try:
                self._resource_monitor.stop()
            except Exception as e:
                self._LOG.error(f"Error stopping Resource Monitor: {e}")
        
        if self._controller:
            try:
                self._controller.finish_run()
            except Exception as e:
                self._LOG.error(f"Error finishing run: {e}")
        
        self._initialized = False
        self._LOG.info("Run integration shutdown complete")
    
    def __repr__(self) -> str:
        return (
            f"PiscesLxRunIntegration("
            f"run_id={self.config.run_id}, "
            f"initialized={self._initialized})"
        )
