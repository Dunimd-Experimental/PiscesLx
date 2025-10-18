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

import time
import threading
from typing import Dict, Any, Optional
from ..log.core import PiscesLxCoreLog
from ..error import PiscesLxCoreObservabilityError
from .service import PiscesLxCoreObservabilityService
from .runtime.monitor import run_monitor_loop

class PiscesLxCoreObservabilityManager:
    """Manages the lifecycle of monitoring sessions, including starting, stopping, and collecting metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the monitoring manager.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        self._monitoring_active = False
        self._metrics_cache = {}
        self._performance_history = []
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self.logger = PiscesLxCoreLog("pisceslx.observability.manager")
        # Retrieve the singleton instance of the core service
        self._service = PiscesLxCoreObservabilityService.instance()
        self.logger.info("observability.manager.init")

    def start_monitoring(self, session_config: Dict[str, Any]) -> bool:
        """Start a monitoring session.

        Args:
            session_config (Dict[str, Any]): Session configuration.

        Returns:
            bool: True if the monitoring session started successfully, False otherwise.
        """
        try:
            session_id = session_config.get("session_id", f"session_{int(time.time())}")
            self.logger.info("observability.monitor.start", session_id=session_id)

            # Prevent duplicate startup
            if self._monitoring_active:
                self.logger.warning("Monitoring session is already running", session_id=session_id)
                return False

            # Avoid duplication with the internal monitoring loop of the core service
            try:
                self._service.disable_internal_monitoring()
            except Exception as e:
                self.logger.warning("Failed to disable internal monitoring", error=str(e), error_class=type(e).__name__)

            # Set monitoring status
            self._monitoring_active = True
            self.config = session_config

            # Start the monitoring thread
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(session_id,),
                daemon=True
            )
            self._monitoring_thread.start()

            self.logger.success("observability.monitor.started", session_id=session_id)
            return True

        except Exception as e:
            self.logger.error("observability.monitor.start.error", error=str(e), error_class=type(e).__name__)
            self._monitoring_active = False
            return False

    def stop_monitoring(self) -> bool:
        """Stop the ongoing monitoring session.

        Returns:
            bool: True if the monitoring session stopped successfully, False otherwise.
        """
        if not self._monitoring_active:
            self.logger.error("Monitoring session is not active")
            return False

        self.logger.info("observability.monitor.stop")
        self._stop_monitoring.set()

        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)

        self._monitoring_active = False
        self.logger.success("observability.monitor.stopped")
        return True

    def get_system_metrics(self) -> Dict[str, Any]:
        """Retrieve the current system metrics.

        Returns:
            Dict[str, Any]: System metrics, including CPU usage, memory usage, throughput, etc.

        Raises:
            PiscesLxCoreObservabilityError: If failed to get system metrics.
        """
        try:
            # Prefer real-time system usage from psutil to avoid always-0 defaults
            cpu_percent = 0.0
            mem_percent = 0.0
            try:
                import psutil  # type: ignore
                # Use a short interval for a quick yet meaningful sample
                cpu_percent = float(psutil.cpu_percent(interval=0.1))
                mem_percent = float(psutil.virtual_memory().percent)
            except Exception:
                # Fallback: keep 0.0 if psutil is not available
                pass

            # Existing performance metrics (requests/latency/throughput/errors)
            pm = self._service.metrics_collector.get_current_metrics()

            # Optionally, try to compute average latency from registry histogram if collector has no data
            p95_latency = pm.p95_latency
            p99_latency = pm.p99_latency
            try:
                if (p95_latency == 0.0 or p99_latency == 0.0):
                    reg = self._service.metrics_registry
                    grouped = reg.snapshot_grouped()
                    lat = grouped.get("histograms", {}).get("llm_latency_ms", {})
                    total_sum = float(lat.get("sum", 0.0))
                    total_cnt = float(lat.get("count", 0.0))
                    if total_cnt > 0.0 and p95_latency == 0.0:
                        # Use avg as a coarse proxy if percentiles are unavailable
                        p95_latency = total_sum / total_cnt
                        p99_latency = max(p99_latency, p95_latency)
            except Exception:
                pass

            return {
                "timestamp": int(time.time()),
                "cpu": {"usage_percent": cpu_percent},
                "memory": {"usage_percent": mem_percent},
                "throughput": pm.throughput,
                "p95_latency": p95_latency,
                "p99_latency": p99_latency,
                "error_rate": pm.error_rate,
                "health_score": pm.health_score,
            }
        except Exception as e:
            self.logger.error("observability.metrics.system.error", error=str(e), error_class=type(e).__name__)
            raise PiscesLxCoreObservabilityError(f"system metrics error: {e}")

    def get_cached_metrics(self, session_id: str, limit: int = 100) -> list:
        """Get the cached metric history for a specific session.

        Args:
            session_id (str): Session ID.
            limit (int, optional): Maximum number of records to return. Defaults to 100.

        Returns:
            list: List of cached metrics.
        """
        metrics = self._metrics_cache.get(session_id, [])
        return metrics[-limit:] if metrics else []

    def get_performance_history(self, limit: int = 1000) -> list:
        """Get the performance history records.

        Args:
            limit (int, optional): Maximum number of records to return. Defaults to 1000.

        Returns:
            list: List of performance history records.
        """
        return self._performance_history[-limit:] if self._performance_history else []

    def is_monitoring_active(self) -> bool:
        """Check if the monitoring session is currently active.

        Returns:
            bool: True if monitoring is active, False otherwise.
        """
        return self._monitoring_active

    def _monitoring_loop(self, session_id: str) -> None:
        """Main monitoring loop that periodically collects and caches metrics.

        Args:
            session_id (str): Session ID.
        """
        self.logger.info("observability.monitor.loop.start", session_id=session_id)

        # Determine interval from config with a safe fallback
        try:
            interval = float(self.config.get("interval", 10.0))
        except Exception:
            interval = 10.0

        # Delegate to the core runtime monitor loop which:
        # - Collects current metrics via the intelligent collector
        # - Runs lightweight collectors (CPU/GPU/Memory, etc.)
        # - Performs state transitions and routes alerts
        # This ensures real metrics are recorded instead of remaining at defaults.
        try:
            run_monitor_loop(self._service, self._stop_monitoring, interval)
        except Exception as e:
            self.logger.error("observability.monitor.loop.error", error=str(e), error_class=type(e).__name__, session_id=session_id)

    def _cache_metrics(self, session_id: str, metrics: Dict[str, Any]) -> None:
        """Cache the collected metrics for a specific session.

        Args:
            session_id (str): Session ID.
            metrics (Dict[str, Any]): Metric data.
        """
        if session_id not in self._metrics_cache:
            self._metrics_cache[session_id] = []

        self._metrics_cache[session_id].append(metrics)

        # Limit the cache size to prevent excessive memory usage
        max_cache_size = 1000
        if len(self._metrics_cache[session_id]) > max_cache_size:
            self._metrics_cache[session_id] = self._metrics_cache[session_id][-max_cache_size:]

    def _record_performance_history(self, metrics: Dict[str, Any]) -> None:
        """Record the performance history based on the collected metrics.

        Args:
            metrics (Dict[str, Any]): Metric data.
        """
        self._performance_history.append({
            "timestamp": metrics.get("timestamp", int(time.time())),
            "cpu_usage": metrics.get("cpu", {}).get("usage_percent", 0),
            "memory_usage": metrics.get("memory", {}).get("usage_percent", 0),
        })

        # Limit the size of the history records to prevent excessive memory usage
        max_history_size = 10000
        if len(self._performance_history) > max_history_size:
            self._performance_history = self._performance_history[-max_history_size:]

    def clear_cache(self) -> None:
        """Clear both the metrics cache and performance history.
        """
        self._metrics_cache.clear()
        self._performance_history.clear()
        self.logger.info("observability.cache.cleared")

    def get_service(self) -> PiscesLxCoreObservabilityService:
        """Retrieve the core service instance.

        Returns:
            PiscesLxCoreObservabilityService: The core service instance.
        """
        return self._service