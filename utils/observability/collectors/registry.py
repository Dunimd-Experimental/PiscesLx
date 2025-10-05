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

import os
import time
from .base import BaseCollector
from typing import Any, Dict, List, Optional

class _RuntimeCollector(BaseCollector):
    """A collector class responsible for collecting runtime metrics and managing health status."""
    _name = "runtime"

    def collect(self, registry: Any) -> None:
        """Collect runtime metrics and export the collector's health status.

        This method measures the time taken to collect metrics, handles both success and error cases,
        and records the collection duration. It's designed to be exception-safe.

        Args:
            registry (Any): The registry object where the collected metrics will be stored.
        """
        start = time.time()
        status = "success"
        try:
            from .runtime import collect_runtime_metrics
            collect_runtime_metrics(registry)
            self._mark_success(time.time())
            try:
                self._export_health(registry)
            except Exception:
                pass
        except Exception:
            status = "error"
            self._mark_error(time.time())
            try:
                self._export_health(registry)
            except Exception:
                pass
        finally:
            try:
                dt = max(0.0, time.time() - start)
                registry.observe("collector_duration_seconds", dt, labels={"collector": self._name, "status": status})
            except Exception:
                pass

class _GpuCollector(BaseCollector):
    """A collector class responsible for collecting GPU metrics and managing health status."""
    _name = "gpu"

    def collect(self, registry: Any) -> None:
        """Collect GPU metrics and export the collector's health status.

        This method measures the time taken to collect metrics, handles both success and error cases,
        and records the collection duration. It's designed to be exception-safe.

        Args:
            registry (Any): The registry object where the collected metrics will be stored.
        """
        start = time.time()
        status = "success"
        try:
            from .gpu import collect_gpu_metrics
            collect_gpu_metrics(registry)
            self._mark_success(time.time())
            try:
                self._export_health(registry)
            except Exception:
                pass
        except Exception:
            status = "error"
            self._mark_error(time.time())
            try:
                self._export_health(registry)
            except Exception:
                pass
        finally:
            try:
                dt = max(0.0, time.time() - start)
                registry.observe("collector_duration_seconds", dt, labels={"collector": self._name, "status": status})
            except Exception:
                pass

class _CpuCollector(BaseCollector):
    """A collector class responsible for collecting CPU metrics and managing health status."""
    _name = "cpu"

    def collect(self, registry: Any) -> None:
        """Collect CPU metrics and export the collector's health status.

        This method measures the time taken to collect metrics, handles both success and error cases,
        and records the collection duration. It's designed to be exception-safe.

        Args:
            registry (Any): The registry object where the collected metrics will be stored.
        """
        start = time.time()
        status = "success"
        try:
            from .cpu import collect_cpu_metrics
            collect_cpu_metrics(registry)
            self._mark_success(time.time())
            try:
                self._export_health(registry)
            except Exception:
                pass
        except Exception:
            status = "error"
            self._mark_error(time.time())
            try:
                self._export_health(registry)
            except Exception:
                pass
        finally:
            try:
                dt = max(0.0, time.time() - start)
                registry.observe("collector_duration_seconds", dt, labels={"collector": self._name, "status": status})
            except Exception:
                pass

    def _export_health(self, registry: Any) -> None:
        """Export the health-related metrics of the collector.

        This method exports metrics including the current health status, total number of failures,
        and the timestamp of the last successful collection.

        Args:
            registry (Any): The registry object where the health metrics will be stored.
        """
        try:
            labels = {"collector": self._name}
            registry.gauge("collector_health", labels=["collector"]).set(1.0 if self._failures == 0 else 0.0, labels)
            registry.gauge("collector_failures_total", labels=["collector"]).set(float(self._failures), labels)
            registry.gauge("collector_last_success_ts", labels=["collector"]).set(float(self._last_ok), labels)
        except Exception:
            pass

class CollectorRegistry:
    """A class for managing a collection of metric collectors."""

    def __init__(self) -> None:
        """Initialize the collector registry with an empty list of collectors."""
        self._collectors: List[BaseCollector] = []

    def register(self, c: BaseCollector) -> None:
        """Register a collector in the registry.

        Args:
            c (BaseCollector): The collector to be registered.
        """
        self._collectors.append(c)

    def list(self) -> List[BaseCollector]:
        """Retrieve a list of all registered collectors.

        Returns:
            List[BaseCollector]: A list containing all registered collectors.
        """
        return list(self._collectors)

_default_registry: Optional[CollectorRegistry] = None

def get_default_collector_registry() -> CollectorRegistry:
    """Retrieve the default collector registry.

    If the default registry does not exist, it will be created and initialized with default collectors.
    The enabled status of these collectors can be controlled by the "PISCES_OBS_COLLECTORS" environment variable.

    Returns:
        CollectorRegistry: The default collector registry.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = CollectorRegistry()
        # Register default collectors
        rc = _RuntimeCollector()
        gc = _GpuCollector()
        cc = _CpuCollector()
        # Enable collectors based on the "PISCES_OBS_COLLECTORS" environment variable
        try:
            wanted = os.environ.get("PISCES_OBS_COLLECTORS", "").strip()
            if wanted:
                allow = {p.strip().lower() for p in wanted.split(',') if p.strip()}
                rc.set_enabled("runtime" in allow)
                gc.set_enabled("gpu" in allow)
                cc.set_enabled("cpu" in allow)
        except Exception:
            pass
        _default_registry.register(rc)
        _default_registry.register(gc)
        _default_registry.register(cc)
    return _default_registry
