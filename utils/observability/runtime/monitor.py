#!/usr/bin/env python3

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

import time
from typing import Any
from utils.hooks.bus import get_global_hook_bus

def run_monitor_loop(service: Any, stop_event, interval: float) -> None:
    """Continuously run the observability monitoring loop until stopped.

    This function executes a loop that periodically collects metrics, checks for drift alerts,
    runs lightweight collectors, performs state transitions, and routes alerts.

    Args:
        service (Any): An instance of PiscesLxCoreObservabilityService containing monitoring components.
        stop_event (threading.Event): Event used to signal the loop to stop.
        interval (float): Base interval in seconds between each iteration of the loop.
    """
    logger = service.logger
    # metrics_registry is an attribute holding the registry instance
    reg = service.metrics_registry

    while not stop_event.is_set():
        try:
            start_ts = time.time()

            # Collect current metrics from the intelligent collector
            metrics = service.metrics_collector.get_current_metrics()

            # Check and handle drift alerts on a best-effort basis
            try:
                if bool(getattr(metrics, "p95_drift_alert", False)):
                    ratio = float(getattr(metrics, "p95_drift", 0.0))
                    logger.warning("latency.p95.drift", ratio=ratio, p95_ms=metrics.p95_latency)
                    try:
                        get_global_hook_bus().emit("latency.p95.drift", ratio=ratio, p95_ms=metrics.p95_latency)
                    except Exception as e:
                        logger.error("Failed to emit latency.p95.drift event", error=str(e))
                if bool(getattr(metrics, "p99_drift_alert", False)):
                    ratio99 = float(getattr(metrics, "p99_drift", 0.0))
                    logger.warning("latency.p99.drift", ratio=ratio99, p99_ms=metrics.p99_latency)
                    try:
                        get_global_hook_bus().emit("latency.p99.drift", ratio=ratio99, p99_ms=metrics.p99_latency)
                    except Exception as e:
                        logger.error("Failed to emit latency.p99.drift event", error=str(e))
            except Exception as e:
                logger.error("Failed to check and handle drift alerts", error=str(e))

            # Run lightweight collectors from the registry on a best-effort basis
            try:
                from ..collectors.registry import get_default_collector_registry
                for c in get_default_collector_registry().list():
                    if getattr(c, "enabled", lambda: True)():
                        try:
                            c.collect(reg)
                        except Exception as e:
                            logger.error("Failed to run collector", collector=c.__class__.__name__, error=str(e))
            except Exception as e:
                logger.error("Failed to get and run lightweight collectors", error=str(e))

            # Perform state transition based on the collected metrics
            service._intelligent_state_transition(metrics)

            # Route alerts based on the collected metrics
            service._check_alerts(metrics)

            # Calculate remaining time and sleep until the next iteration
            elapsed = time.time() - start_ts
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

        except Exception as e:
            logger.error("observability.monitor.loop.error", error=str(e))
            time.sleep(max(1.0, interval * 2))
