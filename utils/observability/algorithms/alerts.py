#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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
from typing import Dict, Any, List

def build_alerts(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a list of alert dictionaries based on the provided system metrics and environment thresholds.
    
    Args:
        metrics (Dict[str, Any]): A dictionary containing various system metrics.
            Expected keys include 'p95_latency', 'p99_latency', 'error_rate',
            'p95_drift_alert', 'p99_drift_alert', 'p95_drift', 'p99_drift',
            'cpu_usage', 'memory_usage', and 'throughput'.

    Returns:
        List[Dict[str, Any]]: A list of alert dictionaries. Each dictionary contains
            information about a triggered alert, such as type, value, and threshold.
    """
    alerts: List[Dict[str, Any]] = []
    try:
        # Retrieve alert thresholds from environment variables with default values
        p95_thr = float(os.environ.get("PISCES_ALERT_P95_MS", "1000"))
        p99_thr = float(os.environ.get("PISCES_ALERT_P99_MS", "1500"))
        err_thr = float(os.environ.get("PISCES_ALERT_ERR_RATE", "0.15"))
        cpu_thr = float(os.environ.get("PISCES_ALERT_CPU_PCT", "90"))
        mem_thr = float(os.environ.get("PISCES_ALERT_MEM_PCT", "90"))
        thr_low = float(os.environ.get("PISCES_ALERT_THROUGHPUT_LOW", "0.1"))  # req/s

        # Check for high p95 latency alert
        p95_latency = float(metrics.get("p95_latency", 0.0))
        if p95_latency > p95_thr:
            alerts.append({"type": "latency_p95", "value_ms": p95_latency, "threshold": p95_thr})

        # Check for high p99 latency alert
        p99_latency = float(metrics.get("p99_latency", 0.0))
        if p99_latency > p99_thr:
            alerts.append({"type": "latency_p99", "value_ms": p99_latency, "threshold": p99_thr})

        # Check for high error rate alert
        error_rate = float(metrics.get("error_rate", 0.0))
        if error_rate > err_thr:
            alerts.append({"type": "error_rate", "value": error_rate, "threshold": err_thr})

        # Check for p95 latency drift alert
        if bool(metrics.get("p95_drift_alert", False)):
            alerts.append({"type": "latency_p95_drift", "ratio": float(metrics.get("p95_drift", 0.0))})

        # Check for p99 latency drift alert
        if bool(metrics.get("p99_drift_alert", False)):
            alerts.append({"type": "latency_p99_drift", "ratio": float(metrics.get("p99_drift", 0.0))})

        # Check for high CPU usage alert
        try:
            cpu_usage = float(metrics.get("cpu_usage", 0.0))
            if cpu_usage >= cpu_thr:
                alerts.append({"type": "cpu_high", "value": cpu_usage, "threshold": cpu_thr})
        except Exception:
            pass

        # Check for high memory usage alert
        try:
            memory_usage = float(metrics.get("memory_usage", 0.0))
            if memory_usage >= mem_thr:
                alerts.append({"type": "memory_high", "value": memory_usage, "threshold": mem_thr})
        except Exception:
            pass

        # Check for low throughput alert
        try:
            throughput = float(metrics.get("throughput", 0.0))
            if throughput > 0 and throughput < thr_low:
                alerts.append({"type": "throughput_low", "value": throughput, "threshold": thr_low})
        except Exception:
            pass
    except Exception:
        pass
    return alerts
