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
from typing import Dict, Any
from utils import PiscesLxCoreLog
from .metrics import PiscesLxCoreMetricsRegistry

TRY_EXPORTERS = ("prom", "otlp")

def _read_gauge(reg: PiscesLxCoreMetricsRegistry, name: str, labels: Dict[str, str]) -> float:
    """Retrieves the value of a gauge metric with specified labels.

    If an exception occurs during the retrieval process, returns 0.0.

    Args:
        reg (PiscesLxCoreMetricsRegistry): The metrics registry instance.
        name (str): The name of the gauge metric.
        labels (Dict[str, str]): The labels of the gauge metric.

    Returns:
        float: The value of the gauge metric. Returns 0.0 if an exception occurs.
    """
    try:
        # Get or create a gauge metric with the "exporter" label dimension
        g = reg.gauge(name, labels=["exporter"])
        return float(g.snapshot(labels))
    except Exception:
        return 0.0

def _read_gauge_simple(reg: PiscesLxCoreMetricsRegistry, name: str) -> float:
    """Retrieves the value of a gauge metric without labels.

    If an exception occurs during the retrieval process, returns 0.0.

    Args:
        reg (PiscesLxCoreMetricsRegistry): The metrics registry instance.
        name (str): The name of the gauge metric.

    Returns:
        float: The value of the gauge metric. Returns 0.0 if an exception occurs.
    """
    try:
        # Get or create a gauge metric without labels
        g = reg.gauge(name)
        return float(g.snapshot({}))
    except Exception:
        return 0.0

def human_ts(ts: float) -> str:
    """Converts a timestamp to a human-readable string.

    If the timestamp is 0, returns "n/a". If an exception occurs during conversion,
    returns the original timestamp as a string.

    Args:
        ts (float): The timestamp to convert.

    Returns:
        str: A human-readable timestamp string. Returns "n/a" if the timestamp is 0,
             or the original timestamp string if an exception occurs.
    """
    if not ts:
        return "n/a"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return str(ts)

def check_exporters(reg: PiscesLxCoreMetricsRegistry) -> Dict[str, Any]:
    """Checks the status of all exporters defined in TRY_EXPORTERS.

    Retrieves the last success and error timestamps for each exporter,
    and converts them to human-readable format.

    Args:
        reg (PiscesLxCoreMetricsRegistry): The metrics registry instance.

    Returns:
        Dict[str, Any]: A dictionary containing the last success and error timestamps
                        of each exporter, both in raw and human-readable formats.
    """
    info: Dict[str, Any] = {}
    for exp in TRY_EXPORTERS:
        last_ok = _read_gauge(reg, "exporter.last_success_timestamp_seconds", {"exporter": exp})
        last_err = _read_gauge(reg, "exporter.last_error_timestamp_seconds", {"exporter": exp})
        info[exp] = {
            "last_success": last_ok,
            "last_success_h": human_ts(last_ok),
            "last_error": last_err,
            "last_error_h": human_ts(last_err),
        }
    return info

def check_jsonl(reg: PiscesLxCoreMetricsRegistry) -> Dict[str, Any]:
    """Performs a best-effort check of the JSONL output status by accessing internal fields.

    If an exception occurs during the check, logs the error and returns an empty dictionary.

    Args:
        reg (PiscesLxCoreMetricsRegistry): The metrics registry instance.

    Returns:
        Dict[str, Any]: A dictionary containing information about the JSONL output,
                        such as path, size, modification time, and rotation settings.
                        Returns an empty dictionary if an exception occurs.
    """
    out = {}
    try:
        path = getattr(reg, "_out_path", None)
        out["path"] = path or "n/a"
        if path and os.path.exists(path):
            st = os.stat(path)
            out["size_bytes"] = st.st_size
            out["mtime_h"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
        else:
            out["size_bytes"] = 0
            out["mtime_h"] = "n/a"
        out["rotate_max_bytes"] = getattr(reg, "_rotate_max_bytes", None)
        out["rotate_max_age_sec"] = getattr(reg, "_rotate_max_age_sec", None)
    except Exception as e:
        log = PiscesLxCoreLog("pisceslx.observability.selfcheck")
        log.error("Failed to check JSONL output status", error=str(e))
    return out

def main() -> int:
    """Main function to perform a self-check of observability components.

    Logs information about exporters, JSONL output, and system rates.
    Returns 0 to indicate successful execution.

    Returns:
        int: 0 indicating successful execution.
    """
    reg = PiscesLxCoreMetricsRegistry.instance()
    log = PiscesLxCoreLog("pisceslx.observability.selfcheck")

    log.info("selfcheck.start")
    exporters = check_exporters(reg)
    for name, meta in exporters.items():
        enabled = (os.environ.get("PISCES_OBS_DISABLE_EXPORTERS", "0").lower() in ("0", "false"))
        if name == "otlp":
            endpoint = os.environ.get("PISCES_OTLP_ENDPOINT") or "n/a"
            proto = os.environ.get("PISCES_OTLP_PROTOCOL", "json")
            log.info("selfcheck.exporter", name=name, enabled=enabled, endpoint=endpoint, protocol=proto, last_ok=meta.get("last_success_h"), last_err=meta.get("last_error_h"))
        elif name == "prom":
            base_dir = os.environ.get("PISCES_BASE_DIR", "<auto>")
            log.info("selfcheck.exporter", name=name, enabled=enabled, base_dir=base_dir, last_ok=meta.get("last_success_h"), last_err=meta.get("last_error_h"))

    js = check_jsonl(reg)
    log.info("selfcheck.jsonl", path=js.get("path"), size_bytes=js.get("size_bytes"), mtime=js.get("mtime_h"), rotate_max_bytes=js.get("rotate_max_bytes"), rotate_max_age_sec=js.get("rotate_max_age_sec"))

    # Rates summary (best-effort)
    rates = {
        "net_rx_bps": _read_gauge_simple(reg, "runtime_net_rx_bytes_per_sec"),
        "net_tx_bps": _read_gauge_simple(reg, "runtime_net_tx_bytes_per_sec"),
        "disk_read_bps": _read_gauge_simple(reg, "runtime_disk_read_bytes_per_sec"),
        "disk_write_bps": _read_gauge_simple(reg, "runtime_disk_write_bytes_per_sec"),
        "disk_read_iops": _read_gauge_simple(reg, "runtime_disk_read_iops"),
        "disk_write_iops": _read_gauge_simple(reg, "runtime_disk_write_iops"),
    }
    log.info("selfcheck.rates", rates)

    log.success("selfcheck.done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
