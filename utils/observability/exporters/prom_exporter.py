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
import json
import time
import threading
from utils import PiscesLxCoreLog
from typing import Dict, Any, Optional, List
from ..metrics import PiscesLxCoreMetricsRegistry

class PromTextfileExporter:
    """Exporter that periodically writes metrics to a text file in Prometheus format."""

    def __init__(self, output_path: str, interval_sec: int = 15) -> None:
        """Initialize the PromTextfileExporter instance.

        Args:
            output_path (str): Path to the output file where metrics will be written.
            interval_sec (int, optional): Interval in seconds between each write operation. Defaults to 15.
        """
        self.output_path = output_path
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._log = PiscesLxCoreLog("pisceslx.exporter.prom")

    def start(self) -> None:
        """Start the exporter thread.
        
        If the thread is already running, this method does nothing.
        Creates the necessary directory for the output file and starts the thread that periodically writes metrics.
        """
        if self._thread and self._thread.is_alive():
            return
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        try:
            self._log.info("exporter.prom.start", path=self.output_path, interval=self.interval_sec)
        except Exception as e:
            self._log.error("Failed to log exporter start event", error=str(e))

    def stop(self) -> None:
        """Stop the exporter thread.
        
        Sets the stop event to signal the thread to stop and logs the stop event.
        """
        self._stop.set()
        try:
            self._log.info("exporter.prom.stop")
        except Exception as e:
            self._log.error("Failed to log exporter stop event", error=str(e))

    def _loop(self) -> None:
        """Main loop of the exporter thread.
        
        Periodically fetches metrics, renders them in Prometheus format, and writes them to the output file.
        Implements exponential backoff in case of errors.
        """
        reg = PiscesLxCoreMetricsRegistry.instance()
        max_backoff = 60.0
        try:
            env_val = os.environ.get("PISCES_EXPORTER_BACKOFF_MAX_SEC")
            if env_val:
                max_backoff = max(1.0, float(env_val))
        except Exception as e:
            self._log.error("Failed to parse PISCES_EXPORTER_BACKOFF_MAX_SEC environment variable", error=str(e))
        backoff = 1.0
        while not self._stop.is_set():
            try:
                _t0 = time.time()
                snap = reg.snapshot_grouped()
                meta = reg.get_metric_info()
                hraw = reg.histograms_snapshot_raw()
                content = self._render_prom_text(snap, meta, hraw)
                with self._lock:
                    with open(self.output_path, "w", encoding="utf-8") as f:
                        f.write(content)
                backoff = 1.0  # reset on success
                try:
                    reg.gauge("exporter.last_success_timestamp_seconds", help_text="Last success epoch seconds", labels=["exporter"]).set(time.time(), labels={"exporter": "prom"})
                except Exception as e:
                    self._log.error("Failed to set last success timestamp metric", error=str(e))
                try:
                    dur = max(0.0, time.time() - _t0)
                    reg.observe("exporter_duration_seconds", dur, labels={"exporter": "prom", "status": "success"})
                except Exception as e:
                    self._log.error("Failed to observe successful export duration", error=str(e))
                try:
                    self._log.success("exporter.prom.flush.ok", bytes=len(content))
                except Exception as e:
                    self._log.error("Failed to log successful export", error=str(e))
            except Exception as e:
                try:
                    # exporter error metric
                    reg.inc("exporter.errors", 1.0, labels={"exporter": "prom"})
                except Exception as ex:
                    self._log.error("Failed to increment exporter error metric", error=str(ex))
                try:
                    reg.gauge("exporter.last_error_timestamp_seconds", help_text="Last error epoch seconds", labels=["exporter"]).set(time.time(), labels={"exporter": "prom"})
                except Exception as ex:
                    self._log.error("Failed to set last error timestamp metric", error=str(ex))
                try:
                    # duration for failed attempt (best-effort)
                    dur = max(0.0, time.time() - _t0) if '_t0' in locals() else 0.0
                    reg.observe("exporter_duration_seconds", dur, labels={"exporter": "prom", "status": "error"})
                except Exception as ex:
                    self._log.error("Failed to observe failed export duration", error=str(ex))
                try:
                    self._log.error("exporter.prom.flush.error")
                except Exception as ex:
                    self._log.error("Failed to log export error", error=str(ex))
                # backoff wait
                self._stop.wait(backoff)
                backoff = min(max_backoff, backoff * 2.0)
                continue
            # normal wait
            self._stop.wait(self.interval_sec)

    def _render_prom_text(self, snap: Dict[str, Any], meta: Dict[str, Any], hraw: Dict[str, Any]) -> str:
        """Render metrics into Prometheus text format.

        Args:
            snap (Dict[str, Any]): Grouped snapshot of metrics.
            meta (Dict[str, Any]): Metric metadata including type and help text.
            hraw (Dict[str, Any]): Raw snapshot of histogram metrics.

        Returns:
            str: Metrics in Prometheus text format.
        """
        lines: List[str] = []
        # Add HELP and TYPE information for each metric
        for name, info in (meta or {}).items():
            metric_type = info.get("type")
            help_text = (info.get("help") or "").replace("\n", " ")
            mname = self._sanitize(name)
            if metric_type in ("counter", "gauge"):
                if help_text:
                    lines.append(f"# HELP {mname} {help_text}")
                lines.append(f"# TYPE {mname} {metric_type}")
            elif metric_type == "histogram":
                if help_text:
                    lines.append(f"# HELP {mname} {help_text}")
                lines.append(f"# TYPE {mname} histogram")
        # Add gauge metrics
        for name, val in (snap.get("gauges") or {}).items():
            try:
                v = float(val)
            except (ValueError, TypeError):
                continue
            lines.append(f"{self._sanitize(name)} {v}")
        # Add counter metrics
        for name, val in (snap.get("counters") or {}).items():
            try:
                v = float(val)
            except (ValueError, TypeError):
                continue
            base = self._sanitize(name)
            if not base.endswith("_total"):
                base = base + "_total"
            lines.append(f"{base} {v}")
        # Add histogram metrics
        for name, label_map in (hraw or {}).items():
            base = self._sanitize(name)
            if not isinstance(label_map, dict):
                continue
            for label_key, data in label_map.items():
                try:
                    buckets = data.get("buckets", {})
                    total_count = int(data.get("count", 0))
                    total_sum = float(data.get("sum", 0.0))
                except (ValueError, TypeError):
                    continue
                # Render cumulative buckets including +Inf
                for le in sorted(buckets.keys()):
                    try:
                        c = int(buckets[le])
                    except (ValueError, TypeError):
                        continue
                    lines.append(f"{base}_bucket{self._append_label(label_key, 'le', str(le))} {c}")
                # +Inf bucket equals to total_count
                lines.append(f"{base}_bucket{self._append_label(label_key, 'le', '+Inf')} {total_count}")
                # Add sum and count metrics
                lines.append(f"{base}_sum{label_key or ''} {total_sum}")
                lines.append(f"{base}_count{label_key or ''} {total_count}")
        return "\n".join(lines) + "\n"

    def _sanitize(self, name: str) -> str:
        """Sanitize metric names to comply with Prometheus naming rules.

        Prometheus metric name rules: [a-zA-Z_:][a-zA-Z0-9_:]*

        Args:
            name (str): The original metric name.

        Returns:
            str: The sanitized metric name.
        """
        s = name.replace(".", "_").replace("-", "_")
        return s

    def _append_label(self, label_key: str, k: str, v: str) -> str:
        """Append or construct a label string by injecting a key-value pair into an existing label string.

        Args:
            label_key (str): Existing label string, e.g., '{a="b"}'.
            k (str): New label key.
            v (str): New label value.

        Returns:
            str: New label string with the injected key-value pair.
        """
        if not label_key:
            return "{" + f"{k}={json.dumps(v)}" + "}"
        # Inject the new key-value pair before the last '}'
        try:
            if label_key.startswith("{") and label_key.endswith("}"):
                inner = label_key[1:-1]
                sep = "," if inner.strip() else ""
                return "{" + inner + sep + f"{k}={json.dumps(v)}" + "}"
        except Exception as e:
            self._log.error("Failed to inject new label key-value pair, using fallback", error=str(e))
        # Fallback: construct label string only with the new key-value pair
        return "{" + f"{k}={json.dumps(v)}" + "}"
