#!/usr/bin/env/python3

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

import os
import re
import time
import json
import threading
from pathlib import Path
from utils import PiscesLxCoreLog
from collections import defaultdict
from utils.error import PiscesLxCoreObservabilityError
from typing import Dict, Any, Optional, List, Union, DefaultDict

_LOGGER = PiscesLxCoreLog("pisceslx.metrics")

_LABEL_NAME_REGEX = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
_MAX_LABEL_VALUE_LENGTH = 1024

def _validate_label_name(name: str) -> None:
    """
    Validates if a label name complies with the Prometheus naming standard.

    Args:
        name (str): The label name to be validated.

    Raises:
        ValueError: If the label name does not match the Prometheus naming standard.
    """
    if not _LABEL_NAME_REGEX.match(name):
        raise ValueError(f"Invalid label name: {name}")

def _format_labels(labels: Optional[Dict[str, str]]) -> str:
    """
    Formats a dictionary of labels into a Prometheus-style string.

    Args:
        labels (Optional[Dict[str, str]]): A dictionary containing label names and their corresponding values. 
                                         Can be None.

    Returns:
        str: A formatted string of labels enclosed in curly braces. Returns an empty string if labels is None or empty.

    Raises:
        ValueError: If the label name is invalid or the label value exceeds the maximum allowed length.
    """
    if not labels:
        return ""
    
    # Validate each label name and check value length
    for name, value in labels.items():
        _validate_label_name(name)
        if len(str(value)) > _MAX_LABEL_VALUE_LENGTH:
            raise ValueError(f"Label value too long: {name}={value}")
    
    # Sort labels by name to ensure consistent output
    sorted_labels = sorted(labels.items())
    # Format each label as 'name="value"' with proper escaping
    label_pairs = []
    for name, value in sorted_labels:
        escaped_value = str(value).replace('"', '\"')
        label_pairs.append(f'{name}="{escaped_value}"')
    return "{" + ",".join(label_pairs) + "}"

class Counter:
    """
    A thread-safe counter that supports labels. Each label combination maintains an independent count.
    """

    def __init__(self, name: str, help_text: str = "", labels: Optional[List[str]] = None) -> None:
        """
        Initialize a Counter instance.

        Args:
            name (str): The name of the counter.
            help_text (str, optional): Help text describing the counter. Defaults to an empty string.
            labels (Optional[List[str]], optional): List of label names. Defaults to None.

        Raises:
            ValueError: If any label name does not match the Prometheus naming standard.
        """
        self.name = name
        self.help_text = help_text
        self.labels = labels or []
        self._lock = threading.RLock()
        self._values: DefaultDict[str, float] = defaultdict(float)
        
        # Validate each label name to ensure it complies with the Prometheus naming standard
        for label in self.labels:
            _validate_label_name(label)

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increase the counter value by the specified amount for the given label combination.

        Args:
            amount (float, optional): The amount to increase the counter by. Defaults to 1.0.
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.

        Raises:
            ValueError: If a label name is invalid or the label value exceeds the maximum allowed length.
        """
        with self._lock:
            label_key = _format_labels(labels)
            self._values[label_key] += amount

    def snapshot(self, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Get the current value of the counter for the specified label combination.

        Args:
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.

        Returns:
            float: The current value of the counter for the specified label combination.

        Raises:
            ValueError: If a label name is invalid or the label value exceeds the maximum allowed length.
        """
        with self._lock:
            label_key = _format_labels(labels)
            return self._values[label_key]

    def snapshot_all(self) -> Dict[str, float]:
        """
        Get snapshots of all label combinations and their corresponding counter values.

        Returns:
            Dict[str, float]: A dictionary containing all label combinations and their corresponding counter values.
        """
        with self._lock:
            return dict(self._values)

    def reset(self, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Reset the counter value. If labels are provided, reset only the value for that label combination; 
        otherwise, reset all values.

        Args:
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. 
                If None, reset all counter values. Defaults to None.

        Raises:
            ValueError: If a label name is invalid or the label value exceeds the maximum allowed length.
        """
        with self._lock:
            if labels is None:
                self._values.clear()
            else:
                label_key = _format_labels(labels)
                self._values[label_key] = 0.0


class Gauge:
    """
    An enterprise-grade thread-safe gauge that supports labels.
    """

    def __init__(self, name: str, help_text: str = "", labels: Optional[List[str]] = None) -> None:
        """
        Initialize a Gauge instance.

        Args:
            name (str): The name of the gauge.
            help_text (str, optional): Help text for the gauge. Defaults to "".
            labels (Optional[List[str]], optional): List of label names. Defaults to None.
        """
        self.name = name
        self.help_text = help_text
        self.labels = labels or []
        self._lock = threading.RLock()
        self._values: Dict[str, float] = {}
        
        for label in self.labels:
            _validate_label_name(label)

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set the gauge value.

        Args:
            value (float): The value to set.
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.
        """
        with self._lock:
            label_key = _format_labels(labels)
            self._values[label_key] = value

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increase the gauge value.

        Args:
            amount (float, optional): The amount to increase. Defaults to 1.0.
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.
        """
        with self._lock:
            label_key = _format_labels(labels)
            self._values[label_key] = self._values.get(label_key, 0.0) + amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Decrease the gauge value.

        Args:
            amount (float, optional): The amount to decrease. Defaults to 1.0.
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.
        """
        with self._lock:
            label_key = _format_labels(labels)
            self._values[label_key] = self._values.get(label_key, 0.0) - amount

    def snapshot(self, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Get the current value of the gauge with specified labels.

        Args:
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.

        Returns:
            float: The current value of the gauge.
        """
        with self._lock:
            label_key = _format_labels(labels)
            return self._values.get(label_key, 0.0)

    def snapshot_all(self) -> Dict[str, float]:
        """
        Get snapshots of all label combinations.

        Returns:
            Dict[str, float]: A dictionary containing all label combinations and their corresponding values.
        """
        with self._lock:
            return self._values.copy()

    def reset(self, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Reset the gauge.

        Args:
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. If None, reset all values. Defaults to None.
        """
        with self._lock:
            if labels is None:
                self._values.clear()
            else:
                label_key = _format_labels(labels)
                if label_key in self._values:
                    del self._values[label_key]

class Histogram:
    """
    A thread-safe histogram that supports labels and configurable buckets.
    """

    def __init__(self, name: str, buckets: Optional[List[float]] = None, help_text: str = "", 
                 labels: Optional[List[str]] = None) -> None:
        """
        Initialize a Histogram instance.

        Args:
            name (str): The name of the histogram.
            buckets (Optional[List[float]], optional): List of bucket values. If None, default buckets will be used.
            help_text (str, optional): Help text describing the histogram. Defaults to an empty string.
            labels (Optional[List[str]], optional): List of label names. Defaults to an empty list.

        Raises:
            ValueError: If any label name does not match the Prometheus naming standard.
        """
        self.name = name
        self.help_text = help_text
        self.labels = labels or []
        self._lock = threading.RLock()
        
        # Use default buckets if no buckets are provided
        self._buckets = buckets or [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 
            2.5, 5.0, 7.5, 10.0, 25.0, 50.0, 75.0, 100.0, 250.0, 500.0, 1000.0
        ]
        
        # Sort and remove duplicate bucket values
        self._buckets = sorted(set(self._buckets))
        
        # Store histogram data for each label combination
        self._data: Dict[str, Dict[str, Union[float, Dict[float, float]]]] = {}
        
        # Validate each label name to ensure it complies with the Prometheus naming standard
        for label in self.labels:
            _validate_label_name(label)

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record an observed value in the histogram.

        Args:
            value (float): The observed value to record.
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.

        Raises:
            ValueError: If a label name is invalid or the label value exceeds the maximum allowed length.
        """
        with self._lock:
            label_key = _format_labels(labels)
            
            # Initialize data for the label combination if it doesn't exist
            if label_key not in self._data:
                self._data[label_key] = {
                    "sum": 0.0,
                    "count": 0.0,
                    "buckets": {b: 0.0 for b in self._buckets}
                }
            
            data = self._data[label_key]
            data["sum"] += value
            data["count"] += 1
            
            # Increment the count of each bucket that includes the observed value
            for b in self._buckets:
                if value <= b:
                    data["buckets"][b] += 1

    def snapshot(self, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """
        Get a snapshot of the histogram data for the specified label combination.

        Args:
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.

        Returns:
            Dict[str, float]: A dictionary containing statistical information including sum, count, 
                            average, and percentiles (p50, p95, p99).

        Raises:
            ValueError: If a label name is invalid or the label value exceeds the maximum allowed length.
        """
        with self._lock:
            label_key = _format_labels(labels)
            
            if label_key not in self._data:
                return {
                    "sum": 0.0,
                    "count": 0.0,
                    "buckets": {b: 0.0 for b in self._buckets},
                    "avg": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                }
            
            data = self._data[label_key]
            count = data["count"]
            
            p50 = self._calculate_percentile(data["buckets"], count, 0.5)
            p95 = self._calculate_percentile(data["buckets"], count, 0.95)
            p99 = self._calculate_percentile(data["buckets"], count, 0.99)
            
            return {
                "sum": data["sum"],
                "count": count,
                "buckets": data["buckets"].copy(),
                "avg": data["sum"] / count if count > 0 else 0.0,
                "p50": p50,
                "p95": p95,
                "p99": p99,
            }

    def snapshot_all(self) -> Dict[str, Dict[str, float]]:
        """
        Get snapshots of the histogram data for all label combinations.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary containing snapshots of all label combinations,
                                       each with statistical information.
        """
        with self._lock:
            result = {}
            for label_key, data in self._data.items():
                count = data["count"]
                p50 = self._calculate_percentile(data["buckets"], count, 0.5)
                p95 = self._calculate_percentile(data["buckets"], count, 0.95)
                p99 = self._calculate_percentile(data["buckets"], count, 0.99)
                
                result[label_key] = {
                    "sum": data["sum"],
                    "count": count,
                    "buckets": data["buckets"].copy(),
                    "avg": data["sum"] / count if count > 0 else 0.0,
                    "p50": p50,
                    "p95": p95,
                    "p99": p99,
                }
            return result

    def _calculate_percentile(self, buckets: Dict[float, float], total_count: float, percentile: float) -> float:
        """
        Calculate the specified percentile based on bucket counts.

        Args:
            buckets (Dict[float, float]): A dictionary of bucket values and their corresponding counts.
            total_count (float): The total count of all observations.
            percentile (float): The percentile to calculate, a value between 0 and 1.

        Returns:
            float: The calculated percentile value.
        """
        if total_count == 0:
            return 0.0
            
        target_count = total_count * percentile
        cumulative_count = 0.0
        
        # Iterate through sorted buckets to find the percentile value
        for bucket_value in sorted(buckets.keys()):
            cumulative_count += buckets[bucket_value]
            if cumulative_count >= target_count:
                return bucket_value
                
        return self._buckets[-1] if self._buckets else 0.0

    def reset(self, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Reset the histogram data. If labels are provided, reset only the data for that label combination;
        otherwise, reset all data.

        Args:
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. 
                If None, reset all histogram data. Defaults to None.

        Raises:
            ValueError: If a label name is invalid or the label value exceeds the maximum allowed length.
        """
        with self._lock:
            if labels is None:
                self._data.clear()
            else:
                label_key = _format_labels(labels)
                if label_key in self._data:
                    del self._data[label_key]

class PiscesLxCoreMetricsRegistry:
    """
    A registry for managing metrics, including counters, gauges, and histograms.
    It supports thread-safe operations and periodic flushing of metrics to files.
    """

    def __init__(self) -> None:
        """
        Initialize a PiscesLxCoreMetricsRegistry instance.
        Set up storage for metrics, threading locks, and configuration parameters.
        """
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self._lock = threading.RLock()
        self._flush_thread: Optional[threading.Thread] = None
        self._flush_stop_event = threading.Event()
        self._out_path: Optional[str] = None
        self._flush_interval: float = 10.0
        try:
            # Get the maximum file size for rotation from environment variable, default to 10MB
            self._rotate_max_bytes: int = int(float(os.environ.get("PISCES_METRICS_ROTATE_MB", "10")) * 1024 * 1024)
        except Exception:
            self._rotate_max_bytes = 10 * 1024 * 1024
        try:
            # Get the maximum file age for rotation from environment variable, default to 60 minutes
            self._rotate_max_age_sec: int = int(float(os.environ.get("PISCES_METRICS_ROTATE_AGE_MIN", "60")) * 60)
        except Exception:
            self._rotate_max_age_sec = 60 * 60
        self._last_rotate_time: float = time.time()
        self._global_labels = self._parse_global_labels(os.environ.get("PISCES_GLOBAL_LABELS", ""))

    # ---- singleton accessor (for compatibility with callers using .instance()) ----
    _global_instance: Optional["PiscesLxCoreMetricsRegistry"] = None
    _global_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "PiscesLxCoreMetricsRegistry":
        """
        Get the singleton instance of PiscesLxCoreMetricsRegistry.

        Returns:
            PiscesLxCoreMetricsRegistry: The singleton instance of the registry.
        """
        with cls._global_lock:
            if cls._global_instance is None:
                cls._global_instance = cls()
            return cls._global_instance

    def counter(self, name: str, help_text: str = "", labels: Optional[List[str]] = None) -> Counter:
        """
        Get or create a counter. If the counter does not exist, create a new one.

        Args:
            name (str): The name of the counter.
            help_text (str, optional): Help text for the counter. Defaults to "".
            labels (Optional[List[str]], optional): List of label names. Defaults to None.

        Returns:
            Counter: A wrapped Counter instance.
        """
        with self._lock:
            if name not in self.counters:
                self.counters[name] = Counter(name, help_text, labels)
            return _CounterWrapper(self, self.counters[name])

    def gauge(self, name: str, help_text: str = "", labels: Optional[List[str]] = None) -> Gauge:
        """
        Get or create a gauge. If the gauge does not exist, create a new one.

        Args:
            name (str): The name of the gauge.
            help_text (str, optional): Help text for the gauge. Defaults to "".
            labels (Optional[List[str]], optional): List of label names. Defaults to None.

        Returns:
            Gauge: A wrapped Gauge instance.
        """
        with self._lock:
            if name not in self.gauges:
                self.gauges[name] = Gauge(name, help_text, labels)
            return _GaugeWrapper(self, self.gauges[name])

    def histogram(self, name: str, buckets: Optional[List[float]] = None, help_text: str = "", 
                  labels: Optional[List[str]] = None) -> Histogram:
        """
        Get or create a histogram. If the histogram does not exist, create a new one.

        Args:
            name (str): The name of the histogram.
            buckets (Optional[List[float]], optional): List of bucket values. Defaults to None.
            help_text (str, optional): Help text for the histogram. Defaults to "".
            labels (Optional[List[str]], optional): List of label names. Defaults to None.

        Returns:
            Histogram: A wrapped Histogram instance.
        """
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(name, buckets, help_text, labels)
            return _HistogramWrapper(self, self.histograms[name])

    def set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set the value of a gauge. If the gauge does not exist, it will be created.

        Args:
            name (str): The name of the gauge.
            value (float): The value to set.
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.
        """
        self.gauge(name).set(value, self._merge_global_labels(labels))

    def inc(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increase the value of a counter. If the counter does not exist, it will be created.

        Args:
            name (str): The name of the counter.
            amount (float, optional): The amount to increase. Defaults to 1.0.
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.
        """
        self.counter(name).inc(amount, self._merge_global_labels(labels))

    def observe(self, name: str, value: float, buckets: Optional[List[float]] = None, 
                labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record an observed value in a histogram. If the histogram does not exist, it will be created.

        Args:
            name (str): The name of the histogram.
            value (float): The observed value to record.
            buckets (Optional[List[float]], optional): List of bucket values. Defaults to None.
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.
        """
        self.histogram(name, buckets).observe(value, self._merge_global_labels(labels))

    def snapshot(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of all metrics.

        Returns:
            Dict[str, Any]: A dictionary containing snapshots of all counters, gauges, and histograms.
            If a metric has no data, it will return a default value.
        """
        with self._lock:
            out: Dict[str, Any] = {}
            
            for k, c in self.counters.items():
                snap = c.snapshot_all()
                if snap:
                    out[k] = snap
                else:
                    out[k] = 0.0
                    
            for k, g in self.gauges.items():
                snap = g.snapshot_all()
                if snap:
                    out[k] = snap
                else:
                    out[k] = 0.0
                    
            for k, h in self.histograms.items():
                snap = h.snapshot_all()
                if snap:
                    out[k] = snap
                else:
                    out[k] = {
                        "sum": 0.0,
                        "count": 0.0,
                        "buckets": {},
                        "avg": 0.0,
                        "p50": 0.0,
                        "p95": 0.0,
                        "p99": 0.0,
                    }
                    
            return out

    def histograms_snapshot_raw(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Return raw snapshots of all histograms (including all label combinations) for exporters to convert as needed.
        Structure:
        { hist_name: { label_key: {"sum": value, "count": value, "buckets":{le:count}, ... }, ... }, ... }

        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: A dictionary containing raw snapshots of all histograms.
            If an error occurs, an empty dictionary will be returned for that histogram.
        """
        with self._lock:
            result: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for name, h in self.histograms.items():
                try:
                    snap = h.snapshot_all()
                    result[name] = snap
                except Exception:
                    result[name] = {}
            return result

    def start_periodic_flush(self, output_dir: str, filename: str = "metrics.jsonl", interval_sec: int = 10) -> None:
        """
        Start a background thread to periodically flush snapshots to a JSONL file.

        Args:
            output_dir (str): The output directory for the JSONL file.
            filename (str, optional): The name of the JSONL file. Defaults to "metrics.jsonl".
            interval_sec (int, optional): The flush interval in seconds. Defaults to 10.
        """
        with self._lock:
            if self._flush_thread is not None and self._flush_thread.is_alive():
                _LOGGER.debug("Flush thread already running")
                return
                
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                _LOGGER.error("Failed to create output directory", dir=output_dir, error=str(e))
                
            self._out_path = os.path.join(output_dir, filename)
            self._flush_interval = float(interval_sec)
            self._flush_stop_event.clear()

            self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True, name="metrics-flush")
            self._flush_thread.start()
            _LOGGER.info("Started metrics flush thread", path=self._out_path, interval=interval_sec)

    def stop_periodic_flush(self) -> None:
        """
        Stop the background flush thread.
        """
        with self._lock:
            if self._flush_thread is not None:
                self._flush_stop_event.set()
                self._flush_thread.join(timeout=5.0)
                self._flush_thread = None
                self._flush_stop_event.clear()
                _LOGGER.info("Stopped metrics flush thread")

    def _flush_loop(self) -> None:
        """
        The background flush loop. Periodically call flush_once until the stop event is set.
        """
        while not self._flush_stop_event.is_set():
            try:
                self.flush_once()
            except Exception as e:
                _LOGGER.error("Failed to flush metrics", error=str(e), error_class=type(e).__name__)
            self._flush_stop_event.wait(self._flush_interval)

    def flush_once(self) -> None:
        """
        Perform a single flush to the file.
        Take a snapshot of all metrics and write it to the output file.
        Check if the file needs to be rotated based on size and age.
        """
        if not self._out_path:
            return
            
        ts = time.time()
        snapshot = {
            "ts": ts,
            "counters": {k: v.snapshot_all() for k, v in self.counters.items()},
            "gauges": {k: v.snapshot_all() for k, v in self.gauges.items()},
            "histograms": {k: v.snapshot_all() for k, v in self.histograms.items()},
        }
        line = json.dumps(snapshot, ensure_ascii=False)
        
        try:
            with self._lock:
                try:
                    need_rotate = False
                    if os.path.exists(self._out_path):
                        sz = os.path.getsize(self._out_path)
                        if sz >= self._rotate_max_bytes:
                            need_rotate = True
                    if (time.time() - self._last_rotate_time) >= self._rotate_max_age_sec:
                        need_rotate = True
                    if need_rotate:
                        base = Path(self._out_path)
                        ts_suffix = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                        rotated = base.with_name(f"{base.stem}-{ts_suffix}{base.suffix}")
                        try:
                            os.replace(str(base), str(rotated))
                        except FileNotFoundError:
                            _LOGGER.warning("File not found during rotation", path=str(base))
                        self._last_rotate_time = time.time()
                except Exception as e:
                    _LOGGER.warning("metrics rotation check failed", error=str(e), error_class=type(e).__name__)
                with open(self._out_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            _LOGGER.error("Failed to write metrics to file", path=self._out_path, error=str(e), error_class=type(e).__name__)

    def snapshot_grouped(self) -> Dict[str, Any]:
        """
        Return a grouped snapshot structure for exporters to use.
        Structure:
        {
          "counters": {name: total_value},
          "gauges": {name: avg_value},
          "histograms": {name: {"sum": total_sum, "count": total_count}}
        }

        Returns:
            Dict[str, Any]: A grouped snapshot of all metrics.
        """
        with self._lock:
            out: Dict[str, Any] = {"counters": {}, "gauges": {}, "histograms": {}}
            for k, c in self.counters.items():
                snap = c.snapshot_all()
                if isinstance(snap, dict):
                    total = 0.0
                    for v in snap.values():
                        try:
                            total += float(v)
                        except Exception:
                            pass
                    out["counters"][k] = total
                else:
                    try:
                        out["counters"][k] = float(snap)
                    except Exception:
                        out["counters"][k] = 0.0
            for k, g in self.gauges.items():
                snap = g.snapshot_all()
                if isinstance(snap, dict) and snap:
                    vals = []
                    for v in snap.values():
                        try:
                            vals.append(float(v))
                        except Exception:
                            pass
                    out["gauges"][k] = (sum(vals) / len(vals)) if vals else 0.0
                else:
                    try:
                        out["gauges"][k] = float(snap)
                    except Exception:
                        out["gauges"][k] = 0.0
            for k, h in self.histograms.items():
                snap = h.snapshot_all()
                total_sum = 0.0
                total_count = 0.0
                if isinstance(snap, dict):
                    for v in snap.values():
                        try:
                            total_sum += float(v.get("sum", 0.0))
                            total_count += float(v.get("count", 0.0))
                        except Exception:
                            pass
                out["histograms"][k] = {"sum": total_sum, "count": total_count}
            return out

    def reset_all(self) -> None:
        """
        Reset all metrics to their initial state.
        """
        with self._lock:
            for counter in self.counters.values():
                counter.reset()
            for gauge in self.gauges.values():
                gauge.reset()
            for histogram in self.histograms.values():
                histogram.reset()
            _LOGGER.info("Reset all metrics")

    def get_metric_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata of all metrics.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing metadata of all counters, gauges, and histograms,
            including type, help text, labels, and buckets (for histograms).
        """
        with self._lock:
            info = {}
            
            for name, counter in self.counters.items():
                info[name] = {
                    "type": "counter",
                    "help": counter.help_text,
                    "labels": counter.labels,
                }
                
            for name, gauge in self.gauges.items():
                info[name] = {
                    "type": "gauge", 
                    "help": gauge.help_text,
                    "labels": gauge.labels,
                }
                
            for name, histogram in self.histograms.items():
                info[name] = {
                    "type": "histogram",
                    "help": histogram.help_text,
                    "labels": histogram.labels,
                    "buckets": histogram._buckets,
                }
                
            return info

    def _parse_global_labels(self, s: str) -> Dict[str, str]:
        """
        Parse global labels from a string. The string can contain label pairs separated by commas,
        with each pair separated by ':' or '='.

        Args:
            s (str): The string containing global labels.

        Returns:
            Dict[str, str]: A dictionary of parsed global labels.
        """
        labels: Dict[str, str] = {}
        try:
            s = (s or "").strip()
            if not s:
                return labels
            parts = [p for p in s.split(',') if p.strip()]
            for p in parts:
                if ':' in p:
                    k, v = p.split(':', 1)
                elif '=' in p:
                    k, v = p.split('=', 1)
                else:
                    continue
                k = k.strip()
                v = v.strip()
                if not k:
                    continue
                _validate_label_name(k)
                labels[k] = v
        except Exception as e:
            _LOGGER.warning("invalid PISCES_GLOBAL_LABELS", error=str(e))
        return labels

    def _merge_global_labels(self, labels: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """
        Merge global labels with the given labels. Global labels will be overridden by the given labels if there are conflicts.

        Args:
            labels (Optional[Dict[str, str]]): The labels to merge with global labels.

        Returns:
            Optional[Dict[str, str]]: The merged labels. If there are no global labels, return the original labels.
        """
        if not self._global_labels:
            return labels
        merged = dict(self._global_labels)
        if labels:
            merged.update(labels)
        return merged

class _CounterWrapper:
    """
    A wrapper class for the Counter that enforces the inclusion of global labels in all operations.
    """
    def __init__(self, registry: PiscesLxCoreMetricsRegistry, inner: Counter) -> None:
        """
        Initialize a _CounterWrapper instance.

        Args:
            registry (PiscesLxCoreMetricsRegistry): The metrics registry containing global labels and configuration.
            inner (Counter): The inner Counter instance to be wrapped.
        """
        self._r = registry
        self._c = inner
        self.name = inner.name
        self.help_text = inner.help_text
        self.labels = inner.labels

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increase the counter value by the specified amount, with global labels merged into the provided labels.

        Args:
            amount (float, optional): The amount to increase the counter by. Defaults to 1.0.
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.
        """
        return self._c.inc(amount, self._r._merge_global_labels(labels))

    def snapshot(self, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Get the current value of the counter for the specified label combination, 
        with global labels merged into the provided labels.

        Args:
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. Defaults to None.

        Returns:
            float: The current value of the counter for the specified label combination.
        """
        return self._c.snapshot(self._r._merge_global_labels(labels))

    def snapshot_all(self) -> Dict[str, float]:
        """
        Get snapshots of all label combinations and their corresponding counter values.
        Note that this method does not merge global labels as it returns all existing combinations.

        Returns:
            Dict[str, float]: A dictionary containing all label combinations and their corresponding counter values.
        """
        return self._c.snapshot_all()

    def reset(self, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Reset the counter value. If labels are provided, reset only the value for that label combination,
        with global labels merged into the provided labels; otherwise, reset all values.

        Args:
            labels (Optional[Dict[str, str]], optional): Label key-value pairs. 
                If None, reset all counter values. Defaults to None.
        """
        return self._c.reset(self._r._merge_global_labels(labels))

class _GaugeWrapper:
    """
    A wrapper class for the Gauge that enforces the inclusion of global labels in operations.
    """
    def __init__(self, registry: PiscesLxCoreMetricsRegistry, inner: Gauge) -> None:
        """
        Initialize a _GaugeWrapper instance.

        Args:
            registry (PiscesLxCoreMetricsRegistry): The metrics registry containing global labels.
            inner (Gauge): The inner Gauge instance to be wrapped.
        """
        self._r = registry
        self._g = inner
        self.name = inner.name
        self.help_text = inner.help_text
        self.labels = inner.labels

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set the gauge value with the provided labels merged with global labels.

        Args:
            value (float): The value to set for the gauge.
            labels (Optional[Dict[str, str]]): Key-value pairs of labels. Defaults to None.
        """
        self._g.set(value, self._r._merge_global_labels(labels))

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increase the gauge value by the specified amount with the provided labels merged with global labels.

        Args:
            amount (float): The amount to increase the gauge value by. Defaults to 1.0.
            labels (Optional[Dict[str, str]]): Key-value pairs of labels. Defaults to None.
        """
        self._g.inc(amount, self._r._merge_global_labels(labels))

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Decrease the gauge value by the specified amount with the provided labels merged with global labels.

        Args:
            amount (float): The amount to decrease the gauge value by. Defaults to 1.0.
            labels (Optional[Dict[str, str]]): Key-value pairs of labels. Defaults to None.
        """
        self._g.dec(amount, self._r._merge_global_labels(labels))

    def snapshot(self, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Get the current value of the gauge with the provided labels merged with global labels.

        Args:
            labels (Optional[Dict[str, str]]): Key-value pairs of labels. Defaults to None.

        Returns:
            float: The current value of the gauge for the specified label combination.
        """
        return self._g.snapshot(self._r._merge_global_labels(labels))

    def snapshot_all(self) -> Dict[str, float]:
        """
        Get snapshots of all label combinations stored in the inner Gauge instance.
        Note: This method does not merge global labels as it retrieves all existing combinations.

        Returns:
            Dict[str, float]: A dictionary containing all label combinations and their corresponding values.
        """
        return self._g.snapshot_all()

    def reset(self, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Reset the gauge value for the specified label combination with global labels merged.
        If no labels are provided, reset all gauge values.

        Args:
            labels (Optional[Dict[str, str]]): Key-value pairs of labels. Defaults to None.
        """
        self._g.reset(self._r._merge_global_labels(labels))

class _HistogramWrapper:
    """
    A wrapper class for Histogram that merges global labels with input labels before operations.
    """
    def __init__(self, registry: PiscesLxCoreMetricsRegistry, inner: Histogram) -> None:
        """
        Initialize a _HistogramWrapper instance.

        Args:
            registry (PiscesLxCoreMetricsRegistry): The metrics registry containing global labels.
            inner (Histogram): The inner Histogram instance to be wrapped.
        """
        self._r = registry
        self._h = inner
        self.name = inner.name
        self.help_text = inner.help_text
        self.labels = inner.labels

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record an observed value in the histogram after merging global labels with input labels.

        Args:
            value (float): The observed value to record.
            labels (Optional[Dict[str, str]]): Label key-value pairs. Defaults to None.
        """
        self._h.observe(value, self._r._merge_global_labels(labels))

    def snapshot(self, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """
        Get a snapshot of the histogram data for the specified label combination after merging global labels.

        Args:
            labels (Optional[Dict[str, str]]): Label key-value pairs. Defaults to None.

        Returns:
            Dict[str, float]: A dictionary containing statistical information such as sum, count, average, and percentiles.
        """
        return self._h.snapshot(self._r._merge_global_labels(labels))

    def snapshot_all(self) -> Dict[str, Dict[str, float]]:
        """
        Get snapshots of the histogram data for all label combinations.

        Note: Global labels are not merged in this method as it retrieves all existing data.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary containing snapshots of all label combinations.
        """
        return self._h.snapshot_all()

    def reset(self, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Reset the histogram data for the specified label combination after merging global labels.
        If labels are None, reset all histogram data.

        Args:
            labels (Optional[Dict[str, str]]): Label key-value pairs. Defaults to None.
        """
        self._h.reset(self._r._merge_global_labels(labels))

    def __del__(self) -> None:
        """
        Destructor to ensure resource cleanup.
        Attempts to stop the periodic flush if it's running.
        """
        try:
            self.stop_periodic_flush()
        except Exception as e:
            _LOGGER.warning(f"Error occurred while stopping periodic flush in _HistogramWrapper destructor: {str(e)}")


_registry: Optional[PiscesLxCoreMetricsRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> PiscesLxCoreMetricsRegistry:
    """
    Retrieve the singleton instance of the metrics registry.
    If the instance does not exist, create a new one.

    Returns:
        PiscesLxCoreMetricsRegistry: The singleton instance of the metrics registry.
    """
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = PiscesLxCoreMetricsRegistry()
        return _registry


def reset_registry() -> None:
    """
    Reset the singleton metrics registry (mainly for testing purposes).
    Stops the periodic flush and clears the registry instance.
    """
    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.stop_periodic_flush()
            _registry = None
