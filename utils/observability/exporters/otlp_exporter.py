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
import urllib.request
from typing import Optional
from utils import PiscesLxCoreLog
from ..metrics import PiscesLxCoreMetricsRegistry

class OtlpExporter:
    """
    A class responsible for exporting metrics to an OTLP (OpenTelemetry Protocol) endpoint periodically.
    Supports multiple protocols for sending metrics data.
    """

    def __init__(self, endpoint: str, interval_sec: int = 15) -> None:
        """
        Initialize the OTLP exporter.

        Args:
            endpoint (str): The endpoint URL to which metrics will be sent.
            interval_sec (int, optional): The interval in seconds between metric exports. Defaults to 15.
        """
        self.endpoint = endpoint.rstrip('/') + '/v1/metrics'
        self.interval = interval_sec
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._log = PiscesLxCoreLog("pisceslx.exporter.otlp")

    def start(self) -> None:
        """
        Start the exporter thread.
        If the thread is already running, this method does nothing.
        """
        if self._th and self._th.is_alive():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        try:
            self._log.info("exporter.otlp.start", endpoint=self.endpoint, interval=self.interval)
        except Exception as e:
            self._log.error("Failed to log exporter start", error=str(e))

    def stop(self) -> None:
        """
        Stop the exporter thread.
        """
        self._stop.set()
        try:
            self._log.info("exporter.otlp.stop")
        except Exception as e:
            self._log.error("Failed to log exporter stop", error=str(e))

    def _loop(self) -> None:
        """
        The main loop of the exporter thread.
        Periodically fetches metrics and sends them to the OTLP endpoint using the specified protocol.
        Implements exponential backoff in case of sending failures.
        """
        # Get the metrics registry instance
        reg = PiscesLxCoreMetricsRegistry.instance()
        # Maximum backoff time in seconds, shared with prom exporter
        max_backoff = 60.0
        try:
            env_val = os.environ.get("PISCES_EXPORTER_BACKOFF_MAX_SEC")
            if env_val:
                max_backoff = max(1.0, float(env_val))
        except Exception as e:
            self._log.error("Failed to parse PISCES_EXPORTER_BACKOFF_MAX_SEC", error=str(e))

        backoff = 1.0
        while not self._stop.is_set():
            try:
                _t0 = time.time()
                # Get a grouped snapshot of metrics
                snap = reg.snapshot_grouped()
                # Get the OTLP protocol from environment variable, default to 'json'
                proto = os.environ.get("PISCES_OTLP_PROTOCOL", "json").lower()
                sent = False

                if proto in ("grpc",):
                    sent = self._post_grpc(self.endpoint, snap)
                    if not sent:
                        sent = self._post_http_protobuf(self.endpoint, snap)
                        if not sent:
                            sent = self._post_json(self.endpoint, snap)
                elif proto in ("http_protobuf", "http-protobuf", "otlp_http_protobuf"):
                    sent = self._post_http_protobuf(self.endpoint, snap)
                    if not sent:
                        sent = self._post_json(self.endpoint, snap)
                else:
                    sent = self._post_json(self.endpoint, snap)

                if not sent:
                    raise RuntimeError("otlp send failed")

                backoff = 1.0
                try:
                    reg.gauge("exporter.last_success_timestamp_seconds", help_text="Last success epoch seconds", labels=["exporter"]).set(time.time(), labels={"exporter": "otlp"})
                except Exception as e:
                    self._log.error("Failed to set last success timestamp", error=str(e))

                try:
                    dur = max(0.0, time.time() - _t0)
                    reg.observe("exporter_duration_seconds", dur, labels={"exporter": "otlp", "status": "success"})
                except Exception as e:
                    self._log.error("Failed to record successful export duration", error=str(e))

                try:
                    self._log.success("exporter.otlp.flush.ok", protocol=proto, counters=len(snap.get("counters", {})), gauges=len(snap.get("gauges", {})), histograms=len(snap.get("histograms", {})))
                except Exception as e:
                    self._log.error("Failed to log successful export", error=str(e))

            except Exception as e:
                try:
                    reg.inc("exporter.errors", 1.0, labels={"exporter": "otlp"})
                except Exception as exc:
                    self._log.error("Failed to increment error count", error=str(exc))

                try:
                    reg.gauge("exporter.last_error_timestamp_seconds", help_text="Last error epoch seconds", labels=["exporter"]).set(time.time(), labels={"exporter": "otlp"})
                except Exception as exc:
                    self._log.error("Failed to set last error timestamp", error=str(exc))

                try:
                    dur = max(0.0, time.time() - _t0) if '_t0' in locals() else 0.0
                    reg.observe("exporter_duration_seconds", dur, labels={"exporter": "otlp", "status": "error"})
                except Exception as exc:
                    self._log.error("Failed to record failed export duration", error=str(exc))

                try:
                    self._log.error("exporter.otlp.flush.error", endpoint=self.endpoint)
                except Exception as exc:
                    self._log.error("Failed to log export error", error=str(exc))

                self._stop.wait(backoff)
                backoff = min(max_backoff, backoff * 2.0)
                continue

            self._stop.wait(self.interval)

    def _post_json(self, url: str, payload: dict) -> bool:
        """
        Send metrics data to the specified URL in JSON format.

        Args:
            url (str): The target URL to send data to.
            payload (dict): The metrics data to send.

        Returns:
            bool: True if the request succeeds with a 2xx status code, False otherwise.
        """
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=2) as resp:
                code = getattr(resp, 'status', None) or getattr(resp, 'code', None) or 0
                return 200 <= int(code) < 300
        except Exception as e:
            self._log.error("Failed to send JSON data", url=url, error=str(e))
            return False

    def _post_http_protobuf(self, url: str, grouped_snap: dict) -> bool:
        """
        Attempt to send metrics data in OTLP HTTP Protobuf format.
        Returns True on success, False if dependencies are unavailable or an error occurs.

        Args:
            url (str): The target URL to send data to.
            grouped_snap (dict): The grouped snapshot of metrics data.

        Returns:
            bool: True if the data is sent successfully, False otherwise.
        """
        try:
            from google.protobuf.json_format import MessageToJson
            from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
            from opentelemetry.proto.metrics.v1.metrics_pb2 import (
                ResourceMetrics, ScopeMetrics, Metric, Gauge, Sum, NumberDataPoint, Histogram, HistogramDataPoint
            )
        except Exception as e:
            self._log.error("Failed to import HTTP Protobuf dependencies", error=str(e))
            return False

        try:
            ts = int(time.time() * 1e9)
            req = ExportMetricsServiceRequest()
            rm = req.resource_metrics.add()
            sm = rm.scope_metrics.add()

            # Process gauge metrics
            for name, val in (grouped_snap.get("gauges") or {}).items():
                try:
                    v = float(val)
                except Exception:
                    continue
                m = sm.metrics.add(name=name)
                g = Gauge()
                dp = NumberDataPoint(time_unix_nano=ts)
                dp.as_double = v
                g.data_points.append(dp)
                m.gauge.CopyFrom(g)

            # Process counter metrics
            for name, val in (grouped_snap.get("counters") or {}).items():
                try:
                    v = float(val)
                except Exception:
                    continue
                m = sm.metrics.add(name=name)
                s = Sum()
                s.is_monotonic = False
                dp = NumberDataPoint(time_unix_nano=ts)
                dp.as_double = v
                s.data_points.append(dp)
                m.sum.CopyFrom(s)

            # Process histogram metrics
            for name, hv in (grouped_snap.get("histograms") or {}).items():
                try:
                    total_sum = float(hv.get("sum", 0.0))
                    total_count = int(hv.get("count", 0))
                except Exception:
                    continue
                m = sm.metrics.add(name=name)
                h = Histogram()
                dp = HistogramDataPoint(time_unix_nano=ts)
                dp.sum = total_sum
                dp.count = total_count
                h.data_points.append(dp)
                m.histogram.CopyFrom(h)

            data = req.SerializeToString()
            headers = {
                'Content-Type': 'application/x-protobuf'
            }
            http_req = urllib.request.Request(url, data=data, headers=headers)
            urllib.request.urlopen(http_req, timeout=2)
            return True
        except Exception as e:
            self._log.error("Failed to send HTTP Protobuf data", url=url, error=str(e))
            return False

    def _post_grpc(self, url: str, grouped_snap: dict) -> bool:
        """
        Attempt to send metrics data via gRPC endpoint (e.g., http://host:4317 or host:4317).
        Returns True on success, False if dependencies are unavailable or an error occurs.

        Args:
            url (str): The target URL to send data to.
            grouped_snap (dict): The grouped snapshot of metrics data.

        Returns:
            bool: True if the data is sent successfully, False otherwise.
        """
        try:
            import grpc  # type: ignore
            from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
            from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2_grpc import MetricsServiceStub
            from opentelemetry.proto.metrics.v1.metrics_pb2 import (
                ResourceMetrics, ScopeMetrics, Metric, Gauge, Sum, NumberDataPoint, Histogram, HistogramDataPoint
            )
        except Exception as e:
            self._log.error("Failed to import gRPC dependencies", error=str(e))
            return False

        try:
            endpoint = url
            # Remove scheme and path from the endpoint
            if endpoint.startswith("http://"):
                endpoint = endpoint[len("http://"):]
            if endpoint.startswith("https://"):
                endpoint = endpoint[len("https://"):]
            if "/" in endpoint:
                endpoint = endpoint.split("/", 1)[0]

            channel = grpc.insecure_channel(endpoint)
            stub = MetricsServiceStub(channel)
            ts = int(time.time() * 1e9)
            req = ExportMetricsServiceRequest()
            rm = req.resource_metrics.add()
            sm = rm.scope_metrics.add()

            # Process gauge metrics
            for name, val in (grouped_snap.get("gauges") or {}).items():
                try:
                    v = float(val)
                except Exception:
                    continue
                m = sm.metrics.add(name=name)
                g = Gauge()
                dp = NumberDataPoint(time_unix_nano=ts)
                dp.as_double = v
                g.data_points.append(dp)
                m.gauge.CopyFrom(g)

            # Process counter metrics
            for name, val in (grouped_snap.get("counters") or {}).items():
                try:
                    v = float(val)
                except Exception:
                    continue
                m = sm.metrics.add(name=name)
                s = Sum()
                s.is_monotonic = False
                dp = NumberDataPoint(time_unix_nano=ts)
                dp.as_double = v
                s.data_points.append(dp)
                m.sum.CopyFrom(s)

            # Process histogram metrics
            for name, hv in (grouped_snap.get("histograms") or {}).items():
                try:
                    total_sum = float(hv.get("sum", 0.0))
                    total_count = int(hv.get("count", 0))
                except Exception:
                    continue
                m = sm.metrics.add(name=name)
                h = Histogram()
                dp = HistogramDataPoint(time_unix_nano=ts)
                dp.sum = total_sum
                dp.count = total_count
                h.data_points.append(dp)
                m.histogram.CopyFrom(h)

            stub.Export(req, timeout=2)
            return True
        except Exception as e:
            self._log.error("Failed to send gRPC data", url=url, error=str(e))
            return False
