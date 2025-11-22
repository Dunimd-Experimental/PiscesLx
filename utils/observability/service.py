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
import json
import uuid
import time
import socket
import atexit
import random
import platform
import threading
from enum import Enum
from pathlib import Path
from datetime import datetime
import threading as _threading
from utils.log.core import PiscesLxCoreLog
from dataclasses import dataclass, field
from .reporter import PiscesLxCoreReporter
from collections import defaultdict, deque
from .algorithms.alerts import build_alerts
from .runtime.monitor import run_monitor_loop
from utils.cache.core import get_default_cache
from .metrics import PiscesLxCoreMetricsRegistry
from typing import Optional, Dict, Any, List, Callable
from .exporters.prom_exporter import PromTextfileExporter
from utils.hooks.types import PiscesLxCoreFunctionListener
from .collectors.registry import get_default_collector_registry
from utils.hooks.bus import get_global_hook_bus, PiscesLxCoreHookBus
from .reports.writer import build_device_report_payload, build_session_report_payload
from .algorithms.trends import compute_ewma, update_baseline, mad_stats, is_outlier_mad, EnterpriseTrendAnalyzer, TrendState

def _tools_request_success_listener(payload):
    """
    Listener for successful tool requests. Records the request count and latency metrics.

    Args:
        payload (dict): The payload containing request information, which may include 
                        'provider', 'model', 'route', 'component', 'latency_ms', 
                        'start_time', and 'end_time'.
    """
    try:
        # Extract relevant labels from the payload
        labels = {}
        for k in ('provider', 'model', 'route', 'component'):
            v = payload.get(k)
            if v is not None:
                labels[k] = str(v)
        # Set the default scope to 'tools'
        labels.setdefault('scope', 'tools')

        # Calculate latency from the payload
        lat = payload.get('latency_ms')
        if lat is None:
            st = payload.get('start_time')
            et = payload.get('end_time')
            if st is not None and et is not None:
                try:
                    lat = float(et) - float(st)
                    lat = lat * 1000.0
                except Exception:
                    lat = 0.0
            else:
                lat = 0.0

        # Get the metrics registry instance and record metrics
        reg = PiscesLxCoreMetricsRegistry.instance()
        reg.counter('llm_requests').inc(1.0, labels)
        reg.histogram('llm_latency_ms', labels=list(labels.keys())).observe(float(lat), labels)
    except Exception as e:
        logger = PiscesLxCoreLog("pisceslx.observability.service")
        logger.error("Error in _tools_request_success_listener", error=str(e))

def _tools_request_error_listener(payload):
    """
    Listener for failed tool requests. Records the request count, error count, and latency metrics.

    Args:
        payload (dict): The payload containing request information, which may include 
                        'provider', 'model', 'route', 'component', 'latency_ms', 
                        'start_time', and 'end_time'.
    """
    try:
        # Extract relevant labels from the payload
        labels = {}
        for k in ('provider', 'model', 'route', 'component'):
            v = payload.get(k)
            if v is not None:
                labels[k] = str(v)
        # Set the default scope to 'tools'
        labels.setdefault('scope', 'tools')

        # Calculate latency from the payload
        lat = payload.get('latency_ms')
        if lat is None:
            st = payload.get('start_time')
            et = payload.get('end_time')
            if st is not None and et is not None:
                try:
                    lat = float(et) - float(st)
                    lat = lat * 1000.0
                except Exception:
                    lat = 0.0
            else:
                lat = 0.0

        # Get the metrics registry instance and record metrics
        reg = PiscesLxCoreMetricsRegistry.instance()
        reg.counter('llm_requests').inc(1.0, labels)
        reg.counter('llm_errors').inc(1.0, labels)
        reg.histogram('llm_latency_ms', labels=list(labels.keys())).observe(float(lat), labels)
    except Exception as e:
        logger = PiscesLxCoreLog("pisceslx.observability.service")
        logger.error("Error in _tools_request_error_listener", error=str(e))

class ServiceState(Enum):
    """
    Defines the lifecycle states of a service, used for service observability management.
    """
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    FAILED = "failed"

class TraceContext:
    """Distributed tracing context."""
    
    def __init__(self, trace_id: str = None, span_id: str = None, parent_span_id: str = None):
        """
        Initialize a new trace context.

        Args:
            trace_id (str, optional): The ID of the trace. If not provided, a new UUID will be generated.
            span_id (str, optional): The ID of the span. If not provided, a new UUID will be generated.
            parent_span_id (str, optional): The ID of the parent span. Defaults to None.
        """
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = span_id or str(uuid.uuid4())
        self.parent_span_id = parent_span_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.tags: Dict[str, Any] = {}
        self.logs: List[Dict[str, Any]] = []
    
    def finish(self):
        """Finish the current span by recording the end time."""
        self.end_time = time.time()
    
    def set_tag(self, key: str, value: Any):
        """
        Set a tag on the trace context.

        Args:
            key (str): The tag key.
            value (Any): The tag value.
        """
        self.tags[key] = value
    
    def log(self, event: str, attributes: Dict[str, Any] = None):
        """
        Record a log entry in the trace context.

        Args:
            event (str): The name of the event.
            attributes (Dict[str, Any], optional): Additional attributes associated with the event. Defaults to None.
        """
        self.logs.append({
            "timestamp": time.time(),
            "event": event,
            "attributes": attributes or {}
        })
    
    def duration(self) -> float:
        """
        Get the duration of the span in seconds.

        Returns:
            float: The duration of the span. If the span is finished, returns the difference between end_time and start_time.
                   Otherwise, returns the difference between current time and start_time.
        """
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

class Tracer:
    """Distributed tracer for tracking and managing trace spans."""
    
    def __init__(self):
        """Initialize the Tracer instance.
        
        Creates a logger for tracing operations, initializes a dictionary to store active spans,
        and sets up a lock for thread-safe operations.
        """
        self.logger = PiscesLxCoreLog("pisceslx.observability.tracer")
        self.active_spans: Dict[str, TraceContext] = {}
        self.lock = threading.Lock()
    
    def start_span(self, operation_name: str, parent_context: TraceContext = None) -> TraceContext:
        """Start a new trace span.

        Args:
            operation_name (str): The name of the operation to be traced.
            parent_context (TraceContext, optional): The parent trace context. 
                If provided, the new span will be part of the same trace. Defaults to None.

        Returns:
            TraceContext: The newly created trace context representing the span.
        """
        if parent_context:
            context = TraceContext(
                trace_id=parent_context.trace_id,
                parent_span_id=parent_context.span_id
            )
        else:
            context = TraceContext()
        
        context.set_tag("operation", operation_name)
        
        with self.lock:
            self.active_spans[context.span_id] = context
        
        self.logger.debug("TRACE_SPAN_STARTED", trace_id=context.trace_id, span_id=context.span_id, operation=operation_name)
        
        return context
    
    def finish_span(self, context: TraceContext):
        """Finish a trace span.

        Marks the span as finished, removes it from the active spans, and logs the completion.

        Args:
            context (TraceContext): The trace context representing the span to finish.
        """
        context.finish()
        
        with self.lock:
            self.active_spans.pop(context.span_id, None)
        
        self.logger.debug("TRACE_SPAN_FINISHED", trace_id=context.trace_id, span_id=context.span_id, duration=context.duration())
    
    def get_active_spans(self) -> Dict[str, TraceContext]:
        """Get all active trace spans.

        Returns:
            Dict[str, TraceContext]: A copy of the dictionary containing all active trace spans.
        """
        with self.lock:
            return self.active_spans.copy()

@dataclass
class PerformanceMetrics:
    """
    Represents real-time performance metrics used for intelligent analysis.

    Attributes:
        request_count (int): Total number of requests. Defaults to 0.
        error_count (int): Total number of failed requests. Defaults to 0.
        avg_latency (float): Average latency in milliseconds. Defaults to 0.0.
        p95_latency (float): 95th percentile latency in milliseconds. Defaults to 0.0.
        p99_latency (float): 99th percentile latency in milliseconds. Defaults to 0.0.
        throughput (float): Throughput in requests per second. Defaults to 0.0.
        memory_usage (float): Memory usage percentage. Defaults to 0.0.
        cpu_usage (float): CPU usage percentage. Defaults to 0.0.
        timestamp (datetime): The time when the metrics were recorded. Defaults to the current UTC time.
    """
    request_count: int = 0
    error_count: int = 0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def error_rate(self) -> float:
        """
        Calculate the error rate, which is the ratio of failed requests to total requests.

        Returns:
            float: The error rate, ranging from 0 to 1.
        """
        return self.error_count / max(self.request_count, 1)
    
    @property
    def health_score(self) -> float:
        """
        Calculate the service health score (0-100).
        Prioritize error rate, then latency against SLO, and consider throughput only when there were requests.
        If there is no recent activity, return 100.0.

        Returns:
            float: The service health score, ranging from 0 to 100.
        """
        # No activity: treat as healthy
        if (
            self.request_count <= 0 and self.error_count <= 0 and
            self.p95_latency <= 0.0 and self.p99_latency <= 0.0 and
            self.throughput <= 0.0
        ):
            return 100.0

        score = 100.0

        # 1) Error rate penalty (primary)
        # Up to 50 points penalty when error_rate approaches 100%
        try:
            score -= min(50.0, max(0.0, self.error_rate * 100.0))
        except Exception:
            pass

        # 2) Latency penalty vs SLO (baseline SLO: p95 <= 300ms)
        if self.p95_latency > 0.0:
            lat_pen = 0.0
            if self.p95_latency > 300.0:
                # 300ms => 0; 1000ms => ~30; cap at 40
                lat_pen = (self.p95_latency - 300.0) / 700.0 * 30.0
            score -= min(40.0, max(0.0, lat_pen))

        # 3) Throughput penalty (only if there were requests)
        if self.request_count > 0:
            if self.throughput < 1.0:
                score -= 10.0
            elif self.throughput < 5.0:
                score -= 5.0

        # 4) Light resource pressure penalty (if cpu/memory provided in percent [0-100])
        try:
            if self.cpu_usage > 0.0:
                score -= min(5.0, max(0.0, (self.cpu_usage - 90.0) * 0.5))
            if self.memory_usage > 0.0:
                score -= min(5.0, max(0.0, (self.memory_usage - 90.0) * 0.5))
        except Exception:
            pass

        return max(0.0, min(100.0, score))

@dataclass 
class AnomalyPattern:
    """
    Represents a detected anomaly pattern. This class is used to record information about detected anomalies.

    Attributes:
        pattern_type (str): The type of anomaly pattern.
        severity (str): The severity level of the anomaly.
        description (str): A description of the anomaly.
        affected_components (List[str]): A list of components affected by the anomaly.
        suggested_actions (List[str]): A list of suggested actions to take.
        confidence (float): The confidence level of the anomaly detection.
        timestamp (datetime): The timestamp when the anomaly was recorded, defaulting to the current UTC time.
    """
    pattern_type: str
    severity: str
    description: str
    affected_components: List[str]
    suggested_actions: List[str]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AlertRule:
    """
    Represents an alert rule. This class defines the conditions under which an alert should be triggered.

    Attributes:
        name (str): The name of the alert rule.
        metric (str): The metric to monitor.
        threshold (float): The threshold value for the metric.
        operator (str): The comparison operator ('>', '<', '>=', '<=', '==', '!=').
        duration (int): The duration (in seconds) for which the condition must hold.
        severity (str): The severity level of the alert ('low', 'medium', 'high', 'critical').
        enabled (bool): Whether the rule is enabled. Defaults to True.
        description (str): A description of the alert rule. Defaults to an empty string.
    """
    name: str
    metric: str
    threshold: float
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    duration: int  # Duration in seconds
    severity: str  # 'low', 'medium', 'high', 'critical'
    enabled: bool = True
    description: str = ""

@dataclass
class Alert:
    """
    Represents an alert instance. This class holds information about a triggered alert.

    Attributes:
        rule_name (str): The name of the alert rule that triggered this alert.
        metric (str): The metric that triggered the alert.
        value (float): The actual value of the metric.
        threshold (float): The threshold value of the metric.
        severity (str): The severity level of the alert.
        timestamp (datetime): The timestamp when the alert was triggered, defaulting to the current UTC time.
        resolved (bool): Whether the alert has been resolved. Defaults to False.
        resolved_at (Optional[datetime]): The timestamp when the alert was resolved. Defaults to None.
    """
    rule_name: str
    metric: str
    value: float
    threshold: float
    severity: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class AlertManager:
    """Manages alert rules and monitors metrics to trigger or resolve alerts."""
    
    def __init__(self):
        """Initialize the AlertManager instance.
        
        Sets up a logger, initializes storage for alert rules, active alerts, and alert history.
        Also initializes a lock for thread-safe operations, notification callbacks, and starts 
        with monitoring disabled. Finally, sets up default alert rules.
        """
        self.logger = PiscesLxCoreLog("pisceslx.observability.alerts")
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.lock = threading.RLock()
        self.notification_callbacks: List[Callable[[Alert], None]] = []
        self._running = False
        self._monitor_thread = None
        self._setup_default_rules()
    
    def add_rule(self, rule: AlertRule):
        """Add a new alert rule to the manager.

        Args:
            rule (AlertRule): The alert rule to be added.
        """
        with self.lock:
            self.rules[rule.name] = rule
        self.logger.info("Alert rule added", event="ALERT_RULE_ADDED", rule=rule.name)
    
    def start_monitoring(self):
        """Start the alert monitoring loop in a separate thread."""
        if not self._running:
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            self.logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop the alert monitoring loop and wait for the monitoring thread to finish."""
        if self._running:
            self._running = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5)
            self.logger.info("Alert monitoring stopped")
    
    def _setup_default_rules(self):
        """Set up default alert rules for system monitoring."""
        default_rules = [
            AlertRule("gpu_memory_high", "gpu_memory_usage", 0.9, ">", 300, "WARNING", description="High GPU memory usage"),
            AlertRule("cpu_usage_high", "cpu_usage", 0.8, ">", 600, "WARNING", description="High CPU usage"),
            AlertRule("memory_usage_high", "memory_usage", 0.85, ">", 600, "WARNING", description="High memory usage"),
            AlertRule("disk_usage_high", "disk_usage", 0.9, ">", 1800, "CRITICAL", description="High disk usage"),
            AlertRule("error_rate_high", "error_rate", 0.05, ">", 300, "CRITICAL", description="High error rate"),
            AlertRule("response_time_high", "response_time", 1000, ">", 300, "WARNING", description="High response time"),
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def _monitor_loop(self):
        """Main monitoring loop that periodically checks all alert rules.
        
        The loop runs as long as self._running is True, checking all rules every 30 seconds.
        If an error occurs during the check, it logs the error and waits for 60 seconds before retrying.
        """
        while self._running:
            try:
                self._check_all_rules()
                time.sleep(30)  # Check all rules every 30 seconds
            except Exception as e:
                self.logger.error("Alert check failed", error=str(e))
                time.sleep(60)  # Wait longer when an error occurs
    
    def _check_all_rules(self):
        """Check all enabled alert rules against the current system metrics."""
        try:
            metrics = self._get_current_metrics()
            for rule_name, rule in self.rules.items():
                if rule.metric in metrics:
                    value = metrics[rule.metric]
                    self.evaluate_metric(rule.metric, value)
        except Exception as e:
            self.logger.error("Rule check failed", error=str(e))
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get the current system metrics.
        
        Tries to collect system metrics using psutil. If collection fails, returns default values.

        Returns:
            Dict[str, float]: A dictionary containing current system metrics.
        """
        try:
            import psutil
            return {
                "gpu_memory_usage": self._get_gpu_memory_usage(),
                "cpu_usage": psutil.cpu_percent(interval=1) / 100.0,
                "memory_usage": psutil.virtual_memory().percent / 100.0,
                "disk_usage": psutil.disk_usage('/').percent / 100.0,
                "error_rate": self._get_error_rate(),
                "response_time": self._get_avg_response_time_using_metrics_registry(),
            }
        except Exception as e:
            self.logger.warning(f"Failed to get metrics, using default values. Error: {e}")
            return {
                "gpu_memory_usage": 0.0,
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "disk_usage": 0.0,
                "error_rate": 0.0,
                "response_time": 0.0,
            }
    
    def _get_gpu_memory_usage(self) -> float:
        """Get the current GPU memory usage percentage.
        
        Uses nvidia-smi to get GPU memory usage. If the command fails, returns 0.0.

        Returns:
            float: The current GPU memory usage percentage.
        """
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                used, total = map(float, result.stdout.strip().split(','))
                return used / total if total > 0 else 0.0
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory usage. Error: {e}")
        return 0.0
    
    def _get_avg_response_time_using_metrics_registry(self) -> float:
        """Get average response time (ms) from aggregated latency histogram for tools scope.
        
        Uses the metrics registry to get the sum and count of latency values, then calculates the average.
        If the metrics cannot be retrieved or the count is zero, returns 0.0.

        Returns:
            float: The average response time in milliseconds.
        """
        try:
            reg = PiscesLxCoreMetricsRegistry.instance()
            grouped = reg.snapshot_grouped()
            histos = grouped.get("histograms", {})
            lat = histos.get("llm_latency_ms", {})

            total_sum = float(lat.get("sum", 0.0))
            total_cnt = float(lat.get("count", 0.0))

            if total_cnt <= 0.0:
                return 0.0
            return total_sum / total_cnt
        except Exception as e:
            self.logger.warning(f"Failed to get average response time. Error: {e}")
            return 0.0
    
    def _get_error_rate(self) -> float:
        """Get the current error rate based on actual metrics collection.
        
        Returns:
            float: The current error rate (0.0-1.0), calculated from recent request metrics.
        """
        try:
            # Get metrics registry to access actual error counts
            reg = PiscesLxCoreMetricsRegistry.instance()
            
            # Get total requests and errors from metrics
            request_metrics = reg.counter('llm_requests')
            error_metrics = reg.counter('llm_errors')
            
            if request_metrics is None or error_metrics is None:
                # Fallback to internal tracking if metrics not available
                return self._calculate_internal_error_rate()
            
            # Calculate error rate from metrics
            total_requests = request_metrics.get_value()
            total_errors = error_metrics.get_value()
            
            if total_requests == 0:
                return 0.0
                
            error_rate = total_errors / total_requests
            return min(max(error_rate, 0.0), 1.0)  # Clamp between 0-1
            
        except Exception as e:
            self.logger.warning(f"Failed to get error rate from metrics, using internal calculation: {e}")
            return self._calculate_internal_error_rate()
    
    def _calculate_internal_error_rate(self) -> float:
        """Calculate error rate using internal request tracking.
        
        Returns:
            float: The calculated error rate based on recent request history.
        """
        try:
            # Use the IntelligentMetricsCollector if available
            if hasattr(self, '_metrics_collector') and self._metrics_collector:
                collector = self._metrics_collector
                if hasattr(collector, 'errors') and hasattr(collector, 'latencies'):
                    error_count = len(collector.errors)
                    total_count = len(collector.latencies)
                    
                    if total_count == 0:
                        return 0.0
                        
                    return error_count / total_count
            
            # Fallback: check component health tracking
            if hasattr(self, '_component_health'):
                total_healthy = 0
                total_failed = 0
                
                for component_data in self._component_health.values():
                    total_healthy += component_data.get('healthy', 0)
                    total_failed += component_data.get('failed', 0)
                
                total_requests = total_healthy + total_failed
                if total_requests == 0:
                    return 0.0
                    
                return total_failed / total_requests
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Internal error rate calculation failed: {e}")
            return 0.0
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule from the manager.

        Args:
            rule_name (str): The name of the alert rule to remove.
        """
        with self.lock:
            self.rules.pop(rule_name, None)
        self.logger.info("Alert rule removed", event="ALERT_RULE_REMOVED", rule=rule_name)
    
    def add_notification_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function to be called when an alert is triggered or resolved.

        Args:
            callback (Callable[[Alert], None]): The callback function to add.
        """
        self.notification_callbacks.append(callback)
    
    def evaluate_metric(self, metric_name: str, value: float):
        """Evaluate a metric against all relevant alert rules and trigger or resolve alerts accordingly.

        Args:
            metric_name (str): The name of the metric to evaluate.
            value (float): The current value of the metric.
        """
        with self.lock:
            for rule in self.rules.values():
                if not rule.enabled or rule.metric != metric_name:
                    continue
                
                # Check if the condition is met
                condition_met = False
                if rule.operator == '>':
                    condition_met = value > rule.threshold
                elif rule.operator == '<':
                    condition_met = value < rule.threshold
                elif rule.operator == '>=':
                    condition_met = value >= rule.threshold
                elif rule.operator == '<=':
                    condition_met = value <= rule.threshold
                elif rule.operator == '==':
                    condition_met = value == rule.threshold
                elif rule.operator == '!=':
                    condition_met = value != rule.threshold
                
                alert_id = f"{rule.name}_{metric_name}"
                
                if condition_met:
                    # Check if there's already an active alert
                    if alert_id not in self.active_alerts:
                        # Create a new alert
                        alert = Alert(
                            rule_name=rule.name,
                            metric=metric_name,
                            value=value,
                            threshold=rule.threshold,
                            severity=rule.severity
                        )
                        self.active_alerts[alert_id] = alert
                        self.alert_history.append(alert)
                        
                        # Notify callbacks
                        for callback in self.notification_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                self.logger.error("Alert notification error", event="ALERT_NOTIFICATION_ERROR", error=str(e))
                        
                        self.logger.alert("Alert triggered", event="ALERT_TRIGGERED", rule=rule.name, metric=metric_name, value=value, threshold=rule.threshold, severity=rule.severity)
                else:
                    # Check if we need to resolve an alert
                    if alert_id in self.active_alerts:
                        alert = self.active_alerts[alert_id]
                        alert.resolved = True
                        alert.resolved_at = datetime.utcnow()
                        
                        # Notify callbacks
                        for callback in self.notification_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                self.logger.error("Alert notification error", event="ALERT_NOTIFICATION_ERROR", error=str(e))
                        
                        self.logger.info("Alert resolved", event="ALERT_RESOLVED", rule=rule.name, metric=metric_name, value=value, threshold=rule.threshold)
                        
                        # Remove from active alerts
                        del self.active_alerts[alert_id]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts.

        Returns:
            List[Alert]: A list of all active alerts.
        """
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 50) -> List[Alert]:
        """Get a limited number of the most recent alerts from the alert history.

        Args:
            limit (int, optional): The maximum number of alerts to return. Defaults to 50.

        Returns:
            List[Alert]: A list of the most recent alerts.
        """
        with self.lock:
            return list(self.alert_history)[-limit:]

class IntelligentMetricsCollector:
    """
    A metrics collector and analysis engine that leverages algorithms to collect and analyze service performance metrics.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize the metrics collector.

        Args:
            window_size (int): The size of the metrics collection window. Defaults to 1000.
        """
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.throughput_samples = deque(maxlen=window_size)
        self.component_health = defaultdict(lambda: {"healthy": 0, "failed": 0})
        self.anomaly_patterns = deque(maxlen=100)
        self._lock = threading.Lock()
        # Exponential Weighted Moving Average (EWMA) alpha value
        self._ewma_alpha: float = 0.2
        self._ewma_p95: float = 0.0
        self._baseline_p95: Optional[float] = None
        self._baseline_p99: Optional[float] = None
        self._ewma_error: float = 0.0
        
    def record_request(self, latency: float, component: str, success: bool = True, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record request metrics and perform performance analysis.

        Args:
            latency (float): Request latency in milliseconds.
            component (str): The name of the component that generated the request.
            success (bool): Whether the request was successful. Defaults to True.
            labels (Optional[Dict[str, str]]): Optional additional labels for metrics emission, 
                e.g., provider/model/route.
        """
        with self._lock:
            self.latencies.append(latency)
            self.throughput_samples.append(time.time())
            
            if success:
                self.component_health[component]["healthy"] += 1
            else:
                self.component_health[component]["failed"] += 1
                self.errors.append(1)
            
            # Emit Prometheus-compatible metrics: request count, error count, and latency histogram
            try:
                reg = PiscesLxCoreMetricsRegistry.instance()
                lab = {"component": component}
                if labels:
                    # Safely merge user labels
                    lab.update({k: str(v) for k, v in labels.items()})
                # Ensure scope=tools is set as an additional safeguard during emission
                lab.setdefault("scope", "tools")
                if random.random() <= getattr(self, "_req_sampling", 1.0):
                    reg.counter("llm_requests").inc(1.0, lab)
                    if not success:
                        reg.counter("llm_errors").inc(1.0, lab)
                    # Latency percentile histogram
                    # Include dynamic label keys to maintain consistency
                    reg.histogram("llm_latency_ms", labels=list(lab.keys())).observe(float(latency), lab)
            except Exception as e:
                logger = PiscesLxCoreLog("pisceslx.observability.service")
                logger.error("Error emitting Prometheus metrics in record_request", error=str(e))
            
    def _detect_anomalies(self, latency: float, component: str, success: bool) -> None:
        """
        Perform anomaly detection using Median Absolute Deviation (MAD) and error rate thresholds.

        Args:
            latency (float): Current request latency in milliseconds.
            component (str): The name of the component that generated the request.
            success (bool): Whether the request was successful.
        """
        if len(self.latencies) < 20:
            return
        recent = list(self.latencies)[-50:]
        median, mad = mad_stats(recent)
        # MAD outlier detection
        if is_outlier_mad(latency, median, mad, k=6.0):
            anomaly = AnomalyPattern(
                pattern_type="latency_spike",
                severity="high" if is_outlier_mad(latency, median, mad, k=10.0) else "medium",
                description=f"MAD spike in {component}: {latency:.2f}ms (median: {median:.2f}ms, MAD: {mad:.2f})",
                affected_components=[component],
                suggested_actions=["Check resource usage", "Review recent changes"],
                confidence=0.8,
            )
            self.anomaly_patterns.append(anomaly)
        # High error rate anomaly detection within the window
        err_cnt = sum(self.errors)
        err_rate = err_cnt / max(len(self.latencies), 1)
        if err_rate > 0.2:
            anomaly = AnomalyPattern(
                pattern_type="high_error_rate",
                severity="critical" if err_rate > 0.4 else "high",
                description=f"High error rate: {err_rate:.1%}",
                affected_components=[component],
                suggested_actions=["Investigate recent failures", "Increase logging"],
                confidence=min(0.95, err_rate * 2.0),
            )
            self.anomaly_patterns.append(anomaly)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics and perform performance analysis.

        Returns:
            PerformanceMetrics: An object representing current performance metrics.
        """
        with self._lock:
            if not self.latencies:
                return PerformanceMetrics()
            latencies = list(self.latencies)
            latencies_sorted = sorted(latencies)
            
            # Calculate percentiles
            p95_idx = int(len(latencies_sorted) * 0.95)
            p99_idx = int(len(latencies_sorted) * 0.99)
            
            # Calculate throughput (requests/second)
            if len(self.throughput_samples) >= 2:
                time_span = max(self.throughput_samples) - min(self.throughput_samples)
                throughput = len(self.throughput_samples) / max(time_span, 1)
            else:
                throughput = 0
                
            # Get system metrics
            try:
                import psutil
                memory = psutil.virtual_memory().percent
                cpu = psutil.cpu_percent(interval=0.1)
            except ImportError:
                memory = cpu = 0.0
                logger = PiscesLxCoreLog("pisceslx.observability.service")
                logger.warning("psutil not found, using default values for system metrics")
            
            pm = PerformanceMetrics(
                request_count=len(latencies),
                error_count=sum(self.errors),
                avg_latency=sum(latencies) / len(latencies),
                p95_latency=latencies_sorted[min(p95_idx, len(latencies_sorted)-1)],
                p99_latency=latencies_sorted[min(p99_idx, len(latencies_sorted)-1)],
                throughput=throughput,
                memory_usage=memory,
                cpu_usage=cpu
            )
            # Update local metrics of the collector and add drift/EWMA metadata as attributes
            try:
                # Update EWMA
                self._ewma_p95 = compute_ewma(float(pm.p95_latency), float(self._ewma_p95), self._ewma_alpha)
                self._ewma_p99 = compute_ewma(float(pm.p99_latency), float(self._ewma_p99), self._ewma_alpha)
                # Error rate EWMA
                err_rate = sum(self.errors) / max(len(self.latencies), 1)
                self._ewma_error = self._ewma_alpha * float(err_rate) + (1.0 - self._ewma_alpha) * float(self._ewma_error)
                # Update baselines
                self._baseline_p95 = update_baseline(self._baseline_p95, float(pm.p95_latency), beta=0.01)
                self._baseline_p99 = update_baseline(self._baseline_p99, float(pm.p99_latency), beta=0.01)
                # Calculate drift
                base95 = self._baseline_p95 or pm.p95_latency
                drift95 = (pm.p95_latency - base95) / max(base95, 1e-6)
                base99 = self._baseline_p99 or pm.p99_latency
                drift99 = (pm.p99_latency - base99) / max(base99, 1e-6)
                # Add attributes for monitoring loop and alerts
                pm.ewma_p95 = self._ewma_p95  # type: ignore
                pm.ewma_p99 = self._ewma_p99  # type: ignore
                pm.ewma_error_rate = self._ewma_error  # type: ignore
                pm.p95_drift = float(drift95)  # type: ignore
                pm.p95_drift_alert = bool(drift95 > 0.3)  # type: ignore
                pm.p99_drift = float(drift99)  # type: ignore
                pm.p99_drift_alert = bool(drift99 > 0.3)  # type: ignore
            except Exception as e:
                logger = PiscesLxCoreLog("pisceslx.observability.service")
                logger.error("Error updating EWMA and baseline metrics", error=str(e))
            return pm
    
    def get_anomaly_insights(self) -> List[AnomalyPattern]:
        """
        Get potential anomaly insights.

        Returns:
            List[AnomalyPattern]: A list of detected anomaly patterns.
        """
        with self._lock:
            return list(self.anomaly_patterns)

class AutoDiscoveryEngine:
    """
    Automated discovery engine for detecting services and dependencies in the runtime environment.
    """
    
    def __init__(self):
        """
        Initialize the automated discovery engine.
        """
        self.discovered_services = {}
        self.service_health = {}
        self._discovery_lock = threading.Lock()
        self.logger = PiscesLxCoreLog("pisceslx.observability.auto_discovery")
        
    def discover_environment(self) -> Dict[str, Any]:
        """
        Automatically discover the runtime environment and services.

        Returns:
            Dict[str, Any]: A dictionary containing runtime environment, services, dependencies, 
                           resources information, and recommended configurations.
        """
        env_info = {
            "runtime": self._detect_runtime_environment(),
            "services": self._scan_for_services(),
            "dependencies": self._analyze_dependencies(),
            "resources": self._assess_resource_availability()
        }
        
        # Generate intelligent configuration recommendations
        env_info["recommended_config"] = self._generate_config_recommendations(env_info)
        
        return env_info
    
    def _detect_runtime_environment(self) -> Dict[str, Any]:
        """
        Detect the runtime environment and perform performance analysis.

        Returns:
            Dict[str, Any]: A dictionary containing runtime environment information.
        """
        runtime = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "is_container": self._is_running_in_container(),
            "is_cloud": self._detect_cloud_environment(),
            "resource_constraints": self._assess_resource_availability()
        }
        
        # Add optimizations for specific environments
        if runtime["is_container"]:
            runtime["optimizations"] = ["Enable memory-aware caching", "Use lightweight logging"]
        if runtime["is_cloud"]:
            runtime["optimizations"] = ["Enable distributed tracing", "Use cloud-native monitoring"]
            
        return runtime
    
    def _is_running_in_container(self) -> bool:
        """
        Automatically detect if the application is running inside a container.

        Returns:
            bool: True if running inside a container, False otherwise.
        """
        return (
            os.path.exists("/.dockerenv") or
            (os.path.exists("/proc/1/cgroup") and "docker" in open("/proc/1/cgroup").read())
        )
    
    def _detect_cloud_environment(self) -> str:
        """
        Detect the cloud provider using an intelligent heuristic approach.

        Returns:
            str: The name of the detected cloud provider. Returns "unknown" if not detected.
        """
        # Check for AWS
        if os.environ.get("AWS_REGION") or os.path.exists("/sys/hypervisor/uuid"):
            return "aws"
        # Check for GCP
        if os.environ.get("GOOGLE_CLOUD_PROJECT") or os.path.exists("/sys/class/dmi/id/product_name"):
            with open("/sys/class/dmi/id/product_name") as f:
                if "Google" in f.read():
                    return "gcp"
        # Check for Azure
        if os.environ.get("AZURE_SUBSCRIPTION_ID") or os.path.exists("/sys/class/dmi/id/chassis_asset_tag"):
            with open("/sys/class/dmi/id/chassis_asset_tag") as f:
                if "Microsoft" in f.read():
                    return "azure"
        return "unknown"
    
    def _scan_for_services(self) -> List[Dict[str, Any]]:
        """
        Automatically scan for available services.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing information about available services.
        """
        services = []
        
        # Check common AI/ML services
        service_checks = [
            ("redis", self._check_redis_availability),
            ("elasticsearch", self._check_elasticsearch_availability),
            ("prometheus", self._check_prometheus_availability),
            ("grafana", self._check_grafana_availability),
            ("tensorboard", self._check_tensorboard_availability)
        ]
        
        for service_name, check_func in service_checks:
            try:
                if check_func():
                    services.append({
                        "name": service_name,
                        "status": "available",
                        "recommended_integration": self._get_integration_recommendation(service_name)
                    })
            except Exception as e:
                self.logger.debug(
                    f"Service check failed for {service_name}",
                    event="SERVICE_CHECK_FAILED",
                    service=service_name,
                    error=str(e)
                )
        
        return services
    
    def _check_redis_availability(self) -> bool:
        """
        Check if the Redis service is available.

        Returns:
            bool: True if the Redis service is available, False otherwise.
        """
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
            r.ping()
            return True
        except Exception:
            return False
    
    def _check_elasticsearch_availability(self) -> bool:
        """
        Check if the Elasticsearch service is available.

        Returns:
            bool: True if the Elasticsearch service is available, False otherwise.
        """
        try:
            import requests
            response = requests.get('http://localhost:9200/_cluster/health', timeout=1)
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_prometheus_availability(self) -> bool:
        """
        Check if the Prometheus service is available.

        Returns:
            bool: True if the Prometheus service is available, False otherwise.
        """
        try:
            import requests
            response = requests.get('http://localhost:9090/-/healthy', timeout=1)
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_grafana_availability(self) -> bool:
        """
        Check if the Grafana service is available.

        Returns:
            bool: True if the Grafana service is available, False otherwise.
        """
        try:
            import requests
            response = requests.get('http://localhost:3000/api/health', timeout=1)
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_tensorboard_availability(self) -> bool:
        """
        Check if the TensorBoard service is available.

        Returns:
            bool: True if the TensorBoard service is available, False otherwise.
        """
        try:
            import requests
            response = requests.get('http://localhost:6006/data/logdir', timeout=1)
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_integration_recommendation(self, service_name: str) -> List[str]:
        """
        Get intelligent integration recommendations for a given service.

        Args:
            service_name (str): The name of the service.

        Returns:
            List[str]: A list of integration recommendations.
        """
        recommendations = {
            "redis": [
                "Enable Redis metrics collection",
                "Configure Redis connection pooling",
                "Set up Redis health checks"
            ],
            "elasticsearch": [
                "Enable Elasticsearch metrics collection",
                "Configure search query logging",
                "Set up index health monitoring"
            ],
            "prometheus": [
                "Configure Prometheus scraping",
                "Set up alerting rules",
                "Enable remote write"
            ],
            "grafana": [
                "Import recommended dashboards",
                "Configure data source connections",
                "Set up alert notifications"
            ],
            "tensorboard": [
                "Enable TensorBoard logging",
                "Configure experiment tracking",
                "Set up model visualization"
            ]
        }
        return recommendations.get(service_name, ["Enable service metrics collection"])
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """
        Analyze project dependencies.

        Returns:
            Dict[str, Any]: A dictionary containing information about installed packages and critical dependencies.
        """
        try:
            import pkg_resources
            installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
            return {
                "installed_packages": installed_packages,
                "critical_deps": self._check_critical_dependencies(installed_packages)
            }
        except Exception:
            return {"error": "Failed to analyze dependencies"}
    
    def _check_critical_dependencies(self, installed_packages: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Check the version compatibility of critical dependencies.

        Args:
            installed_packages (Dict[str, str]): A dictionary of installed packages and their versions.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing information about critical dependencies.
        """
        critical_deps = []
        # Check critical dependencies
        critical_packages = {
            "torch": ">=1.9.0",
            "transformers": ">=4.0.0",
            "numpy": ">=1.19.0"
        }
        
        for package, min_version in critical_packages.items():
            if package in installed_packages:
                critical_deps.append({
                    "package": package,
                    "version": installed_packages[package],
                    "status": "ok"
                })
            else:
                critical_deps.append({
                    "package": package,
                    "version": "not found",
                    "status": "missing"
                })
                
        return critical_deps
    
    def _assess_resource_availability(self) -> Dict[str, Any]:
        """
        Evaluate the availability of system resources.

        Returns:
            Dict[str, Any]: A dictionary containing information about CPU, memory, and disk resources.
        """
        try:
            import psutil
            # Get CPU information
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory information
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            available_memory_gb = memory.available / (1024**3)
            
            # Get disk information
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            available_disk_gb = disk.free / (1024**3)
            
            return {
                "cpu": {
                    "count": cpu_count,
                    "usage_percent": cpu_percent
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": available_memory_gb,
                    "usage_percent": memory_percent
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "available_gb": available_disk_gb,
                    "usage_percent": disk_percent
                }
            }
        except Exception:
            return {"error": "Failed to assess resources"}
    
    def _generate_config_recommendations(self, env_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate configuration recommendations based on environment information.

        Args:
            env_info (Dict[str, Any]): A dictionary containing environment information.

        Returns:
            Dict[str, Any]: A dictionary containing configuration recommendations.
        """
        recommendations = {}
        
        # Recommendations based on CPU core count
        cpu_count = env_info.get("resources", {}).get("cpu", {}).get("count", 0)
        if cpu_count > 0:
            recommendations["threading"] = {
                "max_workers": min(32, max(4, cpu_count * 2)),
                "description": f"Recommended max workers based on {cpu_count} CPU cores"
            }
        
        # Recommendations based on available memory
        available_memory = env_info.get("resources", {}).get("memory", {}).get("available_gb", 0)
        if available_memory > 0:
            if available_memory < 4:
                recommendations["memory"] = {
                    "strategy": "conservative",
                    "description": "Low memory environment, use memory-conservative settings"
                }
            elif available_memory > 32:
                recommendations["memory"] = {
                    "strategy": "aggressive",
                    "description": "High memory environment, can use aggressive caching"
                }
        
        # Recommendations based on cloud environment
        cloud_provider = env_info.get("runtime", {}).get("is_cloud", "unknown")
        if cloud_provider != "unknown":
            recommendations["cloud"] = {
                "provider": cloud_provider,
                "optimizations": [
                    "Enable cloud-native metrics collection",
                    "Use managed services where possible",
                    "Configure auto-scaling"
                ]
            }
            
        return recommendations

class PiscesLxCoreObservabilityService:
    """
    Core class for the Pisces Lx observability service, providing unified monitoring, 
    metric collection, anomaly detection, and reporting capabilities.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        """
        Initialize the observability service.
        """
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.logger = PiscesLxCoreLog("pisceslx.observability.service")
            self.state = ServiceState.INITIALIZING
            self.state_history = deque(maxlen=100)
            self.metrics_collector = IntelligentMetricsCollector()
            self.metrics_registry = PiscesLxCoreMetricsRegistry.instance()
            self.collector_registry = get_default_collector_registry()
            try:
                from utils.fs.core import PiscesLxCoreFS as _FS
                _fs = _FS()
                self.reporter = PiscesLxCoreReporter(str(_fs.reports_dir()))
            except Exception:
                # Fallback (will be normalized by FS layer elsewhere when writing)
                self.reporter = PiscesLxCoreReporter(".pisceslx/reports")
            self.hook_bus = get_global_hook_bus()
            try:
                self.hook_bus.registry.register_listener(
                    PiscesLxCoreFunctionListener(callback=_tools_request_success_listener, event_types=['tools.request.success'])
                )
                self.hook_bus.registry.register_listener(
                    PiscesLxCoreFunctionListener(callback=_tools_request_error_listener, event_types=['tools.request.error'])
                )
            except Exception as e:
                self.logger.warning("Failed to register tool request listeners", error=str(e))
            self.cache = get_default_cache()
            self.auto_discovery = AutoDiscoveryEngine()
            self.trend_analyzer = EnterpriseTrendAnalyzer()
            self.tracer = Tracer()
            self.alert_manager = AlertManager()
            # Connect alert manager to metrics collector for error rate calculation
            self.alert_manager._metrics_collector = self.metrics_collector
            
            # Register default alert rules
            self._setup_default_alerts()
            
            # Runtime metadata
            self.runtime_meta = {}
            self.started_at = datetime.utcnow().isoformat() + "Z"
            
            # Internal monitoring
            self._internal_monitoring = True
            self._monitoring_thread = None
            self._stop_monitoring = threading.Event()
            
            # HTTP self-check server
            self._http_server = None
            self._http_thread = None
            
            # Exporters
            self._prom_exporter = None
            self._otlp_exporter = None
            
            # Alert system
            self._alerts = build_alerts({})
            
            # Performance optimization
            self._sampling_rate = 1.0
            self._adaptive_sampling = True
            
            # Record the initial state transition
            self._record_state_transition(ServiceState.INITIALIZING, ServiceState.HEALTHY)
            self.state = ServiceState.HEALTHY

        # High error rate alert
        self.alert_manager.add_rule(AlertRule(
            name="high_error_rate",
            metric="error_rate",
            threshold=0.1,
            operator=">",
            duration=60,
            severity="high",
            description="Error rate exceeds 10%"
        ))
        
        # High latency alert
        self.alert_manager.add_rule(AlertRule(
            name="high_latency",
            metric="p95_latency",
            threshold=500,
            operator=">",
            duration=60,
            severity="medium",
            description="P95 latency exceeds 500ms"
        ))
        
        # Low health score alert
        self.alert_manager.add_rule(AlertRule(
            name="low_health_score",
            metric="health_score",
            threshold=80,
            operator="<",
            duration=120,
            severity="high",
            description="Health score below 80"
        ))
    
    def get_logger(self):
        """
        Get the logger instance.

        Returns:
            PiscesLxCoreLog: The logger instance.
        """
        return self.logger
    
    @classmethod
    def instance(cls):
        """
        Get the singleton instance of the observability service.

        Returns:
            PiscesLxCoreObservabilityService: The service instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _setup_default_alerts(self) -> None:
        """Register default alert rules on the AlertManager.
        This is idempotent; re-adding with same names will overwrite.
        """
        try:
            # Error rate alert (critical if too high)
            self.alert_manager.add_rule(AlertRule(
                name="high_error_rate",
                metric="error_rate",
                threshold=0.10,
                operator=">",
                duration=60,
                severity="high",
                description="Error rate exceeds 10%"
            ))
            # Latency alert on p95
            self.alert_manager.add_rule(AlertRule(
                name="high_latency",
                metric="p95_latency",
                threshold=500.0,
                operator=">",
                duration=60,
                severity="medium",
                description="P95 latency exceeds 500ms"
            ))
            # Health score too low
            self.alert_manager.add_rule(AlertRule(
                name="low_health_score",
                metric="health_score",
                threshold=80.0,
                operator="<",
                duration=120,
                severity="high",
                description="Health score below 80"
            ))
        except Exception as e:
            self.logger.warning("default.alerts.setup.failed", error=str(e))
    
    def _record_state_transition(self, old_state: ServiceState, new_state: ServiceState, reason: str = ""):
        """
        Record the service state transition.

        Args:
            old_state (ServiceState): The old state.
            new_state (ServiceState): The new state.
            reason (str): The reason for the state transition.
        """
        transition = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "from": old_state.value,
            "to": new_state.value,
            "reason": reason
        }
        self.state_history.append(transition)
        self.logger.info("Service state transition", event="service.state.transition", **transition)
    
    def start_internal_monitoring(self, interval: float = 10.0):
        """
        Start the internal monitoring loop.

        Args:
            interval (float): The monitoring interval in seconds.
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
            
        self._internal_monitoring = True
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("observability.internal.monitoring.started", interval=interval)
    
    def stop_internal_monitoring(self):
        """
        Stop the internal monitoring loop.
        """
        self._internal_monitoring = False
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        self.logger.info("observability.internal.monitoring.stopped")
    
    def disable_internal_monitoring(self):
        """
        Disable internal monitoring.
        """
        self._internal_monitoring = False
    
    def _monitoring_loop(self, interval: float):
        """
        Internal monitoring loop.

        Args:
            interval (float): The monitoring interval in seconds.
        """
        while self._internal_monitoring and not self._stop_monitoring.is_set():
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Run health checks
                self._run_health_checks()
                
                # Update trend analysis
                self._update_trend_analysis()
                
                # Check alerts
                self._check_alerts()
                
                # Wait for the next cycle
                self._stop_monitoring.wait(interval)
            except Exception as e:
                self.logger.error("observability.internal.monitoring.error", error=str(e))
                self._stop_monitoring.wait(min(interval, 5.0))  # Shorten wait time on error
    
    def _collect_metrics(self):
        """
        Collect system and application metrics.
        """
        try:
            # Run default collectors
            for collector in self.collector_registry.list():
                try:
                    collector.collect(self.metrics_registry)
                except Exception as e:
                    self.logger.error("collector.error", collector=getattr(collector, '_name', 'unknown'), error=str(e), exc_info=True)
            
            # Collect runtime metadata
            self._collect_runtime_metadata()
        except Exception as e:
            self.logger.error("metrics.collection.error", error=str(e), exc_info=True)
    
    def _collect_runtime_metadata(self):
        """
        Collect runtime metadata.
        """
        try:
            import psutil
            import platform
            
            # Get system information
            self.runtime_meta["system"] = {
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
                "python_version": platform.python_version(),
                "pid": os.getpid()
            }
            
            # Get CPU information
            cpu_info = {
                "count": psutil.cpu_count(),
                "usage_percent": psutil.cpu_percent(interval=0.1)
            }
            self.runtime_meta["cpu"] = cpu_info
            
            # Get memory information
            memory = psutil.virtual_memory()
            memory_info = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "usage_percent": memory.percent
            }
            self.runtime_meta["memory"] = memory_info
            
            # Get GPU information (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                gpu_info = []
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_info.append({
                        "index": i,
                        "memory_total_gb": mem_info.total / (1024**3),
                        "memory_used_gb": mem_info.used / (1024**3),
                        "memory_usage_percent": (mem_info.used / mem_info.total) * 100,
                        "utilization_percent": util_info.gpu
                    })
                self.runtime_meta["gpu_info"] = gpu_info
            except Exception as e:
                self.logger.warning("GPU information not available", error=str(e))
        except Exception as e:
            self.logger.error("runtime.metadata.collection.error", error=str(e))
    
    def _run_health_checks(self):
        """
        Run health checks.
        """
        try:
            metrics = self.metrics_collector.get_current_metrics()
            health_score = metrics.health_score
            
            # Update service state
            old_state = self.state
            if health_score >= 90:
                new_state = ServiceState.HEALTHY
            elif health_score >= 70:
                new_state = ServiceState.DEGRADED
            else:
                new_state = ServiceState.FAILED
                
            if old_state != new_state:
                self._record_state_transition(old_state, new_state, f"Health score: {health_score:.1f}")
                self.state = new_state
                
            # Record health metrics
            self.metrics_registry.gauge("service_health_score").set(health_score)
            self.metrics_registry.gauge("service_state").set(float(list(ServiceState).index(self.state)))
        except Exception as e:
            self.logger.error("health.check.error", error=str(e))
    
    def _update_trend_analysis(self):
        """
        Update trend analysis.
        """
        try:
            metrics = self.metrics_collector.get_current_metrics()
            # Analyze P95 latency trend
            if metrics.p95_latency > 0:
                analysis = self.trend_analyzer.analyze(metrics.p95_latency)
                # Record trend metrics
                self.metrics_registry.gauge("latency_trend_ewma").set(analysis.ewma)
                self.metrics_registry.gauge("latency_trend_baseline").set(analysis.baseline)
                self.metrics_registry.gauge("latency_drift_ratio").set(analysis.drift_ratio)
        except Exception as e:
            self.logger.error("trend.analysis.error", error=str(e))
    
    def _check_alerts(self, metrics: PerformanceMetrics = None):
        """
        Check and trigger alerts.
        
        Args:
            metrics (PerformanceMetrics): Current performance metrics. If None, will be fetched internally.
        """
        try:
            # Get metrics if not provided
            if metrics is None:
                metrics = self.metrics_collector.get_current_metrics()
            
            # Use alert manager to check alerts with current metrics
            self.alert_manager._check_all_rules()
            
            # Compatible old alert logic
            if hasattr(self, '_alerts'):
                # Check high error rate alert
                if metrics.error_rate > 0.1:
                    self._alerts.trigger("high_error_rate", {
                        "error_rate": metrics.error_rate,
                        "threshold": 0.1
                    })
                
                # Check high latency alert
                if metrics.p95_latency > 500:  # 500ms
                    self._alerts.trigger("high_latency", {
                        "p95_latency": metrics.p95_latency,
                        "threshold": 500
                    })
                    
                # Check low health score alert
                if metrics.health_score < 80:
                    self._alerts.trigger("low_health_score", {
                        "health_score": metrics.health_score,
                        "threshold": 80
                    })
            
            # Log metrics in consistent format with monitor
            self.logger.info(
                "New observability metrics",
                cpu_usage=metrics.cpu_usage,
                memory_usage=metrics.memory_usage,
                throughput=metrics.throughput,
                p95_latency=metrics.p95_latency,
                p99_latency=metrics.p99_latency,
                error_rate=metrics.error_rate,
                health_score=metrics.health_score
            )
            
        except Exception as e:
            self.logger.error("alert check error", error=str(e))
    
    def write_device_report(self, data: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """
        Generate a device report.

        Args:
            data (Dict[str, Any]): Device data.
            session_id (Optional[str]): Session ID.

        Returns:
            str: The path of the generated report file.
        """
        try:
            # Use the new report generator with device data directly
            from .reports.generator import ReportGenerator
            generator = ReportGenerator()
            # The new generator expects device_id and duration_hours, but we're passing device data
            # For now, use the old reporter approach which handles device data correctly
            return generator._generate_device_report_from_data(data, session_id)
        except Exception as e:
            self.logger.error("write_device_report.error", error=str(e))
            # Fall back to the old reporter
            if hasattr(self, 'reporter'):
                return self.reporter.write_device_report(data, session_id)
            raise

    def get_current_performance_metrics(self):
        """
        Get current performance metrics from the metrics collector.
        
        Returns:
            PerformanceMetrics: Current performance metrics object.
        """
        try:
            return self.metrics_collector.get_current_metrics()
        except Exception as e:
            self.logger.error("get_current_performance_metrics.error", error=str(e))
            return None
    
    def get_health_snapshot(self) -> Dict[str, Any]:
        """
        Get a health snapshot of the service.

        Returns:
            Dict[str, Any]: Health snapshot data.
        """
        try:
            metrics = self.metrics_collector.get_current_metrics()
            return {
                "state": self.state.value,
                "health_score": metrics.health_score,
                "timestamp": metrics.timestamp.isoformat() + "Z",
                "metrics": {
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "error_rate": metrics.error_rate,
                    "throughput": metrics.throughput,
                    "p95_latency": metrics.p95_latency,
                    "p99_latency": metrics.p99_latency
                },
                "runtime_meta": self.runtime_meta
            }
        except Exception as e:
            self.logger.error("health.snapshot.error", error=str(e))
            return {
                "state": self.state.value,
                "error": str(e)
            }
    
    def start_http_server(self, host: str = "localhost", port: int = 8080):
        """
        Start the HTTP self-check server.

        Args:
            host (str): Server host address.
            port (int): Server port.
        """
        if self._http_server:
            return
            
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class ObservabilityHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self._handle_health_check()
                elif self.path == "/metrics":
                    self._handle_metrics()
                elif self.path == "/":
                    self._handle_root()
                elif self.path == "/alerts":
                    self._handle_alerts()
                else:
                    self._handle_not_found()
            
            def _handle_health_check(self):
                try:
                    service = PiscesLxCoreObservabilityService.instance()
                    health = service.get_health_snapshot()
                    self._send_json_response(200, health)
                except Exception as e:
                    self._send_json_response(500, error=str(e))
            
            def _handle_metrics(self):
                try:
                    service = PiscesLxCoreObservabilityService.instance()
                    snapshot = service.metrics_registry.snapshot()
                    self._send_json_response(200, snapshot)
                except Exception as e:
                    self._send_json_response(500, error=str(e))
            
            def _handle_alerts(self):
                try:
                    service = PiscesLxCoreObservabilityService.instance()
                    alerts = service.alert_manager.get_active_alerts()
                    alert_data = [
                        {
                            "rule_name": alert.rule_name,
                            "metric": alert.metric,
                            "value": alert.value,
                            "threshold": alert.threshold,
                            "severity": alert.severity,
                            "timestamp": alert.timestamp.isoformat() + "Z",
                            "resolved": alert.resolved
                        }
                        for alert in alerts
                    ]
                    self._send_json_response(200, {"alerts": alert_data})
                except Exception as e:
                    self._send_json_response(500, error=str(e))
            
            def _handle_root(self):
                try:
                    service = PiscesLxCoreObservabilityService.instance()
                    health = service.get_health_snapshot()
                    response = {
                        "service": "PiscesL1 Observability Service",
                        "status": "running",
                        "health": health,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                    self._send_json_response(200, response)
                except Exception as e:
                    self._send_json_response(500, error=str(e))
            
            def _handle_not_found(self):
                self._send_json_response(404, {"error": "Not found"})
            
            def _send_json_response(self, status_code: int, data: Dict[str, Any]):
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))
            
            def log_message(self, format, *args):
                # Disable default logging
                pass
        
        try:
            self._http_server = HTTPServer((host, port), ObservabilityHandler)
            self._http_thread = threading.Thread(
                target=self._http_server.serve_forever,
                daemon=True
            )
            self._http_thread.start()
            self.logger.success("http.server.started", host=host, port=port)
        except Exception as e:
            self.logger.error("http.server.start.error", error=str(e))
    
    def stop_http_server(self):
        """
        Stop the HTTP self-check server.
        """
        if self._http_server:
            try:
                self._http_server.shutdown()
                self._http_server.server_close()
                self._http_server = None
                self.logger.info("http.server.stopped")
            except Exception as e:
                self.logger.error("http.server.stop.error", error=str(e))
    
    def start_prometheus_exporter(self, path: str = "/tmp/metrics.prom"):
        """
        Start the Prometheus text file exporter.

        Args:
            path (str): Export file path.
        """
        try:
            # Normalize path under project .pisceslx/observability
            try:
                from utils.fs.core import PiscesLxCoreFS as _FS
                path = str(_FS().normalize_under_category('observability', path))
            except Exception:
                pass
            self._prom_exporter = PromTextfileExporter(path)
            self._prom_exporter.start()
            self.logger.success("prometheus.exporter.started", path=path)
        except Exception as e:
            self.logger.error("prometheus.exporter.start.error", error=str(e))
    
    def stop_prometheus_exporter(self):
        """
        Stop the Prometheus text file exporter.
        """
        if self._prom_exporter:
            try:
                self._prom_exporter.stop()
                self._prom_exporter = None
                self.logger.info("prometheus.exporter.stopped")
            except Exception as e:
                self.logger.error("prometheus.exporter.stop.error", error=str(e))
    
    def start_otlp_exporter(self, endpoint: str, interval_sec: int = 15):
        """
        Start the OTLP exporter.

        Args:
            endpoint (str): OTLP endpoint URL.
            interval_sec (int): Export interval in seconds.
        """
        try:
            from .exporters.otlp_exporter import OtlpExporter
            self._otlp_exporter = OtlpExporter(endpoint, interval_sec)
            self._otlp_exporter.start()
            self.logger.success("otlp.exporter.started", endpoint=endpoint)
        except Exception as e:
            self.logger.error("otlp.exporter.start.error", error=str(e))
    
    def stop_otlp_exporter(self):
        """
        Stop the OTLP exporter.
        """
        if self._otlp_exporter:
            try:
                self._otlp_exporter.stop()
                self._otlp_exporter = None
                self.logger.info("otlp.exporter.stopped")
            except Exception as e:
                self.logger.error("otlp.exporter.stop.error", error=str(e))
    
    def generate_device_report(self) -> Dict[str, Any]:
        """
        Generate a device report.

        Returns:
            Dict[str, Any]: Device report data.
        """
        try:
            # Use the new report generator
            from .reports.generator import ReportGenerator
            generator = ReportGenerator()
            
            # Get device ID and running duration
            device_id = f"pisces-l1-{self.started_at}"
            duration_hours = int((datetime.utcnow() - datetime.fromisoformat(self.started_at.replace('Z', '+00:00'))).total_seconds() / 3600)
            
            # Generate device report
            report = generator.generate_device_report(device_id, duration_hours)
            return report
        except Exception as e:
            self.logger.error("device report generation error", error=str(e))
            # Fall back to the old report logic
            try:
                # Collect the latest metrics
                metrics = self.metrics_collector.get_current_metrics()
                
                # Build report data
                report_data = {
                    "report_type": "device",
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "service_state": self.state.value,
                    "metrics": {
                        "request_count": metrics.request_count,
                        "error_count": metrics.error_count,
                        "avg_latency": metrics.avg_latency,
                        "p95_latency": metrics.p95_latency,
                        "p99_latency": metrics.p99_latency,
                        "throughput": metrics.throughput,
                        "memory_usage": metrics.memory_usage,
                        "cpu_usage": metrics.cpu_usage,
                        "health_score": metrics.health_score
                    }
                }
                
                # Use the report writer to enhance the report data
                if hasattr(self, 'reporter'):
                    enhanced_report = build_device_report_payload(self, report_data)
                    report = self.reporter.generate_report("device", enhanced_report)
                    return report
                else:
                    return report_data
            except Exception as fallback_error:
                self.logger.error("device report fallback error", error=str(fallback_error))
                return {"error": str(e)}
    
    def generate_session_report(self) -> Dict[str, Any]:
        """
        Generate a session report.

        Returns:
            Dict[str, Any]: Session report data.
        """
        try:
            # Use the new report generator
            from .reports.generator import ReportGenerator
            generator = ReportGenerator()
            
            # Get session information
            session_id = f"session-{self.started_at}"
            start_time = datetime.fromisoformat(self.started_at.replace('Z', '+00:00'))
            end_time = datetime.utcnow()
            
            # Generate session report
            report = generator.generate_session_report(session_id, start_time, end_time)
            return report
        except Exception as e:
            self.logger.error("session report generation error", error=str(e))
            # Fall back to the old report logic
            try:
                # Build report data
                report_data = {
                    "report_type": "session",
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "session_started_at": self.started_at
                }
                
                # Use the report writer to enhance the report data
                if hasattr(self, 'reporter'):
                    enhanced_report = build_session_report_payload(self, report_data)
                    report = self.reporter.generate_report("session", enhanced_report)
                    return report
                else:
                    return report_data
            except Exception as fallback_error:
                self.logger.error("session report fallback error", error=str(fallback_error))
                return {"error": str(e)}
    
    def start_trace(self, operation_name: str, parent_context: TraceContext = None) -> TraceContext:
        """
        Start a new distributed trace.

        Args:
            operation_name (str): Operation name.
            parent_context (TraceContext): Parent trace context.

        Returns:
            TraceContext: The trace context.
        """
        return self.tracer.start_span(operation_name, parent_context)
    
    def finish_trace(self, context: TraceContext):
        """
        Finish a distributed trace.

        Args:
            context (TraceContext): The trace context.
        """
        self.tracer.finish_span(context)
    
    def get_tracer(self) -> Tracer:
        """
        Get the tracer instance.

        Returns:
            Tracer: The tracer instance.
        """
        return self.tracer
    
    def get_alert_manager(self) -> AlertManager:
        """
        Get the alert manager instance.

        Returns:
            AlertManager: The alert manager instance.
        """
        return self.alert_manager
    
    def add_alert_rule(self, rule: AlertRule):
        """
        Add an alert rule.

        Args:
            rule (AlertRule): The alert rule.
        """
        self.alert_manager.add_rule(rule)
    
    def remove_alert_rule(self, rule_name: str):
        """
        Remove an alert rule.

        Args:
            rule_name (str): The name of the alert rule.
        """
        self.alert_manager.remove_rule(rule_name)
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get all active alerts.

        Returns:
            List[Alert]: List of active alerts.
        """
        return self.alert_manager.get_active_alerts()
    
    def _calculate_resource_efficiency(self) -> float:
        """
        Calculate the resource efficiency metric.

        Returns:
            float: Resource efficiency score (0-100).
        """
        try:
            metrics = self.metrics_collector.get_current_metrics()
            
            # CPU efficiency: Lower usage leads to higher score (ideal range: 30-70%)
            cpu_efficiency = 0.0
            if 30 <= metrics.cpu_usage <= 70:
                cpu_efficiency = 100.0
            elif metrics.cpu_usage < 30:
                cpu_efficiency = 50.0 + (metrics.cpu_usage / 30.0) * 50.0
            else:  # > 70%
                cpu_efficiency = max(0.0, 100.0 - ((metrics.cpu_usage - 70.0) / 30.0) * 100.0)
            
            # Memory efficiency: Lower usage leads to higher score (ideal range: 40-80%)
            memory_efficiency = 0.0
            if 40 <= metrics.memory_usage <= 80:
                memory_efficiency = 100.0
            elif metrics.memory_usage < 40:
                memory_efficiency = 60.0 + (metrics.memory_usage / 40.0) * 40.0
            else:  # > 80%
                memory_efficiency = max(0.0, 100.0 - ((metrics.memory_usage - 80.0) / 20.0) * 100.0)
            
            # Throughput efficiency: Higher throughput is better (based on experience)
            throughput_efficiency = 0.0
            if metrics.throughput > 0:
                if metrics.throughput >= 100:  # 100 req/s or higher
                    throughput_efficiency = 100.0
                else:
                    throughput_efficiency = min(100.0, metrics.throughput)
            
            # Error penalty
            error_penalty = metrics.error_rate * 100.0
            
            # Comprehensive calculation (weighted average)
            total_efficiency = (
                cpu_efficiency * 0.3 +
                memory_efficiency * 0.3 +
                throughput_efficiency * 0.2 -
                error_penalty * 0.2
            )
            
            return max(0.0, min(100.0, total_efficiency))
            
        except Exception as e:
            self.logger.error("calculate resource efficiency error", error=str(e))
            return 0.0

    def _intelligent_state_transition(self, metrics: PerformanceMetrics) -> None:
        """
        Intelligently transition service state based on current metrics.
        
        This method analyzes current performance metrics and determines if the service
        should transition to a different operational state (e.g., from HEALTHY to DEGRADED,
        or from DEGRADED to HEALTHY).
        
        Args:
            metrics (PerformanceMetrics): Current performance metrics.
        """
        try:
            current_state = self.state
            new_state = current_state
            
            # Define state transition thresholds
            if metrics.health_score >= 90 and metrics.error_rate < 0.01:
                new_state = ServiceState.HEALTHY
            elif metrics.health_score >= 70 and metrics.error_rate < 0.05:
                new_state = ServiceState.STABLE
            elif metrics.health_score >= 50 or metrics.error_rate < 0.1:
                new_state = ServiceState.DEGRADED
            else:
                new_state = ServiceState.CRITICAL
            
            # Only transition if state has changed
            if new_state != current_state:
                self._record_state_transition(current_state, new_state, metrics)
                self.state = new_state
                
                # Emit state change event
                try:
                    from utils.hooks.bus import get_global_hook_bus
                    get_global_hook_bus().emit(
                        "service.state.changed",
                        from_state=current_state.value,
                        to_state=new_state.value,
                        metrics=metrics.__dict__
                    )
                except Exception as e:
                    self.logger.error("Failed to emit state change event", error=str(e))
                    
        except Exception as e:
            self.logger.error("observability.service.state.transition.error", error=str(e))

    def _cleanup(self):
        """
        Clean up resources.
        """
        try:
            # Stop monitoring
            self.stop_internal_monitoring()
            
            # Stop HTTP server
            self.stop_http_server()
            
            # Stop exporters
            self.stop_prometheus_exporter()
            self.stop_otlp_exporter()
            
            self.logger.info("observability.service.cleaned.up")
        except Exception as e:
            self.logger.error("observability.service.cleanup.error", error=str(e))

