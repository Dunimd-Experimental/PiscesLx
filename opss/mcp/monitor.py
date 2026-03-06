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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

import threading
import time
import psutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import deque
from pathlib import Path

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from .types import (
    POPSSMCPConfiguration,
    POPSSMCPHealthStatus,
    POPSSMCPPerformanceMetrics,
    POPSSMCPModuleStatus
)

class POPSSMCPHealthCheckStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class POPSSMCPComponentHealth:
    component_name: str
    status: POPSSMCPHealthCheckStatus
    last_check: datetime
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class POPSSMCPMonitorAlert:
    alert_id: str
    severity: str
    component: str
    message: str
    timestamp: datetime
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class POPSSMCPMonitor:
    def __init__(self, config: Optional[POPSSMCPConfiguration] = None):
        self.config = config or POPSSMCPConfiguration()
        self._LOG = self._configure_logging()
        
        self._lock = threading.RLock()
        
        self.health_status = POPSSMCPHealthStatus(
            status="healthy",
            component_statuses={},
            alerts=[],
            recommendations=[],
            last_updated=datetime.now()
        )
        
        self.performance_metrics = POPSSMCPPerformanceMetrics()
        
        self.component_health: Dict[str, POPSSMCPComponentHealth] = {}
        
        self.alerts: List[POPSSMCPMonitorAlert] = []
        self.alert_history: deque = deque(maxlen=1000)
        
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._monitoring_active = False
        
        self._health_checks: Dict[str, Callable] = {}
        self._register_default_health_checks()
        
        self._performance_history: Dict[str, deque] = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'execution_times': deque(maxlen=100),
            'queue_sizes': deque(maxlen=100),
        }
        
        self._last_health_check = datetime.now()
        self._health_check_interval = self.config.monitoring_interval
        
        self._thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 75.0,
            'memory_critical': 90.0,
            'execution_time_warning': 5.0,
            'execution_time_critical': 30.0,
            'queue_size_warning': 100,
            'queue_size_critical': 500,
            'error_rate_warning': 0.05,
            'error_rate_critical': 0.15,
        }
        
        self._notification_handlers: Dict[str, List[Callable]] = {
            'warning': [],
            'critical': [],
            'info': [],
        }
        
        self._LOG.info("POPSSMCPMonitor initialized")
    
    def _configure_logging(self) -> PiscesLxLogger:
        _LOG = PiscesLxLogger("PiscesLx.Opss.MCP",file_path=get_log_file("PiscesLx.Opss.MCP"), enable_file=True)
        return _LOG
    
    def _register_default_health_checks(self):
        self._health_checks = {
            'cpu_usage': self._check_cpu_health,
            'memory_usage': self._check_memory_health,
            'disk_usage': self._check_disk_health,
            'execution_latency': self._check_execution_latency,
            'queue_status': self._check_queue_health,
        }
    
    def start_monitoring(self):
        if self._monitoring_active:
            self._LOG.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._stop_event.clear()
        
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="piscesl1_mcp_monitor",
            daemon=True
        )
        self._monitoring_thread.start()
        
        self._LOG.info("Monitoring started")
    
    def stop_monitoring(self):
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        self._stop_event.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        self._LOG.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        while not self._stop_event.is_set():
            try:
                self.run_health_check()
                
                self._collect_performance_metrics()
                
                self._check_thresholds()
                
                self._update_alerts()
                
            except Exception as e:
                self._LOG.error(f"Error in monitoring loop: {e}")
            
            self._stop_event.wait(self.config.monitoring_interval)
    
    def run_health_check(self) -> POPSSMCPHealthStatus:
        with self._lock:
            overall_status = "healthy"
            component_statuses = {}
            recommendations = []
            
            for check_name, check_func in self._health_checks.items():
                try:
                    result = check_func()
                    
                    status = POPSSMCPHealthCheckStatus.UNKNOWN
                    message = ""
                    
                    if isinstance(result, tuple) and len(result) >= 2:
                        status, message = result[0], result[1]
                    elif isinstance(result, dict):
                        status = result.get('status', POPSSMCPHealthCheckStatus.UNKNOWN)
                        message = result.get('message', "")
                    
                    component_health = POPSSMCPComponentHealth(
                        component_name=check_name,
                        status=status,
                        last_check=datetime.now(),
                        message=message,
                        history=self.component_health.get(check_name, POPSSMCPComponentHealth(
                            component_name=check_name,
                            status=POPSSMCPHealthCheckStatus.UNKNOWN,
                            last_check=datetime.now()
                        )).history[-9:] + [{
                            'status': status.value if hasattr(status, 'value') else str(status),
                            'timestamp': datetime.now().isoformat(),
                            'message': message
                        }]
                    )
                    
                    self.component_health[check_name] = component_health
                    component_statuses[check_name] = status.value if hasattr(status, 'value') else str(status)
                    
                    if status == POPSSMCPHealthCheckStatus.UNHEALTHY:
                        overall_status = "unhealthy"
                    elif status == POPSSMCPHealthCheckStatus.DEGRADED and overall_status == "healthy":
                        overall_status = "degraded"
                    
                    if status == POPSSMCPHealthCheckStatus.DEGRADED:
                        recommendations.append(f"Component '{check_name}' is degraded: {message}")
                    
                except Exception as e:
                    self._LOG.error(f"Health check failed for {check_name}: {e}")
                    component_statuses[check_name] = "error"
                    overall_status = "degraded"
            
            self.health_status = POPSSMCPHealthStatus(
                status=overall_status,
                component_statuses=component_statuses,
                alerts=[alert.alert_id for alert in self.alerts if not alert.resolved],
                recommendations=recommendations,
                last_updated=datetime.now()
            )
            
            self._last_health_check = datetime.now()
            
            return self.health_status
    
    def _check_cpu_health(self) -> Tuple[POPSSMCPHealthCheckStatus, str]:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            self._performance_history['cpu_usage'].append({
                'value': cpu_percent,
                'timestamp': datetime.now()
            })
            
            if cpu_percent >= self._thresholds['cpu_critical']:
                return POPSSMCPHealthCheckStatus.UNHEALTHY, f"CPU usage critically high: {cpu_percent}%"
            elif cpu_percent >= self._thresholds['cpu_warning']:
                return POPSSMCPHealthCheckStatus.DEGRADED, f"CPU usage high: {cpu_percent}%"
            else:
                return POPSSMCPHealthCheckStatus.HEALTHY, f"CPU usage normal: {cpu_percent}%"
                
        except Exception as e:
            return POPSSMCPHealthCheckStatus.UNKNOWN, f"Unable to check CPU: {str(e)}"
    
    def _check_memory_health(self) -> Tuple[POPSSMCPHealthCheckStatus, str]:
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            self._performance_history['memory_usage'].append({
                'value': memory_percent,
                'timestamp': datetime.now()
            })
            
            if memory_percent >= self._thresholds['memory_critical']:
                return POPSSMCPHealthCheckStatus.UNHEALTHY, f"Memory usage critically high: {memory_percent}%"
            elif memory_percent >= self._thresholds['memory_warning']:
                return POPSSMCPHealthCheckStatus.DEGRADED, f"Memory usage high: {memory_percent}%"
            else:
                return POPSSMCPHealthCheckStatus.HEALTHY, f"Memory usage normal: {memory_percent}%"
                
        except Exception as e:
            return POPSSMCPHealthCheckStatus.UNKNOWN, f"Unable to check memory: {str(e)}"
    
    def _check_disk_health(self) -> Tuple[POPSSMCPHealthCheckStatus, str]:
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = disk_usage.percent
            
            if disk_percent >= 95:
                return POPSSMCPHealthCheckStatus.UNHEALTHY, f"Disk usage critically high: {disk_percent}%"
            elif disk_percent >= 85:
                return POPSSMCPHealthCheckStatus.DEGRADED, f"Disk usage high: {disk_percent}%"
            else:
                return POPSSMCPHealthCheckStatus.HEALTHY, f"Disk usage normal: {disk_percent}%"
                
        except Exception as e:
            return POPSSMCPHealthCheckStatus.UNKNOWN, f"Unable to check disk: {str(e)}"
    
    def _check_execution_latency(self) -> Tuple[POPSSMCPHealthCheckStatus, str]:
        try:
            recent_executions = list(self._performance_history['execution_times'])[-10:]
            
            if not recent_executions:
                return POPSSMCPHealthCheckStatus.HEALTHY, "No recent executions"
            
            avg_latency = sum(e['value'] for e in recent_executions) / len(recent_executions)
            
            if avg_latency >= self._thresholds['execution_time_critical']:
                return POPSSMCPHealthCheckStatus.UNHEALTHY, f"Average execution latency critically high: {avg_latency:.2f}s"
            elif avg_latency >= self._thresholds['execution_time_warning']:
                return POPSSMCPHealthCheckStatus.DEGRADED, f"Average execution latency high: {avg_latency:.2f}s"
            else:
                return POPSSMCPHealthCheckStatus.HEALTHY, f"Average execution latency normal: {avg_latency:.2f}s"
                
        except Exception as e:
            return POPSSMCPHealthCheckStatus.UNKNOWN, f"Unable to check execution latency: {str(e)}"
    
    def _check_queue_health(self) -> Tuple[POPSSMCPHealthCheckStatus, str]:
        try:
            recent_queue_sizes = list(self._performance_history['queue_sizes'])[-10:]
            
            if not recent_queue_sizes:
                return POPSSMCPHealthCheckStatus.HEALTHY, "Queue is empty"
            
            avg_queue_size = sum(e['value'] for e in recent_queue_sizes) / len(recent_queue_sizes)
            
            if avg_queue_size >= self._thresholds['queue_size_critical']:
                return POPSSMCPHealthCheckStatus.UNHEALTHY, f"Queue size critically high: {avg_queue_size:.0f}"
            elif avg_queue_size >= self._thresholds['queue_size_warning']:
                return POPSSMCPHealthCheckStatus.DEGRADED, f"Queue size high: {avg_queue_size:.0f}"
            else:
                return POPSSMCPHealthCheckStatus.HEALTHY, f"Queue size normal: {avg_queue_size:.0f}"
                
        except Exception as e:
            return POPSSMCPHealthCheckStatus.UNKNOWN, f"Unable to check queue health: {str(e)}"
    
    def _collect_performance_metrics(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            self._performance_history['cpu_usage'].append({
                'value': cpu_percent,
                'timestamp': datetime.now()
            })
            
            self._performance_history['memory_usage'].append({
                'value': memory.percent,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self._LOG.error(f"Error collecting performance metrics: {e}")
    
    def _check_thresholds(self):
        try:
            recent_cpu = list(self._performance_history['cpu_usage'])
            if recent_cpu:
                avg_cpu = sum(e['value'] for e in recent_cpu) / len(recent_cpu)
                
                if avg_cpu >= self._thresholds['cpu_critical']:
                    self._create_alert(
                        severity='critical',
                        component='cpu_usage',
                        message=f"CPU usage critically high: {avg_cpu:.1f}%"
                    )
                elif avg_cpu >= self._thresholds['cpu_warning']:
                    self._create_alert(
                        severity='warning',
                        component='cpu_usage',
                        message=f"CPU usage high: {avg_cpu:.1f}%"
                    )
            
            recent_memory = list(self._performance_history['memory_usage'])
            if recent_memory:
                avg_memory = sum(e['value'] for e in recent_memory) / len(recent_memory)
                
                if avg_memory >= self._thresholds['memory_critical']:
                    self._create_alert(
                        severity='critical',
                        component='memory_usage',
                        message=f"Memory usage critically high: {avg_memory:.1f}%"
                    )
                elif avg_memory >= self._thresholds['memory_warning']:
                    self._create_alert(
                        severity='warning',
                        component='memory_usage',
                        message=f"Memory usage high: {avg_memory:.1f}%"
                    )
            
        except Exception as e:
            self._LOG.error(f"Error checking thresholds: {e}")
    
    def _create_alert(self, severity: str, component: str, message: str):
        existing_alerts = [a for a in self.alerts if not a.resolved and a.component == component and a.severity == severity]
        
        if existing_alerts:
            return
        
        alert = POPSSMCPMonitorAlert(
            alert_id=f"alert_{int(time.time() * 1000)}_{component}",
            severity=severity,
            component=component,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        self._LOG.warning(f"Alert created: [{severity}] {component}: {message}")
        
        handlers = self._notification_handlers.get(severity, [])
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                self._LOG.error(f"Error in alert notification handler: {e}")
    
    def _update_alerts(self):
        current_time = datetime.now()
        
        for alert in self.alerts:
            if not alert.resolved:
                resolved = False
                
                if alert.severity == 'critical':
                    check_func = self._health_checks.get(alert.component)
                    if check_func:
                        result = check_func()
                        if isinstance(result, tuple) and len(result) >= 2:
                            status = result[0]
                            if status in [POPSSMCPHealthCheckStatus.HEALTHY, POPSSMCPHealthCheckStatus.DEGRADED]:
                                resolved = True
                
                if resolved:
                    alert.resolved = True
                    alert.resolved_at = current_time
                    
                    self._LOG.info(f"Alert resolved: {alert.alert_id}")
    
    def register_notification_handler(self, severity: str, handler: Callable):
        if severity not in self._notification_handlers:
            self._notification_handlers[severity] = []
        
        self._notification_handlers[severity].append(handler)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: Optional[str] = None) -> bool:
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.acknowledged:
                    alert.acknowledged = True
                    alert.acknowledged_at = datetime.now()
                    alert.acknowledged_by = acknowledged_by
                    
                    self._LOG.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                    return True
            
            return False
    
    def dismiss_alert(self, alert_id: str) -> bool:
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    
                    self._LOG.info(f"Alert dismissed: {alert_id}")
                    return True
            
            return False
    
    def get_health_status(self) -> POPSSMCPHealthStatus:
        return self.health_status
    
    def get_component_health(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if component_name:
                health = self.component_health.get(component_name)
                if health:
                    return {
                        'component_name': health.component_name,
                        'status': health.status.value if hasattr(health.status, 'value') else str(health.status),
                        'message': health.message,
                        'last_check': health.last_check.isoformat(),
                        'metrics': health.metrics,
                        'history': health.history[-10:]
                    }
                return {}
            
            return {
                name: {
                    'component_name': health.component_name,
                    'status': health.status.value if hasattr(health.status, 'value') else str(health.status),
                    'message': health.message,
                    'last_check': health.last_check.isoformat(),
                    'metrics': health.metrics,
                    'history': health.history[-10:]
                }
                for name, health in self.component_health.items()
            }
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            alerts = [a for a in self.alerts if not a.resolved]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            return [
                {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity,
                    'component': alert.component,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'acknowledged': alert.acknowledged,
                    'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    'acknowledged_by': alert.acknowledged_by
                }
                for alert in alerts
            ]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            recent = list(self.alert_history)[-limit:]
            return [
                {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity,
                    'component': alert.component,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved,
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
                }
                for alert in recent
            ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        with self._lock:
            cpu_history = list(self._performance_history['cpu_usage'])
            memory_history = list(self._performance_history['memory_usage'])
            execution_history = list(self._performance_history['execution_times'])
            
            return {
                'cpu': {
                    'current': cpu_history[-1]['value'] if cpu_history else 0,
                    'average': sum(e['value'] for e in cpu_history) / len(cpu_history) if cpu_history else 0,
                    'max': max(e['value'] for e in cpu_history) if cpu_history else 0,
                    'min': min(e['value'] for e in cpu_history) if cpu_history else 0,
                    'samples': len(cpu_history)
                },
                'memory': {
                    'current': memory_history[-1]['value'] if memory_history else 0,
                    'average': sum(e['value'] for e in memory_history) / len(memory_history) if memory_history else 0,
                    'max': max(e['value'] for e in memory_history) if memory_history else 0,
                    'min': min(e['value'] for e in memory_history) if memory_history else 0,
                    'samples': len(memory_history)
                },
                'execution_time': {
                    'current': execution_history[-1]['value'] if execution_history else 0,
                    'average': sum(e['value'] for e in execution_history) / len(execution_history) if execution_history else 0,
                    'max': max(e['value'] for e in execution_history) if execution_history else 0,
                    'min': min(e['value'] for e in execution_history) if execution_history else 0,
                    'samples': len(execution_history)
                },
                'monitoring': {
                    'active': self._monitoring_active,
                    'last_check': self._last_health_check.isoformat() if self._last_health_check else None,
                    'check_interval': self._health_check_interval,
                    'components_monitored': len(self._health_checks)
                }
            }
    
    def record_execution_time(self, execution_time: float):
        self._performance_history['execution_times'].append({
            'value': execution_time,
            'timestamp': datetime.now()
        })
    
    def record_queue_size(self, queue_size: int):
        self._performance_history['queue_sizes'].append({
            'value': queue_size,
            'timestamp': datetime.now()
        })
    
    def set_threshold(self, threshold_name: str, value: float):
        if threshold_name in self._thresholds:
            self._thresholds[threshold_name] = value
            self._LOG.info(f"Threshold updated: {threshold_name} = {value}")
    
    def get_thresholds(self) -> Dict[str, float]:
        return self._thresholds.copy()
    
    def add_health_check(self, name: str, check_func: Callable):
        self._health_checks[name] = check_func
        self._LOG.info(f"Health check added: {name}")
    
    def remove_health_check(self, name: str) -> bool:
        if name in self._health_checks:
            del self._health_checks[name]
            self._LOG.info(f"Health check removed: {name}")
            return True
        return False
    
    def generate_report(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        if time_range is None:
            time_range = timedelta(hours=1)
        
        cutoff_time = datetime.now() - time_range
        
        with self._lock:
            relevant_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_time]
            
            severity_counts = {}
            component_counts = {}
            
            for alert in relevant_alerts:
                severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
                component_counts[alert.component] = component_counts.get(alert.component, 0) + 1
            
            return {
                'report_time': datetime.now().isoformat(),
                'time_range': str(time_range),
                'alert_summary': {
                    'total_alerts': len(relevant_alerts),
                    'by_severity': severity_counts,
                    'by_component': component_counts,
                    'critical_alerts': [a.alert_id for a in relevant_alerts if a.severity == 'critical']
                },
                'health_summary': self.get_component_health(),
                'performance_summary': self.get_performance_summary()
            }
    
    def shutdown(self):
        self.stop_monitoring()
        self._LOG.info("Monitor shutdown complete")
    
    def __enter__(self):
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
