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
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque

from .types import (
    PiscesLxCoreMCPPerformanceMetrics,
    PiscesLxCoreMCPHealthStatus,
    PiscesLxCoreMCPExecutionContext,
    PiscesLxCoreMCPModuleStats,
    PiscesLxCoreMCPModuleStatus
)

from utils.log.core import PiscesLxCoreLog

class PiscesLxCoreMCPMonitor:
    """Performance monitoring and health checking system for PiscesLxCoreMCP.
    
    Provides real-time performance metrics collection, health status monitoring,
    error tracking, and performance optimization recommendations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the MCP monitor with configuration settings.
        
        Args:
            config: Optional configuration dictionary for monitor settings
        """
        self.config = config or {}
        self.logger = self._configure_logging()
        
        # Performance tracking
        self.metrics = PiscesLxCoreMCPPerformanceMetrics()
        self.health_status = PiscesLxCoreMCPHealthStatus()
        self.execution_history = deque(maxlen=1000)  # Keep last 1000 executions
        
        # Component tracking
        self.component_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.error_log: List[Dict[str, Any]] = []
        self.performance_alerts: List[str] = []
        
        # Timing and uptime tracking
        self.start_time = time.time()
        self.last_health_check = datetime.now()
        
        # Thread safety
        self._lock = threading.RLock()
        self._monitoring_active = True
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _configure_logging(self) -> PiscesLxCoreLog:
        """Configure structured logging for the monitor.
        
        Returns:
            Configured logger instance
        """
        logger = PiscesLxCoreLog("PiscesLx.Utils.MCP.Monitor")
        return logger
    
    def _start_background_monitoring(self):
        """Start background monitoring thread for periodic health checks."""
        def monitor_loop():
            while self._monitoring_active:
                try:
                    self._perform_health_check()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"Health check error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        self.logger.info("Background monitoring started")
    
    def record_execution_start(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Record the start of a tool execution.
        
        Args:
            tool_name: Name of the tool being executed
            arguments: Input arguments for the tool
            
        Returns:
            Execution context ID for tracking
        """
        execution_id = f"exec_{int(time.time() * 1000000)}"
        
        with self._lock:
            context = PiscesLxCoreMCPExecutionContext(
                session_id=execution_id,
                tool_name=tool_name,
                arguments=arguments,
                start_time=datetime.now()
            )
            
            self.execution_history.append(context)
            self.metrics.total_executions += 1
            
            # Update component metrics
            if tool_name not in self.component_metrics:
                self.component_metrics[tool_name] = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'failed_executions': 0,
                    'average_execution_time': 0.0,
                    'last_used': datetime.now()
                }
            
            self.component_metrics[tool_name]['total_executions'] += 1
            self.component_metrics[tool_name]['last_used'] = datetime.now()
        
        self.logger.debug(f"Execution started: {tool_name} (ID: {execution_id})")
        return execution_id
    
    def record_execution_complete(self, execution_id: str, result: Any = None, error: Optional[str] = None):
        """Record the completion of a tool execution.
        
        Args:
            execution_id: Execution context ID
            result: Execution result (if successful)
            error: Error message (if failed)
        """
        with self._lock:
            # Find the execution context
            context = None
            for ctx in self.execution_history:
                if ctx.session_id == execution_id:
                    context = ctx
                    break
            
            if not context:
                self.logger.warning(f"Execution context not found: {execution_id}")
                return
            
            # Update context
            context.end_time = datetime.now()
            context.result = result
            context.error = error
            
            execution_time = (context.end_time - context.start_time).total_seconds()
            tool_name = context.tool_name
            
            # Update metrics
            if error:
                self.metrics.failed_executions += 1
                self.component_metrics[tool_name]['failed_executions'] += 1
                self._record_error(tool_name, error, execution_id)
            else:
                self.metrics.successful_executions += 1
                self.component_metrics[tool_name]['successful_executions'] += 1
            
            # Update execution time metrics
            self._update_execution_time_metrics(tool_name, execution_time)
            
            # Log completion
            status = "failed" if error else "success"
            self.logger.debug(f"Execution {status}: {tool_name} (ID: {execution_id}, Time: {execution_time:.4f}s)")
    
    def _update_execution_time_metrics(self, tool_name: str, execution_time: float):
        """Update execution time metrics for a tool.
        
        Args:
            tool_name: Name of the tool
            execution_time: Execution time in seconds
        """
        component = self.component_metrics[tool_name]
        
        # Calculate running average
        current_avg = component['average_execution_time']
        total_execs = component['total_executions']
        
        if total_execs == 1:
            component['average_execution_time'] = execution_time
        else:
            # Weighted average: 70% old average, 30% new time
            component['average_execution_time'] = (current_avg * 0.7) + (execution_time * 0.3)
        
        # Update global average
        total_time = sum(c['average_execution_time'] * c['total_executions'] 
                        for c in self.component_metrics.values())
        total_execs = sum(c['total_executions'] for c in self.component_metrics.values())
        
        if total_execs > 0:
            self.metrics.average_execution_time = total_time / total_execs
    
    def _record_error(self, tool_name: str, error: str, execution_id: str):
        """Record an error occurrence.
        
        Args:
            tool_name: Name of the tool that failed
            error: Error message
            execution_id: Associated execution ID
        """
        error_record = {
            'timestamp': datetime.now(),
            'tool_name': tool_name,
            'error': error,
            'execution_id': execution_id
        }
        
        self.error_log.append(error_record)
        
        # Keep only recent errors (last 100)
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]
        
        # Update error rates
        self._update_error_rates()
        
        # Check for error rate alerts
        self._check_error_rate_alerts(tool_name)
    
    def _update_error_rates(self):
        """Update error rates for tools and overall system."""
        total_errors = len(self.error_log)
        total_executions = self.metrics.total_executions
        
        if total_executions > 0:
            self.metrics.error_rate = total_errors / total_executions
        
        # Update individual tool error rates
        for tool_name, component in self.component_metrics.items():
            total = component['total_executions']
            failed = component['failed_executions']
            
            if total > 0:
                # Calculate error rate from recent history
                recent_errors = sum(1 for e in self.error_log 
                                  if e['tool_name'] == tool_name)
                component['error_rate'] = recent_errors / max(total, 1)
    
    def _check_error_rate_alerts(self, tool_name: str):
        """Check if error rates warrant alerts.
        
        Args:
            tool_name: Name of the tool to check
        """
        if tool_name not in self.component_metrics:
            return
        
        component = self.component_metrics[tool_name]
        error_rate = component.get('error_rate', 0.0)
        total_executions = component['total_executions']
        
        # Only alert if we have sufficient data (at least 10 executions)
        if total_executions < 10:
            return
        
        if error_rate > 0.5:  # 50% error rate
            alert = f"High error rate for {tool_name}: {error_rate:.1%}"
            if alert not in self.performance_alerts:
                self.performance_alerts.append(alert)
                self.logger.warning(alert)
        elif error_rate > 0.2:  # 20% error rate
            alert = f"Elevated error rate for {tool_name}: {error_rate:.1%}"
            self.logger.info(alert)
    
    def _perform_health_check(self):
        """Perform comprehensive health check of the MCP system."""
        try:
            with self._lock:
                self._check_system_health()
                self._generate_performance_recommendations()
                self.last_health_check = datetime.now()
                self.metrics.last_health_check = self.last_health_check
                self.metrics.uptime = time.time() - self.start_time
            
            self.logger.debug("Health check completed")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.health_status.status = "unhealthy"
            self.health_status.alerts.append(f"Health check error: {str(e)}")
    
    def _check_system_health(self):
        """Check overall system health status."""
        alerts = []
        component_statuses = {}
        
        # Check error rate
        if self.metrics.error_rate > 0.1:  # 10% error rate
            alerts.append(f"High system error rate: {self.metrics.error_rate:.1%}")
            status = "degraded"
        elif self.metrics.error_rate > 0.05:  # 5% error rate
            status = "degraded"
        else:
            status = "healthy"
        
        # Check individual components
        for tool_name, component in self.component_metrics.items():
            tool_status = "healthy"
            tool_alerts = []
            
            error_rate = component.get('error_rate', 0.0)
            if error_rate > 0.5:
                tool_status = "unhealthy"
                tool_alerts.append(f"High error rate: {error_rate:.1%}")
            elif error_rate > 0.2:
                tool_status = "degraded"
                tool_alerts.append(f"Elevated error rate: {error_rate:.1%}")
            
            # Check if tool has been used recently (last 24 hours)
            last_used = component.get('last_used')
            if last_used and (datetime.now() - last_used).total_seconds() > 86400:
                tool_alerts.append("Tool not used in 24 hours")
            
            component_statuses[tool_name] = tool_status
            alerts.extend([f"{tool_name}: {alert}" for alert in tool_alerts])
        
        # Update health status
        self.health_status.status = status
        self.health_status.alerts = alerts[:10]  # Keep only first 10 alerts
        self.health_status.component_statuses = component_statuses
        self.health_status.last_updated = datetime.now()
    
    def _generate_performance_recommendations(self):
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze execution patterns
        if self.metrics.total_executions > 100:
            avg_time = self.metrics.average_execution_time
            if avg_time > 5.0:  # Average execution time > 5 seconds
                recommendations.append("Consider optimizing slow-running tools")
            
            if self.metrics.error_rate > 0.05:  # Error rate > 5%
                recommendations.append("Review and improve error handling")
        
        # Analyze tool-specific performance
        for tool_name, component in self.component_metrics.items():
            error_rate = component.get('error_rate', 0.0)
            avg_time = component.get('average_execution_time', 0.0)
            
            if error_rate > 0.3:  # Tool error rate > 30%
                recommendations.append(f"Review {tool_name} implementation - high error rate")
            
            if avg_time > 10.0:  # Tool average time > 10 seconds
                recommendations.append(f"Optimize {tool_name} performance")
        
        # Memory usage recommendations
        if len(self.execution_history) >= 1000:
            recommendations.append("Consider increasing execution history retention")
        
        self.health_status.recommendations = recommendations[:5]  # Keep top 5
    
    def get_performance_metrics(self) -> PiscesLxCoreMCPPerformanceMetrics:
        """Get current performance metrics.
        
        Returns:
            Current performance metrics
        """
        with self._lock:
            # Update uptime before returning
            self.metrics.uptime = time.time() - self.start_time
            return self.metrics
    
    def get_health_status(self) -> PiscesLxCoreMCPHealthStatus:
        """Get current health status.
        
        Returns:
            Current health status
        """
        with self._lock:
            return self.health_status
    
    def get_component_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get component performance metrics.
        
        Args:
            tool_name: Optional specific tool name, or all components if None
            
        Returns:
            Component metrics dictionary
        """
        with self._lock:
            if tool_name:
                return self.component_metrics.get(tool_name, {})
            return dict(self.component_metrics)
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent error records.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of error records
        """
        with self._lock:
            return self.error_log[-limit:] if self.error_log else []
    
    def reset_metrics(self):
        """Reset performance metrics."""
        with self._lock:
            self.metrics = PiscesLxCoreMCPPerformanceMetrics()
            self.component_metrics.clear()
            self.error_log.clear()
            self.performance_alerts.clear()
            self.start_time = time.time()
            
            self.logger.info("Performance metrics reset")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        self.logger.info("Monitoring stopped")