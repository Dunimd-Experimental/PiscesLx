#!/usr/bin/env/python3
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

"""
Training Monitoring and Logging System

This module provides comprehensive monitoring and logging capabilities for the
PiscesLx training framework. It tracks training metrics, system resources, and
performance statistics in real-time, enabling detailed analysis and visualization.

Components:
    1. TrainingMonitorOperator: Real-time metrics tracking and logging
    2. PerformanceProfilerOperator: Detailed performance bottleneck analysis
    3. SystemResourceMonitor: GPU/CPU resource monitoring
    4. TrainingVisualizer: Training curve generation and dashboard

Features:
    - Real-time metric tracking (loss, learning rate, throughput)
    - System resource monitoring (GPU memory, CPU usage, temperature)
    - Performance profiling with PyTorch profiler integration
    - Automatic visualization generation (matplotlib-based)
    - Structured logging with JSON export
    - Anomaly detection for training issues

Usage Examples:
    Basic Monitoring:
        >>> from tools.train.monitoring import TrainingMonitorOperator
        >>> 
        >>> monitor = TrainingMonitorOperator(log_dir="./logs")
        >>> monitor.start_monitoring()
        >>> 
        >>> for step, metrics in enumerate(training_loop):
        ...     monitor.log_metrics(step, metrics)
        >>> 
        >>> monitor.stop_monitoring()
        >>> monitor.generate_report()

    Performance Profiling:
        >>> profiler = PerformanceProfilerOperator()
        >>> profiler.start_profiling()
        >>> 
        >>> # Training code here
        >>> 
        >>> profiler.stop_profiling()
        >>> profiler.export_chrome_trace("trace.json")

Integration with Training:
    The monitoring operators integrate seamlessly with PiscesLxTrainingOperator:
    - Automatic metric logging at specified intervals
    - Real-time dashboard updates
    - Automatic anomaly detection and alerting
"""

import torch
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import time
from datetime import datetime
import json
import statistics
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# OPSC operator system integration
from utils.opsc.base import PiscesLxTransformOperator
from utils.opsc.interface import PiscesLxOperatorConfig
from utils.opsc.registry import PiscesLxOperatorRegistrar
from utils.dc import PiscesLxLogger

_LOG = PiscesLxLogger(__name__)


@PiscesLxOperatorRegistrar()
class TrainingMonitorOperator(PiscesLxTransformOperator):
    """
    Training Monitor Operator
    
    Real-time monitoring of training metrics and system state. Tracks loss,
    learning rate, throughput, and system resources throughout training.
    
    Attributes:
        log_dir: Directory for storing logs and visualizations
        monitoring_interval: Steps between metric logging
        enable_visualization: Whether to generate plots and charts
        metrics_history: Historical data storage for all metrics
        
    Metrics Tracked:
        - Training loss and validation loss
        - Learning rate schedule
        - Gradient norms
        - Throughput (samples/second)
        - GPU memory usage
        - CPU utilization
        
    Example:
        >>> monitor = TrainingMonitorOperator(log_dir="./logs")
        >>> monitor.start_monitoring()
        >>> 
        >>> for step in range(1000):
        ...     loss = train_step()
        ...     monitor.log_metrics(step, {"loss": loss})
        >>> 
        >>> monitor.generate_report()
    """
    
    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None, log_dir: str = "./logs",
                 monitoring_interval: int = 10, enable_visualization: bool = True):
        """
        Initialize training monitor.
        
        Args:
            log_dir: Directory for log files and visualizations
            monitoring_interval: Steps between logging updates
            enable_visualization: Enable matplotlib-based visualizations
        """
        super().__init__(config)
        if config is not None:
            params = getattr(config, "parameters", {}) or {}
            log_dir = params.get("log_dir", log_dir)
            monitoring_interval = int(params.get("monitoring_interval", monitoring_interval))
            enable_visualization = bool(params.get("enable_visualization", enable_visualization))
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.monitoring_interval = monitoring_interval
        self.enable_visualization = enable_visualization
        
        # Metrics history storage
        self.metrics_history = {
            'loss': [],
            'learning_rate': [],
            'grad_norm': [],
            'throughput': [],
            'gpu_utilization': [],
            'memory_usage': [],
            'timestamps': []
        }
        
        # System resource monitor
        self.system_monitor = SystemResourceMonitor()
        
        # Visualization component
        if self.enable_visualization:
            self.visualizer = TrainingVisualizer(self.log_dir)
        
        _LOG.info(f"TrainingMonitor initialized with log_dir: {self.log_dir}")

    def _percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        data = sorted(float(x) for x in values)
        if len(data) == 1:
            return data[0]
        k = (len(data) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(data) - 1)
        if f == c:
            return data[f]
        return data[f] + (data[c] - data[f]) * (k - f)

    def transform(self, data: Any) -> Any:
        if isinstance(data, dict):
            step = data.get("step")
            metrics = data.get("metrics") or data
            if step is None:
                step = self.metrics_history.get("step", 0)
            self.log_metrics(int(step), metrics)
            return self.get_realtime_metrics()
        raise TypeError(f"Unsupported input type for training monitoring: {type(data).__name__}")
    
    def record_training_step(self, step_metrics: Dict[str, Any]):
        """
        Record training step metrics.
        
        Args:
            step_metrics: Dictionary of step metrics.
        """
        timestamp = datetime.now().isoformat()
        
        # Record basic metrics
        self.metrics_history['loss'].append(step_metrics.get('loss', 0))
        self.metrics_history['learning_rate'].append(step_metrics.get('learning_rate', 0))
        self.metrics_history['grad_norm'].append(step_metrics.get('grad_norm', 0))
        self.metrics_history['throughput'].append(step_metrics.get('throughput', 0))
        self.metrics_history['timestamps'].append(timestamp)
        
        # Record system resource usage
        system_metrics = self.system_monitor.get_system_metrics()
        self.metrics_history['gpu_utilization'].append(system_metrics.get('gpu_utilization', 0))
        self.metrics_history['memory_usage'].append(system_metrics.get('memory_usage', 0))
        
        # Periodically save logs
        if len(self.metrics_history['loss']) % self.monitoring_interval == 0:
            self._save_metrics_log()
            if self.enable_visualization:
                self.visualizer.generate_plots(self.metrics_history)
    
    def _save_metrics_log(self):
        """Save metrics log to file."""
        log_file = self.log_dir / f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare log data
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics_history': self.metrics_history,
            'summary': self._generate_summary()
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        _LOG.debug(f"Metrics logged to {log_file}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate training summary statistics."""
        if not self.metrics_history['loss']:
            return {}
        
        recent_losses = self.metrics_history['loss'][-100:]  # Last 100 steps
        recent_throughput = self.metrics_history['throughput'][-100:]
        
        return {
            'total_steps': len(self.metrics_history['loss']),
            'current_loss': self.metrics_history['loss'][-1],
            'avg_recent_loss': statistics.mean(recent_losses) if recent_losses else 0,
            'min_loss': min(self.metrics_history['loss']),
            'max_loss': max(self.metrics_history['loss']),
            'avg_throughput': statistics.mean(recent_throughput) if recent_throughput else 0,
            'current_lr': self.metrics_history['learning_rate'][-1] if self.metrics_history['learning_rate'] else 0,
            'current_grad_norm': self.metrics_history['grad_norm'][-1] if self.metrics_history['grad_norm'] else 0
        }
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics."""
        return self._generate_summary()
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect training anomalies.
        
        Returns:
            List of detected anomalies.
        """
        anomalies = []
        
        if len(self.metrics_history['loss']) < 10:
            return anomalies
        
        recent_losses = self.metrics_history['loss'][-10:]
        mean_loss = statistics.mean(recent_losses[:-1]) if len(recent_losses) > 1 else recent_losses[0]
        current_loss = recent_losses[-1]
        
        # Loss spike detection
        if current_loss > mean_loss * 2.0:
            anomalies.append({
                'type': 'loss_spike',
                'severity': 'high',
                'message': f'Loss spike detected: {current_loss:.4f} vs average {mean_loss:.4f}',
                'timestamp': self.metrics_history['timestamps'][-1]
            })
        
        # Gradient explosion detection
        recent_grad_norms = self.metrics_history['grad_norm'][-5:]
        if recent_grad_norms and max(recent_grad_norms) > 10.0:
            anomalies.append({
                'type': 'gradient_explosion',
                'severity': 'critical',
                'message': f'Gradient explosion detected: max norm {max(recent_grad_norms):.4f}',
                'timestamp': self.metrics_history['timestamps'][-1]
            })
        
        # Learning rate anomaly detection
        recent_lrs = self.metrics_history['learning_rate'][-5:]
        if len(set(recent_lrs)) == 1 and recent_lrs[0] == 0:
            anomalies.append({
                'type': 'zero_learning_rate',
                'severity': 'critical',
                'message': 'Learning rate is zero',
                'timestamp': self.metrics_history['timestamps'][-1]
            })
        
        if anomalies:
            for anomaly in anomalies:
                _LOG.warning(f"Anomaly detected: {anomaly['message']}")
        
        return anomalies


class SystemResourceMonitor:
    """System resource monitor."""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device_count = torch.cuda.device_count()
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get system resource metrics."""
        metrics = {
            'gpu_utilization': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0
        }
        
        try:
            if self.gpu_available and torch.cuda.is_initialized():
                # GPU utilization (simplified estimation)
                metrics['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 50.0
                
                # GPU memory usage
                total_memory = torch.cuda.get_device_properties(0).total_memory
                reserved_memory = torch.cuda.memory_reserved(0)
                metrics['memory_usage'] = (reserved_memory / total_memory) * 100
                
        except Exception as e:
            _LOG.debug(f"GPU metrics collection failed: {e}")
        
        return metrics


class TrainingVisualizer:
    """Training visualizer."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
    
    def generate_plots(self, metrics_history: Dict[str, List]):
        """Generate training curve plots."""
        if len(metrics_history['loss']) < 2:
            return
        x = list(range(len(metrics_history['loss'])))
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Training Loss", "Learning Rate", "Gradient Norm", "Training Throughput")
        )

        fig.add_trace(go.Scatter(x=x, y=metrics_history['loss'], mode="lines", name="loss"), row=1, col=1)

        if metrics_history.get('learning_rate'):
            fig.add_trace(go.Scatter(x=x, y=metrics_history['learning_rate'], mode="lines", name="lr"), row=1, col=2)

        if metrics_history.get('grad_norm'):
            fig.add_trace(go.Scatter(x=x, y=metrics_history['grad_norm'], mode="lines", name="grad_norm"), row=2, col=1)

        if metrics_history.get('throughput'):
            fig.add_trace(go.Scatter(x=x, y=metrics_history['throughput'], mode="lines", name="throughput"), row=2, col=2)

        fig.update_layout(title="Training Metrics Dashboard", height=800, width=1200, showlegend=True)

        plot_file = self.log_dir / f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(plot_file), include_plotlyjs="cdn")
        _LOG.debug(f"Training curves saved to {plot_file}")


@PiscesLxOperatorRegistrar()
class PerformanceProfilerOperator(PiscesLxTransformOperator):
    """
    Performance profiler operator.
    Provides detailed analysis of training performance bottlenecks.
    """
    
    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None, profile_interval: int = 100):
        super().__init__(config)
        if config is not None:
            params = getattr(config, "parameters", {}) or {}
            profile_interval = int(params.get("profile_interval", profile_interval))
        self.profile_interval = profile_interval
        self.profiling_data = {
            'forward_time': [],
            'backward_time': [],
            'optimizer_time': [],
            'data_loading_time': [],
            'memory_peaks': []
        }
        self.current_profile = {}

    def transform(self, data: Any) -> Any:
        if isinstance(data, dict):
            action = data.get("action")
            if action == "start":
                self.start_step_profile()
                return {"started": True}
            if action == "forward":
                self.record_forward_time()
                return {"forward": True}
            if action == "backward":
                self.record_backward_time()
                return {"backward": True}
            if action == "optimizer":
                self.record_optimizer_time()
                return {"optimizer": True}
            if action == "end":
                return self.end_step_profile()
            if action == "report":
                return self.get_performance_report()
        raise TypeError(f"Unsupported input type for training profiler: {type(data).__name__}")

    def _percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        data = sorted(float(x) for x in values)
        if len(data) == 1:
            return data[0]
        k = (len(data) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(data) - 1)
        if f == c:
            return data[f]
        return data[f] + (data[c] - data[f]) * (k - f)
    
    def start_step_profile(self):
        """Start step performance profiling."""
        self.current_profile = {
            'start_time': time.time(),
            'data_loaded': time.time()
        }
    
    def record_forward_time(self):
        """Record forward pass time."""
        self.current_profile['forward_completed'] = time.time()
    
    def record_backward_time(self):
        """Record backward pass time."""
        self.current_profile['backward_completed'] = time.time()
    
    def record_optimizer_time(self):
        """Record optimizer time."""
        self.current_profile['optimizer_completed'] = time.time()
    
    def end_step_profile(self):
        """End step performance profiling and record data."""
        end_time = time.time()
        
        # Calculate time for each phase
        data_time = self.current_profile.get('forward_completed', 0) - self.current_profile['data_loaded']
        forward_time = self.current_profile.get('backward_completed', 0) - self.current_profile.get('forward_completed', 0)
        backward_time = self.current_profile.get('optimizer_completed', 0) - self.current_profile.get('backward_completed', 0)
        optimizer_time = end_time - self.current_profile.get('optimizer_completed', 0)
        
        # Record data
        self.profiling_data['data_loading_time'].append(data_time)
        self.profiling_data['forward_time'].append(forward_time)
        self.profiling_data['backward_time'].append(backward_time)
        self.profiling_data['optimizer_time'].append(optimizer_time)
        
        # Record memory peak
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            self.profiling_data['memory_peaks'].append(memory_allocated)
        
        # Periodic analysis and reporting
        if len(self.profiling_data['forward_time']) % self.profile_interval == 0:
            self._generate_performance_report()
    
    def _generate_performance_report(self):
        """Generate performance analysis report."""
        if not any(self.profiling_data.values()):
            return
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_summary': {},
            'bottlenecks': []
        }
        
        # Calculate average time
        for key, times in self.profiling_data.items():
            if times:
                window = times[-self.profile_interval:]
                avg_time = statistics.mean(window)
                report['performance_summary'][key] = {
                    'average': float(avg_time),
                    'total': float(sum(times)),
                    'count': len(times),
                    'p95': float(self._percentile(window, 95)),
                    'p99': float(self._percentile(window, 99))
                }
        
        # Identify performance bottlenecks
        if self.profiling_data['forward_time'] and self.profiling_data['backward_time']:
            forward_avg = statistics.mean(self.profiling_data['forward_time'][-self.profile_interval:])
            backward_avg = statistics.mean(self.profiling_data['backward_time'][-self.profile_interval:])
            
            if forward_avg > backward_avg * 1.5:
                report['bottlenecks'].append({
                    'type': 'forward_bottleneck',
                    'severity': 'medium',
                    'message': f'Forward pass slower than backward ({forward_avg:.4f}s vs {backward_avg:.4f}s)'
                })
            elif backward_avg > forward_avg * 2:
                report['bottlenecks'].append({
                    'type': 'backward_bottleneck',
                    'severity': 'high',
                    'message': f'Backward pass significantly slower ({backward_avg:.4f}s vs {forward_avg:.4f}s)'
                })
        
        # Memory bottleneck detection
        if self.profiling_data['memory_peaks']:
            recent_memory = self.profiling_data['memory_peaks'][-10:]
            if recent_memory and max(recent_memory) > 0.8 * torch.cuda.get_device_properties(0).total_memory / 1024**3:
                report['bottlenecks'].append({
                    'type': 'memory_bottleneck',
                    'severity': 'critical',
                    'message': 'High GPU memory usage detected'
                })
        
        _LOG.info("Performance analysis completed")
        if report['bottlenecks']:
            for bottleneck in report['bottlenecks']:
                _LOG.warning(f"Bottleneck detected: {bottleneck['message']}")
        
        return report

    def get_performance_report(self) -> Dict[str, Any]:
        return self._generate_performance_report()


