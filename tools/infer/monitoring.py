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

"""
Inference Monitoring and Logging System
Real-time monitoring of inference process with detailed logging.
"""

import torch
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import time
from datetime import datetime
import json
import psutil
import statistics
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# OPSC operator system integration
from utils.opsc.base import PiscesLxTransformOperator
from utils.opsc.interface import PiscesLxOperatorConfig
from utils.opsc.registry import PiscesLxOperatorRegistrar
from utils.dc import PiscesLxLogger

from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Infer", file_path=get_log_file("PiscesLx.Tools.Infer"), enable_file=True)


@PiscesLxOperatorRegistrar()
class InferenceMonitorOperator(PiscesLxTransformOperator):
    """
    Inference Monitor Operator.
    Real-time monitoring of inference metrics and system state.
    """
    
    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None, log_dir: str = "./inference_logs",
                 monitoring_interval: int = 5, enable_visualization: bool = True):
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
        
        # Monitoring data storage
        self.metrics_history = {
            'latency': [],
            'throughput': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'cpu_usage': [],
            'tokens_per_second': [],
            'request_count': 0,
            'error_count': 0,
            'timestamps': []
        }
        
        # System resource monitor
        self.system_monitor = SystemResourceMonitor()
        
        # Visualization component
        if self.enable_visualization:
            self.visualizer = InferenceVisualizer(self.log_dir)
        
        _LOG.info(f"InferenceMonitor initialized with log_dir: {self.log_dir}")

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
            self.record_inference_request(data)
            return self.get_realtime_metrics()
        raise TypeError(f"Unsupported input type for monitoring: {type(data).__name__}")
    
    def record_inference_request(self, request_metrics: Dict[str, Any]):
        """
        Record inference request metrics.
        
        Args:
            request_metrics: Request metrics dictionary.
        """
        timestamp = datetime.now().isoformat()
        
        # Record basic metrics
        self.metrics_history['latency'].append(request_metrics.get('latency', 0))
        self.metrics_history['throughput'].append(request_metrics.get('throughput', 0))
        self.metrics_history['tokens_per_second'].append(request_metrics.get('tokens_per_second', 0))
        self.metrics_history['timestamps'].append(timestamp)
        self.metrics_history['request_count'] += 1
        
        # Record system resource usage
        system_metrics = self.system_monitor.get_system_metrics()
        self.metrics_history['memory_usage'].append(system_metrics.get('memory_usage', 0))
        self.metrics_history['gpu_utilization'].append(system_metrics.get('gpu_utilization', 0))
        self.metrics_history['cpu_usage'].append(system_metrics.get('cpu_usage', 0))
        
        # Record errors
        if request_metrics.get('error', False):
            self.metrics_history['error_count'] += 1
        
        # Periodically save logs and generate plots
        if self.metrics_history['request_count'] % self.monitoring_interval == 0:
            self._save_metrics_log()
            if self.enable_visualization:
                self.visualizer.generate_plots(self.metrics_history)
    
    def _save_metrics_log(self):
        """Save metrics log to file."""
        log_file = self.log_dir / f"inference_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
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
        """Generate inference summary statistics."""
        if not self.metrics_history['latency']:
            return {}
        
        recent_latencies = self.metrics_history['latency'][-100:]  # Last 100 requests
        recent_throughput = self.metrics_history['throughput'][-100:]
        recent_tokens = self.metrics_history['tokens_per_second'][-100:]
        
        return {
            'total_requests': self.metrics_history['request_count'],
            'error_rate': self.metrics_history['error_count'] / max(self.metrics_history['request_count'], 1),
            'avg_latency': statistics.mean(recent_latencies) if recent_latencies else 0,
            'p95_latency': self._percentile(recent_latencies, 95) if recent_latencies else 0,
            'p99_latency': self._percentile(recent_latencies, 99) if recent_latencies else 0,
            'avg_throughput': statistics.mean(recent_throughput) if recent_throughput else 0,
            'avg_tokens_per_second': statistics.mean(recent_tokens) if recent_tokens else 0,
            'current_memory_usage': self.metrics_history['memory_usage'][-1] if self.metrics_history['memory_usage'] else 0,
            'current_gpu_utilization': self.metrics_history['gpu_utilization'][-1] if self.metrics_history['gpu_utilization'] else 0
        }
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics."""
        return self._generate_summary()
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect inference anomalies.
        
        Returns:
            List of detected anomalies.
        """
        anomalies = []
        
        if len(self.metrics_history['latency']) < 10:
            return anomalies
        
        recent_latencies = self.metrics_history['latency'][-10:]
        mean_latency = statistics.mean(recent_latencies[:-1]) if len(recent_latencies) > 1 else recent_latencies[0]
        current_latency = recent_latencies[-1]
        
        # Latency spike detection
        if current_latency > mean_latency * 3.0:
            anomalies.append({
                'type': 'latency_spike',
                'severity': 'high',
                'message': f'Latency spike detected: {current_latency:.4f}s vs average {mean_latency:.4f}s',
                'timestamp': self.metrics_history['timestamps'][-1]
            })
        
        # Error rate anomaly detection
        if self.metrics_history['request_count'] >= 100:
            error_rate = self.metrics_history['error_count'] / self.metrics_history['request_count']
            if error_rate > 0.05:  # 5% error rate threshold
                anomalies.append({
                    'type': 'high_error_rate',
                    'severity': 'critical',
                    'message': f'High error rate detected: {error_rate:.2%}',
                    'timestamp': self.metrics_history['timestamps'][-1]
                })
        
        # Resource usage anomaly detection
        recent_memory = self.metrics_history['memory_usage'][-5:]
        if recent_memory and max(recent_memory) > 90.0:
            anomalies.append({
                'type': 'high_memory_usage',
                'severity': 'warning',
                'message': f'High memory usage: {max(recent_memory):.1f}%',
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
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'gpu_utilization': 0.0,
            'gpu_memory_usage': 0.0
        }
        
        try:
            if self.gpu_available and torch.cuda.is_initialized():
                # GPU utilization
                metrics['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 50.0
                
                # GPU memory usage
                total_memory = torch.cuda.get_device_properties(0).total_memory
                reserved_memory = torch.cuda.memory_reserved(0)
                metrics['gpu_memory_usage'] = (reserved_memory / total_memory) * 100
                
        except Exception as e:
            _LOG.debug(f"GPU metrics collection failed: {e}")
        
        return metrics


class InferenceVisualizer:
    """Inference visualizer."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
    
    def generate_plots(self, metrics_history: Dict[str, List]):
        """Generate inference performance plots."""
        if len(metrics_history['latency']) < 2:
            return
        x = list(range(len(metrics_history['latency'])))
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Request Latency", "Throughput", "System Resources", "Token Generation Rate")
        )

        fig.add_trace(go.Scatter(x=x, y=metrics_history['latency'], mode="lines", name="latency"), row=1, col=1)

        if metrics_history.get('throughput'):
            fig.add_trace(go.Scatter(x=x, y=metrics_history['throughput'], mode="lines", name="throughput"), row=1, col=2)

        if metrics_history.get('memory_usage'):
            fig.add_trace(go.Scatter(x=x, y=metrics_history['memory_usage'], mode="lines", name="memory"), row=2, col=1)
        if metrics_history.get('gpu_utilization'):
            fig.add_trace(go.Scatter(x=x, y=metrics_history['gpu_utilization'], mode="lines", name="gpu"), row=2, col=1)

        if metrics_history.get('tokens_per_second'):
            fig.add_trace(go.Scatter(x=x, y=metrics_history['tokens_per_second'], mode="lines", name="tok/s"), row=2, col=2)

        fig.update_layout(title="Inference Performance Dashboard", height=800, width=1200, showlegend=True)

        plot_file = self.log_dir / f"inference_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(plot_file), include_plotlyjs="cdn")
        _LOG.debug(f"Inference performance plots saved to {plot_file}")


@PiscesLxOperatorRegistrar()
class PerformanceProfilerOperator(PiscesLxTransformOperator):
    """
    Performance Profiler Operator.
    Provides detailed analysis of inference performance bottlenecks.
    """
    
    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None, profile_interval: int = 50):
        super().__init__(config)
        if config is not None:
            params = getattr(config, "parameters", {}) or {}
            profile_interval = int(params.get("profile_interval", profile_interval))
        self.profile_interval = profile_interval
        self.profiling_data = {
            'tokenization_time': [],
            'model_forward_time': [],
            'decoding_time': [],
            'total_time': [],
            'memory_peaks': []
        }
        self.current_profile = {}

    def transform(self, data: Any) -> Any:
        if isinstance(data, dict):
            action = data.get("action")
            if action == "start":
                self.start_request_profile()
                return {"started": True}
            if action == "tokenized":
                self.record_tokenization_time()
                return {"tokenized": True}
            if action == "forward_start":
                self.record_forward_start()
                return {"forward_start": True}
            if action == "forward_end":
                self.record_forward_end()
                return {"forward_end": True}
            if action == "decoded":
                self.record_decoding_time()
                return {"decoded": True}
            if action == "report":
                return self.get_performance_report()
        raise TypeError(f"Unsupported input type for profiler: {type(data).__name__}")

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
    
    def start_request_profile(self):
        """Start request performance profiling."""
        self.current_profile = {
            'start_time': time.time(),
            'tokenized_time': None,
            'forward_started': None,
            'forward_completed': None,
            'decoded_time': None
        }
    
    def record_tokenization_time(self):
        """Record tokenization time."""
        if self.current_profile:
            self.current_profile['tokenized_time'] = time.time()
    
    def record_forward_start(self):
        """Record forward pass start time."""
        if self.current_profile:
            self.current_profile['forward_started'] = time.time()
    
    def record_forward_end(self):
        """Record forward pass end time."""
        if self.current_profile:
            self.current_profile['forward_completed'] = time.time()
    
    def record_decoding_time(self):
        """Record decoding time."""
        if self.current_profile:
            self.current_profile['decoded_time'] = time.time()
    
    def end_request_profile(self, tokens_generated: int = 0):
        """End request performance profiling and record data."""
        if not self.current_profile:
            return
            
        end_time = time.time()
        
        # Calculate time for each phase
        if self.current_profile['tokenized_time']:
            tokenization_time = self.current_profile['tokenized_time'] - self.current_profile['start_time']
            self.profiling_data['tokenization_time'].append(tokenization_time)
        
        if self.current_profile['forward_started'] and self.current_profile['forward_completed']:
            forward_time = self.current_profile['forward_completed'] - self.current_profile['forward_started']
            self.profiling_data['model_forward_time'].append(forward_time)
        
        if self.current_profile['decoded_time']:
            decoding_time = end_time - self.current_profile['decoded_time']
            self.profiling_data['decoding_time'].append(decoding_time)
        
        total_time = end_time - self.current_profile['start_time']
        self.profiling_data['total_time'].append(total_time)
        
        # Record memory peak
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            self.profiling_data['memory_peaks'].append(memory_allocated)
        
        # Periodic analysis and reporting
        if len(self.profiling_data['total_time']) % self.profile_interval == 0:
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
            if times and key != 'memory_peaks':
                window = times[-self.profile_interval:]
                avg_time = statistics.mean(window)
                report['performance_summary'][key] = {
                    'average': float(avg_time),
                    'p95': float(self._percentile(window, 95)),
                    'p99': float(self._percentile(window, 99))
                }
        
        # Identify performance bottlenecks
        if self.profiling_data['model_forward_time'] and self.profiling_data['total_time']:
            forward_avg = statistics.mean(self.profiling_data['model_forward_time'][-self.profile_interval:])
            total_avg = statistics.mean(self.profiling_data['total_time'][-self.profile_interval:])
            
            if forward_avg / total_avg > 0.7:  # Forward pass accounts for more than 70% of total time
                report['bottlenecks'].append({
                    'type': 'model_bottleneck',
                    'severity': 'high',
                    'message': f'Model forward pass dominates inference time ({forward_avg/total_avg:.1%})'
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


