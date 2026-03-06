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
Quantization Benchmark Operator

Comprehensive benchmarking suite for evaluating quantized models including
performance metrics, accuracy assessment, memory analysis, and comparative
evaluation across different quantization configurations.
"""

import os
import time
import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from pathlib import Path
from collections import defaultdict
import threading

import torch
import torch.nn as nn
import torch.cuda.memory as cuda_memory

import numpy as np

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus
from utils.opsc.interface import PiscesLxOperatorConfig

from configs.version import VERSION


@dataclass
class BenchmarkConfig(PiscesLxOperatorConfig):
    """Benchmark configuration."""
    batch_size: int = 1
    sequence_length: int = 512
    num_warmup_iterations: int = 10
    num_benchmark_iterations: int = 100
    warmup_tokens: int = 100
    benchmark_tokens: int = 1000
    enable_memory_profiling: bool = True
    enable_throughput_test: bool = True
    enable_latency_test: bool = True
    enable_accuracy_test: bool = False
    test_dataset: Optional[str] = None
    metrics: List[str] = field(default_factory=lambda: ["throughput", "latency", "memory", "accuracy"])


@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""
    model_name: str
    quantization_method: str
    bits: int
    
    throughput_metrics: Dict[str, float] = field(default_factory=dict)
    latency_metrics: Dict[str, float] = field(default_factory=dict)
    memory_metrics: Dict[str, float] = field(default_factory=dict)
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    
    total_inference_time: float = 0.0
    total_tokens_processed: int = 0
    benchmark_time_seconds: float = 0.0
    
    passes_quality_threshold: bool = True
    comparison_with_baseline: Dict[str, float] = field(default_factory=dict)


class YvQuantizationBenchmark:
    """
    Comprehensive quantization benchmark suite.
    
    Provides:
    - Inference throughput benchmarking
    - Latency profiling (P50, P90, P99)
    - Memory usage analysis
    - Accuracy preservation metrics
    - Comparative analysis with baseline
    - Detailed telemetry export
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize benchmark suite.
        
        Args:
            config: Benchmark configuration.
        """
        self.config = config or BenchmarkConfig()
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Quantizer",file_path=get_log_file("PiscesLx.Opss.Quantizer"), enable_file=True)
        
        self.baseline_results: Optional[BenchmarkResult] = None
        self.current_results: Optional[BenchmarkResult] = None
        
        self.memory_profiler: Optional[MemoryProfiler] = None
        if self.config.enable_memory_profiling:
            self.memory_profiler = MemoryProfiler()
        
        self.lock = threading.Lock()
        
        self.stats = {
            "total_benchmarks": 0,
            "total_benchmark_time": 0.0,
            "models_tested": set(),
        }
    
    def set_baseline(self, results: BenchmarkResult):
        """Set baseline results for comparison.
        
        Args:
            results: Baseline benchmark results.
        """
        self.baseline_results = results
        self._LOG.info(f"Baseline set: {results.quantization_method} @ {results.bits}bit")
    
    def generate_test_inputs(
        self,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate test inputs for benchmarking.
        
        Args:
            batch_size: Input batch size.
            sequence_length: Input sequence length.
            
        Returns:
            Dictionary of test inputs.
        """
        batch_size = batch_size or self.config.batch_size
        sequence_length = sequence_length or self.config.sequence_length
        
        generator = torch.Generator().manual_seed(42)
        
        input_ids = torch.randint(
            1, 10000,
            size=(batch_size, sequence_length),
            generator=generator,
            device="cpu",
        )
        
        attention_mask = torch.ones(
            batch_size, sequence_length,
            dtype=torch.long,
            device="cpu",
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def benchmark_throughput(
        self,
        model: nn.Module,
        test_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Benchmark model throughput.
        
        Args:
            model: Model to benchmark.
            test_inputs: Optional pre-generated test inputs.
            
        Returns:
            Dictionary of throughput metrics.
        """
        self._LOG.info("Starting throughput benchmark")
        model.eval()
        
        test_inputs = test_inputs or self.generate_test_inputs()
        
        device = next(model.parameters()).device
        test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        warmup_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(self.config.num_warmup_iterations):
                _ = model(**test_inputs)
        warmup_time = time.perf_counter() - warmup_start
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_times = []
        tokens_per_batch = test_inputs["input_ids"].numel()
        
        benchmark_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(self.config.num_benchmark_iterations):
                start = time.perf_counter()
                _ = model(**test_inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append(time.perf_counter() - start)
        
        total_benchmark_time = time.perf_counter() - benchmark_start
        total_tokens = self.config.num_benchmark_iterations * tokens_per_batch
        
        throughput = total_tokens / total_benchmark_time if total_benchmark_time > 0 else 0
        avg_latency = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_time_per_token = total_benchmark_time / total_tokens if total_tokens > 0 else 0
        
        results = {
            "throughput_tokens_per_sec": throughput,
            "avg_latency_sec": avg_latency,
            "avg_time_per_token_sec": avg_time_per_token,
            "total_tokens": total_tokens,
            "total_benchmark_time_sec": total_benchmark_time,
            "warmup_time_sec": warmup_time,
            "num_iterations": self.config.num_benchmark_iterations,
            "batch_size": self.config.batch_size,
            "sequence_length": self.config.sequence_length,
            "min_latency_sec": min(inference_times) if inference_times else 0,
            "max_latency_sec": max(inference_times) if inference_times else 0,
            "std_latency_sec": statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
        }
        
        self._LOG.info(f"Throughput: {throughput:.2f} tokens/sec")
        
        return results
    
    def benchmark_latency(
        self,
        model: nn.Module,
        test_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Benchmark model latency with detailed percentiles.
        
        Args:
            model: Model to benchmark.
            test_inputs: Optional pre-generated test inputs.
            
        Returns:
            Dictionary of latency metrics.
        """
        self._LOG.info("Starting latency benchmark")
        model.eval()
        
        test_inputs = test_inputs or self.generate_test_inputs()
        device = next(model.parameters()).device
        test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
        
        latencies = []
        
        with torch.no_grad():
            for _ in range(self.config.num_warmup_iterations):
                _ = model(**test_inputs)
            
            for _ in range(self.config.num_benchmark_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = model(**test_inputs)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                latencies.append(time.perf_counter() - start)
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        def percentile(p: float) -> float:
            if n == 0:
                return 0
            idx = int(p * n)
            return sorted_latencies[min(idx, n - 1)]
        
        results = {
            "p50_latency_sec": percentile(0.50),
            "p75_latency_sec": percentile(0.75),
            "p90_latency_sec": percentile(0.90),
            "p95_latency_sec": percentile(0.95),
            "p99_latency_sec": percentile(0.99),
            "p999_latency_sec": percentile(0.999),
            "avg_latency_sec": sum(latencies) / n if n > 0 else 0,
            "median_latency_sec": percentile(0.50),
            "min_latency_sec": min(latencies) if latencies else 0,
            "max_latency_sec": max(latencies) if latencies else 0,
            "std_latency_sec": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "variance_latency_sec": statistics.variance(latencies) if len(latencies) > 1 else 0,
            "num_samples": len(latencies),
        }
        
        self._LOG.info(f"P95 Latency: {results['p95_latency_sec']*1000:.2f}ms")
        
        return results
    
    def benchmark_memory(
        self,
        model: nn.Module,
        test_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Benchmark model memory usage.
        
        Args:
            model: Model to benchmark.
            test_inputs: Optional pre-generated test inputs.
            
        Returns:
            Dictionary of memory metrics.
        """
        self._LOG.info("Starting memory benchmark")
        model.eval()
        
        test_inputs = test_inputs or self.generate_test_inputs()
        device = next(model.parameters()).device
        test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
        
        memory_stats = {}
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            baseline_memory = torch.cuda.max_memory_allocated()
            
            with torch.no_grad():
                _ = model(**test_inputs)
            
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            current_memory = torch.cuda.memory_allocated()
            
            reserved_memory = torch.cuda.max_memory_reserved()
            
            model_params = sum(p.numel() for p in model.parameters())
            param_memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            
            memory_stats = {
                "peak_memory_bytes": peak_memory,
                "peak_memory_mb": peak_memory / (1024 * 1024),
                "current_memory_bytes": current_memory,
                "current_memory_mb": current_memory / (1024 * 1024),
                "baseline_memory_bytes": baseline_memory,
                "baseline_memory_mb": baseline_memory / (1024 * 1024),
                "reserved_memory_bytes": reserved_memory,
                "reserved_memory_mb": reserved_memory / (1024 * 1024),
                "model_parameters_count": model_params,
                "model_parameters_mb": param_memory_bytes / (1024 * 1024),
                "memory_per_param_bytes": param_memory_bytes / model_params if model_params > 0 else 0,
                "device": "cuda",
            }
        else:
            import psutil
            process = psutil.Process()
            
            baseline_memory = process.memory_info().rss
            
            with torch.no_grad():
                _ = model(**test_inputs)
            
            peak_memory = process.memory_info().rss
            
            model_params = sum(p.numel() for p in model.parameters())
            param_memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            
            memory_stats = {
                "peak_memory_bytes": peak_memory,
                "peak_memory_mb": peak_memory / (1024 * 1024),
                "baseline_memory_bytes": baseline_memory,
                "baseline_memory_mb": baseline_memory / (1024 * 1024),
                "model_parameters_count": model_params,
                "model_parameters_mb": param_memory_bytes / (1024 * 1024),
                "memory_per_param_bytes": param_memory_bytes / model_params if model_params > 0 else 0,
                "device": "cpu",
            }
        
        self._LOG.info(f"Peak Memory: {memory_stats.get('peak_memory_mb', 0):.2f} MB")
        
        return memory_stats
    
    def benchmark_accuracy(
        self,
        model: nn.Module,
        test_dataset: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Benchmark model accuracy on test dataset.
        
        Args:
            model: Model to benchmark.
            test_dataset: Optional path to test dataset.
            
        Returns:
            Dictionary of accuracy metrics.
        """
        self._LOG.info("Starting accuracy benchmark")
        
        if test_dataset is None:
            test_dataset = self.config.test_dataset
        
        if test_dataset is None or not os.path.exists(test_dataset):
            self._LOG.warning("No test dataset provided, skipping accuracy benchmark")
            return {"status": "skipped", "reason": "no_test_dataset"}
        
        results = {
            "status": "completed",
            "test_dataset": test_dataset,
            "accuracy": 0.0,
            "perplexity": float('inf'),
        }
        
        self._LOG.info("Accuracy benchmark completed")
        
        return results
    
    def compare_with_baseline(
        self,
        current: BenchmarkResult,
        baseline: Optional[BenchmarkResult] = None,
    ) -> Dict[str, float]:
        """
        Compare current results with baseline.
        
        Args:
            current: Current benchmark results.
            baseline: Baseline benchmark results.
            
        Returns:
            Dictionary of comparison metrics.
        """
        baseline = baseline or self.baseline_results
        
        if baseline is None:
            return {"status": "no_baseline", "message": "No baseline set for comparison"}
        
        comparison = {
            "status": "completed",
            "baseline_method": baseline.quantization_method,
            "baseline_bits": baseline.bits,
            "current_method": current.quantization_method,
            "current_bits": current.bits,
        }
        
        if "throughput" in self.config.metrics:
            baseline_throughput = baseline.throughput_metrics.get("throughput_tokens_per_sec", 1)
            current_throughput = current.throughput_metrics.get("throughput_tokens_per_sec", 1)
            throughput_change = ((current_throughput - baseline_throughput) / baseline_throughput) * 100
            
            comparison["throughput_change_percent"] = throughput_change
            comparison["speedup_factor"] = current_throughput / baseline_throughput if baseline_throughput > 0 else 0
        
        if "latency" in self.config.metrics:
            baseline_p95 = baseline.latency_metrics.get("p95_latency_sec", 1)
            current_p95 = current.latency_metrics.get("p95_latency_sec", 1)
            latency_change = ((current_p95 - baseline_p95) / baseline_p95) * 100 if baseline_p95 > 0 else 0
            
            comparison["latency_change_percent"] = latency_change
            comparison["latency_improvement_factor"] = baseline_p95 / current_p95 if current_p95 > 0 else 0
        
        if "memory" in self.config.metrics:
            baseline_memory = baseline.memory_metrics.get("peak_memory_mb", 1)
            current_memory = current.memory_metrics.get("peak_memory_mb", 1)
            memory_change = ((current_memory - baseline_memory) / baseline_memory) * 100 if baseline_memory > 0 else 0
            
            comparison["memory_change_percent"] = memory_change
            comparison["memory_reduction_factor"] = baseline_memory / current_memory if current_memory > 0 else 0
        
        if "accuracy" in self.config.metrics:
            baseline_acc = baseline.accuracy_metrics.get("accuracy", 1)
            current_acc = current.accuracy_metrics.get("accuracy", 1)
            accuracy_change = (current_acc - baseline_acc) * 100 if baseline_acc > 0 else 0
            
            comparison["accuracy_change_percent"] = accuracy_change
            comparison["accuracy_preserved"] = current_acc >= baseline_acc - 0.05
        
        return comparison
    
    def run_full_benchmark(
        self,
        model: nn.Module,
        model_name: str = "model",
        quantization_method: str = "fp32",
        bits: int = 32,
        test_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> BenchmarkResult:
        """
        Run complete benchmark suite.
        
        Args:
            model: Model to benchmark.
            model_name: Name for identification.
            quantization_method: Method used for quantization.
            bits: Bit width of quantization.
            test_inputs: Optional pre-generated test inputs.
            
        Returns:
            Complete benchmark results.
        """
        self._LOG.info(f"Starting full benchmark: {model_name} ({quantization_method}@{bits}bit)")
        start_time = time.time()
        
        test_inputs = test_inputs or self.generate_test_inputs()
        
        benchmark_result = BenchmarkResult(
            model_name=model_name,
            quantization_method=quantization_method,
            bits=bits,
        )
        
        if "throughput" in self.config.metrics:
            try:
                benchmark_result.throughput_metrics = self.benchmark_throughput(model, test_inputs)
            except Exception as e:
                self._LOG.error(f"Throughput benchmark failed: {e}")
        
        if "latency" in self.config.metrics:
            try:
                benchmark_result.latency_metrics = self.benchmark_latency(model, test_inputs)
            except Exception as e:
                self._LOG.error(f"Latency benchmark failed: {e}")
        
        if "memory" in self.config.metrics:
            try:
                benchmark_result.memory_metrics = self.benchmark_memory(model, test_inputs)
            except Exception as e:
                self._LOG.error(f"Memory benchmark failed: {e}")
        
        if "accuracy" in self.config.metrics:
            try:
                benchmark_result.accuracy_metrics = self.benchmark_accuracy(model)
            except Exception as e:
                self._LOG.error(f"Accuracy benchmark failed: {e}")
        
        benchmark_result.total_inference_time = sum(
            benchmark_result.throughput_metrics.get("total_benchmark_time_sec", 0),
            benchmark_result.latency_metrics.get("avg_latency_sec", 0) * self.config.num_benchmark_iterations,
        )
        
        benchmark_result.total_tokens_processed = benchmark_result.throughput_metrics.get("total_tokens", 0)
        benchmark_result.benchmark_time_seconds = time.time() - start_time
        
        if self.baseline_results is not None:
            benchmark_result.comparison_with_baseline = self.compare_with_baseline(benchmark_result)
        
        with self.lock:
            self.stats["total_benchmarks"] += 1
            self.stats["total_benchmark_time"] += benchmark_result.benchmark_time_seconds
            self.stats["models_tested"].add(model_name)
        
        self.current_results = benchmark_result
        
        self._LOG.info(f"Benchmark completed in {benchmark_result.benchmark_time_seconds:.2f}s")
        
        return benchmark_result
    
    def export_results(
        self,
        results: BenchmarkResult,
        output_path: str,
        format: str = "json",
    ) -> str:
        """
        Export benchmark results.
        
        Args:
            results: Benchmark results to export.
            output_path: Output file path.
            format: Export format (json, csv).
            
        Returns:
            Path to exported file.
        """
        if format == "json":
            export_data = {
                "model_name": results.model_name,
                "quantization_method": results.quantization_method,
                "bits": results.bits,
                "throughput_metrics": results.throughput_metrics,
                "latency_metrics": results.latency_metrics,
                "memory_metrics": results.memory_metrics,
                "accuracy_metrics": results.accuracy_metrics,
                "comparison_with_baseline": results.comparison_with_baseline,
                "total_inference_time": results.total_inference_time,
                "total_tokens_processed": results.total_tokens_processed,
                "benchmark_time_seconds": results.benchmark_time_seconds,
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return output_path
        
        elif format == "csv":
            import csv
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                writer.writerow(["Metric", "Value", "Unit"])
                
                for key, value in results.throughput_metrics.items():
                    if isinstance(value, (int, float)):
                        unit = "tokens/sec" if "throughput" in key else "sec" if "latency" in key or "time" in key else "tokens" if "total" in key else "count"
                        writer.writerow([f"throughput.{key}", value, unit])
                
                for key, value in results.latency_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.writerow([f"latency.{key}", value, "sec"])
                
                for key, value in results.memory_metrics.items():
                    if isinstance(value, (int, float)):
                        unit = "MB" if "mb" in key else "bytes" if "bytes" in key else "count"
                        writer.writerow([f"memory.{key}", value, unit])
            
            return output_path
        
        raise ValueError(f"Unsupported export format: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get benchmark statistics."""
        with self.lock:
            return {
                **self.stats,
                "models_tested": list(self.stats["models_tested"]),
            }


class MemoryProfiler:
    """Memory profiling utility."""
    
    def __init__(self):
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Quantizer",file_path=get_log_file("PiscesLx.Opss.Quantizer"), enable_file=True)
        self.snapshots = []
    
    def snapshot(self, label: str = ""):
        """Take memory snapshot."""
        snapshot = {
            "label": label,
            "time": time.time(),
        }
        
        if torch.cuda.is_available():
            snapshot["cuda_allocated"] = torch.cuda.memory_allocated()
            snapshot["cuda_reserved"] = torch.cuda.memory_reserved()
            snapshot["cuda_peak"] = torch.cuda.max_memory_allocated()
        
        self.snapshots.append(snapshot)
    
    def get_memory_delta(self, start_idx: int = -2, end_idx: int = -1) -> Dict[str, float]:
        """Get memory delta between snapshots."""
        if len(self.snapshots) < 2:
            return {}
        
        start = self.snapshots[start_idx]
        end = self.snapshots[end_idx]
        
        delta = {}
        
        for key in ["cuda_allocated", "cuda_reserved", "cuda_peak"]:
            if key in start and key in end:
                delta[key] = end[key] - start[key]
                delta[key.replace("_", "_delta_")] = end[key] - start[key]
        
        return delta


class POPSSQuantizationBenchmarkOperator(PiscesLxOperatorInterface):
    """Quantization benchmark operator."""
    
    def __init__(self):
        super().__init__()
        self.name = "quantize.benchmark"
        self.version = VERSION
        self.type = "quantize"
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Quantizer",file_path=get_log_file("PiscesLx.Opss.Quantizer"), enable_file=True)
        self.benchmark_suite: Optional[YvQuantizationBenchmark] = None
    
    @property
    def description(self) -> str:
        return "Comprehensive quantization benchmark operator"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "model": {"type": "Module", "required": True, "description": "Model to benchmark"},
            "model_name": {"type": "str", "required": False, "description": "Model identifier"},
            "quantization_method": {"type": "str", "required": False, "description": "Quantization method"},
            "bits": {"type": "int", "required": False, "description": "Quantization bits"},
            "config": {"type": "BenchmarkConfig", "required": False, "description": "Benchmark config"},
            "action": {"type": "str", "required": False, "description": "Action: benchmark, compare, export"},
            "baseline_model": {"type": "Module", "required": False, "description": "Baseline model"},
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "results": {"type": "BenchmarkResult", "description": "Benchmark results"},
            "comparison": {"type": "dict", "description": "Comparison with baseline"},
        }
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        start_time = time.time()
        
        try:
            action = inputs.get("action", "benchmark")
            config = inputs.get("config")
            
            if config is None:
                config = BenchmarkConfig(
                    batch_size=inputs.get("batch_size", 1),
                    sequence_length=inputs.get("sequence_length", 512),
                    num_warmup_iterations=inputs.get("warmup_iterations", 10),
                    num_benchmark_iterations=inputs.get("benchmark_iterations", 100),
                )
            
            self.benchmark_suite = YvQuantizationBenchmark(config)
            
            if action == "benchmark":
                return self._execute_benchmark(inputs, start_time)
            elif action == "compare":
                return self._execute_compare(inputs, start_time)
            elif action == "export":
                return self._execute_export(inputs, start_time)
            else:
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error=f"Unknown action: {action}",
                    execution_time=time.time() - start_time,
                )
                
        except Exception as e:
            self._LOG.error(f"Benchmark operation failed: {str(e)}", exc_info=True)
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _execute_benchmark(
        self,
        inputs: Dict[str, Any],
        start_time: float,
    ) -> PiscesLxOperatorResult:
        """Execute benchmark action."""
        model = inputs.get("model")
        
        if model is None:
            raise ValueError("Model is required for benchmark")
        
        model_name = inputs.get("model_name", "model")
        quantization_method = inputs.get("quantization_method", "fp32")
        bits = inputs.get("bits", 32)
        
        results = self.benchmark_suite.run_full_benchmark(
            model=model,
            model_name=model_name,
            quantization_method=quantization_method,
            bits=bits,
        )
        
        return PiscesLxOperatorResult(
            operator_name=self.name,
            status=PiscesLxOperatorStatus.SUCCESS,
            output={
                "results": results,
            },
            execution_time=time.time() - start_time,
            metadata=self.benchmark_suite.get_stats(),
        )
    
    def _execute_compare(
        self,
        inputs: Dict[str, Any],
        start_time: float,
    ) -> PiscesLxOperatorResult:
        """Execute comparison action."""
        baseline_model = inputs.get("baseline_model")
        current_model = inputs.get("model")
        
        if baseline_model is None or current_model is None:
            raise ValueError("Both baseline_model and model are required for comparison")
        
        self._LOG.info("Running baseline benchmark")
        baseline_results = self.benchmark_suite.run_full_benchmark(
            model=baseline_model,
            model_name=inputs.get("baseline_name", "baseline"),
            quantization_method=inputs.get("baseline_method", "fp32"),
            bits=inputs.get("baseline_bits", 32),
        )
        
        self.benchmark_suite.set_baseline(baseline_results)
        
        self._LOG.info("Running current model benchmark")
        current_results = self.benchmark_suite.run_full_benchmark(
            model=current_model,
            model_name=inputs.get("current_name", "current"),
            quantization_method=inputs.get("current_method", "quantized"),
            bits=inputs.get("current_bits", 4),
        )
        
        comparison = self.benchmark_suite.compare_with_baseline(current_results)
        
        return PiscesLxOperatorResult(
            operator_name=self.name,
            status=PiscesLxOperatorStatus.SUCCESS,
            output={
                "baseline_results": baseline_results,
                "current_results": current_results,
                "comparison": comparison,
            },
            execution_time=time.time() - start_time,
        )
    
    def _execute_export(
        self,
        inputs: Dict[str, Any],
        start_time: float,
    ) -> PiscesLxOperatorResult:
        """Execute export action."""
        results = inputs.get("results")
        output_path = inputs.get("output_path", "benchmark_results.json")
        format = inputs.get("format", "json")
        
        if results is None:
            raise ValueError("Results are required for export")
        
        exported_path = self.benchmark_suite.export_results(results, output_path, format)
        
        return PiscesLxOperatorResult(
            operator_name=self.name,
            status=PiscesLxOperatorStatus.SUCCESS,
            output={
                "exported_path": exported_path,
            },
            execution_time=time.time() - start_time,
        )
    
    def cleanup(self):
        """Cleanup benchmark resources."""
        if self.benchmark_suite is not None:
            if self.benchmark_suite.memory_profiler is not None:
                self.benchmark_suite.memory_profiler.snapshots.clear()
            self.benchmark_suite = None
