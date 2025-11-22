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

import time
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import gc
import psutil
import os

from ..core import PiscesLxCoreLog, PiscesLxCoreException
from .config import QuantizationConfig

logger = PiscesLxCoreLog("quantization.benchmark")


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    # Performance metrics
    inference_time_ms: float
    throughput_samples_per_sec: float
    memory_usage_mb: float
    peak_memory_mb: float
    
    # Accuracy metrics
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    top1_accuracy: Optional[float] = None
    top5_accuracy: Optional[float] = None
    
    # Model info
    model_size_mb: float = 0.0
    parameter_count: int = 0
    
    # Comparison metrics
    speedup_ratio: Optional[float] = None
    memory_reduction_ratio: Optional[float] = None
    accuracy_degradation: Optional[float] = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    num_runs: int = 10
    warmup_runs: int = 3
    batch_size: int = 1
    sequence_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "float16"
    enable_memory_tracking: bool = True
    enable_profiling: bool = False


class ModelBenchmark:
    """Comprehensive benchmarking suite for quantized models."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.baseline_metrics = None
        self.results_history = []
    
    def benchmark_model(
        self,
        model: nn.Module,
        test_data: Optional[torch.Tensor] = None,
        model_name: str = "model",
        is_baseline: bool = False
    ) -> BenchmarkMetrics:
        """
        Run comprehensive benchmark on a model.
        
        Args:
            model: Model to benchmark
            test_data: Optional test data for accuracy evaluation
            model_name: Name for logging
            is_baseline: Whether this is the baseline model
            
        Returns:
            Benchmark metrics
        """
        try:
            logger.info(f"Starting benchmark for {model_name}")
            
            # Prepare model
            model = model.to(self.config.device)
            model.eval()
            
            # Generate test data if not provided
            if test_data is None:
                test_data = self._generate_test_data()
            
            # Warmup runs
            logger.info(f"Running {self.config.warmup_runs} warmup iterations")
            self._warmup_model(model, test_data)
            
            # Performance benchmarking
            logger.info(f"Running {self.config.num_runs} benchmark iterations")
            performance_metrics = self._benchmark_performance(model, test_data)
            
            # Memory benchmarking
            if self.config.enable_memory_tracking:
                memory_metrics = self._benchmark_memory(model, test_data)
                performance_metrics.update(memory_metrics)
            
            # Model size analysis
            size_metrics = self._analyze_model_size(model)
            performance_metrics.update(size_metrics)
            
            # Create metrics object
            metrics = BenchmarkMetrics(**performance_metrics)
            
            # Store baseline if this is baseline
            if is_baseline:
                self.baseline_metrics = metrics
                logger.info("Stored baseline metrics for comparison")
            else:
                # Calculate comparison metrics
                metrics = self._calculate_comparison_metrics(metrics)
            
            # Log results
            self._log_benchmark_results(model_name, metrics)
            
            # Store in history
            self.results_history.append({
                "model_name": model_name,
                "metrics": asdict(metrics),
                "timestamp": time.time()
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}", error=str(e))
            raise PiscesLxCoreException(f"Benchmark failed: {str(e)}")
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        test_data: Optional[torch.Tensor] = None,
        baseline_model: Optional[str] = None
    ) -> Dict[str, BenchmarkMetrics]:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of model names to models
            test_data: Optional test data
            baseline_model: Name of baseline model for comparison
            
        Returns:
            Dictionary of model names to metrics
        """
        try:
            results = {}
            
            # Determine baseline
            if baseline_model is None:
                baseline_model = list(models.keys())[0]
            
            # Benchmark baseline first
            if baseline_model in models:
                logger.info(f"Benchmarking baseline model: {baseline_model}")
                baseline_metrics = self.benchmark_model(
                    models[baseline_model],
                    test_data,
                    baseline_model,
                    is_baseline=True
                )
                results[baseline_model] = baseline_metrics
            
            # Benchmark other models
            for model_name, model in models.items():
                if model_name == baseline_model:
                    continue
                
                logger.info(f"Benchmarking model: {model_name}")
                metrics = self.benchmark_model(
                    model,
                    test_data,
                    model_name,
                    is_baseline=False
                )
                results[model_name] = metrics
            
            return results
            
        except Exception as e:
            logger.error("Model comparison failed", error=str(e))
            raise PiscesLxCoreException(f"Model comparison failed: {str(e)}")
    
    def _warmup_model(self, model: nn.Module, test_data: torch.Tensor) -> None:
        """Run warmup iterations."""
        with torch.no_grad():
            for _ in range(self.config.warmup_runs):
                _ = model(test_data)
                if self.config.device == "cuda":
                    torch.cuda.synchronize()
    
    def _benchmark_performance(self, model: nn.Module, test_data: torch.Tensor) -> Dict[str, float]:
        """Benchmark inference performance."""
        times = []
        
        with torch.no_grad():
            for _ in range(self.config.num_runs):
                if self.config.device == "cuda":
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = model(test_data)
                
                if self.config.device == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        throughput = (self.config.batch_size * self.config.num_runs) / (sum(times) / 1000)
        
        return {
            "inference_time_ms": avg_time,
            "throughput_samples_per_sec": throughput
        }
    
    def _benchmark_memory(self, model: nn.Module, test_data: torch.Tensor) -> Dict[str, float]:
        """Benchmark memory usage."""
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Run inference
            with torch.no_grad():
                _ = model(test_data)
            
            # Get memory statistics
            memory_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            
            return {
                "memory_usage_mb": memory_usage,
                "peak_memory_mb": peak_memory
            }
        else:
            # CPU memory tracking
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                "memory_usage_mb": memory_info.rss / (1024 * 1024),  # MB
                "peak_memory_mb": memory_info.rss / (1024 * 1024)  # Approximation
            }
    
    def _analyze_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Analyze model size and parameter count."""
        param_count = sum(p.numel() for p in model.parameters())
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        return {
            "model_size_mb": model_size,
            "parameter_count": param_count
        }
    
    def _generate_test_data(self) -> torch.Tensor:
        """Generate synthetic test data."""
        vocab_size = 1000
        return torch.randint(
            0, vocab_size,
            (self.config.batch_size, self.config.sequence_length),
            device=self.config.device
        )
    
    def _calculate_comparison_metrics(self, metrics: BenchmarkMetrics) -> BenchmarkMetrics:
        """Calculate comparison metrics against baseline."""
        if self.baseline_metrics is None:
            return metrics
        
        # Speedup ratio
        if metrics.inference_time_ms > 0 and self.baseline_metrics.inference_time_ms > 0:
            metrics.speedup_ratio = self.baseline_metrics.inference_time_ms / metrics.inference_time_ms
        
        # Memory reduction
        if metrics.memory_usage_mb > 0 and self.baseline_metrics.memory_usage_mb > 0:
            metrics.memory_reduction_ratio = self.baseline_metrics.memory_usage_mb / metrics.memory_usage_mb
        
        # Accuracy degradation (if accuracy is available)
        if (metrics.accuracy is not None and 
            self.baseline_metrics.accuracy is not None and
            self.baseline_metrics.accuracy > 0):
            metrics.accuracy_degradation = (
                (self.baseline_metrics.accuracy - metrics.accuracy) / self.baseline_metrics.accuracy
            ) * 100
        
        return metrics
    
    def _log_benchmark_results(self, model_name: str, metrics: BenchmarkMetrics) -> None:
        """Log benchmark results."""
        logger.info(f"Benchmark results for {model_name}")
        logger.info(f"  Inference time: {metrics.inference_time_ms:.2f}ms")
        logger.info(f"  Throughput: {metrics.throughput_samples_per_sec:.2f} samples/sec")
        logger.info(f"  Memory usage: {metrics.memory_usage_mb:.2f}MB")
        logger.info(f"  Model size: {metrics.model_size_mb:.2f}MB")
        logger.info(f"  Parameters: {metrics.parameter_count:,}")
        
        if metrics.speedup_ratio is not None:
            logger.info(f"  Speedup vs baseline: {metrics.speedup_ratio:.2f}x")
        
        if metrics.memory_reduction_ratio is not None:
            logger.info(f"  Memory reduction vs baseline: {metrics.memory_reduction_ratio:.2f}x")
        
        if metrics.accuracy_degradation is not None:
            logger.info(f"  Accuracy degradation: {metrics.accuracy_degradation:.2f}%")
    
    def save_results(self, filepath: Union[str, Path]) -> None:
        """Save benchmark results to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            results_data = {
                "config": asdict(self.config),
                "baseline_metrics": asdict(self.baseline_metrics) if self.baseline_metrics else None,
                "results_history": self.results_history,
                "timestamp": time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"Benchmark results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
            raise PiscesLxCoreException(f"Failed to save results: {str(e)}")


class QuantizationBenchmarkSuite:
    """Complete benchmarking suite for quantization analysis."""
    
    def __init__(self):
        self.benchmark = ModelBenchmark()
        self.quantization_results = {}
    
    def benchmark_quantization_pipeline(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        quantization_config: QuantizationConfig,
        test_data: Optional[torch.Tensor] = None,
        model_name: str = "quantized_model"
    ) -> Dict[str, Any]:
        """
        Benchmark complete quantization pipeline.
        
        Args:
            original_model: Original full-precision model
            quantized_model: Quantized model
            quantization_config: Configuration used for quantization
            test_data: Optional test data
            model_name: Name for the quantized model
            
        Returns:
            Comprehensive benchmark results
        """
        try:
            logger.info("Starting quantization pipeline benchmark")
            
            # Compare models
            models = {
                "original": original_model,
                model_name: quantized_model
            }
            
            results = self.benchmark.compare_models(models, test_data, baseline_model="original")
            
            # Add quantization-specific analysis
            analysis = self._analyze_quantization_benefits(
                results["original"],
                results[model_name],
                quantization_config
            )
            
            # Compile comprehensive results
            comprehensive_results = {
                "model_comparison": results,
                "quantization_analysis": analysis,
                "config": quantization_config.__dict__,
                "recommendations": self._generate_recommendations(results[model_name])
            }
            
            self.quantization_results[model_name] = comprehensive_results
            
            return comprehensive_results
            
        except Exception as e:
            logger.error("Quantization pipeline benchmark failed", error=str(e))
            raise PiscesLxCoreException(f"Pipeline benchmark failed: {str(e)}")
    
    def _analyze_quantization_benefits(
        self,
        baseline_metrics: BenchmarkMetrics,
        quantized_metrics: BenchmarkMetrics,
        config: QuantizationConfig
    ) -> Dict[str, Any]:
        """Analyze quantization benefits and trade-offs."""
        analysis = {
            "performance_improvement": {
                "speedup": quantized_metrics.speedup_ratio,
                "memory_reduction": quantized_metrics.memory_reduction_ratio,
                "size_reduction": baseline_metrics.model_size_mb / quantized_metrics.model_size_mb
            },
            "quality_assessment": {
                "accuracy_degradation": quantized_metrics.accuracy_degradation,
                "acceptable": self._is_accuracy_acceptable(quantized_metrics)
            },
            "efficiency_score": self._calculate_efficiency_score(
                baseline_metrics, quantized_metrics
            ),
            "quantization_method_analysis": {
                "method": config.method,
                "bits": config.bits,
                "granularity": config.granularity,
                "effectiveness": self._assess_quantization_effectiveness(config)
            }
        }
        
        return analysis
    
    def _is_accuracy_acceptable(self, metrics: BenchmarkMetrics, threshold: float = 5.0) -> bool:
        """Check if accuracy degradation is acceptable."""
        if metrics.accuracy_degradation is None:
            return True  # No accuracy data available
        
        return abs(metrics.accuracy_degradation) <= threshold
    
    def _calculate_efficiency_score(self, baseline: BenchmarkMetrics, quantized: BenchmarkMetrics) -> float:
        """Calculate overall efficiency score."""
        score = 0.0
        
        # Speed contribution (40%)
        if quantized.speedup_ratio is not None:
            score += min(quantized.speedup_ratio, 3.0) * 0.4
        
        # Memory contribution (30%)
        if quantized.memory_reduction_ratio is not None:
            score += min(quantized.memory_reduction_ratio, 3.0) * 0.3
        
        # Accuracy penalty (30%)
        if quantized.accuracy_degradation is not None:
            accuracy_penalty = max(0, abs(quantized.accuracy_degradation) / 10.0)
            score += (1.0 - min(accuracy_penalty, 1.0)) * 0.3
        else:
            score += 0.3  # No accuracy penalty if no data
        
        return min(score, 1.0)
    
    def _assess_quantization_effectiveness(self, config: QuantizationConfig) -> str:
        """Assess the effectiveness of the quantization method."""
        if config.bits <= 4:
            return "Aggressive - High compression, potential quality loss"
        elif config.bits <= 8:
            return "Moderate - Good balance of compression and quality"
        else:
            return "Conservative - Minimal compression, high quality"
    
    def _generate_recommendations(self, metrics: BenchmarkMetrics) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        if metrics.speedup_ratio is not None and metrics.speedup_ratio < 1.0:
            recommendations.append("Consider optimizing model architecture for better performance")
        
        if metrics.memory_reduction_ratio is not None and metrics.memory_reduction_ratio < 1.5:
            recommendations.append("Consider more aggressive quantization for better memory savings")
        
        if metrics.accuracy_degradation is not None and abs(metrics.accuracy_degradation) > 5.0:
            recommendations.append("Consider less aggressive quantization to preserve accuracy")
        
        if not recommendations:
            recommendations.append("Quantization configuration appears optimal")
        
        return recommendations


# Convenience functions
def benchmark_model(
    model: nn.Module,
    test_data: Optional[torch.Tensor] = None,
    model_name: str = "model",
    config: Optional[BenchmarkConfig] = None
) -> BenchmarkMetrics:
    """Convenience function to benchmark a single model."""
    benchmark = ModelBenchmark(config)
    return benchmark.benchmark_model(model, test_data, model_name)


def compare_models(
    models: Dict[str, nn.Module],
    test_data: Optional[torch.Tensor] = None,
    baseline_model: Optional[str] = None,
    config: Optional[BenchmarkConfig] = None
) -> Dict[str, BenchmarkMetrics]:
    """Convenience function to compare multiple models."""
    benchmark = ModelBenchmark(config)
    return benchmark.compare_models(models, test_data, baseline_model)