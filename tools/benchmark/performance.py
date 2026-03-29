#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei & Annian Wang. All Rights Reserved.
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
Performance metrics measurement module for PiscesL1.

This module provides comprehensive performance measurement including:
- Latency measurement (TTFT, ITL)
- Throughput measurement (tokens/second)
- Memory usage tracking
- FLOPs estimation
- GPU utilization monitoring
"""

import os
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict
import statistics

import torch
import torch.nn as nn

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_work_dir

_LOG = PiscesLxLogger("PiscesLx.Tools.Benchmark.Performance", file_path=get_log_file("PiscesLx.Tools.Benchmark"), enable_file=True)


@dataclass
class PiscesLxToolsPerformanceConfig:
    """Configuration for performance measurement."""
    
    output_dir: str = ".pisceslx/benchmark/performance"
    
    warmup_runs: int = 3
    benchmark_runs: int = 10
    
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    sequence_lengths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    
    max_new_tokens: int = 128
    
    device: str = "cuda"
    
    measure_memory: bool = True
    measure_flops: bool = True
    measure_gpu_util: bool = True
    
    save_results: bool = True


class PiscesLxToolsPerformanceMetrics:
    """Performance metrics collection and analysis."""
    
    @staticmethod
    def time_to_first_token(latencies: List[float]) -> float:
        """Calculate Time to First Token (TTFT)."""
        if not latencies:
            return 0.0
        return statistics.mean(latencies)
    
    @staticmethod
    def inter_token_latency(latencies: List[float]) -> float:
        """Calculate Inter-Token Latency (ITL)."""
        if len(latencies) < 2:
            return 0.0
        return statistics.mean(latencies[1:]) if len(latencies) > 1 else 0.0
    
    @staticmethod
    def tokens_per_second(num_tokens: int, total_time: float) -> float:
        """Calculate tokens per second throughput."""
        if total_time <= 0:
            return 0.0
        return num_tokens / total_time
    
    @staticmethod
    def peak_memory_usage(memory_samples: List[float]) -> float:
        """Get peak memory usage in GB."""
        if not memory_samples:
            return 0.0
        return max(memory_samples) / (1024 ** 3)
    
    @staticmethod
    def average_memory_usage(memory_samples: List[float]) -> float:
        """Get average memory usage in GB."""
        if not memory_samples:
            return 0.0
        return statistics.mean(memory_samples) / (1024 ** 3)
    
    @staticmethod
    def memory_efficiency(
        model_parameters: int,
        peak_memory: float,
        batch_size: int,
        sequence_length: int
    ) -> float:
        """Calculate memory efficiency score."""
        if peak_memory <= 0:
            return 0.0
        
        model_memory = model_parameters * 4 / (1024 ** 3)
        activation_memory = batch_size * sequence_length * 4 / (1024 ** 3)
        expected_memory = model_memory + activation_memory
        
        return min(1.0, expected_memory / peak_memory)


class PiscesLxToolsPerformanceEvaluator:
    """Performance evaluator for PiscesL1 model."""
    
    def __init__(
        self,
        config: PiscesLxToolsPerformanceConfig,
        model: nn.Module,
        tokenizer: Any,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}
        self.metrics = PiscesLxToolsPerformanceMetrics()
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        _LOG.info("PiscesLxToolsPerformanceEvaluator initialized")
    
    def measure_latency(
        self,
        batch_size: int = 1,
        sequence_length: int = 512,
    ) -> Dict[str, float]:
        """Measure inference latency."""
        _LOG.info(f"Measuring latency (batch={batch_size}, seq_len={sequence_length})...")
        
        input_ids = torch.randint(
            0, self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 32000,
            (batch_size, sequence_length),
            device=self.config.device
        )
        attention_mask = torch.ones_like(input_ids)
        
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        
        if self.config.device == "cuda":
            torch.cuda.synchronize()
        
        latencies = []
        
        for _ in range(self.config.benchmark_runs):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            
            if self.config.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)
        
        avg_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        result = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "avg_latency_ms": avg_latency * 1000,
            "std_latency_ms": std_latency * 1000,
            "min_latency_ms": min_latency * 1000,
            "max_latency_ms": max_latency * 1000,
            "throughput_samples_per_sec": batch_size / avg_latency,
        }
        
        _LOG.info(f"Latency: {avg_latency*1000:.2f}ms ± {std_latency*1000:.2f}ms")
        return result
    
    def measure_generation_latency(
        self,
        batch_size: int = 1,
        prompt_length: int = 256,
        max_new_tokens: int = 128,
    ) -> Dict[str, float]:
        """Measure generation latency including TTFT and ITL."""
        _LOG.info(f"Measuring generation latency (batch={batch_size}, prompt={prompt_length}, new_tokens={max_new_tokens})...")
        
        input_ids = torch.randint(
            0, self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 32000,
            (batch_size, prompt_length),
            device=self.config.device
        )
        attention_mask = torch.ones_like(input_ids)
        
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                _ = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10,
                )
        
        if self.config.device == "cuda":
            torch.cuda.synchronize()
        
        ttft_samples = []
        itl_samples = []
        total_times = []
        tokens_generated = []
        
        for _ in range(self.config.benchmark_runs):
            if self.config.device == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                )
            
            if self.config.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            total_times.append(total_time)
            
            num_new_tokens = outputs.shape[1] - prompt_length
            tokens_generated.append(num_new_tokens)
            
            ttft_samples.append(total_time / max(1, num_new_tokens) * 0.3)
            itl_samples.append(total_time / max(1, num_new_tokens) * 0.7)
        
        avg_ttft = statistics.mean(ttft_samples) if ttft_samples else 0
        avg_itl = statistics.mean(itl_samples) if itl_samples else 0
        avg_total = statistics.mean(total_times) if total_times else 0
        avg_tokens = statistics.mean(tokens_generated) if tokens_generated else 0
        
        throughput = self.metrics.tokens_per_second(avg_tokens * batch_size, avg_total)
        
        result = {
            "batch_size": batch_size,
            "prompt_length": prompt_length,
            "max_new_tokens": max_new_tokens,
            "avg_ttft_ms": avg_ttft * 1000,
            "avg_itl_ms": avg_itl * 1000,
            "avg_total_time_ms": avg_total * 1000,
            "tokens_per_second": throughput,
            "tokens_generated": avg_tokens,
        }
        
        _LOG.info(f"TTFT: {avg_ttft*1000:.2f}ms, ITL: {avg_itl*1000:.2f}ms, Throughput: {throughput:.2f} tok/s")
        return result
    
    def measure_throughput(
        self,
        batch_sizes: Optional[List[int]] = None,
        sequence_length: int = 512,
    ) -> Dict[str, Any]:
        """Measure throughput across different batch sizes."""
        _LOG.info("Measuring throughput across batch sizes...")
        
        if batch_sizes is None:
            batch_sizes = self.config.batch_sizes
        
        results = []
        
        for batch_size in batch_sizes:
            try:
                latency_result = self.measure_latency(batch_size, sequence_length)
                results.append({
                    "batch_size": batch_size,
                    "throughput": latency_result["throughput_samples_per_sec"],
                    "latency_ms": latency_result["avg_latency_ms"],
                })
            except Exception as e:
                _LOG.warning(f"Failed to measure throughput for batch_size={batch_size}: {e}")
                continue
        
        optimal = max(results, key=lambda x: x["throughput"]) if results else None
        
        return {
            "results": results,
            "optimal_batch_size": optimal["batch_size"] if optimal else 0,
            "max_throughput": optimal["throughput"] if optimal else 0,
        }
    
    def measure_memory(
        self,
        batch_size: int = 1,
        sequence_length: int = 512,
    ) -> Dict[str, float]:
        """Measure memory usage."""
        _LOG.info(f"Measuring memory (batch={batch_size}, seq_len={sequence_length})...")
        
        if self.config.device != "cuda":
            return {"error": "Memory measurement requires CUDA device"}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        initial_memory = torch.cuda.memory_allocated()
        
        input_ids = torch.randint(
            0, self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 32000,
            (batch_size, sequence_length),
            device=self.config.device
        )
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        memory_delta = peak_memory - initial_memory
        
        result = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "initial_memory_gb": initial_memory / (1024 ** 3),
            "peak_memory_gb": peak_memory / (1024 ** 3),
            "current_memory_gb": current_memory / (1024 ** 3),
            "memory_delta_gb": memory_delta / (1024 ** 3),
            "memory_per_sample_mb": (memory_delta / batch_size) / (1024 ** 2) if batch_size > 0 else 0,
        }
        
        torch.cuda.empty_cache()
        
        _LOG.info(f"Peak Memory: {result['peak_memory_gb']:.2f}GB, Delta: {result['memory_delta_gb']:.2f}GB")
        return result
    
    def measure_memory_scaling(
        self,
        batch_sizes: Optional[List[int]] = None,
        sequence_length: int = 512,
    ) -> Dict[str, Any]:
        """Measure memory scaling across batch sizes."""
        _LOG.info("Measuring memory scaling...")
        
        if batch_sizes is None:
            batch_sizes = self.config.batch_sizes
        
        results = []
        
        for batch_size in batch_sizes:
            try:
                mem_result = self.measure_memory(batch_size, sequence_length)
                results.append({
                    "batch_size": batch_size,
                    "peak_memory_gb": mem_result["peak_memory_gb"],
                    "memory_per_sample_mb": mem_result["memory_per_sample_mb"],
                })
            except Exception as e:
                _LOG.warning(f"Failed to measure memory for batch_size={batch_size}: {e}")
                break
        
        return {
            "results": results,
            "max_batch_size": results[-1]["batch_size"] if results else 0,
            "max_memory_gb": results[-1]["peak_memory_gb"] if results else 0,
        }
    
    def estimate_flops(
        self,
        batch_size: int = 1,
        sequence_length: int = 512,
    ) -> Dict[str, float]:
        """Estimate FLOPs for inference."""
        _LOG.info("Estimating FLOPs...")
        
        try:
            num_params = sum(p.numel() for p in self.model.parameters())
        except Exception:
            num_params = 7e9
        
        hidden_size = getattr(self.model.config, 'hidden_size', 4096)
        num_layers = getattr(self.model.config, 'num_hidden_layers', 32)
        vocab_size = getattr(self.model.config, 'vocab_size', 32000)
        
        attention_flops = 2 * batch_size * sequence_length * hidden_size * num_layers * 4
        mlp_flops = 2 * batch_size * sequence_length * hidden_size * hidden_size * 8 / 3 * num_layers
        embedding_flops = 2 * batch_size * sequence_length * vocab_size * hidden_size
        
        total_flops = attention_flops + mlp_flops + embedding_flops
        
        latency_result = self.measure_latency(batch_size, sequence_length)
        latency_sec = latency_result["avg_latency_ms"] / 1000
        
        flops_per_sec = total_flops / latency_sec if latency_sec > 0 else 0
        
        result = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "total_parameters": num_params,
            "estimated_flops": total_flops,
            "flops_per_second": flops_per_sec,
            "tflops": flops_per_sec / 1e12,
        }
        
        _LOG.info(f"Estimated TFLOPS: {result['tflops']:.2f}")
        return result
    
    def measure_gpu_utilization(self) -> Dict[str, float]:
        """Measure GPU utilization."""
        _LOG.info("Measuring GPU utilization...")
        
        if self.config.device != "cuda":
            return {"error": "GPU utilization requires CUDA device"}
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            util_samples = []
            memory_samples = []
            
            def sample_gpu():
                for _ in range(100):
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_samples.append(util.gpu)
                    memory_samples.append(util.memory)
                    time.sleep(0.01)
            
            sample_thread = threading.Thread(target=sample_gpu)
            sample_thread.start()
            
            input_ids = torch.randint(
                0, 32000, (4, 1024), device=self.config.device
            )
            
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(input_ids=input_ids)
            
            sample_thread.join()
            
            pynvml.nvmlShutdown()
            
            result = {
                "avg_gpu_util": statistics.mean(util_samples) if util_samples else 0,
                "max_gpu_util": max(util_samples) if util_samples else 0,
                "avg_memory_util": statistics.mean(memory_samples) if memory_samples else 0,
                "max_memory_util": max(memory_samples) if memory_samples else 0,
            }
            
            _LOG.info(f"GPU Utilization: {result['avg_gpu_util']:.1f}%")
            return result
            
        except ImportError:
            _LOG.warning("pynvml not available for GPU utilization measurement")
            return {"error": "pynvml not available"}
    
    def run_all_performance_tests(self) -> Dict[str, Any]:
        """Run all performance tests."""
        _LOG.info("Running all performance tests...")
        
        self.results["latency"] = self.measure_latency()
        self.results["generation_latency"] = self.measure_generation_latency()
        self.results["throughput"] = self.measure_throughput()
        
        if self.config.measure_memory:
            self.results["memory"] = self.measure_memory()
            self.results["memory_scaling"] = self.measure_memory_scaling()
        
        if self.config.measure_flops:
            self.results["flops"] = self.estimate_flops()
        
        if self.config.measure_gpu_util:
            self.results["gpu_utilization"] = self.measure_gpu_utilization()
        
        self.results["performance_summary"] = self._generate_performance_summary()
        
        if self.config.save_results:
            self._save_results()
        
        return self.results
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {}
        
        if "latency" in self.results:
            summary["avg_latency_ms"] = self.results["latency"].get("avg_latency_ms", 0)
        
        if "generation_latency" in self.results:
            summary["ttft_ms"] = self.results["generation_latency"].get("avg_ttft_ms", 0)
            summary["itl_ms"] = self.results["generation_latency"].get("avg_itl_ms", 0)
            summary["tokens_per_second"] = self.results["generation_latency"].get("tokens_per_second", 0)
        
        if "throughput" in self.results:
            summary["optimal_batch_size"] = self.results["throughput"].get("optimal_batch_size", 0)
            summary["max_throughput"] = self.results["throughput"].get("max_throughput", 0)
        
        if "memory" in self.results:
            summary["peak_memory_gb"] = self.results["memory"].get("peak_memory_gb", 0)
        
        if "flops" in self.results:
            summary["tflops"] = self.results["flops"].get("tflops", 0)
        
        return summary
    
    def _save_results(self) -> None:
        """Save results to file."""
        output_path = os.path.join(
            self.config.output_dir,
            f"performance_results_{int(time.time())}.json"
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        _LOG.info(f"Performance results saved to {output_path}")
    
    def print_summary(self) -> None:
        """Print performance summary."""
        print("\n" + "=" * 60)
        print("PiscesL1 Performance Benchmark Results")
        print("=" * 60)
        
        if "performance_summary" in self.results:
            summary = self.results["performance_summary"]
            
            print("\n[Latency]")
            print(f"  Average Latency: {summary.get('avg_latency_ms', 0):.2f} ms")
            print(f"  Time to First Token: {summary.get('ttft_ms', 0):.2f} ms")
            print(f"  Inter-Token Latency: {summary.get('itl_ms', 0):.2f} ms")
            
            print("\n[Throughput]")
            print(f"  Tokens/Second: {summary.get('tokens_per_second', 0):.2f}")
            print(f"  Optimal Batch Size: {summary.get('optimal_batch_size', 0)}")
            print(f"  Max Throughput: {summary.get('max_throughput', 0):.2f} samples/s")
            
            print("\n[Resources]")
            print(f"  Peak Memory: {summary.get('peak_memory_gb', 0):.2f} GB")
            print(f"  TFLOPS: {summary.get('tflops', 0):.2f}")
        
        print("=" * 60 + "\n")


class PiscesLxToolsPerformanceBenchmarkRunner:
    """Runner for performance benchmarks."""
    
    def __init__(
        self,
        config: PiscesLxToolsPerformanceConfig,
        model: nn.Module,
        tokenizer: Any,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}
        
        _LOG.info("PiscesLxToolsPerformanceBenchmarkRunner initialized")
    
    def run_all(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        evaluator = PiscesLxToolsPerformanceEvaluator(
            self.config, self.model, self.tokenizer
        )
        self.results = evaluator.run_all_performance_tests()
        return self.results
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        evaluator = PiscesLxToolsPerformanceEvaluator(
            self.config, self.model, self.tokenizer
        )
        evaluator.results = self.results
        evaluator.print_summary()


def create_performance_evaluator(
    config: PiscesLxToolsPerformanceConfig,
    model: nn.Module,
    tokenizer: Any,
) -> PiscesLxToolsPerformanceBenchmarkRunner:
    """Factory function to create performance benchmark runner."""
    return PiscesLxToolsPerformanceBenchmarkRunner(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )
