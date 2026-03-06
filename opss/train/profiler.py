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
Training Profiler Operator - Performance Analysis and Optimization
Based on tools/train/profiler.py

This module provides comprehensive performance profiling capabilities for
training deep learning models. It analyzes timing, memory usage, and
computational bottlenecks to help optimize training efficiency.

Key Features:
    - Timing analysis (forward pass, backward pass, data loading)
    - Memory profiling (peak memory, memory growth patterns)
    - Detailed operation-level profiling
    - Bottleneck identification and recommendations
    - Throughput calculation (samples/sec, tokens/sec)
"""

import torch
import time
from typing import Any, Dict, Optional, List, Tuple
from configs.version import VERSION
from ops.core.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorConfig


class ProfilingConfig(PiscesLxOperatorConfig):
    """
    Configuration for training performance profiling.
    
    Attributes:
        profile_memory: Enable memory usage profiling. Default: True
        profile_time: Enable timing analysis. Default: True
        warmup_steps: Number of warmup steps before profiling. Default: 3
        profile_steps: Number of steps to profile. Default: 10
        detailed_ops: Enable detailed operation-level profiling. Default: True
    """
    profile_memory: bool = True
    profile_time: bool = True
    warmup_steps: int = 3
    profile_steps: int = 10
    detailed_ops: bool = True


class ProfilingOperator(PiscesLxOperatorInterface):
    """
    Training Performance Profiling Operator.
    
    This operator provides comprehensive performance analysis for training
    deep learning models. It measures timing, memory usage, and identifies
    computational bottlenecks.
    
    The profiling process includes:
        1. Model warmup to stabilize performance
        2. Timing analysis for each training step
        3. Memory profiling to track GPU memory usage
        4. Detailed operation-level profiling
        5. Bottleneck identification and recommendations
    
    Example:
        >>> config = ProfilingConfig(profile_steps=20, profile_memory=True)
        >>> operator = ProfilingOperator()
        >>> result = operator.execute({
        ...     "model": model,
        ...     "dataloader": train_loader,
        ...     "config": config
        ... })
        >>> if result.success:
        ...     timing = result.data["timing_analysis"]
        ...     memory = result.data["memory_analysis"]
    """
    
    def __init__(self):
        """Initialize the profiling operator."""
        super().__init__()
        self.name = "training_profiler"
        self.version = VERSION
        
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute performance profiling.
        
        Performs comprehensive profiling of model training performance,
        including timing, memory, and operation-level analysis.
        
        Args:
            inputs: Dictionary containing profiling inputs
                - model: The model to profile (nn.Module)
                - dataloader: Training data loader
                - config: Profiling configuration (ProfilingConfig)
                - forward_fn: Optional custom forward function
        
        Returns:
            PiscesLxOperatorResult: Result containing
                - timing_analysis: Step timing statistics
                - memory_analysis: Memory usage statistics
                - throughput_analysis: Throughput metrics
                - bottleneck_analysis: Identified bottlenecks
                - detailed_operations: Operation-level profiling (if enabled)
        
        Raises:
            ValueError: If model or dataloader is not provided
        """
        try:
            model = inputs.get("model")
            dataloader = inputs.get("dataloader")
            config = inputs.get("config", ProfilingConfig())
            forward_fn = inputs.get("forward_fn", self._default_forward)
            
            if not model or not dataloader:
                raise ValueError("Model and dataloader are required for profiling")
            
            profiling_results = self._profile_model(model, dataloader, config, forward_fn)
            
            return PiscesLxOperatorResult(
                success=True,
                data=profiling_results,
                metadata={
                    "operator": self.name,
                    "version": self.version,
                    "profiled_steps": config.profile_steps,
                    "memory_profiled": config.profile_memory,
                    "time_profiled": config.profile_time
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                success=False,
                error=str(e),
                metadata={
                    "operator": self.name,
                    "version": self.version,
                    "error_type": type(e).__name__
                }
            )
    
    def _profile_model(self, 
                      model: torch.nn.Module,
                      dataloader,
                      config: ProfilingConfig,
                      forward_fn) -> Dict[str, Any]:
        """
        Profile model performance.
        
        Executes warmup and profiling steps, collecting timing and
        memory statistics throughout the process.
        
        Args:
            model: The neural network model
            dataloader: Training data loader
            config: Profiling configuration
            forward_fn: Forward pass function
        
        Returns:
            Dictionary containing all profiling results
        """
        model.eval()
        device = next(model.parameters()).device
        
        timing_stats = []
        memory_stats = []
        detailed_ops = {} if config.detailed_ops else None
        
        self._warmup_model(model, dataloader, config.warmup_steps, forward_fn, device)
        
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                if step >= config.profile_steps:
                    break
                
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                if config.profile_time:
                    step_timing = self._profile_step_timing(model, batch, forward_fn)
                    timing_stats.append(step_timing)
                
                if config.profile_memory:
                    step_memory = self._profile_step_memory(model, batch, forward_fn)
                    memory_stats.append(step_memory)
                
                if config.detailed_ops:
                    step_ops = self._profile_detailed_operations(model, batch, forward_fn)
                    for op_name, op_stats in step_ops.items():
                        if op_name not in detailed_ops:
                            detailed_ops[op_name] = []
                        detailed_ops[op_name].append(op_stats)
        
        results = {
            "timing_analysis": self._aggregate_timing_stats(timing_stats) if timing_stats else {},
            "memory_analysis": self._aggregate_memory_stats(memory_stats) if memory_stats else {},
            "throughput_analysis": self._calculate_throughput(timing_stats, dataloader) if timing_stats else {},
            "bottleneck_analysis": self._identify_bottlenecks(timing_stats, memory_stats) if timing_stats or memory_stats else {}
        }
        
        if detailed_ops:
            results["detailed_operations"] = self._aggregate_detailed_ops(detailed_ops)
        
        return results
    
    def _warmup_model(self, model, dataloader, warmup_steps: int, forward_fn, device):
        """
        Warmup model for profiling.
        
        Runs a few forward passes to warm up CUDA kernels and
        stabilize performance before profiling.
        
        Args:
            model: The neural network model
            dataloader: Training data loader
            warmup_steps: Number of warmup steps
            forward_fn: Forward pass function
            device: Target device
        """
        model.train()
        for step, batch in enumerate(dataloader):
            if step >= warmup_steps:
                break
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            _ = forward_fn(model, batch)
        model.eval()
    
    def _profile_step_timing(self, model, batch, forward_fn) -> Dict[str, float]:
        """
        Profile step timing.
        
        Measures the time taken for forward pass and total step time.
        
        Args:
            model: The neural network model
            batch: Input batch
            forward_fn: Forward pass function
        
        Returns:
            Dictionary with timing statistics
        """
        start_time = time.perf_counter()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_start = time.perf_counter()
        outputs = forward_fn(model, batch)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_time = time.perf_counter() - forward_start
        
        total_time = time.perf_counter() - start_time
        
        return {
            "total_time": total_time,
            "forward_time": forward_time,
            "overhead_time": total_time - forward_time
        }
    
    def _profile_step_memory(self, model, batch, forward_fn) -> Dict[str, float]:
        """
        Profile step memory usage.
        
        Measures GPU memory usage during forward pass.
        
        Args:
            model: The neural network model
            batch: Input batch
            forward_fn: Forward pass function
        
        Returns:
            Dictionary with memory statistics in MB
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        initial_memory = torch.cuda.memory_allocated()
        
        outputs = forward_fn(model, batch)
        
        peak_memory = torch.cuda.max_memory_allocated()
        
        return {
            "initial_memory_mb": initial_memory / 1024 / 1024,
            "peak_memory_mb": peak_memory / 1024 / 1024,
            "memory_increase_mb": (peak_memory - initial_memory) / 1024 / 1024
        }
    
    def _profile_detailed_operations(self, model, batch, forward_fn) -> Dict[str, Dict]:
        """
        Profile detailed operations.
        
        Uses PyTorch profiler to get detailed timing for each operation.
        
        Args:
            model: The neural network model
            batch: Input batch
            forward_fn: Forward pass function
        
        Returns:
            Dictionary mapping operation names to statistics
        """
        if not torch.cuda.is_available():
            return {}
        
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            outputs = forward_fn(model, batch)
        
        op_stats = {}
        for evt in prof.function_events:
            op_name = evt.name
            if op_name not in op_stats:
                op_stats[op_name] = {
                    "count": 0,
                    "total_time_ms": 0,
                    "avg_time_ms": 0
                }
            
            op_stats[op_name]["count"] += 1
            op_stats[op_name]["total_time_ms"] += evt.cuda_time_total / 1000
        
        for op_name in op_stats:
            op_stats[op_name]["avg_time_ms"] = (
                op_stats[op_name]["total_time_ms"] / op_stats[op_name]["count"]
            )
        
        return op_stats
    
    def _default_forward(self, model, batch):
        """
        Default forward pass function.
        
        Handles common batch formats for language models.
        
        Args:
            model: The neural network model
            batch: Input batch dictionary
        
        Returns:
            Model outputs
        """
        if "input_ids" in batch:
            return model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
        else:
            return model(**batch)
    
    def _aggregate_timing_stats(self, timing_stats: List[Dict]) -> Dict[str, float]:
        """
        Aggregate timing statistics.
        
        Computes mean and standard deviation of timing metrics.
        
        Args:
            timing_stats: List of per-step timing statistics
        
        Returns:
            Aggregated timing statistics
        """
        if not timing_stats:
            return {}
        
        total_times = [stat["total_time"] for stat in timing_stats]
        forward_times = [stat["forward_time"] for stat in timing_stats]
        overhead_times = [stat["overhead_time"] for stat in timing_stats]
        
        return {
            "avg_total_time": sum(total_times) / len(total_times),
            "avg_forward_time": sum(forward_times) / len(forward_times),
            "avg_overhead_time": sum(overhead_times) / len(overhead_times),
            "total_time_std": torch.std(torch.tensor(total_times)).item() if len(total_times) > 1 else 0,
            "forward_time_std": torch.std(torch.tensor(forward_times)).item() if len(forward_times) > 1 else 0
        }
    
    def _aggregate_memory_stats(self, memory_stats: List[Dict]) -> Dict[str, float]:
        """
        Aggregate memory statistics.
        
        Computes mean and standard deviation of memory metrics.
        
        Args:
            memory_stats: List of per-step memory statistics
        
        Returns:
            Aggregated memory statistics
        """
        if not memory_stats or "error" in memory_stats[0]:
            return {"error": "Memory profiling not available"}
        
        peak_memories = [stat["peak_memory_mb"] for stat in memory_stats]
        memory_increases = [stat["memory_increase_mb"] for stat in memory_stats]
        
        return {
            "avg_peak_memory_mb": sum(peak_memories) / len(peak_memories),
            "max_peak_memory_mb": max(peak_memories),
            "avg_memory_increase_mb": sum(memory_increases) / len(memory_increases),
            "memory_increase_std": torch.std(torch.tensor(memory_increases)).item() if len(memory_increases) > 1 else 0
        }
    
    def _calculate_throughput(self, timing_stats: List[Dict], dataloader) -> Dict[str, float]:
        """
        Calculate throughput metrics.
        
        Computes samples per second and tokens per second.
        
        Args:
            timing_stats: List of per-step timing statistics
            dataloader: Training data loader
        
        Returns:
            Throughput metrics
        """
        if not timing_stats:
            return {}
        
        avg_step_time = sum(stat["total_time"] for stat in timing_stats) / len(timing_stats)
        
        batch_size = getattr(dataloader, 'batch_size', 1)
        samples_per_second = batch_size / avg_step_time if avg_step_time > 0 else 0
        tokens_per_second = samples_per_second * getattr(dataloader.dataset, 'max_length', 512)
        
        return {
            "samples_per_second": samples_per_second,
            "tokens_per_second": tokens_per_second,
            "steps_per_second": 1 / avg_step_time if avg_step_time > 0 else 0
        }
    
    def _identify_bottlenecks(self, timing_stats: List[Dict], memory_stats: List[Dict]) -> Dict[str, Any]:
        """
        Identify performance bottlenecks.
        
        Analyzes timing and memory statistics to identify potential
        performance bottlenecks.
        
        Args:
            timing_stats: List of per-step timing statistics
            memory_stats: List of per-step memory statistics
        
        Returns:
            Dictionary with bottleneck analysis
        """
        bottlenecks = {}
        
        if timing_stats:
            forward_times = [stat["forward_time"] for stat in timing_stats]
            overhead_times = [stat["overhead_time"] for stat in timing_stats]
            
            if sum(forward_times) > sum(overhead_times):
                bottlenecks["primary_bottleneck"] = "forward_computation"
                bottlenecks["forward_percentage"] = sum(forward_times) / (sum(forward_times) + sum(overhead_times)) * 100
            else:
                bottlenecks["primary_bottleneck"] = "overhead"
                bottlenecks["overhead_percentage"] = sum(overhead_times) / (sum(forward_times) + sum(overhead_times)) * 100
        
        if memory_stats and "error" not in memory_stats[0]:
            peak_memories = [stat["peak_memory_mb"] for stat in memory_stats]
            max_memory = max(peak_memories)
            
            if max_memory > 0.8 * self._get_gpu_memory_gb() * 1024:
                bottlenecks["memory_bottleneck"] = "high_memory_usage"
                bottlenecks["peak_memory_utilization"] = max_memory / (self._get_gpu_memory_gb() * 1024) * 100
        
        return bottlenecks
    
    def _aggregate_detailed_ops(self, detailed_ops: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Aggregate detailed operation statistics.
        
        Combines per-step operation statistics and sorts by total time.
        
        Args:
            detailed_ops: Dictionary mapping operation names to per-step stats
        
        Returns:
            Aggregated operation statistics (top 20 by time)
        """
        aggregated = {}
        
        for op_name, op_list in detailed_ops.items():
            total_counts = sum(op["count"] for op in op_list)
            total_times = sum(op["total_time_ms"] for op in op_list)
            avg_times = [op["avg_time_ms"] for op in op_list]
            
            aggregated[op_name] = {
                "total_count": total_counts,
                "total_time_ms": total_times,
                "avg_time_ms": sum(avg_times) / len(avg_times),
                "time_std_ms": torch.std(torch.tensor(avg_times)).item() if len(avg_times) > 1 else 0
            }
        
        sorted_ops = sorted(aggregated.items(), key=lambda x: x[1]["total_time_ms"], reverse=True)
        return dict(sorted_ops[:20])
    
    def _get_gpu_memory_gb(self) -> float:
        """
        Get total GPU memory in GB.
        
        Returns:
            Total GPU memory in gigabytes, or 0 if CUDA unavailable
        """
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        return 0
    
    def validate_config(self, config: Any) -> bool:
        """
        Validate configuration parameters.
        
        Ensures all configuration values are within valid ranges.
        
        Args:
            config: Configuration to validate
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, ProfilingConfig):
            return False
            
        return (config.warmup_steps >= 0 and 
                config.profile_steps > 0 and
                config.profile_steps > config.warmup_steps)


def profile_training(model: torch.nn.Module,
                    dataloader,
                    profile_memory: bool = True,
                    profile_time: bool = True,
                    profile_steps: int = 10) -> Dict[str, Any]:
    """
    Convenience function for training performance profiling.
    
    Provides a simple interface for profiling model training without
    explicitly creating operator instances.
    
    Args:
        model: The model to profile
        dataloader: Training data loader
        profile_memory: Enable memory profiling
        profile_time: Enable timing profiling
        profile_steps: Number of steps to profile
    
    Returns:
        Dictionary containing profiling results
    
    Raises:
        RuntimeError: If profiling fails
    
    Example:
        >>> results = profile_training(model, train_loader, profile_steps=20)
        >>> print(f"Average forward time: {results['timing_analysis']['avg_forward_time']:.4f}s")
    """
    operator = ProfilingOperator()
    config = ProfilingConfig(
        profile_memory=profile_memory,
        profile_time=profile_time,
        profile_steps=profile_steps
    )
    
    inputs = {
        "model": model,
        "dataloader": dataloader,
        "config": config
    }
    
    result = operator.execute(inputs)
    
    if result.success:
        return result.data
    else:
        raise RuntimeError(f"Profiling failed: {result.error}")
