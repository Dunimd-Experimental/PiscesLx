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

"""
Heterogeneous Executor for CPU/GPU Hybrid Execution

This module provides an execution engine that can intelligently schedule
operators across heterogeneous compute devices (CPU, GPU, NPU, etc.),
maximizing resource utilization and minimizing execution time.

Key Components:
    - PiscesLxDeviceTypePreference: Device type selection strategy
    - PiscesLxExecutionTarget: Target device specification
    - PiscesLxHeterogeneousConfig: Configuration for heterogeneous execution
    - PiscesLxHeterogeneousExecutor: Main hybrid execution engine
    - PiscesLxOffloadManager: CPU-GPU memory offload management
    - PiscesLxLoadBalancer: Load balancing across devices

Design Principles:
    1. Device Awareness: Automatically detect and utilize available devices
    2. Workload Matching: Match operators to optimal device types
    3. Memory Efficiency: Offload unused data to CPU memory
    4. Overlap Optimization: Overlap CPU and GPU computation
    5. Graceful Degradation: Fall back to CPU when GPU unavailable

Usage:
    executor = PiscesLxHeterogeneousExecutor()
    executor.register_operator(MyGPUOperator, preferred_device="gpu")
    executor.register_operator(MyCPUOperator, preferred_device="cpu")

    result = await executor.execute_hybrid("my_operator", inputs)
"""

import time
import asyncio
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set, Callable, Union
from queue import PriorityQueue

from .interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig
)
from .registry import PiscesLxOperatorRegistry, PiscesLxOperatorRegistryHub
from .resources import (
    PiscesLxGPUScheduler, PiscesLxResourceRequest, PiscesLxResourceAllocation,
    PiscesLxResourceEstimator, PiscesLxAllocationPriority
)
from ..dc import (
    PiscesLxLogger, PiscesLxMetrics, PiscesLxTracing,
    PiscesLxDevice, PiscesLxDeviceType, PiscesLxDeviceStatus,
    PiscesLxSystemMonitor
)


class PiscesLxDeviceTypePreference(Enum):
    """
    Device type selection strategy for operator execution.

    Strategies:
        PREFER_GPU: Use GPU if available, fall back to CPU
        PREFER_CPU: Use CPU primarily, GPU only for specific ops
        GPU_ONLY: Only execute on GPU, fail if unavailable
        CPU_ONLY: Only execute on CPU
        AUTO: Automatically select based on operator characteristics
        BALANCED: Balance load across all available devices
    """
    PREFER_GPU = "prefer_gpu"
    PREFER_CPU = "prefer_cpu"
    GPU_ONLY = "gpu_only"
    CPU_ONLY = "cpu_only"
    AUTO = "auto"
    BALANCED = "balanced"


class PiscesLxExecutionTarget(Enum):
    """
    Target device for execution.

    Targets:
        CPU: Execute on CPU cores
        GPU: Execute on GPU device
        NPU: Execute on Neural Processing Unit
        TPU: Execute on Tensor Processing Unit
        ANY: Execute on any available device
        PARALLEL: Execute in parallel across multiple devices
    """
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    TPU = "tpu"
    ANY = "any"
    PARALLEL = "parallel"


@dataclass
class PiscesLxHeterogeneousConfig:
    """
    Configuration for heterogeneous execution.

    Attributes:
        device_preference: Default device selection strategy
        max_cpu_workers: Maximum CPU worker threads
        max_gpu_workers: Maximum concurrent GPU operations
        enable_offloading: Enable CPU-GPU memory offloading
        offload_threshold_gb: Memory threshold for offloading (GB)
        enable_overlap: Enable CPU-GPU computation overlap
        gpu_memory_fraction: Fraction of GPU memory to use
        cpu_memory_fraction: Fraction of system RAM to use
        scheduling_interval_ms: Device scheduling interval
        load_balance_strategy: Load balancing strategy
        fallback_enabled: Enable fallback to CPU on GPU failure
    """
    device_preference: PiscesLxDeviceTypePreference = PiscesLxDeviceTypePreference.AUTO
    max_cpu_workers: int = 8
    max_gpu_workers: int = 4
    enable_offloading: bool = True
    offload_threshold_gb: float = 4.0
    enable_overlap: bool = True
    gpu_memory_fraction: float = 0.9
    cpu_memory_fraction: float = 0.5
    scheduling_interval_ms: int = 100
    load_balance_strategy: str = "least_loaded"
    fallback_enabled: bool = True


@dataclass
class PiscesLxDeviceLoad:
    """
    Load information for a compute device.

    Attributes:
        device_id: Device identifier
        device_type: Type of device
        compute_utilization: Compute utilization (0-1)
        memory_utilization: Memory utilization (0-1)
        queue_depth: Number of pending tasks
        avg_latency_ms: Average task latency
        is_available: Whether device is available
    """
    device_id: str
    device_type: PiscesLxDeviceType
    compute_utilization: float = 0.0
    memory_utilization: float = 0.0
    queue_depth: int = 0
    avg_latency_ms: float = 0.0
    is_available: bool = True

    @property
    def load_score(self) -> float:
        """
        Calculate overall load score for scheduling decisions.

        Lower score = less loaded = better target.

        Returns:
            Load score (0-1, lower is better)
        """
        if not self.is_available:
            return float('inf')

        return (
            self.compute_utilization * 0.4 +
            self.memory_utilization * 0.3 +
            min(self.queue_depth / 10, 1.0) * 0.2 +
            min(self.avg_latency_ms / 1000, 1.0) * 0.1
        )


@dataclass
class PiscesLxScheduledTask:
    """
    A task scheduled for heterogeneous execution.

    Attributes:
        task_id: Unique task identifier
        operator_name: Name of operator to execute
        inputs: Input data
        config: Execution configuration
        target_device: Target device for execution
        priority: Task priority
        submitted_at: Submission timestamp
        started_at: Execution start timestamp
        dependencies: Task IDs that must complete first
    """
    task_id: str
    operator_name: str
    inputs: Dict[str, Any]
    config: Dict[str, Any]
    target_device: Optional[PiscesLxExecutionTarget] = None
    priority: int = 0
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)

    def __lt__(self, other: 'PiscesLxScheduledTask') -> bool:
        """Compare tasks by priority for priority queue."""
        return self.priority < other.priority


class PiscesLxOffloadManager:
    """
    CPU-GPU memory offload management.

    Manages offloading of tensors between CPU and GPU memory to
    enable larger models and batch sizes than GPU memory allows.

    Features:
        - Automatic offload of unused tensors
        - Prefetch for anticipated tensor needs
        - LRU-based eviction policy
        - Memory pressure handling

    Attributes:
        gpu_memory_limit: Maximum GPU memory to use
        cpu_memory_limit: Maximum CPU memory to use
        gpu_tensors: Tensors currently on GPU
        cpu_tensors: Tensors offloaded to CPU
        access_history: Access history for LRU eviction
    """

    def __init__(
        self,
        gpu_memory_limit_gb: float = 16.0,
        cpu_memory_limit_gb: float = 64.0
    ):
        """
        Initialize offload manager.

        Args:
            gpu_memory_limit_gb: GPU memory limit in GB
            cpu_memory_limit_gb: CPU memory limit in GB
        """
        self.gpu_memory_limit = gpu_memory_limit_gb * 1024 ** 3
        self.cpu_memory_limit = cpu_memory_limit_gb * 1024 ** 3
        self.gpu_tensors: Dict[str, Tuple[Any, int]] = {}
        self.cpu_tensors: Dict[str, Tuple[Any, int]] = {}
        self.access_history: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._metrics = PiscesLxMetrics()
        self._logger = PiscesLxLogger("PiscesLx.Core.OPSC.OffloadManager", file_path=get_log_file("PiscesLx.Core.OPSC.OffloadManager"), enable_file=True)

    def put(
        self,
        tensor_id: str,
        tensor: Any,
        device: str = "gpu"
    ) -> bool:
        """
        Store a tensor on specified device.

        Args:
            tensor_id: Unique tensor identifier
            tensor: Tensor to store
            device: Target device ("gpu" or "cpu")

        Returns:
            True if stored successfully
        """
        with self._lock:
            try:
                import torch
                size_bytes = tensor.numel() * tensor.element_size()
            except (ImportError, AttributeError):
                size_bytes = 0

            if device == "gpu":
                current_usage = sum(s for _, s in self.gpu_tensors.values())
                if current_usage + size_bytes > self.gpu_memory_limit:
                    self._evict_gpu(size_bytes)

                self.gpu_tensors[tensor_id] = (tensor, size_bytes)
                self._metrics.gauge("offload_gpu_memory_gb", current_usage / (1024 ** 3))
            else:
                current_usage = sum(s for _, s in self.cpu_tensors.values())
                if current_usage + size_bytes > self.cpu_memory_limit:
                    self._evict_cpu(size_bytes)

                self.cpu_tensors[tensor_id] = (tensor, size_bytes)
                self._metrics.gauge("offload_cpu_memory_gb", current_usage / (1024 ** 3))

            self.access_history[tensor_id] = time.time()
            return True

    def get(
        self,
        tensor_id: str,
        target_device: str = "gpu"
    ) -> Optional[Any]:
        """
        Retrieve a tensor, optionally moving to target device.

        Args:
            tensor_id: Tensor identifier
            target_device: Desired device ("gpu" or "cpu")

        Returns:
            Tensor or None if not found
        """
        with self._lock:
            self.access_history[tensor_id] = time.time()

            if tensor_id in self.gpu_tensors:
                tensor, _ = self.gpu_tensors[tensor_id]
                if target_device == "cpu":
                    return self._to_cpu(tensor)
                return tensor

            if tensor_id in self.cpu_tensors:
                tensor, _ = self.cpu_tensors[tensor_id]
                if target_device == "gpu":
                    return self._to_gpu(tensor)
                return tensor

            return None

    def _to_cpu(self, tensor: Any) -> Any:
        """Move tensor to CPU."""
        try:
            import torch
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                return tensor.cpu()
        except ImportError:
            pass
        return tensor

    def _to_gpu(self, tensor: Any) -> Any:
        """Move tensor to GPU."""
        try:
            import torch
            if isinstance(tensor, torch.Tensor) and not tensor.is_cuda:
                return tensor.cuda()
        except ImportError:
            pass
        return tensor

    def _evict_gpu(self, needed_bytes: int) -> int:
        """
        Evict tensors from GPU to make room.

        Args:
            needed_bytes: Bytes needed

        Returns:
            Bytes freed
        """
        freed = 0
        sorted_by_access = sorted(
            self.access_history.items(),
            key=lambda x: x[1]
        )

        for tensor_id, _ in sorted_by_access:
            if tensor_id not in self.gpu_tensors:
                continue

            tensor, size = self.gpu_tensors.pop(tensor_id)
            self.cpu_tensors[tensor_id] = (self._to_cpu(tensor), size)
            freed += size
            self._metrics.counter("offload_gpu_evictions")

            if freed >= needed_bytes:
                break

        return freed

    def _evict_cpu(self, needed_bytes: int) -> int:
        """
        Evict tensors from CPU memory.

        Args:
            needed_bytes: Bytes needed

        Returns:
            Bytes freed
        """
        freed = 0
        sorted_by_access = sorted(
            self.access_history.items(),
            key=lambda x: x[1]
        )

        for tensor_id, _ in sorted_by_access:
            if tensor_id not in self.cpu_tensors:
                continue

            _, size = self.cpu_tensors.pop(tensor_id)
            self.access_history.pop(tensor_id, None)
            freed += size
            self._metrics.counter("offload_cpu_evictions")

            if freed >= needed_bytes:
                break

        return freed

    def prefetch(
        self,
        tensor_ids: List[str],
        target_device: str = "gpu"
    ) -> None:
        """
        Prefetch tensors to target device.

        Args:
            tensor_ids: List of tensor IDs to prefetch
            target_device: Target device
        """
        for tensor_id in tensor_ids:
            self.get(tensor_id, target_device)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get offload manager statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            gpu_usage = sum(s for _, s in self.gpu_tensors.values())
            cpu_usage = sum(s for _, s in self.cpu_tensors.values())

            return {
                "gpu_tensors": len(self.gpu_tensors),
                "cpu_tensors": len(self.cpu_tensors),
                "gpu_memory_gb": gpu_usage / (1024 ** 3),
                "cpu_memory_gb": cpu_usage / (1024 ** 3),
                "gpu_limit_gb": self.gpu_memory_limit / (1024 ** 3),
                "cpu_limit_gb": self.cpu_memory_limit / (1024 ** 3),
                "gpu_utilization": gpu_usage / self.gpu_memory_limit if self.gpu_memory_limit > 0 else 0,
                "cpu_utilization": cpu_usage / self.cpu_memory_limit if self.cpu_memory_limit > 0 else 0
            }


class PiscesLxLoadBalancer:
    """
    Load balancer for heterogeneous device scheduling.

    Implements various strategies for distributing work across
    available compute devices.

    Strategies:
        - least_loaded: Select device with lowest load
        - round_robin: Rotate through available devices
        - weighted: Weight by device capability
        - locality_aware: Consider data locality
    """

    def __init__(
        self,
        strategy: str = "least_loaded"
    ):
        """
        Initialize load balancer.

        Args:
            strategy: Load balancing strategy
        """
        self.strategy = strategy
        self._round_robin_index = 0
        self._device_loads: Dict[str, PiscesLxDeviceLoad] = {}
        self._lock = threading.RLock()
        self._logger = PiscesLxLogger("PiscesLx.Core.OPSC.LoadBalancer", file_path=get_log_file("PiscesLx.Core.OPSC.LoadBalancer"), enable_file=True)

    def update_device_load(
        self,
        device_id: str,
        device_type: PiscesLxDeviceType,
        compute_util: float,
        memory_util: float,
        queue_depth: int,
        avg_latency_ms: float = 0.0
    ) -> None:
        """
        Update load information for a device.

        Args:
            device_id: Device identifier
            device_type: Type of device
            compute_util: Compute utilization (0-1)
            memory_util: Memory utilization (0-1)
            queue_depth: Number of pending tasks
            avg_latency_ms: Average task latency
        """
        with self._lock:
            self._device_loads[device_id] = PiscesLxDeviceLoad(
                device_id=device_id,
                device_type=device_type,
                compute_utilization=compute_util,
                memory_utilization=memory_util,
                queue_depth=queue_depth,
                avg_latency_ms=avg_latency_ms
            )

    def select_device(
        self,
        preferred_type: Optional[PiscesLxExecutionTarget] = None,
        exclude: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Select the best device for a task.

        Args:
            preferred_type: Preferred device type
            exclude: Device IDs to exclude

        Returns:
            Selected device ID or None
        """
        exclude = exclude or set()

        with self._lock:
            available = [
                load for load in self._device_loads.values()
                if load.device_id not in exclude and load.is_available
            ]

            if preferred_type:
                type_map = {
                    PiscesLxExecutionTarget.GPU: PiscesLxDeviceType.GPU,
                    PiscesLxExecutionTarget.CPU: PiscesLxDeviceType.CPU,
                    PiscesLxExecutionTarget.NPU: PiscesLxDeviceType.NPU,
                    PiscesLxExecutionTarget.TPU: PiscesLxDeviceType.TPU,
                }
                target_type = type_map.get(preferred_type)
                if target_type:
                    available = [l for l in available if l.device_type == target_type]

            if not available:
                return None

            if self.strategy == "least_loaded":
                best = min(available, key=lambda l: l.load_score)
                return best.device_id

            elif self.strategy == "round_robin":
                available_list = [l.device_id for l in available]
                if available_list:
                    selected = available_list[self._round_robin_index % len(available_list)]
                    self._round_robin_index += 1
                    return selected

            elif self.strategy == "weighted":
                import random
                weights = [1.0 / (l.load_score + 0.1) for l in available]
                total = sum(weights)
                weights = [w / total for w in weights]
                selected = random.choices(available, weights=weights)[0]
                return selected.device_id

            else:
                return available[0].device_id

    def get_device_load(self, device_id: str) -> Optional[PiscesLxDeviceLoad]:
        """
        Get load information for a device.

        Args:
            device_id: Device identifier

        Returns:
            Device load or None
        """
        return self._device_loads.get(device_id)

    def get_all_loads(self) -> Dict[str, PiscesLxDeviceLoad]:
        """
        Get load information for all devices.

        Returns:
            Dictionary of device loads
        """
        return dict(self._device_loads)


class PiscesLxHeterogeneousExecutor:
    """
    Main heterogeneous execution engine.

    This class provides a unified interface for executing operators
    across heterogeneous compute devices, handling device selection,
    memory management, and load balancing automatically.

    Features:
        - Automatic device selection based on operator characteristics
        - CPU-GPU hybrid execution with overlap optimization
        - Memory offloading for large models
        - Load balancing across devices
        - Graceful fallback on device failures
        - Comprehensive monitoring and metrics

    Attributes:
        config: Heterogeneous execution configuration
        registry: Operator registry
        gpu_scheduler: GPU resource scheduler
        offload_manager: Memory offload manager
        load_balancer: Device load balancer
        cpu_pool: CPU thread pool
        task_queue: Priority queue for pending tasks
        active_tasks: Currently executing tasks
    """

    def __init__(
        self,
        registry: Optional[PiscesLxOperatorRegistry] = None,
        config: Optional[PiscesLxHeterogeneousConfig] = None
    ):
        """
        Initialize heterogeneous executor.

        Args:
            registry: Operator registry (default: global registry)
            config: Execution configuration
        """
        self.registry = registry or PiscesLxOperatorRegistryHub.get_registry()
        self.config = config or PiscesLxHeterogeneousConfig()
        self.gpu_scheduler = PiscesLxGPUScheduler()
        self.resource_estimator = PiscesLxResourceEstimator()
        self.offload_manager = PiscesLxOffloadManager()
        self.load_balancer = PiscesLxLoadBalancer(
            strategy=self.config.load_balance_strategy
        )

        self.cpu_pool = ThreadPoolExecutor(max_workers=self.config.max_cpu_workers)
        self.task_queue: PriorityQueue = PriorityQueue()
        self.active_tasks: Dict[str, PiscesLxScheduledTask] = {}
        self.completed_tasks: Dict[str, PiscesLxOperatorResult] = {}

        self._next_task_id = 0
        self._lock = threading.RLock()
        self._metrics = PiscesLxMetrics()
        self._tracing = PiscesLxTracing()
        self._logger = PiscesLxLogger("PiscesLx.Core.OPSC.HeterogeneousExecutor", file_path=get_log_file("PiscesLx.Core.OPSC.HeterogeneousExecutor"), enable_file=True)

        self._operator_device_preference: Dict[str, PiscesLxExecutionTarget] = {}
        self._system_monitor = PiscesLxSystemMonitor()
        self._initialize_device_loads()

    def _initialize_device_loads(self) -> None:
        """Initialize device load tracking."""
        self.load_balancer.update_device_load(
            device_id="cpu-0",
            device_type=PiscesLxDeviceType.CPU,
            compute_util=0.0,
            memory_util=0.0,
            queue_depth=0
        )

        cpu_info = self._system_monitor.get_cpu_info()
        if cpu_info:
            pass

    def register_operator(
        self,
        operator_class: type,
        preferred_device: PiscesLxExecutionTarget = PiscesLxExecutionTarget.ANY
    ) -> None:
        """
        Register an operator with device preference.

        Args:
            operator_class: Operator class to register
            preferred_device: Preferred execution device
        """
        self.registry.register(operator_class)

        try:
            temp_instance = operator_class()
            operator_name = temp_instance.name
            self._operator_device_preference[operator_name] = preferred_device
        except Exception:
            pass

    def _determine_target_device(
        self,
        operator_name: str,
        inputs: Dict[str, Any]
    ) -> PiscesLxExecutionTarget:
        """
        Determine the best target device for an operator.

        Args:
            operator_name: Operator name
            inputs: Input data

        Returns:
            Target device for execution
        """
        preference = self._operator_device_preference.get(operator_name)

        if preference and preference != PiscesLxExecutionTarget.ANY:
            return preference

        if self.config.device_preference == PiscesLxDeviceTypePreference.GPU_ONLY:
            return PiscesLxExecutionTarget.GPU
        elif self.config.device_preference == PiscesLxDeviceTypePreference.CPU_ONLY:
            return PiscesLxExecutionTarget.CPU
        elif self.config.device_preference == PiscesLxDeviceTypePreference.PREFER_CPU:
            return PiscesLxExecutionTarget.CPU

        gpu_keywords = ["attention", "linear", "conv", "matmul", "embedding", "softmax", "layernorm"]
        cpu_keywords = ["json", "text", "tokenize", "parse", "io", "file"]

        operator_lower = operator_name.lower()

        for keyword in cpu_keywords:
            if keyword in operator_lower:
                return PiscesLxExecutionTarget.CPU

        for keyword in gpu_keywords:
            if keyword in operator_lower:
                return PiscesLxExecutionTarget.GPU

        return PiscesLxExecutionTarget.CPU

    def execute_sync(
        self,
        operator_name: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> PiscesLxOperatorResult:
        """
        Execute an operator synchronously on the best available device.

        Args:
            operator_name: Name of operator to execute
            inputs: Input data
            config: Optional execution configuration

        Returns:
            Execution result
        """
        start_time = time.time()
        config = config or {}

        target_device = self._determine_target_device(operator_name, inputs)

        span = self._tracing.start_span(
            f"hetero_exec_{operator_name}",
            attributes={
                "operator": operator_name,
                "target_device": target_device.value
            }
        )

        try:
            if target_device == PiscesLxExecutionTarget.GPU:
                result = self._execute_on_gpu(operator_name, inputs, config)
            elif target_device == PiscesLxExecutionTarget.CPU:
                result = self._execute_on_cpu(operator_name, inputs, config)
            else:
                result = self._execute_on_any(operator_name, inputs, config)

            if result.is_failed() and self.config.fallback_enabled:
                if target_device == PiscesLxExecutionTarget.GPU:
                    self._logger.warning(
                        "hetero_fallback_to_cpu",
                        operator=operator_name,
                        error=result.error
                    )
                    result = self._execute_on_cpu(operator_name, inputs, config)

            self._metrics.histogram(
                "hetero_execution_time_ms",
                (time.time() - start_time) * 1000
            )

            return result

        except Exception as e:
            self._tracing.end_span(span, status="error", error_message=str(e))
            return PiscesLxOperatorResult(
                operator_name=operator_name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _execute_on_gpu(
        self,
        operator_name: str,
        inputs: Dict[str, Any],
        config: Dict[str, Any]
    ) -> PiscesLxOperatorResult:
        """
        Execute operator on GPU.

        Args:
            operator_name: Operator name
            inputs: Input data
            config: Configuration

        Returns:
            Execution result
        """
        start_time = time.time()

        input_shape = self._get_input_shape(inputs)
        resource_request = self.resource_estimator.estimate_operator_resources(
            operator_name, input_shape, config
        )

        allocation = self.gpu_scheduler.allocate(resource_request)

        if not allocation.granted:
            return PiscesLxOperatorResult(
                operator_name=operator_name,
                status=PiscesLxOperatorStatus.FAILED,
                error="GPU resource allocation failed",
                execution_time=time.time() - start_time
            )

        try:
            with self.gpu_scheduler.device_context(allocation) as device_id:
                self._update_device_load(f"gpu-{device_id}", queue_delta=1)

                operator = self.registry.create_instance(operator_name)
                if not operator:
                    return PiscesLxOperatorResult(
                        operator_name=operator_name,
                        status=PiscesLxOperatorStatus.FAILED,
                        error=f"Operator '{operator_name}' not found",
                        execution_time=time.time() - start_time
                    )

                result = operator.execute(inputs, **config)

                self._update_device_load(f"gpu-{device_id}", queue_delta=-1)

                return result

        except Exception as e:
            self._logger.error("gpu_execution_failed", operator=operator_name, error=str(e))
            return PiscesLxOperatorResult(
                operator_name=operator_name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )
        finally:
            self.gpu_scheduler.release(allocation)

    def _execute_on_cpu(
        self,
        operator_name: str,
        inputs: Dict[str, Any],
        config: Dict[str, Any]
    ) -> PiscesLxOperatorResult:
        """
        Execute operator on CPU.

        Args:
            operator_name: Operator name
            inputs: Input data
            config: Configuration

        Returns:
            Execution result
        """
        start_time = time.time()

        self._update_device_load("cpu-0", queue_delta=1)

        try:
            operator = self.registry.create_instance(operator_name)
            if not operator:
                return PiscesLxOperatorResult(
                    operator_name=operator_name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error=f"Operator '{operator_name}' not found",
                    execution_time=time.time() - start_time
                )

            result = operator.execute(inputs, **config)
            return result

        except Exception as e:
            self._logger.error("cpu_execution_failed", operator=operator_name, error=str(e))
            return PiscesLxOperatorResult(
                operator_name=operator_name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )
        finally:
            self._update_device_load("cpu-0", queue_delta=-1)

    def _execute_on_any(
        self,
        operator_name: str,
        inputs: Dict[str, Any],
        config: Dict[str, Any]
    ) -> PiscesLxOperatorResult:
        """
        Execute operator on any available device.

        Args:
            operator_name: Operator name
            inputs: Input data
            config: Configuration

        Returns:
            Execution result
        """
        device_id = self.load_balancer.select_device()

        if device_id and device_id.startswith("gpu-"):
            return self._execute_on_gpu(operator_name, inputs, config)
        else:
            return self._execute_on_cpu(operator_name, inputs, config)

    async def execute_async(
        self,
        operator_name: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> PiscesLxOperatorResult:
        """
        Execute an operator asynchronously.

        Args:
            operator_name: Operator name
            inputs: Input data
            config: Configuration

        Returns:
            Execution result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.cpu_pool,
            self.execute_sync,
            operator_name,
            inputs,
            config
        )

    def execute_batch(
        self,
        executions: List[Dict[str, Any]]
    ) -> List[PiscesLxOperatorResult]:
        """
        Execute multiple operators concurrently.

        Distributes tasks across available devices for parallel execution.

        Args:
            executions: List of execution specifications

        Returns:
            List of results in completion order
        """
        futures: List[Future] = []
        results: List[PiscesLxOperatorResult] = []

        for exec_spec in executions:
            future = self.cpu_pool.submit(
                self.execute_sync,
                exec_spec["operator"],
                exec_spec.get("inputs", {}),
                exec_spec.get("config", {})
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result(timeout=300)
                results.append(result)
            except Exception as e:
                results.append(PiscesLxOperatorResult(
                    operator_name="unknown",
                    status=PiscesLxOperatorStatus.FAILED,
                    error=str(e)
                ))

        return results

    def execute_pipeline(
        self,
        pipeline: List[Dict[str, Any]]
    ) -> List[PiscesLxOperatorResult]:
        """
        Execute a pipeline with CPU-GPU overlap optimization.

        Overlaps CPU preprocessing with GPU computation for
        improved throughput.

        Args:
            pipeline: Pipeline specification

        Returns:
            List of results for each stage
        """
        results: List[PiscesLxOperatorResult] = []
        previous_output = None

        for i, stage in enumerate(pipeline):
            inputs = stage.get("inputs", {}).copy()
            if previous_output is not None:
                inputs["previous_output"] = previous_output

            if self.config.enable_overlap and i < len(pipeline) - 1:
                next_stage = pipeline[i + 1]
                next_target = self._determine_target_device(
                    next_stage["operator"],
                    {}
                )

                if next_target == PiscesLxExecutionTarget.GPU:
                    self._prefetch_for_next_stage(next_stage)

            result = self.execute_sync(
                stage["operator"],
                inputs,
                stage.get("config", {})
            )

            results.append(result)

            if not result.is_success() and i < len(pipeline) - 1:
                break

            previous_output = result.output

        return results

    def _prefetch_for_next_stage(self, stage: Dict[str, Any]) -> None:
        """
        Prefetch data for next pipeline stage.

        Args:
            stage: Next stage specification
        """
        inputs = stage.get("inputs", {})
        tensor_ids = inputs.get("_tensor_ids", [])

        if tensor_ids:
            self.offload_manager.prefetch(tensor_ids, "gpu")

    def _get_input_shape(self, inputs: Dict[str, Any]) -> Tuple[int, ...]:
        """
        Get shape of input data.

        Args:
            inputs: Input dictionary

        Returns:
            Shape tuple
        """
        for key, value in inputs.items():
            try:
                import torch
                if isinstance(value, torch.Tensor):
                    return tuple(value.shape)
            except ImportError:
                pass

            if hasattr(value, "shape"):
                return tuple(value.shape)

            if isinstance(value, (list, tuple)) and len(value) > 0:
                return (len(value),)

        return (1,)

    def _update_device_load(
        self,
        device_id: str,
        queue_delta: int = 0
    ) -> None:
        """
        Update device load information.

        Args:
            device_id: Device identifier
            queue_delta: Change in queue depth
        """
        current_load = self.load_balancer.get_device_load(device_id)

        if current_load:
            self.load_balancer.update_device_load(
                device_id=device_id,
                device_type=current_load.device_type,
                compute_util=current_load.compute_utilization,
                memory_util=current_load.memory_utilization,
                queue_depth=current_load.queue_depth + queue_delta,
                avg_latency_ms=current_load.avg_latency_ms
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get executor statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "config": {
                "device_preference": self.config.device_preference.value,
                "max_cpu_workers": self.config.max_cpu_workers,
                "max_gpu_workers": self.config.max_gpu_workers,
                "enable_offloading": self.config.enable_offloading,
                "enable_overlap": self.config.enable_overlap
            },
            "gpu_scheduler": self.gpu_scheduler.get_stats(),
            "offload_manager": self.offload_manager.get_stats(),
            "load_balancer": {
                device_id: {
                    "device_type": load.device_type.value,
                    "compute_util": load.compute_utilization,
                    "memory_util": load.memory_utilization,
                    "queue_depth": load.queue_depth,
                    "load_score": load.load_score
                }
                for device_id, load in self.load_balancer.get_all_loads().items()
            },
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks)
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health check results
        """
        gpu_health = self.gpu_scheduler.health_check()
        cpu_usage = self._system_monitor.get_cpu_usage()
        mem_usage = self._system_monitor.get_memory_usage()

        return {
            "healthy": gpu_health.get("healthy", True),
            "gpu": gpu_health,
            "cpu": {
                "usage_percent": cpu_usage,
                "memory_percent": mem_usage
            },
            "executor_stats": self.get_stats()
        }

    def shutdown(self) -> None:
        """
        Shutdown the executor.

        Releases all resources and stops worker threads.
        """
        self.cpu_pool.shutdown(wait=True)
        self.gpu_scheduler.cleanup_expired()

        self._logger.info("heterogeneous_executor_shutdown")
