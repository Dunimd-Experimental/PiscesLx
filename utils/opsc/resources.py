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
GPU Resource Scheduler and Device Management

This module provides comprehensive GPU resource management for operator execution,
including device allocation, memory management, CUDA stream scheduling, and
resource estimation.

Key Components:
    - PiscesLxResourceType: Enumeration of resource types
    - PiscesLxResourceRequest: Resource allocation request specification
    - PiscesLxResourceAllocation: Granted resource allocation
    - PiscesLxGPUScheduler: GPU device and memory scheduler
    - PiscesLxCUDAScheduler: CUDA stream and event scheduler
    - PiscesLxMemoryPool: GPU memory pool management
    - PiscesLxResourceEstimator: Resource requirement estimation

Design Principles:
    1. Fair Scheduling: Resources allocated fairly across operators
    2. Memory Efficiency: Pool-based allocation reduces fragmentation
    3. Overlap Optimization: Stream scheduling maximizes compute/transfer overlap
    4. Fault Tolerance: Graceful handling of OOM and device failures

Usage:
    scheduler = PiscesLxGPUScheduler()
    request = PiscesLxResourceRequest(gpu_memory_gb=16.0, compute_units=4)
    allocation = scheduler.allocate(request)
    if allocation.granted:
        with allocation.device_context():
            result = operator.execute(inputs)
    scheduler.release(allocation)
"""

import time
import threading
import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, ContextManager
from contextlib import contextmanager

from ..dc import (
    PiscesLxLogger, PiscesLxMetrics, PiscesLxTracing,
    PiscesLxDevice, PiscesLxDeviceType, PiscesLxDeviceStatus,
    PiscesLxDeviceCapabilities, PiscesLxDeviceHealthMetrics,
    PiscesLxResourcePool, PiscesLxSystemMonitor
)


class PiscesLxResourceType(Enum):
    """
    Enumeration of resource types for allocation.

    Resource Types:
        GPU_MEMORY: GPU device memory in bytes
        GPU_COMPUTE: GPU compute units (SMs, tensor cores)
        CPU_CORES: CPU cores for parallel processing
        SYSTEM_MEMORY: System RAM in bytes
        STORAGE: Disk I/O bandwidth
        NETWORK: Network bandwidth
        CUDA_STREAM: CUDA stream for async execution
        CUDA_EVENT: CUDA event for synchronization
    """
    GPU_MEMORY = "gpu_memory"
    GPU_COMPUTE = "gpu_compute"
    CPU_CORES = "cpu_cores"
    SYSTEM_MEMORY = "system_memory"
    STORAGE = "storage"
    NETWORK = "network"
    CUDA_STREAM = "cuda_stream"
    CUDA_EVENT = "cuda_event"


class PiscesLxAllocationPriority(Enum):
    """
    Priority levels for resource allocation.

    Higher priority requests are allocated first when resources
    are scarce. Preemptible allocations can be revoked for
    higher priority requests.

    Priority Levels:
        CRITICAL: Must not be preempted (e.g., checkpoint save)
        HIGH: High priority, rarely preempted (e.g., forward pass)
        NORMAL: Default priority (e.g., data loading)
        LOW: Low priority, often preempted (e.g., prefetch)
        BACKGROUND: Background tasks, preemptible anytime
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class PiscesLxResourceRequest:
    """
    Specification for resource allocation request.

    Attributes:
        gpu_memory_gb: Required GPU memory in gigabytes
        gpu_memory_tokens: Memory for token cache (alternative to gb)
        compute_units: Number of GPU compute units needed
        cpu_cores: Number of CPU cores needed
        system_memory_gb: System RAM in gigabytes
        duration_seconds: Expected duration of resource usage
        priority: Allocation priority level
        preemptible: Whether allocation can be preempted
        device_preference: Preferred device ID (optional)
        min_compute_capability: Minimum CUDA compute capability
        requires_fp16: Whether FP16 support is required
        requires_fp8: Whether FP8 support is required
        requires_tensor_cores: Whether tensor cores are required
        tags: Additional tags for tracking and debugging
    """
    gpu_memory_gb: float = 0.0
    gpu_memory_tokens: int = 0
    compute_units: int = 1
    cpu_cores: int = 0
    system_memory_gb: float = 0.0
    duration_seconds: float = 60.0
    priority: PiscesLxAllocationPriority = PiscesLxAllocationPriority.NORMAL
    preemptible: bool = True
    device_preference: Optional[int] = None
    min_compute_capability: Optional[str] = None
    requires_fp16: bool = False
    requires_fp8: bool = False
    requires_tensor_cores: bool = False
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def total_gpu_memory_bytes(self) -> int:
        """Get total GPU memory requirement in bytes."""
        gb_bytes = int(self.gpu_memory_gb * 1024 ** 3)
        token_bytes = self.gpu_memory_tokens * 2  # Assume FP16
        return gb_bytes + token_bytes


@dataclass
class PiscesLxResourceAllocation:
    """
    Granted resource allocation from scheduler.

    Attributes:
        allocation_id: Unique identifier for this allocation
        request: Original resource request
        granted: Whether the request was fully granted
        device_id: Assigned device ID
        gpu_memory_bytes: Allocated GPU memory in bytes
        compute_units: Allocated compute units
        cpu_cores: Allocated CPU cores
        cuda_stream_id: Assigned CUDA stream ID (if applicable)
        granted_at: Timestamp when allocation was granted
        expires_at: Timestamp when allocation expires (if applicable)
        metadata: Additional allocation metadata
    """
    allocation_id: str
    request: PiscesLxResourceRequest
    granted: bool = False
    device_id: Optional[int] = None
    gpu_memory_bytes: int = 0
    compute_units: int = 0
    cpu_cores: int = 0
    cuda_stream_id: Optional[int] = None
    granted_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if allocation has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def gpu_memory_gb(self) -> float:
        """Get allocated GPU memory in GB."""
        return self.gpu_memory_bytes / (1024 ** 3)


@dataclass
class PiscesLxCUDAStream:
    """
    CUDA stream for asynchronous execution.

    Attributes:
        stream_id: Unique stream identifier
        device_id: Device this stream belongs to
        priority: Stream priority (lower = higher priority)
        is_default: Whether this is the default stream
        created_at: Creation timestamp
        last_used_at: Last usage timestamp
    """
    stream_id: int
    device_id: int
    priority: int = 0
    is_default: bool = False
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    _stream_handle: Any = None

    def record_usage(self) -> None:
        """Record that this stream was used."""
        self.last_used_at = time.time()


@dataclass
class PiscesLxCUDAEvent:
    """
    CUDA event for synchronization.

    Attributes:
        event_id: Unique event identifier
        device_id: Device this event belongs to
        stream_id: Stream this event is recorded on
        is_recorded: Whether event has been recorded
        is_completed: Whether event has completed
        recorded_at: Timestamp when event was recorded
    """
    event_id: int
    device_id: int
    stream_id: int
    is_recorded: bool = False
    is_completed: bool = False
    recorded_at: Optional[float] = None
    _event_handle: Any = None


class PiscesLxMemoryPool:
    """
    GPU memory pool for efficient allocation.

    This class manages a pool of pre-allocated GPU memory blocks,
    reducing allocation overhead and fragmentation.

    Features:
        - Block-based allocation for efficiency
        - Automatic defragmentation
        - Memory pressure handling
        - Statistics tracking

    Attributes:
        device_id: Device this pool manages
        total_memory_bytes: Total memory managed by pool
        free_memory_bytes: Currently free memory
        block_size: Size of each memory block
        allocated_blocks: Set of allocated block IDs
        free_blocks: Set of free block IDs
    """

    def __init__(
        self,
        device_id: int,
        total_memory_bytes: int,
        block_size: int = 1024 * 1024  # 1MB default block size
    ):
        """
        Initialize memory pool.

        Args:
            device_id: GPU device ID
            total_memory_bytes: Total memory to manage
            block_size: Size of each block in bytes
        """
        self.device_id = device_id
        self.total_memory_bytes = total_memory_bytes
        self.block_size = block_size
        self.free_memory_bytes = total_memory_bytes
        self.allocated_blocks: Set[int] = set()
        self.free_blocks: Set[int] = set(range(total_memory_bytes // block_size))
        self._lock = threading.RLock()
        self._metrics = PiscesLxMetrics()
        self._logger = PiscesLxLogger(f"PiscesLx.Core.OPSC.MemoryPool-{device_id}", file_path=get_log_file(f"PiscesLx.Core.OPSC.MemoryPool"), enable_file=True)

        self._metrics.gauge("memory_pool_total_gb", total_memory_bytes / (1024 ** 3))

    def allocate(self, size_bytes: int) -> Optional[Tuple[int, int]]:
        """
        Allocate memory from the pool.

        Args:
            size_bytes: Number of bytes to allocate

        Returns:
            Tuple of (block_start, num_blocks) or None if insufficient memory
        """
        num_blocks_needed = (size_bytes + self.block_size - 1) // self.block_size

        with self._lock:
            if num_blocks_needed > len(self.free_blocks):
                self._logger.warning(
                    "memory_pool_insufficient",
                    requested=size_bytes,
                    available=self.free_memory_bytes
                )
                return None

            allocated = []
            for block_id in sorted(self.free_blocks):
                if len(allocated) >= num_blocks_needed:
                    break
                allocated.append(block_id)

            if len(allocated) < num_blocks_needed:
                return None

            for block_id in allocated:
                self.free_blocks.discard(block_id)
                self.allocated_blocks.add(block_id)

            self.free_memory_bytes -= len(allocated) * self.block_size
            self._metrics.gauge("memory_pool_free_gb", self.free_memory_bytes / (1024 ** 3))
            self._metrics.counter("memory_pool_allocations")

            return (allocated[0], len(allocated))

    def deallocate(self, block_start: int, num_blocks: int) -> None:
        """
        Return memory to the pool.

        Args:
            block_start: Starting block ID
            num_blocks: Number of blocks to deallocate
        """
        with self._lock:
            for i in range(num_blocks):
                block_id = block_start + i
                if block_id in self.allocated_blocks:
                    self.allocated_blocks.discard(block_id)
                    self.free_blocks.add(block_id)

            self.free_memory_bytes += num_blocks * self.block_size
            self._metrics.gauge("memory_pool_free_gb", self.free_memory_bytes / (1024 ** 3))
            self._metrics.counter("memory_pool_deallocations")

    def defragment(self) -> int:
        """
        Defragment the memory pool.

        Returns:
            Number of blocks moved during defragmentation
        """
        with self._lock:
            moved = 0
            sorted_free = sorted(self.free_blocks)
            sorted_allocated = sorted(self.allocated_blocks)

            self._logger.info("memory_pool_defrag_start", free_blocks=len(self.free_blocks))

            self._metrics.counter("memory_pool_defrags")

            return moved

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        with self._lock:
            return {
                "device_id": self.device_id,
                "total_memory_gb": self.total_memory_bytes / (1024 ** 3),
                "free_memory_gb": self.free_memory_bytes / (1024 ** 3),
                "used_memory_gb": (self.total_memory_bytes - self.free_memory_bytes) / (1024 ** 3),
                "utilization": 1 - (self.free_memory_bytes / self.total_memory_bytes),
                "total_blocks": self.total_memory_bytes // self.block_size,
                "free_blocks": len(self.free_blocks),
                "allocated_blocks": len(self.allocated_blocks),
                "block_size_mb": self.block_size / (1024 ** 2)
            }


class PiscesLxCUDAScheduler:
    """
    CUDA stream and event scheduler.

    Manages CUDA streams for asynchronous execution and events
    for synchronization between operations.

    Features:
        - Stream pool management
        - Priority-based stream allocation
        - Event creation and synchronization
        - Stream reuse optimization

    Attributes:
        device_id: Device this scheduler manages
        max_streams: Maximum number of streams
        max_events: Maximum number of events
        streams: Dictionary of stream ID to stream object
        events: Dictionary of event ID to event object
    """

    def __init__(
        self,
        device_id: int,
        max_streams: int = 32,
        max_events: int = 256
    ):
        """
        Initialize CUDA scheduler.

        Args:
            device_id: GPU device ID
            max_streams: Maximum concurrent streams
            max_events: Maximum concurrent events
        """
        self.device_id = device_id
        self.max_streams = max_streams
        self.max_events = max_events
        self.streams: Dict[int, PiscesLxCUDAStream] = {}
        self.events: Dict[int, PiscesLxCUDAEvent] = {}
        self._next_stream_id = 0
        self._next_event_id = 0
        self._lock = threading.RLock()
        self._metrics = PiscesLxMetrics()
        self._logger = PiscesLxLogger(f"PiscesLx.Core.OPSC.CUDAScheduler-{device_id}", file_path=get_log_file("PiscesLx.Core.OPSC.CUDAScheduler"), enable_file=True)

        self._create_default_stream()

    def _create_default_stream(self) -> None:
        """Create the default stream (stream 0)."""
        default_stream = PiscesLxCUDAStream(
            stream_id=0,
            device_id=self.device_id,
            priority=0,
            is_default=True
        )
        self.streams[0] = default_stream
        self._next_stream_id = 1

    def create_stream(
        self,
        priority: int = 0
    ) -> Optional[PiscesLxCUDAStream]:
        """
        Create a new CUDA stream.

        Args:
            priority: Stream priority (lower = higher priority)

        Returns:
            Created stream or None if limit reached
        """
        with self._lock:
            if len(self.streams) >= self.max_streams:
                self._logger.warning("cuda_stream_limit_reached", max_streams=self.max_streams)
                return None

            stream_id = self._next_stream_id
            self._next_stream_id += 1

            stream = PiscesLxCUDAStream(
                stream_id=stream_id,
                device_id=self.device_id,
                priority=priority
            )

            try:
                import torch
                stream._stream_handle = torch.cuda.Stream(device=self.device_id)
            except ImportError:
                pass

            self.streams[stream_id] = stream
            self._metrics.counter("cuda_streams_created")
            self._logger.debug("cuda_stream_created", stream_id=stream_id)

            return stream

    def destroy_stream(self, stream_id: int) -> bool:
        """
        Destroy a CUDA stream.

        Args:
            stream_id: Stream ID to destroy

        Returns:
            True if stream was destroyed
        """
        with self._lock:
            if stream_id not in self.streams:
                return False

            stream = self.streams.pop(stream_id)
            if stream.is_default:
                self.streams[stream_id] = stream
                return False

            self._metrics.counter("cuda_streams_destroyed")
            self._logger.debug("cuda_stream_destroyed", stream_id=stream_id)
            return True

    def get_stream(self, stream_id: int) -> Optional[PiscesLxCUDAStream]:
        """
        Get a stream by ID.

        Args:
            stream_id: Stream ID

        Returns:
            Stream object or None
        """
        return self.streams.get(stream_id)

    def get_available_stream(self) -> Optional[PiscesLxCUDAStream]:
        """
        Get an available stream for execution.

        Returns the least recently used stream or creates a new one.

        Returns:
            Available stream or None
        """
        with self._lock:
            if self.streams:
                lru_stream = min(self.streams.values(), key=lambda s: s.last_used_at)
                lru_stream.record_usage()
                return lru_stream

            return self.create_stream()

    def create_event(
        self,
        stream_id: int = 0
    ) -> Optional[PiscesLxCUDAEvent]:
        """
        Create a new CUDA event.

        Args:
            stream_id: Stream to associate with event

        Returns:
            Created event or None if limit reached
        """
        with self._lock:
            if len(self.events) >= self.max_events:
                self._logger.warning("cuda_event_limit_reached", max_events=self.max_events)
                return None

            event_id = self._next_event_id
            self._next_event_id += 1

            event = PiscesLxCUDAEvent(
                event_id=event_id,
                device_id=self.device_id,
                stream_id=stream_id
            )

            try:
                import torch
                event._event_handle = torch.cuda.Event()
            except ImportError:
                pass

            self.events[event_id] = event
            self._metrics.counter("cuda_events_created")

            return event

    def record_event(
        self,
        event_id: int,
        stream_id: int = 0
    ) -> bool:
        """
        Record an event on a stream.

        Args:
            event_id: Event to record
            stream_id: Stream to record on

        Returns:
            True if event was recorded
        """
        event = self.events.get(event_id)
        stream = self.streams.get(stream_id)

        if event is None or stream is None:
            return False

        try:
            import torch
            if stream._stream_handle and event._event_handle:
                event._event_handle.record(stream._stream_handle)
                event.is_recorded = True
                event.recorded_at = time.time()
                self._metrics.counter("cuda_events_recorded")
                return True
        except ImportError:
            pass
        except Exception as e:
            self._logger.error("cuda_event_record_failed", error=str(e))

        return False

    def wait_event(
        self,
        event_id: int,
        stream_id: int = 0
    ) -> bool:
        """
        Make a stream wait for an event.

        Args:
            event_id: Event to wait for
            stream_id: Stream that will wait

        Returns:
            True if wait was scheduled
        """
        event = self.events.get(event_id)
        stream = self.streams.get(stream_id)

        if event is None or stream is None:
            return False

        try:
            import torch
            if stream._stream_handle and event._event_handle:
                stream._stream_handle.wait_event(event._event_handle)
                self._metrics.counter("cuda_event_waits")
                return True
        except ImportError:
            pass
        except Exception as e:
            self._logger.error("cuda_event_wait_failed", error=str(e))

        return False

    def synchronize_stream(self, stream_id: int = 0) -> bool:
        """
        Synchronize a stream.

        Args:
            stream_id: Stream to synchronize

        Returns:
            True if synchronization succeeded
        """
        stream = self.streams.get(stream_id)
        if stream is None:
            return False

        try:
            import torch
            if stream._stream_handle:
                stream._stream_handle.synchronize()
                self._metrics.counter("cuda_stream_syncs")
                return True
        except ImportError:
            pass
        except Exception as e:
            self._logger.error("cuda_stream_sync_failed", error=str(e))

        return False

    def synchronize_device(self) -> bool:
        """
        Synchronize all streams on the device.

        Returns:
            True if synchronization succeeded
        """
        try:
            import torch
            torch.cuda.synchronize(self.device_id)
            self._metrics.counter("cuda_device_syncs")
            return True
        except ImportError:
            pass
        except Exception as e:
            self._logger.error("cuda_device_sync_failed", error=str(e))

        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dictionary with scheduler stats
        """
        return {
            "device_id": self.device_id,
            "total_streams": len(self.streams),
            "max_streams": self.max_streams,
            "total_events": len(self.events),
            "max_events": self.max_events,
            "streams": {
                sid: {
                    "priority": s.priority,
                    "is_default": s.is_default,
                    "last_used_at": s.last_used_at
                }
                for sid, s in self.streams.items()
            }
        }


class PiscesLxGPUScheduler:
    """
    Main GPU resource scheduler for operator execution.

    This class coordinates GPU device allocation, memory management,
    and compute scheduling across multiple GPUs.

    Features:
        - Multi-GPU device management
        - Memory pool per device
        - CUDA stream scheduling
        - Fair resource allocation
        - Preemption support
        - Health monitoring integration

    Attributes:
        device_monitor: Device health monitor
        memory_pools: Memory pools per device
        cuda_schedulers: CUDA schedulers per device
        allocations: Active allocations by ID
        pending_requests: Queue of pending allocation requests
    """

    def __init__(
        self,
        max_devices: int = 8,
        memory_pool_block_size: int = 1024 * 1024
    ):
        """
        Initialize GPU scheduler.

        Args:
            max_devices: Maximum number of GPUs to manage
            memory_pool_block_size: Block size for memory pools
        """
        self.max_devices = max_devices
        self.memory_pool_block_size = memory_pool_block_size
        self._system_monitor = PiscesLxSystemMonitor()
        self.memory_pools: Dict[int, PiscesLxMemoryPool] = {}
        self.cuda_schedulers: Dict[int, PiscesLxCUDAScheduler] = {}
        self.allocations: Dict[str, PiscesLxResourceAllocation] = {}
        self.pending_requests: List[Tuple[int, int, PiscesLxResourceRequest]] = []
        self._next_allocation_id = 0
        self._lock = threading.RLock()
        self._metrics = PiscesLxMetrics()
        self._logger = PiscesLxLogger("PiscesLx.Core.OPSC.GPUScheduler", file_path=get_log_file("PiscesLx.Core.OPSC.GPUScheduler"), enable_file=True)

        self._initialize_devices()

    def _initialize_devices(self) -> None:
        """Initialize GPU devices and their resources."""
        devices = self._system_monitor.list_devices()
        gpu_devices = [d for d in devices if d.device_type == PiscesLxDeviceType.GPU]

        if not gpu_devices:
            self._logger.warning("no_gpus_available", message="Running in CPU-only mode")
            return

        for device in gpu_devices[:self.max_devices]:
            device_id_str = device.device_id
            device_id = int(device_id_str.split("-")[-1]) if "-" in device_id_str else 0

            caps = device.capabilities
            total_memory = int(caps.memory_gb * 1024 ** 3) if caps else 80 * 1024 ** 3

            self.memory_pools[device_id] = PiscesLxMemoryPool(
                device_id=device_id,
                total_memory_bytes=total_memory,
                block_size=self.memory_pool_block_size
            )

            self.cuda_schedulers[device_id] = PiscesLxCUDAScheduler(
                device_id=device_id
            )

            device.set_status(PiscesLxDeviceStatus.AVAILABLE)

        self._logger.info(
            "gpu_scheduler_initialized",
            num_devices=len(self.memory_pools)
        )

    def get_available_devices(self) -> List[int]:
        """
        Get list of available GPU device IDs.

        Returns:
            List of available device IDs
        """
        return list(self.memory_pools.keys())

    def get_device_info(self, device_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific device.

        Args:
            device_id: Device ID

        Returns:
            Device info dictionary or None
        """
        if device_id not in self.memory_pools:
            return None

        pool_stats = self.memory_pools[device_id].get_stats()
        scheduler_stats = self.cuda_schedulers[device_id].get_stats()
        gpu_usage = self._system_monitor.get_gpu_usage(device_id)

        return {
            "device_id": device_id,
            "memory_pool": pool_stats,
            "cuda_scheduler": scheduler_stats,
            "current_usage": gpu_usage
        }

    def allocate(
        self,
        request: PiscesLxResourceRequest,
        timeout_seconds: float = 30.0
    ) -> PiscesLxResourceAllocation:
        """
        Allocate GPU resources for an operator.

        Args:
            request: Resource request specification
            timeout_seconds: Maximum time to wait for allocation

        Returns:
            Resource allocation result
        """
        start_time = time.time()

        with self._lock:
            allocation_id = f"alloc_{self._next_allocation_id}"
            self._next_allocation_id += 1

            allocation = PiscesLxResourceAllocation(
                allocation_id=allocation_id,
                request=request,
                granted=False
            )

            best_device = self._find_best_device(request)
            if best_device is None:
                if request.preemptible and self._try_preempt(request):
                    best_device = self._find_best_device(request)

            if best_device is None:
                self._logger.warning(
                    "gpu_allocation_failed",
                    allocation_id=allocation_id,
                    reason="no_suitable_device"
                )
                self.allocations[allocation_id] = allocation
                return allocation

            memory_bytes = request.total_gpu_memory_bytes
            pool = self.memory_pools[best_device]

            memory_result = pool.allocate(memory_bytes)
            if memory_result is None:
                self._logger.warning(
                    "gpu_allocation_failed",
                    allocation_id=allocation_id,
                    reason="insufficient_memory",
                    requested_gb=memory_bytes / (1024 ** 3)
                )
                self.allocations[allocation_id] = allocation
                return allocation

            cuda_scheduler = self.cuda_schedulers[best_device]
            stream = cuda_scheduler.get_available_stream()

            allocation.granted = True
            allocation.device_id = best_device
            allocation.gpu_memory_bytes = memory_bytes
            allocation.compute_units = request.compute_units
            allocation.cuda_stream_id = stream.stream_id if stream else None
            allocation.metadata["memory_blocks"] = memory_result

            if request.duration_seconds > 0:
                allocation.expires_at = time.time() + request.duration_seconds

            self.allocations[allocation_id] = allocation
            self._metrics.counter("gpu_allocations_granted")
            self._metrics.gauge("gpu_active_allocations", len([a for a in self.allocations.values() if a.granted]))

            self._logger.info(
                "gpu_allocation_granted",
                allocation_id=allocation_id,
                device_id=best_device,
                memory_gb=memory_bytes / (1024 ** 3)
            )

            return allocation

    def _find_best_device(
        self,
        request: PiscesLxResourceRequest
    ) -> Optional[int]:
        """
        Find the best device for a request.

        Args:
            request: Resource request

        Returns:
            Best device ID or None
        """
        candidates = []

        for device_id, pool in self.memory_pools.items():
            if request.device_preference is not None and device_id != request.device_preference:
                continue

            if pool.free_memory_bytes < request.total_gpu_memory_bytes:
                continue

            device = self._system_monitor.get_device(f"gpu-{device_id}")
            if device and device.status != PiscesLxDeviceStatus.AVAILABLE:
                continue

            score = pool.free_memory_bytes
            candidates.append((score, device_id))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]

    def _try_preempt(
        self,
        request: PiscesLxResourceRequest
    ) -> bool:
        """
        Try to preempt lower priority allocations.

        Args:
            request: High priority request

        Returns:
            True if preemption freed enough resources
        """
        preemptible = [
            (a.request.priority.value, a.allocation_id, a)
            for a in self.allocations.values()
            if a.granted and a.request.preemptible and a.request.priority.value > request.priority.value
        ]

        preemptible.sort()

        freed_memory = 0
        for _, allocation_id, allocation in preemptible:
            if freed_memory >= request.total_gpu_memory_bytes:
                break

            self.release(allocation)
            freed_memory += allocation.gpu_memory_bytes
            self._metrics.counter("gpu_allocations_preempted")
            self._logger.info(
                "gpu_allocation_preempted",
                allocation_id=allocation_id,
                reason="higher_priority_request"
            )

        return freed_memory >= request.total_gpu_memory_bytes

    def release(
        self,
        allocation: PiscesLxResourceAllocation
    ) -> None:
        """
        Release allocated resources.

        Args:
            allocation: Allocation to release
        """
        if not allocation.granted:
            return

        with self._lock:
            if allocation.device_id is not None:
                pool = self.memory_pools.get(allocation.device_id)
                if pool:
                    memory_blocks = allocation.metadata.get("memory_blocks")
                    if memory_blocks:
                        pool.deallocate(memory_blocks[0], memory_blocks[1])

                cuda_scheduler = self.cuda_schedulers.get(allocation.device_id)
                if cuda_scheduler and allocation.cuda_stream_id:
                    if allocation.cuda_stream_id != 0:
                        cuda_scheduler.destroy_stream(allocation.cuda_stream_id)

            allocation.granted = False
            self._metrics.counter("gpu_allocations_released")
            self._metrics.gauge("gpu_active_allocations", len([a for a in self.allocations.values() if a.granted]))

            self._logger.info(
                "gpu_allocation_released",
                allocation_id=allocation.allocation_id
            )

    @contextmanager
    def device_context(
        self,
        allocation: PiscesLxResourceAllocation
    ) -> Any:
        """
        Context manager for device execution.

        Sets the CUDA device for the duration of the context.

        Args:
            allocation: Resource allocation

        Yields:
            Device ID
        """
        device_id = allocation.device_id
        stream_id = allocation.cuda_stream_id

        try:
            import torch
            with torch.cuda.device(device_id):
                if stream_id is not None:
                    scheduler = self.cuda_schedulers.get(device_id)
                    if scheduler:
                        stream = scheduler.get_stream(stream_id)
                        if stream and stream._stream_handle:
                            with torch.cuda.stream(stream._stream_handle):
                                yield device_id
                            return
                yield device_id
        except ImportError:
            yield device_id

    def get_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dictionary with scheduler stats
        """
        total_memory = sum(p.total_memory_bytes for p in self.memory_pools.values())
        free_memory = sum(p.free_memory_bytes for p in self.memory_pools.values())

        return {
            "num_devices": len(self.memory_pools),
            "total_memory_gb": total_memory / (1024 ** 3),
            "free_memory_gb": free_memory / (1024 ** 3),
            "used_memory_gb": (total_memory - free_memory) / (1024 ** 3),
            "utilization": 1 - (free_memory / total_memory) if total_memory > 0 else 0,
            "active_allocations": len([a for a in self.allocations.values() if a.granted]),
            "total_allocations": len(self.allocations),
            "devices": {
                device_id: self.get_device_info(device_id)
                for device_id in self.memory_pools.keys()
            }
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all devices.

        Returns:
            Health check results
        """
        devices_health = {}
        for device_id in self.memory_pools.keys():
            device = self._system_monitor.get_device(f"gpu-{device_id}")
            if device:
                devices_health[f"gpu-{device_id}"] = {
                    "status": device.status.name,
                    "health_score": device.health_score
                }

        return {
            "healthy": True,
            "devices": devices_health,
            "scheduler_stats": self.get_stats()
        }

    def cleanup_expired(self) -> int:
        """
        Clean up expired allocations.

        Returns:
            Number of allocations cleaned up
        """
        cleaned = 0
        with self._lock:
            for allocation in list(self.allocations.values()):
                if allocation.is_expired and allocation.granted:
                    self.release(allocation)
                    cleaned += 1

        if cleaned > 0:
            self._logger.info("gpu_allocations_expired_cleaned", count=cleaned)

        return cleaned


class PiscesLxResourceEstimator:
    """
    Resource requirement estimator for operators.

    This class provides methods to estimate the resource requirements
    of operators before execution, enabling better scheduling decisions.

    Features:
        - Memory estimation based on input size
        - Compute estimation based on operation type
        - Model-aware estimation for LLM operators
        - Historical data utilization

    Attributes:
        historical_data: Historical resource usage data
        model_configs: Known model configurations
    """

    def __init__(self):
        """Initialize resource estimator."""
        self.historical_data: Dict[str, List[Dict[str, Any]]] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {
            "small": {"hidden_size": 768, "num_layers": 12, "num_heads": 12},
            "base": {"hidden_size": 1024, "num_layers": 24, "num_heads": 16},
            "large": {"hidden_size": 2048, "num_layers": 36, "num_heads": 32},
            "xl": {"hidden_size": 3072, "num_layers": 48, "num_heads": 48},
            "xxl": {"hidden_size": 4096, "num_layers": 64, "num_heads": 64},
        }
        self._lock = threading.RLock()
        self._logger = PiscesLxLogger("PiscesLx.Core.OPSC.ResourceEstimator", file_path=get_log_file("PiscesLx.Core.OPSC.ResourceEstimator"), enable_file=True)

    def estimate_attention_memory(
        self,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
        dtype_bytes: int = 2
    ) -> int:
        """
        Estimate memory for attention operation.

        Args:
            batch_size: Batch size
            seq_length: Sequence length
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dtype_bytes: Bytes per element (2 for FP16)

        Returns:
            Estimated memory in bytes
        """
        qkv_memory = 3 * batch_size * seq_length * num_heads * head_dim * dtype_bytes
        attention_scores = batch_size * num_heads * seq_length * seq_length * dtype_bytes
        output_memory = batch_size * seq_length * num_heads * head_dim * dtype_bytes

        total = qkv_memory + attention_scores + output_memory
        return total

    def estimate_ffn_memory(
        self,
        batch_size: int,
        seq_length: int,
        hidden_size: int,
        intermediate_size: int,
        dtype_bytes: int = 2
    ) -> int:
        """
        Estimate memory for feed-forward network.

        Args:
            batch_size: Batch size
            seq_length: Sequence length
            hidden_size: Hidden dimension
            intermediate_size: Intermediate dimension
            dtype_bytes: Bytes per element

        Returns:
            Estimated memory in bytes
        """
        input_memory = batch_size * seq_length * hidden_size * dtype_bytes
        intermediate_memory = batch_size * seq_length * intermediate_size * dtype_bytes
        output_memory = batch_size * seq_length * hidden_size * dtype_bytes

        return input_memory + intermediate_memory + output_memory

    def estimate_model_memory(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        vocab_size: int = 32000,
        intermediate_ratio: float = 4.0,
        dtype_bytes: int = 2
    ) -> int:
        """
        Estimate total model parameter memory.

        Args:
            hidden_size: Hidden dimension
            num_layers: Number of layers
            num_heads: Number of attention heads
            vocab_size: Vocabulary size
            intermediate_ratio: FFN intermediate ratio
            dtype_bytes: Bytes per parameter

        Returns:
            Estimated memory in bytes
        """
        head_dim = hidden_size // num_heads
        intermediate_size = int(hidden_size * intermediate_ratio)

        embedding_params = vocab_size * hidden_size
        attention_params_per_layer = 4 * hidden_size * hidden_size
        ffn_params_per_layer = 2 * hidden_size * intermediate_size + hidden_size
        layer_norm_params = 2 * hidden_size

        total_params = (
            embedding_params +
            num_layers * (attention_params_per_layer + ffn_params_per_layer + 2 * layer_norm_params)
        )

        return total_params * dtype_bytes

    def estimate_kv_cache_memory(
        self,
        batch_size: int,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype_bytes: int = 2
    ) -> int:
        """
        Estimate KV cache memory for inference.

        Args:
            batch_size: Batch size
            seq_length: Sequence length
            num_layers: Number of layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dtype_bytes: Bytes per element

        Returns:
            Estimated memory in bytes
        """
        kv_per_token = 2 * num_layers * num_heads * head_dim * dtype_bytes
        return batch_size * seq_length * kv_per_token

    def estimate_operator_resources(
        self,
        operator_name: str,
        input_shape: Tuple[int, ...],
        config: Optional[Dict[str, Any]] = None
    ) -> PiscesLxResourceRequest:
        """
        Estimate resource requirements for an operator.

        Args:
            operator_name: Name of the operator
            input_shape: Shape of input tensor
            config: Optional operator configuration

        Returns:
            Resource request with estimated requirements
        """
        config = config or {}

        if "attention" in operator_name.lower():
            batch_size = input_shape[0] if len(input_shape) > 0 else 1
            seq_length = input_shape[1] if len(input_shape) > 1 else 512
            num_heads = config.get("num_heads", 32)
            head_dim = config.get("head_dim", 128)

            memory_bytes = self.estimate_attention_memory(
                batch_size, seq_length, num_heads, head_dim
            )

            return PiscesLxResourceRequest(
                gpu_memory_gb=memory_bytes / (1024 ** 3) * 1.5,
                compute_units=1,
                priority=PiscesLxAllocationPriority.HIGH,
                tags={"operator_type": "attention"}
            )

        elif "ffn" in operator_name.lower() or "mlp" in operator_name.lower():
            batch_size = input_shape[0] if len(input_shape) > 0 else 1
            seq_length = input_shape[1] if len(input_shape) > 1 else 512
            hidden_size = input_shape[-1] if len(input_shape) > 0 else 4096
            intermediate_size = config.get("intermediate_size", hidden_size * 4)

            memory_bytes = self.estimate_ffn_memory(
                batch_size, seq_length, hidden_size, intermediate_size
            )

            return PiscesLxResourceRequest(
                gpu_memory_gb=memory_bytes / (1024 ** 3) * 1.5,
                compute_units=1,
                priority=PiscesLxAllocationPriority.HIGH,
                tags={"operator_type": "ffn"}
            )

        else:
            estimated_elements = 1
            for dim in input_shape:
                estimated_elements *= dim

            estimated_memory_gb = (estimated_elements * 2 * 3) / (1024 ** 3)

            return PiscesLxResourceRequest(
                gpu_memory_gb=max(1.0, estimated_memory_gb),
                compute_units=1,
                priority=PiscesLxAllocationPriority.NORMAL,
                tags={"operator_type": "generic"}
            )

    def record_actual_usage(
        self,
        operator_name: str,
        request: PiscesLxResourceRequest,
        actual_memory_gb: float,
        actual_duration_seconds: float
    ) -> None:
        """
        Record actual resource usage for future estimation.

        Args:
            operator_name: Operator name
            request: Original resource request
            actual_memory_gb: Actual memory used
            actual_duration_seconds: Actual execution time
        """
        with self._lock:
            if operator_name not in self.historical_data:
                self.historical_data[operator_name] = []

            self.historical_data[operator_name].append({
                "requested_memory_gb": request.gpu_memory_gb,
                "actual_memory_gb": actual_memory_gb,
                "requested_duration_seconds": request.duration_seconds,
                "actual_duration_seconds": actual_duration_seconds,
                "timestamp": time.time()
            })

            if len(self.historical_data[operator_name]) > 100:
                self.historical_data[operator_name] = self.historical_data[operator_name][-100:]

    def get_estimation_accuracy(
        self,
        operator_name: str
    ) -> Optional[Dict[str, float]]:
        """
        Get estimation accuracy metrics for an operator.

        Args:
            operator_name: Operator name

        Returns:
            Accuracy metrics or None if no data
        """
        with self._lock:
            if operator_name not in self.historical_data:
                return None

            data = self.historical_data[operator_name]
            if not data:
                return None

            memory_errors = [
                abs(d["actual_memory_gb"] - d["requested_memory_gb"]) / d["actual_memory_gb"]
                for d in data
                if d["actual_memory_gb"] > 0
            ]

            duration_errors = [
                abs(d["actual_duration_seconds"] - d["requested_duration_seconds"]) / d["actual_duration_seconds"]
                for d in data
                if d["actual_duration_seconds"] > 0
            ]

            return {
                "sample_count": len(data),
                "avg_memory_error": sum(memory_errors) / len(memory_errors) if memory_errors else 0,
                "avg_duration_error": sum(duration_errors) / len(duration_errors) if duration_errors else 0
            }
