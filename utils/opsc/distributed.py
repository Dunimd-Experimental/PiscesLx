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
Distributed Operator Base Classes

This module provides base classes for distributed operators supporting various
parallelism strategies essential for large-scale LLM training and inference.

Parallelism Strategies:
    - Tensor Parallelism (TP): Split tensors across devices for large model layers
    - Pipeline Parallelism (PP): Split model stages across devices for depth scaling
    - Expert Parallelism (EP): Distribute MoE experts across devices
    - Data Parallelism (DP): Replicate model across devices for batch scaling

Key Components:
    - PiscesLxParallelismType: Enumeration of parallelism strategies
    - PiscesLxParallelConfig: Configuration for parallel execution
    - PiscesLxDistributedOperator: Base class for distributed operators
    - PiscesLxTensorParallelOperator: TP-specific operator base
    - PiscesLxPipelineParallelOperator: PP-specific operator base
    - PiscesLxExpertParallelOperator: EP-specific operator base
    - PiscesLxDataParallelOperator: DP-specific operator base

Design Principles:
    1. Strategy Pattern: Each parallelism type has dedicated operator base
    2. Composition: Operators can combine multiple parallelism strategies
    3. Fault Tolerance: Built-in support for node failures and recovery
    4. Communication Optimization: Overlap computation with communication

Usage:
    class MyTPOperator(PiscesLxTensorParallelOperator):
        @property
        def name(self) -> str:
            return "my_tp_operator"

        def forward_tp(self, inputs, tp_group) -> Any:
            return self.process_with_tp(inputs, tp_group)
"""

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from .interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig
)
from ..dc import (
    PiscesLxLogger, PiscesLxMetrics, PiscesLxTracing,
    PiscesLxDevice, PiscesLxDeviceType, PiscesLxDeviceStatus, PiscesLxSystemMonitor
)


class PiscesLxParallelismType(Enum):
    """
    Enumeration of distributed parallelism strategies.

    Each strategy addresses different scaling challenges in LLM training:

    Tensor Parallelism (TP):
        - Splits individual tensor operations across devices
        - Best for: Large layers that don't fit on single device
        - Communication: All-reduce after each layer
        - Example: Split attention heads across GPUs

    Pipeline Parallelism (PP):
        - Splits model stages across devices
        - Best for: Deep models with sequential stages
        - Communication: Point-to-point between stages
        - Example: Encoder on GPU 0, Decoder on GPU 1

    Expert Parallelism (EP):
        - Distributes MoE experts across devices
        - Best for: Mixture-of-Experts models
        - Communication: All-to-all for expert routing
        - Example: 8 experts across 4 GPUs (2 per GPU)

    Data Parallelism (DP):
        - Replicates model across devices
        - Best for: Large batch training
        - Communication: All-reduce for gradients
        - Example: 4 replicas processing different batches
    """
    TENSOR = "tensor_parallel"
    PIPELINE = "pipeline_parallel"
    EXPERT = "expert_parallel"
    DATA = "data_parallel"
    HYBRID = "hybrid"


class PiscesLxCommunicationType(Enum):
    """
    Enumeration of collective communication operations.

    Communication Types:
        ALL_REDUCE: Sum/average across all ranks, result on all
        ALL_GATHER: Gather data from all ranks, result on all
        REDUCE_SCATTER: Reduce then scatter result across ranks
        ALL_TO_ALL: Each rank sends different data to each other rank
        BROADCAST: One rank sends to all others
        POINT_TO_POINT: Direct send/recv between two ranks
    """
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"
    POINT_TO_POINT = "p2p"


@dataclass
class PiscesLxParallelConfig:
    """
    Configuration for distributed parallel execution.

    This configuration defines how an operator should be distributed
    across multiple devices or nodes.

    Attributes:
        world_size: Total number of participating processes
        rank: Current process rank (0 to world_size-1)
        local_rank: Rank within the current node
        tp_degree: Tensor parallelism degree (devices per TP group)
        pp_degree: Pipeline parallelism degree (number of stages)
        ep_degree: Expert parallelism degree (devices per EP group)
        dp_degree: Data parallelism degree (number of replicas)
        backend: Communication backend ("nccl", "gloo", "nccl+gloo")
        timeout_seconds: Timeout for collective operations
        enable_overlap: Enable compute-communication overlap
        gradient_checkpointing: Enable gradient checkpointing for memory
    """
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    tp_degree: int = 1
    pp_degree: int = 1
    ep_degree: int = 1
    dp_degree: int = 1
    backend: str = "nccl"
    timeout_seconds: float = 300.0
    enable_overlap: bool = True
    gradient_checkpointing: bool = False

    @property
    def is_distributed(self) -> bool:
        """Check if this is a distributed configuration."""
        return self.world_size > 1

    @property
    def tp_rank(self) -> int:
        """Get tensor parallel rank within TP group."""
        return self.rank % self.tp_degree

    @property
    def pp_rank(self) -> int:
        """Get pipeline parallel rank (stage index)."""
        return (self.rank // (self.tp_degree * self.dp_degree)) % self.pp_degree

    @property
    def dp_rank(self) -> int:
        """Get data parallel rank within DP group."""
        return self.rank // (self.tp_degree * self.pp_degree)

    @property
    def ep_rank(self) -> int:
        """Get expert parallel rank within EP group."""
        return self.rank % self.ep_degree

    def get_tp_group(self) -> List[int]:
        """Get ranks in the same tensor parallel group."""
        pp_idx = self.rank // (self.tp_degree * self.dp_degree)
        dp_idx = self.rank // (self.tp_degree * self.pp_degree * self.dp_degree)
        base = pp_idx * self.tp_degree * self.dp_degree + dp_idx * self.tp_degree
        return list(range(base, base + self.tp_degree))

    def get_pp_group(self) -> List[int]:
        """Get ranks in the same pipeline parallel group."""
        return list(range(0, self.world_size, self.tp_degree * self.dp_degree))

    def get_dp_group(self) -> List[int]:
        """Get ranks in the same data parallel group."""
        group_size = self.tp_degree * self.pp_degree
        base = (self.rank // group_size) * group_size
        return [base + i for i in range(group_size)]

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate configuration consistency."""
        total = self.tp_degree * self.pp_degree * self.dp_degree
        if total != self.world_size:
            return False, f"TP({self.tp_degree}) * PP({self.pp_degree}) * DP({self.dp_degree}) = {total} != world_size({self.world_size})"
        if self.rank < 0 or self.rank >= self.world_size:
            return False, f"Rank {self.rank} out of range [0, {self.world_size})"
        return True, None


@dataclass
class PiscesLxCommunicationStats:
    """
    Statistics for collective communication operations.

    Tracks timing, volume, and efficiency metrics for distributed
    communication operations.

    Attributes:
        comm_type: Type of communication operation
        bytes_transferred: Total bytes transferred
        latency_ms: Communication latency in milliseconds
        bandwidth_gbps: Achieved bandwidth in Gbps
        overlap_ratio: Ratio of overlapped compute time
    """
    comm_type: PiscesLxCommunicationType
    bytes_transferred: int = 0
    latency_ms: float = 0.0
    bandwidth_gbps: float = 0.0
    overlap_ratio: float = 0.0
    timestamp: float = field(default_factory=time.time)


class PiscesLxDistributedOperator(PiscesLxOperatorInterface, ABC):
    """
    Abstract base class for distributed operators.

    This class extends PiscesLxOperatorInterface with distributed computing
    capabilities, including process groups, collective communication,
    and fault tolerance.

    Attributes:
        parallel_config: Configuration for parallel execution
        comm_stats: Communication statistics tracker
        device_monitor: Device health monitoring
        metrics: Performance metrics collector

    Lifecycle:
        1. Initialize with parallel configuration
        2. setup() initializes process groups and communication
        3. execute() runs distributed computation
        4. teardown() releases distributed resources

    Thread Safety:
        - Communication operations are thread-safe via backend
        - Metrics collection uses thread-safe data structures
    """

    def __init__(
        self,
        config: Optional[PiscesLxOperatorConfig] = None,
        parallel_config: Optional[PiscesLxParallelConfig] = None
    ):
        """
        Initialize distributed operator.

        Args:
            config: Base operator configuration
            parallel_config: Distributed execution configuration
        """
        super().__init__(config)
        self.parallel_config = parallel_config or PiscesLxParallelConfig()
        self.comm_stats: List[PiscesLxCommunicationStats] = []
        self.system_monitor = PiscesLxSystemMonitor()
        self.metrics = PiscesLxMetrics()
        self._comm_lock = threading.RLock()
        self._process_group = None
        self._is_initialized = False

    @property
    @abstractmethod
    def parallelism_type(self) -> PiscesLxParallelismType:
        """
        Get the parallelism strategy type for this operator.

        Returns:
            PiscesLxParallelismType enum value
        """
        raise NotImplementedError("Distributed operator must implement parallelism_type")

    @property
    def is_master(self) -> bool:
        """Check if this is the master rank (rank 0)."""
        return self.parallel_config.rank == 0

    @property
    def world_size(self) -> int:
        """Get total number of participating processes."""
        return self.parallel_config.world_size

    @property
    def rank(self) -> int:
        """Get current process rank."""
        return self.parallel_config.rank

    def setup(self) -> None:
        """
        Initialize distributed resources.

        Sets up process groups, communication backends, and device monitoring.
        Must be called before execute().
        """
        if self._is_setup:
            return

        super().setup()

        if self.parallel_config.is_distributed:
            self._initialize_process_group()
            self._setup_device_monitoring()

        self._is_initialized = True
        self.logger.info(
            "distributed_setup",
            parallelism=self.parallelism_type.value,
            world_size=self.world_size,
            rank=self.rank
        )

    def _initialize_process_group(self) -> None:
        """
        Initialize the distributed process group.

        Creates process groups for the configured parallelism strategy.
        Override this method to customize group creation.
        """
        try:
            import torch
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.parallel_config.backend,
                    timeout=self.parallel_config.timeout_seconds
                )

            self._process_group = dist.group.WORLD
            self.logger.info("process_group_initialized", backend=self.parallel_config.backend)

        except ImportError:
            self.logger.warning("torch_not_available", message="Running in simulation mode")
            self._process_group = None
        except Exception as e:
            self.logger.error("process_group_init_failed", error=str(e))
            self._process_group = None

    def _setup_device_monitoring(self) -> None:
        """
        Set up device health monitoring for distributed execution.

        Registers local devices with the monitor for health tracking.
        """
        device_id = f"gpu-{self.parallel_config.local_rank}"
        device = PiscesLxDevice(
            device_id=device_id,
            device_type=PiscesLxDeviceType.GPU,
            name=f"GPU-{self.parallel_config.local_rank}"
        )
        device.set_status(PiscesLxDeviceStatus.AVAILABLE)

    def all_reduce(
        self,
        tensor: Any,
        average: bool = True,
        async_op: bool = False
    ) -> Any:
        """
        Perform all-reduce collective operation.

        Sums or averages a tensor across all ranks, with the result
        available on all ranks.

        Args:
            tensor: Input tensor to reduce
            average: If True, divide by world_size after sum
            async_op: If True, return async work handle

        Returns:
            Reduced tensor (or async handle if async_op=True)
        """
        start_time = time.time()
        result = tensor

        try:
            import torch.distributed as dist

            if self._process_group is not None and dist.is_initialized():
                if async_op:
                    handle = dist.all_reduce(tensor, async_op=True)
                    self._record_comm_stats(
                        PiscesLxCommunicationType.ALL_REDUCE,
                        tensor.numel() * tensor.element_size(),
                        time.time() - start_time
                    )
                    return handle
                else:
                    dist.all_reduce(tensor)
                    if average:
                        tensor.div_(self.world_size)
                    self._record_comm_stats(
                        PiscesLxCommunicationType.ALL_REDUCE,
                        tensor.numel() * tensor.element_size(),
                        time.time() - start_time
                    )
                    result = tensor

        except ImportError:
            pass
        except Exception as e:
            self.logger.error("all_reduce_failed", error=str(e))

        return result

    def all_gather(
        self,
        tensor: Any,
        dim: int = 0
    ) -> Any:
        """
        Perform all-gather collective operation.

        Gathers tensors from all ranks and concatenates along specified dimension.

        Args:
            tensor: Input tensor to gather
            dim: Dimension to concatenate along

        Returns:
            Gathered tensor from all ranks
        """
        start_time = time.time()
        result = tensor

        try:
            import torch
            import torch.distributed as dist

            if self._process_group is not None and dist.is_initialized():
                gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
                dist.all_gather(gathered, tensor)
                result = torch.cat(gathered, dim=dim)
                self._record_comm_stats(
                    PiscesLxCommunicationType.ALL_GATHER,
                    tensor.numel() * tensor.element_size() * self.world_size,
                    time.time() - start_time
                )

        except ImportError:
            pass
        except Exception as e:
            self.logger.error("all_gather_failed", error=str(e))

        return result

    def broadcast(
        self,
        tensor: Any,
        src_rank: int = 0
    ) -> Any:
        """
        Broadcast tensor from source rank to all ranks.

        Args:
            tensor: Tensor to broadcast (significant only on src_rank)
            src_rank: Source rank for broadcast

        Returns:
            Broadcast tensor
        """
        start_time = time.time()

        try:
            import torch.distributed as dist

            if self._process_group is not None and dist.is_initialized():
                dist.broadcast(tensor, src=src_rank)
                self._record_comm_stats(
                    PiscesLxCommunicationType.BROADCAST,
                    tensor.numel() * tensor.element_size(),
                    time.time() - start_time
                )

        except ImportError:
            pass
        except Exception as e:
            self.logger.error("broadcast_failed", error=str(e))

        return tensor

    def send(
        self,
        tensor: Any,
        dst_rank: int
    ) -> None:
        """
        Send tensor to destination rank (point-to-point).

        Args:
            tensor: Tensor to send
            dst_rank: Destination rank
        """
        start_time = time.time()

        try:
            import torch.distributed as dist

            if self._process_group is not None and dist.is_initialized():
                dist.send(tensor, dst=dst_rank)
                self._record_comm_stats(
                    PiscesLxCommunicationType.POINT_TO_POINT,
                    tensor.numel() * tensor.element_size(),
                    time.time() - start_time
                )

        except ImportError:
            pass
        except Exception as e:
            self.logger.error("send_failed", dst=dst_rank, error=str(e))

    def recv(
        self,
        tensor: Any,
        src_rank: int
    ) -> Any:
        """
        Receive tensor from source rank (point-to-point).

        Args:
            tensor: Pre-allocated tensor to receive into
            src_rank: Source rank

        Returns:
            Received tensor
        """
        start_time = time.time()

        try:
            import torch.distributed as dist

            if self._process_group is not None and dist.is_initialized():
                dist.recv(tensor, src=src_rank)
                self._record_comm_stats(
                    PiscesLxCommunicationType.POINT_TO_POINT,
                    tensor.numel() * tensor.element_size(),
                    time.time() - start_time
                )

        except ImportError:
            pass
        except Exception as e:
            self.logger.error("recv_failed", src=src_rank, error=str(e))

        return tensor

    def _record_comm_stats(
        self,
        comm_type: PiscesLxCommunicationType,
        bytes_transferred: int,
        elapsed_seconds: float
    ) -> None:
        """
        Record communication statistics.

        Args:
            comm_type: Type of communication
            bytes_transferred: Total bytes transferred
            elapsed_seconds: Time taken for operation
        """
        latency_ms = elapsed_seconds * 1000
        bandwidth_gbps = (bytes_transferred * 8) / (elapsed_seconds * 1e9) if elapsed_seconds > 0 else 0

        stats = PiscesLxCommunicationStats(
            comm_type=comm_type,
            bytes_transferred=bytes_transferred,
            latency_ms=latency_ms,
            bandwidth_gbps=bandwidth_gbps
        )

        with self._comm_lock:
            self.comm_stats.append(stats)
            if len(self.comm_stats) > 1000:
                self.comm_stats = self.comm_stats[-1000:]

        self.metrics.histogram("comm_latency_ms", latency_ms)
        self.metrics.gauge("comm_bandwidth_gbps", bandwidth_gbps)

    def get_comm_summary(self) -> Dict[str, Any]:
        """
        Get summary of communication statistics.

        Returns:
            Dictionary with aggregated communication metrics
        """
        with self._comm_lock:
            if not self.comm_stats:
                return {"total_ops": 0}

            total_bytes = sum(s.bytes_transferred for s in self.comm_stats)
            avg_latency = sum(s.latency_ms for s in self.comm_stats) / len(self.comm_stats)
            avg_bandwidth = sum(s.bandwidth_gbps for s in self.comm_stats) / len(self.comm_stats)

            by_type: Dict[str, List[PiscesLxCommunicationStats]] = {}
            for s in self.comm_stats:
                key = s.comm_type.value
                if key not in by_type:
                    by_type[key] = []
                by_type[key].append(s)

            return {
                "total_ops": len(self.comm_stats),
                "total_bytes": total_bytes,
                "total_gb": total_bytes / (1024 ** 3),
                "avg_latency_ms": avg_latency,
                "avg_bandwidth_gbps": avg_bandwidth,
                "by_type": {
                    k: {
                        "count": len(v),
                        "total_bytes": sum(s.bytes_transferred for s in v)
                    }
                    for k, v in by_type.items()
                }
            }

    def barrier(self) -> None:
        """
        Synchronize all ranks at a barrier.

        Blocks until all ranks reach this point.
        """
        try:
            import torch.distributed as dist

            if self._process_group is not None and dist.is_initialized():
                dist.barrier()

        except ImportError:
            pass
        except Exception as e:
            self.logger.error("barrier_failed", error=str(e))

    def teardown(self) -> None:
        """
        Release distributed resources.

        Cleans up process groups, communication handles, and device monitoring.
        """
        if self._is_torn_down:
            return

        if self.parallel_config.is_distributed:
            self._cleanup_process_group()

        self.comm_stats.clear()
        self.system_monitor = None
        self._is_initialized = False

        super().teardown()

    def _cleanup_process_group(self) -> None:
        """
        Clean up process group resources.
        """
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()

        except ImportError:
            pass
        except Exception as e:
            self.logger.error("process_group_cleanup_failed", error=str(e))

        self._process_group = None


class PiscesLxTensorParallelOperator(PiscesLxDistributedOperator, ABC):
    """
    Base class for tensor parallel operators.

    Tensor parallelism splits individual tensor operations across devices.
    Each device holds a shard of the weight matrices and performs
    computations on its portion, with results combined via all-reduce.

    Key Features:
        - Column-parallel and row-parallel linear layers
        - Split attention heads across devices
        - All-reduce for combining partial results

    Usage:
        class ParallelAttention(PiscesLxTensorParallelOperator):
            def forward_tp(self, inputs, tp_group):
                qkv = self.compute_qkv(inputs)  # Split across TP ranks
                attn_output = self.attention(qkv)
                return self.all_reduce(attn_output)
    """

    @property
    def parallelism_type(self) -> PiscesLxParallelismType:
        return PiscesLxParallelismType.TENSOR

    @property
    def tp_rank(self) -> int:
        """Get tensor parallel rank."""
        return self.parallel_config.tp_rank

    @property
    def tp_degree(self) -> int:
        """Get tensor parallelism degree."""
        return self.parallel_config.tp_degree

    @property
    def tp_group(self) -> List[int]:
        """Get ranks in the tensor parallel group."""
        return self.parallel_config.get_tp_group()

    def column_parallel_linear(
        self,
        input_tensor: Any,
        weight: Any,
        bias: Optional[Any] = None
    ) -> Any:
        """
        Column-parallel linear transformation.

        Splits output features across TP ranks. Each rank computes
        a portion of the output, then all-gather combines results.

        Args:
            input_tensor: Input tensor (replicated across ranks)
            weight: Full weight matrix (each rank has a shard)
            bias: Optional bias (sharded like output)

        Returns:
            Output tensor (sharded across TP ranks)
        """
        try:
            import torch

            local_output = torch.nn.functional.linear(input_tensor, weight, bias)
            return local_output

        except ImportError:
            return None

    def row_parallel_linear(
        self,
        input_tensor: Any,
        weight: Any,
        bias: Optional[Any] = None,
        reduce_output: bool = True
    ) -> Any:
        """
        Row-parallel linear transformation.

        Splits input features across TP ranks. Each rank computes
        partial results, then all-reduce combines them.

        Args:
            input_tensor: Input tensor (sharded across ranks)
            weight: Weight matrix (each rank has a shard)
            bias: Optional bias (replicated, added after reduce)
            reduce_output: If True, all-reduce the output

        Returns:
            Output tensor (replicated across TP ranks)
        """
        try:
            import torch

            local_output = torch.nn.functional.linear(input_tensor, weight)

            if reduce_output:
                local_output = self.all_reduce(local_output, average=False)

            if bias is not None:
                local_output = local_output + bias

            return local_output

        except ImportError:
            return None

    def scatter_tensor(
        self,
        tensor: Any,
        dim: int = 0
    ) -> Any:
        """
        Scatter tensor across TP ranks.

        Splits tensor along specified dimension and distributes
        to each rank.

        Args:
            tensor: Full tensor (only valid on rank 0)
            dim: Dimension to split along

        Returns:
            Local shard of the tensor
        """
        try:
            import torch

            if self.is_master:
                chunks = torch.chunk(tensor, self.tp_degree, dim=dim)
                for i, chunk in enumerate(chunks):
                    if i != self.tp_rank:
                        self.send(chunk, i)
                return chunks[self.tp_rank]
            else:
                shape = list(tensor.shape)
                shape[dim] = shape[dim] // self.tp_degree
                local = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
                return self.recv(local, 0)

        except ImportError:
            return tensor

    def gather_tensor(
        self,
        tensor: Any,
        dim: int = 0
    ) -> Any:
        """
        Gather tensor from all TP ranks.

        Combines sharded tensors from all ranks into a single tensor.

        Args:
            tensor: Local shard
            dim: Dimension to concatenate along

        Returns:
            Full gathered tensor
        """
        return self.all_gather(tensor, dim=dim)


class PiscesLxPipelineParallelOperator(PiscesLxDistributedOperator, ABC):
    """
    Base class for pipeline parallel operators.

    Pipeline parallelism splits model stages across devices.
    Each device holds a complete stage and passes activations
    to the next stage via point-to-point communication.

    Key Features:
        - Stage-based execution
        - Micro-batch scheduling (GPipe, 1F1B)
        - Point-to-point activation passing
        - Gradient checkpointing for memory efficiency

    Pipeline Schedules:
        - GPipe: Fill pipeline, then drain
        - 1F1B: One forward, one backward alternating
        - Interleaved: Multiple stages per device

    Usage:
        class PipelineStage(PiscesLxPipelineParallelOperator):
            def forward_stage(self, inputs):
                output = self.stage_forward(inputs)
                if not self.is_last_stage:
                    self.send(output, self.next_stage)
                return output
    """

    @property
    def parallelism_type(self) -> PiscesLxParallelismType:
        return PiscesLxParallelismType.PIPELINE

    @property
    def pp_rank(self) -> int:
        """Get pipeline stage index."""
        return self.parallel_config.pp_rank

    @property
    def pp_degree(self) -> int:
        """Get number of pipeline stages."""
        return self.parallel_config.pp_degree

    @property
    def is_first_stage(self) -> bool:
        """Check if this is the first pipeline stage."""
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        """Check if this is the last pipeline stage."""
        return self.pp_rank == self.pp_degree - 1

    @property
    def prev_stage_rank(self) -> Optional[int]:
        """Get rank of previous pipeline stage."""
        if self.is_first_stage:
            return None
        return self.rank - self.parallel_config.tp_degree * self.parallel_config.dp_degree

    @property
    def next_stage_rank(self) -> Optional[int]:
        """Get rank of next pipeline stage."""
        if self.is_last_stage:
            return None
        return self.rank + self.parallel_config.tp_degree * self.parallel_config.dp_degree

    def send_activation(
        self,
        activation: Any,
        micro_batch_id: int = 0
    ) -> None:
        """
        Send activation to next pipeline stage.

        Args:
            activation: Activation tensor to send
            micro_batch_id: Micro-batch identifier for tracking
        """
        if self.next_stage_rank is not None:
            self.metrics.counter("pp_activations_sent")
            self.send(activation, self.next_stage_rank)

    def recv_activation(
        self,
        shape: Any,
        dtype: Any = None,
        micro_batch_id: int = 0
    ) -> Any:
        """
        Receive activation from previous pipeline stage.

        Args:
            shape: Shape of expected activation tensor
            dtype: Data type of activation
            micro_batch_id: Micro-batch identifier for tracking

        Returns:
            Received activation tensor
        """
        if self.prev_stage_rank is not None:
            self.metrics.counter("pp_activations_recv")
            try:
                import torch
                tensor = torch.zeros(shape, dtype=dtype)
                return self.recv(tensor, self.prev_stage_rank)
            except ImportError:
                return None
        return None

    def send_gradient(
        self,
        gradient: Any,
        micro_batch_id: int = 0
    ) -> None:
        """
        Send gradient to previous pipeline stage (backward pass).

        Args:
            gradient: Gradient tensor to send
            micro_batch_id: Micro-batch identifier
        """
        if self.prev_stage_rank is not None:
            self.metrics.counter("pp_gradients_sent")
            self.send(gradient, self.prev_stage_rank)

    def recv_gradient(
        self,
        shape: Any,
        dtype: Any = None,
        micro_batch_id: int = 0
    ) -> Any:
        """
        Receive gradient from next pipeline stage (backward pass).

        Args:
            shape: Shape of expected gradient tensor
            dtype: Data type of gradient
            micro_batch_id: Micro-batch identifier

        Returns:
            Received gradient tensor
        """
        if self.next_stage_rank is not None:
            self.metrics.counter("pp_gradients_recv")
            try:
                import torch
                tensor = torch.zeros(shape, dtype=dtype)
                return self.recv(tensor, self.next_stage_rank)
            except ImportError:
                return None
        return None


class PiscesLxExpertParallelOperator(PiscesLxDistributedOperator, ABC):
    """
    Base class for expert parallel operators (MoE).

    Expert parallelism distributes MoE experts across devices.
    Tokens are routed to the appropriate expert device via
    all-to-all communication.

    Key Features:
        - Expert sharding across devices
        - Token routing and dispatch
        - All-to-all communication for expert dispatch
        - Load balancing across experts

    Usage:
        class MoELayer(PiscesLxExpertParallelOperator):
            def forward_ep(self, inputs, routing_weights):
                dispatched = self.dispatch_to_experts(inputs, routing_weights)
                expert_output = self.compute_experts(dispatched)
                return self.combine_from_experts(expert_output, routing_weights)
    """

    @property
    def parallelism_type(self) -> PiscesLxParallelismType:
        return PiscesLxParallelismType.EXPERT

    @property
    def ep_rank(self) -> int:
        """Get expert parallel rank."""
        return self.parallel_config.ep_rank

    @property
    def ep_degree(self) -> int:
        """Get expert parallelism degree."""
        return self.parallel_config.ep_degree

    @property
    def ep_group(self) -> List[int]:
        """Get ranks in the expert parallel group."""
        return list(range(self.ep_degree))

    def dispatch_to_experts(
        self,
        tokens: Any,
        routing_weights: Any,
        expert_indices: Any
    ) -> Tuple[Any, Any]:
        """
        Dispatch tokens to their assigned experts.

        Uses all-to-all communication to send tokens to the
        devices hosting their target experts.

        Args:
            tokens: Input token embeddings
            routing_weights: Soft routing weights for each expert
            expert_indices: Assigned expert index for each token

        Returns:
            Tuple of (dispatched_tokens, dispatch_metadata)
        """
        start_time = time.time()

        try:
            import torch
            import torch.distributed as dist

            batch_size = tokens.shape[0]
            tokens_per_rank = batch_size // self.ep_degree

            dispatched_tokens = torch.zeros_like(tokens[:tokens_per_rank])

            if self._process_group is not None and dist.is_initialized():
                dist.all_to_all_single(dispatched_tokens, tokens)

            self._record_comm_stats(
                PiscesLxCommunicationType.ALL_TO_ALL,
                tokens.numel() * tokens.element_size(),
                time.time() - start_time
            )

            return dispatched_tokens, {
                "original_batch_size": batch_size,
                "tokens_per_rank": tokens_per_rank
            }

        except ImportError:
            return tokens, {}
        except Exception as e:
            self.logger.error("dispatch_failed", error=str(e))
            return tokens, {}

    def combine_from_experts(
        self,
        expert_outputs: Any,
        routing_weights: Any,
        dispatch_metadata: Dict[str, Any]
    ) -> Any:
        """
        Combine expert outputs back to original token order.

        Uses all-to-all communication to gather outputs from
        all expert devices.

        Args:
            expert_outputs: Output from local experts
            routing_weights: Routing weights for combining
            dispatch_metadata: Metadata from dispatch step

        Returns:
            Combined output tensor in original token order
        """
        start_time = time.time()

        try:
            import torch
            import torch.distributed as dist

            batch_size = dispatch_metadata.get("original_batch_size", expert_outputs.shape[0])
            combined = torch.zeros(
                (batch_size,) + expert_outputs.shape[1:],
                dtype=expert_outputs.dtype,
                device=expert_outputs.device
            )

            if self._process_group is not None and dist.is_initialized():
                dist.all_to_all_single(combined, expert_outputs)

            self._record_comm_stats(
                PiscesLxCommunicationType.ALL_TO_ALL,
                combined.numel() * combined.element_size(),
                time.time() - start_time
            )

            return combined

        except ImportError:
            return expert_outputs
        except Exception as e:
            self.logger.error("combine_failed", error=str(e))
            return expert_outputs

    def compute_expert_load_balance(
        self,
        expert_indices: Any,
        num_experts: int
    ) -> Dict[str, float]:
        """
        Compute load balancing metrics for expert routing.

        Args:
            expert_indices: Expert assignment for each token
            num_experts: Total number of experts

        Returns:
            Dictionary with load balancing metrics
        """
        try:
            import torch

            expert_counts = torch.bincount(
                expert_indices.flatten(),
                minlength=num_experts
            ).float()

            total_tokens = expert_indices.numel()
            ideal_count = total_tokens / num_experts

            load_variance = expert_counts.var().item()
            max_load = expert_counts.max().item()
            min_load = expert_counts.min().item()

            aux_loss = num_experts * (expert_counts / total_tokens).var()

            return {
                "load_variance": load_variance,
                "max_load": max_load,
                "min_load": min_load,
                "ideal_load": ideal_count,
                "aux_loss": aux_loss.item()
            }

        except ImportError:
            return {}
        except Exception as e:
            self.logger.error("load_balance_compute_failed", error=str(e))
            return {}


class PiscesLxDataParallelOperator(PiscesLxDistributedOperator, ABC):
    """
    Base class for data parallel operators.

    Data parallelism replicates the model across devices, with each
    device processing a different batch of data. Gradients are
    synchronized via all-reduce.

    Key Features:
        - Model replication across devices
        - Gradient synchronization
        - Distributed optimizer support
        - Gradient accumulation for large batches

    Usage:
        class DPModel(PiscesLxDataParallelOperator):
            def forward_dp(self, inputs):
                output = self.model(inputs)
                return output

            def backward_dp(self, loss):
                loss.backward()
                self.sync_gradients()
    """

    @property
    def parallelism_type(self) -> PiscesLxParallelismType:
        return PiscesLxParallelismType.DATA

    @property
    def dp_rank(self) -> int:
        """Get data parallel rank."""
        return self.parallel_config.dp_rank

    @property
    def dp_degree(self) -> int:
        """Get data parallelism degree."""
        return self.parallel_config.dp_degree

    @property
    def dp_group(self) -> List[int]:
        """Get ranks in the data parallel group."""
        return self.parallel_config.get_dp_group()

    def sync_gradients(
        self,
        parameters: Optional[List[Any]] = None,
        average: bool = True
    ) -> None:
        """
        Synchronize gradients across DP ranks.

        Performs all-reduce on gradients to ensure all replicas
        have the same gradient values before optimizer step.

        Args:
            parameters: List of parameters to sync (default: all)
            average: If True, average gradients across ranks
        """
        try:
            import torch

            if parameters is None:
                parameters = [p for p in torch.nn.Module.parameters(self) if p.grad is not None]

            for param in parameters:
                if param.grad is not None:
                    self.all_reduce(param.grad, average=average)

            self.metrics.counter("dp_gradient_syncs")

        except ImportError:
            pass
        except Exception as e:
            self.logger.error("gradient_sync_failed", error=str(e))

    def sync_model(
        self,
        parameters: Optional[List[Any]] = None,
        src_rank: int = 0
    ) -> None:
        """
        Synchronize model parameters from source rank.

        Broadcasts parameters to ensure all replicas have identical
        weights. Typically used at initialization or checkpoint load.

        Args:
            parameters: List of parameters to sync (default: all)
            src_rank: Source rank for broadcast
        """
        try:
            import torch

            if parameters is None:
                parameters = list(torch.nn.Module.parameters(self))

            for param in parameters:
                self.broadcast(param.data, src_rank=src_rank)

            self.metrics.counter("dp_model_syncs")

        except ImportError:
            pass
        except Exception as e:
            self.logger.error("model_sync_failed", error=str(e))

    def average_metrics(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Average metrics across DP ranks.

        Useful for reporting consistent training metrics.

        Args:
            metrics: Dictionary of metric values

        Returns:
            Averaged metrics dictionary
        """
        averaged = {}
        try:
            import torch

            for key, value in metrics.items():
                tensor = torch.tensor([value])
                self.all_reduce(tensor, average=True)
                averaged[key] = tensor.item()

        except ImportError:
            averaged = metrics
        except Exception as e:
            self.logger.error("metrics_average_failed", error=str(e))
            averaged = metrics

        return averaged


class PiscesLxHybridParallelOperator(PiscesLxDistributedOperator, ABC):
    """
    Base class for hybrid parallel operators combining multiple strategies.

    Hybrid parallelism combines TP, PP, EP, and DP for maximum scalability.
    This is the most common approach for training very large models.

    Common Configurations:
        - 3D: TP + PP + DP (e.g., Megatron-LM)
        - 4D: TP + PP + EP + DP (e.g., DeepSeek-V3)
        - 2D: TP + DP (e.g., single-node multi-GPU)

    Usage:
        class HybridModel(PiscesLxHybridParallelOperator):
            def forward_hybrid(self, inputs):
                x = self.tensor_parallel_layer(inputs)
                x = self.pipeline_stage(x)
                return x
    """

    @property
    def parallelism_type(self) -> PiscesLxParallelismType:
        return PiscesLxParallelismType.HYBRID

    def __init__(
        self,
        config: Optional[PiscesLxOperatorConfig] = None,
        parallel_config: Optional[PiscesLxParallelConfig] = None
    ):
        super().__init__(config, parallel_config)
        self._tp_operator: Optional[PiscesLxTensorParallelOperator] = None
        self._pp_operator: Optional[PiscesLxPipelineParallelOperator] = None
        self._ep_operator: Optional[PiscesLxExpertParallelOperator] = None
        self._dp_operator: Optional[PiscesLxDataParallelOperator] = None

    def setup(self) -> None:
        """Initialize hybrid parallel components."""
        super().setup()

        if self.parallel_config.tp_degree > 1:
            self._tp_operator = PiscesLxTensorParallelOperator(
                config=self.config,
                parallel_config=self.parallel_config
            )
            self._tp_operator.setup()

        if self.parallel_config.pp_degree > 1:
            self._pp_operator = PiscesLxPipelineParallelOperator(
                config=self.config,
                parallel_config=self.parallel_config
            )
            self._pp_operator.setup()

        if self.parallel_config.ep_degree > 1:
            self._ep_operator = PiscesLxExpertParallelOperator(
                config=self.config,
                parallel_config=self.parallel_config
            )
            self._ep_operator.setup()

        if self.parallel_config.dp_degree > 1:
            self._dp_operator = PiscesLxDataParallelOperator(
                config=self.config,
                parallel_config=self.parallel_config
            )
            self._dp_operator.setup()

    @property
    def tp_op(self) -> Optional[PiscesLxTensorParallelOperator]:
        """Get tensor parallel operator."""
        return self._tp_operator

    @property
    def pp_op(self) -> Optional[PiscesLxPipelineParallelOperator]:
        """Get pipeline parallel operator."""
        return self._pp_operator

    @property
    def ep_op(self) -> Optional[PiscesLxExpertParallelOperator]:
        """Get expert parallel operator."""
        return self._ep_operator

    @property
    def dp_op(self) -> Optional[PiscesLxDataParallelOperator]:
        """Get data parallel operator."""
        return self._dp_operator

    def get_parallel_topology(self) -> Dict[str, Any]:
        """
        Get the parallel topology configuration.

        Returns:
            Dictionary describing the hybrid parallelism setup
        """
        return {
            "world_size": self.world_size,
            "tp_degree": self.parallel_config.tp_degree,
            "pp_degree": self.parallel_config.pp_degree,
            "ep_degree": self.parallel_config.ep_degree,
            "dp_degree": self.parallel_config.dp_degree,
            "rank": self.rank,
            "tp_rank": self.parallel_config.tp_rank,
            "pp_rank": self.parallel_config.pp_rank,
            "ep_rank": self.parallel_config.ep_rank,
            "dp_rank": self.parallel_config.dp_rank,
            "tp_group": self.parallel_config.get_tp_group(),
            "pp_group": self.parallel_config.get_pp_group(),
            "dp_group": self.parallel_config.get_dp_group()
        }
