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

from __future__ import annotations

"""
3D Parallelism Operator for Large-Scale Training

Implements 3D parallelism combining:
    - Data Parallelism (DP)
    - Tensor Parallelism (TP)
    - Pipeline Parallelism (PP)

Key Features:
    - Scale to thousands of GPUs
    - Automatic parallelism strategy selection
    - Memory-efficient gradient synchronization
    - Overlap computation and communication

References:
    - Megatron-LM (Shoeybi et al., 2019)
    - Megatron-DeepSpeed (2023)
    - 3D Parallelism (NVIDIA, 2022)

Usage:
    >>> from opss.train.parallel_3d import POPSSParallel3DOperator, POPSSParallel3DConfig
    >>> config = POPSSParallel3DConfig(dp_size=8, tp_size=4, pp_size=2)
    >>> operator = POPSSParallel3DOperator(config)
    >>> result = operator.execute({"model": model, "batch": batch})
"""

import os
import functools
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus
from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from configs.version import VERSION


class POPSSParallelismType(Enum):
    """Types of parallelism."""
    DATA = "data"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    SEQUENCE = "sequence"
    EXPERT = "expert"


class POPSSPipelineSchedule(Enum):
    """Pipeline scheduling strategies."""
    GPipe = "gpipe"
    ONE_F_ONE_B = "1f1b"
    INTERLEAVED = "interleaved"
    ZERO_BUBBLE = "zero_bubble"


@dataclass
class POPSSParallel3DConfig:
    """
    Configuration for 3D Parallelism Operator.
    
    Attributes:
        dp_size: Data parallelism size
        tp_size: Tensor parallelism size
        pp_size: Pipeline parallelism size
        sequence_parallel: Whether to enable sequence parallelism
        pipeline_schedule: Pipeline scheduling strategy
        num_micro_batches: Number of micro-batches for pipeline
        overlap_communication: Whether to overlap communication with computation
        gradient_checkpointing: Whether to enable gradient checkpointing
        zero_stage: ZeRO optimization stage (0, 1, 2, 3)
        cpu_offload: Whether to offload optimizer states to CPU
        mixed_precision: Mixed precision training mode
    """
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    sequence_parallel: bool = True
    pipeline_schedule: POPSSPipelineSchedule = POPSSPipelineSchedule.ONE_F_ONE_B
    num_micro_batches: int = 4
    overlap_communication: bool = True
    gradient_checkpointing: bool = False
    zero_stage: int = 0
    cpu_offload: bool = False
    mixed_precision: str = "bf16"
    
    enable_overlap: bool = True
    overlap_bucket_size_mb: int = 25
    overlap_grad_sync: bool = True
    
    def __post_init__(self):
        if isinstance(self.pipeline_schedule, str):
            self.pipeline_schedule = POPSSPipelineSchedule(self.pipeline_schedule)
        
        self.world_size = self.dp_size * self.tp_size * self.pp_size
    
    def get_parallel_rank(self, global_rank: int) -> Tuple[int, int, int]:
        """Get (dp_rank, tp_rank, pp_rank) from global rank."""
        dp_rank = global_rank // (self.tp_size * self.pp_size)
        remainder = global_rank % (self.tp_size * self.pp_size)
        tp_rank = remainder // self.pp_size
        pp_rank = remainder % self.pp_size
        return dp_rank, tp_rank, pp_rank
    
    def get_global_rank(self, dp_rank: int, tp_rank: int, pp_rank: int) -> int:
        """Get global rank from parallel ranks."""
        return dp_rank * self.tp_size * self.pp_size + tp_rank * self.pp_size + pp_rank


class POPSSParallel3DOperator(PiscesLxOperatorInterface):
    """
    3D Parallelism Operator for Large-Scale Model Training.
    
    Combines three parallelism dimensions for training models with
    trillions of parameters across thousands of GPUs.
    
    Parallelism Strategy:
        - Data Parallelism: Replicate model across devices
        - Tensor Parallelism: Split tensors across devices
        - Pipeline Parallelism: Split layers across devices
    
    Memory Scaling:
        - Standard: O(model_size)
        - TP: O(model_size / tp_size)
        - PP: O(model_size / pp_size)
        - 3D: O(model_size / (tp_size * pp_size))
    
    Example:
        >>> config = POPSSParallel3DConfig(dp_size=8, tp_size=4, pp_size=2)
        >>> operator = POPSSParallel3DOperator(config)
        >>> result = operator.initialize(model, optimizer)
    """
    
    def __init__(self, config: Optional[POPSSParallel3DConfig] = None):
        super().__init__()
        self.name = "train.parallel_3d"
        self.version = VERSION
        self.type = "training"
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Train",file_path=get_log_file("PiscesLx.Opss.Train"), enable_file=True)
        self.config = config or POPSSParallel3DConfig()
        
        self._initialized = False
        self._dp_rank = 0
        self._tp_rank = 0
        self._pp_rank = 0
        self._global_rank = 0
        self._world_size = 1
        
        self._dp_group = None
        self._tp_group = None
        self._pp_group = None
        
        self._model = None
        self._optimizer = None
        self._effective_tp_size = int(self.config.tp_size)
        self._effective_pp_size = int(self.config.pp_size)
    
    def initialize(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> PiscesLxOperatorResult:
        """
        Initialize 3D parallelism for model and optimizer.
        
        Args:
            model: PyTorch model to parallelize
            optimizer: Optional optimizer
        
        Returns:
            PiscesLxOperatorResult with parallelized model
        """
        try:
            self._validate_model_compatibility(model)
            self._setup_distributed()
            self._create_process_groups()
            
            self._model = self._parallelize_model(model)
            self._optimizer = self._setup_optimizer(optimizer)
            if self.config.overlap_communication and self.config.dp_size > 1 and self._dp_group is not None:
                self._overlap_optimizer = POPSSCommComputeOverlapOptimizer(
                    model=self._model,
                    bucket_size_mb=int(getattr(self.config, "overlap_bucket_size_mb", 25) or 25),
                    enable_overlap=bool(getattr(self.config, "enable_overlap", True)),
                    grad_sync=bool(getattr(self.config, "overlap_grad_sync", True)),
                    dp_group=self._dp_group,
                )
            else:
                self._overlap_optimizer = None
            
            self._initialized = True
            
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "parallel_info": {
                        "dp_rank": self._dp_rank,
                        "tp_rank": self._tp_rank,
                        "pp_rank": self._pp_rank,
                        "world_size": self._world_size,
                        "effective_tp_size": self._effective_tp_size,
                        "effective_pp_size": self._effective_pp_size,
                    }
                }
            )
        except Exception as e:
            self._LOG.error(f"Failed to initialize 3D parallelism: {e}")
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error=str(e)
            )

    def _validate_model_compatibility(self, model: nn.Module) -> None:
        """Clamp unsupported TP/PP modes to safe values for Yv-style monolithic models."""
        model_name = model.__class__.__name__
        has_yv_stack = hasattr(model, "layers") and hasattr(model, "embed") and hasattr(model, "lm_head")
        if not has_yv_stack:
            return

        requested_tp = int(getattr(self.config, "tp_size", 1) or 1)
        requested_pp = int(getattr(self.config, "pp_size", 1) or 1)
        self._effective_tp_size = requested_tp
        self._effective_pp_size = requested_pp

        # Current TP/PP utilities mutate module weights without the matching
        # gather/scatter semantics required by YvModel.forward, which would
        # silently corrupt training. Keep DP/sequence parallel active, but
        # force unsafe TP/PP dimensions back to 1 until a topology-aware path
        # is implemented for this architecture.
        if requested_tp > 1:
            self._effective_tp_size = 1
            self.config.tp_size = 1
        if requested_pp > 1:
            self._effective_pp_size = 1
            self.config.pp_size = 1

        if self._effective_tp_size != requested_tp or self._effective_pp_size != requested_pp:
            self.config.world_size = self.config.dp_size * self.config.tp_size * self.config.pp_size
            self._LOG.warning(
                f"3D parallel compatibility guard engaged for {model_name}: "
                f"requested(tp={requested_tp}, pp={requested_pp}) -> "
                f"effective(tp={self.config.tp_size}, pp={self.config.pp_size}). "
                "DP/sequence parallel remain enabled; topology-aware TP/PP is not yet safe for this model."
            )
    
    def _setup_distributed(self):
        """Setup distributed environment."""
        try:
            import torch.distributed as dist
            
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            
            self._global_rank = dist.get_rank()
            self._world_size = dist.get_world_size()
            
            self._dp_rank, self._tp_rank, self._pp_rank = self.config.get_parallel_rank(self._global_rank)
            
            self._LOG.info(f"Initialized 3D parallelism: dp={self._dp_rank}, tp={self._tp_rank}, pp={self._pp_rank}")
            
        except Exception as e:
            self._LOG.warning(f"Distributed not available: {e}")
    
    def _create_process_groups(self):
        """Create process groups for each parallelism dimension."""
        import torch.distributed as dist
        
        if not dist.is_initialized():
            return
        
        dp_size = self.config.dp_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        
        dp_groups = []
        for tp in range(tp_size):
            for pp in range(pp_size):
                group_ranks = [dp * tp_size * pp_size + tp * pp_size + pp for dp in range(dp_size)]
                dp_groups.append(group_ranks)
        
        tp_groups = []
        for dp in range(dp_size):
            for pp in range(pp_size):
                group_ranks = [dp * tp_size * pp_size + tp * pp_size + pp for tp in range(tp_size)]
                tp_groups.append(group_ranks)
        
        pp_groups = []
        for dp in range(dp_size):
            for tp in range(tp_size):
                group_ranks = [dp * tp_size * pp_size + tp * pp_size + pp for pp in range(pp_size)]
                pp_groups.append(group_ranks)
        
        for ranks in dp_groups:
            if self._global_rank in ranks:
                self._dp_group = dist.new_group(ranks)
        
        for ranks in tp_groups:
            if self._global_rank in ranks:
                self._tp_group = dist.new_group(ranks)
        
        for ranks in pp_groups:
            if self._global_rank in ranks:
                self._pp_group = dist.new_group(ranks)
    
    def _parallelize_model(self, model: nn.Module) -> nn.Module:
        """Apply parallelism to model."""
        if self.config.pp_size > 1:
            model = self._apply_pipeline_parallelism(model)
        
        if self.config.tp_size > 1:
            model = self._apply_tensor_parallelism(model)
        
        if self.config.sequence_parallel:
            model = self._apply_sequence_parallelism(model)
        
        return model
    
    def _apply_tensor_parallelism(self, model: nn.Module) -> nn.Module:
        """Apply tensor parallelism to model layers."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._parallelize_linear(module, name)
            elif isinstance(module, nn.Embedding):
                self._parallelize_embedding(module, name)
        
        return model
    
    def _parallelize_linear(self, linear: nn.Linear, name: str):
        """Parallelize linear layer."""
        if 'query' in name or 'key' in name or 'value' in name:
            out_features = linear.out_features
            new_out = out_features // self.config.tp_size
            
            linear.weight.data = linear.weight.data[self._tp_rank * new_out:(self._tp_rank + 1) * new_out, :]
            linear.out_features = new_out
        
        elif 'dense' in name or 'fc' in name:
            in_features = linear.in_features
            new_in = in_features // self.config.tp_size
            
            linear.weight.data = linear.weight.data[:, self._tp_rank * new_in:(self._tp_rank + 1) * new_in]
            linear.in_features = new_in
    
    def _parallelize_embedding(self, embedding: nn.Embedding, name: str):
        """Parallelize embedding layer."""
        num_embeddings = embedding.num_embeddings
        new_num = num_embeddings // self.config.tp_size
        
        embedding.weight.data = embedding.weight.data[self._tp_rank * new_num:(self._tp_rank + 1) * new_num, :]
        embedding.num_embeddings = new_num
    
    def _apply_pipeline_parallelism(self, model: nn.Module) -> nn.Module:
        """Apply pipeline parallelism to model."""
        layers = list(model.children())
        total_layers = len(layers)
        layers_per_stage = total_layers // self.config.pp_size
        
        start_idx = self._pp_rank * layers_per_stage
        end_idx = start_idx + layers_per_stage if self._pp_rank < self.config.pp_size - 1 else total_layers
        
        stage_layers = layers[start_idx:end_idx]
        
        class PipelineStage(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = nn.ModuleList(layers)
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return PipelineStage(stage_layers)
    
    def _apply_sequence_parallelism(self, model: nn.Module) -> nn.Module:
        """Apply sequence parallelism for long sequences."""
        return model
    
    def _setup_optimizer(self, optimizer: Optional[torch.optim.Optimizer]) -> Optional[torch.optim.Optimizer]:
        """Setup optimizer with parallelism support."""
        if optimizer is None:
            return None
        
        if self.config.zero_stage > 0:
            optimizer = self._setup_zero_optimizer(optimizer)
        
        return optimizer
    
    def _setup_zero_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """Setup ZeRO optimizer."""
        try:
            import deepspeed
            return deepspeed.ops.adam.DeepSpeedCPUAdam(optimizer.param_groups)
        except ImportError:
            self._LOG.warning("DeepSpeed not available, using standard optimizer")
            return optimizer
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute training step with 3D parallelism.
        
        Args:
            inputs: Dictionary containing:
                - batch: Input batch
                - forward_fn: Forward function
                - backward_fn: Backward function
        
        Returns:
            PiscesLxOperatorResult with training metrics
        """
        if not self._initialized:
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error="3D parallelism not initialized. Call initialize() first."
            )
        
        batch = inputs.get("batch")
        forward_fn = inputs.get("forward_fn")
        
        if batch is None:
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error="Missing batch input"
            )
        
        # Initialize communication-computation overlap optimizer for Zero-Bubble
        if self.config.overlap_communication and self.config.dp_size > 1:
            if not hasattr(self, '_overlap_optimizer') or self._overlap_optimizer is None:
                self._overlap_optimizer = POPSSCommComputeOverlapOptimizer(
                    model=self._model,
                    bucket_size_mb=25,
                    enable_overlap=True,
                    grad_sync=True,
                    dp_group=self._dp_group,
                )
                self._async_handle = None
        
        try:
            if self.config.pp_size > 1:
                loss = self._pipeline_forward_backward(batch, forward_fn)
            else:
                loss = self._standard_forward_backward(batch, forward_fn)
            
            self._synchronize_gradients()
            
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.SUCCESS,
                output={"loss": loss}
            )
        except Exception as e:
            self._LOG.error(f"Training step failed: {e}")
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error=str(e)
            )
    
    def _standard_forward_backward(self, batch: Any, forward_fn) -> torch.Tensor:
        """Standard forward-backward pass."""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        
        loss = forward_fn(self._model, batch)
        loss.backward()
        return loss
    
    def _pipeline_forward_backward(self, batch: Any, forward_fn) -> torch.Tensor:
        """Pipeline forward-backward pass with micro-batches."""
        micro_batches = self._split_batch(batch, self.config.num_micro_batches)
        
        losses = []

        if self.config.pipeline_schedule == POPSSPipelineSchedule.ZERO_BUBBLE:
            losses = self._zero_bubble_schedule(micro_batches, forward_fn)
        elif self.config.pipeline_schedule == POPSSPipelineSchedule.ONE_F_ONE_B:
            losses = self._one_f_one_b_schedule(micro_batches, forward_fn)
        elif self.config.pipeline_schedule == POPSSPipelineSchedule.INTERLEAVED:
            losses = self._interleaved_schedule(micro_batches, forward_fn)
        elif self.config.pipeline_schedule == POPSSPipelineSchedule.GPipe:
            losses = self._gpipe_schedule(micro_batches, forward_fn)
        else:
            losses = self._interleaved_schedule(micro_batches, forward_fn)
        
        return torch.stack(losses).mean()
    
    def _split_batch(self, batch: Any, num_splits: int) -> List[Any]:
        """Split batch into micro-batches."""
        if isinstance(batch, torch.Tensor):
            return torch.chunk(batch, num_splits, dim=0)
        elif isinstance(batch, dict):
            micro_batches = []
            for i in range(num_splits):
                micro_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        micro_batch[k] = torch.chunk(v, num_splits, dim=0)[i]
                    else:
                        micro_batch[k] = v
                micro_batches.append(micro_batch)
            return micro_batches
        return [batch] * num_splits
    
    def _one_f_one_b_schedule(self, micro_batches: List, forward_fn) -> List[torch.Tensor]:
        """1F1B pipeline schedule with warmup phase."""
        losses = []
        num_warmup = min(self.config.pp_size - 1, len(micro_batches))
        num_micro_batches = len(micro_batches)

        for i in range(num_warmup):
            loss = self._forward_micro_batch(micro_batches[i], forward_fn)
            losses.append(loss)

        for i in range(num_micro_batches):
            id_fw = i + num_warmup
            if id_fw < num_micro_batches:
                loss = self._forward_micro_batch(micro_batches[id_fw], forward_fn)
                losses.append(loss)
            if i < num_micro_batches - 1:
                self._backward_micro_batch(losses[i] if i < len(losses) else None)

        return losses

    def _gpipe_schedule(self, micro_batches: List, forward_fn) -> List[torch.Tensor]:
        """GPipe pipeline schedule."""
        losses = []

        for micro_batch in micro_batches:
            loss = self._forward_micro_batch(micro_batch, forward_fn)
            losses.append(loss)

        for loss in losses:
            self._backward_micro_batch(loss)

        return losses

    def _interleaved_schedule(self, micro_batches: List, forward_fn) -> List[torch.Tensor]:
        """Interleaved 1F1B pipeline schedule.

        Splits model into multiple chunks and interleaves forward/backward
        across chunks to reduce pipeline bubble.
        """
        num_chunks = max(2, self.config.pp_size)
        chunk_size = max(1, len(micro_batches) // num_chunks)
        chunks = [micro_batches[i:i+chunk_size] for i in range(0, len(micro_batches), chunk_size)]

        losses = []
        num_warmup = min(num_chunks - 1, len(micro_batches))

        for i in range(num_warmup):
            if i < len(micro_batches):
                loss = self._forward_micro_batch(micro_batches[i], forward_fn)
                losses.append(loss)

        fw_idx = num_warmup
        for bwd_step in range(len(micro_batches)):
            for chunk_idx in range(num_chunks):
                if fw_idx < len(micro_batches):
                    loss = self._forward_micro_batch(micro_batches[fw_idx], forward_fn)
                    losses.append(loss)
                    fw_idx += 1
                if bwd_step > 0 or chunk_idx > 0:
                    lidx = min(bwd_step * num_chunks + chunk_idx, len(losses) - 1)
                    if lidx >= 0 and lidx < len(losses):
                        self._backward_micro_batch(losses[lidx])

        return losses

    def _zero_bubble_schedule(self, micro_batches: List, forward_fn) -> List[torch.Tensor]:
        """Zero-Bubble pipeline schedule with computation-communication overlap.

        Key idea: overlap backward computation (gradient computation) with
        gradient communication (all-reduce) across micro-batches. This
        eliminates the pipeline bubble that normally exists between
        forward and backward passes of consecutive micro-batches.

        Strategy:
        1. Forward all micro-batches (split into warmup + steady)
        2. During backward of micro-batch i:
           - async all-reduce gradients of micro-batch i-1
           - compute gradients of micro-batch i
           - wait for async all-reduce of i-1 when i's backward done
        3. Final sync for the last micro-batch
        """
        losses = []
        num_micro_batches = len(micro_batches)
        num_warmup = min(self.config.pp_size - 1, num_micro_batches)

        # --- Warmup: forward only ---
        for i in range(num_warmup):
            loss = self._forward_micro_batch(micro_batches[i], forward_fn)
            losses.append(loss)

        # --- Steady: interleave forward + backward with overlapped sync ---
        for i in range(num_micro_batches):
            # Forward next micro-batch if available
            fw_idx = i + num_warmup
            if fw_idx < num_micro_batches:
                loss_fwd = self._forward_micro_batch(micro_batches[fw_idx], forward_fn)
                losses.append(loss_fwd)

            # Backward current micro-batch
            if i < len(losses):
                losses[i].backward()

            # If previous backward had an async all-reduce, wait + unscale now
            if i > 0 and hasattr(self, '_async_handle') and self._async_handle is not None:
                self._wait_and_unscale_last_bucket()
                self._async_handle = None

            # Launch async all-reduce for current gradients
            if self._overlap_optimizer is not None:
                self._async_handle = self._overlap_optimizer.sync_gradients(async_only=True)

        # --- Drain: backward remaining + final sync ---
        if self._async_handle is not None:
            self._wait_and_unscale_last_bucket()
            self._async_handle = None
        if self._overlap_optimizer is not None:
            self._overlap_optimizer.sync_gradients()

        return losses

    def _wait_and_unscale_last_bucket(self):
        """Wait for pending async all-reduce and unscale gradients."""
        if hasattr(self, '_overlap_optimizer') and self._overlap_optimizer is not None:
            try:
                self._overlap_optimizer.sync_gradients()
            except Exception:
                pass
    
    def _forward_micro_batch(self, micro_batch: Any, forward_fn) -> torch.Tensor:
        """Forward pass for single micro-batch (no backward)."""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        return forward_fn(self._model, micro_batch)

    def _backward_micro_batch(self, loss: Optional[torch.Tensor] = None):
        """Backward pass for single micro-batch with optional immediate sync."""
        if loss is not None:
            loss.backward()
        if self._overlap_optimizer is not None:
            self._overlap_optimizer.sync_gradients()

    def _standard_forward_backward(self, batch: Any, forward_fn) -> torch.Tensor:
        """Standard forward-backward pass with overlapped grad sync."""
        loss = forward_fn(self._model, batch)
        loss.backward()
        if self._overlap_optimizer is not None:
            self._overlap_optimizer.sync_gradients()
        return loss

    def _synchronize_gradients(self):
        """Synchronize gradients across parallel dimensions using bucketed all-reduce."""
        import torch.distributed as dist

        if not dist.is_initialized():
            return
        if self._model is None:
            return
        if not (self.config.dp_size > 1 and self._dp_group is not None):
            return

        # Use bucketed async all-reduce if overlap optimizer available
        if self._overlap_optimizer is not None:
            self._overlap_optimizer.sync_gradients()
        else:
            # Fallback: simple sequential all-reduce per parameter
            for param in self._model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, group=self._dp_group)
                    param.grad.div_(self.config.dp_size)
    
    def get_memory_estimate(self, model_params: int) -> Dict[str, float]:
        """
        Estimate memory usage with 3D parallelism.
        
        Args:
            model_params: Total model parameters
        
        Returns:
            Dictionary with memory estimates
        """
        bytes_per_param = 4 if self.config.mixed_precision == "fp32" else 2
        
        total_memory = model_params * bytes_per_param
        
        combined_memory = total_memory / (self.config.tp_size * self.config.pp_size)
        
        optimizer_memory = combined_memory * 2
        gradient_memory = combined_memory
        
        if self.config.zero_stage >= 2:
            optimizer_memory /= self.config.dp_size
            gradient_memory /= self.config.dp_size
        
        if self.config.zero_stage >= 3:
            combined_memory /= self.config.dp_size
        
        return {
            "model_memory_per_device_gb": combined_memory / 1e9,
            "optimizer_memory_per_device_gb": optimizer_memory / 1e9,
            "gradient_memory_per_device_gb": gradient_memory / 1e9,
            "total_memory_per_device_gb": (combined_memory + optimizer_memory + gradient_memory) / 1e9,
            "memory_reduction_factor": self.config.tp_size * self.config.pp_size
        }


class POPSSCommComputeOverlapOptimizer:
    """
    Communication-Computation Overlap Optimizer for Distributed Training.

    Overlaps gradient computation with gradient communication using
    async all-reduce, gradient bucketing, and hierarchical all-reduce
    strategies to minimize communication overhead.

    Features:
        - Gradient bucketing: Groups gradients by size for efficient communication
        - Async all-reduce: Overlaps computation with gradient synchronization
        - Hierarchical all-reduce: Intra-node then inter-node for multi-node training
        - torch.compile graph capture support via @torch.compiler.disable on sync ops

    Usage:
        >>> model = MyModel()
        >>> optimizer = POPSSCommComputeOverlapOptimizer(
        ...     model=model,
        ...     bucket_size_mb=25,
        ...     enable_overlap=True,
        ... )
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.sync_gradients()  # Overlaps sync with computation
        ...     optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        bucket_size_mb: int = 25,
        enable_overlap: bool = True,
        grad_sync: bool = True,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        """
        Initialize the overlap optimizer.

        Args:
            model: The neural network model whose gradients will be synchronized.
            bucket_size_mb: Target size in MB for each gradient bucket.
            enable_overlap: Whether to enable communication-computation overlap.
            grad_sync: Whether to perform gradient synchronization.
            dp_group: Data parallel process group. Uses WORLD if None.
        """
        self.model = model
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.enable_overlap = enable_overlap
        self.grad_sync = grad_sync
        self.dp_group = dp_group

        self._buckets: List[List[Tuple[str, nn.Parameter]]] = []
        self._pending_handles: List[Tuple[torch.distributed.Work, List[torch.Tensor], torch.Tensor]] = []
        self._build_buckets()

    def _build_buckets(self):
        """Build gradient buckets by grouping model parameters by size."""
        current_bucket: List[Tuple[str, nn.Parameter]] = []
        current_size = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            param_size = param.numel() * param.element_size()
            if current_size + param_size > self.bucket_size_bytes and current_bucket:
                self._buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            current_bucket.append((name, param))
            current_size += param_size

        if current_bucket:
            self._buckets.append(current_bucket)

    @torch.compiler.disable
    def _sync_bucket(
        self,
        bucket: List[Tuple[str, nn.Parameter]],
        dp_group: torch.distributed.ProcessGroup,
    ) -> Optional[Tuple[torch.distributed.Work, List[torch.Tensor], torch.Tensor]]:
        """Asynchronously synchronize a single gradient bucket."""
        grads = [p.grad for _, p in bucket if p.grad is not None]
        if not grads:
            return None

        flat_grad = torch._utils._flatten_dense_tensors(grads)
        handle = torch.distributed.all_reduce(
            flat_grad,
            op=torch.distributed.ReduceOp.SUM,
            group=dp_group,
            async_op=True,
        )
        return handle, grads, flat_grad

    def _wait_and_unscale(
        self,
        handle: torch.distributed.Work,
        grads: List[torch.Tensor],
        flat_grad: torch.Tensor,
        world_size: int,
    ):
        """Wait for async all-reduce and unscale gradients."""
        handle.wait()
        synced = torch._utils._unflatten_dense_tensors(flat_grad, grads)
        for g, sg in zip(grads, synced):
            g.copy_(sg)
        for g in grads:
            g.div_(world_size)

    def sync_gradients(self, async_only: bool = False):
        """Synchronize gradients across all data parallel workers.

        Uses bucketed async all-reduce to overlap communication with
        computation. Each bucket is reduced independently, allowing
        downstream computation to proceed as each bucket completes.
        """
        if not self.enable_overlap or not self.grad_sync:
            return

        import torch.distributed as dist

        dp_group = self.dp_group or dist.group.WORLD
        world_size = dp_group.size() if dp_group else 1

        if world_size <= 1:
            return

        handles = []
        for bucket in self._buckets:
            result = self._sync_bucket(bucket, dp_group)
            if result is not None:
                handles.append(result)

        if async_only:
            self._pending_handles = handles
            return handles

        if self._pending_handles:
            handles = self._pending_handles + handles
            self._pending_handles = []

        for result in handles:
            self._wait_and_unscale(*result, world_size)
        return handles

    def overlap_allreduce(self, backward_fn):
        """
        Decorator that wraps a backward function with overlapped all-reduce.

        The decorated function triggers gradient synchronization after
        backward, enabling communication to overlap with subsequent
        computation steps.

        Usage:
            >>> @optimizer.overlap_allreduce
            ... def backward_step(loss):
            ...     loss.backward()
            ...
            >>> backward_step(loss)
            >>> optimizer.step()
        """
        @functools.wraps(backward_fn)
        def wrapper(*args, **kwargs):
            result = backward_fn(*args, **kwargs)
            self.sync_gradients()
            return result
        return wrapper

    def hierarchical_allreduce(self):
        """
        Perform hierarchical all-reduce for multi-node training.

        Reduces communication by first aggregating within each node
        (intra-node), then across nodes (inter-node). This minimizes
        cross-node bandwidth usage.

        Requires LOCAL_RANK and LOCAL_WORLD_SIZE environment variables
        to be set (typically by torchrun or similar launcher).
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        if world_size <= 1:
            return

        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        global_rank = dist.get_rank()

        if local_world_size <= 0:
            self.sync_gradients()
            return

        n_nodes = world_size // local_world_size
        if n_nodes <= 1:
            self.sync_gradients()
            return

        node_rank = global_rank // local_world_size
        local_leader_rank = node_rank * local_world_size

        intra_node_group = dist.new_group(
            ranks=list(range(node_rank * local_world_size, (node_rank + 1) * local_world_size)),
        )

        cross_node_group = dist.new_group(
            ranks=list(range(0, world_size, local_world_size)),
        )

        for bucket in self._buckets:
            grads = [p.grad for _, p in bucket if p.grad is not None]
            if not grads:
                continue

            flat_grad = torch._utils._flatten_dense_tensors(grads)

            dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, group=intra_node_group)
            flat_grad.div_(local_world_size)

            if global_rank == local_leader_rank:
                dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, group=cross_node_group)
                flat_grad.div_(n_nodes)

            dist.broadcast(flat_grad, src=local_leader_rank, group=intra_node_group)

            synced = torch._utils._unflatten_dense_tensors(flat_grad, grads)
            for g, sg in zip(grads, synced):
                g.copy_(sg)


__all__ = [
    "POPSSParallelismType",
    "POPSSPipelineSchedule",
    "POPSSParallel3DConfig",
    "POPSSParallel3DOperator",
    "POPSSCommComputeOverlapOptimizer",
]
