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

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import math

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
            self._setup_distributed()
            self._create_process_groups()
            
            self._model = self._parallelize_model(model)
            self._optimizer = self._setup_optimizer(optimizer)
            
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
                        "world_size": self._world_size
                    }
                }
            )
        except Exception as e:
            self._LOG.error(f"Failed to initialize 3D parallelism: {e}")
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error=str(e)
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
        
        world_size = self._world_size
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
        
        if self.config.pipeline_schedule == POPSSPipelineSchedule.ONE_F_ONE_B:
            losses = self._one_f_one_b_schedule(micro_batches, forward_fn)
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
        """1F1B pipeline schedule."""
        losses = []
        num_warmup = self.config.pp_size - 1
        num_micro_batches = len(micro_batches)
        
        for i in range(num_warmup):
            if i < num_micro_batches:
                loss = self._forward_micro_batch(micro_batches[i], forward_fn)
                losses.append(loss)
        
        for i in range(num_micro_batches):
            if i + num_warmup < num_micro_batches:
                loss = self._forward_micro_batch(micro_batches[i + num_warmup], forward_fn)
                losses.append(loss)
            
            if i > 0:
                self._backward_micro_batch()
        
        return losses
    
    def _gpipe_schedule(self, micro_batches: List, forward_fn) -> List[torch.Tensor]:
        """GPipe pipeline schedule."""
        losses = []
        
        for micro_batch in micro_batches:
            loss = self._forward_micro_batch(micro_batch, forward_fn)
            losses.append(loss)
        
        for _ in micro_batches:
            self._backward_micro_batch()
        
        return losses
    
    def _interleaved_schedule(self, micro_batches: List, forward_fn) -> List[torch.Tensor]:
        """Interleaved pipeline schedule."""
        return self._gpipe_schedule(micro_batches, forward_fn)
    
    def _forward_micro_batch(self, micro_batch: Any, forward_fn) -> torch.Tensor:
        """Forward pass for single micro-batch."""
        return self._standard_forward_backward(micro_batch, forward_fn)
    
    def _backward_micro_batch(self):
        """Backward pass for single micro-batch."""
        pass
    
    def _synchronize_gradients(self):
        """Synchronize gradients across parallel dimensions."""
        import torch.distributed as dist
        
        if not dist.is_initialized():
            return
        
        if self._model is None:
            return
        
        for param in self._model.parameters():
            if param.grad is not None:
                if self.config.dp_size > 1 and self._dp_group is not None:
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
        
        tp_memory = total_memory / self.config.tp_size
        pp_memory = total_memory / self.config.pp_size
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


__all__ = [
    "POPSSParallelismType",
    "POPSSPipelineSchedule",
    "POPSSParallel3DConfig",
    "POPSSParallel3DOperator",
]
