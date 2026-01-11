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

"""
FSDP Distributed Training for PiscesL1 Ruchbah Model.

PyTorch-native fully sharded data parallelism with ZeRO-style
parameter sharding. No DeepSpeed compilation required.

Key Features:
- ZeRO-2: Gradient and optimizer state sharding
- ZeRO-3: Full parameter sharding
- Mixed precision training (FP16/BF16)
- Gradient checkpointing
- Local gradient accumulation
- Automatic device placement
"""

import os
import sys
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    wrap,
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
logger = PiscesLxCoreLog("pisceslx.tools.train.fsdp_train")


@dataclass
class PiscesLxFSDPConfig:
    """FSDP training configuration."""
    
    sharding_strategy: str = "full_shard"
    mixed_precision: str = "bf16"
    backward_prefetch: str = "pre_forward"
    cpu_offload: bool = False
    gradient_checkpointing: bool = True
    checkpoint_interval: int = 1000
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    def __post_init__(self):
        if self.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            logger.warning("BF16 not supported, falling back to FP16")
            self.mixed_precision = "fp16"
    
    @classmethod
    def from_args(cls, args: Any) -> "PiscesLxFSDPConfig":
        config_dict = {}
        
        if getattr(args, "fsdp_strategy", None):
            config_dict["sharding_strategy"] = args.fsdp_strategy
        
        if getattr(args, "mixed_precision", None):
            config_dict["mixed_precision"] = args.mixed_precision
        
        if getattr(args, "gradient_checkpointing", None):
            config_dict["gradient_checkpointing"] = args.gradient_checkpointing
        
        if getattr(args, "lr", None):
            config_dict["learning_rate"] = args.lr
        
        if getattr(args, "weight_decay", None):
            config_dict["weight_decay"] = args.weight_decay
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sharding_strategy": self.sharding_strategy,
            "mixed_precision": self.mixed_precision,
            "backward_prefetch": self.backward_prefetch,
            "cpu_offload": self.cpu_offload,
            "gradient_checkpointing": self.gradient_checkpointing,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }


def get_sharding_strategy(name: str) -> ShardingStrategy:
    """Map string name to ShardingStrategy."""
    strategies = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    return strategies.get(name, ShardingStrategy.FULL_SHARD)


def get_mixed_precision(dtype: str) -> Optional[MixedPrecision]:
    """Create MixedPrecision policy."""
    if dtype == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif dtype == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    return None


def get_backward_prefetch(name: str) -> Optional[BackwardPrefetch]:
    """Map string name to BackwardPrefetch."""
    prefs = {
        "pre_forward": BackwardPrefetch.PRE_FORWARD,
        "post_forward": BackwardPrefetch.POST_FORWARD,
    }
    return prefs.get(name)


class PiscesLxFSDPTrainer:
    """FSDP distributed trainer for PiscesL1.
    
    Provides PyTorch-native distributed training with:
    - Full ZeRO-2 sharding (gradient + optimizer state)
    - Optional ZeRO-3 (full parameter sharding)
    - Mixed precision training
    - Gradient checkpointing
    """
    
    def __init__(self, config: Optional[PiscesLxFSDPConfig] = None):
        """Initialize FSDP trainer.
        
        Args:
            config: FSDP configuration
        """
        self.config = config or PiscesLxFSDPConfig()
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.scaler = None
        self.local_rank = 0
        self.world_size = 1
        self.rank = 0
        self.device = None
        
        self._check_torch_distributed()
    
    def _check_torch_distributed(self):
        """Check if torch distributed is available."""
        if not dist.is_available():
            logger.warning("torch.distributed not available")
    
    def _setup_distributed(self, local_rank: int = 0, backend: str = "nccl"):
        """Setup distributed training environment.
        
        Args:
            local_rank: Local GPU rank
            backend: Distributed backend (nccl for GPU, gloo for CPU)
        """
        if "SLURM_PROCID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            local_rank = rank % torch.cuda.device_count()
            sys.argv = [sys.argv[0]] + sys.argv[1:]
        else:
            rank = local_rank
        
        self.local_rank = local_rank
        self.rank = rank
        
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
            backend = "nccl"
        else:
            self.device = torch.device("cpu")
            backend = "gloo"
        
        if self.world_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    world_size=self.world_size,
                    rank=self.rank,
                )
        
        logger.info(f"FSDP distributed setup: rank {self.rank}/{self.world_size}")
    
    def _create_auto_wrap_policy(self):
        """Create auto-wrap policy for FSDP."""
        try:
            from model.modeling import RuchbahTransformerBlock
            
            return transformer_auto_wrap_policy(
                {
                    RuchbahTransformerBlock,
                }
            )
        except ImportError:
            logger.warning("Using size-based wrap policy")
            return size_based_auto_wrap_policy(min_num_params=1e6)
    
    def _wrap_model(self, model: nn.Module) -> FSDP:
        """Wrap model with FSDP.
        
        Args:
            model: Original model
            
        Returns:
            FSDP-wrapped model
        """
        auto_wrap_policy = self._create_auto_wrap_policy()
        
        mixed_precision = get_mixed_precision(self.config.mixed_precision)
        backward_prefetch = get_backward_prefetch(self.config.backward_prefetch)
        
        cpu_offload = CPUOffload(offload_params=self.config.cpu_offload) if self.config.cpu_offload else None
        
        fsdp_model = FSDP(
            model,
            sharding_strategy=get_sharding_strategy(self.config.sharding_strategy),
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            backward_prefetch=backward_prefetch,
            cpu_offload=cpu_offload,
            device_id=self.device if self.device.type == "cuda" else None,
        )
        
        logger.info(f"Model wrapped with FSDP ({self.config.sharding_strategy})")
        
        return fsdp_model
    
    def _create_optimizer(self, model: FSDP) -> torch.optim.Optimizer:
        """Create optimizer for FSDP model.
        
        Args:
            model: FSDP model
            
        Returns:
            Optimizer instance
        """
        parameters = model.parameters()
        
        if self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
                eps=self.config.eps,
            )
        elif self.config.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "galore":
            from utils.optim.galore import RuchbahGalore
            optimizer = RuchbahGalore(
                parameters,
                lr=self.config.learning_rate,
                rank=64,
                update_interval=100,
            )
        else:
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        
        logger.info(f"Optimizer created: {self.config.optimizer_type}")
        
        return optimizer
    
    def _setup_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """Setup gradient scaler for FP16 training."""
        if self.config.mixed_precision == "fp16":
            scaler = torch.cuda.amp.GradScaler()
            logger.info("FP16 GradScaler initialized")
            return scaler
        return None
    
    def setup(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        num_workers: int = 4,
        local_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Setup training environment.
        
        Args:
            model: PiscesL1 model
            train_dataset: Training dataset
            batch_size: Batch size per GPU
            learning_rate: Learning rate
            num_workers: DataLoader workers
            local_rank: Local GPU rank
            world_size: Total number of processes
        """
        self.world_size = world_size
        self._setup_distributed(local_rank)
        
        self.config.learning_rate = learning_rate
        
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True,
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        self.model = self._wrap_model(model)
        self.optimizer = self._create_optimizer(self.model)
        self.scaler = self._setup_scaler()
        
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.success(f"FSDP trainer initialized on rank {self.rank}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            batch: Input batch
            
        Returns:
            Loss dict
        """
        self.model.train()
        
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        input_ids = batch.get("input_ids")
        labels = batch.get("labels", input_ids)
        
        loss_dict = {}
        
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.get("loss", outputs[0]) if isinstance(outputs, tuple) else outputs.get("loss", outputs)
                if isinstance(loss, dict):
                    total_loss = sum(loss.values())
                else:
                    total_loss = loss
            
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.get("loss", outputs[0]) if isinstance(outputs, tuple) else outputs.get("loss", outputs)
            if isinstance(loss, dict):
                total_loss = sum(loss.values())
            else:
                total_loss = loss
            
            total_loss.backward()
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        if isinstance(loss, dict):
            loss_dict = {k: v.item() for k, v in loss.items()}
            loss_dict["total"] = total_loss.item()
        else:
            loss_dict["loss"] = loss.item()
        
        return loss_dict
    
    def train_epoch(self, epoch: int = 0) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch in self.train_loader:
            loss_dict = self.train_step(batch)
            total_loss += loss_dict.get("total", loss_dict.get("loss", 0))
            num_batches += 1
            
            if self.rank == 0 and num_batches % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {num_batches}, Loss: {loss_dict.get('loss', 0):.4f}")
        
        epoch_time = time.time() - start_time
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        metrics = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "num_batches": num_batches,
            "time_seconds": epoch_time,
        }
        
        if self.rank == 0:
            logger.success(f"Epoch {epoch} complete: loss={avg_loss:.4f}, time={epoch_time:.2f}s")
        
        return metrics
    
    def save_checkpoint(self, checkpoint_dir: str, epoch: int = 0) -> None:
        """Save training checkpoint.
        
        Args:
            checkpoint_dir: Checkpoint directory
            epoch: Current epoch
        """
        if self.rank != 0:
            return
        
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_path)
        
        torch.save({
            "epoch": epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
        }, checkpoint_path / "training_state.pt")
        
        logger.success(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """Load training checkpoint.
        
        Args:
            checkpoint_dir: Checkpoint directory
            
        Returns:
            Starting epoch
        """
        checkpoint_path = Path(checkpoint_dir)
        
        if (checkpoint_path / "training_state.pt").exists():
            state = torch.load(checkpoint_path / "training_state.pt")
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            logger.info(f"Loaded checkpoint from epoch {state['epoch']}")
            return state["epoch"]
        
        return 0
    
    def cleanup(self) -> None:
        """Cleanup distributed training resources."""
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        logger.info("FSDP training cleanup complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def fsdp_launcher(
    model: nn.Module,
    train_dataset: Dataset,
    config: PiscesLxFSDPConfig,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    checkpoint_dir: Optional[str] = None,
):
    """Launch FSDP training on multiple GPUs.
    
    Args:
        model: PiscesL1 model
        train_dataset: Training dataset
        config: FSDP configuration
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        learning_rate: Learning rate
        checkpoint_dir: Checkpoint save directory
    """
    world_size = torch.cuda.device_count()
    
    mp.spawn(
        _fsdp_worker,
        nprocs=world_size,
        args=(
            model,
            train_dataset,
            config,
            epochs,
            batch_size,
            learning_rate,
            checkpoint_dir,
        ),
        daemon=False,
    )


def _fsdp_worker(
    local_rank: int,
    model: nn.Module,
    train_dataset: Dataset,
    config: PiscesLxFSDPConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_dir: Optional[str],
):
    """Worker function for FSDP training.
    
    Args:
        local_rank: Local GPU rank
        model: PiscesL1 model
        train_dataset: Training dataset
        config: FSDP configuration
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        learning_rate: Learning rate
        checkpoint_dir: Checkpoint directory
    """
    trainer = PiscesLxFSDPTrainer(config)
    
    trainer.setup(
        model=model,
        train_dataset=train_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        local_rank=local_rank,
        world_size=1,
    )
    
    start_epoch = 0
    if checkpoint_dir:
        start_epoch = trainer.load_checkpoint(checkpoint_dir)
    
    for epoch in range(start_epoch, epochs):
        trainer.train_epoch(epoch)
        
        if checkpoint_dir and (epoch + 1) % 1 == 0:
            trainer.save_checkpoint(checkpoint_dir, epoch)
    
    trainer.cleanup()


def create_fsdp_trainer(config: Optional[PiscesLxFSDPConfig] = None) -> PiscesLxFSDPTrainer:
    """Factory function to create FSDP trainer.
    
    Args:
        config: FSDP configuration
        
    Returns:
        FSDP trainer instance
    """
    return PiscesLxFSDPTrainer(config)


def benchmark_fsdp_train(
    trainer: PiscesLxFSDPTrainer,
    num_steps: int = 100,
) -> Dict[str, float]:
    """Benchmark FSDP training throughput.
    
    Args:
        trainer: FSDP trainer instance
        num_steps: Number of benchmark steps
        
    Returns:
        Benchmark results
    """
    import time
    
    trainer.model.train()
    
    times = []
    losses = []
    
    for i, batch in enumerate(trainer.train_loader):
        if i >= num_steps:
            break
        
        start = time.time()
        loss_dict = trainer.train_step(batch)
        elapsed = time.time() - start
        
        times.append(elapsed)
        losses.append(loss_dict.get("loss", 0))
        
        if trainer.rank == 0:
            logger.info(f"Step {i+1}/{num_steps}: loss={losses[-1]:.4f}, time={elapsed:.3f}s")
    
    avg_time = sum(times) / len(times)
    avg_loss = sum(losses) / len(losses)
    
    throughput = trainer.config.learning_rate / avg_time if avg_time > 0 else 0
    
    stats = {
        "num_steps": num_steps,
        "avg_time_per_step": avg_time,
        "avg_loss": avg_loss,
        "throughput": throughput,
    }
    
    if trainer.rank == 0:
        logger.success("Benchmark complete:")
        logger.success(f"  Avg step time: {avg_time*1000:.2f}ms")
        logger.success(f"  Avg loss: {avg_loss:.4f}")
        logger.success(f"  Throughput: {throughput:.2f} steps/sec")
    
    return stats
