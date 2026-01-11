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
FSDP Training Runner for PiscesL1.

PyTorch-native distributed training without DeepSpeed compilation.

Usage:
    # Single GPU training
    python -m tools.train.fsdp_runner --model-path ./checkpoints/ruchbah --data-path ./data

    # Multi-GPU training
    torchrun --nproc_per_node=4 -m tools.train.fsdp_runner --model-path ./checkpoints/ruchbah

    # Custom configuration
    python -m tools.train.fsdp_runner --model-path ./checkpoints/ruchbah \
        --epochs 3 --batch-size 4 --lr 1e-4 \
        --mixed-precision bf16 \
        --sharding-strategy full_shard
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
from .fsdp_train import (
    PiscesLxFSDPTrainer,
    PiscesLxFSDPConfig,
    fsdp_launcher,
    create_fsdp_trainer,
)

logger = PiscesLxCoreLog("pisceslx.tools.train.fsdp_runner")


class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration."""
    
    def __init__(self, data_path: str, max_length: int = 2048):
        """Initialize dataset.
        
        Args:
            data_path: Path to text data or JSONL file
            max_length: Maximum sequence length
        """
        self.data = []
        self.max_length = max_length
        
        if os.path.isfile(data_path):
            if data_path.endswith('.jsonl'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        text = item.get('text', item.get('content', item))
                        if isinstance(text, str):
                            self.data.append(text)
            else:
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.data.append(line.strip())
        else:
            logger.warning(f"Data path not found: {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = {
            'input_ids': text,
            'text': text,
        }
        return encoding


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PiscesL1 FSDP Training Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single GPU
    python -m tools.train.fsdp_runner --model-path ./checkpoints/ruchbah --data-path ./data

    # Multi-GPU (using torchrun)
    torchrun --nproc_per_node=4 -m tools.train.fsdp_runner --model-path ./checkpoints/ruchbah

    # Custom training
    python -m tools.train.fsdp_runner --model-path ./checkpoints/ruchbah \
        --epochs 3 --batch-size 4 --lr 1e-4 \
        --mixed-precision bf16 \
        --sharding-strategy full_shard

Environment Variables:
    CUDA_VISIBLE_DEVICES: GPU selection
    SLURM_PROCID: Process ID (for SLURM cluster)
    SLURM_NTASKS: Number of tasks
        """
    )
    
    parser.add_argument("--model-path", type=str, default="./checkpoints/ruchbah",
                        help="Path to PiscesL1 model checkpoint")
    parser.add_argument("--data-path", type=str, default="./data/train.txt",
                        help="Path to training data")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to training config JSON")
    
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Maximum gradient norm")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Learning rate warmup steps")
    
    parser.add_argument("--mixed-precision", type=str, default="bf16",
                        choices=["fp16", "bf16", "none"],
                        help="Mixed precision training")
    parser.add_argument("--sharding-strategy", type=str, default="full_shard",
                        choices=["full_shard", "shard_grad_op", "no_shard"],
                        help="FSDP sharding strategy")
    parser.add_argument("--backward-prefetch", type=str, default="pre_forward",
                        choices=["pre_forward", "post_forward"],
                        help="Backward prefetch strategy")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Enable CPU offload for parameters")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing")
    
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd", "galore"],
                        help="Optimizer type")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/fsdp",
                        help="Checkpoint save directory")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                        help="Steps between checkpoints")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Steps between logs")
    parser.add_argument("--eval-interval", type=int, default=1000,
                        help="Steps between evaluations")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    
    parser.add_argument("--test", action="store_true",
                        help="Run quick test (10 steps)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark (100 steps)")
    
    parser.add_argument("--local-rank", type=int, default=0,
                        help="Local rank (for torchrun)")
    parser.add_argument("--world-size", type=int, default=1,
                        help="World size (for torchrun)")
    
    return parser.parse_args()


def create_model(model_path: str) -> nn.Module:
    """Create PiscesL1 model.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        RuchbahModel instance
    """
    try:
        from model.config import RuchbahConfig
        from model.modeling import RuchbahModel
        
        logger.info(f"Loading model from: {model_path}")
        
        config = RuchbahConfig.from_pretrained(model_path)
        model = RuchbahModel.from_pretrained(model_path)
        
        logger.success("Model loaded successfully")
        return model
        
    except ImportError as e:
        logger.error(f"Failed to import model: {e}")
        raise


def create_dataset(data_path: str) -> Dataset:
    """Create training dataset.
    
    Args:
        data_path: Path to training data
        
    Returns:
        Dataset instance
    """
    if os.path.exists(data_path):
        dataset = SimpleTextDataset(data_path)
        logger.info(f"Dataset created with {len(dataset)} samples")
        return dataset
    else:
        logger.warning(f"Data path not found: {data_path}")
        logger.info("Creating dummy dataset for testing...")
        return SimpleTextDataset("dummy")


def run_single_gpu_training(args) -> None:
    """Run training on single GPU."""
    logger.info("Starting single-GPU training")
    
    config = PiscesLxFSDPConfig(
        sharding_strategy=args.sharding_strategy,
        mixed_precision=args.mixed_precision,
        backward_prefetch=args.backward_prefetch,
        cpu_offload=args.cpu_offload,
        gradient_checkpointing=args.gradient_checkpointing,
        optimizer_type=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    
    trainer = create_fsdp_trainer(config)
    
    model = create_model(args.model_path)
    dataset = create_dataset(args.data_path)
    
    if len(dataset) == 0:
        logger.warning("Empty dataset, skipping training")
        return
    
    trainer.setup(
        model=model,
        train_dataset=dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        local_rank=0,
        world_size=1,
    )
    
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    num_steps = args.epochs * len(trainer.train_loader)
    
    if args.test:
        num_steps = 10
        logger.info("Running test (10 steps)")
    elif args.benchmark:
        num_steps = 100
        logger.info("Running benchmark (100 steps)")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            trainer.train_epoch(epoch)
            
            if (epoch + 1) % 1 == 0 and args.checkpoint_dir:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                trainer.save_checkpoint(args.checkpoint_dir, epoch)
            
            if num_steps >= len(trainer.train_loader) * (epoch + 1):
                break
                
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if args.checkpoint_dir:
            trainer.save_checkpoint(args.checkpoint_dir, "interrupted")
    
    trainer.cleanup()


def run_distributed_training(args) -> None:
    """Run distributed training on multiple GPUs."""
    logger.info(f"Starting distributed training with {args.world_size} processes")
    
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    
    config = PiscesLxFSDPConfig(
        sharding_strategy=args.sharding_strategy,
        mixed_precision=args.mixed_precision,
        backward_prefetch=args.backward_prefetch,
        cpu_offload=args.cpu_offload,
        gradient_checkpointing=args.gradient_checkpointing,
        optimizer_type=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    
    if world_size == 1:
        run_single_gpu_training(args)
        return
    
    if local_rank == 0:
        logger.info(f"Distributed training config:")
        logger.info(f"  Sharding: {args.sharding_strategy}")
        logger.info(f"  Precision: {args.mixed_precision}")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Learning rate: {args.lr}")
    
    fsdp_launcher(
        model=None,
        train_dataset=None,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )


def main():
    """Main entry point."""
    args = parse_args()
    
    torch.manual_seed(args.seed)
    
    print(f"\n{'='*60}")
    print("PiscesL1 FSDP Training Runner")
    print(f"{'='*60}")
    
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if world_size > 1 and "SLURM_PROCID" not in os.environ:
        print(f"\n{world_size} GPUs detected.")
        print(f"Use: torchrun --nproc_per_node={world_size} -m tools.train.fsdp_runner ...")
        print(f"\nOr run single-GPU training:")
    
    args.world_size = world_size
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Sharding: {args.sharding_strategy}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"\n{'='*60}\n")
    
    if args.test:
        logger.info("Test mode enabled")
    elif args.benchmark:
        logger.info("Benchmark mode enabled")
    
    try:
        if torch.cuda.is_available():
            run_single_gpu_training(args)
        else:
            logger.warning("CUDA not available, using CPU")
            run_single_gpu_training(args)
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
