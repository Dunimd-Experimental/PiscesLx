#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
Supervised Fine-Tuning (SFT) training pipeline for PiscesL1.

This module implements complete SFT training with:
- Standard language modeling loss
- Multi-modal instruction tuning support
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16/BF16)
- Learning rate scheduling
- Checkpoint saving and resuming
- Logging and evaluation
"""

import os
import sys
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

import dms_core
PiscesLxCoreLog = dms_core.log.get_logger
from utils import PiscesLxCoreConfigManager, PiscesLxCoreCheckpointManager

logger = PiscesLxCoreLog("pisceslx.tools.train.sft")


@dataclass
class PiscesL1SFTConfig:
    """SFT training configuration."""
    
    model_path: str = "./checkpoints/ruchbah"
    output_dir: str = "./checkpoints/sft_output"
    
    train_data: str = "./data/train.jsonl"
    val_data: str = "./data/val.jsonl"
    
    batch_size: int = 4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    
    learning_rate: float = 1e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 10000
    max_grad_norm: float = 1.0
    
    use_fp16: bool = False
    use_bf16: bool = True
    
    use_gradient_checkpointing: bool = True
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    
    save_total_limit: int = 3
    
    local_rank: int = 0
    world_size: int = 1
    master_port: int = 29500
    
    def __post_init__(self):
        if self.use_fp16 and self.use_bf16:
            self.use_bf16 = False


class PiscesL1SFTDataset(Dataset):
    """Dataset for SFT training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_seq_length: int = 4096,
        ignore_index: int = -100,
    ):
        """Initialize SFT dataset.
        
        Args:
            data_path: Path to training data (JSONL format).
            tokenizer: Tokenizer for encoding text.
            max_seq_length: Maximum sequence length.
            ignore_index: Token index to ignore in loss computation.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ignore_index = ignore_index
        
        self.samples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load training data from JSONL file."""
        samples = []
        
        if not os.path.exists(data_path):
            logger.warning(f"Data file not found: {data_path}")
            return samples
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        return samples
    
    def _format_sample(self, sample: Dict[str, Any]) -> str:
        """Format a single sample for training."""
        messages = sample.get("messages", [])
        
        if not messages:
            return sample.get("text", "")
        
        formatted_text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted_text += f"System: {content}\n"
            elif role == "user":
                formatted_text += f"User: {content}\n"
            elif role == "assistant":
                formatted_text += f"Assistant: {content}\n"
        
        return formatted_text.strip()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        sample = self.samples[idx]
        text = self._format_sample(sample)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PiscesL1SFTTrainer:
    """SFT trainer for PiscesL1 model."""
    
    def __init__(
        self,
        config: PiscesL1SFTConfig,
        model: nn.Module,
        tokenizer: Any,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
    ):
        """Initialize SFT trainer.
        
        Args:
            config: SFT training configuration.
            model: PiscesL1 model.
            tokenizer: Tokenizer for the model.
            optimizer: Optimizer for training.
            scheduler: Learning rate scheduler.
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.scaler = None
        if config.use_fp16 or config.use_bf16:
            self.scaler = GradScaler()
        
        self.global_step = 0
        self.total_loss = 0.0
        
        self.checkpoint_manager = PiscesLxCoreCheckpointManager()
        
        logger.info("PiscesL1SFTTrainer initialized")
    
    def _create_optimizer(self, config: PiscesL1SFTConfig) -> Optimizer:
        """Create optimizer for training."""
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.max_grad_norm,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
        )
        
        return optimizer
    
    def _create_scheduler(self, config: PiscesL1SFTConfig, num_training_steps: int) -> LRScheduler:
        """Create learning rate scheduler."""
        warmup_steps = config.warmup_steps
        max_steps = num_training_steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                return max(
                    config.min_lr_ratio,
                    float(max_steps - step) / float(max(1, max_steps - warmup_steps))
                )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda,
        )
        
        return scheduler
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> Dict[str, Any]:
        """Run SFT training.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Optional validation dataset.
            
        Returns:
            Training metrics dictionary.
        """
        config = self.config
        
        logger.info("Starting SFT training...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        
        num_batches = len(train_loader)
        num_training_steps = num_batches * (config.max_steps // num_batches)
        
        if self.optimizer is None:
            self.optimizer = self._create_optimizer(config)
        
        if self.scheduler is None:
            self.scheduler = self._create_scheduler(config, num_training_steps)
        
        if config.use_gradient_checkpointing:
            self.model.set_gradient_checkpointing(True)
        
        self.model.train()
        
        accumulation_steps = config.gradient_accumulation_steps
        
        training_metrics = {
            "total_steps": 0,
            "avg_loss": 0.0,
            "learning_rate": 0.0,
            "grad_norm": 0.0,
        }
        
        start_time = time.time()
        
        for epoch in range(sys.maxsize):
            for batch_idx, batch in enumerate(train_loader):
                if self.global_step >= config.max_steps:
                    break
                
                self.global_step += 1
                
                batch = {k: v.cuda() for k, v in batch.items()}
                
                with autocast(
                    enabled=(config.use_fp16 or config.use_bf16),
                    dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
                ):
                    outputs = self.model(**batch)
                    loss = outputs.get("loss", outputs[0] if isinstance(outputs, tuple) else outputs)
                    
                    if isinstance(loss, dict):
                        total_loss = sum(v for v in loss.values() if isinstance(v, torch.Tensor))
                    else:
                        total_loss = loss
                    
                    loss = total_loss / accumulation_steps
                
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if self.global_step % accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        config.max_grad_norm,
                    )
                    
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                
                self.total_loss += total_loss.item()
                
                if self.global_step % 100 == 0:
                    avg_loss = self.total_loss / self.global_step
                    current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else config.learning_rate
                    
                    logger.info(
                        f"Step {self.global_step}/{config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Time: {time.time() - start_time:.1f}s"
                    )
                
                if self.global_step % config.checkpoint_interval == 0:
                    self._save_checkpoint()
                
                if val_dataset is not None and self.global_step % config.eval_interval == 0:
                    val_metrics = self.evaluate(val_dataset)
                    logger.info(f"Validation at step {self.global_step}: {val_metrics}")
                    self.model.train()
            
            logger.info(f"Epoch {epoch + 1} completed")
        
        training_metrics = {
            "total_steps": self.global_step,
            "avg_loss": self.total_loss / max(1, self.global_step),
            "total_time": time.time() - start_time,
        }
        
        self._save_checkpoint(is_final=True)
        
        logger.info(f"SFT training completed: {training_metrics}")
        
        return training_metrics
    
    def evaluate(self, val_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the model on validation set.
        
        Args:
            val_dataset: Validation dataset.
            
        Returns:
            Evaluation metrics dictionary.
        """
        self.model.eval()
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.cuda() for k, v in batch.items()}
                
                with autocast(
                    enabled=(self.config.use_fp16 or self.config.use_bf16),
                    dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16,
                ):
                    outputs = self.model(**batch)
                    loss = outputs.get("loss", outputs[0] if isinstance(outputs, tuple) else outputs)
                    
                    if isinstance(loss, dict):
                        total_loss += sum(v for v in loss.values() if isinstance(v, torch.Tensor)).item()
                    else:
                        total_loss += loss.item()
                    
                    num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        
        return {"eval_loss": avg_loss}
    
    def _save_checkpoint(self, is_final: bool = False) -> None:
        """Save training checkpoint."""
        config = self.config
        
        if config.local_rank != 0:
            return
        
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_name = f"checkpoint_{self.global_step}"
        if is_final:
            checkpoint_name = "final_model"
        
        checkpoint_path = output_dir / checkpoint_name
        
        try:
            self.model.save_pretrained(str(checkpoint_path))
            self.tokenizer.save_pretrained(str(checkpoint_path))
            
            torch.save({
                "global_step": self.global_step,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "config": config.__dict__,
            }, checkpoint_path / "training_state.pt")
            
            logger.success(f"Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def save_model(self, output_path: str) -> None:
        """Save the trained model.
        
        Args:
            output_path: Path to save the model.
        """
        if self.config.local_rank != 0:
            return
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))
            
            logger.success(f"Model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")


def create_sft_trainer(
    config: PiscesL1SFTConfig,
    model: nn.Module,
    tokenizer: Any,
) -> PiscesL1SFTTrainer:
    """Factory function to create SFT trainer.
    
    Args:
        config: SFT configuration.
        model: PiscesL1 model.
        tokenizer: Model tokenizer.
        
    Returns:
        Initialized SFT trainer.
    """
    trainer = PiscesL1SFTTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )
    
    return trainer


def sft_main(args):
    """Main entry point for SFT training."""
    config = PiscesL1SFTConfig.from_args(args)
    
    logger.info(f"SFT Config: {config}")
    
    if config.local_rank == 0:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    from transformers import AutoTokenizer
    from model.modeling import RuchbahModel
    from model.config import RuchbahConfig
    
    logger.info(f"Loading tokenizer from {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    logger.info(f"Loading model from {config.model_path}")
    model_config = RuchbahConfig.from_json(os.path.join(config.model_path, "config.json"))
    model = RuchbahModel(model_config)
    model = model.from_pretrained(config.model_path)
    
    model = model.cuda()
    
    if config.world_size > 1:
        model = DDP(model, device_ids=[config.local_rank])
    
    train_dataset = PiscesL1SFTDataset(
        data_path=config.train_data,
        tokenizer=tokenizer,
        max_seq_length=4096,
    )
    
    val_dataset = None
    if config.val_data and os.path.exists(config.val_data):
        val_dataset = PiscesL1SFTDataset(
            data_path=config.val_data,
            tokenizer=tokenizer,
            max_seq_length=4096,
        )
    
    trainer = create_sft_trainer(config, model, tokenizer)
    
    metrics = trainer.train(train_dataset, val_dataset)
    
    logger.success(f"SFT training finished: {metrics}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT training for PiscesL1")
    
    parser.add_argument("--model_path", type=str, default="./checkpoints/ruchbah")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft_output")
    parser.add_argument("--train_data", type=str, default="./data/train.jsonl")
    parser.add_argument("--val_data", type=str, default="./data/val.jsonl")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--local_rank", type=int, default=0)
    
    args = parser.parse_args()
    
    sft_main(args)
