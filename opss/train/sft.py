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
Supervised Fine-Tuning (SFT) Operator Implementation

Complete implementation of SFT training as a standardized operator.
Based on the original PiscesL1 SFT training pipeline.
"""

import os
import sys
import json
import time
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

from utils.dc import PiscesLxLogger
from configs.version import VERSION

from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus


@dataclass
class POPSSSFTTrainingConfig:
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
    
    max_seq_length: int = 4096
    ignore_index: int = -100
    
    def __post_init__(self):
        if self.use_fp16 and self.use_bf16:
            self.use_bf16 = False


class POPSSSFTDataset(Dataset):
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
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Train.Sft.Dataset", file_path=get_log_file("PiscesLx.Opss.Train.Sft.Dataset"), enable_file=True)
        self._LOG.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load training data from JSONL file."""
        samples = []
        
        if not os.path.exists(data_path):
            self._LOG.warning(f"Data file not found: {data_path}")
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


class _SFTTrainingOperatorImpl(PiscesLxOperatorInterface):
    """Complete SFT training operator implementation."""
    
    def __init__(self):
        super().__init__()
        self._name = "sft.training"
        self._version = VERSION
        self.type = "training"
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Train.Sft", file_path=get_log_file("PiscesLx.Opss.Train.Sft"), enable_file=True)
        
    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def description(self) -> str:
        return "Complete Supervised Fine-Tuning training operator with full PiscesL1 pipeline"
        
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "model": {"type": "torch.nn.Module", "required": True, "description": "PiscesL1 model to train"},
            "tokenizer": {"type": "object", "required": True, "description": "Model tokenizer"},
            "train_data_path": {"type": "str", "required": True, "description": "Path to training data (JSONL)"},
            "val_data_path": {"type": "str", "required": False, "description": "Path to validation data"},
            "config": {"type": "POPSSSFTTrainingConfig", "required": False, "description": "Training configuration"},
            "optimizer": {"type": "torch.optim.Optimizer", "required": False, "description": "Custom optimizer"},
            "scheduler": {"type": "torch.optim.lr_scheduler.LRScheduler", "required": False, "description": "Custom scheduler"}
        }
        
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "metrics": {"type": "dict", "description": "Training metrics and statistics"},
            "model_state": {"type": "dict", "description": "Final model state dict"},
            "checkpoint_path": {"type": "str", "description": "Path to saved checkpoint"},
            "training_history": {"type": "list", "description": "Detailed training history"}
        }
        
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        required_keys = ['model', 'tokenizer', 'train_data_path']
        for key in required_keys:
            if key not in inputs or inputs[key] is None:
                self._LOG.error(f"Missing required parameter: {key}")
                return False
                
        if not isinstance(inputs['model'], nn.Module):
            self._LOG.error("Model must be a torch.nn.Module")
            return False
            
        if not os.path.exists(inputs['train_data_path']):
            self._LOG.error(f"Training data not found: {inputs['train_data_path']}")
            return False
            
        if 'val_data_path' in inputs and inputs['val_data_path']:
            if not os.path.exists(inputs['val_data_path']):
                self._LOG.warning(f"Validation data not found: {inputs['val_data_path']}")
                
        return True
        
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """Execute complete SFT training pipeline."""
        start_time = time.time()
        
        try:
            if not self.validate_inputs(inputs):
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Invalid input parameters",
                    execution_time=time.time() - start_time
                )
            
            model = inputs['model']
            tokenizer = inputs['tokenizer']
            train_data_path = inputs['train_data_path']
            val_data_path = inputs.get('val_data_path')
            custom_config = inputs.get('config')
            custom_optimizer = inputs.get('optimizer')
            custom_scheduler = inputs.get('scheduler')
            
            if custom_config:
                config = custom_config
            else:
                config = POPSSSFTTrainingConfig(
                    train_data=train_data_path,
                    val_data=val_data_path or "",
                    output_dir=f"./checkpoints/sft_{int(time.time())}"
                )
            
            self._LOG.info(f"Starting SFT training with config: {config}")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            self._LOG.info(f"Using device: {device}")
            
            train_dataset = POPSSSFTDataset(
                data_path=config.train_data,
                tokenizer=tokenizer,
                max_seq_length=config.max_seq_length,
                ignore_index=config.ignore_index
            )
            
            val_dataset = None
            if config.val_data and os.path.exists(config.val_data):
                val_dataset = POPSSSFTDataset(
                    data_path=config.val_data,
                    tokenizer=tokenizer,
                    max_seq_length=config.max_seq_length,
                    ignore_index=config.ignore_index
                )
            
            trainer = self._create_trainer(
                config, model, tokenizer, 
                custom_optimizer, custom_scheduler
            )
            
            metrics = self._run_training(trainer, train_dataset, val_dataset, config, device)
            
            checkpoint_path = self._save_model(trainer, config)
            
            execution_time = time.time() - start_time
            
            result_data = {
                'metrics': metrics,
                'model_state': model.state_dict(),
                'checkpoint_path': checkpoint_path,
                'training_history': getattr(trainer, 'training_history', [])
            }
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=result_data,
                execution_time=execution_time,
                metadata={
                    'config': config.__dict__,
                    'final_loss': metrics.get('final_loss', 0.0),
                    'total_steps': metrics.get('total_steps', 0)
                }
            )
            
        except Exception as e:
            self._LOG.error(f"SFT training failed: {str(e)}", exc_info=True)
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _create_trainer(self, config, model, tokenizer, custom_optimizer=None, custom_scheduler=None):
        """Create SFT trainer with all components."""
        class SFTTrainer:
            def __init__(self, config, model, tokenizer, custom_optimizer=None, custom_scheduler=None):
                self.config = config
                self.model = model
                self.tokenizer = tokenizer
                self.optimizer = custom_optimizer
                self.scheduler = custom_scheduler
                self.scaler = None
                self.global_step = 0
                self.total_loss = 0.0
                self.training_history = []
                self._LOG = PiscesLxLogger("PiscesLx.Opss.Train.Sft.Trainer", file_path=get_log_file("PiscesLx.Opss.Train.Sft.Trainer"), enable_file=True)
                
                if config.use_fp16 or config.use_bf16:
                    self.scaler = GradScaler()
                
                if self.optimizer is None:
                    self.optimizer = self._create_optimizer()
                
                self.checkpoint_manager = None
                self._LOG.info("SFTTrainer initialized")
            
            def _create_optimizer(self):
                """Create AdamW optimizer with weight decay."""
                no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
                
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": self.config.max_grad_norm,
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
                    lr=self.config.learning_rate,
                )
                
                return optimizer
            
            def _create_scheduler(self, num_training_steps):
                """Create learning rate scheduler."""
                warmup_steps = self.config.warmup_steps
                max_steps = num_training_steps
                
                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    else:
                        return max(
                            self.config.min_lr_ratio,
                            float(max_steps - step) / float(max(1, max_steps - warmup_steps))
                        )
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda,
                )
                
                return scheduler
        
        return SFTTrainer(config, model, tokenizer, custom_optimizer, custom_scheduler)
    
    def _run_training(self, trainer, train_dataset, val_dataset, config, device):
        """Execute the main training loop."""
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
        
        if trainer.scheduler is None:
            trainer.scheduler = trainer._create_scheduler(num_training_steps)
        
        if config.use_gradient_checkpointing:
            if hasattr(trainer.model, 'set_gradient_checkpointing'):
                trainer.model.set_gradient_checkpointing(True)
        
        trainer.model.train()
        accumulation_steps = config.gradient_accumulation_steps
        
        training_metrics = {
            "total_steps": 0,
            "final_loss": 0.0,
            "learning_rate": config.learning_rate,
            "grad_norm": 0.0,
        }
        
        start_time = time.time()
        
        for epoch in range(sys.maxsize):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if trainer.global_step >= config.max_steps:
                    break
                
                trainer.global_step += 1
                
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with autocast(
                    enabled=(config.use_fp16 or config.use_bf16),
                    dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
                ):
                    outputs = trainer.model(**batch)
                    loss = outputs.get("loss", outputs[0] if isinstance(outputs, tuple) else outputs)
                    
                    if isinstance(loss, dict):
                        total_loss = sum(v for v in loss.values() if isinstance(v, torch.Tensor))
                    else:
                        total_loss = loss
                    
                    loss = total_loss / accumulation_steps
                
                if trainer.scaler is not None:
                    trainer.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if trainer.global_step % accumulation_steps == 0:
                    if trainer.scaler is not None:
                        trainer.scaler.unscale_(trainer.optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainer.model.parameters(),
                        config.max_grad_norm,
                    )
                    
                    if trainer.scaler is not None:
                        trainer.scaler.step(trainer.optimizer)
                        trainer.scaler.update()
                    else:
                        trainer.optimizer.step()
                    
                    if trainer.scheduler is not None:
                        trainer.scheduler.step()
                    
                    trainer.optimizer.zero_grad()
                
                trainer.total_loss += total_loss.item()
                epoch_loss += total_loss.item()
                epoch_batches += 1
                
                if trainer.global_step % 100 == 0:
                    avg_loss = trainer.total_loss / trainer.global_step
                    current_lr = trainer.scheduler.get_last_lr()[0] if trainer.scheduler else config.learning_rate
                    
                    trainer._LOG.info(
                        f"Step {trainer.global_step}/{config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Time: {time.time() - start_time:.1f}s"
                    )
                
                if trainer.global_step % config.checkpoint_interval == 0:
                    self._save_checkpoint(trainer, config, is_intermediate=True)
                
                if val_dataset is not None and trainer.global_step % config.eval_interval == 0:
                    val_metrics = self._evaluate(trainer, val_dataset, config, device)
                    trainer._LOG.info(f"Validation at step {trainer.global_step}: {val_metrics}")
                    trainer.model.train()
            
            if epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                trainer.training_history.append({
                    'epoch': epoch + 1,
                    'avg_loss': avg_epoch_loss,
                    'steps': trainer.global_step,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                trainer._LOG.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")
            
            if trainer.global_step >= config.max_steps:
                break
        
        training_metrics.update({
            "total_steps": trainer.global_step,
            "final_loss": trainer.total_loss / max(1, trainer.global_step),
            "total_time": time.time() - start_time,
        })
        
        return training_metrics
    
    def _evaluate(self, trainer, val_dataset, config, device):
        """Evaluate model on validation set."""
        trainer.model.eval()
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with autocast(
                    enabled=(config.use_fp16 or config.use_bf16),
                    dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
                ):
                    outputs = trainer.model(**batch)
                    loss = outputs.get("loss", outputs[0] if isinstance(outputs, tuple) else outputs)
                    
                    if isinstance(loss, dict):
                        total_loss += sum(v for v in loss.values() if isinstance(v, torch.Tensor)).item()
                    else:
                        total_loss += loss.item()
                    
                    num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        return {"eval_loss": avg_loss}
    
    def _save_checkpoint(self, trainer, config, is_intermediate=False):
        """Save training checkpoint."""
        if config.local_rank != 0:
            return
        
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_name = f"checkpoint_{trainer.global_step}"
        if not is_intermediate:
            checkpoint_name = "final_model"
        
        checkpoint_path = output_dir / checkpoint_name
        
        try:
            trainer.model.save_pretrained(str(checkpoint_path))
            trainer.tokenizer.save_pretrained(str(checkpoint_path))
            
            torch.save({
                "global_step": trainer.global_step,
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
                "config": config.__dict__,
            }, checkpoint_path / "training_state.pt")
            
            trainer._LOG.info(f"Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            trainer._LOG.error(f"Failed to save checkpoint: {e}")
    
    def _save_model(self, trainer, config):
        """Save final trained model."""
        if config.local_rank != 0:
            return ""
        
        output_path = Path(config.output_dir) / "final_model"
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            trainer.model.save_pretrained(str(output_path))
            trainer.tokenizer.save_pretrained(str(output_path))
            
            trainer._LOG.info(f"Model saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            trainer._LOG.error(f"Failed to save model: {e}")
            return ""


class POPSSSFTTrainingOperator(_SFTTrainingOperatorImpl):
    pass


__all__ = [
    "POPSSSFTTrainingConfig",
    "POPSSSFTDataset",
    "POPSSSFTTrainingOperator",
]
