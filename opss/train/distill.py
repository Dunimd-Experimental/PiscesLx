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
Knowledge Distillation Training Operator

This module provides a comprehensive training operator for knowledge distillation.
The Teacher model is injected by the training engine, allowing maximum flexibility
for users to choose any teacher model.

Key Features:
    - Teacher-agnostic design (teacher injected by training engine)
    - Progressive distillation strategies
    - Mixed precision training
    - Gradient checkpointing
    - Distributed training support
    - Comprehensive logging and metrics

Distillation Strategies:
    - Logits-only: For remote API teachers
    - Full: Logits + hidden states + attentions
    - Progressive: Layer-by-layer knowledge transfer
    - Contrastive: For text-only teachers

Usage:
    from opss.train.distill import (
        POPSSDistillationConfig,
        POPSSDistillationOperator,
    )
    from opss.train.distill_provider import (
        POPSSTeacherProviderFactory,
        POPSSTeacherConfig,
    )
    
    # User creates and configures the teacher provider
    teacher_config = POPSSTeacherConfig(
        provider_type="local",
        model_path="./models/teacher-7b"
    )
    teacher_provider = POPSSTeacherProviderFactory.create(teacher_config)
    
    # Training engine injects teacher to distillation operator
    config = POPSSDistillationConfig(
        student_model_path="./models/student-7b",
        output_dir="./outputs/distilled"
    )
    
    operator = POPSSDistillationOperator()
    result = operator.execute({
        "config": config,
        "teacher_provider": teacher_provider,  # Injected by training engine
    })
"""

import os
import sys
import json
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_work_dir
from configs.version import VERSION
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus

from .distill_provider import POPSSTeacherProvider
from .distill_loss import (
    POPSSDistillationLossConfig,
    POPSSDistillationLoss,
)


@dataclass
class POPSSDistillationConfig:
    """Configuration for knowledge distillation training.
    
    Note: Teacher model is NOT defined here. The training engine injects
    the teacher provider at runtime, allowing users to choose any teacher.
    
    Attributes:
        student_model_path: Path to student model.
        output_dir: Output directory for checkpoints.
        
        train_data: Path to training data.
        val_data: Path to validation data.
        
        batch_size: Global batch size.
        micro_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Gradient accumulation steps.
        
        learning_rate: Learning rate.
        min_lr_ratio: Minimum LR ratio for decay.
        warmup_steps: Warmup steps.
        max_steps: Maximum training steps.
        max_grad_norm: Gradient clipping norm.
        
        distillation_mode: Distillation mode (logits/full/progressive/contrastive).
        temperature: Temperature for soft labels.
        
        alpha: Logits loss weight.
        beta: Hidden state loss weight.
        gamma: Attention loss weight.
        delta: Layer-wise loss weight.
        epsilon: Task loss weight.
        
        output_hidden_states: Whether to request hidden states from teacher.
        output_attentions: Whether to request attentions from teacher.
        
        use_fp16: Use FP16 mixed precision.
        use_bf16: Use BF16 mixed precision.
        use_gradient_checkpointing: Enable gradient checkpointing.
        
        checkpoint_interval: Steps between checkpoints.
        eval_interval: Steps between evaluations.
        log_interval: Steps between logging.
        
        save_total_limit: Maximum checkpoints to keep.
        
        local_rank: Local rank for distributed training.
        world_size: World size for distributed training.
        
        max_seq_length: Maximum sequence length.
        ignore_index: Index to ignore in loss.
    """
    
    student_model_path: str = ".pisceslx/ckpt"
    output_dir: str = ".pisceslx/distill_output"
    
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
    
    distillation_mode: str = "full"
    temperature: float = 2.0
    
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.3
    delta: float = 0.2
    epsilon: float = 1.0
    
    output_hidden_states: bool = True
    output_attentions: bool = True
    
    use_fp16: bool = False
    use_bf16: bool = True
    use_gradient_checkpointing: bool = True
    
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 10
    
    save_total_limit: int = 3
    
    local_rank: int = 0
    world_size: int = 1
    master_port: int = 29502
    
    max_seq_length: int = 4096
    ignore_index: int = -100
    
    def __post_init__(self):
        if self.use_fp16 and self.use_bf16:
            self.use_bf16 = False


class POPSSDistillationDataset(Dataset):
    """Dataset for knowledge distillation training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_seq_length: int = 4096,
        ignore_index: int = -100,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ignore_index = ignore_index
        self.samples = self._load_data(data_path)
        self._LOG = PiscesLxLogger(
            "PiscesLx.Distill.Dataset",
            file_path=get_log_file("PiscesLx.Distill.Dataset"),
            enable_file=True,
        )
        self._LOG.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        samples = []
        if not os.path.exists(data_path):
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


class _DistillationOperatorImpl(PiscesLxOperatorInterface):
    """Knowledge distillation training operator implementation.
    
    The teacher model is injected by the training engine at runtime,
    allowing maximum flexibility for users to choose any teacher.
    """
    
    def __init__(self):
        super().__init__()
        self._name = "distillation.training"
        self._version = VERSION
        self.type = "training"
        self._LOG = PiscesLxLogger(
            "PiscesLx.Distill.Operator",
            file_path=get_log_file("PiscesLx.Distill.Operator"),
            enable_file=True,
        )
        
        self.teacher_provider: Optional[POPSSTeacherProvider] = None
        self.student_model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[LRScheduler] = None
        self.scaler: Optional[GradScaler] = None
        
        self.config: Optional[POPSSDistillationConfig] = None
        self.global_step: int = 0
        self.best_loss: float = float('inf')
    
    def initialize(
        self,
        config: POPSSDistillationConfig,
        teacher_provider: POPSSTeacherProvider,
    ) -> None:
        """Initialize distillation training components.
        
        Args:
            config: Distillation configuration.
            teacher_provider: Teacher model provider injected by training engine.
                             User decides which teacher to use.
        """
        self.config = config
        self.teacher_provider = teacher_provider
        
        self._LOG.info("Initializing distillation training...")
        self._LOG.info("Teacher provider injected by training engine")
        
        self._init_student()
        self._init_tokenizer()
        self._init_optimizer()
        self._init_loss()
        
        if not self.teacher_provider.is_available():
            raise RuntimeError("Teacher provider is not available")
        
        self._LOG.info("Distillation training initialized successfully")
    
    def _init_student(self) -> None:
        """Initialize student model."""
        self._LOG.info(f"Loading student model from {self.config.student_model_path}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            dtype = torch.bfloat16 if self.config.use_bf16 else torch.float16
            
            self.student_model = AutoModelForCausalLM.from_pretrained(
                self.config.student_model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            
            device = torch.device(f"cuda:{self.config.local_rank}" if torch.cuda.is_available() else "cpu")
            self.student_model = self.student_model.to(device)
            
            if self.config.use_gradient_checkpointing:
                self.student_model.gradient_checkpointing_enable()
            
            self.student_model.train()
            
        except ImportError:
            raise ImportError("transformers library required for student model loading")
        
        self._LOG.info("Student model loaded successfully")
    
    def _init_tokenizer(self) -> None:
        """Initialize tokenizer."""
        try:
            from transformers import AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.student_model_path,
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        except ImportError:
            raise ImportError("transformers library required for tokenizer")
    
    def _init_optimizer(self) -> None:
        """Initialize optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            return self.config.min_lr_ratio + (1 - self.config.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda,
        )
        
        if self.config.use_fp16:
            self.scaler = GradScaler()
    
    def _init_loss(self) -> None:
        """Initialize distillation loss."""
        loss_config = POPSSDistillationLossConfig(
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            delta=self.config.delta,
            epsilon=self.config.epsilon,
            ignore_index=self.config.ignore_index,
        )
        self.distill_loss = POPSSDistillationLoss(loss_config)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step."""
        input_ids = batch["input_ids"].to(self.student_model.device)
        labels = batch["labels"].to(self.student_model.device)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_provider.get_all_outputs(input_ids)
        
        with autocast(enabled=self.config.use_fp16 or self.config.use_bf16, dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16):
            student_outputs = self.student_model(
                input_ids,
                output_hidden_states=self.config.output_hidden_states,
                output_attentions=self.config.output_attentions,
            )
            
            student_outputs_dict = {
                "logits": student_outputs.logits,
                "hidden_states": student_outputs.hidden_states if hasattr(student_outputs, 'hidden_states') else None,
                "attentions": student_outputs.attentions if hasattr(student_outputs, 'attentions') else None,
            }
            
            losses = self.distill_loss(teacher_outputs, student_outputs_dict, labels)
        
        loss = losses['total']
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(),
                self.config.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        os.makedirs(path, exist_ok=True)
        
        self.student_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        checkpoint = {
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config.__dict__,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, os.path.join(path, "training_state.pt"))
        
        self._LOG.info(f"Checkpoint saved to {path}")
    
    def execute(self, params: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Execute distillation training.
        
        Args:
            params: Dictionary containing:
                - config: POPSSDistillationConfig
                - teacher_provider: POPSSTeacherProvider (injected by training engine)
                
        Returns:
            Training result.
        """
        config = params.get("config")
        teacher_provider = params.get("teacher_provider")
        
        if config is None:
            config = POPSSDistillationConfig()
        
        if isinstance(config, dict):
            config = POPSSDistillationConfig(**config)
        
        if teacher_provider is None:
            raise ValueError(
                "teacher_provider is required. "
                "The training engine must inject the teacher provider. "
                "Example: operator.execute({'config': config, 'teacher_provider': my_teacher})"
            )
        
        self.initialize(config, teacher_provider)
        
        dataset = POPSSDistillationDataset(
            config.train_data,
            self.tokenizer,
            config.max_seq_length,
            config.ignore_index,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.micro_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        self._LOG.info(f"Starting distillation training for {config.max_steps} steps")
        
        start_time = time.time()
        accumulated_loss = 0.0
        
        while self.global_step < config.max_steps:
            for batch in dataloader:
                if self.global_step >= config.max_steps:
                    break
                
                losses = self.train_step(batch)
                accumulated_loss += losses['total']
                
                self.global_step += 1
                
                if self.global_step % config.log_interval == 0:
                    avg_loss = accumulated_loss / config.log_interval
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed
                    
                    self._LOG.info(
                        f"Step {self.global_step}/{config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Speed: {steps_per_sec:.2f} steps/s | "
                        f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                    )
                    accumulated_loss = 0.0
                
                if self.global_step % config.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(
                        config.output_dir,
                        f"checkpoint-{self.global_step}"
                    )
                    self.save_checkpoint(checkpoint_path)
        
        final_checkpoint_path = os.path.join(config.output_dir, "final")
        self.save_checkpoint(final_checkpoint_path)
        
        total_time = time.time() - start_time
        
        self._LOG.info(f"Distillation training completed in {total_time:.2f} seconds")
        
        if self.teacher_provider is not None:
            self.teacher_provider.close()
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            data={
                "final_step": self.global_step,
                "training_time": total_time,
                "output_path": final_checkpoint_path,
            }
        )


class POPSSDistillationOperator:
    """Facade for distillation training operator.
    
    The teacher model is injected by the training engine at runtime.
    Users have full control over which teacher to use.
    
    Example:
        # User creates teacher provider
        from opss.train.distill_provider import POPSSTeacherProviderFactory, POPSSTeacherConfig
        
        teacher_config = POPSSTeacherConfig(
            provider_type="local",
            model_path="./models/deepseek-r1-7b"
        )
        teacher = POPSSTeacherProviderFactory.create(teacher_config)
        
        # Training engine injects teacher to operator
        config = POPSSDistillationConfig(student_model_path="./models/student")
        operator = POPSSDistillationOperator()
        result = operator.execute({
            "config": config,
            "teacher_provider": teacher,  # Injected here
        })
    """
    
    def __init__(self):
        self._impl = _DistillationOperatorImpl()
    
    def execute(self, params: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Execute distillation training with injected teacher.
        
        Args:
            params: Must contain 'teacher_provider' key with POPSSTeacherProvider instance.
        """
        return self._impl.execute(params)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return self._impl.train_step(batch)
    
    def save_checkpoint(self, path: str) -> None:
        self._impl.save_checkpoint(path)


__all__ = [
    "POPSSDistillationConfig",
    "POPSSDistillationDataset",
    "POPSSDistillationOperator",
]
