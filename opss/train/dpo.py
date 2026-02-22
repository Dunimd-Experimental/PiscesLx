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
Direct Preference Optimization (DPO) Operator Implementation

Complete implementation of DPO training as a standardized operator.
Based on the original PiscesL1 DPO training pipeline.

DPO Algorithm Overview:
    DPO directly optimizes language models to align with human preferences
    without requiring an explicit reward model. It uses preference pairs
    (chosen vs rejected responses) to train the policy model.

    The core loss function is:
    L_DPO = -E_{(x,y_w,y_l)~D}[log σ(β * (log π(y_w|x) - log π(y_l|x)))]
    
    where:
    - x: input prompt
    - y_w: chosen (preferred) response
    - y_l: rejected (less preferred) response
    - β: temperature parameter controlling alignment strength
    - π: policy model

Features:
    - Full DPO training pipeline with preference data
    - Support for multiple loss types (sigmoid, hinge, IPO)
    - Reference model for KL regularization
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation and checkpointing
    - Distributed training support
    - Comprehensive logging and metrics

Usage:
    from ops.train.dpo import DPOTrainingConfig, DPOTrainingOperator
    
    config = DPOTrainingConfig(
        model_path="./checkpoints/sft_model",
        ref_model_path="./checkpoints/base_model",
        beta=0.1,
        learning_rate=5e-7
    )
    
    operator = DPOTrainingOperator()
    result = operator.execute({
        "model": policy_model,
        "ref_model": reference_model,
        "tokenizer": tokenizer,
        "train_data_path": "./data/preferences.jsonl"
    })
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
import torch.nn.functional as F
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
class POPSSDPOTrainingConfig:
    """
    Configuration for DPO training.
    
    Attributes:
        model_path: Path to the policy model checkpoint
        ref_model_path: Path to the reference model (for KL regularization)
        output_dir: Directory for saving outputs and checkpoints
        
        train_data: Path to training preference data (JSONL format)
        val_data: Path to validation preference data
        
        batch_size: Global batch size across all GPUs
        micro_batch_size: Batch size per GPU
        gradient_accumulation_steps: Number of steps to accumulate gradients
        
        learning_rate: Initial learning rate for optimizer
        min_lr_ratio: Minimum learning rate ratio for decay
        warmup_steps: Number of warmup steps for learning rate scheduler
        max_steps: Maximum number of training steps
        max_grad_norm: Maximum gradient norm for clipping
        
        beta: Temperature parameter for DPO loss (higher = stronger alignment)
        label_smoothing: Label smoothing factor for loss computation
        loss_type: Type of DPO loss function (sigmoid, hinge, ipo)
        
        use_fp16: Whether to use FP16 mixed precision training
        use_bf16: Whether to use BF16 mixed precision training
        
        use_gradient_checkpointing: Enable gradient checkpointing to save memory
        checkpoint_interval: Steps between saving checkpoints
        eval_interval: Steps between evaluation
        
        save_total_limit: Maximum number of checkpoints to keep
        
        local_rank: Local rank for distributed training
        world_size: Total number of processes for distributed training
        master_port: Port for distributed training communication
        
        max_prompt_length: Maximum length of input prompts
        max_response_length: Maximum length of responses
    """
    
    model_path: str = "./checkpoints/sft_model"
    ref_model_path: str = "./checkpoints/ruchbah"
    output_dir: str = "./checkpoints/dpo_output"
    
    train_data: str = "./data/preferences.jsonl"
    val_data: str = "./data/val_preferences.jsonl"
    
    batch_size: int = 4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    
    learning_rate: float = 5e-7
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 2000
    max_grad_norm: float = 1.0
    
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo
    
    use_fp16: bool = False
    use_bf16: bool = True
    
    use_gradient_checkpointing: bool = True
    checkpoint_interval: int = 500
    eval_interval: int = 250
    
    save_total_limit: int = 3
    
    local_rank: int = 0
    world_size: int = 1
    master_port: int = 29501
    
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure only one mixed precision format is used
        if self.use_fp16 and self.use_bf16:
            self.use_bf16 = False


class POPSSDPODataset(Dataset):
    """
    Dataset for DPO training with preference pairs.
    
    This dataset loads preference data in JSONL format where each sample
    contains a prompt, chosen response, and rejected response.
    
    Data Format (JSONL):
        {
            "prompt": "User query or instruction",
            "chosen": "Preferred assistant response",
            "rejected": "Less preferred assistant response"
        }
    
    Attributes:
        tokenizer: Tokenizer for encoding text
        max_prompt_length: Maximum prompt sequence length
        max_response_length: Maximum response sequence length
        samples: List of loaded preference samples
        logger: Logger instance
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_prompt_length: int = 1024,
        max_response_length: int = 1024,
    ):
        """
        Initialize DPO dataset.
        
        Args:
            data_path: Path to preference data (JSONL format)
            tokenizer: Tokenizer for encoding text
            max_prompt_length: Maximum prompt length
            max_response_length: Maximum response length
        """
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        
        self.samples = self._load_data(data_path)
        self._LOG = get_logger("pisceslx.ops.train.dpo.dataset")
        self._LOG.info(f"Loaded {len(self.samples)} preference samples from {data_path}")
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Load preference data from JSONL file.
        
        Args:
            data_path: Path to JSONL file
            
        Returns:
            List of preference samples
        """
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
                    # Validate required fields
                    if all(key in sample for key in ['prompt', 'chosen', 'rejected']):
                        samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single preference sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing tokenized tensors for both chosen and rejected responses
        """
        sample = self.samples[idx]
        prompt = sample['prompt']
        chosen_response = sample['chosen']
        rejected_response = sample['rejected']
        
        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        prompt_ids = prompt_encoding["input_ids"].squeeze(0)
        prompt_mask = prompt_encoding["attention_mask"].squeeze(0)
        
        # Tokenize chosen response
        chosen_encoding = self.tokenizer(
            chosen_response,
            max_length=self.max_response_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        chosen_ids = chosen_encoding["input_ids"].squeeze(0)
        chosen_mask = chosen_encoding["attention_mask"].squeeze(0)
        
        # Tokenize rejected response
        rejected_encoding = self.tokenizer(
            rejected_response,
            max_length=self.max_response_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_ids = rejected_encoding["input_ids"].squeeze(0)
        rejected_mask = rejected_encoding["attention_mask"].squeeze(0)
        
        # Combine prompt with responses
        chosen_input_ids = torch.cat([prompt_ids, chosen_ids], dim=0)
        chosen_attention_mask = torch.cat([prompt_mask, chosen_mask], dim=0)
        chosen_labels = chosen_input_ids.clone()
        # Mask prompt tokens in labels (only compute loss on response tokens)
        chosen_labels[:len(prompt_ids)] = -100
        
        rejected_input_ids = torch.cat([prompt_ids, rejected_ids], dim=0)
        rejected_attention_mask = torch.cat([prompt_mask, rejected_mask], dim=0)
        rejected_labels = rejected_input_ids.clone()
        # Mask prompt tokens in labels
        rejected_labels[:len(prompt_ids)] = -100
        
        return {
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels,
        }


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int, dim: int = -1) -> torch.Tensor:
    """
    Pad tensor to specified length.
    
    Args:
        tensor: Input tensor to pad
        length: Target length
        pad_value: Value to use for padding
        dim: Dimension to pad along
        
    Returns:
        Padded tensor
    """
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype)], dim=dim)


class POPSSDPOLoggingCallback:
    """
    Logging callback for DPO training.
    
    Tracks and logs training metrics including loss, rewards, and accuracy.
    
    Attributes:
        logger: Logger instance for output
        start_time: Training start timestamp
    """
    
    def __init__(self, logger):
        """
        Initialize logging callback.
        
        Args:
            logger: Logger instance
        """
        self._LOG = logger
        self.start_time = time.time()
        
    def on_step_end(self, step: int, loss: float, chosen_rewards: float, rejected_rewards: float, 
                   accuracy: float, learning_rate: float):
        """
        Log training step information.
        
        Args:
            step: Current training step
            loss: Current loss value
            chosen_rewards: Average reward for chosen responses
            rejected_rewards: Average reward for rejected responses
            accuracy: Preference prediction accuracy
            learning_rate: Current learning rate
        """
        if step % 10 == 0:
            elapsed_time = time.time() - self.start_time
            self._LOG.info(
                f"Step {step} | "
                f"Loss: {loss:.4f} | "
                f"Chosen Rewards: {chosen_rewards:.4f} | "
                f"Rejected Rewards: {rejected_rewards:.4f} | "
                f"Accuracy: {accuracy:.4f} | "
                f"LR: {learning_rate:.2e} | "
                f"Time: {elapsed_time:.2f}s"
            )


class POPSSDPOTrainingOperator(PiscesLxOperatorInterface):
    """
    Direct Preference Optimization Training Operator.
    
    Implements the complete DPO training algorithm as a standardized operator.
    DPO trains language models to align with human preferences using preference pairs
    without requiring an explicit reward model.
    
    Algorithm:
        The core DPO loss is computed as:
        L = -E[log σ(β * (log π(y_w|x) - log π(y_l|x) - (log π_ref(y_w|x) - log π_ref(y_l|x))))]
        
        where:
        - π: policy model (being trained)
        - π_ref: reference model (frozen)
        - y_w: chosen response
        - y_l: rejected response
        - β: temperature parameter
    
    Features:
        - Multiple loss types: sigmoid, hinge, IPO
        - Reference model for KL regularization
        - Mixed precision training (FP16/BF16)
        - Gradient accumulation
        - Distributed training support
        - Comprehensive metrics tracking
    
    Attributes:
        config: POPSSDPOTrainingConfig instance
        logger: Logger instance
        device: Training device (cuda/cpu)
        is_distributed: Whether using distributed training
    
    Example:
        >>> config = POPSSDPOTrainingConfig(beta=0.1, learning_rate=5e-7)
        >>> operator = POPSSDPOTrainingOperator(config)
        >>> result = operator.execute({
        ...     "model": policy_model,
        ...     "ref_model": ref_model,
        ...     "tokenizer": tokenizer,
        ...     "train_data_path": "./data/train.jsonl"
        ... })
    """
    
    def __init__(self, config: Optional[POPSSDPOTrainingConfig] = None):
        """
        Initialize DPO training operator.
        
        Args:
            config: DPO training configuration. If None, uses default config.
        """
        super().__init__()
        self.config = config or POPSSDPOTrainingConfig()
        self._LOG = get_logger("pisceslx.ops.train.dpo")
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_distributed = torch.distributed.is_initialized() if hasattr(torch.distributed, 'is_initialized') else False
        
        self._LOG.info(f"DPO Training Operator initialized on device: {self.device}")
    
    @property
    def name(self) -> str:
        """Return operator name."""
        return "dpo_training"
    
    @property
    def version(self) -> str:
        """Return operator version."""
        return VERSION
    
    @property
    def description(self) -> str:
        """Return operator description."""
        return "Direct Preference Optimization training operator for alignment"
    
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Return input schema for this operator.
        
        Returns:
            Dictionary describing required and optional inputs
        """
        return {
            "model": {
                "type": "torch.nn.Module",
                "required": True,
                "description": "Policy model to train"
            },
            "ref_model": {
                "type": "torch.nn.Module",
                "required": True,
                "description": "Reference model (frozen)"
            },
            "tokenizer": {
                "type": "Any",
                "required": True,
                "description": "Tokenizer for encoding text"
            },
            "train_data_path": {
                "type": "str",
                "required": True,
                "description": "Path to training preference data"
            },
            "val_data_path": {
                "type": "str",
                "required": False,
                "description": "Path to validation preference data"
            }
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Return output schema for this operator.
        
        Returns:
            Dictionary describing outputs
        """
        return {
            "model": {
                "type": "torch.nn.Module",
                "description": "Trained policy model"
            },
            "final_loss": {
                "type": "float",
                "description": "Final training loss"
            },
            "training_steps": {
                "type": "int",
                "description": "Number of training steps completed"
            }
        }
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate input parameters.
        
        Args:
            inputs: Dictionary of input parameters
            
        Returns:
            True if inputs are valid, False otherwise
        """
        required = ["model", "ref_model", "tokenizer", "train_data_path"]
        for key in required:
            if key not in inputs:
                self._LOG.error(f"Missing required input: {key}")
                return False
        
        # Validate data path exists
        if not os.path.exists(inputs["train_data_path"]):
            self._LOG.error(f"Training data not found: {inputs['train_data_path']}")
            return False
        
        return True
    
    def _compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.
        
        Args:
            policy_chosen_logps: Log probabilities from policy model for chosen responses
            policy_rejected_logps: Log probabilities from policy model for rejected responses
            reference_chosen_logps: Log probabilities from reference model for chosen responses
            reference_rejected_logps: Log probabilities from reference model for rejected responses
            
        Returns:
            Tuple of (loss tensor, metrics dictionary)
        """
        # Compute log ratios
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        
        # Compute logits for DPO loss
        logits = self.config.beta * (policy_logratios - reference_logratios)
        
        # Compute loss based on loss type
        if self.config.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.config.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        elif self.config.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.config.beta)) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        loss = losses.mean()
        
        # Compute metrics
        chosen_rewards = self.config.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.config.beta * (policy_rejected_logps - reference_rejected_logps)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        metrics = {
            "loss": loss.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "accuracy": accuracy.item(),
        }
        
        return loss, metrics
    
    def _get_batch_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities for a batch.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for computing log probs
            
        Returns:
            Log probabilities for each sequence
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("Model outputs must contain 'logits' for DPO log-prob computation")
        
        # Shift logits and labels for next token prediction
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        per_token_logps = torch.gather(
            log_probs,
            dim=2,
            index=labels.unsqueeze(2)
        ).squeeze(2)
        
        # Mask out padding tokens
        loss_mask = (labels != -100).float()
        per_token_logps = per_token_logps * loss_mask
        
        # Sum log probs over sequence
        return per_token_logps.sum(dim=1)
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute DPO training.
        
        Args:
            inputs: Dictionary containing:
                - model: Policy model to train
                - ref_model: Reference model (frozen)
                - tokenizer: Tokenizer
                - train_data_path: Path to training data
                - val_data_path: Path to validation data (optional)
            
        Returns:
            PiscesLxOperatorResult with training results
        """
        start_time = time.time()
        
        # Validate inputs
        if not self.validate_inputs(inputs):
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error="Invalid inputs",
                execution_time=time.time() - start_time
            )
        
        try:
            # Extract inputs
            model = inputs["model"].to(self.device)
            ref_model = inputs["ref_model"].to(self.device)
            tokenizer = inputs["tokenizer"]
            train_data_path = inputs["train_data_path"]
            val_data_path = inputs.get("val_data_path")
            
            # Freeze reference model
            for param in ref_model.parameters():
                param.requires_grad = False
            ref_model.eval()
            
            # Create dataset and dataloader
            train_dataset = POPSSDPODataset(
                train_data_path,
                tokenizer,
                self.config.max_prompt_length,
                self.config.max_response_length
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.micro_batch_size,
                shuffle=True,
                num_workers=0
            )
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
            
            # Setup learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_steps,
                eta_min=self.config.learning_rate * self.config.min_lr_ratio
            )
            
            # Setup mixed precision
            scaler = GradScaler() if self.config.use_fp16 else None
            
            # Setup logging callback
            logging_callback = POPSSDPOLoggingCallback(self._LOG)
            
            # Training loop
            model.train()
            global_step = 0
            total_loss = 0.0
            
            self._LOG.info("Starting DPO training...")
            
            while global_step < self.config.max_steps:
                for batch in train_loader:
                    if global_step >= self.config.max_steps:
                        break
                    
                    # Move batch to device
                    chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                    chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                    chosen_labels = batch["chosen_labels"].to(self.device)
                    
                    rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                    rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                    rejected_labels = batch["rejected_labels"].to(self.device)
                    
                    # Compute policy log probabilities
                    with torch.cuda.amp.autocast(enabled=self.config.use_fp16 or self.config.use_bf16):
                        policy_chosen_logps = self._get_batch_logps(
                            model, chosen_input_ids, chosen_attention_mask, chosen_labels
                        )
                        policy_rejected_logps = self._get_batch_logps(
                            model, rejected_input_ids, rejected_attention_mask, rejected_labels
                        )
                        
                        # Compute reference log probabilities
                        with torch.no_grad():
                            reference_chosen_logps = self._get_batch_logps(
                                ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels
                            )
                            reference_rejected_logps = self._get_batch_logps(
                                ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels
                            )
                        
                        # Compute DPO loss
                        loss, metrics = self._compute_dpo_loss(
                            policy_chosen_logps,
                            policy_rejected_logps,
                            reference_chosen_logps,
                            reference_rejected_logps
                        )
                        
                        # Scale loss for gradient accumulation
                        loss = loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    total_loss += loss.item()
                    
                    # Update weights after accumulation
                    if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.config.max_grad_norm
                        )
                        
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        optimizer.zero_grad()
                        scheduler.step()
                    
                    # Logging
                    if global_step % 10 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        logging_callback.on_step_end(
                            global_step,
                            metrics["loss"],
                            metrics["chosen_rewards"],
                            metrics["rejected_rewards"],
                            metrics["accuracy"],
                            current_lr
                        )
                    
                    global_step += 1
            
            self._LOG.info(f"DPO training completed. Total steps: {global_step}")
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                outputs={
                    "model": model,
                    "final_loss": total_loss / global_step if global_step > 0 else 0.0,
                    "training_steps": global_step
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self._LOG.error(f"DPO training failed: {str(e)}")
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )

__all__ = [
    "POPSSDPOTrainingConfig",
    "POPSSDPODataset",
    "POPSSDPOTrainingOperator",
    "POPSSDPOLoggingCallback"
]
