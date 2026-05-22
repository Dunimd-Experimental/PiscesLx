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
        model_path=".pisceslx/ckpt",
        ref_model_path=".pisceslx/ckpt",
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
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

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
    
    model_path: str = ".pisceslx/ckpt"
    ref_model_path: str = ".pisceslx/ckpt"
    output_dir: str = ".pisceslx/ckpt"
    
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
    
    use_fp8: bool = False
    
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


@dataclass
class POPSSOPDConfig:
    """
    On-Policy Distillation (OPD) Configuration.

    OPD enables knowledge distillation from a teacher model to a student model
    using on-policy rollouts. The teacher generates outputs, and the student
    learns from the teacher's logits via KL divergence and cross-entropy.

    Attributes:
        teacher_model_path: Path to the teacher model checkpoint
        student_model_path: Path to the student model checkpoint
        opd_temperature: Distillation temperature for softening distributions
        opd_kl_weight: KL divergence weight in the combined loss
        opd_ce_weight: Cross-entropy weight in the combined loss
        opd_online: Whether teacher generates on-policy (vs using cached outputs)
        num_rollouts: Number of rollouts per prompt for advantage estimation
    """
    teacher_model_path: str = ""
    student_model_path: str = ""
    opd_temperature: float = 1.0
    opd_kl_weight: float = 0.5
    opd_ce_weight: float = 0.5
    opd_online: bool = True
    num_rollouts: int = 4

    def __post_init__(self):
        if self.opd_kl_weight < 0 or self.opd_ce_weight < 0:
            raise ValueError("KL and CE weights must be non-negative")
        if self.opd_temperature <= 0:
            raise ValueError("Distillation temperature must be positive")


@dataclass
class POPSSGRMConfig:
    """
    Generative Reward Model (GRM) Configuration.

    GRM wraps the base language model with a reward head that outputs scalar
    reward scores. It can be trained via preference pairs or used implicitly
    through DPO policy log-probability ratios.

    Attributes:
        grm_model_path: Path to the GRM model checkpoint
        grm_learning_rate: Learning rate for GRM training
        grm_hidden_size: Hidden size of the reward head MLP
        grm_num_layers: Number of MLP layers in the reward head
        grm_use_implicit_reward: Whether to use implicit reward from DPO policy
    """
    grm_model_path: str = ""
    grm_learning_rate: float = 1e-6
    grm_hidden_size: int = 4096
    grm_num_layers: int = 2
    grm_use_implicit_reward: bool = True


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
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Train",file_path=get_log_file("PiscesLx.Opss.Train"), enable_file=True)
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


def opd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    config: POPSSOPDConfig,
    advantages: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute On-Policy Distillation (OPD) loss combining KL divergence and cross-entropy.

    The OPD loss enables knowledge transfer from a teacher model to a student model
    by minimizing the divergence between their output distributions, optionally weighted
    by advantage estimates from a reward model.

    Args:
        teacher_logits: Logits from the teacher model [batch, seq_len, vocab]
        student_logits: Logits from the student model [batch, seq_len, vocab]
        teacher_log_probs: Log probabilities from teacher [batch]
        student_log_probs: Log probabilities from student [batch]
        config: OPD configuration
        advantages: Optional advantage weights for each sample [batch]

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    temp = config.opd_temperature

    teacher_soft = F.log_softmax(teacher_logits / temp, dim=-1)
    student_soft = F.softmax(student_logits / temp, dim=-1)

    kl_div = F.kl_div(
        teacher_soft,
        student_soft,
        log_target=False,
        reduction="batchmean",
    ) * (temp ** 2)

    ce_loss = -(teacher_log_probs * student_log_probs).mean()

    loss = config.opd_kl_weight * kl_div + config.opd_ce_weight * ce_loss

    if advantages is not None:
        advantages = advantages.detach()
        advantage_weight = advantages - advantages.mean()
        advantage_weight = advantage_weight / (advantage_weight.std() + 1e-8)
        advantage_weight = torch.sigmoid(advantage_weight / temp)
        weighted_kl = (
            F.kl_div(
                teacher_soft,
                student_soft,
                log_target=False,
                reduction="none",
            ).sum(dim=-1).mean(dim=-1)
            * advantage_weight
        ).mean() * (temp ** 2)
        weighted_ce = -(teacher_log_probs * student_log_probs * advantage_weight).mean()
        loss = config.opd_kl_weight * weighted_kl + config.opd_ce_weight * weighted_ce

    metrics = {
        "opd_loss": loss.item(),
        "kl_div": kl_div.item(),
        "ce_loss": ce_loss.item(),
    }

    return loss, metrics


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
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Train",file_path=get_log_file("PiscesLx.Opss.Train"), enable_file=True)
        
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
                    fp8_context = te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling(
                        margin=0, interval=1, fp8_format=Format.HYBRID, amax_history_len=1024, amax_compute_algo="max",
                    )) if self.config.use_fp8 else autocast(
                        enabled=self.config.use_fp16 or self.config.use_bf16
                    )
                    
                    with fp8_context:
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


class POPSSOPDTrainer:
    """
    On-Policy Distillation (OPD) Trainer.

    Implements knowledge distillation from a teacher model to a student model
    using on-policy rollouts. The teacher generates outputs, and the student
    learns from the teacher's logits via KL divergence, optionally weighted
    by advantage estimates from a reward model (GRM).

    Key Features:
        - Teacher generates on-policy rollouts for distillation
        - KL divergence + cross-entropy combined loss
        - Advantage-weighted distillation using GRM scores
        - Supports both online (teacher generates) and offline (cached) modes

    Example:
        >>> config = POPSSOPDConfig(
        ...     teacher_model_path=".pisceslx/teacher/ckpt",
        ...     student_model_path=".pisceslx/student/ckpt",
        ...     opd_temperature=2.0,
        ...     opd_kl_weight=0.7,
        ...     opd_ce_weight=0.3,
        ... )
        >>> trainer = POPSSOPDTrainer(
        ...     teacher_model=teacher,
        ...     student_model=student,
        ...     config=config,
        ... )
        >>> result = trainer.train(prompts=train_prompts, num_epochs=10)
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: Optional[POPSSOPDConfig] = None,
        teacher_optimizer: Optional[torch.optim.Optimizer] = None,
        student_optimizer: Optional[torch.optim.Optimizer] = None,
        tokenizer=None,
        grm_model: Optional["POPSSGenerativeRewardModel"] = None,
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config or POPSSOPDConfig()
        self.tokenizer = tokenizer
        self.grm_model = grm_model

        if student_optimizer is None:
            self.student_optimizer = torch.optim.AdamW(
                student_model.parameters(),
                lr=1e-6,
                weight_decay=0.01,
            )
        else:
            self.student_optimizer = student_optimizer

        self.teacher_optimizer = teacher_optimizer
        self.training_history = []

    def _generate_teacher_rollouts(
        self,
        prompt: str,
        num_rollouts: int,
    ) -> List[Dict[str, Any]]:
        """
        Generate on-policy rollouts using the teacher model.

        Args:
            prompt: Input prompt for generation
            num_rollouts: Number of rollouts to generate

        Returns:
            List of rollout dictionaries with logits, log_probs, and text
        """
        device = next(self.teacher_model.parameters()).device
        rollouts = []

        self.teacher_model.eval()
        with torch.no_grad():
            for _ in range(num_rollouts):
                if self.tokenizer:
                    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
                else:
                    input_ids = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device)

                all_logits = []
                generated_ids = input_ids.clone()
                past_key_values = None

                max_gen_tokens = 512
                for _ in range(max_gen_tokens):
                    if generated_ids.shape[1] > 1:
                        model_input = generated_ids[:, -1:]
                    else:
                        model_input = generated_ids

                    if hasattr(self.teacher_model, 'forward'):
                        outputs = self.teacher_model(
                            input_ids=model_input,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    else:
                        outputs = self.teacher_model(generated_ids)

                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None

                    all_logits.append(logits[:, -1:, :])
                    next_token_logits = logits[:, -1, :] / max(self.config.opd_temperature, 1.0)
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                    if self.tokenizer and next_token.item() == self.tokenizer.eos_token_id:
                        break

                if all_logits:
                    teacher_logits = torch.cat(all_logits, dim=1)
                else:
                    teacher_logits = torch.zeros(1, 1, 1, device=device)

                if self.tokenizer:
                    response_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                else:
                    response_text = "".join(chr(c) for c in generated_ids[0].tolist())

                full_text = prompt + response_text
                if self.tokenizer:
                    full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(device)
                else:
                    full_ids = torch.tensor([[ord(c) for c in full_text]], dtype=torch.long, device=device)

                with torch.no_grad():
                    full_outputs = self.teacher_model(full_ids)
                    teacher_logits_full = full_outputs.logits if hasattr(full_outputs, 'logits') else full_outputs[0]
                    teacher_log_probs_full = F.log_softmax(teacher_logits_full, dim=-1)
                    teacher_token_lps = teacher_log_probs_full[:, :-1, :].gather(2, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                    teacher_lp = teacher_token_lps.sum()

                rollout = {
                    "prompt": prompt,
                    "response": response_text,
                    "teacher_logits": teacher_logits,
                    "teacher_log_prob": teacher_lp,
                    "full_text": full_text,
                }

                if self.grm_model is not None:
                    reward = self.grm_model.compute_grm_reward(prompt, response_text)
                    rollout["grm_reward"] = reward

                rollouts.append(rollout)

        return rollouts

    def _compute_student_log_probs(
        self,
        full_text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute student model logits and log probabilities for a given text.

        Args:
            full_text: Full prompt + response text

        Returns:
            Tuple of (student_logits, student_log_prob)
        """
        device = next(self.student_model.parameters()).device

        if self.tokenizer:
            input_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(device)
        else:
            input_ids = torch.tensor([[ord(c) for c in full_text]], dtype=torch.long, device=device)

        self.student_model.eval()
        with torch.no_grad():
            outputs = self.student_model(input_ids)
            student_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            student_token_lps = student_log_probs[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            student_lp = student_token_lps.sum()

        return student_logits, student_lp

    def train(
        self,
        prompts: List[str],
        num_epochs: int = 1,
        save_dir: Optional[str] = None,
        save_every: int = 100,
    ) -> Dict[str, Any]:
        """
        Train the student model using on-policy distillation from the teacher.

        Args:
            prompts: List of training prompts
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N steps

        Returns:
            Training statistics dictionary
        """
        all_stats = {
            "opd_losses": [],
            "kl_divergences": [],
            "ce_losses": [],
        }

        step = 0
        for epoch in range(num_epochs):
            for prompt in prompts:
                rollouts = self._generate_teacher_rollouts(prompt, self.config.num_rollouts)

                advantages = None
                if self.grm_model is not None and rollouts[0].get("grm_reward") is not None:
                    rewards = torch.tensor([r["grm_reward"] for r in rollouts], dtype=torch.float32)
                    if rewards.numel() > 1:
                        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                student_logits_list = []
                student_lp_list = []
                teacher_logits_list = []
                teacher_lp_list = []

                for rollout in rollouts:
                    s_logits, s_lp = self._compute_student_log_probs(rollout["full_text"])
                    student_logits_list.append(s_logits)
                    student_lp_list.append(s_lp)
                    teacher_logits_list.append(rollout["teacher_logits"])
                    teacher_lp_list.append(rollout["teacher_log_prob"])

                max_seq_len = max(logit.shape[1] for logit in student_logits_list)
                vocab_size = student_logits_list[0].shape[-1]

                padded_student_logits = []
                padded_teacher_logits = []
                for s_logits, t_logits in zip(student_logits_list, teacher_logits_list):
                    if s_logits.shape[1] < max_seq_len:
                        pad_len = max_seq_len - s_logits.shape[1]
                        s_pad = torch.zeros(1, pad_len, vocab_size, device=s_logits.device)
                        t_pad = torch.zeros(1, pad_len, vocab_size, device=t_logits.device)
                        padded_student_logits.append(torch.cat([s_logits, s_pad], dim=1))
                        padded_teacher_logits.append(torch.cat([t_logits, t_pad], dim=1))
                    else:
                        padded_student_logits.append(s_logits[:, :max_seq_len, :])
                        padded_teacher_logits.append(t_logits[:, :max_seq_len, :])

                stacked_student_logits = torch.cat(padded_student_logits, dim=0)
                stacked_teacher_logits = torch.cat(padded_teacher_logits, dim=0)
                stacked_student_lp = torch.stack(student_lp_list)
                stacked_teacher_lp = torch.stack(teacher_lp_list)

                adv_tensor = advantages.to(stacked_student_logits.device) if advantages is not None else None

                loss, metrics = opd_loss(
                    teacher_logits=stacked_teacher_logits,
                    student_logits=stacked_student_logits,
                    teacher_log_probs=stacked_teacher_lp,
                    student_log_probs=stacked_student_lp,
                    config=self.config,
                    advantages=adv_tensor,
                )

                self.student_optimizer.zero_grad()
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    self.student_optimizer.step()

                all_stats["opd_losses"].append(metrics["opd_loss"])
                all_stats["kl_divergences"].append(metrics["kl_div"])
                all_stats["ce_losses"].append(metrics["ce_loss"])

                step += 1

                if save_dir and step % save_every == 0:
                    self._save_checkpoint(save_dir, step)

        self.training_history.append(all_stats)

        return {
            "mean_opd_loss": sum(all_stats["opd_losses"]) / len(all_stats["opd_losses"]) if all_stats["opd_losses"] else 0,
            "mean_kl": sum(all_stats["kl_divergences"]) / len(all_stats["kl_divergences"]) if all_stats["kl_divergences"] else 0,
            "mean_ce": sum(all_stats["ce_losses"]) / len(all_stats["ce_losses"]) if all_stats["ce_losses"] else 0,
            "total_steps": step,
        }

    def _save_checkpoint(self, save_dir: str, step: int):
        """Save a training checkpoint."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "step": step,
            "student_state_dict": self.student_model.state_dict(),
            "student_optimizer_state_dict": self.student_optimizer.state_dict(),
            "config": self.config.__dict__,
        }

        path = os.path.join(save_dir, f"opd_checkpoint_{step}.pt")
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")

        self.student_model.load_state_dict(checkpoint["student_state_dict"])
        self.student_optimizer.load_state_dict(checkpoint["student_optimizer_state_dict"])

        return checkpoint["step"]


class POPSSGenerativeRewardModel(nn.Module):
    """
    Generative Reward Model (GRM) wrapping a base LM with a reward head.

    The GRM extends a language model with a learned reward head that outputs
    scalar reward scores for any input/output pair. It can be trained via
    preference pairs (like DPO but as a reward model) or used implicitly
    through DPO policy log-probability ratios.

    Key Features:
        - LM backbone with a learned MLP reward head
        - Outputs scalar reward scores for response evaluation
        - Supports both explicit training (preference pairs) and implicit rewards
        - Can be used as a reward signal for GRPO/OPD training

    Architecture:
        Base LM → Hidden States → [Reward Head MLP] → Scalar Reward

    Example:
        >>> grm = POPSSGenerativeRewardModel(
        ...     base_model=language_model,
        ...     config=POPSSGRMConfig(grm_hidden_size=4096, grm_num_layers=2),
        ... )
        >>> reward = grm.compute_grm_reward(prompt="Hello", response="Hi there!")
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: Optional[POPSSGRMConfig] = None,
    ):
        super().__init__()
        self.config = config or POPSSGRMConfig()
        self.base_model = base_model
        self.hidden_size = self.config.grm_hidden_size

        reward_layers = []
        input_dim = self.hidden_size
        for i in range(self.config.grm_num_layers):
            output_dim = self.hidden_size if i < self.config.grm_num_layers - 1 else 1
            reward_layers.append(nn.Linear(input_dim, output_dim))
            if i < self.config.grm_num_layers - 1:
                reward_layers.append(nn.GELU())
                reward_layers.append(nn.Dropout(0.1))
            input_dim = output_dim

        self.reward_head = nn.Sequential(*reward_layers)

        self._reward_optimizer: Optional[torch.optim.Optimizer] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the base model and reward head.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Reward scores [batch, 1]
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        elif isinstance(outputs, dict) and "hidden_states" in outputs:
            hidden = outputs["hidden_states"][-1]
        else:
            last_hidden = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            hidden = last_hidden

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            hidden = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            hidden = hidden.mean(dim=1)

        reward = self.reward_head(hidden)
        return reward

    def compute_grm_reward(
        self,
        prompt: str,
        response: str,
        tokenizer=None,
    ) -> float:
        """
        Compute scalar reward for a prompt-response pair.

        Args:
            prompt: Input prompt text
            response: Generated response text
            tokenizer: Tokenizer for encoding

        Returns:
            Scalar reward score
        """
        self.eval()
        device = next(self.parameters()).device

        full_text = prompt + response
        if tokenizer:
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
        else:
            input_ids = torch.tensor([[ord(c) for c in full_text]], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            reward = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        return reward.squeeze().item()

    def train_reward_model(
        self,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Train the reward model on preference pairs using a ranking loss.

        The loss encourages higher rewards for chosen responses than rejected ones.

        Args:
            chosen_input_ids: Token IDs for chosen responses [batch, seq_len]
            rejected_input_ids: Token IDs for rejected responses [batch, seq_len]
            chosen_attention_mask: Attention mask for chosen [batch, seq_len]
            rejected_attention_mask: Attention mask for rejected [batch, seq_len]

        Returns:
            Dictionary of training metrics
        """
        chosen_rewards = self.forward(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
        )
        rejected_rewards = self.forward(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
        )

        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        if self._reward_optimizer is None:
            self._reward_optimizer = torch.optim.AdamW(
                self.reward_head.parameters(),
                lr=self.config.grm_learning_rate,
            )

        self._reward_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_head.parameters(), 1.0)
        self._reward_optimizer.step()

        accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return {
            "grm_loss": loss.item(),
            "grm_accuracy": accuracy.item(),
            "mean_chosen_reward": chosen_rewards.mean().item(),
            "mean_rejected_reward": rejected_rewards.mean().item(),
        }


__all__ = [
    "POPSSDPOTrainingConfig",
    "POPSSDPODataset",
    "POPSSDPOTrainingOperator",
    "POPSSDPOLoggingCallback",
    "POPSSOPDConfig",
    "POPSSOPDTrainer",
    "POPSSGRMConfig",
    "POPSSGenerativeRewardModel",
    "opd_loss",
]
