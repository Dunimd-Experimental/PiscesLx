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
Direct Preference Optimization (DPO) training pipeline for PiscesL1.

This module implements DPO training with:
- Preference pair loss (chosen vs rejected responses)
- Reference model for KL divergence regularization
- Beta hyperparameter for policy/reference tradeoff
- Mixed precision training support
- Checkpoint management
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

logger = PiscesLxCoreLog("pisceslx.tools.train.dpo")


@dataclass
class PiscesL1DPOConfig:
    """DPO training configuration."""
    
    model_path: str = "./checkpoints/ruchbah"
    ref_model_path: str = "./checkpoints/sft_output/final_model"
    output_dir: str = "./checkpoints/dpo_output"
    
    train_data: str = "./data/preference_train.jsonl"
    val_data: str = "./data/preference_val.jsonl"
    
    beta: float = 0.1
    reference_alpha: float = 1.0
    loss_type: str = "sigmoid"
    
    batch_size: int = 4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    
    learning_rate: float = 5e-6
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
    
    def __post_init__(self):
        if self.use_fp16 and self.use_bf16:
            self.use_bf16 = False


class PiscesL1DPODataset(Dataset):
    """Dataset for DPO training with preference pairs."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_seq_length: int = 2048,
    ):
        """Initialize DPO dataset.
        
        Args:
            data_path: Path to preference data (JSONL format).
            tokenizer: Tokenizer for encoding text.
            max_seq_length: Maximum sequence length.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        self.samples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.samples)} preference pairs from {data_path}")
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load preference data from JSONL file."""
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
                    if "chosen" in sample and "rejected" in sample:
                        samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        return samples
    
    def _format_pair(self, prompt: str, response: str) -> str:
        """Format a prompt-response pair for encoding."""
        return f"User: {prompt}\nAssistant: {response}"
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair sample."""
        sample = self.samples[idx]
        
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        
        chosen_text = self._format_pair(prompt, chosen)
        rejected_text = self._format_pair(prompt, rejected)
        
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
        }


class PiscesL1DPOTrainer:
    """DPO trainer for PiscesL1 model."""
    
    def __init__(
        self,
        config: PiscesL1DPOConfig,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer: Any,
        optimizer: Optional[Optimizer] = None,
    ):
        """Initialize DPO trainer.
        
        Args:
            config: DPO training configuration.
            model: Policy model to train.
            ref_model: Reference model for KL regularization.
            tokenizer: Tokenizer for the model.
            optimizer: Optimizer for training.
        """
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        
        self.optimizer = optimizer
        
        self.scaler = None
        if config.use_fp16 or config.use_bf16:
            self.scaler = GradScaler()
        
        self.global_step = 0
        self.total_loss = 0.0
        
        self.checkpoint_manager = PiscesLxCoreCheckpointManager()
        
        logger.info("PiscesL1DPOTrainer initialized")
    
    def _create_optimizer(self, config: PiscesL1DPOConfig) -> Optimizer:
        """Create optimizer for training."""
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
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
    
    def _compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        beta: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss.
        
        Args:
            policy_chosen_logps: Log probabilities for chosen responses from policy model.
            policy_rejected_logps: Log probabilities for rejected responses from policy model.
            ref_chosen_logps: Log probabilities for chosen responses from reference model.
            ref_rejected_logps: Log probabilities for rejected responses from reference model.
            beta: Temperature parameter for KL regularization.
            
        Returns:
            Tuple of (loss tensor, loss components dictionary).
        """
        policy_advantages = policy_chosen_logps - policy_rejected_logps
        ref_advantages = ref_chosen_logps - ref_rejected_logps
        
        logits = beta * (policy_advantages - ref_advantages)
        
        if self.config.loss_type == "sigmoid":
            losses = -torch.nn.functional.logsigmoid(logits)
        elif self.config.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        else:
            losses = -torch.nn.functional.logsigmoid(logits)
        
        chosen_probs = torch.sigmoid(policy_advantages)
        kl_chosen = policy_chosen_logps - ref_chosen_logps
        kl_rejected = policy_rejected_logps - ref_rejected_logps
        
        loss_dict = {
            "loss": losses.mean(),
            "policy_advantage": policy_advantages.mean().item(),
            "ref_advantage": ref_advantages.mean().item(),
            "kl_chosen": kl_chosen.mean().item(),
            "kl_rejected": kl_rejected.mean().item(),
            "chosen_accuracy": (chosen_probs > 0.5).float().mean().item(),
        }
        
        return losses.mean(), loss_dict
    
    @torch.no_grad()
    def _get_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for given sequences.
        
        Args:
            model: Model to compute log probabilities.
            input_ids: Token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            
        Returns:
            Log probabilities tensor [batch].
        """
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = output.get("logits", output)
        
        if logits.dim() == 3:
            logits = logits[:, :-1, :]
            input_ids = input_ids[:, 1:]
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        logps = torch.gather(
            log_probs,
            dim=-1,
            index=input_ids.unsqueeze(-1),
        ).squeeze(-1)
        
        mask = attention_mask[:, 1:].float()
        
        logps = (logps * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        
        return logps
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> Dict[str, Any]:
        """Run DPO training.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Optional validation dataset.
            
        Returns:
            Training metrics dictionary.
        """
        config = self.config
        
        logger.info("Starting DPO training...")
        
        self.ref_model.eval()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        
        if self.optimizer is None:
            self.optimizer = self._create_optimizer(config)
        
        if config.use_gradient_checkpointing:
            self.model.set_gradient_checkpointing(True)
        
        self.model.train()
        
        accumulation_steps = config.gradient_accumulation_steps
        
        training_metrics = {
            "total_steps": 0,
            "avg_loss": 0.0,
            "policy_advantage": 0.0,
            "kl_chosen": 0.0,
        }
        
        start_time = time.time()
        
        for epoch in range(sys.maxsize):
            for batch_idx, batch in enumerate(train_loader):
                if self.global_step >= config.max_steps:
                    break
                
                self.global_step += 1
                
                chosen_ids = batch["chosen_input_ids"].cuda()
                chosen_mask = batch["chosen_attention_mask"].cuda()
                rejected_ids = batch["rejected_input_ids"].cuda()
                rejected_mask = batch["rejected_attention_mask"].cuda()
                
                batch_size = chosen_ids.shape[0]
                
                with autocast(
                    enabled=(config.use_fp16 or config.use_bf16),
                    dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
                ):
                    policy_chosen_logps = self._get_logps(
                        self.model, chosen_ids, chosen_mask
                    )
                    policy_rejected_logps = self._get_logps(
                        self.model, rejected_ids, rejected_mask
                    )
                    
                    with torch.no_grad():
                        ref_chosen_logps = self._get_logps(
                            self.ref_model, chosen_ids, chosen_mask
                        )
                        ref_rejected_logps = self._get_logps(
                            self.ref_model, rejected_ids, rejected_mask
                        )
                    
                    loss, loss_dict = self._compute_dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        ref_chosen_logps,
                        ref_rejected_logps,
                        config.beta,
                    )
                    
                    loss = loss / accumulation_steps
                
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
                    
                    self.optimizer.zero_grad()
                
                self.total_loss += loss_dict["loss"].item()
                
                if self.global_step % 100 == 0:
                    avg_loss = self.total_loss / self.global_step
                    
                    logger.info(
                        f"Step {self.global_step}/{config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Policy Adv: {loss_dict['policy_advantage']:.4f} | "
                        f"KL: {loss_dict['kl_chosen']:.4f} | "
                        f"Acc: {loss_dict['chosen_accuracy']:.2%} | "
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
        
        logger.info(f"DPO training completed: {training_metrics}")
        
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
        total_advantage = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        self.ref_model.eval()
        
        with torch.no_grad():
            for batch in val_loader:
                chosen_ids = batch["chosen_input_ids"].cuda()
                chosen_mask = batch["chosen_attention_mask"].cuda()
                rejected_ids = batch["rejected_input_ids"].cuda()
                rejected_mask = batch["rejected_attention_mask"].cuda()
                
                with autocast(
                    enabled=(self.config.use_fp16 or self.config.use_bf16),
                    dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16,
                ):
                    policy_chosen_logps = self._get_logps(
                        self.model, chosen_ids, chosen_mask
                    )
                    policy_rejected_logps = self._get_logps(
                        self.model, rejected_ids, rejected_mask
                    )
                    
                    ref_chosen_logps = self._get_logps(
                        self.ref_model, chosen_ids, chosen_mask
                    )
                    ref_rejected_logps = self._get_logps(
                        self.ref_model, rejected_ids, rejected_mask
                    )
                    
                    _, loss_dict = self._compute_dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        ref_chosen_logps,
                        ref_rejected_logps,
                        self.config.beta,
                    )
                    
                    total_loss += loss_dict["loss"].item()
                    total_advantage += loss_dict["policy_advantage"]
                    total_accuracy += loss_dict["chosen_accuracy"]
                    num_batches += 1
        
        return {
            "eval_loss": total_loss / max(1, num_batches),
            "eval_advantage": total_advantage / max(1, num_batches),
            "eval_accuracy": total_accuracy / max(1, num_batches),
        }
    
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
                "config": config.__dict__,
            }, checkpoint_path / "training_state.pt")
            
            logger.success(f"Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def save_model(self, output_path: str) -> None:
        """Save the trained model."""
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


def create_dpo_trainer(
    config: PiscesL1DPOConfig,
    model: nn.Module,
    ref_model: nn.Module,
    tokenizer: Any,
) -> PiscesL1DPOTrainer:
    """Factory function to create DPO trainer.
    
    Args:
        config: DPO configuration.
        model: Policy model.
        ref_model: Reference model.
        tokenizer: Model tokenizer.
        
    Returns:
        Initialized DPO trainer.
    """
    trainer = PiscesL1DPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    
    return trainer


def dpo_main(args):
    """Main entry point for DPO training."""
    config = PiscesL1DPOConfig.from_args(args)
    
    logger.info(f"DPO Config: {config}")
    
    if config.local_rank == 0:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    from transformers import AutoTokenizer
    from model.modeling import RuchbahModel
    from model.config import RuchbahConfig
    
    logger.info(f"Loading tokenizer from {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    logger.info(f"Loading policy model from {config.model_path}")
    model_config = RuchbahConfig.from_json(os.path.join(config.model_path, "config.json"))
    model = RuchbahModel(model_config)
    model = model.from_pretrained(config.model_path)
    
    logger.info(f"Loading reference model from {config.ref_model_path}")
    ref_model = RuchbahModel(model_config)
    ref_model = ref_model.from_pretrained(config.ref_model_path)
    
    model = model.cuda()
    ref_model = ref_model.cuda()
    
    ref_model.requires_grad_(False)
    
    if config.world_size > 1:
        model = DDP(model, device_ids=[config.local_rank])
    
    train_dataset = PiscesL1DPODataset(
        data_path=config.train_data,
        tokenizer=tokenizer,
        max_seq_length=2048,
    )
    
    val_dataset = None
    if config.val_data and os.path.exists(config.val_data):
        val_dataset = PiscesL1DPODataset(
            data_path=config.val_data,
            tokenizer=tokenizer,
            max_seq_length=2048,
        )
    
    trainer = create_dpo_trainer(config, model, ref_model, tokenizer)
    
    metrics = trainer.train(train_dataset, val_dataset)
    
    logger.success(f"DPO training finished: {metrics}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DPO training for PiscesL1")
    
    parser.add_argument("--model_path", type=str, default="./checkpoints/ruchbah")
    parser.add_argument("--ref_model_path", type=str, default="./checkpoints/sft_output/final_model")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/dpo_output")
    parser.add_argument("--train_data", type=str, default="./data/preference_train.jsonl")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--local_rank", type=int, default=0)
    
    args = parser.parse_args()
    
    dpo_main(args)
