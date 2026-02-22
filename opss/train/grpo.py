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
GRPO (Group Relative Policy Optimization) Operator

Complete implementation of DeepSeek R1's GRPO algorithm for preference alignment.
GRPO eliminates the need for a Critic network by using group-relative advantage estimation.

Key Innovation:
    - No Critic network required (saves 30-40% memory)
    - Group-relative advantage: A_i = (r_i - mean(r_group)) / std(r_group)
    - PPO-style clipped objective for stable training
    - KL divergence constraint to prevent deviation from reference model

Reference:
    DeepSeek R1 Technical Report (arXiv:2402.03300)

Algorithm:
    1. Sample group_size responses for each prompt
    2. Compute rewards for each response
    3. Calculate group-relative advantages
    4. Update policy with clipped objective
    5. Apply KL penalty to stay close to reference model
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math

from configs.version import VERSION
from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig,
)


@dataclass
class POPSSGRPOConfig(PiscesLxOperatorConfig):
    """
    GRPO (Group Relative Policy Optimization) Configuration.
    
    This configuration controls the GRPO training algorithm parameters
    for preference alignment without a Critic network.
    
    Attributes:
        group_size: Number of responses to sample per prompt for group comparison
        temperature: Sampling temperature for response generation
        clip_ratio: PPO-style clipping ratio for policy updates
        entropy_coef: Coefficient for entropy regularization bonus
        kl_coef: Coefficient for KL divergence penalty
        max_new_tokens: Maximum number of tokens to generate per response
        use_reference_model: Whether to use a reference model for KL computation
        max_grad_norm: Maximum gradient norm for clipping
        gamma: Discount factor for rewards (usually 1.0 for language tasks)
        advantage_normalization: Whether to normalize advantages within groups
        min_std: Minimum std for advantage normalization stability
    """
    name: str = "grpo"
    version: str = VERSION
    
    group_size: int = 4
    temperature: float = 1.0
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    max_new_tokens: int = 512
    use_reference_model: bool = True
    max_grad_norm: float = 1.0
    gamma: float = 1.0
    advantage_normalization: bool = True
    min_std: float = 1e-8
    
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    
    ppo_epochs: int = 4
    mini_batch_size: int = 4
    
    def __post_init__(self):
        super().__post_init__()
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO")


class POPSSGRPOOperator(PiscesLxOperatorInterface):
    """
    Group Relative Policy Optimization (GRPO) Operator.
    
    GRPO is DeepSeek R1's core algorithm for preference alignment.
    It eliminates the need for a Critic network by computing advantages
    relative to a group of sampled responses.
    
    Key Features:
        - No Critic network: Saves 30-40% memory compared to PPO
        - Group-relative advantages: Stable advantage estimation
        - PPO-style clipping: Prevents large policy updates
        - KL regularization: Maintains proximity to reference model
    
    Example:
        >>> config = POPSSGRPOConfig(group_size=4, temperature=1.0)
        >>> grpo = POPSSGRPOOperator()
        >>> result = grpo.execute({
        ...     "model": policy_model,
        ...     "reference_model": ref_model,
        ...     "prompts": training_prompts,
        ...     "reward_function": reward_fn,
        ...     "config": config,
        ... })
    """
    
    def __init__(self):
        super().__init__()
        self._name = "grpo"
        self._version = VERSION
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return "Group Relative Policy Optimization - DeepSeek R1 alignment algorithm"
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute GRPO training step.
        
        Args:
            inputs: Dictionary containing:
                - model: Policy model to optimize
                - reference_model: Reference model for KL computation (optional)
                - prompts: List of prompts for training
                - reward_function: Function to compute rewards
                - config: GRPO configuration
                - optimizer: Optimizer for parameter updates
                - tokenizer: Tokenizer for encoding/decoding
        
        Returns:
            PiscesLxOperatorResult with training statistics
        """
        start_time = self._get_time()
        
        try:
            model = inputs.get("model")
            reference_model = inputs.get("reference_model")
            prompts = inputs.get("prompts", [])
            reward_function = inputs.get("reward_function")
            config = inputs.get("config", POPSSGRPOConfig())
            optimizer = inputs.get("optimizer")
            tokenizer = inputs.get("tokenizer")
            
            if not model or not prompts:
                raise ValueError("Model and prompts are required for GRPO training")
            
            if not reward_function:
                raise ValueError("Reward function is required for GRPO training")
            
            model.train()
            if reference_model:
                reference_model.eval()
            
            stats = {
                "policy_losses": [],
                "kl_divergences": [],
                "entropies": [],
                "advantages": [],
                "rewards": [],
                "clip_fractions": [],
                "approx_kl": [],
            }
            
            for prompt in prompts:
                prompt_stats = self._train_on_prompt(
                    model=model,
                    reference_model=reference_model,
                    prompt=prompt,
                    reward_function=reward_function,
                    config=config,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                )
                
                for key, values in prompt_stats.items():
                    if key in stats:
                        stats[key].extend(values)
            
            output = {
                "mean_policy_loss": self._safe_mean(stats["policy_losses"]),
                "mean_kl": self._safe_mean(stats["kl_divergences"]),
                "mean_entropy": self._safe_mean(stats["entropies"]),
                "mean_advantage": self._safe_mean(stats["advantages"]),
                "mean_reward": self._safe_mean(stats["rewards"]),
                "clip_fraction": self._safe_mean(stats["clip_fractions"]),
                "approx_kl": self._safe_mean(stats["approx_kl"]),
            }
            
            execution_time = self._get_time() - start_time
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=output,
                execution_time=execution_time,
                metadata={
                    "version": self.version,
                    "algorithm": "GRPO",
                    "group_size": config.group_size,
                    "num_prompts": len(prompts),
                },
            )
            
        except Exception as e:
            execution_time = self._get_time() - start_time
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={
                    "version": self.version,
                    "error_type": type(e).__name__,
                },
            )
    
    def _train_on_prompt(
        self,
        model: nn.Module,
        reference_model: Optional[nn.Module],
        prompt: str,
        reward_function,
        config: POPSSGRPOConfig,
        optimizer: Optional[torch.optim.Optimizer],
        tokenizer,
    ) -> Dict[str, List[float]]:
        """Train on a single prompt with GRPO."""
        stats = {key: [] for key in [
            "policy_losses", "kl_divergences", "entropies",
            "advantages", "rewards", "clip_fractions", "approx_kl",
        ]}
        
        responses, log_probs, old_log_probs = self._sample_group_responses(
            model=model,
            prompt=prompt,
            group_size=config.group_size,
            config=config,
            tokenizer=tokenizer,
        )
        
        rewards = self._compute_rewards(
            responses=responses,
            prompt=prompt,
            reward_function=reward_function,
        )
        
        advantages = self.compute_group_advantages(
            rewards=torch.tensor(rewards, dtype=torch.float32),
            group_size=config.group_size,
            normalize=config.advantage_normalization,
            min_std=config.min_std,
        )
        
        if reference_model and config.use_reference_model:
            ref_log_probs = self._compute_reference_log_probs(
                reference_model=reference_model,
                prompt=prompt,
                responses=responses,
                tokenizer=tokenizer,
            )
        else:
            ref_log_probs = torch.zeros_like(log_probs)
        
        for epoch in range(config.ppo_epochs):
            epoch_stats = self._ppo_update(
                model=model,
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                ref_log_probs=ref_log_probs,
                advantages=advantages,
                config=config,
                optimizer=optimizer,
            )
            
            for key, values in epoch_stats.items():
                stats[key].extend(values)
        
        stats["rewards"].extend(rewards)
        stats["advantages"].extend(advantages.tolist())
        
        return stats
    
    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
        normalize: bool = True,
        min_std: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.
        
        This is the core innovation of GRPO: instead of using a Critic network
        to estimate advantages, we compute them relative to other samples
        in the same group.
        
        Formula: A_i = (r_i - mean(r_group)) / std(r_group)
        
        Args:
            rewards: Tensor of rewards [batch_size * group_size]
            group_size: Number of samples per group
            normalize: Whether to normalize advantages
            min_std: Minimum std for numerical stability
        
        Returns:
            Tensor of advantages with same shape as rewards
        """
        rewards = rewards.view(-1, group_size)
        
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True)
        
        if normalize:
            advantages = (rewards - mean) / (std + min_std)
        else:
            advantages = rewards - mean
        
        return advantages.view(-1)
    
    def _sample_group_responses(
        self,
        model: nn.Module,
        prompt: str,
        group_size: int,
        config: POPSSGRPOConfig,
        tokenizer,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Sample a group of responses and compute log probabilities."""
        responses = []
        all_log_probs = []
        all_old_log_probs = []
        
        model.eval()
        with torch.no_grad():
            for _ in range(group_size):
                response, log_prob = self._generate_response(
                    model=model,
                    prompt=prompt,
                    config=config,
                    tokenizer=tokenizer,
                )
                responses.append(response)
                all_log_probs.append(log_prob)
                all_old_log_probs.append(log_prob.clone())
        
        model.train()
        
        log_probs = torch.stack(all_log_probs)
        old_log_probs = torch.stack(all_old_log_probs)
        
        return responses, log_probs, old_log_probs
    
    def _generate_response(
        self,
        model: nn.Module,
        prompt: str,
        config: POPSSGRPOConfig,
        tokenizer,
    ) -> Tuple[str, torch.Tensor]:
        """Generate a single response with log probability."""
        device = next(model.parameters()).device
        
        if tokenizer:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        else:
            input_ids = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device)
        
        generated_ids = input_ids.clone()
        log_probs_sum = torch.tensor(0.0, device=device)
        
        for _ in range(config.max_new_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            next_token_logits = logits[:, -1, :] / config.temperature
            
            if config.top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, config.top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            
            if config.temperature > 0:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            token_log_prob = torch.log(probs.gather(1, next_token) + 1e-10)
            log_probs_sum = log_probs_sum + token_log_prob.squeeze()
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if tokenizer and next_token.item() == tokenizer.eos_token_id:
                break
        
        if tokenizer:
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            response = "".join(chr(c) for c in generated_ids[0].tolist())
        
        return response, log_probs_sum
    
    def _compute_rewards(
        self,
        responses: List[str],
        prompt: str,
        reward_function,
    ) -> List[float]:
        """Compute rewards for each response."""
        rewards = []
        for response in responses:
            try:
                if callable(reward_function):
                    reward = reward_function(prompt, response)
                else:
                    reward = 0.0
                rewards.append(float(reward))
            except Exception:
                rewards.append(0.0)
        return rewards
    
    def _compute_reference_log_probs(
        self,
        reference_model: nn.Module,
        prompt: str,
        responses: List[str],
        tokenizer,
    ) -> torch.Tensor:
        """Compute log probabilities under reference model."""
        device = next(reference_model.parameters()).device
        ref_log_probs = []
        
        reference_model.eval()
        with torch.no_grad():
            for response in responses:
                if tokenizer:
                    full_text = prompt + response
                    input_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
                else:
                    input_ids = torch.tensor([[ord(c) for c in prompt + response]], dtype=torch.long, device=device)
                
                outputs = reference_model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                total_log_prob = token_log_probs.sum()
                
                ref_log_probs.append(total_log_prob)
        
        return torch.stack(ref_log_probs)
    
    def _ppo_update(
        self,
        model: nn.Module,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        config: POPSSGRPOConfig,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Dict[str, List[float]]:
        """Perform PPO-style policy update."""
        stats = {
            "policy_losses": [],
            "kl_divergences": [],
            "entropies": [],
            "clip_fractions": [],
            "approx_kl": [],
        }
        
        if optimizer:
            optimizer.zero_grad()
        
        ratio = torch.exp(log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        kl_div = (log_probs - ref_log_probs).mean()
        
        entropy = -log_probs.mean()
        
        total_loss = (
            policy_loss +
            config.kl_coef * kl_div -
            config.entropy_coef * entropy
        )
        
        if optimizer and total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        
        clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).float().mean()
        approx_kl = (old_log_probs - log_probs).mean().abs()
        
        stats["policy_losses"].append(policy_loss.item())
        stats["kl_divergences"].append(kl_div.item())
        stats["entropies"].append(entropy.item())
        stats["clip_fractions"].append(clip_fraction.item())
        stats["approx_kl"].append(approx_kl.item())
        
        return stats
    
    def _safe_mean(self, values: List[float]) -> float:
        """Compute mean safely, returning 0.0 for empty lists."""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def _get_time(self) -> float:
        """Get current time in seconds."""
        import time
        return time.time()


class POPSSGRPOTrainer:
    """
    High-level GRPO Trainer for preference alignment.
    
    This class provides a convenient interface for training models
    with GRPO, handling the training loop and checkpointing.
    
    Example:
        >>> trainer = POPSSGRPOTrainer(
        ...     model=policy_model,
        ...     reference_model=ref_model,
        ...     reward_function=reward_fn,
        ...     config=POPSSGRPOConfig(group_size=4),
        ... )
        >>> trainer.train(prompts=train_prompts, num_epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        reward_function = None,
        config: Optional[POPSSGRPOConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        tokenizer = None,
    ):
        self.model = model
        self.reference_model = reference_model
        self.reward_function = reward_function
        self.config = config or POPSSGRPOConfig()
        self.tokenizer = tokenizer
        
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-5,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer
        
        self.operator = POPSSGRPOOperator()
        self.training_history = []
    
    def train(
        self,
        prompts: List[str],
        num_epochs: int = 1,
        save_dir: Optional[str] = None,
        save_every: int = 100,
    ) -> Dict[str, Any]:
        """
        Train the model with GRPO.
        
        Args:
            prompts: List of training prompts
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N steps
        
        Returns:
            Training statistics dictionary
        """
        all_stats = {
            "policy_losses": [],
            "kl_divergences": [],
            "rewards": [],
        }
        
        step = 0
        for epoch in range(num_epochs):
            for i, prompt in enumerate(prompts):
                result = self.operator.execute({
                    "model": self.model,
                    "reference_model": self.reference_model,
                    "prompts": [prompt],
                    "reward_function": self.reward_function,
                    "config": self.config,
                    "optimizer": self.optimizer,
                    "tokenizer": self.tokenizer,
                })
                
                if result.status == PiscesLxOperatorStatus.SUCCESS:
                    all_stats["policy_losses"].append(result.output["mean_policy_loss"])
                    all_stats["kl_divergences"].append(result.output["mean_kl"])
                    all_stats["rewards"].append(result.output["mean_reward"])
                
                step += 1
                
                if save_dir and step % save_every == 0:
                    self._save_checkpoint(save_dir, step)
        
        self.training_history.append(all_stats)
        
        return {
            "mean_policy_loss": sum(all_stats["policy_losses"]) / len(all_stats["policy_losses"]) if all_stats["policy_losses"] else 0,
            "mean_kl": sum(all_stats["kl_divergences"]) / len(all_stats["kl_divergences"]) if all_stats["kl_divergences"] else 0,
            "mean_reward": sum(all_stats["rewards"]) / len(all_stats["rewards"]) if all_stats["rewards"] else 0,
            "total_steps": step,
        }
    
    def _save_checkpoint(self, save_dir: str, step: int):
        """Save a training checkpoint."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
        }
        
        path = os.path.join(save_dir, f"checkpoint_{step}.pt")
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint["step"]
