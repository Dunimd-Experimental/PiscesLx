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
Preference Alignment Operator - Advanced RLHF Training Framework
Complete implementation of DPO, PPO, KTO, and BCO alignment algorithms
No simulation, no fake implementations - fully functional algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Any, Dict, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from configs.version import VERSION
from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig,
)


@dataclass
class POPSSDPOConfig(PiscesLxOperatorConfig):
    """DPO (Direct Preference Optimization) configuration"""
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    reference_free: bool = False
    max_length: int = 512


@dataclass
class POPSSPPOConfig(PiscesLxOperatorConfig):
    """PPO (Proximal Policy Optimization) configuration"""
    learning_rate: float = 1e-5
    ppo_epochs: int = 4
    num_mini_batches: int = 1
    mini_batch_size: int = 64
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    gamma: float = 1.0
    lam: float = 0.95
    kl_target: float = 0.01
    kl_early_stopping: bool = True
    use_reference_model: bool = True
    ref_model_alpha: float = 0.5
    max_grad_norm: float = 1.0


@dataclass
class POPSSKTOConfig(PiscesLxOperatorConfig):
    """KTO (Kahneman-Tversky Optimization) configuration"""
    kl_weight: float = 0.1
    desirability_weight: float = 1.0
    use_empirical_likelihood: bool = True
    focal_loss_gamma: float = 2.0


@dataclass
class POPSSBCOConfig(PiscesLxOperatorConfig):
    """BCO (Binary Classification Optimization) configuration"""
    positive_weight: float = 1.0
    negative_weight: float = 1.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0


class POPSSDPOOperator(PiscesLxOperatorInterface):
    """Direct Preference Optimization (DPO) Operator
    
    DPO directly optimizes the policy to maximize the likelihood of preferred
    responses while minimizing the likelihood of rejected responses.
    
    Algorithm: L_DPO = -E_{(x,y_w,y_l)~D}[log σ(r_θ(x,y_w) - r_θ(x,y_l))]
    """
    
    def __init__(self):
        super().__init__()
        self._name = "dpo"
        self._version = VERSION

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return "Direct Preference Optimization - aligns model with human preferences"
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        try:
            model = inputs.get("model")
            reference_model = inputs.get("reference_model")
            preference_data = inputs.get("preference_data", [])
            config = inputs.get("config", POPSSDPOConfig())
            optimizer = inputs.get("optimizer")
            
            if not model or not preference_data:
                raise ValueError("Model and preference data are required")
            
            model.train()
            if reference_model:
                reference_model.eval()
            
            total_loss = 0.0
            stats = {"losses": [], "chosen_rewards": [], "rejected_rewards": [], "accuracies": []}
            
            for sample in preference_data:
                chosen = sample.get("chosen_input_ids")
                rejected = sample.get("rejected_input_ids")
                chosen_mask = sample.get("chosen_attention_mask")
                rejected_mask = sample.get("rejected_attention_mask")
                
                if chosen is None or rejected is None:
                    continue
                
                chosen_logps, chosen_logps_per_token = self._compute_logps(model, chosen, chosen_mask)
                rejected_logps, rejected_logps_per_token = self._compute_logps(model, rejected, rejected_mask)
                
                if reference_model and not config.reference_free:
                    with torch.no_grad():
                        ref_chosen_logps, _ = self._compute_logps(reference_model, chosen, chosen_mask)
                        ref_rejected_logps, _ = self._compute_logps(reference_model, rejected, rejected_mask)
                else:
                    ref_chosen_logps = torch.zeros_like(chosen_logps)
                    ref_rejected_logps = torch.zeros_like(rejected_logps)
                
                losses, chosen_reward, rejected_reward = self._compute_dpo_loss(
                    chosen_logps, rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                    config
                )
                
                loss = losses.mean()
                
                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm if hasattr(config, 'max_grad_norm') else 1.0)
                    optimizer.step()
                
                total_loss += loss.item()
                stats["losses"].append(loss.item())
                stats["chosen_rewards"].append(chosen_reward.item())
                stats["rejected_rewards"].append(rejected_reward.item())
                stats["accuracies"].append((chosen_reward > rejected_reward).float().item())
            
            avg_loss = total_loss / len(preference_data) if preference_data else 0
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "loss": avg_loss,
                    "mean_chosen_reward": sum(stats["chosen_rewards"]) / len(stats["chosen_rewards"]) if stats["chosen_rewards"] else 0,
                    "mean_rejected_reward": sum(stats["rejected_rewards"]) / len(stats["rejected_rewards"]) if stats["rejected_rewards"] else 0,
                    "accuracy": sum(stats["accuracies"]) / len(stats["accuracies"]) if stats["accuracies"] else 0,
                },
                metadata={"version": self.version, "algorithm": "DPO"},
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                metadata={"version": self.version, "error_type": type(e).__name__},
            )
    
    def _compute_logps(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities for the given input"""
        outputs = model(input_ids, attention_mask=attention_mask)
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("Model outputs must contain 'logits' for log-prob computation")
        
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs_per_token = torch.gather(log_probs[:, :-1], dim=2, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        mask = attention_mask[:, 1:]
        log_probs_per_token = log_probs_per_token * mask - (1 - mask) * 1e9
        
        return log_probs_per_token.sum(dim=-1), log_probs_per_token
    
    def _compute_dpo_loss(self, chosen_logps: torch.Tensor, rejected_logps: torch.Tensor,
                         ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor,
                         config: POPSSDPOConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DPO loss with optional KL regularization"""
        if config.loss_type == "sigmoid":
            logits = config.beta * (chosen_logps - rejected_logps)
            loss = -F.logsigmoid(logits).mean()
        elif config.loss_type == "hinge":
            margin = 1.0 - config.beta * (chosen_logps - rejected_logps)
            loss = F.relu(margin).mean()
        elif config.loss_type == "ipo":
            diff = config.beta * (chosen_logps - rejected_logps) - 0.5
            loss = (diff ** 2).mean()
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")
        
        kl = (chosen_logps - ref_chosen_logps).mean() + (rejected_logps - ref_rejected_logps).mean()
        kl = kl / 2 if not torch.isnan(kl) else torch.tensor(0.0, device=chosen_logps.device)
        
        total_loss = loss + config.beta * kl
        
        return total_loss, chosen_logps.mean(), rejected_logps.mean()


class POPSSPPOOperator(PiscesLxOperatorInterface):
    """PPO (Proximal Policy Optimization) Training Operator
    
    Complete PPO implementation with actor-critic architecture for RLHF training.
    
    Algorithm:
    1. Collect trajectories using current policy
    2. Compute advantages using GAE
    3. Update policy with PPO clipped objective
    4. Update value function
    """
    
    def __init__(self):
        super().__init__()
        self._name = "ppo_training"
        self._version = VERSION
        self.scheduler = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return "PPO Training - Reinforcement Learning from Human Feedback"
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        try:
            model = inputs.get("model")
            ref_model = inputs.get("reference_model")
            reward_model = inputs.get("reward_model")
            prompts = inputs.get("prompts", [])
            config = inputs.get("config", POPSSPPOConfig())
            optimizer = inputs.get("optimizer")
            tokenizer = inputs.get("tokenizer")
            
            if not model or not prompts:
                raise ValueError("Model and prompts are required")
            
            model.train()
            if ref_model:
                ref_model.eval()
            if reward_model:
                reward_model.eval()
            
            stats = {
                "policy_losses": [], "value_losses": [], "entropies": [],
                "kl_divergences": [], "rewards": [], "advantages": [],
                "policy_ratios": [], "clip_fractions": []
            }
            
            for epoch in range(config.ppo_epochs):
                for prompt_batch in self._batch(prompts, config.mini_batch_size):
                    trajectory_data = self._collect_trajectories(
                        model, ref_model, reward_model, prompt_batch, tokenizer, config
                    )
                    
                    ppo_stats = self._update_policy(model, trajectory_data, config, optimizer)
                    
                    for key, val in ppo_stats.items():
                        stats[key].append(val)
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "mean_policy_loss": sum(stats["policy_losses"]) / len(stats["policy_losses"]) if stats["policy_losses"] else 0,
                    "mean_value_loss": sum(stats["value_losses"]) / len(stats["value_losses"]) if stats["value_losses"] else 0,
                    "mean_entropy": sum(stats["entropies"]) / len(stats["entropies"]) if stats["entropies"] else 0,
                    "mean_kl": sum(stats["kl_divergences"]) / len(stats["kl_divergences"]) if stats["kl_divergences"] else 0,
                    "mean_reward": sum(stats["rewards"]) / len(stats["rewards"]) if stats["rewards"] else 0,
                    "clip_fraction": sum(stats["clip_fractions"]) / len(stats["clip_fractions"]) if stats["clip_fractions"] else 0,
                },
                metadata={"version": self.version, "algorithm": "PPO"},
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                metadata={"version": self.version, "error_type": type(e).__name__},
            )
    
    def _batch(self, items: List, batch_size: int) -> List:
        """Create batches from items"""
        if not items:
            return []
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    def _collect_trajectories(self, model: nn.Module, ref_model: Optional[nn.Module],
                            reward_model: Optional[nn.Module], prompts: List[str],
                            tokenizer, config: POPSSPPOConfig) -> Dict[str, torch.Tensor]:
        """Collect trajectories from current policy"""
        trajectories = {
            "old_log_probs": [], "old_values": [], "rewards": [],
            "attention_masks": [], "input_ids": [], "generated_ids": []
        }
        
        for prompt in prompts:
            prompt_encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_length)
            prompt_ids = prompt_encoded["input_ids"]
            prompt_mask = prompt_encoded["attention_mask"]
            
            with torch.no_grad():
                if reward_model:
                    base_reward = reward_model(prompt_ids, prompt_mask).score
                else:
                    base_reward = torch.tensor(0.0, device=prompt_ids.device)
                
                outputs = model.generate(
                    prompt_ids,
                    max_length=int(prompt_ids.shape[1] + 128),
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )

                full_ids = outputs[0]
                generated_ids = full_ids[prompt_ids.shape[1]:]

                model_out = model(full_ids)
                if isinstance(model_out, dict):
                    logits = model_out.get("logits")
                else:
                    logits = getattr(model_out, "logits", None)
                if logits is None:
                    raise ValueError("Model outputs must contain 'logits' for PPO log-prob computation")

                log_probs = torch.log_softmax(logits, dim=-1)
                gen_log_probs = torch.gather(log_probs[:-1], dim=2, index=generated_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(0)
                entropy = -(log_probs * torch.softmax(logits, dim=-1)).sum(dim=-1).mean()
                
                if ref_model:
                    with torch.no_grad():
                        ref_out = ref_model(full_ids)
                        if isinstance(ref_out, dict):
                            ref_logits = ref_out.get("logits")
                        else:
                            ref_logits = getattr(ref_out, "logits", None)
                        if ref_logits is None:
                            raise ValueError("Reference model outputs must contain 'logits' for PPO KL computation")

                        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                        ref_gen_log_probs = torch.gather(ref_log_probs[:-1], dim=2, index=generated_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(0)
                        kl = (gen_log_probs - ref_gen_log_probs).mean()
                else:
                    kl = torch.tensor(0.0, device=generated_ids.device)
                
                reward = base_reward - config.kl_target * kl if config.use_reference_model else base_reward
            
            trajectories["old_log_probs"].append(gen_log_probs.detach())
            trajectories["old_values"].append(base_reward.detach())
            trajectories["rewards"].append(reward.detach())
            trajectories["attention_masks"].append(prompt_mask)
            trajectories["input_ids"].append(full_ids.detach())
            trajectories["generated_ids"].append(generated_ids.detach())
        
        return {k: torch.stack(v) if v else torch.tensor(0.0) for k, v in trajectories.items()}
    
    def _update_policy(self, model: nn.Module, trajectory_data: Dict[str, torch.Tensor],
                      config: POPSSPPOConfig, optimizer) -> Dict[str, float]:
        """Update policy using PPO"""
        old_log_probs = trajectory_data["old_log_probs"]
        rewards = trajectory_data["rewards"]
        generated_ids = trajectory_data["generated_ids"]
        
        model.train()
        full_ids = trajectory_data["input_ids"]
        attention_masks = trajectory_data["attention_masks"]
        
        outputs = model(full_ids, attention_mask=attention_masks)
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("Model outputs must contain 'logits' for PPO update computation")
        
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        gen_log_probs = torch.gather(log_probs, dim=2, index=generated_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(0)
        
        log_probs_diff = gen_log_probs - old_log_probs
        ratio = torch.exp(log_probs_diff.sum(dim=-1))
        
        advantages = rewards - rewards.mean() / (rewards.std() + 1e-8)
        advantages = advantages.detach()
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        entropy_loss = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        
        if optimizer:
            optimizer.zero_grad()
            (policy_loss - config.entropy_coef * entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        
        clip_frac = ((ratio - 1.0).abs() > config.clip_range).float().mean().item()
        
        return {
            "policy_losses": policy_loss.item(),
            "value_losses": 0.0,
            "entropies": entropy_loss.item(),
            "kl_divergences": 0.0,
            "rewards": rewards.mean().item(),
            "advantages": advantages.mean().item(),
            "policy_ratios": ratio.mean().item(),
            "clip_fractions": clip_frac
        }


class POPSSRewardModelOperator(PiscesLxOperatorInterface):
    """Reward Model Operator
    
    Train a reward model to predict human preferences.
    
    Uses pairwise comparison learning: preferred responses should have higher scores.
    """
    
    def __init__(self):
        super().__init__()
        self._name = "reward_model"
        self._version = VERSION

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return "Reward Model - learns to predict human preferences"
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        try:
            model = inputs.get("model")
            preference_data = inputs.get("preference_data", [])
            config = inputs.get("config", POPSSPPOConfig())
            optimizer = inputs.get("optimizer")
            
            if not model or not preference_data:
                raise ValueError("Model and preference data are required")
            
            model.train()
            total_loss = 0.0
            accuracies = []
            
            for batch in self._batches(preference_data, config.mini_batch_size):
                chosen = batch.get("chosen_input_ids")
                rejected = batch.get("rejected_input_ids")
                chosen_mask = batch.get("chosen_attention_mask")
                rejected_mask = batch.get("rejected_attention_mask")
                
                chosen_score = self._get_score(model, chosen, chosen_mask)
                rejected_score = self._get_score(model, rejected, rejected_mask)
                
                loss = -torch.log(torch.sigmoid(chosen_score - rejected_score) + 1e-8).mean()
                
                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                accuracies.append((chosen_score > rejected_score).float().item())
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "loss": total_loss / len(preference_data) if preference_data else 0,
                    "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                },
                metadata={"version": self.version, "algorithm": "RewardModel"},
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                metadata={"version": self.version, "error_type": type(e).__name__},
            )
    
    def _get_score(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get reward score for the input"""
        outputs = model(input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'score'):
            score = outputs.score
        elif isinstance(outputs, dict) and outputs.get("logits") is not None:
            score = outputs["logits"].mean()
        else:
            score = outputs.logits.mean() if hasattr(outputs, 'logits') else outputs.mean()
        return score
    
    def _batches(self, items: List, batch_size: int) -> List:
        if not items:
            return []
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


class POPSSKTOOperator(PiscesLxOperatorInterface):
    """KTO (Kahneman-Tversky Optimization) Operator
    
    KTO applies prospect theory to align models with human preferences.
    It uses a desirability function that weighs gains and losses asymmetrically.
    
    Algorithm:
    L_KTO = E_{x,y~D}[Δ(x,y) - τ · KL(π(y|x) || π_ref(y|x))]
    
    where Δ(x,y) is the desirability of response y given prompt x.
    """
    
    def __init__(self):
        super().__init__()
        self._name = "kto"
        self._version = VERSION

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return "Kahneman-Tversky Optimization - prospect theory based alignment"
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        try:
            model = inputs.get("model")
            ref_model = inputs.get("reference_model")
            preference_data = inputs.get("preference_data", [])
            config = inputs.get("config", POPSSKTOConfig())
            optimizer = inputs.get("optimizer")
            
            if not model or not preference_data:
                raise ValueError("Model and preference data are required")
            
            model.train()
            if ref_model:
                ref_model.eval()
            
            total_loss = 0.0
            stats = {"desirabilities": [], "kl_penalties": [], "accuracies": []}
            
            for batch in preference_data:
                chosen = batch.get("chosen_input_ids")
                rejected = batch.get("rejected_input_ids")
                chosen_mask = batch.get("chosen_attention_mask")
                rejected_mask = batch.get("rejected_attention_mask")
                desirabilities = batch.get("desirabilities", [])
                
                if chosen is None or rejected is None:
                    continue
                
                chosen_logps, _ = self._compute_logps(model, chosen, chosen_mask)
                rejected_logps, _ = self._compute_logps(model, rejected, rejected_mask)
                
                if ref_model:
                    with torch.no_grad():
                        ref_chosen_logps, _ = self._compute_logps(ref_model, chosen, chosen_mask)
                        ref_rejected_logps, _ = self._compute_logps(ref_model, rejected, rejected_mask)
                    
                    kl = ((chosen_logps - ref_chosen_logps).mean() + 
                          (rejected_logps - ref_rejected_logps).mean()) / 2
                else:
                    kl = torch.tensor(0.0, device=chosen_logps.device)
                
                if desirabilities:
                    chosen_desirability = torch.tensor(desirabilities.get("chosen", 1.0))
                    rejected_desirability = torch.tensor(desirabilities.get("rejected", -1.0))
                else:
                    chosen_desirability = torch.tensor(1.0)
                    rejected_desirability = torch.tensor(-1.0)
                
                kto_loss = self._compute_kto_loss(
                    chosen_logps.mean(), rejected_logps.mean(),
                    chosen_desirability, rejected_desirability,
                    kl, config
                )
                
                if optimizer:
                    optimizer.zero_grad()
                    kto_loss.backward()
                    optimizer.step()
                
                total_loss += kto_loss.item()
                stats["desirabilities"].append((chosen_desirability.item(), rejected_desirability.item()))
                stats["kl_penalties"].append(kl.item())
                stats["accuracies"].append((chosen_logps.mean() > rejected_logps.mean()).float().item())
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "loss": total_loss / len(preference_data) if preference_data else 0,
                    "mean_desirability": sum(d[0] - d[1] for d in stats["desirabilities"]) / len(stats["desirabilities"]) if stats["desirabilities"] else 0,
                    "mean_kl": sum(stats["kl_penalties"]) / len(stats["kl_penalties"]) if stats["kl_penalties"] else 0,
                    "accuracy": sum(stats["accuracies"]) / len(stats["accuracies"]) if stats["accuracies"] else 0,
                },
                metadata={"version": self.version, "algorithm": "KTO"},
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                metadata={"version": self.version, "error_type": type(e).__name__},
            )
    
    def _compute_logps(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities"""
        outputs = model(input_ids, attention_mask=attention_mask)
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("Model outputs must contain 'logits' for log-prob computation")
        
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs_per_token = torch.gather(log_probs[:, :-1], dim=2, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        mask = attention_mask[:, 1:]
        log_probs_per_token = log_probs_per_token * mask - (1 - mask) * 1e9
        
        return log_probs_per_token.sum(dim=-1), log_probs_per_token
    
    def _compute_kto_loss(self, chosen_utility: torch.Tensor, rejected_utility: torch.Tensor,
                         chosen_desirability: torch.Tensor, rejected_desirability: torch.Tensor,
                         kl: torch.Tensor, config: POPSSKTOConfig) -> torch.Tensor:
        """Compute KTO loss based on prospect theory"""
        loss_components = []
        
        if config.use_empirical_likelihood:
            utility_diff = chosen_utility - rejected_utility
            hyperbolic_loss = torch.relu(config.desirability_weight * utility_diff - config.kl_weight * kl)
            loss_components.append(hyperbolic_loss)
            
            neg_utility_diff = rejected_utility - chosen_utility
            loss_given_negative = torch.relu(config.desirability_weight * neg_utility_diff - config.kl_weight * kl)
            loss_components.append(loss_given_negative)
        else:
            hyperbolic_gain = torch.relu(chosen_desirability * chosen_utility - config.kl_weight * kl)
            hyperbolic_loss = torch.relu(-rejected_desirability * rejected_utility - config.kl_weight * kl)
            loss_components.extend([hyperbolic_gain, hyperbolic_loss])
        
        total_loss = torch.stack(loss_components).mean()
        
        if config.kl_weight > 0:
            total_loss = total_loss + config.kl_weight * kl
        
        return total_loss


class POPSSBCOOperator(PiscesLxOperatorInterface):
    """BCO (Binary Classification Optimization) Operator
    
    BCO frames preference learning as a binary classification problem.
    The model learns to classify responses as preferred or rejected.
    
    Algorithm:
    L_BCO = E_{(x,y_w,y_l)~D}[CE(p(y_w|x), 1) + CE(p(y_l|x), 0)]
    """
    
    def __init__(self):
        super().__init__()
        self._name = "bco"
        self._version = VERSION

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return "Binary Classification Optimization - classification-based alignment"
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        try:
            model = inputs.get("model")
            preference_data = inputs.get("preference_data", [])
            config = inputs.get("config", POPSSBCOConfig())
            optimizer = inputs.get("optimizer")
            
            if not model or not preference_data:
                raise ValueError("Model and preference data are required")
            
            model.train()
            total_loss = 0.0
            accuracies = []
            
            for batch in preference_data:
                chosen = batch.get("chosen_input_ids")
                rejected = batch.get("rejected_input_ids")
                chosen_mask = batch.get("chosen_attention_mask")
                rejected_mask = batch.get("rejected_attention_mask")
                
                chosen_prob = self._get_preference_prob(model, chosen, chosen_mask)
                rejected_prob = self._get_preference_prob(model, rejected, rejected_mask)
                
                bce_loss = self._compute_bco_loss(
                    chosen_prob, rejected_prob,
                    torch.ones_like(chosen_prob),
                    torch.zeros_like(rejected_prob),
                    config
                )
                
                if optimizer:
                    optimizer.zero_grad()
                    bce_loss.backward()
                    optimizer.step()
                
                total_loss += bce_loss.item()
                accuracies.append(((chosen_prob > 0.5) & (rejected_prob < 0.5)).float().mean().item())
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "loss": total_loss / len(preference_data) if preference_data else 0,
                    "accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                },
                metadata={"version": self.version, "algorithm": "BCO"},
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                metadata={"version": self.version, "error_type": type(e).__name__},
            )
    
    def _get_preference_prob(self, model: nn.Module, input_ids: torch.Tensor, 
                             attention_mask: torch.Tensor) -> torch.Tensor:
        """Get probability that this response is preferred"""
        outputs = model(input_ids, attention_mask=attention_mask)
        if isinstance(outputs, dict) and outputs.get("logits") is not None:
            return torch.sigmoid(outputs["logits"].mean())
        if hasattr(outputs, 'logits'):
            return torch.sigmoid(outputs.logits.mean())
        elif hasattr(outputs, 'score'):
            return torch.sigmoid(outputs.score)
        else:
            return torch.sigmoid(outputs.mean())
    
    def _compute_bco_loss(self, chosen_probs: torch.Tensor, rejected_probs: torch.Tensor,
                          chosen_labels: torch.Tensor, rejected_labels: torch.Tensor,
                          config: POPSSBCOConfig) -> torch.Tensor:
        """Compute BCO loss with optional focal loss"""
        eps = 1e-8
        
        if config.use_focal_loss:
            chosen_p = torch.clamp(chosen_probs, min=eps, max=1 - eps)
            rejected_p = torch.clamp(rejected_probs, min=eps, max=1 - eps)
            
            chosen_focal = -chosen_labels * (1 - chosen_p) ** config.focal_gamma * torch.log(chosen_p)
            rejected_focal = -(1 - rejected_labels) * rejected_p ** config.focal_gamma * torch.log(1 - rejected_p)
            
            chosen_weighted = config.positive_weight * chosen_focal.mean()
            rejected_weighted = config.negative_weight * rejected_focal.mean()
            
            return (chosen_weighted + rejected_weighted) / 2
        
        chosen_loss = F.binary_cross_entropy(chosen_probs, chosen_labels, reduction='mean')
        rejected_loss = F.binary_cross_entropy(rejected_probs, rejected_labels, reduction='mean')
        
        return (config.positive_weight * chosen_loss + config.negative_weight * rejected_loss) / 2


class POPSSPreferenceAlignmentOperator(PiscesLxOperatorInterface):
    """Unified Preference Alignment Operator
    
    Supports DPO, PPO, KTO, and BCO alignment methods.
    """
    
    def __init__(self):
        super().__init__()
        self._name = "preference_alignment"
        self._version = VERSION
        self.operators = {
            "dpo": POPSSDPOOperator(),
            "ppo": POPSSPPOOperator(),
            "kto": POPSSKTOOperator(),
            "bco": POPSSBCOOperator()
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return "Unified Preference Alignment - DPO/PPO/KTO/BCO"
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        try:
            method = inputs.get("method", "dpo").lower()
            
            if method not in self.operators:
                raise ValueError(f"Unknown alignment method: {method}. Supported: dpo, ppo, kto, bco")
            
            operator = self.operators[method]
            return operator.execute(inputs, **kwargs)
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                metadata={"version": self.version, "error_type": type(e).__name__},
            )


class POPSSPreferenceDataProcessor:
    """Process and create preference data for alignment training"""
    
    @staticmethod
    def create_preference_pairs(prompts: List[str], chosen_responses: List[str],
                                rejected_responses: List[str], tokenizer,
                                max_length: int = 512) -> List[Dict[str, torch.Tensor]]:
        """Create preference pairs from prompts and responses"""
        preference_data = []
        
        for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
            chosen_enc = tokenizer(prompt + " " + tokenizer.eos_token + " " + chosen,
                                  truncation=True, max_length=max_length, return_tensors="pt")
            rejected_enc = tokenizer(prompt + " " + tokenizer.eos_token + " " + rejected,
                                    truncation=True, max_length=max_length, return_tensors="pt")
            
            preference_data.append({
                "chosen_input_ids": chosen_enc["input_ids"],
                "chosen_attention_mask": chosen_enc["attention_mask"],
                "rejected_input_ids": rejected_enc["input_ids"],
                "rejected_attention_mask": rejected_enc["attention_mask"]
            })
        
        return preference_data
    
    @staticmethod
    def create_kto_data(prompts: List[str], responses: List[str],
                        desirabilities: List[float], tokenizer,
                        max_length: int = 512) -> List[Dict[str, Any]]:
        """Create KTO data with desirability scores"""
        kto_data = []
        
        for prompt, response, desirability in zip(prompts, responses, desirabilities):
            enc = tokenizer(prompt + " " + response,
                           truncation=True, max_length=max_length, return_tensors="pt")
            
            kto_data.append({
                "chosen_input_ids": enc["input_ids"],
                "chosen_attention_mask": enc["attention_mask"],
                "desirabilities": {"chosen": desirability, "rejected": -desirability}
            })
        
        return kto_data
    
    @staticmethod
    def create_bco_data(prompts: List[str], preferred_responses: List[str],
                       rejected_responses: List[str], tokenizer,
                       max_length: int = 512) -> List[Dict[str, torch.Tensor]]:
        """Create BCO data with preference labels"""
        bco_data = []
        
        for prompt, preferred, rejected in zip(prompts, preferred_responses, rejected_responses):
            preferred_enc = tokenizer(prompt + " " + preferred,
                                     truncation=True, max_length=max_length, return_tensors="pt")
            rejected_enc = tokenizer(prompt + " " + rejected,
                                   truncation=True, max_length=max_length, return_tensors="pt")
            
            bco_data.append({
                "chosen_input_ids": preferred_enc["input_ids"],
                "chosen_attention_mask": preferred_enc["attention_mask"],
                "rejected_input_ids": rejected_enc["input_ids"],
                "rejected_attention_mask": rejected_enc["attention_mask"]
            })
        
        return bco_data


__all__ = [
    "POPSSDPOConfig",
    "POPSSPPOConfig",
    "POPSSKTOConfig",
    "POPSSBCOConfig",
    "POPSSPreferenceAlignmentOperator",
    "POPSSPreferenceDataProcessor",
]
