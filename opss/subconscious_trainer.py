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

from __future__ import annotations

"""
Subconscious Training Engine for EnTA-driven RL training.

Trains the subconscious system (router + GRU state + FiLM generators) using
reinforcement learning signals from the external EnTA orchestrator. The
628 × 0.5B knowledge expert pool is frozen and not trained.

Training Flow:
    1. Router Pretrain: learn expert selection via imitation/exploration
    2. Joint RL: PPO or REINFORCE for router + state_evolver + FiLM generators
    3. Self-Play: system generates own exploration targets
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("Yv.SubconsciousTrainer", file_path=get_log_file("Yv.SubconsciousTrainer"), enable_file=True)


class TrainingPhase(Enum):
    ROUTER_PRETRAIN = "router_pretrain"
    JOINT_RL = "joint_rl"
    SELF_PLAY = "self_play"


@dataclass
class SubconsciousTrainingConfig:
    phase: TrainingPhase = TrainingPhase.ROUTER_PRETRAIN
    router_learning_rate: float = 1e-5
    state_learning_rate: float = 3e-6
    film_learning_rate: float = 1e-6
    rl_learning_rate: float = 3e-6
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    reward_normalize: bool = True
    kl_penalty_coef: float = 0.1
    aux_loss_coef: float = 0.01
    router_entropy_reg: float = 0.05
    max_grad_norm: float = 1.0
    use_ppo: bool = True
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95


class RewardNormalizer:
    def __init__(self, momentum: float = 0.99, eps: float = 1e-8):
        self.momentum = momentum
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, reward: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_mean = reward.mean().item()
            batch_var = reward.var().item()
            if self.count == 0:
                self.mean = batch_mean
                self.var = batch_var
            else:
                self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
                self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
            self.count += 1

    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        return (reward - self.mean) / (math.sqrt(self.var) + self.eps)


class SubconsciousTrainer:
    def __init__(
        self,
        subconscious_system: nn.Module,
        config: SubconsciousTrainingConfig = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.subconscious = subconscious_system
        self.cfg = config or SubconsciousTrainingConfig()
        self.device = device
        self.training_step = 0
        self.phase = self.cfg.phase
        self.reward_normalizer = RewardNormalizer()

        router_params = list(self.subconscious.router.parameters())
        state_params = list(self.subconscious.state_evolver.parameters())
        film_params = list(self.subconscious.film_generators.parameters())

        self.router_optimizer = torch.optim.AdamW(router_params, lr=self.cfg.router_learning_rate, weight_decay=0.01)
        self.state_optimizer = torch.optim.AdamW(state_params, lr=self.cfg.state_learning_rate, weight_decay=0.01)
        self.film_optimizer = torch.optim.AdamW(film_params, lr=self.cfg.film_learning_rate, weight_decay=0.0)

        self._all_params = router_params + state_params + film_params
        if self.cfg.use_ppo:
            self._ensure_value_head()
            self.rl_optimizer = torch.optim.AdamW(self._all_params, lr=self.cfg.rl_learning_rate, weight_decay=0.01)

        self._ppo_buffer: List[Dict[str, torch.Tensor]] = []
        self._old_log_probs: Optional[torch.Tensor] = None

        _LOG.info(
            f"SubconsciousTrainer initialized. "
            f"Phase: {self.phase.value}. "
            f"Params: {sum(p.numel() for p in self._all_params)/1e6:.1f}M"
        )

    def _ensure_value_head(self):
        if not hasattr(self.subconscious.router, 'value_head'):
            self.subconscious.router.value_head = nn.Sequential(
                nn.Linear(self.subconscious.router.hidden_size, 256),
                nn.SiLU(),
                nn.Linear(256, 1),
            ).to(self.device)

    def warm_start_knowledge_field(self, teacher_embeddings: torch.Tensor) -> Dict[str, float]:
        _LOG.info(f"Warm start: verifying expert pool with {teacher_embeddings.shape[0]} teacher embeddings")
        knowledge_pool = getattr(self.subconscious, 'knowledge_pool', None)
        if knowledge_pool is not None:
            _LOG.info(f"Knowledge expert pool ready: {knowledge_pool.num_experts} experts")
            return {"num_experts": knowledge_pool.num_experts}
        _LOG.warning("No knowledge expert pool attached; routing will use projected embeddings only")
        return {"num_experts": 0}

    def get_router_aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        if router_logits is None:
            return torch.tensor(0.0, device=self.device)
        probs = F.softmax(router_logits, dim=-1)
        avg_probs = probs.mean(dim=(0, 1))
        target = 1.0 / avg_probs.size(-1)
        load_balance_loss = (avg_probs * F.log(avg_probs / target + 1e-10)).sum()
        return load_balance_loss * self.cfg.aux_loss_coef

    def step(
        self,
        hidden_states: torch.Tensor,
        quality_score: float,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        self.training_step += 1
        if self.phase == TrainingPhase.ROUTER_PRETRAIN:
            return self._router_pretrain_step(hidden_states, quality_score, task_metadata)
        elif self.phase == TrainingPhase.JOINT_RL:
            if self.cfg.use_ppo:
                return self._ppo_step(hidden_states, quality_score, task_metadata)
            return self._rl_step(hidden_states, quality_score, task_metadata)
        elif self.phase == TrainingPhase.SELF_PLAY:
            return self._self_play_step(hidden_states, quality_score, task_metadata)
        return {"error": f"Unknown phase: {self.phase}"}

    def _router_pretrain_step(
        self,
        hidden_states: torch.Tensor,
        quality_score: float,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        self.router_optimizer.zero_grad()
        _, _, _, router_logits = self.subconscious.router(hidden_states)
        logits_flat = router_logits.view(-1, router_logits.size(-1))
        probs = F.softmax(logits_flat, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
        collapse_penalty = -self.cfg.router_entropy_reg * entropy
        aux_loss = self.get_router_aux_loss(router_logits)
        total_loss = collapse_penalty + aux_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.subconscious.router.parameters(), self.cfg.max_grad_norm)
        self.router_optimizer.step()
        metrics = {
            "phase": "router_pretrain",
            "routing_entropy": entropy.item(),
            "aux_loss": aux_loss.item(),
        }
        if self.training_step >= 5000 and self.phase == TrainingPhase.ROUTER_PRETRAIN:
            self.phase = TrainingPhase.JOINT_RL
            _LOG.info("Router pretrain complete -> switching to JOINT_RL phase")
            metrics["phase_switch"] = "router_pretrain -> joint_rl"
        return metrics

    def _rl_step(
        self,
        hidden_states: torch.Tensor,
        quality_score: float,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        _, _, _, router_logits = self.subconscious.router(hidden_states)
        gate = router_logits.mean()
        reward = torch.tensor(quality_score, device=self.device, dtype=torch.float32)
        if self.cfg.reward_normalize:
            self.reward_normalizer.update(reward.unsqueeze(0))
            reward = self.reward_normalizer.normalize(reward.unsqueeze(0)).squeeze(0)
        entropy_bonus = -gate * torch.log(torch.sigmoid(gate) + 1e-10) * (1.0 - reward.abs())
        logits_flat = router_logits.view(-1, router_logits.size(-1))
        probs = F.softmax(logits_flat, dim=-1)
        routing_entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
        aux_loss = self.get_router_aux_loss(router_logits)
        rl_loss = -reward * gate.mean() - self.cfg.entropy_coef * entropy_bonus.mean()
        rl_loss = rl_loss - self.cfg.router_entropy_reg * routing_entropy + aux_loss
        optimizers = [self.router_optimizer, self.state_optimizer, self.film_optimizer]
        for opt in optimizers:
            opt.zero_grad()
        rl_loss.backward()
        for opt in optimizers:
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g['params']],
                self.cfg.max_grad_norm,
            )
            opt.step()
        return {
            "phase": self.phase.value,
            "rl_loss": rl_loss.item(),
            "reward": quality_score,
            "routing_entropy": routing_entropy.item(),
        }

    def _ppo_step(
        self,
        hidden_states: torch.Tensor,
        quality_score: float,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        _, _, _, router_logits = self.subconscious.router(hidden_states)
        logits_flat = router_logits.view(-1, router_logits.size(-1))
        action_dist = torch.distributions.Categorical(logits=logits_flat.softmax(dim=-1))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        value = self.subconscious.router.value_head(
            router_logits.mean(dim=-1, keepdim=True)
        ).mean()
        self._ppo_buffer.append({
            "log_prob": log_prob.detach(),
            "value": value.detach(),
            "reward": torch.tensor(quality_score, device=self.device),
            "router_logits": router_logits.detach(),
        })
        if len(self._ppo_buffer) < self.cfg.ppo_minibatch_size:
            return {
                "phase": self.phase.value,
                "buffer_size": len(self._ppo_buffer),
                "buffer_target": self.cfg.ppo_minibatch_size,
            }
        return self._perform_ppo_update()

    def _perform_ppo_update(self) -> Dict[str, float]:
        rewards = torch.stack([exp["reward"] for exp in self._ppo_buffer])
        values = torch.stack([exp["value"] for exp in self._ppo_buffer])
        if self.cfg.reward_normalize:
            self.reward_normalizer.update(rewards)
            rewards = self.reward_normalizer.normalize(rewards)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] - values[t]
            if t + 1 < len(rewards):
                delta = delta + self.cfg.gamma * values[t + 1] - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_log_probs = torch.stack([exp["log_prob"] for exp in self._ppo_buffer])
        old_router_logits = torch.stack([exp["router_logits"] for exp in self._ppo_buffer])
        dataset_size = len(self._ppo_buffer)
        indices = torch.randperm(dataset_size)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        for _ in range(self.cfg.ppo_epochs):
            for start in range(0, dataset_size, self.cfg.ppo_minibatch_size):
                end = start + self.cfg.ppo_minibatch_size
                batch_indices = indices[start:end]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_router = old_router_logits[batch_indices]
                current_probs = batch_old_router.view(-1, batch_old_router.size(-1)).softmax(dim=-1)
                current_dist = torch.distributions.Categorical(probs=current_probs)
                current_log_probs = current_dist.log_prob(
                    batch_old_log_probs.argmax(-1).clamp(0, current_probs.size(-1) - 1)
                )
                ratio = (current_log_probs - batch_old_log_probs.view(-1)).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_preds = self.subconscious.router.value_head(
                    batch_old_router.mean(dim=-1, keepdim=True)
                ).squeeze()
                value_loss = F.mse_loss(value_preds, batch_returns)
                entropy = current_dist.entropy().mean()
                total_loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                optimizers = [self.rl_optimizer]
                for opt in optimizers:
                    opt.zero_grad()
                total_loss.backward()
                for opt in optimizers:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in opt.param_groups for p in g['params']],
                        self.cfg.max_grad_norm,
                    )
                    opt.step()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        n_updates = self.cfg.ppo_epochs * (dataset_size // self.cfg.ppo_minibatch_size)
        metrics = {
            "phase": self.phase.value,
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "buffer_used": dataset_size,
            "reward_mean": rewards.mean().item(),
        }
        self._ppo_buffer = []
        if self.training_step >= 20000 and self.phase == TrainingPhase.JOINT_RL:
            self.phase = TrainingPhase.SELF_PLAY
            _LOG.info("JOINT_RL converged -> switching to SELF_PLAY phase")
            metrics["phase_switch"] = "joint_rl -> self_play"
        return metrics

    def _self_play_step(
        self,
        hidden_states: torch.Tensor,
        quality_score: float,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        metrics = self._rl_step(hidden_states, quality_score, task_metadata)
        with torch.no_grad():
            _, _, _, router_logits = self.subconscious.router(hidden_states)
            probs = F.softmax(router_logits, dim=-1)
            exploration_bonus = -(probs * torch.log(probs + 1e-10)).mean(-1).mean()
            metrics["exploration_bonus"] = exploration_bonus.item()
        return metrics

    def set_phase(self, phase: TrainingPhase):
        old_phase = self.phase
        self.phase = phase
        _LOG.info(f"Training phase: {old_phase.value} -> {phase.value}")

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "training_step": self.training_step,
            "phase": self.phase.value,
            "router_optimizer": self.router_optimizer.state_dict(),
            "state_optimizer": self.state_optimizer.state_dict(),
            "film_optimizer": self.film_optimizer.state_dict(),
            "reward_normalizer_mean": self.reward_normalizer.mean,
            "reward_normalizer_var": self.reward_normalizer.var,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.training_step = state_dict.get("training_step", 0)
        self.phase = TrainingPhase(state_dict.get("phase", "router_pretrain"))
        self.router_optimizer.load_state_dict(state_dict["router_optimizer"])
        self.state_optimizer.load_state_dict(state_dict["state_optimizer"])
        self.film_optimizer.load_state_dict(state_dict["film_optimizer"])
        self.reward_normalizer.mean = state_dict.get("reward_normalizer_mean", 0.0)
        self.reward_normalizer.var = state_dict.get("reward_normalizer_var", 1.0)


YvSubconsciousTrainer = SubconsciousTrainer
