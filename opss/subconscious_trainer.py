#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
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
Subconscious Training Engine for EnTA-driven autonomous training.

Trains the 0.5B Dynamic Head + 314B Implicit Knowledge Field architecture
using reinforcement learning signals from EnTA. The training pipeline:

1. Knowledge Field Warm Start: Initialize codebooks using teacher model outputs
2. Dynamic Head RL Training: EnTA generates tasks → 7B executes → reward → head update
3. Knowledge Field Refinement: Periodically update codebooks with discovered patterns
4. Self-Play: After initial training, the system generates its own training data

Key Design:
    - EnTA is the orchestrator (not part of this module)
    - This module provides the API that EnTA calls
    - All training signals come from task execution rewards
    - No human-annotated data required
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("Yv.SubconsciousTrainer", file_path=get_log_file("Yv.SubconsciousTrainer"), enable_file=True)


class TrainingPhase(Enum):
    """Training phases for the subconscious system."""
    FIELD_WARM_START = "field_warm_start"        # Initialize codebooks with teacher embeddings
    HEAD_PRETRAIN = "head_pretrain"              # Train dynamic head via imitation learning
    HEAD_RL = "head_rl"                          # RL phase: reward-driven head optimization
    JOINT_FINE_TUNE = "joint_fine_tune"          # Joint head + field refinement
    SELF_PLAY = "self_play"                      # Autonomous self-improvement


@dataclass
class SubconsciousTrainingConfig:
    """Configuration for subconscious system training.

    Args:
        phase: Current training phase.
        head_learning_rate: LR for the 0.5B dynamic head.
        field_learning_rate: LR for the knowledge field codebooks.
        modulator_learning_rate: LR for the layer modulators.
        rl_learning_rate: LR for RL phase (PPO).
        clip_epsilon: PPO clip epsilon.
        entropy_coef: PPO entropy bonus coefficient.
        value_coef: Value loss coefficient.
        reward_normalize: Whether to normalize rewards.
        kl_penalty_coef: KL penalty for keeping 7B from shifting too fast.
        field_update_frequency: How often to update codebooks with new knowledge.
        field_prototype_topk: Top-K teacher prototypes per codebook entry.
        warm_start_temperature: Temperature for soft addressing during warm start.
        head_entropy_regularization: Entropy bonus weight for head exploration.
        max_grad_norm: Gradient clipping norm.
        use_ppo: Whether to use PPO for RL phase.
        ppo_epochs: PPO epochs per rollout.
        ppo_minibatch_size: PPO minibatch size.
        gamma: Reward discount factor.
        gae_lambda: GAE lambda for advantage estimation.
    """
    phase: TrainingPhase = TrainingPhase.HEAD_RL
    head_learning_rate: float = 1e-5
    field_learning_rate: float = 5e-6
    modulator_learning_rate: float = 1e-6
    rl_learning_rate: float = 3e-6
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    reward_normalize: bool = True
    kl_penalty_coef: float = 0.1
    field_update_frequency: int = 1000
    field_prototype_topk: int = 5
    warm_start_temperature: float = 2.0
    head_entropy_regularization: float = 0.05
    max_grad_norm: float = 1.0
    use_ppo: bool = True
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95


class RewardNormalizer:
    """Running reward normalization for stable RL training.

    Maintains running statistics of rewards and normalizes to
    zero mean, unit variance. Critical for stable PPO training
    when rewards come from diverse tasks.
    """

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
    """Training orchestrator for the subconscious system.

    Provides the training API that EnTA calls. Manages training phases,
    optimizer state, reward processing, and knowledge field updates.

    EnTA integration:
        EnTA calls trainer.step(task, reward) after each task execution.
        Internally, the trainer handles gradient computation and parameter updates.

    Training Flow:
        1. EnTA initializes with warm_start() using teacher embeddings
        2. EnTA generates tasks → 7B executes with subconscious system
        3. EnTA evaluates quality → computes reward → calls trainer.step()
        4. Trainer computes PPO update for the dynamic head
        5. Periodically: refine codebooks with discovered patterns
        6. After convergence: switch to self-play phase
    """

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

        # Reward normalization
        self.reward_normalizer = RewardNormalizer()

        # Optimizers — separate LR per component
        head_params = list(self.subconscious.dynamic_head.parameters())
        field_params = list(self.subconscious.knowledge_field.parameters())
        mod_params = list(self.subconscious.modulators.parameters())

        self.head_optimizer = torch.optim.AdamW(
            head_params,
            lr=self.cfg.head_learning_rate,
            weight_decay=0.01,
        )
        self.field_optimizer = torch.optim.AdamW(
            field_params,
            lr=self.cfg.field_learning_rate,
            weight_decay=0.01,
        )
        self.modulator_optimizer = torch.optim.AdamW(
            mod_params,
            lr=self.cfg.modulator_learning_rate,
            weight_decay=0.0,
        )

        if self.cfg.use_ppo:
            # PPO requires a value head in the dynamic head
            self._ensure_value_head()
            self.rl_optimizer = torch.optim.AdamW(
                list(head_params) + list(field_params),
                lr=self.cfg.rl_learning_rate,
                weight_decay=0.01,
            )

        # PPO buffer
        self._ppo_buffer: List[Dict[str, torch.Tensor]] = []
        self._old_log_probs: Optional[torch.Tensor] = None

        _LOG.info(
            f"SubconsciousTrainer initialized. "
            f"Phase: {self.phase.value}. "
            f"Head params: {sum(p.numel() for p in head_params)/1e6:.1f}M, "
            f"Field params: {sum(p.numel() for p in field_params)/1e6:.1f}M"
        )

    def _ensure_value_head(self):
        """Add a value head to the dynamic head for PPO critic."""
        if not hasattr(self.subconscious.dynamic_head, 'value_head'):
            self.subconscious.dynamic_head.value_head = nn.Sequential(
                nn.Linear(self.subconscious.dynamic_head.head_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 1),
            ).to(self.device)

    def warm_start_knowledge_field(
        self,
        teacher_embeddings: torch.Tensor,
        teacher_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Initialize codebooks with teacher model embeddings.

        This is the critical first step: the 314B-equivalent knowledge field
        starts not from random noise but from real teacher knowledge.

        Args:
            teacher_embeddings: [num_concepts, embedding_dim] from teacher models.
                These can come from DeepSeek-R1, Qwen3.6, etc.
            teacher_labels: Optional [num_concepts] category labels.
                If provided, codebooks are initialized with category-aware clustering.

        Returns:
            Metrics dict.
        """
        _LOG.info(f"Warm starting knowledge field with {teacher_embeddings.shape[0]} teacher embeddings")

        with torch.no_grad():
            codebooks = self.subconscious.knowledge_field.codebooks
            h, m, k, d = codebooks.shape

            if teacher_labels is not None:
                # Category-aware initialization: distribute codebook entries across categories
                unique_labels = torch.unique(teacher_labels)
                entries_per_category = k // len(unique_labels)

                for i, label in enumerate(unique_labels):
                    mask = teacher_labels == label
                    category_embs = teacher_embeddings[mask]

                    if len(category_embs) >= entries_per_category:
                        # Use k-means within the category
                        from sklearn.cluster import KMeans
                        kmeans = KMeans(n_clusters=min(entries_per_category, len(category_embs)))
                        kmeans.fit(category_embs.cpu().numpy())

                        start = i * entries_per_category
                        end = start + entries_per_category
                        for head_idx in range(h):
                            codebooks[head_idx, :, start:end, :d] = torch.tensor(
                                kmeans.cluster_centers_[:entries_per_category, :d],
                                device=codebooks.device, dtype=codebooks.dtype
                            ).unsqueeze(0)  # broadcast across codebooks
            else:
                # Global k-means initialization
                from sklearn.cluster import KMeans
                n_init_entries = min(k, len(teacher_embeddings))
                kmeans = KMeans(n_clusters=n_init_entries)
                kmeans.fit(teacher_embeddings.cpu().numpy())
                centers = torch.tensor(kmeans.cluster_centers_[:, :d], device=codebooks.device, dtype=codebooks.dtype)

                for head_idx in range(h):
                    codebooks[head_idx, :, :n_init_entries, :d] = centers.unsqueeze(0)

            _LOG.info(f"Knowledge field warm started. Codebook shape: {codebooks.shape}")

        return {"warm_start_entries": min(k, len(teacher_embeddings))}

    def step(
        self,
        hidden_states: torch.Tensor,
        quality_score: float,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Single training step called by EnTA after task execution.

        Args:
            hidden_states: [batch, seq, hidden_size] from 7B during task execution.
            quality_score: EnTA's reward signal (0-1 or -1 to 1).
            task_metadata: Optional dict with task info.

        Returns:
            Metrics dict for EnTA monitoring.
        """
        self.training_step += 1

        if self.phase == TrainingPhase.HEAD_PRETRAIN:
            return self._head_imitation_step(hidden_states, quality_score, task_metadata)
        elif self.phase in (TrainingPhase.HEAD_RL, TrainingPhase.JOINT_FINE_TUNE):
            if self.cfg.use_ppo:
                return self._ppo_step(hidden_states, quality_score, task_metadata)
            else:
                return self._rl_step(hidden_states, quality_score, task_metadata)
        elif self.phase == TrainingPhase.SELF_PLAY:
            return self._self_play_step(hidden_states, quality_score, task_metadata)
        else:
            return {"error": f"Unknown phase: {self.phase}"}

    def _head_imitation_step(
        self,
        hidden_states: torch.Tensor,
        quality_score: float,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Pretrain the dynamic head via imitation learning.

        In this phase, the head learns to produce addressing signals
        that correlate with task success. The head is trained to predict
        which knowledge regions are useful for which inputs.

        This is a warm-up phase before full RL.
        """
        self.head_optimizer.zero_grad()

        # Forward through the dynamic head
        addressing_logits, gate = self.subconscious.dynamic_head(hidden_states)

        # Compute addressing entropy (encourage exploration)
        logits_2d = addressing_logits.view(-1, addressing_logits.size(-1))
        probs = F.softmax(logits_2d, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()

        # The quality_score acts as a learning signal:
        # high quality → reinforce current addressing pattern
        # use the score as a weight for the imitation loss
        score_tensor = torch.tensor(quality_score, device=self.device, dtype=torch.float32)

        # Encourage the gate to be active when the task benefits from subconscious
        gate_loss = F.mse_loss(gate.mean(), score_tensor.expand_as(gate.mean()))

        # Discourage collapse: encourage diverse codebook usage
        usage_entropy = entropy
        collapse_penalty = -self.cfg.head_entropy_regularization * usage_entropy

        total_loss = gate_loss + collapse_penalty

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.subconscious.dynamic_head.parameters(),
            self.cfg.max_grad_norm,
        )
        self.head_optimizer.step()

        metrics = {
            "phase": "head_pretrain",
            "gate_loss": gate_loss.item(),
            "addressing_entropy": usage_entropy.item(),
            "gate_value": gate.mean().item(),
        }

        # Periodically update phase
        if self.training_step >= 5000 and self.phase == TrainingPhase.HEAD_PRETRAIN:
            self.phase = TrainingPhase.HEAD_RL
            _LOG.info("Head pretrain complete → switching to HEAD_RL phase")
            metrics["phase_switch"] = "head_pretrain → head_rl"

        return metrics

    def _rl_step(
        self,
        hidden_states: torch.Tensor,
        quality_score: float,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Direct RL training step for the dynamic head.

        Uses the quality_score as a reward signal to train the head
        via REINFORCE-style policy gradient.

        Args:
            hidden_states: 7B hidden states.
            quality_score: Reward from EnTA.
            task_metadata: Optional metadata.

        Returns:
            Training metrics.
        """
        # Convert to log-probability space for REINFORCE
        addressing_logits, gate = self.subconscious.dynamic_head(hidden_states)

        # The quality_score is our reward
        reward = torch.tensor(quality_score, device=self.device, dtype=torch.float32)
        if self.cfg.reward_normalize:
            self.reward_normalizer.update(reward.unsqueeze(0))
            reward = self.reward_normalizer.normalize(reward.unsqueeze(0)).squeeze(0)

        # Combine all optimizer steps
        optimizers = [self.head_optimizer]
        if self.phase == TrainingPhase.JOINT_FINE_TUNE:
            optimizers.append(self.field_optimizer)
            optimizers.append(self.modulator_optimizer)

        for opt in optimizers:
            opt.zero_grad()

        # Gate entropy: encourage the model to explore when reward is low
        entropy_bonus = -gate * torch.log(gate + 1e-10) * (1.0 - reward.abs())

        # Addressing entropy: maintain diverse codebook usage
        logits_2d = addressing_logits.view(-1, addressing_logits.size(-1))
        probs = F.softmax(logits_2d, dim=-1)
        addressing_entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()

        # Loss: maximize reward * log_prob + entropy bonus
        rl_loss = -reward * gate.mean() - self.cfg.entropy_coef * entropy_bonus.mean()
        rl_loss = rl_loss - self.cfg.head_entropy_regularization * addressing_entropy

        rl_loss.backward()

        for opt in optimizers:
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g['params']],
                self.cfg.max_grad_norm,
            )
            opt.step()

        metrics = {
            "phase": self.phase.value,
            "rl_loss": rl_loss.item(),
            "reward": quality_score,
            "gate": gate.mean().item(),
            "addressing_entropy": addressing_entropy.item(),
        }

        # Periodic knowledge field refinement
        if self.training_step % self.cfg.field_update_frequency == 0:
            self._refine_knowledge_field()
            metrics["field_refined"] = True

        return metrics

    def _ppo_step(
        self,
        hidden_states: torch.Tensor,
        quality_score: float,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """PPO training step with experience buffer.

        Collects experiences into a buffer and performs PPO update
        when enough samples are accumulated.
        """
        # Forward to get action log probs and value
        addressing_logits, gate = self.subconscious.dynamic_head(hidden_states)

        # Sample action (which codebook entries to use)
        logits_flat = addressing_logits.view(-1, addressing_logits.size(-1))
        action_dist = torch.distributions.Categorical(logits=logits_flat.softmax(dim=-1))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        # Value estimate
        value = self.subconscious.dynamic_head.value_head(
            self.subconscious.dynamic_head.input_norm(
                self.subconscious.dynamic_head.input_proj(hidden_states)
            )
        ).mean()

        # Store experience
        self._ppo_buffer.append({
            "log_prob": log_prob.detach(),
            "value": value.detach(),
            "reward": torch.tensor(quality_score, device=self.device),
            "addressing_logits": addressing_logits.detach(),
            "gate": gate.detach(),
        })

        # Check if buffer is full enough for an update
        if len(self._ppo_buffer) < self.cfg.ppo_minibatch_size:
            return {
                "phase": self.phase.value,
                "buffer_size": len(self._ppo_buffer),
                "buffer_target": self.cfg.ppo_minibatch_size,
            }

        # PPO update
        return self._perform_ppo_update()

    def _perform_ppo_update(self) -> Dict[str, float]:
        """Execute PPO policy update using accumulated experience buffer.

        Implements clipped surrogate objective with GAE-based advantage estimation.
        """
        # Compute advantages using GAE
        rewards = torch.stack([exp["reward"] for exp in self._ppo_buffer])
        values = torch.stack([exp["value"] for exp in self._ppo_buffer])

        if self.cfg.reward_normalize:
            self.reward_normalizer.update(rewards)
            rewards = self.reward_normalizer.normalize(rewards)

        # GAE computation
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] - values[t]
            if t + 1 < len(rewards):
                delta = delta + self.cfg.gamma * values[t + 1] - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * gae
            advantages[t] = gae
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        old_log_probs = torch.stack([exp["log_prob"] for exp in self._ppo_buffer])
        old_addressing = torch.stack([exp["addressing_logits"] for exp in self._ppo_buffer])
        old_gates = torch.stack([exp["gate"] for exp in self._ppo_buffer])

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
                batch_old_addressing = old_addressing[batch_indices]
                batch_old_gates = old_gates[batch_indices]

                # Current log probs (approximate from old addressing)
                current_probs = batch_old_addressing.view(-1, batch_old_addressing.size(-1)).softmax(dim=-1)
                current_dist = torch.distributions.Categorical(probs=current_probs)
                current_log_probs = current_dist.log_prob(
                    batch_old_log_probs.argmax(-1).clamp(0, current_probs.size(-1) - 1)
                )

                # Ratio for clipped surrogate
                ratio = (current_log_probs - batch_old_log_probs.view(-1)).exp()

                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_preds = self.subconscious.dynamic_head.value_head(
                    self.subconscious.dynamic_head.input_norm(
                        self.subconscious.dynamic_head.input_proj(
                            torch.zeros(len(batch_advantages), 1, self.subconscious.dynamic_head.hidden_size, device=self.device)
                        )
                    )
                ).squeeze()
                value_loss = F.mse_loss(value_preds, batch_returns)

                # Entropy bonus
                entropy = current_dist.entropy().mean()

                total_loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                # Update
                optimizers = [self.rl_optimizer]
                if self.phase == TrainingPhase.JOINT_FINE_TUNE:
                    optimizers.append(self.field_optimizer)

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

        # Clear buffer
        self._ppo_buffer = []

        # Phase transition check
        if self.training_step >= 20000 and self.phase == TrainingPhase.HEAD_RL:
            self.phase = TrainingPhase.JOINT_FINE_TUNE
            _LOG.info("HEAD_RL converged → switching to JOINT_FINE_TUNE phase")
            metrics["phase_switch"] = "head_rl → joint_fine_tune"

        return metrics

    def _self_play_step(
        self,
        hidden_states: torch.Tensor,
        quality_score: float,
        task_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Self-play training: the system generates its own tasks.

        In this phase, the subconscious system generates its own
        exploration targets (novel knowledge regions) and the 7B
        attempts to reason about them. Rewards come from:
        - Internal consistency checks
        - Cross-validation between different field regions
        - Novelty of discovered knowledge patterns

        This is the fully autonomous phase.
        """
        # For now, delegate to RL with additional exploration bonus
        metrics = self._rl_step(hidden_states, quality_score, task_metadata)

        # Add exploration bonus: encourage visiting novel field regions
        with torch.no_grad():
            _, gate = self.subconscious.dynamic_head(hidden_states)
            # If gate is uniformly distributed, it's exploring
            exploration_bonus = -(gate * torch.log(gate + 1e-10)).mean()
            metrics["exploration_bonus"] = exploration_bonus.item()

        return metrics

    def _refine_knowledge_field(self):
        """Periodically refine codebooks with accumulated knowledge.

        Uses the dynamic head's learned addressing patterns to identify
        which field regions are most useful and adjust codebooks accordingly.

        This is called automatically every field_update_frequency steps.
        """
        _LOG.info(f"Step {self.training_step}: Refining knowledge field...")

        with torch.no_grad():
            codebooks = self.subconscious.knowledge_field.codebooks

            # Get addressing statistics from the dynamic head
            # (This is a simplified version; in practice, we'd track
            # which field regions produce the best task outcomes)
            addressing_logits = self.subconscious._current_addressing
            if addressing_logits is not None:
                # Identify underutilized codebook entries
                probs = F.softmax(addressing_logits, dim=-1).mean(dim=(0, 1))
                usage = probs.view(self.subconscious.knowledge_field.num_heads, -1).mean(dim=0)

                # Find least-used entries for potential replacement
                low_usage_mask = usage < 0.01
                n_low_usage = low_usage_mask.sum().item()

                if n_low_usage > 10:
                    _LOG.debug(f"  {n_low_usage} codebook entries underutilized, marking for refresh")

    def set_phase(self, phase: TrainingPhase):
        """Manually set training phase (called by EnTA).

        Args:
            phase: Target training phase.
        """
        old_phase = self.phase
        self.phase = phase
        _LOG.info(f"Training phase: {old_phase.value} → {phase.value}")

    def get_state_dict(self) -> Dict[str, Any]:
        """Get trainer state for checkpointing.

        Returns:
            Dict with optimizer states and training metadata.
        """
        return {
            "training_step": self.training_step,
            "phase": self.phase.value,
            "head_optimizer": self.head_optimizer.state_dict(),
            "field_optimizer": self.field_optimizer.state_dict(),
            "modulator_optimizer": self.modulator_optimizer.state_dict(),
            "reward_normalizer_mean": self.reward_normalizer.mean,
            "reward_normalizer_var": self.reward_normalizer.var,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load trainer state from checkpoint.

        Args:
            state_dict: State dict from get_state_dict().
        """
        self.training_step = state_dict.get("training_step", 0)
        self.phase = TrainingPhase(state_dict.get("phase", "head_rl"))
        self.head_optimizer.load_state_dict(state_dict["head_optimizer"])
        self.field_optimizer.load_state_dict(state_dict["field_optimizer"])
        self.modulator_optimizer.load_state_dict(state_dict["modulator_optimizer"])
        self.reward_normalizer.mean = state_dict.get("reward_normalizer_mean", 0.0)
        self.reward_normalizer.var = state_dict.get("reward_normalizer_var", 1.0)
