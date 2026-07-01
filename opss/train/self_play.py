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
YvSelfPlayTrainer: Self-Play Reinforcement Learning with Self-Rewarding.

Implements a complete self-play training loop where the model generates
responses, evaluates them (self-critique), and trains on self-generated
preference pairs. The algorithm follows SPIN/Self-Rewarding paradigm:

    1. GENERATE:  Model produces num_samples responses per prompt
    2. CRITIQUE:  Model scores each response (self-reward / self-critic)
    3. SELECT:    Create preference pairs from best/worst responses
    4. TRAIN:     Update model via GRPO/DPO on the preference pairs
    5. REPEAT:    Iterate for num_rounds rounds

Key Design:
    - No external reward model required — uses model's own judgment
    - Optional POPSSGenerativeRewardModel head for denser reward signal
    - GRPO-style group-relative advantage when num_samples >= 2
    - Self-critique via perplexity/confidence/log-probability scoring

Reference:
    SPIN: Self-Play Fine-Tuning (arXiv:2401.01335, 2024)
    Self-Rewarding Language Models (NeurIPS 2024)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from configs.version import VERSION
from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig,
)


@dataclass
class POPSSSelfPlayConfig(PiscesLxOperatorConfig):
    name: str = "self_play"
    version: str = VERSION

    # Self-play loop
    num_rounds: int = 3
    num_samples: int = 4
    temperature: float = 0.8
    top_p: float = 0.9
    max_generation_length: int = 1024

    # Scoring
    score_method: str = "self_critic"
    self_critic_prompt: str = "Rate the quality of the following response on a scale of 1 to 10:"

    # Training
    grpo_group_size: int = 4
    grpo_temperature: float = 1.0
    grpo_clip_ratio: float = 0.2
    grpo_kl_coef: float = 0.1
    dpo_beta: float = 0.1

    # Efficiency
    max_prompt_length: int = 2048
    batch_size: int = 8
    learning_rate: float = 1e-6
    use_reward_model: bool = False
    reward_model_update_interval: int = 2

    # Generation diversity
    diversity_penalty: float = 0.05
    use_mmr_diversity: bool = True
    mmr_lambda: float = 0.5


class POPSSSelfPlayTrainer:
    """
    Self-Play Trainer orchestrating generate → critique → train → repeat.

    The trainer can operate in two modes:
        1. Self-Critic mode (default): model scores its own generations
        2. Reward Model mode: external POPSSGenerativeRewardModel scores

    Example:
        >>> config = POPSSSelfPlayConfig(num_rounds=3, num_samples=4)
        >>> trainer = YvSelfPlayTrainer(model, tokenizer, config)
        >>> result = trainer.train(prompts=training_prompts)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[POPSSSelfPlayConfig] = None,
        reward_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        reference_model: Optional[nn.Module] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or POPSSSelfPlayConfig()
        self.reward_model = reward_model
        self.reference_model = reference_model
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        self.round = 0
        self.history = []
        self._setup_scoring()

    def _setup_scoring(self):
        if self.config.use_reward_model and self.reward_model is not None:
            self._score_fn = self._score_with_reward_model
        else:
            self._score_fn = self._score_with_self_critic

    def _score_with_reward_model(
        self, prompt: str, responses: List[str]
    ) -> List[float]:
        scores = []
        for resp in responses:
            full_text = prompt + resp
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            with torch.no_grad():
                reward = self.reward_model(**inputs)
            scores.append(reward.item())
        return scores

    def _score_with_self_critic(
        self, prompt: str, responses: List[str]
    ) -> List[float]:
        scores = []
        for resp in responses:
            critic_text = f"{self.config.self_critic_prompt}\n\n{prompt}\n{resp}"
            inputs = self.tokenizer(critic_text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs["logits"]
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                confidence = 1.0 - (entropy / np.log(logits.size(-1)))
                length_norm = inputs["input_ids"].shape[1]
            scores.append((confidence.item() * length_norm) / max(length_norm, 1.0))
        return scores

    def _generate_responses(self, prompt: str) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_prompt_length)
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        responses = []
        for _ in range(self.config.num_samples):
            with torch.no_grad():
                gen_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=self.config.max_generation_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                )
            gen_text = self.tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            responses.append(gen_text)

        scores = self._score_fn(prompt, responses)

        if self.config.use_mmr_diversity:
            scores = self._apply_mmr_diversity(responses, scores)

        return list(zip(responses, scores))

    def _apply_mmr_diversity(
        self, responses: List[str], scores: List[float]
    ) -> List[float]:
        if len(responses) < 2:
            return scores

        response_embeds = []
        for resp in responses:
            inputs = self.tokenizer(resp, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
                if isinstance(out, dict):
                    hidden = out.get("hidden_states", [None])[-1]
                else:
                    hidden = out.hidden_states[-1] if hasattr(out, 'hidden_states') else None
                if hidden is not None:
                    response_embeds.append(hidden.mean(dim=1).detach())
                else:
                    response_embeds.append(None)

        adjusted = list(scores)
        for i in range(len(responses)):
            if response_embeds[i] is None:
                continue
            diversity_penalty = 0.0
            for j in range(len(responses)):
                if i != j and response_embeds[j] is not None:
                    sim = F.cosine_similarity(response_embeds[i], response_embeds[j]).item()
                    diversity_penalty += max(0, sim - 0.5) * self.config.mmr_lambda
            adjusted[i] = scores[i] - diversity_penalty

        return adjusted

    def _create_preference_pairs(
        self, scored_responses: List[Tuple[str, float]]
    ) -> List[Tuple[str, str, float]]:
        pairs = []
        if len(scored_responses) < 2:
            return pairs

        sorted_resp = sorted(scored_responses, key=lambda x: x[1], reverse=True)
        best = sorted_resp[0]
        worst = sorted_resp[-1]
        pairs.append((best[0], worst[0], best[1] - worst[1]))
        return pairs

    def _compute_grpo_advantages(
        self, rewards: List[float]
    ) -> List[float]:
        if len(rewards) < 2:
            return [0.0] * len(rewards)
        r = torch.tensor(rewards)
        mean = r.mean()
        std = r.std().clamp(min=1e-6)
        advantages = (r - mean) / std
        return advantages.tolist()

    def _train_on_preferences(
        self, prompt: str, pairs: List[Tuple[str, str, float]]
    ) -> Dict[str, float]:
        if not pairs:
            return {"loss": 0.0, "accuracy": 0.0}

        total_loss = 0.0
        n_correct = 0

        for chosen_resp, rejected_resp, margin in pairs:
            chosen_text = prompt + chosen_resp
            rejected_text = prompt + rejected_resp

            chosen_inputs = self.tokenizer(chosen_text, return_tensors="pt", truncation=True, max_length=self.config.max_prompt_length + self.config.max_generation_length)
            rejected_inputs = self.tokenizer(rejected_text, return_tensors="pt", truncation=True, max_length=self.config.max_prompt_length + self.config.max_generation_length)

            chosen_ids = chosen_inputs["input_ids"].to(next(self.model.parameters()).device)
            rejected_ids = rejected_inputs["input_ids"].to(next(self.model.parameters()).device)
            chosen_mask = chosen_inputs["attention_mask"].to(next(self.model.parameters()).device)
            rejected_mask = rejected_inputs["attention_mask"].to(next(self.model.parameters()).device)

            chosen_out = self.model(input_ids=chosen_ids, attention_mask=chosen_mask, labels=chosen_ids)
            rejected_out = self.model(input_ids=rejected_ids, attention_mask=rejected_mask, labels=rejected_ids)

            chosen_loss = chosen_out["loss"]
            rejected_loss = rejected_out["loss"]

            dpo_loss = -F.logsigmoid(self.config.dpo_beta * (rejected_loss - chosen_loss)).mean()
            total_loss = total_loss + dpo_loss

            if chosen_loss < rejected_loss:
                n_correct = n_correct + 1

        avg_loss = total_loss / max(len(pairs), 1)

        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": avg_loss.item(),
            "accuracy": n_correct / max(len(pairs), 1),
        }

    def train_step(
        self, prompts: List[str]
    ) -> Dict[str, Any]:
        self.model.train()
        total_metrics = {"loss": 0.0, "accuracy": 0.0, "avg_reward": 0.0, "n_prompts": 0}

        for prompt in prompts:
            scored_responses = self._generate_responses(prompt)
            if len(scored_responses) < 2:
                continue

            pairs = self._create_preference_pairs(scored_responses)
            if not pairs:
                continue

            metrics = self._train_on_preferences(prompt, pairs)
            avg_reward = sum(s for _, s in scored_responses) / len(scored_responses)

            for k in total_metrics:
                if k in metrics:
                    total_metrics[k] = total_metrics[k] + metrics[k]
            total_metrics["avg_reward"] = total_metrics["avg_reward"] + avg_reward
            total_metrics["n_prompts"] = total_metrics["n_prompts"] + 1

        n = max(total_metrics["n_prompts"], 1)
        for k in ["loss", "accuracy", "avg_reward"]:
            total_metrics[k] = total_metrics[k] / n

        return total_metrics

    def train(self, prompts: List[str]) -> Dict[str, Any]:
        final_metrics = {"rounds_completed": 0, "final_loss": 0.0, "final_reward": 0.0}
        self.round = 0

        for rnd in range(self.config.num_rounds):
            self.round = rnd + 1
            round_metrics = self.train_step(prompts)

            self.history.append({
                "round": self.round,
                "metrics": round_metrics,
            })

            final_metrics["rounds_completed"] = self.round
            final_metrics["final_loss"] = round_metrics["loss"]
            final_metrics["final_reward"] = round_metrics["avg_reward"]

        return final_metrics

    def save_checkpoint(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "round": self.round,
            "history": self.history,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.round = ckpt["round"]
        self.history = ckpt["history"]
