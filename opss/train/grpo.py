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
GRPO (Group Relative Policy Optimization) Operator

Complete implementation of DeepSeek R1's GRPO algorithm for preference alignment.
GRPO eliminates the need for a Critic network by using group-relative advantage estimation.

Key Innovation:
    - No Critic network required (saves 30-40% memory)
    - Group-relative advantage: A_i = (r_i - mean(r_group)) / std(r_group)
    - PPO-style clipped objective for stable training
    - KL divergence constraint to prevent deviation from reference model

Reference:
    DeepSeek R1 Technical Report (arXiv:2501.12948)

Algorithm:
    1. Sample group_size responses for each prompt
    2. Compute rewards for each response
    3. Calculate group-relative advantages
    4. Update policy with clipped objective
    5. Apply KL penalty to stay close to reference model
"""

import contextlib
import math

import torch

from .dapo import YvDAPO
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

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

    use_fp8: bool = False

    # FP16/AMP loss scaling
    use_amp: bool = False
    loss_scale: float = 128.0
    loss_scale_window: int = 2000
    min_loss_scale: float = 1.0
    max_loss_scale: float = 32768.0

    ppo_epochs: int = 4
    mini_batch_size: int = 4

    enable_self_verification: bool = False
    verification_weight: float = 0.3
    max_refinement_iterations: int = 3
    refinement_reward_threshold: float = 0.01

    # DAPO: Decoupled Clipping and Dynamic Sampling
    use_dapo: bool = False
    dapo_epsilon_low: float = 0.2
    dapo_epsilon_high: float = 0.4
    dapo_diversity_threshold: float = 0.3

    # ── iGRPO: Two-Stage Self-Conditioning (Yv Architecture, Dunimd Team) ──
    # First stage generates draft trajectories; second stage conditions
    # on the best draft to produce refined trajectories.  Advantage is
    # computed relative to the best draft (self-conditioned baseline).
    use_igrpo: bool = False
    igrpo_draft_ratio: float = 0.5    # fraction of group used as drafts
    igrpo_conditioning_strength: float = 0.3  # blend weight for draft baseline
    igrpo_draft_temperature: float = 1.2      # higher temperature for draft diversity
    igrpo_refinement_temperature: float = 0.8 # lower temperature for focused refinement
    igrpo_self_consistent_weight: float = 0.2 # weight for self-consistency bonus

    # ── GraphPO: Graph-Based Trajectory Exploration ──────────────
    # Expands trajectories into a decision graph, branching at key
    # decision points (tool calls, reasoning forks).  Rewards are
    # propagated back through graph paths for structured credit
    # assignment.
    use_graphpo: bool = False
    graphpo_max_branches: int = 3       # max branches per decision node
    graphpo_depth_penalty: float = 0.05 # penalty per depth level
    graphpo_exploration_bonus: float = 0.1  # bonus for visiting new nodes
    graphpo_reward_discount: float = 0.95   # discount factor for backprop
    graphpo_top_paths: int = 5          # keep top-K paths for training

    # ── CoDaPO: Confidence/Difficulty Adaptive (Yv Architecture, Dunimd Team) ──
    # Dynamically adjusts clipping range, KL penalty, and temperature
    # based on per-response confidence scores and task difficulty.
    use_codapo: bool = False
    codapo_confidence_threshold_low: float = 0.3
    codapo_confidence_threshold_high: float = 0.7
    codapo_clip_low_confidence: float = 0.1    # tighter clip for low confidence
    codapo_clip_high_confidence: float = 0.3   # wider clip for high confidence
    codapo_kl_low_confidence: float = 0.05     # lower KL for uncertain responses
    codapo_kl_high_confidence: float = 0.15    # higher KL for confident responses
    codapo_difficulty_adapt_temperature: bool = True

    # ── GRPO-VPS: Verifiable Process Supervision (Yv Architecture, Dunimd Team) ──
    # Splits responses into reasoning steps, assigns per-step verifiable
    # rewards (correctness + process quality), and combines with outcome
    # reward for finer-grained supervision.
    use_vps: bool = False
    vps_outcome_weight: float = 0.6
    vps_process_weight: float = 0.4
    vps_step_delimiter: str = "\n"
    vps_min_steps: int = 1
    vps_quality_scale: float = 1.0

    # ── MMR-GRPO: Diversity-aware Multi-Model Refinement (Yv Architecture, Dunimd Team) ──
    # Maintains a diversity buffer of recent successful trajectories.
    # During training, mixes current policy trajectories with diverse
    # historical ones and rewards diverse trajectories to prevent mode
    # collapse in reinforcement learning.
    use_mmr_grpo: bool = False
    mmr_buffer_size: int = 32
    mmr_diversity_weight: float = 0.2
    mmr_similarity_threshold: float = 0.85
    mmr_mix_ratio: float = 0.3
    mmr_embedding_dim: int = 64

    # ── TR-GRPO: Token-Level Reward Weighting (Yv Architecture, Dunimd Team) ──
    # Assigns per-token advantages based on token importance/surprise
    # estimated from policy probability change.  Provides more fine-grained
    # credit assignment than sequence-level GRPO.
    use_tr_grpo: bool = False
    tr_importance_scale: float = 1.0
    tr_importance_bias: float = 0.1
    tr_token_clip_ratio: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO")


@dataclass
class POPSSAgenticRLConfig(PiscesLxOperatorConfig):
    """
    Agentic RL Post-Training Configuration.

    Extends GRPO with agent-environment interaction capabilities.
    Enables multi-step agent rollouts with tool use and task completion rewards.

    Attributes:
        agent_rollout_steps: Number of agent steps per rollout
        max_tool_calls: Maximum tool calls per episode
        tool_call_reward: Reward per successful tool call
        task_completion_reward: Reward for task completion
        efficiency_penalty: Penalty per unnecessary step
        use_agentic_grpo: Whether to enable agentic GRPO training
        grpo_config: Underlying GRPO configuration for group-based advantage
    """
    name: str = "agentic_rl"
    version: str = VERSION

    agent_rollout_steps: int = 8
    max_tool_calls: int = 20
    tool_call_reward: float = 0.1
    task_completion_reward: float = 1.0
    efficiency_penalty: float = -0.01
    use_agentic_grpo: bool = False

    grpo_config: POPSSGRPOConfig = field(default_factory=POPSSGRPOConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.agent_rollout_steps < 1:
            raise ValueError("agent_rollout_steps must be at least 1")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")


@dataclass
class TrajectoryEntry:
    """Entry in the MMR-GRPO diversity buffer.

    Stores the embedding of a trajectory and its associated rewards
    for diversity-aware sampling and mode-collapse prevention.
    """
    embedding: torch.Tensor
    reward: float = 0.0
    diversity_score: float = 0.0


# Paper: DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", arXiv:2501.12948
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
        self._mmr_diversity_buffer = None
    
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
            
            # Determine active algorithm mode
            alg_modes = []
            if getattr(config, 'use_igrpo', False):
                alg_modes.append("iGRPO")
            if getattr(config, 'use_graphpo', False):
                alg_modes.append("GraphPO")
            if getattr(config, 'use_codapo', False):
                alg_modes.append("CoDaPO")
            if getattr(config, 'use_dapo', False):
                alg_modes.append("DAPO")
            if getattr(config, 'use_vps', False):
                alg_modes.append("GRPO-VPS")
            if getattr(config, 'use_mmr_grpo', False):
                alg_modes.append("MMR-GRPO")
            if getattr(config, 'use_tr_grpo', False):
                alg_modes.append("TR-GRPO")
            if not alg_modes:
                alg_modes.append("GRPO")

            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=output,
                execution_time=execution_time,
                metadata={
                    "version": self.version,
                    "algorithm": "+".join(alg_modes),
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
            "verification_rewards", "refinement_iterations",
        ]}

        # ── Route through selected algorithm mode ─────────────────
        use_igrpo = getattr(config, 'use_igrpo', False)
        use_graphpo = getattr(config, 'use_graphpo', False)
        use_codapo = getattr(config, 'use_codapo', False)
        use_vps = getattr(config, 'use_vps', False)
        use_mmr_grpo = getattr(config, 'use_mmr_grpo', False)
        use_tr_grpo = getattr(config, 'use_tr_grpo', False)

        if use_igrpo:
            # iGRPO: Two-stage self-conditioned sampling
            (responses, log_probs, old_log_probs,
             consistency_scores) = self._igrpo_two_stage_sample(
                model=model, prompt=prompt, config=config,
                tokenizer=tokenizer, reward_function=reward_function,
            )
            refined_responses, verification_rewards = self._apply_iterative_refinement(
                model=model, prompt=prompt, responses=responses,
                config=config, tokenizer=tokenizer,
            )
            final_responses = refined_responses if config.enable_self_verification else responses
            combined_rewards = self._compute_rewards(final_responses, prompt, reward_function)
            rewards_tensor = torch.tensor(combined_rewards, dtype=torch.float32)
            advantages = self._igrpo_self_conditioned_advantages(
                rewards=rewards_tensor, consistency_scores=consistency_scores,
                group_size=config.group_size, config=config,
            )
        elif use_graphpo:
            # GraphPO: Graph-based trajectory expansion
            (responses, log_probs, old_log_probs,
             graph_rewards) = self._graphpo_expand_trajectories(
                model=model, prompt=prompt, config=config,
                tokenizer=tokenizer, reward_function=reward_function,
            )
            refined_responses, verification_rewards = self._apply_iterative_refinement(
                model=model, prompt=prompt, responses=responses,
                config=config, tokenizer=tokenizer,
            )
            final_responses = refined_responses if config.enable_self_verification else responses
            rewards_tensor = graph_rewards
            advantages = self._graphpo_compute_advantages(
                rewards=rewards_tensor, group_size=config.group_size, config=config,
            )
        else:
            # Standard GRPO sampling (with optional VPS, MMR, TR extensions)
            responses, log_probs, old_log_probs = self._sample_group_responses(
                model=model, prompt=prompt, group_size=config.group_size,
                config=config, tokenizer=tokenizer,
            )
            refined_responses, verification_rewards = self._apply_iterative_refinement(
                model=model, prompt=prompt, responses=responses,
                config=config, tokenizer=tokenizer,
            )
            if config.enable_self_verification and any(v > 0 for v in verification_rewards):
                final_responses = refined_responses
                combined_rewards = []
                for i, (original_reward, refined_reward) in enumerate(zip(
                    self._compute_rewards(responses, prompt, reward_function),
                    verification_rewards
                )):
                    combined_reward = original_reward + config.verification_weight * refined_reward
                    combined_rewards.append(combined_reward)
            else:
                final_responses = responses
                combined_rewards = self._compute_rewards(
                    responses=responses, prompt=prompt, reward_function=reward_function,
                )
            rewards_tensor = torch.tensor(combined_rewards, dtype=torch.float32)

            # ── VPS: Verifiable Process Supervision ──
            if use_vps:
                rewards_tensor, vps_scores = self._compute_vps_rewards(
                    sequences=final_responses, rewards=rewards_tensor,
                    tokenizer=tokenizer, config=config,
                    prompt=prompt, reward_function=reward_function,
                )
                stats["vps_step_scores"] = [s.tolist() for s in vps_scores]

            # ── MMR: Diversity-aware Multi-Model Refinement ──
            if use_mmr_grpo:
                self._mmr_update_buffer(
                    trajectories=final_responses, rewards=rewards_tensor,
                    config=config, tokenizer=tokenizer,
                    device=rewards_tensor.device,
                )
                rewards_tensor = self._mmr_diversify_rewards(
                    responses=final_responses, rewards=rewards_tensor,
                    config=config, tokenizer=tokenizer,
                    device=rewards_tensor.device,
                )

            # ── Advantage computation (TR or standard) ──
            if use_tr_grpo:
                token_lp, token_old_lp, token_mask = self._compute_token_level_log_probs(
                    model=model, prompt=prompt, responses=final_responses,
                    tokenizer=tokenizer, config=config,
                )
                token_adv = self._compute_token_advantages(
                    old_log_probs=token_old_lp, new_log_probs=token_lp,
                    rewards=rewards_tensor, mask=token_mask, config=config,
                )
                tr_data = (token_lp, token_old_lp, token_mask, token_adv)
                advantages = token_adv.mean(dim=-1)
            else:
                tr_data = None
                advantages = self.compute_group_advantages(
                    rewards=rewards_tensor, group_size=config.group_size,
                    normalize=config.advantage_normalization, min_std=config.min_std,
                )

        # ── Reference log probabilities ─────────────────────────
        if use_tr_grpo and not use_igrpo and not use_graphpo:
            if reference_model and config.use_reference_model:
                token_ref_lp, _, _ = self._compute_token_level_log_probs(
                    model=reference_model, prompt=prompt,
                    responses=final_responses, tokenizer=tokenizer, config=config,
                )
            else:
                token_ref_lp = torch.zeros_like(token_lp)
            ref_log_probs = token_ref_lp
        elif reference_model and config.use_reference_model:
            ref_log_probs = self._compute_reference_log_probs(
                reference_model=reference_model, prompt=prompt,
                responses=final_responses, tokenizer=tokenizer, config=config,
            )
        else:
            ref_log_probs = torch.zeros_like(log_probs)

        if config.enable_self_verification and not use_igrpo and not use_graphpo and not use_tr_grpo:
            refined_log_probs = []
            for response in final_responses:
                if tokenizer:
                    full_text = prompt + response
                    input_ids = tokenizer.encode(full_text, return_tensors="pt").to(next(model.parameters()).device)
                else:
                    input_ids = torch.tensor([[ord(c) for c in prompt + response]], dtype=torch.long, device=next(model.parameters()).device)
                fp8_context = te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling(
                    margin=0, interval=1, fp8_format=Format.HYBRID, amax_history_len=1024, amax_compute_algo="max",
                )) if config and config.use_fp8 else contextlib.nullcontext()
                with fp8_context:
                    outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                log_probs_response = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs_response[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                refined_log_probs.append(token_log_probs.sum())
            refined_log_probs_tensor = torch.stack(refined_log_probs)
            refined_log_probs_tensor = torch.where(
                torch.isfinite(refined_log_probs_tensor),
                refined_log_probs_tensor,
                torch.zeros_like(refined_log_probs_tensor),
            )
            log_probs = refined_log_probs_tensor

        for epoch in range(config.ppo_epochs):
            if use_tr_grpo and not use_igrpo and not use_graphpo:
                token_lp, token_old_lp, token_mask, token_adv = tr_data
                epoch_stats = self._tr_ppo_update(
                    model=model,
                    token_log_probs=token_lp,
                    token_old_log_probs=token_old_lp,
                    token_ref_log_probs=ref_log_probs,
                    token_advantages=token_adv,
                    mask=token_mask,
                    config=config,
                    optimizer=optimizer,
                )
            elif use_codapo:
                epoch_stats = self._codapo_adaptive_update(
                    model=model, log_probs=log_probs, old_log_probs=old_log_probs,
                    ref_log_probs=ref_log_probs, advantages=advantages,
                    config=config, optimizer=optimizer,
                )
            else:
                epoch_stats = self._ppo_update(
                    model=model, log_probs=log_probs, old_log_probs=old_log_probs,
                    ref_log_probs=ref_log_probs, advantages=advantages,
                    config=config, optimizer=optimizer,
                )
            for key, values in epoch_stats.items():
                stats[key].extend(values)

        stats["rewards"].extend(combined_rewards if not use_graphpo else rewards_tensor.tolist())
        stats["advantages"].extend(advantages.tolist())
        stats["verification_rewards"].extend(verification_rewards)
        stats["refinement_iterations"].append(sum(1 for v in verification_rewards if v > 0))

        return stats
    
    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
        normalize: bool = True,
        min_std: float = 1e-8,
    ) -> torch.Tensor:
        rewards = rewards.view(-1, group_size)
        rewards = torch.where(torch.isfinite(rewards), rewards, torch.zeros_like(rewards))

        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True).clamp(min=min_std)

        if normalize:
            advantages = (rewards - mean) / std
        else:
            advantages = rewards - mean

        advantages = torch.where(torch.isfinite(advantages), advantages, torch.zeros_like(advantages))
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

        past_key_values = None
        log_probs_sum = torch.tensor(0.0, device=device)
        generated_ids = input_ids

        fp8_context = te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling(
            margin=0, interval=1, fp8_format=Format.HYBRID, amax_history_len=1024, amax_compute_algo="max",
        )) if config and config.use_fp8 else contextlib.nullcontext()

        with fp8_context:
            for _ in range(config.max_new_tokens):
                outputs = model(
                    input_ids=generated_ids[:, -1:] if generated_ids.shape[1] > 1 else generated_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                ) if hasattr(model, 'forward') else model(generated_ids)

                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None

                next_token_logits = logits[:, -1, :] / config.temperature if config.temperature > 0 else logits[:, -1, :]

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

                log_probs_for_token = F.log_softmax(next_token_logits, dim=-1)

                if config.temperature > 0:
                    probs = torch.exp(log_probs_for_token)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(log_probs_for_token, dim=-1, keepdim=True)

                token_log_prob = log_probs_for_token.gather(1, next_token)
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

    def _compute_verification_reward(
        self,
        original_response: str,
        refined_response: str,
        prompt: str,
    ) -> float:
        """Compute consistency-based verification reward between original and refined response."""
        original_lower = original_response.lower()
        refined_lower = refined_response.lower()

        original_words = set(original_lower.split())
        refined_words = set(refined_lower.split())

        if len(original_words) == 0:
            return 0.0

        word_overlap = len(original_words & refined_words) / max(len(original_words), len(refined_words))

        if original_lower == refined_lower:
            consistency_score = 1.0
        elif original_lower in refined_lower or refined_lower in original_lower:
            consistency_score = 0.9
        else:
            semantic_similarity = word_overlap
            consistency_score = semantic_similarity * 0.5

        return consistency_score

    def _refine_response(
        self,
        model: nn.Module,
        prompt: str,
        original_response: str,
        config: POPSSGRPOConfig,
        tokenizer,
    ) -> Tuple[str, float]:
        """Refine a response through self-verification and improvement."""
        device = next(model.parameters()).device

        current_response = original_response
        best_response = original_response
        best_score = 0.0

        refinement_history = [original_response]

        for iteration in range(config.max_refinement_iterations):
            verification_prompt = f"{prompt}\n\nOriginal response: {current_response}\n\nPlease verify if the response is correct and provide an improved version if needed:"

            if tokenizer:
                input_ids = tokenizer.encode(verification_prompt, return_tensors="pt").to(device)
            else:
                input_ids = torch.tensor([[ord(c) for c in verification_prompt]], dtype=torch.long, device=device)

            max_new_tokens = min(len(current_response.split()) * 2, config.max_new_tokens)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=config.temperature * 0.8,
                    top_p=config.top_p,
                )

            refined_text = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            if hasattr(refined_text, 'generated_text'):
                refined_response = refined_text.generated_text
            else:
                if tokenizer:
                    refined_response = tokenizer.decode(refined_text[0], skip_special_tokens=True)
                else:
                    refined_response = "".join(chr(c) for c in refined_text[0].tolist())

            verification_reward = self._compute_verification_reward(
                original_response=current_response,
                refined_response=refined_response,
                prompt=prompt,
            )

            if verification_reward > best_score:
                best_score = verification_reward
                best_response = refined_response
                current_response = refined_response
            else:
                break

            refinement_history.append(refined_response)

        return best_response, best_score

    def _apply_iterative_refinement(
        self,
        model: nn.Module,
        prompt: str,
        responses: List[str],
        config: POPSSGRPOConfig,
        tokenizer,
    ) -> Tuple[List[str], List[float]]:
        """Apply iterative refinement to responses with self-verification."""
        if not config.enable_self_verification:
            return responses, [0.0] * len(responses)

        refined_responses = []
        verification_rewards = []

        for original_response in responses:
            refined_response, verification_reward = self._refine_response(
                model=model,
                prompt=prompt,
                original_response=original_response,
                config=config,
                tokenizer=tokenizer,
            )
            refined_responses.append(refined_response)
            verification_rewards.append(verification_reward)

        return refined_responses, verification_rewards
    
    def _compute_reference_log_probs(
        self,
        reference_model: nn.Module,
        prompt: str,
        responses: List[str],
        tokenizer,
        config: Optional[POPSSGRPOConfig] = None,
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
                
                fp8_context = te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling(
                    margin=0, interval=1, fp8_format=Format.HYBRID, amax_history_len=1024, amax_compute_algo="max",
                )) if config and config.use_fp8 else contextlib.nullcontext()
                
                with fp8_context:
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
        """Perform PPO-style policy update with optional DAPO."""
        stats = {
            "policy_losses": [],
            "kl_divergences": [],
            "entropies": [],
            "clip_fractions": [],
            "approx_kl": [],
        }

        if optimizer:
            optimizer.zero_grad()

        # DAPO: Decoupled clipping and dynamic sampling
        if getattr(config, 'use_dapo', False):
            dapo = YvDAPO(
                epsilon_low=config.dapo_epsilon_low,
                epsilon_high=config.dapo_epsilon_high,
                beta=config.kl_coef,
                diversity_threshold=config.dapo_diversity_threshold
            )

            # Recompute advantages with DAPO decoupled normalization
            advantages_dapo = dapo.compute_decoupled_advantages(advantages)

            # Compute DAPO policy loss with asymmetric clipping
            total_loss, metrics = dapo.compute_policy_loss(
                log_probs=log_probs,
                ref_log_probs=ref_log_probs,
                advantages=advantages_dapo
            )

            policy_loss = torch.tensor(metrics["policy_loss"], device=log_probs.device)
            kl_div = torch.tensor(metrics["kl_div"], device=log_probs.device)
            entropy = -log_probs.mean()

            if optimizer and total_loss.requires_grad:
                scaled_loss = self._maybe_scale_loss(total_loss, config)
                scaled_loss.backward()
                self._maybe_unscale_grads(model, config)
                max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            clip_fraction = ((torch.exp((log_probs - old_log_probs).clamp(min=-10.0, max=10.0)) - 1.0).abs() > config.clip_ratio).float().mean()
            approx_kl = (old_log_probs - log_probs).mean().abs()

            stats["policy_losses"].append(policy_loss.item())
            stats["kl_divergences"].append(kl_div.item())
            stats["entropies"].append(entropy.item())
            stats["clip_fractions"].append(clip_fraction.item())
            stats["approx_kl"].append(approx_kl.item())

            return stats

        # Standard PPO update
        ratio = torch.exp((log_probs - old_log_probs).clamp(min=-10.0, max=10.0))

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        kl_div = (log_probs - ref_log_probs).clamp(min=-10.0, max=10.0).mean()

        entropy = -(log_probs.clamp(min=-10.0).mean())

        total_loss = (
            policy_loss +
            config.kl_coef * kl_div -
            config.entropy_coef * entropy
        )

        if optimizer and total_loss.requires_grad:
            scaled_loss = self._maybe_scale_loss(total_loss, config)
            scaled_loss.backward()
            self._maybe_unscale_grads(model, config)
            max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).float().mean()
        approx_kl = (old_log_probs - log_probs).mean().abs()

        stats["policy_losses"].append(policy_loss.item())
        stats["kl_divergences"].append(kl_div.item())
        stats["entropies"].append(entropy.item())
        stats["clip_fractions"].append(clip_fraction.item())
        stats["approx_kl"].append(approx_kl.item())

        return stats
    
    # ── iGRPO: Two-Stage Self-Conditioning ──────────────────────────────

    def _igrpo_two_stage_sample(
        self,
        model: nn.Module,
        prompt: str,
        config: POPSSGRPOConfig,
        tokenizer,
        reward_function,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two-stage self-conditioned sampling for iGRPO.

        Stage 1: Generate draft responses at high temperature.
        Stage 2: Select best drafts, condition refinement at lower temperature.
        Returns responses, log_probs, old_log_probs, and self-consistency scores.
        """
        draft_size = max(1, int(config.group_size * config.igrpo_draft_ratio))
        refine_size = config.group_size - draft_size

        # Stage 1: Draft generation
        draft_responses = []
        draft_log_probs = []
        model.eval()
        with torch.no_grad():
            orig_temperature = getattr(config, 'temperature', 1.0)
            config.temperature = getattr(config, 'igrpo_draft_temperature', 1.2)
            for _ in range(draft_size):
                response, log_prob = self._generate_response(
                    model=model, prompt=prompt, config=config, tokenizer=tokenizer,
                )
                draft_responses.append(response)
                draft_log_probs.append(log_prob)
            config.temperature = orig_temperature

        # Score drafts and select best for conditioning
        draft_rewards = self._compute_rewards(draft_responses, prompt, reward_function)
        draft_scores = torch.tensor(draft_rewards)
        best_draft_idx = draft_scores.argmax().item()
        best_draft = draft_responses[best_draft_idx]

        # Stage 2: Refinement conditioned on best draft
        conditioned_prompt = prompt + "\n<reference_draft>" + best_draft + "</reference_draft>"
        refine_responses = []
        refine_log_probs = []
        orig_temperature = getattr(config, 'temperature', 1.0)
        config.temperature = getattr(config, 'igrpo_refinement_temperature', 0.8)
        model.eval()
        with torch.no_grad():
            for _ in range(refine_size):
                response, log_prob = self._generate_response(
                    model=model, prompt=conditioned_prompt, config=config, tokenizer=tokenizer,
                )
                refine_responses.append(response)
                refine_log_probs.append(log_prob)
        config.temperature = orig_temperature

        # Combine: drafts + refinements
        all_responses = draft_responses + refine_responses
        all_log_probs = torch.stack(draft_log_probs + refine_log_probs)
        all_old_log_probs = all_log_probs.detach().clone()

        # Self-consistency scores: lexical overlap between each response and best draft
        best_draft_words = set(best_draft.lower().split())
        consistency_scores = []
        for resp in all_responses:
            resp_words = set(resp.lower().split())
            if len(best_draft_words) > 0:
                overlap = len(best_draft_words & resp_words) / len(best_draft_words)
            else:
                overlap = 0.0
            consistency_scores.append(overlap)
        consistency_tensor = torch.tensor(consistency_scores)

        return all_responses, all_log_probs, all_old_log_probs, consistency_tensor

    def _igrpo_self_conditioned_advantages(
        self,
        rewards: torch.Tensor,
        consistency_scores: torch.Tensor,
        group_size: int,
        config: POPSSGRPOConfig,
    ) -> torch.Tensor:
        """Compute self-conditioned advantages with draft baseline.

        A_i = (r_i - (1-λ)*mean(r_group) - λ*consistency_i) / std(r_group + ε)
        where λ = igrpo_conditioning_strength
        """
        rewards = rewards.view(-1, group_size)
        rewards = torch.where(torch.isfinite(rewards), rewards, torch.zeros_like(rewards))
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True).clamp(min=config.min_std)

        consistency_bonus = getattr(config, 'igrpo_self_consistent_weight', 0.2) * consistency_scores.view(-1, group_size)
        cond_strength = getattr(config, 'igrpo_conditioning_strength', 0.3)
        baseline = (1.0 - cond_strength) * mean + cond_strength * consistency_bonus
        advantages = (rewards - baseline) / std
        advantages = torch.where(torch.isfinite(advantages), advantages, torch.zeros_like(advantages))
        return advantages.view(-1)

    # ── GraphPO: Graph-Based Trajectory Exploration ─────────────────────

    def _graphpo_expand_trajectories(
        self,
        model: nn.Module,
        prompt: str,
        config: POPSSGRPOConfig,
        tokenizer,
        reward_function,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand a set of trajectories into a decision graph.

        At each decision point (tool call / reasoning fork), branch into
        multiple continuations.  Rewards are propagated back through the
        graph paths, and the top-K paths are returned for training.
        """
        import itertools

        # Generate initial responses (root nodes)
        root_responses = []
        root_log_probs = []
        model.eval()
        with torch.no_grad():
            for _ in range(config.group_size):
                response, log_prob = self._generate_response(
                    model=model, prompt=prompt, config=config, tokenizer=tokenizer,
                )
                root_responses.append(response)
                root_log_probs.append(log_prob)

        # Expand each root into a tree by branching at decision points
        all_paths = []
        for root_resp, root_lp in zip(root_responses, root_log_probs):
            paths = self._graphpo_branch_from(
                model=model, prompt=prompt, response=root_resp,
                log_prob=root_lp, depth=0, config=config, tokenizer=tokenizer,
            )
            all_paths.extend(paths)

        # Score all paths
        final_responses = [p["response"] for p in all_paths]
        rewards = self._compute_rewards(final_responses, prompt, reward_function)

        for i, path in enumerate(all_paths):
            path["reward"] = rewards[i] if i < len(rewards) else 0.0
            depth_penalty = config.graphpo_depth_penalty * path["depth"]
            path["discounted_reward"] = path["reward"] * (config.graphpo_reward_discount ** path["depth"]) - depth_penalty

        # Sort by discounted reward, take top-k
        all_paths.sort(key=lambda p: p["discounted_reward"], reverse=True)
        top_paths = all_paths[:config.graphpo_top_paths]

        # If fewer than group_size paths, pad with root responses
        while len(top_paths) < config.group_size:
            idx = len(top_paths) % len(root_responses)
            top_paths.append({
                "response": root_responses[idx],
                "log_prob": root_log_probs[idx],
                "reward": 0.0,
                "discounted_reward": 0.0,
                "depth": 0,
            })

        graph_responses = [p["response"] for p in top_paths]
        graph_log_probs = torch.stack([p["log_prob"] for p in top_paths])
        graph_old_probs = graph_log_probs.detach().clone()
        graph_rewards = torch.tensor([p["discounted_reward"] for p in top_paths])

        return graph_responses, graph_log_probs, graph_old_probs, graph_rewards

    def _graphpo_branch_from(
        self,
        model: nn.Module,
        prompt: str,
        response: str,
        log_prob: torch.Tensor,
        depth: int,
        config: POPSSGRPOConfig,
        tokenizer,
    ) -> List[dict]:
        """Recursively expand branches at decision points in a response."""
        if depth >= config.graphpo_max_branches:
            return [{"response": response, "log_prob": log_prob, "depth": depth}]

        # Detect decision points: tool calls, reasoning transitions
        decision_points = []
        for keyword in ["```tool", "Therefore", "Alternatively", "In conclusion"]:
            idx = response.find(keyword)
            if idx >= 0:
                decision_points.append(idx)

        if not decision_points:
            return [{"response": response, "log_prob": log_prob, "depth": depth}]

        # Branch at the first decision point
        branch_idx = decision_points[0]
        prefix = response[:branch_idx]
        suffix = response[branch_idx:]

        branches = []
        branch_prompts = [
            prefix + "\n[Branch: First approach] ",
            prefix + "\n[Branch: Alternative approach] ",
            prefix + "\n[Branch: Refine and verify] ",
        ]

        model.eval()
        with torch.no_grad():
            for bp in branch_prompts[:config.graphpo_max_branches]:
                orig_temp = config.temperature
                config.temperature = 0.9 + (depth * 0.05)  # increase temperature with depth
                branch_response, branch_lp = self._generate_response(
                    model=model, prompt=prompt + "\n" + bp, config=config, tokenizer=tokenizer,
                )
                config.temperature = orig_temp

                combined = bp + branch_response
                combined_lp = log_prob + branch_lp.to(log_prob.device)

                sub_branches = self._graphpo_branch_from(
                    model=model, prompt=prompt,
                    response=combined, log_prob=combined_lp,
                    depth=depth + 1, config=config, tokenizer=tokenizer,
                )
                branches.extend(sub_branches)

        return branches

    def _graphpo_compute_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
        config: POPSSGRPOConfig,
    ) -> torch.Tensor:
        """Compute graph-structure-aware advantages."""
        rewards = rewards.view(-1, group_size)
        rewards = torch.where(torch.isfinite(rewards), rewards, torch.zeros_like(rewards))
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True).clamp(min=config.min_std)
        exploration_bonus = getattr(config, 'graphpo_exploration_bonus', 0.1) * (1.0 - (std / (std + 1.0)))
        advantages = (rewards - mean) / std + exploration_bonus
        advantages = torch.where(torch.isfinite(advantages), advantages, torch.zeros_like(advantages))
        return advantages.view(-1)

    # ── CoDaPO: Confidence/Difficulty Adaptive ──────────────────────────

    def _codapo_compute_confidence_scores(
        self,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        log_probs_safe = torch.where(
            torch.isfinite(log_probs), log_probs, torch.full_like(log_probs, -10.0),
        )
        normalised = log_probs_safe / log_probs_safe.abs().clamp(min=1.0).detach()
        confidence = torch.exp(normalised.clamp(min=-10.0, max=0.0))
        cmin = confidence.min()
        cmax = confidence.max()
        if cmax > cmin:
            return (confidence - cmin) / (cmax - cmin)
        return torch.zeros_like(confidence)

    def _codapo_adaptive_clip_ratio(
        self,
        confidence: torch.Tensor,
        config: POPSSGRPOConfig,
    ) -> torch.Tensor:
        """Adjust clip ratio per response based on confidence."""
        low_clip = getattr(config, 'codapo_clip_low_confidence', 0.1)
        high_clip = getattr(config, 'codapo_clip_high_confidence', 0.3)
        adaptive = low_clip + (high_clip - low_clip) * confidence
        return adaptive.clamp(min=1e-4, max=1.0)

    def _codapo_adaptive_kl_coef(
        self,
        confidence: torch.Tensor,
        config: POPSSGRPOConfig,
    ) -> torch.Tensor:
        """Adjust KL coefficient per response based on confidence."""
        low_kl = getattr(config, 'codapo_kl_low_confidence', 0.05)
        high_kl = getattr(config, 'codapo_kl_high_confidence', 0.15)
        adaptive = low_kl + (high_kl - low_kl) * confidence
        return adaptive.clamp(min=1e-4, max=1.0)

    def _codapo_adaptive_update(
        self,
        model: nn.Module,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        config: POPSSGRPOConfig,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Dict[str, List[float]]:
        """CoDaPO policy update with confidence/difficulty adaptive parameters."""
        stats = {
            "policy_losses": [], "kl_divergences": [], "entropies": [],
            "clip_fractions": [], "approx_kl": [], "confidence_scores": [],
        }

        confidence = self._codapo_compute_confidence_scores(log_probs)
        adaptive_clip = self._codapo_adaptive_clip_ratio(confidence, config)
        adaptive_kl = self._codapo_adaptive_kl_coef(confidence, config)

        if optimizer:
            optimizer.zero_grad()

        ratio = torch.exp((log_probs - old_log_probs).clamp(min=-10.0, max=10.0))
        surr1 = ratio * advantages
        clipped_ratio_low = 1.0 - adaptive_clip.view(-1, 1)
        clipped_ratio_high = 1.0 + adaptive_clip.view(-1, 1)
        surr2 = torch.clamp(ratio, clipped_ratio_low, clipped_ratio_high) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        kl_div = (log_probs - ref_log_probs).clamp(min=-10.0, max=10.0).mean()
        adaptive_kl_mean = (adaptive_kl * (log_probs - ref_log_probs).abs().clamp(max=10.0)).mean()
        entropy = -(log_probs.clamp(min=-10.0).mean())

        total_loss = (
            policy_loss +
            config.kl_coef * adaptive_kl_mean -
            config.entropy_coef * entropy
        )

        if optimizer and total_loss.requires_grad:
            scaled_loss = self._maybe_scale_loss(total_loss, config)
            scaled_loss.backward()
            self._maybe_unscale_grads(model, config)
            max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).float().mean()
        approx_kl = (old_log_probs - log_probs).mean().abs()

        stats["policy_losses"].append(policy_loss.item())
        stats["kl_divergences"].append(kl_div.item())
        stats["entropies"].append(entropy.item())
        stats["clip_fractions"].append(clip_fraction.item())
        stats["approx_kl"].append(approx_kl.item())
        stats["confidence_scores"].append(confidence.mean().item())

        return stats

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  GRPO-VPS: Verifiable Process Supervision (Yv Architecture, Dunimd Team)         ║
    # ╚══════════════════════════════════════════════════════════════════╝

    def _split_into_steps(
        self,
        sequence: str,
        delimiter: str = "\n",
        min_steps: int = 1,
    ) -> List[str]:
        """Split a response sequence into reasoning steps.

        Each step is a segment delimited by *delimiter*.  Falling back
        to period-based splitting when the delimiter yields fewer than
        *min_steps* steps.
        """
        raw = [s.strip() for s in sequence.split(delimiter) if s.strip()]
        if len(raw) < min_steps:
            raw = [s.strip() for s in sequence.replace(".", "\n").split("\n") if s.strip()]
        return raw if raw else [sequence]

    def _compute_step_quality_scores(
        self,
        steps: List[str],
        model: Optional[nn.Module] = None,
        tokenizer=None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Heuristic process-quality scores for each step.

        Combines three signals:
          1. Step confidence  – average token log-prob (if model given)
          2. Step coherence   – bigram overlap with adjacent steps
          3. Step position    – slight decay for later steps
        Returns a normalised tensor of shape ``(len(steps),)``.
        """
        n = len(steps)
        scores = torch.ones(n, dtype=torch.float32)

        # 1. Model confidence per step
        if model is not None and tokenizer is not None and device is not None:
            model.eval()
            with torch.no_grad():
                for i, step in enumerate(steps):
                    if not step.strip():
                        continue
                    ids = tokenizer.encode(step, return_tensors="pt").to(device)
                    outputs = model(ids)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    lp = F.log_softmax(logits, dim=-1)
                    token_lp = lp[:, :-1, :].gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                    conf = token_lp.mean().exp().clamp(0.0, 1.0)
                    scores[i] = scores[i] * (0.5 + 0.5 * conf)

        # 2. Coherence: bigram overlap with neighbour
        for i in range(n):
            if i > 0:
                prev_bigrams = set(zip(steps[i - 1].split(), steps[i - 1].split()[1:]))
                cur_bigrams = set(zip(steps[i].split(), steps[i].split()[1:]))
                if prev_bigrams:
                    overlap = len(prev_bigrams & cur_bigrams) / max(len(prev_bigrams), 1)
                    scores[i] = scores[i] * (0.7 + 0.3 * overlap)

        # 3. Position decay (earlier steps slightly preferred)
        pos_weight = torch.linspace(1.0, 0.9, n)
        scores = scores * pos_weight

        # Normalise to [0, 1]
        scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        lo, hi = scores.min(), scores.max()
        if hi > lo:
            scores = (scores - lo) / (hi - lo)
        return scores

    def _compute_vps_rewards(
        self,
        sequences: List[str],
        rewards: torch.Tensor,
        tokenizer=None,
        config: Optional[POPSSGRPOConfig] = None,
        prompt: str = "",
        reward_function=None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """GRPO-VPS: Verifiable Process Supervision.

        Splits every response into reasoning steps and computes a
        per-step process reward.  The final per-sequence reward is a
        convex combination of the original outcome reward and the mean
        step process reward:

            final = outcome * W_outcome + mean(step_rewards) * W_process

        Returns the modified rewards tensor and a list of per-step
        score tensors (one per sequence).
        """
        cfg = config or POPSSGRPOConfig()
        w_out = cfg.vps_outcome_weight
        w_proc = cfg.vps_process_weight
        delimiter = cfg.vps_step_delimiter
        min_steps = cfg.vps_min_steps
        quality_scale = cfg.vps_quality_scale

        device = rewards.device
        modified = []
        all_step_scores = []

        outcomes_pos = rewards.clamp(min=0.0)

        for idx, seq in enumerate(sequences):
            steps = self._split_into_steps(seq, delimiter, min_steps)
            step_scores = self._compute_step_quality_scores(
                steps, tokenizer=tokenizer,
            )
            step_scores = step_scores.to(device)

            step_rewards = step_scores * quality_scale * outcomes_pos[idx]
            mean_step_r = step_rewards.mean()

            combined = w_out * outcomes_pos[idx] + w_proc * mean_step_r
            modified.append(combined)
            all_step_scores.append(step_scores.cpu())

        return torch.stack(modified).to(device), all_step_scores

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  MMR-GRPO: Diversity-aware Multi-Model Refinement (Yv Architecture, Dunimd Team) ║
    # ╚══════════════════════════════════════════════════════════════════╝

    def _mmr_ensure_buffer(self, config: POPSSGRPOConfig):
        """Lazily initialise the diversity buffer."""
        if self._mmr_diversity_buffer is None:
            self._mmr_diversity_buffer = {
                "entries": [],
                "maxlen": config.mmr_buffer_size,
            }

    def _mmr_compute_embedding(
        self,
        response: str,
        tokenizer,
        dim: int = 64,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compute a fixed-size embedding for a response string.

        Uses token-frequency hashing (feature hashing) to produce a
        *dim*-dimensional vector without requiring a forward pass.
        """
        if tokenizer is not None:
            ids = tokenizer.encode(response, add_special_tokens=False)
        else:
            ids = [ord(c) for c in response[:2048]]

        emb = torch.zeros(dim)
        for tid in ids:
            h = hash(str(tid)) % dim
            sign = 1 if (hash(str(tid)) // dim) % 2 == 0 else -1
            emb[h] += sign
        # Normalise
        norm = emb.norm()
        if norm > 1e-8:
            emb = emb / norm
        return emb

    def _mmr_diversity_score(
        self,
        embedding: torch.Tensor,
        buffer_entries: List[TrajectoryEntry],
    ) -> float:
        """Compute the minimum cosine distance to any entry in the buffer."""
        if not buffer_entries:
            return 1.0
        all_embs = torch.stack([e.embedding.to(embedding.device) for e in buffer_entries])
        sims = F.cosine_similarity(embedding.unsqueeze(0), all_embs)
        return (1.0 - sims).min().item()

    def _mmr_update_buffer(
        self,
        trajectories: List[str],
        rewards: torch.Tensor,
        config: POPSSGRPOConfig,
        tokenizer=None,
        device: Optional[torch.device] = None,
    ):
        """Update the diversity buffer with new trajectory entries.

        Each new trajectory is scored for diversity against the existing
        buffer.  Low-diversity trajectories (similarity above threshold)
        are discarded.  The buffer is kept at ``mmr_buffer_size`` by
        evicting the lowest-diversity entries when full.
        """
        self._mmr_ensure_buffer(config)
        buf = self._mmr_diversity_buffer
        threshold = config.mmr_similarity_threshold
        dim = config.mmr_embedding_dim

        for resp, r in zip(trajectories, rewards):
            emb = self._mmr_compute_embedding(resp, tokenizer, dim, device)
            div = self._mmr_diversity_score(emb, buf["entries"])

            # Discard if too similar to existing entries
            if div < (1.0 - threshold) and len(buf["entries"]) > 0:
                continue

            entry = TrajectoryEntry(embedding=emb, reward=float(r), diversity_score=div)
            buf["entries"].append(entry)

        # Prune to maxlen: keep the most diverse entries
        if len(buf["entries"]) > buf["maxlen"]:
            buf["entries"].sort(key=lambda e: e.diversity_score, reverse=True)
            buf["entries"] = buf["entries"][: buf["maxlen"]]

    def _mmr_diversify_rewards(
        self,
        responses: List[str],
        rewards: torch.Tensor,
        config: POPSSGRPOConfig,
        tokenizer=None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Apply diversity bonus to rewards.

        Responses that are more diverse (relative to the buffer) receive
        a positive bonus; responses that are too similar receive a penalty.
        """
        self._mmr_ensure_buffer(config)
        buf = self._mmr_diversity_buffer
        dim = config.mmr_embedding_dim
        div_weight = config.mmr_diversity_weight

        modified = []
        for resp, r in zip(responses, rewards):
            emb = self._mmr_compute_embedding(resp, tokenizer, dim, device)
            div = self._mmr_diversity_score(emb, buf["entries"])
            bonus = div_weight * div
            modified.append(r.item() * (1.0 + bonus))

        return torch.tensor(modified, device=rewards.device)

    def _mmr_mix_from_buffer(
        self,
        responses: List[str],
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        config: POPSSGRPOConfig,
        tokenizer=None,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Mix a fraction of responses from the diversity buffer.

        Replaces a portion (``mmr_mix_ratio``) of the current batch with
        entries sampled from the buffer, prioritising high-diversity and
        high-reward trajectories.
        """
        self._mmr_ensure_buffer(config)
        buf = self._mmr_diversity_buffer
        if not buf["entries"]:
            return responses, log_probs, old_log_probs

        device = log_probs.device
        n_replace = max(1, int(len(responses) * config.mmr_mix_ratio))
        n_replace = min(n_replace, len(buf["entries"]), len(responses))

        # Weighted sampling: diversity * reward
        entries = buf["entries"]
        weights = torch.tensor(
            [e.diversity_score * max(e.reward, 0.01) for e in entries],
            dtype=torch.float32,
        )
        weights = F.softmax(weights, dim=0)
        idxs = torch.multinomial(weights, n_replace, replacement=False).tolist()

        # Replace the last n_replace responses with buffer entries
        mixed_resp = list(responses)
        mixed_lp = log_probs.clone()
        mixed_old = old_log_probs.clone()

        for i, buf_idx in enumerate(idxs):
            target = len(responses) - n_replace + i
            entry = entries[buf_idx]
            # We don't have the original log prob for buffer entries,
            # so we assign a proxy based on the reward.
            proxy_lp = torch.tensor(max(math.log(entry.reward + 1e-6), -10.0), device=device)
            mixed_resp[target] = f"<diverse_trajectory_{buf_idx}>"
            mixed_lp[target] = proxy_lp
            mixed_old[target] = proxy_lp.detach()

        return mixed_resp, mixed_lp, mixed_old

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  TR-GRPO: Token-Level Reward Weighting (Yv Architecture, Dunimd Team)            ║
    # ╚══════════════════════════════════════════════════════════════════╝

    def _compute_token_level_log_probs(
        self,
        model: nn.Module,
        prompt: str,
        responses: List[str],
        tokenizer,
        config: POPSSGRPOConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-token log probabilities for every response.

        Returns
        -------
        token_log_probs : Tensor[batch, max_len]
        token_old_probs : Tensor[batch, max_len]  (detached copy)
        mask           : Tensor[batch, max_len]   (1 = valid token)
        """
        device = next(model.parameters()).device
        max_len = 0
        all_tokens = []
        prompt_lens = []

        model.eval()
        with torch.no_grad():
            for resp in responses:
                full = prompt + resp
                if tokenizer:
                    ids = tokenizer.encode(full, return_tensors="pt").to(device)
                else:
                    ids = torch.tensor(
                        [[ord(c) for c in full]], dtype=torch.long, device=device
                    )

                plen = len(tokenizer.encode(prompt)) if tokenizer else len(prompt)
                prompt_lens.append(plen)

                fp8_context = (
                    te.fp8_autocast(
                        enabled=True,
                        fp8_recipe=DelayedScaling(
                            margin=0,
                            interval=1,
                            fp8_format=Format.HYBRID,
                            amax_history_len=1024,
                            amax_compute_algo="max",
                        ),
                    )
                    if config and config.use_fp8
                    else contextlib.nullcontext()
                )

                with fp8_context:
                    outputs = model(ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                lp = F.log_softmax(logits, dim=-1)
                token_lp = lp[:, :-1, :].gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(0)

                # Response tokens only
                resp_lp = token_lp[plen - 1 :]
                all_tokens.append(resp_lp)
                max_len = max(max_len, resp_lp.shape[0])

        # Pad to max_len
        padded_lp = []
        padded_mask = []
        for lp in all_tokens:
            pad = max_len - lp.shape[0]
            if pad > 0:
                lp = F.pad(lp, (0, pad), value=0.0)
                mask = torch.cat([torch.ones(lp.shape[0] - pad, device=device),
                                  torch.zeros(pad, device=device)])
            else:
                mask = torch.ones(max_len, device=device)
            padded_lp.append(lp)
            padded_mask.append(mask)

        token_lp = torch.stack(padded_lp)
        mask = torch.stack(padded_mask)
        return token_lp, token_lp.detach().clone(), mask

    def _compute_token_advantages(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        config: POPSSGRPOConfig,
    ) -> torch.Tensor:
        """TR-GRPO: Token-level advantage computation.

        Per-token importance is estimated from the absolute change in
        log-probability:  importance = |new_lp - old_lp|.

        Per-token advantages are:

            A_{i,t} = importance_{i,t} * A_i  (scaled and biased)

        where A_i is the group-relative advantage at the sequence level.
        """
        device = old_log_probs.device
        group_size = old_log_probs.shape[0]
        scale = config.tr_importance_scale
        bias = config.tr_importance_bias

        rewards = rewards.to(device)
        reshaped = rewards.view(-1, group_size)
        reshaped = torch.where(torch.isfinite(reshaped), reshaped, torch.zeros_like(reshaped))
        mean = reshaped.mean(dim=-1, keepdim=True)
        std = reshaped.std(dim=-1, keepdim=True).clamp(min=getattr(config, 'min_std', 1e-8))
        seq_advantages = ((reshaped - mean) / std).view(-1)
        seq_advantages = torch.where(torch.isfinite(seq_advantages), seq_advantages, torch.zeros_like(seq_advantages))

        importance = (new_log_probs - old_log_probs).abs().detach()
        importance = torch.where(torch.isfinite(importance), importance, torch.zeros_like(importance))

        imp_max = importance.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        importance_norm = importance / imp_max

        token_adv = importance_norm * seq_advantages.unsqueeze(-1)
        token_adv = token_adv * scale + bias * seq_advantages.unsqueeze(-1).sign()
        token_adv = token_adv * mask

        return token_adv

    def _tr_ppo_update(
        self,
        model: nn.Module,
        token_log_probs: torch.Tensor,
        token_old_log_probs: torch.Tensor,
        token_ref_log_probs: torch.Tensor,
        token_advantages: torch.Tensor,
        mask: torch.Tensor,
        config: POPSSGRPOConfig,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Dict[str, List[float]]:
        """TR-GRPO: Per-token PPO-style policy update.

        Applies the clipped surrogate objective token-wise, then
        aggregates via masked mean.
        """
        stats = {
            "policy_losses": [],
            "kl_divergences": [],
            "entropies": [],
            "clip_fractions": [],
            "approx_kl": [],
        }
        clip = config.tr_token_clip_ratio

        if optimizer:
            optimizer.zero_grad()

        ratio = torch.exp((token_log_probs - token_old_log_probs).clamp(min=-10.0, max=10.0))

        surr1 = ratio * token_advantages
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * token_advantages

        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * mask).sum() / mask.sum()

        kl_div = (token_log_probs - token_ref_log_probs).clamp(min=-10.0, max=10.0)
        kl_div = (kl_div * mask).sum() / mask.sum()

        entropy = -(token_log_probs.clamp(min=-10.0) * mask).sum() / mask.sum()

        total_loss = (
            policy_loss
            + config.kl_coef * kl_div
            - config.entropy_coef * entropy
        )

        if optimizer and total_loss.requires_grad:
            scaled_loss = self._maybe_scale_loss(total_loss, config)
            scaled_loss.backward()
            self._maybe_unscale_grads(model, config)
            max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).float().mean()
        approx_kl = (old_log_probs - log_probs).mean().abs()

        stats["policy_losses"].append(policy_loss.item())
        stats["kl_divergences"].append(kl_div.item())
        stats["entropies"].append(entropy.item())
        stats["clip_fractions"].append(clip_fraction.item())
        stats["approx_kl"].append(approx_kl.item())

        return stats

    def _maybe_scale_loss(self, loss: torch.Tensor, config: POPSSGRPOConfig) -> torch.Tensor:
        if getattr(config, 'use_amp', False):
            scale = getattr(config, 'loss_scale', 128.0)
            return loss * scale
        return loss

    def _maybe_unscale_grads(self, model: nn.Module, config: POPSSGRPOConfig) -> None:
        if getattr(config, 'use_amp', False):
            scale = getattr(config, 'loss_scale', 128.0)
            has_inf = False
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(scale)
                    if not has_inf:
                        has_inf = bool(torch.isinf(p.grad).any() or torch.isnan(p.grad).any())
            if has_inf:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

    def _safe_mean(self, values: List[float]) -> float:
        """Compute mean safely, returning 0.0 for empty lists."""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def _get_time(self) -> float:
        """Get current time in seconds."""
        import time
        return time.time()


# Paper: DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", arXiv:2501.12948
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


class POPSSAgenticRLTrainer:
    """
    Agentic RL Post-Training Trainer.

    Extends GRPO with agent-environment interaction for training LMs as agents.
    Collects multi-step agent trajectories, computes tool/task/efficiency rewards,
    and uses group-relative advantage estimation for policy updates.

    Key Features:
        - Multi-step agent rollouts with tool call environment
        - Agent-specific reward computation (tool success, task completion, efficiency)
        - Group-relative advantage based on trajectory quality
        - Agent-specific PPO-style policy updates

    Example:
        >>> config = POPSSAgenticRLConfig(agent_rollout_steps=8, max_tool_calls=20)
        >>> trainer = POPSSAgenticRLTrainer(
        ...     model=policy_model,
        ...     reference_model=ref_model,
        ...     env_reward_function=reward_fn,
        ...     config=config,
        ... )
        >>> result = trainer.train(prompts=train_prompts, num_epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        env_reward_function=None,
        config: Optional[POPSSAgenticRLConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        tokenizer=None,
        tool_executor=None,
    ):
        self.model = model
        self.reference_model = reference_model
        self.env_reward_function = env_reward_function
        self.config = config or POPSSAgenticRLConfig()
        self.tokenizer = tokenizer
        self.tool_executor = tool_executor

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-5,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer

        self.grpo_operator = POPSSGRPOOperator()
        self.training_history = []

    def _collect_agent_trajectory(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Collect a single agent trajectory with multiple steps.

        The agent interacts with an environment by generating actions (tool calls),
        receiving observations, and accumulating rewards. The trajectory is treated
        as a "response group" for GRPO-style advantage computation.

        When ``self.encre_trainer`` is set, the trajectory is produced by
        the full EnTA rollout loop (multi-step tool use, Rust sandbox,
        reward shaping).  Otherwise the lightweight in-process sampler
        is used.

        Args:
            prompt: Initial prompt for the agent

        Returns:
            Dictionary containing trajectory data
        """
        trajectory = {
            "prompt": prompt,
            "steps": [],
            "tool_calls": 0,
            "tool_successes": 0,
            "task_completed": False,
            "total_reward": 0.0,
            "step_rewards": [],
            "log_probs": [],
            "responses": [],
        }

        current_context = prompt
        total_tool_calls = 0
        task_completed = False

        for step in range(self.config.agent_rollout_steps):
            response, log_prob = self._generate_agent_step(current_context)

            trajectory["responses"].append(response)
            trajectory["log_probs"].append(log_prob)

            tool_call_count = response.count("```tool")
            tool_success = 0
            if tool_call_count > 0 and self.tool_executor is not None:
                for _ in range(tool_call_count):
                    try:
                        self.tool_executor(response)
                        tool_success += 1
                    except Exception:
                        pass
                total_tool_calls += tool_call_count
                trajectory["tool_successes"] += tool_success

            trajectory["tool_calls"] = total_tool_calls

            step_reward = 0.0
            if tool_success > 0:
                step_reward += self.config.tool_call_reward * tool_success
            if total_tool_calls > self.config.max_tool_calls:
                break
            if self.env_reward_function is not None:
                try:
                    env_reward = self.env_reward_function(current_context, response)
                    if env_reward >= self.config.task_completion_reward * 0.9:
                        task_completed = True
                        trajectory["task_completed"] = True
                        step_reward += self.config.task_completion_reward
                    step_reward += env_reward * 0.1
                except Exception:
                    pass

            step_penalty = self.config.efficiency_penalty * max(0, step - 2)
            step_reward += step_penalty
            trajectory["step_rewards"].append(step_reward)

            current_context = current_context + "\n" + response
            if task_completed:
                break

        trajectory["total_reward"] = sum(trajectory["step_rewards"])
        return trajectory

    def _generate_agent_step(
        self,
        context: str,
    ) -> Tuple[str, torch.Tensor]:
        """Generate a single agent step response."""
        device = next(self.model.parameters()).device

        if self.tokenizer:
            input_ids = self.tokenizer.encode(context, return_tensors="pt").to(device)
        else:
            input_ids = torch.tensor([[ord(c) for c in context]], dtype=torch.long, device=device)

        past_key_values = None
        log_probs_sum = torch.tensor(0.0, device=device)
        generated_ids = input_ids

        max_gen_tokens = min(self.config.grpo_config.max_new_tokens, 256)

        for _ in range(max_gen_tokens):
            if generated_ids.shape[1] > 1:
                model_input = generated_ids[:, -1:]
            else:
                model_input = generated_ids

            if hasattr(self.model, 'forward'):
                outputs = self.model(
                    input_ids=model_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs = self.model(generated_ids)

            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None

            next_token_logits = logits[:, -1, :]
            temp = self.config.grpo_config.temperature
            if temp > 0:
                next_token_logits = next_token_logits / temp

            log_probs_for_token = F.log_softmax(next_token_logits, dim=-1)

            if temp > 0:
                probs = torch.exp(log_probs_for_token)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(log_probs_for_token, dim=-1, keepdim=True)

            token_log_prob = log_probs_for_token.gather(1, next_token)
            log_probs_sum = log_probs_sum + token_log_prob.squeeze()

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if self.tokenizer and next_token.item() == self.tokenizer.eos_token_id:
                break

        if self.tokenizer:
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            response = "".join(chr(c) for c in generated_ids[0].tolist())

        return response, log_probs_sum

    def _compute_agent_advantages(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Compute group-relative advantages based on agent trajectory quality.

        Args:
            trajectories: List of agent trajectory dictionaries

        Returns:
            Tensor of advantages, one per trajectory
        """
        rewards = torch.tensor([t["total_reward"] for t in trajectories], dtype=torch.float32)
        group_size = len(trajectories)

        return self.grpo_operator.compute_group_advantages(
            rewards=rewards,
            group_size=group_size,
            normalize=True,
            min_std=1e-8,
        )

    def agentic_rl_update(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Perform agent-specific PPO-style policy update using trajectory data.

        This method implements the core agentic RL update, using trajectory-level
        log probabilities and group-relative advantages computed from agent rewards.

        Args:
            trajectories: List of agent trajectory dictionaries

        Returns:
            Dictionary of training statistics
        """
        advantages = self._compute_agent_advantages(trajectories)

        trajectory_log_probs = []
        for traj in trajectories:
            combined_log_prob = torch.stack(traj["log_probs"]).sum() if traj["log_probs"] else torch.tensor(0.0)
            trajectory_log_probs.append(combined_log_prob)

        log_probs_tensor = torch.stack([lp.to(advantages.device) if lp.device != advantages.device else lp for lp in trajectory_log_probs])
        old_log_probs = log_probs_tensor.detach().clone()

        if self.reference_model is not None and self.config.grpo_config.use_reference_model:
            ref_log_probs_list = []
            self.reference_model.eval()
            with torch.no_grad():
                for traj in trajectories:
                    full_text = traj["prompt"] + " ".join(traj["responses"])
                    device = next(self.reference_model.parameters()).device
                    if self.tokenizer:
                        input_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(device)
                    else:
                        input_ids = torch.tensor([[ord(c) for c in full_text]], dtype=torch.long, device=device)
                    outputs = self.reference_model(input_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    log_probs = F.log_softmax(logits, dim=-1)
                    token_log_probs = log_probs[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                    ref_log_probs_list.append(token_log_probs.sum())
            ref_log_probs = torch.stack(ref_log_probs_list)
        else:
            ref_log_probs = torch.zeros_like(log_probs_tensor)

        grpo_config = self.config.grpo_config
        ratio = torch.exp((log_probs_tensor - old_log_probs).clamp(min=-10.0, max=10.0))

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - grpo_config.clip_ratio, 1.0 + grpo_config.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        kl_div = (log_probs_tensor - ref_log_probs).clamp(min=-10.0, max=10.0).mean()
        entropy = -(log_probs_tensor.clamp(min=-10.0).mean())

        total_loss = (
            policy_loss
            + grpo_config.kl_coef * kl_div
            - grpo_config.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        if total_loss.requires_grad:
            scaled_loss = self._maybe_scale_loss(total_loss, grpo_config)
            scaled_loss.backward()
            self._maybe_unscale_grads(self.model, grpo_config)
            max_grad_norm = getattr(grpo_config, 'max_grad_norm', 1.0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()

        clip_fraction = ((ratio - 1.0).abs() > grpo_config.clip_ratio).float().mean()
        approx_kl = (old_log_probs - log_probs_tensor).mean().abs()

        return {
            "policy_loss": policy_loss.item(),
            "kl_divergence": kl_div.item(),
            "entropy": entropy.item(),
            "mean_advantage": advantages.mean().item(),
            "clip_fraction": clip_fraction.item(),
            "approx_kl": approx_kl.item(),
            "total_loss": total_loss.item(),
        }

    def train(
        self,
        prompts: List[str],
        num_epochs: int = 1,
        num_trajectories_per_prompt: int = 4,
        save_dir: Optional[str] = None,
        save_every: int = 100,
    ) -> Dict[str, Any]:
        """
        Train the model with Agentic RL.

        Args:
            prompts: List of training prompts
            num_epochs: Number of training epochs
            num_trajectories_per_prompt: Number of trajectories per prompt (group size)
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N steps

        Returns:
            Training statistics dictionary
        """
        all_stats = {
            "policy_losses": [],
            "kl_divergences": [],
            "rewards": [],
            "task_completion_rate": [],
        }

        step = 0
        for epoch in range(num_epochs):
            for prompt in prompts:
                trajectories = []
                for _ in range(num_trajectories_per_prompt):
                    traj = self._collect_agent_trajectory(prompt)
                    trajectories.append(traj)

                stats = self.agentic_rl_update(trajectories)

                all_stats["policy_losses"].append(stats["policy_loss"])
                all_stats["kl_divergences"].append(stats["kl_divergence"])

                mean_reward = sum(t["total_reward"] for t in trajectories) / max(len(trajectories), 1)
                all_stats["rewards"].append(mean_reward)

                completion_rate = sum(1 for t in trajectories if t["task_completed"]) / max(len(trajectories), 1)
                all_stats["task_completion_rate"].append(completion_rate)

                step += 1

                if save_dir and step % save_every == 0:
                    self._save_checkpoint(save_dir, step)

        self.training_history.append(all_stats)

        return {
            "mean_policy_loss": sum(all_stats["policy_losses"]) / len(all_stats["policy_losses"]) if all_stats["policy_losses"] else 0,
            "mean_kl": sum(all_stats["kl_divergences"]) / len(all_stats["kl_divergences"]) if all_stats["kl_divergences"] else 0,
            "mean_reward": sum(all_stats["rewards"]) / len(all_stats["rewards"]) if all_stats["rewards"] else 0,
            "task_completion_rate": sum(all_stats["task_completion_rate"]) / len(all_stats["task_completion_rate"]) if all_stats["task_completion_rate"] else 0,
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

        path = os.path.join(save_dir, f"agentic_checkpoint_{step}.pt")
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["step"]


