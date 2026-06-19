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

import contextlib

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

# Optional: when the EnTA training pipeline is enabled, the agentic RL
# trainer uses a :class:`YvEncreTrainer` (when provided via
# ``inputs['encre_trainer']``) as the rollout engine -- replacing the
# lightweight in-process sampler with the full EnCRE multi-step agent
# loop (tool use, sandbox enforcement, reward shaping).  The downstream
# GRPO update is unchanged.
try:
    from model.agentic.enta import YvEncreTrainer  # noqa: F401
    _ENTA_AVAILABLE = True
except Exception:  # noqa: BLE001
    YvEncreTrainer = None  # type: ignore[assignment]
    _ENTA_AVAILABLE = False


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

    # EnTA integration: when True, the agentic RL trainer routes every
    # rollout through a :class:`YvEncreTrainer` instance (passed via
    # ``POPSSAgenticRLTrainer(encre_trainer=...)``) instead of the
    # lightweight in-process sampler.  This brings the full EnCRE tool
    # palette (bash, file_*, grep, glob, web_*, ...) and the Rust
    # sandbox into the GRPO loop.
    use_encre_rollout: bool = False
    encre_use_roundtable: bool = False
    encre_system_prompt: str = ""

    grpo_config: POPSSGRPOConfig = field(default_factory=POPSSGRPOConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.agent_rollout_steps < 1:
            raise ValueError("agent_rollout_steps must be at least 1")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")


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
            "verification_rewards", "refinement_iterations",
        ]}

        responses, log_probs, old_log_probs = self._sample_group_responses(
            model=model,
            prompt=prompt,
            group_size=config.group_size,
            config=config,
            tokenizer=tokenizer,
        )

        refined_responses, verification_rewards = self._apply_iterative_refinement(
            model=model,
            prompt=prompt,
            responses=responses,
            config=config,
            tokenizer=tokenizer,
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
                responses=responses,
                prompt=prompt,
                reward_function=reward_function,
            )

        rewards_tensor = torch.tensor(combined_rewards, dtype=torch.float32)

        advantages = self.compute_group_advantages(
            rewards=rewards_tensor,
            group_size=config.group_size,
            normalize=config.advantage_normalization,
            min_std=config.min_std,
        )

        if reference_model and config.use_reference_model:
            ref_log_probs = self._compute_reference_log_probs(
                reference_model=reference_model,
                prompt=prompt,
                responses=final_responses,
                tokenizer=tokenizer,
                config=config,
            )
        else:
            ref_log_probs = torch.zeros_like(log_probs)

        if config.enable_self_verification:
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
            log_probs = refined_log_probs_tensor

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

        stats["rewards"].extend(combined_rewards)
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
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

            clip_fraction = ((torch.exp(log_probs - old_log_probs) - 1.0).abs() > config.clip_ratio).float().mean()
            approx_kl = (old_log_probs - log_probs).mean().abs()

            stats["policy_losses"].append(metrics["policy_loss"])
            stats["kl_divergences"].append(metrics["kl_div"])
            stats["entropies"].append(entropy.item())
            stats["clip_fractions"].append(clip_fraction.item())
            stats["approx_kl"].append(approx_kl.item())

            return stats

        # Standard PPO update
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
        encre_trainer: Optional[Any] = None,
    ):
        self.model = model
        self.reference_model = reference_model
        self.env_reward_function = env_reward_function
        self.config = config or POPSSAgenticRLConfig()
        self.tokenizer = tokenizer
        self.tool_executor = tool_executor
        # EnTA integration: when ``config.use_encre_rollout`` is True and
        # an ``encre_trainer`` is supplied, every agent trajectory is
        # produced by the EnCRE rollout loop (full tool palette, Rust
        # sandbox, reward shaping) instead of the in-process sampler.
        # Fall back to the original behaviour when the trainer is missing
        # or the integration is disabled.
        self.encre_trainer = encre_trainer if (
            self.config.use_encre_rollout and encre_trainer is not None
        ) else None

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
        the full EnCRE rollout loop (multi-step tool use, Rust sandbox,
        reward shaping).  Otherwise the lightweight in-process sampler
        is used.

        Args:
            prompt: Initial prompt for the agent

        Returns:
            Dictionary containing trajectory data
        """
        # ── EnTA path ───────────────────────────────────────────
        if self.encre_trainer is not None:
            return self._collect_encre_trajectory(prompt)

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

            probs = F.softmax(next_token_logits, dim=-1)

            if temp > 0:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            token_log_prob = torch.log(probs.gather(1, next_token) + 1e-10)
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
        ratio = torch.exp(log_probs_tensor - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - grpo_config.clip_ratio, 1.0 + grpo_config.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        kl_div = (log_probs_tensor - ref_log_probs).mean()
        entropy = -log_probs_tensor.mean()

        total_loss = (
            policy_loss
            + grpo_config.kl_coef * kl_div
            - grpo_config.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grpo_config.max_grad_norm)
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

    # ── EnTA integration helpers ────────────────────────────────

    def _collect_encre_trajectory(self, prompt: str) -> Dict[str, Any]:
        """Collect a single agent trajectory via the EnCRE rollout loop.

        This is the EnTA integration point.  Instead of the lightweight
        in-process sampler, the trajectory is produced by
        :meth:`YvEncreTrainer.rollout` (or, when
        ``self.config.encre_use_roundtable`` is True, by
        :meth:`YvEncreTrainer.run_with_roundtable`).

        The returned trajectory has the same shape as the original
        in-process trajectory so that the downstream GRPO update does
        not need to be aware of the integration.
        """
        trajectory: Dict[str, Any] = {
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

        if self.encre_trainer is None:
            return trajectory

        try:
            if self.config.encre_use_roundtable:
                result = self.encre_trainer.run_with_roundtable(
                    [(prompt, "")],
                    optimizer=None,
                    system=self.config.encre_system_prompt or None,
                )
            else:
                result = self.encre_trainer.run_adversarial_batch(
                    [(prompt, "")], optimizer=None
                )
        except Exception as exc:  # noqa: BLE001
            # Surface the failure on the trajectory but keep the GRPO
            # loop alive -- a failed rollout is a reward-zero sample.
            trajectory["total_reward"] = 0.0
            trajectory["step_rewards"] = [0.0]
            trajectory["log_probs"] = [
                torch.tensor(0.0, device=next(self.model.parameters()).device)
            ]
            trajectory["responses"] = [f"<<encre_error: {exc}>>"]
            return trajectory

        # Normalise the EnCRE trainer output to the trajectory shape
        # expected by ``agentic_rl_update``.  The trainer's contract is
        # ``{"items": [...], "trajectories": [...], "loss": ...}`` where
        # each trajectory carries ``final_text`` (the model output) and
        # ``total_reward``.  We collapse both shapes into (response,
        # step_reward) pairs.
        items = (
            result.get("items", []) if isinstance(result, dict) else []
        )
        responses: list[str] = []
        step_rewards: list[float] = []
        for item in items:
            responses.append(str(item.get("response", item.get("reference", ""))))
            step_rewards.append(float(item.get("reward", 0.0)))

        trajectories = (
            result.get("trajectories", []) if isinstance(result, dict) else []
        )
        for t in trajectories:
            text = str(t.get("final_text", "")) or str(t.get("reference", ""))
            if not text:
                continue
            responses.append(text)
            step_rewards.append(float(t.get("total_reward", 0.0)))

        if not responses:
            responses = [str(prompt)]
            step_rewards = [0.0]

        tool_calls = sum(str(r).count("```tool") for r in responses)
        task_completed = any(
            float(s) >= self.config.task_completion_reward * 0.9
            for s in step_rewards
        )

        device = next(self.model.parameters()).device
        log_probs: list[torch.Tensor] = []
        for resp in responses:
            log_probs.append(
                torch.tensor(0.0, device=device)
            )  # EnCRE consumes its own log-prob stream; placeholder.

        trajectory.update(
            {
                "steps": list(zip(responses, step_rewards)),
                "tool_calls": tool_calls,
                "tool_successes": sum(1 for s in step_rewards if s > 0),
                "task_completed": task_completed,
                "total_reward": float(sum(step_rewards)),
                "step_rewards": step_rewards,
                "log_probs": log_probs,
                "responses": responses,
            }
        )
        return trajectory
