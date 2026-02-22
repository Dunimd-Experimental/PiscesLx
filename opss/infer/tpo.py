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
TPO (Test-Time Preference Optimization) Implementation

Complete implementation of TPO for inference-time preference alignment.
TPO aligns model outputs with preferences during inference without modifying model weights.

Key Innovation:
    - No training required: Alignment happens at inference time
    - Iterative refinement: Multiple rounds of generation and feedback
    - Textual feedback: Uses language feedback instead of scalar rewards
    - Prompt optimization: Refines prompts rather than model weights

Reference:
    "Test-Time Preference Optimization: On-the-fly Alignment via Iterative Textual Feedback"
    (arXiv:2501.12895)

Algorithm:
    1. Generate initial response from model
    2. Get textual feedback based on preference
    3. Refine prompt with feedback
    4. Generate improved response
    5. Repeat until satisfaction or max iterations

Benefits:
    - Zero training cost
    - Immediate adaptation to new preferences
    - No model modification
    - Flexible preference specification
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import re

from configs.version import VERSION
from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig,
)


class POPSSTPOFeedbackType(Enum):
    """Types of feedback for TPO."""
    TEXTUAL = "textual"
    NUMERIC = "numeric"
    HYBRID = "hybrid"


@dataclass
class POPSSTPOConfig(PiscesLxOperatorConfig):
    """
    TPO (Test-Time Preference Optimization) Configuration.
    
    This configuration controls the TPO inference-time alignment parameters.
    
    Attributes:
        max_iterations: Maximum number of refinement iterations
        temperature: Sampling temperature for generation
        reward_threshold: Reward threshold for early stopping
        feedback_type: Type of feedback (textual/numeric/hybrid)
        improvement_prompt: Template for improvement prompts
        use_best_of_n: Whether to use best-of-N selection
        n_candidates: Number of candidates for best-of-N
        early_stopping: Whether to stop early when threshold is met
        verbose: Whether to print intermediate results
        max_new_tokens: Maximum tokens to generate per response
        top_p: Nucleus sampling probability
        top_k: Top-k sampling parameter
        feedback_model: Model for generating feedback (optional)
        reward_model: Model for computing rewards (optional)
    """
    name: str = "tpo"
    version: str = VERSION
    
    max_iterations: int = 3
    temperature: float = 0.7
    reward_threshold: float = 0.8
    feedback_type: str = "textual"
    use_best_of_n: bool = False
    n_candidates: int = 4
    early_stopping: bool = True
    verbose: bool = False
    max_new_tokens: int = 512
    top_p: float = 0.95
    top_k: int = 50
    
    improvement_prompt: str = (
        "The following response needs improvement based on the preference: {preference}\n\n"
        "Original response: {response}\n\n"
        "Feedback: {feedback}\n\n"
        "Please provide an improved response that better aligns with the preference:"
    )
    
    feedback_prompt: str = (
        "Evaluate the following response based on the given preference.\n\n"
        "Preference: {preference}\n\n"
        "Response: {response}\n\n"
        "Provide specific feedback on how to improve the response to better match the preference. "
        "Focus on concrete, actionable suggestions."
    )
    
    reward_prompt: str = (
        "Rate how well the following response matches the given preference on a scale of 0 to 1.\n\n"
        "Preference: {preference}\n\n"
        "Response: {response}\n\n"
        "Output only a single number between 0 and 1:"
    )
    
    def __post_init__(self):
        super().__post_init__()
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")


class POPSSTPOOperator(PiscesLxOperatorInterface):
    """
    Test-Time Preference Optimization (TPO) Operator.
    
    TPO provides inference-time preference alignment without modifying model weights.
    It iteratively refines responses based on textual feedback.
    
    Key Features:
        - Zero training cost
        - Iterative refinement
        - Textual feedback
        - Best-of-N selection
        - Early stopping
    
    Example:
        >>> config = POPSSTPOConfig(max_iterations=3, reward_threshold=0.8)
        >>> tpo = POPSSTPOOperator()
        >>> aligned_response = tpo.align_at_inference(
        ...     model=llm,
        ...     prompt="Write a Python function to sort a list",
        ...     preference="Code should be efficient and well-documented",
        ... )
    """
    
    def __init__(self):
        super().__init__()
        self._name = "tpo"
        self._version = VERSION
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return "Test-Time Preference Optimization - Inference-time alignment"
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute TPO alignment.
        
        Args:
            inputs: Dictionary containing:
                - model: Language model
                - prompt: Input prompt
                - preference: Preference specification
                - config: TPO configuration
                - tokenizer: Tokenizer (optional)
                - feedback_model: Model for feedback (optional)
                - reward_model: Model for rewards (optional)
        
        Returns:
            PiscesLxOperatorResult with aligned response
        """
        start_time = self._get_time()
        
        try:
            model = inputs.get("model")
            prompt = inputs.get("prompt", "")
            preference = inputs.get("preference", "")
            config = inputs.get("config", POPSSTPOConfig())
            tokenizer = inputs.get("tokenizer")
            feedback_model = inputs.get("feedback_model")
            reward_model = inputs.get("reward_model")
            
            if model is None:
                raise ValueError("Model is required for TPO alignment")
            
            aligned_response, stats = self.align_at_inference(
                model=model,
                prompt=prompt,
                preference=preference,
                config=config,
                tokenizer=tokenizer,
                feedback_model=feedback_model or model,
                reward_model=reward_model or model,
            )
            
            execution_time = self._get_time() - start_time
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "aligned_response": aligned_response,
                    "iterations": stats["iterations"],
                    "final_reward": stats["final_reward"],
                    "reward_history": stats["reward_history"],
                },
                execution_time=execution_time,
                metadata={
                    "version": self.version,
                    "algorithm": "TPO",
                    "max_iterations": config.max_iterations,
                    "early_stopped": stats["early_stopped"],
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
    
    def align_at_inference(
        self,
        model: nn.Module,
        prompt: str,
        preference: str,
        config: Optional[POPSSTPOConfig] = None,
        tokenizer = None,
        feedback_model: Optional[nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Align model output with preference at inference time.
        
        Args:
            model: Language model
            prompt: Input prompt
            preference: Preference specification
            config: TPO configuration
            tokenizer: Tokenizer
            feedback_model: Model for generating feedback
            reward_model: Model for computing rewards
        
        Returns:
            Tuple of (aligned_response, statistics)
        """
        if config is None:
            config = POPSSTPOConfig()
        
        feedback_model = feedback_model or model
        reward_model = reward_model or model
        
        stats = {
            "iterations": 0,
            "final_reward": 0.0,
            "reward_history": [],
            "early_stopped": False,
            "responses": [],
        }
        
        if config.use_best_of_n:
            return self._best_of_n_align(
                model=model,
                prompt=prompt,
                preference=preference,
                config=config,
                tokenizer=tokenizer,
                reward_model=reward_model,
            )
        
        current_prompt = prompt
        best_response = None
        best_reward = -float('inf')
        
        for iteration in range(config.max_iterations):
            stats["iterations"] = iteration + 1
            
            response = self.generate_response(
                model=model,
                prompt=current_prompt,
                config=config,
                tokenizer=tokenizer,
            )
            stats["responses"].append(response)
            
            reward = self.compute_reward(
                model=reward_model,
                response=response,
                preference=preference,
                config=config,
                tokenizer=tokenizer,
            )
            stats["reward_history"].append(reward)
            
            if reward > best_reward:
                best_reward = reward
                best_response = response
            
            if config.early_stopping and reward >= config.reward_threshold:
                stats["early_stopped"] = True
                break
            
            if iteration < config.max_iterations - 1:
                feedback = self.get_textual_feedback(
                    model=feedback_model,
                    response=response,
                    preference=preference,
                    config=config,
                    tokenizer=tokenizer,
                )
                
                current_prompt = self.refine_prompt(
                    original_prompt=prompt,
                    response=response,
                    feedback=feedback,
                    preference=preference,
                    config=config,
                )
        
        stats["final_reward"] = best_reward
        
        return best_response or response, stats
    
    def _best_of_n_align(
        self,
        model: nn.Module,
        prompt: str,
        preference: str,
        config: POPSSTPOConfig,
        tokenizer,
        reward_model: nn.Module,
    ) -> Tuple[str, Dict[str, Any]]:
        """Best-of-N selection for alignment."""
        candidates = []
        rewards = []
        
        for _ in range(config.n_candidates):
            response = self.generate_response(
                model=model,
                prompt=prompt,
                config=config,
                tokenizer=tokenizer,
            )
            reward = self.compute_reward(
                model=reward_model,
                response=response,
                preference=preference,
                config=config,
                tokenizer=tokenizer,
            )
            candidates.append(response)
            rewards.append(reward)
        
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
        
        stats = {
            "iterations": 1,
            "final_reward": rewards[best_idx],
            "reward_history": rewards,
            "early_stopped": False,
            "responses": candidates,
        }
        
        return candidates[best_idx], stats
    
    def generate_response(
        self,
        model: nn.Module,
        prompt: str,
        config: POPSSTPOConfig,
        tokenizer,
    ) -> str:
        """
        Generate response from model.
        
        Args:
            model: Language model
            prompt: Input prompt
            config: TPO configuration
            tokenizer: Tokenizer
        
        Returns:
            Generated response string
        """
        model.eval()
        device = next(model.parameters()).device
        
        if tokenizer:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        else:
            input_ids = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            if hasattr(model, 'generate'):
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature if config.temperature > 0 else 1.0,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.temperature > 0,
                )
            else:
                outputs = self._manual_generate(
                    model=model,
                    input_ids=input_ids,
                    config=config,
                )
        
        if tokenizer:
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
        else:
            response = "".join(chr(c) for c in outputs[0].tolist()[len(prompt):])
        
        return response
    
    def _manual_generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        config: POPSSTPOConfig,
    ) -> torch.Tensor:
        """Manual generation for models without generate method."""
        generated = input_ids.clone()
        
        for _ in range(config.max_new_tokens):
            outputs = model(generated)
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
            
            generated = torch.cat([generated, next_token], dim=-1)
        
        return generated
    
    def get_textual_feedback(
        self,
        model: nn.Module,
        response: str,
        preference: str,
        config: POPSSTPOConfig,
        tokenizer,
    ) -> str:
        """
        Get textual feedback on response.
        
        Args:
            model: Language model
            response: Model response
            preference: Preference specification
            config: TPO configuration
            tokenizer: Tokenizer
        
        Returns:
            Textual feedback string
        """
        feedback_prompt = config.feedback_prompt.format(
            preference=preference,
            response=response,
        )
        
        feedback = self.generate_response(
            model=model,
            prompt=feedback_prompt,
            config=config,
            tokenizer=tokenizer,
        )
        
        return feedback
    
    def compute_reward(
        self,
        model: nn.Module,
        response: str,
        preference: str,
        config: POPSSTPOConfig,
        tokenizer,
    ) -> float:
        """
        Compute reward score for response.
        
        Args:
            model: Language model
            response: Model response
            preference: Preference specification
            config: TPO configuration
            tokenizer: Tokenizer
        
        Returns:
            Reward score between 0 and 1
        """
        reward_prompt = config.reward_prompt.format(
            preference=preference,
            response=response,
        )
        
        reward_response = self.generate_response(
            model=model,
            prompt=reward_prompt,
            config=config,
            tokenizer=tokenizer,
        )
        
        try:
            numbers = re.findall(r'[0-9]*\.?[0-9]+', reward_response)
            if numbers:
                reward = float(numbers[0])
                return max(0.0, min(1.0, reward))
        except ValueError:
            pass
        
        positive_indicators = ['good', 'great', 'excellent', 'perfect', 'well', 'correct', 'yes']
        negative_indicators = ['bad', 'poor', 'wrong', 'incorrect', 'no', 'missing', 'lacks']
        
        response_lower = reward_response.lower()
        positive_count = sum(1 for word in positive_indicators if word in response_lower)
        negative_count = sum(1 for word in negative_indicators if word in response_lower)
        
        if positive_count + negative_count > 0:
            return positive_count / (positive_count + negative_count)
        
        return 0.5
    
    def refine_prompt(
        self,
        original_prompt: str,
        response: str,
        feedback: str,
        preference: str,
        config: POPSSTPOConfig,
    ) -> str:
        """
        Refine prompt based on feedback.
        
        Args:
            original_prompt: Original input prompt
            response: Previous model response
            feedback: Textual feedback
            preference: Preference specification
            config: TPO configuration
        
        Returns:
            Refined prompt string
        """
        improvement_prompt = config.improvement_prompt.format(
            preference=preference,
            response=response,
            feedback=feedback,
        )
        
        refined_prompt = f"{original_prompt}\n\n{improvement_prompt}"
        
        return refined_prompt


class POPSSTPOAligner:
    """
    High-level TPO Aligner for easy preference alignment.
    
    Example:
        >>> aligner = POPSSTPOAligner(model, tokenizer)
        >>> aligned = aligner.align(
        ...     prompt="Write a sorting function",
        ...     preference="Code should be efficient and documented",
        ... )
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer = None,
        config: Optional[POPSSTPOConfig] = None,
        feedback_model: Optional[nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or POPSSTPOConfig()
        self.feedback_model = feedback_model or model
        self.reward_model = reward_model or model
        self.operator = POPSSTPOOperator()
    
    def align(
        self,
        prompt: str,
        preference: str,
        **kwargs,
    ) -> str:
        """
        Align model output with preference.
        
        Args:
            prompt: Input prompt
            preference: Preference specification
            **kwargs: Additional configuration overrides
        
        Returns:
            Aligned response string
        """
        config = self._merge_config(kwargs)
        
        result = self.operator.execute({
            "model": self.model,
            "prompt": prompt,
            "preference": preference,
            "config": config,
            "tokenizer": self.tokenizer,
            "feedback_model": self.feedback_model,
            "reward_model": self.reward_model,
        })
        
        if result.status == PiscesLxOperatorStatus.SUCCESS:
            return result.output["aligned_response"]
        else:
            raise RuntimeError(f"TPO alignment failed: {result.error}")
    
    def align_batch(
        self,
        prompts: List[str],
        preference: str,
        **kwargs,
    ) -> List[str]:
        """
        Align multiple prompts with the same preference.
        
        Args:
            prompts: List of input prompts
            preference: Preference specification
            **kwargs: Additional configuration overrides
        
        Returns:
            List of aligned responses
        """
        aligned_responses = []
        
        for prompt in prompts:
            aligned = self.align(prompt, preference, **kwargs)
            aligned_responses.append(aligned)
        
        return aligned_responses
    
    def _merge_config(self, kwargs: Dict[str, Any]) -> POPSSTPOConfig:
        """Merge kwargs with default config."""
        config_dict = {
            "name": self.config.name,
            "version": self.config.version,
            "max_iterations": kwargs.get("max_iterations", self.config.max_iterations),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "reward_threshold": kwargs.get("reward_threshold", self.config.reward_threshold),
            "feedback_type": kwargs.get("feedback_type", self.config.feedback_type),
            "use_best_of_n": kwargs.get("use_best_of_n", self.config.use_best_of_n),
            "n_candidates": kwargs.get("n_candidates", self.config.n_candidates),
            "early_stopping": kwargs.get("early_stopping", self.config.early_stopping),
            "verbose": kwargs.get("verbose", self.config.verbose),
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "improvement_prompt": kwargs.get("improvement_prompt", self.config.improvement_prompt),
            "feedback_prompt": kwargs.get("feedback_prompt", self.config.feedback_prompt),
            "reward_prompt": kwargs.get("reward_prompt", self.config.reward_prompt),
        }
        
        return POPSSTPOConfig(**config_dict)
    
    def set_preference_templates(
        self,
        improvement_prompt: Optional[str] = None,
        feedback_prompt: Optional[str] = None,
        reward_prompt: Optional[str] = None,
    ):
        """Set custom preference templates."""
        if improvement_prompt:
            self.config.improvement_prompt = improvement_prompt
        if feedback_prompt:
            self.config.feedback_prompt = feedback_prompt
        if reward_prompt:
            self.config.reward_prompt = reward_prompt


class POPSSTPOPreferenceLibrary:
    """
    Library of common preferences for TPO alignment.
    """
    
    CODE_QUALITY = "Code should be efficient, well-documented, and follow best practices"
    
    CODE_SECURITY = "Code should be secure, handle edge cases, and not have vulnerabilities"
    
    CODE_READABLE = "Code should be readable, with clear variable names and structure"
    
    RESPONSE_CONCISE = "Response should be concise and to the point"
    
    RESPONSE_DETAILED = "Response should be detailed and comprehensive"
    
    RESPONSE_FRIENDLY = "Response should be friendly and helpful"
    
    RESPONSE_PROFESSIONAL = "Response should be professional and formal"
    
    RESPONSE_ACCURATE = "Response should be factually accurate and precise"
    
    RESPONSE_SAFE = "Response should be safe and not contain harmful content"
    
    MATH_STEP_BY_STEP = "Solution should show step-by-step reasoning"
    
    @staticmethod
    def custom(description: str) -> str:
        """Create a custom preference."""
        return description
    
    @staticmethod
    def combine(*preferences: str) -> str:
        """Combine multiple preferences."""
        return " and ".join(preferences)
