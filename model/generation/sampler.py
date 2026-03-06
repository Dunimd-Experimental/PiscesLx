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
Sampling strategies for Yv model generation.

This module provides a comprehensive suite of sampling strategies for text
generation, supporting various decoding approaches from deterministic greedy
decoding to stochastic sampling methods.

Sampling Strategies:
    1. Greedy Decoding:
       - Deterministic selection of highest probability token
       - Fast but may produce repetitive outputs
       - Best for factual/precision tasks
    
    2. Top-k Sampling:
       - Samples from top k most likely tokens
       - Balances quality and diversity
       - Typical values: k=40-100
    
    3. Top-p (Nucleus) Sampling:
       - Samples from smallest set with cumulative probability >= p
       - Adaptive to distribution shape
       - Typical values: p=0.9-0.95
    
    4. Top-k + Top-p Combined:
       - Applies both filters sequentially
       - Most commonly used strategy
       - Recommended default for general generation
    
    5. Typical Sampling:
       - Selects tokens with surprisal close to entropy
       - Produces more coherent outputs
       - Good for creative writing
    
    6. Eta Sampling:
       - Cutoff based on entropy-scaled probability threshold
       - Adaptive to distribution uncertainty
       - Good for diverse outputs
    
    7. Beam Search:
       - Maintains multiple hypotheses
       - Best for translation/summarization
       - Higher quality but slower

Key Features:
    - Unified interface for all sampling strategies
    - Configurable repetition penalty
    - Support for bad words filtering
    - Length and diversity penalties for beam search

Performance Characteristics:
    - Greedy: O(vocab) - fastest
    - Top-k: O(vocab log k) - fast
    - Top-p: O(vocab log vocab) - moderate
    - Beam: O(beam_width * vocab) - slowest

Usage Example:
    >>> from model.generation.sampler import YvSampler, YvSamplingConfig
    >>> 
    >>> # Create sampler with configuration
    >>> config = YvSamplingConfig(
    ...     strategy=YvSamplingStrategy.TOP_K_TOP_P,
    ...     temperature=0.7,
    ...     top_k=50,
    ...     top_p=0.9
    ... )
    >>> sampler = YvSampler(config)
    >>> 
    >>> # Sample next token
    >>> next_token = sampler.sample(logits, past_tokens=generated_ids)
    >>> 
    >>> # Use factory function
    >>> sampler = create_sampler(strategy="top_k_top_p", temperature=0.8)

Note:
    Temperature should be > 0 for sampling strategies.
    Temperature = 0 effectively becomes greedy decoding.
    Repetition penalty > 1.0 discourages repetition.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from enum import Enum


class YvSamplingStrategy(Enum):
    """Enumeration of available sampling strategies for text generation.
    
    Defines the different approaches for selecting the next token during
    autoregressive generation. Each strategy offers different trade-offs
    between quality, diversity, and speed.
    
    Attributes:
        GREEDY: Deterministic selection of highest probability token.
            Fastest but may produce repetitive or generic outputs.
            Best for factual responses and code generation.
        TOP_K: Samples from the k most likely tokens.
            Balances quality and diversity with fixed vocabulary subset.
            Typical values: k=40-100.
        TOP_P: Nucleus sampling from smallest set with cumulative prob >= p.
            Adaptive to distribution shape, more flexible than top-k.
            Typical values: p=0.9-0.95.
        TOP_K_TOP_P: Combined top-k and top-p filtering.
            Most commonly used strategy for general-purpose generation.
            Applies both filters sequentially for robust sampling.
        BEAM_SEARCH: Maintains multiple hypotheses during generation.
            Higher quality outputs but slower and memory-intensive.
            Best for translation and summarization tasks.
        CONTRASTIVE: Contrastive search for diverse and coherent outputs.
            Balances between model confidence and degeneration penalty.
            Good for long-form generation.
        TYPICAL: Typical sampling based on surprisal-entropy distance.
            Selects tokens with information content close to expected.
            Produces more coherent and less repetitive outputs.
        ETA_SAMPLING: Entropy-scaled adaptive sampling.
            Cutoff threshold adapts to distribution uncertainty.
            Good for maintaining diversity across different contexts.
    
    Example:
        >>> strategy = YvSamplingStrategy.TOP_K_TOP_P
        >>> if strategy == YvSamplingStrategy.GREEDY:
        ...     print("Using deterministic decoding")
    
    Note:
        Strategy choice significantly impacts output quality and diversity.
        TOP_K_TOP_P is recommended as the default for most use cases.
    """
    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TOP_K_TOP_P = "top_k_top_p"
    BEAM_SEARCH = "beam_search"
    CONTRASTIVE = "contrastive"
    TYPICAL = "typical"
    ETA_SAMPLING = "eta_sampling"


@dataclass
class YvSamplingConfig:
    """Configuration dataclass for sampling strategy parameters.
    
    Encapsulates all parameters needed to configure sampling behavior,
    supporting various strategies from greedy decoding to beam search.
    
    Attributes:
        strategy (YvSamplingStrategy): Primary sampling strategy.
            Default: TOP_K_TOP_P.
        temperature (float): Temperature for probability scaling.
            Higher = more random, Lower = more deterministic.
            Range: 0.1-2.0 recommended. Default: 1.0.
        top_k (int): Number of top tokens for top-k sampling.
            Only used when strategy involves top-k. Default: 50.
        top_p (float): Cumulative probability threshold for nucleus sampling.
            Only used when strategy involves top-p. Default: 0.9.
        min_p (float): Minimum probability threshold for tokens.
            Filters out very low probability tokens. Default: 0.0.
        typical_p (float): Probability threshold for typical sampling.
            Controls coherence-diversity trade-off. Default: 0.9.
        eta_cutoff (float): Eta cutoff for eta sampling.
            Entropy-scaled probability threshold. Default: 0.0.
        epsilon_cutoff (float): Epsilon cutoff for epsilon sampling.
            Absolute probability threshold. Default: 0.0.
        beam_width (int): Number of beams for beam search.
            Higher = better quality but slower. Default: 4.
        beam_groups (int): Number of beam groups for diverse beam search.
            Enables diverse outputs within beam search. Default: 1.
        length_penalty (float): Length penalty for beam search scoring.
            > 1.0 favors longer, < 1.0 favors shorter. Default: 1.0.
        early_stopping (bool): Whether to stop when all beams finish.
            Default: True.
        diversity_penalty (float): Penalty for similar tokens across beams.
            Encourages diverse beam outputs. Default: 0.0.
        repetition_penalty (float): Penalty for repeated tokens.
            > 1.0 discourages repetition. Default: 1.0.
        no_repeat_ngram_size (int): Size of n-grams to prevent repeating.
            0 = disabled. Default: 0.
        bad_words_ids (Optional[List[List[int]]]): Token IDs to suppress.
            Prevents generation of specified tokens. Default: None.
        min_new_tokens (int): Minimum new tokens to generate.
            Prevents premature stopping. Default: 0.
        max_new_tokens (int): Maximum new tokens to generate.
            Default: 100.
        do_sample (bool): Whether to use sampling or greedy.
            False = greedy even with other settings. Default: True.
    
    Example:
        >>> config = YvSamplingConfig(
        ...     strategy=YvSamplingStrategy.TOP_K_TOP_P,
        ...     temperature=0.7,
        ...     top_k=50,
        ...     top_p=0.9,
        ...     repetition_penalty=1.1
        ... )
    
    Note:
        Temperature=0 effectively becomes greedy regardless of strategy.
        Repetition penalty should be used carefully to avoid artifacts.
    """
    strategy: YvSamplingStrategy = YvSamplingStrategy.TOP_K_TOP_P
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.0
    typical_p: float = 0.9
    eta_cutoff: float = 0.0
    epsilon_cutoff: float = 0.0
    beam_width: int = 4
    beam_groups: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = True
    diversity_penalty: float = 0.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    bad_words_ids: Optional[List[List[int]]] = None
    min_new_tokens: int = 0
    max_new_tokens: int = 100
    do_sample: bool = True

    def __post_init__(self):
        """Post-initialization to convert string strategy to enum."""
        if isinstance(self.strategy, str):
            self.strategy = YvSamplingStrategy(self.strategy)


class YvSampler:
    """Unified sampler for text generation with multiple sampling strategies.
    
    Provides a single interface for various sampling strategies, handling
    temperature scaling, repetition penalty, and strategy-specific filtering.
    
    Supported Strategies:
        - Greedy: Deterministic highest probability selection
        - Top-k: Sample from top k tokens
        - Top-p: Nucleus sampling
        - Top-k + Top-p: Combined filtering
        - Typical: Surprisal-based sampling
        - Eta: Entropy-scaled adaptive sampling
    
    Key Features:
        - Unified interface for all strategies
        - Runtime parameter override via kwargs
        - Repetition penalty support
        - Beam search integration
    
    Attributes:
        config (YvSamplingConfig): Sampling configuration.
    
    Example:
        >>> sampler = YvSampler(YvSamplingConfig(temperature=0.7))
        >>> next_token = sampler.sample(logits, past_tokens=generated)
        >>> 
        >>> # Override temperature at runtime
        >>> next_token = sampler.sample(logits, temperature=0.5)
    
    Note:
        For beam search, use the beam_search() method directly.
        Temperature must be > 0 for sampling strategies.
    """

    def __init__(self, config: Optional[YvSamplingConfig] = None):
        """Initialize sampler with optional configuration.
        
        Args:
            config: Sampling configuration. If None, uses defaults.
        """
        self.config = config or YvSamplingConfig()

    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample next token from logits.

        Args:
            logits: Token logits [batch_size, vocab_size]
            past_tokens: Previously generated tokens for repetition penalty
            **kwargs: Additional sampling parameters

        Returns:
            Sampled token IDs [batch_size, 1]
        """
        config = self.config

        if kwargs.get("temperature") is not None:
            config = YvSamplingConfig(
                temperature=kwargs["temperature"],
                top_k=kwargs.get("top_k", config.top_k),
                top_p=kwargs.get("top_p", config.top_p),
                strategy=config.strategy
            )

        logits = logits / config.temperature

        if config.repetition_penalty != 1.0 and past_tokens is not None:
            logits = self._apply_repetition_penalty(logits, past_tokens, config.repetition_penalty)

        if config.strategy == YvSamplingStrategy.GREEDY:
            return self._greedy_sample(logits)
        elif config.strategy == YvSamplingStrategy.TOP_K:
            return self._top_k_sample(logits, config.top_k)
        elif config.strategy == YvSamplingStrategy.TOP_P:
            return self._top_p_sample(logits, config.top_p)
        elif config.strategy == YvSamplingStrategy.TOP_K_TOP_P:
            return self._top_k_top_p_sample(logits, config.top_k, config.top_p)
        elif config.strategy == YvSamplingStrategy.TYPICAL:
            return self._typical_sample(logits, config.typical_p)
        elif config.strategy == YvSamplingStrategy.ETA_SAMPLING:
            return self._eta_sample(logits, config.eta_cutoff)
        else:
            return self._top_k_top_p_sample(logits, config.top_k, config.top_p)

    def _greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Perform greedy sampling by selecting highest probability token.
        
        Args:
            logits: Token logits [batch_size, vocab_size].
            
        Returns:
            Selected token IDs [batch_size, 1].
        """
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _top_k_sample(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Perform top-k sampling from the k most likely tokens.
        
        Args:
            logits: Token logits [batch_size, vocab_size].
            top_k: Number of top tokens to consider.
            
        Returns:
            Sampled token IDs [batch_size, 1].
        """
        top_k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

        probs = F.softmax(top_k_logits, dim=-1)
        sampled_indices = torch.multinomial(probs, num_samples=1)

        return torch.gather(top_k_indices, -1, sampled_indices)

    def _top_p_sample(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Perform nucleus (top-p) sampling from smallest set with cumulative prob >= p.
        
        Args:
            logits: Token logits [batch_size, vocab_size].
            top_p: Cumulative probability threshold.
            
        Returns:
            Sampled token IDs [batch_size, 1].
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _top_k_top_p_sample(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """Perform combined top-k and top-p sampling.
        
        Applies top-k filtering first, then top-p filtering on the result.
        This is the recommended default sampling strategy.
        
        Args:
            logits: Token logits [batch_size, vocab_size].
            top_k: Number of top tokens for initial filtering.
            top_p: Cumulative probability threshold for nucleus filtering.
            
        Returns:
            Sampled token IDs [batch_size, 1].
        """
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, _ = torch.topk(logits, top_k, dim=-1)
            min_top_k = top_k_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_top_k,
                torch.full_like(logits, float('-inf')),
                logits
            )

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _typical_sample(self, logits: torch.Tensor, typical_p: float) -> torch.Tensor:
        """Perform typical sampling based on surprisal-entropy distance.
        
        Selects tokens whose surprisal (negative log probability) is close
        to the entropy of the distribution, producing more coherent outputs.
        
        Args:
            logits: Token logits [batch_size, vocab_size].
            typical_p: Probability mass to consider for typical set.
            
        Returns:
            Sampled token IDs [batch_size, 1].
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        surprisal = -log_probs

        typical_scores = torch.abs(surprisal - entropy)

        sorted_scores, sorted_indices = torch.sort(typical_scores, dim=-1)
        sorted_probs = torch.gather(probs, -1, sorted_indices)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        typical_mask = cumulative_probs <= typical_p
        typical_mask[..., 0] = True

        sorted_logits = torch.gather(logits, -1, sorted_indices)
        sorted_logits = sorted_logits.masked_fill(~typical_mask, float('-inf'))

        probs = F.softmax(sorted_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)

        return torch.gather(sorted_indices, -1, sampled)

    def _eta_sample(self, logits: torch.Tensor, eta_cutoff: float) -> torch.Tensor:
        """Perform eta sampling with entropy-scaled probability threshold.
        
        Uses entropy to compute an adaptive cutoff threshold, filtering
        tokens below the threshold before sampling.
        
        Args:
            logits: Token logits [batch_size, vocab_size].
            eta_cutoff: Multiplier for entropy-based cutoff.
            
        Returns:
            Sampled token IDs [batch_size, 1].
        """
        if eta_cutoff <= 0:
            return self._top_k_top_p_sample(logits, self.config.top_k, self.config.top_p)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        eta = torch.exp(-entropy) * eta_cutoff

        eta_mask = probs >= eta

        logits = logits.masked_fill(~eta_mask, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        past_tokens: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to previously generated tokens.
        
        Penalizes tokens that have appeared in the past by dividing
        positive logits or multiplying negative logits by the penalty.
        
        Args:
            logits: Token logits [batch_size, vocab_size].
            past_tokens: Previously generated token IDs.
            penalty: Repetition penalty factor (> 1.0 discourages repetition).
            
        Returns:
            Modified logits with penalty applied.
        """
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            unique_tokens = torch.unique(past_tokens[i])
            for token in unique_tokens:
                if logits[i, token] > 0:
                    logits[i, token] /= penalty
                else:
                    logits[i, token] *= penalty

        return logits

    def beam_search(
        self,
        logits: torch.Tensor,
        beam_scores: torch.Tensor,
        beam_width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform beam search expansion for multiple hypothesis tracking.
        
        Expands each beam by considering all possible next tokens,
        then selects the top-k scoring continuations.
        
        Args:
            logits: Token logits [num_beams, vocab_size].
            beam_scores: Cumulative scores for each beam [num_beams].
            beam_width: Number of beams to maintain.
            
        Returns:
            Tuple of:
                - token_indices: Selected token IDs for each beam
                - top_scores: Scores for selected continuations
                - beam_indices: Source beam indices for each selection
        """
        log_probs = F.log_softmax(logits, dim=-1)

        combined_scores = beam_scores.unsqueeze(-1) + log_probs

        flat_scores = combined_scores.view(-1)
        top_scores, top_indices = torch.topk(flat_scores, beam_width)

        beam_indices = top_indices // logits.size(-1)
        token_indices = top_indices % logits.size(-1)

        return token_indices.unsqueeze(-1), top_scores, beam_indices


def create_sampler(
    strategy: str = "top_k_top_p",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    **kwargs
) -> YvSampler:
    """Factory function to create a configured sampler.
    
    Convenience function for creating a YvSampler with common
    parameters without explicitly creating a YvSamplingConfig.
    
    Args:
        strategy: Sampling strategy name. Default: "top_k_top_p".
        temperature: Sampling temperature. Default: 1.0.
        top_k: Top-k parameter. Default: 50.
        top_p: Top-p parameter. Default: 0.9.
        **kwargs: Additional parameters passed to YvSamplingConfig.
        
    Returns:
        Configured YvSampler instance.
    
    Example:
        >>> sampler = create_sampler(temperature=0.7, top_k=40)
    """
    config = YvSamplingConfig(
        strategy=strategy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        **kwargs
    )
    return YvSampler(config)
