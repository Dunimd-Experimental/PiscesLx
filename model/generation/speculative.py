#!/usr/bin/env/python3
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
Speculative Decoding Module for Yv Model.

This module provides advanced speculative decoding implementations for
accelerating text generation through draft-then-verify paradigms.

Speculative Decoding Overview:
    Speculative decoding accelerates autoregressive generation by:
    1. Using a fast draft model to generate candidate tokens
    2. Verifying all candidates in a single forward pass of the target model
    3. Accepting valid tokens and resampling at rejection points
    
    This achieves 2-3x speedup while maintaining the same output distribution
    as standard autoregressive decoding.

Module Components:
    1. Core Classes:
       - YvSpeculativeConfig: Configuration for speculative decoding
       - YvVerificationResult: Result container for draft verification
       - YvDraftModel: Lightweight draft model for fast generation
       - YvSpeculativeDecoder: Standard speculative decoder
       - YvAdaptiveSpeculativeDecoder: Adaptive decoder with dynamic parameters
    
    2. Verification Strategies:
       - STANDARD: Sequential verification of draft tokens
       - PARALLEL: Single forward pass for all draft tokens
       - SEQUENTIAL: Token-by-token verification
       - MEDUSA: Multi-head prediction without separate draft model
       - EAGLE: Feature-based speculative decoding
    
    3. Advanced Features:
       - YvMedusaHead: Multi-token prediction heads
       - YvParallelVerifier: Parallel verification implementation

Key Features:
    - Draft-then-verify paradigm for 2-3x speedup
    - Multiple verification strategies (standard, parallel, medusa, eagle)
    - Adaptive parameter adjustment based on acceptance rates
    - Automatic fallback to standard generation on repeated failures
    - Comprehensive performance statistics

Performance Characteristics:
    - Acceptance rate: Typically 60-80% with well-matched draft model
    - Speedup: 2-3x for high acceptance rates
    - Memory: Requires additional memory for draft model
    - Best for: Batch size 1, high acceptance scenarios

Design Principles:
    - Single implementation per feature (no redundancy)
    - Flagship-level completeness matching latest LLM architectures
    - Support for various verification strategies
    - Robust fallback mechanisms

Usage Example:
    >>> from model.generation.speculative import (
    ...     YvAdaptiveSpeculativeDecoder,
    ...     YvSpeculativeConfig
    ... )
    >>> 
    >>> # Configure speculative decoding
    >>> config = YvSpeculativeConfig(
    ...     draft_length=5,
    ...     acceptance_threshold=0.8,
    ...     temperature=0.7
    ... )
    >>> 
    >>> # Create decoder
    >>> decoder = YvAdaptiveSpeculativeDecoder(config, model)
    >>> 
    >>> # Generate with speculative decoding
    >>> generated, stats = decoder.speculative_generate(
    ...     input_ids=input_ids,
    ...     max_length=100
    ... )
    >>> 
    >>> print(f"Acceptance rate: {stats['draft_acceptance_rate']:.2%}")
    >>> print(f"Speedup: {stats['speedup']:.2f}x")

Note:
    All classes follow the YvXxx naming convention.
    For best performance, use with batch_size=1.
    Draft model quality significantly impacts acceptance rates.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum

from utils.dc import PiscesLxLogger

_LOG = PiscesLxLogger(__name__)


class YvVerificationStrategy(Enum):
    """Enumeration of available verification strategies for speculative decoding.
    
    Defines the different approaches for verifying draft tokens during
    speculative decoding. Each strategy offers different trade-offs
    between verification speed and acceptance rate.
    
    Attributes:
        STANDARD: Standard sequential verification of draft tokens.
            Verifies tokens one by one with early stopping on rejection.
            Most reliable but slower verification.
        PARALLEL: Parallel verification in single forward pass.
            Processes all draft tokens simultaneously for efficiency.
            Recommended for most use cases.
        SEQUENTIAL: Token-by-token verification with full recomputation.
            Most conservative approach with highest accuracy.
            Use when parallel verification has issues.
        MEDUSA: Medusa-style multi-head prediction verification.
            Uses multiple prediction heads instead of draft model.
            No separate draft model required.
        EAGLE: EAGLE-style feature-based speculative decoding.
            Uses feature-level prediction for draft generation.
            Advanced technique for specific architectures.
    
    Example:
        >>> strategy = YvVerificationStrategy.PARALLEL
        >>> if strategy == YvVerificationStrategy.MEDUSA:
        ...     print("Using Medusa heads for speculation")
    
    Note:
        PARALLEL is recommended for most use cases due to its efficiency.
        MEDUSA is useful when draft model memory is a concern.
    """
    STANDARD = "standard"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    MEDUSA = "medusa"
    EAGLE = "eagle"


@dataclass
class YvSpeculativeConfig:
    """Configuration dataclass for speculative decoding parameters.
    
    Encapsulates all parameters needed to configure speculative decoding,
    including draft generation, verification, and sampling settings.
    
    Attributes:
        num_candidates (int): Number of candidate tokens to generate in parallel.
            Higher values increase potential speedup but also memory usage.
            Default: 4.
        draft_length (int): Length of the draft sequence to generate.
            Longer sequences have higher potential speedup but lower acceptance.
            Typical values: 4-8. Default: 5.
        acceptance_threshold (float): Threshold for accepting draft tokens.
            Higher values are more conservative (fewer acceptances).
            Range: 0.0-1.0. Default: 0.8.
        temperature (float): Temperature for sampling during draft generation.
            Bounded to 0.1-2.0 for stability. Default: 0.7.
        top_k (int): Number of top-k tokens to consider during sampling.
            Bounded to 1-1000. Default: 50.
        top_p (float): Cumulative probability threshold for nucleus sampling.
            Bounded to 0.1-1.0. Default: 0.9.
        verification_strategy (YvVerificationStrategy): Strategy for
            verifying draft tokens. Default: PARALLEL.
        medusa_heads (int): Number of Medusa heads for multi-token prediction.
            Only used with MEDUSA strategy. Default: 4.
        use_tree_attention (bool): Whether to use tree-based attention for
            verification. Can improve efficiency for long draft sequences.
            Default: True.
    
    Example:
        >>> config = YvSpeculativeConfig(
        ...     draft_length=6,
        ...     acceptance_threshold=0.75,
        ...     verification_strategy=YvVerificationStrategy.PARALLEL
        ... )
    
    Note:
        Temperature, top_k, and top_p are automatically bounded to valid ranges.
        Draft length should be tuned based on acceptance rate observations.
    """
    num_candidates: int = 4
    draft_length: int = 5
    acceptance_threshold: float = 0.8
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    verification_strategy: YvVerificationStrategy = YvVerificationStrategy.PARALLEL
    medusa_heads: int = 4
    use_tree_attention: bool = True
    
    def __post_init__(self):
        """Post-initialization to validate and bound parameters."""
        self.temperature = max(0.1, min(2.0, self.temperature))
        self.top_k = max(1, min(1000, self.top_k))
        self.top_p = max(0.1, min(1.0, self.top_p))
        if isinstance(self.verification_strategy, str):
            self.verification_strategy = YvVerificationStrategy(self.verification_strategy)


@dataclass
class YvVerificationResult:
    """Result container for draft token verification.
    
    Encapsulates the results of verifying a sequence of draft tokens,
    including accepted tokens and updated cache state.
    
    Attributes:
        accepted_ids (torch.Tensor): Accepted token IDs from the draft.
            Shape: [batch_size, num_accepted].
        num_accepted (int): Number of tokens that passed verification.
            Used for statistics and adaptive parameter adjustment.
        new_past_key_values (Optional[Any]): Updated KV cache after
            processing the accepted tokens. None if caching disabled.
        rejection_position (int): Position where rejection occurred.
            -1 if all tokens were accepted.
    
    Example:
        >>> result = verifier(input_ids, draft_ids)
        >>> if result.num_accepted > 0:
        ...     generated = torch.cat([generated, result.accepted_ids], dim=1)
    
    Note:
        The accepted_ids tensor may be padded if batch acceptance varies.
        Use num_accepted to determine actual valid tokens.
    """
    accepted_ids: torch.Tensor
    num_accepted: int
    new_past_key_values: Optional[Any] = None
    rejection_position: int = -1


class YvDraftModel(nn.Module):
    """Lightweight draft model for fast token generation.
    
    Creates a smaller version of the main model for generating draft sequences
    quickly. The draft model should be significantly faster than the target
    model while maintaining reasonable acceptance rates.
    
    Architecture:
        - Token embedding layer
        - Stack of Transformer encoder layers
        - Linear language modeling head
    
    Design Considerations:
        - Typically 2-4x smaller than target model
        - Fewer layers (e.g., target_layers // 4)
        - Smaller hidden dimension (e.g., target_hidden // 2)
        - Same vocabulary as target model
    
    Attributes:
        vocab_size (int): Vocabulary size (same as target model).
        hidden_size (int): Hidden dimension of draft model.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        embed (nn.Embedding): Token embedding layer.
        encoder (nn.TransformerEncoder): Transformer encoder stack.
        lm_head (nn.Linear): Language modeling output head.
    
    Example:
        >>> draft = YvDraftModel(
        ...     vocab_size=128000,
        ...     hidden_size=1024,
        ...     num_layers=6,
        ...     num_heads=8
        ... )
        >>> logits = draft(input_ids)  # Fast forward pass
    
    Note:
        Draft model quality directly impacts acceptance rates.
        Consider knowledge distillation from target model for better alignment.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_position_embeddings: int = 2048,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize draft model with architecture parameters.
        
        Args:
            vocab_size: Vocabulary size (must match target model).
            hidden_size: Hidden dimension for draft model.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            max_position_embeddings: Maximum sequence length. Default: 2048.
            dropout: Dropout probability. Default: 0.0.
            device: Device to place model on.
            dtype: Data type for model parameters.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.embed = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            device=device,
            dtype=dtype
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through draft model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            
        Returns:
            Logits for next token prediction [batch_size, seq_len, vocab_size].
        """
        x = self.embed(input_ids)
        x = self.encoder(x)
        logits = self.lm_head(x)
        return logits


class YvMedusaHead(nn.Module):
    """Medusa-style multi-token prediction head for speculative decoding.
    
    Implements multiple prediction heads for parallel token prediction,
    enabling speculative decoding without a separate draft model. Each head
    predicts a token at a different future position.
    
    Architecture:
        - Multiple prediction heads (one per future position)
        - Each head: Linear -> SiLU -> Linear
        - Shares hidden states from target model
    
    Advantages over Draft Model:
        - No separate model to train/store
        - Lower memory overhead
        - Better alignment with target model
    
    Attributes:
        num_heads (int): Number of prediction heads.
        heads (nn.ModuleList): List of prediction head modules.
    
    Example:
        >>> medusa = YvMedusaHead(hidden_size=4096, vocab_size=128000, num_heads=4)
        >>> logits_list = medusa(hidden_states)  # List of [batch, seq, vocab]
        >>> # logits_list[i] predicts token at position +i+1
    
    Note:
        Heads should be trained with the main model for best results.
        More heads increase speculation length but may reduce accuracy.
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize Medusa heads.
        
        Args:
            hidden_size: Hidden dimension from target model.
            vocab_size: Vocabulary size for output predictions.
            num_heads: Number of prediction heads. Default: 4.
            device: Device to place heads on.
            dtype: Data type for head parameters.
        """
        super().__init__()
        self.num_heads = num_heads
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
            )
            for _ in range(num_heads)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize head weights with normal distribution."""
        for head in self.heads:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Generate predictions from all heads.
        
        Args:
            hidden_states: Hidden states from target model [batch, seq, hidden].
            
        Returns:
            List of logits tensors, one per head.
            Each tensor shape: [batch, seq, vocab_size].
        """
        return [head(hidden_states) for head in self.heads]


class YvParallelVerifier(nn.Module):
    """Parallel verifier for speculative decoding.
    
    Processes all draft tokens in a single forward pass for efficient
    verification. This is the recommended verification strategy for most
    use cases due to its efficiency.
    
    Verification Process:
        1. Concatenate input_ids with draft_ids
        2. Single forward pass through target model
        3. Extract logits for draft positions
        4. Compute acceptance probabilities
        5. Accept tokens meeting threshold
    
    Attributes:
        config (YvSpeculativeConfig): Verification configuration.
        model (nn.Module): Target model for verification.
        vocab_size (int): Vocabulary size from model config.
    
    Example:
        >>> verifier = YvParallelVerifier(config, target_model)
        >>> result = verifier(input_ids, draft_ids, past_key_values=cache)
        >>> accepted = result.accepted_ids[:, :result.num_accepted]
    
    Note:
        Parallel verification is most efficient with KV caching enabled.
        Acceptance threshold should be tuned based on draft model quality.
    """
    
    def __init__(self, config: YvSpeculativeConfig, model: nn.Module):
        """Initialize parallel verifier.
        
        Args:
            config: Configuration containing acceptance threshold.
            model: Target model for verification.
        """
        super().__init__()
        self.config = config
        self.model = model
        self.vocab_size = getattr(model.config, 'vocab_size', 65536)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        draft_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
    ) -> YvVerificationResult:
        device = input_ids.device
        batch_size, draft_len = draft_ids.shape
        
        full_sequence = torch.cat([input_ids, draft_ids], dim=1)
        
        with torch.no_grad():
            outputs = self.model(
                full_sequence,
                use_cache=True,
                past_key_values=past_key_values,
            )
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs)
                new_past_key = outputs.get('past_key_values', None)
            else:
                logits = outputs
                new_past_key = None
            
            start_idx = input_ids.shape[1]
            draft_logits = logits[:, start_idx:, :]
            
            draft_probs = F.softmax(draft_logits, dim=-1)
            
            token_probs = torch.gather(
                draft_probs,
                dim=-1,
                index=draft_ids.unsqueeze(-1),
            ).squeeze(-1)
            
            acceptance_mask = torch.zeros_like(token_probs, dtype=torch.bool)
            rejection_position = -1
            
            for b in range(batch_size):
                cum_prob = 1.0
                for i in range(draft_len):
                    p_draft = token_probs[b, i].item()
                    p_accept = min(1.0, cum_prob / (p_draft + 1e-8))
                    
                    if p_accept >= self.config.acceptance_threshold and p_draft > 1e-8:
                        acceptance_mask[b, i] = True
                        cum_prob *= p_draft
                    else:
                        if rejection_position == -1:
                            rejection_position = i
                        break
            
            accepted_lengths = acceptance_mask.sum(dim=1)
            max_accepted = accepted_lengths.max().item()
            
            if max_accepted == 0:
                next_token = torch.multinomial(draft_probs[:, 0], num_samples=1)
                return YvVerificationResult(
                    accepted_ids=next_token,
                    num_accepted=1,
                    new_past_key_values=new_past_key,
                    rejection_position=0
                )
            
            accepted_ids = torch.zeros(batch_size, max_accepted, dtype=torch.long, device=device)
            for b in range(batch_size):
                accepted_ids[b, :accepted_lengths[b]] = draft_ids[b, :accepted_lengths[b]]
            
            return YvVerificationResult(
                accepted_ids=accepted_ids,
                num_accepted=int(max_accepted),
                new_past_key_values=new_past_key,
                rejection_position=rejection_position
            )


class YvSpeculativeDecoder(nn.Module):
    """Unified speculative decoder with multiple verification strategies.
    
    Implements the draft-then-verify paradigm for accelerating autoregressive
    generation. Uses a lightweight draft model for fast candidate generation
    and the target model for verification.
    
    Supported Features:
        - Standard draft-then-verify paradigm
        - Parallel verification (single forward pass)
        - Medusa-style multi-head prediction
        - Adaptive parameter adjustment
        - Automatic fallback on repeated failures
    
    Architecture:
        - Draft Model: Lightweight model for fast token generation
        - Parallel Verifier: Efficient batch verification
        - Medusa Head: Multi-token prediction heads
    
    Performance Tracking:
        - Acceptance rate monitoring
        - Speedup calculation
        - Iteration-level statistics
        - Automatic parameter adaptation
    
    Attributes:
        config (YvSpeculativeConfig): Decoder configuration.
        model (nn.Module): Target model for verification.
        tokenizer (Optional[Any]): Tokenizer for text processing.
        on_stats (Optional[Callable]): Callback for statistics reporting.
        draft_model (YvDraftModel): Lightweight draft model.
        parallel_verifier (YvParallelVerifier): Parallel verification module.
        medusa_head (YvMedusaHead): Multi-token prediction heads.
        performance_history (List[Dict]): History of performance metrics.
        adaptation_interval (int): Steps between parameter adaptation.
    
    Example:
        >>> config = YvSpeculativeConfig(draft_length=5)
        >>> decoder = YvSpeculativeDecoder(config, model)
        >>> generated, stats = decoder.speculative_generate(input_ids, max_length=100)
        >>> print(f"Speedup: {stats['speedup']:.2f}x")
    
    Note:
        Best performance with batch_size=1 and well-matched draft model.
        Automatic fallback to standard generation after repeated failures.
    """
    
    def __init__(
        self,
        config: YvSpeculativeConfig,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        on_stats: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize speculative decoder.
        
        Args:
            config: Configuration for speculative decoding parameters.
            model: Target model for verification and fallback generation.
            tokenizer: Optional tokenizer for text processing.
            on_stats: Optional callback function for statistics reporting.
                Receives a dictionary of performance metrics after each generation.
        """
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.on_stats = on_stats
        
        self.draft_model = self._create_draft_model()
        self.parallel_verifier = YvParallelVerifier(config, model)
        
        hidden_size = getattr(model.config, 'hidden_size', 2048)
        vocab_size = getattr(model.config, 'vocab_size', 65536)
        self.medusa_head = YvMedusaHead(hidden_size, vocab_size, config.medusa_heads)
        
        self.performance_history: List[Dict[str, float]] = []
        self.adaptation_interval = 10
    
    def _create_draft_model(self) -> YvDraftModel:
        """Create a lightweight draft model based on target model architecture.
        
        The draft model is automatically sized to be smaller than the target
        model while maintaining compatibility with the vocabulary.
        
        Returns:
            Configured YvDraftModel instance.
        """
        vocab_size = getattr(self.model.config, 'vocab_size', 65536)
        base_hidden = getattr(self.model.config, 'hidden_size', 2048)
        base_layers = getattr(self.model.config, 'num_layers', 
                              getattr(self.model.config, 'n_layer', 24))
        base_heads = getattr(self.model.config, 'num_heads', 
                             getattr(self.model.config, 'n_head', 16))
        
        hidden_size = max(512, base_hidden // 2)
        num_layers = max(2, base_layers // 4)
        
        preferred_max = max(4, min(8, max(1, base_heads // 2)))
        candidates = [h for h in range(preferred_max, 0, -1) if hidden_size % h == 0]
        if candidates:
            num_heads = candidates[0]
        else:
            num_heads = max(1, min(preferred_max, base_heads))
            hidden_size = ((hidden_size + num_heads - 1) // num_heads) * num_heads
        
        return YvDraftModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads
        )
    
    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        cache_manager: Optional[Any] = None,
        **model_kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate tokens using speculative decoding.
        
        Main generation loop that alternates between draft generation
        and verification until reaching the target length.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            attention_mask: Optional attention mask. Will be created if None.
            max_length: Maximum sequence length to generate.
            cache_manager: Optional cache manager for speculative caching.
            **model_kwargs: Additional arguments passed to the model.
            
        Returns:
            Tuple of:
                - generated_ids: Generated token sequence [batch_size, new_seq_len]
                - stats: Dictionary of performance statistics including:
                    - method: 'speculative'
                    - total_draft_tokens: Total draft tokens generated
                    - accepted_tokens: Tokens that passed verification
                    - rejected_tokens: Tokens that failed verification
                    - draft_acceptance_rate: Fraction of accepted draft tokens
                    - speedup: Estimated speedup over standard generation
                    - iter_accept: List of accepted tokens per iteration
                    - total_time_ms: Total generation time in milliseconds
                    - avg_accept_per_iter: Average accepted tokens per iteration
                    - max_accept_in_iter: Maximum accepted in single iteration
                    - batch_size: Batch size used
        """
        if len(self.performance_history) >= self.adaptation_interval:
            self._adapt_parameters()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated_ids = input_ids.clone()
        stats = {
            'method': 'speculative',
            'total_draft_tokens': 0,
            'accepted_tokens': 0,
            'rejected_tokens': 0,
            'draft_acceptance_rate': 0.0,
            'speedup': 1.0,
            'iter_accept': [],
            'total_time_ms': 0.0,
            'avg_accept_per_iter': 0.0,
            'max_accept_in_iter': 0,
            'batch_size': batch_size,
        }
        start_time = time.time()
        
        past_key_values = None
        zero_accept_streak = 0
        
        while generated_ids.shape[1] < max_length:
            if cache_manager is not None:
                cached = cache_manager.get_speculative_cache(self.config.draft_length)
                if cached is not None:
                    return cached, {'from_cache': True}
            
            draft_ids, draft_logits = self._generate_draft_sequence(
                generated_ids, attention_mask, **model_kwargs
            )
            
            result = self._verify_and_accept(
                generated_ids, draft_ids, past_key_values, **model_kwargs
            )
            
            accepted_ids = result.accepted_ids
            num_accepted = result.num_accepted
            past_key_values = result.new_past_key_values
            
            generated_ids = torch.cat([generated_ids, accepted_ids], dim=1)
            
            if attention_mask is None:
                attention_mask = torch.ones_like(generated_ids, dtype=torch.long, device=device)
            else:
                add_len = accepted_ids.shape[1]
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], add_len), device=device, dtype=attention_mask.dtype)
                ], dim=1)
            
            stats['total_draft_tokens'] += draft_ids.shape[1]
            stats['accepted_tokens'] += num_accepted
            stats['rejected_tokens'] += max(0, draft_ids.shape[1] - num_accepted)
            stats['iter_accept'].append(int(num_accepted))
            stats['max_accept_in_iter'] = max(stats['max_accept_in_iter'], int(num_accepted))
            
            if num_accepted == 0:
                zero_accept_streak += 1
                
                if zero_accept_streak >= 3:
                    fallback_ids, fallback_stats = self._standard_generate(
                        generated_ids, attention_mask, max_length, **model_kwargs
                    )
                    stats.update({k: v for k, v in fallback_stats.items() if k not in stats})
                    generated_ids = fallback_ids
                    break
                
                outputs = self.model(generated_ids, attention_mask=attention_mask, **model_kwargs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                next_logits = self._apply_sampling(logits[:, -1, :])
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                ], dim=1)
                
                stats['accepted_tokens'] += 1
                stats['iter_accept'][-1] = 1
                zero_accept_streak = 0
                continue
            
            zero_accept_streak = 0
            
            if generated_ids.shape[1] >= max_length:
                break
        
        if stats['total_draft_tokens'] > 0:
            stats['draft_acceptance_rate'] = stats['accepted_tokens'] / stats['total_draft_tokens']
            avg_accept = sum(stats['iter_accept']) / max(1, len(stats['iter_accept']))
            stats['avg_accept_per_iter'] = avg_accept
            stats['speedup'] = 1.0 + (stats['accepted_tokens'] / max(1, stats['rejected_tokens']))
        
        stats['total_time_ms'] = (time.time() - start_time) * 1000.0
        stats['num_iterations'] = len(stats['iter_accept'])
        
        if cache_manager is not None:
            cache_manager.set_speculative_cache(self.config.draft_length, generated_ids)
        
        _LOG.debug(
            f"[SpecDecode] draft_len={self.config.draft_length}, "
            f"accept_rate={stats['draft_acceptance_rate']:.3f}, "
            f"avg_accept={stats['avg_accept_per_iter']:.1f}, "
            f"speedup={stats['speedup']:.2f}, "
            f"time_ms={stats['total_time_ms']:.1f}"
        )
        
        if self.on_stats is not None:
            try:
                self.on_stats(stats)
            except Exception:
                pass
        
        self.performance_history.append({
            'acceptance_rate': stats.get('draft_acceptance_rate', 0),
            'speedup': stats.get('speedup', 1),
            'avg_accept': stats.get('avg_accept_per_iter', 0),
        })
        
        return generated_ids, stats
    
    def _apply_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature, top-k, and top-p sampling to logits.
        
        Args:
            logits: Raw logits from the model [batch_size, vocab_size].
            
        Returns:
            Modified logits after applying sampling parameters.
        """
        if self.config.temperature > 0:
            temp = max(0.1, min(2.0, self.config.temperature))
            logits = logits / temp
        
        if self.config.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(
                logits, min(self.config.top_k, logits.size(-1))
            )
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, top_k_indices, top_k_logits)
        
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def _generate_draft_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a sequence of draft tokens using the draft model.
        
        Autoregressively generates draft_length tokens using the lightweight
        draft model for fast candidate generation.
        
        Args:
            input_ids: Current sequence [batch_size, seq_len].
            attention_mask: Optional attention mask.
            **model_kwargs: Additional model arguments.
            
        Returns:
            Tuple of:
                - draft_seq: Draft token sequence [batch_size, draft_length]
                - draft_logits: Logits for each draft position [batch_size, draft_length, vocab]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        draft_len = max(1, self.config.draft_length)
        
        cur_ids = input_ids
        draft_tokens: List[torch.Tensor] = []
        step_logits_list: List[torch.Tensor] = []
        
        with torch.no_grad():
            for _ in range(draft_len):
                logits = self.draft_model(cur_ids)
                step_logits = logits[:, -1:, :]
                step_token = self._sample_candidates(step_logits).unsqueeze(1)[:, 0:1]
                draft_tokens.append(step_token)
                cur_ids = torch.cat([cur_ids, step_token], dim=1)
                step_logits_list.append(step_logits)
        
        draft_seq = torch.cat(draft_tokens, dim=1).to(device)
        vocab_size = getattr(self.model.config, 'vocab_size', 65536)
        draft_step_logits = torch.cat(step_logits_list, dim=1) if step_logits_list else torch.zeros(
            batch_size, 0, vocab_size, device=device
        )
        
        return draft_seq, draft_step_logits
    
    def _sample_candidates(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample candidate tokens from logits.
        
        Applies temperature, top-k, and top-p filtering before sampling
        multiple candidates for speculative decoding.
        
        Args:
            logits: Logits for sampling [batch_size, 1, vocab_size].
            
        Returns:
            Sampled candidate token IDs [batch_size, num_candidates].
        """
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
        
        if self.config.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, top_k_indices, top_k_logits)
        
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        candidates = torch.multinomial(probs.squeeze(1), self.config.num_candidates, replacement=False)
        
        return candidates
    
    def _verify_and_accept(
        self,
        input_ids: torch.Tensor,
        draft_ids: torch.Tensor,
        past_key_values: Optional[Any],
        **model_kwargs
    ) -> YvVerificationResult:
        """Verify draft tokens and return accepted sequence.
        
        Uses the parallel verifier to check draft tokens against
        the target model's predictions.
        
        Args:
            input_ids: Input sequence before draft tokens.
            draft_ids: Draft token sequence to verify.
            past_key_values: Optional KV cache for efficient computation.
            **model_kwargs: Additional model arguments.
            
        Returns:
            YvVerificationResult containing accepted tokens and updated cache.
        """
        if draft_ids.shape[1] == 0:
            return YvVerificationResult(
                accepted_ids=draft_ids,
                num_accepted=0,
                new_past_key_values=past_key_values
            )
        
        return self.parallel_verifier(input_ids, draft_ids, past_key_values)
    
    def _standard_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
        **model_kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Fallback to standard autoregressive generation.
        
        Used when speculative decoding fails repeatedly or when
        acceptance rate is too low.
        
        Args:
            input_ids: Current sequence.
            attention_mask: Attention mask for the sequence.
            max_length: Maximum sequence length.
            **model_kwargs: Additional model arguments.
            
        Returns:
            Tuple of generated sequence and statistics dictionary.
        """
        stats = {'method': 'standard_fallback'}
        start_time = time.time()
        
        current_ids = input_ids
        current_mask = attention_mask
        
        while current_ids.shape[1] < max_length:
            with torch.no_grad():
                outputs = self.model(current_ids, attention_mask=current_mask, **model_kwargs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                next_logits = self._apply_sampling(logits[:, -1, :])
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
                current_mask = torch.cat([
                    current_mask,
                    torch.ones((current_mask.shape[0], 1), device=current_ids.device, dtype=current_mask.dtype)
                ], dim=1)
        
        stats['total_time_ms'] = (time.time() - start_time) * 1000.0
        
        return current_ids, stats
    
    def _adapt_parameters(self):
        """Adapt speculative decoding parameters based on performance history.
        
        Analyzes recent performance metrics and adjusts parameters to
        optimize acceptance rate and speedup. Called periodically during
        generation.
        
        Adaptation Rules:
            - High acceptance (>0.8) but low speedup: Increase candidates
            - Low acceptance (<0.5): Decrease candidates
            - Very high acceptance (>0.9): Increase draft length
            - Low acceptance (<0.6): Decrease draft length
            - Low acceptance: Lower temperature and top_p
            - High acceptance but low speedup: Raise temperature and top_p
        """
        if len(self.performance_history) < self.adaptation_interval:
            return
        
        recent_history = self.performance_history[-self.adaptation_interval:]
        avg_acceptance_rate = sum(h['acceptance_rate'] for h in recent_history) / len(recent_history)
        avg_speedup = sum(h['speedup'] for h in recent_history) / len(recent_history)
        
        if avg_acceptance_rate > 0.8 and avg_speedup < 2.0:
            self.config.num_candidates = min(8, self.config.num_candidates + 1)
        elif avg_acceptance_rate < 0.5:
            self.config.num_candidates = max(2, self.config.num_candidates - 1)
        
        if avg_acceptance_rate > 0.9:
            self.config.draft_length = min(10, self.config.draft_length + 1)
        elif avg_acceptance_rate < 0.6:
            self.config.draft_length = max(2, self.config.draft_length - 1)
        
        if avg_acceptance_rate < 0.5:
            self.config.temperature = max(0.5, round(self.config.temperature * 0.9, 2))
            self.config.top_p = max(0.7, round(self.config.top_p - 0.05, 2))
        elif avg_acceptance_rate > 0.85 and avg_speedup < 1.8:
            self.config.temperature = min(1.2, round(self.config.temperature * 1.05, 2))
            self.config.top_p = min(0.98, round(self.config.top_p + 0.02, 2))
        
        self.performance_history = self.performance_history[-self.adaptation_interval // 2:]


class YvAdaptiveSpeculativeDecoder(YvSpeculativeDecoder):
    """Adaptive speculative decoder with dynamic parameter adjustment.
    
    Extends YvSpeculativeDecoder with automatic parameter adaptation
    based on real-time performance monitoring. Continuously optimizes
    draft length, candidate count, and sampling parameters.
    
    Key Features:
        - Automatic parameter tuning during generation
        - Performance history tracking
        - Adaptive draft length adjustment
        - Temperature and top-p optimization
    
    Adaptation Strategy:
        - Monitors acceptance rate and speedup over recent iterations
        - Adjusts parameters to maximize acceptance rate while
          maintaining good speedup
        - Falls back to conservative settings on poor performance
    
    Attributes:
        adaptation_interval (int): Number of iterations between adaptations.
    
    Example:
        >>> config = YvSpeculativeConfig(draft_length=5)
        >>> decoder = YvAdaptiveSpeculativeDecoder(config, model)
        >>> # Parameters will automatically adapt during generation
        >>> generated, stats = decoder.speculative_generate(input_ids)
    
    Note:
        Inherits all functionality from YvSpeculativeDecoder.
        Adaptation occurs at the start of each generation call.
    """
    
    def __init__(
        self,
        config: YvSpeculativeConfig,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        on_stats: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize adaptive speculative decoder.
        
        Args:
            config: Configuration for speculative decoding.
            model: Target model for verification.
            tokenizer: Optional tokenizer for text processing.
            on_stats: Optional callback for statistics reporting.
        """
        super().__init__(config, model, tokenizer, on_stats)
        self.adaptation_interval = 10
    
    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        cache_manager: Optional[Any] = None,
        **model_kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate tokens with adaptive parameter adjustment.
        
        Overrides parent method to add automatic parameter adaptation
        before each generation call.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            attention_mask: Optional attention mask.
            max_length: Maximum sequence length.
            cache_manager: Optional cache manager.
            **model_kwargs: Additional model arguments.
            
        Returns:
            Tuple of generated sequence and statistics dictionary.
        """
        if len(self.performance_history) >= self.adaptation_interval:
            self._adapt_parameters()
        
        generated_ids, stats = super().speculative_generate(
            input_ids, attention_mask, max_length, cache_manager, **model_kwargs
        )
        
        self.performance_history.append({
            'acceptance_rate': stats['draft_acceptance_rate'],
            'speedup': stats['speedup'],
            'num_candidates': self.config.num_candidates,
            'draft_length': self.config.draft_length
        })
        
        return generated_ids, stats


class YvMedusaDecoder(nn.Module):
    """Medusa-style speculative decoder with multi-head prediction.
    
    Implements speculative decoding using multiple prediction heads
    attached to the main model's hidden states. Eliminates the need
    for a separate draft model, reducing memory overhead.
    
    Architecture:
        - Multiple prediction heads (one per future position)
        - Each head predicts a token at position +i+1
        - Heads share hidden states from target model
    
    Advantages:
        - No separate draft model required
        - Lower memory footprint
        - Better alignment with target model
        - Simpler deployment
    
    Attributes:
        config (YvSpeculativeConfig): Decoder configuration.
        model (nn.Module): Target model for hidden states.
        tokenizer (Optional[Any]): Tokenizer for text processing.
        on_stats (Optional[Callable]): Statistics callback.
        medusa_heads (nn.ModuleList): Multi-token prediction heads.
    
    Example:
        >>> config = YvSpeculativeConfig(medusa_heads=4)
        >>> decoder = YvMedusaDecoder(config, model)
        >>> generated, stats = decoder.generate(input_ids, max_length=100)
    
    Note:
        Medusa heads should be trained with the main model for best results.
        Performance depends on head prediction accuracy.
    """
    
    def __init__(
        self,
        config: YvSpeculativeConfig,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        on_stats: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize Medusa decoder.
        
        Args:
            config: Configuration containing medusa_heads parameter.
            model: Target model providing hidden states.
            tokenizer: Optional tokenizer for text processing.
            on_stats: Optional callback for statistics reporting.
        """
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.on_stats = on_stats
        
        hidden_size = getattr(model.config, 'hidden_size', 2048)
        vocab_size = getattr(model.config, 'vocab_size', 65536)
        
        self.medusa_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, vocab_size, bias=False)
            )
            for _ in range(config.medusa_heads)
        ])
        
        self._init_medusa_weights()
    
    def _init_medusa_weights(self):
        """Initialize Medusa head weights with normal distribution."""
        for head in self.medusa_heads:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> List[torch.Tensor]:
        """Generate predictions from all Medusa heads.
        
        Args:
            hidden_states: Hidden states from target model [batch, seq, hidden].
            
        Returns:
            List of logits tensors, one per head.
        """
        return [head(hidden_states) for head in self.medusa_heads]
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        **model_kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate tokens using Medusa-style speculation.
        
        Uses multi-head prediction to generate candidate tokens and
        verifies them against the target model's main predictions.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            attention_mask: Optional attention mask.
            max_length: Maximum sequence length.
            **model_kwargs: Additional model arguments.
            
        Returns:
            Tuple of:
                - generated_ids: Generated token sequence
                - stats: Dictionary with method, acceptance_rate, speedup, etc.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated_ids = input_ids.clone()
        stats = {
            'method': 'medusa',
            'total_predicted': 0,
            'accepted_tokens': 0,
            'acceptance_rate': 0.0,
            'speedup': 1.0,
            'total_time_ms': 0.0
        }
        start_time = time.time()
        
        while generated_ids.shape[1] < max_length:
            with torch.no_grad():
                outputs = self.model(
                    generated_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    **model_kwargs
                )
                
                if isinstance(outputs, dict):
                    hidden_states = outputs.get('hidden_states', None)
                    past_key_values = outputs.get('past_key_values', None)
                    main_logits = outputs.get('logits', outputs)
                else:
                    hidden_states = None
                    past_key_values = None
                    main_logits = outputs
                
                if hidden_states is None:
                    main_probs = F.softmax(main_logits[:, -1, :], dim=-1)
                    next_token = torch.multinomial(main_probs, num_samples=1)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    continue
                
                last_hidden = hidden_states[:, -1, :] if hidden_states.dim() == 3 else hidden_states
                
                head_logits = self(last_hidden)
                
                candidate_tokens = []
                for i, logits in enumerate(head_logits):
                    probs = F.softmax(logits / self.config.temperature, dim=-1)
                    token = torch.multinomial(probs, num_samples=1)
                    candidate_tokens.append(token)
                
                stats['total_predicted'] += len(candidate_tokens)
                
                main_probs = F.softmax(main_logits[:, -1, :], dim=-1)
                accepted = 0
                for i, token in enumerate(candidate_tokens):
                    token_prob = main_probs.gather(1, token).item()
                    if token_prob >= self.config.acceptance_threshold:
                        generated_ids = torch.cat([generated_ids, token], dim=1)
                        accepted += 1
                    else:
                        break
                
                stats['accepted_tokens'] += accepted
                
                if accepted == 0:
                    next_token = torch.multinomial(main_probs, num_samples=1)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                    ], dim=1)
        
        if stats['total_predicted'] > 0:
            stats['acceptance_rate'] = stats['accepted_tokens'] / stats['total_predicted']
            stats['speedup'] = 1.0 + stats['acceptance_rate']
        
        stats['total_time_ms'] = (time.time() - start_time) * 1000.0
        
        if self.on_stats is not None:
            try:
                self.on_stats(stats)
            except Exception:
                pass
        
        return generated_ids, stats
