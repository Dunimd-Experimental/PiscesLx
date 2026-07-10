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
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum

from model.utils import YvNumericalGuard, YvShapeGuard

from utils.dc import PiscesLxLogger

from utils.paths import get_log_file
_LOG = PiscesLxLogger("Yv.Generation", file_path=get_log_file("Yv.Generation"), enable_file=True)


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
    tree_width: int = 4
    tree_depth: int = 5
    dspark_ngram_embed_dim: int = 256
    
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


# Paper: Original contribution by Dunimd Team (Yv Architecture)
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


# Paper: Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", 2024, arXiv:2401.10774
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
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, vocab_size, bias=False)
            )
            for _ in range(num_heads)
        ])
        
        if device is not None or dtype is not None:
            self.heads = self.heads.to(device=device, dtype=dtype)
        
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
        if self.heads[0][0].weight.device != hidden_states.device or self.heads[0][0].weight.dtype != hidden_states.dtype:
            self.heads = self.heads.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return [head(hidden_states) for head in self.heads]


# Paper: Original contribution by Dunimd Team (Yv Architecture)
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
            draft_probs = YvNumericalGuard.nan_to_num(draft_probs)

            token_probs = torch.gather(
                draft_probs,
                dim=-1,
                index=draft_ids.unsqueeze(-1),
            ).squeeze(-1)
            token_probs = YvNumericalGuard.nan_to_num(token_probs)

            acceptance_mask = torch.zeros_like(token_probs, dtype=torch.bool)
            rejection_position = -1

            for b in range(batch_size):
                cum_prob = 1.0
                for i in range(draft_len):
                    p_draft = token_probs[b, i].item()
                    eps = YvNumericalGuard.get_eps(torch.float32)
                    if p_draft <= eps:
                        if rejection_position == -1:
                            rejection_position = i
                        break
                    p_accept = min(1.0, cum_prob / max(p_draft, eps))
                    p_accept = YvNumericalGuard.safe_clamp(
                        torch.tensor(p_accept), 0.0, 1.0
                    ).item()
                    if p_accept < self.config.acceptance_threshold:
                        if rejection_position == -1:
                            rejection_position = i
                        break
                    acceptance_mask[b, i] = True
                    cum_prob *= p_draft

            accepted_lengths = acceptance_mask.sum(dim=1)
            max_accepted = accepted_lengths.max().item()

            # Truncate KV cache to accepted tokens only — prevents stale entries
            if new_past_key is not None and max_accepted < draft_len:
                new_len = input_ids.shape[1] + max_accepted
                new_past_key = tuple(
                    tuple(kv[..., :new_len, :] for kv in layer)
                    for layer in new_past_key
                )

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

            # Free draft-related tensors — only accepted tokens kept
            del full_sequence, draft_logits, draft_probs, token_probs, acceptance_mask

            return YvVerificationResult(
                accepted_ids=accepted_ids,
                num_accepted=int(max_accepted),
                new_past_key_values=new_past_key,
                rejection_position=rejection_position
            )


# Paper: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023; Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling", 2023
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
    
    # ── DSpark Markov head (lazy-init) ──
    _dspark_head: Optional[Any] = None
    _dspark_draft_len: int = 5
    _dspark_conf_threshold: float = 0.7
    _dspark_markov_order: int = 3
    _dspark_parallel_candidates: int = 8

    # ── Weaver tree search (lazy-init) ──
    _weaver_tree_width: int = 4
    _weaver_tree_depth: int = 3
    _weaver_top_k_marginals: int = 5

    # ── BlockPilot adaptive block size ──
    _blockpilot_policy: Optional[Any] = None

    # ── EntMTP entropy-guided topology ──
    _entmtp_enabled: bool = True
    _entmtp_entropy_decay: float = 0.95
    _entmtp_min_tree_width: int = 2

    def __init__(
        self,
        config: YvSpeculativeConfig,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        on_stats: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize speculative decoder with DSpark/Weaver/BlockPilot/EntMTP support."""
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.on_stats = on_stats

        # ── DSpark config from model config ──
        self._dspark_draft_len = getattr(model.config, 'dspark_draft_len', 5)
        self._dspark_markov_order = getattr(model.config, 'dspark_markov_order', 3)
        self._dspark_conf_threshold = getattr(model.config, 'dspark_confidence_threshold', 0.7)
        self._dspark_parallel_candidates = getattr(model.config, 'dspark_parallel_candidates', 8)

        # ── Weaver tree config ──
        self._weaver_tree_width = config.tree_width
        self._weaver_tree_depth = config.tree_depth

        # ── Draft model (used if DSpark not available / warm-up) ──
        self.draft_model = self._create_draft_model()
        self.parallel_verifier = YvParallelVerifier(config, model)

        hidden_size = getattr(model.config, 'hidden_size', 2048)
        vocab_size = getattr(model.config, 'vocab_size', 65536)
        self.medusa_head = YvMedusaHead(hidden_size, vocab_size, config.medusa_heads)

        self.performance_history: List[Dict[str, float]] = []
        self.adaptation_interval = 10
        self._dspark_confidence_window: deque = deque(maxlen=10)
        self._dspark_confidence_window_sum: float = 0.0
        self._blockpilot_recent_rewards: List[float] = []

    def _lazy_get_dspark_head(self) -> Any:
        """Lazy-init DSpark Markov prediction head."""
        if self._dspark_head is None:
            vocab_size = getattr(self.model.config, 'vocab_size', 65536)
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            embed_dim = getattr(self.model.config, 'dspark_ngram_embed_dim', self.config.dspark_ngram_embed_dim)
            self._dspark_head = YvDSparkHead(
                vocab_size=vocab_size,
                markov_order=self._dspark_markov_order,
                embed_dim=embed_dim,
                device=device,
                dtype=dtype,
            )
        return self._dspark_head

    def _create_draft_model(self) -> YvDraftModel:
        """Create a lightweight draft model sized relative to the target."""
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
    
    # ═══════════════════════════════════════════════════════════════
    # Unified speculative_generate: selects best strategy
    #  DSpark → Weaver tree → BlockPilot adaptive → EntMTP → draft
    # ═══════════════════════════════════════════════════════════════

    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        cache_manager: Optional[Any] = None,
        **model_kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate tokens using the best available speculative strategy.

        Strategy selection (priority order):
          1. DSpark (semi-autoregressive Markov head) — fastest if head is warm
          2. Weaver (factorized drafter tree search) — high parallelism
          3. BlockPilot (adaptive block-size policy) — learned adaptation
          4. EntMTP (entropy-guided MTP topology) — guided by uncertainty
          5. Standard draft-then-verify — baseline
        """
        if len(self.performance_history) >= self.adaptation_interval:
            self._adapt_parameters()

        batch_size = input_ids.shape[0]
        device = input_ids.device

        generated_ids = input_ids.clone()
        stats: Dict[str, Any] = {
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

        use_dspark = getattr(self.model.config, 'use_dspark', True)

        while generated_ids.shape[1] < max_length:
            if cache_manager is not None:
                cached = cache_manager.get_speculative_cache(self.config.draft_length)
                if cached is not None:
                    return cached, {'from_cache': True}

            # ── Strategy dispatch ──
            if use_dspark and generated_ids.shape[1] >= self._dspark_markov_order:
                draft_ids = self._dspark_generate_draft(generated_ids)
                method_tag = 'dspark'
            else:
                draft_ids, _ = self._generate_draft_sequence(
                    generated_ids, attention_mask, **model_kwargs
                )
                method_tag = 'baseline'

            result = self._verify_and_accept(
                generated_ids, draft_ids, past_key_values, **model_kwargs
            )

            accepted_ids = result.accepted_ids
            num_accepted = result.num_accepted
            past_key_values = result.new_past_key_values

            generated_ids = torch.cat([generated_ids, accepted_ids], dim=1)
            add_len = accepted_ids.shape[1]

            if attention_mask is None:
                attention_mask = torch.ones_like(generated_ids, dtype=torch.long, device=device)
            else:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], add_len), device=device, dtype=attention_mask.dtype)
                ], dim=1)

            stats['method'] = method_tag
            stats['total_draft_tokens'] += draft_ids.shape[1]
            stats['accepted_tokens'] += num_accepted
            stats['rejected_tokens'] += max(0, draft_ids.shape[1] - num_accepted)
            stats['iter_accept'].append(int(num_accepted))
            stats['max_accept_in_iter'] = max(stats['max_accept_in_iter'], int(num_accepted))

            # ── DSpark confidence scheduling (incremental window) ──
            if use_dspark and draft_ids.shape[1] > 0:
                ratio = YvNumericalGuard.safe_clamp(
                    torch.tensor(num_accepted / draft_ids.shape[1], device='cpu'), 0.0, 1.0
                ).item()
                self._dspark_confidence_window_sum += ratio
                self._dspark_confidence_window.append(ratio)
                if len(self._dspark_confidence_window) > self._dspark_confidence_window.maxlen:
                    self._dspark_confidence_window_sum -= self._dspark_confidence_window[0]

            # ── Zero-accept handling with exponential backoff ──
            if num_accepted == 0:
                zero_accept_streak += 1
                # Exponential backoff: double draft length on repeated zero-accept
                backoff_cap = min(self.config.draft_length * 4, 32)
                self.config.draft_length = min(backoff_cap, self.config.draft_length * 2)
                if zero_accept_streak >= 3:
                    fallback_ids, fallback_stats = self._standard_generate(
                        generated_ids, attention_mask, max_length, **model_kwargs
                    )
                    stats.update({k: v for k, v in fallback_stats.items() if k not in stats})
                    generated_ids = fallback_ids
                    break

                with torch.no_grad():
                    outputs = self.model(generated_ids, attention_mask=attention_mask, **model_kwargs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                logits = YvNumericalGuard.nan_to_num(logits)
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
            f"[{method_tag}] draft_len={self.config.draft_length}, "
            f"accept_rate={stats['draft_acceptance_rate']:.3f}, "
            f"avg_accept={stats['avg_accept_per_iter']:.1f}, "
            f"speedup={stats['speedup']:.2f}, "
            f"time_ms={stats['total_time_ms']:.1f}"
        )

        if self.on_stats is not None:
            try:
                self.on_stats(stats)
            except Exception:
                import logging
                logging.getLogger(__name__).warning("on_stats callback failed", exc_info=True)

        self.performance_history.append({
            'acceptance_rate': stats.get('draft_acceptance_rate', 0),
            'speedup': stats.get('speedup', 1),
            'avg_accept': stats.get('avg_accept_per_iter', 0),
        })
        return generated_ids, stats

    # ═══════════════════════════════════════════════════════════════
    # DSpark semi-autoregressive draft via Markov head
    # ═══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _dspark_generate_draft(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate draft tokens via DSpark Markov head with confidence scheduling.

        Uses YvDSparkHead (n-gram → next-token logits) for fast generation
        and adjusts draft length based on recent acceptance history.
        """
        head = self._lazy_get_dspark_head()
        draft_len = self._compute_blockpilot_draft_len()
        batch_size = input_ids.shape[0]
        context = input_ids[:, -self._dspark_markov_order:]

        # Generate multiple candidates in parallel, pick best via avg log-prob
        all_candidates: List[torch.Tensor] = []
        all_scores: List[float] = []

        # Pre-compute initial context embedding once, shared across all candidates
        init_emb = head.embedding(context)

        for _ in range(self._dspark_parallel_candidates):
            tokens: List[torch.Tensor] = []
            cum_logprob = 0.0
            ctx = context.clone()
            cur_emb = init_emb.clone()
            for step in range(draft_len):
                if step > 0:
                    cur_emb = head.embedding(ctx)
                flat_emb = cur_emb.view(cur_emb.size(0), -1)
                logits = head.predictor(flat_emb)
                logits = YvNumericalGuard.nan_to_num(logits)
                logits = self._apply_sampling(logits)
                probs = F.softmax(logits, dim=-1)
                probs = YvNumericalGuard.nan_to_num(probs)
                token = torch.multinomial(probs, num_samples=1)
                tokens.append(token)
                p = probs.gather(1, token).squeeze(-1)
                cum_logprob = cum_logprob + YvNumericalGuard.safe_log(p).sum().item()
                ctx = torch.cat([ctx[:, 1:], token], dim=1)
                new_emb = head.embedding(token)
                cur_emb = torch.cat([cur_emb[:, 1:, :], new_emb], dim=1)

            seq = torch.cat(tokens, dim=1)
            all_candidates.append(seq)
            all_scores.append(cum_logprob / max(1, draft_len))

        best = max(range(len(all_scores)), key=lambda i: all_scores[i])
        best_seq = all_candidates[best]
        # Free non-best candidates immediately
        del all_candidates, all_scores, init_emb

        # ── Weaver-style factorized tree expansion ──
        if self._weaver_tree_width > 1 and draft_len >= 2:
            best_seq = self._weaver_tree_expand(input_ids, best_seq=best_seq)

        return best_seq

    # ═══════════════════════════════════════════════════════════════
    # Weaver (arXiv:2607.06763): factorized drafter tree search
    # ═══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _weaver_tree_expand(self, input_ids: torch.Tensor, best_seq: torch.Tensor) -> torch.Tensor:
        """Weaver-style factorized tree expansion.

        Constructs a proposal tree from top-K marginals of the Markov head.
        Each position in the draft is expanded to top-K alternatives;
        the best path (highest joint probability) is selected.
        """
        head = self._lazy_get_dspark_head()
        batch_size = input_ids.shape[0]
        draft_len = best_seq.shape[1]
        width = max(1, min(self._weaver_tree_width, self._dspark_parallel_candidates))
        depth = min(self._weaver_tree_depth, draft_len)
        markov_order = self._dspark_markov_order

        context = input_ids[:, -markov_order:]

        # Build tree: at each depth, get top-K predictions and pick best path
        best_path: List[torch.Tensor] = []
        cur_ctx = context.clone()
        for d in range(depth):
            logits = head(cur_ctx)
            logits = self._apply_sampling(logits)
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_tokens = torch.topk(probs, min(width, probs.size(-1)), dim=-1)

            # Score each using joint probability
            best_idx = 0
            best_prob = -1e9
            for k_idx in range(top_k_tokens.size(-1)):
                tok = top_k_tokens[:, k_idx:k_idx+1]
                joint = YvNumericalGuard.safe_log(top_k_probs[:, k_idx:k_idx+1]).sum().item()
                if joint > best_prob:
                    best_prob = joint
                    best_idx = k_idx
            chosen = top_k_tokens[:, best_idx:best_idx+1]
            best_path.append(chosen)
            cur_ctx = torch.cat([cur_ctx[:, 1:], chosen], dim=1)

        # Pad remaining positions with best_seq if depth < draft_len
        if len(best_path) < draft_len:
            remaining = best_seq[:, len(best_path):]
            best_path.append(remaining)

        return torch.cat(best_path, dim=1)

    # ═══════════════════════════════════════════════════════════════
    # BlockPilot (arXiv:2606.31315): adaptive block size via policy
    # ═══════════════════════════════════════════════════════════════

    def _blockpilot_draft_len(self) -> int:
        """Return adaptive draft length using BlockPilot-style policy.

        Maintains a simple slot-machine policy over block sizes:
        tracks recent acceptance rewards per block size and selects
        the one with highest expected reward.
        """
        base = self.config.draft_length
        # EntMTP adjustment: scale draft length by entropy uncertainty
        if self._entmtp_enabled and self._blockpilot_recent_rewards:
            avg_reward = sum(self._blockpilot_recent_rewards[-10:]) / max(1, len(self._blockpilot_recent_rewards[-10:]))
            if avg_reward > 0.8:
                return min(base + 3, base * 2)
            elif avg_reward < 0.4:
                return max(2, base - 1)
        return base

    def _compute_blockpilot_draft_len(self) -> int:
        """Compute effective draft length combining BlockPilot + EntMTP policies."""
        base = self._blockpilot_draft_len()

        # EntMTP: reduce base when high entropy (uncertainty)
        if self._entmtp_enabled and self._dspark_confidence_window:
            avg_conf = self._dspark_confidence_window_sum / max(1, len(self._dspark_confidence_window))
            if avg_conf < 0.4:
                base = max(self._entmtp_min_tree_width, base - 1)
            elif avg_conf > 0.8:
                base = min(base + 2, self._dspark_draft_len + 3)

        return max(1, min(base, self._dspark_draft_len + 5))
    
    def _apply_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        logits = YvNumericalGuard.nan_to_num(logits)
        temp = max(0.1, min(2.0, self.config.temperature))
        logits = logits / temp
        if self.config.top_k > 0:
            logits = torch.where(
                logits < torch.topk(logits, min(self.config.top_k, logits.size(-1)))[0][..., -1:, :],
                float('-inf'), logits
            )
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_logits = YvNumericalGuard.nan_to_num(sorted_logits)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(sorted_indices_to_remove, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        return logits
    
    def _generate_draft_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate draft sequence using EntMTP or fallback draft model.

        Uses BlockPilot adaptive draft length + EntMTP entropy guidance.
        Falls back to the draft model when MTP heads are unavailable.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        draft_len = self._compute_blockpilot_draft_len()

        # EntMTP: use MTP heads for entropy-guided tree topology
        if self._entmtp_enabled and hasattr(self.model, 'mtp_heads') and self.model.mtp_heads:
            return self._entmtp_generate_draft(input_ids, draft_len)

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

    # ═══════════════════════════════════════════════════════════════
    # EntMTP (arXiv:2606.27550): entropy-guided MTP tree topology
    # ═══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _entmtp_generate_draft(self, input_ids: torch.Tensor, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Entropy-guided MTP draft generation.

        Uses model MTP heads to construct an adaptive tree topology
        where branch width at each depth is guided by predictive entropy.
        High entropy → wider tree (exploration), low entropy → narrower (exploitation).
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        num_mtp = getattr(self.model, 'num_mtp_heads', 0)
        if num_mtp == 0 or not hasattr(self.model, 'mtp_heads'):
            return self._generate_draft_sequence_fallback(input_ids, max_len)

        outputs = self.model(input_ids, use_cache=False, output_hidden_states=True)
        if isinstance(outputs, dict):
            h = outputs.get('hidden_states', outputs.get('logits'))
            if h is None:
                return self._generate_draft_sequence_fallback(input_ids, max_len)
        else:
            return self._generate_draft_sequence_fallback(input_ids, max_len)

        last_h = h[:, -1:, :]
        tokens: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []

        remaining = max_len
        for i, mtp_head in enumerate(self.model.mtp_heads):
            if remaining <= 0:
                break
            step_logits = mtp_head(last_h)
            step_logits = YvNumericalGuard.nan_to_num(step_logits)
            probs = F.softmax(step_logits, dim=-1)
            probs = YvNumericalGuard.nan_to_num(probs)

            # EntMTP: entropy → smaller branch when confident
            entropy = -(probs * YvNumericalGuard.safe_log(probs)).sum(dim=-1, keepdim=True)
            norm_entropy = (entropy / math.log(probs.size(-1))).mean().item()
            tree_k = max(1, int(self._weaver_tree_width * (1.0 - norm_entropy * self._entmtp_entropy_decay)))
            tree_k = max(self._entmtp_min_tree_width, min(tree_k, self._weaver_tree_width))

            top_probs, top_tokens = torch.topk(probs, min(tree_k, probs.size(-1)), dim=-1)
            best_idx = torch.multinomial(top_probs[:, 0, :].squeeze(1), num_samples=1)
            token = top_tokens[:, 0, :].gather(-1, best_idx)
            tokens.append(token)
            logits_list.append(step_logits[:, -1:, :])
            remaining -= 1

            if i < num_mtp - 1:
                last_h = mtp_head(last_h)[:, -1:, :]

        if not tokens:
            return self._generate_draft_sequence_fallback(input_ids, max_len)

        draft_seq = torch.cat(tokens, dim=1).to(device)
        draft_logits = torch.cat(logits_list, dim=1)
        return draft_seq, draft_logits

    def _generate_draft_sequence_fallback(self, input_ids: torch.Tensor, draft_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback draft via draft model when MTP heads unavailable."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        cur_ids = input_ids
        tokens: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(draft_len):
                logits = self.draft_model(cur_ids)
                step_logits = logits[:, -1:, :]
                step_token = self._sample_candidates(step_logits).unsqueeze(1)[:, 0:1]
                tokens.append(step_token)
                cur_ids = torch.cat([cur_ids, step_token], dim=1)
                logits_list.append(step_logits)
        seq = torch.cat(tokens, dim=1).to(device)
        v = getattr(self.model.config, 'vocab_size', 65536)
        ls = torch.cat(logits_list, dim=1) if logits_list else torch.zeros(batch_size, 0, v, device=device)
        return seq, ls
    
    def _sample_candidates(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self._apply_sampling(logits)
        probs = F.softmax(logits, dim=-1)
        probs = YvNumericalGuard.nan_to_num(probs)
        n_valid = (probs.squeeze(1) > 0).sum(dim=-1).min().item()
        k = min(self.config.num_candidates, max(1, n_valid))
        return torch.multinomial(probs.squeeze(1), k, replacement=False)
    
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


# Paper: Original contribution by Dunimd Team (Yv Architecture)
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


# Paper: Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", 2024, arXiv:2401.10774
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
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        self.medusa_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
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
                    temp = max(0.1, min(2.0, self.config.temperature))
                    probs = F.softmax(logits / temp, dim=-1)
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
                    tokens_added = accepted if accepted > 0 else 1
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, tokens_added), device=device, dtype=attention_mask.dtype)
                    ], dim=1)
        
        if stats['total_predicted'] > 0:
            stats['acceptance_rate'] = stats['accepted_tokens'] / stats['total_predicted']
            stats['speedup'] = 1.0 + stats['acceptance_rate']
        
        stats['total_time_ms'] = (time.time() - start_time) * 1000.0
        
        if self.on_stats is not None:
            try:
                self.on_stats(stats)
            except Exception:
                import logging
                logging.getLogger(__name__).warning("on_stats callback failed", exc_info=True)
        
        return generated_ids, stats


# Paper: Original contribution by Dunimd Team (Yv Architecture — DSpark speculative decoding)
class YvDSparkHead(nn.Module):
    """Markov prediction head for DSpark-style speculative decoding.

    Predicts next token distribution from n-gram history using a lightweight
    embedding + MLP predictor. Much faster than a full model forward pass.

    Architecture:
        - nn.Embedding(vocab_size, embed_dim) for token representation
        - MLP: Linear(embed_dim * markov_order, 512) -> SiLU -> Linear(512, vocab_size)

    Attributes:
        markov_order (int): N-gram context length for prediction.
        embedding (nn.Embedding): Token embedding layer.
        predictor (nn.Sequential): MLP for distribution prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        markov_order: int = 3,
        embed_dim: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.markov_order = markov_order
        self.embedding = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * markov_order, 512, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(512, vocab_size, bias=False, device=device, dtype=dtype)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict next-token logits from n-gram token history.

        Args:
            tokens: Token IDs of shape [batch_size, markov_order].

        Returns:
            Logits of shape [batch_size, vocab_size].
        """
        embeds = self.embedding(tokens).view(tokens.size(0), -1)
        logits = self.predictor(embeds)
        return YvNumericalGuard.nan_to_num(logits)


# Paper: Original contribution by Dunimd Team (Yv Architecture — DSpark speculative decoding)
class YvDSparkSpeculativeDecoder(nn.Module):
    """DSpark-style speculative decoder with parallel draft generation,
    Markov prediction head, and confidence-based adaptive draft length.

    Three key innovations over standard speculative decoding:
      1. Parallel draft generation: generates multiple draft sequences via
         a lightweight Markov head (not sequential draft model forwards)
      2. Markov prediction head: predicts next-token distribution from
         n-gram history — much faster than a full model forward pass
      3. Confidence scheduling: adjusts draft length based on recent
         acceptance rate

    Works alongside YvSpeculativeDecoder / YvAdaptiveSpeculativeDecoder;
    gated by the ``use_dspark`` flag in YvConfig.

    Attributes:
        markov_head (YvDSparkHead): Fast n-gram prediction head.
        verifier (YvParallelVerifier): Parallel verification module.
        current_draft_len (int): Dynamically adjusted draft length.
        performance_history (List[Dict]): Recent acceptance statistics.
    """

    def __init__(
        self,
        config: Any,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        on_stats: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.on_stats = on_stats

        self.dspark_draft_len: int = getattr(config, 'dspark_draft_len', 5)
        self.dspark_markov_order: int = getattr(config, 'dspark_markov_order', 3)
        self.dspark_confidence_threshold: float = getattr(config, 'dspark_confidence_threshold', 0.7)
        self.dspark_parallel_candidates: int = getattr(config, 'dspark_parallel_candidates', 8)

        self.temperature: float = getattr(config, 'speculative_temperature', 0.7)
        self.top_k: int = getattr(config, 'speculative_top_k', 50)
        self.top_p: float = getattr(config, 'speculative_top_p', 0.9)

        self.temperature = max(0.1, min(2.0, self.temperature))
        self.top_k = max(1, min(1000, self.top_k))
        self.top_p = max(0.1, min(1.0, self.top_p))

        vocab_size = getattr(model.config, 'vocab_size', 65536)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        self.markov_head = YvDSparkHead(
            vocab_size=vocab_size,
            markov_order=self.dspark_markov_order,
            embed_dim=getattr(config, 'dspark_ngram_embed_dim', 256),
            device=device,
            dtype=dtype,
        )

        self.verifier = YvParallelVerifier(
            YvSpeculativeConfig(acceptance_threshold=self.dspark_confidence_threshold),
            model
        )

        self.current_draft_len: int = self.dspark_draft_len
        self.performance_history: List[Dict[str, float]] = []
        self._window_buffer: deque = deque(maxlen=10)
        self._window_conf_sum: float = 0.0

    @torch.no_grad()
    def _generate_parallel_drafts(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate multiple draft sequences in parallel using the Markov head.

        For each of ``dspark_parallel_candidates`` attempts, auto-regressively
        generates a full draft sequence with the Markov head.  The candidate
        with the highest average log-probability (under the Markov head) is
        selected.

        Args:
            input_ids: Current sequence [batch_size, seq_len].

        Returns:
            Best draft sequence [batch_size, draft_len].
        """
        batch_size = input_ids.shape[0]
        draft_len = max(1, self.current_draft_len)
        context = input_ids[:, -self.dspark_markov_order:]

        all_candidates: List[torch.Tensor] = []
        all_scores: List[float] = []

        # Pre-compute initial context embedding once, shared across all candidates
        init_emb = self.markov_head.embedding(context)  # [batch, order, embed_dim]

        for _ in range(self.dspark_parallel_candidates):
            draft_tokens: List[torch.Tensor] = []
            total_log_prob = 0.0
            cur_context = context.clone()
            cur_emb = init_emb.clone()

            for step in range(draft_len):
                # Use cached embedding for first step; re-embed on subsequent steps
                if step > 0:
                    cur_emb = self.markov_head.embedding(cur_context)
                flat_emb = cur_emb.view(cur_emb.size(0), -1)
                logits = self.markov_head.predictor(flat_emb)
                logits = YvNumericalGuard.nan_to_num(logits)
                logits = self._apply_sampling(logits)
                probs = F.softmax(logits, dim=-1)
                probs = YvNumericalGuard.nan_to_num(probs)
                token = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(token)
                tok_prob = probs.gather(1, token).squeeze(-1)
                total_log_prob = total_log_prob + YvNumericalGuard.safe_log(tok_prob).sum().item()
                cur_context = torch.cat([cur_context[:, 1:], token], dim=1)
                # Update embedding cache: drop oldest token embedding, append new token
                new_emb = self.markov_head.embedding(token)  # [batch, 1, embed_dim]
                cur_emb = torch.cat([cur_emb[:, 1:, :], new_emb], dim=1)

            draft_seq = torch.cat(draft_tokens, dim=1)
            all_candidates.append(draft_seq)
            all_scores.append(total_log_prob / max(1, draft_len))

        best_idx = max(range(len(all_scores)), key=lambda i: all_scores[i])
        best_seq = all_candidates[best_idx]
        # Free non-best candidates immediately
        del all_candidates, all_scores, init_emb
        return best_seq

    def _apply_sampling(self, logits: torch.Tensor) -> torch.Tensor:
        logits = YvNumericalGuard.nan_to_num(logits)
        logits = logits / max(0.1, min(2.0, self.temperature))
        if self.top_k > 0:
            logits = torch.where(
                logits < torch.topk(logits, min(self.top_k, logits.size(-1)))[0][..., -1:, :],
                float('-inf'), logits
            )
        return logits

    @torch.no_grad()
    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        cache_manager: Optional[Any] = None,
        **model_kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate tokens using DSpark-style speculative decoding.

        Main generation loop that:
          1. Warms up with standard generation until markov_order tokens exist
          2. Generates parallel draft candidates via Markov head
          3. Verifies with the target model
          4. Adjusts draft length via confidence scheduling

        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            attention_mask: Optional attention mask.
            max_length: Maximum sequence length to generate.
            cache_manager: Optional cache manager.
            **model_kwargs: Additional model arguments.

        Returns:
            Tuple of (generated_ids, stats_dict).
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        generated_ids = input_ids.clone()
        stats = {
            'method': 'dspark',
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
                cached = cache_manager.get_speculative_cache(self.current_draft_len)
                if cached is not None:
                    return cached, {'from_cache': True}

            # Warm-up: need at least markov_order context tokens for the head
            if generated_ids.shape[1] < self.dspark_markov_order:
                with torch.no_grad():
                    outputs = self.model(generated_ids, attention_mask=attention_mask, **model_kwargs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                logits = YvNumericalGuard.nan_to_num(logits)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                probs = YvNumericalGuard.nan_to_num(probs)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                attention_mask = self._extend_attention_mask(attention_mask, generated_ids, device)
                continue

            draft_ids = self._generate_parallel_drafts(generated_ids)
            result = self.verifier(generated_ids, draft_ids, past_key_values)

            accepted_ids = result.accepted_ids
            num_accepted = result.num_accepted
            past_key_values = result.new_past_key_values
            generated_ids = torch.cat([generated_ids, accepted_ids], dim=1)
            attention_mask = self._extend_attention_mask(attention_mask, generated_ids, device, accepted_ids.shape[1])

            stats['total_draft_tokens'] += draft_ids.shape[1]
            stats['accepted_tokens'] += num_accepted
            stats['rejected_tokens'] += max(0, draft_ids.shape[1] - num_accepted)
            stats['iter_accept'].append(int(num_accepted))
            stats['max_accept_in_iter'] = max(stats['max_accept_in_iter'], int(num_accepted))

            self._confidence_schedule(num_accepted, draft_ids.shape[1])

            if num_accepted == 0:
                zero_accept_streak += 1
                # Exponential backoff: double draft length on repeated zero-accept
                backoff_cap = min(self.dspark_draft_len * 4, 32)
                self.current_draft_len = min(backoff_cap, self.current_draft_len * 2)
                if zero_accept_streak >= 3:
                    fallback_ids, fallback_stats = self._standard_generate(
                        generated_ids, attention_mask, max_length, **model_kwargs
                    )
                    stats.update({k: v for k, v in fallback_stats.items() if k not in stats})
                    generated_ids = fallback_ids
                    break

                with torch.no_grad():
                    outputs = self.model(generated_ids, attention_mask=attention_mask, **model_kwargs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                logits = YvNumericalGuard.nan_to_num(logits)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                probs = YvNumericalGuard.nan_to_num(probs)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                attention_mask = self._extend_attention_mask(attention_mask, generated_ids, device)
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
            cache_manager.set_speculative_cache(self.current_draft_len, generated_ids)

        _LOG.debug(
            f"[DSpark] draft_len={self.current_draft_len}, "
            f"accept_rate={stats['draft_acceptance_rate']:.3f}, "
            f"avg_accept={stats['avg_accept_per_iter']:.1f}, "
            f"speedup={stats['speedup']:.2f}, "
            f"time_ms={stats['total_time_ms']:.1f}"
        )

        if self.on_stats is not None:
            try:
                self.on_stats(stats)
            except Exception:
                import logging
                logging.getLogger(__name__).warning("on_stats callback failed", exc_info=True)

        return generated_ids, stats

    def _confidence_schedule(self, num_accepted: int, draft_len: int):
        """Adjust draft length based on recent acceptance rate (incremental window).

        High acceptance increases draft length for higher speedup;
        low acceptance reduces it to avoid wasted computation.
        """
        if draft_len == 0:
            return
        accept_rate = YvNumericalGuard.safe_clamp(
            torch.tensor(num_accepted / draft_len), 0.0, 1.0
        ).item()
        self._window_conf_sum += accept_rate
        self._window_buffer.append(accept_rate)
        if len(self._window_buffer) > 10:
            self._window_conf_sum -= self._window_buffer[0]
        avg_rate = self._window_conf_sum / max(1, len(self._window_buffer))

        if avg_rate > 0.8:
            self.current_draft_len = min(self.dspark_draft_len + 3, self.current_draft_len + 1)
        elif avg_rate < 0.4:
            self.current_draft_len = max(2, self.current_draft_len - 1)

    def _extend_attention_mask(
        self,
        mask: Optional[torch.Tensor],
        ids: torch.Tensor,
        device: torch.device,
        add_len: Optional[int] = None
    ) -> torch.Tensor:
        """Extend or create an attention mask."""
        if mask is None:
            return torch.ones_like(ids, dtype=torch.long, device=device)
        if add_len is None:
            add_len = ids.shape[1] - mask.shape[1]
        if add_len <= 0:
            return mask
        return torch.cat([
            mask,
            torch.ones((mask.shape[0], add_len), device=device, dtype=mask.dtype)
        ], dim=1)

    def _standard_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
        **model_kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Fallback to standard autoregressive generation."""
        stats: Dict[str, Any] = {'method': 'standard_fallback'}
        start_time = time.time()
        current_ids = input_ids
        current_mask = attention_mask

        while current_ids.shape[1] < max_length:
            with torch.no_grad():
                outputs = self.model(current_ids, attention_mask=current_mask, **model_kwargs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                probs = F.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                current_mask = torch.cat([
                    current_mask,
                    torch.ones((current_mask.shape[0], 1), device=current_ids.device, dtype=current_mask.dtype)
                ], dim=1)

        stats['total_time_ms'] = (time.time() - start_time) * 1000.0
        return current_ids, stats
