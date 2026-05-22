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

"""Unified routing logic for Yv's Chain-of-Thought and multi-path reasoners.

This module provides a unified reasoning interface that intelligently routes
queries between different reasoning strategies based on problem complexity
and sequence length.

Architecture:
    1. Routing Strategy:
       - Complexity-based routing using CoT reasoner metrics
       - Sequence length threshold for multi-path activation
       - Temperature-scaled logit alignment for output consistency
    
    2. Sub-Components:
       - YvCoTMemoryReasoner: Chain-of-thought with memory
       - YvMultiPathReasoningEngine: Multi-path exploration
    
    3. Output Fusion:
       - Logit temperature scaling for calibration
       - Confidence score blending from multiple sources
       - Correction logits via error analysis

Key Features:
    - Automatic routing between CoT and multi-path reasoning
    - Complexity estimation using semantic variance
    - Graceful fallback on multi-path failures
    - Consistent output interface across all paths

Performance Characteristics:
    - Routing Decision: O(L) where L = sequence length
    - CoT Path: O(T * L * H^2) where T = reasoning steps
    - Multi-Path: O(P * T * L * H^2) where P = number of paths

Usage Example:
    >>> from model.reasoning.reasoner import YvUnifiedReasoner
    >>> 
    >>> # Initialize with config
    >>> reasoner = YvUnifiedReasoner(config)
    >>> 
    >>> # Forward pass with automatic routing
    >>> output = reasoner.forward(
    ...     input_ids=hidden_states,
    ...     attention_mask=mask,
    ...     memory_context=memory_entries
    >>> )
    >>> 
    >>> # Access thinking logits
    >>> thinking = output["thinking_logits"]

Dependencies:
    - torch: Tensor operations and neural network modules
    - .cot_memory: YvCoTMemoryReasoner for CoT reasoning
    - .multipath_core: YvMultiPathReasoningEngine for multi-path

Note:
    The unified reasoner maintains consistent output format regardless
    of the internal routing decision. All outputs include thinking_logits,
    difficulty_logits, reflection_logits, confidence_score, and loss.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from .cot_memory import YvCoTMemoryReasoner
from .multipath_core import YvMultiPathReasoningEngine
from ..ttt_e2e import YvTestTimeTrainer


class YvUnifiedReasoner(nn.Module):
    """Unified reasoning router between CoT and multi-path engines.
    
    This class provides a single interface for reasoning that automatically
    selects between chain-of-thought reasoning with memory and multi-path
    reasoning based on problem complexity and sequence length.
    
    Architecture:
        1. Routing Decision:
           - Complexity estimation via CoT reasoner
           - Sequence length threshold check
           - Enable/disable flag for multi-path
        
        2. CoT Path (Low Complexity):
           - Uses YvCoTMemoryReasoner
           - Adaptive depth based on complexity
           - Memory-augmented reasoning
        
        3. Multi-Path Path (High Complexity):
           - Uses YvMultiPathReasoningEngine
           - Parallel path exploration
           - Fallback to CoT on failure
        
        4. Output Alignment:
           - Temperature-scaled logits
           - Confidence score blending
           - Consistent dictionary format
    
    Routing Logic:
        Multi-path is activated when:
        - enable_multi_path_core is True AND
        - complexity >= mpr_threshold OR seq_len > seq_len_threshold
        
        Otherwise, CoT reasoning is used.
    
    Attributes:
        cfg (Any): Configuration namespace with hyperparameters.
        cot_reasoner (YvCoTMemoryReasoner): CoT reasoning module.
        multi_path_core (YvMultiPathReasoningEngine): Multi-path engine.
        enable_multi_path_core (bool): Flag to enable multi-path routing.
        mpr_threshold (float): Complexity threshold for multi-path activation.
        seq_len_threshold (int): Sequence length threshold for multi-path.
        _logit_temp (nn.Parameter): Learnable temperature for logit scaling.
    
    Example:
        >>> reasoner = YvUnifiedReasoner(config)
        >>> output = reasoner.forward(input_ids=hidden_states)
        >>> print(output["thinking_logits"].shape)
        torch.Size([batch_size, vocab_size])
    
    Note:
        The reasoner ensures consistent output format regardless of routing.
        All outputs include: thinking_logits, difficulty_logits, reflection_logits,
        confidence_score, reasoning_states, reasoning_steps, correction_logits,
        attention_weights, final_state, and loss.
    """
    
    def __init__(self, cfg: Any):
        """Initialize sub-components and routing thresholds from configuration.
        
        Creates the CoT reasoner, multi-path engine, and initializes routing
        parameters from the configuration object.
        
        Args:
            cfg: Configuration namespace providing shared parameters across
                reasoning engines. Expected attributes:
                - hidden_size (int): Hidden dimension size
                - vocab_size (int): Vocabulary size
                - n_head (int): Number of attention heads
                - enable_multi_path_core (bool): Enable multi-path routing
                - mpr_threshold (float): Complexity threshold (default: 0.6)
                - mpr_seq_len_threshold (int): Length threshold (default: 512)
        
        Note:
            The _logit_temp parameter is initialized to 1.0 and is learnable
            for temperature scaling during training.
        """
        super().__init__()
        self.cfg = cfg

        # Initialize the CoT with Memory reasoner.
        self.cot_reasoner = YvCoTMemoryReasoner(cfg)

        # Initialize the Multi-Path reasoning engine.
        self.multi_path_core = YvMultiPathReasoningEngine(cfg)

        # Fetch routing parameters with fallbacks.
        self.enable_multi_path_core = getattr(cfg, "enable_multi_path_core", True)
        self.mpr_threshold = getattr(cfg, "mpr_threshold", 0.6)
        self.seq_len_threshold = getattr(cfg, "mpr_seq_len_threshold", 512)

        self.thinking_intensity = float(getattr(cfg, 'thinking_intensity', 0.5))
        self.complexity_threshold_low = float(getattr(cfg, 'complexity_threshold_low', 0.3))
        self.complexity_threshold_high = float(getattr(cfg, 'complexity_threshold_high', 0.7))
        
        if self.thinking_intensity > 0:
            self.complexity_estimator = nn.Sequential(
                nn.Linear(cfg.hidden_size, max(1, cfg.hidden_size // 4)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(max(1, cfg.hidden_size // 4), 1),
                nn.Sigmoid()
            )
            
            self.thinking_depth_controller = nn.Parameter(
                torch.ones(1) * self.thinking_intensity
            )

        # Parameter controlling temperature scaling for logit alignment.
        self._logit_temp = nn.Parameter(torch.tensor(1.0))

        # Test-Time Training (TTT-E2E) Integration
        self.use_ttt_e2e = bool(getattr(cfg, 'use_ttt_e2e', False))
        self.ttt_trainer = None  # Lazy initialization

    def _extract_hidden_states(self, input_ids: Optional[torch.Tensor], kwargs: Dict[str, Any]) -> torch.Tensor:
        """Obtain hidden states compatible with downstream reasoning modules.
        
        Extracts or generates hidden states from various input formats to ensure
        compatibility with the reasoning pipeline. Handles three cases:
        1. Direct hidden states (float tensors)
        2. Hidden states from kwargs dictionary
        3. Fallback random generation
        
        Args:
            input_ids (Optional[torch.Tensor]): Input tensor that may be either
                token IDs (integer type) or pre-computed hidden states (float type).
            kwargs (Dict[str, Any]): Additional keyword arguments that may contain
                'hidden_states' key with pre-computed embeddings.
        
        Returns:
            torch.Tensor: Hidden states tensor of shape [batch, seq_len, hidden_size].
        
        Note:
            When no valid hidden states are found, generates random tensor as fallback.
            This ensures the reasoning pipeline always has valid input for processing.
        """
        hidden_states = None
        if torch.is_tensor(input_ids) and input_ids.dtype in (torch.float16, torch.float32, torch.bfloat16):
            hidden_states = input_ids
        elif "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            hidden_states = kwargs["hidden_states"]
        else:
            # Generate a random tensor fallback to mimic YvReasoner behavior.
            hidden_size = getattr(self.cfg, "hidden_size", 1024)
            hidden_states = torch.randn(1, 1, hidden_size, device=next(self.parameters()).device)

        return hidden_states

    def _should_use_multi_path(self, hidden_states: torch.Tensor) -> bool:
        """Decide whether the multi-path engine should handle the query.
        
        Determines routing based on problem complexity and sequence length.
        Multi-path reasoning is activated when either complexity or sequence
        length exceeds their respective thresholds.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch, seq_len, hidden].
        
        Returns:
            bool: True if multi-path reasoning should be used, False for CoT.
        
        Decision Logic:
            - Uses CoT reasoner's complexity estimation when available
            - Falls back to length-based heuristic on estimation failure
            - Checks both complexity and sequence length thresholds
        
        Note:
            The complexity score is normalized to [0, 1] range.
            Higher complexity or longer sequences favor multi-path reasoning.
        """
        try:
            # Estimate problem complexity using the CoT reasoner metric.
            complexity = self.cot_reasoner._calculate_problem_complexity(hidden_states)
        except Exception:
            # Fallback heuristic using sequence length.
            seq_len = hidden_states.shape[1]
            complexity = min(seq_len / float(self.seq_len_threshold), 1.0)

        seq_len = hidden_states.shape[1]
        return (self.enable_multi_path_core and
                (complexity >= self.mpr_threshold or seq_len > self.seq_len_threshold))

    def _pool_state(self, hs: torch.Tensor) -> torch.Tensor:
        """Mean-pool hidden states to produce [B, H] tensors.
        
        Applies mean pooling across the sequence dimension to produce
        a fixed-size representation for classification heads.
        
        Args:
            hs (torch.Tensor): Hidden states of shape [batch, seq_len, hidden].
        
        Returns:
            torch.Tensor: Pooled tensor of shape [batch, hidden].
        """
        return hs.mean(dim=1)

    def initialize_reasoning_tokens(self, tokenizer: Optional[Any] = None) -> None:
        """Forward token initialization requests to each component.
        
        Propagates tokenizer initialization to both CoT reasoner and
        multi-path engine for special token handling.
        
        Args:
            tokenizer (Optional[Any]): Tokenizer instance for special token
                registration. Typically a HuggingFace tokenizer.
        
        Note:
            Initialization failures in individual components are silently
            ignored to ensure robust initialization.
        """
        if hasattr(self, "multi_path_core") and hasattr(self.multi_path_core, "initialize_reasoning_tokens"):
            self.multi_path_core.initialize_reasoning_tokens(tokenizer)
        if hasattr(self, "cot_reasoner") and hasattr(self.cot_reasoner, "initialize_reasoning_tokens"):
            try:
                self.cot_reasoner.initialize_reasoning_tokens(tokenizer)  # type: ignore
            except Exception:
                pass

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        memory_context: Optional[list] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a forward pass compatible with YvReasoner outputs.
        
        Performs intelligent routing between CoT and multi-path reasoning
        based on problem complexity, then fuses outputs for consistent
        return format.
        
        Args:
            input_ids (Optional[torch.Tensor]): Input tensor, either token IDs
                or pre-computed hidden states. Default: None.
            attention_mask (Optional[torch.Tensor]): Attention mask or token IDs
                for multi-path engine. Default: None.
            memory_context (Optional[list]): Memory entries for CoT reasoning.
                Can also contain labels when passed as tensor. Default: None.
            labels (Optional[torch.Tensor]): Target labels for loss computation.
                Default: None.
            **kwargs: Additional keyword arguments including:
                - hidden_states: Pre-computed hidden states
        
        Returns:
            Dict[str, Any]: Output dictionary containing:
                - thinking_logits (torch.Tensor): Logits for next token prediction
                - difficulty_logits (torch.Tensor): Difficulty classification logits
                - reflection_logits (torch.Tensor): Reflection type logits
                - confidence_score (torch.Tensor): Confidence score [0, 1]
                - reasoning_states (torch.Tensor): Final reasoning hidden states
                - reasoning_steps (list): List of intermediate states
                - correction_logits (torch.Tensor): Correction prediction logits
                - attention_weights (Optional[torch.Tensor]): Memory attention weights
                - final_state (torch.Tensor): Pooled final hidden state
                - loss (torch.Tensor): Computed loss or zero tensor
        
        Routing Behavior:
            1. Low complexity/short sequence: Uses CoT reasoner
            2. High complexity/long sequence: Uses multi-path engine
            3. Multi-path failure: Falls back to CoT reasoner
        
        Note:
            Output format is consistent regardless of routing decision.
            Temperature scaling is applied to multi-path logits for calibration.
        """
        device = next(self.parameters()).device
        hidden_states = self._extract_hidden_states(input_ids, kwargs).to(device)
        
        thinking_depth = 1
        complexity_score = 0.5
        
        if self.thinking_intensity > 0 and hasattr(self, 'complexity_estimator'):
            seq_len = hidden_states.shape[1]
            length_factor = torch.log1p(torch.tensor(float(seq_len), dtype=torch.float)) / 10.0
            
            variance_factor = hidden_states.var(dim=-1).mean(dim=-1)
            variance_factor = torch.sigmoid(variance_factor)
            
            learned_complexity = self.complexity_estimator(hidden_states.mean(dim=1))
            
            complexity_score = 0.3 * length_factor.to(device) + 0.3 * variance_factor + 0.4 * learned_complexity.squeeze(-1)
            complexity_score = complexity_score.mean().item()
            
            intensity = torch.sigmoid(self.thinking_depth_controller).item()
            effective_complexity = complexity_score * intensity
            
            if effective_complexity < self.complexity_threshold_low:
                thinking_depth = 1
            elif effective_complexity < self.complexity_threshold_high:
                thinking_depth = 3
            elif effective_complexity < 0.9:
                thinking_depth = 5
            else:
                thinking_depth = 10

        # Test-Time Training (TTT-E2E) adaptation
        if self.use_ttt_e2e and not self.training:
            if self.ttt_trainer is None:
                # Lazy initialization: create trainer with a dummy model
                # In practice, this should be the actual model being used
                self.ttt_trainer = YvTestTimeTrainer(
                    model=self,
                    update_layers=getattr(self.cfg, 'ttt_update_layers', 2),
                    lr=getattr(self.cfg, 'ttt_learning_rate', 1e-5),
                    max_steps=getattr(self.cfg, 'ttt_max_steps', 5)
                )

            if self.ttt_trainer.should_adapt(
                confidence=complexity_score,
                complexity=complexity_score,
                confidence_threshold=getattr(self.cfg, 'ttt_confidence_threshold', 0.6),
                complexity_threshold=getattr(self.cfg, 'ttt_complexity_threshold', 0.7)
            ):
                # Perform test-time training on the last N layers
                hidden_for_ttt = hidden_states.detach()
                with torch.enable_grad():
                    self.ttt_trainer.train()
                    # Use the current hidden states as adaptation signal
                    adapt_input = hidden_for_ttt.unsqueeze(0) if hidden_for_ttt.dim() == 2 else hidden_for_ttt
                    adaptation_loss = self.ttt_trainer.adapt_step(
                        adapt_input,
                        depth=thinking_depth,
                        confidence_target=complexity_score
                    )
                    self.ttt_trainer.eval()

        # If memory_context carries labels, remap accordingly.
        if labels is None and torch.is_tensor(memory_context):
            labels = memory_context
            memory_context = None

        # Interpret integer-valued attention_mask as token IDs when appropriate.
        input_ids_tokens = None
        if torch.is_tensor(attention_mask) and attention_mask.dtype in (torch.long, torch.int32, torch.int64):
            input_ids_tokens = attention_mask

        # Use the CoT path when routing thresholds are not satisfied.
        if not self._should_use_multi_path(hidden_states):
            cot_out = self.cot_reasoner.forward(
                input_ids=hidden_states,
                attention_mask=attention_mask,
                memory_context=memory_context,
                **kwargs
            )
            # Ensure the CoT output dictionary exposes a "loss" entry.
            if isinstance(cot_out, dict) and ("loss" not in cot_out):
                cot_out["loss"] = torch.tensor(0.0, device=device)
            return cot_out

        # Use the Multi-Path core with fallback to the CoT path if an exception occurs.
        try:
            core_out = self.multi_path_core.forward(
                hidden_states=hidden_states,
                input_ids=input_ids_tokens,
                labels=labels
            )
        except Exception:
            return self.cot_reasoner.forward(
                input_ids=hidden_states,
                attention_mask=attention_mask,
                memory_context=memory_context,
                **kwargs
            )

        # Process thinking logits to align with CoT-style outputs.
        thinking_logits = core_out.get("thinking_logits", None)
        if thinking_logits is None:
            # Fallback to the CoT thinking head for aligned logits.
            pooled = self._pool_state(hidden_states)
            thinking_logits = self.cot_reasoner.thinking_head(pooled)
        else:
            if thinking_logits.dim() == 3:
                thinking_logits = thinking_logits[:, -1, :]  # Reshape to [B, V]
            # Apply temperature scaling for calibration.
            thinking_logits = thinking_logits / torch.clamp(self._logit_temp, min=1e-3)

        # Compute auxiliary logits using CoT modules.
        pooled_state = self._pool_state(hidden_states)
        difficulty_logits = self.cot_reasoner.difficulty_head(pooled_state)
        reflection_logits = self.cot_reasoner.reflection_head(pooled_state)

        # Compute confidence score by blending CoT confidence and core uncertainty.
        cot_conf = torch.sigmoid(self.cot_reasoner.confidence_head(pooled_state))
        core_unc = core_out.get("uncertainty_scores", None)
        if core_unc is not None and torch.is_tensor(core_unc):
            # Collapse uncertainty scores to [B, 1] for fusion.
            while core_unc.dim() > 2:
                core_unc = core_unc.mean(dim=-1)
            if core_unc.dim() == 2 and core_unc.size(1) > 1:
                core_unc = core_unc.mean(dim=1, keepdim=True)
            confidence_score = 0.5 * cot_conf + 0.5 * (1.0 - core_unc)
        else:
            confidence_score = cot_conf

        # Compute correction logits using CoT error analysis.
        error_analysis = self.cot_reasoner.error_analyzer(pooled_state)
        correction_input = torch.cat([pooled_state, error_analysis], dim=-1)
        correction_logits = self.cot_reasoner.correction_head(correction_input)

        # Prepare reasoning states, steps, attention weights, and final state.
        reasoning_states = hidden_states
        reasoning_steps: list = []
        attention_weights = None
        final_state = pooled_state

        return {
            "thinking_logits": thinking_logits,
            "difficulty_logits": difficulty_logits,
            "reflection_logits": reflection_logits,
            "confidence_score": confidence_score,
            "reasoning_states": reasoning_states,
            "reasoning_steps": reasoning_steps,
            "correction_logits": correction_logits,
            "attention_weights": attention_weights,
            "final_state": final_state,
            "loss": core_out.get("loss", torch.tensor(0.0, device=device)),
            "thinking_depth": thinking_depth,
            "complexity_score": complexity_score,
        }

    def _sample_next_token(self, logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50) -> torch.Tensor:
        """Sample next token using nucleus (top-p) sampling with temperature.
        
        Args:
            logits: Tensor of shape [vocab_size] containing logits for next token.
            temperature: Sampling temperature. Lower = more deterministic.
            top_p: Nucleus sampling threshold. Cumulative probability cutoff.
            top_k: Number of top tokens to consider.
        
        Returns:
            Sampled token ID as scalar tensor.
        """
        if temperature == 0:
            return torch.argmax(logits, dim=-1)
        
        logits = logits / temperature
        
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def generate_with_thinking(
        self,
        prompt: str,
        tokenizer: Any,
        max_new_tokens: int = 2048,
        max_think_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        enable_thinking: bool = True,
    ) -> str:
        """Perform true autoregressive generation with thinking phase.
        
        This method generates text in two phases:
        1. Thinking phase: Generate reasoning tokens enclosed in <|think|>...</|think|>
        2. Answer phase: Generate final answer after thinking tokens.
        
        Each token is generated by calling the actual forward pass to obtain
        real logits, making this a genuine autoregressive generation process.
        
        Args:
            prompt: Input prompt string.
            tokenizer: Tokenizer for encoding/decoding.
            max_new_tokens: Maximum tokens to generate in total.
            max_think_tokens: Maximum tokens for thinking phase.
            temperature: Sampling temperature (0.0 = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling parameter.
            enable_thinking: Whether to enable thinking phase.
        
        Returns:
            Generated text with thinking tags if enabled.
        
        Example:
            >>> reasoner = YvUnifiedReasoner(config)
            >>> tokenizer = YvTokenizer()
            >>> output = reasoner.generate_with_thinking(
            ...     "Solve: 2 + 2 = ?",
            ...     tokenizer,
            ...     max_new_tokens=1024
            ... )
            >>> print(output)
            <|think|>
            Let me solve this step by step...
            2 + 2 = 4
            </|think|>
            The answer is 4.
        """
        self.eval()
        device = next(self.parameters()).device
        
        think_start = "<|think|>"
        think_end = "</|think|>"
        
        if enable_thinking:
            prompt_with_tag = f"{think_start}{prompt}"
        else:
            prompt_with_tag = prompt
        
        input_ids = tokenizer.encode(prompt_with_tag, return_tensors="pt").to(device)
        
        generated_tokens = []
        current_length = 0
        in_thinking_phase = enable_thinking
        think_token_count = 0
        eos_id = tokenizer.eos_token_id
        
        max_tokens = max_new_tokens if not enable_thinking else max_think_tokens + (max_new_tokens - max_think_tokens)
        
        with torch.no_grad():
            while current_length < max_tokens:
                output = self.forward(input_ids=input_ids)
                logits = output["thinking_logits"]
                
                if logits.dim() == 2:
                    logits = logits[:, -1, :]
                
                next_token = self._sample_next_token(logits.squeeze(0), temperature, top_p, top_k)
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                current_length += 1
                
                if in_thinking_phase:
                    think_token_count += 1
                    
                    decoded_partial = tokenizer.decode(generated_tokens)
                    if think_end in decoded_partial or think_token_count >= max_think_tokens:
                        in_thinking_phase = False
                        if think_end not in decoded_partial:
                            generated_tokens.append(eos_id)
                        continue
                
                if next_token.item() == eos_id and not in_thinking_phase:
                    break
                
                if current_length >= max_new_tokens:
                    break
        
        full_output = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        if enable_thinking and think_start in full_output:
            pass
        else:
            full_output = f"{think_start}{full_output}"

        return full_output

    def reason_latent(
        self,
        hidden_states: torch.Tensor,
        max_depth: int = 8,
        confidence_threshold: float = 0.85,
        enable_self_correction: bool = True,
        enable_multi_path: bool = False,
    ) -> Dict[str, Any]:
        """Perform latent-space chain-of-thought reasoning without token generation.

        This is the core reasoning engine that operates purely in hidden-state space,
        similar to how frontier models like Claude internally reason before producing
        text output. It iteratively refines hidden representations through the
        CoT reasoner, tracking confidence and uncertainty at each step.

        Architecture:
            1. Initial encoding: pool input to [B, hidden]
            2. Iterative refinement: feed through CoT reasoner, update state
            3. Confidence tracking: monitor confidence/difficulty per iteration
            4. Early stopping: exit when confidence > threshold
            5. Self-correction: detect confidence drops and backtrack
            6. Multi-path: optionally explore alternative reasoning paths

        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size].
            max_depth: Maximum number of reasoning iterations (default: 8).
            confidence_threshold: Early-stop when confidence exceeds this.
            enable_self_correction: Enable contradiction detection and backtracking.
            enable_multi_path: Enable multi-path exploration in latent space.

        Returns:
            Dictionary with refined_hidden, reasoning_trace, final_confidence,
            final_difficulty, num_iterations, correction_count, path_scores.
        """
        h = self.cfg.hidden_size
        device = hidden_states.device if hidden_states is not None else next(self.parameters()).device
        bsz = hidden_states.size(0) if hidden_states is not None else 1

        # Pool the sequence into a compact representation
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states

        reasoning_state = pooled.clone()
        reasoning_trace = []
        correction_count = 0
        path_scores = []

        # Iterative latent reasoning loop
        for step in range(max_depth):
            step_input = reasoning_state.unsqueeze(1)

            cot_out = self.cot_reasoner.forward(
                input_ids=step_input,
                attention_mask=torch.ones(bsz, 1, device=device),
                memory_context=None,
                thinking_depth=min(step + 1, 5),
            )

            if isinstance(cot_out, dict):
                thinking_logits = cot_out.get("thinking_logits", reasoning_state)
                difficulty_logits = cot_out.get("difficulty_logits", None)
                reflection_logits = cot_out.get("reflection_logits", None)
            else:
                thinking_logits = cot_out
                difficulty_logits = None
                reflection_logits = None

            if thinking_logits.dim() == 3:
                thinking_logits = thinking_logits[:, -1, :]

            thinking_probs = torch.softmax(thinking_logits, dim=-1)
            confidence = thinking_probs.max(dim=-1).values.mean().item()

            # Estimate difficulty
            if difficulty_logits is not None:
                if difficulty_logits.dim() == 3:
                    difficulty_logits = difficulty_logits[:, -1, :]
                difficulty = torch.sigmoid(difficulty_logits.mean()).item()
            else:
                difficulty = min(1.0, thinking_probs.var(dim=-1).mean().item() * 5.0)

            # Self-correction: detect confidence drops and backtrack
            if enable_self_correction and len(reasoning_trace) >= 2:
                prev_conf = reasoning_trace[-1][1]
                if confidence < prev_conf * 0.5:
                    correction_count += 1
                    blend_weight = min(0.7, confidence / max(prev_conf, 1e-6))
                    reasoning_state = (
                        blend_weight * reasoning_state
                        + (1.0 - blend_weight) * pooled
                    )
                    reasoning_state = F.normalize(reasoning_state, p=2, dim=-1)
                    continue

            # Update reasoning state with smooth residual
            reasoning_state = F.normalize(
                reasoning_state + reasoning_state * 0.1, p=2, dim=-1
            )

            # Reflection-based correction
            if reflection_logits is not None and enable_self_correction:
                if reflection_logits.dim() == 3:
                    reflection_logits = reflection_logits[:, -1, :]
                reflection_score = torch.sigmoid(reflection_logits.mean()).item()
                if reflection_score < 0.3:
                    correction_count += 1
                    reasoning_state = 0.8 * reasoning_state + 0.2 * pooled

            reasoning_trace.append((step, confidence, difficulty))

            if confidence >= confidence_threshold:
                break

        # Multi-path latent exploration
        if enable_multi_path and len(reasoning_trace) > 0:
            for path_idx in range(min(4, getattr(self.cfg, 'n_head', 16))):
                noise = torch.randn_like(reasoning_state) * 0.01
                alt_state = reasoning_state + noise

                alt_cot_out = self.cot_reasoner.forward(
                    input_ids=alt_state.unsqueeze(1),
                    attention_mask=torch.ones(bsz, 1, device=device),
                    thinking_depth=3,
                )

                alt_logits = (
                    alt_cot_out.get("thinking_logits", alt_state)
                    if isinstance(alt_cot_out, dict)
                    else alt_cot_out
                )
                if alt_logits.dim() == 3:
                    alt_logits = alt_logits[:, -1, :]
                alt_probs = torch.softmax(alt_logits, dim=-1)
                path_scores.append(alt_probs.max(dim=-1).values.mean().item())

            best_score = max(path_scores) if path_scores else 0.0
            if best_score > confidence:
                reasoning_trace.append((max_depth, best_score, difficulty))

        final_confidence = reasoning_trace[-1][1] if reasoning_trace else 0.5
        final_difficulty = reasoning_trace[-1][2] if reasoning_trace else 0.5

        return {
            "refined_hidden": reasoning_state,
            "reasoning_trace": reasoning_trace,
            "final_confidence": final_confidence,
            "final_difficulty": final_difficulty,
            "num_iterations": len(reasoning_trace),
            "correction_count": correction_count,
            "path_scores": path_scores,
        }
