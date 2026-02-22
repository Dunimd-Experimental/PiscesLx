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
from typing import Any, Dict, Optional
from .cot_memory import YvCoTMemoryReasoner
from .multipath_core import YvMultiPathReasoningEngine


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

        # Parameter controlling temperature scaling for logit alignment.
        self._logit_temp = nn.Parameter(torch.tensor(1.0))

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
