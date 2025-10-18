#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, Optional
from .cot_memory import ArcticCoTMemoryReasoner
from .multipath_core import ArcticMultiPathReasoningEngine

class ArcticUnifiedReasoner(nn.Module):
    """
    A unified reasoning module that routes input between the Chain-of-Thought (CoT) with Memory and Multi-Path core 
    based on problem complexity and sequence length. It maintains a stable external forward pass signature and returns 
    a consistent output dictionary format.
    """

    def __init__(self, cfg):
        """
        Initialize the ArcticUnifiedReasoner.

        Args:
            cfg: Configuration object containing parameters for initializing the reasoning engines.
        """
        super().__init__()
        self.cfg = cfg

        # Initialize the CoT with Memory reasoner
        self.cot_reasoner = ArcticCoTMemoryReasoner(cfg)

        # Initialize the Multi-Path reasoning engine
        self.multi_path_core = ArcticMultiPathReasoningEngine(cfg)

        # Get configuration parameters with default values
        self.enable_multi_path_core = getattr(cfg, "enable_multi_path_core", True)
        self.mpr_threshold = getattr(cfg, "mpr_threshold", 0.6)
        self.seq_len_threshold = getattr(cfg, "mpr_seq_len_threshold", 512)

        # Initialize parameter for logits scale alignment
        self._logit_temp = nn.Parameter(torch.tensor(1.0))

    def _extract_hidden_states(self, input_ids: Optional[torch.Tensor], kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Extract hidden states compatible with both reasoning engines. If no valid hidden states are found, 
        generate a minimal random tensor as fallback.

        Args:
            input_ids (Optional[torch.Tensor]): Input tensor that may contain hidden states.
            kwargs (Dict[str, Any]): Additional keyword arguments that may contain the "hidden_states" key.

        Returns:
            torch.Tensor: Hidden states tensor.
        """
        hidden_states = None
        if torch.is_tensor(input_ids) and input_ids.dtype in (torch.float16, torch.float32, torch.bfloat16):
            hidden_states = input_ids
        elif "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            hidden_states = kwargs["hidden_states"]
        else:
            # Fallback to generate a random tensor to mimic ArcticReasoner behavior
            hidden_size = getattr(self.cfg, "hidden_size", 1024)
            hidden_states = torch.randn(1, 1, hidden_size, device=next(self.parameters()).device)

        return hidden_states

    def _should_use_multi_path(self, hidden_states: torch.Tensor) -> bool:
        """
        Determine whether to use the Multi-Path core based on the problem complexity and sequence length.

        Args:
            hidden_states (torch.Tensor): Hidden states tensor used to calculate problem complexity.

        Returns:
            bool: True if the Multi-Path core should be used, False otherwise.
        """
        try:
            # Calculate problem complexity using the CoT reasoner
            complexity = self.cot_reasoner._calculate_problem_complexity(hidden_states)
        except Exception:
            # Use sequence length as a fallback heuristic to estimate complexity
            seq_len = hidden_states.shape[1]
            complexity = min(seq_len / float(self.seq_len_threshold), 1.0)

        seq_len = hidden_states.shape[1]
        return (self.enable_multi_path_core and
                (complexity >= self.mpr_threshold or seq_len > self.seq_len_threshold))

    def _pool_state(self, hs: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on the input tensor to obtain a tensor of shape [B, H], 
        which is compatible with the CoT heads.

        Args:
            hs (torch.Tensor): Input tensor with shape [B, T, H].

        Returns:
            torch.Tensor: Mean-pooled tensor with shape [B, H].
        """
        return hs.mean(dim=1)

    def initialize_reasoning_tokens(self, tokenizer=None):
        """
        Initialize special tokens for reasoning across all components that support this operation.

        Args:
            tokenizer: Tokenizer object used to initialize reasoning tokens. Defaults to None.
        """
        if hasattr(self, "multi_path_core") and hasattr(self.multi_path_core, "initialize_reasoning_tokens"):
            self.multi_path_core.initialize_reasoning_tokens(tokenizer)
        if hasattr(self, "cot_reasoner") and hasattr(self.cot_reasoner, "initialize_reasoning_tokens"):
            try:
                self.cot_reasoner.initialize_reasoning_tokens(tokenizer)  # type: ignore
            except Exception:
                pass

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                memory_context: Optional[list] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Perform a forward pass through the unified reasoning module, maintaining compatibility with ArcticReasoner's signature.

        Args:
            input_ids (Optional[torch.Tensor]): Input token IDs. Defaults to None.
            attention_mask (Optional[torch.Tensor]): Attention mask. Defaults to None.
            memory_context (Optional[list]): Memory context. Defaults to None.
            labels (Optional[torch.Tensor]): Ground truth labels. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary containing reasoning outputs with keys:
                thinking_logits, difficulty_logits, reflection_logits, confidence_score,
                reasoning_states, reasoning_steps, correction_logits, attention_weights, final_state, loss.
        """
        device = next(self.parameters()).device
        hidden_states = self._extract_hidden_states(input_ids, kwargs).to(device)

        # If memory_context is a tensor and labels is None, remap memory_context to labels
        if labels is None and torch.is_tensor(memory_context):
            labels = memory_context
            memory_context = None

        # If attention_mask contains token IDs, interpret it as input_ids_tokens
        input_ids_tokens = None
        if torch.is_tensor(attention_mask) and attention_mask.dtype in (torch.long, torch.int32, torch.int64):
            input_ids_tokens = attention_mask

        # Use the CoT path if the routing condition is not met or the Multi-Path core is disabled
        if not self._should_use_multi_path(hidden_states):
            cot_out = self.cot_reasoner.forward(
                input_ids=hidden_states,
                attention_mask=attention_mask,
                memory_context=memory_context,
                **kwargs
            )
            # Ensure the output dictionary contains the "loss" key
            if isinstance(cot_out, dict) and ("loss" not in cot_out):
                cot_out["loss"] = torch.tensor(0.0, device=device)
            return cot_out

        # Use the Multi-Path core, fallback to the CoT path if it fails
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

        # Process thinking_logits to match the shape of CoT outputs
        thinking_logits = core_out.get("thinking_logits", None)
        if thinking_logits is None:
            # Fallback to use the CoT thinking head on the pooled state
            pooled = self._pool_state(hidden_states)
            thinking_logits = self.cot_reasoner.thinking_head(pooled)
        else:
            if thinking_logits.dim() == 3:
                thinking_logits = thinking_logits[:, -1, :]  # Reshape to [B, V]
            # Apply temperature scaling to thinking_logits
            thinking_logits = thinking_logits / torch.clamp(self._logit_temp, min=1e-3)

        # Compute auxiliary logits using CoT modules
        pooled_state = self._pool_state(hidden_states)
        difficulty_logits = self.cot_reasoner.difficulty_head(pooled_state)
        reflection_logits = self.cot_reasoner.reflection_head(pooled_state)

        # Compute confidence score by fusing CoT confidence and core uncertainty
        cot_conf = torch.sigmoid(self.cot_reasoner.confidence_head(pooled_state))
        core_unc = core_out.get("uncertainty_scores", None)
        if core_unc is not None and torch.is_tensor(core_unc):
            # Reduce core_unc to a tensor of shape [B, 1]
            while core_unc.dim() > 2:
                core_unc = core_unc.mean(dim=-1)
            if core_unc.dim() == 2 and core_unc.size(1) > 1:
                core_unc = core_unc.mean(dim=1, keepdim=True)
            confidence_score = 0.5 * cot_conf + 0.5 * (1.0 - core_unc)
        else:
            confidence_score = cot_conf

        # Compute correction logits using CoT error analysis
        error_analysis = self.cot_reasoner.error_analyzer(pooled_state)
        correction_input = torch.cat([pooled_state, error_analysis], dim=-1)
        correction_logits = self.cot_reasoner.correction_head(correction_input)

        # Prepare reasoning states, steps, attention weights, and final state
        reasoning_states = hidden_states  # Use the last integrated state
        reasoning_steps = []  # Use an empty list as a placeholder
        attention_weights = None  # No cross-attention in this path
        final_state = pooled_state  # Shape: [B, H]

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