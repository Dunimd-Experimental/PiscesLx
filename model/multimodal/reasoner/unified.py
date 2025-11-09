#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

"""Unified routing logic for Arctic's Chain-of-Thought and multi-path reasoners.

The module defines :class:`ArcticUnifiedReasoner`, which selects between the
CoT memory reasoner and the multi-path reasoning engine based on problem
complexity and sequence length while preserving a consistent forward interface.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, Optional
from .cot_memory import ArcticCoTMemoryReasoner
from .multipath_core import ArcticMultiPathReasoningEngine

class ArcticUnifiedReasoner(nn.Module):
    """Route queries between CoT-with-memory and multi-path reasoning cores."""

    def __init__(self, cfg: Any):
        """Initialize sub-components and routing thresholds from configuration.

        Args:
            cfg: Configuration namespace providing shared parameters across reasoning engines.
        """
        super().__init__()
        self.cfg = cfg

        # Initialize the CoT with Memory reasoner.
        self.cot_reasoner = ArcticCoTMemoryReasoner(cfg)

        # Initialize the Multi-Path reasoning engine.
        self.multi_path_core = ArcticMultiPathReasoningEngine(cfg)

        # Fetch routing parameters with fallbacks.
        self.enable_multi_path_core = getattr(cfg, "enable_multi_path_core", True)
        self.mpr_threshold = getattr(cfg, "mpr_threshold", 0.6)
        self.seq_len_threshold = getattr(cfg, "mpr_seq_len_threshold", 512)

        # Parameter controlling temperature scaling for logit alignment.
        self._logit_temp = nn.Parameter(torch.tensor(1.0))

    def _extract_hidden_states(self, input_ids: Optional[torch.Tensor], kwargs: Dict[str, Any]) -> torch.Tensor:
        """Obtain hidden states compatible with downstream reasoning modules."""
        hidden_states = None
        if torch.is_tensor(input_ids) and input_ids.dtype in (torch.float16, torch.float32, torch.bfloat16):
            hidden_states = input_ids
        elif "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            hidden_states = kwargs["hidden_states"]
        else:
            # Generate a random tensor fallback to mimic ArcticReasoner behavior.
            hidden_size = getattr(self.cfg, "hidden_size", 1024)
            hidden_states = torch.randn(1, 1, hidden_size, device=next(self.parameters()).device)

        return hidden_states

    def _should_use_multi_path(self, hidden_states: torch.Tensor) -> bool:
        """Decide whether the multi-path engine should handle the query."""
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
        """Mean-pool hidden states to produce [B, H] tensors."""
        return hs.mean(dim=1)

    def initialize_reasoning_tokens(self, tokenizer: Optional[Any] = None) -> None:
        """Forward token initialization requests to each component."""
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
        """Execute a forward pass compatible with ArcticReasoner outputs."""
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
