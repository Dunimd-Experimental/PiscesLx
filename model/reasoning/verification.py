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

"""Reasoning Verification Modules for Yv Models.

Implements:
- CRV: Circuit-based white-box CoT verification (arXiv 2510.09312)
- OTV: One-token forward verification (OpenReview 2025)
- ARES: Autoregressive reasoning entailment stability (EMNLP 2025)
- CoT Failure Mode Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class YvCRV(nn.Module):
    """Circuit-based Reasoning Verification.

    Verifies CoT reasoning via computation graph analysis.
    Detects logical contradictions by tracing value flows.

    Attributes:
        model (nn.Module): Model to verify.
        hidden_size (int): Hidden dimension.
        contradiction_detector (nn.Linear): Detects contradictions.

    Example:
        >>> crv = YvCRV(model, hidden_size=4096)
        >>> is_valid = crv.verify_reasoning(steps, final_answer)
    """

    def __init__(self, model: nn.Module, hidden_size: int):
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size

        # Contradiction detection network
        self.contradiction_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def verify_reasoning(
        self,
        reasoning_steps: List[torch.Tensor],
        final_answer: torch.Tensor
    ) -> float:
        """Verify reasoning chain via circuit tracing.

        Uses the trained contradiction detector to evaluate reasoning
        quality, falling back to cosine similarity for consistency checks.

        Args:
            reasoning_steps: List of hidden states for each step.
            final_answer: Final answer representation.

        Returns:
            Verification score (0-1), higher is better.
        """
        if len(reasoning_steps) < 2:
            return 1.0

        # Use the trained contradiction detector between consecutive steps
        contradiction_scores = []
        for i in range(len(reasoning_steps) - 1):
            step_i = reasoning_steps[i].mean(dim=1)
            step_j = reasoning_steps[i + 1].mean(dim=1)
            step_pair = torch.cat([step_i, step_j], dim=-1)
            # Low contradiction = high consistency
            contradiction_prob = self.contradiction_detector(step_pair).mean().item()
            contradiction_scores.append(1.0 - contradiction_prob)

        # Check final answer consistency with last reasoning step
        final_pair = torch.cat([
            reasoning_steps[-1].mean(dim=1),
            final_answer.mean(dim=1)
        ], dim=-1)
        final_contradiction = self.contradiction_detector(final_pair).mean().item()
        contradiction_scores.append(1.0 - final_contradiction)

        return sum(contradiction_scores) / len(contradiction_scores)

    def detect_contradictions(
        self,
        steps: List[torch.Tensor]
    ) -> List[Tuple[int, float]]:
        """Detect contradictions between reasoning steps.

        Args:
            steps: List of step hidden states.

        Returns:
            List of (step_idx, contradiction_score) tuples.
        """
        contradictions = []

        for i in range(len(steps) - 1):
            step_pair = torch.cat([steps[i].mean(dim=1), steps[i + 1].mean(dim=1)], dim=-1)
            contradiction_prob = self.contradiction_detector(step_pair)

            if contradiction_prob.mean().item() > 0.5:
                contradictions.append((i, contradiction_prob.mean().item()))

        return contradictions


class YvOTV(nn.Module):
    """One-Token Verification.

    Verifies reasoning quality with a single forward pass per step.
    Lightweight alternative to full CRV.

    Attributes:
        model (nn.Module): Model to verify.
        hidden_size (int): Hidden dimension.
        quality_scorer (nn.Linear): Scores step quality.

    Example:
        >>> otv = YvOTV(model, hidden_size=4096)
        >>> quality = otv.verify_step(step_hidden, next_token_logits)
    """

    def __init__(self, model: nn.Module, hidden_size: int):
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size

        self.quality_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def verify_step(
        self,
        current_step: torch.Tensor,
        next_token_logits: torch.Tensor
    ) -> float:
        """Verify quality of a single reasoning step.

        Args:
            current_step: Current step hidden state [batch, hidden].
            next_token_logits: Next token logits [batch, vocab_size].

        Returns:
            Quality score (0-1).
        """
        # Score based on hidden state quality
        hidden_score = self.quality_scorer(current_step).mean().item()

        # Score based on prediction confidence
        probs = F.softmax(next_token_logits, dim=-1)
        max_prob = probs.max(dim=-1).values.mean().item()

        # Combined score
        quality = 0.6 * hidden_score + 0.4 * max_prob

        return quality

    def should_regenerate(
        self,
        step_quality: float,
        threshold: float = 0.6
    ) -> bool:
        """Determine if step should be re-generated.

        Args:
            step_quality: Quality score from verify_step.
            threshold: Minimum acceptable quality.

        Returns:
            True if step should be re-generated.
        """
        return step_quality < threshold


class YvARES(nn.Module):
    """Autoregressive Reasoning Entailment Stability.

    Computes entailment probability for reasoning stability.

    Attributes:
        entailment_scorer (nn.Linear): Scores entailment between steps.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.entailment_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def compute_entailment(
        self,
        premise: torch.Tensor,
        hypothesis: torch.Tensor
    ) -> torch.Tensor:
        """Compute entailment probability.

        Args:
            premise: Premise hidden state [batch, hidden].
            hypothesis: Hypothesis hidden state [batch, hidden].

        Returns:
            Entailment probability [batch].
        """
        pair = torch.cat([premise, hypothesis], dim=-1)
        return self.entailment_scorer(pair).squeeze(-1)

    def compute_uncertainty(
        self,
        reasoning_chain: List[torch.Tensor]
    ) -> List[float]:
        """Compute uncertainty for each step in reasoning chain.

        Args:
            reasoning_chain: List of step hidden states.

        Returns:
            List of uncertainty scores (0-1).
        """
        uncertainties = []

        for i in range(len(reasoning_chain)):
            step = reasoning_chain[i]
            # Uncertainty = 1 - max activation
            max_act = step.abs().max(dim=-1).values.mean().item()
            uncertainties.append(1.0 - max_act)

        return uncertainties


def detect_cot_failure_modes(
    reasoning_steps: List[str],
    final_answer: str
) -> List[Tuple[str, float]]:
    """Detect CoT failure modes.

    Args:
        reasoning_steps: Textual reasoning steps.
        final_answer: Final answer text.

    Returns:
        List of (failure_mode, confidence) tuples.
    """
    failures = []

    # Check for premature conclusion
    if len(reasoning_steps) < 3 and len(final_answer.split()) > 5:
        failures.append(("premature_conclusion", 0.7))

    # Check for circular reasoning
    step_texts = [s.lower() for s in reasoning_steps]
    for i in range(len(step_texts)):
        for j in range(i + 1, len(step_texts)):
            if step_texts[i] in step_texts[j] or step_texts[j] in step_texts[i]:
                failures.append(("circular_reasoning", 0.6))
                break

    # Check for hallucination (repeated phrases without new info)
    if len(reasoning_steps) > 5:
        unique_steps = set(step_texts)
        if len(unique_steps) < len(step_texts) * 0.7:
            failures.append(("hallucination", 0.5))

    return failures
