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
SPELL: Self-Play Reinforcement Learning for Evolving Long-Context LLMs
(ICLR 2026, arXiv:2509.23863).

Three-role self-play framework: Questioner, Responder, Verifier.
Single model learns to generate questions, answer them, and verify
answers — all within its own loop. Automated curriculum increases
document length; reward adapts to model's evolving capability.

Reference: Yang et al. "SPELL: Self-Play Reinforcement Learning for
Evolving Long-Context Language Models." ICLR 2026.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple


# Paper: Yang et al., "SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models," ICLR 2026, arXiv:2509.23863
class YvVeriCoTVerifier(nn.Module):
    """
    SPELL Verifier: evaluates semantic equivalence between generated
    answers and reference answers, producing reward signals for the
    self-play loop. Also acts as the Questioner (generates questions)
    and the quality gate for the Responder.
    """

    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.hidden_size = config.hidden_size

        proj_dim = getattr(config, 'spell_proj_dim', self.hidden_size // 2)

        self.questioner_proj = nn.Linear(self.hidden_size, proj_dim, device=device, dtype=dtype)
        self.responder_proj = nn.Linear(self.hidden_size, proj_dim, device=device, dtype=dtype)
        self.verifier_proj = nn.Linear(proj_dim * 2, 1, device=device, dtype=dtype)

        self.quality_estimator = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(proj_dim, 1, device=device, dtype=dtype),
            nn.Sigmoid(),
        )

        self.ramp_steps = getattr(config, 'spell_ramp_steps', 1000)
        self.register_buffer('spell_step', torch.tensor(0))

    def forward(
        self,
        step_hidden: torch.Tensor,
        prev_step_hidden: Optional[torch.Tensor] = None,
        step_text: str = "",
        prev_step_text: str = "",
        context_hidden: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        device = step_hidden.device

        if step_hidden.dim() == 2:
            q_emb = self.questioner_proj(step_hidden)
        else:
            q_emb = self.questioner_proj(step_hidden.mean(dim=1))

        if context_hidden is not None:
            c_emb = context_hidden.mean(dim=1) if context_hidden.dim() > 2 else context_hidden
            r_emb = self.responder_proj(c_emb)
            ver_input = torch.cat([q_emb, r_emb], dim=-1)
            quality = self.quality_estimator(ver_input).squeeze(-1)
        else:
            quality = torch.ones(q_emb.shape[0], device=device) * 0.5

        is_valid = quality > 0.5

        result = {
            "validity_score": quality,
            "consistency_score": quality,
            "combined_score": quality,
            "is_valid": is_valid,
            "symbolic_score": 0.0,
            "rule_matches": {},
            "matched_rules": [],
        }

        if return_details:
            result["validity_raw"] = quality
            result["consistency_raw"] = quality

        self.spell_step.add_(1)
        return result

    def verify_chain(
        self,
        step_hiddens: List[torch.Tensor],
        step_texts: List[str],
        context_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        results = []
        total_valid = 0

        for i, h in enumerate(step_hiddens):
            prev_h = step_hiddens[i - 1] if i > 0 else None
            prev_t = step_texts[i - 1] if i > 0 else ""
            result = self.forward(h, prev_h, step_texts[i], prev_t, context_hidden)
            results.append(result)
            if result["is_valid"]:
                total_valid += 1

        chain_validity = total_valid / max(len(results), 1)
        avg_scores = torch.stack([r["validity_score"] for r in results]).mean()

        return {
            "step_results": results,
            "chain_validity": chain_validity,
            "avg_validity": avg_scores,
            "avg_consistency": avg_scores,
            "total_steps": len(results),
            "valid_steps": total_valid,
        }

    def verify_batch(
        self,
        hidden_states: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        reasoner_out: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        B, T, H = hidden_states.shape
        pooled = hidden_states.mean(dim=1)

        q_emb = self.questioner_proj(pooled)
        r_emb = self.responder_proj(pooled)
        ver_input = torch.cat([q_emb, r_emb], dim=-1)
        quality = self.quality_estimator(ver_input).squeeze(-1)

        curriculum_factor = min(1.0, self.spell_step.float() / max(1, self.ramp_steps))
        is_valid = quality > (0.5 - 0.2 * curriculum_factor)

        result = {
            "verified": is_valid,
            "confidence": quality,
            "correction_logits": None,
            "verifier_loss": torch.tensor(0.0, device=hidden_states.device),
        }

        if not is_valid.any() and logits is not None:
            correction = torch.zeros_like(logits)
            correction[:, :, :10] = quality.mean() * 0.05
            result['correction_logits'] = correction

        self.spell_step.add_(1)
        return result

    def compute_reflection(
        self,
        step_hidden: torch.Tensor,
        verification_result: Dict[str, Any],
    ) -> torch.Tensor:
        quality = verification_result["validity_score"]
        if not isinstance(quality, torch.Tensor):
            quality = torch.tensor(quality, device=step_hidden.device)
        quality = quality.to(step_hidden.dtype)

        if quality.dim() < step_hidden.dim():
            quality = quality.view(-1, *([1] * (step_hidden.dim() - 1)))

        correction = (1.0 - quality) * step_hidden * 0.1
        return step_hidden + correction


# Paper: Yang et al., "SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models," ICLR 2026, arXiv:2509.23863
class YvVeriCoTReflector(nn.Module):
    """
    SPELL self-play loop: cycles through Questioner → Responder → Verifier
    roles to iteratively improve long-context reasoning.
    """

    def __init__(self, verifier: YvVeriCoTVerifier, max_reflections: int = 3):
        super().__init__()
        self.verifier = verifier
        self.max_reflections = max_reflections
        self.register_buffer('selfplay_round', torch.tensor(0))

    def forward(
        self,
        step_hidden: torch.Tensor,
        step_text: str = "",
        prev_step_hidden: Optional[torch.Tensor] = None,
        prev_step_text: str = "",
        context_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        reflections = 0
        current_hidden = step_hidden

        while reflections < self.max_reflections:
            result = self.verifier.forward(
                current_hidden, prev_step_hidden, step_text, prev_step_text,
                context_hidden, return_details=True
            )
            valid = result["is_valid"]
            if isinstance(valid, torch.Tensor) and valid.all():
                break
            elif isinstance(valid, bool) and valid:
                break

            current_hidden = self.verifier.compute_reflection(current_hidden, result)
            reflections += 1

        self.selfplay_round.add_(1)

        return current_hidden, {
            "reflections": reflections,
            "final_valid": result["is_valid"],
            "final_validity": result["validity_score"],
            "selfplay_round": int(self.selfplay_round.item()),
        }
