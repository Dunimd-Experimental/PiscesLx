#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
EnTARoundTable — Multi-teacher consensus via swarm orchestration.

For creative and ambiguous tasks (writing, design, complex planning),
single teacher output can be biased. The round table orchestrates
multiple teacher models to debate, critique, and converge on higher-
quality training data.

Uses Encre's swarm infrastructure:
    - Each teacher model is an EncreTeammate with a specific role
    - The EncreOrchestrator manages the discussion flow
    - EncreConsensus formalizes the agreement/voting process

Phases:
    1. Proposal: Each teacher generates their approach independently
    2. Critique: Teachers review each other's proposals
    3. Debate: Open discussion to identify strengths/weaknesses
    4. Synthesis: Combine best elements into final output
    5. Verification: Cross-check the synthesized result
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("encre.enta")


@dataclass
class RoundTableResult:
    """Result from a round-table discussion.

    Args:
        consensus: The synthesized training data.
        proposals: Individual teacher proposals.
        vote_counts: How each teacher voted.
        discussion_steps: Number of discussion rounds.
        quality_score: Estimated quality of the consensus.
        metadata: Additional discussion metadata.
    """
    consensus: Dict[str, Any]
    proposals: Dict[str, str] = field(default_factory=dict)
    vote_counts: Dict[str, int] = field(default_factory=dict)
    discussion_steps: int = 0
    quality_score: float = 0.0
    metadata: Dict = field(default_factory=dict)


class EnTARoundTable:
    """Multi-teacher round-table for generating high-quality training data.

    Orchestrates multiple teacher models through a structured discussion
    protocol to produce training data that surpasses any single teacher.
    """

    def __init__(
        self,
        teacher_backends: Dict[str, Any],
        max_rounds: int = 3,
        min_teachers: int = 2,
    ):
        """
        Args:
            teacher_backends: Dict mapping model name to backend instance.
            max_rounds: Maximum discussion rounds.
            min_teachers: Minimum teachers needed for round-table.
        """
        self.backends = teacher_backends
        self.max_rounds = max_rounds
        self.min_teachers = min_teachers
        self.model_names = list(teacher_backends.keys())

        logger.info(
            f"RoundTable initialized: {len(self.model_names)} teachers "
            f"({', '.join(self.model_names)}), max_rounds={max_rounds}"
        )

    async def discuss(
        self,
        task: Any,
        initial_data: Optional[Dict[str, Any]] = None,
    ) -> RoundTableResult:
        """Run a full round-table discussion on a task.

        Args:
            task: The TaskTemplate to discuss.
            initial_data: Optional initial training data from primary teacher.

        Returns:
            RoundTableResult with consensus output.
        """
        if len(self.backends) < self.min_teachers:
            logger.warning(
                f"Not enough teachers ({len(self.backends)} < {self.min_teachers}), "
                "falling back to single teacher"
            )
            return RoundTableResult(
                consensus=initial_data or {},
                quality_score=0.5,
            )

        # Phase 1: Independent proposals
        proposals = await self._collect_proposals(task, initial_data)

        # Phase 2 + 3: Critique and debate
        for round_idx in range(self.max_rounds):
            critiques = await self._collect_critiques(task, proposals)
            revisions = await self._collect_revisions(task, critiques, proposals)
            proposals.update(revisions)

        # Phase 4: Synthesis
        consensus = await self._synthesize(task, proposals)

        # Phase 5: Quality scoring
        quality = await self._score_quality(task, consensus)

        return RoundTableResult(
            consensus=consensus,
            proposals={name: str(p)[:200] for name, p in proposals.items()},
            discussion_steps=self.max_rounds,
            quality_score=quality,
        )

    async def _collect_proposals(
        self,
        task: Any,
        initial_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Each teacher independently generates a proposal.

        Args:
            task: The task to address.
            initial_data: Optional seed data from primary teacher.

        Returns:
            Dict mapping model name to proposal text.
        """
        proposals = {}

        prompt = (
            f"You are participating in a round-table discussion. "
            f"The task is: {task.description}\n"
        )
        if initial_data:
            prompt += f"\nInitial approach: {str(initial_data)[:500]}\n"
        prompt += (
            "\nGenerate your independent solution. Be thorough and "
            "explain your reasoning step by step."
        )

        async def _get_proposal(name: str) -> tuple:
            backend = self.backends[name]
            try:
                response = await backend.generate(prompt)
                return name, response
            except Exception as e:
                logger.error(f"Round-table proposal from {name} failed: {e}")
                return name, ""

        results = await asyncio.gather(*[
            _get_proposal(name) for name in self.model_names
        ])
        proposals = dict(results)
        return proposals

    async def _collect_critiques(
        self,
        task: Any,
        proposals: Dict[str, str],
    ) -> Dict[str, str]:
        """Teachers critique each other's proposals.

        Args:
            task: The original task.
            proposals: All teacher proposals.

        Returns:
            Dict mapping model name to critique text.
        """
        critiques = {}

        for name in self.model_names:
            others = {k: v for k, v in proposals.items() if k != name}
            if not others:
                continue

            others_summary = "\n\n".join(
                f"=== {n}'s proposal ===\n{p[:500]}"
                for n, p in others.items()
            )

            prompt = (
                f"Review the following proposals for the task: {task.description}\n\n"
                f"{others_summary}\n\n"
                "Provide constructive criticism for each proposal. "
                "What are their strengths? What could be improved? "
                "Be specific and actionable."
            )

            backend = self.backends[name]
            try:
                response = await backend.generate(prompt)
                critiques[name] = response
            except Exception as e:
                logger.error(f"Round-table critique from {name} failed: {e}")

        return critiques

    async def _collect_revisions(
        self,
        task: Any,
        critiques: Dict[str, str],
        proposals: Dict[str, str],
    ) -> Dict[str, str]:
        """Teachers revise their proposals based on critiques.

        Args:
            task: The original task.
            critiques: Critiques from peers.
            proposals: Original proposals.

        Returns:
            Dict mapping model name to revised proposal.
        """
        revisions = {}

        for name in self.model_names:
            critique = critiques.get(name, "")
            if not critique:
                continue

            my_proposal = proposals.get(name, "")

            prompt = (
                f"Task: {task.description}\n\n"
                f"Your original proposal:\n{my_proposal[:500]}\n\n"
                f"Peer feedback:\n{critique[:500]}\n\n"
                "Revise your proposal based on this feedback. "
                "Address the critiques while keeping your strengths."
            )

            backend = self.backends[name]
            try:
                response = await backend.generate(prompt)
                revisions[name] = response
            except Exception as e:
                logger.error(f"Round-table revision from {name} failed: {e}")

        return revisions

    async def _synthesize(
        self,
        task: Any,
        proposals: Dict[str, str],
    ) -> Dict[str, Any]:
        """Combine proposals into a single consensus output.

        Args:
            task: The original task.
            proposals: All (revised) proposals.

        Returns:
            Synthesized training data dict.
        """
        if not proposals:
            return {"error": "no proposals"}

        # Use the first available teacher as synthesizer
        synthesizer_name = self.model_names[0]
        backend = self.backends[synthesizer_name]

        all_proposals = "\n\n".join(
            f"=== {name}'s solution ===\n{text[:800]}"
            for name, text in proposals.items() if text
        )

        prompt = (
            f"Task: {task.description}\n\n"
            f"Multiple AI teachers have proposed solutions:\n\n"
            f"{all_proposals}\n\n"
            "Synthesize the best elements from all solutions into "
            "a single, superior answer. Include:\n"
            "1. The best approach\n"
            "2. Step-by-step reasoning\n"
            "3. Final answer\n"
            "4. Why this combination is better than any individual solution"
        )

        try:
            synthesis = await backend.generate(prompt)
            return {
                "synthesis": synthesis,
                "num_contributors": len(proposals),
                "method": "round_table_consensus",
            }
        except Exception as e:
            logger.error(f"Round-table synthesis failed: {e}")
            return {"error": str(e)}

    async def _score_quality(
        self,
        task: Any,
        consensus: Dict[str, Any],
    ) -> float:
        """Score the quality of the consensus output.

        Uses a different teacher than the synthesizer to avoid bias.

        Args:
            task: The original task.
            consensus: The synthesized result.

        Returns:
            Quality score from 0.0 to 1.0.
        """
        if "error" in consensus:
            return 0.0

        # Use second teacher as scorer if available
        scorer = self.model_names[1] if len(self.model_names) > 1 else self.model_names[0]
        backend = self.backends[scorer]

        synthesis_text = str(consensus.get("synthesis", ""))[:500]

        prompt = (
            f"Rate the quality of this solution for the task: {task.description}\n\n"
            f"Solution:\n{synthesis_text}\n\n"
            "Rate from 0.0 to 1.0 based on: correctness, completeness, "
                "clarity. Return ONLY a number."
        )

        try:
            response = await backend.generate(prompt)
            score = float(response.strip()[:4])
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5
