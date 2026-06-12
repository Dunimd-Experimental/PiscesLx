#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from encre.swarm.mailbox import EncreMailbox


@dataclass
class Proposal:
    id: str
    title: str
    description: str
    options: list[str] = field(default_factory=list)
    proposed_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "options": self.options,
            "proposed_by": self.proposed_by,
        }


@dataclass
class Vote:
    voter_id: str
    choice: str
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsensusResult:
    proposal_id: str
    winner: str
    vote_counts: dict[str, int]
    is_consensus: bool  # True if > 2/3 majority
    votes: list[Vote] = field(default_factory=list)
    summary: str = ""


class EncreConsensus:
    """Consensus protocols for multi-agent decision making.

    Two modes:
    1. Proposal-Vote: One agent proposes, N agents vote. Majority wins.
    2. Debate-Mediation: Two agents debate opposing positions, mediator decides.
    """

    def __init__(self) -> None:
        self._votes: dict[str, list[Vote]] = {}
        self._results: dict[str, ConsensusResult] = {}
        self._mailbox = EncreMailbox(owner_id="consensus")

    def create_proposal(
        self,
        title: str,
        description: str,
        options: list[str],
        proposed_by: str = "",
    ) -> Proposal:
        pid = str(uuid.uuid4())[:8]
        return Proposal(
            id=pid,
            title=title,
            description=description,
            options=options,
            proposed_by=proposed_by,
        )

    def cast_vote(
        self,
        proposal_id: str,
        voter_id: str,
        choice: str,
        reasoning: str = "",
    ) -> Vote:
        vote = Vote(voter_id=voter_id, choice=choice, reasoning=reasoning)
        self._votes.setdefault(proposal_id, []).append(vote)
        return vote

    def tally(self, proposal: Proposal) -> ConsensusResult:
        votes = self._votes.get(proposal.id, [])
        counts: dict[str, int] = {opt: 0 for opt in proposal.options}
        for v in votes:
            if v.choice in counts:
                counts[v.choice] += 1

        total = sum(counts.values())
        winner = max(counts, key=counts.get) if counts else (proposal.options[0] if proposal.options else "")
        winner_count = counts.get(winner, 0)
        is_consensus = total > 0 and (winner_count / total) >= 0.67

        summary = f"Result: {winner} ({winner_count}/{total} votes)"
        if is_consensus:
            summary += " — CONSENSUS reached"
        else:
            summary += " — no clear consensus"

        result = ConsensusResult(
            proposal_id=proposal.id,
            winner=winner,
            vote_counts=counts,
            is_consensus=is_consensus,
            votes=list(votes),
            summary=summary,
        )
        self._results[proposal.id] = result
        return result

    async def run_proposal_vote(
        self,
        proposal: Proposal,
        voters: list["EncreTeammate"],
        timeout: float = 60.0,
    ) -> ConsensusResult:
        """Send proposal to all voters, collect responses, tally.

        Each voter's agent evaluates the proposal independently and sends a
        vote back through the mailbox.  Results are tallied and consensus is
        declared when a > 2/3 supermajority is reached.
        """
        from encre.swarm.teammate import EncreTeammate
        from encre.agent import EncreAgent
        from encre.config import EncreConfig
        from encre.utils.types import TextDelta

        # Step 1 — Deliver the proposal to every voter's mailbox so each
        # voter has a durable record of what is being decided.
        proposal_msg = (
            f"PROPOSAL [{proposal.id}]: {proposal.title}\n"
            f"{proposal.description}\n"
            f"Options: {', '.join(proposal.options)}\n"
            f"Choose exactly one option from the list above. "
            f"Reply with ONLY the option text."
        )
        for voter in voters:
            await self._mailbox.send(voter.mailbox, proposal_msg)

        # Step 2 — Run a brief agent per voter concurrently.  Each agent
        # receives the proposal content directly in its prompt (equivalent
        # to reading the mailbox) and returns a single choice.
        async def _voter_task(voter: EncreTeammate) -> Vote | None:
            try:
                vote_prompt = (
                    f"You are voter '{voter.name}' (id: {voter.teammate_id}).\n"
                    f"You have received the following proposal:\n\n"
                    f"Title: {proposal.title}\n"
                    f"Description: {proposal.description}\n"
                    f"Options: {', '.join(proposal.options)}\n\n"
                    f"Choose exactly ONE option from the list. "
                    f"Reply with ONLY the exact option text, nothing else."
                )
                config = EncreConfig(max_turns=3, permission_mode="bypass")
                agent = EncreAgent(config=config)
                parts: list[str] = []
                async for event in agent.run(vote_prompt):
                    if isinstance(event, TextDelta) and event.text:
                        parts.append(event.text)
                raw = "".join(parts).strip()

                # Normalise: try exact match first, then case-insensitive
                choice = ""
                for opt in proposal.options:
                    if raw == opt:
                        choice = opt
                        break
                if not choice:
                    for opt in proposal.options:
                        if opt.lower() in raw.lower():
                            choice = opt
                            break

                # Send vote back to the consensus mailbox so external
                # observers (and the mailbox audit trail) see it.
                if choice:
                    await voter.mailbox.send(self._mailbox, choice)

                return Vote(
                    voter_id=voter.teammate_id,
                    choice=choice or raw,
                    reasoning=raw if choice else "",
                )
            except asyncio.CancelledError:
                return None
            except Exception:
                return None

        tasks = [asyncio.create_task(_voter_task(v)) for v in voters]

        # Step 3 — Wait for all voters to respond (or timeout).
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            for t in tasks:
                if not t.done():
                    t.cancel()

        # Step 4 — Harvest votes from completed tasks.
        results: list[Vote] = []
        for t in tasks:
            try:
                if t.done() and not t.cancelled():
                    vote = t.result()
                    if vote is not None and vote.choice:
                        results.append(vote)
            except Exception:
                pass

        # Step 5 — Also drain any late-arriving mailbox messages.
        deadline = time.time() + 2.0  # short grace window after tasks finish
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(
                    self._mailbox.receive(),
                    timeout=max(0.1, deadline - time.time()),
                )
                if msg:
                    content = msg.content.strip()
                    for opt in proposal.options:
                        if opt.lower() in content.lower():
                            # Avoid duplicates — check if this sender already voted
                            if not any(v.voter_id == msg.sender for v in results):
                                results.append(Vote(
                                    voter_id=msg.sender,
                                    choice=opt,
                                    reasoning=content,
                                ))
                            break
            except asyncio.TimeoutError:
                break

        # Register and tally.
        for v in results:
            self._votes.setdefault(proposal.id, []).append(v)

        return self.tally(proposal)

    async def debate_mediate(
        self,
        topic: str,
        debaters: list["EncreTeammate"],
        mediator: "EncreTeammate",
        rounds: int = 3,
    ) -> ConsensusResult:
        """Multi-round debate followed by mediator adjudication.

        1. Each debater receives the topic and states their initial position.
        2. For *rounds* iterations positions are exchanged and rebuttals made.
        3. A mediator agent reviews the full transcript and declares a winner.

        Returns a ``ConsensusResult`` whose ``winner`` is the name (or id)
        of the winning debater, ``DRAW``, or ``UNDECIDED``.
        """
        from encre.swarm.teammate import EncreTeammate
        from encre.agent import EncreAgent
        from encre.config import EncreConfig
        from encre.utils.types import TextDelta

        if len(debaters) < 2:
            return ConsensusResult(
                proposal_id="",
                winner="UNDECIDED",
                vote_counts={},
                is_consensus=False,
                summary="Debate requires at least two debaters.",
            )

        transcript: list[str] = [
            f"=== DEBATE: {topic} ===",
            f"Debaters: {', '.join(d.name for d in debaters)}",
            f"Mediator: {mediator.name}",
            f"Rounds: {rounds}",
        ]

        # Each debater is assigned a label A, B, C, …
        labels = [chr(ord("A") + i) for i in range(len(debaters))]
        debater_label: dict[str, str] = {
            d.teammate_id: labels[i] for i, d in enumerate(debaters)
        }
        label_to_name: dict[str, str] = {
            labels[i]: d.name for i, d in enumerate(debaters)
        }

        # ----------------------------------------------------------------
        # Helper: run a brief agent and return its text output
        # ----------------------------------------------------------------
        async def _run_agent(prompt: str, max_turns: int = 5) -> str:
            config = EncreConfig(max_turns=max_turns, permission_mode="bypass")
            agent = EncreAgent(config=config)
            parts: list[str] = []
            try:
                async for event in agent.run(prompt):
                    if isinstance(event, TextDelta) and event.text:
                        parts.append(event.text)
            except Exception:
                pass
            return "".join(parts).strip()

        # ----------------------------------------------------------------
        # Round 0 — Opening statements
        # ----------------------------------------------------------------
        opening_statements: dict[str, str] = {}
        transcript.append("\n--- OPENING STATEMENTS ---")
        for i, debater in enumerate(debaters):
            label = labels[i]
            opponent_labels = [l for l in labels if l != label]
            prompt = (
                f"You are debating: {topic}\n"
                f"Your label: {label}\n"
                f"Opponents: {', '.join(opponent_labels)}\n"
                f"Make your opening statement. Argue for your position "
                f"in 3-5 sentences. Be clear and persuasive."
            )
            try:
                statement = await asyncio.wait_for(
                    _run_agent(prompt), timeout=120.0
                )
            except asyncio.TimeoutError:
                statement = "[timeout]"
            opening_statements[debater.teammate_id] = statement
            transcript.append(f"{label} ({debater.name}): {statement}")

        # ----------------------------------------------------------------
        # Rounds 1..N — Rebuttals (each debater sees opponent arguments)
        # ----------------------------------------------------------------
        for r in range(1, rounds + 1):
            transcript.append(f"\n--- ROUND {r} ---")
            round_statements: dict[str, str] = {}

            for i, debater in enumerate(debaters):
                label = labels[i]
                # Gather all opponent statements from previous round or openings
                opponent_args_parts: list[str] = []
                for j, other in enumerate(debaters):
                    if other.teammate_id == debater.teammate_id:
                        continue
                    prev = round_statements.get(other.teammate_id) or opening_statements.get(other.teammate_id, "")
                    opponent_args_parts.append(
                        f"{labels[j]} ({other.name}): {prev[:2000]}"
                    )
                opponent_args = "\n\n".join(opponent_args_parts)

                rebuttal_prompt = (
                    f"You are debating: {topic}\n"
                    f"Your label: {label}\n\n"
                    f"Your opponent{'s' if len(debaters) > 2 else ''} argued:\n"
                    f"{opponent_args}\n\n"
                    f"This is round {r} of {rounds}. "
                    f"Rebutt your opponent{'s' if len(debaters) > 2 else ''} "
                    f"and strengthen your own position. "
                    f"Keep it to 3-5 sentences."
                )
                try:
                    rebuttal = await asyncio.wait_for(
                        _run_agent(rebuttal_prompt), timeout=120.0
                    )
                except asyncio.TimeoutError:
                    rebuttal = "[timeout]"
                round_statements[debater.teammate_id] = rebuttal
                transcript.append(f"{label} ({debater.name}): {rebuttal}")

            # Update opening_statements to include latest arguments for next round
            for tid, stmt in round_statements.items():
                opening_statements[tid] = stmt

        # ----------------------------------------------------------------
        # Mediation — mediator reviews the full transcript and decides
        # ----------------------------------------------------------------
        transcript.append("\n--- MEDIATOR DECISION ---")
        options_for_mediator = [d.name for d in debaters] + ["DRAW"]
        verdict_prompt = (
            f"You are the mediator for this debate.\n\n"
            f"Topic: {topic}\n\n"
            f"Full debate transcript:\n"
            f"{chr(10).join(transcript)}\n\n"
            f"Based on the strength of arguments presented, which debater "
            f"has the stronger position?\n"
            f"Reply with ONLY the winner's name ({', '.join(d.name for d in debaters)}) "
            f"or 'DRAW' if the debate is tied. "
            f"Then on a new line, provide a brief justification."
        )
        try:
            verdict_text = await asyncio.wait_for(
                _run_agent(verdict_prompt, max_turns=8), timeout=120.0
            )
        except asyncio.TimeoutError:
            verdict_text = "DRAW\n[Mediator timed out]"

        transcript.append(f"Mediator ({mediator.name}): {verdict_text}")

        # Parse the verdict
        verdict_lines = verdict_text.strip().split("\n")
        winner_raw = verdict_lines[0].strip() if verdict_lines else "DRAW"
        justification = "\n".join(verdict_lines[1:]).strip() if len(verdict_lines) > 1 else ""

        # Normalise winner name
        winner = "DRAW"
        for d in debaters:
            if d.name.lower() in winner_raw.lower():
                winner = d.name
                break

        # Build vote-like counts for the result
        vote_counts: dict[str, int] = {d.name: 0 for d in debaters}
        vote_counts["DRAW"] = 0
        if winner in vote_counts:
            vote_counts[winner] = 1
        else:
            vote_counts["DRAW"] = 1
            winner = "DRAW"

        is_consensus = winner != "DRAW"

        proposal_id = str(uuid.uuid4())[:8]
        full_transcript = "\n".join(transcript)
        summary = (
            f"Debate winner: {winner}\n"
            f"Justification: {justification}\n\n"
            f"--- Full transcript ---\n"
            f"{full_transcript}"
        )

        result = ConsensusResult(
            proposal_id=proposal_id,
            winner=winner,
            vote_counts=vote_counts,
            is_consensus=is_consensus,
            summary=summary,
        )
        self._results[proposal_id] = result
        return result
