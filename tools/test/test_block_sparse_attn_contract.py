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
Executable form of docs/proofs/05 (block-sparse attention).

Four properties are asserted against the same weights, because each one is a claim the proof
makes about this implementation and each can be broken independently of the others:

1. **Exactness.** With a key budget of at least one block the layer must equal dense causal
   attention elementwise - the sparse path is an optimisation, not an approximation, wherever
   it drops nothing.
2. **Causal.** Perturbing one token must leave every earlier output bit-identical, including
   the *selection* map, which is why the query side of the block score uses a single
   representative row rather than a pooled block.
3. **Finite.** A row the caller's mask forbids entirely must still soften to a distribution,
   not to NaN; its own key is always in the selected set by construction.
4. **Honest.** A causal mask is idempotent here, and a mask that conflicts with causality is
   still honoured - a mask that is silently dropped would pass (1)-(3) unnoticed.
"""

from typing import Callable, List, Tuple

import torch

from model.config import YvConfig
from model.core.token_sparse_attn import YvTokenSparseAttention

__all__ = ["PiscesLxBlockSparseAttentionContract"]


class PiscesLxBlockSparseAttentionContract:
    """The four claims of docs/proofs/05, checked on one instance."""

    BATCH = 2
    SEQ = 16
    HIDDEN = 64
    HEADS = 4
    TOLERANCE = 1e-5

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []
        cfg = YvConfig(
            hidden_size=self.HIDDEN,
            n_head=self.HEADS,
            tactic_block_size=self.SEQ,
            tactic_blocks_per_query=1,
        )
        self.module = YvTokenSparseAttention(cfg, device=torch.device("cpu"),
                                            dtype=torch.float32)
        self.module.eval()
        torch.manual_seed(0)
        self.states = torch.randn(self.BATCH, self.SEQ, self.HIDDEN)

    def record(self, name: str, passed: bool, detail: str) -> None:
        self.results.append((name, passed, detail))

    def _dense_causal(self, states: torch.Tensor) -> torch.Tensor:
        """Reference dense causal attention reusing this module's own projections."""
        batch, seq, _ = states.shape
        heads, head_dim = self.module.num_heads, self.module.head_dim
        q = self.module.q_proj(states).view(batch, seq, heads, head_dim).transpose(1, 2)
        k = self.module.k_proj(states).view(batch, seq, heads, head_dim).transpose(1, 2)
        v = self.module.v_proj(states).view(batch, seq, heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1)
        scores = scores.masked_fill(future, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        return self.module.o_proj(out.transpose(1, 2).contiguous().view(batch, seq, self.HIDDEN))

    def check_exact_when_budget_covers_blocks(self) -> None:
        """One block of size SEQ with a budget of one block reads exactly the causal set."""
        with torch.no_grad():
            sparse = self.module(self.states, None)
            dense = self._dense_causal(self.states)
        delta = float((sparse - dense).abs().max())
        self.record("exact::equals dense causal", delta < self.TOLERANCE,
                    f"max|diff| = {delta:.3e}")

    def check_causality_is_structural(self) -> None:
        """A later token must not reach an earlier output, selection map included."""
        perturbed = self.states.clone()
        perturbed[:, self.SEQ - 3] += 5.0
        with torch.no_grad():
            base = self.module(self.states, None)
            moved = self.module(perturbed, None)
        spread = (moved - base).abs().amax(dim=(0, 2))
        frozen = float(spread[:self.SEQ - 3].max())
        later = float(spread[self.SEQ - 3:].max())
        self.record("causal::past bit-identical", frozen == 0.0,
                    f"max past drift = {frozen:.3e}")
        self.record("causal::future reacts", later > 0.0,
                    f"max future drift = {later:.3e}")

    def check_fully_masked_row_is_finite(self) -> None:
        """An entirely forbidden row must produce a finite output, never NaN."""
        forbidden = torch.full((self.BATCH, self.SEQ), float('-inf'))
        forbidden[0, ::2] = 0.0
        with torch.no_grad():
            out = self.module(self.states, forbidden)
        self.record("finite::no NaN on a dead row", bool(torch.isfinite(out).all()),
                    f"sample 1 max|out| = {float(out[1].abs().max()):.3e}")

    def check_mask_is_honoured(self) -> None:
        """Causal is idempotent; a mask conflicting with causality still changes the result."""
        with torch.no_grad():
            base = self.module(self.states, None)
            causal = torch.full((self.SEQ, self.SEQ), -1e4)
            causal.triu_(1)
            redundant = float((base - self.module(self.states, causal)).norm())
            past_forbidden = torch.zeros(self.SEQ, self.SEQ)
            past_forbidden[:, 0] = -1e4
            honoured = float((base - self.module(self.states, past_forbidden)).norm())
        self.record("mask::causal is idempotent", redundant == 0.0,
                    f"|delta| = {redundant:.3e}")
        self.record("mask::non-redundant mask applied", honoured > 1e-6,
                    f"|delta| = {honoured:.3e} when a past key is forbidden")

    def run(self) -> int:
        """Execute every check; non-zero exit on any failure."""
        self.check_exact_when_budget_covers_blocks()
        self.check_causality_is_structural()
        self.check_fully_masked_row_is_finite()
        self.check_mask_is_honoured()
        failed = [name for name, ok, _ in self.results if not ok]
        for name, ok, detail in self.results:
            print(f"[{'PASS' if ok else 'FAIL'}] {name:<40} {detail}")
        print(f"\n{len(self.results) - len(failed)}/{len(self.results)} checks passed")
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(PiscesLxBlockSparseAttentionContract().run())
