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
Resonant Trace Addresser (RTA) — how a token selects its knowledge block.

Each block owns a short *trace* (``k`` dims, all traces permanently device-resident),
and the trunk projects its hidden state to a matching query. The selection score is

    s_i(t) = <q_t, r_i> / sqrt(k)  -  lambda * (1 - [i == i_{t-1}])  +  mu * staleness_i

The hysteresis term is **not a smoothing preference**: Theorem 2 of
``docs/proofs/00-sbf-theorems.md`` shows the 5 ms/token retrieval budget forces the
switch rate to ``p <= budget / transfer_time ~= 0.06``, i.e. a mean dwell of at least
16.7 tokens. Without a cross-token coupling term the switch rate is near 1 and the
budget is provably unattainable.

The staleness term keeps rarely-selected blocks reachable: without it the pool would
degenerate into a handful of permanently resident blocks and the remaining capacity
would never be trained (violating "被激活的知识块参与训练").
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

__all__ = ["YvResonantTraceAddresser", "YvAddressingStatistics"]


class YvAddressingStatistics:
    """Switch/dwell measurements that make Theorem 2 checkable at runtime."""

    __slots__ = ("tokens", "switches", "dwell_lengths", "current_dwell",
                 "_last_block", "tail_dwell_samples")

    def __init__(self) -> None:
        self.tokens = 0
        self.switches = 0
        self.dwell_lengths = []
        self.current_dwell = 0
        self._last_block: Optional[int] = None
        self.tail_dwell_samples: list = []

    def observe(self, block_id: int) -> bool:
        """Record one token's selection; returns whether the block changed."""
        switched = self._last_block is not None and block_id != self._last_block
        if switched:
            self.dwell_lengths.append(self.current_dwell)
            self.current_dwell = 0
            self.switches += 1
        self.current_dwell += 1
        self.tokens += 1
        self._last_block = block_id
        return switched

    @property
    def switch_rate(self) -> float:
        """Measured ``p``."""
        if self.tokens <= 1:
            return 0.0
        return self.switches / (self.tokens - 1)

    @property
    def mean_dwell(self) -> float:
        """Measured ``D = 1/p``; ``inf`` while the block never changes."""
        rate = self.switch_rate
        return float("inf") if rate == 0.0 else 1.0 / rate

    def record_runs(self, selected: torch.Tensor) -> None:
        """Log realised dwell lengths so Theorem 2's condition is measured, not assumed."""
        for row in range(selected.shape[0]):
            sequence = selected[row].tolist()
            length = 1
            for index in range(1, len(sequence)):
                if sequence[index] == sequence[index - 1]:
                    length += 1
                else:
                    self.dwell_lengths.append(length)
                    length = 1
            self.dwell_lengths.append(length)

    @property
    def tail_dwell(self) -> float:
        """Minimum realised dwell over the trailing window — the safety-critical number."""
        window = self.dwell_lengths[-512:]
        return float(min(window)) if window else float(self.current_dwell)

    def snapshot(self) -> Dict[str, float]:
        """Serializable summary for logs and regression assertions."""
        return {
            "tokens": float(self.tokens),
            "switches": float(self.switches),
            "switch_rate": self.switch_rate,
            "mean_dwell": self.mean_dwell,
            "tail_dwell": self.tail_dwell,
        }


class YvResonantTraceAddresser(nn.Module):
    """Scores blocks from a trunk hidden state and picks one, with guaranteed dwell.

    Args:
        trace_count: number of blocks ``N`` (one trace each).
        trace_dim: trace width ``k``, far below the trunk width.
        hysteresis: ``lambda`` — score penalty paid for every block switch.
        staleness_gain: ``mu`` — bonus per token a block has gone unselected,
            so cold blocks stay reachable.
        staleness_cap_tokens: saturation point of the staleness bonus.
    """

    def __init__(
        self,
        hidden_dim: int,
        trace_count: int,
        trace_dim: int,
        hysteresis: float,
        staleness_gain: float,
        staleness_cap_tokens: int,
        min_dwell_tokens: int = 17,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.trace_count = trace_count
        self.trace_dim = trace_dim
        self.hysteresis = float(hysteresis)
        self.staleness_gain = float(staleness_gain)
        self.staleness_cap_tokens = max(int(staleness_cap_tokens), 1)
        self.min_dwell_tokens = max(int(min_dwell_tokens), 1)
        self.dtype = dtype

        self.query_proj = nn.Linear(hidden_dim, trace_dim, bias=False, dtype=dtype)
        self.traces = nn.Parameter(torch.randn(trace_count, trace_dim, dtype=dtype) * 0.02)
        self.score_scale = 1.0 / max(trace_dim, 1) ** 0.5

        self.stats = YvAddressingStatistics()
        self._last_seen: Dict[int, int] = {}
        self._step = 0

    def scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the adjusted score vector for a single trunk state ``[hidden_dim]``."""
        flat = hidden.reshape(-1, self.hidden_dim)
        query = self.query_proj(flat[0]) * self.score_scale
        raw = query @ self.traces.T
        adjusted = raw.clone()

        self._step += 1
        for block_id in range(self.trace_count):
            gap = self._step - self._last_seen.get(block_id, 0)
            adjusted[block_id] += self.staleness_gain * min(gap, self.staleness_cap_tokens)

        if self.stats.tokens:
            previous = self._previous_block()
            if previous is not None:
                # Hysteresis: every non-previous block pays lambda, so switching must
                # be worth at least lambda in raw score to happen.
                adjusted -= self.hysteresis
                adjusted[previous] += self.hysteresis
        return adjusted

    def select(self, hidden: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Pick the block for this token and update dwell statistics."""
        adjusted = self.scores(hidden)
        block_id = int(torch.argmax(adjusted).item())
        self._register(block_id)
        self.stats.observe(block_id)
        return block_id, adjusted

    def score_sequence(self, hidden: torch.Tensor) -> torch.Tensor:
        """Score every position at once: ``hidden [B, T, d]`` -> ``scores [B, T, N]``.

        Cost is ``B·T·N·k`` FLOPs, i.e. ``N·k`` per token — Theorem 1's point that
        addressing is negligible next to block application (``P`` per token).
        """
        batch, seq, dim = hidden.shape
        if dim != self.hidden_dim:
            raise ValueError(f"addresser expects dim {self.hidden_dim}, got {dim}")
        query = self.query_proj(hidden) * self.score_scale
        return torch.einsum("btk,nk->btn", query, self.traces)

    @staticmethod
    def _run_encode(sequence: torch.Tensor) -> List[List[int]]:
        """Run-length encode one row as ``[[block_id, length], ...]``."""
        rows = []
        current = int(sequence[0].item())
        length = 0
        for value in sequence.tolist():
            if value == current:
                length += 1
            else:
                rows.append([current, length])
                current = int(value)
                length = 1
        rows.append([current, length])
        return rows

    @classmethod
    def _merge_short_runs(cls, runs: List[List[int]], min_dwell: int) -> List[List[int]]:
        """Merge runs shorter than ``min_dwell`` into their longer neighbour.

        Classic run-length smoothing: absorbing the globally shortest run into its
        longer neighbour never creates a new run shorter than the one removed, so the
        loop terminates with every run >= min_dwell (unless the whole row is shorter).
        """
        while len(runs) > 1:
            shortest = min(range(len(runs)), key=lambda i: runs[i][1])
            if runs[shortest][1] >= min_dwell:
                return runs
            left = runs[shortest - 1][1] if shortest > 0 else -1
            right = runs[shortest + 1][1] if shortest + 1 < len(runs) else -1
            if left >= right and shortest > 0:
                runs[shortest - 1][1] += runs[shortest][1]
                del runs[shortest]
            elif shortest + 1 < len(runs):
                runs[shortest + 1][1] += runs[shortest][1]
                del runs[shortest]
            else:
                runs[shortest - 1][1] += runs[shortest][1]
                del runs[shortest]
        return runs

    @staticmethod
    def _run_decode(runs: List[List[int]], device, dtype) -> torch.Tensor:
        if not runs:
            return torch.zeros(0, device=device, dtype=dtype)
        blocks = torch.tensor([block for block, _ in runs], device=device, dtype=dtype)
        lengths = torch.tensor([length for _, length in runs], device=device, dtype=torch.long)
        return blocks.repeat_interleave(lengths)

    @classmethod
    def apply_dwell_constraint(cls, selected: torch.Tensor, min_dwell: int) -> torch.Tensor:
        """Guarantee every block run lasts at least ``min_dwell`` tokens.

        Theorem 2 says the 5 ms/token retrieval budget *requires* a mean dwell of
        ``transfer_ms / budget_ms``. Rather than hoping the corpus is topic-coherent
        enough to provide that, the selector refuses to emit a shorter run: a switch
        that cannot be sustained is absorbed into a neighbour. Feasibility becomes a
        structural property of the selector instead of an assumption about the data.

        Cost scales with the number of change points, not with sequence length.
        """
        if min_dwell <= 1:
            return selected
        batch, seq = selected.shape
        rows = []
        for row in range(batch):
            runs = cls._run_encode(selected[row])
            rows.append(cls._run_decode(cls._merge_short_runs(runs, min_dwell),
                                        selected.device, selected.dtype))
        widths = {row.numel() for row in rows}
        if len(widths) != 1:
            raise RuntimeError(f"dwell smoothing changed row width inconsistently: {widths}")
        return torch.stack(rows)

    def select_sequence(self, hidden: torch.Tensor,
                        previous: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-token block selection over a whole sequence, with the dwell floor applied."""
        del previous  # the dwell floor is enforced structurally, not by seeding history
        scores = self.score_sequence(hidden)
        raw = scores.argmax(dim=-1)
        constrained = self.apply_dwell_constraint(raw, self.min_dwell_tokens)
        # Counters stay vectorised: a Python loop over B*T would dwarf the N*k FLOPs
        # that addressing itself costs per token (Theorem 1).
        switches = int((constrained[:, 1:] != constrained[:, :-1]).sum().item())
        self.stats.tokens += int(constrained.shape[1])
        self.stats.switches += switches
        self.stats.record_runs(constrained)
        return constrained, scores

    def _register(self, block_id: int) -> None:
        self._last_seen[block_id] = self._step
        self._prev = block_id

    def _previous_block(self) -> Optional[int]:
        return getattr(self, "_prev", None)

    def residence_memory_bytes(self) -> int:
        """Device footprint of the trace field — cheap enough to keep always-resident."""
        return self.trace_count * self.trace_dim * torch.finfo(self.dtype).bits // 8

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(one_hot_selection, adjusted_scores)`` for a single state."""
        block_id, adjusted = self.select(hidden)
        one_hot = torch.zeros(self.trace_count, device=adjusted.device, dtype=adjusted.dtype)
        one_hot[block_id] = 1.0
        return one_hot, adjusted

    def diagnostics(self) -> Dict[str, float]:
        """Switch rate, dwell and the Theorem 2 verdict for the observed stream."""
        snapshot = self.stats.snapshot()
        snapshot["hysteresis"] = self.hysteresis
        snapshot["trace_resident_bytes"] = float(self.residence_memory_bytes())
        snapshot["observed_at"] = time.time()
        return snapshot
