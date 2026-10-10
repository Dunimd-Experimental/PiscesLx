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
Subconscious Block Field — the "查" (lookup) half of compute-search separation.

Design and proofs: ``docs/design/00-subconscious-block-field.md``,
``docs/proofs/00-sbf-theorems.md`` (Lemma 0, Theorems 1-6).

A block is an **associative memory read**, not a data record: it holds ``M`` memory
rows of width ``d`` plus query/key/value/output projections, and produces a bounded
multiplicative modulation of the trunk hidden state. Every parameter participates at
activation time, so ``block_params`` is capacity that is actually used rather than dead
weights.

Two invariants enforced by construction (they are the premises of Theorem 5):
  * ``‖γ‖_∞ ≤ γ_max`` — γ is a scaled ``tanh``, so the bound holds for any input.
  * ``‖β‖ ≤ c·‖h‖`   — β is renormalised onto the sphere and rescaled by ``c·‖h‖``.
Therefore ``‖Δh‖ ≤ α(γ_max + c)‖h‖ = κ‖h‖``: retrieving a *wrong* block can only
under-help, never blow the trunk up.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .block_addresser import YvResonantTraceAddresser
from ..utils.device import YvDeviceResolver

__all__ = [
    "YvKnowledgeBlockModule",
    "YvResidentBlockPool",
    "YvSubconsciousBlockField",
]


class YvKnowledgeBlockModule(nn.Module):
    """One 0.5B-class knowledge block: associative memory + bounded modulation head.

    Args:
        hidden_dim: trunk width ``d``.
        memory_rows: ``M``, rows of associative memory; contributes ``M·d`` parameters.
        rank: internal projection rank ``r`` for the q/k/v/output head.
        gamma_max: bound on the multiplicative gate (assumption A1 of Theorem 5).
        beta_bound: bound constant ``c`` on the additive branch, relative to ``‖h‖``.
        rank: width of the scoring bottleneck; values are read at full ``hidden_dim``.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_rows: int,
        rank: int,
        gamma_max: float,
        beta_bound: float,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_rows = memory_rows
        self.rank = rank
        self.gamma_max = float(gamma_max)
        self.beta_bound = float(beta_bound)

        # Associative memory: each row is a stored knowledge slot, read by attention.
        # torch.empty would leave the row contents undefined; an all-zero read silently
        # produces zero modulation, i.e. a block that looks loaded but knows nothing.
        self.memory = nn.Parameter(torch.empty(memory_rows, hidden_dim, dtype=dtype))
        nn.init.normal_(self.memory, mean=0.0, std=hidden_dim ** -0.5)
        self.memory_norm = nn.LayerNorm(hidden_dim, dtype=dtype)

        # Rank-bottleneck scoring: <W_q h, W_k m> over all M rows.
        self.query = nn.Linear(hidden_dim, rank, bias=False, dtype=dtype)
        self.key = nn.Linear(hidden_dim, rank, bias=False, dtype=dtype)
        # Read values are memory rows (hidden_dim wide); one projection writes them back.
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)

        self._log_temperature = nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=True)
        self.score_floor = 1.0 / max(memory_rows, 1)
        self.beta_epsilon = 1e-6
        self.score_scale = 1.0 / max(rank, 1) ** 0.5

    @property
    def param_count(self) -> int:
        """Total parameters — must match ``block_params`` for the capacity claim to hold."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read memory and emit the bounded modulation ``(gamma, beta)`` for ``hidden``."""
        squeeze = hidden.dim() == 2
        if squeeze:
            hidden = hidden.unsqueeze(1)
        batch, seq, dim = hidden.shape
        if dim != self.hidden_dim:
            raise ValueError(f"block expects dim {self.hidden_dim}, got {dim}")


        work = hidden.to(self.memory.dtype)
        normalized = self.memory_norm(self.memory)
        query = self.query(work) * self.score_scale
        key = self.key(normalized)
        scores = torch.einsum("bsr,mr->bsm", query, key) * self._temperature()
        # Softmax over up to ~1e5 memory rows is done in FP32: the running sums of an
        # FP16 softmax overflow long before the distribution is normalised.
        weights = torch.softmax(scores.float(), dim=-1).to(self.memory.dtype)
        read = torch.einsum("bsm,md->bsd", weights, normalized)
        projected = self.out(read)

        # Premise A1(i): the gate is bounded for every input, by construction.
        gamma = self.gamma_max * torch.tanh(projected)

        # Premise A1(ii): beta is placed on a sphere of radius c*||h||, so ||beta|| = c*||h||.
        norm_read = projected.norm(dim=-1, keepdim=True)
        norm_hidden = work.norm(dim=-1, keepdim=True)
        # An additive denominator, not clamp_min: d/dx [x/(|x|)] is undefined at x = 0 and
        # produced NaN gradients once the read vector collapsed. x/(|x| + eps) is smooth
        # everywhere and still satisfies |beta| <= c*|h| because |x|/(|x|+eps) < 1.
        beta = self.beta_bound * norm_hidden * projected / (norm_read + self.beta_epsilon)
        if squeeze:
            return gamma.squeeze(1), beta.squeeze(1)
        return gamma, beta

    def _temperature(self) -> torch.Tensor:
        """Learned score temperature, floored so FP16 scores cannot overflow."""
        return self._log_temperature.exp().clamp_min(self.score_floor)


class YvResidentBlockPool(nn.Module):
    """Device-resident working set of ``slots`` blocks with λ-biased LRU eviction.

    Only pool members can receive gradients (they are the *activated* blocks). Blocks
    outside the pool hold their last committed weights on the host and contribute
    exactly zero gradient for the step — "not yet updated", never "frozen".
    """

    def __init__(
        self,
        slots: int,
        block_factory,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if slots < 1:
            raise ValueError(f"pool needs at least 1 slot, got {slots}")
        self.slots = slots
        self.dtype = dtype
        self.device = YvDeviceResolver.resolve(device)
        self._block_factory = block_factory
        # OrderedDict gives O(1) move-to-end, i.e. LRU recency.
        self._order: "OrderedDict[int, float]" = OrderedDict()
        self._slots: Dict[int, Optional[YvKnowledgeBlockModule]] = {
            index: None for index in range(slots)
        }
        self._by_block: Dict[int, int] = {}
        self.stats: Dict[str, float] = {"hits": 0, "misses": 0, "evictions": 0,
                                        "switches": 0, "tokens": 0}

    def resident_block_ids(self) -> List[int]:
        """Block ids currently on device."""
        return sorted(self._by_block)

    def slot_module(self, block_id: int) -> Optional[YvKnowledgeBlockModule]:
        """Return the device-resident module for ``block_id`` if present."""
        slot = self._by_block.get(block_id)
        if slot is None:
            return None
        return self._slots[slot]

    def touch(self, block_id: int) -> None:
        """Record recency for λ-biased eviction (recently used = stickier)."""
        self._order.pop(block_id, None)
        self._order[block_id] = time.time()

    def install(self, block_id: int, module: YvKnowledgeBlockModule,
                evict_callback=None) -> Optional[int]:
        """Place a block on device, evicting the least-sticky slot when full.

        Returns the evicted block id (to be committed back to the host), else None.
        """
        if block_id in self._by_block:
            self.touch(block_id)
            return None
        victim_slot: Optional[int] = None
        for slot, module_present in self._slots.items():
            if module_present is None:
                victim_slot = slot
                break
        evicted: Optional[int] = None
        if victim_slot is None:
            # Ordered by recency, so popitem(last=False) yields the least-sticky resident
            # block. The dict maps block_id -> recency, while _by_block maps block_id -> slot;
            # conflating them was the cause of a KeyError on eviction.
            victim_block, _recency = self._order.popitem(last=False)
            victim_slot = self._by_block.pop(victim_block)
            evicted = victim_block
            self.stats["evictions"] += 1
            if evict_callback is not None:
                evict_callback(victim_block, self._slots[victim_slot])
        self._slots[victim_slot] = module.to(self.device)
        module.block_id = block_id  # type: ignore[attr-defined]
        self._by_block[block_id] = victim_slot
        self.touch(block_id)
        return evicted

    def observe_token(self, block_id: int, previous: Optional[int]) -> bool:
        """Update switch/dwell statistics — Theorem 2 is measured, not assumed."""
        self.stats["tokens"] += 1
        switched = previous is not None and block_id != previous
        if switched:
            self.stats["switches"] += 1
            self.stats["misses"] += 1 if block_id not in self._by_block else 0
        else:
            self.stats["hits"] += 1
        return switched

    def switch_rate(self) -> float:
        """Measured ``p``; Theorem 2 requires ``p ≤ budget / transfer_time``."""
        tokens = self.stats["tokens"]
        if tokens <= 1:
            return 0.0
        return self.stats["switches"] / (tokens - 1)

    def mean_dwell(self) -> float:
        """Measured ``D = 1/p``; Theorem 2 requires ``D ≥ 16.7`` at 5 ms/token."""
        rate = self.switch_rate()
        return float("inf") if rate == 0.0 else 1.0 / rate

    def forward(self, block_id: int, hidden: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Apply the resident block, or return None when it is not on device."""
        module = self.slot_module(block_id)
        if module is None:
            return None
        return module(hidden)


class YvSubconsciousBlockField(nn.Module):
    """Whole "查" subsystem: host backing field + resident pool + addresser wiring.

    The backing store holds block *parameter payloads* on host (pinned when available);
    these are the model's own weights, saved with the checkpoint — not an external
    database. The resident pool is the only part on device, sized by the memory budget.
    """

    def __init__(
        self,
        cfg,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.device = YvDeviceResolver.resolve(device)
        self.block_count: int = int(cfg.block_count)
        self.memory_rows: int = int(cfg.block_memory_rows)
        self.hidden_dim: int = int(cfg.hidden_size)
        self.rank: int = int(cfg.block_rank)
        self.gamma_max: float = float(cfg.block_gamma_max)
        self.beta_bound: float = float(cfg.block_beta_bound)
        self.alpha: float = float(cfg.block_inject_alpha)
        self._host_payloads: Dict[int, Dict[str, torch.Tensor]] = {}
        self._transfer_lock = threading.Lock()
        self._inflight: Dict[int, Any] = {}
        self._last_block_id: Optional[int] = None
        self.pool = YvResidentBlockPool(
            slots=int(cfg.block_pool_slots),
            block_factory=self._make_block,
            dtype=dtype,
            device=self.device,
        )
        # One trace field only: the addresser owns it. Duplicating it here would give
        # two sources of truth for the same addressing manifold.
        self.addresser = YvResonantTraceAddresser(
            hidden_dim=self.hidden_dim,
            trace_count=self.block_count,
            trace_dim=int(cfg.block_trace_dim),
            hysteresis=float(cfg.block_hysteresis),
            staleness_gain=float(cfg.block_staleness_gain),
            staleness_cap_tokens=int(cfg.block_staleness_cap_tokens),
            min_dwell_tokens=int(cfg.block_min_dwell_tokens),
            dtype=dtype,
        ).to(self.device)

    def _make_block(self) -> YvKnowledgeBlockModule:
        return YvKnowledgeBlockModule(
            hidden_dim=self.hidden_dim,
            memory_rows=self.memory_rows,
            rank=self.rank,
            gamma_max=self.gamma_max,
            beta_bound=self.beta_bound,
            dtype=self.dtype,
        )

    def register_host_block(self, block_id: int, state: Dict[str, torch.Tensor]) -> None:
        """Store a block's parameter payload in the backing field (host memory)."""
        with self._transfer_lock:
            self._host_payloads[block_id] = {
                key: value.detach().cpu().clone() for key, value in state.items()
            }

    def commit_block(self, block_id: int, module: YvKnowledgeBlockModule) -> None:
        """Write a resident block's (possibly updated) weights back to the host field.

        This is ABGB item 1 (block-serial commit): a block leaves device only after its
        weights and optimiser payload have been written back, so no block is ever
        concurrently owned by two updaters.
        """
        self.register_host_block(block_id, module.state_dict())

    def load_block(self, block_id: int) -> Optional[YvKnowledgeBlockModule]:
        """Instantiate a block on device and load its host payload, if present."""
        payload = self._host_payloads.get(block_id)
        module = self._make_block()
        if payload is not None:
            # strict=True on purpose: a silently partial load is how a block ends up
            # "resident" while carrying no knowledge at all.
            module.load_state_dict(payload, strict=True)
        return module.to(self.device)

    def address(self, hidden: torch.Tensor) -> int:
        """Select the block for one hidden state (delegates to the addresser)."""
        block_id, _ = self.addresser.select(hidden)
        return block_id

    def address_sequence(self, hidden: torch.Tensor) -> torch.Tensor:
        """Dwell-guaranteed per-token selection for ``hidden [B, T, d]``."""
        selected, _ = self.addresser.select_sequence(hidden)
        return selected

    def modulate(self, hidden: torch.Tensor, block_id: int) -> torch.Tensor:
        """Apply ``h + α(γ ⊙ h + β)`` for one block, honouring Theorem 5's bound."""
        output = self.pool(block_id, hidden)
        if output is None:
            return hidden
        gamma, beta = output
        base = hidden if hidden.dim() == gamma.dim() else hidden.unsqueeze(1)
        delta = self.alpha * (gamma * base + beta)
        return base + delta

    def relative_perturbation_bound(self) -> float:
        """``κ = α(γ_max + c)`` — the guaranteed worst-case relative change (Theorem 5)."""
        return self.alpha * (self.gamma_max + self.beta_bound)

    def read_and_modulate(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Select a block per token, make it resident, and apply the modulation.

        Single application per forward, not per layer: a block read costs ~1.1e9 FLOPs per
        token at full scale (see docs/design 00 §3.2), which is 8.6% of the 1500 tok/s
        budget; repeating it every layer would forfeit the throughput target outright.

        Args:
            hidden: trunk states ``[B, T, d]``.

        Returns:
            The modulated states and a diagnostics dict (selection, pool statistics).
        """
        if hidden.dim() != 3:
            raise ValueError(f"read_and_modulate expects [B,T,d], got {tuple(hidden.shape)}")
        selected, _ = self.addresser.select_sequence(hidden.to(self.pool.dtype))
        batch, seq = selected.shape
        working = hidden.clone()

        # Group positions by block so each resident block is applied in one batched pass.
        flat_ids = selected.reshape(-1)
        unique_ids = torch.unique(flat_ids)
        for block_tensor in unique_ids:
            block_id = int(block_tensor.item())
            module = self._ensure_resident(block_id)
            if module is None:
                continue
            mask = (flat_ids == block_tensor)
            positions = mask.nonzero(as_tuple=True)[0]
            gathered = working.reshape(-1, self.hidden_dim)[positions]
            gamma, beta = module(gathered)
            delta = self.alpha * (gamma * gathered + beta)
            updated = working.reshape(-1, self.hidden_dim).index_copy(
                0, positions, gathered + delta
            )
            working = updated.reshape(batch, seq, self.hidden_dim)

        diagnostics = {
            "selected": selected,
            "distinct_blocks": int(unique_ids.numel()),
            "resident_blocks": self.pool.resident_block_ids(),
            "switch_rate": self.pool.switch_rate(),
            "mean_dwell": self.pool.mean_dwell(),
        }
        return working, diagnostics

    def _ensure_resident(self, block_id: int) -> Optional[YvKnowledgeBlockModule]:
        """Return the device-resident module for ``block_id``, loading it if needed.

        Transfer is synchronous here; the CUDA-stream pipeline of design §4 is the next
        step, which is what turns the 83 ms whole-block move into an amortised cost.
        """
        module = self.pool.slot_module(block_id)
        if module is not None:
            return module
        loaded = self.load_block(block_id)
        if loaded is None:
            return None
        self.pool.install(block_id, loaded, evict_callback=self.commit_block)
        return self.pool.slot_module(block_id)

    def measurable_capacity(self) -> Dict[str, float]:
        """Report capacity/participation so the 314B claim can be checked, not asserted."""
        resident_params = sum(
            module.param_count for module in self.pool._slots.values()
            if module is not None
        )
        return {
            "block_count": float(self.block_count),
            "params_per_block": float(self.memory_rows * self.hidden_dim
                                      + 2 * self.hidden_dim * self.rank
                                      + self.hidden_dim * self.hidden_dim),
            "equivalent_capacity": float(self.block_count * self.memory_rows * self.hidden_dim),
            "resident_params": float(resident_params),
            "host_blocks": float(len(self._host_payloads)),
            "switch_rate": self.pool.switch_rate(),
            "mean_dwell": self.pool.mean_dwell(),
        }
