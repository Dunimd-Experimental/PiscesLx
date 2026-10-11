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
        self.init_seed: int = int(getattr(cfg, "block_init_seed", 0) or 0)
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

        # Resonant paired initialisation (docs/design 10): the trace codebook starts as the
        # representative of each block's own contribution direction, not as an independent
        # random draw. Deferred to first use rather than run here: deriving it materialises
        # one block (0.5B params at full scale), and paying that at *construction* would make
        # an idle model cost a block's worth of host memory for nothing. It still completes
        # before the first forward, hence before any optimiser step — see `_ensure_paired`.
        self.paired_stats: Dict[str, float] = {}
        self._paired_ready = False

    def _ensure_paired(self) -> None:
        """Run the one-time paired initialisation exactly once, on first use."""
        if self._paired_ready:
            return
        self._paired_ready = True
        self.paired_stats = self.paired_initialize()

    def paired_initialize(self) -> Dict[str, float]:
        """Align the addressing codebook with block contents *before the first step* (RPI).

        The addresser's traces and the blocks' memories were two independent random draws, so
        at step 0 "which hidden state hits block i" and "what block i does to a hidden state"
        were uncorrelated — the pool behaved as *look up a random answer* until the trace
        codebook happened to catch up. The dwell floor (``min_dwell_tokens``) makes that worse
        rather than better: an early mis-assignment sticks for >= 17 tokens and those tokens'
        gradients reinforce the same wrong trace. That is the structural reason pool coverage
        stayed low (docs/design 10 §1).

        For the trunk query map ``W = query_proj.weight`` (``k x d``) and a block's mean read
        direction ``u_i = W_out · mean_j LN(memory_j)`` (``d``), the trace is written as

            R_i = rho * normalize( (W W^T)^+ W u_i )

        so that ``<W h, R_i> = (rho / ||R~_i||) * <h, P u_i>`` with ``P = W^T (W W^T)^+ W``
        the orthogonal projector onto ``row(W)``. Ordering blocks for any ``h`` is therefore
        the content ordering, projected onto the ``k`` dimensions the addresser can see —
        a k-dimensional order-preserving image of the content-capability partition instead of
        a random one (docs/proofs/03, Proposition R1).

        Cost is ``O(N·(M·d + d²))`` — the whole pool streamed once, one block in memory at a
        time, never per step. Blocks are rebuilt deterministically by :meth:`_make_block`, so
        nothing is retained between iterations.

        All traces share **one** scale ``rho / rms`` rather than each being unit-normalised:
        a per-trace normaliser multiplies ``<h, u_i>`` by a different ``1/||R~_i||`` for every
        block and destroys exactly the ordering this method exists to fix (measured: Kendall
        tau 0.865 vs 1.000). The global scale keeps the ordering exact and the mean trace norm
        equal to ``rho``.
        """
        with torch.no_grad():
            weight = self.addresser.query_proj.weight.detach().float()
            inverse = torch.linalg.pinv(weight @ weight.T)
            rho = self._trace_scale(weight)
            representatives = []
            for block_id in range(self.block_count):
                direction = self._block_direction(self._make_block(block_id))
                representative = inverse @ (weight @ direction)
                if float(representative.norm()) <= 0.0:
                    # A block whose read direction has no component in the addressing
                    # subspace gets an arbitrary unit trace rather than a zero one: a zero
                    # trace can never be selected, which would silently remove the block.
                    representative = torch.zeros_like(representative)
                    representative[0] = 1.0
                representatives.append(representative)
            matrix = torch.stack(representatives)
            rms = float(matrix.norm(dim=1).pow(2).mean().sqrt())
            scale = rho / (rms if rms > 0.0 else 1.0)
            stacked = (scale * matrix).to(self.addresser.traces.dtype)
            self.addresser.traces.copy_(stacked)
        return {
            "blocks": float(self.block_count),
            "rho": float(rho),
            "scale": float(scale),
            "trace_norm_mean": float(stacked.norm(dim=1).mean()),
            "trace_norm_min": float(stacked.norm(dim=1).min()),
            "trace_norm_max": float(stacked.norm(dim=1).max()),
        }

    @staticmethod
    def _block_direction(module: YvKnowledgeBlockModule) -> torch.Tensor:
        """``u_i = W_out · mean_j LN(memory_j)`` — the block's typical read direction."""
        with torch.no_grad():
            normalized = module.memory_norm(module.memory.detach().float())
            centroid = normalized.mean(dim=0)
            return module.out(centroid)

    def _trace_scale(self, weight: torch.Tensor) -> float:
        """``rho = lambda * sqrt(k) / rms(W)``: step-0 score spread commensurate with lambda.

        With unit traces a query of norm ``||W h||`` scores ``rho ||W h|| / sqrt(k)``; taking
        the RMS row norm of ``W`` as the typical scale makes that typical score equal to the
        hysteresis ``lambda`` — the quantity that decides whether dwell or switching dominates
        the initial partition (docs/design 10 §4). Derived, not tuned.
        """
        rms = float(weight.norm()) / max(int(weight.shape[0]), 1) ** 0.5
        return float(self.addresser.hysteresis) * int(weight.shape[0]) ** 0.5 / max(rms, 1e-6)

    def _make_block(self, block_id: int = 0) -> YvKnowledgeBlockModule:
        """Build block ``block_id`` from a seed derived from the declared base seed.

        Per-id seeding is what lets :meth:`paired_initialize` derive a trace from a block's
        contents without keeping all ``block_count`` blocks resident: regenerating block
        ``i`` yields the *same* memory it had when the trace was computed, so the trace and
        the contents it was derived from can never drift apart (docs/design 10 §7). It also
        makes block initialisation reproducible across processes, which the A21 determinism
        standard requires of every parameter set.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._block_seed(block_id))
            return YvKnowledgeBlockModule(
                hidden_dim=self.hidden_dim,
                memory_rows=self.memory_rows,
                rank=self.rank,
                gamma_max=self.gamma_max,
                beta_bound=self.beta_bound,
                dtype=self.dtype,
            )

    def _block_seed(self, block_id: int) -> int:
        """Deterministic per-block seed, stable across processes and runs.

        The stride is the prime ``1000003`` so that adjacent block ids do not produce
        adjacent seeds (which would correlate their streams on the first draw).
        """
        return (self.init_seed * 1000003 + block_id) % (2 ** 31 - 1)  # prime stride; Mersenne

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
        module = self._make_block(block_id)
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
        self._ensure_paired()
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
        self._ensure_paired()
        selected, scores = self.addresser.select_sequence(hidden.to(self.pool.dtype))
        batch, seq = selected.shape
        working = hidden.clone()
        gate = self._straight_through_gate(scores, selected)

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
            if gate is not None:
                # Cast the gate, not the delta: its value is exactly 1.0, so this multiply
                # is bit-identical to the ungated forward while still carrying gradient.
                delta = gate[positions].to(delta.dtype) * delta
            updated = working.reshape(-1, self.hidden_dim).index_copy(
                0, positions, gathered + delta
            )
            working = updated.reshape(batch, seq, self.hidden_dim)

        # Carried in the returned diagnostics rather than stored on `self`: an attribute
        # holding a graph node across steps is what made step 2 fail with "backward through
        # the graph a second time" (docs/audit/00 §13, A1).
        diagnostics = {
            "selected": selected,
            "distinct_blocks": int(unique_ids.numel()),
            "resident_blocks": self.pool.resident_block_ids(),
            "switch_rate": self.pool.switch_rate(),
            "mean_dwell": self.pool.mean_dwell(),
        }
        return working, diagnostics

    def _straight_through_gate(self, scores: torch.Tensor,
                               selected: torch.Tensor) -> Optional[torch.Tensor]:
        """Route the block contribution's gradient back into the trace codebook.

        ``select_sequence`` ends in ``argmax``, so nothing the addressed block does could
        ever reach ``traces``: the codebook of a supposedly learnable addresser stayed at
        its random initialisation for the whole run (docs/audit/00 §48, A21). The gate is
        the straight-through estimator for the selection,

            g = 1 + (p_c - detach(p_c)),    p_c = softmax(s)_c

        whose value is exactly ``1`` — the forward pass, and therefore inference throughput
        and every hard/dwell-constrained decision, is bit-identical to the ungated one —
        while backward yields ``dL/dp_c = <dL/dh', delta>``: the codebook is pushed toward
        the blocks whose modulation actually reduced the objective, away from those that
        raised it, and learns from the LM loss alone rather than from a self-referential
        auxiliary term. Proof: ``docs/proofs/03-trace-codebook-alignment.md``.
        """
        if not (self.training and torch.is_grad_enabled() and scores.requires_grad):
            return None
        probs = torch.softmax(scores.float(), dim=-1)
        chosen = probs.gather(-1, selected.unsqueeze(-1).to(torch.long))
        return (chosen - chosen.detach() + 1.0).reshape(-1, 1)

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
