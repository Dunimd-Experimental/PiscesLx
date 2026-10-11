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
Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting
(arXiv:2502.12216, ICLR 2026).

Block-sparse attention with an adaptive per-query-block key budget, sized by a
cumulative attention-mass target instead of a fixed count. Three properties are
structural rather than incidental:

* **Causality is built in.** A query block reads key block ``j`` only if
  ``j <= i``, and inside a block only keys at or before the query position. No
  caller-supplied causal mask is needed to keep the layer autoregressive: a 2-D
  causal mask from the trunk reproduces exactly this constraint, so applying one
  is idempotent and never widens the result.
* **No fully masked row is ever softened.** The query's own key block is always
  selected, and a row the padding mask kills entirely falls back to its own key at
  weight one, so every softmax input keeps a finite entry. Softening an all
  ``-inf`` row would emit NaN into the residual stream and destroy the run
  (docs/audit/00 A54/A62).
* **The dense score matrix is never materialised.** Selection runs once per query
  block against pooled key blocks, and attention is evaluated only inside the
  selected blocks, so the quadratic term carries the factor
  ``blocks_per_query * block_size / seq_len`` (docs/proofs/05).
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


# Paper: Zhu et al., "Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs," ICLR 2026, arXiv:2502.12216
class YvTokenSparseAttention(nn.Module):
    """Adaptive block-sparse attention with a cumulative attention-mass target.

    Keys are partitioned into blocks of ``tactic_block_size`` and each block is
    described by the mean of its low-rank keys: pooling is the paper's clustering
    step made exact, since a block *is* the cluster. Every query block scores those
    representatives, keeps the smallest set of blocks whose score mass reaches
    ``tactic_cumulative_target``, and attends only inside that set.

    Reference: Tactic (ICLR 2026) - 5.14x decode attention speedup.
    """

    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.cumulative_target = getattr(config, 'tactic_cumulative_target', 0.8)
        self.n_bins = getattr(config, 'tactic_n_bins', 64)
        self.block_size = int(getattr(config, 'tactic_block_size', 64))
        self.blocks_per_query = int(getattr(config, 'tactic_blocks_per_query', 8))
        # Rank of the *selection* view. Scoring keys through a low-rank prefix is what
        # makes block selection cheaper than the attention it exists to gate.
        self.score_dim = max(1, self.head_dim // 4)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)

    def _key_bias(
        self, attention_mask: Optional[torch.Tensor], batch: int, seq_len: int, padded_len: int
    ) -> torch.Tensor:
        """Per-key additive bias ``[1, 1, 1, 1, padded_len]`` encoding padding only.

        A causal mask is deliberately *not* translated here: block-level causality
        plus the intra-block causal mask reproduce it exactly, whereas reading the
        first row of a causal mask would mark every key invalid. Masks that are
        neither padding-shaped nor constant along the query axis are ignored, and
        columns past ``seq_len`` stay forbidden regardless.
        """
        bias = torch.zeros(1, 1, 1, 1, padded_len, device=self.q_proj.weight.device,
                           dtype=self.q_proj.weight.dtype)
        if padded_len > seq_len:
            bias[..., seq_len:] = float('-inf')
        if attention_mask is None:
            return bias
        mask = attention_mask
        if mask.dim() < 2 or mask.shape[-1] != seq_len or mask.shape[-2] == seq_len:
            return bias
        per_key = mask.reshape(-1, seq_len)
        if per_key.shape[0] not in (1, batch):
            raise ValueError(
                f"padding mask batch {per_key.shape[0]} is neither 1 nor {batch}"
            )
        per_key = per_key.reshape(-1, 1, 1, 1, seq_len).to(bias.dtype)
        if padded_len > seq_len:
            per_key = F.pad(per_key, (0, padded_len - seq_len))
        return bias + per_key

    def _estimate_threshold_via_histogram(self, scores: torch.Tensor) -> torch.Tensor:
        """Return a per-row cumulative-mass threshold, tolerating ``-inf`` entries.

        ``-inf`` marks a forbidden candidate. Folding those onto the lowest bin keeps
        them out of the selection (they can never reach the threshold), while a row
        that is entirely forbidden gets a finite threshold of zero instead of a NaN
        composed from ``inf - inf``.
        """
        B, H, T_q, T_k = scores.shape
        scores_flat = scores.view(B * H * T_q, T_k)

        finite = torch.isfinite(scores_flat)
        any_finite = finite.any(dim=-1, keepdim=True)
        v_min = scores_flat.masked_fill(~finite, float('inf')).amin(dim=-1, keepdim=True)
        v_max = scores_flat.masked_fill(~finite, float('-inf')).amax(dim=-1, keepdim=True)
        v_min = torch.where(any_finite, v_min, torch.zeros_like(v_min))
        v_max = torch.where(any_finite, v_max, torch.zeros_like(v_max))
        range_safe = (v_max - v_min).clamp(min=1e-8)

        shifted = torch.where(finite, scores_flat, v_min)
        normalized = (shifted - v_min) / range_safe
        bin_idx = (normalized * (self.n_bins - 1)).long().clamp(0, self.n_bins - 1)

        bins = torch.zeros(scores_flat.shape[0], self.n_bins, device=scores.device)
        bins.scatter_add_(1, bin_idx, torch.ones_like(bin_idx, dtype=bins.dtype))

        bin_centers = (torch.arange(self.n_bins, device=scores.device).float() + 0.5) / self.n_bins
        bin_values = v_min + bin_centers.unsqueeze(0) * range_safe

        cumsum = bins.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        total = cumsum[:, :1].clamp(min=1)
        cum_frac = cumsum / total

        target_idx = (cum_frac >= self.cumulative_target).float().argmax(dim=-1, keepdim=True)
        target_idx = target_idx.clamp(0, self.n_bins - 1)
        threshold = bin_values.gather(1, target_idx)

        return threshold.view(B, H, T_q, 1)

    def _select_blocks(
        self, query_block_repr: torch.Tensor, key_block_repr: torch.Tensor,
        allowed: torch.Tensor, num_qblocks: int
    ) -> torch.Tensor:
        """Pick each query block's key blocks, always including the diagonal one.

        The returned ids have width ``min(blocks_per_query, num_qblocks)`` and are
        unique per row: the diagonal is forced in by replacing the worst-ranked slot
        whenever the threshold set did not already contain it. A duplicate would count
        that block's attention mass twice, so uniqueness is not cosmetic.
        """
        block_scores = torch.matmul(query_block_repr, key_block_repr.transpose(-2, -1))
        block_scores = block_scores / (self.score_dim ** 0.5)
        block_scores = block_scores.masked_fill(~allowed, float('-inf'))

        threshold = self._estimate_threshold_via_histogram(block_scores)
        keep = block_scores >= threshold

        width = max(1, min(self.blocks_per_query, num_qblocks))
        ranked = block_scores.masked_fill(~keep, float('-inf'))
        block_ids = torch.topk(ranked, k=width, dim=-1).indices

        axis = torch.arange(num_qblocks, device=block_scores.device)
        diagonal = axis.view(1, 1, num_qblocks, 1).expand(*block_scores.shape[:-1], 1)
        has_diagonal = (block_ids == diagonal).any(dim=-1, keepdim=True)
        forced = torch.cat([block_ids[..., :-1], diagonal], dim=-1)
        return torch.where(has_diagonal, block_ids, forced)

    def _additive_mask_bias(
        self, attention_mask: Optional[torch.Tensor], batch: int, seq_len: int,
        num_qblocks: int, block: int, width: int,
        key_global: torch.Tensor, query_global: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Gather a caller-supplied square additive mask onto the selected (query, key) pairs.

        A square mask is additive over pairs, not a key-side vector, so it cannot ride the
        padding lane (that would forbid every key) and it cannot be ignored either: it may
        carry ALiBi slopes or a window pattern. It is therefore gathered with the same ids the
        keys used. The gathered tensor has the score tensor's shape, which is the price of
        honouring arbitrary pair structure; positions past ``seq_len`` are clamped here and
        forbidden by the padding bias instead.
        """
        if attention_mask is None or attention_mask.dim() < 2:
            return None
        mask = attention_mask
        if tuple(mask.shape[-2:]) != (seq_len, seq_len):
            return None
        rows = mask.reshape(-1, seq_len, seq_len)
        if rows.shape[0] not in (1, batch):
            raise ValueError(
                f"square mask batch {rows.shape[0]} is neither 1 nor {batch}"
            )
        flat = rows.reshape(rows.shape[0], seq_len * seq_len)
        # The two indices live on different axes of the score tile: the query index varies over
        # the key slots (last axis), the key index over the queries inside a block (one before
        # it). Inserting the axes that way lets the arithmetic broadcast into ``[B, H, n, block,
        # width*block]`` instead of forcing both onto the same rank.
        q_index = query_global.reshape(1, 1, num_qblocks, block, 1).clamp(max=seq_len - 1)
        k_index = key_global.unsqueeze(-2).clamp(max=seq_len - 1)
        row_index = torch.zeros(1, 1, 1, 1, 1, dtype=torch.long, device=mask.device)
        if rows.shape[0] == batch:
            row_index = row_index + torch.arange(
                batch, device=mask.device).view(batch, 1, 1, 1, 1)
        offset = (row_index * (seq_len * seq_len) + q_index * seq_len + k_index).reshape(-1)
        gathered = flat.reshape(-1).gather(0, offset)
        return gathered.view(batch, self.num_heads, num_qblocks, block, width * block)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, T, H, D).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, T, H, D).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, T, H, D).transpose(1, 2)

        block = max(1, min(self.block_size, T))
        num_qblocks = -(-T // block)
        padded_len = num_qblocks * block
        if padded_len > T:
            # Queries are padded alongside keys: the block layout is a reshape of both axes,
            # so a query axis left at T would not divide by block_size.
            Q = F.pad(Q, (0, 0, 0, padded_len - T))
            K = F.pad(K, (0, 0, 0, padded_len - T))
            V = F.pad(V, (0, 0, 0, padded_len - T))

        # Selection view: mean-pooling the low-rank keys over a block is the paper's
        # clustering made exact (a block is the cluster), and it costs one reduction.
        lr = self.score_dim
        key_block_repr = K[..., :lr].view(B, H, num_qblocks, block, lr).mean(dim=3)

        # Query side uses only the *first* row of each block. Pooling the whole block would
        # let the selection for row 8 depend on rows 9..15, which is exactly the leak a
        # causal decoder must not have; one representative row is at or before every row in
        # its block, so the choice is causal by construction. The diagonal block is added
        # unconditionally below, so no query loses its own neighbourhood.
        query_block_repr = Q[:, :, ::block, :lr]

        axis = torch.arange(num_qblocks, device=hidden_states.device)
        block_start = (axis * block).view(1, 1, 1, num_qblocks)
        query_start = (axis * block).view(1, 1, num_qblocks, 1)
        # Strictly earlier blocks only: block i's own keys are never scored, which keeps the
        # selection independent of any key at or after the block's first query.
        allowed = block_start < query_start

        block_ids = self._select_blocks(query_block_repr, key_block_repr, allowed, num_qblocks)
        width = block_ids.shape[-1]

        key_bias = self._key_bias(attention_mask, B, T, padded_len)
        slot_index = torch.arange(block, device=hidden_states.device)
        key_global = block_ids.unsqueeze(-1) * block + slot_index.view(1, 1, 1, 1, block)
        key_global = key_global.reshape(B, H, num_qblocks, width * block)
        query_global = torch.arange(padded_len, device=hidden_states.device).view(
            1, 1, num_qblocks, block)
        intra_causal = key_global.unsqueeze(-2) <= query_global.unsqueeze(-1)

        scores = torch.empty(B, H, num_qblocks, block, width * block,
                             device=hidden_states.device, dtype=Q.dtype)
        values = torch.empty(B, H, num_qblocks, width * block, D,
                             device=hidden_states.device, dtype=V.dtype)

        # One slot at a time. Gathering every slot's keys at once would rebuild a
        # [B, H, T, width*block, D] temporary, which is the very term the block layout
        # exists to avoid.
        q_blocks = Q.view(B, H, num_qblocks, block, D)
        k_flat = K.reshape(B, H, padded_len, D)
        v_flat = V.reshape(B, H, padded_len, D)
        for slot in range(width):
            ids = block_ids[..., slot].unsqueeze(-1) * block + slot_index.view(1, 1, 1, block)
            ids = ids.reshape(B, H, num_qblocks * block).unsqueeze(-1).expand(
                B, H, num_qblocks * block, D)
            selected_k = k_flat.gather(2, ids).view(B, H, num_qblocks, block, D)
            selected_v = v_flat.gather(2, ids).view(B, H, num_qblocks, block, D)
            scores[..., slot * block:(slot + 1) * block] = torch.matmul(
                q_blocks, selected_k.transpose(-2, -1)) / (D ** 0.5)
            values[..., slot * block:(slot + 1) * block, :] = selected_v

        # The bias is indexed by *global* key position, while ``scores`` runs over the
        # selected keys only, so it has to be gathered by the same ids the keys used.
        key_bias = self._key_bias(attention_mask, B, T, padded_len).reshape(-1, padded_len)
        if key_bias.shape[0] == 1:
            key_bias = key_bias.expand(B, padded_len)
        key_bias = key_bias.gather(1, key_global.reshape(B, -1)).view(
            B, H, num_qblocks, width * block).unsqueeze(3)
        scores = scores + key_bias

        additive = self._additive_mask_bias(
            attention_mask, B, T, num_qblocks, block, width, key_global, query_global)
        if additive is not None:
            scores = scores + additive
        scores = scores.masked_fill(~intra_causal, float('-inf'))

        # A row whose every candidate is forbidden would soften to NaN. Its own key is in
        # the selected set by construction, so pin that single entry to zero and hand the
        # row the value of its own token at weight one.
        diagonal_slot = (block_ids == axis.view(1, 1, num_qblocks, 1)).float().argmax(dim=-1)
        row_pos = torch.arange(block, device=hidden_states.device).view(1, 1, 1, block)
        own_key = (diagonal_slot.unsqueeze(-1) * block + row_pos).unsqueeze(-1)
        own_key = own_key.expand(B, H, num_qblocks, block, 1)
        alive = torch.isfinite(scores).any(dim=-1, keepdim=True)
        own_present = torch.zeros_like(scores).scatter_(-1, own_key, 1.0) > 0
        scores = scores.masked_fill(~alive & own_present, 0.0)

        # fp16 softmax over long rows loses range on Turing; accumulate it in fp32 and
        # cast back, which is exact for the distribution actually applied to the values.
        attn = F.softmax(scores.float(), dim=-1).to(values.dtype)
        out = torch.matmul(attn, values).reshape(B, H, padded_len, D)
        out = out[:, :, :T, :].transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        return self.o_proj(out)
