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
Subconscious Knowledge System for PiscesLx/Yv.

Implements the 0.5B Dynamic Head + 314B-equivalent Implicit Knowledge Field
architecture for subconscious knowledge injection. The system separates
"conscious" reasoning (the 1M context window) from "subconscious" knowledge
(volatile, memory-address-speed knowledge retrieval that influences computation
without appearing in the token sequence).

Architecture Overview:
    1. YvImplicitKnowledgeField (314B-equivalent storage, ~0.27B actual params)
       - Product-quantized codebook structure for massive virtual capacity
       - Navigable via learned addressing, not fixed parameter indices
       - Knowledge is "where you are" in the field, not "what you store"

    2. YvDynamicHead (0.5B navigation head, ~0.23B params)
       - Memory-address-speed router over the knowledge field
       - Projects 7B hidden state into navigation coordinates
       - Selects and retrieves relevant knowledge in O(1)

    3. YvSubconsciousInjector (zero additional params on 7B)
       - Layer-wise modulation of 7B computation
       - NOT cross-attention, NOT token injection
       - FiLM-style: scale & shift each layer's hidden flow
       - Truly subconscious: the model never "sees" the knowledge as tokens

Key Design:
    - Volatile: knowledge is retrieved fresh each forward pass (like RAM)
    - Navigation-based: 0.5B head learns to "go to" the right knowledge
    - Parallel to context: subconscious channel is orthogonal to 1M context
    - Trainable via RL: reward = improvement in 7B reasoning quality

Memory Addressing Analogy:
    1M context = RAM (persistent, organized, addressable by position)
    Subconscious = CPU cache (fast, volatile, transparently influences execution)
    7B core = ALU (pure computation, doesn't store data)
    314B field = disk (massive, slow, paged in by the cache)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from utils.dtype_safe import qr_safe

_LOG = PiscesLxLogger("Yv.Subconscious", file_path=get_log_file("Yv.Subconscious"), enable_file=True)


# Paper: Original contribution by Dunimd Team (Yv Architecture — subconscious system)
class YvImplicitKnowledgeField(nn.Module):
    """314B-equivalent implicit knowledge field via product-quantized codebooks.

    Represents an enormous knowledge space using multiple small codebooks.
    The combined addressing space of all codebooks is equivalent to what a
    314B parameter model would encode, but the actual storage is only ~0.27B.

    Architecture:
        M codebooks each with K entries of dimension D.
        Knowledge is retrieved by selecting one entry from each codebook
        (via soft addressing) and combining them into a unified representation.

        Total virtual combinations: K^M
        With M=16, K=131072, D=128: 131072^16 ≈ 10^80 combinations

    Key Properties:
        - Navigable: similar addresses retrieve similar knowledge
        - Continuous: differentiable soft addressing enables gradient flow
        - Massive capacity: combinatorial explosion without parameter explosion
        - Volatile: retrieved knowledge is computed fresh each forward pass

    Addressing (fixed, was ~17B -> ~0):
        The field receives *query vectors* from the dynamic head, not pre-computed
        logits.  Logits are computed on the fly as dot products between queries
        and codebook entries, eliminating the 17B-parameter address_proj.

    Args:
        num_codebooks: Number of product-quantized codebooks (M).
        codebook_size: Number of entries per codebook (K).
        codebook_dim: Dimension of each codebook entry (D).
        knowledge_dim: Output dimension of retrieved knowledge.
        num_heads: Number of attention heads for multi-head addressing.
    """

    def __init__(
        self,
        num_codebooks: int = 16,
        codebook_size: int = 131072,
        codebook_dim: int = 128,
        knowledge_dim: int = 256,
        num_heads: int = 8,
        top_k_entries: int = 64,
        min_top_k_entries: int = 16,
        max_top_k_entries: int = 64,
        dynamic_topk_scale: float = 2048.0,
        score_chunk_size: int = 2048,
        query_chunk_size: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.knowledge_dim = knowledge_dim
        self.num_heads = num_heads
        self.top_k_entries = max(1, min(top_k_entries, codebook_size))
        self.min_top_k_entries = max(1, min(min_top_k_entries, codebook_size))
        self.max_top_k_entries = max(self.min_top_k_entries, min(max_top_k_entries, codebook_size))
        self.dynamic_topk_scale = max(1.0, float(dynamic_topk_scale))
        self.score_chunk_size = max(1, score_chunk_size)
        self.query_chunk_size = max(1, query_chunk_size)

        # Multi-head codebooks: each head has its own set of codebooks
        # This increases representational capacity without increasing K or M
        self.head_dim = knowledge_dim // num_heads

        # Codebook parameters: [num_heads, num_codebooks, codebook_size, codebook_dim]
        # Total params: num_heads * num_codebooks * codebook_size * codebook_dim
        # = 8 * 16 * 131072 * 128 ≈ 2.147B
        self.codebooks = nn.Parameter(
            torch.randn(num_heads, num_codebooks, codebook_size, codebook_dim, device=device, dtype=dtype)
            * 0.02
        )

        # Structured diversification: each head's codebook entries are
        # orthonormalised after random init so that the soft-addressing
        # logits span the full representational space from the start.
        self.reset_parameters(num_heads, num_codebooks, codebook_size, codebook_dim, device, dtype)

        # Output projection: combines multi-head knowledge into knowledge_dim
        self.output_proj = nn.Linear(
            knowledge_dim, knowledge_dim, bias=False, device=device, dtype=dtype
        )

        # Layer norm for output stability
        self.norm = nn.LayerNorm(knowledge_dim, device=device, dtype=dtype)

        _LOG.info(
            f"YvImplicitKnowledgeField: "
            f"{num_heads}x{num_codebooks}x{codebook_size}x{codebook_dim} "
            f"= {self._param_count():.2f}B actual, "
            f"{codebook_size ** num_codebooks:.1e} virtual combinations"
        )

    def _param_count(self) -> float:
        return self.codebooks.numel() / 1e9

    def forward(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve knowledge from the field given query vectors.

        Instead of receiving pre-computed logits (which would require a ~17B
        parameter projection), this method receives compact query vectors and
        computes logits as dot products with the codebook entries, then does
        soft-addressing and retrieval in one fused step.

        Args:
            queries: [batch, seq, num_heads, num_codebooks, codebook_dim]
                Query vectors produced by YvDynamicHead.

        Returns:
            knowledge: [batch, seq, knowledge_dim]
                Retrieved knowledge representation.
        """
        B, T, H, M, D = queries.shape

        # ── Step 1: compute logits as dot products ────────────────────
        # queries:     [B, T, H, M, D]
        # codebooks:   [H, M, K, D]
        # logits:      [B, T, H, M, K]
        logits = torch.einsum('bthmd,hmkd->bthmk', queries, self.codebooks)

        # ── Step 2: soft-addressing ──────────────────────────────────
        # Temperature annealing for sharper addressing during training
        if self.training:
            temp = max(0.5, 1.0 - self._get_training_progress() * 0.5)
            addressing_weights = F.softmax(logits / temp, dim=-1)
        else:
            addressing_weights = F.softmax(logits, dim=-1)  # [B, T, H, M, K]

        # ── Step 3: weighted sum retrieval ───────────────────────────
        # addressing_weights: [B, T, H, M, K]
        # codebooks:          [H, M, K, D]
        # retrieved:          [B, T, H, M, D]
        retrieved = torch.einsum(
            'bthmk,hmkd->bthmd',
            addressing_weights,
            self.codebooks
        )

        # ── Step 4: combine codebooks and heads ──────────────────────
        # Sum across codebooks within each head
        knowledge_per_head = retrieved.sum(dim=3)           # [B, T, H, D]
        # Concatenate heads → knowledge_dim  (= H * D)
        knowledge = knowledge_per_head.reshape(B, T, self.knowledge_dim)

        # Output projection and normalize
        knowledge = self.output_proj(knowledge)
        knowledge = self.norm(knowledge)

        return knowledge

    def _resolve_sparse_budget(self, seq_len: int) -> Tuple[int, int]:
        """Adapt retrieval budget to sequence length and training mode."""
        if self.training:
            target_top_k = int(self.dynamic_topk_scale / max(1, seq_len))
            target_top_k = max(self.min_top_k_entries, min(self.max_top_k_entries, target_top_k))
            target_top_k = min(target_top_k, self.top_k_entries, self.codebook_size)

            shrink = max(1, seq_len // max(1, int(self.dynamic_topk_scale)))
            query_chunk = max(32, self.query_chunk_size // shrink)
        else:
            target_top_k = min(self.max_top_k_entries, self.top_k_entries, self.codebook_size)
            query_chunk = self.query_chunk_size

        return target_top_k, max(1, query_chunk)

    def _sparse_retrieve(self, queries: torch.Tensor, temperature: float) -> torch.Tensor:
        """Chunked top-k retrieval to avoid materialising full [B, T, H, M, K] logits."""
        B, T, H, M, D = queries.shape
        bt = B * T
        hm = H * M
        top_k, query_chunk_size = self._resolve_sparse_budget(T)

        flat_queries = queries.reshape(bt, hm, D).transpose(0, 1).contiguous()
        flat_codebooks = self.codebooks.reshape(hm, self.codebook_size, D)
        retrieved = queries.new_zeros((hm, bt, D))

        for hm_idx in range(hm):
            query_hm = flat_queries[hm_idx]
            codebook_hm = flat_codebooks[hm_idx]

            for q_start in range(0, bt, query_chunk_size):
                q_end = min(q_start + query_chunk_size, bt)
                q_chunk = query_hm[q_start:q_end]
                top_vals = None
                top_indices = None

                for cb_start in range(0, self.codebook_size, self.score_chunk_size):
                    cb_end = min(cb_start + self.score_chunk_size, self.codebook_size)
                    cb_chunk = codebook_hm[cb_start:cb_end]
                    scores = q_chunk @ cb_chunk.transpose(0, 1)
                    local_k = min(top_k, cb_end - cb_start)
                    chunk_vals, chunk_idx = torch.topk(scores, k=local_k, dim=-1)
                    chunk_idx = chunk_idx + cb_start

                    if top_vals is None:
                        top_vals = chunk_vals
                        top_indices = chunk_idx
                    else:
                        merged_vals = torch.cat([top_vals, chunk_vals], dim=-1)
                        merged_idx = torch.cat([top_indices, chunk_idx], dim=-1)
                        top_vals, merged_pos = torch.topk(merged_vals, k=top_k, dim=-1)
                        top_indices = merged_idx.gather(-1, merged_pos)

                selected = codebook_hm.index_select(0, top_indices.reshape(-1))
                selected = selected.view(q_end - q_start, top_k, D)
                weights = F.softmax(top_vals / temperature, dim=-1)
                retrieved[hm_idx, q_start:q_end] = (weights.unsqueeze(-1) * selected).sum(dim=1)

        return retrieved.transpose(0, 1).reshape(B, T, H, M, D)

    def forward(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve knowledge from the field given query vectors with sparse lookup."""
        B, T, H, M, D = queries.shape
        temp = max(0.5, 1.0 - self._get_training_progress() * 0.5) if self.training else 1.0
        retrieved = self._sparse_retrieve(queries, temp)
        knowledge_per_head = retrieved.sum(dim=3)
        knowledge = knowledge_per_head.reshape(B, T, self.knowledge_dim)
        knowledge = self.output_proj(knowledge)
        knowledge = self.norm(knowledge)
        return knowledge

    def _get_training_progress(self) -> float:
        """Estimate training progress for temperature scheduling."""
        if hasattr(self, '_training_step'):
            return min(1.0, self._training_step / 50000)
        return 0.0

    def set_training_step(self, step: int):
        self._training_step = step

    def reset_parameters(
        self,
        num_heads: int,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Structured initialisation of the codebook parameters.

        Three strategies applied sequentially:

        1. **Random orthonormal rows** — each codebook matrix is initialised
           with a random orthogonal matrix so that the K entries span the
           D-dimensional space uniformly.  This prevents degenerate addressing
           where all queries collapse to the same logit pattern.

        2. **Per-head diversification** — each head receives an independent
           random seed, ensuring that different heads learn different
           knowledge decompositions from the start.

        3. **Zero-mean, unit-variance scaling** — after orthogonalisation the
           entries are scaled to have roughly unit variance along the
           codebook_dim axis, matching the query vectors' expected scale.
        """
        import math
        for h in range(num_heads):
            for m in range(num_codebooks):
                # Create a random Gaussian matrix
                W = torch.randn(codebook_size, codebook_dim, device=device, dtype=dtype)
                # QR decomposition → orthonormal rows (up to K > D tolerance).
                # `qr_safe` upcasts to fp32 internally so this works on
                # fp16/bf16 models where torch.linalg.qr's CUDA path is
                # not implemented.
                Q = qr_safe(W)
                # Ensure Q has the right shape (K x D)
                Q = Q[:, :codebook_dim]
                # Scale so that ‖row‖² ≈ codebook_dim (unit variance per dim)
                scale = math.sqrt(codebook_dim) * 0.02
                with torch.no_grad():
                    self.codebooks[h, m].copy_(Q * scale)

    def extra_repr(self) -> str:
        return (
            f"num_codebooks={self.num_codebooks}, "
            f"codebook_size={self.codebook_size}, "
            f"codebook_dim={self.codebook_dim}, "
            f"param_count={self._param_count():.3f}B"
        )


# Paper: Original contribution by Dunimd Team (Yv Architecture — dynamic navigation head)
class YvDynamicHead(nn.Module):
    """0.5B dynamic navigation head for implicit knowledge field addressing.

    Projects the 7B core's current reasoning state into compact query vectors
    that address the implicit knowledge field via dot-product (not a giant
    logit projection).  Designed for memory-address-speed operation with
    minimal latency overhead.

    Architecture (fixed parameter explosion):
        - Input projection: hidden_size -> head_dim
        - Lightweight transformer encoder (2 layers, 4 heads)
        - **Query projection**: head_dim -> num_heads * num_codebooks * codebook_dim
          (was head_dim -> num_heads * num_codebooks * codebook_size, which
           created a ~17B-parameter layer; now ~16.8M — 1000x reduction)
        - Logits are computed on-the-fly in YvImplicitKnowledgeField via
          einsum('bthmd,hmkd->bthmk') — no extra parameters needed.

    Key Properties:
        - Fast: O(1) routing via learned projection + lightweight processing
        - Context-aware: addressing depends on current reasoning state
        - Differentiable: enables end-to-end training with RL
        - Memory-address-speed: no iterative search, single forward pass

    Args:
        hidden_size: 7B core's hidden dimension.
        num_codebooks: Number of codebooks in the knowledge field.
        codebook_dim: Dimension of each codebook entry.
        num_heads: Number of heads in the knowledge field.
        head_dim: Internal dimension for the navigation head.
        num_layers: Number of lightweight transformer layers.
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        num_codebooks: int = 16,
        codebook_dim: int = 128,
        num_heads: int = 8,
        head_dim: int = 1024,
        num_layers: int = 2,
        num_attn_heads: int = 4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_codebooks = num_codebooks
        self.codebook_dim = codebook_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Input projection: 7B hidden -> head internal dimension
        self.input_proj = nn.Linear(hidden_size, head_dim, bias=False, device=device, dtype=dtype)
        self.input_norm = nn.LayerNorm(head_dim, device=device, dtype=dtype)

        # Lightweight transformer encoder for context-aware addressing
        # Using small dimensions to keep this at ~0.23B params total.
        #
        # NOTE: torch.nn.TransformerEncoderLayer only accepts 'relu' or
        # 'gelu' for its `activation` argument; SiLU/swish is not
        # supported. We pick 'gelu' as the closest smooth nonlinearity
        # to SiLU.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=head_dim,
            nhead=num_attn_heads,
            dim_feedforward=head_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Query projection (FIXED: was 17B, now ~16.8M) ──────────────
        # Instead of projecting to H*M*K logits (K=131072), project to
        # H*M*D compact query vectors (D=128).  The actual logits are
        # computed as dot products with codebook entries inside the
        # knowledge field — no extra parameters needed.
        #
        #   Before: head_dim * H * M * K = 1024 * 8 * 16 * 131072 ≈ 17.2B
        #   After:  head_dim * H * M * D = 1024 * 8 * 16 * 128   ≈ 16.8M
        self.query_proj = nn.Linear(
            head_dim,
            num_heads * num_codebooks * codebook_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # Query normalisation: stabilises the dot-product addressing
        self.query_norm = nn.RMSNorm(codebook_dim, device=device, dtype=dtype)

        # Context gating: decides how much subconscious to apply
        # Based on current reasoning uncertainty
        self.context_gate = nn.Sequential(
            nn.Linear(head_dim, head_dim // 4, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(head_dim // 4, 1, device=device, dtype=dtype),
            nn.Sigmoid(),
        )

        _LOG.info(
            f"YvDynamicHead: {self._param_count():.3f}B params, "
            f"hidden={hidden_size}, head_dim={head_dim}, "
            f"num_layers={num_layers}, "
            f"query_dim={num_codebooks}x{codebook_dim} (no more 17B logit proj)"
        )

    def _param_count(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e9

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Navigate the knowledge field based on current reasoning state.

        Produces compact query vectors (not logits).  The knowledge field
        computes logits on-the-fly via dot-product with its codebooks.

        Args:
            hidden_states: [batch, seq, hidden_size] from 7B core.

        Returns:
            queries: [batch, seq, num_heads, num_codebooks, codebook_dim]
                Compact query vectors for dot-product addressing.
            gate: [batch, seq, 1]
                Context gating value (how much subconscious to apply).
        """
        # Project to head dimension
        x = self.input_proj(hidden_states)
        x = self.input_norm(x)

        # Lightweight context encoding
        x = self.encoder(x)

        # Generate query vectors (NOT logits)
        # [B, T, H * M * D]
        queries_flat = self.query_proj(x)

        # Reshape to [B, T, H, M, D]
        B, T, _ = queries_flat.shape
        queries = queries_flat.reshape(
            B, T, self.num_heads, self.num_codebooks, self.codebook_dim
        )

        # Normalise queries for stable dot-product
        queries = self.query_norm(queries)

        # Compute context gate
        gate = self.context_gate(x)  # [B, T, 1]

        return queries, gate


# Paper: Original contribution by Dunimd Team (Yv Architecture — subconscious modulator)
class YvSubconsciousModulator(nn.Module):
    """Layer-wise subconscious modulation for 7B transformer layers.

    Injects retrieved knowledge into a single transformer layer's computation
    via FiLM-style modulation. This is NOT cross-attention — the knowledge
    never becomes part of the token sequence. Instead, it subtly shifts
    how the layer processes its inputs.

    Mechanism:
        retrieved_knowledge -> learned affine transforms -> FiLM parameters
        h_attn = h_attn * (1 + gamma_attn) + beta_attn  (modulate attention output)
        h_mlp  = h_mlp  * (1 + gamma_mlp)  + beta_mlp   (modulate FFN output)

    The modulation is applied AFTER the sublayer but BEFORE the residual.
    This ensures the residual stream carries modulated information forward.

    Key Properties:
        - Zero extra tokens: never increases sequence length
        - Parallel to attention: doesn't compete with attention compute
        - Volatile: computed fresh each forward pass, no persistent state
        - Subtle: initialized near-identity (gamma ≈ 0, beta ≈ 0)

    Args:
        hidden_size: Model hidden dimension.
        knowledge_dim: Dimension of retrieved knowledge.
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        knowledge_dim: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Attention modulation: knowledge -> gamma_attn, beta_attn
        self.attn_mod = nn.Linear(knowledge_dim, hidden_size * 2, bias=False, device=device, dtype=dtype)

        # MLP modulation: knowledge -> gamma_mlp, beta_mlp
        self.mlp_mod = nn.Linear(knowledge_dim, hidden_size * 2, bias=False, device=device, dtype=dtype)

        # Initialize to near-zero so the model starts without subconscious
        # and gradually learns to use it
        nn.init.zeros_(self.attn_mod.weight)
        nn.init.zeros_(self.mlp_mod.weight)

    def forward(
        self,
        attn_output: torch.Tensor,
        mlp_output: torch.Tensor,
        knowledge: torch.Tensor,
        gate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply subconscious modulation to attention and MLP outputs.

        Args:
            attn_output: [batch, seq, hidden_size] Post-attention (before residual).
            mlp_output: [batch, seq, hidden_size] Post-MLP (before residual).
            knowledge: [batch, seq, knowledge_dim] Retrieved subconscious knowledge.
            gate: [batch, seq, 1] Context gating value.

        Returns:
            attn_output: Modulated attention output.
            mlp_output: Modulated MLP output.
        """
        # Compute attention modulation parameters
        attn_params = self.attn_mod(knowledge)  # [B, T, hidden * 2]
        gamma_attn, beta_attn = attn_params.chunk(2, dim=-1)

        # Compute MLP modulation parameters
        mlp_params = self.mlp_mod(knowledge)  # [B, T, hidden * 2]
        gamma_mlp, beta_mlp = mlp_params.chunk(2, dim=-1)

        # Apply gate: only modulate when the gate says to
        gamma_attn = gamma_attn * gate
        beta_attn = beta_attn * gate
        gamma_mlp = gamma_mlp * gate
        beta_mlp = beta_mlp * gate

        # Apply FiLM-style modulation
        # The (1 + gamma) form ensures the output is near-identical when gamma ≈ 0
        attn_output = attn_output * (1.0 + gamma_attn) + beta_attn
        mlp_output = mlp_output * (1.0 + gamma_mlp) + beta_mlp

        return attn_output, mlp_output

    def modulate_attn_only(
        self,
        attn_output: torch.Tensor,
        knowledge: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate ONLY the attention output (efficient, no MLP compute).

        Args:
            attn_output: [batch, seq, hidden_size] Post-attention (before residual).
            knowledge: [batch, seq, knowledge_dim] Retrieved subconscious knowledge.
            gate: [batch, seq, 1] Context gating value.

        Returns:
            Modulated attention output.
        """
        params = self.attn_mod(knowledge)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma * gate
        beta = beta * gate
        return attn_output * (1.0 + gamma) + beta

    def modulate_mlp_only(
        self,
        mlp_output: torch.Tensor,
        knowledge: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate ONLY the MLP output (efficient, no attention compute).

        Args:
            mlp_output: [batch, seq, hidden_size] Post-MLP (before residual).
            knowledge: [batch, seq, knowledge_dim] Retrieved subconscious knowledge.
            gate: [batch, seq, 1] Context gating value.

        Returns:
            Modulated MLP output.
        """
        params = self.mlp_mod(knowledge)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma * gate
        beta = beta * gate
        return mlp_output * (1.0 + gamma) + beta


# Paper: Original contribution by Dunimd Team (Yv Architecture — subconscious system)
class YvSubconsciousSystem(nn.Module):
    """Complete subconscious knowledge system: 0.5B head + 314B field + injection.

    Orchestrates the full subconscious pipeline:
    1. Receive 7B hidden state
    2. Dynamic head navigates the knowledge field (O(1))
    3. Retrieved knowledge is injected into each transformer layer
    4. Knowledge is volatile (discarded after forward pass)

    This runs in parallel with the 1M context window and never interacts
    with it directly. The subconscious and conscious channels are orthogonal.

    Total extra parameters: ~0.5B
    - 0.27B: Implicit knowledge field (codebooks)
    - 0.23B: Dynamic head (addressing network)
    - ≈0B: Modulators (no extra storage per layer, computed per-forward)

    Args:
        hidden_size: 7B core hidden dimension.
        num_layers: Number of 7B transformer layers.
        knowledge_dim: Dimension of retrieved knowledge.
        num_codebooks: Codebook count for the knowledge field.
        codebook_size: Entries per codebook.
        codebook_dim: Dimension per codebook entry.
        num_field_heads: Attention heads in the knowledge field.
        head_dim: Dynamic head internal dimension.
        head_num_layers: Dynamic head transformer depth.
        head_num_attn_heads: Dynamic head attention heads.
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        num_layers: int = 32,
        knowledge_dim: int = 256,
        num_codebooks: int = 16,
        codebook_size: int = 131072,
        codebook_dim: int = 128,
        num_field_heads: int = 8,
        head_dim: int = 1024,
        head_num_layers: int = 2,
        head_num_attn_heads: int = 4,
        top_k_entries: int = 64,
        min_top_k_entries: int = 16,
        max_top_k_entries: int = 64,
        dynamic_topk_scale: float = 2048.0,
        score_chunk_size: int = 2048,
        query_chunk_size: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # 314B-equivalent implicit knowledge field
        self.knowledge_field = YvImplicitKnowledgeField(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            knowledge_dim=knowledge_dim,
            num_heads=num_field_heads,
            top_k_entries=top_k_entries,
            min_top_k_entries=min_top_k_entries,
            max_top_k_entries=max_top_k_entries,
            dynamic_topk_scale=dynamic_topk_scale,
            score_chunk_size=score_chunk_size,
            query_chunk_size=query_chunk_size,
            device=device,
            dtype=dtype,
        )

        # 0.5B dynamic navigation head
        # NOTE: codebook_dim is passed instead of codebook_size so the head
        # produces compact query vectors (~16.8M params) rather than giant
        # logits (~17.2B params).
        self.dynamic_head = YvDynamicHead(
            hidden_size=hidden_size,
            num_codebooks=num_codebooks,
            codebook_dim=codebook_dim,  # was codebook_size — fixed param explosion
            num_heads=num_field_heads,
            head_dim=head_dim,
            num_layers=head_num_layers,
            num_attn_heads=head_num_attn_heads,
            device=device,
            dtype=dtype,
        )

        # Layer-wise subconscious modulators
        self.modulators = nn.ModuleList([
            YvSubconsciousModulator(
                hidden_size=hidden_size,
                knowledge_dim=knowledge_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # Knowledge shift: tracks the "position" in the knowledge field
        # This is the subconscious equivalent of position_ids in the context
        # Shift is learned so the model can track knowledge flow across layers
        self.knowledge_shift = nn.Parameter(torch.zeros(1, 1, knowledge_dim, device=device, dtype=dtype))

        # Cache for current forward pass (volatile, cleared after each forward)
        self._current_knowledge: Optional[torch.Tensor] = None
        self._current_gate: Optional[torch.Tensor] = None
        self._film_param_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        total_params = sum(p.numel() for p in self.parameters())
        _LOG.info(
            f"YvSubconsciousSystem: {total_params / 1e9:.3f}B total params, "
            f"virtual knowledge capacity = {codebook_size ** num_codebooks:.1e}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass of the subconscious system.

        This is called ONCE per model forward pass. The retrieved knowledge
        is cached and then consumed by each transformer layer's modulator.

        Pipeline:
            1. Dynamic head encodes hidden state → compact query vectors
            2. Knowledge field computes dot-product logits → soft-addressing → retrieval
            3. Retrieved knowledge is cached for layer-wise consumption

        Args:
            hidden_states: [batch, seq, hidden_size] From the 7B core
                (typically from the first few layers or the embedding).

        Returns:
            knowledge: [batch, seq, knowledge_dim] Cached knowledge.
            gate: [batch, seq, 1] Context gate.
        """
        # 1. Navigate the knowledge field → compact query vectors (no 17B proj)
        queries, gate = self.dynamic_head(hidden_states)

        # 2. Retrieve knowledge from the field via dot-product addressing
        #    (logits computed on-the-fly inside the field, zero extra params)
        knowledge = self.knowledge_field(queries)

        # Cache for layer-wise consumption
        # (knowledge shift is applied per-layer in modulate_layer / get_film_params)
        self._current_knowledge = knowledge
        self._current_gate = gate
        self._film_param_cache.clear()

        return knowledge, gate

    def modulate_layer(
        self,
        layer_idx: int,
        attn_output: torch.Tensor,
        mlp_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply subconscious modulation to a single layer.

        Called by each YvTransformerBlock during its forward pass.
        Consumes the cached knowledge from the current forward step.

        Args:
            layer_idx: Index of the current transformer layer.
            attn_output: Post-attention output.
            mlp_output: Post-MLP output.

        Returns:
            Modulated attention and MLP outputs.
        """
        if self._current_knowledge is None or self._current_gate is None:
            return attn_output, mlp_output

        # Shift knowledge for this layer so each layer gets a slightly
        # different "view" of the same subconscious knowledge
        layer_shift = self.knowledge_shift * layer_idx * 0.01
        knowledge = self._current_knowledge + layer_shift

        modulator = self.modulators[layer_idx]
        return modulator(attn_output, mlp_output, knowledge, self._current_gate)

    def modulate_attn(
        self,
        layer_idx: int,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate ONLY the attention output (no dummy MLP tensor needed).

        Efficient variant of :meth:`modulate_layer` that only computes the
        attention FiLM transform — no wasteful computation for MLP.

        Args:
            layer_idx: Index of the current transformer layer.
            attn_output: Post-attention output (before residual).

        Returns:
            Modulated attention output.
        """
        if self._current_knowledge is None or self._current_gate is None:
            return attn_output
        layer_shift = self.knowledge_shift * layer_idx * 0.01
        knowledge = self._current_knowledge + layer_shift
        return self.modulators[layer_idx].modulate_attn_only(
            attn_output, knowledge, self._current_gate
        )

    def modulate_mlp(
        self,
        layer_idx: int,
        mlp_output: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate ONLY the MLP output (no dummy attention tensor needed).

        Efficient variant of :meth:`modulate_layer` that only computes the
        MLP FiLM transform — no wasteful computation for attention.

        Args:
            layer_idx: Index of the current transformer layer.
            mlp_output: Post-MLP output (before residual).

        Returns:
            Modulated MLP output.
        """
        if self._current_knowledge is None or self._current_gate is None:
            return mlp_output
        layer_shift = self.knowledge_shift * layer_idx * 0.01
        knowledge = self._current_knowledge + layer_shift
        return self.modulators[layer_idx].modulate_mlp_only(
            mlp_output, knowledge, self._current_gate
        )

    def clear_cache(self):
        """Clear volatile subconscious cache after forward pass."""
        self._current_knowledge = None
        self._current_gate = None
        self._film_param_cache.clear()

    def get_film_params(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute FiLM scale and shift for a specific layer.

        This is used by :class:`YvDualInjector` to obtain the subconscious
        modulation parameters without applying them, so the caller can decide
        how to blend the FiLM path with the raw hidden stream.

        Args:
            hidden_states: [batch, seq, hidden_size] from the 7B core.
            layer_idx: Index of the transformer layer being injected.

        Returns:
            Dict with keys ``scale`` and ``shift``, each of shape
            [batch, seq, hidden_size].
        """
        cached = self._film_param_cache.get(layer_idx)
        if cached is not None:
            return cached

        gate = self._current_gate
        knowledge = self._current_knowledge
        if knowledge is None or gate is None:
            queries, gate = self.dynamic_head(hidden_states)
            knowledge = self.knowledge_field(queries)
            self._current_knowledge = knowledge
            self._current_gate = gate
            self._film_param_cache.clear()
        knowledge = knowledge + self.knowledge_shift * (1 + layer_idx * 0.01)

        attn_params = self.modulators[layer_idx].attn_mod(knowledge)
        mlp_params = self.modulators[layer_idx].mlp_mod(knowledge)
        gate_val = torch.sigmoid(attn_params.mean(dim=-1, keepdim=True))
        params = gate_val * attn_params + (1 - gate_val) * mlp_params
        scale, shift = params.chunk(2, dim=-1)
        scale = scale * gate
        shift = shift * gate
        out = {"scale": scale, "shift": shift}
        self._film_param_cache[layer_idx] = out
        return out

    def get_knowledge(self) -> Optional[torch.Tensor]:
        """Get current cached knowledge for debugging/inspection."""
        return self._current_knowledge

    def get_gate(self) -> Optional[torch.Tensor]:
        """Get current cached gate for debugging/inspection."""
        return self._current_gate

    def extra_repr(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"total_params={total/1e9:.3f}B"
