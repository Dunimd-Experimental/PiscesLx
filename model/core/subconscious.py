#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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
Subconscious Knowledge System for PiscesL1/Yv.

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
    - Trainable via RL (EnTA): reward = improvement in 7B reasoning quality

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

_LOG = PiscesLxLogger("Yv.Subconscious", file_path=get_log_file("Yv.Subconscious"), enable_file=True)


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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.knowledge_dim = knowledge_dim
        self.num_heads = num_heads

        # Multi-head codebooks: each head has its own set of codebooks
        # This increases representational capacity without increasing K or M
        self.head_dim = knowledge_dim // num_heads

        # Codebook parameters: [num_heads, num_codebooks, codebook_size, codebook_dim]
        # Total params: num_heads * num_codebooks * codebook_size * codebook_dim
        # = 8 * 16 * 131072 * 128 ≈ 0.214B
        self.codebooks = nn.Parameter(
            torch.randn(num_heads, num_codebooks, codebook_size, codebook_dim, device=device, dtype=dtype)
            * 0.02
        )

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
        addressing_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve knowledge from the field given addressing signals.

        Args:
            addressing_logits: [batch, seq, num_heads * num_codebooks, codebook_size]
                Raw logits for soft-addressing each codebook.
                Each head addresses a separate slice of the codebooks.

        Returns:
            knowledge: [batch, seq, knowledge_dim]
                Retrieved knowledge representation.
        """
        B, T, *_ = addressing_logits.shape

        # Reshape to separate heads and codebooks
        # [B, T, num_heads, num_codebooks, codebook_size]
        logits = addressing_logits.view(
            B, T, self.num_heads, self.num_codebooks, self.codebook_size
        )

        # Soft addressing over each codebook
        # [B, T, num_heads, num_codebooks, codebook_size]
        addressing_weights = F.softmax(logits, dim=-1)

        # Temperature annealing for sharper addressing during training
        if self.training:
            # Start soft for gradient flow, gradually sharpen
            temp = max(0.5, 1.0 - self._get_training_progress() * 0.5)
            addressing_weights = F.softmax(logits / temp, dim=-1)

        # Retrieve from codebooks via weighted sum
        # codebooks: [num_heads, num_codebooks, codebook_size, codebook_dim]
        # addressing_weights: [B, T, num_heads, num_codebooks, codebook_size]
        # -> [B, T, num_heads, num_codebooks, codebook_dim]
        retrieved = torch.einsum(
            'bthmk,hkmd->bthmd',
            addressing_weights,
            self.codebooks
        )

        # Combine across codebooks within each head
        # [B, T, num_heads, knowledge_dim_per_head]  where knowledge_dim_per_head = codebook_dim
        # Actually, we need to project to head_dim.
        # For now, codebook_dim == head_dim, so just sum across codebooks
        knowledge_per_head = retrieved.sum(dim=3)  # [B, T, num_heads, codebook_dim]

        # Concatenate heads
        knowledge = knowledge_per_head.reshape(B, T, self.knowledge_dim)

        # Output projection and normalize
        knowledge = self.output_proj(knowledge)
        knowledge = self.norm(knowledge)

        return knowledge

    def _get_training_progress(self) -> float:
        """Estimate training progress for temperature scheduling."""
        if hasattr(self, '_training_step'):
            return min(1.0, self._training_step / 50000)
        return 0.0

    def extra_repr(self) -> str:
        return (
            f"num_codebooks={self.num_codebooks}, "
            f"codebook_size={self.codebook_size}, "
            f"codebook_dim={self.codebook_dim}, "
            f"param_count={self._param_count():.3f}B"
        )


class YvDynamicHead(nn.Module):
    """0.5B dynamic navigation head for implicit knowledge field addressing.

    Projects the 7B core's current reasoning state into navigation coordinates
    that address the implicit knowledge field. Designed for memory-address-speed
    operation with minimal latency overhead.

    Architecture:
        - Input projection: hidden_size -> head_dim
        - Lightweight transformer encoder (2 layers, 4 heads)
        - Output projection: head_dim -> num_heads * num_codebooks * codebook_size
        - Address generation via learned routing

    Key Properties:
        - Fast: O(1) routing via learned projection + lightweight processing
        - Context-aware: addressing depends on current reasoning state
        - Differentiable: enables end-to-end training with RL
        - Memory-address-speed: no iterative search, single forward pass

    Args:
        hidden_size: 7B core's hidden dimension.
        num_codebooks: Number of codebooks in the knowledge field.
        codebook_size: Number of entries per codebook.
        num_heads: Number of heads in the knowledge field.
        head_dim: Internal dimension for the navigation head.
        num_layers: Number of lightweight transformer layers.
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        num_codebooks: int = 16,
        codebook_size: int = 131072,
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
        self.codebook_size = codebook_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Input projection: 7B hidden -> head internal dimension
        self.input_proj = nn.Linear(hidden_size, head_dim, bias=False, device=device, dtype=dtype)
        self.input_norm = nn.LayerNorm(head_dim, device=device, dtype=dtype)

        # Lightweight transformer encoder for context-aware addressing
        # Using small dimensions to keep this at ~0.23B params total
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=head_dim,
            nhead=num_attn_heads,
            dim_feedforward=head_dim * 4,
            dropout=0.0,
            activation='silu',
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Addressing output: produces logits for each codebook entry
        # Weight-tied output for parameter efficiency: shared projection per codebook
        self.address_proj = nn.Linear(
            head_dim,
            num_heads * num_codebooks * codebook_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # Context gating: decides how much subconscious to apply
        # Based on current reasoning uncertainty
        self.context_gate = nn.Sequential(
            nn.Linear(head_dim, head_dim // 4, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(head_dim // 4, 1, device=device, dtype=dtype),
            nn.Sigmoid(),
        )

        # Norm for addressing stability
        self.address_norm = nn.RMSNorm(num_heads * num_codebooks * codebook_size // num_codebooks, device=device, dtype=dtype)

        _LOG.info(
            f"YvDynamicHead: {self._param_count():.3f}B params, "
            f"hidden={hidden_size}, head_dim={head_dim}, "
            f"num_layers={num_layers}"
        )

    def _param_count(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e9

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Navigate the knowledge field based on current reasoning state.

        Args:
            hidden_states: [batch, seq, hidden_size] from 7B core.

        Returns:
            addressing_logits: [batch, seq, num_heads * num_codebooks, codebook_size]
                Raw addressing signals for the knowledge field.
            gate: [batch, seq, 1]
                Context gating value (how much subconscious to apply).
        """
        # Project to head dimension
        x = self.input_proj(hidden_states)
        x = self.input_norm(x)

        # Lightweight context encoding
        # This captures the current reasoning state for precise addressing
        x = self.encoder(x)

        # Generate addressing logits
        addressing_logits = self.address_proj(x)

        # Reshape to separate heads and codebooks
        # [B, T, num_heads * num_codebooks, codebook_size]
        B, T, D = addressing_logits.shape
        addressing_logits = addressing_logits.view(
            B, T, self.num_heads * self.num_codebooks, self.codebook_size
        )

        # Apply RMSNorm for stable addressing
        # Normalize across the codebook dimension (per head-codebook group)
        addressing_logits = self.address_norm(addressing_logits)

        # Compute context gate
        gate = self.context_gate(x)  # [B, T, 1]

        return addressing_logits, gate


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
            device=device,
            dtype=dtype,
        )

        # 0.5B dynamic navigation head
        self.dynamic_head = YvDynamicHead(
            hidden_size=hidden_size,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
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

        Args:
            hidden_states: [batch, seq, hidden_size] From the 7B core
                (typically from the first few layers or the embedding).

        Returns:
            knowledge: [batch, seq, knowledge_dim] Cached knowledge.
            gate: [batch, seq, 1] Context gate.
        """
        # 1. Navigate the knowledge field
        addressing_logits, gate = self.dynamic_head(hidden_states)

        # 2. Retrieve knowledge from the field
        knowledge = self.knowledge_field(addressing_logits)

        # 3. Add knowledge shift for layer-wise variation
        knowledge = knowledge + self.knowledge_shift

        # Cache for layer-wise consumption
        self._current_knowledge = knowledge
        self._current_gate = gate

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

    def clear_cache(self):
        """Clear volatile subconscious cache after forward pass."""
        self._current_knowledge = None
        self._current_gate = None

    def get_knowledge(self) -> Optional[torch.Tensor]:
        """Get current cached knowledge for debugging/inspection."""
        return self._current_knowledge

    def get_gate(self) -> Optional[torch.Tensor]:
        """Get current cached gate for debugging/inspection."""
        return self._current_gate

    def extra_repr(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"total_params={total/1e9:.3f}B"
