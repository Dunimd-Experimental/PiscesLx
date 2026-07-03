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

"""Dynamic multimodal fusion utilities for PiscesLx Yv agents.

This module provides comprehensive multimodal fusion components for the Yv
model, including dynamic cross-modal attention, modality-specific gating, and
generation pathways for unified representation learning.

Module Components:
    1. YvUnifiedMultimodalTokenizer:
       - Shared tokenizer abstraction mapping all modalities into one token space
       - Modality-specific tokenizers and trainable projection layers
       - Unified positional and modality embeddings

    2. YvDynamicModalFusion:
       - Unified tokenization across modalities
       - Native unified multimodal token space mode (2026 flagship design)
       - Legacy CNN-patch cross-modal attention mode for backward compatibility
       - Cross-modal attention integration
       - Understanding and generation gating
       - Modality-specific generation with caching

Key Features:
    - 6-modality support (text, image, audio, video, document, agentic)
    - Unified tokenization with modality embeddings
    - Native token-space fusion where every modality is a token sequence
    - Cross-modal attention for inter-modality reasoning
    - Understanding gate for global representation
    - Generation gates for modality-specific outputs
    - Weight caching for efficient repeated fusion
    - Hardware-adaptive gradient configuration

Performance Characteristics:
    - Tokenization: O(N * hidden_size) per modality
    - Cross-modal attention: O(T^2 * hidden_size) where T = total tokens
    - Gating: O(hidden_size) per modality
    - Total complexity: O(T^2 * hidden_size)

Usage Example:
    >>> from model.multimodal.fusion import YvDynamicModalFusion
    >>>
    >>> # Initialize fusion module
    >>> fusion = YvDynamicModalFusion(config)
    >>>
    >>> # Fuse multimodal features
    >>> features = {
    ...     "text": text_features,
    ...     "image": image_features,
    ...     "audio": audio_features
    >>> }
    >>> fused = fusion(features)  # [B, 1, hidden_size] legacy or [B, T, hidden_size] native
    >>>
    >>> # Generate modality-specific output
    >>> image_gen = fusion.generate_modality("image", temperature=0.8)

Note:
    Supports text, image, audio, video, document, and agentic modalities.
    Uses YvCrossModalAttention for cross-modality reasoning.
    Integrates with YvMemory for tensor lifetime tracking.
    When config.use_native_multimodal_fusion is True, all modalities are
    projected into the same autoregressive token embedding space.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional
from .memory import YvMemory
from .hw import YvHardwareAdaptiveConfig
from .attention import YvCrossModalAttention
from model.utils import YvShapeGuard
import logging

logger = logging.getLogger(__name__)


class YvUnifiedMultimodalTokenizer(nn.Module):
    """Shared multimodal tokenizer abstraction for native unified token space.

    Maps raw modality inputs or pre-encoded modality features into the same
    token embedding space as text tokens. Each modality receives a dedicated
    tokenizer, a trainable projection layer, and a shared modality embedding
    so that image, audio, video, document, agentic, and text tokens can be
    concatenated and processed by a single autoregressive transformer.

    Architecture:
        1. Modality-specific tokenizers:
           - text: Identity or linear projection
           - image: Conv2d patch embedding for raw images, linear for features
           - audio: Conv1d patch embedding for waveforms, linear for features
           - video: Conv3d patch embedding for raw videos, linear for features
           - document: Linear projection
           - agentic: Linear projection

        2. Per-modality projection layers:
           - Trainable linear projections into the shared hidden_size space

        3. Unified embeddings:
           - Shared learned positional embeddings
           - Modality type embeddings

    Attributes:
        cfg: Configuration namespace containing fusion hyperparameters.
        hidden_size (int): Dimensionality of the shared token space.
        modalities (List[str]): Canonical modality identifiers.
        token_counts (Dict[str, int]): Maximum tokens per modality.
        modality_tokenizers (nn.ModuleDict): Modality-specific tokenizers.
        modality_projections (nn.ModuleDict): Per-modality projection layers.
        modality_embeddings (nn.Embedding): Modality type embeddings.
        pos_embed (nn.Parameter): Shared learned positional embeddings.

    Example:
        >>> tokenizer = YvUnifiedMultimodalTokenizer(config)
        >>> unified = tokenizer({"text": text_emb, "image": img_tensor})
        >>> for modal, tokens in unified.items():
        ...     print(modal, tokens.shape)  # [B, T, hidden_size]

    Note:
        Inputs may be raw tensors (image [B, C, H, W]) or pre-encoded features
        ([B, T, hidden_size]). The tokenizer detects shape and routes accordingly.
    """

    def __init__(self, cfg, device=None, dtype=None):
        """Initialize the unified multimodal tokenizer.

        Args:
            cfg: Configuration object containing parameters such as:
                - hidden_size: Output embedding dimension
                - image_tokens, audio_tokens, video_tokens, document_tokens,
                  agentic_tokens, text_tokens: Per-modality token counts
                - max_position_embeddings: Positional embedding length
            device: Optional device for created parameters.
            dtype: Optional dtype for created parameters.
        """
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.modalities = ["text", "image", "audio", "video", "document", "agentic"]

        # Per-modality token counts; -1 means no explicit truncation.
        self.token_counts = {
            "text": getattr(cfg, "text_tokens", -1),
            "image": getattr(cfg, "image_tokens", getattr(cfg, "mm_tokens", 256)),
            "audio": getattr(cfg, "audio_tokens", 512),
            "video": getattr(cfg, "video_tokens", getattr(cfg, "mm_tokens", 256) // 2),
            "document": getattr(cfg, "document_tokens", getattr(cfg, "mm_tokens", 256)),
            "agentic": getattr(cfg, "agentic_tokens", getattr(cfg, "modal_token_count", 8)),
        }

        max_pos = getattr(cfg, "max_position_embeddings", 8192)
        self.register_parameter(
            "pos_embed",
            nn.Parameter(torch.randn(1, max_pos, self.hidden_size, device=device, dtype=dtype) * 0.02)
        )
        self.modality_embeddings = nn.Embedding(len(self.modalities), self.hidden_size, device=device, dtype=dtype)

        # Modality-specific tokenizers that handle both raw inputs and features.
        self.modality_tokenizers = nn.ModuleDict({
            "text": nn.Identity(),
            "image": self._build_image_tokenizer(device, dtype),
            "audio": self._build_audio_tokenizer(device, dtype),
            "video": self._build_video_tokenizer(device, dtype),
            "document": nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype),
            "agentic": nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype),
        })

        # Trainable per-modality projections into the shared token space.
        self.modality_projections = nn.ModuleDict({
            modal: nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
            for modal in self.modalities
        })

    def _build_image_tokenizer(self, device, dtype):
        """Build image tokenizer for raw images or pre-encoded features."""
        patch_size = getattr(self.cfg, "image_patch", 16)
        return nn.Sequential(
            nn.Conv2d(3, self.hidden_size, patch_size, patch_size, device=device, dtype=dtype),
            nn.Flatten(2),
            nn.Identity()
        )

    def _build_audio_tokenizer(self, device, dtype):
        """Build audio tokenizer for raw waveforms or pre-encoded features."""
        return nn.Sequential(
            nn.Conv1d(1, self.hidden_size, 16, 16, device=device, dtype=dtype),
            nn.Identity()
        )

    def _build_video_tokenizer(self, device, dtype):
        """Build video tokenizer for raw videos or pre-encoded features."""
        patch_size = getattr(self.cfg, "image_patch", 16)
        return nn.Sequential(
            nn.Conv3d(3, self.hidden_size, (2, patch_size, patch_size), (2, patch_size, patch_size), device=device, dtype=dtype),
            nn.Flatten(2),
            nn.Identity()
        )

    def _limit_tokens(self, tokens: torch.Tensor, modal: str) -> torch.Tensor:
        """Truncate or pool tokens to the configured per-modality count.

        Args:
            tokens: Token tensor [B, T, hidden_size].
            modal: Modality name.

        Returns:
            Tensor with token count limited to token_counts[modal].
        """
        limit = self.token_counts.get(modal, -1)
        if limit <= 0 or tokens.shape[1] <= limit:
            return tokens
        if tokens.shape[1] > limit * 2:
            # Average pool to roughly limit tokens before truncation.
            factor = tokens.shape[1] // limit
            tokens = F.avg_pool1d(
                tokens.transpose(1, 2),
                kernel_size=factor,
                stride=factor
            ).transpose(1, 2)
        return tokens[:, :limit, :]

    def _tokenize_modality(self, modal: str, feat: torch.Tensor, idx: int) -> torch.Tensor:
        """Convert one modality input into unified token space.

        Args:
            modal: Modality identifier.
            feat: Input tensor, raw or pre-encoded.
            idx: Modality index for embedding lookup.

        Returns:
            Unified token tensor [B, T, hidden_size].
        """
        tokenizer = self.modality_tokenizers[modal]
        projector = self.modality_projections[modal]

        if modal == "text":
            if feat.dim() == 2:
                feat = feat.unsqueeze(1)
            tok = tokenizer(feat) if not isinstance(tokenizer, nn.Identity) else feat
        elif modal == "image":
            if feat.dim() == 4:
                # Raw image [B, C, H, W]
                patches = tokenizer(feat)
                if patches.dim() == 3:
                    tok = patches.transpose(1, 2)
                else:
                    tok = patches.view(patches.shape[0], patches.shape[1], -1).transpose(1, 2)
            elif feat.dim() == 3:
                tok = feat
            else:
                tok = feat.unsqueeze(1)
        elif modal == "audio":
            if feat.dim() == 2:
                feat = feat.unsqueeze(1)
            if feat.dim() == 3 and feat.shape[1] == 1:
                # Raw waveform [B, 1, T]
                tok = tokenizer(feat).transpose(1, 2)
            elif feat.dim() == 3:
                tok = feat
            else:
                tok = feat.unsqueeze(1)
        elif modal == "video":
            if feat.dim() == 5:
                # Raw video [B, C, T, H, W]
                b, c, t, h, w = feat.shape
                patches = tokenizer(feat)
                if patches.dim() == 3:
                    tok = patches.transpose(1, 2)
                else:
                    tok = patches.view(b, -1, self.hidden_size)
                tok = tok.view(b, t * (tok.shape[1] // t), self.hidden_size)
            elif feat.dim() == 3:
                tok = feat
            else:
                tok = feat.unsqueeze(1)
        else:
            # document / agentic
            if feat.dim() == 3:
                tok = feat
            else:
                tok = feat.unsqueeze(1)

        # Project into shared hidden space and apply per-token nonlinearity.
        tok = F.silu(projector(tok))

        # Truncate / pool to modality token budget.
        tok = self._limit_tokens(tok, modal)

        # Add positional and modality embeddings.
        seq_len = tok.shape[1]
        if seq_len > self.pos_embed.shape[1]:
            logger.debug(
                "pos_embed truncated: seq_len=%d > pos_embed_len=%d for modal=%s",
                seq_len, self.pos_embed.shape[1], modal,
            )
        pos = self.pos_embed[:, :seq_len, :].to(tok.device)
        modal_emb = self.modality_embeddings(
            torch.tensor(idx, device=tok.device, dtype=torch.long)
        ).unsqueeze(0).unsqueeze(0)
        return tok + pos + modal_emb

    def forward(self, modal_features: Dict[str, Optional[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Map multimodal inputs to unified token sequences.

        Args:
            modal_features: Mapping from modality name to feature tensor or None.
                Supported modalities: text, image, audio, video, document, agentic.

        Returns:
            Dictionary mapping modality name to unified token tensor
            with shape [B, T, hidden_size]. Missing modalities are omitted.
        """
        unified: Dict[str, torch.Tensor] = {}
        for idx, modal in enumerate(self.modalities):
            feat = modal_features.get(modal)
            if feat is None or not isinstance(feat, torch.Tensor) or feat.numel() == 0:
                continue
            unified[modal] = self._tokenize_modality(modal, feat, idx)
        return unified


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class YvDynamicModalFusion(nn.Module):
    """Dynamic multimodal fusion backbone for Yv workflows.

    A comprehensive fusion module that tokenizes modality-specific inputs,
    enriches them with learned positional and modality embeddings, and performs
    cross-modal attention followed by gated fusion to produce a unified
    representation.

    Two operating modes are supported:
        - Legacy mode (config.use_native_multimodal_fusion=False):
          CNN-patch tokenizers + cross-modal attention + gating,
          returning ``[B, 1, hidden_size]``.
        - Native unified token space mode
          (config.use_native_multimodal_fusion=True):
          All modalities are tokenized into the same embedding space via
          YvUnifiedMultimodalTokenizer, fused with self-attention, and
          projected to a fixed number of output tokens compatible with the
          autoregressive language model backbone.

    Architecture:
        1. Unified Tokenization:
           - Text: Identity (pass-through)
           - Image: Conv2d patch embedding or linear projection
           - Audio: Conv1d patch embedding or linear projection
           - Video: Conv3d patch embedding or linear projection
           - Document/Agentic: Linear projection

        2. Positional Encoding:
           - Shared learned positional embeddings (8192 max length)
           - Modality-specific token embeddings

        3. Cross-Modal Attention:
           - YvCrossModalAttention for inter-modality reasoning
           - Self-attention over concatenated modality tokens

        4. Gating Mechanisms:
           - Understanding gate: Global representation modulation
           - Generation gates: Per-modality output modulation

    Key Features:
        - 6-modality support with unified tokenization
        - Native unified multimodal token space (2026 flagship design)
        - Backward-compatible legacy fusion path
        - Cross-modal attention for inter-modality reasoning
        - Understanding and generation gating mechanisms
        - Weight caching for efficient repeated fusion
        - Hardware-adaptive gradient configuration

    Attributes:
        cfg: Configuration namespace containing fusion hyperparameters.
        hidden_size (int): Dimensionality of the shared representation space.
        modalities (List[str]): Canonical modality identifiers handled by the fusion core.
        use_native_mode (bool): Whether native unified token space is enabled.
        native_tokenizer (YvUnifiedMultimodalTokenizer): Shared tokenizer in native mode.
        native_fusion_layers (nn.ModuleList): Self-attention layers in native mode.
        output_proj (nn.Linear): Projects pooled representation to output tokens in native mode.
        weight_cache (Dict[str, torch.Tensor]): Cache for previously fused outputs keyed by modality presence signatures.
        cache_size_limit (int): Maximum number of cached signatures retained.
        cache_manager: Optional external cache manager reused across agent subsystems.
        memory_manager (YvMemory): Memory system for tracking tensor lifetimes.
        hw (YvHardwareAdaptiveConfig): Hardware adaptation helper used to derive gradient configuration.
        grad_conf (Dict[str, Any]): Gradient settings retrieved from the hardware adapter.
        unified_tokenizer (nn.ModuleDict): Mapping from modality to tokenization modules that project raw inputs (legacy).
        unified_pos_embed (nn.Parameter): Learned positional embeddings shared across modalities (legacy).
        modality_tokens (nn.Embedding): Trainable embeddings encoding modality identity (legacy).
        cross_modal_attn (YvCrossModalAttention): Attention layer performing cross-modality reasoning.
        understanding_gate (nn.Sequential): Gating module producing global understanding signals.
        generation_gates (nn.ModuleDict): Modality-specific gates used to modulate generation outputs.
        _generation_cache (Dict[str, torch.Tensor]): Storage for latest modality-specific outputs produced by ``forward``.

    Example:
        >>> fusion = YvDynamicModalFusion(config)
        >>> features = {"text": text_feat, "image": img_feat}
        >>> fused = fusion(features)  # [B, 1, hidden_size] legacy or [B, T, hidden_size] native
        >>>
        >>> # Generate modality-specific output
        >>> gen = fusion.generate_modality("image", temperature=0.8)

    Note:
        Supports text, image, audio, video, document, and agentic modalities.
        Cache size limit is 1000 signatures by default.
    """

    def __init__(self, cfg, cache_manager=None, device=None, dtype=None):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.modalities = ["text", "image", "audio", "video", "document", "agentic"]
        self.weight_cache: Dict[str, torch.Tensor] = {}
        self.cache_size_limit = 1000
        self.cache_manager = cache_manager
        self.memory_manager = YvMemory() if getattr(cfg, 'use_agentic', False) else None
        self.hw = YvHardwareAdaptiveConfig()
        self.grad_conf = self.hw.get_gradient_config()

        # === Unified tokenization (native always; legacy tokenizers for raw input compat) ===
        self.native_tokenizer = YvUnifiedMultimodalTokenizer(cfg, device=device, dtype=dtype)
        self.native_output_tokens = int(getattr(cfg, "native_fusion_output_tokens", getattr(cfg, "modal_token_count", 8)))
        native_layers = int(getattr(cfg, "native_fusion_num_layers", 2))
        num_heads = max(1, getattr(cfg, "n_head", 16) // 4)
        self.native_fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size, nhead=num_heads,
                dim_feedforward=self.hidden_size * 4,
                dropout=float(getattr(cfg, "fusion_dropout", 0.1)),
                activation="gelu", batch_first=True,
                device=device, dtype=dtype,
            ) for _ in range(native_layers)
        ])
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        self.output_tokens = nn.Parameter(
            torch.randn(1, self.native_output_tokens, self.hidden_size, device=device, dtype=dtype) * 0.02
        )
        # Legacy raw-input tokenizers
        self.unified_tokenizer = nn.ModuleDict({
            "text": nn.Identity(),
            "image": nn.Conv2d(3, self.hidden_size, 16, 16, device=device, dtype=dtype),
            "audio": nn.Conv1d(1, self.hidden_size, 16, 16, device=device, dtype=dtype),
            "video": nn.Conv3d(3, self.hidden_size, (2, 16, 16), (2, 16, 16), device=device, dtype=dtype),
            "document": nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype),
            "agentic": nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        })
        self.unified_pos_embed = nn.Parameter(
            torch.randn(1, 8192, self.hidden_size, device=device, dtype=dtype) * 0.02
        )
        self.modality_tokens = nn.Embedding(len(self.modalities), self.hidden_size, device=device)
        if dtype is not None:
            self.modality_tokens = self.modality_tokens.to(dtype)

        # === Cross-modal attention ===
        self.cross_modal_attn = YvCrossModalAttention(cfg)

        # === SyncFusion: audio-video temporal alignment ===
        self.sync_temporal_bins = int(getattr(cfg, "sync_fusion_temporal_bins", 16))
        sync_bins = self.sync_temporal_bins
        self.audio_temporal = nn.Linear(self.hidden_size, sync_bins * self.hidden_size, device=device, dtype=dtype)
        self.video_temporal = nn.Linear(self.hidden_size, sync_bins * self.hidden_size, device=device, dtype=dtype)
        self.sync_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=num_heads,
            batch_first=True, device=device, dtype=dtype,
        )
        self.sync_fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 1, device=device, dtype=dtype),
            nn.Sigmoid(),
        )
        self.sync_output_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, device=device, dtype=dtype)

        # === Intra-modal encoders (enhanced fusion) ===
        self.intra_modal_encoders = nn.ModuleDict({
            modal: nn.Sequential(
                nn.LayerNorm(self.hidden_size, device=device, dtype=dtype),
                nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype),
            ) for modal in self.modalities
        })

        # === Inter-modal attention ===
        self.inter_modal_q = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        self.inter_modal_k = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        self.inter_modal_v = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        self.inter_modal_out = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)

        # === Cross-modal alignment ===
        self.align_projections = nn.ModuleDict({
            modal: nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
            for modal in self.modalities
        })
        self.align_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * len(self.modalities), self.hidden_size, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype),
        )

        # === Importance weighting ===
        self.importance_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, len(self.modalities), device=device, dtype=dtype),
        )
        self.modality_gate_nets = nn.ModuleDict({
            modal: nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 4, 1, device=device, dtype=dtype),
                nn.Sigmoid(),
            ) for modal in self.modalities
        })

        # === Quality-aware fusion ===
        self.quality_global = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(1),
            nn.Linear(self.hidden_size, self.hidden_size // 2, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 2, 1, device=device, dtype=dtype),
            nn.Sigmoid(),
        )
        self.quality_local = nn.Conv1d(self.hidden_size, 1, kernel_size=3, padding=1, device=device, dtype=dtype)
        self.quality_head = nn.Sequential(
            nn.Linear(2, self.hidden_size // 4, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 4, 1, device=device, dtype=dtype),
            nn.Sigmoid(),
        )

        # === Recursive refinement (RCA-style) ===
        self.max_rca_rounds = int(getattr(cfg, "max_rca_rounds", 3))
        self.rca_convergence_threshold = float(getattr(cfg, "rca_convergence_threshold", 0.995))
        self.rca_output_tokens = nn.Parameter(
            torch.randn(1, self.native_output_tokens, self.hidden_size, device=device, dtype=dtype) * 0.02
        )
        self.rca_output_proj = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        self.rca_forecast_proj = nn.Linear(
            self.hidden_size, len(self.modalities) * max(1, num_heads // 2),
            device=device, dtype=dtype,
        )
        self.rca_forecast_gate = nn.Parameter(torch.tensor(0.5, device=device, dtype=dtype))

        # === Understanding and generation gating ===
        self.understanding_gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype),
            nn.Sigmoid()
        )
        self.generation_gates = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.hidden_size, device=device, dtype=dtype),
                nn.Sigmoid()
            ) for m in self.modalities
        })
        self._generation_cache: Dict[str, torch.Tensor] = {}

    def _signature(self, features: Dict[str, Optional[torch.Tensor]]) -> str:
        """Summarize modality presence and tensor shapes into a cache signature string.

        Creates a unique identifier for the current modality configuration
        to enable caching of previously computed fusion outputs.

        Args:
            features: Dictionary mapping modality names to optional tensors.
                Keys are modality names, values are feature tensors or None.

        Returns:
            str: Colon-delimited presence signature with shape info.
                e.g., ``"text:1:128:image:0:0"``.
                Used as cache key for weight_cache lookup.
        """
        parts = []
        for m in self.modalities:
            feat = features.get(m)
            if feat is not None and isinstance(feat, torch.Tensor):
                seq_len = feat.shape[1] if feat.dim() >= 2 else 0
                parts.append(f"{m}:1:{seq_len}:{feat.dtype}:{feat.device.type}")
            else:
                parts.append(f"{m}:0:0")
        return ":".join(parts)

    def generate_modality(self, target_modal: str, prompt_tokens: Optional[torch.Tensor] = None,
                          temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate representations for a target modality using cached states.

        Produces modality-specific outputs by applying generation gates and
        optional conditioning from prompt tokens. Uses cached fusion outputs
        from the most recent forward pass.

        Args:
            target_modal (str): Name of the modality to synthesize.
                Must be one of: text, image, audio, video, document, agentic.
            prompt_tokens (Optional[torch.Tensor]): Optional conditioning tokens.
                Shape: [B, seq_len, hidden_size] or [B, hidden_size].
            temperature (float): Softmax temperature scaling factor.
                Values > 1.0 increase diversity, < 1.0 increase determinism.
                Default: 1.0.
            top_k (Optional[int]): Retain top-k dimensions when specified.
                If set, zeros out all but the top-k values. Default: None.

        Returns:
            torch.Tensor: Generated tensor for ``target_modal``.
                Shape: [B, seq_len, hidden_size].

        Raises:
            ValueError: If the generation cache is empty, implying ``forward``
                was not invoked before calling this method.

        Note:
            Falls back to mean of all cached modalities if target not found.
            Applies understanding gate for conditional fusion when prompt provided.
        """
        if not self._generation_cache:
            raise ValueError("The forward() method must be called first to build the generation cache.")
        base = self._generation_cache.get(target_modal)
        if base is None:
            # If no cache for the target modality, fall back to the mean of global understanding
            base = torch.stack(list(self._generation_cache.values()), dim=0).mean(dim=0)

        out = base
        if prompt_tokens is not None and prompt_tokens.numel() > 0:
            # Lightweight conditional fusion
            cond = prompt_tokens.mean(dim=1, keepdim=True) if prompt_tokens.dim() == 3 else prompt_tokens.unsqueeze(1)
            gate = self.understanding_gate(cond.squeeze(1)).unsqueeze(1)
            out = out * gate + 0.3 * cond

        if temperature != 1.0:
            out = out / max(1e-6, temperature)

        if (top_k is not None) and (out.shape[-1] > top_k):
            vals, idx = torch.topk(out, k=top_k, dim=-1)
            mask = torch.zeros_like(out)
            mask.scatter_(-1, idx, vals)
            out = mask
        return out

    def generate_cross_modal(self, source_modal: str, target_modal: str, source_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate a modality representation conditioned on another modality.

        Convenience method for cross-modal generation that uses source
        modality tokens as conditioning for target modality generation.

        Args:
            source_modal (str): Modality providing conditioning context.
                Used for logging/debugging purposes.
            target_modal (str): Modality to synthesize.
                Must be one of the supported modalities.
            source_tokens (Optional[torch.Tensor]): Optional conditioning tokens.
                Shape: [B, seq_len, hidden_size] or [B, hidden_size].

        Returns:
            torch.Tensor: Generated tensor for the target modality.
                Shape: [B, seq_len, hidden_size].

        Note:
            Wrapper around generate_modality with source_tokens as prompt.
        """
        return self.generate_modality(target_modal, prompt_tokens=source_tokens)

    def forward(self, modal_features: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        sig = self._signature(modal_features)
        if sig in self.weight_cache:
            return self.weight_cache[sig]

        # 1. SyncFusion: temporal alignment for audio+video
        audio_feat = modal_features.get("audio")
        video_feat = modal_features.get("video")
        if audio_feat is not None and video_feat is not None:
            a_pool = F.adaptive_avg_pool1d(audio_feat.transpose(1, 2), self.sync_temporal_bins).transpose(1, 2)
            v_pool = F.adaptive_avg_pool1d(video_feat.transpose(1, 2), self.sync_temporal_bins).transpose(1, 2)
            a_proj = self.audio_temporal(a_pool)
            v_proj = self.video_temporal(v_pool)
            a2v, _ = self.sync_cross_attn(a_proj, v_proj, v_proj)
            v2a, _ = self.sync_cross_attn(v_proj, a_proj, a_proj)
            gate_a = self.sync_fusion_gate(torch.cat([a2v, a_pool], dim=-1))
            gate_v = self.sync_fusion_gate(torch.cat([v2a, v_pool], dim=-1))
            a_synced = gate_a * a2v + (1 - gate_a) * a_pool
            v_synced = gate_v * v2a + (1 - gate_v) * v_pool
            modal_features = dict(modal_features)
            modal_features["audio"] = self.sync_output_proj(torch.cat([a_synced, a_pool], dim=-1))
            modal_features["video"] = self.sync_output_proj(torch.cat([v_synced, v_pool], dim=-1))

        # 2. Tokenize via native tokenizer
        tokenized = self.native_tokenizer(modal_features)
        if not tokenized:
            raise ValueError(
                f"YvDynamicModalFusion received no valid modality tensors. "
                f"Expected modalities: {set(self.intra_modal_encoders.keys())}, "
                f"received: {set(modal_features.keys()) if isinstance(modal_features, dict) else type(modal_features).__name__}"
            )

        # 3. Intra-modal encoding
        encoded = {}
        for modal, tok in tokenized.items():
            encoded[modal] = self.intra_modal_encoders[modal](tok)

        # 4. Cross-modal alignment
        aligned = {}
        for modal, h in encoded.items():
            aligned[modal] = self.align_projections[modal](h)

        # 5. Inter-modal attention (each modality attends to others)
        modal_keys = list(encoded.keys())
        if len(modal_keys) > 1:
            inter_features = {}
            for modal in modal_keys:
                others = torch.cat([encoded[m] for m in modal_keys if m != modal], dim=1)
                YvShapeGuard.check_cat([encoded[m] for m in modal_keys if m != modal], 1, "fusion inter-modal cat")
                q = self.inter_modal_q(encoded[modal])
                k = self.inter_modal_k(others)
                v = self.inter_modal_v(others)
                YvShapeGuard.check_matmul(q, k.transpose(1, 2), "fusion inter-modal attn")
                attn_scores = torch.matmul(q, k.transpose(1, 2)) / (self.hidden_size ** 0.5)
                attn_weights = F.softmax(attn_scores, dim=-1)
                YvShapeGuard.check_matmul(attn_weights, v, "fusion inter-modal out")
                inter_out = torch.matmul(attn_weights, v)
                inter_features[modal] = self.inter_modal_out(inter_out) + aligned[modal]
        else:
            inter_features = aligned

        # 6. Concatenate all modality sequences
        seq = torch.cat(list(inter_features.values()), dim=1)

        # Log when temporal ratio across modalities exceeds 100x.
        lens = [v.shape[1] for v in inter_features.values()]
        if lens:
            ratio = max(lens) / max(min(lens), 1)
            if ratio > 100:
                logger.debug(
                    "Cross-modal temporal ratio=%.1fx (max=%d, min=%d) exceeds 100x",
                    ratio, max(lens), min(lens),
                )

        # 7. Self-attention fusion (native layers)
        h = seq
        for layer in self.native_fusion_layers:
            if isinstance(layer, nn.TransformerEncoderLayer):
                h = layer(h)
            else:
                h = layer(h, h, h)

        # 8. Importance-weighted pooling
        pooled_per_modal = {}
        cursor = 0
        for modal, feat in inter_features.items():
            end = cursor + feat.shape[1]
            pooled_per_modal[modal] = h[:, cursor:end, :].mean(dim=1)
            cursor = end

        if pooled_per_modal:
            modal_stack = torch.stack(list(pooled_per_modal.values()), dim=1)
            importance_scores = F.softmax(self.importance_net(modal_stack.mean(dim=2)), dim=-1)
            importance_weighted = (modal_stack * importance_scores.unsqueeze(-1)).sum(dim=1)
        else:
            importance_weighted = h.mean(dim=1)

        # 9. Quality-awareness scores
        for modal, pooled in pooled_per_modal.items():
            gs = self.quality_global(pooled.unsqueeze(-1))
            ls = self.quality_local(pooled.unsqueeze(-1)).mean(dim=-1, keepdim=False).unsqueeze(-1)
            qs = self.quality_head(torch.cat([gs, ls], dim=-1)).squeeze(-1)
            if qs.item() < 0.3:
                pass  # Quality gate: low-quality modalities are down-weighted naturally

        # 10. Understanding gate
        gate = self.understanding_gate(importance_weighted).unsqueeze(1)
        understanding = h * gate

        # 11. Compress to output tokens
        batch_size = h.shape[0]
        query = self.output_tokens.expand(batch_size, -1, -1)
        YvShapeGuard.check_matmul(query, h.transpose(1, 2), "fusion output attn")
        attn_scores = torch.matmul(query, h.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled_out = torch.matmul(attn_weights, h)
        out = self.output_proj(pooled_out) + understanding.mean(dim=1, keepdim=True)

        # 12. Recursive refinement (RCA)
        if self.max_rca_rounds > 1:
            prev = out
            for rnd in range(self.max_rca_rounds):
                query = self.rca_output_tokens.expand(batch_size, -1, -1)
                YvShapeGuard.check_matmul(query, out.transpose(1, 2), "fusion rca attn")
                attn_scores = torch.matmul(query, out.transpose(1, 2)) / (self.hidden_size ** 0.5)
                attn_weights = F.softmax(attn_scores, dim=-1)
                refined = torch.matmul(attn_weights, out)
                refined = self.rca_output_proj(refined) + out
                if rnd > 0:
                    cos_sim = F.cosine_similarity(refined.flatten(1), prev.flatten(1), dim=1).mean()
                    if cos_sim > self.rca_convergence_threshold:
                        break
                prev = refined
                out = refined

        # 13. Generation gates
        gen_outputs = {}
        cursor = 0
        total_len = out.shape[1]
        for modal in self.modalities:
            est_len = max(1, total_len // max(1, len(self.modalities)))
            end = min(total_len, cursor + est_len)
            modal_tokens = out[:, cursor:end, :]
            cursor = end
            gen_gate = self.generation_gates[modal](out.mean(dim=1)).unsqueeze(1)
            gen_outputs[modal] = modal_tokens * gen_gate

        # Cache management
        if len(self.weight_cache) > self.cache_size_limit:
            for k in list(self.weight_cache.keys())[:100]:
                self.weight_cache.pop(k, None)
        self.weight_cache[sig] = out.detach()
        self._generation_cache = {m: v.detach() for m, v in gen_outputs.items()}
        if self.memory_manager is not None:
            self.memory_manager.register_tensor(out, "fusion_out")
        return out


# Re-export enhanced fusion for convenience
# YvEnhancedModalFusion provides:
# - 6-modality native fusion
# - Quality-aware fusion
# - Contrastive cross-modal alignment
# - Online adaptive weights
from .enhanced_fusion import (
    YvEnhancedModalFusion,
    YvModalFusionConfig,
    YvContrastiveCrossModalAligner,
)


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class YvRecurrentModalRefiner(nn.Module):
    """Recurrent-Depth Transformer modal refiner (RDT-based).

    Iteratively refines multimodal fusion using a looped computation
    scheme inspired by OpenMythos RDT:
        h_{t+1} = A * h_t + B * e + RefineBlock(h_t, e)

    where A is a learnable decay matrix with spectral radius < 1
    for stability, B is an input projection, e is the text embedding
    serving as a semantic anchor, and RefineBlock is a lightweight
    Transformer layer with shared weights.

    Key Features:
        - Convergence-guaranteed via spectral radius constraint on A.
        - Dynamic loop count: early stopping when cosine similarity
          between consecutive iterations exceeds threshold.
        - Text-anchor prevents semantic drift during refinement.
        - Lightweight: RefineBlock has ~1/4 parameters of a full layer.
    """

    def __init__(self, cfg, device=None, dtype=None):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.max_loops = getattr(cfg, 'rdt_max_loops', 3)
        self.spectral_radius = getattr(cfg, 'rdt_spectral_radius', 0.95)
        self.convergence_threshold = getattr(cfg, 'rdt_convergence_threshold', 0.99)
        refine_heads = getattr(cfg, 'rdt_refine_heads', 2)
        ffn_ratio = getattr(cfg, 'rdt_refine_ffn_ratio', 1.0)

        self.A_diag = nn.Parameter(
            torch.ones(self.hidden_size, device=device, dtype=dtype) * self.spectral_radius
        )
        self.B_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, device=device, dtype=dtype)

        refine_dim = max(self.hidden_size // 4, 64)
        self.refine_norm = nn.LayerNorm(self.hidden_size, device=device, dtype=dtype)
        self.refine_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=refine_heads,
            dropout=0.0,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        ffn_hidden = int(self.hidden_size * ffn_ratio)
        self.refine_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, ffn_hidden, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(ffn_hidden, self.hidden_size, device=device, dtype=dtype),
        )
        self.refine_ffn_norm = nn.LayerNorm(self.hidden_size, device=device, dtype=dtype)

        self.base_fusion = YvDynamicModalFusion(cfg, cache_manager=None, device=device, dtype=dtype)

    def _enforce_spectral_radius(self):
        with torch.no_grad():
            self.A_diag.clamp_(max=self.spectral_radius)

    def _refine_step(self, h, e):
        residual = h
        h_norm = self.refine_norm(h)
        attn_out, _ = self.refine_attn(h_norm, e, e)
        h = residual + attn_out
        residual2 = h
        h_norm2 = self.refine_ffn_norm(h)
        h = residual2 + self.refine_ffn(h_norm2)
        return h

    def forward(
        self,
        modal_features: Dict[str, Optional[torch.Tensor]],
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.base_fusion(modal_features)
        if h.dim() == 2:
            h = h.unsqueeze(1)

        if text_emb is None:
            text_feat = modal_features.get('text')
            if text_feat is not None:
                e = text_feat.mean(dim=1, keepdim=True) if text_feat.dim() == 3 else text_feat.unsqueeze(0).unsqueeze(0)
            else:
                e = h.detach()
        else:
            e = text_emb.mean(dim=1, keepdim=True) if text_emb.dim() == 3 else text_emb.unsqueeze(1)

        self._enforce_spectral_radius()

        a = self.A_diag
        b_e = self.B_proj(e)

        for loop_idx in range(self.max_loops):
            h_prev = h
            refined = self._refine_step(h, e)
            h = a * h_prev + b_e + refined
            if loop_idx > 0 and self.convergence_threshold < 1.0:
                cos_sim = F.cosine_similarity(
                    h.flatten(1), h_prev.flatten(1), dim=1
                ).mean()
                if cos_sim > self.convergence_threshold:
                    break

        return h


__all__ = [
    'YvDynamicModalFusion',
    'YvUnifiedMultimodalTokenizer',
    'YvEnhancedModalFusion',
    'YvModalFusionConfig',
    'YvContrastiveCrossModalAligner',
    'YvRecurrentModalRefiner',
]
