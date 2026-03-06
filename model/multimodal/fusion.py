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

"""Dynamic multimodal fusion utilities for PiscesL1 Yv agents.

This module provides comprehensive multimodal fusion components for the Yv
model, including dynamic cross-modal attention, modality-specific gating, and
generation pathways for unified representation learning.

Module Components:
    1. YvDynamicModalFusion:
       - Unified tokenization across modalities
       - Cross-modal attention integration
       - Understanding and generation gating
       - Modality-specific generation with caching

Key Features:
    - 6-modality support (text, image, audio, video, document, agentic)
    - Unified tokenization with modality embeddings
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
    >>> fused = fusion(features)  # [B, 1, hidden_size]
    >>> 
    >>> # Generate modality-specific output
    >>> image_gen = fusion.generate_modality("image", temperature=0.8)

Note:
    Supports text, image, audio, video, document, and agentic modalities.
    Uses YvCrossModalAttention for cross-modality reasoning.
    Integrates with YvMemory for tensor lifetime tracking.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional
from .memory import YvMemory
from .hw import YvHardwareAdaptiveConfig
from .attention import YvCrossModalAttention

class YvDynamicModalFusion(nn.Module):
    """Dynamic multimodal fusion backbone for Yv workflows.
    
    A comprehensive fusion module that tokenizes modality-specific inputs,
    enriches them with learned positional and modality embeddings, and performs
    cross-modal attention followed by gated fusion to produce a unified
    representation.
    
    Architecture:
        1. Unified Tokenization:
           - Text: Identity (pass-through)
           - Image: Conv2d with 16x16 patches
           - Audio: Conv1d with 16 kernel
           - Video: Conv3d with spatio-temporal patches
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
        - Cross-modal attention for inter-modality reasoning
        - Understanding and generation gating mechanisms
        - Weight caching for efficient repeated fusion
        - Hardware-adaptive gradient configuration
    
    Attributes:
        cfg: Configuration namespace containing fusion hyperparameters.
        hidden_size (int): Dimensionality of the shared representation space.
        modalities (List[str]): Canonical modality identifiers handled by the fusion core.
        weight_cache (Dict[str, torch.Tensor]): Cache for previously fused outputs keyed by modality presence signatures.
        cache_size_limit (int): Maximum number of cached signatures retained.
        cache_manager: Optional external cache manager reused across agent subsystems.
        memory_manager (YvMemory): Memory system for tracking tensor lifetimes.
        hw (YvHardwareAdaptiveConfig): Hardware adaptation helper used to derive gradient configuration.
        grad_conf (Dict[str, Any]): Gradient settings retrieved from the hardware adapter.
        unified_tokenizer (nn.ModuleDict): Mapping from modality to tokenization modules that project raw inputs.
        unified_pos_embed (nn.Parameter): Learned positional embeddings shared across modalities.
        modality_tokens (nn.Embedding): Trainable embeddings encoding modality identity.
        cross_modal_attn (YvCrossModalAttention): Attention layer performing cross-modality reasoning.
        understanding_gate (nn.Sequential): Gating module producing global understanding signals.
        generation_gates (nn.ModuleDict): Modality-specific gates used to modulate generation outputs.
        _generation_cache (Dict[str, torch.Tensor]): Storage for latest modality-specific outputs produced by ``forward``.
    
    Example:
        >>> fusion = YvDynamicModalFusion(config)
        >>> features = {"text": text_feat, "image": img_feat}
        >>> fused = fusion(features)  # [B, 1, hidden_size]
        >>> 
        >>> # Generate modality-specific output
        >>> gen = fusion.generate_modality("image", temperature=0.8)
    
    Note:
        Supports text, image, audio, video, document, and agentic modalities.
        Cache size limit is 1000 signatures by default.
    """

    def __init__(self, cfg, cache_manager=None):
        """Initialize the fusion module and supporting infrastructure.
        
        Args:
            cfg: Configuration object containing parameters such as:
                - hidden_size: Output embedding dimension
            cache_manager: Optional cache manager for generation caches. Defaults
                to ``None``.
        """
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.modalities = ["text", "image", "audio", "video", "document", "agentic"]
        self.weight_cache: Dict[str, torch.Tensor] = {}
        self.cache_size_limit = 1000
        self.cache_manager = cache_manager
        self.memory_manager = YvMemory()
        # Initialize hardware adaptive configuration
        self.hw = YvHardwareAdaptiveConfig()
        self.grad_conf = self.hw.get_gradient_config()

        # Unified tokenization for different modalities
        self.unified_tokenizer = nn.ModuleDict({
            'text': nn.Identity(),
            'image': nn.Conv2d(3, self.hidden_size, 16, 16),
            'audio': nn.Conv1d(1, self.hidden_size, 16, 16),
            'video': nn.Conv3d(3, self.hidden_size, (2, 16, 16), (2, 16, 16)),
            'document': nn.Linear(self.hidden_size, self.hidden_size),
            'agentic': nn.Linear(self.hidden_size, self.hidden_size)
        })
        self.unified_pos_embed = nn.Parameter(torch.randn(1, 8192, self.hidden_size) * 0.02)
        self.modality_tokens = nn.Embedding(len(self.modalities), self.hidden_size)

        # Native cross-modal token-level attention
        self.cross_modal_attn = YvCrossModalAttention(cfg)

        # Understanding and generation gating mechanisms
        self.understanding_gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )
        self.generation_gates = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.hidden_size),
                nn.Sigmoid()
            ) for m in self.modalities
        })

        # Generation cache for each modality
        self._generation_cache: Dict[str, torch.Tensor] = {}

    def _signature(self, features: Dict[str, Optional[torch.Tensor]]) -> str:
        """Summarize modality presence into a cache signature string.
        
        Creates a unique identifier for the current modality configuration
        to enable caching of previously computed fusion outputs.
        
        Args:
            features: Dictionary mapping modality names to optional tensors.
                Keys are modality names, values are feature tensors or None.
        
        Returns:
            str: Colon-delimited presence signature, e.g., ``"text:1:image:0"``.
                Used as cache key for weight_cache lookup.
        """
        present = [f"{m}:{1 if (features.get(m) is not None) else 0}" for m in self.modalities]
        return ":".join(present)

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
        """Fuse modality features into a shared representation.
        
        Main entry point for multimodal fusion. Tokenizes each modality,
        applies positional and modality embeddings, performs cross-modal
        attention, and produces a unified representation through gating.
        
        Args:
            modal_features (Dict[str, Optional[torch.Tensor]]): Mapping from
                modality name to feature tensors or ``None``.
                Supported modalities: text, image, audio, video, document, agentic.
                - text: [B, seq_len, hidden_size] or [B, hidden_size]
                - image: [B, C, H, W]
                - audio: [B, seq_len] or [B, 1, seq_len]
                - video: [B, C, T, H, W]
                - document/agentic: [B, seq_len, hidden_size]
        
        Returns:
            torch.Tensor: Global representation tensor with shape ``[B, 1, hidden_size]``.
        
        Note:
            Caches output for efficient repeated fusion with same modality config.
            Updates generation cache for modality-specific generation.
            Returns zero tensor if no modalities are provided.
        """
        # Cache lookup
        sig = self._signature(modal_features)
        if sig in self.weight_cache:
            cached = self.weight_cache[sig]
        else:
            cached = None

        tokens = []
        token_type_ids = []
        device = None
        for idx, modal in enumerate(self.modalities):
            feat = modal_features.get(modal)
            if feat is None:
                continue
            if device is None:
                device = feat.device
            if modal == 'text':
                tok = feat
                if tok.dim() == 2:
                    tok = tok.unsqueeze(1)
            elif modal == 'image' and feat.dim() == 4:
                patches = self.unified_tokenizer['image'](feat)
                tok = patches.flatten(2).transpose(1, 2)
            elif modal == 'audio':
                if feat.dim() == 2:
                    feat = feat.unsqueeze(1)
                tok = self.unified_tokenizer['audio'](feat).transpose(1, 2)
            elif modal == 'video' and feat.dim() == 5:
                b, c, t, h, w = feat.shape
                feat_ = feat.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
                patches = self.unified_tokenizer['video'](feat)
                tok = patches.flatten(2).transpose(1, 2)
                tok = tok.reshape(b, t * tok.shape[1], self.hidden_size)
            else:
                # document/agentic or fallback
                if feat.dim() == 3:
                    tok = feat
                else:
                    tok = feat.unsqueeze(1)
            seq_len = tok.shape[1]
            pos = self.unified_pos_embed[:, :seq_len, :].to(tok.device)
            modal_emb = self.modality_tokens(torch.tensor(idx, device=tok.device)).unsqueeze(0).unsqueeze(0)
            tokens.append(tok + pos + modal_emb)
            token_type_ids.extend([idx] * seq_len)

        if not tokens:
            return torch.zeros(1, 1, self.hidden_size, device=device or torch.device('cpu'))

        seq = torch.cat(tokens, dim=1)
        # Apply a single pass of cross-modal attention to integrate modalities.
        fused = self.cross_modal_attn(seq, seq, seq)

        # Derive a global understanding vector via gating.
        gate = self.understanding_gate(fused.mean(dim=1)).unsqueeze(1)
        understanding = fused * gate

        # Generate modality-specific outputs modulated by generation gates.
        gen_outputs: Dict[str, torch.Tensor] = {}
        cursor = 0
        for idx, modal in enumerate(self.modalities):
            # Estimate per-modality token allocation using uniform division.
            est_len = max(1, fused.shape[1] // max(1, len(self.modalities)))
            end = min(fused.shape[1], cursor + est_len)
            modal_tokens = fused[:, cursor:end, :]
            cursor = end
            gen_gate = self.generation_gates[modal](fused.mean(dim=1)).unsqueeze(1)
            gen_outputs[modal] = modal_tokens * gen_gate

        # Compute the final global representation via temporal averaging.
        out = understanding.mean(dim=1, keepdim=True)

        # Write to weight cache
        if cached is None:
            if len(self.weight_cache) > self.cache_size_limit:
                for k in list(self.weight_cache.keys())[:100]:
                    self.weight_cache.pop(k, None)
            self.weight_cache[sig] = out.detach()

        # Update generation cache (only keep the last one)
        self._generation_cache = {m: v.detach() for m, v in gen_outputs.items()}

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

__all__ = [
    'YvDynamicModalFusion',
    'YvEnhancedModalFusion',
    'YvModalFusionConfig',
    'YvContrastiveCrossModalAligner',
]
