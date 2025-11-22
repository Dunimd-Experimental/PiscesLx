#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

"""Dynamic multimodal fusion utilities for PiscesL1 Arctic agents.

This module implements :class:`ArcticDynamicModalFusion`, which harmonizes
token representations across modalities, applies cross-modal attention, and
manages gated generation pathways. It integrates with the Arctic memory
manager and hardware adaptation layers to orchestrate caching and gradient
configurations.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional
from .memory import ArcticMemoryManager
from .hw import ArcticHardwareAdaptiveConfig
from .attention import ArcticCrossModalAttention

class ArcticDynamicModalFusion(nn.Module):
    """Dynamic multimodal fusion backbone for Arctic workflows.

    The module tokenizes modality-specific inputs, enriches them with learned
    positional and modality embeddings, and performs cross-modal attention
    followed by gated fusion to produce a unified representation.

    Attributes:
        cfg: Configuration namespace containing fusion hyperparameters.
        hidden_size (int): Dimensionality of the shared representation space.
        modalities (List[str]): Canonical modality identifiers handled by the fusion core.
        weight_cache (Dict[str, torch.Tensor]): Cache for previously fused outputs keyed by modality presence signatures.
        cache_size_limit (int): Maximum number of cached signatures retained.
        cache_manager: Optional external cache manager reused across agent subsystems.
        memory_manager (ArcticMemoryManager): Manager responsible for tracking tensor lifetimes.
        hw (ArcticHardwareAdaptiveConfig): Hardware adaptation helper used to derive gradient configuration.
        grad_conf (Dict[str, Any]): Gradient settings retrieved from the hardware adapter.
        unified_tokenizer (nn.ModuleDict): Mapping from modality to tokenization modules that project raw inputs.
        unified_pos_embed (nn.Parameter): Learned positional embeddings shared across modalities.
        modality_tokens (nn.Embedding): Trainable embeddings encoding modality identity.
        cross_modal_attn (ArcticCrossModalAttention): Attention layer performing cross-modality reasoning.
        understanding_gate (nn.Sequential): Gating module producing global understanding signals.
        generation_gates (nn.ModuleDict): Modality-specific gates used to modulate generation outputs.
        _generation_cache (Dict[str, torch.Tensor]): Storage for latest modality-specific outputs produced by ``forward``.
    """

    def __init__(self, cfg, cache_manager=None):
        """Initialize the fusion module and supporting infrastructure.

        Args:
            cfg: Configuration object containing parameters such as ``hidden_size``.
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
        self.memory_manager = ArcticMemoryManager(enable_background=(cache_manager is None))
        self.memory_manager.start_monitoring()
        # Initialize hardware adaptive configuration
        self.hw = ArcticHardwareAdaptiveConfig()
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
        self.cross_modal_attn = ArcticCrossModalAttention(cfg)

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

        Args:
            features: Dictionary mapping modality names to optional tensors.

        Returns:
            str: Colon-delimited presence signature, e.g., ``"text:1:image:0"``.
        """
        present = [f"{m}:{1 if (features.get(m) is not None) else 0}" for m in self.modalities]
        return ":".join(present)

    def generate_modality(self, target_modal: str, prompt_tokens: Optional[torch.Tensor] = None,
                          temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate representations for a target modality using cached states.

        Args:
            target_modal (str): Name of the modality to synthesize.
            prompt_tokens (Optional[torch.Tensor]): Optional conditioning tokens.
            temperature (float): Softmax temperature scaling factor.
            top_k (Optional[int]): Retain top-k dimensions when specified.

        Returns:
            torch.Tensor: Generated tensor for ``target_modal``.

        Raises:
            ValueError: If the generation cache is empty, implying ``forward`` was not invoked.
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

        Args:
            source_modal (str): Modality providing conditioning context.
            target_modal (str): Modality to synthesize.
            source_tokens (Optional[torch.Tensor]): Optional conditioning tokens.

        Returns:
            torch.Tensor: Generated tensor for the target modality.
        """
        return self.generate_modality(target_modal, prompt_tokens=source_tokens)

    def forward(self, modal_features: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        """Fuse modality features into a shared representation.

        Args:
            modal_features (Dict[str, Optional[torch.Tensor]]): Mapping from
                modality name to feature tensors or ``None``.

        Returns:
            torch.Tensor: Global representation tensor with shape ``[B, 1, hidden_size]``.
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
