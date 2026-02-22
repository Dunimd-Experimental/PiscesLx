#!/usr/bin/env/python3
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

"""Enhanced multimodal fusion components for Yv architecture.

This module provides advanced multimodal fusion components for the Yv
model, including quality-aware fusion, cross-modal attention, and modality
alignment for comprehensive multi-modal representation learning.

Module Components:
    1. YvModalFusionConfig:
       - Configuration dataclass for fusion parameters
       - Modality list and attention settings

    2. _IntraModalEncoder:
       - Transformer encoder for intra-modal processing
       - Self-attention within each modality

    3. _CrossModalAligner:
       - Cross-modal feature alignment
       - Modality-specific projections
       - Feature fusion and residual connections

    4. _InterModalAttention:
       - Cross-modal attention mechanism
       - Multi-head attention across modalities

Key Features:
    - Quality-aware fusion with adaptive weighting
    - Cross-modal attention for feature interaction
    - Modality-specific projection layers
    - Residual connections for gradient flow
    - Support for 6 modalities (text, image, audio, video, document, agentic)

Performance Characteristics:
    - Intra-modal encoding: O(L^2 * hidden_size) per modality
    - Cross-modal alignment: O(N * hidden_size^2) where N = modalities
    - Inter-modal attention: O(N^2 * L * hidden_size)

Usage Example:
    >>> from model.multimodal.enhanced_fusion import YvModalFusionConfig
    >>> 
    >>> # Initialize configuration
    >>> config = YvModalFusionConfig(
    ...     hidden_size=2048,
    ...     num_modalities=6,
    ...     use_quality_aware_fusion=True
    >>> )

Note:
    Default modalities: text, image, audio, video, document, agentic.
    Uses GELU activation and LayerNorm for stability.
    Supports both quality-aware and standard fusion modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class YvModalFusionConfig:
    """Configuration for enhanced multimodal fusion.
    
    A comprehensive configuration dataclass that defines parameters for
    the enhanced multimodal fusion system, including modality settings,
    attention configuration, and fusion options.
    
    Attributes:
        hidden_size (int): Hidden dimension for all projections. Default: 2048.
        num_modalities (int): Number of modalities to fuse. Default: 6.
        modalities (List[str]): List of modality names. Default:
            ["text", "image", "audio", "video", "document", "agentic"].
        num_heads (int): Number of attention heads. Default: 16.
        num_layers (int): Number of fusion layers. Default: 4.
        dropout (float): Dropout probability. Default: 0.1.
        use_quality_aware_fusion (bool): Enable quality-aware weighting. Default: True.
        use_modality_attention (bool): Enable cross-modal attention. Default: True.
        use_cross_modal_alignment (bool): Enable feature alignment. Default: True.
    
    Example:
        >>> config = YvModalFusionConfig(hidden_size=4096, num_modalities=4)
        >>> print(config.modalities)
        ['text', 'image', 'audio', 'video', 'document', 'agentic']
    
    Note:
        Modality list is truncated to num_modalities during processing.
        Quality-aware fusion adapts weights based on feature quality scores.
    """
    hidden_size: int = 2048
    num_modalities: int = 6
    modalities: List[str] = field(default_factory=lambda: [
        "text", "image", "audio", "video", "document", "agentic"
    ])
    num_heads: int = 16
    num_layers: int = 4
    dropout: float = 0.1
    use_quality_aware_fusion: bool = True
    use_modality_attention: bool = True
    use_cross_modal_alignment: bool = True


class _IntraModalEncoder(nn.Module):
    """Transformer encoder for intra-modal feature processing.
    
    A multi-layer transformer encoder that processes features within
    each modality independently, applying self-attention and feed-forward
    layers for intra-modal representation learning.
    
    Architecture:
        - Multiple TransformerEncoderLayers
        - Self-attention with 8 heads
        - GELU activation in feed-forward
        - Batch-first processing
    
    Attributes:
        layers (nn.ModuleList): List of TransformerEncoderLayer modules.
    
    Example:
        >>> encoder = _IntraModalEncoder(hidden_size=2048, num_layers=2)
        >>> output = encoder(input_features)  # [B, L, hidden_size]
    
    Note:
        Uses GELU activation for smooth gradient flow.
        Feed-forward dimension is 4x hidden_size.
    """
    
    def __init__(self, hidden_size: int, num_layers: int = 2):
        """Initialize the intra-modal encoder.
        
        Args:
            hidden_size (int): Hidden dimension for features.
            num_layers (int): Number of transformer layers. Default: 2.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through transformer layers.
        
        Args:
            x (torch.Tensor): Input features [B, L, hidden_size].
        
        Returns:
            torch.Tensor: Encoded features [B, L, hidden_size].
        """
        for layer in self.layers:
            x = layer(x)
        return x


class _CrossModalAligner(nn.Module):
    """Cross-modal feature alignment and fusion module.
    
    Aligns features from different modalities through modality-specific
    projections and performs fusion via concatenation and residual connections.
    
    Architecture:
        1. Modality Projections:
           - Linear projection per modality
           - Projects all modalities to common hidden_size
        
        2. Feature Fusion:
           - Concatenation of all modality features
           - Two-layer MLP with GELU activation
           - LayerNorm for stability
        
        3. Residual Connection:
           - Adds fused features to each modality (scaled by 0.1)
    
    Attributes:
        hidden_size (int): Common hidden dimension.
        num_modalities (int): Number of modalities to align.
        modal_fusion (nn.Sequential): Fusion network.
        {modality}_proj (nn.Linear): Per-modality projection layers.
    
    Example:
        >>> aligner = _CrossModalAligner(hidden_size=2048, num_modalities=6)
        >>> aligned = aligner({"text": text_feat, "image": img_feat})
    
    Note:
        Modality projections are created dynamically based on modality names.
        Residual connection uses 0.1 scaling for stability.
    """
    
    def __init__(self, hidden_size: int, num_modalities: int):
        """Initialize the cross-modal aligner.
        
        Args:
            hidden_size (int): Common hidden dimension for all modalities.
            num_modalities (int): Number of modalities to process.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        
        for i, modality in enumerate(["text", "image", "audio", "video", "document", "agentic"][:num_modalities]):
            setattr(self, f"{modality}_proj", nn.Linear(hidden_size, hidden_size))
        
        self.modal_fusion = nn.Sequential(
            nn.Linear(hidden_size * num_modalities, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
    
    def forward(self, encoded_modals: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Align and fuse features from multiple modalities.
        
        Args:
            encoded_modals (Dict[str, torch.Tensor]): Dictionary mapping
                modality names to feature tensors [B, L, hidden_size].
        
        Returns:
            Dict[str, torch.Tensor]: Aligned features with residual fusion.
        
        Note:
            Features are concatenated, fused, and added back as residuals.
        """
        aligned = {}
        for name, feat in encoded_modals.items():
            if hasattr(self, f"{name}_proj"):
                aligned[name] = getattr(self, f"{name}_proj")(feat)
            else:
                aligned[name] = feat
        
        all_features = torch.cat(list(aligned.values()), dim=-1)
        fused = self.modal_fusion(all_features)
        
        for name in aligned:
            aligned[name] = aligned[name] + fused * 0.1
        
        return aligned


class _InterModalAttention(nn.Module):
    """Cross-modal attention mechanism for feature interaction.
    
    Implements multi-head attention across different modalities, enabling
    features from one modality to attend to features from other modalities.
    
    Architecture:
        - Multi-head attention with configurable heads
        - Cross-modal query-key-value computation
        - Attention weights for modality interaction
    
    Attributes:
        hidden_size (int): Hidden dimension for features.
    
    Example:
        >>> attention = _InterModalAttention(hidden_size=2048, num_modalities=6)
        >>> attended = attention(modal_features)
    
    Note:
        Enables bidirectional attention between all modality pairs.
    """
    
    def __init__(self, hidden_size: int, num_modalities: int, num_heads: int = 8):
        """Initialize the inter-modal attention module.
        
        Args:
            hidden_size (int): Hidden dimension for features.
            num_modalities (int): Number of modalities to process.
            num_heads (int): Number of attention heads. Default: 8.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        self.modal_k_proj = nn.ModuleDict()
        self.modal_v_proj = nn.ModuleDict()
        
        for modality in ["text", "image", "audio", "video", "document", "agentic"][:num_modalities]:
            self.modal_k_proj[modality] = nn.Linear(hidden_size, hidden_size)
            self.modal_v_proj[modality] = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        query_modal: str,
        target_modals: Dict[str, torch.Tensor],
        context_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = target_modals[query_modal].shape
        
        Q = self.q_proj(target_modals[query_modal]).view(B, T, self.num_heads, self.head_dim)
        
        all_K = []
        all_V = []
        for name, feat in target_modals.items():
            K = self.modal_k_proj[name](feat).view(B, -1, self.num_heads, self.head_dim)
            V = self.modal_v_proj[name](feat).view(B, -1, self.num_heads, self.head_dim)
            all_K.append(K)
            all_V.append(V)
        
        K_cat = torch.cat(all_K, dim=1)
        V_cat = torch.cat(all_V, dim=1)
        
        attn_scores = torch.einsum("bthd,bkhd->bhtk", Q, K_cat) / math.sqrt(self.head_dim)
        
        if context_mask is not None:
            extended_mask = torch.ones(B, 1, T, K_cat.size(1), dtype=torch.bool, device=attn_scores.device)
            attn_scores = attn_scores.masked_fill(~extended_mask, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        output = torch.einsum("bhtk,bkhd->bthd", attn_weights, V_cat)
        output = output.contiguous().view(B, T, self.hidden_size)
        output = self.o_proj(output)
        
        attention_map = attn_weights.mean(dim=1)
        
        return output, attention_map


class _ModalityImportanceLearner(nn.Module):
    def __init__(self, hidden_size: int, num_modalities: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        
        self.importance_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        self.gate_net = nn.ModuleDict()
        for modality in ["text", "image", "audio", "video", "document", "agentic"][:num_modalities]:
            self.gate_net[modality] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        encoded_modals: Dict[str, torch.Tensor],
        task_embedding: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        importance_weights = {}
        gate_values = {}
        
        for name, feat in encoded_modals.items():
            pooled = feat.mean(dim=1)
            
            importance_weights[name] = self.importance_net(pooled)[:, list(encoded_modals.keys()).index(name)]
            
            gate_values[name] = self.gate_net[name](feat)
        
        return importance_weights, gate_values


class _MultiScaleQualityAssessor(nn.Module):
    def __init__(self, hidden_size: int, num_modalities: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        
        self.scale_encoders = nn.ModuleDict()
        for modality in ["text", "image", "audio", "video", "document", "agentic"][:num_modalities]:
            self.scale_encoders[modality] = nn.ModuleDict({
                'global': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.GELU(),
                    nn.Linear(hidden_size // 2, hidden_size // 4)
                ),
                'local': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.GELU(),
                    nn.Linear(hidden_size // 2, hidden_size // 4)
                ),
                'temporal': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.GELU(),
                    nn.Linear(hidden_size // 2, hidden_size // 4)
                )
            })
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_size * num_modalities, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_modalities),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        encoded_modals: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        scale_features = {name: {} for name in encoded_modals.keys()}
        
        for name, feat in encoded_modals.items():
            if feat.dim() == 3:
                pooled = feat.mean(dim=1)
                seq_pooled = feat.mean(dim=1)
                temporal_pooled = feat.mean(dim=1) if feat.size(1) > 1 else feat.mean(dim=[1, 2])
            else:
                pooled = feat.mean(dim=1)
                seq_pooled = pooled
                temporal_pooled = pooled
            
            scale_features[name]['global'] = self.scale_encoders[name]['global'](pooled)
            scale_features[name]['local'] = self.scale_encoders[name]['local'](seq_pooled)
            scale_features[name]['temporal'] = self.scale_encoders[name]['temporal'](temporal_pooled)
        
        all_features = []
        for name, scales in scale_features.items():
            combined = torch.cat([
                scales['global'],
                scales['local'],
                scales['temporal']
            ], dim=-1)
            all_features.append(combined)
        
        fused = self.fusion_proj(torch.cat(all_features, dim=-1))
        
        quality_scores = self.quality_head(fused)
        
        quality_dict = {}
        for i, name in enumerate(encoded_modals.keys()):
            quality_dict[name] = {
                'global': torch.sigmoid(scale_features[name]['global'].mean()),
                'local': torch.sigmoid(scale_features[name]['local'].mean()),
                'temporal': torch.sigmoid(scale_features[name]['temporal'].mean()),
                'overall': quality_scores[:, i]
            }
        
        return quality_dict


class _QualityAwareFusion(nn.Module):
    def __init__(self, hidden_size: int, num_modalities: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * num_modalities, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * num_modalities),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * num_modalities, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
    
    def forward(
        self,
        interacted: Dict[str, torch.Tensor],
        importance_weights: Dict[str, torch.Tensor],
        gate_values: Dict[str, torch.Tensor],
        quality_scores: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        batch_size = list(interacted.values())[0].size(0)
        
        modality_features = []
        for name in interacted.keys():
            feat = interacted[name]
            
            importance = importance_weights.get(name, torch.ones(batch_size, 1, device=feat.device))
            gate = gate_values.get(name, torch.ones_like(feat))
            quality = quality_scores.get(name, {}).get('overall', torch.ones(batch_size, 1, device=feat.device))
            
            combined_weight = importance.unsqueeze(-1) * gate * quality.unsqueeze(-1).unsqueeze(-1)
            
            weighted_feat = feat * combined_weight
            
            modality_features.append(weighted_feat)
        
        max_len = max(f.size(1) for f in modality_features)
        
        padded_features = []
        attention_mask = []
        for feat in modality_features:
            if feat.size(1) < max_len:
                padding = torch.zeros(
                    batch_size,
                    max_len - feat.size(1),
                    self.hidden_size,
                    device=feat.device,
                    dtype=feat.dtype
                )
                padded = torch.cat([feat, padding], dim=1)
                mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=feat.device)
                mask[:, :feat.size(1)] = False
            else:
                padded = feat
                mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=feat.device)
            
            padded_features.append(padded)
            attention_mask.append(mask)
        
        batch_attention = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool, device=padded_features[0].device)
        for mask in attention_mask:
            batch_attention = batch_attention | mask.unsqueeze(1)
        
        query = padded_features[0]
        key = torch.cat(padded_features, dim=1)
        value = torch.cat(padded_features, dim=1)
        
        attended, _ = self.fusion_attention(
            query, key, value,
            key_padding_mask=batch_attention.any(dim=-1)
        )
        
        concat_features = torch.cat([query, attended], dim=-1)
        
        output = self.output_proj(concat_features)
        
        return output


class YvEnhancedModalFusion(nn.Module):
    def __init__(self, config: Optional[YvModalFusionConfig] = None):
        super().__init__()
        if config is None:
            config = YvModalFusionConfig()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_modalities = config.num_modalities
        self.modalities = config.modalities
        
        self.intra_modal_encoders = nn.ModuleDict()
        for modality in self.modalities:
            self.intra_modal_encoders[modality] = _IntraModalEncoder(
                hidden_size=self.hidden_size,
                num_layers=config.num_layers // 2
            )
        
        if config.use_cross_modal_alignment:
            self.cross_modal_aligner = _CrossModalAligner(
                hidden_size=self.hidden_size,
                num_modalities=self.num_modalities
            )
        else:
            self.cross_modal_aligner = None
        
        self.inter_modal_attention = _InterModalAttention(
            hidden_size=self.hidden_size,
            num_modalities=self.num_modalities,
            num_heads=config.num_heads
        )
        
        if config.use_modality_attention:
            self.modality_importance = _ModalityImportanceLearner(
                hidden_size=self.hidden_size,
                num_modalities=self.num_modalities
            )
        else:
            self.modality_importance = None
        
        if config.use_quality_aware_fusion:
            self.quality_assessor = _MultiScaleQualityAssessor(
                hidden_size=self.hidden_size,
                num_modalities=self.num_modalities
            )
            
            self.quality_fusion = _QualityAwareFusion(
                hidden_size=self.hidden_size,
                num_modalities=self.num_modalities,
                dropout=config.dropout
            )
        else:
            self.quality_assessor = None
            self.quality_fusion = None
        
        self.output_norm = nn.LayerNorm(self.hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def _standardize_input(
        self,
        modal_features: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        standardized = {}
        for name in self.modalities:
            if name in modal_features:
                feat = modal_features[name]
                if isinstance(feat, dict):
                    feat = feat.get('features', feat)
                
                if not isinstance(feat, torch.Tensor):
                    continue
                
                if feat.dim() == 2:
                    feat = feat.unsqueeze(1)
                
                standardized[name] = feat
        
        return standardized
    
    def forward(
        self,
        modal_features: Dict[str, Any],
        task_type: Optional[str] = None,
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        encoded_modals = self._standardize_input(modal_features)
        
        if not encoded_modals:
            raise ValueError("No valid modal features provided")
        
        intra_encoded = {}
        for name, feat in encoded_modals.items():
            intra_encoded[name] = self.intra_modal_encoders[name](feat)
        
        if self.cross_modal_aligner is not None:
            aligned = self.cross_modal_aligner(intra_encoded)
        else:
            aligned = intra_encoded
        
        interacted = {}
        attention_maps = {}
        for query_modal in aligned.keys():
            result, attention_map = self.inter_modal_attention(
                query_modal=query_modal,
                target_modals=aligned
            )
            interacted[query_modal] = result
            attention_maps[query_modal] = attention_map
        
        importance_weights = None
        gate_values = None
        if self.modality_importance is not None:
            importance_weights, gate_values = self.modality_importance(
                interacted,
                None
            )
        else:
            batch_size = list(interacted.values())[0].size(0)
            importance_weights = {name: torch.ones(batch_size, 1, device=list(interacted.values())[0].device)
                                  for name in interacted.keys()}
            gate_values = {name: torch.ones_like(list(interacted.values())[0])
                           for name in interacted.keys()}
        
        quality_scores = None
        if self.quality_assessor is not None:
            quality_scores = self.quality_assessor(interacted)
        
        modality_sensitivity = {}
        for name in interacted.keys():
            feat = interacted[name]
            sensitivity = torch.norm(feat, dim=-1).mean() / (feat.var(dim=-1).mean() + 1e-8)
            modality_sensitivity[name] = sensitivity
        
        total_sensitivity = sum(modality_sensitivity.values())
        for name in interacted.keys():
            balance_factor = modality_sensitivity[name] / (total_sensitivity + 1e-8)
            importance_weights[name] = importance_weights[name] * (0.5 + 0.5 * balance_factor)
        
        if self.quality_fusion is not None and quality_scores is not None:
            fused_output = self.quality_fusion(
                interacted,
                importance_weights,
                gate_values,
                quality_scores
            )
        else:
            batch_size = list(interacted.values())[0].size(0)
            device = list(interacted.values())[0].device
            
            all_features = []
            for name in interacted.keys():
                importance = importance_weights.get(name, torch.ones(batch_size, 1, device=device))
                gate = gate_values.get(name, torch.ones_like(list(interacted.values())[0]))
                combined_weight = importance.unsqueeze(-1) * gate
                all_features.append(interacted[name] * combined_weight)
            
            max_len = max(f.size(1) for f in all_features)
            padded_features = [
                F.pad(f, (0, 0, 0, max_len - f.size(1)), value=0)
                for f in all_features
            ]
            concat_features = torch.cat(padded_features, dim=-1)
            
            fused_output = self.output_norm(concat_features)
        
        hallucination_scores = {}
        for name, feat in interacted.items():
            activation_norm = torch.norm(feat, dim=-1)
            if activation_norm.numel() > 1:
                anomaly_score = (activation_norm > activation_norm.mean() + 2 * activation_norm.std()).float().mean()
            else:
                anomaly_score = torch.tensor(0.0, device=feat.device)
            hallucination_scores[name] = anomaly_score
        
        for name in hallucination_scores:
            if hallucination_scores[name] > 0.3:
                importance_weights[name] = importance_weights[name] * 0.5
        
        output = self.output_norm(fused_output)
        
        result = {
            'fused_features': output,
            'modality_features': interacted,
            'attention_maps': attention_maps,
            'importance_weights': importance_weights,
            'quality_scores': quality_scores
        }
        
        if return_intermediate:
            result['intra_encoded'] = intra_encoded
            result['aligned'] = aligned
        
        return result


class YvOnlineQualityAdaptation(nn.Module):
    """
    Online Quality Adaptation for modality fusion with reinforcement learning.
    
    Enhancements:
    - Online learning from user feedback
    - Task reward-driven quality weight adjustment
    - Experience replay for stable adaptation
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_modalities: int,
        modalities: List[str],
        learning_rate: float = 0.001,
        memory_size: int = 1000,
        reward_window: int = 100
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.modalities = modalities
        self.learning_rate = learning_rate
        self.reward_window = reward_window
        
        self.quality_scaler = nn.Parameter(torch.ones(num_modalities))
        
        self.quality_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        self.register_buffer('quality_history', torch.zeros(memory_size, num_modalities))
        self.register_buffer('reward_history', torch.zeros(memory_size))
        self.register_buffer('memory_ptr', torch.tensor(0))
        self.register_buffer('episode_reward', torch.tensor(0.0))
        
        self.optimizer = torch.optim.Adam([self.quality_scaler], lr=learning_rate)
        
        self.current_step = 0
        
    def _update_memory(self, quality_weights: torch.Tensor, reward: float):
        ptr = int(self.memory_ptr) % self.quality_history.size(0)
        self.quality_history[ptr] = quality_weights.detach()
        self.reward_history[ptr] = torch.tensor(reward)
        self.memory_ptr += 1
        self.episode_reward += reward
        
    def _sample_experience(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.memory_ptr < 10:
            return None, None
        
        num_samples = min(32, int(self.memory_ptr))
        indices = torch.randperm(int(self.memory_ptr))[:num_samples]
        
        sampled_quality = self.quality_history[indices].mean(dim=0)
        sampled_reward = self.reward_history[indices].mean()
        
        return sampled_quality, sampled_reward
    
    def adapt(
        self,
        fused_output: torch.Tensor,
        task_reward: float,
        end_of_episode: bool = False
    ) -> Dict[str, torch.Tensor]:
        self.current_step += 1
        
        current_quality = torch.softmax(self.quality_scaler, dim=0)
        
        self._update_memory(current_quality, task_reward)
        
        if end_of_episode or self.current_step % 10 == 0:
            sampled_quality, avg_reward = self._sample_experience()
            
            if sampled_quality is not None:
                loss = F.mse_loss(current_quality, sampled_quality) * (-avg_reward)
                loss = loss + 0.01 * torch.norm(current_quality - 1.0 / self.num_modalities)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.quality_scaler], max_norm=1.0)
                self.optimizer.step()
            
            if end_of_episode:
                self.episode_reward = torch.tensor(0.0)
        
        return {
            'quality_weights': current_quality,
            'adaptation_step': self.current_step
        }
    
    def forward(
        self,
        fused_output: torch.Tensor,
        quality_scores: Dict[str, Dict[str, torch.Tensor]],
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = fused_output.size(0)
        
        if task_embedding is not None:
            task_quality = self.quality_net(task_embedding.mean(dim=0))
        else:
            task_quality = torch.ones(batch_size, self.num_modalities, device=fused_output.device) / self.num_modalities
        
        adaptive_quality = torch.softmax(self.quality_scaler, dim=0).unsqueeze(0) * 0.7 + task_quality * 0.3
        
        modality_quality = []
        for i, name in enumerate(self.modalities):
            if name in quality_scores:
                q = quality_scores[name].get('overall', torch.ones(batch_size, 1, device=fused_output.device))
                modality_quality.append(q)
            else:
                modality_quality.append(torch.ones(batch_size, 1, device=fused_output.device))
        
        modality_quality = torch.cat(modality_quality, dim=-1)
        combined_quality = adaptive_quality * modality_quality
        combined_quality = combined_quality / (combined_quality.sum(dim=-1, keepdim=True) + 1e-8)
        
        return combined_quality


class YvTaskAwareModalityImportance(nn.Module):
    """
    Task-Aware Modality Importance Learner with task embedding modulation.
    
    Enhancements:
    - Task embedding for modality weight adjustment
    - Support for task types: reasoning, generation, understanding, etc.
    - Gradient-based importance scoring
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_modalities: int,
        modalities: List[str],
        num_task_types: int = 8,
        task_embed_dim: int = 64
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.modalities = modalities
        self.task_embed_dim = task_embed_dim
        
        self.task_embedding = nn.Embedding(num_task_types, task_embed_dim)
        
        self.task_type_encoder = nn.Sequential(
            nn.Linear(task_embed_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        self.gradient_importance = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self.gate_net = nn.ModuleDict()
        for modality in modalities:
            self.gate_net[modality] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            )
        
        self.modality_to_idx = {m: i for i, m in enumerate(modalities)}
        
    def _compute_gradient_importance(
        self,
        encoded_modals: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        gradient_importance = {}
        
        for name, feat in encoded_modals.items():
            feat = feat.detach().requires_grad_(True)
            grad_score = self.gradient_importance(feat.mean(dim=[0, 1]))
            gradient_importance[name] = torch.sigmoid(grad_score).squeeze()
        
        return gradient_importance
    
    def forward(
        self,
        encoded_modals: Dict[str, torch.Tensor],
        task_type: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, Any]:
        batch_size = list(encoded_modals.values())[0].size(0)
        
        if task_type is not None:
            task_emb = self.task_embedding(task_type)
            task_importance = self.task_type_encoder(task_emb)
        else:
            task_importance = torch.ones(batch_size, self.num_modalities, device=list(encoded_modals.values())[0].device) / self.num_modalities
        
        grad_importance = self._compute_gradient_importance(encoded_modals)
        
        importance_weights = {}
        gate_values = {}
        
        for name, feat in encoded_modals.items():
            idx = self.modality_to_idx.get(name, 0)
            
            task_weight = task_importance[:, idx:idx+1]
            
            grad_score = grad_importance.get(name, torch.tensor(0.5, device=feat.device))
            grad_weight = grad_score.unsqueeze(0).unsqueeze(-1).expand_as(feat)
            
            importance_weights[name] = task_weight
            
            modality_gate = self.gate_net[name](feat)
            gate_values[name] = modality_gate
        
        result = {
            'importance_weights': importance_weights,
            'gate_values': gate_values,
            'task_importance': task_importance,
            'gradient_importance': grad_importance
        }
        
        if return_details:
            result['task_embedding'] = task_emb if task_type is not None else None
        
        return result


class YvContrastiveCrossModalAligner(nn.Module):
    """
    Contrastive Cross-Modal Aligner with negative sampling.
    
    Enhancements:
    - Contrastive learning for better modal alignment
    - Hard negative mining
    - InfoNCE loss optimization
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_modalities: int,
        modalities: List[str],
        temperature: float = 0.1,
        num_negatives: int = 4,
        align_dim: int = 256
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.modalities = modalities
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.align_dim = align_dim
        
        self.modality_projection = nn.ModuleDict()
        for modality in modalities:
            self.modality_projection[modality] = nn.Sequential(
                nn.Linear(hidden_size, align_dim),
                nn.LayerNorm(align_dim),
                nn.ReLU()
            )
        
        self.alignment_head = nn.ModuleDict()
        for modality in modalities:
            self.alignment_head[modality] = nn.Sequential(
                nn.Linear(align_dim, align_dim),
                nn.ReLU(),
                nn.Linear(align_dim, hidden_size)
            )
        
        self.contrastive_loss_fn = nn.CrossEntropyLoss()
        
        self.register_buffer('negative_bank', torch.zeros(1000, align_dim))
        self.register_buffer('negative_ptr', torch.tensor(0))
        
    def _mine_hard_negatives(
        self,
        anchors: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        similarity = torch.matmul(anchors, targets.t())
        hard_negatives = similarity.topk(self.num_negatives, dim=-1, largest=False)[1]
        return hard_negatives
    
    def forward(
        self,
        encoded_modals: Dict[str, torch.Tensor],
        return_loss: bool = True
    ) -> Dict[str, Any]:
        projected = {}
        for name, feat in encoded_modals.items():
            feat_2d = feat.mean(dim=1) if feat.dim() == 3 else feat
            projected[name] = self.modality_projection[name](feat_2d)
        
        aligned = {}
        for name, proj_feat in projected.items():
            aligned[name] = self.alignment_head[name](proj_feat)
        
        loss_dict = {}
        if return_loss and self.training:
            modality_list = list(encoded_modals.keys())
            
            for i, anchor_mod in enumerate(modality_list):
                anchor = projected[anchor_mod]
                positive_mod = modality_list[(i + 1) % len(modality_list)]
                positive = projected[positive_mod]
                
                if len(modality_list) > 2:
                    negative_mods = [m for m in modality_list if m not in [anchor_mod, positive_mod]]
                    if negative_mods:
                        negative = torch.cat([projected[m] for m in negative_mods], dim=0)
                    else:
                        negative = anchor
                else:
                    negative = anchor
                
                anchor_expanded = anchor.unsqueeze(1).expand(-1, positive.size(0), -1)
                positive_expanded = positive.unsqueeze(0).expand(anchor.size(0), -1, -1)
                
                pos_sim = torch.sum(anchor_expanded * positive_expanded, dim=-1) / self.temperature
                
                neg_sim = torch.matmul(anchor, negative.t()) / self.temperature
                
                logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
                labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
                
                loss = self.contrastive_loss_fn(logits, labels)
                loss_dict[f'contrastive_{anchor_mod}'] = loss
            
            total_loss = sum(loss_dict.values())
            loss_dict['total'] = total_loss
        
        return {
            'aligned': aligned,
            'loss': loss_dict.get('total', torch.tensor(0.0)),
            'loss_dict': loss_dict
        }


class YvModalityBenchmark:
    """
    Modality Fusion Benchmark for 6-modal evaluation.
    
    Metrics:
    - Alignment quality
    - Fusion effectiveness
    - Cross-modal reasoning
    - Quality awareness accuracy
    """
    
    def __init__(
        self,
        modalities: List[str] = None,
        device: str = 'cuda'
    ):
        self.modalities = modalities or ["text", "image", "audio", "video", "document", "agentic"]
        self.device = device
        self.metrics = {}
        self.reset()
    
    def reset(self):
        self.metrics = {
            'alignment_scores': {m: [] for m in self.modalities},
            'fusion_scores': [],
            'cross_modal_scores': {m: [] for m in self.modalities},
            'quality_scores': {m: [] for m in self.modalities},
            'importance_scores': {m: [] for m in self.modalities},
            'inference_times': [],
            'memory_usage': []
        }
    
    def evaluate_alignment(
        self,
        aligned_features: Dict[str, torch.Tensor],
        ground_truth: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        scores = {}
        
        for name, feat in aligned_features.items():
            if feat.dim() > 2:
                feat_2d = feat.mean(dim=1)
            else:
                feat_2d = feat
            
            norm = torch.norm(feat_2d, dim=-1)
            scores[f'{name}_norm'] = norm.mean().item()
            
            if ground_truth is not None:
                similarity = F.cosine_similarity(feat_2d, ground_truth, dim=-1)
                scores[f'{name}_similarity'] = similarity.mean().item()
        
        for m, score in scores.items():
            modality = m.split('_')[0]
            if modality in self.metrics['alignment_scores']:
                self.metrics['alignment_scores'][modality].append(score)
        
        return scores
    
    def evaluate_fusion(
        self,
        fused_output: torch.Tensor,
        modality_features: Dict[str, torch.Tensor],
        task_type: Optional[str] = None
    ) -> Dict[str, float]:
        scores = {}
        
        batch_size, seq_len, hidden_dim = fused_output.shape
        scores['fusion_variance'] = fused_output.var(dim=[0, 1]).mean().item()
        
        entropy = -(F.softmax(fused_output.mean(dim=[0, 1]), dim=-1) * 
                    torch.log(F.softmax(fused_output.mean(dim=[0, 1]), dim=-1) + 1e-8)).sum().item()
        scores['fusion_entropy'] = entropy
        
        modality_contribution = {}
        total_norm = fused_output.norm(dim=-1).sum()
        for name, feat in modality_features.items():
            contribution = (feat * fused_output).sum() / (feat.norm() * fused_output.norm() + 1e-8)
            modality_contribution[name] = contribution.item()
        scores['modality_contribution'] = modality_contribution
        
        self.metrics['fusion_scores'].append(scores)
        
        return scores
    
    def evaluate_quality_awareness(
        self,
        quality_scores: Dict[str, Dict[str, torch.Tensor]],
        gt_quality: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        scores = {}
        
        for name, qual in quality_scores.items():
            overall = qual.get('overall', torch.tensor(1.0))
            if isinstance(overall, torch.Tensor):
                overall = overall.mean().item()
            scores[f'{name}_overall_quality'] = overall
            
            if gt_quality is not None and name in gt_quality:
                mse = (overall - gt_quality[name]) ** 2
                scores[f'{name}_quality_mse'] = mse
        
        for m, score in scores.items():
            modality = m.split('_')[0]
            if '_quality' in m and modality in self.metrics['quality_scores']:
                self.metrics['quality_scores'][modality].append(score)
        
        return scores
    
    def get_summary(self) -> Dict[str, Any]:
        summary = {}
        
        for modality, scores in self.metrics['alignment_scores'].items():
            if scores:
                summary[f'{modality}_alignment_mean'] = sum(scores) / len(scores)
                summary[f'{modality}_alignment_std'] = torch.tensor(scores).std().item() if len(scores) > 1 else 0.0
        
        if self.metrics['fusion_scores']:
            fusion_means = {k: [] for k in self.metrics['fusion_scores'][0].keys() 
                          if k != 'modality_contribution'}
            for score_dict in self.metrics['fusion_scores']:
                for k, v in score_dict.items():
                    if k != 'modality_contribution':
                        if isinstance(v, dict):
                            for mk, mv in v.items():
                                fusion_means.setdefault(mk, []).append(mv)
                        else:
                            fusion_means.setdefault(k, []).append(v)
            
            for k, v in fusion_means.items():
                summary[f'fusion_{k}_mean'] = sum(v) / len(v) if v else 0.0
        
        summary['total_samples'] = len(self.metrics['fusion_scores'])
        summary['modality_count'] = len(self.modalities)
        
        return summary
