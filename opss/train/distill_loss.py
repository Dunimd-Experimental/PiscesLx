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

"""
Knowledge Distillation Loss Functions

This module implements comprehensive distillation loss functions for
transferring knowledge from teacher models to student models.

Loss Types:
    - Logits Loss: KL divergence on output distributions
    - Hidden State Loss: MSE on intermediate representations
    - Attention Loss: MSE on attention patterns
    - Layer-wise Loss: Progressive layer alignment
    - Contrastive Loss: For remote API distillation

Key Features:
    - Temperature-scaled soft labels
    - Automatic dimension alignment
    - Multi-layer progressive distillation
    - Gradient-friendly implementations

Usage:
    from opss.train.distill_loss import (
        DistillationLossConfig,
        DistillationLoss,
    )
    
    config = DistillationLossConfig(
        temperature=2.0,
        alpha=1.0,
        beta=0.5,
    )
    loss_fn = DistillationLoss(config)
    loss = loss_fn(teacher_outputs, student_outputs, labels)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file


@dataclass
class DistillationLossConfig:
    """Configuration for distillation loss computation.
    
    Attributes:
        temperature: Temperature for softening distributions.
        alpha: Weight for logits distillation loss.
        beta: Weight for hidden state distillation loss.
        gamma: Weight for attention distillation loss.
        delta: Weight for layer-wise distillation loss.
        epsilon: Weight for task-specific loss.
        ignore_index: Index to ignore in loss computation.
        reduction: Loss reduction method ('mean', 'sum', 'none').
        layer_mapping: Mapping from student layers to teacher layers.
        hidden_proj_dim: Projection dimension for hidden alignment.
        normalize_hidden: Whether to normalize hidden states.
        attention_heads: Number of attention heads for alignment.
    """
    
    temperature: float = 2.0
    
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.3
    delta: float = 0.2
    epsilon: float = 1.0
    
    ignore_index: int = -100
    reduction: str = "mean"
    
    layer_mapping: Optional[Dict[int, int]] = None
    hidden_proj_dim: Optional[int] = None
    normalize_hidden: bool = True
    attention_heads: Optional[int] = None


class LogitsDistillationLoss(nn.Module):
    """Logits-level knowledge distillation loss.
    
    Computes KL divergence between teacher and student output distributions
    with temperature scaling for soft label generation.
    
    The loss is computed as:
        L = T^2 * KL(softmax(teacher_logits / T) || softmax(student_logits / T))
    
    The T^2 factor compensates for the gradient magnitude reduction
    caused by temperature scaling.
    """
    
    def __init__(self, temperature: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
    
    def forward(
        self,
        teacher_logits: Tensor,
        student_logits: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute logits distillation loss.
        
        Args:
            teacher_logits: Teacher output logits [batch, seq, vocab].
            student_logits: Student output logits [batch, seq, vocab].
            labels: Optional labels for masking [batch, seq].
            
        Returns:
            Scalar loss value.
        """
        if teacher_logits.dim() == 2:
            teacher_logits = teacher_logits.unsqueeze(0)
        if student_logits.dim() == 2:
            student_logits = student_logits.unsqueeze(0)
        
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='none',
            log_target=False,
        )
        
        loss = loss.sum(dim=-1)
        
        if labels is not None:
            mask = (labels != self.ignore_index).float()
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1.0)
        else:
            loss = loss.mean()
        
        loss = loss * (self.temperature ** 2)
        
        return loss


class HiddenStateDistillationLoss(nn.Module):
    """Hidden state knowledge distillation loss.
    
    Computes MSE loss between teacher and student hidden states,
    with optional projection for dimension alignment.
    """
    
    def __init__(
        self,
        teacher_dim: Optional[int] = None,
        student_dim: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.projection = None
        
        if teacher_dim is not None and student_dim is not None:
            if teacher_dim != student_dim:
                self.projection = nn.Linear(student_dim, teacher_dim, bias=False)
    
    def forward(
        self,
        teacher_hidden: Tensor,
        student_hidden: Tensor,
    ) -> Tensor:
        """Compute hidden state distillation loss.
        
        Args:
            teacher_hidden: Teacher hidden states [batch, seq, dim].
            student_hidden: Student hidden states [batch, seq, dim].
            
        Returns:
            Scalar loss value.
        """
        if self.projection is not None:
            student_hidden = self.projection(student_hidden)
        
        if self.normalize:
            teacher_hidden = F.normalize(teacher_hidden, dim=-1)
            student_hidden = F.normalize(student_hidden, dim=-1)
        
        loss = F.mse_loss(student_hidden, teacher_hidden)
        
        return loss


class AttentionDistillationLoss(nn.Module):
    """Attention pattern knowledge distillation loss.
    
    Computes MSE loss between teacher and student attention weights,
    transferring attention patterns from teacher to student.
    """
    
    def __init__(
        self,
        teacher_heads: Optional[int] = None,
        student_heads: Optional[int] = None,
    ):
        super().__init__()
        self.teacher_heads = teacher_heads
        self.student_heads = student_heads
    
    def _align_attention(
        self,
        attention: Tensor,
        target_heads: int,
    ) -> Tensor:
        """Align attention to target number of heads.
        
        Args:
            attention: Attention weights [batch, heads, seq, seq].
            target_heads: Target number of attention heads.
            
        Returns:
            Aligned attention weights.
        """
        current_heads = attention.shape[1]
        
        if current_heads == target_heads:
            return attention
        
        if current_heads > target_heads:
            factor = current_heads // target_heads
            attention = attention.view(
                attention.shape[0],
                target_heads,
                factor,
                attention.shape[2],
                attention.shape[3],
            ).mean(dim=2)
        else:
            factor = target_heads // current_heads
            attention = attention.unsqueeze(2).expand(
                -1, -1, factor, -1, -1
            ).reshape(
                attention.shape[0],
                target_heads,
                attention.shape[2],
                attention.shape[3],
            )
        
        return attention
    
    def forward(
        self,
        teacher_attn: Tensor,
        student_attn: Tensor,
    ) -> Tensor:
        """Compute attention distillation loss.
        
        Args:
            teacher_attn: Teacher attention weights [batch, heads, seq, seq].
            student_attn: Student attention weights [batch, heads, seq, seq].
            
        Returns:
            Scalar loss value.
        """
        if teacher_attn.dim() == 3:
            teacher_attn = teacher_attn.unsqueeze(1)
        if student_attn.dim() == 3:
            student_attn = student_attn.unsqueeze(1)
        
        if self.teacher_heads is not None:
            teacher_attn = self._align_attention(teacher_attn, self.teacher_heads)
        if self.student_heads is not None:
            student_attn = self._align_attention(student_attn, self.student_heads)
        
        min_heads = min(teacher_attn.shape[1], student_attn.shape[1])
        teacher_attn = teacher_attn[:, :min_heads]
        student_attn = student_attn[:, :min_heads]
        
        min_seq = min(teacher_attn.shape[2], student_attn.shape[2])
        teacher_attn = teacher_attn[:, :, :min_seq, :min_seq]
        student_attn = student_attn[:, :, :min_seq, :min_seq]
        
        loss = F.mse_loss(student_attn, teacher_attn)
        
        return loss


class LayerWiseDistillationLoss(nn.Module):
    """Layer-wise progressive knowledge distillation loss.
    
    Computes distillation loss at each layer, allowing progressive
    knowledge transfer from teacher to student.
    """
    
    def __init__(
        self,
        layer_mapping: Optional[Dict[int, int]] = None,
        teacher_dim: Optional[int] = None,
        student_dim: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.layer_mapping = layer_mapping
        self.normalize = normalize
        self.hidden_loss = HiddenStateDistillationLoss(
            teacher_dim=teacher_dim,
            student_dim=student_dim,
            normalize=normalize,
        )
    
    def forward(
        self,
        teacher_hiddens: List[Tensor],
        student_hiddens: List[Tensor],
    ) -> Tensor:
        """Compute layer-wise distillation loss.
        
        Args:
            teacher_hiddens: List of teacher hidden states per layer.
            student_hiddens: List of student hidden states per layer.
            
        Returns:
            Scalar loss value.
        """
        if not teacher_hiddens or not student_hiddens:
            return torch.tensor(0.0, device=teacher_hiddens[0].device if teacher_hiddens else 'cpu')
        
        total_loss = 0.0
        num_layers = 0
        
        if self.layer_mapping is not None:
            for student_idx, teacher_idx in self.layer_mapping.items():
                if student_idx < len(student_hiddens) and teacher_idx < len(teacher_hiddens):
                    loss = self.hidden_loss(
                        teacher_hiddens[teacher_idx],
                        student_hiddens[student_idx],
                    )
                    total_loss = total_loss + loss
                    num_layers += 1
        else:
            teacher_layers = len(teacher_hiddens)
            student_layers = len(student_hiddens)
            
            for i, student_hidden in enumerate(student_hiddens):
                teacher_idx = int(i * teacher_layers / student_layers)
                teacher_idx = min(teacher_idx, teacher_layers - 1)
                
                loss = self.hidden_loss(
                    teacher_hiddens[teacher_idx],
                    student_hidden,
                )
                total_loss = total_loss + loss
                num_layers += 1
        
        if num_layers > 0:
            total_loss = total_loss / num_layers
        
        return total_loss


class ContrastiveDistillationLoss(nn.Module):
    """Contrastive distillation loss for remote API teachers.
    
    When only teacher-generated text is available (no logits),
    this loss encourages the student to generate similar outputs.
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
    
    def forward(
        self,
        student_logits: Tensor,
        teacher_ids: Tensor,
        student_ids: Tensor,
    ) -> Tensor:
        """Compute contrastive distillation loss.
        
        Args:
            student_logits: Student output logits [batch, seq, vocab].
            teacher_ids: Teacher-generated token IDs [batch, seq].
            student_ids: Student input token IDs [batch, seq].
            
        Returns:
            Scalar loss value.
        """
        batch_size, seq_len, vocab_size = student_logits.shape
        
        student_logits_flat = student_logits.view(-1, vocab_size)
        teacher_ids_flat = teacher_ids.view(-1)
        
        loss = F.cross_entropy(
            student_logits_flat,
            teacher_ids_flat,
            ignore_index=self.ignore_index,
        )
        
        return loss


class DistillationLoss(nn.Module):
    """Comprehensive knowledge distillation loss.
    
    Combines multiple distillation loss types with configurable weights:
    - Logits distillation (KL divergence)
    - Hidden state distillation (MSE)
    - Attention distillation (MSE)
    - Layer-wise distillation (progressive)
    - Task-specific loss (cross-entropy)
    """
    
    def __init__(self, config: DistillationLossConfig):
        super().__init__()
        self.config = config
        self._LOG = PiscesLxLogger(
            "PiscesLx.Distill.Loss",
            file_path=get_log_file("PiscesLx.Distill.Loss"),
            enable_file=True,
        )
        
        self.logits_loss = LogitsDistillationLoss(
            temperature=config.temperature,
            ignore_index=config.ignore_index,
        )
        
        self.hidden_loss = HiddenStateDistillationLoss(
            normalize=config.normalize_hidden,
        )
        
        self.attn_loss = AttentionDistillationLoss(
            teacher_heads=config.attention_heads,
            student_heads=config.attention_heads,
        )
        
        self.layer_loss = LayerWiseDistillationLoss(
            layer_mapping=config.layer_mapping,
            normalize=config.normalize_hidden,
        )
        
        self.contrastive_loss = ContrastiveDistillationLoss(
            temperature=config.temperature,
            ignore_index=config.ignore_index,
        )
    
    def forward(
        self,
        teacher_outputs: Dict[str, Any],
        student_outputs: Dict[str, Any],
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute total distillation loss.
        
        Args:
            teacher_outputs: Dictionary with teacher outputs:
                - logits: [batch, seq, vocab]
                - hidden_states: Optional[List[Tensor]]
                - attentions: Optional[List[Tensor]]
            student_outputs: Dictionary with student outputs (same format).
            labels: Optional labels for task loss [batch, seq].
            
        Returns:
            Dictionary containing:
                - total: Total weighted loss
                - logits: Logits distillation loss
                - hidden: Hidden state loss (if applicable)
                - attention: Attention loss (if applicable)
                - layer: Layer-wise loss (if applicable)
                - task: Task-specific loss (if labels provided)
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=student_outputs['logits'].device)
        
        if teacher_outputs.get('logits') is not None and student_outputs.get('logits') is not None:
            logits_loss = self.logits_loss(
                teacher_outputs['logits'],
                student_outputs['logits'],
                labels,
            )
            losses['logits'] = logits_loss
            total_loss = total_loss + self.config.alpha * logits_loss
        
        teacher_hiddens = teacher_outputs.get('hidden_states')
        student_hiddens = student_outputs.get('hidden_states')
        
        if teacher_hiddens and student_hiddens:
            if isinstance(teacher_hiddens, (list, tuple)) and isinstance(student_hiddens, (list, tuple)):
                layer_loss = self.layer_loss(teacher_hiddens, student_hiddens)
                losses['layer'] = layer_loss
                total_loss = total_loss + self.config.delta * layer_loss
        
        teacher_attns = teacher_outputs.get('attentions')
        student_attns = student_outputs.get('attentions')
        
        if teacher_attns and student_attns:
            if isinstance(teacher_attns, (list, tuple)) and isinstance(student_attns, (list, tuple)):
                attn_losses = []
                for t_attn, s_attn in zip(teacher_attns, student_attns):
                    if t_attn is not None and s_attn is not None:
                        attn_losses.append(self.attn_loss(t_attn, s_attn))
                
                if attn_losses:
                    attn_loss = torch.stack(attn_losses).mean()
                    losses['attention'] = attn_loss
                    total_loss = total_loss + self.config.gamma * attn_loss
        
        if labels is not None:
            task_loss = F.cross_entropy(
                student_outputs['logits'].view(-1, student_outputs['logits'].shape[-1]),
                labels.view(-1),
                ignore_index=self.config.ignore_index,
            )
            losses['task'] = task_loss
            total_loss = total_loss + self.config.epsilon * task_loss
        
        losses['total'] = total_loss
        
        return losses


__all__ = [
    "DistillationLossConfig",
    "LogitsDistillationLoss",
    "HiddenStateDistillationLoss",
    "AttentionDistillationLoss",
    "LayerWiseDistillationLoss",
    "ContrastiveDistillationLoss",
    "DistillationLoss",
]
