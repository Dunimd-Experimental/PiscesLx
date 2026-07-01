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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def reverse_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Reverse KL divergence: KL(student || teacher).

    Used in OPD for multi-teacher distillation. Reverse KL focuses on
    modes where the teacher has high probability, avoiding mode-averaging.

    Args:
        student_logits: Logits from student model (B, V).
        teacher_logits: Logits from teacher model (B, V).
        temperature: Softmax temperature. Default: 1.0.

    Returns:
        Per-example reverse KL divergence.
    """
    p = F.log_softmax(student_logits / temperature, dim=-1)
    q = F.softmax(teacher_logits / temperature, dim=-1)
    kl = (q * (torch.log(q + 1e-10) - p)).sum(dim=-1)
    return kl * (temperature ** 2)


class YvOPDConfig:
    """Configuration for On-Policy Distillation.

    Args:
        domains: List of domain names (e.g., ['math', 'code', 'agent', 'instruction']).
        temperatures: Per-domain distillation temperatures.
        domain_weights: Per-domain loss weights.
        alpha_ce: Cross-entropy weight. Default: 1.0.
        alpha_kl: KL divergence weight. Default: 1.0.
        use_reverse_kl: Use reverse KL instead of forward KL. Default: True.
        vocab_size: Vocabulary size for full-logits distillation. Default: 151646.
    """

    def __init__(
        self,
        domains: List[str] = None,
        temperatures: List[float] = None,
        domain_weights: List[float] = None,
        alpha_ce: float = 1.0,
        alpha_kl: float = 1.0,
        use_reverse_kl: bool = True,
        vocab_size: int = 151646,
    ):
        self.domains = domains or ['math', 'code', 'agent', 'instruction']
        self.temperatures = temperatures or [2.0, 2.0, 1.5, 1.5]
        self.domain_weights = domain_weights or [1.0, 1.0, 1.0, 1.0]
        self.alpha_ce = alpha_ce
        self.alpha_kl = alpha_kl
        self.use_reverse_kl = use_reverse_kl
        self.vocab_size = vocab_size

        assert len(self.domains) == len(self.temperatures) == len(self.domain_weights)


@torch.no_grad()
def compute_teacher_logits(
    teacher_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_batch_tokens: int = 65536,
) -> torch.Tensor:
    """Compute teacher logits in a memory-efficient manner.

    Supports teacher weight offloading and cached hidden states.

    Args:
        teacher_model: Teacher model (in eval mode).
        input_ids: Input token IDs (B, T).
        attention_mask: Attention mask.
        max_batch_tokens: Max tokens per forward pass.

    Returns:
        Teacher logits (B, T, V).
    """
    teacher_model.eval()
    B, T = input_ids.shape

    with torch.no_grad():
        if B * T <= max_batch_tokens:
            outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            return logits

        logits_list = []
        for start in range(0, T, max_batch_tokens // B):
            end = min(start + max_batch_tokens // B, T)
            chunk = input_ids[:, start:end]
            mask_chunk = attention_mask[:, start:end] if attention_mask is not None else None
            outputs = teacher_model(input_ids=chunk, attention_mask=mask_chunk)
            logits_chunk = outputs['logits'] if isinstance(outputs, dict) else outputs
            logits_list.append(logits_chunk)

        return torch.cat(logits_list, dim=1)


def compute_opd_loss(
    student_logits: torch.Tensor,
    teacher_logits_list: List[torch.Tensor],
    labels: torch.Tensor,
    config: YvOPDConfig,
    domain_idx: int = 0,
) -> Dict[str, torch.Tensor]:
    """Compute OPD loss for a single domain.

    Args:
        student_logits: Student model logits (B, T, V).
        teacher_logits_list: List of teacher logits, one per domain.
        labels: Target labels (B, T).
        config: OPD configuration.
        domain_idx: Current domain index.

    Returns:
        Dict with 'loss', 'ce_loss', 'kl_loss', 'kl_breakdown'.
    """
    B, T, V = student_logits.shape

    # Cross-entropy loss
    ce_loss = F.cross_entropy(
        student_logits.view(-1, V),
        labels.view(-1),
        ignore_index=-100,
        reduction='mean',
    )

    # Multi-teacher reverse KL
    kl_loss = 0.0
    kl_breakdown = {}

    for i, teacher_logits in enumerate(teacher_logits_list):
        if teacher_logits is None:
            continue

        temp = config.temperatures[i]
        weight = config.domain_weights[i]

        if config.use_reverse_kl:
            kld = reverse_kl_divergence(
                student_logits.view(-1, V),
                teacher_logits.view(-1, V),
                temperature=temp,
            )
        else:
            # Forward KL: KL(teacher || student)
            p = F.log_softmax(teacher_logits.view(-1, V) / temp, dim=-1)
            q = F.softmax(student_logits.view(-1, V) / temp, dim=-1)
            kld = (F.softmax(teacher_logits.view(-1, V) / temp, dim=-1) *
                   (F.log_softmax(teacher_logits.view(-1, V) / temp, dim=-1) -
                    F.log_softmax(student_logits.view(-1, V) / temp, dim=-1))).sum(dim=-1)
            kld = kld * (temp ** 2)

        # Mask padding tokens
        valid_mask = (labels.view(-1) != -100).float()
        kld = (kld * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        domain_kl = kld * weight
        kl_loss = kl_loss + domain_kl
        kl_breakdown[config.domains[i]] = domain_kl.item()

    loss = config.alpha_ce * ce_loss + config.alpha_kl * kl_loss

    return {
        'loss': loss,
        'ce_loss': ce_loss.detach(),
        'kl_loss': kl_loss.detach() if isinstance(kl_loss, torch.Tensor) else torch.tensor(kl_loss),
        'kl_breakdown': kl_breakdown,
    }


class YvOPDTrainer:
    """On-Policy Distillation trainer.

    Orchestrates the two-stage OPD pipeline:
    1. Loads domain-specialized teacher checkpoints
    2. Distills knowledge into student via self-sampled trajectories

    Usage:
        >>> trainer = YvOPDTrainer(student_model, opd_config)
        >>> trainer.add_teacher('math', math_teacher_model)
        >>> trainer.add_teacher('code', code_teacher_model)
        >>> loss = trainer.train_step(input_ids, labels)
    """

    def __init__(
        self,
        student_model: nn.Module,
        config: YvOPDConfig = None,
    ):
        self.student = student_model
        self.config = config or YvOPDConfig()
        self.teachers: Dict[str, nn.Module] = {}
        self.teacher_logits_cache: Dict[str, torch.Tensor] = {}
        self.step = 0

    def add_teacher(self, domain: str, teacher_model: nn.Module):
        """Register a domain-specialized teacher.

        Args:
            domain: Domain name (must match config.domains).
            teacher_model: Teacher model.
        """
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        self.teachers[domain] = teacher_model

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        domain_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Single OPD training step.

        Args:
            input_ids: Input tokens (B, T).
            labels: Target tokens (B, T).
            attention_mask: Attention mask.
            domain_idx: Current domain index for weighting.

        Returns:
            Dict with losses.
        """
        self.student.train()

        # Forward student
        student_out = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        student_logits = student_out['logits']

        # Forward teachers (cached)
        teacher_logits_list = []
        for domain in self.config.domains:
            if domain in self.teachers:
                t_logits = compute_teacher_logits(self.teachers[domain], input_ids, attention_mask)
                teacher_logits_list.append(t_logits)
            else:
                teacher_logits_list.append(None)

        loss_dict = compute_opd_loss(
            student_logits=student_logits,
            teacher_logits_list=teacher_logits_list,
            labels=labels,
            config=self.config,
            domain_idx=domain_idx,
        )

        self.step += 1
        return loss_dict


def create_opd_training_config(
    domains: List[str] = None,
    base_temperature: float = 2.0,
) -> YvOPDConfig:
    """Create OPD config with sensible defaults.

    Args:
        domains: Domain names.
        base_temperature: Base distillation temperature.

    Returns:
        YvOPDConfig instance.
    """
    domains = domains or ['math', 'code', 'agent', 'instruction']
    n = len(domains)
    return YvOPDConfig(
        domains=domains,
        temperatures=[base_temperature] * n,
        domain_weights=[1.0 / n] * n,
        alpha_ce=1.0,
        alpha_kl=1.0,
        use_reverse_kl=True,
    )
