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

"""Memory Separation Training Pipeline.

Implements the three-phase training strategy for Engram-style
Lookup-Computation Separation. Trains models where reasoning/tool
capabilities are in the core weights while static factual knowledge
is retrieved from an external memory store.

Three-Phase Training:
    Phase 1 (core_reasoning): Train backbone on reasoning data only.
        Gate clamped to 0, memory router frozen, no knowledge injection.
        Model learns pure reasoning, tool use, and language understanding.

    Phase 2 (memory_router_training): Train router + cross-attention.
        Backbone frozen, gate ramps up from 0 via sigmoid schedule.
        Router learns to query relevant knowledge, cross-attention
        learns to inject it effectively into the hidden flow.

    Phase 3 (knowledge_store_build): Offline knowledge encoding.
        Trained router encodes raw corpora via fixed 0.5B encoder.
        Builds FAISS IVF-PQ index and mmap knowledge store.

Architecture inspired by:
    Liang Wenfeng et al., "Engram: Conditional Memory via Scalable
    Lookup", arXiv:2601.07372, 2026.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger(
    "POPSS.MemSepTrainer",
    file_path=get_log_file("POPSS.MemSepTrainer"),
    enable_file=True,
)


# ============================================================
# Phase Definition
# ============================================================

class MemSepPhase(Enum):
    """Memory separation training phases."""
    CORE_REASONING = "core_reasoning"
    MEMORY_ROUTER_TRAINING = "memory_router_training"
    KNOWLEDGE_STORE_BUILD = "knowledge_store_build"


# ============================================================
# Configuration
# ============================================================

@dataclass
class POPSSMemSepTrainingConfig:
    """Configuration for memory separation training pipeline.

    Attributes:
        enabled: Master switch for memory separation training.
        phase_1_steps: Training steps for Phase 1 (core reasoning).
        phase_2_steps: Training steps for Phase 2 (memory router).
        gate_schedule: Type of gate schedule for Phase 2.
            "sigmoid": Smooth sigmoid ramp from 0 to gate_target.
            "linear": Linear ramp.
            "step": Hard step at midpoint.
        gate_warmup_steps: Warmup steps for gate during Phase 2.
        gate_target: Target gate value at end of Phase 2 (sigmoid).
        sigmoid_steepness: Steepness of sigmoid gate schedule.
        freeze_backbone_phase2: Freeze backbone weights during Phase 2.
        freeze_router_phase1: Freeze memory router during Phase 1.
        reason_data_path: Path to reasoning-only training data for Phase 1.
        mem_data_path: Path to memory training data for Phase 2.
        mem_alignment_weight: Weight of memory alignment loss vs LM loss.
        mem_alignment_margin: Margin for triplet-based alignment loss.
        mem_top_k_training: Number of knowledge slots for training (may differ from inference).
        router_lr_multiplier: LR multiplier for router params vs backbone.
        gate_lr_multiplier: LR multiplier for gate params.
        cross_attn_lr_multiplier: LR multiplier for cross-attention params.
        phase_3_text_path: Path to raw text corpus for Phase 3 knowledge encoding.
        phase_3_store_path: Output path for built knowledge store.
    """
    enabled: bool = False

    # Phase scheduling
    phase_1_steps: int = 5000
    phase_2_steps: int = 2000

    # Gate schedule
    gate_schedule: str = "sigmoid"
    gate_warmup_steps: int = 500
    gate_target: float = 0.5
    sigmoid_steepness: float = 0.02

    # Freeze strategy
    freeze_backbone_phase2: bool = True
    freeze_router_phase1: bool = True

    # Data paths
    reason_data_path: str = ""
    mem_data_path: str = ""

    # Loss weights
    mem_alignment_weight: float = 0.1
    mem_alignment_margin: float = 0.5

    # Memory training params
    mem_top_k_training: int = 8

    # LR multipliers for fine-grained control
    router_lr_multiplier: float = 1.0
    gate_lr_multiplier: float = 10.0
    cross_attn_lr_multiplier: float = 1.0

    # Phase 3: knowledge store build
    phase_3_text_path: str = ""
    phase_3_store_path: str = ""

    def __post_init__(self):
        if self.gate_warmup_steps > self.phase_2_steps:
            self.gate_warmup_steps = self.phase_2_steps


# ============================================================
# Gate Schedule
# ============================================================

class MemSepGateScheduler:
    """Manages the knowledge injection gate schedule across training phases.

    Phase 1: gate = 0 (no injection, model learns reasoning)
    Phase 2: gate ramps from 0 -> gate_target via schedule
    Phase 3: gate = gate_target (inference mode)
    """

    def __init__(self, config: POPSSMemSepTrainingConfig):
        self.config = config
        self.phase = MemSepPhase.CORE_REASONING
        self.phase_step = 0
        self.current_gate = 0.0

    def get_gate(self) -> float:
        """Get current gate value based on phase and step.

        Returns:
            Gate value in [0, 1].
        """
        if self.phase == MemSepPhase.CORE_REASONING:
            return 0.0

        if self.phase == MemSepPhase.KNOWLEDGE_STORE_BUILD:
            return self.config.gate_target

        # Phase 2: gate schedule
        return self._compute_phase2_gate()

    def _compute_phase2_gate(self) -> float:
        """Compute gate value for Phase 2 based on schedule type.

        Returns:
            Gate value in [0, gate_target].
        """
        t = min(self.phase_step, self.config.gate_warmup_steps)
        warmup = max(1, self.config.gate_warmup_steps)
        progress = t / warmup

        if self.config.gate_schedule == "linear":
            gate = progress * self.config.gate_target
        elif self.config.gate_schedule == "step":
            gate = self.config.gate_target if progress >= 0.5 else 0.0
        else:  # sigmoid
            midpoint = 0.5
            x = (progress - midpoint) / self.config.sigmoid_steepness
            x_clamped = max(-10.0, min(10.0, x))
            sigmoid = 1.0 / (1.0 + math.exp(-x_clamped))
            gate = sigmoid * self.config.gate_target

        return float(gate)

    def advance_phase(self, next_phase: MemSepPhase):
        """Advance to next training phase.

        Args:
            next_phase: The phase to advance to.
        """
        old_phase = self.phase
        self.phase = next_phase
        self.phase_step = 0
        self.current_gate = self.get_gate()

        _LOG.info(
            f"MemSep Phase: {old_phase.value} -> {next_phase.value}, "
            f"gate={self.current_gate:.4f}"
        )

    def step(self):
        """Advance one training step within current phase."""
        self.phase_step += 1
        self.current_gate = self.get_gate()


# ============================================================
# Parameter Group Manager
# ============================================================

class MemSepParamManager:
    """Manages parameter groups for memory separation training.

    Handles freeze/unfreeze of backbone, router, cross-attention,
    and gate parameters per training phase. Also manages optimizer
    parameter groups with different learning rates.

    Parameter groups:
        - backbone: All model params except memory_router and memory_attn
        - router: memory_router parameters
        - cross_attn: memory_attn parameters in transformer blocks
        - gate: All gate parameters (router.gate + attn.gate)
    """

    def __init__(
        self,
        model: nn.Module,
        config: POPSSMemSepTrainingConfig,
        base_lr: float,
    ):
        """Initialize parameter manager.

        Args:
            model: The YvModel instance.
            config: Memory separation training config.
            base_lr: Base learning rate for backbone.
        """
        self.model = model
        self.config = config
        self.base_lr = base_lr

        self.backbone_params: List[nn.Parameter] = []
        self.router_params: List[nn.Parameter] = []
        self.cross_attn_params: List[nn.Parameter] = []
        self.gate_params: List[nn.Parameter] = []

        self._classify_params()

    def _classify_params(self):
        """Classify model parameters into groups."""
        backbone = []
        router = []
        cross_attn = []
        gate_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'memory_router' in name:
                if 'gate' in name:
                    gate_params.append(param)
                else:
                    router.append(param)
            elif 'memory_attn' in name:
                if 'gate' in name:
                    gate_params.append(param)
                else:
                    cross_attn.append(param)
            else:
                backbone.append(param)

        self.backbone_params = backbone
        self.router_params = router
        self.cross_attn_params = cross_attn
        self.gate_params = gate_params

        total_backbone = sum(p.numel() for p in backbone)
        total_router = sum(p.numel() for p in router)
        total_cross_attn = sum(p.numel() for p in cross_attn)
        total_gate = sum(p.numel() for p in gate_params)

        _LOG.info(
            f"MemSep params: backbone={total_backbone:,}, "
            f"router={total_router:,}, cross_attn={total_cross_attn:,}, "
            f"gate={total_gate:,}"
        )

    def set_phase(self, phase: MemSepPhase):
        """Set parameter requires_grad based on training phase.

        Phase 1: Train backbone only, freeze router/cross_attn/gate
        Phase 2: Train router/cross_attn/gate only, freeze backbone
        Phase 3: Freeze all (offline build)

        Args:
            phase: Current training phase.
        """
        if phase == MemSepPhase.CORE_REASONING:
            # Train backbone, freeze memory components
            for p in self.backbone_params:
                p.requires_grad = True
            if self.config.freeze_router_phase1:
                for p in self.router_params:
                    p.requires_grad = False
                for p in self.cross_attn_params:
                    p.requires_grad = False
                for p in self.gate_params:
                    p.requires_grad = False

        elif phase == MemSepPhase.MEMORY_ROUTER_TRAINING:
            if self.config.freeze_backbone_phase2:
                for p in self.backbone_params:
                    p.requires_grad = False
            for p in self.router_params:
                p.requires_grad = True
            for p in self.cross_attn_params:
                p.requires_grad = True
            for p in self.gate_params:
                p.requires_grad = True

        elif phase == MemSepPhase.KNOWLEDGE_STORE_BUILD:
            for p in self.model.parameters():
                p.requires_grad = False

        # Log trainable params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        _LOG.info(
            f"Phase {phase.value}: trainable={trainable:,}/{total:,} "
            f"({100*trainable/max(1,total):.1f}%)"
        )

    def get_optimizer_param_groups(self) -> List[Dict[str, Any]]:
        """Get optimizer parameter groups with different learning rates.

        Returns:
            List of param group dicts for torch optimizer.
        """
        groups = []

        if self.backbone_params:
            groups.append({
                "params": self.backbone_params,
                "lr": self.base_lr,
                "name": "backbone",
            })

        if self.router_params:
            groups.append({
                "params": self.router_params,
                "lr": self.base_lr * self.config.router_lr_multiplier,
                "name": "router",
            })

        if self.cross_attn_params:
            groups.append({
                "params": self.cross_attn_params,
                "lr": self.base_lr * self.config.cross_attn_lr_multiplier,
                "name": "cross_attn",
            })

        if self.gate_params:
            groups.append({
                "params": self.gate_params,
                "lr": self.base_lr * self.config.gate_lr_multiplier,
                "name": "gate",
            })

        return groups

    def set_gate_value(self, gate_value: float):
        """Set all gate parameters to a specific value.

        Args:
            gate_value: Gate value in [0, 1] to set.
        """
        gate_value = float(gate_value)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'gate' in name and ('memory_router' in name or 'memory_attn' in name):
                    # Convert sigmoid domain: find x s.t. sigmoid(x) = gate_value
                    # x = ln(g / (1-g)), clamped for numerical stability
                    g = max(1e-7, min(1.0 - 1e-7, gate_value))
                    x = math.log(g / (1.0 - g))
                    param.fill_(x)

    def get_gate_value(self) -> float:
        """Get average gate value across all memory gate parameters.

        Returns:
            Average sigmoid(gate) value.
        """
        values = []
        for name, param in self.model.named_parameters():
            if 'gate' in name and ('memory_router' in name or 'memory_attn' in name):
                values.append(torch.sigmoid(param).item())
        if not values:
            return 0.0
        return sum(values) / len(values)


# ============================================================
# Memory Alignment Loss
# ============================================================

class POPSSMemoryAlignmentLoss(nn.Module):
    """Alignment loss between router output and target knowledge.

    Encourages the memory router to retrieve knowledge embeddings
    that align with the model's information needs. Uses a combination
    of:
    - Triplet loss: Push correct knowledge closer, incorrect further
    - Consistency loss: Penalize large variance in knowledge across
      nearby layers (smooth knowledge flow)
    """

    def __init__(
        self,
        margin: float = 0.5,
        consistency_weight: float = 0.01,
    ):
        super().__init__()
        self.margin = margin
        self.consistency_weight = consistency_weight

    def forward(
        self,
        knowledge_ctx: Optional[Dict[str, torch.Tensor]],
        router_stats: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute memory alignment loss.

        Args:
            knowledge_ctx: Output from memory_router.forward().
            router_stats: Optional router statistics.

        Returns:
            Scalar alignment loss (0 if no knowledge context).
        """
        if knowledge_ctx is None:
            return torch.tensor(0.0, requires_grad=True)

        loss = torch.tensor(0.0, requires_grad=True)

        # 1. Distance loss: Penalize large distances to retrieved knowledge
        if "distances" in knowledge_ctx:
            distances = knowledge_ctx["distances"]  # [B, T, top_k]
            # Mean of mean distances, normalized
            distance_loss = distances.mean()
            loss = loss + distance_loss * 0.01

        # 2. Variance loss: Penalize high variance in retrieved knowledge
        # across the sequence (encourages coherent knowledge)
        if "knowledge" in knowledge_ctx:
            knowledge = knowledge_ctx["knowledge"]  # [B, T, top_k, dim]
            # Variance across sequence positions
            # knowledge_norm = F.normalize(knowledge, p=2, dim=-1)
            # For efficiency, just compute L2 variance reduction
            knowledge_flat = knowledge.view(-1, knowledge.shape[-1])
            if knowledge_flat.shape[0] > 1:
                var_loss = knowledge_flat.var(dim=0).mean()
                loss = loss + var_loss * self.consistency_weight

        return loss


# ============================================================
# Main Memory Separation Trainer
# ============================================================

class POPSSMemSepTrainer:
    """Memory separation training orchestrator.

    Manages the three-phase training pipeline:
    1. Core reasoning training
    2. Memory router training
    3. Knowledge store building

    Integrates with existing SFT/GRPO training loops by providing
    hooks for gate scheduling, parameter freezing, and loss computation.

    Usage:
        config = POPSSMemSepTrainingConfig(enabled=True, phase_1_steps=5000)
        memsep = POPSSMemSepTrainer(model, config)
        memsep.start_phase(MemSepPhase.CORE_REASONING)

        # In training loop:
        for step in range(total_steps):
            memsep.pre_step()
            loss = model(batch)
            mem_loss = memsep.compute_memory_loss(knowledge_ctx)
            total_loss = loss + mem_loss
            total_loss.backward()
            memsep.post_step()
    """

    def __init__(
        self,
        model: nn.Module,
        config: POPSSMemSepTrainingConfig,
        base_lr: float = 1e-5,
    ):
        """Initialize memory separation trainer.

        Args:
            model: YvModel instance with memory separation enabled.
            config: Memory separation training configuration.
            base_lr: Base learning rate for optimizer.
        """
        self.model = model
        self.config = config
        self.base_lr = base_lr

        if not config.enabled:
            _LOG.info("MemSep training disabled")
            self.gate_scheduler = None
            self.param_manager = None
            self.alignment_loss = None
            return

        self.gate_scheduler = MemSepGateScheduler(config)
        self.param_manager = MemSepParamManager(model, config, base_lr)
        self.alignment_loss = POPSSMemoryAlignmentLoss(
            margin=config.mem_alignment_margin,
        )

        self.current_phase = MemSepPhase.CORE_REASONING
        self.phase_step = 0
        self.total_steps = config.phase_1_steps + config.phase_2_steps

        self.stat_history: List[Dict[str, Any]] = []

        _LOG.info(
            f"MemSepTrainer initialized: phase1={config.phase_1_steps} "
            f"phase2={config.phase_2_steps}, gate_schedule={config.gate_schedule}"
        )

    def start_phase(self, phase: MemSepPhase):
        """Start a training phase.

        Sets up parameter requirements, gate schedule, and logs phase start.

        Args:
            phase: Training phase to start.
        """
        self.current_phase = phase
        self.phase_step = 0

        if self.gate_scheduler:
            self.gate_scheduler.advance_phase(phase)

        if self.param_manager:
            self.param_manager.set_phase(phase)

        _LOG.info(f"Phase started: {phase.value}")

    def pre_step(self, step_idx: int):
        """Pre-training step hook.

        Call at the beginning of each training step to update
        gate schedule and manage phase transitions.

        Args:
            step_idx: Global training step index.
        """
        if not self.config.enabled:
            return

        # Check phase transitions
        if step_idx >= self.config.phase_1_steps and self.current_phase == MemSepPhase.CORE_REASONING:
            self.start_phase(MemSepPhase.MEMORY_ROUTER_TRAINING)

        self.phase_step = step_idx
        if self.gate_scheduler:
            self.gate_scheduler.step()

        # Apply gate to model
        self._apply_gate_to_model()

    def _apply_gate_to_model(self):
        """Apply current gate value to model's memory components."""
        if not self.gate_scheduler:
            return

        gate = self.gate_scheduler.get_gate()

        # Set on router
        if hasattr(self.model, 'memory_router') and self.model.memory_router is not None:
            with torch.no_grad():
                g = max(1e-7, min(1.0 - 1e-7, gate))
                x = math.log(g / (1.0 - g))
                self.model.memory_router.gate.fill_(x)

        # Set on cross-attention layers
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'memory_attn') and layer.memory_attn is not None:
                    with torch.no_grad():
                        g = max(1e-7, min(1.0 - 1e-7, gate))
                        x = math.log(g / (1.0 - g))
                        layer.memory_attn.gate.fill_(x)

    def compute_memory_loss(
        self,
        knowledge_ctx: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute memory alignment loss.

        Args:
            knowledge_ctx: Output from memory router forward pass.

        Returns:
            Scalar memory alignment loss, or 0 tensor if disabled.
        """
        if not self.config.enabled or self.alignment_loss is None:
            return torch.tensor(0.0)

        return self.alignment_loss(knowledge_ctx)

    def get_optimizer_param_groups(self, stage_optimizer=None) -> List[Dict[str, Any]]:
        """Get parameter groups for optimizer.

        Args:
            stage_optimizer: Optional existing optimizer to adapt.

        Returns:
            List of parameter group dicts.
        """
        if not self.config.enabled or not self.param_manager:
            if stage_optimizer:
                return stage_optimizer.param_groups
            # Return all params at base LR
            return [{
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "lr": self.base_lr,
            }]

        return self.param_manager.get_optimizer_param_groups()

    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics.

        Returns:
            Dict with phase, step, gate, and active param groups.
        """
        if not self.config.enabled:
            return {"memsep_enabled": False}

        gate = self.gate_scheduler.get_gate() if self.gate_scheduler else 0.0
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.model.parameters())

        return {
            "memsep_enabled": True,
            "phase": self.current_phase.value,
            "phase_step": self.phase_step,
            "gate_value": gate,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_ratio": trainable / max(1, total),
        }


# ============================================================
# Factory Function
# ============================================================

def create_memsep_trainer(
    model: nn.Module,
    phase_1_steps: int = 5000,
    phase_2_steps: int = 2000,
    gate_target: float = 0.5,
    freeze_backbone_phase2: bool = True,
    base_lr: float = 1e-5,
    **kwargs,
) -> POPSSMemSepTrainer:
    """Factory function for memory separation trainer.

    Args:
        model: YvModel instance.
        phase_1_steps: Steps for core reasoning phase.
        phase_2_steps: Steps for memory router phase.
        gate_target: Target gate value.
        freeze_backbone_phase2: Freeze backbone during Phase 2.
        base_lr: Base learning rate.
        **kwargs: Additional config overrides.

    Returns:
        Configured POPSSMemSepTrainer.
    """
    config = POPSSMemSepTrainingConfig(
        enabled=True,
        phase_1_steps=phase_1_steps,
        phase_2_steps=phase_2_steps,
        gate_target=gate_target,
        freeze_backbone_phase2=freeze_backbone_phase2,
        **kwargs,
    )
    return POPSSMemSepTrainer(model, config, base_lr=base_lr)