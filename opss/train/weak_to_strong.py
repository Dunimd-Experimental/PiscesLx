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
Weak-to-Strong Training Operators

Based on ICML 2024: "Weak-to-Strong Generalization: Eliciting Strong Capabilities 
with Weak Supervision" by OpenAI.

Key Discovery:
    A GPT-2 level model can supervise GPT-4, recovering 80% of the performance gap.
    Strong models can correct errors made by weak supervisors.

Key Features:
    - Weak model generates labels for strong model training
    - Confidence-weighted loss for quality filtering
    - Curriculum learning from easy to hard samples
    - Self-correction mechanism for strong model
    - Iterative amplification support

Training Pipeline:
    1. Weak model generates predictions on unlabeled data
    2. Filter by confidence threshold
    3. Strong model learns from weak labels
    4. Strong model can exceed weak model performance

Usage:
    from opss.train.weak_to_strong import (
        POPSSWeakToStrongConfig,
        POPSSWeakToStrongOperator,
    )
    
    config = POPSSWeakToStrongConfig(
        confidence_threshold=0.7,
        use_curriculum=True,
    )
    w2s = POPSSWeakToStrongOperator(weak_model, strong_model, config)
    loss = w2s.train_step(batch)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus
from configs.version import VERSION


class POPSSW2SMode(Enum):
    """Weak-to-strong training mode."""
    STANDARD = "standard"
    CURRICULUM = "curriculum"
    ITERATIVE = "iterative"
    ADVERSARIAL = "adversarial"


@dataclass
class POPSSWeakToStrongConfig:
    """Configuration for weak-to-strong training.
    
    Attributes:
        confidence_threshold: Minimum confidence for weak labels.
        use_curriculum: Enable curriculum learning.
        curriculum_steps: Number of curriculum steps.
        temperature: Temperature for confidence computation.
        label_smoothing: Label smoothing factor.
        loss_type: Loss type ('ce', 'kl', 'js').
        strong_model_lr: Learning rate for strong model.
        weak_model_frozen: Whether to freeze weak model.
        max_grad_norm: Maximum gradient norm.
        use_self_correction: Enable self-correction mechanism.
        correction_weight: Weight for self-correction loss.
    """
    
    confidence_threshold: float = 0.7
    use_curriculum: bool = True
    curriculum_steps: int = 1000
    temperature: float = 1.0
    label_smoothing: float = 0.1
    loss_type: str = "ce"
    strong_model_lr: float = 1e-5
    weak_model_frozen: bool = True
    max_grad_norm: float = 1.0
    use_self_correction: bool = True
    correction_weight: float = 0.3
    
    def __post_init__(self):
        if isinstance(self.loss_type, str):
            self.loss_type = self.loss_type.lower()


class POPSSWeakLabelGenerator:
    """Generates weak labels from weak model predictions.
    
    Handles confidence computation, filtering, and label generation
    for weak-to-strong training.
    """
    
    def __init__(self, weak_model: nn.Module, config: POPSSWeakToStrongConfig):
        self.weak_model = weak_model
        self.config = config
        self._LOG = PiscesLxLogger(
            "PiscesLx.W2S.LabelGen",
            file_path=get_log_file("PiscesLx.W2S.LabelGen"),
            enable_file=True,
        )
        
        # Freeze weak model if configured
        if config.weak_model_frozen:
            for param in self.weak_model.parameters():
                param.requires_grad = False
            self.weak_model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate weak labels with confidence scores.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            
        Returns:
            Tuple of (weak_labels, confidence, valid_mask).
        """
        self.weak_model.eval()
        
        # Forward pass
        outputs = self.weak_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # Compute probabilities
        probs = F.softmax(logits / self.config.temperature, dim=-1)
        
        # Get labels and confidence
        weak_labels = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1)[0]
        
        # Create valid mask based on confidence threshold
        valid_mask = confidence > self.config.confidence_threshold
        
        return weak_labels, confidence, valid_mask
    
    @torch.no_grad()
    def generate_with_features(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Generate weak labels with intermediate features.
        
        Returns dict with labels, confidence, and hidden states.
        """
        self.weak_model.eval()
        
        outputs = self.weak_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        
        probs = F.softmax(logits / self.config.temperature, dim=-1)
        weak_labels = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1)[0]
        valid_mask = confidence > self.config.confidence_threshold
        
        return {
            "weak_labels": weak_labels,
            "confidence": confidence,
            "valid_mask": valid_mask,
            "hidden_states": hidden_states,
            "logits": logits,
        }


class POPSSCurriculumScheduler:
    """Curriculum scheduler for weak-to-strong training.
    
    Gradually increases difficulty by adjusting confidence threshold
    and sample complexity over training steps.
    """
    
    def __init__(self, config: POPSSWeakToStrongConfig):
        self.config = config
        self.current_step = 0
        self.initial_threshold = config.confidence_threshold
        self.final_threshold = 0.5  # Lower threshold = harder samples
        self._LOG = PiscesLxLogger(
            "PiscesLx.W2S.Curriculum",
            file_path=get_log_file("PiscesLx.W2S.Curriculum"),
            enable_file=True,
        )
    
    def step(self):
        """Advance curriculum by one step."""
        self.current_step += 1
    
    def get_current_threshold(self) -> float:
        """Get current confidence threshold based on curriculum progress."""
        if not self.config.use_curriculum:
            return self.config.confidence_threshold
        
        progress = min(1.0, self.current_step / self.config.curriculum_steps)
        
        # Linear interpolation from initial to final threshold
        threshold = self.initial_threshold + (self.final_threshold - self.initial_threshold) * progress
        
        return threshold
    
    def get_difficulty_weight(self) -> float:
        """Get difficulty weight for loss computation."""
        if not self.config.use_curriculum:
            return 1.0
        
        progress = min(1.0, self.current_step / self.config.curriculum_steps)
        
        # Weight increases as we progress (harder samples get more weight)
        return 0.5 + 0.5 * progress


class POPSSSelfCorrection:
    """Self-correction mechanism for strong model.
    
    Allows strong model to identify and correct errors
    in weak labels, enabling super-weak performance.
    """
    
    def __init__(self, config: POPSSWeakToStrongConfig):
        self.config = config
        self._LOG = PiscesLxLogger(
            "PiscesLx.W2S.SelfCorrect",
            file_path=get_log_file("PiscesLx.W2S.SelfCorrect"),
            enable_file=True,
        )
    
    def compute_correction_loss(
        self,
        strong_logits: Tensor,
        weak_labels: Tensor,
        confidence: Tensor,
    ) -> Tensor:
        """Compute self-correction loss.
        
        Encourages strong model to deviate from weak labels
        when it has higher confidence in a different prediction.
        
        Args:
            strong_logits: Strong model logits.
            weak_labels: Weak model labels.
            confidence: Weak model confidence.
            
        Returns:
            Self-correction loss.
        """
        strong_probs = F.softmax(strong_logits, dim=-1)
        strong_labels = strong_probs.argmax(dim=-1)
        strong_confidence = strong_probs.max(dim=-1)[0]
        
        # Identify where strong model disagrees with weak model
        disagreement = (strong_labels != weak_labels).float()
        
        # Only correct when strong model is more confident
        should_correct = (strong_confidence > confidence).float()
        
        # Correction mask
        correction_mask = disagreement * should_correct
        
        if correction_mask.sum() == 0:
            return torch.tensor(0.0, device=strong_logits.device)
        
        # Encourage strong model's prediction
        loss = F.cross_entropy(
            strong_logits.view(-1, strong_logits.size(-1)),
            strong_labels.view(-1),
            reduction='none',
        )
        
        loss = (loss.view(strong_logits.size(0), -1) * correction_mask).mean()
        
        return loss


class _WeakToStrongOperatorImpl(PiscesLxOperatorInterface):
    """Weak-to-strong training operator implementation."""
    
    def __init__(
        self,
        weak_model: nn.Module,
        strong_model: nn.Module,
        config: POPSSWeakToStrongConfig,
    ):
        super().__init__()
        self._name = "training.weak_to_strong"
        self._version = VERSION
        self.type = "training"
        
        self.weak_model = weak_model
        self.strong_model = strong_model
        self.config = config
        
        self.label_generator = POPSSWeakLabelGenerator(weak_model, config)
        self.curriculum = POPSSCurriculumScheduler(config) if config.use_curriculum else None
        self.self_correction = POPSSSelfCorrection(config) if config.use_self_correction else None
        
        self._LOG = PiscesLxLogger(
            "PiscesLx.W2S.Operator",
            file_path=get_log_file("PiscesLx.W2S.Operator"),
            enable_file=True,
        )
        
        # Statistics tracking
        self._stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "corrections": 0,
        }
    
    def execute(self, params: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Execute weak-to-strong training step.
        
        Args:
            params: Dictionary containing:
                - input_ids: Input token IDs
                - attention_mask: Attention mask
                - labels: Optional ground truth labels (for evaluation)
                
        Returns:
            Training result with loss and statistics.
        """
        input_ids = params.get("input_ids")
        attention_mask = params.get("attention_mask")
        labels = params.get("labels")
        
        if input_ids is None:
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error="input_ids is required",
            )
        
        try:
            # Generate weak labels
            weak_outputs = self.label_generator.generate_with_features(
                input_ids, attention_mask
            )
            
            weak_labels = weak_outputs["weak_labels"]
            confidence = weak_outputs["confidence"]
            valid_mask = weak_outputs["valid_mask"]
            
            # Update curriculum threshold
            if self.curriculum is not None:
                current_threshold = self.curriculum.get_current_threshold()
                valid_mask = confidence > current_threshold
            
            # Filter valid samples
            if valid_mask.sum() == 0:
                return PiscesLxOperatorResult(
                    status=PiscesLxOperatorStatus.SUCCESS,
                    data={
                        "loss": torch.tensor(0.0),
                        "valid_samples": 0,
                        "total_samples": input_ids.size(0),
                    }
                )
            
            # Strong model forward
            strong_outputs = self.strong_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            strong_logits = strong_outputs.logits if hasattr(strong_outputs, 'logits') else strong_outputs[0]
            
            # Compute main loss
            main_loss = self._compute_loss(
                strong_logits, weak_labels, confidence, valid_mask
            )
            
            # Add self-correction loss
            total_loss = main_loss
            if self.self_correction is not None:
                correction_loss = self.self_correction.compute_correction_loss(
                    strong_logits, weak_labels, confidence
                )
                total_loss = total_loss + self.config.correction_weight * correction_loss
            
            # Update statistics
            self._stats["total_samples"] += input_ids.size(0)
            self._stats["valid_samples"] += valid_mask.sum().item()
            
            # Advance curriculum
            if self.curriculum is not None:
                self.curriculum.step()
            
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.SUCCESS,
                data={
                    "loss": total_loss,
                    "main_loss": main_loss,
                    "valid_samples": valid_mask.sum().item(),
                    "total_samples": input_ids.size(0),
                    "avg_confidence": confidence[valid_mask].mean().item() if valid_mask.sum() > 0 else 0.0,
                }
            )
            
        except Exception as e:
            self._LOG.error(f"Weak-to-strong training failed: {e}")
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error=str(e),
            )
    
    def _compute_loss(
        self,
        strong_logits: Tensor,
        weak_labels: Tensor,
        confidence: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Compute training loss."""
        if self.config.loss_type == "ce":
            # Cross-entropy with confidence weighting
            loss = F.cross_entropy(
                strong_logits.view(-1, strong_logits.size(-1)),
                weak_labels.view(-1),
                reduction='none',
                label_smoothing=self.config.label_smoothing,
            )
            loss = loss.view(strong_logits.size(0), -1).mean(dim=-1)
            
            # Weight by confidence
            weighted_loss = (loss * confidence * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
            
        elif self.config.loss_type == "kl":
            # KL divergence loss
            weak_probs = F.one_hot(weak_labels, num_classes=strong_logits.size(-1)).float()
            log_weak_probs = torch.log(weak_probs + 1e-8)
            
            loss = F.kl_div(
                F.log_softmax(strong_logits, dim=-1),
                log_weak_probs,
                reduction='none',
            )
            loss = loss.mean(dim=-1)
            weighted_loss = (loss * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
            
        else:
            # Default cross-entropy
            loss = F.cross_entropy(
                strong_logits.view(-1, strong_logits.size(-1)),
                weak_labels.view(-1),
                reduction='mean',
            )
            weighted_loss = loss
        
        return weighted_loss
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self._stats.copy()


class POPSSWeakToStrongOperator:
    """Facade for weak-to-strong training operator.
    
    Enables training a strong model using weak model supervision,
    achieving performance beyond the weak model's capabilities.
    
    Example:
        >>> config = POPSSWeakToStrongConfig(
        ...     confidence_threshold=0.7,
        ...     use_curriculum=True,
        ... )
        >>> w2s = POPSSWeakToStrongOperator(weak_model, strong_model, config)
        >>> 
        >>> for batch in dataloader:
        ...     result = w2s.train_step(batch)
        ...     loss = result.data["loss"]
        ...     loss.backward()
    """
    
    def __init__(
        self,
        weak_model: nn.Module,
        strong_model: nn.Module,
        config: Optional[POPSSWeakToStrongConfig] = None,
    ):
        self.config = config or POPSSWeakToStrongConfig()
        self._impl = _WeakToStrongOperatorImpl(weak_model, strong_model, self.config)
        self._LOG = PiscesLxLogger(
            "PiscesLx.W2S.Facade",
            file_path=get_log_file("PiscesLx.W2S.Facade"),
            enable_file=True,
        )
    
    def execute(self, params: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Execute training step."""
        return self._impl.execute(params)
    
    def train_step(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> PiscesLxOperatorResult:
        """Convenience method for single training step."""
        params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return self.execute(params)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self._impl.get_statistics()


class POPSSIterativeAmplification:
    """Iterative amplification for continuous improvement.
    
    Implements iterative weak-to-strong training where the
    trained strong model becomes the weak model for the next iteration.
    
    Based on "Supervising strong learners by amplifying weak experts"
    (Christiano et al., 2018).
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: POPSSWeakToStrongConfig,
        num_iterations: int = 3,
    ):
        self.base_model = base_model
        self.config = config
        self.num_iterations = num_iterations
        self._LOG = PiscesLxLogger(
            "PiscesLx.W2S.Iterative",
            file_path=get_log_file("PiscesLx.W2S.Iterative"),
            enable_file=True,
        )
        
        self.current_model = base_model
        self.iteration = 0
    
    def step(self, dataloader: DataLoader) -> nn.Module:
        """Perform one iteration of amplification.
        
        Args:
            dataloader: Training data loader.
            
        Returns:
            Amplified model.
        """
        if self.iteration >= self.num_iterations:
            self._LOG.warning(f"Already completed {self.num_iterations} iterations")
            return self.current_model
        
        # Create new strong model (copy of current)
        strong_model = copy.deepcopy(self.current_model)
        
        # Reset parameters for diversity
        for param in strong_model.parameters():
            param.data += torch.randn_like(param) * 0.01
        
        # Train with weak-to-strong
        w2s = POPSSWeakToStrongOperator(
            self.current_model,  # weak model
            strong_model,        # strong model
            self.config,
        )
        
        for batch in dataloader:
            result = w2s.train_step(
                batch["input_ids"],
                batch.get("attention_mask"),
            )
            if result.status == PiscesLxOperatorStatus.SUCCESS:
                loss = result.data["loss"]
                if loss.requires_grad:
                    loss.backward()
        
        self.current_model = strong_model
        self.iteration += 1
        
        self._LOG.info(f"Completed iteration {self.iteration}/{self.num_iterations}")
        
        return self.current_model
    
    def get_current_model(self) -> nn.Module:
        """Get current amplified model."""
        return self.current_model


import copy


__all__ = [
    "POPSSW2SMode",
    "POPSSWeakToStrongConfig",
    "POPSSWeakLabelGenerator",
    "POPSSCurriculumScheduler",
    "POPSSSelfCorrection",
    "POPSSWeakToStrongOperator",
    "POPSSIterativeAmplification",
]
