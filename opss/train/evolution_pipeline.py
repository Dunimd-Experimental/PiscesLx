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
Evolution Pipeline - Complete Self-Evolution Training System

Implements the full pipeline: Distillation -> Growth -> Self-Evolution -> Weak-to-Strong

Based on multiple NeurIPS/ICML/ICLR 2024-2025 papers:
    - Gstack: Stacking Your Transformers (NeurIPS 2024, arXiv:2405.15319)
    - SPIN: Self-Play Fine-Tuning (ICML 2024, arXiv:2401.01335)
    - Weak-to-Strong Generalization (Burns et al., ICML 2024, arXiv:2312.09390)
    - Architect Thyself: Neural Darwinism and Self-Evolving Multimodal Networks (ICLR 2025)

Pipeline Stages:
    1. Distillation: Teacher model -> 0.5B seed model
    2. Growth: 0.5B -> 1B -> 2B -> 4B -> 7B (progressive expansion)
    3. Self-Evolution: SEAL/SPIN self-improvement
    4. Weak-to-Strong: Evolved model -> PiscesLx flagship

Usage:
    from opss.train.evolution_pipeline import (
        POPSSEvolutionConfig,
        POPSSEvolutionPipeline,
    )
    
    config = POPSSEvolutionConfig(
        seed_size="0.5B",
        target_size="7B",
        growth_stages=[
            {"type": "depth", "num_layers": 4},
            {"type": "width", "hidden_size": 2048},
            {"type": "experts", "num_experts": 16},
        ],
    )
    
    pipeline = POPSSEvolutionPipeline(config)
    pl1_model = pipeline.run(teacher_model, dataloader)
"""

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_checkpoint_dir
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus
from configs.version import VERSION

from .growth import (
    POPSSGrowthType,
    POPSSModelGrowthConfig,
    POPSSModelGrowthOperator,
)
from .weak_to_strong import (
    POPSSWeakToStrongConfig,
    POPSSWeakToStrongOperator,
    POPSSIterativeAmplification,
)
from .distill import (
    POPSSDistillationConfig,
    POPSSDistillationOperator,
)


class POPSSEvolutionStage(Enum):
    """Evolution pipeline stages."""
    DISTILL = "distill"
    GROW = "grow"
    EVOLVE = "evolve"
    W2S = "w2s"
    COMPLETE = "complete"


@dataclass
class POPSSGrowthStage:
    """Single growth stage configuration."""
    type: str
    num_layers: int = 0
    hidden_size: int = 0
    num_experts: int = 0
    train_steps: int = 1000
    
    def to_growth_config(self) -> POPSSModelGrowthConfig:
        """Convert to growth config."""
        return POPSSModelGrowthConfig(
            growth_type=self.type,
            num_new_layers=self.num_layers,
            new_hidden_size=self.hidden_size if self.hidden_size > 0 else None,
            num_new_experts=self.num_experts,
        )


@dataclass
class POPSSEvolutionConfig:
    """Complete evolution pipeline configuration.
    
    Attributes:
        seed_size: Initial seed model size (e.g., "0.5B").
        target_size: Target model size (e.g., "7B").
        growth_stages: List of growth stages.
        distill_steps: Number of distillation training steps.
        evolution_steps: Number of self-evolution steps.
        w2s_steps: Number of weak-to-strong training steps.
        checkpoint_dir: Directory for saving checkpoints.
        log_interval: Logging interval in steps.
        eval_interval: Evaluation interval in steps.
        save_interval: Checkpoint save interval in steps.
        use_fp16: Use mixed precision training.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Base learning rate.
        weight_decay: Weight decay coefficient.
        warmup_steps: Warmup steps for learning rate.
    """
    
    seed_size: str = "0.5B"
    target_size: str = "7B"
    growth_stages: List[Dict] = field(default_factory=lambda: [
        {"type": "depth", "num_layers": 4, "train_steps": 500},
        {"type": "width", "hidden_size": 2048, "train_steps": 500},
        {"type": "experts", "num_experts": 16, "train_steps": 500},
        {"type": "depth", "num_layers": 8, "train_steps": 1000},
    ])
    
    distill_steps: int = 2000
    evolution_steps: int = 1000
    w2s_steps: int = 2000
    
    checkpoint_dir: str = ""
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    use_fp16: bool = True
    gradient_accumulation_steps: int = 4
    
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    def __post_init__(self):
        if not self.checkpoint_dir:
            self.checkpoint_dir = str(get_checkpoint_dir("evolution"))
        
        # Convert dict stages to POPSSGrowthStage
        self._parsed_stages = []
        for stage in self.growth_stages:
            if isinstance(stage, dict):
                self._parsed_stages.append(POPSSGrowthStage(**stage))
            elif isinstance(stage, POPSSGrowthStage):
                self._parsed_stages.append(stage)
    
    def get_growth_stages(self) -> List[POPSSGrowthStage]:
        """Get parsed growth stages."""
        return self._parsed_stages


class POPSSEvolutionTracker:
    """Tracks evolution pipeline progress and metrics."""
    
    def __init__(self):
        self.stage = POPSSEvolutionStage.DISTILL
        self.step = 0
        self.stage_step = 0
        self.metrics = {
            "distill_loss": [],
            "growth_loss": [],
            "evolution_loss": [],
            "w2s_loss": [],
        }
        self.start_time = time.time()
        self._LOG = PiscesLxLogger(
            "PiscesLx.Evolution.Tracker",
            file_path=get_log_file("PiscesLx.Evolution.Tracker"),
            enable_file=True,
        )
    
    def set_stage(self, stage: POPSSEvolutionStage):
        """Set current stage."""
        self.stage = stage
        self.stage_step = 0
        self._LOG.info(f"Entered stage: {stage.value}")
    
    def log_step(self, metric_name: str, value: float):
        """Log a metric for current step."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        self.step += 1
        self.stage_step += 1
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of evolution progress."""
        return {
            "stage": self.stage.value,
            "total_steps": self.step,
            "stage_steps": self.stage_step,
            "elapsed_time": self.get_elapsed_time(),
            "metrics": {k: v[-100:] if v else [] for k, v in self.metrics.items()},
        }


class _EvolutionPipelineImpl(PiscesLxOperatorInterface):
    """Evolution pipeline implementation."""
    
    def __init__(self, config: POPSSEvolutionConfig):
        super().__init__()
        self._name = "training.evolution_pipeline"
        self._version = VERSION
        self.type = "training"
        
        self.config = config
        self.tracker = POPSSEvolutionTracker()
        
        self._LOG = PiscesLxLogger(
            "PiscesLx.Evolution.Pipeline",
            file_path=get_log_file("PiscesLx.Evolution.Pipeline"),
            enable_file=True,
        )
        
        # Operators
        self._growth_operator = POPSSModelGrowthOperator()
        self._w2s_operator = None
        self._distill_operator = None
    
    def execute(self, params: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Execute complete evolution pipeline.
        
        Args:
            params: Dictionary containing:
                - teacher_model: Teacher model for distillation
                - seed_model: Optional pre-built seed model
                - dataloader: Training data loader
                - eval_dataloader: Optional evaluation dataloader
                
        Returns:
            Final evolved model.
        """
        teacher_model = params.get("teacher_model")
        seed_model = params.get("seed_model")
        dataloader = params.get("dataloader")
        eval_dataloader = params.get("eval_dataloader")
        
        if dataloader is None:
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error="dataloader is required",
            )
        
        try:
            current_model = seed_model
            
            # Stage 1: Distillation
            if teacher_model is not None and current_model is None:
                self._LOG.info("Stage 1: Distillation")
                self.tracker.set_stage(POPSSEvolutionStage.DISTILL)
                current_model = self._run_distillation(teacher_model, dataloader)
            
            if current_model is None:
                return PiscesLxOperatorResult(
                    status=PiscesLxOperatorStatus.ERROR,
                    error="No seed model available",
                )
            
            # Stage 2: Growth
            self._LOG.info("Stage 2: Progressive Growth")
            self.tracker.set_stage(POPSSEvolutionStage.GROW)
            current_model = self._run_growth(current_model, dataloader)
            
            # Stage 3: Self-Evolution
            self._LOG.info("Stage 3: Self-Evolution")
            self.tracker.set_stage(POPSSEvolutionStage.EVOLVE)
            current_model = self._run_self_evolution(current_model, dataloader)
            
            # Stage 4: Weak-to-Strong
            self._LOG.info("Stage 4: Weak-to-Strong Training")
            self.tracker.set_stage(POPSSEvolutionStage.W2S)
            final_model = self._run_weak_to_strong(current_model, dataloader)
            
            self.tracker.set_stage(POPSSEvolutionStage.COMPLETE)
            
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.SUCCESS,
                data={
                    "model": final_model,
                    "tracker": self.tracker.get_summary(),
                }
            )
            
        except Exception as e:
            self._LOG.error(f"Evolution pipeline failed: {e}")
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error=str(e),
            )
    
    def _run_distillation(
        self,
        teacher_model: nn.Module,
        dataloader: DataLoader,
    ) -> nn.Module:
        """Run distillation stage."""
        from model.config import YvConfig
        from model.core.model import YvModelForCausalLM
        
        # Create seed model
        seed_config = self._get_seed_config()
        student_model = YvModelForCausalLM(seed_config)
        
        # Setup distillation
        distill_config = POPSSDistillationConfig(
            temperature=2.0,
            alpha=0.5,
        )
        
        self._distill_operator = POPSSDistillationOperator(
            teacher_model, student_model, distill_config
        )
        
        # Training loop with mixed precision + gradient accumulation
        optimizer = AdamW(
            student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        use_amp = self.config.use_fp16 and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler('cuda') if use_amp else None
        grad_accum = max(1, self.config.gradient_accumulation_steps)
        max_grad_norm = 1.0
        
        step = 0
        for batch in dataloader:
            if step >= self.config.distill_steps:
                break
            
            with torch.cuda.amp.autocast('cuda', enabled=use_amp):
                result = self._distill_operator.train_step(
                    batch["input_ids"],
                    batch.get("attention_mask"),
                    batch.get("labels"),
                )
            
            if result.status == PiscesLxOperatorStatus.SUCCESS:
                loss = result.data.get("loss", 0)
                if isinstance(loss, Tensor) and loss.requires_grad:
                    loss = loss / grad_accum
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
            
            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                
                if result.status == PiscesLxOperatorStatus.SUCCESS:
                    self.tracker.log_step("distill_loss", loss.item() * grad_accum if isinstance(loss, Tensor) else 0)
            
            step += 1
            
            if step % self.config.log_interval == 0:
                self._LOG.info(f"Distill step {step}/{self.config.distill_steps}")
        
        self._LOG.info(f"Distillation complete: {step} steps")
        return student_model
    
    def _run_growth(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> nn.Module:
        """Run progressive growth stage."""
        growth_stages = self.config.get_growth_stages()
        
        for i, stage in enumerate(growth_stages):
            self._LOG.info(f"Growth stage {i+1}/{len(growth_stages)}: {stage.type}")
            
            # Grow model
            growth_config = stage.to_growth_config()
            result = self._growth_operator.execute({
                "model": model,
                "config": growth_config,
            })
            
            if result.status == PiscesLxOperatorStatus.SUCCESS:
                model = result.data["model"]
            
            # Train after growth
            model = self._train_after_growth(model, dataloader, stage.train_steps)
            
            # Save checkpoint
            self._save_checkpoint(model, f"growth_stage_{i+1}")
        
        return model
    
    def _train_after_growth(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        steps: int,
    ) -> nn.Module:
        """Train model after growth with mixed precision + gradient accumulation."""
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        use_amp = self.config.use_fp16 and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler('cuda') if use_amp else None
        grad_accum = max(1, self.config.gradient_accumulation_steps)
        max_grad_norm = 1.0
        
        model.train()
        step = 0
        
        for batch in dataloader:
            if step >= steps:
                break
            
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels", input_ids)
            
            with torch.cuda.amp.autocast('cuda', enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            if isinstance(loss, Tensor) and loss.requires_grad:
                loss = loss / grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                
                if isinstance(loss, Tensor):
                    self.tracker.log_step("growth_loss", loss.item() * grad_accum)
            
            step += 1
        
        return model
    
    def _run_self_evolution(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> nn.Module:
        """Run self-evolution stage using SEAL/SPIN."""
        from model.reasoning.self_evolution import YvSEAL
        
        # Initialize SEAL for self-evolution
        seal = YvSEAL(
            model,
            confidence_threshold=0.85,
            max_synthetic_samples=100,
        )
        
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        use_amp = self.config.use_fp16 and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler('cuda') if use_amp else None
        grad_accum = max(1, self.config.gradient_accumulation_steps)
        max_grad_norm = 1.0
        
        model.train()
        step = 0
        
        for batch in dataloader:
            if step >= self.config.evolution_steps:
                break
            
            input_ids = batch["input_ids"]
            
            with torch.cuda.amp.autocast('cuda', enabled=use_amp):
                outputs = seal(input_ids)
            
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
                if isinstance(loss, Tensor) and loss.requires_grad:
                    loss = loss / grad_accum
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
            
            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                
                if hasattr(outputs, 'loss') and outputs.loss is not None and isinstance(loss, Tensor):
                    self.tracker.log_step("evolution_loss", loss.item() * grad_accum)
            
            step += 1
            
            if step % self.config.log_interval == 0:
                self._LOG.info(f"Evolution step {step}/{self.config.evolution_steps}")
        
        self._LOG.info(f"Self-evolution complete: {step} steps")
        return model
    
    def _run_weak_to_strong(
        self,
        evolved_model: nn.Module,
        dataloader: DataLoader,
    ) -> nn.Module:
        """Run weak-to-strong training stage."""
        from model.config import YvConfig
        from model.core.model import YvModelForCausalLM
        
        # Create target PL1 model
        target_config = self._get_target_config()
        pl1_model = YvModelForCausalLM(target_config)
        
        # Initialize from evolved model where possible
        pl1_model = self._initialize_from_evolved(pl1_model, evolved_model)
        
        # Setup weak-to-strong training
        w2s_config = POPSSWeakToStrongConfig(
            confidence_threshold=0.7,
            use_curriculum=True,
            use_self_correction=True,
        )
        
        w2s_operator = POPSSWeakToStrongOperator(
            evolved_model,  # weak model
            pl1_model,      # strong model
            w2s_config,
        )
        
        optimizer = AdamW(
            pl1_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        use_amp = self.config.use_fp16 and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler('cuda') if use_amp else None
        grad_accum = max(1, self.config.gradient_accumulation_steps)
        max_grad_norm = 1.0
        
        step = 0
        for batch in dataloader:
            if step >= self.config.w2s_steps:
                break
            
            with torch.cuda.amp.autocast('cuda', enabled=use_amp):
                result = w2s_operator.train_step(
                    batch["input_ids"],
                    batch.get("attention_mask"),
                )
            
            if result.status == PiscesLxOperatorStatus.SUCCESS:
                loss = result.data.get("loss", 0)
                if isinstance(loss, Tensor) and loss.requires_grad:
                    loss = loss / grad_accum
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
            
            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(pl1_model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(pl1_model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                
                if result.status == PiscesLxOperatorStatus.SUCCESS:
                    self.tracker.log_step("w2s_loss", loss.item() * grad_accum if isinstance(loss, Tensor) else 0)
            
            step += 1
            
            if step % self.config.log_interval == 0:
                self._LOG.info(f"W2S step {step}/{self.config.w2s_steps}")
        
        self._LOG.info(f"Weak-to-strong complete: {step} steps")
        return pl1_model
    
    def _get_seed_config(self) -> "YvConfig":
        """Get seed model configuration."""
        from model.config import YvConfig
        
        size_map = {
            "0.5B": {"hidden_size": 512, "n_layer": 8, "n_head": 8},
            "1B": {"hidden_size": 768, "n_layer": 12, "n_head": 12},
        }
        
        size_params = size_map.get(self.config.seed_size, size_map["0.5B"])
        
        return YvConfig(
            hidden_size=size_params["hidden_size"],
            n_layer=size_params["n_layer"],
            n_head=size_params["n_head"],
            vocab_size=64000,
        )
    
    def _get_target_config(self) -> "YvConfig":
        """Get target model configuration."""
        from model.config import YvConfig
        
        size_map = {
            "7B": {"hidden_size": 4096, "n_layer": 32, "n_head": 32},
            "13B": {"hidden_size": 5120, "n_layer": 40, "n_head": 40},
        }
        
        size_params = size_map.get(self.config.target_size, size_map["7B"])
        
        return YvConfig(
            hidden_size=size_params["hidden_size"],
            n_layer=size_params["n_layer"],
            n_head=size_params["n_head"],
            vocab_size=64000,
            moe_num_experts=64,
        )
    
    def _initialize_from_evolved(
        self,
        target_model: nn.Module,
        evolved_model: nn.Module,
    ) -> nn.Module:
        """Initialize target model from evolved model."""
        target_state = target_model.state_dict()
        evolved_state = evolved_model.state_dict()
        
        # Copy matching parameters
        for name, param in evolved_state.items():
            if name in target_state:
                if target_state[name].shape == param.shape:
                    target_state[name] = param
                elif target_state[name].shape[0] == param.shape[0]:
                    # Partial copy for expanded dimensions
                    target_state[name][:param.shape[0]] = param
        
        target_model.load_state_dict(target_state)
        return target_model
    
    def _save_checkpoint(self, model: nn.Module, name: str):
        """Save checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{name}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model.config.__dict__ if hasattr(model, 'config') else {},
            "stage": self.tracker.stage.value,
            "step": self.tracker.step,
        }, checkpoint_path)
        
        self._LOG.info(f"Saved checkpoint: {checkpoint_path}")


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class POPSSEvolutionPipeline:
    """Facade for evolution pipeline.
    
    Complete self-evolution training pipeline implementing:
        1. Knowledge distillation from teacher to seed model
        2. Progressive model growth (depth/width/experts)
        3. Self-evolution using SEAL/SPIN
        4. Weak-to-strong training for final PiscesLx
    
    Example:
        >>> config = POPSSEvolutionConfig(
        ...     seed_size="0.5B",
        ...     target_size="7B",
        ...     distill_steps=2000,
        ...     evolution_steps=1000,
        ...     w2s_steps=2000,
        ... )
        >>> 
        >>> pipeline = POPSSEvolutionPipeline(config)
        >>> result = pipeline.run(
        ...     teacher_model=deepseek_model,
        ...     dataloader=train_dataloader,
        ... )
        >>> 
        >>> if result.status == PiscesLxOperatorStatus.SUCCESS:
        ...     pl1_model = result.data["model"]
    """
    
    def __init__(self, config: Optional[POPSSEvolutionConfig] = None):
        self.config = config or POPSSEvolutionConfig()
        self._impl = _EvolutionPipelineImpl(self.config)
        self._LOG = PiscesLxLogger(
            "PiscesLx.Evolution.Facade",
            file_path=get_log_file("PiscesLx.Evolution.Facade"),
            enable_file=True,
        )
    
    def run(
        self,
        teacher_model: nn.Module,
        dataloader: DataLoader,
        seed_model: Optional[nn.Module] = None,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> PiscesLxOperatorResult:
        """Run complete evolution pipeline.
        
        Args:
            teacher_model: Teacher model for distillation.
            dataloader: Training data loader.
            seed_model: Optional pre-built seed model.
            eval_dataloader: Optional evaluation dataloader.
            
        Returns:
            Result containing final evolved model.
        """
        params = {
            "teacher_model": teacher_model,
            "dataloader": dataloader,
            "seed_model": seed_model,
            "eval_dataloader": eval_dataloader,
        }
        
        return self._impl.execute(params)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current pipeline progress."""
        return self._impl.tracker.get_summary()
    
    def run_distillation_only(
        self,
        teacher_model: nn.Module,
        dataloader: DataLoader,
    ) -> nn.Module:
        """Run only distillation stage."""
        return self._impl._run_distillation(teacher_model, dataloader)
    
    def run_growth_only(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> nn.Module:
        """Run only growth stage."""
        return self._impl._run_growth(model, dataloader)
    
    def run_evolution_only(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> nn.Module:
        """Run only self-evolution stage."""
        return self._impl._run_self_evolution(model, dataloader)
    
    def run_w2s_only(
        self,
        evolved_model: nn.Module,
        dataloader: DataLoader,
    ) -> nn.Module:
        """Run only weak-to-strong stage."""
        return self._impl._run_weak_to_strong(evolved_model, dataloader)


__all__ = [
    "POPSSEvolutionStage",
    "POPSSGrowthStage",
    "POPSSEvolutionConfig",
    "POPSSEvolutionTracker",
    "POPSSEvolutionPipeline",
]
