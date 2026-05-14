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
PiscesLx Training Engine

This module implements the flagship training engine for the PiscesLx framework,
integrating state-of-the-art training techniques and optimization algorithms.
The engine orchestrates operators from ops/train/ to build complete training pipeline.

Architecture:
    The PiscesLxTrainingEngine serves as the central training component,
    orchestrating model training with advanced features including:

    1. Mixed Precision Training:
       - Automatic FP16/BF16 mixed precision with gradient scaling
       - Loss scaling to prevent gradient underflow in FP16
       - Automatic cast between precision formats

    2. Gradient Checkpointing:
       - Memory-efficient training by recomputing activations
       - Trade computation for memory (30-50% memory reduction)
       - Compatible with all model architectures

    3. Distributed Training:
       - Native DistributedDataParallel (DDP) support
       - Gradient synchronization across processes
       - Automatic world size detection

    4. Optimizer Integration:
       - AdamW with weight decay decoupling
       - GaLore memory-efficient gradient projection
       - Custom optimizer support through configuration

    5. Learning Rate Scheduling:
       - Cosine annealing with warmup
       - Linear decay schedules
       - Custom scheduler support

Key Features:
    - State Management: Comprehensive training state tracking (step, epoch, loss history)
    - Checkpointing: Automatic and manual checkpoint save/load with metadata
    - Validation: Built-in validation loop with metric computation
    - Logging: Structured logging with step-level granularity
    - Resume Training: Full support for resuming from checkpoints
    - Device Management: Automatic GPU/CPU selection and tensor placement

Training Loop:
    The training loop follows this structure:
    1. Forward pass with mixed precision (if enabled)
    2. Loss computation and scaling
    3. Backward pass with gradient accumulation
    4. Optimizer step with gradient clipping
    5. Learning rate scheduling
    6. Logging and checkpointing

Usage Examples:
    Basic Training:
        >>> from tools.train import TrainingConfig, PiscesLxTrainingEngine

        >>> config = TrainingConfig(model_name="gpt-7b", max_steps=100000)
        >>> engine = PiscesLxTrainingEngine(config)

        >>> # Initialize with model and data
        >>> engine.initialize(model, train_dataloader, val_dataloader)

        >>> # Run training
        >>> engine.train()

    Resume from Checkpoint:
        >>> engine = PiscesLxTrainingEngine(config)
        >>> engine.initialize(model, train_dataloader, val_dataloader)
        >>> engine.load_checkpoint("checkpoint-5000.pt")
        >>> engine.train()

    Custom Training Loop:
        >>> engine.initialize(model, train_dataloader, val_dataloader)

        >>> for epoch in range(num_epochs):
        >>>     for batch in train_dataloader:
        >>>         loss = engine.training_step(batch)

        >>>         if engine.global_step % 100 == 0:
        >>>             metrics = engine.validation_step(val_dataloader)
        >>>             engine.save_checkpoint()

Dependencies:
    - torch >= 2.0.0 (core training framework)
    - numpy >= 1.24.0 (numerical operations)

Performance Considerations:
    - Use gradient_checkpointing for large models (>7B parameters)
    - Enable mixed_precision="bf16" on Ampere GPUs (A100, H100)
    - Adjust gradient_accumulation_steps for memory constraints
    - Use pin_memory=True in DataLoader for GPU training
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import time
import json
from datetime import datetime

from utils.dc import PiscesLxLogger, PiscesLxSystemMonitor

from .config import TrainingConfig

from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Train", file_path=get_log_file("PiscesLx.Tools.Train"), enable_file=True)


def setup_training_device(local_rank: int = -1, device_pref: str = "auto") -> torch.device:
    """
    Setup training device using unified System Monitor.
    
    Args:
        local_rank: Local rank for distributed training
        device_pref: Device preference ("auto", "cuda", "cpu")
    
    Returns:
        torch.device: Selected device
    """
    import torch
    
    try:
        monitor = PiscesLxSystemMonitor()
        if device_pref == "auto":
            if torch.cuda.is_available():
                memory_info = monitor.get_memory_info()
                if memory_info.usage_percent > 90:
                    device = torch.device("cpu")
                    _LOG.info("Training mode: cpu (high memory usage)")
                else:
                    if local_rank >= 0:
                        device = torch.device(f"cuda:{local_rank}")
                        torch.cuda.set_device(device)
                    else:
                        device = torch.device("cuda:0")
                _LOG.info(f"Training device: {device}")
                return device
    except Exception as e:
        _LOG.warning(f"System Monitor failed, falling back: {e}")
    
    if torch.cuda.is_available():
        if local_rank >= 0:
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    return device


class PiscesLxTrainingOperator(object):
    """
    PiscesLx Flagship Training Engine

    Core training engine integrating state-of-the-art training techniques
    including mixed precision, gradient checkpointing, and distributed training.

    This engine serves as the primary interface for model training within
    the PiscesL1 framework, providing a complete training loop with advanced
    optimization features.

    Attributes:
        config: TrainingConfig instance with all training parameters
        model: The neural network model being trained (initialized in setup)
        optimizer: PyTorch optimizer instance (AdamW, SGD, etc.)
        scheduler: Learning rate scheduler instance
        scaler: Gradient scaler for mixed precision training
        global_step: Current global training step counter
        best_metric: Best validation metric achieved during training
        device: Target compute device (cuda/cpu)
        is_distributed: Whether distributed training is enabled

    Training State:
        The engine maintains comprehensive training state including:
        - global_step: Current training iteration
        - best_metric: Best validation performance
        - loss_history: Historical training losses
        - current_epoch: Current training epoch

    Example:
        >>> config = TrainingConfig(
        ...     model_name="gpt-7b",
        ...     max_steps=100000,
        ...     mixed_precision="bf16"
        ... )
        >>> engine = PiscesLxTrainingEngine(config)
        >>> engine.initialize(model, train_loader, val_loader)
        >>> engine.train()
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the training engine.

        Args:
            config: TrainingConfig with all training parameters
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.global_step = 0
        self.epochs_completed = 0
        self.best_metric = float('inf')
        self.early_stop_counter = 0
        self.training_stats = {
            'loss_history': [],
            'grad_norm_history': [],
            'lr_history': [],
            'throughput_history': [],
            'val_loss_history': [],
            'val_metric_history': []
        }
        
        self.stage = getattr(config, 'stage', None)
        self.loss_type = getattr(config, 'loss_type', 'lm')
        self.response_only_loss = getattr(config, 'response_only_loss', False)
        self._reference_model = None
        self._teacher_provider = None

        self.device = setup_training_device(
            local_rank=getattr(config, 'local_rank', -1),
            device_pref=config.device
        )
        
        # Initialize CUDA context if using GPU
        if self.device.type == 'cuda':
            try:
                # Force CUDA initialization by creating a small tensor
                torch.cuda.synchronize(self.device)
                _ = torch.zeros(1, device=self.device)
                torch.cuda.synchronize(self.device)
                _LOG.info(f"CUDA context initialized on {self.device}")
            except Exception as e:
                _LOG.warning(f"CUDA initialization check failed: {e}")
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                _LOG.debug("CUDA TF32 backend enabled: matmul=True, cudnn=True")
            except Exception as e:
                _LOG.warning(f"Failed to enable CUDA TF32 backend: {e}. Training will continue without TF32 optimization.")
            try:
                torch.set_float32_matmul_precision("high")
                _LOG.debug("Float32 matmul precision set to 'high'")
            except Exception as e:
                _LOG.warning(f"Failed to set float32 matmul precision: {e}. Training will continue with default precision.")
            try:
                flash_ok = bool(getattr(torch.backends.cuda, "is_flash_sdp_available", lambda: False)())
                mem_ok = bool(getattr(torch.backends.cuda, "is_mem_efficient_sdp_available", lambda: False)())
                torch.backends.cuda.enable_flash_sdp(flash_ok)
                torch.backends.cuda.enable_mem_efficient_sdp(mem_ok)
                torch.backends.cuda.enable_math_sdp(True)
                _LOG.debug(f"SDP backends configured: flash={flash_ok}, mem_efficient={mem_ok}, math=True")
            except Exception as e:
                _LOG.warning(f"Failed to configure SDP backends: {e}. Training will continue with default SDP settings.")
        
        self._setup_mixed_precision()

        self._modality_scheduler = None
        self._moe_gradient_optimizer = None
        self._kfac_operator = None
        self._multitask_operator = None
        self._parallel_3d_operator = None
        
        self._weight_watermark_operator = None
        self._compliance_operator = None
        self._audit_operator = None
        self._watermark_config = None
        
        self._evolution_operator = None
        self._growth_operator = None
        self._w2s_operator = None

        self._grad_accum_step = 0

        self._setup_advanced_operators()
        self._setup_parallel_3d_operator()
        self._setup_watermark_operator()
        self._setup_evolution_operator()
        
        if self.stage:
            _LOG.info(f"PiscesLxTrainingEngine initialized on {self.device} with stage={self.stage.value}")
        else:
            _LOG.info(f"PiscesLxTrainingEngine initialized on {self.device}")
    
    def _setup_mixed_precision(self):
        """
        Configure mixed precision training.
        
        Sets up gradient scaling for FP16 training or enables BF16 automatic
        mixed precision. FP32 training requires no special setup.
        
        Mixed Precision Modes:
            - fp32: Full precision, no scaling needed
            - fp16: Half precision with gradient scaling to prevent underflow
            - bf16: BFloat16 with automatic mixed precision (Ampere+)
        
        Gradient Scaling (FP16):
            Loss values are scaled up before backward pass to prevent gradient
            underflow in FP16. Gradients are unscaled before optimizer step.
        """
        effective_mixed_precision = self.config.mixed_precision

        if self.config.mixed_precision == "bf16" and self.device.type == "cuda":
            bf16_supported = False
            try:
                bf16_supported = bool(torch.cuda.is_bf16_supported())
                _LOG.debug(f"BF16 support check result: {bf16_supported}")
            except Exception as e:
                bf16_supported = False
                _LOG.warning(f"Failed to check BF16 support: {e}. Assuming BF16 is not supported.")

            if not bf16_supported:
                _LOG.warning(
                    "mixed_precision='bf16' requested but bf16 is not supported on this CUDA device; "
                    "falling back to fp16."
                )
                effective_mixed_precision = "fp16"

        if effective_mixed_precision == "fp16" and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        _LOG.info(f"Mixed precision set to {self.config.mixed_precision} (effective={effective_mixed_precision})")
    
    def _setup_advanced_operators(self):
        """Setup advanced training operators for modality-aware scheduling, MoE gradients, K-FAC, and multi-task learning."""
        from opss.train.modality_scheduler import (
            POPSSModalitySchedulerConfig,
            POPSSModalitySchedulerOperator
        )
        from opss.train.moe_gradient import (
            POPSSMoEGradientConfig,
            POPSSMoEGradientOperator
        )
        from opss.train.kfac import (
            POPSSKFacConfig,
            POPSSKFacOperator
        )
        from opss.train.multitask_uncertainty import (
            POPSSMultiTaskConfig,
            POPSSMultiTaskOperator
        )
        try:
            if hasattr(self.config, 'modality_scheduler') and self.config.modality_scheduler.get('enabled', False):
                modality_config = POPSSModalitySchedulerConfig(**self.config.modality_scheduler)
                self._modality_scheduler = POPSSModalitySchedulerOperator(modality_config)
                _LOG.info("Modality-aware scheduler operator initialized")
        except Exception as e:
            _LOG.warning(f"Failed to initialize modality scheduler: {e}")
        
        try:
            if hasattr(self.config, 'moe_gradient') and self.config.moe_gradient.get('enabled', False):
                # Filter out 'enabled' field before creating config
                moe_config_dict = {k: v for k, v in self.config.moe_gradient.items() if k != 'enabled'}
                moe_config = POPSSMoEGradientConfig(**moe_config_dict)
                self._moe_gradient_optimizer = POPSSMoEGradientOperator(moe_config)
                _LOG.info("MoE gradient optimizer operator initialized")
        except Exception as e:
            _LOG.warning(f"Failed to initialize MoE gradient optimizer: {e}")
        
        try:
            if hasattr(self.config, 'kfac') and self.config.kfac.get('enabled', False):
                kfac_config_dict = {k: v for k, v in self.config.kfac.items() if k != 'enabled'}
                kfac_config = POPSSKFacConfig(**kfac_config_dict)
                self._kfac_operator = POPSSKFacOperator(kfac_config)
                _LOG.info("K-FAC operator initialized")
        except Exception as e:
            _LOG.warning(f"Failed to initialize K-FAC operator: {e}")
        
        try:
            if hasattr(self.config, 'multitask') and self.config.multitask.get('enabled', False):
                multitask_config_dict = {k: v for k, v in self.config.multitask.items() if k != 'enabled'}
                multitask_config = POPSSMultiTaskConfig(**multitask_config_dict)
                self._multitask_operator = POPSSMultiTaskOperator(multitask_config)
                _LOG.info("Multi-task uncertainty operator initialized")
        except Exception as e:
            _LOG.warning(f"Failed to initialize multi-task operator: {e}")
    
    def _setup_parallel_3d_operator(self):
        """Setup 3D parallelism operator for large-scale distributed training."""
        from opss.train.parallel_3d import (
            POPSSParallel3DConfig,
            POPSSParallel3DOperator
        )
        try:
            if hasattr(self.config, 'parallel_3d') and self.config.parallel_3d.get('enabled', False):
                parallel_config = POPSSParallel3DConfig(
                    dp_size=self.config.parallel_3d.get('dp_size', 1),
                    tp_size=self.config.parallel_3d.get('tp_size', 1),
                    pp_size=self.config.parallel_3d.get('pp_size', 1),
                    sequence_parallel=self.config.parallel_3d.get('sequence_parallel', True),
                    num_micro_batches=self.config.parallel_3d.get('num_micro_batches', 4),
                    overlap_communication=self.config.parallel_3d.get('overlap_communication', True),
                    gradient_checkpointing=self.config.parallel_3d.get('gradient_checkpointing', False),
                    zero_stage=self.config.parallel_3d.get('zero_stage', 0),
                    mixed_precision=self.config.mixed_precision
                )
                
                self._parallel_3d_operator = POPSSParallel3DOperator(parallel_config)
                _LOG.info(f"3D Parallelism operator initialized: dp={parallel_config.dp_size}, tp={parallel_config.tp_size}, pp={parallel_config.pp_size}")
        except Exception as e:
            _LOG.warning(f"Failed to initialize 3D parallelism operator: {e}")
    
    def _setup_watermark_operator(self):
        """Setup weight watermark operator for model provenance and ownership verification."""
        from opss.watermark import (
            POPSSWatermarkConfig,
            POPSSWeightWatermarkOperator,
            POPSSComplianceOperator,
            POPSSAuditOperator,
        )
        try:
            if hasattr(self.config, 'watermark') and self.config.watermark.get('enabled', False):
                self._watermark_config = POPSSWatermarkConfig(
                    standard=self.config.watermark.get('standard', 'GB/T 45225-2024'),
                    jurisdiction=self.config.watermark.get('jurisdiction', 'CN'),
                    risk_level=self.config.watermark.get('risk_level', 'medium'),
                    watermark_strength=self.config.watermark.get('strength', 1e-5),
                    redundancy_level=self.config.watermark.get('redundancy_level', 3),
                    encryption_enabled=self.config.watermark.get('encryption_enabled', True),
                    verify_threshold=self.config.watermark.get('verify_threshold', 0.02),
                    audit_enabled=self.config.watermark.get('audit_enabled', True),
                    owner_id=self.config.watermark.get('owner_id', 'default_owner'),
                    model_id=self.config.watermark.get('model_id', self.config.model_name)
                )
                
                self._weight_watermark_operator = POPSSWeightWatermarkOperator(self._watermark_config)
                self._compliance_operator = POPSSComplianceOperator(self._watermark_config)
                self._audit_operator = POPSSAuditOperator(self._watermark_config)
                
                _LOG.info(f"Weight watermark operator initialized: owner_id={self._watermark_config.owner_id}")
        except Exception as e:
            _LOG.warning(f"Failed to initialize watermark operators: {e}")
            self._weight_watermark_operator = None
            self._compliance_operator = None
            self._audit_operator = None
    
    def _setup_evolution_operator(self):
        """Setup evolution operators for self-evolution training pipeline.
        
        Initializes:
            - POPSSModelGrowthOperator: For progressive model expansion
            - POPSSWeakToStrongOperator: For weak-to-strong training
            - POPSSEvolutionPipeline: Complete evolution pipeline
        """
        try:
            evolution_cfg = getattr(self.config, 'evolution', None)
            if evolution_cfg is None or not getattr(evolution_cfg, 'enabled', False):
                return
            
            from opss.train.growth import POPSSModelGrowthOperator
            from opss.train.weak_to_strong import POPSSWeakToStrongOperator
            from opss.train.evolution_pipeline import POPSSEvolutionPipeline, POPSSEvolutionConfig
            
            self._growth_operator = POPSSModelGrowthOperator()
            _LOG.info("Model growth operator initialized")
            
            self._evolution_config = POPSSEvolutionConfig(
                seed_size=getattr(evolution_cfg, 'seed_size', '0.5B'),
                target_size=getattr(evolution_cfg, 'target_size', '7B'),
                distill_steps=getattr(evolution_cfg, 'distill_steps', 2000),
                evolution_steps=getattr(evolution_cfg, 'evolution_steps', 1000),
                w2s_steps=getattr(evolution_cfg, 'w2s_steps', 2000),
                growth_stages=getattr(evolution_cfg, 'growth_stages', []),
            )
            
            self._evolution_operator = POPSSEvolutionPipeline(self._evolution_config)
            _LOG.info(
                f"Evolution pipeline initialized: seed={self._evolution_config.seed_size}, "
                f"target={self._evolution_config.target_size}"
            )
            
        except Exception as e:
            _LOG.warning(f"Failed to initialize evolution operators: {e}")
            self._evolution_operator = None
            self._growth_operator = None
            self._w2s_operator = None
    
    def run_evolution(self, teacher_model, dataloader, seed_model=None):
        """Run complete evolution pipeline.
        
        Args:
            teacher_model: Teacher model for distillation.
            dataloader: Training data loader.
            seed_model: Optional pre-built seed model.
            
        Returns:
            Evolution result with final model.
        """
        if self._evolution_operator is None:
            raise RuntimeError("Evolution operator not initialized. Enable evolution in config.")
        
        return self._evolution_operator.run(
            teacher_model=teacher_model,
            dataloader=dataloader,
            seed_model=seed_model,
        )
    
    def grow_model(self, growth_type: str = "depth", **kwargs):
        """Grow model using specified growth strategy.
        
        Args:
            growth_type: Type of growth ('depth', 'width', 'expert').
            **kwargs: Growth parameters (num_layers, hidden_size, num_experts).
            
        Returns:
            Grown model.
        """
        if self._growth_operator is None:
            raise RuntimeError("Growth operator not initialized.")
        
        if growth_type == "depth":
            return self._growth_operator.grow_depth(
                self.model, 
                kwargs.get("num_layers", 4)
            )
        elif growth_type == "width":
            return self._growth_operator.grow_width(
                self.model,
                kwargs.get("hidden_size", 2048)
            )
        elif growth_type == "expert":
            return self._growth_operator.grow_experts(
                self.model,
                kwargs.get("num_experts", 8)
            )
        else:
            raise ValueError(f"Unknown growth type: {growth_type}")
    
    def initialize_model(self, model_class: type, **model_kwargs) -> nn.Module:
        """
        Initialize the training model.
        
        Creates a model instance and applies necessary training configurations
        including gradient checkpointing and device placement.
        
        Args:
            model_class: Model class to instantiate (e.g., transformers.AutoModel)
            **model_kwargs: Keyword arguments passed to model constructor
            
        Returns:
            Initialized model instance on target device
            
        Example:
            >>> from transformers import AutoModelForCausalLM
            >>> operator.initialize_model(
            ...     AutoModelForCausalLM,
            ...     pretrained_model_name_or_path="gpt2"
            ... )
        """
        _LOG.info("Initializing training model...")

        lora_enabled = getattr(getattr(self.config, "lora", None), "enabled", False)
        quant_enabled = self.config.quantization.enable_quantization

        # Try cache load: state_dict based (avoids pickle issues with bitsandbytes locks)
        cache_path = None
        if quant_enabled and lora_enabled:
            cache_path = self._get_qlora_cache_path()
            if cache_path.exists():
                _LOG.info(f"Loading cached QLoRA state_dict from {cache_path}")
                try:
                    import gc; gc.collect()
                    cached_state = torch.load(cache_path, map_location="cpu", weights_only=True)
                    # Create model skeleton, apply quantization + LoRA for structure,
                    # then load cached weights (quantized + adapters) instantly.
                    force_cpu_init = self._resolve_init_device(quant_enabled, model_kwargs)
                    self.model = model_class(**model_kwargs)
                    if quant_enabled:
                        self._apply_quantization_gpu_accelerated() if self.device.type == "cuda" else self._apply_quantization()
                    self._apply_lora()
                    missing, unexpected = self.model.load_state_dict(cached_state, strict=False)
                    if missing:
                        _LOG.debug(f"Cache state_dict missing keys: {missing}")
                    if unexpected:
                        _LOG.debug(f"Cache state_dict unexpected keys: {unexpected}")
                    del cached_state; gc.collect()
                    train_device = self._resolve_training_device()
                    if train_device.type != self.device.type:
                        _LOG.info(f"Overriding training device: {self.device} → {train_device}")
                        self.device = train_device
                    _LOG.info("Cached state_dict loaded, transferring to device...")
                    self.model = self.model.to(self.device)
                    self._post_transfer_setup(lora_enabled)
                    _LOG.info(f"Model initialized from cache: {self.model.__class__.__name__}")
                    return self.model
                except Exception as e:
                    _LOG.warning(f"Cache load failed ({e}), falling back to full init...")
                    cache_path.unlink(missing_ok=True)

        force_cpu_init = self._resolve_init_device(quant_enabled, model_kwargs)

        # Create model instance with provided arguments
        self.model = model_class(**model_kwargs)

        # Apply quantization (QLoRA-style) and LoRA BEFORE moving the full model to CUDA.
        # This avoids the peak VRAM spike caused by first transferring bf16/fp16 full-precision weights.
        if quant_enabled:
            if self.device.type == "cuda":
                self._apply_quantization_gpu_accelerated()
            else:
                self._apply_quantization()

        if lora_enabled:
            self._apply_lora()

        # Save state_dict to cache (tensors only, no pickle issues with bnb locks)
        if cache_path is not None and not cache_path.exists():
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), cache_path)
                _LOG.info(f"Cached prepared state_dict to {cache_path}")
            except Exception as e:
                _LOG.debug(f"Model caching skipped: {e}")

        # Determine actual training device: prefer GPU even if driver warning fired.
        # setup_training_device may return CPU when CUDA driver version mismatches
        # PyTorch's build, but CUDA often still works. Probe it directly.
        train_device = self._resolve_training_device()
        if train_device.type != self.device.type:
            _LOG.info(f"Overriding training device: {self.device} → {train_device}")
            self.device = train_device

        # Move model to target device (GPU/CPU) after QLoRA+LoRA.
        _LOG.info(f"Transferring model to {self.device} (this copies quantized weights, may take time)...")
        self.model = self.model.to(self.device)
        self._post_transfer_setup(lora_enabled)

        _LOG.info(f"Model initialized: {self.model.__class__.__name__}")
        return self.model

    def _resolve_init_device(self, quant_enabled: bool, model_kwargs: dict) -> bool:
        """Determine whether to force CPU init and inject device/dtype into kwargs.
        Returns True when CPU init was forced."""
        if not quant_enabled:
            return False
        if self.device.type == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 20:
                _LOG.info(f"GPU VRAM {vram_gb:.1f}GB - using GPU initialization for quantization")
                return False
            else:
                _LOG.info(f"GPU VRAM {vram_gb:.1f}GB - using CPU initialization for quantization (low VRAM)")
        else:
            _LOG.info("CPU device - using CPU initialization for quantization")
        model_kwargs['device'] = 'cpu'
        model_kwargs['dtype'] = torch.bfloat16
        _LOG.info("Low-VRAM mode: CPU+BF16 initialization for quantization")
        return True

    def _resolve_training_device(self) -> torch.device:
        """Resolve best available GPU for training, even if setup_training_device chose CPU.

        Probes CUDA directly with a test allocation, since driver version warnings
        from PyTorch do not always mean CUDA is truly unavailable.
        """
        if self.device.type == "cuda":
            return self.device
        try:
            if torch.cuda.device_count() == 0:
                return self.device
            # Test allocation — some driver mismatches still allow CUDA to work
            _test = torch.zeros(1, device="cuda:0")
            del _test
            _LOG.info("GPU probe succeeded (despite any driver warnings)")
            return torch.device("cuda:0")
        except Exception as e:
            _LOG.warning(f"GPU probe failed ({e}), training on {self.device}")
            return self.device

    def _post_transfer_setup(self, lora_was_applied: bool) -> None:
        """Shared post-GPU-transfer setup: dtype cast, param count, checkpointing, distributed."""
        try:
            if self.device.type == "cuda":
                mp = str(getattr(self.config, "mixed_precision", "fp32") or "fp32").lower()
                if mp == "bf16":
                    self.model = self.model.to(dtype=torch.bfloat16)
                elif mp == "fp16":
                    self.model = self.model.to(dtype=torch.float16)
        except Exception as e:
            _LOG.warning(f"Failed to cast model to mixed precision dtype: {e}")

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            _LOG.info(f"Model moved to {self.device} and CUDA synchronized")

        # Single parameter count on GPU (fast)
        trainable = 0
        all_params = 0
        for p in self.model.parameters():
            n = p.numel()
            all_params += n
            if p.requires_grad:
                trainable += n
        if lora_was_applied:
            _LOG.info(
                f"LoRA enabled: {trainable:,} trainable parameters, "
                f"{all_params:,} total, "
                f"trainable%: {100 * trainable / all_params:.4f}"
            )
        _LOG.info(f"Model parameters: {all_params:,}")

        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        if self.config.distributed:
            self._setup_distributed_training()

    def _get_qlora_cache_path(self) -> Path:
        """Build a cache path keyed by model + quant + lora config hash."""
        import hashlib, json
        from pathlib import Path

        lora_cfg = getattr(self.config, "lora", None)
        key_data = {
            "model": self.config.model_name,
            "quant_method": str(getattr(self.config.quantization, "quant_method", "nf4")),
            "bits": int(getattr(self.config.quantization, "bits", 4)),
            "lora_r": int(getattr(lora_cfg, "r", 8)) if lora_cfg else 0,
            "lora_alpha": int(getattr(lora_cfg, "lora_alpha", 16)) if lora_cfg else 0,
            "lora_targets": "_".join(sorted(getattr(lora_cfg, "target_modules", []))) if lora_cfg else "",
        }
        key_str = json.dumps(key_data, sort_keys=True)
        cache_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]
        output_dir = str(getattr(self.config, "output_dir", ".pisceslx/ckpt") or ".pisceslx/ckpt")
        return Path(output_dir) / ".qlora_cache" / f"model_{cache_hash}.pt"

    def invalidate_model_cache(self) -> None:
        """Delete cached model so next init rebuilds from scratch."""
        cache_path = self._get_qlora_cache_path()
        if cache_path.exists():
            cache_path.unlink()
            _LOG.info(f"Cache invalidated: {cache_path}")

    def _apply_lora(self) -> None:
        from peft import LoraConfig as _PeftLoraConfig, get_peft_model

        lora_cfg = getattr(self.config, "lora", None)
        if lora_cfg is None:
            return

        target_modules = getattr(lora_cfg, "target_modules", None)
        if not target_modules:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        target_modules_list = list(target_modules)
        lora_r = int(getattr(lora_cfg, "r", 8))
        lora_alpha = int(getattr(lora_cfg, "lora_alpha", 16))
        lora_dropout = float(getattr(lora_cfg, "lora_dropout", 0.05))
        lora_bias = str(getattr(lora_cfg, "bias", "none"))
        
        peft_cfg = _PeftLoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules_list,
            bias=lora_bias,
            task_type="CAUSAL_LM",
        )
        
        _LOG.info(f"Applying LoRA: r={lora_r}, alpha={lora_alpha}, target_modules={len(target_modules_list)}")
        self.model = get_peft_model(self.model, peft_cfg)
        _LOG.info("LoRA adapters injected, moving model to target device...")
    
    def _apply_quantization(self):
        """
        Apply model quantization based on configuration.
        
        Quantizes model weights to lower precision (INT4/INT8/FP8/NF4)
        to reduce memory usage and improve inference speed.
        
        For CPU initialization, uses optimized parallel quantization
        to minimize initialization time while keeping memory low.
        """
        method = str(getattr(self.config.quantization, "quant_method", "nf4") or "nf4").lower()
        bits = int(getattr(self.config.quantization, "bits", 4) or 4)
        group_size = int(getattr(self.config.quantization, "group_size", 128) or 128)

        if bits == 4 and method in {"nf4", "int4", "fp4"}:
            try:
                import bitsandbytes as bnb
                import torch.nn as nn
                import warnings
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import threading
                warnings.filterwarnings('ignore', message='.*_check_is_size.*')

                quant_type = "nf4" if method == "nf4" else "fp4"
                compute_dtype = torch.bfloat16 if str(getattr(self.config, "mixed_precision", "bf16")).lower() == "bf16" else torch.float16

                _LOG.info(f"Quantizing linear layers with {quant_type} (CPU-optimized parallel mode)...")

                skip_modules = set()
                linear_modules = []
                conv_modules = []
                
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.MultiheadAttention):
                        for child_name, _ in module.named_modules():
                            if child_name:
                                skip_modules.add(f"{name}.{child_name}")
                            else:
                                skip_modules.add(name)
                    elif isinstance(module, nn.Linear) and name not in skip_modules:
                        linear_modules.append((name, module))
                    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                        conv_modules.append((name, module))

                total_linear = len(linear_modules)
                _LOG.info(f"Found {total_linear} linear layers to quantize")

                quantized_results = {}
                quantized_results_lock = threading.Lock()
                progress_counter = [0]
                log_interval = max(1, total_linear // 10)

                def quantize_single_layer(name: str, module: nn.Linear):
                    weight_data = module.weight.data.clone()
                    bias_data = module.bias.data.clone() if module.bias is not None else None
                    in_features = module.in_features
                    out_features = module.out_features
                    has_bias = module.bias is not None
                    
                    quantized_weight = bnb.nn.Params4bit(
                        weight_data,
                        requires_grad=False,
                        quant_type=quant_type,
                    )
                    
                    with quantized_results_lock:
                        progress_counter[0] += 1
                        if progress_counter[0] % log_interval == 0 or progress_counter[0] == total_linear:
                            _LOG.info(f"Quantization progress: {progress_counter[0]}/{total_linear} ({100*progress_counter[0]//total_linear}%)")
                    
                    return (name, quantized_weight, bias_data, in_features, out_features, has_bias)

                max_workers = min(8, max(1, total_linear // 50))
                _LOG.info(f"Using {max_workers} parallel workers for quantization")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(quantize_single_layer, name, module): name 
                        for name, module in linear_modules
                    }
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            name = result[0]
                            quantized_results[name] = result
                        except Exception as e:
                            failed_name = futures[future]
                            _LOG.warning(f"Failed to quantize {failed_name}: {e}")

                _LOG.info("Applying quantized weights to model...")
                linear4bit_count = 0
                
                for name, module in linear_modules:
                    if name not in quantized_results:
                        continue
                    
                    _, quantized_weight, bias_data, in_features, out_features, has_bias = quantized_results[name]
                    
                    parent = self.model
                    parts = name.split('.')
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    child_name = parts[-1] if parts else name
                    
                    new_mod = bnb.nn.Linear4bit(
                        in_features,
                        out_features,
                        bias=has_bias,
                        quant_type=quant_type,
                        compress_statistics=False,
                        compute_dtype=compute_dtype,
                    )
                    new_mod.weight = quantized_weight
                    if has_bias:
                        new_mod.bias.data = bias_data
                    setattr(parent, child_name, new_mod)
                    linear4bit_count += 1

                conv_quantized_count = 0
                for name, module in conv_modules:
                    if module.weight.dtype != compute_dtype:
                        module.weight.data = module.weight.data.to(compute_dtype)
                        if module.bias is not None:
                            module.bias.data = module.bias.data.to(compute_dtype)
                        conv_quantized_count += 1

                del quantized_results
                
                for p in self.model.parameters():
                    p.requires_grad = False

                trainable = sum(int(p.numel()) for p in self.model.parameters() if p.requires_grad)

                _LOG.info(
                    "bitsandbytes 4bit conversion finished",
                    linear4bit_layers=int(linear4bit_count),
                    conv_dtype_optimized=int(conv_quantized_count),
                    trainable_params=int(trainable),
                )

                _LOG.info(
                    "Model quantization applied successfully",
                    method=f"bitsandbytes:{quant_type}",
                    bits=bits,
                    group_size=group_size,
                )
                return
            except Exception as e:
                _LOG.warning(f"bitsandbytes 4bit quantization requested but failed; falling back: {e}")

    def _apply_quantization_gpu_accelerated(self):
        """
        GPU-accelerated quantization using layer-by-layer GPU processing.
        
        Strategy:
        1. Move layer weights to GPU
        2. Quantize on GPU (faster than CPU)
        3. Move quantized weights back to CPU
        4. Repeat for all layers
        
        This is faster than pure CPU quantization while keeping memory low.
        """
        method = str(getattr(self.config.quantization, "quant_method", "nf4") or "nf4").lower()
        bits = int(getattr(self.config.quantization, "bits", 4) or 4)
        group_size = int(getattr(self.config.quantization, "group_size", 128) or 128)

        if bits == 4 and method in {"nf4", "int4", "fp4"}:
            try:
                import bitsandbytes as bnb
                import torch.nn as nn
                import warnings
                warnings.filterwarnings('ignore', message='.*_check_is_size.*')

                quant_type = "nf4" if method == "nf4" else "fp4"
                compute_dtype = torch.bfloat16 if str(getattr(self.config, "mixed_precision", "bf16")).lower() == "bf16" else torch.float16

                _LOG.info(f"Quantizing linear layers with {quant_type} (GPU-accelerated mode)...")

                skip_modules = set()
                linear_modules = []
                
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.MultiheadAttention):
                        for child_name, _ in module.named_modules():
                            if child_name:
                                skip_modules.add(f"{name}.{child_name}")
                            else:
                                skip_modules.add(name)
                    elif isinstance(module, nn.Linear) and name not in skip_modules:
                        linear_modules.append((name, module))

                total_linear = len(linear_modules)
                _LOG.info(f"Found {total_linear} linear layers to quantize")
                _LOG.info("Using GPU-accelerated quantization (layer by layer)")

                linear4bit_count = 0
                log_interval = max(1, total_linear // 10)
                
                for idx, (name, module) in enumerate(linear_modules):
                    if idx % log_interval == 0 or idx == total_linear - 1:
                        _LOG.info(f"GPU Quantization progress: {idx + 1}/{total_linear} ({100 * (idx + 1) // total_linear}%)")
                    
                    try:
                        parent = self.model
                        parts = name.split('.')
                        for part in parts[:-1]:
                            parent = getattr(parent, part)
                        child_name = parts[-1] if parts else name
                        
                        weight_data = module.weight.data.clone()
                        bias_data = module.bias.data.clone() if module.bias is not None else None
                        in_features = module.in_features
                        out_features = module.out_features
                        has_bias = module.bias is not None
                        
                        quantized_weight = bnb.nn.Params4bit(
                            weight_data,
                            requires_grad=False,
                            quant_type=quant_type,
                        )
                        
                        new_mod = bnb.nn.Linear4bit(
                            in_features,
                            out_features,
                            bias=has_bias,
                            quant_type=quant_type,
                            compress_statistics=False,
                            compute_dtype=compute_dtype,
                        )
                        new_mod.weight = quantized_weight
                        if has_bias:
                            new_mod.bias.data = bias_data
                        
                        setattr(parent, child_name, new_mod)
                        linear4bit_count += 1
                        
                    except Exception as e:
                        _LOG.warning(f"Failed to quantize layer {name}: {e}")

                for p in self.model.parameters():
                    p.requires_grad = False

                trainable = sum(int(p.numel()) for p in self.model.parameters() if p.requires_grad)

                _LOG.info(
                    "GPU-accelerated 4bit conversion finished",
                    linear4bit_layers=int(linear4bit_count),
                    trainable_params=int(trainable),
                )
                _LOG.info(
                    "Model quantization applied successfully",
                    method=f"bitsandbytes:{quant_type}",
                    bits=bits,
                    group_size=group_size,
                )
                return
            except Exception as e:
                _LOG.warning(f"GPU-accelerated quantization failed: {e}, falling back to CPU")
                self._apply_quantization()

        try:
            from ops.quantize.core import QuantizationOperator

            quant_op = QuantizationOperator()
            quant_config = {
                "method": method,
                "bits": bits,
                "group_size": group_size,
                "symmetric": bool(getattr(self.config.quantization, "symmetric", False)),
            }

            self.model = quant_op.apply_quantization(self.model, quant_config)
            _LOG.info("Model quantization applied successfully", method=str(method), bits=int(bits), group_size=int(group_size))

        except Exception as e:
            _LOG.warning(f"Quantization failed: {e}")
    
    def _enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for memory-efficient training.
        
        Gradient checkpointing trades computation for memory by recomputing
        activations during backward pass instead of storing them.
        
        Memory Savings:
            Typically reduces memory usage by 30-50% for transformer models.
            Exact savings depend on model architecture and sequence length.
        """
        try:
            # For Transformer models with built-in support
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            else:
                # Manual enable for custom modules
                for module in self.model.modules():
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = True
                        
            _LOG.info("Gradient checkpointing enabled")
        except Exception as e:
            _LOG.warning(f"Failed to enable gradient checkpointing: {e}")
    
    def _setup_distributed_training(self):
        """
        Setup distributed training with DistributedDataParallel.
        
        Initializes process group and wraps model for multi-GPU training.
        Requires torchrun or mpirun for process launching.
        
        Backend:
            Uses NCCL backend for GPU communication (fastest for CUDA).
            Falls back to Gloo if NCCL is not available.
        """
        try:
            import torch.distributed as dist
            
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            
            # Wrap model with DistributedDataParallel
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[torch.cuda.current_device()]
            )
            
            _LOG.info("Distributed training setup completed")
        except Exception as e:
            _LOG.error(f"Distributed training setup failed: {e}")
    
    def initialize_optimizer(self, optimizer_class: Optional[type] = None, **optimizer_kwargs) -> torch.optim.Optimizer:
        """
        Initialize the optimizer with Ink as the default unified optimizer.
        
        The Ink optimizer integrates INT8/INT4 state compression, sparse gradients,
        and GaLore/FP4/ROOT techniques for maximum memory efficiency and throughput.
        
        Args:
            optimizer_class: Optimizer class (ignored when using Ink).
            **optimizer_kwargs: Optimizer parameters.
            
        Returns:
            Initialized optimizer instance.
        """
        _LOG.info("Initializing Ink optimizer...")
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        try:
            from opss.optim.ink import POPSSInkOptimizer, POPSSInkConfig
            
            ink_section = getattr(self.config, 'ink_optimizer', None) or self.config
            
            ink_config = POPSSInkConfig(
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps,
                max_grad_norm=getattr(self.config.optimizer, 'max_grad_norm', 1.0),
                momentum_bits=getattr(ink_section, 'momentum_bits', 8),
                momentum_block_size=getattr(ink_section, 'momentum_block_size', 128),
                variance_bits=getattr(ink_section, 'variance_bits', 8),
                variance_block_size=getattr(ink_section, 'variance_block_size', 256),
                sparse_ratio=getattr(ink_section, 'sparse_ratio', 0.01),
                sparse_warmup_steps=getattr(ink_section, 'sparse_warmup_steps', 1000),
                sparse_adaptive=True,
                ortho_momentum=getattr(ink_section, 'ortho_momentum', 0.9),
                galore_rank=getattr(ink_section, 'galore_rank', 160),
                galore_update_proj_gap=getattr(ink_section, 'galore_update_proj_gap', 200),
                galore_quantization_bits=getattr(ink_section, 'galore_quantization_bits', 8),
                galore_min_rank=getattr(ink_section, 'galore_min_rank', 40),
                galore_max_rank=getattr(ink_section, 'galore_max_rank', 320),
                galore_rank_adapt_interval=getattr(ink_section, 'galore_rank_adapt_interval', 1000),
                galore_rank_adapt_threshold=getattr(ink_section, 'galore_rank_adapt_threshold', 0.1),
                galore_memory_efficient=getattr(ink_section, 'galore_memory_efficient', True),
                galore_moe_expert_only=getattr(ink_section, 'galore_moe_expert_only', False),
                fp4_block_size=getattr(ink_section, 'fp4_block_size', 16),
                fp4_stochastic_rounding=getattr(ink_section, 'fp4_stochastic_rounding', True),
                fp4_master_weights_dtype=getattr(ink_section, 'fp4_master_weights_dtype', 'fp32'),
                root_ortho_steps=getattr(ink_section, 'root_ortho_steps', 5),
                root_soft_threshold=getattr(ink_section, 'root_soft_threshold', 0.1),
                root_spectral_norm_clip=getattr(ink_section, 'root_spectral_norm_clip', 1.0),
                root_min_dim_for_ortho=getattr(ink_section, 'root_min_dim_for_ortho', 16),
                gradient_bits=getattr(ink_section, 'gradient_bits', 8),
                gradient_block_size=getattr(ink_section, 'gradient_block_size', 128),
                gradient_sparse_ratio=getattr(ink_section, 'gradient_sparse_ratio', 0.01),
                kv_cache_bits=getattr(ink_section, 'kv_cache_bits', 8),
                kv_cache_block_size=getattr(ink_section, 'kv_cache_block_size', 64),
                max_experts_on_gpu=getattr(ink_section, 'max_experts_on_gpu', 4),
                moe_offload_threshold=getattr(ink_section, 'moe_offload_threshold', 0.8),
                moe_lru_cache_size=getattr(ink_section, 'moe_lru_cache_size', 8),
                checkpoint_transformer=getattr(ink_section, 'checkpoint_transformer', True),
                checkpoint_ratio=getattr(ink_section, 'checkpoint_ratio', 0.5),
                checkpoint_preserve_ratio=getattr(ink_section, 'checkpoint_preserve_ratio', 0.3),
            )

            self.optimizer = POPSSInkOptimizer(trainable_params, config=ink_config)
            self._ink_config = ink_config

            memory_stats = self.optimizer.get_memory_stats()
            _LOG.info(
                f"Ink optimizer initialized: "
                f"momentum={ink_config.momentum_bits}bit, "
                f"variance={ink_config.variance_bits}bit, "
                f"sparse_ratio={ink_config.sparse_ratio}, "
                f"gradient={ink_config.gradient_bits}bit, "
                f"kv_cache={ink_config.kv_cache_bits}bit, "
                f"moe_gpu={ink_config.max_experts_on_gpu}, "
                f"checkpoint_ratio={ink_config.checkpoint_ratio}, "
                f"compression_ratio={memory_stats['compression_ratio']:.2f}x, "
                f"ortho_momentum={ink_config.ortho_momentum}, "
                f"galore_rank={ink_config.galore_rank}"
            )
            
        except Exception as e:
            _LOG.warning(f"Ink optimizer initialization failed, falling back to AdamW: {e}")
            
            default_params = {
                'lr': self.config.optimizer.learning_rate,
                'weight_decay': self.config.optimizer.weight_decay,
                'betas': self.config.optimizer.betas,
                'eps': self.config.optimizer.eps
            }
            default_params.update(optimizer_kwargs)
            
            opt_name = str(getattr(self.config.optimizer, "name", "adamw") or "adamw").lower()
            if opt_name == "sgd":
                optimizer_class = torch.optim.SGD
            else:
                optimizer_class = torch.optim.AdamW
            
            self.optimizer = optimizer_class(trainable_params, **default_params)
            _LOG.info(f"Fallback optimizer {self.optimizer.__class__.__name__} initialized")
        
        self._initialize_scheduler()
        
        return self.optimizer
    
    def _initialize_scheduler(self):
        """Initialize learning rate scheduler with warmup support."""
        try:
            from opss.train.lr_scheduler import (
                POPSSLRSchedulerOperator,
                POPSSLRSchedulerConfig,
                POPSSSchedulerType
            )
            
            warmup_steps = self.config.scheduler.warmup_steps
            if warmup_steps == 0 and self.config.scheduler.warmup_ratio > 0:
                warmup_steps = int(self.config.scheduler.warmup_ratio * self.config.max_steps)
            
            scheduler_type_map = {
                'cosine': POPSSSchedulerType.COSINE,
                'linear': POPSSSchedulerType.LINEAR,
                'polynomial': POPSSSchedulerType.POLYNOMIAL,
                'inverse_square': POPSSSchedulerType.INVERSE_SQUARE,
                'step': POPSSSchedulerType.STEP,
                'exponential': POPSSSchedulerType.EXPONENTIAL,
            }
            
            scheduler_type = scheduler_type_map.get(
                self.config.scheduler.name.lower(), 
                POPSSSchedulerType.COSINE
            )
            
            lr_config = POPSSLRSchedulerConfig(
                type=scheduler_type,
                initial_lr=self.config.optimizer.learning_rate,
                min_lr=self.config.optimizer.learning_rate * self.config.scheduler.min_lr_ratio,
                max_lr=self.config.optimizer.learning_rate,
                warmup_steps=warmup_steps,
                warmup_type=str(getattr(self.config.scheduler, "warmup_type", "linear") or "linear"),
                total_steps=self.config.max_steps
            )
            
            self._lr_scheduler_operator = POPSSLRSchedulerOperator(lr_config)
            self.scheduler = self._lr_scheduler_operator

            # IMPORTANT: Ensure the optimizer starts at the warmup-start LR.
            # Otherwise, the optimizer keeps the initial max_lr until the first optimizer_step()
            # (under gradient accumulation), then the scheduler suddenly drops it to warmup LR,
            # which looks like an unexpected LR collapse in logs.
            try:
                if self.optimizer is not None and hasattr(self.scheduler, "execute"):
                    self.scheduler.execute({
                        "step": 0,
                        "optimizer": self.optimizer,
                        "reset": True,
                    })
                    _LOG.debug("Learning rate scheduler reset successfully")
            except Exception as e:
                _LOG.warning(f"Failed to reset learning rate scheduler: {e}. Training will continue, but scheduler state may be inconsistent.")
            
            _LOG.info(
                f"Learning rate scheduler {self.config.scheduler.name} initialized "
                f"with warmup_steps={warmup_steps}, "
                f"min_lr_ratio={self.config.scheduler.min_lr_ratio}"
            )
            
        except ImportError:
            _LOG.warning("LRSchedulerOperator not available, falling back to PyTorch schedulers")
            self._initialize_scheduler_fallback()
        except Exception as e:
            _LOG.error(f"Scheduler initialization failed: {e}")
            self.scheduler = None
    
    def _initialize_scheduler_fallback(self):
        """Fallback to PyTorch built-in schedulers (no warmup support)."""
        try:
            if self.config.scheduler.name.lower() == 'cosine':
                from torch.optim.lr_scheduler import CosineAnnealingLR
                
                T_max = self.config.scheduler.decay_steps or self.config.max_steps
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=T_max,
                    eta_min=self.config.optimizer.learning_rate * self.config.scheduler.min_lr_ratio
                )
                
            elif self.config.scheduler.name.lower() == 'linear':
                from torch.optim.lr_scheduler import LinearLR
                
                self.scheduler = LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=self.config.scheduler.min_lr_ratio,
                    total_iters=self.config.max_steps
                )
            
            _LOG.info(f"Fallback scheduler {self.config.scheduler.name} initialized (no warmup)")
            
        except Exception as e:
            _LOG.error(f"Fallback scheduler initialization failed: {e}")
            self.scheduler = None
    
    def forward_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute forward pass with stage-aware loss computation.

        Args:
            batch: Input batch data.

        Returns:
            Dictionary containing loss and other metrics.
        """
        # Check if FP4 training is enabled
        if hasattr(self, '_fp4_operator') and self._fp4_operator is not None:
            try:
                # Use FP4 execute method for the entire forward-backward step
                result = self._fp4_operator.execute({
                    "model": self.model,
                    "batch": batch,
                    "config": getattr(self, '_fp4_config', POPSSFP4Config()),
                    "optimizer": self.optimizer,
                    "step": self.global_step,
                })
                
                if result.status == PiscesLxOperatorStatus.SUCCESS:
                    return {
                        'loss': torch.tensor(result.output['loss'], device=self.device),
                        'grad_norm': result.output.get('grad_norm', 0.0),
                        'scale_factor': result.output.get('scale_factor', 1.0),
                    }
                else:
                    _LOG.warning(f"FP4 execution failed: {result.error}")
                    # Fall back to standard forward pass
            except Exception as e:
                _LOG.warning(f"FP4 execution error: {e}")
                # Fall back to standard forward pass
        
        # Standard forward pass (fallback)
        non_blocking = False
        try:
            non_blocking = bool(getattr(getattr(self.config, "data", None), "pin_memory", False)) and self.device.type == "cuda"
        except Exception as e:
            non_blocking = False
            _LOG.warning(f"Failed to check pin_memory setting: {e}. Using blocking transfer.")
        batch = {k: v.to(self.device, non_blocking=non_blocking) for k, v in batch.items()}
        
        if self.config.mixed_precision in {"fp16", "bf16"} and self.device.type == "cuda":
            if self.config.mixed_precision == "bf16":
                bf16_supported = False
                try:
                    bf16_supported = bool(torch.cuda.is_bf16_supported())
                except Exception as e:
                    bf16_supported = False
                    _LOG.warning(f"Failed to check BF16 support in training_step: {e}. Assuming BF16 is not supported.")

                if not bf16_supported:
                    _LOG.warning(
                        "mixed_precision='bf16' requested but bf16 is not supported on this CUDA device; "
                        "falling back to fp16."
                    )
                    autocast_dtype = torch.float16
                else:
                    autocast_dtype = torch.bfloat16
            else:
                autocast_dtype = torch.float16

            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                return self._stage_forward(batch)

        return self._stage_forward(batch)
    
    def _stage_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Stage-aware forward pass that computes loss based on training stage.
        
        Args:
            batch: Input batch data
            
        Returns:
            Dictionary containing loss and outputs
        """
        from .config import TrainingStage
        
        if self.stage == TrainingStage.ALIGNMENT_DPO:
            return self._compute_dpo_forward(batch)
        elif self.stage == TrainingStage.ALIGNMENT_PPO:
            return self._compute_ppo_forward(batch)
        elif self.stage == TrainingStage.ALIGNMENT_ORPO:
            return self._compute_orpo_forward(batch)
        elif self.stage in [TrainingStage.SFT, TrainingStage.SPECIALIZED]:
            return self._compute_sft_forward(batch)
        else:
            outputs = self.model(**batch)
            if self.response_only_loss and 'response_mask' in batch:
                outputs['loss'] = self._apply_response_mask_loss(outputs, batch)
            return outputs
    
    def _compute_sft_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        SFT forward pass with response-only loss masking.
        
        Args:
            batch: Input batch with optional response_mask
            
        Returns:
            Outputs with masked loss
        """
        outputs = self.model(**batch)
        
        if 'response_mask' in batch:
            outputs['loss'] = self._apply_response_mask_loss(outputs, batch)
        elif self.response_only_loss and 'labels' in batch:
            labels = batch['labels']
            prompt_mask = (labels != -100).float()
            if 'attention_mask' in batch:
                first_non_pad = (batch['attention_mask'] == 1).float().argmax(dim=1)
                for i, start in enumerate(first_non_pad):
                    prompt_mask[i, :int(start) + 50] = 0
            outputs['loss'] = self._compute_masked_lm_loss(outputs, labels, prompt_mask)
        
        return outputs
    
    def _apply_response_mask_loss(self, outputs: Dict[str, Any], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply response-only mask to language model loss.
        
        Args:
            outputs: Model outputs with logits
            batch: Input batch with response_mask
            
        Returns:
            Masked loss tensor
        """
        logits = outputs.get('logits')
        labels = batch.get('labels')
        response_mask = batch.get('response_mask')
        
        if logits is None or labels is None:
            return outputs.get('loss', torch.tensor(0.0, device=self.device))
        
        if response_mask is None:
            return outputs.get('loss', torch.tensor(0.0, device=self.device))
        
        return self._compute_masked_lm_loss(outputs, labels, response_mask)
    
    def _compute_masked_lm_loss(self, outputs: Dict[str, Any], labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked language model loss.
        
        Args:
            outputs: Model outputs with logits
            labels: Target labels
            mask: Loss mask (1 for positions to compute loss, 0 for ignore)
            
        Returns:
            Masked loss tensor
        """
        import torch.nn.functional as F
        
        logits = outputs.get('logits')
        if logits is None:
            return torch.tensor(0.0, device=self.device)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )
        
        loss = loss.view(shift_labels.shape)
        masked_loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)
        
        return masked_loss
    
    def _compute_dpo_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        DPO (Direct Preference Optimization) forward pass.
        
        Computes DPO loss: L = -log sigmoid(beta * (log p_chosen - log p_rejected))
        
        Args:
            batch: Input batch with chosen/rejected sequences
            
        Returns:
            Dictionary with DPO loss
        """
        import torch.nn.functional as F
        
        chosen_ids = batch.get('chosen_input_ids')
        rejected_ids = batch.get('rejected_input_ids')
        chosen_mask = batch.get('chosen_attention_mask')
        rejected_mask = batch.get('rejected_attention_mask')
        
        if chosen_ids is None or rejected_ids is None:
            outputs = self.model(**batch)
            return outputs
        
        chosen_outputs = self.model(input_ids=chosen_ids, attention_mask=chosen_mask)
        rejected_outputs = self.model(input_ids=rejected_ids, attention_mask=rejected_mask)
        
        chosen_log_probs = self._get_sequence_log_probs(chosen_outputs, chosen_ids, chosen_mask)
        rejected_log_probs = self._get_sequence_log_probs(rejected_outputs, rejected_ids, rejected_mask)
        
        if self._reference_model is not None:
            with torch.no_grad():
                ref_chosen_outputs = self._reference_model(input_ids=chosen_ids, attention_mask=chosen_mask)
                ref_rejected_outputs = self._reference_model(input_ids=rejected_ids, attention_mask=rejected_mask)
                ref_chosen_log_probs = self._get_sequence_log_probs(ref_chosen_outputs, chosen_ids, chosen_mask)
                ref_rejected_log_probs = self._get_sequence_log_probs(ref_rejected_outputs, rejected_ids, rejected_mask)
            chosen_log_probs = chosen_log_probs - ref_chosen_log_probs
            rejected_log_probs = rejected_log_probs - ref_rejected_log_probs
        
        beta = getattr(self.config, 'beta', 0.1)
        loss = -F.logsigmoid(beta * (chosen_log_probs - rejected_log_probs)).mean()
        
        return {
            'loss': loss,
            'chosen_log_prob': chosen_log_probs.mean().item(),
            'rejected_log_prob': rejected_log_probs.mean().item(),
            'logits': chosen_outputs.get('logits')
        }
    
    def _compute_ppo_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        PPO (Proximal Policy Optimization) forward pass placeholder.
        
        Note: Full PPO requires rollout generation and value model.
        This is a simplified version for integration.
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary with PPO-related outputs
        """
        outputs = self.model(**batch)
        return outputs
    
    def _compute_orpo_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        ORPO (Odds Ratio Preference Optimization) forward pass.
        
        ORPO combines SFT loss with odds ratio loss without requiring a reference model.
        
        Args:
            batch: Input batch with chosen/rejected sequences
            
        Returns:
            Dictionary with ORPO loss
        """
        import torch.nn.functional as F
        
        chosen_ids = batch.get('chosen_input_ids')
        rejected_ids = batch.get('rejected_input_ids')
        chosen_mask = batch.get('chosen_attention_mask')
        rejected_mask = batch.get('rejected_attention_mask')
        
        if chosen_ids is None or rejected_ids is None:
            outputs = self.model(**batch)
            return outputs
        
        chosen_outputs = self.model(input_ids=chosen_ids, attention_mask=chosen_mask)
        rejected_outputs = self.model(input_ids=rejected_ids, attention_mask=rejected_mask)
        
        chosen_log_probs = self._get_sequence_log_probs(chosen_outputs, chosen_ids, chosen_mask)
        rejected_log_probs = self._get_sequence_log_probs(rejected_outputs, rejected_ids, rejected_mask)
        
        sft_loss = -chosen_log_probs.mean()
        
        log_odds_ratio = chosen_log_probs - rejected_log_probs
        orpo_loss = -F.logsigmoid(log_odds_ratio).mean()
        
        lambda_orpo = getattr(self.config, 'lambda_orpo', 0.1)
        loss = sft_loss + lambda_orpo * orpo_loss
        
        return {
            'loss': loss,
            'sft_loss': sft_loss.item(),
            'orpo_loss': orpo_loss.item(),
            'logits': chosen_outputs.get('logits')
        }
    
    def _get_sequence_log_probs(self, outputs: Dict[str, Any], input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute log probabilities for sequences.
        
        Args:
            outputs: Model outputs with logits
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Log probabilities summed over sequence length
        """
        import torch.nn.functional as F
        
        logits = outputs.get('logits')
        if logits is None:
            return torch.tensor(0.0, device=self.device)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].contiguous()
            token_log_probs = token_log_probs * shift_mask
        
        return token_log_probs.sum(dim=-1)
    
    def set_reference_model(self, ref_model):
        """
        Set reference model for DPO/PPO training.
        
        Args:
            ref_model: Reference model (frozen copy of initial model)
        """
        self._reference_model = ref_model
        if self._reference_model is not None:
            self._reference_model.eval()
            for param in self._reference_model.parameters():
                param.requires_grad = False
            _LOG.info("Reference model set for alignment training")
    
    def set_teacher_provider(self, teacher_provider):
        """
        Set teacher provider for knowledge distillation.
        
        Args:
            teacher_provider: POPSSTeacherProvider instance for distillation
        """
        self._teacher_provider = teacher_provider
        if self._teacher_provider is not None:
            _LOG.info(
                "Teacher provider set for distillation",
                provider_type=type(teacher_provider).__name__
            )
    
    def get_teacher_provider(self):
        """
        Get the teacher provider for distillation.
        
        Returns:
            Teacher provider instance or None
        """
        return self._teacher_provider
    
    def is_distillation_enabled(self):
        """
        Check if distillation is enabled.
        
        Returns:
            bool: True if teacher provider is set
        """
        return self._teacher_provider is not None
    
    def backward_pass(self, loss: torch.Tensor) -> float:
        """
        Execute backward pass with advanced gradient processing.
        
        Args:
            loss: Loss value from forward pass
            
        Returns:
            Gradient norm after processing
        """
        if self.scaler is not None:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()
        
        grad_norm = self._compute_gradient_norm()
        
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        max_grad_norm = 1.0
        try:
            max_grad_norm = float(getattr(getattr(self.config, "optimizer", None), "max_grad_norm", 1.0) or 1.0)
        except Exception as e:
            max_grad_norm = 1.0
            _LOG.warning(f"Failed to parse max_grad_norm: {e}. Using default value 1.0.")
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
        
        # Optimized: batch all auxiliary backward passes together
        aux_loss_total = 0.0
        
        # Collect GaLore gradients (no backward needed, just projection)
        if hasattr(self, '_galore_adapter') and self._galore_adapter is not None:
            try:
                gradients = {name: param.grad for name, param in self.model.named_parameters() 
                           if param.grad is not None}
                if gradients:
                    self.model, galore_stats = self._galore_adapter.step(self.model, gradients)
                    _LOG.debug(f"GaLore adapter step completed, stats: {galore_stats}")
            except Exception as e:
                _LOG.warning(f"GaLore adapter step failed: {e}. Skipping GaLore gradient projection.")
        
        # Collect MoE auxiliary loss
        if self._moe_gradient_optimizer is not None:
            try:
                moe_result = self._moe_gradient_optimizer.execute({
                    "model": self.model,
                    "step": self.global_step
                })
                if moe_result.is_success() and moe_result.output:
                    aux_val = moe_result.output.get('total_auxiliary_loss', 0.0)
                    aux_loss_total += aux_val
                    if aux_val != 0.0:
                        _LOG.debug(f"MoE gradient optimizer step completed, aux_loss: {aux_val}")
            except Exception as e:
                _LOG.warning(f"MoE gradient optimizer execute failed: {e}. Skipping MoE auxiliary loss collection.")
        
        # Collect weight watermark regularization loss
        if self._weight_watermark_operator is not None:
            try:
                wm_result = self._weight_watermark_operator._regularize({"model": self.model})
                if wm_result.is_success() and wm_result.output.get("regularization_loss") is not None:
                    aux_loss_total += wm_result.output["regularization_loss"].item()
                    _LOG.debug(f"Weight watermark regularization completed, loss: {wm_result.output['regularization_loss'].item()}")
            except Exception as e:
                _LOG.warning(f"Weight watermark regularization failed: {e}. Skipping watermark regularization loss.")
        
        # Single backward for all auxiliary losses (much more efficient)
        if aux_loss_total > 0:
            aux_loss_tensor = torch.tensor(aux_loss_total, device=self.device)
            if self.scaler is not None:
                self.scaler.scale(aux_loss_tensor).backward()
            else:
                aux_loss_tensor.backward()
        
        # K-FAC preconditioning (no backward needed)
        if self._kfac_operator is not None:
            try:
                self._kfac_operator.execute({
                    "model": self.model,
                    "step": self.global_step,
                    "backward_pass": True
                })
            except Exception as e:
                pass
        
        return grad_norm
    
    def _step_modality_scheduler(self):
        """Step the modality-aware scheduler if enabled."""
        if self._modality_scheduler is not None:
            try:
                self._modality_scheduler.step()
            except Exception as e:
                _LOG.warning(f"Modality scheduler step failed: {e}")
    
    def get_advanced_operator_stats(self) -> Dict[str, Any]:
        """Get statistics from all advanced operators.
        
        Returns:
            Dictionary containing statistics from modality scheduler,
            MoE gradient optimizer, K-FAC operator, and multi-task operator.
        """
        stats = {}
        
        if self._modality_scheduler is not None:
            stats['modality_scheduler'] = {
                'lr': self._modality_scheduler.get_lr()
            }
        
        if self._moe_gradient_optimizer is not None:
            stats['moe_gradient'] = self._moe_gradient_optimizer.get_gradient_statistics()
        
        if self._kfac_operator is not None:
            stats['kfac'] = self._kfac_operator.get_statistics()
        
        if self._multitask_operator is not None:
            stats['multitask'] = self._multitask_operator.get_statistics()
        
        if self._parallel_3d_operator is not None:
            stats['parallel_3d'] = {
                'enabled': True,
                'dp_size': self._parallel_3d_operator.config.dp_size,
                'tp_size': self._parallel_3d_operator.config.tp_size,
                'pp_size': self._parallel_3d_operator.config.pp_size,
                'world_size': self._parallel_3d_operator.config.world_size
            }
        
        if self._evolution_operator is not None:
            stats['evolution'] = {
                'enabled': True,
                'seed_size': self._evolution_config.seed_size if hasattr(self, '_evolution_config') else '0.5B',
                'target_size': self._evolution_config.target_size if hasattr(self, '_evolution_config') else '7B',
                'progress': self._evolution_operator.get_progress() if hasattr(self._evolution_operator, 'get_progress') else {},
            }
        
        return stats
    
    def optimizer_step(self):
        """Execute optimizer step with proper scheduler update."""
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        # Update learning rate
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'execute'):
                # LRSchedulerOperator from opss.train.lr_scheduler
                result = self.scheduler.execute({
                    # Align scheduler step to the *upcoming* optimizer step under gradient accumulation.
                    # global_step is incremented after optimizer_step(), so we pass global_step + 1 here.
                    'step': int(self.global_step) + 1,
                    'optimizer': self.optimizer
                })
                if not result.is_success():
                    _LOG.warning(f"Scheduler step failed: {result.error}")
            else:
                # PyTorch built-in scheduler
                self.scheduler.step()
        
        self.global_step += 1
    
    def _compute_gradient_norm(self) -> float:
        """Compute gradient norm efficiently."""
        # Optimized: use torch.no_grad and avoid Python loops
        with torch.no_grad():
            total_norm_sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.data.norm(2).item() ** 2
            return total_norm_sq ** 0.5
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute a complete training step.
        
        Args:
            batch: Input batch data.
            
        Returns:
            Training metrics dictionary.
        """
        # Use CUDA events for accurate timing without synchronization overhead
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start_time = time.time()

        grad_accum_steps = int(getattr(self.config, "gradient_accumulation_steps", 1) or 1)
        if grad_accum_steps < 1:
            grad_accum_steps = 1
        
        # Optimized: direct forward/backward without 3D parallelism overhead
        outputs = self.forward_pass(batch)
        loss = outputs['loss']
        if grad_accum_steps > 1:
            loss = loss / grad_accum_steps
        grad_norm = self.backward_pass(loss)
        
        # Optimized: skip multitask operator overhead (rarely used)
        
        self._grad_accum_step += 1
        did_step = (self._grad_accum_step % grad_accum_steps == 0)

        # Optimizer step
        if did_step:
            self.optimizer_step()
        
        # Record end event for accurate GPU timing
        if self.device.type == "cuda":
            end_event.record()

        # Calculate throughput (use CPU time for async execution, GPU time available via events)
        step_time = time.time() - start_time
        
        # Always compute token count and loss for accurate logging / epoch stats
        if 'input_ids' in batch:
            try:
                if 'attention_mask' in batch:
                    tokens = int(batch['attention_mask'].sum().item())
                else:
                    tokens = int(batch['input_ids'].numel())
            except Exception as e:
                tokens = 0
                _LOG.warning(f"Failed to compute token count: {e}. Setting tokens to 0.")
        else:
            tokens = 0

        throughput = batch['input_ids'].size(0) / step_time if ('input_ids' in batch and step_time > 0) else 0.0
        token_throughput = float(tokens) / step_time if (tokens and step_time > 0) else 0.0

        loss_scalar = float(loss.detach().item())
        if grad_accum_steps > 1:
            loss_scalar = loss_scalar * grad_accum_steps

        # Record detailed stats only on logging boundaries (avoids inflating history buffer)
        log_steps = int(getattr(self.config, 'log_steps', 100) or 100)
        if self.global_step % log_steps == 0:
            self._record_training_stats(loss_scalar, grad_norm, throughput)

        return {
            'loss': loss_scalar,
            'grad_norm': grad_norm,
            'learning_rate': self._get_current_lr(),
            'throughput': throughput,
            'token_throughput': token_throughput,
            'global_step': self.global_step,
            'step_time': step_time
        }
    
    def _record_training_stats(self, loss: float, grad_norm: float, throughput: float):
        """Record training statistics."""
        self.training_stats['loss_history'].append(loss)
        self.training_stats['grad_norm_history'].append(grad_norm)
        self.training_stats['lr_history'].append(self._get_current_lr())
        self.training_stats['throughput_history'].append(throughput)
        
        # Keep only the last 1000 records
        max_records = 1000
        if len(self.training_stats['loss_history']) > max_records:
            for key in self.training_stats:
                self.training_stats[key] = self.training_stats[key][-max_records:]
    
    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]['lr']
        return self.config.optimizer.learning_rate
    
    def save_checkpoint(self, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save training checkpoint.
        
        Args:
            filepath: Checkpoint file path.
            metadata: Additional metadata.
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'config': self.config.to_dict(),
            'training_stats': self.training_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        _LOG.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            filepath: Checkpoint file path.
            
        Returns:
            Checkpoint information.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.training_stats = checkpoint.get('training_stats', {})
        
        _LOG.info(f"Checkpoint loaded from {filepath}")
        return checkpoint
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        Get training progress information.
        
        Returns:
            Training progress dictionary.
        """
        return {
            'global_step': self.global_step,
            'progress_percentage': (self.global_step / self.config.max_steps) * 100,
            'current_loss': self.training_stats['loss_history'][-1] if self.training_stats['loss_history'] else 0,
            'current_lr': self._get_current_lr(),
            'best_metric': self.best_metric,
            'recent_throughput': (
                sum(self.training_stats['throughput_history'][-10:]) / 10 
                if len(self.training_stats['throughput_history']) >= 10 else 0
            )
        }
    
    def export_model(self, filepath: str, export_format: str = "torch"):
        """
        Export trained model.
        
        Args:
            filepath: Export file path.
            export_format: Export format ('torch', 'onnx', 'safetensors').
        """
        if export_format.lower() == "torch":
            torch.save(self.model.state_dict(), filepath)
        elif export_format.lower() == "onnx":
            # ONNX export requires example input
            dummy_input = torch.randn(1, 512, dtype=torch.long, device=self.device)
            torch.onnx.export(self.model, dummy_input, filepath)
        elif export_format.lower() == "safetensors":
            try:
                from safetensors.torch import save_file
                save_file(self.model.state_dict(), filepath)
            except ImportError:
                _LOG.warning("safetensors not installed, falling back to torch format")
                torch.save(self.model.state_dict(), filepath)
        
        _LOG.info(f"Model exported to {filepath} in {export_format} format")

    def validation_step(self, val_dataloader) -> Dict[str, float]:
        """
        Execute validation step.

        Args:
            val_dataloader: Validation dataloader.

        Returns:
            Validation metrics dictionary.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.config.mixed_precision == "fp16":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = self.model(**batch)
                elif self.config.mixed_precision == "bf16":
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)

                loss = outputs.get('loss', outputs.get('loss'))
                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1

        self.model.train()

        avg_loss = total_loss / max(num_batches, 1)
        self.training_stats['val_loss_history'].append(avg_loss)

        if avg_loss < self.best_metric:
            self.best_metric = avg_loss
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        _LOG.info(f"Validation loss: {avg_loss:.4f}, Best: {self.best_metric:.4f}")

        return {'val_loss': avg_loss}

    def train(self, train_dataloader, val_dataloader=None):
        """
        Execute complete training workflow.

        Args:
            train_dataloader: Training dataloader.
            val_dataloader: Validation dataloader (optional).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        self.model.train()
        max_steps = int(getattr(self.config, "max_steps", 0) or 0)
        if max_steps <= 0:
            _LOG.warning("Training skipped: max_steps <= 0", max_steps=max_steps)
            return

        log_steps = int(getattr(self.config, "log_steps", 10) or 10)
        save_steps = int(getattr(self.config, "save_steps", 0) or 0)
        eval_steps = int(getattr(self.config, "eval_steps", 0) or 0)

        output_dir = str(getattr(self.config, "output_dir", ".pisceslx/ckpt") or ".pisceslx/ckpt")
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            _LOG.debug(f"Output directory created/verified: {output_dir}")
        except Exception as e:
            _LOG.error(f"Failed to create output directory {output_dir}: {e}. Training may fail when saving checkpoints.")

        patience = getattr(self.config, "early_stopping_patience", None)
        if patience is not None:
            try:
                patience = int(patience)
                _LOG.debug(f"Early stopping patience: {patience}")
            except Exception as e:
                patience = None
                _LOG.warning(f"Failed to parse early_stopping_patience: {e}. Early stopping disabled.")

        _LOG.info("Starting training", max_steps=max_steps, output_dir=output_dir)

        while self.global_step < max_steps:
            self.epochs_completed += 1

            for batch in train_dataloader:
                step_result = self.train_step(batch)

                if log_steps > 0 and self.global_step % log_steps == 0:
                    _LOG.info(
                        f"Epoch {self.epochs_completed} | "
                        f"Step {self.global_step}/{max_steps} | "
                        f"Loss: {step_result['loss']:.4f} | "
                        f"Grad Norm: {step_result['grad_norm']:.4f} | "
                        f"LR: {step_result['learning_rate']:.2e} | "
                        f"Throughput: {step_result['throughput']:.2f} samples/s"
                    )

                if eval_steps > 0 and self.global_step % eval_steps == 0 and val_dataloader is not None:
                    self.validation_step(val_dataloader)

                if save_steps > 0 and self.global_step % save_steps == 0:
                    self.save_checkpoint(
                        f"{output_dir}/checkpoint-{self.global_step}.pt"
                    )

                if self.global_step >= max_steps:
                    _LOG.info("Training completed: max steps reached")
                    return

                if patience is not None and self.early_stop_counter >= patience:
                    _LOG.info(f"Early stopping triggered at step {self.global_step}")
                    return

        _LOG.info("Training completed")

    def should_stop_early(self) -> bool:
        """Check if early stopping should be triggered."""
        patience = getattr(self.config, "early_stopping_patience", None)
        if patience is None:
            return False
        try:
            patience = int(patience)
            _LOG.debug(f"Early stopping check: early_stop_counter={self.early_stop_counter}, patience={patience}")
        except Exception as e:
            _LOG.warning(f"Failed to parse patience in should_stop_early: {e}. Returning False.")
            return False
        return self.early_stop_counter >= patience
    
    def verify_weights(self) -> Tuple[float, bool]:
        """
        Verify model ownership through weight watermark detection.
        
        This method checks if the model weights contain the embedded watermark
        by computing correlation scores against the owner codebook.
        
        Returns:
            Tuple of (verification_score, passed)
            
        Raises:
            RuntimeError: If watermark operator not initialized
        """
        if self._weight_watermark_operator is None:
            raise RuntimeError("Weight watermark operator not initialized. Enable watermark in config.")
        
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            result = self._weight_watermark_operator._verify({"model": self.model})
            
            if result.is_success():
                score = result.output.get("verification_score", 0.0)
                passed = result.output.get("passed", False)
                
                _LOG.info(f"Weight verification: score={score:.4f}, threshold={self._watermark_config.verify_threshold:.4f}, passed={passed}")
                
                if self._audit_operator is not None:
                    self._audit_operator.log_operation(
                        operation="verify",
                        content_type="weight",
                        result="success" if passed else "failed",
                        metadata={
                            "verification_score": score,
                            "threshold": self._watermark_config.verify_threshold,
                            "model_id": self._watermark_config.model_id
                        }
                    )
                
                return score, passed
            else:
                _LOG.warning(f"Weight verification failed: {result.error}")
                return 0.0, False
                
        except Exception as e:
            _LOG.error(f"Weight verification error: {e}")
            if self._audit_operator is not None:
                self._audit_operator.log_operation(
                    operation="verify",
                    content_type="weight",
                    result="failed",
                    metadata={"error": str(e)}
                )
            return 0.0, False
    
    def get_watermark_stats(self) -> Dict[str, Any]:
        """Get watermark operator statistics."""
        if self._weight_watermark_operator is not None:
            stats = self._weight_watermark_operator._get_stats({})
            if stats.is_success():
                return stats.output
        return {"watermark_enabled": self._weight_watermark_operator is not None}
    
    def validate_watermark_compliance(self, jurisdiction: str = None) -> Dict[str, Any]:
        """
        Validate watermark configuration against compliance requirements.
        
        Args:
            jurisdiction: Target jurisdiction for validation
            
        Returns:
            Compliance validation report
        """
        if self._compliance_operator is None:
            return {"valid": False, "message": "Compliance operator not initialized"}
        
        try:
            result = self._compliance_operator._validate({
                "content_type": "weight",
                "jurisdiction": jurisdiction or self._watermark_config.jurisdiction.code,
                "config": self._watermark_config
            })
            
            if result.is_success():
                return result.output
            return {"valid": False, "error": result.error}
            
        except Exception as e:
            _LOG.error(f"Compliance validation error: {e}")
            return {"valid": False, "error": str(e)}
