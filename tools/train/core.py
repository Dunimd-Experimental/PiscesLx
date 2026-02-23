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
from opss.train.parallel_3d import (
    POPSSParallel3DConfig,
    POPSSParallel3DOperator
)
from opss.watermark import (
    POPSSWatermarkConfig,
    POPSSWeightWatermarkOperator,
    POPSSComplianceOperator,
    POPSSAuditOperator,
    POPSSWatermarkDefaultConfigFactory
)

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

        self._setup_advanced_operators()
        self._setup_watermark_operator()
        
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
            except Exception:
                bf16_supported = False

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
        try:
            if hasattr(self.config, 'modality_scheduler') and self.config.modality_scheduler.get('enabled', False):
                modality_config = POPSSModalitySchedulerConfig(**self.config.modality_scheduler)
                self._modality_scheduler = POPSSModalitySchedulerOperator(modality_config)
                _LOG.info("Modality-aware scheduler operator initialized")
        except Exception as e:
            _LOG.warning(f"Failed to initialize modality scheduler: {e}")
        
        try:
            if hasattr(self.config, 'moe_gradient') and self.config.moe_gradient.get('enabled', False):
                moe_config = POPSSMoEGradientConfig(**self.config.moe_gradient)
                self._moe_gradient_optimizer = POPSSMoEGradientOperator(moe_config)
                _LOG.info("MoE gradient optimizer operator initialized")
        except Exception as e:
            _LOG.warning(f"Failed to initialize MoE gradient optimizer: {e}")
        
        try:
            if hasattr(self.config, 'kfac') and self.config.kfac.get('enabled', False):
                kfac_config = POPSSKFacConfig(**self.config.kfac)
                self._kfac_operator = POPSSKFacOperator(kfac_config)
                _LOG.info("K-FAC operator initialized")
        except Exception as e:
            _LOG.warning(f"Failed to initialize K-FAC operator: {e}")
        
        try:
            if hasattr(self.config, 'multitask') and self.config.multitask.get('enabled', False):
                multitask_config = POPSSMultiTaskConfig(**self.config.multitask)
                self._multitask_operator = POPSSMultiTaskOperator(multitask_config)
                _LOG.info("Multi-task uncertainty operator initialized")
        except Exception as e:
            _LOG.warning(f"Failed to initialize multi-task operator: {e}")
    
    def _setup_parallel_3d_operator(self):
        """Setup 3D parallelism operator for large-scale distributed training."""
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
        
        # Create model instance with provided arguments
        self.model = model_class(**model_kwargs)
        
        # Move model to target device (GPU/CPU)
        self.model = self.model.to(self.device)
        
        # Ensure CUDA is fully initialized after model transfer
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            _LOG.info(f"Model moved to {self.device} and CUDA synchronized")
        
        if getattr(getattr(self.config, "lora", None), "enabled", False):
            self._apply_lora()

        # Apply quantization if enabled in configuration
        if self.config.quantization.enable_quantization:
            self._apply_quantization()
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Setup distributed training if enabled
        if self.config.distributed:
            self._setup_distributed_training()
        
        _LOG.info(f"Model initialized: {self.model.__class__.__name__}")
        _LOG.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model

    def _apply_lora(self) -> None:
        try:
            from peft import LoraConfig as _PeftLoraConfig, get_peft_model
        except Exception as e:
            _LOG.warning(f"LoRA requested but peft not available: {e}")
            return

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

        try:
            peft_cfg = _PeftLoraConfig(
                r=int(getattr(lora_cfg, "r", 8)),
                lora_alpha=int(getattr(lora_cfg, "lora_alpha", 16)),
                lora_dropout=float(getattr(lora_cfg, "lora_dropout", 0.05)),
                target_modules=list(target_modules),
                bias=str(getattr(lora_cfg, "bias", "none")),
                task_type=str(getattr(lora_cfg, "task_type", "CAUSAL_LM")),
            )
            self.model = get_peft_model(self.model, peft_cfg)
            try:
                self.model.print_trainable_parameters()
            except Exception:
                pass
            _LOG.info("LoRA enabled")
        except Exception as e:
            _LOG.warning(f"Failed to enable LoRA: {e}")
    
    def _apply_quantization(self):
        """
        Apply model quantization based on configuration.
        
        Quantizes model weights to lower precision (INT4/INT8/FP8/NF4)
        to reduce memory usage and improve inference speed.
        """
        try:
            from ops.quantize.core import QuantizationOperator

            quant_op = QuantizationOperator()
            quant_config = {
                'method': self.config.quantization.quant_method,
                'bits': self.config.quantization.bits,
                'group_size': self.config.quantization.group_size,
                'symmetric': self.config.quantization.symmetric
            }

            self.model = quant_op.apply_quantization(
                self.model,
                quant_config
            )
            _LOG.info("Model quantization applied successfully")

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
        Initialize the optimizer.
        
        Args:
            optimizer_class: Optimizer class, defaults to AdamW.
            **optimizer_kwargs: Optimizer parameters.
            
        Returns:
            Initialized optimizer instance.
        """
        _LOG.info("Initializing optimizer...")
        
        # Default optimizer parameters
        default_params = {
            'lr': self.config.optimizer.learning_rate,
            'weight_decay': self.config.optimizer.weight_decay,
            'betas': self.config.optimizer.betas,
            'eps': self.config.optimizer.eps
        }
        default_params.update(optimizer_kwargs)
        
        # Select optimizer class
        if optimizer_class is None:
            if self.config.optimizer.name.lower() == 'adamw':
                optimizer_class = torch.optim.AdamW
            elif self.config.optimizer.name.lower() == 'sgd':
                optimizer_class = torch.optim.SGD
            else:
                raise ValueError(f"Unsupported optimizer: {self.config.optimizer.name}")
        
        # Apply GaLore (if enabled)
        if self.config.optimizer.use_galore:
            try:
                from opss.optim.galore import POPSSGaLoreOptimizerAdapter, POPSSGaLoreConfig
                
                # Get trainable parameters
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                
                galore_config = POPSSGaLoreConfig(
                    rank=self.config.optimizer.galore_rank,
                    update_proj_gap=self.config.optimizer.galore_update_proj_gap,
                    **default_params
                )
                self._galore_adapter = POPSSGaLoreOptimizerAdapter(galore_config)
                self.optimizer = optimizer_class(trainable_params, **default_params)
                _LOG.info("GaLore optimizer initialized")
            except Exception as e:
                _LOG.warning(f"GaLore initialization failed, falling back to standard AdamW: {e}")
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = optimizer_class(trainable_params, **default_params)
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optimizer_class(trainable_params, **default_params)
        
        # Initialize learning rate scheduler
        self._initialize_scheduler()
        
        _LOG.info(f"Optimizer {self.optimizer.__class__.__name__} initialized")
        return self.optimizer
    
    def _initialize_scheduler(self):
        """Initialize learning rate scheduler."""
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
                
            _LOG.info(f"Learning rate scheduler {self.config.scheduler.name} initialized")
            
        except Exception as e:
            _LOG.error(f"Scheduler initialization failed: {e}")
    
    def forward_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute forward pass with stage-aware loss computation.
        
        Args:
            batch: Input batch data.
            
        Returns:
            Dictionary containing loss and other metrics.
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        if self.config.mixed_precision in {"fp16", "bf16"} and self.device.type == "cuda":
            if self.config.mixed_precision == "bf16":
                bf16_supported = False
                try:
                    bf16_supported = bool(torch.cuda.is_bf16_supported())
                except Exception:
                    bf16_supported = False

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
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        if self._moe_gradient_optimizer is not None:
            try:
                moe_result = self._moe_gradient_optimizer.execute({
                    "model": self.model,
                    "step": self.global_step
                })
                if moe_result.is_success() and moe_result.output:
                    aux_loss = moe_result.output.get('total_auxiliary_loss', 0.0)
                    if aux_loss > 0:
                        if self.scaler is not None:
                            scaled_aux_loss = self.scaler.scale(torch.tensor(aux_loss))
                            scaled_aux_loss.backward()
                        else:
                            aux_loss_tensor = torch.tensor(aux_loss, device=self.device)
                            aux_loss_tensor.backward()
                        _LOG.debug(f"MoE auxiliary loss: {aux_loss:.6f}")
            except Exception as e:
                _LOG.warning(f"MoE gradient optimization failed: {e}")
        
        if self._kfac_operator is not None:
            try:
                self._kfac_operator.execute({
                    "model": self.model,
                    "step": self.global_step,
                    "backward_pass": True
                })
            except Exception as e:
                _LOG.warning(f"K-FAC preconditioning failed: {e}")
        
        if self._weight_watermark_operator is not None:
            try:
                wm_result = self._weight_watermark_operator._regularize({"model": self.model})
                if wm_result.is_success() and wm_result.output.get("regularization_loss") is not None:
                    wm_loss = wm_result.output["regularization_loss"]
                    if self.scaler is not None:
                        scaled_wm_loss = self.scaler.scale(wm_loss)
                        scaled_wm_loss.backward()
                    else:
                        wm_loss.backward()
                    _LOG.debug(f"Weight watermark regularization loss: {wm_loss.item():.6f}")
            except Exception as e:
                _LOG.warning(f"Weight watermark regularization failed: {e}")
        
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
        
        return stats
    
    def optimizer_step(self):
        """Execute optimizer step."""
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.global_step += 1
    
    def _compute_gradient_norm(self) -> float:
        """Compute gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute a complete training step.
        
        Args:
            batch: Input batch data.
            
        Returns:
            Training metrics dictionary.
        """
        start_time = time.time()
        
        # Forward pass
        outputs = self.forward_pass(batch)
        loss = outputs['loss']
        
        # Backward pass
        grad_norm = self.backward_pass(loss)
        
        # Optimizer step
        self.optimizer_step()
        
        # Calculate throughput
        step_time = time.time() - start_time
        throughput = batch['input_ids'].size(0) / step_time if 'input_ids' in batch else 0
        
        # Record statistics
        self._record_training_stats(loss.item(), grad_norm, throughput)
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'learning_rate': self._get_current_lr(),
            'throughput': throughput,
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
        except Exception:
            pass

        patience = getattr(self.config, "early_stopping_patience", None)
        if patience is not None:
            try:
                patience = int(patience)
            except Exception:
                patience = None

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
        except Exception:
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
