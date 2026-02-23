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
Training Configuration Management System

Centralized configuration management for all training parameters in the PiscesLx
training framework. This module provides a hierarchical configuration system using
dataclasses, enabling type-safe, validated, and serializable configuration objects.

Configuration Hierarchy:
    The configuration system follows a hierarchical structure:
    
    TrainingConfig (Root)
    ├── OptimizerConfig: Optimizer hyperparameters and GaLore settings
    ├── SchedulerConfig: Learning rate scheduling and warmup configuration
    ├── DataConfig: Data loading, batching, and preprocessing settings
    └── QuantizationConfig: Quantization-aware training parameters

Design Principles:
    - Type Safety: All configurations use Python dataclasses with type hints
    - Default Values: Sensible defaults for all parameters to enable quick start
    - Serialization: Full support for JSON/YAML serialization and deserialization
    - Validation: Built-in validation through __post_init__ methods
    - Extensibility: Easy to extend with new configuration categories

Key Features:
    - GaLore Integration: Memory-efficient training with gradient low-rank projection
    - Mixed Precision: FP16/BF16 training configuration
    - Quantization: Support for INT4/INT8/FP8/NF4 quantization methods
    - Distributed Training: World size and gradient accumulation settings
    - Special Training Modes: DPO, SFT, Preference Alignment, Multi-task

Usage Examples:
    Basic Configuration:
        >>> from tools.train.config import TrainingConfig
        >>> 
        >>> config = TrainingConfig(
        ...     model_name="gpt-7b",
        ...     output_dir="./checkpoints",
        ...     max_steps=100000,
        ...     learning_rate=1e-4
        ... )

    Advanced Configuration with GaLore:
        >>> from tools.train.config import OptimizerConfig, SchedulerConfig
        >>> 
        >>> config = TrainingConfig(
        ...     model_name="llama-70b",
        ...     optimizer=OptimizerConfig(
        ...         name="adamw",
        ...         learning_rate=5e-5,
        ...         use_galore=True,
        ...         galore_rank=512,
        ...         galore_update_proj_gap=200
        ...     ),
        ...     scheduler=SchedulerConfig(
        ...         name="cosine",
        ...         warmup_steps=2000,
        ...         min_lr_ratio=0.1
        ...     )
        ... )

    Configuration Persistence:
        >>> # Save configuration
        >>> config.save_to_json("config.json")
        >>> 
        >>> # Load configuration
        >>> loaded_config = TrainingConfig.load_from_json("config.json")

Integration with Training System:
    The configuration objects are passed to various operators in the training system:
    - PiscesLxTrainingOperator: Uses TrainingConfig for initialization
    - TrainingOrchestrator: Centralizes configuration management
    - TrainingPipelineOperator: Accesses training hyperparameters
    - Optimizer Operators: Read optimizer-specific settings

Version Compatibility:
    Configurations include version information for backward compatibility.
    Future versions will support automatic migration of older configurations.

Attributes:
    See individual dataclass documentation for detailed attribute descriptions.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path
import json
import yaml

from configs.version import VERSION

TRAIN_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" / "train"


class TrainingStage(str, Enum):
    """
    Training Stage Enumeration
    
    Defines all supported training stages in the PiscesLx framework.
    Each stage has specific configurations and loss functions.
    
    Stages:
        PRETRAIN: Initial pre-training on large-scale corpus
        CONTINUED_PRETRAIN: Continued pre-training for domain adaptation
        SFT: Supervised Fine-Tuning on instruction data
        ALIGNMENT_DPO: Direct Preference Optimization alignment
        ALIGNMENT_PPO: PPO-based reinforcement learning alignment
        ALIGNMENT_ORPO: ORPO (Odds Ratio Preference Optimization) alignment
        SPECIALIZED: Task-specific fine-tuning
    """
    PRETRAIN = "pretrain"
    CONTINUED_PRETRAIN = "continued_pretrain"
    SFT = "sft"
    ALIGNMENT_DPO = "alignment_dpo"
    ALIGNMENT_PPO = "alignment_ppo"
    ALIGNMENT_ORPO = "alignment_orpo"
    SPECIALIZED = "specialized"


def get_stage_config_path(stage: TrainingStage) -> Path:
    """
    Get the configuration file path for a training stage.
    
    Args:
        stage: TrainingStage enum value
        
    Returns:
        Path to the stage configuration file
    """
    return TRAIN_CONFIG_DIR / f"{stage.value}.yaml"


def load_stage_config(stage: TrainingStage) -> Dict[str, Any]:
    """
    Load configuration for a training stage from file.
    
    Args:
        stage: TrainingStage enum value
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If the stage config file does not exist
    """
    config_path = get_stage_config_path(stage)
    if not config_path.exists():
        raise FileNotFoundError(f"Stage config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # Replace {{VERSION}} placeholder with actual version
    if "version" in config and config["version"] == "{{VERSION}}":
        config["version"] = VERSION
    
    return config


def list_available_stages() -> List[str]:
    """
    List all available training stage configurations.
    
    Returns:
        List of available stage names
    """
    if not TRAIN_CONFIG_DIR.exists():
        return []
    
    return [
        p.stem for p in TRAIN_CONFIG_DIR.glob("*.yaml")
        if p.stem != "default"
    ]


GLOBAL_DEFAULTS: Dict[str, Any] = {
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "flash_attention": True,
    "distributed": False,
    "world_size": 1,
    "gradient_accumulation_steps": 1,
    "moe_gradient": {"enabled": True},
    "kfac": {"enabled": False},
    "multitask": {"enabled": False},
    "watermark": {"enabled": False},
    "modality_scheduler": {"enabled": False},
}

STAGE_DEFAULTS: Dict[TrainingStage, Dict[str, Any]] = {
    TrainingStage.PRETRAIN: {
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "batch_size": 512,
        "sequence_length": 4096,
        "warmup_ratio": 0.01,
        "min_lr_ratio": 0.1,
        "loss_type": "lm",
        "response_only_loss": False,
        "moe_gradient": {"enabled": True},
        "kfac": {"enabled": False},
        "multitask": {"enabled": False},
        "packing": True,
    },
    TrainingStage.CONTINUED_PRETRAIN: {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "batch_size": 256,
        "sequence_length": 4096,
        "warmup_ratio": 0.02,
        "min_lr_ratio": 0.1,
        "loss_type": "lm",
        "response_only_loss": False,
        "moe_gradient": {"enabled": True},
        "kfac": {"enabled": False},
        "multitask": {"enabled": False},
    },
    TrainingStage.SFT: {
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "batch_size": 128,
        "sequence_length": 2048,
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.1,
        "loss_type": "lm",
        "response_only_loss": True,
        "moe_gradient": {"enabled": True},
        "kfac": {"enabled": True},
        "multitask": {"enabled": False},
    },
    TrainingStage.ALIGNMENT_DPO: {
        "learning_rate": 1e-6,
        "weight_decay": 0.0,
        "batch_size": 32,
        "sequence_length": 512,
        "warmup_ratio": 0.1,
        "min_lr_ratio": 0.1,
        "loss_type": "dpo",
        "beta": 0.1,
        "reference_free": False,
        "moe_gradient": {"enabled": False},
        "kfac": {"enabled": False},
        "multitask": {"enabled": False},
    },
    TrainingStage.ALIGNMENT_PPO: {
        "learning_rate": 1e-6,
        "weight_decay": 0.0,
        "batch_size": 16,
        "sequence_length": 512,
        "warmup_ratio": 0.1,
        "min_lr_ratio": 0.1,
        "loss_type": "ppo",
        "ppo_epochs": 4,
        "clip_range": 0.2,
        "kl_coef": 0.1,
        "moe_gradient": {"enabled": False},
        "kfac": {"enabled": False},
        "multitask": {"enabled": False},
    },
    TrainingStage.ALIGNMENT_ORPO: {
        "learning_rate": 5e-6,
        "weight_decay": 0.0,
        "batch_size": 32,
        "sequence_length": 512,
        "warmup_ratio": 0.1,
        "min_lr_ratio": 0.1,
        "loss_type": "orpo",
        "lambda_orpo": 0.1,
        "moe_gradient": {"enabled": False},
        "kfac": {"enabled": False},
        "multitask": {"enabled": False},
    },
    TrainingStage.SPECIALIZED: {
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "batch_size": 64,
        "sequence_length": 2048,
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.1,
        "loss_type": "lm",
        "response_only_loss": True,
        "moe_gradient": {"enabled": True},
        "kfac": {"enabled": True},
        "multitask": {"enabled": True},
    },
}


@dataclass
class OptimizerConfig:
    """
    Optimizer Configuration
    
    Configuration class for optimizer hyperparameters including support for
    advanced optimizers like GaLore (Gradient Low-Rank Projection).
    
    Attributes:
        name: Optimizer name (adamw, sgd, etc.)
        learning_rate: Initial learning rate for optimization
        weight_decay: L2 regularization coefficient
        betas: Beta coefficients for Adam optimizers (momentum, adaptive lr)
        eps: Small constant for numerical stability
        use_galore: Whether to enable GaLore memory-efficient training
        galore_rank: Rank for low-rank gradient projection in GaLore
        galore_update_proj_gap: Steps between projection matrix updates
    
    GaLore Configuration:
        GaLore reduces memory usage during training by projecting gradients
        to a low-rank subspace. Recommended settings:
        - galore_rank: 128-512 (higher for larger models)
        - galore_update_proj_gap: 200-500 (balance between accuracy and speed)
    
    Example:
        >>> # Standard AdamW
        >>> config = OptimizerConfig(name="adamw", learning_rate=1e-4)
        >>> 
        >>> # GaLore-enabled training for large models
        >>> config = OptimizerConfig(
        ...     name="adamw",
        ...     learning_rate=5e-5,
        ...     use_galore=True,
        ...     galore_rank=256,
        ...     galore_update_proj_gap=200
        ... )
    """
    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    use_galore: bool = False
    galore_rank: int = 128
    galore_update_proj_gap: int = 200


@dataclass
class SchedulerConfig:
    """
    Learning Rate Scheduler Configuration
    
    Configuration for learning rate scheduling during training, supporting
    various scheduling strategies including warmup and decay.
    
    Attributes:
        name: Scheduler type (cosine, linear, polynomial, etc.)
        warmup_steps: Number of warmup steps with linear learning rate increase
        warmup_ratio: Ratio of total steps for warmup (alternative to warmup_steps)
        min_lr_ratio: Minimum learning rate as ratio of initial learning rate
        decay_steps: Total steps for learning rate decay (defaults to max_steps)
    
    Scheduling Strategies:
        - cosine: Cosine annealing schedule (recommended for most cases)
        - linear: Linear decay from initial to minimum learning rate
        - polynomial: Polynomial decay with configurable power
        - constant: No decay, constant learning rate after warmup
    
    Warmup Strategy:
        Warmup gradually increases learning rate from 0 to initial_lr over
        warmup_steps. This stabilizes early training and prevents divergence.
        
    Example:
        >>> # Cosine schedule with warmup
        >>> config = SchedulerConfig(
        ...     name="cosine",
        ...     warmup_steps=1000,
        ...     min_lr_ratio=0.1
        ... )
        >>> 
        >>> # Linear decay without warmup
        >>> config = SchedulerConfig(
        ...     name="linear",
        ...     warmup_steps=0,
        ...     min_lr_ratio=0.0
        ... )
    """
    name: str = "cosine"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.0
    decay_steps: Optional[int] = None


@dataclass
class DataConfig:
    """
    Data Loading Configuration
    
    Configuration for data loading, batching, and preprocessing during training.
    Optimizes data pipeline for maximum throughput and minimal GPU idle time.
    
    Attributes:
        batch_size: Number of samples per batch (global batch size)
        sequence_length: Maximum sequence length for tokenized inputs
        num_workers: Number of subprocesses for data loading
        pin_memory: Whether to pin memory in DataLoader for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        datasets: List of dataset configurations for multi-dataset training
    
    Performance Optimization:
        - pin_memory=True: Essential for GPU training, enables async data transfer
        - num_workers: Set to 4-8 for CPU-bound preprocessing, 0 for GPU-bound
        - prefetch_factor: Higher values (2-4) reduce data loading bottlenecks
        - batch_size: Consider gradient_accumulation_steps for effective batch size
    
    Multi-Dataset Training:
        The datasets list supports training on multiple datasets simultaneously:
        >>> config = DataConfig(
        ...     batch_size=32,
        ...     datasets=[
        ...         {"name": "dataset1", "path": "./data1", "weight": 0.6},
        ...         {"name": "dataset2", "path": "./data2", "weight": 0.4}
        ...     ]
        ... )
    
    Example:
        >>> # Standard configuration for GPU training
        >>> config = DataConfig(
        ...     batch_size=16,
        ...     sequence_length=2048,
        ...     num_workers=4,
        ...     pin_memory=True,
        ...     prefetch_factor=2
        ... )
    """
    batch_size: int = 32
    sequence_length: int = 2048
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    datasets: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class QuantizationConfig:
    """
    Quantization Configuration
    
    Configuration for quantization-aware training (QAT) and post-training
    quantization. Supports multiple quantization methods and bit widths.
    
    Attributes:
        enable_quantization: Whether to enable quantization during training
        quant_method: Quantization method (int4, int8, fp8, nf4)
        bits: Number of bits for quantization (4, 8)
        group_size: Group size for grouped quantization
        symmetric: Whether to use symmetric quantization
        enable_fp8_linear: Enable FP8 linear layers (requires Hopper GPU)
    
    Quantization Methods:
        - int4: 4-bit integer quantization (highest compression, some accuracy loss)
        - int8: 8-bit integer quantization (good balance of speed and accuracy)
        - fp8: 8-bit floating point (NVIDIA Hopper, good for inference)
        - nf4: 4-bit Normal Float (QLoRA, recommended for fine-tuning)
    
    Grouped Quantization:
        Group size determines the granularity of quantization. Smaller groups
        (64-128) provide better accuracy but slightly lower speed.
    
    Symmetric vs Asymmetric:
        - Symmetric: Zero point is always 0, simpler but may lose precision
        - Asymmetric: Custom zero point, better for non-centered distributions
    
    Example:
        >>> # NF4 quantization for QLoRA fine-tuning
        >>> config = QuantizationConfig(
        ...     enable_quantization=True,
        ...     quant_method="nf4",
        ...     bits=4,
        ...     group_size=64
        ... )
        >>> 
        >>> # INT8 quantization for inference
        >>> config = QuantizationConfig(
        ...     enable_quantization=True,
        ...     quant_method="int8",
        ...     bits=8,
        ...     symmetric=True
        ... )
    """
    enable_quantization: bool = False
    quant_method: str = "nf4"  # int4, int8, fp8, nf4
    bits: int = 4
    group_size: int = 128
    symmetric: bool = False
    enable_fp8_linear: bool = False


@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """
    Complete Training Configuration
    
    Root configuration class that aggregates all training parameters into a
    single, cohesive configuration object. This is the primary interface for
    configuring the PiscesLx training system.
    
    Attributes:
        # Basic Configuration
        model_name: Name identifier for the model being trained
        output_dir: Directory for saving checkpoints and logs
        max_steps: Maximum number of training steps
        save_steps: Steps between checkpoint saves
        eval_steps: Steps between evaluation runs
        log_steps: Steps between logging updates
        
        # Hardware Configuration
        device: Compute device (cuda, cpu, mps)
        mixed_precision: Mixed precision mode (fp32, fp16, bf16)
        gradient_checkpointing: Enable gradient checkpointing for memory saving
        flash_attention: Enable Flash Attention for faster training
        
        # Distributed Configuration
        distributed: Enable distributed training
        world_size: Total number of training processes
        gradient_accumulation_steps: Steps to accumulate gradients before update
        
        # Component Configurations
        optimizer: OptimizerConfig instance
        scheduler: SchedulerConfig instance
        data: DataConfig instance
        quantization: QuantizationConfig instance
        
        # Training Stage
        stage: TrainingStage enumeration for stage-aware training
        
        # Special Training Modes
        enable_dpo: Enable Direct Preference Optimization
        enable_sft: Enable Supervised Fine-Tuning mode
        enable_pref_align: Enable preference alignment training
        enable_multitask: Enable multi-task training
    """
    model_name: str = "pisceslx-base"
    output_dir: str = ".pisceslx/ckpt"
    max_steps: int = 100000
    save_steps: int = 1000
    eval_steps: int = 500
    log_steps: int = 10
    
    device: str = "cuda"
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    
    distributed: bool = False
    world_size: int = 1
    gradient_accumulation_steps: int = 1
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    stage: Optional[TrainingStage] = None
    
    loss_type: str = "lm"
    response_only_loss: bool = False
    beta: float = 0.1
    reference_free: bool = False
    ppo_epochs: int = 4
    clip_range: float = 0.2
    kl_coef: float = 0.1
    lambda_orpo: float = 0.1
    packing: bool = False
    
    moe_gradient: Dict[str, Any] = field(default_factory=lambda: {"enabled": True})
    moe: Dict[str, Any] = field(default_factory=lambda: {
        "routing_temperature": 1.30,
        "temperature_min": 0.80,
        "temperature_max": 3.00,
        "expert_temperature_max": 5.00,
        "expert_load_balance_threshold": 0.15,
    })
    kfac: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    multitask: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    watermark: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    modality_scheduler: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    
    enable_dpo: bool = False
    enable_sft: bool = False
    enable_pref_align: bool = False
    enable_multitask: bool = False
    
    parallel_3d: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "dp_size": 1,
        "tp_size": 1,
        "pp_size": 1,
        "sequence_parallel": True,
        "num_micro_batches": 4,
        "overlap_communication": True,
        "gradient_checkpointing": False,
        "zero_stage": 0
    })
    
    def __post_init__(self):
        """Post initialization - do not auto-load config file."""
        pass
    
    def load_stage_config(self, stage: TrainingStage):
        """
        Load configuration from stage config file.
        
        This method must be called explicitly when user specifies --stage.
        
        Args:
            stage: The training stage to load configuration for
        """
        self.stage = stage
        try:
            stage_config = load_stage_config(self.stage)
            self._apply_stage_file_config(stage_config)
        except FileNotFoundError:
            pass
    
    def _apply_stage_file_config(self, config: Dict[str, Any]):
        """Apply configuration from loaded stage file."""
        if "optimizer" in config:
            opt = config["optimizer"]
            if "learning_rate" in opt:
                self.optimizer.learning_rate = opt["learning_rate"]
            if "weight_decay" in opt:
                self.optimizer.weight_decay = opt["weight_decay"]
            if "betas" in opt:
                self.optimizer.betas = tuple(opt["betas"])
            if "eps" in opt:
                self.optimizer.eps = opt["eps"]
            if "use_galore" in opt:
                self.optimizer.use_galore = opt["use_galore"]
            if "galore_rank" in opt:
                self.optimizer.galore_rank = opt["galore_rank"]
            if "galore_update_proj_gap" in opt:
                self.optimizer.galore_update_proj_gap = opt["galore_update_proj_gap"]
        
        if "scheduler" in config:
            sched = config["scheduler"]
            if "warmup_ratio" in sched:
                self.scheduler.warmup_ratio = sched["warmup_ratio"]
            if "min_lr_ratio" in sched:
                self.scheduler.min_lr_ratio = sched["min_lr_ratio"]
            if "name" in sched:
                self.scheduler.name = sched["name"]
        
        if "data" in config:
            data = config["data"]
            if "batch_size" in data:
                self.data.batch_size = data["batch_size"]
            if "sequence_length" in data:
                self.data.sequence_length = data["sequence_length"]
            if "num_workers" in data:
                self.data.num_workers = data["num_workers"]
            if "pin_memory" in data:
                self.data.pin_memory = data["pin_memory"]
            if "prefetch_factor" in data:
                self.data.prefetch_factor = data["prefetch_factor"]
        
        if "training" in config:
            train = config["training"]
            if "max_steps" in train:
                self.max_steps = train["max_steps"]
            if "save_steps" in train:
                self.save_steps = train["save_steps"]
            if "eval_steps" in train:
                self.eval_steps = train["eval_steps"]
            if "log_steps" in train:
                self.log_steps = train["log_steps"]
            if "gradient_accumulation_steps" in train:
                self.gradient_accumulation_steps = train["gradient_accumulation_steps"]
            if "gradient_checkpointing" in train:
                self.gradient_checkpointing = train["gradient_checkpointing"]
            if "mixed_precision" in train:
                self.mixed_precision = train["mixed_precision"]
            if "flash_attention" in train:
                self.flash_attention = train["flash_attention"]
            if "packing" in train:
                self.packing = train["packing"]

        if "moe" in config:
            moe_cfg = config["moe"]
            if isinstance(moe_cfg, dict):
                self.moe.update(moe_cfg)
        
        if "loss" in config:
            loss = config["loss"]
            if "loss_type" in loss:
                self.loss_type = loss["loss_type"]
            if "response_only_loss" in loss:
                self.response_only_loss = loss["response_only_loss"]
            if "beta" in loss:
                self.beta = loss["beta"]
            if "reference_free" in loss:
                self.reference_free = loss["reference_free"]
            if "ppo_epochs" in loss:
                self.ppo_epochs = loss["ppo_epochs"]
            if "clip_range" in loss:
                self.clip_range = loss["clip_range"]
            if "kl_coef" in loss:
                self.kl_coef = loss["kl_coef"]
            if "lambda_orpo" in loss:
                self.lambda_orpo = loss["lambda_orpo"]
        
        if "advanced_operators" in config:
            adv = config["advanced_operators"]
            if "moe_gradient" in adv:
                self.moe_gradient = adv["moe_gradient"]
            if "kfac" in adv:
                self.kfac = adv["kfac"]
            if "multitask" in adv:
                self.multitask = adv["multitask"]
            if "watermark" in adv:
                self.watermark = adv["watermark"]
            if "modality_scheduler" in adv:
                self.modality_scheduler = adv["modality_scheduler"]
        
        if "distributed" in config:
            dist = config["distributed"]
            if "enabled" in dist:
                self.distributed = dist["enabled"]
            if "world_size" in dist:
                self.world_size = dist["world_size"]
            if "parallel_3d" in dist:
                self.parallel_3d = dist["parallel_3d"]
        
        if "lora" in config:
            lora = config["lora"]
            if "enabled" in lora:
                self.lora.enabled = lora["enabled"]
            if "r" in lora:
                self.lora.r = lora["r"]
            if "lora_alpha" in lora:
                self.lora.lora_alpha = lora["lora_alpha"]
            if "lora_dropout" in lora:
                self.lora.lora_dropout = lora["lora_dropout"]
            if "target_modules" in lora:
                self.lora.target_modules = lora["target_modules"]
        
        if "output" in config:
            out = config["output"]
            if "output_dir" in out:
                self.output_dir = out["output_dir"]
            if "model_name" in out:
                self.model_name = out["model_name"]
    
    def switch_stage(self, new_stage: TrainingStage, **kwargs):
        """
        Switch to a new training stage.
        
        Args:
            new_stage: The target training stage
            **kwargs: Additional configuration overrides
        """
        self.load_stage_config(new_stage)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Serializes the entire configuration hierarchy into a nested dictionary
        suitable for JSON serialization or YAML export.
        
        Returns:
            Dictionary representation of the configuration with all nested
            components (optimizer, scheduler, data, quantization) expanded.
            
        Example:
            >>> config = TrainingConfig(model_name="test")
            >>> config_dict = config.to_dict()
            >>> print(config_dict["model_name"])  # "test"
            >>> print(config_dict["optimizer"]["name"])  # "adamw"
        """
        return {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "max_steps": self.max_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "log_steps": self.log_steps,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "flash_attention": self.flash_attention,
            "distributed": self.distributed,
            "world_size": self.world_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer": {
                "name": self.optimizer.name,
                "learning_rate": self.optimizer.learning_rate,
                "weight_decay": self.optimizer.weight_decay,
                "betas": self.optimizer.betas,
                "eps": self.optimizer.eps,
                "use_galore": self.optimizer.use_galore,
                "galore_rank": self.optimizer.galore_rank,
                "galore_update_proj_gap": self.optimizer.galore_update_proj_gap
            },
            "scheduler": {
                "name": self.scheduler.name,
                "warmup_steps": self.scheduler.warmup_steps,
                "warmup_ratio": self.scheduler.warmup_ratio,
                "min_lr_ratio": self.scheduler.min_lr_ratio,
                "decay_steps": self.scheduler.decay_steps
            },
            "data": {
                "batch_size": self.data.batch_size,
                "sequence_length": self.data.sequence_length,
                "num_workers": self.data.num_workers,
                "pin_memory": self.data.pin_memory,
                "prefetch_factor": self.data.prefetch_factor,
                "datasets": self.data.datasets
            },
            "quantization": {
                "enable_quantization": self.quantization.enable_quantization,
                "quant_method": self.quantization.quant_method,
                "bits": self.quantization.bits,
                "group_size": self.quantization.group_size,
                "symmetric": self.quantization.symmetric,
                "enable_fp8_linear": self.quantization.enable_fp8_linear
            },
            "lora": {
                "enabled": self.lora.enabled,
                "r": self.lora.r,
                "lora_alpha": self.lora.lora_alpha,
                "lora_dropout": self.lora.lora_dropout,
                "target_modules": self.lora.target_modules,
                "bias": self.lora.bias,
                "task_type": self.lora.task_type,
            },
            "special_modes": {
                "enable_dpo": self.enable_dpo,
                "enable_sft": self.enable_sft,
                "enable_pref_align": self.enable_pref_align,
                "enable_multitask": self.enable_multitask
            },
            "stage": {
                "current": self.stage.value if self.stage else "pretrain",
                "loss_type": self.loss_type,
                "response_only_loss": self.response_only_loss,
                "beta": self.beta,
                "reference_free": self.reference_free,
                "ppo_epochs": self.ppo_epochs,
                "clip_range": self.clip_range,
                "kl_coef": self.kl_coef,
                "lambda_orpo": self.lambda_orpo,
                "packing": self.packing,
            },
            "advanced_operators": {
                "moe_gradient": self.moe_gradient,
                "moe": self.moe,
                "kfac": self.kfac,
                "multitask": self.multitask,
                "watermark": self.watermark,
                "modality_scheduler": self.modality_scheduler,
            },
            "parallel_3d": self.parallel_3d
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """
        Create configuration object from dictionary.
        
        Deserializes a configuration from a dictionary, typically loaded from
        JSON or YAML configuration files. Missing keys use default values.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            TrainingConfig instance with values from dictionary
            
        Example:
            >>> config_dict = {
            ...     "model_name": "gpt-7b",
            ...     "max_steps": 50000,
            ...     "optimizer": {"learning_rate": 5e-5}
            ... }
            >>> config = TrainingConfig.from_dict(config_dict)
        """
        config = cls()
        
        # Basic configuration
        config.model_name = config_dict.get("model_name", config.model_name)
        config.output_dir = config_dict.get("output_dir", config.output_dir)
        config.max_steps = config_dict.get("max_steps", config.max_steps)
        config.save_steps = config_dict.get("save_steps", config.save_steps)
        config.eval_steps = config_dict.get("eval_steps", config.eval_steps)
        config.log_steps = config_dict.get("log_steps", config.log_steps)
        config.device = config_dict.get("device", config.device)
        config.mixed_precision = config_dict.get("mixed_precision", config.mixed_precision)
        config.gradient_checkpointing = config_dict.get("gradient_checkpointing", config.gradient_checkpointing)
        config.flash_attention = config_dict.get("flash_attention", config.flash_attention)
        config.distributed = config_dict.get("distributed", config.distributed)
        config.world_size = config_dict.get("world_size", config.world_size)
        config.gradient_accumulation_steps = config_dict.get("gradient_accumulation_steps", config.gradient_accumulation_steps)
        
        # Optimizer configuration
        if "optimizer" in config_dict:
            opt_dict = config_dict["optimizer"]
            config.optimizer.name = opt_dict.get("name", config.optimizer.name)
            config.optimizer.learning_rate = opt_dict.get("learning_rate", config.optimizer.learning_rate)
            config.optimizer.weight_decay = opt_dict.get("weight_decay", config.optimizer.weight_decay)
            config.optimizer.betas = tuple(opt_dict.get("betas", config.optimizer.betas))
            config.optimizer.eps = opt_dict.get("eps", config.optimizer.eps)
            config.optimizer.use_galore = opt_dict.get("use_galore", config.optimizer.use_galore)
            config.optimizer.galore_rank = opt_dict.get("galore_rank", config.optimizer.galore_rank)
            config.optimizer.galore_update_proj_gap = opt_dict.get("galore_update_proj_gap", config.optimizer.galore_update_proj_gap)
        
        # Scheduler configuration
        if "scheduler" in config_dict:
            sched_dict = config_dict["scheduler"]
            config.scheduler.name = sched_dict.get("name", config.scheduler.name)
            config.scheduler.warmup_steps = sched_dict.get("warmup_steps", config.scheduler.warmup_steps)
            config.scheduler.warmup_ratio = sched_dict.get("warmup_ratio", config.scheduler.warmup_ratio)
            config.scheduler.min_lr_ratio = sched_dict.get("min_lr_ratio", config.scheduler.min_lr_ratio)
            config.scheduler.decay_steps = sched_dict.get("decay_steps", config.scheduler.decay_steps)
        
        # Data configuration
        if "data" in config_dict:
            data_dict = config_dict["data"]
            config.data.batch_size = data_dict.get("batch_size", config.data.batch_size)
            config.data.sequence_length = data_dict.get("sequence_length", config.data.sequence_length)
            config.data.num_workers = data_dict.get("num_workers", config.data.num_workers)
            config.data.pin_memory = data_dict.get("pin_memory", config.data.pin_memory)
            config.data.prefetch_factor = data_dict.get("prefetch_factor", config.data.prefetch_factor)
            config.data.datasets = data_dict.get("datasets", config.data.datasets)
        
        # Quantization configuration
        if "quantization" in config_dict:
            quant_dict = config_dict["quantization"]
            config.quantization.enable_quantization = quant_dict.get("enable_quantization", config.quantization.enable_quantization)
            config.quantization.quant_method = quant_dict.get("quant_method", config.quantization.quant_method)
            config.quantization.bits = quant_dict.get("bits", config.quantization.bits)
            config.quantization.group_size = quant_dict.get("group_size", config.quantization.group_size)
            config.quantization.symmetric = quant_dict.get("symmetric", config.quantization.symmetric)
            config.quantization.enable_fp8_linear = quant_dict.get("enable_fp8_linear", config.quantization.enable_fp8_linear)

        if "lora" in config_dict:
            lora_dict = config_dict["lora"]
            config.lora.enabled = lora_dict.get("enabled", config.lora.enabled)
            config.lora.r = lora_dict.get("r", config.lora.r)
            config.lora.lora_alpha = lora_dict.get("lora_alpha", config.lora.lora_alpha)
            config.lora.lora_dropout = lora_dict.get("lora_dropout", config.lora.lora_dropout)
            config.lora.target_modules = lora_dict.get("target_modules", config.lora.target_modules)
            config.lora.bias = lora_dict.get("bias", config.lora.bias)
            config.lora.task_type = lora_dict.get("task_type", config.lora.task_type)
        
        # Special mode configuration
        if "special_modes" in config_dict:
            modes_dict = config_dict["special_modes"]
            config.enable_dpo = modes_dict.get("enable_dpo", config.enable_dpo)
            config.enable_sft = modes_dict.get("enable_sft", config.enable_sft)
            config.enable_pref_align = modes_dict.get("enable_pref_align", config.enable_pref_align)
            config.enable_multitask = modes_dict.get("enable_multitask", config.enable_multitask)
        
        # Stage configuration
        if "stage" in config_dict:
            stage_dict = config_dict["stage"]
            stage_value = stage_dict.get("current", "pretrain")
            config.stage = TrainingStage(stage_value)
            config.loss_type = stage_dict.get("loss_type", config.loss_type)
            config.response_only_loss = stage_dict.get("response_only_loss", config.response_only_loss)
            config.beta = stage_dict.get("beta", config.beta)
            config.reference_free = stage_dict.get("reference_free", config.reference_free)
            config.ppo_epochs = stage_dict.get("ppo_epochs", config.ppo_epochs)
            config.clip_range = stage_dict.get("clip_range", config.clip_range)
            config.kl_coef = stage_dict.get("kl_coef", config.kl_coef)
            config.lambda_orpo = stage_dict.get("lambda_orpo", config.lambda_orpo)
            config.packing = stage_dict.get("packing", config.packing)
        elif "stage" in config_dict and isinstance(config_dict["stage"], str):
            config.stage = TrainingStage(config_dict["stage"])
        
        # Advanced operators configuration
        if "advanced_operators" in config_dict:
            adv_dict = config_dict["advanced_operators"]
            config.moe_gradient = adv_dict.get("moe_gradient", config.moe_gradient)
            config.moe = adv_dict.get("moe", config.moe)
            config.kfac = adv_dict.get("kfac", config.kfac)
            config.multitask = adv_dict.get("multitask", config.multitask)
            config.watermark = adv_dict.get("watermark", config.watermark)
            config.modality_scheduler = adv_dict.get("modality_scheduler", config.modality_scheduler)
        
        return config
    
    def save_to_json(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Serializes the configuration to a JSON file with pretty printing
        and UTF-8 encoding for international characters.
        
        Args:
            filepath: Path to the output JSON file
            
        Example:
            >>> config = TrainingConfig(model_name="my-model")
            >>> config.save_to_json("config.json")
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'TrainingConfig':
        """
        Load configuration from JSON file.
        
        Deserializes a configuration from a JSON file. The file must contain
        a valid configuration dictionary.
        
        Args:
            filepath: Path to the input JSON file
            
        Returns:
            TrainingConfig instance loaded from file
            
        Example:
            >>> config = TrainingConfig.load_from_json("config.json")
            >>> print(config.model_name)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def load_from_yaml(cls, filepath: str) -> 'TrainingConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}
        
        # Replace {{VERSION}} placeholder with actual version
        if "version" in config_dict and config_dict["version"] == "{{VERSION}}":
            config_dict["version"] = VERSION
        
        if not isinstance(config_dict, dict):
            config_dict = {}
        return cls.from_dict(config_dict)

    def save_to_yaml(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def apply_cli_overrides(self, args: Dict[str, Any]) -> None:
        seq_len = args.get("seq_len")
        if seq_len is not None:
            try:
                self.data.sequence_length = int(seq_len)
            except Exception:
                pass

        if args.get("no_quant"):
            self.quantization.enable_quantization = False
        if args.get("quant") or args.get("force_quant"):
            self.quantization.enable_quantization = True
        quant_bits = args.get("quant_bits")
        if quant_bits is not None:
            try:
                self.quantization.bits = int(quant_bits)
            except Exception:
                pass

        if args.get("force_lora"):
            self.lora.enabled = True
