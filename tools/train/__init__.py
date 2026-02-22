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
PiscesLx Training Toolkit - Operator-based Flagship Training System

This module provides a comprehensive, operator-based training solution built on
the OPSC (Operator-based Standardized Computing) architecture. It integrates
state-of-the-art training techniques and optimization algorithms into a unified,
modular framework.

Architecture Overview:
    The training system follows a layered architecture:
    
    1. Core Layer (core.py):
       - PiscesLxTrainingOperator: Main training operator with mixed precision,
         gradient checkpointing, and distributed training support
    
    2. Configuration Layer (config.py):
       - TrainingConfig: Centralized configuration management for all training
         parameters including optimizer, scheduler, data, and quantization settings
    
    3. Orchestration Layer (orchestrator.py):
       - TrainingOrchestrator: High-level coordinator that manages the entire
         training lifecycle, component initialization, and workflow execution
    
    4. Pipeline Layer (pipeline.py):
       - TrainingPipelineOperator: End-to-end training pipeline with epoch
         management, evaluation loops, and callback systems
    
    5. Optimization Layer (optimizers.py):
       - GaLoreOptimizerOperator: Memory-efficient low-rank gradient projection
       - LionOptimizerOperator: Evolved sign momentum optimization
       - SophiaOptimizerOperator: Second-order clipped stochastic optimization
       - AdaptiveGradientClippingOperator: Dynamic gradient clipping
       - GradientNoiseInjectionOperator: Noise injection for improved generalization
    
    6. Quantization Layer (quantization.py):
       - QuantizationAwareTrainingOperator: QAT with fake quantization
       - MixedPrecisionTrainingOperator: FP16/BF16 automatic mixed precision
       - KnowledgeDistillationOperator: Teacher-student knowledge transfer
       - GradientAccumulationOperator: Large batch simulation
       - ModelPruningOperator: Structured and unstructured pruning
    
    7. Multi-task Layer (multitask.py):
       - MultiTaskTrainingOperator: Dynamic task weighting and knowledge sharing
       - TaskBalancingOperator: Fair task sampling and resource allocation
       - ContinualLearningOperator: EWC-based catastrophic forgetting prevention
       - TransferLearningOperator: Cross-task knowledge transfer
    
    8. Monitoring Layer (monitoring.py):
       - TrainingMonitorOperator: Real-time metrics tracking and visualization
       - PerformanceProfilerOperator: Detailed performance bottleneck analysis
       - SystemResourceMonitor: GPU/CPU resource monitoring
       - TrainingVisualizer: Training curve generation and dashboard

Key Features:
    - Operator Pattern: All components implement standardized operator interfaces
      for seamless composition and interoperability
    - Mixed Precision: Automatic FP16/BF16 mixed precision training with
      gradient scaling and loss scaling
    - Distributed Training: Native support for DistributedDataParallel (DDP)
      and Fully Sharded Data Parallel (FSDP)
    - Memory Optimization: Gradient checkpointing, activation checkpointing,
      and GaLore low-rank optimization
    - Quantization: Quantization-aware training, post-training quantization,
      and dynamic quantization
    - Multi-task Learning: Dynamic task weighting, task balancing, and
      continual learning with EWC
    - Advanced Optimizers: GaLore, Lion, Sophia, and adaptive gradient clipping
    - Comprehensive Monitoring: Real-time metrics, anomaly detection, and
      performance profiling

Usage Examples:
    Basic Training:
        >>> from tools.train import TrainingConfig, TrainingOrchestrator
        >>> 
        >>> config = TrainingConfig(
        ...     model_name="my-model",
        ...     output_dir="./checkpoints",
        ...     max_steps=100000,
        ...     learning_rate=1e-4
        ... )
        >>> 
        >>> orchestrator = TrainingOrchestrator(config)
        >>> orchestrator.initialize_training(
        ...     model_class=MyModel,
        ...     train_dataloader_factory=train_loader_factory,
        ...     val_dataloader_factory=val_loader_factory
        ... )
        >>> results = orchestrator.start_training(epochs=3)

    Advanced Configuration:
        >>> from tools.train.config import OptimizerConfig, SchedulerConfig
        >>> 
        >>> config = TrainingConfig(
        ...     optimizer=OptimizerConfig(
        ...         name="adamw",
        ...         learning_rate=5e-5,
        ...         use_galore=True,
        ...         galore_rank=128
        ...     ),
        ...     scheduler=SchedulerConfig(
        ...         name="cosine",
        ...         warmup_steps=1000,
        ...         min_lr_ratio=0.1
        ...     ),
        ...     quantization=QuantizationConfig(
        ...         enable_quantization=True,
        ...         quant_method="int8"
        ...     )
        ... )

Integration with OPSC:
    All operators in this module are registered with the OPSC registry and
    implement standardized interfaces (PiscesLxBaseOperator, PiscesLxTransformOperator).
    This enables:
    - Dynamic operator discovery and loading
    - Standardized input/output schemas
    - Automatic operator composition and chaining
    - Unified logging and monitoring

Dependencies:
    - torch >= 2.0.0
    - numpy >= 1.24.0
    - matplotlib >= 3.7.0 (for visualization)
    - PyYAML >= 6.0 (for configuration files)
    - safetensors >= 0.3.0 (for model export)

Version History:
    - 1.0.0: Initial release with core training, optimization, and monitoring
"""

from .core import PiscesLxTrainingOperator
from .config import TrainingConfig
from .orchestrator import PiscesLxTrainOrchestrator
from .watermark import TrainingWatermarkIntegrationOperator, TrainingPipelineWatermarkOperator

from configs.version import VERSION, AUTHOR

__version__ = VERSION
__author__ = AUTHOR

__all__ = [
    "PiscesLxTrainingOperator",
    "TrainingConfig", 
    "PiscesLxTrainOrchestrator",
    "TrainingWatermarkIntegrationOperator",
    "TrainingPipelineWatermarkOperator"
]
