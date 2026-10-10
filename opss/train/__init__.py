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
Training Operators - Flagship-level Training Optimization Components

This module provides comprehensive training operators implementing state-of-the-art
optimization techniques for large language model training.

Available Operators:
    - PiscesLxKFacOperator: K-FAC natural gradient preconditioning
    - PiscesLxMoEGradientOperator: MoE expert gradient optimization
    - PiscesLxModalitySchedulerOperator: Modality-aware learning rate scheduling
    - PiscesLxMultiTaskOperator: Multi-task learning with uncertainty weighting
    - PiscesLxGaLoreOperator: Gradient low-rank projection optimization

Key Features:
    - Second-order optimization with K-FAC
    - Expert gradient scaling for MoE models (10^4 scale handling)
    - Independent LR schedules for vision/audio/text modalities
    - Automatic task uncertainty-based loss balancing
    - Memory-efficient large model training

Architecture:
    All operators inherit from PiscesLxOperatorInterface and follow the
    OPSC (Operator-based Standardized Component) pattern for consistency
    and composability.

Usage Examples:
    K-FAC Optimization:
    >>> from opss.train import PiscesLxKFacOperator, PiscesLxKFacConfig
    >>> kfac_op = PiscesLxKFacOperator(config)
    >>> precond_grads = kfac_op.execute(gradients, layer_params)
    
    MoE Gradient Optimization:
    >>> from opss.train import PiscesLxMoEGradientOperator
    >>> moe_op = PiscesLxMoEGradientOperator()
    >>> optimized_grads = moe_op.execute(gradients, expert_indices)
    
    Modality-Aware Scheduling:
    >>> from opss.train import PiscesLxModalitySchedulerOperator
    >>> scheduler = PiscesLxModalitySchedulerOperator()
    >>> scheduler.step(modality='vision')

See Also:
    - utils.opsc.interface: Base operator interface
    - tools.train.core: Training engine using these operators
    - opss.optim.galore: Related GaLore optimization operator
"""

import sys
import importlib
from pathlib import Path

from configs.version import VERSION, AUTHOR

_LAZY_SYMBOLS = {
    "PiscesLxKFacOperator": (".kfac", "PiscesLxKFacOperator"),
    "PiscesLxKFacConfig": (".kfac", "PiscesLxKFacConfig"),
    "PiscesLxKFacFacade": (".kfac", "PiscesLxKFacFacade"),
    "PiscesLxMoEGradientOperator": (".moe_gradient", "PiscesLxMoEGradientOperator"),
    "PiscesLxMoEGradientConfig": (".moe_gradient", "PiscesLxMoEGradientConfig"),
    "PiscesLxExpertGradientClipper": (".moe_gradient", "PiscesLxExpertGradientClipper"),
    "PiscesLxModalitySchedulerOperator": (".modality_scheduler", "PiscesLxModalitySchedulerOperator"),
    "PiscesLxModalitySchedulerConfig": (".modality_scheduler", "PiscesLxModalitySchedulerConfig"),
    "PiscesLxModalityType": (".modality_scheduler", "PiscesLxModalityType"),
    "PiscesLxModalitySchedulerFacade": (".modality_scheduler", "PiscesLxModalitySchedulerFacade"),
    "PiscesLxMultiTaskOperator": (".multitask_uncertainty", "PiscesLxMultiTaskOperator"),
    "PiscesLxMultiTaskConfig": (".multitask_uncertainty", "PiscesLxMultiTaskConfig"),
    "PiscesLxTaskUncertaintyWeighting": (".multitask_uncertainty", "PiscesLxTaskUncertaintyWeighting"),
    "PiscesLxMultiTaskFacade": (".multitask_uncertainty", "PiscesLxMultiTaskFacade"),
    "PiscesLxTaskType": (".multitask_uncertainty", "PiscesLxTaskType"),
    "PiscesLxSFTTrainingOperator": (".sft", "PiscesLxSFTTrainingOperator"),
    "PiscesLxSFTTrainingConfig": (".sft", "PiscesLxSFTTrainingConfig"),
    "PiscesLxSFTDataset": (".sft", "PiscesLxSFTDataset"),
    "PiscesLxDPOTrainingOperator": (".dpo", "PiscesLxDPOTrainingOperator"),
    "PiscesLxDPOTrainingConfig": (".dpo", "PiscesLxDPOTrainingConfig"),
    "PiscesLxDPODataset": (".dpo", "PiscesLxDPODataset"),
    "PiscesLxDPOLoggingCallback": (".dpo", "PiscesLxDPOLoggingCallback"),
    "PiscesLxPreferenceAlignmentOperator": (".pref_align", "PiscesLxPreferenceAlignmentOperator"),
    "PiscesLxDPOConfig": (".pref_align", "PiscesLxDPOConfig"),
    "PiscesLxPPOConfig": (".pref_align", "PiscesLxPPOConfig"),
    "PiscesLxKTOConfig": (".pref_align", "PiscesLxKTOConfig"),
    "PiscesLxBCOConfig": (".pref_align", "PiscesLxBCOConfig"),
    "PiscesLxPreferenceDataProcessor": (".pref_align", "PiscesLxPreferenceDataProcessor"),
    "POPSSParallelismType": (".parallel_3d", "POPSSParallelismType"),
    "PiscesLxPipelineSchedule": (".parallel_3d", "PiscesLxPipelineSchedule"),
    "PiscesLxParallel3DConfig": (".parallel_3d", "PiscesLxParallel3DConfig"),
    "PiscesLxParallel3DOperator": (".parallel_3d", "PiscesLxParallel3DOperator"),
    "PiscesLxGRPOOperator": (".grpo", "PiscesLxGRPOOperator"),
    "PiscesLxGRPOConfig": (".grpo", "PiscesLxGRPOConfig"),
    "PiscesLxGRPOTrainer": (".grpo", "PiscesLxGRPOTrainer"),
    "PiscesLxRLVROperator": (".rlvr", "PiscesLxRLVROperator"),
    "PiscesLxRLVRConfig": (".rlvr", "PiscesLxRLVRConfig"),
    "PiscesLxRLVRDataset": (".rlvr", "PiscesLxRLVRDataset"),
    "PiscesLxRLVRTrainer": (".rlvr", "PiscesLxRLVRTrainer"),
    "PiscesLxRLVRVerifierType": (".rlvr", "PiscesLxRLVRVerifierType"),
    "PiscesLxSchedulerType": (".lr_scheduler", "PiscesLxSchedulerType"),
    "PiscesLxLRSchedulerConfig": (".lr_scheduler", "PiscesLxLRSchedulerConfig"),
    "PiscesLxLRSchedulerOperator": (".lr_scheduler", "PiscesLxLRSchedulerOperator"),
    "PiscesLxCosineWarmupScheduler": (".lr_scheduler", "PiscesLxCosineWarmupScheduler"),
    "PiscesLxLinearWarmupScheduler": (".lr_scheduler", "PiscesLxLinearWarmupScheduler"),
    "PiscesLxInverseSquareRootScheduler": (".lr_scheduler", "PiscesLxInverseSquareRootScheduler"),
    "PiscesLxTeacherProviderType": ("tools.data.distill.offline_teacher_provider", "PiscesLxTeacherProviderType"),
    "PiscesLxTeacherConfig": ("tools.data.distill.offline_teacher_provider", "PiscesLxTeacherConfig"),
    "PiscesLxTeacherProvider": ("tools.data.distill.offline_teacher_provider", "PiscesLxTeacherProvider"),
    "PiscesLxLocalTeacherProvider": ("tools.data.distill.offline_teacher_provider", "PiscesLxLocalTeacherProvider"),
    "PiscesLxServerTeacherProvider": ("tools.data.distill.offline_teacher_provider", "PiscesLxServerTeacherProvider"),
    "PiscesLxRemoteTeacherProvider": ("tools.data.distill.offline_teacher_provider", "PiscesLxRemoteTeacherProvider"),
    "PiscesLxTeacherProviderFactory": ("tools.data.distill.offline_teacher_provider", "PiscesLxTeacherProviderFactory"),
    "PiscesLxDistillationLossConfig": (".distill_loss", "PiscesLxDistillationLossConfig"),
    "PiscesLxLogitsDistillationLoss": (".distill_loss", "PiscesLxLogitsDistillationLoss"),
    "PiscesLxHiddenStateDistillationLoss": (".distill_loss", "PiscesLxHiddenStateDistillationLoss"),
    "PiscesLxAttentionDistillationLoss": (".distill_loss", "PiscesLxAttentionDistillationLoss"),
    "PiscesLxLayerWiseDistillationLoss": (".distill_loss", "PiscesLxLayerWiseDistillationLoss"),
    "PiscesLxContrastiveDistillationLoss": (".distill_loss", "PiscesLxContrastiveDistillationLoss"),
    "PiscesLxDistillationLoss": (".distill_loss", "PiscesLxDistillationLoss"),
    "PiscesLxDistillationConfig": (".distill", "PiscesLxDistillationConfig"),
    "PiscesLxDistillationDataset": (".distill", "PiscesLxDistillationDataset"),
    "PiscesLxDistillationOperator": (".distill", "PiscesLxDistillationOperator"),
    "PiscesLxGrowthType": (".growth", "PiscesLxGrowthType"),
    "PiscesLxModelGrowthConfig": (".growth", "PiscesLxModelGrowthConfig"),
    "PiscesLxOptimalTransportAligner": (".growth", "PiscesLxOptimalTransportAligner"),
    "PiscesLxDepthGrower": (".growth", "PiscesLxDepthGrower"),
    "PiscesLxWidthGrower": (".growth", "PiscesLxWidthGrower"),
    "PiscesLxExpertGrower": (".growth", "PiscesLxExpertGrower"),
    "PiscesLxModelGrowthOperator": (".growth", "PiscesLxModelGrowthOperator"),
    "PiscesLxW2SMode": (".weak_to_strong", "PiscesLxW2SMode"),
    "PiscesLxWeakToStrongConfig": (".weak_to_strong", "PiscesLxWeakToStrongConfig"),
    "PiscesLxWeakLabelGenerator": (".weak_to_strong", "PiscesLxWeakLabelGenerator"),
    "PiscesLxCurriculumScheduler": (".weak_to_strong", "PiscesLxCurriculumScheduler"),
    "PiscesLxSelfCorrection": (".weak_to_strong", "PiscesLxSelfCorrection"),
    "PiscesLxWeakToStrongOperator": (".weak_to_strong", "PiscesLxWeakToStrongOperator"),
    "PiscesLxIterativeAmplification": (".weak_to_strong", "PiscesLxIterativeAmplification"),
    "PiscesLxEvolutionStage": (".evolution_pipeline", "PiscesLxEvolutionStage"),
    "PiscesLxGrowthStage": (".evolution_pipeline", "PiscesLxGrowthStage"),
    "PiscesLxEvolutionConfig": (".evolution_pipeline", "PiscesLxEvolutionConfig"),
    "PiscesLxEvolutionTracker": (".evolution_pipeline", "PiscesLxEvolutionTracker"),
    "PiscesLxEvolutionPipeline": (".evolution_pipeline", "PiscesLxEvolutionPipeline"),
    "PiscesLxMemSepTrainer": (".memsep", "PiscesLxMemSepTrainer"),
    "PiscesLxMemSepTrainingConfig": (".memsep", "PiscesLxMemSepTrainingConfig"),
    "PiscesLxMemSepPhase": (".memsep", "PiscesLxMemSepPhase"),
    "PiscesLxMemSepGateScheduler": (".memsep", "PiscesLxMemSepGateScheduler"),
    "PiscesLxMemoryAlignmentLoss": (".memsep", "PiscesLxMemoryAlignmentLoss"),
    "create_memsep_trainer": (".memsep", "create_memsep_trainer"),
    "PiscesLxSelfPlayConfig": (".self_play", "PiscesLxSelfPlayConfig"),
    "PiscesLxSelfPlayTrainer": (".self_play", "PiscesLxSelfPlayTrainer"),
    "PiscesLxGaLoreOperator": ("opss.optim.galore", "PiscesLxGaLoreOperator"),
    "PiscesLxGaLoreConfig": ("opss.optim.galore", "PiscesLxGaLoreConfig"),
}


def __getattr__(name):
    if name in _LAZY_SYMBOLS:
        submod, attr = _LAZY_SYMBOLS[name]
        module = importlib.import_module(submod, __name__)
        val = getattr(module, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = VERSION
__author__ = AUTHOR

__all__ = [
    "PiscesLxKFacOperator",
    "PiscesLxKFacConfig",
    "PiscesLxKFacFacade",
    "PiscesLxMoEGradientOperator",
    "PiscesLxMoEGradientConfig",
    "PiscesLxExpertGradientClipper",
    "PiscesLxModalitySchedulerOperator",
    "PiscesLxModalitySchedulerConfig",
    "PiscesLxModalityType",
    "PiscesLxModalitySchedulerFacade",
    "PiscesLxMultiTaskOperator",
    "PiscesLxMultiTaskConfig",
    "PiscesLxTaskUncertaintyWeighting",
    "PiscesLxMultiTaskFacade",
    "PiscesLxTaskType",
    "PiscesLxSFTTrainingOperator",
    "PiscesLxSFTTrainingConfig",
    "PiscesLxSFTDataset",
    "PiscesLxDPOTrainingOperator",
    "PiscesLxDPOTrainingConfig",
    "PiscesLxDPODataset",
    "PiscesLxDPOLoggingCallback",
    "PiscesLxPreferenceAlignmentOperator",
    "PiscesLxDPOConfig",
    "PiscesLxPPOConfig",
    "PiscesLxKTOConfig",
    "PiscesLxBCOConfig",
    "PiscesLxPreferenceDataProcessor",
    "POPSSParallelismType",
    "PiscesLxPipelineSchedule",
    "PiscesLxParallel3DConfig",
    "PiscesLxParallel3DOperator",
    "PiscesLxGRPOOperator",
    "PiscesLxGRPOConfig",
    "PiscesLxGRPOTrainer",
    "PiscesLxRLVROperator",
    "PiscesLxRLVRConfig",
    "PiscesLxRLVRDataset",
    "PiscesLxRLVRTrainer",
    "PiscesLxRLVRVerifierType",
    "PiscesLxSchedulerType",
    "PiscesLxLRSchedulerConfig",
    "PiscesLxLRSchedulerOperator",
    "PiscesLxCosineWarmupScheduler",
    "PiscesLxLinearWarmupScheduler",
    "PiscesLxInverseSquareRootScheduler",
    "PiscesLxTeacherProviderType",
    "PiscesLxTeacherConfig",
    "PiscesLxTeacherProvider",
    "PiscesLxLocalTeacherProvider",
    "PiscesLxServerTeacherProvider",
    "PiscesLxRemoteTeacherProvider",
    "PiscesLxTeacherProviderFactory",
    "PiscesLxDistillationLossConfig",
    "PiscesLxLogitsDistillationLoss",
    "PiscesLxHiddenStateDistillationLoss",
    "PiscesLxAttentionDistillationLoss",
    "PiscesLxLayerWiseDistillationLoss",
    "PiscesLxContrastiveDistillationLoss",
    "PiscesLxDistillationLoss",
    "PiscesLxDistillationConfig",
    "PiscesLxDistillationDataset",
    "PiscesLxDistillationOperator",
    "PiscesLxGrowthType",
    "PiscesLxModelGrowthConfig",
    "PiscesLxOptimalTransportAligner",
    "PiscesLxDepthGrower",
    "PiscesLxWidthGrower",
    "PiscesLxExpertGrower",
    "PiscesLxModelGrowthOperator",
    "PiscesLxW2SMode",
    "PiscesLxWeakToStrongConfig",
    "PiscesLxWeakLabelGenerator",
    "PiscesLxCurriculumScheduler",
    "PiscesLxSelfCorrection",
    "PiscesLxWeakToStrongOperator",
    "PiscesLxIterativeAmplification",
    "PiscesLxEvolutionStage",
    "PiscesLxGrowthStage",
    "PiscesLxEvolutionConfig",
    "PiscesLxEvolutionTracker",
    "PiscesLxEvolutionPipeline",
    "PiscesLxMemSepTrainer",
    "PiscesLxMemSepTrainingConfig",
    "PiscesLxMemSepPhase",
    "PiscesLxMemSepGateScheduler",
    "PiscesLxMemoryAlignmentLoss",
    "create_memsep_trainer",
    "PiscesLxSelfPlayConfig",
    "PiscesLxSelfPlayTrainer",
    "PiscesLxGaLoreOperator",
    "PiscesLxGaLoreConfig",

    "PiscesLxSubconsciousTrainer",
    "PiscesLxSubconsciousTrainingConfig",
    "PiscesLxTrainingPhase",
]
