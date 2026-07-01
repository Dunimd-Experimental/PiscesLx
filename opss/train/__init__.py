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
    - POPSSKFacOperator: K-FAC natural gradient preconditioning
    - POPSSMoEGradientOperator: MoE expert gradient optimization
    - POPSSModalitySchedulerOperator: Modality-aware learning rate scheduling
    - POPSSMultiTaskOperator: Multi-task learning with uncertainty weighting
    - POPSSGaLoreOperator: Gradient low-rank projection optimization

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
    >>> from opss.train import POPSSKFacOperator, POPSSKFacConfig
    >>> kfac_op = POPSSKFacOperator(config)
    >>> precond_grads = kfac_op.execute(gradients, layer_params)
    
    MoE Gradient Optimization:
    >>> from opss.train import POPSSMoEGradientOperator
    >>> moe_op = POPSSMoEGradientOperator()
    >>> optimized_grads = moe_op.execute(gradients, expert_indices)
    
    Modality-Aware Scheduling:
    >>> from opss.train import POPSSModalitySchedulerOperator
    >>> scheduler = POPSSModalitySchedulerOperator()
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
    "POPSSKFacOperator": (".kfac", "POPSSKFacOperator"),
    "POPSSKFacConfig": (".kfac", "POPSSKFacConfig"),
    "POPSSKFacFacade": (".kfac", "POPSSKFacFacade"),
    "POPSSMoEGradientOperator": (".moe_gradient", "POPSSMoEGradientOperator"),
    "POPSSMoEGradientConfig": (".moe_gradient", "POPSSMoEGradientConfig"),
    "POPSSExpertGradientClipper": (".moe_gradient", "POPSSExpertGradientClipper"),
    "POPSSModalitySchedulerOperator": (".modality_scheduler", "POPSSModalitySchedulerOperator"),
    "POPSSModalitySchedulerConfig": (".modality_scheduler", "POPSSModalitySchedulerConfig"),
    "POPSSModalityType": (".modality_scheduler", "POPSSModalityType"),
    "POPSSModalitySchedulerFacade": (".modality_scheduler", "POPSSModalitySchedulerFacade"),
    "POPSSMultiTaskOperator": (".multitask_uncertainty", "POPSSMultiTaskOperator"),
    "POPSSMultiTaskConfig": (".multitask_uncertainty", "POPSSMultiTaskConfig"),
    "POPSSTaskUncertaintyWeighting": (".multitask_uncertainty", "POPSSTaskUncertaintyWeighting"),
    "POPSSMultiTaskFacade": (".multitask_uncertainty", "POPSSMultiTaskFacade"),
    "POPSSTaskType": (".multitask_uncertainty", "POPSSTaskType"),
    "POPSSSFTTrainingOperator": (".sft", "POPSSSFTTrainingOperator"),
    "POPSSSFTTrainingConfig": (".sft", "POPSSSFTTrainingConfig"),
    "POPSSSFTDataset": (".sft", "POPSSSFTDataset"),
    "POPSSDPOTrainingOperator": (".dpo", "POPSSDPOTrainingOperator"),
    "POPSSDPOTrainingConfig": (".dpo", "POPSSDPOTrainingConfig"),
    "POPSSDPODataset": (".dpo", "POPSSDPODataset"),
    "POPSSDPOLoggingCallback": (".dpo", "POPSSDPOLoggingCallback"),
    "POPSSPreferenceAlignmentOperator": (".pref_align", "POPSSPreferenceAlignmentOperator"),
    "POPSSDPOConfig": (".pref_align", "POPSSDPOConfig"),
    "POPSSPPOConfig": (".pref_align", "POPSSPPOConfig"),
    "POPSSKTOConfig": (".pref_align", "POPSSKTOConfig"),
    "POPSSBCOConfig": (".pref_align", "POPSSBCOConfig"),
    "POPSSPreferenceDataProcessor": (".pref_align", "POPSSPreferenceDataProcessor"),
    "POPSSParallelismType": (".parallel_3d", "POPSSParallelismType"),
    "POPSSPipelineSchedule": (".parallel_3d", "POPSSPipelineSchedule"),
    "POPSSParallel3DConfig": (".parallel_3d", "POPSSParallel3DConfig"),
    "POPSSParallel3DOperator": (".parallel_3d", "POPSSParallel3DOperator"),
    "POPSSGRPOOperator": (".grpo", "POPSSGRPOOperator"),
    "POPSSGRPOConfig": (".grpo", "POPSSGRPOConfig"),
    "POPSSGRPOTrainer": (".grpo", "POPSSGRPOTrainer"),
    "POPSSRLVROperator": (".rlvr", "POPSSRLVROperator"),
    "POPSSRLVRConfig": (".rlvr", "POPSSRLVRConfig"),
    "POPSSRLVRDataset": (".rlvr", "POPSSRLVRDataset"),
    "POPSSRLVRTrainer": (".rlvr", "POPSSRLVRTrainer"),
    "POPSSRLVRVerifierType": (".rlvr", "POPSSRLVRVerifierType"),
    "POPSSSchedulerType": (".lr_scheduler", "POPSSSchedulerType"),
    "POPSSLRSchedulerConfig": (".lr_scheduler", "POPSSLRSchedulerConfig"),
    "POPSSLRSchedulerOperator": (".lr_scheduler", "POPSSLRSchedulerOperator"),
    "POPSSCosineWarmupScheduler": (".lr_scheduler", "POPSSCosineWarmupScheduler"),
    "POPSSLinearWarmupScheduler": (".lr_scheduler", "POPSSLinearWarmupScheduler"),
    "POPSSInverseSquareRootScheduler": (".lr_scheduler", "POPSSInverseSquareRootScheduler"),
    "POPSSTeacherProviderType": (".distill_provider", "POPSSTeacherProviderType"),
    "POPSSTeacherConfig": (".distill_provider", "POPSSTeacherConfig"),
    "POPSSTeacherProvider": (".distill_provider", "POPSSTeacherProvider"),
    "POPSSLocalTeacherProvider": (".distill_provider", "POPSSLocalTeacherProvider"),
    "POPSSServerTeacherProvider": (".distill_provider", "POPSSServerTeacherProvider"),
    "POPSSRemoteTeacherProvider": (".distill_provider", "POPSSRemoteTeacherProvider"),
    "POPSSTeacherProviderFactory": (".distill_provider", "POPSSTeacherProviderFactory"),
    "POPSSDistillationLossConfig": (".distill_loss", "POPSSDistillationLossConfig"),
    "POPSSLogitsDistillationLoss": (".distill_loss", "POPSSLogitsDistillationLoss"),
    "POPSSHiddenStateDistillationLoss": (".distill_loss", "POPSSHiddenStateDistillationLoss"),
    "POPSSAttentionDistillationLoss": (".distill_loss", "POPSSAttentionDistillationLoss"),
    "POPSSLayerWiseDistillationLoss": (".distill_loss", "POPSSLayerWiseDistillationLoss"),
    "POPSSContrastiveDistillationLoss": (".distill_loss", "POPSSContrastiveDistillationLoss"),
    "POPSSDistillationLoss": (".distill_loss", "POPSSDistillationLoss"),
    "POPSSDistillationConfig": (".distill", "POPSSDistillationConfig"),
    "POPSSDistillationDataset": (".distill", "POPSSDistillationDataset"),
    "POPSSDistillationOperator": (".distill", "POPSSDistillationOperator"),
    "POPSSGrowthType": (".growth", "POPSSGrowthType"),
    "POPSSModelGrowthConfig": (".growth", "POPSSModelGrowthConfig"),
    "POPSSOptimalTransportAligner": (".growth", "POPSSOptimalTransportAligner"),
    "POPSSDepthGrower": (".growth", "POPSSDepthGrower"),
    "POPSSWidthGrower": (".growth", "POPSSWidthGrower"),
    "POPSSExpertGrower": (".growth", "POPSSExpertGrower"),
    "POPSSModelGrowthOperator": (".growth", "POPSSModelGrowthOperator"),
    "POPSSW2SMode": (".weak_to_strong", "POPSSW2SMode"),
    "POPSSWeakToStrongConfig": (".weak_to_strong", "POPSSWeakToStrongConfig"),
    "POPSSWeakLabelGenerator": (".weak_to_strong", "POPSSWeakLabelGenerator"),
    "POPSSCurriculumScheduler": (".weak_to_strong", "POPSSCurriculumScheduler"),
    "POPSSSelfCorrection": (".weak_to_strong", "POPSSSelfCorrection"),
    "POPSSWeakToStrongOperator": (".weak_to_strong", "POPSSWeakToStrongOperator"),
    "POPSSIterativeAmplification": (".weak_to_strong", "POPSSIterativeAmplification"),
    "POPSSEvolutionStage": (".evolution_pipeline", "POPSSEvolutionStage"),
    "POPSSGrowthStage": (".evolution_pipeline", "POPSSGrowthStage"),
    "POPSSEvolutionConfig": (".evolution_pipeline", "POPSSEvolutionConfig"),
    "POPSSEvolutionTracker": (".evolution_pipeline", "POPSSEvolutionTracker"),
    "POPSSEvolutionPipeline": (".evolution_pipeline", "POPSSEvolutionPipeline"),
    "POPSSMemSepTrainer": (".memsep", "POPSSMemSepTrainer"),
    "POPSSMemSepTrainingConfig": (".memsep", "POPSSMemSepTrainingConfig"),
    "MemSepPhase": (".memsep", "MemSepPhase"),
    "MemSepGateScheduler": (".memsep", "MemSepGateScheduler"),
    "POPSSMemoryAlignmentLoss": (".memsep", "POPSSMemoryAlignmentLoss"),
    "create_memsep_trainer": (".memsep", "create_memsep_trainer"),
    "POPSSSelfPlayConfig": (".self_play", "POPSSSelfPlayConfig"),
    "POPSSSelfPlayTrainer": (".self_play", "POPSSSelfPlayTrainer"),
    "POPSSGaLoreOperator": ("opss.optim.galore", "POPSSGaLoreOperator"),
    "POPSSGaLoreConfig": ("opss.optim.galore", "POPSSGaLoreConfig"),
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
    "POPSSKFacOperator",
    "POPSSKFacConfig",
    "POPSSKFacFacade",
    "POPSSMoEGradientOperator",
    "POPSSMoEGradientConfig",
    "POPSSExpertGradientClipper",
    "POPSSModalitySchedulerOperator",
    "POPSSModalitySchedulerConfig",
    "POPSSModalityType",
    "POPSSModalitySchedulerFacade",
    "POPSSMultiTaskOperator",
    "POPSSMultiTaskConfig",
    "POPSSTaskUncertaintyWeighting",
    "POPSSMultiTaskFacade",
    "POPSSTaskType",
    "POPSSSFTTrainingOperator",
    "POPSSSFTTrainingConfig",
    "POPSSSFTDataset",
    "POPSSDPOTrainingOperator",
    "POPSSDPOTrainingConfig",
    "POPSSDPODataset",
    "POPSSDPOLoggingCallback",
    "POPSSPreferenceAlignmentOperator",
    "POPSSDPOConfig",
    "POPSSPPOConfig",
    "POPSSKTOConfig",
    "POPSSBCOConfig",
    "POPSSPreferenceDataProcessor",
    "POPSSParallelismType",
    "POPSSPipelineSchedule",
    "POPSSParallel3DConfig",
    "POPSSParallel3DOperator",
    "POPSSGRPOOperator",
    "POPSSGRPOConfig",
    "POPSSGRPOTrainer",
    "POPSSRLVROperator",
    "POPSSRLVRConfig",
    "POPSSRLVRDataset",
    "POPSSRLVRTrainer",
    "POPSSRLVRVerifierType",
    "POPSSSchedulerType",
    "POPSSLRSchedulerConfig",
    "POPSSLRSchedulerOperator",
    "POPSSCosineWarmupScheduler",
    "POPSSLinearWarmupScheduler",
    "POPSSInverseSquareRootScheduler",
    "POPSSTeacherProviderType",
    "POPSSTeacherConfig",
    "POPSSTeacherProvider",
    "POPSSLocalTeacherProvider",
    "POPSSServerTeacherProvider",
    "POPSSRemoteTeacherProvider",
    "POPSSTeacherProviderFactory",
    "POPSSDistillationLossConfig",
    "POPSSLogitsDistillationLoss",
    "POPSSHiddenStateDistillationLoss",
    "POPSSAttentionDistillationLoss",
    "POPSSLayerWiseDistillationLoss",
    "POPSSContrastiveDistillationLoss",
    "POPSSDistillationLoss",
    "POPSSDistillationConfig",
    "POPSSDistillationDataset",
    "POPSSDistillationOperator",
    "POPSSGrowthType",
    "POPSSModelGrowthConfig",
    "POPSSOptimalTransportAligner",
    "POPSSDepthGrower",
    "POPSSWidthGrower",
    "POPSSExpertGrower",
    "POPSSModelGrowthOperator",
    "POPSSW2SMode",
    "POPSSWeakToStrongConfig",
    "POPSSWeakLabelGenerator",
    "POPSSCurriculumScheduler",
    "POPSSSelfCorrection",
    "POPSSWeakToStrongOperator",
    "POPSSIterativeAmplification",
    "POPSSEvolutionStage",
    "POPSSGrowthStage",
    "POPSSEvolutionConfig",
    "POPSSEvolutionTracker",
    "POPSSEvolutionPipeline",
    "POPSSMemSepTrainer",
    "POPSSMemSepTrainingConfig",
    "MemSepPhase",
    "MemSepGateScheduler",
    "POPSSMemoryAlignmentLoss",
    "create_memsep_trainer",
    "POPSSSelfPlayConfig",
    "POPSSSelfPlayTrainer",
    "POPSSGaLoreOperator",
    "POPSSGaLoreConfig",

    "SubconsciousTrainer",
    "SubconsciousTrainingConfig",
    "TrainingPhase",
]
