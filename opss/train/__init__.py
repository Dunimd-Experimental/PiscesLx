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
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.version import VERSION, AUTHOR

from .kfac import (
    POPSSKFacOperator,
    POPSSKFacConfig,
    POPSSKFacFacade,
)

from .moe_gradient import (
    POPSSMoEGradientOperator,
    POPSSMoEGradientConfig,
    POPSSExpertGradientClipper,
)

from .modality_scheduler import (
    POPSSModalitySchedulerOperator,
    POPSSModalitySchedulerConfig,
    POPSSModalityType,
    POPSSModalitySchedulerFacade,
)

from .multitask_uncertainty import (
    POPSSMultiTaskOperator,
    POPSSMultiTaskConfig,
    POPSSTaskUncertaintyWeighting,
    POPSSMultiTaskFacade,
    POPSSTaskType,
)

from .sft import (
    POPSSSFTTrainingOperator,
    POPSSSFTTrainingConfig,
    POPSSSFTDataset,
)

from .dpo import (
    POPSSDPOTrainingOperator,
    POPSSDPOTrainingConfig,
    POPSSDPODataset,
    POPSSDPOLoggingCallback,
)

from .pref_align import (
    POPSSPreferenceAlignmentOperator,
    POPSSDPOConfig,
    POPSSPPOConfig,
    POPSSKTOConfig,
    POPSSBCOConfig,
    POPSSPreferenceDataProcessor,
)

from .parallel_3d import (
    POPSSParallelismType,
    POPSSPipelineSchedule,
    POPSSParallel3DConfig,
    POPSSParallel3DOperator,
)

from .grpo import (
    POPSSGRPOOperator,
    POPSSGRPOConfig,
    POPSSGRPOTrainer,
)

from .rlvr import (
    POPSSRLVROperator,
    POPSSRLVRConfig,
    POPSSRLVRDataset,
    POPSSRLVRTrainer,
    POPSSRLVRVerifierType,
)

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
]
