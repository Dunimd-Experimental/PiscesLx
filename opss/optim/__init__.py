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
Optimizers - Flagship-level Optimization Components

This module provides comprehensive optimizer operators implementing state-of-the-art
optimization techniques for large language model training.

Available Operators:
    - POPSSGaLoreOperator: Gradient low-rank projection optimization
    - POPSSROOTOperator: Robust orthogonalized optimizer (Adam stability + Muon speed)
    - POPSSFP4Operator: Native FP4 training for extreme memory efficiency

Key Features:
    - Memory-efficient training with GaLore low-rank projection
    - 2-3x faster convergence with ROOT orthogonalization
    - 50% memory savings over FP8 with FP4 quantization
    - Stochastic rounding for numerical accuracy
    - Spectral norm clipping for stability

Architecture:
    All operators inherit from PiscesLxOperatorInterface and follow the
    OPSC (Operator-based Standardized Component) pattern for consistency
    and composability.

Usage Examples:
    GaLore Optimization:
    >>> from opss.optim import POPSSGaLoreOperator, POPSSGaLoreConfig
    >>> galore = POPSSGaLoreOperator()
    >>> result = galore.execute({"model": model, "gradients": grads})
    
    ROOT Optimizer:
    >>> from opss.optim import POPSSROOTOptimizer, POPSSROOTConfig
    >>> optimizer = POPSSROOTOptimizer(model.parameters(), lr=1e-3)
    
    FP4 Training:
    >>> from opss.optim import POPSSFP4Trainer, POPSSFP4Config
    >>> trainer = POPSSFP4Trainer(model, config)
    >>> trainer.train(dataloader, num_epochs=10)

See Also:
    - utils.opsc.interface: Base operator interface
    - opss.train: Training operators using these optimizers
"""

import sys
from pathlib import Path

from configs.version import VERSION, AUTHOR

from .galore import (
    POPSSGaLoreOperator,
    POPSSGaLoreConfig,
    POPSSGaLoreOptimizerAdapter,
)

from .root import (
    POPSSROOTOperator,
    POPSSROOTConfig,
    POPSSROOTOptimizer,
    POPSSROOTScheduler,
)

from .fp4 import (
    POPSSFP4Operator,
    POPSSFP4Config,
    POPSSFP4Trainer,
    POPSSFP4Quantizer,
    POPSSFP4Linear,
    FP4_E2M1_VALUES,
    FP4_E2M1_MAX,
    FP4_E2M1_MIN,
)

__version__ = VERSION
__author__ = AUTHOR

__all__ = [
    "POPSSGaLoreOperator",
    "POPSSGaLoreConfig",
    "POPSSGaLoreOptimizerAdapter",
    "POPSSROOTOperator",
    "POPSSROOTConfig",
    "POPSSROOTOptimizer",
    "POPSSROOTScheduler",
    "POPSSFP4Operator",
    "POPSSFP4Config",
    "POPSSFP4Trainer",
    "POPSSFP4Quantizer",
    "POPSSFP4Linear",
    "FP4_E2M1_VALUES",
    "FP4_E2M1_MAX",
    "FP4_E2M1_MIN",
]
