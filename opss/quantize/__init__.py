#!/usr/bin/env/python3
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
Quantization Operators Module

Comprehensive quantization algorithms for model compression and inference optimization.
Provides various quantization methods with operator-style interfaces.

Quantization Methods:
    - BitsAndBytes: 4-bit/8-bit quantization with NF4 format
    - Dynamic: Post-training dynamic quantization
    - Static: Post-training static quantization with calibration
    - GPTQ: Gradient-based Post-Training Quantization
    - AWQ: Activation-aware Weight Quantization
    - KV Cache: Key-Value cache quantization for long context
    - BitNet v2: 1.58-bit quantization

Operator Classes:
    POPSSQuantizationEngineOperator: Main operator for model quantization
    POPSSGPTQOperator: GPTQ quantization implementation
    POPSSAWQOperator: AWQ quantization operator
    POPSSSmoothQuantOperator: SmoothQuant operator
    POPSSQuantizationPipelineOperator: Full quantization pipeline

Usage:
    from opss.quantize import POPSSQuantizationPipelineOperator, POPSSQuantizationPipelineConfig

    config = POPSSQuantizationPipelineConfig(method="gptq", bits=4, group_size=128)
    operator = POPSSQuantizationPipelineOperator()
    result = operator.execute({
        "model": model,
        "config": config,
        "calibration_data": calib_data
    })
    quantized_model = result.output["model"]
"""

from .methods import POPSSQuantizationConfig, POPSSQuantizationOperatorFactory
from .calibration import (
    POPSSCalibrationConfig,
    POPSSCalibrationDataLoaderOperator,
    POPSSActivationCollectorOperator,
)
from .advanced import POPSSAdvancedQuantizationConfig, POPSSSensitivityAnalysisOperator, POPSSAdaptiveBitAllocationOperator
from .pipeline import POPSSQuantizationPipelineConfig, POPSSQuantizationPipelineOperator
from .engine import POPSSQuantizationEngineConfig, POPSSQuantizationEngineOperator

__all__ = [
    "POPSSQuantizationConfig",
    "POPSSQuantizationOperatorFactory",
    "POPSSCalibrationConfig",
    "POPSSCalibrationDataLoaderOperator",
    "POPSSActivationCollectorOperator",
    "POPSSAdvancedQuantizationConfig",
    "POPSSSensitivityAnalysisOperator",
    "POPSSAdaptiveBitAllocationOperator",
    "POPSSQuantizationPipelineConfig",
    "POPSSQuantizationPipelineOperator",
    "POPSSQuantizationEngineConfig",
    "POPSSQuantizationEngineOperator",
]
