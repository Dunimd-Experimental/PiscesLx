#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import os
import gc
import json
import time
import torch
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple
# Use dms_core logging exclusively
import dms_core
PiscesLxCoreLog = dms_core.log.get_logger
from utils.error import PiscesLxCoreValidationError, PiscesLxCoreIOError, PiscesLxCoreMemoryError

# Define constants locally to avoid circular imports
ERROR = "🔴"
RIGHT = "✅"

logger = PiscesLxCoreLog("PiscesLx.Core.Quantization.Core")

class QuantizationMethod(Enum):
    """Enumeration of available quantization methods."""
    BITSANDBYTES = "bitsandbytes"  # BitsAndBytes quantization method
    DYNAMIC = "dynamic"            # Dynamic quantization method
    STATIC = "static"              # Static quantization method
    GPTQ = "gptq"                  # GPTQ quantization method
    AWQ = "awq"                    # AWQ quantization method
    SQUEEZELLM = "squeezellm"      # SqueezeLLM quantization method
    KV_CACHE = "kv_cache"          # KV cache quantization method
    BITNET_V2 = "bitnet_v2"        # BitNet v2 quantization method

class QuantizationGranularity(Enum):
    """Enumeration of quantization granularity levels."""
    PER_TENSOR = "per_tensor"  # Quantization at the tensor level
    PER_CHANNEL = "per_channel"  # Quantization at the channel level
    PER_GROUP = "per_group"    # Quantization at the group level
    PER_TOKEN = "per_token"    # Quantization at the token level

@dataclass
class QuantizationConfig:
    """Configuration class for model quantization.
    
    Attributes:
        method (QuantizationMethod): The quantization method to use. Defaults to BITSANDBYTES.
        bits (int): The number of bits to use for quantization. Defaults to 8.
        granularity (QuantizationGranularity): The granularity level of quantization. Defaults to PER_CHANNEL.
        group_size (int): The size of each group for group-wise quantization. Defaults to 128.
        symmetric (bool): Whether to use symmetric quantization. Defaults to True.
        calibration_dataset (Optional[str]): Path to the calibration dataset. Defaults to None.
        num_calibration_samples (int): Number of samples to use for calibration. Defaults to 128.
        enable_kv_cache_quant (bool): Whether to enable KV cache quantization. Defaults to False.
        kv_cache_bits (int): The number of bits to use for KV cache quantization. Defaults to 8.
        mixed_precision (bool): Whether to use mixed precision quantization. Defaults to False.
        sensitive_layers (List[str]): List of layer names that are sensitive to quantization. Defaults to None.
        preserve_accuracy_layers (List[str]): List of layer names where accuracy should be preserved. Defaults to None.
    """
    method: QuantizationMethod = QuantizationMethod.BITSANDBYTES
    bits: int = 8
    granularity: QuantizationGranularity = QuantizationGranularity.PER_CHANNEL
    group_size: int = 128
    symmetric: bool = True
    calibration_dataset: Optional[str] = None
    num_calibration_samples: int = 128
    enable_kv_cache_quant: bool = False
    kv_cache_bits: int = 8
    mixed_precision: bool = False
    sensitive_layers: List[str] = None
    preserve_accuracy_layers: List[str] = None
    
    def __post_init__(self):
        """Initialize default empty lists for sensitive_layers and preserve_accuracy_layers if they are None."""
        if self.sensitive_layers is None:
            self.sensitive_layers = []
        if self.preserve_accuracy_layers is None:
            self.preserve_accuracy_layers = []

@dataclass
class QuantizationMetrics:
    """Metrics to evaluate the performance and effects of quantization.
    
    Attributes:
        original_size_mb (float): The size of the original model in megabytes. Defaults to 0.0.
        quantized_size_mb (float): The size of the quantized model in megabytes. Defaults to 0.0.
        compression_ratio (float): The compression ratio of the model after quantization. Defaults to 1.0.
        accuracy_drop (float): The drop in model accuracy after quantization. Defaults to 0.0.
        inference_speedup (float): The speedup of model inference after quantization. Defaults to 1.0.
        memory_reduction (float): The reduction in memory usage after quantization. Defaults to 1.0.
        calibration_time_seconds (float): The time taken for calibration in seconds. Defaults to 0.0.
        quantization_time_seconds (float): The time taken for quantization in seconds. Defaults to 0.0.
    """
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 1.0
    accuracy_drop: float = 0.0
    inference_speedup: float = 1.0
    memory_reduction: float = 1.0
    calibration_time_seconds: float = 0.0
    quantization_time_seconds: float = 0.0