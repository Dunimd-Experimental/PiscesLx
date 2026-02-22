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
Inference Operators - Flagship-level Inference Acceleration Components

This module provides comprehensive inference operators implementing state-of-the-art
optimization techniques for large language model inference.

Available Operators:
    - PiscesLxMoERuntimeOperator: Mixture of Experts runtime optimization
    - SpeculativeDecodingOperator: Accelerated generation via speculative decoding
    - VLLMInferenceOperator: High-performance vLLM backend integration
    - SamplingOperator: Advanced sampling strategies

Key Features:
    - Adaptive expert routing with temperature scaling
    - Expert computation caching for token reuse
    - Load balancing across experts
    - Dynamic capacity management
    - Batch expert sharing across requests
    - Prefix KV cache optimization

Architecture:
    All operators inherit from PiscesLxOperatorInterface and follow the
    OPSC (Operator-based Standardized Component) pattern for consistency
    and composability.

Usage Examples:
    MoE Runtime:
    >>> from opss.infer import PiscesLxMoERuntimeOperator, MoERuntimeConfig
    >>> config = MoERuntimeConfig(top_k=2, routing_temp=1.12)
    >>> operator = PiscesLxMoERuntimeOperator(config)
    >>> result = operator.execute({'hidden_states': hidden_states, 'action': 'route'})
    
    Speculative Decoding:
    >>> from opss.infer import SpeculativeDecodingOperator
    >>> spec_op = SpeculativeDecodingOperator()
    >>> spec_op.set_model(model)
    >>> result = spec_op.execute({'input_ids': input_ids, 'gamma': 4})

See Also:
    - utils.opsc.interface: Base operator interface
    - tools.infer.core: Inference engine using these operators
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.version import VERSION, AUTHOR

from .moe_runtime import (
    POPSSMoERuntimeOperator,
    POPSSMoERuntimeConfig,
    POPSSMoERoutingStrategy,
)

from .speculative import (
    POPSSSpeculativeDecodingOperator,
    POPSSSpeculativeConfig,
    POPSSAssistedDecodingOperator,
)

from .vllm import (
    POPSSVLLMInferenceOperator,
    POPSSVLLMConfig,
    POPSSMultiLoRAOperator,
    POPSSPrefixCachingOperator,
    POPSSChunkedPrefillOperator,
)

from .sampling import (
    POPSSSamplingOperator,
    POPSSSamplingConfig,
    POPSSBeamSearchOperator,
)

from .check import (
    POPSSGPUCheckOperator,
)

from .native_ruchbah import (
    POPSSNativeInferenceOperator,
)

from .ring_attention import (
    POPSSRingTopology,
    POPSSRingAttentionConfig,
    POPSSRingAttentionOperator,
    POPSSRingAttentionLayer,
)

from .flash_attention import (
    POPSSFlashAttentionBackend,
    POPSSPrecisionMode,
    POPSSFlashAttention3Config,
    POPSSFlashAttention3Operator,
    POPSSFlashAttention3Layer,
)

from .mamba import (
    POPSSSSMType,
    POPSSMambaConfig,
    POPSSMambaOperator,
    POPSSMambaLayer,
    POPSSMambaMixer,
    POPSSMamba2Operator,
)

from .tpo import (
    POPSSTPOOperator,
    POPSSTPOConfig,
    POPSSTPOAligner,
    POPSSTPOFeedbackType,
    POPSSTPOPreferenceLibrary,
)

__version__ = VERSION
__author__ = AUTHOR

__all__ = [
    "POPSSMoERuntimeOperator",
    "POPSSMoERuntimeConfig",
    "POPSSMoERoutingStrategy",
    "POPSSSpeculativeDecodingOperator",
    "POPSSSpeculativeConfig",
    "POPSSAssistedDecodingOperator",
    "POPSSVLLMInferenceOperator",
    "POPSSVLLMConfig",
    "POPSSMultiLoRAOperator",
    "POPSSPrefixCachingOperator",
    "POPSSChunkedPrefillOperator",
    "POPSSSamplingOperator",
    "POPSSSamplingConfig",
    "POPSSBeamSearchOperator",
    "POPSSGPUCheckOperator",
    "POPSSNativeInferenceOperator",
    "POPSSRingTopology",
    "POPSSRingAttentionConfig",
    "POPSSRingAttentionOperator",
    "POPSSRingAttentionLayer",
    "POPSSFlashAttentionBackend",
    "POPSSPrecisionMode",
    "POPSSFlashAttention3Config",
    "POPSSFlashAttention3Operator",
    "POPSSFlashAttention3Layer",
    "POPSSSSMType",
    "POPSSMambaConfig",
    "POPSSMambaOperator",
    "POPSSMambaLayer",
    "POPSSMambaMixer",
    "POPSSMamba2Operator",
    "POPSSTPOOperator",
    "POPSSTPOConfig",
    "POPSSTPOAligner",
    "POPSSTPOFeedbackType",
    "POPSSTPOPreferenceLibrary",
]
