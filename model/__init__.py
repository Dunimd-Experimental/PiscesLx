#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

from .config import ArcticConfig
from .h2o_attention import ArcticH2OAttention
from .moe_dynamic import ArcticDynamicMoELayer, ArcticExpertChoiceRouter
from .moe import ArcticMoEGate, ArcticStableMoEGate, ArcticMoELayer
from .yarn_rope import ArcticYaRNRotaryEmbedding
from .speculative_decoder import ArcticSpeculativeConfig, ArcticSpeculativeDecoder, ArcticAdaptiveSpeculativeDecoder
from .modeling import (
    ArcticUnifiedCacheManager,
    ArcticRMSNorm,
    ArcticRotaryEmbedding,
    ArcticAttention,
    ArcticTransformerBlock,
    ArcticModel,
)
from .multimodal.reasoner.multipath_core import ArcticMultiPathReasoningEngine
from .multimodal.reasoner.multipath_meta import ArcticMultiPathMetaLearner
from .multimodal.reasoner.enhancer import ArcticMultiModalReasoningEnhancer

# Merge MCP public classes into model package exports - Updated to use utils.mcp
from utils.mcp import PiscesLxCoreMCPPlaza as ArcticOptimizedMCPServer
from utils.mcp.xml_utils import PiscesLxCoreMCPXMLParser as ArcticMCPTranslationLayer
from utils.mcp.types import PiscesLxCoreMCPAgenticCall as ArcticAgenticCall

__all__ = [
    "ArcticConfig",
    "ArcticH2OAttention",
    "ArcticDynamicMoELayer",
    "ArcticExpertChoiceRouter",
    "ArcticMoEGate",
    "ArcticStableMoEGate",
    "ArcticMoELayer",
    "ArcticYaRNRotaryEmbedding",
    "ArcticSpeculativeConfig",
    "ArcticSpeculativeDecoder",
    "ArcticAdaptiveSpeculativeDecoder",
    "ArcticUnifiedCacheManager",
    "ArcticRMSNorm",
    "ArcticRotaryEmbedding",
    "ArcticAttention",
    "ArcticTransformerBlock",
    "ArcticModel",
    "ArcticMultiPathReasoningEngine",
    "ArcticMultiPathMetaLearner",
    "ArcticMultiModalReasoningEnhancer",
    "ArcticOptimizedMCPServer",
    "ArcticMCPTranslationLayer",
    "ArcticAgenticCall",
]