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
PiscesLx Inference Toolkit - Operator-based Flagship Inference System

This module provides a comprehensive inference solution based on the OPSC (Operator-based
Standardized Component) architecture. It integrates state-of-the-art inference acceleration
techniques including vLLM, speculative decoding, and various quantization methods.

The inference system is designed with a modular operator architecture, allowing users to
compose different inference strategies by combining various operators. Each operator is
self-contained, well-documented, and follows standardized interfaces.

Key Components:
    - PiscesLxInferenceEngine: Core inference engine with multi-backend support
    - PiscesLxInferService: OpenAI-compatible backend service
    - InferenceConfig: Comprehensive configuration management for inference parameters
    - InferenceOrchestrator: High-level coordinator managing the complete inference pipeline

Architecture:
    The inference system follows a layered architecture:
    1. Operator Layer: Individual inference operators (core, acceleration, quantization)
    2. Pipeline Layer: End-to-end inference pipeline management
    3. Orchestration Layer: High-level coordination and resource management
    4. Service Layer: OpenAI-compatible HTTP API server

API Endpoints:
    Text Endpoints:
        - GET  /v1/models: List available models
        - POST /v1/chat/completions: Chat completion (supports streaming)
        - POST /v1/completions: Legacy text completion
        - POST /v1/embeddings: Text embedding generation

    Multimodal Endpoints:
        - POST /v1/images/generations: Image generation
        - POST /v1/images/edits: Image editing (detection + segmentation)
        - POST /v1/videos/generations: Video generation
        - POST /v1/audio/speech: Text-to-speech synthesis

    Agent Endpoints:
        - POST /v1/agents/execute: Agent execution
        - GET  /v1/agents/list: Agent list
        - POST /v1/agents/swarm: Swarm coordination execution
        - POST /v1/tools/execute: Tool execution
        - GET  /v1/tools/list: Tool list
        - POST /v1/tools/search: Tool search

    Runtime Endpoints:
        - GET  /v1/runs: Run list
        - GET  /v1/runs/{run_id}: Run details
        - POST /v1/runs/{run_id}/control: Run control

Usage Examples:
    Basic inference:
    >>> from tools.infer import PiscesLxInferenceEngine, InferenceConfig
    >>> config = InferenceConfig(model_path="path/to/model")
    >>> engine = PiscesLxInferenceEngine(config)
    >>> result = engine.generate("Hello, world!")
    
    Start backend service:
    >>> from tools.infer import PiscesLxInferService
    >>> service = PiscesLxInferService(host="127.0.0.1", port=8000, model_size="7B")
    >>> service.serve()
    
    Using orchestrator:
    >>> from tools.infer import InferenceOrchestrator
    >>> orchestrator = InferenceOrchestrator(config)
    >>> orchestrator.initialize_inference("path/to/model")
    >>> result = orchestrator.run_inference("Hello, world!")

Dependencies:
    - torch: Core deep learning framework
    - vllm: Optional, for high-performance inference acceleration
    - transformers: For model loading and tokenization
    - fastapi: For HTTP API server
    - uvicorn: ASGI server

Version History:
    - 1.0.0: Initial release with vLLM integration and quantization support
    - 2.0.0: Added OpenAI-compatible backend service with OPSS integration

See Also:
    - ops.infer: Lower-level inference operators
    - utils.opsc: Operator standardization framework
"""

from .core import PiscesLxInferenceEngine
from .config import (
    InferenceConfig,
    MoEConfig,
    ModelSpec,
    OPSSConfig,
    RunConfig,
    ServiceConfig,
    MODEL_SPECS,
    get_model_spec,
    list_available_models,
)
from .server import PiscesLxInferService, PiscesLxBackendServer, create_service
from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    VideoGenerationRequest,
    VideoGenerationResponse,
    AgentExecuteRequest,
    AgentExecuteResponse,
    SwarmExecuteRequest,
    SwarmExecuteResponse,
    ToolExecuteRequest,
    ToolExecuteResponse,
)
from .model_router import PiscesLxModelRouter, PiscesLxRoutingStrategy
from .agent_interceptor import PiscesLxAgentInterceptor, PiscesLxAgentMode
from .opss_integration import PiscesLxOPSSIntegration
from .run_integration import PiscesLxRunIntegration

from .orchestrator import InferenceOrchestrator
from .watermark import InferenceWatermarkIntegrationOperator, InferencePipelineWatermarkOperator

from configs.version import VERSION, AUTHOR

__version__ = VERSION
__author__ = AUTHOR

__all__ = [
    "PiscesLxInferenceEngine",
    "PiscesLxInferService",
    "PiscesLxBackendServer",
    "create_service",
    "InferenceConfig",
    "MoEConfig",
    "ModelSpec",
    "OPSSConfig",
    "RunConfig",
    "ServiceConfig",
    "MODEL_SPECS",
    "get_model_spec",
    "list_available_models",
    "InferenceOrchestrator",
    "InferenceWatermarkIntegrationOperator",
    "InferencePipelineWatermarkOperator",
    "PiscesLxModelRouter",
    "PiscesLxRoutingStrategy",
    "PiscesLxAgentInterceptor",
    "PiscesLxAgentMode",
    "PiscesLxOPSSIntegration",
    "PiscesLxRunIntegration",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "CompletionRequest",
    "CompletionResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
    "AgentExecuteRequest",
    "AgentExecuteResponse",
    "SwarmExecuteRequest",
    "SwarmExecuteResponse",
    "ToolExecuteRequest",
    "ToolExecuteResponse",
]
