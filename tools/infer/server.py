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
PiscesLx Backend Inference Service - Flagship OpenAI-Compatible API Server

This module provides a complete backend inference service with OpenAI-compatible
API endpoints, OPSS integration, and multimodal support.

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
        - GET  /v1/runs/{run_id}/logs: Run logs
        - POST /v1/runs/{run_id}/control: Run control

    System Endpoints:
        - GET  /healthz: Health check
        - GET  /readyz: Readiness check
        - GET  /stats: Service statistics

Architecture:
    PiscesLxInferService
    ├── FastAPI Application
    │   ├── Lifespan management
    │   ├── Route registration
    │   └── Error handling
    ├── PiscesLxInferenceEngine
    │   ├── Model loading
    │   ├── Text generation
    │   ├── Streaming support
    │   └── Multimodal operations
    ├── PiscesLxOPSSIntegration
    │   ├── MCP Plaza
    │   ├── Swarm Coordinator
    │   ├── Orchestrator
    │   └── MCP Bridge
    ├── PiscesLxRunIntegration
    │   ├── Run Controller
    │   ├── Resource Monitor
    │   └── Heartbeat Monitor
    └── PiscesLxAgentInterceptor
        ├── Pattern detection
        └── Request routing

Usage:
    >>> from tools.infer.server import PiscesLxInferService
    >>> service = PiscesLxInferService(
    ...     host="127.0.0.1",
    ...     port=8000,
    ...     model_size="7B"
    ... )
    >>> service.serve()
"""

import asyncio
import base64
import json
import time
import uuid
import io
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from .config import (
    InferenceConfig,
    ServiceConfig,
    ModelSpec,
    MODEL_SPECS,
    OPSSConfig,
    RunConfig,
    get_model_spec,
    list_available_models,
)
from .core import PiscesLxInferenceEngine
from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    PiscesLxChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageData,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoData,
    AgentExecuteRequest,
    AgentExecuteResponse,
    SwarmExecuteRequest,
    SwarmExecuteResponse,
    ToolExecuteRequest,
    ToolExecuteResponse,
    ToolInfo,
    ToolListResponse,
    ToolSearchRequest,
    RunInfo,
    RunListResponse,
    RunControlRequest,
    RunControlResponse,
    ModelInfo,
    ModelListResponse,
    ServiceStats,
    HealthResponse,
    ReadyResponse,
    ErrorResponse,
)
from .model_router import PiscesLxModelRouter, PiscesLxRoutingStrategy
from .agent_interceptor import PiscesLxAgentInterceptor, PiscesLxAgentMode
from .opss_integration import PiscesLxOPSSIntegration
from .run_integration import PiscesLxRunIntegration


_LOG = PiscesLxLogger("PiscesLx.Tools.Infer", file_path=get_log_file("PiscesLx.Tools.Infer"), enable_file=True)


class PiscesLxInferService:
    """
    PiscesLx Flagship Backend Inference Service.
    
    This service provides a complete OpenAI-compatible API with:
    - Text generation (sync and streaming)
    - Embeddings
    - Image/Video/Audio generation
    - Agent execution (single, swarm, orchestrated)
    - Tool execution
    - Runtime management
    
    Features:
        - Full OpenAI API compatibility
        - SSE streaming support
        - Agent XML pattern interception
        - OPSS component integration
        - Resource monitoring
        - Health/readiness checks
    
    Example:
        >>> service = PiscesLxInferService(
        ...     host="127.0.0.1",
        ...     port=8000,
        ...     model_size="7B"
        ... )
        >>> service.serve()
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        model_size: str = "7B",
        model_path: Optional[str] = None,
        config: Optional[ServiceConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
    ):
        """
        Initialize the inference service.
        
        Args:
            host: Service host address
            port: Service port
            model_size: Default model size
            model_path: Path to model checkpoint
            config: Complete service configuration
            inference_config: Inference engine configuration
        """
        self.host = host
        self.port = port
        self.model_size = model_size
        self.model_path = model_path
        
        self.config = config or ServiceConfig(
            model_size=model_size,
            host=host,
            port=port,
        )
        
        self.inference_config = inference_config or InferenceConfig()
        if model_path:
            self.inference_config.model.model_path = model_path
        
        self._engine: Optional[PiscesLxInferenceEngine] = None
        self._model_router: Optional[PiscesLxModelRouter] = None
        self._agent_interceptor: Optional[PiscesLxAgentInterceptor] = None
        self._opss: Optional[PiscesLxOPSSIntegration] = None
        self._run: Optional[PiscesLxRunIntegration] = None
        self._app: Optional[FastAPI] = None
        
        self._start_time = time.time()
        self._request_count = 0
        self._success_count = 0
        self._error_count = 0
        self._latencies: List[float] = []
        
        self._LOG = PiscesLxLogger("PiscesLx.Tools.Infer", file_path=get_log_file("PiscesLx.Tools.Infer"), enable_file=True)
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Application lifespan manager for startup and shutdown."""
        self._LOG.info("Starting PiscesLxInferService...")
        
        self._engine = PiscesLxInferenceEngine(self.inference_config)
        
        self._model_router = PiscesLxModelRouter(
            default_size=self.model_size,
            strategy=PiscesLxRoutingStrategy.EXPLICIT,
        )
        
        self._agent_interceptor = PiscesLxAgentInterceptor()
        
        if self.config.opss:
            self._opss = PiscesLxOPSSIntegration(self.config.opss)
            self._opss.initialize()
        
        if self.config.run:
            self._run = PiscesLxRunIntegration(self.config.run)
            self._run.initialize({
                "model_size": self.model_size,
                "host": self.host,
                "port": self.port,
            })
        
        self._LOG.info(f"PiscesLxInferService started on {self.host}:{self.port}")
        
        yield
        
        self._LOG.info("Shutting down PiscesLxInferService...")
        
        if self._run:
            self._run.shutdown()
        
        if self._opss:
            self._opss.shutdown()
        
        self._LOG.info("PiscesLxInferService shutdown complete")
    
    def create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        self._app = FastAPI(
            title="PiscesLx Inference Service",
            description="Flagship OpenAI-Compatible API with OPSS Integration",
            version="1.0.0",
            lifespan=self.lifespan,
        )
        
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._register_routes()
        return self._app
    
    def _register_routes(self):
        """Register all API routes."""
        
        # === Model Endpoints ===
        @self._app.get("/v1/models", response_model=ModelListResponse)
        async def list_models():
            return self._list_models()
        
        # === Chat Endpoints ===
        @self._app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            return await self._chat_completions(request)
        
        @self._app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            return await self._completions(request)
        
        # === Embedding Endpoints ===
        @self._app.post("/v1/embeddings", response_model=EmbeddingResponse)
        async def embeddings(request: EmbeddingRequest):
            return await self._embeddings(request)
        
        # === Image Endpoints ===
        @self._app.post("/v1/images/generations", response_model=ImageGenerationResponse)
        async def image_generations(request: ImageGenerationRequest):
            return await self._image_generations(request)
        
        @self._app.post("/v1/images/edits")
        async def image_edits(request: dict):
            return await self._image_edits(request)
        
        # === Video Endpoints ===
        @self._app.post("/v1/videos/generations", response_model=VideoGenerationResponse)
        async def video_generations(request: VideoGenerationRequest):
            return await self._video_generations(request)
        
        # === Audio Endpoints ===
        @self._app.post("/v1/audio/speech")
        async def audio_speech(request: dict):
            return await self._audio_speech(request)
        
        # === Agent Endpoints ===
        @self._app.post("/v1/agents/execute", response_model=AgentExecuteResponse)
        async def agents_execute(request: AgentExecuteRequest):
            return await self._agents_execute(request)
        
        @self._app.get("/v1/agents/list")
        async def agents_list():
            return await self._agents_list()
        
        @self._app.post("/v1/agents/swarm", response_model=SwarmExecuteResponse)
        async def agents_swarm(request: SwarmExecuteRequest):
            return await self._agents_swarm(request)
        
        # === Tool Endpoints ===
        @self._app.post("/v1/tools/execute", response_model=ToolExecuteResponse)
        async def tools_execute(request: ToolExecuteRequest):
            return await self._tools_execute(request)
        
        @self._app.get("/v1/tools/list", response_model=ToolListResponse)
        async def tools_list(category: Optional[str] = None):
            return await self._tools_list(category)
        
        @self._app.post("/v1/tools/search")
        async def tools_search(request: ToolSearchRequest):
            return await self._tools_search(request)
        
        # === Run Endpoints ===
        @self._app.get("/v1/runs", response_model=RunListResponse)
        async def runs_list():
            return self._runs_list()
        
        @self._app.get("/v1/runs/{run_id}")
        async def run_get(run_id: str):
            return self._run_get(run_id)
        
        @self._app.post("/v1/runs/{run_id}/control", response_model=RunControlResponse)
        async def run_control(run_id: str, request: RunControlRequest):
            return self._run_control(run_id, request)
        
        # === Health Endpoints ===
        @self._app.get("/healthz")
        async def healthz():
            return {"status": "healthy"}
        
        @self._app.get("/readyz")
        async def readyz():
            return self._readyz()
        
        @self._app.get("/stats")
        async def stats():
            return self._stats()
    
    # === Model Endpoints Implementation ===
    
    def _list_models(self) -> ModelListResponse:
        """GET /v1/models"""
        models = []
        for size in list_available_models():
            models.append(ModelInfo(
                id=f"piscesl1-{size.lower()}",
                object="model",
                created=int(self._start_time),
                owned_by="piscesl1",
            ))
        
        return ModelListResponse(data=models)
    
    # === Chat Endpoints Implementation ===
    
    async def _chat_completions(self, request: ChatCompletionRequest):
        """POST /v1/chat/completions"""
        start_time = time.time()
        self._request_count += 1
        
        if self._run:
            self._run.increment_request_count()
            self._run.log_event("chat_request", {"model": request.model})
        
        try:
            prompt = self._extract_prompt(request.messages)
            
            intercept_result = self._agent_interceptor.intercept(prompt)
            
            if intercept_result.mode == PiscesLxAgentMode.SWARM:
                result = await self._handle_swarm_request(request, intercept_result)
                self._record_success(start_time)
                return result
            
            elif intercept_result.mode == PiscesLxAgentMode.ORCHESTRATED:
                result = await self._handle_orchestrate_request(request, intercept_result)
                self._record_success(start_time)
                return result
            
            elif intercept_result.mode == PiscesLxAgentMode.TOOL_CALL:
                result = await self._handle_tool_request(request, intercept_result)
                self._record_success(start_time)
                return result
            
            elif intercept_result.mode == PiscesLxAgentMode.SINGLE:
                result = await self._handle_single_agent_request(request, intercept_result)
                self._record_success(start_time)
                return result
            
            if request.stream:
                return StreamingResponse(
                    self._stream_chat(request, prompt),
                    media_type="text/event-stream"
                )
            
            response_text = self._engine.generate(
                prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_new_tokens=request.max_tokens,
            )
            
            prompt_tokens = self._engine.count_tokens(prompt)
            completion_tokens = self._engine.count_tokens(response_text)
            
            self._record_success(start_time)
            
            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            })
            
        except Exception as e:
            self._record_error(start_time)
            self._LOG.error(f"Chat completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _stream_chat(self, request: ChatCompletionRequest, prompt: str) -> AsyncGenerator[str, None]:
        """SSE streaming for chat completions."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
        try:
            for token in self._engine.generate_stream(
                prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_new_tokens=request.max_tokens,
            ):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            self._LOG.error(f"Streaming failed: {e}")
            error_chunk = {
                "id": completion_id,
                "object": "error",
                "error": {"message": str(e)}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    async def _completions(self, request: CompletionRequest) -> JSONResponse:
        """POST /v1/completions"""
        start_time = time.time()
        self._request_count += 1
        
        try:
            prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
            
            results = []
            for prompt in prompts:
                response_text = self._engine.generate(
                    prompt,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    max_new_tokens=request.max_tokens,
                )
                results.append(response_text)
            
            self._record_success(start_time)
            
            return JSONResponse({
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "text": text,
                    "index": i,
                    "finish_reason": "stop"
                } for i, text in enumerate(results)]
            })
            
        except Exception as e:
            self._record_error(start_time)
            raise HTTPException(status_code=500, detail=str(e))
    
    # === Embedding Endpoints Implementation ===
    
    async def _embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """POST /v1/embeddings"""
        start_time = time.time()
        self._request_count += 1
        
        try:
            inputs = request.input if isinstance(request.input, list) else [request.input]
            
            embeddings = []
            for i, text in enumerate(inputs):
                emb = self._engine.embed(text)
                embeddings.append(EmbeddingData(
                    object="embedding",
                    index=i,
                    embedding=emb.tolist() if hasattr(emb, 'tolist') else list(emb),
                ))
            
            total_tokens = sum(self._engine.count_tokens(t) for t in inputs)
            
            self._record_success(start_time)
            
            return EmbeddingResponse(
                data=embeddings,
                model=request.model,
                usage=ChatCompletionUsage(
                    prompt_tokens=total_tokens,
                    total_tokens=total_tokens,
                )
            )
            
        except Exception as e:
            self._record_error(start_time)
            raise HTTPException(status_code=500, detail=str(e))
    
    # === Image Endpoints Implementation ===
    
    async def _image_generations(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """POST /v1/images/generations"""
        start_time = time.time()
        self._request_count += 1
        
        try:
            images = []
            for i in range(request.n):
                image_tensor = self._engine.generate_image(
                    request.prompt,
                    size=request.size,
                )
                
                image_bytes = self._tensor_to_bytes(image_tensor)
                b64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                images.append(ImageData(b64_json=b64_image))
            
            self._record_success(start_time)
            
            return ImageGenerationResponse(data=images)
            
        except Exception as e:
            self._record_error(start_time)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _image_edits(self, request: dict) -> JSONResponse:
        """POST /v1/images/edits"""
        start_time = time.time()
        self._request_count += 1
        
        try:
            operation = request.get("operation", "edit")
            image_b64 = request.get("image", "")
            prompt = request.get("prompt", "")
            
            image_bytes = base64.b64decode(image_b64)
            
            if operation == "detect":
                results = self._engine.detect_objects(image_bytes)
                self._record_success(start_time)
                return JSONResponse({"detections": results})
            
            elif operation == "segment":
                mask = self._engine.segment_image(image_bytes)
                mask_bytes = self._tensor_to_bytes(mask)
                mask_b64 = base64.b64encode(mask_bytes).decode('utf-8')
                self._record_success(start_time)
                return JSONResponse({"mask": mask_b64})
            
            else:
                self._record_success(start_time)
                return JSONResponse({"message": "Edit operation not implemented"})
            
        except Exception as e:
            self._record_error(start_time)
            raise HTTPException(status_code=500, detail=str(e))
    
    # === Video Endpoints Implementation ===
    
    async def _video_generations(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        """POST /v1/videos/generations"""
        start_time = time.time()
        self._request_count += 1
        
        try:
            video_tensor = self._engine.generate_video(
                request.prompt,
                duration=request.duration,
                fps=request.fps,
            )
            
            video_bytes = self._tensor_to_bytes(video_tensor)
            b64_video = base64.b64encode(video_bytes).decode('utf-8')
            
            self._record_success(start_time)
            
            return VideoGenerationResponse(
                data=[VideoData(b64_json=b64_video, duration=float(request.duration))]
            )
            
        except Exception as e:
            self._record_error(start_time)
            raise HTTPException(status_code=500, detail=str(e))
    
    # === Audio Endpoints Implementation ===
    
    async def _audio_speech(self, request: dict) -> JSONResponse:
        """POST /v1/audio/speech"""
        start_time = time.time()
        self._request_count += 1
        
        try:
            text = request.get("input", "")
            duration = request.get("duration", 5.0)
            
            audio_tensor = self._engine.generate_audio(text, duration=duration)
            
            audio_bytes = self._tensor_to_bytes(audio_tensor)
            b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            
            self._record_success(start_time)
            
            return JSONResponse({
                "created": int(time.time()),
                "data": [{"b64_json": b64_audio}]
            })
            
        except Exception as e:
            self._record_error(start_time)
            raise HTTPException(status_code=500, detail=str(e))
    
    # === Agent Endpoints Implementation ===
    
    async def _agents_execute(self, request: AgentExecuteRequest) -> AgentExecuteResponse:
        """POST /v1/agents/execute"""
        start_time = time.time()
        self._request_count += 1
        
        try:
            result = await self._engine.execute_agent({
                "agent": request.agent,
                "task": request.task,
                "tools": request.tools,
                "context": request.context,
            })
            
            self._record_success(start_time)
            
            return AgentExecuteResponse(
                agent=request.agent,
                success=result.get("success", False),
                result=result.get("output", result.get("result", "")),
                metadata=result.get("metadata", {}),
            )
            
        except Exception as e:
            self._record_error(start_time)
            return AgentExecuteResponse(
                agent=request.agent,
                success=False,
                error=str(e),
            )
    
    async def _agents_list(self) -> JSONResponse:
        """GET /v1/agents/list"""
        agents = []
        
        if self._opss:
            agents = self._opss.list_agents()
        
        return JSONResponse({
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "agent_type": a.agent_type,
                    "name": a.name,
                    "capabilities": a.capabilities,
                    "status": a.status,
                }
                for a in agents
            ]
        })
    
    async def _agents_swarm(self, request: SwarmExecuteRequest) -> SwarmExecuteResponse:
        """POST /v1/agents/swarm"""
        start_time = time.time()
        self._request_count += 1
        
        try:
            task_id = await self._engine.orchestrate_agents(
                task=request.description,
                context={
                    "task_type": request.task_type,
                    "input_data": request.input_data,
                    "topology": request.topology,
                }
            )
            
            self._record_success(start_time)
            
            return SwarmExecuteResponse(
                task_id=task_id.get("task_id", str(uuid.uuid4())),
                status="submitted",
                topology=request.topology,
            )
            
        except Exception as e:
            self._record_error(start_time)
            raise HTTPException(status_code=500, detail=str(e))
    
    # === Tool Endpoints Implementation ===
    
    async def _tools_execute(self, request: ToolExecuteRequest) -> ToolExecuteResponse:
        """POST /v1/tools/execute"""
        start_time = time.time()
        self._request_count += 1
        
        try:
            result = await self._engine.execute_tool(
                request.tool_name,
                request.arguments
            )
            
            self._record_success(start_time)
            
            return ToolExecuteResponse(
                tool_name=request.tool_name,
                success=True,
                result=result,
            )
            
        except Exception as e:
            self._record_error(start_time)
            return ToolExecuteResponse(
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )
    
    async def _tools_list(self, category: Optional[str] = None) -> ToolListResponse:
        """GET /v1/tools/list"""
        tools = self._engine.list_tools(category=category)
        
        return ToolListResponse(
            tools=[
                ToolInfo(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    category=t.get("category"),
                    parameters=t.get("parameters"),
                )
                for t in tools
            ],
            total=len(tools),
        )
    
    async def _tools_search(self, request: ToolSearchRequest) -> JSONResponse:
        """POST /v1/tools/search"""
        tools = self._engine.search_tools(request.query)
        
        return JSONResponse({
            "tools": [
                {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "category": t.get("category"),
                }
                for t in tools[:request.limit]
            ]
        })
    
    # === Run Endpoints Implementation ===
    
    def _runs_list(self) -> RunListResponse:
        """GET /v1/runs"""
        runs = []
        if self._run:
            runs = self._run.list_runs()
        
        return RunListResponse(
            runs=[
                RunInfo(
                    run_id=r.run_id,
                    run_dir=r.run_dir,
                    status=r.status,
                    phase=r.phase,
                    created_at=r.created_at,
                    updated_at=r.updated_at,
                    pid=r.pid,
                )
                for r in runs
            ],
            total=len(runs),
        )
    
    def _run_get(self, run_id: str) -> JSONResponse:
        """GET /v1/runs/{run_id}"""
        if not self._run:
            return JSONResponse({"error": "Run integration not available"}, status_code=503)
        
        run = self._run.get_run(run_id)
        if not run:
            return JSONResponse({"error": f"Run {run_id} not found"}, status_code=404)
        
        return JSONResponse({
            "run_id": run.run_id,
            "run_dir": run.run_dir,
            "status": run.status,
            "phase": run.phase,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
            "pid": run.pid,
        })
    
    def _run_control(self, run_id: str, request: RunControlRequest) -> RunControlResponse:
        """POST /v1/runs/{run_id}/control"""
        if not self._run:
            return RunControlResponse(
                success=False,
                run_id=run_id,
                action=request.action,
                message="Run integration not available",
            )
        
        success = False
        message = ""
        
        if request.action == "pause":
            success = self._run.pause()
            message = "Run paused" if success else "Failed to pause"
        elif request.action == "resume":
            success = self._run.resume()
            message = "Run resumed" if success else "Failed to resume"
        elif request.action == "cancel":
            success = self._run.cancel()
            message = "Run cancelled" if success else "Failed to cancel"
        elif request.action == "kill":
            success = self._run.kill_run(run_id)
            message = "Run killed" if success else "Failed to kill"
        else:
            message = f"Unknown action: {request.action}"
        
        return RunControlResponse(
            success=success,
            run_id=run_id,
            action=request.action,
            message=message,
        )
    
    # === Agent Mode Handlers ===
    
    async def _handle_swarm_request(self, request, intercept_result) -> JSONResponse:
        """Handle swarm mode request."""
        agent_request = intercept_result.agent_requests[0]
        
        result = await self._engine.orchestrate_agents(
            task=agent_request.get("task", ""),
            context={
                "mode": agent_request.get("mode", "hierarchical"),
                "original_content": intercept_result.original_content,
            }
        )
        
        return JSONResponse({
            "mode": "swarm",
            "task_id": result.get("task_id", str(uuid.uuid4())),
            "status": "submitted",
            "result": result.get("output", ""),
        })
    
    async def _handle_orchestrate_request(self, request, intercept_result) -> JSONResponse:
        """Handle orchestrated mode request."""
        agent_request = intercept_result.agent_requests[0]
        
        result = await self._engine.orchestrate_agents(
            task=agent_request.get("task", ""),
            context={
                "strategy": agent_request.get("strategy", "dynamic"),
            }
        )
        
        return JSONResponse({
            "mode": "orchestrated",
            "success": result.get("success", False),
            "output": result.get("output", ""),
            "execution_time": result.get("execution_time", 0),
        })
    
    async def _handle_tool_request(self, request, intercept_result) -> JSONResponse:
        """Handle tool call request."""
        results = []
        
        for tool_call in intercept_result.tool_calls:
            try:
                result = await self._engine.execute_tool(
                    tool_call["tool_name"],
                    tool_call["arguments"]
                )
                results.append({
                    "tool_name": tool_call["tool_name"],
                    "success": True,
                    "result": result,
                })
            except Exception as e:
                results.append({
                    "tool_name": tool_call["tool_name"],
                    "success": False,
                    "error": str(e),
                })
        
        return JSONResponse({
            "mode": "tool_call",
            "results": results,
        })
    
    async def _handle_single_agent_request(self, request, intercept_result) -> JSONResponse:
        """Handle single agent request."""
        results = []
        
        for agent_request in intercept_result.agent_requests:
            result = await self._engine.execute_agent({
                "agent": agent_request.get("agent_name"),
                "task": intercept_result.processed_content,
            })
            results.append(result)
        
        return JSONResponse({
            "mode": "single_agent",
            "results": results,
        })
    
    # === Health Endpoints Implementation ===
    
    def _readyz(self) -> Dict[str, Any]:
        """GET /readyz"""
        if self._engine is None:
            return {"ready": False, "reason": "engine_not_initialized"}
        
        return {"ready": True}
    
    def _stats(self) -> Dict[str, Any]:
        """GET /stats"""
        uptime = time.time() - self._start_time
        
        stats = {
            "uptime_seconds": uptime,
            "request_count": self._request_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "qps": self._request_count / uptime if uptime > 0 else 0,
            "model_size": self.model_size,
        }
        
        if self._latencies:
            sorted_latencies = sorted(self._latencies)
            stats["latency_p50_ms"] = sorted_latencies[len(sorted_latencies) // 2] * 1000
            stats["latency_p95_ms"] = sorted_latencies[int(len(sorted_latencies) * 0.95)] * 1000
        
        if self._engine:
            stats["engine_status"] = self._engine.get_inference_stats()
            stats["model_info"] = self._engine.get_model_info()
        
        if self._run:
            stats["run_status"] = self._run.get_status()
        
        if self._opss:
            stats["opss_status"] = self._opss.get_status()
        
        return stats
    
    # === Utility Methods ===
    
    def _extract_prompt(self, messages: List[PiscesLxChatMessage]) -> str:
        """Extract prompt from messages."""
        parts = []
        for msg in messages:
            role = msg.role
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            parts.append(f"[{role}]: {content}")
        return "\n".join(parts)
    
    def _tensor_to_bytes(self, tensor) -> bytes:
        """Convert tensor to bytes."""
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()
    
    def _record_success(self, start_time: float):
        """Record successful request."""
        latency = time.time() - start_time
        self._success_count += 1
        self._latencies.append(latency)
        
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-500:]
        
        if self._run:
            self._run.record_metric({"latency_s": latency})
    
    def _record_error(self, start_time: float):
        """Record failed request."""
        self._error_count += 1
        
        if self._run:
            self._run.increment_error_count()
    
    def serve(self):
        """Start the FastAPI server."""
        import uvicorn
        
        app = self.create_app()
        uvicorn.run(app, host=self.host, port=self.port)
    
    def serve_async(self):
        """Start the FastAPI server asynchronously."""
        import uvicorn
        import asyncio
        
        app = self.create_app()
        config = uvicorn.Config(app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        
        return server.serve()


class PiscesLxBackendServer:
    """
    PiscesLx Backend Server - CLI Entry Point Wrapper.
    
    This class wraps PiscesLxInferService for use as a CLI command
    in manage.py. It handles argument parsing and service configuration.
    
    Attributes:
        args: Parsed command-line arguments
        service: The underlying inference service
    """
    
    def __init__(self, args):
        """
        Initialize backend server from CLI arguments.
        
        Args:
            args: Parsed argparse namespace containing CLI arguments
        """
        self.args = args
        self._service = None
        
        self.model_size = getattr(args, 'model_size', '7B')
        self.host = getattr(args, 'host', '127.0.0.1')
        self.port = getattr(args, 'port', 8000)
        self.workers = getattr(args, 'workers', 1)
        self.max_concurrency = getattr(args, 'max_concurrency', 2)
        self.request_timeout = getattr(args, 'request_timeout', 120.0)
        self.api_key = getattr(args, 'api_key', None)
        self.cors_origins = getattr(args, 'cors_origins', '*')
        self.log_level = getattr(args, 'log_level', 'INFO')
        self.enable_opss = not getattr(args, 'disable_opss', False)
        self.enable_agent_intercept = not getattr(args, 'disable_agent_intercept', False)
        self.serve_config = getattr(args, 'serve_config', None)
        self.model_path = getattr(args, 'model_path', None)
        
        self._LOG = PiscesLxLogger("PiscesLx.Tools.Infer", file_path=get_log_file("PiscesLx.Tools.Infer"), enable_file=True)
    
    def _build_config(self) -> ServiceConfig:
        """Build service configuration from CLI arguments."""
        config = ServiceConfig(
            model_size=self.model_size,
            host=self.host,
            port=self.port,
        )
        
        config.run.max_concurrency = self.max_concurrency
        config.run.request_timeout_s = self.request_timeout
        
        if self.api_key:
            config.api_key = self.api_key
        
        if self.cors_origins:
            origins = [o.strip() for o in self.cors_origins.split(',')]
            config.cors_origins = origins
        
        if self.enable_opss:
            config.opss.enable_mcp_plaza = True
            config.opss.enable_swarm_coordinator = True
            config.opss.enable_orchestrator = True
            config.opss.enable_mcp_bridge = True
        
        return config
    
    def run(self):
        """Start the backend server."""
        logging.basicConfig(
            level=getattr(logging, self.log_level, logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self._LOG.info(f"Starting PiscesLx Backend Server...")
        self._LOG.info(f"Model Size: {self.model_size}")
        self._LOG.info(f"Host: {self.host}")
        self._LOG.info(f"Port: {self.port}")
        self._LOG.info(f"Workers: {self.workers}")
        self._LOG.info(f"OPSS Enabled: {self.enable_opss}")
        self._LOG.info(f"Agent Intercept: {self.enable_agent_intercept}")
        
        config = self._build_config()
        
        self._service = PiscesLxInferService(
            host=self.host,
            port=self.port,
            model_size=self.model_size,
            model_path=self.model_path,
            config=config,
        )
        
        if self.enable_agent_intercept and self._service._agent_interceptor:
            pass
        
        if self.workers > 1:
            import uvicorn
            uvicorn.run(
                self._service.create_app(),
                host=self.host,
                port=self.port,
                workers=self.workers,
            )
        else:
            self._service.serve()


def create_service(
    host: str = "127.0.0.1",
    port: int = 8000,
    model_size: str = "7B",
    model_path: Optional[str] = None,
    **kwargs
) -> PiscesLxInferService:
    """
    Factory function to create an inference service.
    
    Args:
        host: Service host address
        port: Service port
        model_size: Model size
        model_path: Path to model checkpoint
        **kwargs: Additional configuration
    
    Returns:
        Configured PiscesLxInferService instance
    """
    return PiscesLxInferService(
        host=host,
        port=port,
        model_size=model_size,
        model_path=model_path,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PiscesLx Inference Service")
    parser.add_argument("--host", default="127.0.0.1", help="Service host")
    parser.add_argument("--port", type=int, default=8000, help="Service port")
    parser.add_argument("--model_size", default="7B", help="Model size")
    parser.add_argument("--model_path", default=None, help="Model checkpoint path")
    
    args = parser.parse_args()
    
    service = PiscesLxInferService(
        host=args.host,
        port=args.port,
        model_size=args.model_size,
        model_path=args.model_path,
    )
    
    service.serve()
