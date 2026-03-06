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
OpenAI-Compatible API Protocol Definitions

This module defines request and response models for OpenAI-compatible API endpoints.
All models use Pydantic for validation and serialization.

Protocol Hierarchy:
    Request Models:
        - ChatCompletionRequest: Chat completion with messages
        - CompletionRequest: Legacy text completion
        - EmbeddingRequest: Text embedding generation
        - ImageGenerationRequest: Image generation from text
        - VideoGenerationRequest: Video generation from text
        - AudioSpeechRequest: Text-to-speech synthesis
        - AgentExecuteRequest: Agent execution request
        - ToolExecuteRequest: Tool execution request
        - RunControlRequest: Run control commands

    Response Models:
        - ChatCompletionResponse: Chat completion response
        - ChatCompletionChunk: Streaming chunk
        - EmbeddingResponse: Embedding vectors
        - ImageGenerationResponse: Generated images
        - VideoGenerationResponse: Generated videos
        - AgentExecuteResponse: Agent execution result
        - ToolExecuteResponse: Tool execution result

Usage:
    >>> from tools.infer.protocol import ChatCompletionRequest
    >>> request = ChatCompletionRequest(
    ...     model="piscesl1-7b",
    ...     messages=[{"role": "user", "content": "Hello"}]
    ... )
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import time
import uuid


class PiscesLxChatMessage(BaseModel):
    """Chat message in OpenAI format."""
    role: str = Field(..., description="Message role: system, user, assistant, or tool")
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="Message content")
    name: Optional[str] = Field(None, description="Name for tool messages")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls from assistant")


class PiscesLxResponseFormat(BaseModel):
    """Response format specification."""
    type: str = Field("text", description="Format type: text or json_object")


class PiscesLxFunctionCall(BaseModel):
    """Function call specification."""
    name: str = Field(..., description="Function name")
    arguments: str = Field(..., description="Function arguments as JSON string")


class PiscesLxTool(BaseModel):
    """Tool definition."""
    type: str = Field("function", description="Tool type")
    function: Dict[str, Any] = Field(..., description="Function definition")


class ChatCompletionRequest(BaseModel):
    """Chat completion request in OpenAI format."""
    model: str = Field("piscesl1-7b", description="Model ID to use")
    messages: List[PiscesLxChatMessage] = Field(..., description="List of chat messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(50, ge=0, description="Top-k sampling parameter")
    n: Optional[int] = Field(1, ge=1, le=10, description="Number of completions to generate")
    stream: Optional[bool] = Field(False, description="Enable streaming response")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    max_tokens: Optional[int] = Field(2048, ge=1, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Token logit biases")
    user: Optional[str] = Field(None, description="User identifier")
    response_format: Optional[PiscesLxResponseFormat] = Field(None, description="Response format")
    tools: Optional[List[PiscesLxTool]] = Field(None, description="Available tools")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice mode")
    repetition_penalty: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="Repetition penalty")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    class Config:
        extra = "allow"


class ChatCompletionChoice(BaseModel):
    """Single choice in chat completion response."""
    index: int = Field(..., description="Choice index")
    message: PiscesLxChatMessage = Field(..., description="Generated message")
    finish_reason: str = Field("stop", description="Reason for finishing")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities")


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(0, description="Tokens in prompt")
    completion_tokens: int = Field(0, description="Tokens in completion")
    total_tokens: int = Field(0, description="Total tokens")


class ChatCompletionResponse(BaseModel):
    """Chat completion response in OpenAI format."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = Field("chat.completion")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field("piscesl1-7b")
    choices: List[ChatCompletionChoice] = Field(default_factory=list)
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint")

    class Config:
        extra = "allow"


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in streaming chunk."""
    role: Optional[str] = Field(None, description="Message role")
    content: Optional[str] = Field(None, description="Content delta")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool call deltas")


class ChatCompletionChunkChoice(BaseModel):
    """Single choice in streaming chunk."""
    index: int = Field(0, description="Choice index")
    delta: ChatCompletionChunkDelta = Field(default_factory=ChatCompletionChunkDelta)
    finish_reason: Optional[str] = Field(None, description="Finish reason")


class ChatCompletionChunk(BaseModel):
    """Streaming chunk for chat completion."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = Field("chat.completion.chunk")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field("piscesl1-7b")
    choices: List[ChatCompletionChunkChoice] = Field(default_factory=list)
    system_fingerprint: Optional[str] = Field(None)


class CompletionRequest(BaseModel):
    """Legacy text completion request."""
    model: str = Field("piscesl1-7b")
    prompt: Union[str, List[str]] = Field(..., description="Text prompt(s)")
    max_tokens: Optional[int] = Field(2048)
    temperature: Optional[float] = Field(0.7)
    top_p: Optional[float] = Field(0.9)
    top_k: Optional[int] = Field(50)
    n: Optional[int] = Field(1)
    stream: Optional[bool] = Field(False)
    stop: Optional[Union[str, List[str]]] = Field(None)
    echo: Optional[bool] = Field(False)
    logprobs: Optional[int] = Field(None)
    presence_penalty: Optional[float] = Field(0.0)
    frequency_penalty: Optional[float] = Field(0.0)
    best_of: Optional[int] = Field(1)
    user: Optional[str] = Field(None)

    class Config:
        extra = "allow"


class CompletionChoice(BaseModel):
    """Single choice in text completion."""
    text: str = Field("", description="Generated text")
    index: int = Field(0)
    finish_reason: str = Field("stop")
    logprobs: Optional[Dict[str, Any]] = Field(None)


class CompletionResponse(BaseModel):
    """Legacy text completion response."""
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = Field("text_completion")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field("piscesl1-7b")
    choices: List[CompletionChoice] = Field(default_factory=list)
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


class EmbeddingRequest(BaseModel):
    """Text embedding request."""
    model: str = Field("piscesl1-7b")
    input: Union[str, List[str]] = Field(..., description="Text to embed")
    encoding_format: Optional[str] = Field("float", description="Encoding format")
    dimensions: Optional[int] = Field(None, description="Output dimensions")
    user: Optional[str] = Field(None)

    class Config:
        extra = "allow"


class EmbeddingData(BaseModel):
    """Single embedding result."""
    object: str = Field("embedding")
    index: int = Field(0)
    embedding: List[float] = Field(default_factory=list)


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    object: str = Field("list")
    data: List[EmbeddingData] = Field(default_factory=list)
    model: str = Field("piscesl1-7b")
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


class ImageGenerationRequest(BaseModel):
    """Image generation request."""
    model: str = Field("piscesl1-7b")
    prompt: str = Field(..., description="Image description")
    n: Optional[int] = Field(1, ge=1, le=10, description="Number of images")
    size: Optional[str] = Field("1024x1024", description="Image size")
    quality: Optional[str] = Field("standard", description="Image quality")
    response_format: Optional[str] = Field("b64_json", description="Response format")
    style: Optional[str] = Field("vivid", description="Image style")
    user: Optional[str] = Field(None)

    class Config:
        extra = "allow"


class ImageData(BaseModel):
    """Generated image data."""
    url: Optional[str] = Field(None, description="Image URL")
    b64_json: Optional[str] = Field(None, description="Base64 encoded image")
    revised_prompt: Optional[str] = Field(None, description="Revised prompt")


class ImageGenerationResponse(BaseModel):
    """Image generation response."""
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageData] = Field(default_factory=list)


class ImageEditRequest(BaseModel):
    """Image edit request with detection and segmentation."""
    model: str = Field("piscesl1-7b")
    image: str = Field(..., description="Base64 encoded image")
    prompt: str = Field(..., description="Edit instruction")
    mask: Optional[str] = Field(None, description="Mask for inpainting")
    n: Optional[int] = Field(1)
    size: Optional[str] = Field("1024x1024")
    response_format: Optional[str] = Field("b64_json")
    operation: Optional[str] = Field("edit", description="Operation: edit, detect, segment")

    class Config:
        extra = "allow"


class VideoGenerationRequest(BaseModel):
    """Video generation request."""
    model: str = Field("piscesl1-7b")
    prompt: str = Field(..., description="Video description")
    duration: Optional[int] = Field(5, ge=1, le=60, description="Duration in seconds")
    fps: Optional[int] = Field(24, ge=1, le=60, description="Frames per second")
    resolution: Optional[str] = Field("1080p", description="Video resolution")
    aspect_ratio: Optional[str] = Field("16:9", description="Aspect ratio")
    n: Optional[int] = Field(1, ge=1, le=4)
    response_format: Optional[str] = Field("b64_json")
    user: Optional[str] = Field(None)

    class Config:
        extra = "allow"


class VideoData(BaseModel):
    """Generated video data."""
    b64_json: Optional[str] = Field(None, description="Base64 encoded video")
    url: Optional[str] = Field(None, description="Video URL")
    duration: Optional[float] = Field(None, description="Video duration")


class VideoGenerationResponse(BaseModel):
    """Video generation response."""
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[VideoData] = Field(default_factory=list)


class AudioSpeechRequest(BaseModel):
    """Text-to-speech request."""
    model: str = Field("piscesl1-7b")
    input: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field("default", description="Voice ID")
    response_format: Optional[str] = Field("mp3", description="Audio format")
    speed: Optional[float] = Field(1.0, ge=0.25, le=4.0, description="Speech speed")
    duration: Optional[float] = Field(None, description="Target duration")

    class Config:
        extra = "allow"


class AudioSpeechResponse(BaseModel):
    """Text-to-speech response."""
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[Dict[str, Any]] = Field(default_factory=list)


class AgentExecuteRequest(BaseModel):
    """Agent execution request."""
    agent: str = Field(..., description="Agent name or ID")
    task: str = Field(..., description="Task description")
    tools: Optional[List[str]] = Field(None, description="Tool names to use")
    context: Optional[Dict[str, Any]] = Field(None, description="Execution context")
    mode: Optional[str] = Field("single", description="Execution mode: single, swarm, orchestrated")
    timeout: Optional[float] = Field(60.0, description="Timeout in seconds")
    stream: Optional[bool] = Field(False, description="Enable streaming")

    class Config:
        extra = "allow"


class AgentExecuteResponse(BaseModel):
    """Agent execution response."""
    agent: str = Field(..., description="Agent name")
    success: bool = Field(True, description="Execution success")
    result: Optional[str] = Field(None, description="Execution result")
    output: Optional[Dict[str, Any]] = Field(None, description="Structured output")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Execution metadata")
    error: Optional[str] = Field(None, description="Error message if failed")


class SwarmExecuteRequest(BaseModel):
    """Swarm execution request."""
    task_type: str = Field("general", description="Task type")
    description: str = Field(..., description="Task description")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data")
    topology: Optional[str] = Field("hierarchical", description="Swarm topology")
    max_agents: Optional[int] = Field(10, description="Maximum agents")
    timeout: Optional[float] = Field(300.0, description="Timeout in seconds")

    class Config:
        extra = "allow"


class SwarmExecuteResponse(BaseModel):
    """Swarm execution response."""
    task_id: str = Field(..., description="Task ID")
    status: str = Field("submitted", description="Task status")
    topology: Optional[str] = Field(None, description="Topology used")
    agents_involved: Optional[List[str]] = Field(None, description="Agents involved")


class ToolExecuteRequest(BaseModel):
    """Tool execution request."""
    tool_name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    timeout: Optional[float] = Field(30.0, description="Timeout in seconds")
    session_id: Optional[str] = Field(None, description="Session ID for stateful tools")

    class Config:
        extra = "allow"


class ToolExecuteResponse(BaseModel):
    """Tool execution response."""
    tool_name: str = Field(..., description="Tool name")
    success: bool = Field(True, description="Execution success")
    result: Optional[Any] = Field(None, description="Tool result")
    error: Optional[str] = Field(None, description="Error message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Execution metadata")


class ToolInfo(BaseModel):
    """Tool information."""
    name: str = Field(..., description="Tool name")
    description: str = Field("", description="Tool description")
    category: Optional[str] = Field(None, description="Tool category")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters schema")


class ToolListResponse(BaseModel):
    """Tool list response."""
    tools: List[ToolInfo] = Field(default_factory=list)
    total: int = Field(0, description="Total tools")


class ToolSearchRequest(BaseModel):
    """Tool search request."""
    query: str = Field(..., description="Search query")
    category: Optional[str] = Field(None, description="Filter by category")
    limit: Optional[int] = Field(10, description="Maximum results")


class RunInfo(BaseModel):
    """Run information."""
    run_id: str = Field(..., description="Run ID")
    run_dir: Optional[str] = Field(None, description="Run directory")
    status: Optional[str] = Field(None, description="Run status")
    phase: Optional[str] = Field(None, description="Run phase")
    created_at: Optional[str] = Field(None, description="Creation time")
    updated_at: Optional[str] = Field(None, description="Update time")
    pid: Optional[int] = Field(None, description="Process ID")


class RunListResponse(BaseModel):
    """Run list response."""
    runs: List[RunInfo] = Field(default_factory=list)
    total: int = Field(0)


class RunControlRequest(BaseModel):
    """Run control request."""
    action: str = Field(..., description="Control action: pause, resume, cancel, kill, reload")
    payload: Optional[Dict[str, Any]] = Field(None, description="Action payload")


class RunControlResponse(BaseModel):
    """Run control response."""
    success: bool = Field(True)
    run_id: str = Field(..., description="Run ID")
    action: str = Field(..., description="Action performed")
    message: Optional[str] = Field(None)


class ModelInfo(BaseModel):
    """Model information."""
    id: str = Field(..., description="Model ID")
    object: str = Field("model", description="Object type")
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = Field("piscesl1", description="Model owner")
    permission: List[Dict[str, Any]] = Field(default_factory=list)
    root: Optional[str] = Field(None)
    parent: Optional[str] = Field(None)


class ModelListResponse(BaseModel):
    """Model list response."""
    object: str = Field("list")
    data: List[ModelInfo] = Field(default_factory=list)


class ServiceStats(BaseModel):
    """Service statistics."""
    uptime_seconds: float = Field(0, description="Service uptime")
    request_count: int = Field(0, description="Total requests")
    success_count: int = Field(0, description="Successful requests")
    error_count: int = Field(0, description="Failed requests")
    qps: float = Field(0, description="Queries per second")
    latency_p50_ms: Optional[float] = Field(None, description="P50 latency")
    latency_p95_ms: Optional[float] = Field(None, description="P95 latency")
    model_size: Optional[str] = Field(None)
    engine_status: Optional[Dict[str, Any]] = Field(None)
    resource: Optional[Dict[str, Any]] = Field(None)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("healthy")
    run_id: Optional[str] = Field(None)
    cpu_percent: Optional[float] = Field(None)
    memory_mb: Optional[float] = Field(None)
    gpu_memory_mb: Optional[float] = Field(None)


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool = Field(False)
    run_id: Optional[str] = Field(None)
    reason: Optional[str] = Field(None)


class ErrorResponse(BaseModel):
    """Error response."""
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    type: Optional[str] = Field(None, description="Error type")
    param: Optional[str] = Field(None, description="Parameter that caused error")
