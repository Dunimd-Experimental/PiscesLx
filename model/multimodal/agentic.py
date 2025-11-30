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

"""High-level orchestration helpers for Ruchbah multimodal agent behaviors.

This module defines :class:`RuchbahAgentic`, the coordination layer that binds
perception, memory, reasoning, and tool execution for the PiscesL1 system. It
acts as the compatibility surface between legacy MCP integrations and newer
PiscesAgent-based flows, managing smart routing, capability registration, and
context-aware action planning.
"""

import uuid
import json
import torch
import numpy as np
from torch import nn
from dataclasses import asdict
from .types import RuchbahMCPMessage
from .reasoner import RuchbahUnifiedReasoner
from .audio import RuchbahAudioEncoder
from .vision import RuchbahVisionEncoder
# Use dms_core logging exclusively
import dms_core
PiscesLxCoreLog = dms_core.log.get_logger
from typing import Dict, Any, Union, List, Callable
from .mcp import (
    PiscesLxCoreMCPProtocol,
    RuchbahMCPToolRegistry,
    PiscesLxCoreMCPTreeSearchReasoner,
)
from .types import (
    RuchbahAgenticState,
    RuchbahAgenticAction,
    RuchbahAgenticObservation,
    RuchbahAgenticMemory,
    RuchbahMCPMessageType,
)

logger = PiscesLxCoreLog("Ruchbah.Core.Agentic", file_path="logs/RuchbahCore.log")

class RuchbahAgentic(nn.Module):
    """Multimodal control surface that orchestrates the Ruchbah agent runtime.

    The controller unifies perception, long-term memory, structured reasoning,
    and tool execution under the MCP (Model Context Protocol) contract. It can
    operate in a standalone configuration with local encoders or attach to an
    upstream Pisces base model for shared infrastructure. Smart routing logic
    selects between native and external tool executors while maintaining rich
    telemetry on decision quality.

    Attributes:
        cfg: Configuration namespace that enumerates model hyper-parameters and
            feature toggles consumed by encoders and the reasoning engine.
        tokenizer: Optional tokenizer leveraged for textual normalization and
            embedding fallback routines.
        agentic_id (str): Stable identifier used across MCP message exchanges.
        memory (RuchbahAgenticMemory): Persistent store that captures
            observations, actions, and reflections for contextual retrieval.
        smart_routing_enabled (bool): Flag toggling intelligent routing between
            native and external tool execution surfaces.
        performance_monitor (Dict[str, Any]): Diagnostics counters tracking
            registration events, resource usage, and uptime metadata.
        execution_stats (Dict[str, int]): Aggregated counts of routing decisions
            partitioned by execution mode.
    """

    def __init__(self, cfg, tokenizer=None, model=None, agentic_id: str = None):
        """Construct the agent controller and wire supporting infrastructure.

        Args:
            cfg: Configuration object describing encoder dimensions, reasoning
                backends, and routing defaults.
            tokenizer: Optional tokenizer used to embed textual observations and
                tool outputs when language models are available.
            model: Optional Pisces base model providing shared encoders and
                reasoning modules. When supplied, the controller links via weak
                references to avoid affecting module registration.
            agentic_id (str, optional): Pre-assigned identifier. A UUID v4 is
                generated when the parameter is omitted.
        """
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        import weakref
        self._model_ref = None  # Weak reference placeholder; resolved post-init.
        self.agentic_id = agentic_id or str(uuid.uuid4())
        
        # Consume shared modules from the provided base model when possible; otherwise
        # instantiate standalone components to maintain feature parity.
        if model is not None:
            # Store a weak reference to avoid registering the model as a submodule.
            self._base_model_ref = weakref.ref(model)
            self._model_ref = self._base_model_ref
        else:
            # Standalone agent mode: construct local reasoning and perception stacks.
            self._base_model_ref = None
            self._model_ref = None
            self._reasoner = RuchbahUnifiedReasoner(cfg)
            self._vision_encoder = RuchbahVisionEncoder(cfg)
            self._audio_encoder = RuchbahAudioEncoder(cfg)
        
        self.tree_reasoner = PiscesLxCoreMCPTreeSearchReasoner(None, tokenizer) if tokenizer else None

        # MCP agent infrastructure
        self.memory = RuchbahAgenticMemory([], [], [])
        
        # Initialize multipath reasoning backend
        from .reasoner.multipath_core import RuchbahMultiPathReasoningEngine
        self.reasoning_engine = RuchbahMultiPathReasoningEngine(cfg)
        
        # Enhanced MCP tool registry integrating the multipath reasoning engine
        self.mcp_tools = RuchbahMCPToolRegistry(
            self.agentic_id, 
            self._handle_mcp_message,
            reasoning_engine=self.reasoning_engine
        )
        
        # Dual-track MCP protocol to coordinate native and external execution
        self.mcp_protocol = PiscesLxCoreMCPProtocol(self.agentic_id, self.mcp_tools)
        
        self.state = RuchbahAgenticState.IDLE
        self.mcp_peers: Dict[str, Dict[str, Any]] = {}
        self.mcp_capabilities: Dict[str, Dict[str, Any]] = {}
        
        # Smart routing status and counters
        self.smart_routing_enabled = True
        self.execution_stats = {
            "native_executions": 0,
            "external_executions": 0,
            "total_executions": 0,
            "routing_decisions": 0
        }
        
        # Performance monitoring state
        from datetime import datetime
        self.performance_monitor = {
            "start_time": datetime.now(),
            "peak_memory_usage": 0,
            "total_tools_registered": 0
        }
        
        # Coordinate marking support flag for vision-assisted operations
        self._coordinate_detection_enabled = True
        
        # Action prediction heads
        self.action_type_head = nn.Linear(cfg.hidden_size, 10)  # Predict 10 action categories
        self.action_param_head = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.confidence_head = nn.Linear(cfg.hidden_size, 1)
        
        # MCP message handlers
        self.mcp_handlers = {
            RuchbahMCPMessageType.OBSERVATION.value: self._handle_observation,
            RuchbahMCPMessageType.ACTION.value: self._handle_action,
            RuchbahMCPMessageType.TOOL_CALL.value: self._handle_tool_call,
            RuchbahMCPMessageType.TOOL_RESULT.value: self._handle_tool_result,
            RuchbahMCPMessageType.CAPABILITY_REGISTER.value: self._handle_capability_register,
            RuchbahMCPMessageType.SYNC_REQUEST.value: self._handle_sync_request,
            RuchbahMCPMessageType.SYNC_RESPONSE.value: self._handle_sync_response,
        }

    @property
    def base_model(self):
        """Access the upstream Pisces base model when linked.

        Returns:
            Optional[nn.Module]: Resolved base model if the weak reference is
            still alive, otherwise ``None``.
        """
        return self._base_model_ref() if self._base_model_ref else None

    @property
    def reasoner(self):
        """Return the active reasoning module bound to the controller.

        Returns:
            RuchbahUnifiedReasoner: Shared base-model reasoner when available, or
            the locally instantiated fallback instance.
        """
        if self._base_model_ref:
            return self._base_model_ref().reasoner
        return self._reasoner

    @property
    def vision_encoder(self):
        """Retrieve the vision encoder responsible for image embeddings.

        Returns:
            RuchbahVisionEncoder: Encoder sourced from the base model when
            possible, falling back to the local implementation.
        """
        if self._base_model_ref:
            return self._base_model_ref().vision
        return self._vision_encoder

    @property
    def audio_encoder(self):
        """Expose the audio encoder for waveform or spectrogram processing.

        Returns:
            RuchbahAudioEncoder: Base-model audio encoder if present, else the
            locally provisioned encoder.
        """
        if self._base_model_ref:
            return self._base_model_ref().audio
        return self._audio_encoder

    async def register_capability(self, name: str, description: str,
                                  parameters: Dict[str, Any], handler: Callable):
        """Expose a capability to MCP peers via the shared registry.

        Args:
            name (str): Public identifier for the capability entry.
            description (str): Human-readable explanation of the capability.
            parameters (Dict[str, Any]): Schema describing accepted arguments.
            handler (Callable): Coroutine handling incoming invocations.

        Returns:
            None
        """
        await self.mcp_tools.register_capability(name, description, parameters, handler)
    
    async def register_native_tool(self, name: str, description: str,
                                  parameters: Dict[str, Any], handler: Callable):
        """Register a native tool that can be executed without routing indirection.

        Args:
            name (str): Tool identifier used by callers.
            description (str): Summary of the tool's behavior.
            parameters (Dict[str, Any]): Expected argument specification.
            handler (Callable): Implementation invoked for native execution.

        Returns:
            Any: Resulting registry metadata produced by the MCP tool registry.
        """
        result = await self.mcp_tools.register_native_tool(name, description, parameters, handler)
        self.performance_monitor["total_tools_registered"] += 1
        return result
    
    async def execute_tool_with_smart_routing(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool with intelligent routing between native and external execution.
        
        Args:
            tool_name (str): Name of the tool to execute.
            **kwargs: Keyword arguments forwarded to the tool handler.
            
        Returns:
            Dict[str, Any]: Execution artifact containing result payload and metadata.
        """
        if not self.smart_routing_enabled:
            # Fallback: execute the tool directly without smart routing heuristics.
            result = await self.mcp_tools.handle_tool_call(tool_name, **kwargs)
            return {"result": result, "execution_mode": "direct", "routed": False}
        
        # Dispatch via dual-track MCP protocol to select native or external execution paths.
        execution_result = await self.mcp_protocol.execute_tool_with_fallback(
            tool_name, kwargs, "tool_executor"
        )
        
        # Update routing statistics to reflect this decision point.
        self.execution_stats["total_executions"] += 1
        self.execution_stats["routing_decisions"] += 1
        
        if execution_result.get("execution_mode") == "native":
            self.execution_stats["native_executions"] += 1
        elif execution_result.get("execution_mode") == "external":
            self.execution_stats["external_executions"] += 1
        
        return execution_result
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Collect execution metrics and derived analytics for monitoring.

        Returns:
            Dict[str, Any]: Aggregated statistics including routing efficiency,
            health indicators, and tool registry counters.
        """
        tool_stats = self.mcp_tools.get_execution_stats()
        
        # Compute routing efficiency metrics for diagnostic purposes.
        total_routing = self.execution_stats["routing_decisions"]
        native_ratio = self.execution_stats["native_executions"] / total_routing if total_routing > 0 else 0
        external_ratio = self.execution_stats["external_executions"] / total_routing if total_routing > 0 else 0
        
        # Derive optimization suggestions based on observed execution profiles.
        performance_suggestions = []
        if native_ratio < 0.3 and self.execution_stats["total_executions"] > 10:
            performance_suggestions.append("Consider converting more tools to native execution for better performance")
        if tool_stats["performance_metrics"]["average_execution_time"] > 1.0:
            performance_suggestions.append("High average execution time detected - review tool implementations")
        if tool_stats["performance_metrics"]["efficiency_rating"] == "low":
            performance_suggestions.append("Low efficiency rating - optimize tool selection and execution paths")
        
        # Measure agent runtime for health reporting.
        uptime = (datetime.now() - self.performance_monitor["start_time"]).total_seconds()
        
        return {
            "agent_stats": self.execution_stats,
            "tool_registry_stats": tool_stats,
            "smart_routing_enabled": self.smart_routing_enabled,
            "total_tools_available": len(self.mcp_tools.tools),
            "native_tools_available": len(self.mcp_tools._native_tools),
            "routing_efficiency": {
                "native_ratio": native_ratio,
                "external_ratio": external_ratio,
                "routing_success_rate": 1.0 if total_routing > 0 else 0.0,
                "optimization_potential": max(0.0, 1.0 - native_ratio)
            },
            "performance_suggestions": performance_suggestions,
            "system_health": {
                "status": "healthy" if tool_stats["performance_metrics"]["performance_score"] > 0.6 else "needs_attention",
                "last_updated": datetime.now().isoformat(),
                "recommendations": performance_suggestions[:2],
                "uptime_seconds": uptime,
                "tools_registration_rate": self.performance_monitor["total_tools_registered"] / max(1, uptime / 60)
            },
            "performance_monitor": self.performance_monitor
        }
    
    def reset_all_statistics(self):
        """Clear accumulated execution statistics and monitoring counters."""
        self.execution_stats = {
            "native_executions": 0,
            "external_executions": 0,
            "total_executions": 0,
            "routing_decisions": 0
        }
        self.mcp_tools.reset_execution_stats()
        
        # Reset performance monitor counters while preserving inception timestamp.
        start_time = self.performance_monitor["start_time"]
        self.performance_monitor = {
            "start_time": start_time,
            "peak_memory_usage": 0,
            "total_tools_registered": 0
        }
        print("[RuchbahAgentic] All execution statistics reset")
    
    def enable_smart_routing(self, enabled: bool = True):
        """Toggle smart routing heuristics for tool execution.

        Args:
            enabled (bool, optional): Desired routing state. Defaults to ``True``.

        Returns:
            None
        """
        self.smart_routing_enabled = enabled
        print(f"[RuchbahAgentic] Smart routing {'enabled' if enabled else 'disabled'}")
        
        # Record state transitions to enable historical analysis of routing toggles.
        if not hasattr(self, 'routing_state_changes'):
            self.routing_state_changes = []
        self.routing_state_changes.append({
            "timestamp": datetime.now().isoformat(),
            "enabled": enabled,
            "total_executions": self.execution_stats["total_executions"]
        })
    
    def get_routing_history(self) -> List[Dict[str, Any]]:
        """Get smart routing state change history.

        Returns:
            List[Dict[str, Any]]: Chronological records of smart routing toggle
            events with timestamps and execution counts.
        """
        return getattr(self, 'routing_state_changes', [])

    async def _handle_mcp_message(self, message: RuchbahMCPMessage) -> RuchbahMCPMessage:
        """Route an MCP message to the corresponding handler coroutine.

        Args:
            message (RuchbahMCPMessage): Envelope containing the message type and payload.

        Returns:
            RuchbahMCPMessage: Response produced by the handler or an error state
            update when the message type is unsupported.
        """
        handler = self.mcp_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        else:
            return PiscesLxCoreMCPProtocol.create_message(
                RuchbahMCPMessageType.STATE_UPDATE,
                self.agent_id,
                {"error": f"Unknown message type: {message.message_type}"}
            )

    async def _handle_observation(self, message: RuchbahMCPMessage) -> RuchbahMCPMessage:
        """Persist an observation and acknowledge processing.

        Args:
            message (RuchbahMCPMessage): Observation payload emitted by a peer.

        Returns:
            RuchbahMCPMessage: Observation acknowledgement carrying embedding metadata.
        """
        observation_data = message.payload
        observation = RuchbahAgenticObservation(
            modality=observation_data["modality"],
            content=observation_data["content"],
            metadata=observation_data.get("metadata", {})
        )
        
        self.memory.add_observation(observation)
        obs_embedding = self.process_observation(observation)
        
        return PiscesLxCoreMCPProtocol.create_message(
            RuchbahMCPMessageType.OBSERVATION,
            self.agent_id,
            {
                "status": "processed",
                "embedding_shape": list(obs_embedding.shape) if torch.is_tensor(obs_embedding) else None,
                "observation_id": str(uuid.uuid4())
            }
        )

    async def _handle_action(self, message: RuchbahMCPMessage) -> RuchbahMCPMessage:
        """Plan an action in response to an MCP action request.

        Args:
            message (RuchbahMCPMessage): Action request populated with contextual data.

        Returns:
            RuchbahMCPMessage: Action response including the serialized plan and agent state.
        """
        action_data = message.payload
        context = {
            "observation": action_data.get("observation"),
            "available_tools": list(self.mcp_capabilities.keys())
        }
        
        action = await self.plan_action(context)
        self.memory.add_action(action)
        
        return PiscesLxCoreMCPProtocol.create_message(
            RuchbahMCPMessageType.ACTION,
            self.agent_id,
            {
                "action": asdict(action),
                "agent_state": self.state.value
            }
        )

    async def _handle_tool_call(self, message: RuchbahMCPMessage) -> RuchbahMCPMessage:
        """Process an MCP tool invocation and report execution metadata.

        Args:
            message (RuchbahMCPMessage): Serialized tool call request including
                tool name, parameters, and optional routing hints.

        Returns:
            RuchbahMCPMessage: Response annotated with success flag, execution
            mode, and serialized result payload.
        """
        tool_data = message.payload
        tool_name = tool_data["tool_name"]
        parameters = tool_data["parameters"]
        execution_mode = tool_data.get("execution_mode", "auto")
        
        # Prefer smart routing when enabled and the requested mode allows indirection.
        if self.smart_routing_enabled and execution_mode in ["auto", "native", "external"]:
            execution_result = await self.execute_tool_with_smart_routing(tool_name, **parameters)
            result = execution_result.get("result", execution_result)
            actual_execution_mode = execution_result.get("execution_mode", "unknown")
            success = execution_result.get("success", True)
        else:
            # Fallback to direct capability execution when routing is bypassed.
            if tool_name in self.mcp_capabilities:
                handler = self.mcp_capabilities[tool_name]["handler"]
                result = await handler(**parameters)
                actual_execution_mode = "traditional"
                success = True
            else:
                result = f"Tool {tool_name} not found"
                actual_execution_mode = "traditional"
                success = False
        
        # Create and return the tool result message with execution metadata
        return PiscesLxCoreMCPProtocol.create_message(
            RuchbahMCPMessageType.TOOL_RESULT,
            self.agent_id,
            {
                "tool_name": tool_name,
                "result": result,
                "success": success,
                "execution_mode": actual_execution_mode,
                "smart_routing_used": self.smart_routing_enabled
            }
        )

    async def _handle_tool_result(self, message: RuchbahMCPMessage) -> RuchbahMCPMessage:
        """Record a tool execution result and signal completion.

        Args:
            message (RuchbahMCPMessage): Tool result payload referencing the originating agent.

        Returns:
            RuchbahMCPMessage: State update confirming the tool result has been assimilated.
        """
        result_data = message.payload
        
        observation = RuchbahAgenticObservation(
            modality="tool_result",
            content=result_data,
            metadata={"source": message.agent_id}
        )
        
        self.memory.add_observation(observation)
        
        return PiscesLxCoreMCPProtocol.create_message(
            RuchbahMCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "tool_result_processed", "result_id": str(uuid.uuid4())}
        )

    async def _handle_capability_register(self, message: RuchbahMCPMessage) -> RuchbahMCPMessage:
        """Store a peer capability advertised over MCP.

        Args:
            message (RuchbahMCPMessage): Capability registration announcement.

        Returns:
            RuchbahMCPMessage: State update acknowledging the capability.
        """
        capability_data = message.payload
        capability_name = capability_data["capability"]
        
        if message.agent_id != self.agent_id:
            self.mcp_capabilities[f"{message.agent_id}.{capability_name}"] = capability_data
        
        return PiscesLxCoreMCPProtocol.create_message(
            RuchbahMCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "capability_registered", "capability": capability_name}
        )

    async def _handle_sync_request(self, message: RuchbahMCPMessage) -> RuchbahMCPMessage:
        """Respond to synchronization requests from peers.

        Args:
            message (RuchbahMCPMessage): Synchronization request specifying desired data.

        Returns:
            RuchbahMCPMessage: Sync response containing capability or state data, or
            an error descriptor when the request type is unsupported.
        """
        sync_type = message.payload.get("type")
        
        if sync_type == "capability_discovery":
            return PiscesLxCoreMCPProtocol.create_message(
                RuchbahMCPMessageType.SYNC_RESPONSE,
                self.agent_id,
                {
                    "type": "capabilities",
                    "capabilities": list(self.mcp_capabilities.keys())
                }
            )
        elif sync_type == "state_sync":
            return PiscesLxCoreMCPProtocol.create_message(
                RuchbahMCPMessageType.SYNC_RESPONSE,
                self.agent_id,
                {
                    "type": "state",
                    "state": self.state.value,
                    "memory_summary": self._summarize_memory()
                }
            )
        
        return PiscesLxCoreMCPProtocol.create_message(
            RuchbahMCPMessageType.SYNC_RESPONSE,
            self.agent_id,
            {"error": "Unknown sync type"}
        )

    async def _handle_sync_response(self, message: RuchbahMCPMessage) -> RuchbahMCPMessage:
        """Consolidate synchronization data received from a peer.

        Args:
            message (RuchbahMCPMessage): Synchronization response carrying peer metadata.

        Returns:
            RuchbahMCPMessage: State update marking the synchronization cycle as complete.
        """
        response_data = message.payload
        
        if response_data.get("type") == "capabilities":
            self.mcp_peers[message.agent_id] = {
                "capabilities": response_data.get("capabilities", [])
            }
        
        return PiscesLxCoreMCPProtocol.create_message(
            RuchbahMCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "sync_completed", "peer_id": message.agent_id}
        )

    def process_observation(self, observation: RuchbahAgenticObservation) -> torch.Tensor:
        """Project an incoming observation into the shared embedding space.

        Args:
            observation (RuchbahAgenticObservation): Structured observation with
                modality label, content payload, and optional metadata.

        Returns:
            torch.Tensor: Embedding tensor compatible with downstream reasoning
            modules. A zero tensor is returned when the modality cannot be
            processed.
        """
        if observation.modality == "text":
            if hasattr(self, 'tokenizer') and self.tokenizer:
                tokens = self.tokenizer.encode(str(observation.content), return_tensors="pt")
                if self.base_model:
                    return self.base_model.embed_tokens(tokens)
                else:
                    return torch.randn(1, tokens.size(1), self.cfg.hidden_size)
            else:
                return torch.zeros(1, 1, self.cfg.hidden_size)
        
        elif observation.modality == "image":
            if isinstance(observation.content, str):  # Image path
                image_tensor = self.vision_encoder.process_image(observation.content)
                if image_tensor is not None:
                    image_tensor = image_tensor.unsqueeze(0)
                    return self.vision_encoder(image_tensor)
            elif torch.is_tensor(observation.content):
                return self.vision_encoder(observation.content)
        
        elif observation.modality == "audio":
            if isinstance(observation.content, str):  # Audio path
                audio_tensor = self.audio_encoder.process_audio(observation.content)
                if audio_tensor is not None:
                    return self.audio_encoder(audio_tensor)
            elif torch.is_tensor(observation.content):
                return self.audio_encoder(observation.content)
        
        elif observation.modality == "tool_result":
            # Convert structured tool output into a textual embedding surrogate.
            import json
            result_str = json.dumps(observation.content)
            if hasattr(self, 'tokenizer') and self.tokenizer:
                tokens = self.tokenizer.encode(result_str, return_tensors="pt")
                if self.base_model:
                    return self.base_model.embed_tokens(tokens)
                else:
                    return torch.randn(1, tokens.size(1), self.cfg.hidden_size)
        
        # Fallback to zero tensor
        return torch.zeros(1, 1, self.cfg.hidden_size)

    async def plan_action(self, context: Dict[str, Any]) -> RuchbahAgenticAction:
        """Generate an agent action using retrieval-augmented reasoning.

        The planner retrieves contextual memories, encodes the query, and invokes
        the configured reasoning backend to derive an action distribution. When a
        linked base model is available, the routine leverages its reasoning head;
        otherwise it falls back to stochastic sampling on the local heads.

        Args:
            context (Dict[str, Any]): Dictionary describing the latest
                observation and declared tool set.

        Returns:
            RuchbahAgenticAction: Structured action containing type, parameters,
            confidence, and an explanatory reasoning trace.
        """
        # Retrieve semantically relevant memories to enrich the reasoning context.
        memory_context = self.memory.get_context_with_retrieval(
            query=str(context), 
            k=5, 
            include_compressed=True
        )
        
        # Encode the textualized context for similarity search against memories.
        query_embedding = self._encode_query(str(context))
        
        # Perform semantic retrieval to isolate the top-k memories by affinity.
        relevant_memories = self.memory.semantic_search(
            query_embedding=query_embedding,
            k=3,
            threshold=0.7
        )
        
        # Extract structured key/value representations for subsequent reasoning steps.
        memory_keys = self._extract_memory_keys(relevant_memories)
        memory_values = self._extract_memory_values(relevant_memories)
        
        # Construct the consolidated reasoning payload consumed by the LLM head.
        enhanced_input = self._prepare_enhanced_reasoning_input(
            context=context,
            memory_context=memory_context,
            memory_keys=memory_keys,
            memory_values=memory_values,
            query_embedding=query_embedding
        )
        
        # Execute multi-step chain-of-thought reasoning to plan the response.
        with torch.no_grad():
            if self.base_model and hasattr(self, 'reasoner'):
                # Use the shared base-model reasoner to incorporate memory context.
                reasoning_output = self.reasoner(
                    enhanced_input,
                    memory_context=memory_context.get("embeddings", None)
                )
                
                # Collect logits from the multi-step thinking heads.
                thinking_logits = reasoning_output.get("thinking_logits", reasoning_output.get("logits"))
                difficulty_logits = reasoning_output.get("difficulty_logits")
                reflection_logits = reasoning_output.get("reflection_logits")
                confidence_logits = reasoning_output.get("confidence_logits")
                
                # Derive action probabilities from the terminal thinking token.
                action_logits = self.action_type_head(thinking_logits[:, -1])
                action_probs = torch.softmax(action_logits, dim=-1)
                action_type_idx = torch.argmax(action_probs, dim=-1).item()
                
                # Blend base and reflection confidences when available.
                base_confidence = torch.sigmoid(self.confidence_head(thinking_logits[:, -1])).item()
                reflection_confidence = torch.sigmoid(reflection_logits[:, -1]).item() if reflection_logits is not None else base_confidence
                confidence = (base_confidence + reflection_confidence) / 2
                
                # Estimate difficulty tier for downstream parameter decoding.
                if difficulty_logits is not None:
                    difficulty = torch.softmax(difficulty_logits[:, -1], dim=-1)
                    difficulty_level = torch.argmax(difficulty, dim=-1).item()
                else:
                    difficulty_level = 2  # Default medium
                
            else:
                # Fallback path: sample logits from random embeddings when no base model exists.
                action_logits = self.action_type_head(torch.randn(1, self.cfg.hidden_size))
                action_probs = torch.softmax(action_logits, dim=-1)
                action_type_idx = torch.argmax(action_probs, dim=-1).item()
                confidence = 0.5
                difficulty_level = 2
            
            # Enumerate supported action types in deterministic order.
            action_types = [
                "respond", "use_tool", "ask_clarification", "reflect", 
                "search_memory", "plan_next", "wait", "verify", 
                "correct_action", "explore", "deep_think", "summarize"
            ]
            action_type = action_types[action_type_idx]
            
            # Generate action parameters conditioned on the reasoning head.
            if self.base_model and hasattr(self, 'reasoner'):
                param_embedding = self.action_param_head(thinking_logits[:, -1])
            else:
                param_embedding = self.action_param_head(torch.randn(1, self.cfg.hidden_size))
            
            action_params = self._decode_enhanced_action_params(
                param_embedding, 
                action_type, 
                difficulty_level,
                confidence
            )
            
            # Assemble a reasoning trace to surface retrieval and decision factors.
            reasoning_trace = self._generate_reasoning_trace(
                context=context,
                memory_summary=memory_context.get("memory_summary", {}),
                action_type=action_type,
                confidence=confidence,
                difficulty_level=difficulty_level,
                relevant_memories=relevant_memories
            )
            
            return RuchbahAgenticAction(
                action_type=action_type,
                parameters=action_params,
                confidence=confidence,
                reasoning=reasoning_trace
            )

    def _prepare_reasoning_input(self, context: Dict[str, Any], memory_context: Dict[str, List]) -> Dict[str, Any]:
        """Assemble the minimal payload required by the baseline reasoner.

        Args:
            context (Dict[str, Any]): Current conversational or environmental context.
            memory_context (Dict[str, List]): Retrieved memory slices and metadata.

        Returns:
            Dict[str, Any]: Dictionary formatted for the legacy reasoner interface.
        """
        return {
            "context": context,
            "memory": memory_context,
            "agent_state": self.state.value
        }

    def _decode_action_params(self, param_embedding: torch.Tensor, action_type: str) -> Dict[str, Any]:
        """Decode legacy action parameters from a latent embedding.

        Args:
            param_embedding (torch.Tensor): Latent representation produced by the
                action parameter head.
            action_type (str): Selected action label guiding decoding heuristics.

        Returns:
            Dict[str, Any]: Dictionary describing the decoded parameters for the
            legacy execution path.
        """
        return {
            "embedding": param_embedding.detach().cpu().numpy().tolist(),
            "decoded_from": action_type,
            "confidence": torch.sigmoid(self.confidence_head(param_embedding)).item()
        }

    def _encode_query(self, query: str) -> torch.Tensor:
        """Encode a textual query into the controller's semantic space.

        Args:
            query (str): Query string representing the current planning context.

        Returns:
            torch.Tensor: Semantic embedding of shape ``[1, hidden_size]``. Falls
            back to a hash-based representation when tokenizer or base model are
            unavailable.
        """
        if hasattr(self, 'tokenizer') and self.tokenizer and self.base_model:
            tokens = self.tokenizer.encode(query, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                embeddings = self.base_model.embed_tokens(tokens)
                # Use mean pooling for query embedding
                query_embedding = embeddings.mean(dim=1)
                return query_embedding
        else:
            # Fallback: use simple hash-based encoding when language models are unavailable.
            import hashlib
            hash_obj = hashlib.md5(query.encode())
            hash_bytes = hash_obj.digest()
            hash_tensor = torch.tensor([int(b) for b in hash_bytes], dtype=torch.float32)
            # Normalize to hidden size
            query_embedding = hash_tensor.unsqueeze(0) / 255.0
            if query_embedding.size(-1) != self.cfg.hidden_size:
                # Pad or truncate to match hidden size
                if query_embedding.size(-1) < self.cfg.hidden_size:
                    padding = torch.zeros(1, self.cfg.hidden_size - query_embedding.size(-1))
                    query_embedding = torch.cat([query_embedding, padding], dim=-1)
                else:
                    query_embedding = query_embedding[:, :self.cfg.hidden_size]
            return query_embedding

    def _extract_memory_keys(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Create coarse key phrases summarizing retrieved memories.

        Args:
            memories (List[Dict[str, Any]]): Retrieved memory entries.

        Returns:
            List[str]: List of phrases used as shorthand keys during reasoning.
        """
        keys = []
        for memory in memories:
            if "content" in memory:
                content = str(memory["content"])
                # Extract deterministic prefix of words for a lightweight key.
                key_phrases = content.split()[:10]
                keys.append(" ".join(key_phrases))
            else:
                keys.append("memory_entry")
        return keys

    def _extract_memory_values(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Surface raw memory values for downstream inspection.

        Args:
            memories (List[Dict[str, Any]]): Retrieved memory entries.

        Returns:
            List[str]: Stringified representation of the memory contents.
        """
        values = []
        for memory in memories:
            if "content" in memory:
                values.append(str(memory["content"]))
            else:
                values.append(str(memory))
        return values

    def _prepare_enhanced_reasoning_input(self, context: Dict[str, Any], 
                                        memory_context: Dict[str, List],
                                        memory_keys: List[str],
                                        memory_values: List[str],
                                        query_embedding: torch.Tensor) -> Dict[str, Any]:
        """Build the enriched reasoning payload consumed by enhanced CoT logic.

        Args:
            context (Dict[str, Any]): Foreground context provided by the caller.
            memory_context (Dict[str, List]): Retrieval response with embeddings and summaries.
            memory_keys (List[str]): Shortened descriptors derived from memories.
            memory_values (List[str]): Detailed memory contents for justification.
            query_embedding (torch.Tensor): Semantic embedding describing the query.

        Returns:
            Dict[str, Any]: Payload containing all features required by multi-path
            reasoning heads.
        """
        return {
            "context": context,
            "memory_context": memory_context,
            "memory_keys": memory_keys,
            "memory_values": memory_values,
            "query_embedding": query_embedding,
            "agent_state": self.state.value,
            "timestamp": str(uuid.uuid4())[:8]  # Lightweight correlation token.
        }

    def _decode_enhanced_action_params(self, param_embedding: torch.Tensor, 
                                     action_type: str, 
                                     difficulty_level: int,
                                     confidence: float) -> Dict[str, Any]:
        """Decode enhanced action parameters with difficulty and confidence signals.

        Args:
            param_embedding (torch.Tensor): Latent vector produced by the action
                parameter head.
            action_type (str): Selected action label guiding specialization.
            difficulty_level (int): Discrete difficulty tier inferred by the reasoner.
            confidence (float): Confidence value in ``[0, 1]``.

        Returns:
            Dict[str, Any]: Structured parameters tailored to the selected action.
        """
        params = {
            "embedding": param_embedding.detach().cpu().numpy().tolist(),
            "action_type": action_type,
            "difficulty_level": difficulty_level,
            "confidence": confidence,
            "timestamp": str(uuid.uuid4())[:8]
        }
        
        # Add action-specific parameters according to the selected action template.
        if action_type == "use_tool":
            params.update({
                "tool_name": "default_tool",
                "tool_parameters": {},
                "retry_count": 0
            })
        elif action_type == "reflect":
            params.update({
                "reflection_type": "self_analysis",
                "focus_areas": ["accuracy", "efficiency", "completeness"]
            })
        elif action_type == "search_memory":
            params.update({
                "search_query": "relevant_memories",
                "max_results": 5,
                "include_compressed": True
            })
        elif action_type == "deep_think":
            params.update({
                "thinking_steps": min(4 + difficulty_level, 8),
                "exploration_depth": min(2 + difficulty_level // 2, 5),
                "validation_required": confidence < 0.7
            })
        
        return params

    def _generate_reasoning_trace(self, context: Dict[str, Any],
                                memory_summary: Dict[str, int],
                                action_type: str,
                                confidence: float,
                                difficulty_level: int,
                                relevant_memories: List[Dict[str, Any]]) -> str:
        """Produce a human-readable reasoning trace for auditing.

        Args:
            context (Dict[str, Any]): Caller-provided action context.
            memory_summary (Dict[str, int]): Aggregated memory counts for telemetry.
            action_type (str): Chosen action label.
            confidence (float): Confidence score emitted by the reasoner.
            difficulty_level (int): Difficulty tier derived from logits.
            relevant_memories (List[Dict[str, Any]]): Memories contributing to the decision.

        Returns:
            str: Pipe-delimited summary capturing context length, memory usage,
            difficulty, and action justification.
        """
        trace_parts = []
        
        # Context analysis
        trace_parts.append(f"Context Analysis: Processing {len(str(context))} characters of input")
        
        # Memory integration
        total_memories = memory_summary.get("total_count", 0)
        retrieved_count = len(relevant_memories)
        trace_parts.append(f"Memory Integration: Retrieved {retrieved_count} relevant memories from {total_memories} total")
        
        # Difficulty assessment
        difficulty_labels = ["very_easy", "easy", "medium", "hard", "very_hard"]
        difficulty_label = difficulty_labels[min(difficulty_level, len(difficulty_labels)-1)]
        trace_parts.append(f"Difficulty Assessment: {difficulty_label} (level {difficulty_level})")
        
        # Action selection reasoning
        trace_parts.append(f"Action Selection: Chose '{action_type}' with {confidence:.2f} confidence")
        
        # Memory influence
        if relevant_memories:
            memory_types = [mem.get("type", "unknown") for mem in relevant_memories]
            type_counts = {}
            for t in memory_types:
                type_counts[t] = type_counts.get(t, 0) + 1
            trace_parts.append(f"Memory Influence: {type_counts}")
        
        # Reasoning summary
        trace_parts.append("Reasoning Complete: Enhanced CoT with semantic memory integration")
        
        return " | ".join(trace_parts)

    def _summarize_memory(self) -> Dict[str, int]:
        """Summarize stored memories for synchronization or diagnostics.

        Returns:
            Dict[str, int]: Counts of observations, actions, and reflections.
        """
        return {
            "observations": len(self.memory.observations),
            "actions": len(self.memory.actions),
            "reflections": len(self.memory.reflections)
        }

    def detect_objects(self, image_input: Union[str, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """Run vision encoder detection pipeline and expose summarized results.

        Args:
            image_input (Union[str, torch.Tensor, np.ndarray]): Image path or tensor.

        Returns:
            Dict[str, Any]: Dictionary with detected objects, image dimensions, and
            diagnostics. Empty structures are returned when detection fails.
        """
        if not self._coordinate_detection_enabled:
            return {"objects": [], "image_size": [0, 0], "num_objects": 0}
        
        try:
            # Process image through vision encoder
            if isinstance(image_input, str):
                image_tensor = self.vision_encoder.process_image(image_input)
                if image_tensor is None:
                    return {"objects": [], "image_size": [0, 0], "num_objects": 0}
                image_tensor = image_tensor.unsqueeze(0)
            elif isinstance(image_input, np.ndarray):
                image_tensor = torch.from_numpy(image_input).float()
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.unsqueeze(0)
            else:
                image_tensor = image_input
            
            # Get detection results from vision encoder
            with torch.no_grad():
                detection_results = self.vision_encoder(image_tensor)
            
            if "detection_results" not in detection_results:
                return {"objects": [], "image_size": [0, 0], "num_objects": 0}
            
            results = detection_results["detection_results"]
            objects = []
            
            # Process bounding boxes and coordinates.
            if "boxes" in results and "labels" in results:
                boxes = results["boxes"].cpu().numpy()
                labels = results["labels"].cpu().numpy()
                scores = results.get("scores", torch.ones(len(boxes))).cpu().numpy()
                
                # Convert patch indices into absolute image coordinates.
                img_coords = self.vision_encoder.convert_patch_to_image_coords(boxes)
                
                for i, (box, label, score) in enumerate(zip(img_coords, labels, scores)):
                    if score > 0.5:  # Confidence threshold
                        x_min, y_min, x_max, y_max = box
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        
                        objects.append({
                            "class": f"class_{label}",
                            "confidence": float(score),
                            "coordinates": [float(x_center), float(y_center)],
                            "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)]
                        })
            
            # Capture approximate image dimensions for coordinate normalization.
            if isinstance(image_input, str):
                from PIL import Image
                with Image.open(image_input) as img:
                    width, height = img.size
            else:
                # Assume square image for tensor inputs
                width = height = 224
            
            return {
                "objects": objects,
                "image_size": [width, height],
                "num_objects": len(objects)
            }
            
        except Exception as e:
            return {"objects": [], "image_size": [0, 0], "num_objects": 0, "error": str(e)}

    def get_coordinates(self, image_input: Union[str, torch.Tensor, np.ndarray], 
                         target_object: str = None) -> List[List[float]]:
        """Extract object centroid coordinates from detection results.

        Args:
            image_input (Union[str, torch.Tensor, np.ndarray]): Image input forwarded to detection.
            target_object (str, optional): Filter criterion for object class names.

        Returns:
            List[List[float]]: Coordinate pairs of detected objects satisfying the filter.
        """
        detection_results = self.detect_objects(image_input)
        
        coordinates = []
        for obj in detection_results.get("objects", []):
            if target_object is None or target_object.lower() in obj["class"].lower():
                coordinates.append(obj["coordinates"])
        
        return coordinates

    def point_to_object(self, image_input: Union[str, torch.Tensor, np.ndarray], 
                       object_description: str) -> Dict[str, Any]:
        """Identify an object aligned with the textual description.

        Args:
            image_input (Union[str, torch.Tensor, np.ndarray]): Image to analyze.
            object_description (str): Free-form description of the target object.

        Returns:
            Dict[str, Any]: Result indicating success, matched object metadata, and
            pointing coordinates when available.
        """
        detection_results = self.detect_objects(image_input)
        
        # Simple matching based on description
        best_match = None
        highest_confidence = 0
        
        for obj in detection_results.get("objects", []):
            # Simple keyword matching for now
            obj_class = obj["class"].lower()
            description_lower = object_description.lower()
            
            if any(keyword in obj_class or obj_class in keyword 
                   for keyword in description_lower.split()):
                if obj["confidence"] > highest_confidence:
                    best_match = obj
                    highest_confidence = obj["confidence"]
        
        if best_match:
            return {
                "found": True,
                "object": best_match,
                "point_coordinates": best_match["coordinates"],
                "message": f"Found {object_description} at coordinates {best_match['coordinates']}"
            }
        else:
            return {
                "found": False,
                "point_coordinates": None,
                "message": f"Could not find {object_description} in the image"
            }

    def enable_coordinate_detection(self, enabled: bool = True):
        """Toggle coordinate detection support in the vision subsystem.

        Args:
            enabled (bool, optional): Whether to enable coordinate detection.
                Defaults to ``True``.
        """
        self._coordinate_detection_enabled = enabled
        if hasattr(self.vision_encoder, 'enable_detection'):
            self.vision_encoder.enable_detection(enabled)
