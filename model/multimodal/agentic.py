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

import uuid
import json
import torch
import numpy as np
from torch import nn
from dataclasses import asdict
from .types import PiscesLxCoreMCPMessage
from .reasoner import ArcticUnifiedReasoner
from .audio import ArcticAudioEncoder
from .vision import ArcticVisionEncoder
from utils.log.core import PiscesLxCoreLog
from typing import Dict, Any, Union, List, Callable
from .mcp import PiscesLxCoreMCPProtocol, ArcticMCPToolRegistry, PiscesLxCoreMCPTreeSearchReasoner
from .types import ArcticAgenticState, PiscesLxCoreAgenticAction, PiscesLxCoreAgenticObservation, ArcticAgenticMemory, PiscesLxCoreMCPMessageType

logger = PiscesLxCoreLog("Arctic.Core.Agentic", file_path="logs/ArcticCore.log")

class ArcticAgentic(nn.Module):
    """
    Represents an agent in the PiscesL1 system with integrated reasoning, perception, and action capabilities.
    Supports multimodal perception, advanced reasoning, tool usage, memory management, and MCP protocol.
    """
    
    def __init__(self, cfg, tokenizer=None, model=None, agentic_id: str = None):
        """
        Initialize the ArcticAgent instance.

        Args:
            cfg: Configuration object containing necessary parameters.
            tokenizer: Tokenizer for text processing, defaults to None.
            model: Base model instance, defaults to None.
            agentic_id: Unique identifier for the agentic, defaults to None (a UUID will be generated).
        """
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        import weakref
        self._model_ref = None  # Placeholder for weak reference to the model
        self.agentic_id = agentic_id or str(uuid.uuid4())
        
        # Use provided model or initialize components for stand - alone mode
        if model is not None:
            # Store a weak reference to avoid registering the model as a sub - module
            self._base_model_ref = weakref.ref(model)
            self._model_ref = self._base_model_ref
        else:
            # Stand - alone agent: no linked PiscesModel to avoid reference cycles
            self._base_model_ref = None
            self._model_ref = None
            self._reasoner = ArcticUnifiedReasoner(cfg)
            self._vision_encoder = ArcticVisionEncoder(cfg)
            self._audio_encoder = ArcticAudioEncoder(cfg)
        
        self.tree_reasoner = PiscesLxCoreMCPTreeSearchReasoner(None, tokenizer) if tokenizer else None

        # MCP Agent infrastructure
        self.memory = ArcticAgenticMemory([], [], [])
        
        # 初始化推理引擎
        from .reasoner.multipath_core import ArcticMultiPathReasoningEngine
        self.reasoning_engine = ArcticMultiPathReasoningEngine(cfg)
        
        # 创建增强的MCP工具注册表（集成8路径推理引擎）
        self.mcp_tools = ArcticMCPToolRegistry(
            self.agentic_id, 
            self._handle_mcp_message,
            reasoning_engine=self.reasoning_engine
        )
        
        # 创建双轨MCP协议
        self.mcp_protocol = PiscesLxCoreMCPProtocol(self.agentic_id, self.mcp_tools)
        
        self.state = ArcticAgenticState.IDLE
        self.mcp_peers: Dict[str, Dict[str, Any]] = {}
        self.mcp_capabilities: Dict[str, Dict[str, Any]] = {}
        
        # 智能路由状态
        self.smart_routing_enabled = True
        self.execution_stats = {
            "native_executions": 0,
            "external_executions": 0,
            "total_executions": 0,
            "routing_decisions": 0
        }
        
        # 性能监控
        from datetime import datetime
        self.performance_monitor = {
            "start_time": datetime.now(),
            "peak_memory_usage": 0,
            "total_tools_registered": 0
        }
        
        # Coordinate marking support
        self._coordinate_detection_enabled = True
        
        # Action prediction heads
        self.action_type_head = nn.Linear(cfg.hidden_size, 10)  # Predict 10 types of actions
        self.action_param_head = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.confidence_head = nn.Linear(cfg.hidden_size, 1)
        
        # MCP message handlers
        self.mcp_handlers = {
            PiscesLxCoreMCPMessageType.OBSERVATION.value: self._handle_observation,
            PiscesLxCoreMCPMessageType.ACTION.value: self._handle_action,
            PiscesLxCoreMCPMessageType.TOOL_CALL.value: self._handle_tool_call,
            PiscesLxCoreMCPMessageType.TOOL_RESULT.value: self._handle_tool_result,
            PiscesLxCoreMCPMessageType.CAPABILITY_REGISTER.value: self._handle_capability_register,
            PiscesLxCoreMCPMessageType.SYNC_REQUEST.value: self._handle_sync_request,
            PiscesLxCoreMCPMessageType.SYNC_RESPONSE.value: self._handle_sync_response,
        }

    @property
    def base_model(self):
        """
        Get the base model instance through the weak reference.

        Returns:
            The base model instance if the weak reference is valid, otherwise None.
        """
        return self._base_model_ref() if self._base_model_ref else None

    @property
    def reasoner(self):
        """
        Get the reasoner instance. Use the reasoner of the base model if available, 
        otherwise use the local reasoner.

        Returns:
            The reasoner instance.
        """
        if self._base_model_ref:
            return self._base_model_ref().reasoner
        return self._reasoner

    @property
    def vision_encoder(self):
        """
        Get the vision encoder instance. Use the vision encoder of the base model if available, 
        otherwise use the local vision encoder.

        Returns:
            The vision encoder instance.
        """
        if self._base_model_ref:
            return self._base_model_ref().vision
        return self._vision_encoder

    @property
    def audio_encoder(self):
        """
        Get the audio encoder instance. Use the audio encoder of the base model if available, 
        otherwise use the local audio encoder.

        Returns:
            The audio encoder instance.
        """
        if self._base_model_ref:
            return self._base_model_ref().audio
        return self._audio_encoder

    async def register_capability(self, name: str, description: str, 
                                  parameters: Dict[str, Any], handler: Callable):
        """
        Register a capability via the MCP protocol.

        Args:
            name: Name of the capability.
            description: Description of the capability.
            parameters: Parameters of the capability.
            handler: Handler function for the capability.
        """
        await self.mcp_tools.register_capability(name, description, parameters, handler)
    
    async def register_native_tool(self, name: str, description: str, 
                                  parameters: Dict[str, Any], handler: Callable):
        """Register a native tool with direct execution capability.
        
        Args:
            name: Name of the tool.
            description: Description of the tool.
            parameters: Parameters of the tool.
            handler: Native function handler for direct execution.
        """
        result = await self.mcp_tools.register_native_tool(name, description, parameters, handler)
        self.performance_monitor["total_tools_registered"] += 1
        return result
    
    async def execute_tool_with_smart_routing(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute tool with intelligent routing between native and external execution.
        
        Args:
            tool_name: Name of the tool to execute.
            **kwargs: Arguments to pass to the tool.
            
        Returns:
            Dict containing execution result and metadata.
        """
        if not self.smart_routing_enabled:
            # 回退到普通工具调用
            result = await self.mcp_tools.handle_tool_call(tool_name, **kwargs)
            return {"result": result, "execution_mode": "direct", "routed": False}
        
        # 使用双轨MCP协议进行智能执行
        execution_result = await self.mcp_protocol.execute_tool_with_fallback(
            tool_name, kwargs, "tool_executor"
        )
        
        # 更新执行统计
        self.execution_stats["total_executions"] += 1
        self.execution_stats["routing_decisions"] += 1
        
        if execution_result.get("execution_mode") == "native":
            self.execution_stats["native_executions"] += 1
        elif execution_result.get("execution_mode") == "external":
            self.execution_stats["external_executions"] += 1
        
        return execution_result
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics with advanced analytics."""
        tool_stats = self.mcp_tools.get_execution_stats()
        
        # 计算智能路由效率
        total_routing = self.execution_stats["routing_decisions"]
        native_ratio = self.execution_stats["native_executions"] / total_routing if total_routing > 0 else 0
        external_ratio = self.execution_stats["external_executions"] / total_routing if total_routing > 0 else 0
        
        # 性能优化建议
        performance_suggestions = []
        if native_ratio < 0.3 and self.execution_stats["total_executions"] > 10:
            performance_suggestions.append("Consider converting more tools to native execution for better performance")
        if tool_stats["performance_metrics"]["average_execution_time"] > 1.0:
            performance_suggestions.append("High average execution time detected - review tool implementations")
        if tool_stats["performance_metrics"]["efficiency_rating"] == "low":
            performance_suggestions.append("Low efficiency rating - optimize tool selection and execution paths")
        
        # 系统运行时间
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
                "routing_success_rate": 1.0 if total_routing > 0 else 0.0,  # 简化版本
                "optimization_potential": max(0.0, 1.0 - native_ratio)  # 原生执行提升空间
            },
            "performance_suggestions": performance_suggestions,
            "system_health": {
                "status": "healthy" if tool_stats["performance_metrics"]["performance_score"] > 0.6 else "needs_attention",
                "last_updated": datetime.now().isoformat(),
                "recommendations": performance_suggestions[:2],  # 前两条建议
                "uptime_seconds": uptime,
                "tools_registration_rate": self.performance_monitor["total_tools_registered"] / max(1, uptime / 60)  # 每分钟注册速率
            },
            "performance_monitor": self.performance_monitor
        }
    
    def reset_all_statistics(self):
        """Reset all execution statistics."""
        self.execution_stats = {
            "native_executions": 0,
            "external_executions": 0,
            "total_executions": 0,
            "routing_decisions": 0
        }
        self.mcp_tools.reset_execution_stats()
        
        # 重置性能监控但保留启动时间
        start_time = self.performance_monitor["start_time"]
        self.performance_monitor = {
            "start_time": start_time,
            "peak_memory_usage": 0,
            "total_tools_registered": 0
        }
        print("[ArcticAgentic] All execution statistics reset")
    
    def enable_smart_routing(self, enabled: bool = True):
        """Enable or disable smart routing functionality."""
        self.smart_routing_enabled = enabled
        print(f"[ArcticAgentic] Smart routing {'enabled' if enabled else 'disabled'}")
        
        # 记录状态变更到统计信息
        if not hasattr(self, 'routing_state_changes'):
            self.routing_state_changes = []
        self.routing_state_changes.append({
            "timestamp": datetime.now().isoformat(),
            "enabled": enabled,
            "total_executions": self.execution_stats["total_executions"]
        })
    
    def get_routing_history(self) -> List[Dict[str, Any]]:
        """Get smart routing state change history."""
        return getattr(self, 'routing_state_changes', [])

    async def _handle_mcp_message(self, message: PiscesLxCoreMCPMessage) -> PiscesLxCoreMCPMessage:
        """
        Handle incoming MCP messages.

        Args:
            message: Incoming MCP message.

        Returns:
            An MCP message as a response. If the message type is unknown, 
            return an error message.
        """
        handler = self.mcp_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        else:
            return PiscesLxCoreMCPProtocol.create_message(
                PiscesLxCoreMCPMessageType.STATE_UPDATE,
                self.agent_id,
                {"error": f"Unknown message type: {message.message_type}"}
            )

    async def _handle_observation(self, message: PiscesLxCoreMCPMessage) -> PiscesLxCoreMCPMessage:
        """
        Handle MCP observation messages.

        Args:
            message: MCP observation message.

        Returns:
            An MCP observation message indicating the processing status.
        """
        observation_data = message.payload
        observation = PiscesLxCoreAgenticObservation(
            modality=observation_data["modality"],
            content=observation_data["content"],
            metadata=observation_data.get("metadata", {})
        )
        
        self.memory.add_observation(observation)
        obs_embedding = self.process_observation(observation)
        
        return PiscesLxCoreMCPProtocol.create_message(
            PiscesLxCoreMCPMessageType.OBSERVATION,
            self.agent_id,
            {
                "status": "processed",
                "embedding_shape": list(obs_embedding.shape) if torch.is_tensor(obs_embedding) else None,
                "observation_id": str(uuid.uuid4())
            }
        )

    async def _handle_action(self, message: PiscesLxCoreMCPMessage) -> PiscesLxCoreMCPMessage:
        """
        Handle MCP action messages.

        Args:
            message: MCP action message.

        Returns:
            An MCP action message containing the planned action and the agent's state.
        """
        action_data = message.payload
        context = {
            "observation": action_data.get("observation"),
            "available_tools": list(self.mcp_capabilities.keys())
        }
        
        action = await self.plan_action(context)
        self.memory.add_action(action)
        
        return PiscesLxCoreMCPProtocol.create_message(
            PiscesLxCoreMCPMessageType.ACTION,
            self.agent_id,
            {
                "action": asdict(action),
                "agent_state": self.state.value
            }
        )

    async def _handle_tool_call(self, message: PiscesLxCoreMCPMessage) -> PiscesLxCoreMCPMessage:
        """
        Handle MCP tool call messages with smart routing support.

        Args:
            message: MCP tool call message.

        Returns:
            An MCP tool result message indicating the execution result. 
            If the tool is not found, return an error message.
        """
        tool_data = message.payload
        tool_name = tool_data["tool_name"]
        parameters = tool_data["parameters"]
        execution_mode = tool_data.get("execution_mode", "auto")
        
        # 使用智能路由执行工具调用
        if self.smart_routing_enabled and execution_mode in ["auto", "native", "external"]:
            execution_result = await self.execute_tool_with_smart_routing(tool_name, **parameters)
            result = execution_result.get("result", execution_result)
            actual_execution_mode = execution_result.get("execution_mode", "unknown")
            success = execution_result.get("success", True)
        else:
            # 回退到传统工具调用
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
            ArcticMCPMessageType.TOOL_RESULT,
            self.agent_id,
            {
                "tool_name": tool_name,
                "result": result,
                "success": success,
                "execution_mode": actual_execution_mode,
                "smart_routing_used": self.smart_routing_enabled
            }
        )

    async def _handle_tool_result(self, message: PiscesLxCoreMCPMessage) -> PiscesLxCoreMCPMessage:
        """
        Handle MCP tool result messages.

        Args:
            message: MCP tool result message.

        Returns:
            An MCP state update message indicating that the tool result has been processed.
        """
        result_data = message.payload
        
        observation = PiscesLxCoreAgenticObservation(
            modality="tool_result",
            content=result_data,
            metadata={"source": message.agent_id}
        )
        
        self.memory.add_observation(observation)
        
        return PiscesLxCoreMCPProtocol.create_message(
            PiscesLxCoreMCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "tool_result_processed", "result_id": str(uuid.uuid4())}
        )

    async def _handle_capability_register(self, message: PiscesLxCoreMCPMessage) -> PiscesLxCoreMCPMessage:
        """
        Handle capability registration from other agents.

        Args:
            message: MCP message for capability registration.

        Returns:
            An MCP state update message indicating that the capability has been registered.
        """
        capability_data = message.payload
        capability_name = capability_data["capability"]
        
        if message.agent_id != self.agent_id:
            self.mcp_capabilities[f"{message.agent_id}.{capability_name}"] = capability_data
        
        return PiscesLxCoreMCPProtocol.create_message(
            PiscesLxCoreMCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "capability_registered", "capability": capability_name}
        )

    async def _handle_sync_request(self, message: PiscesLxCoreMCPMessage) -> PiscesLxCoreMCPMessage:
        """
        Handle synchronization requests.

        Args:
            message: MCP synchronization request message.

        Returns:
            An MCP synchronization response message containing the requested information. 
            If the sync type is unknown, return an error message.
        """
        sync_type = message.payload.get("type")
        
        if sync_type == "capability_discovery":
            return PiscesLxCoreMCPProtocol.create_message(
                PiscesLxCoreMCPMessageType.SYNC_RESPONSE,
                self.agent_id,
                {
                    "type": "capabilities",
                    "capabilities": list(self.mcp_capabilities.keys())
                }
            )
        elif sync_type == "state_sync":
            return PiscesLxCoreMCPProtocol.create_message(
                ArcticMCPMessageType.SYNC_RESPONSE,
                self.agent_id,
                {
                    "type": "state",
                    "state": self.state.value,
                    "memory_summary": self._summarize_memory()
                }
            )
        
        return PiscesLxCoreMCPProtocol.create_message(
            ArcticMCPMessageType.SYNC_RESPONSE,
            self.agent_id,
            {"error": "Unknown sync type"}
        )

    async def _handle_sync_response(self, message: PiscesLxCoreMCPMessage) -> PiscesLxCoreMCPMessage:
        """
        Handle synchronization responses.

        Args:
            message: MCP synchronization response message.

        Returns:
            An MCP state update message indicating that the synchronization is completed.
        """
        response_data = message.payload
        
        if response_data.get("type") == "capabilities":
            self.mcp_peers[message.agent_id] = {
                "capabilities": response_data.get("capabilities", [])
            }
        
        return PiscesLxCoreMCPProtocol.create_message(
            ArcticMCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "sync_completed", "peer_id": message.agent_id}
        )

    def process_observation(self, observation: PiscesLxCoreAgenticObservation) -> torch.Tensor:
        """
        Process multimodal observations into a unified representation.

        Args:
            observation: A PiscesLxCoreAgenticObservation instance containing observation data.

        Returns:
            A torch.Tensor representing the observation embedding. 
            If processing fails, return a zero tensor.
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
            # Convert tool results to embedding
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

    async def plan_action(self, context: Dict[str, Any]) -> PiscesLxCoreAgenticAction:
        """
        Generate an action based on the current context and enhanced reasoning.

        Args:
            context: A dictionary containing the current context and available tools.

        Returns:
            A PiscesLxCoreAgenticAction instance representing the planned action.
        """
        # Get enhanced context from memory with semantic search
        memory_context = self.memory.get_context_with_retrieval(
            query=str(context), 
            k=5, 
            include_compressed=True
        )
        
        # Encode query for semantic search
        query_embedding = self._encode_query(str(context))
        
        # Perform semantic memory retrieval
        relevant_memories = self.memory.semantic_search(
            query_embedding=query_embedding,
            k=3,
            threshold=0.7
        )
        
        # Extract memory keys and values for enhanced reasoning
        memory_keys = self._extract_memory_keys(relevant_memories)
        memory_values = self._extract_memory_values(relevant_memories)
        
        # Prepare enhanced reasoning input
        enhanced_input = self._prepare_enhanced_reasoning_input(
            context=context,
            memory_context=memory_context,
            memory_keys=memory_keys,
            memory_values=memory_values,
            query_embedding=query_embedding
        )
        
        # Use enhanced reasoner for multi - step CoT reasoning
        with torch.no_grad():
            if self.base_model and hasattr(self, 'reasoner'):
                # Enhanced reasoning with memory context
                reasoning_output = self.reasoner(
                    enhanced_input,
                    memory_context=memory_context.get("embeddings", None)
                )
                
                # Multi - step CoT processing
                thinking_logits = reasoning_output.get("thinking_logits", reasoning_output.get("logits"))
                difficulty_logits = reasoning_output.get("difficulty_logits")
                reflection_logits = reasoning_output.get("reflection_logits")
                confidence_logits = reasoning_output.get("confidence_logits")
                
                # Enhanced action prediction
                action_logits = self.action_type_head(thinking_logits[:, -1])
                action_probs = torch.softmax(action_logits, dim=-1)
                action_type_idx = torch.argmax(action_probs, dim=-1).item()
                
                # Enhanced confidence calculation with reflection
                base_confidence = torch.sigmoid(self.confidence_head(thinking_logits[:, -1])).item()
                reflection_confidence = torch.sigmoid(reflection_logits[:, -1]).item() if reflection_logits is not None else base_confidence
                confidence = (base_confidence + reflection_confidence) / 2
                
                # Difficulty - aware reasoning
                if difficulty_logits is not None:
                    difficulty = torch.softmax(difficulty_logits[:, -1], dim=-1)
                    difficulty_level = torch.argmax(difficulty, dim=-1).item()
                else:
                    difficulty_level = 2  # Default medium
                
            else:
                # Fallback for stand - alone mode
                action_logits = self.action_type_head(torch.randn(1, self.cfg.hidden_size))
                action_probs = torch.softmax(action_logits, dim=-1)
                action_type_idx = torch.argmax(action_probs, dim=-1).item()
                confidence = 0.5
                difficulty_level = 2
            
            # Enhanced action types with deep thinking
            action_types = [
                "respond", "use_tool", "ask_clarification", "reflect", 
                "search_memory", "plan_next", "wait", "verify", 
                "correct_action", "explore", "deep_think", "summarize"
            ]
            action_type = action_types[action_type_idx]
            
            # Generate enhanced action parameters
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
            
            # Generate detailed reasoning trace
            reasoning_trace = self._generate_reasoning_trace(
                context=context,
                memory_summary=memory_context.get("memory_summary", {}),
                action_type=action_type,
                confidence=confidence,
                difficulty_level=difficulty_level,
                relevant_memories=relevant_memories
            )
            
            return PiscesLxCoreAgenticAction(
                action_type=action_type,
                parameters=action_params,
                confidence=confidence,
                reasoning=reasoning_trace
            )

    def _prepare_reasoning_input(self, context: Dict[str, Any], memory_context: Dict[str, List]) -> Dict[str, Any]:
        """
        Prepare input for the reasoner.

        Args:
            context: Current context dictionary.
            memory_context: Memory context dictionary.

        Returns:
            A dictionary containing the input for the reasoner.
        """
        return {
            "context": context,
            "memory": memory_context,
            "agent_state": self.state.value
        }

    def _decode_action_params(self, param_embedding: torch.Tensor, action_type: str) -> Dict[str, Any]:
        """
        Decode action parameters from the embedding.

        Args:
            param_embedding: Action parameter embedding tensor.
            action_type: Type of the action.

        Returns:
            A dictionary containing the decoded action parameters.
        """
        return {
            "embedding": param_embedding.detach().cpu().numpy().tolist(),
            "decoded_from": action_type,
            "confidence": torch.sigmoid(self.confidence_head(param_embedding)).item()
        }

    def _encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a query string into a semantic embedding.

        Args:
            query: Query string to be encoded.

        Returns:
            A torch.Tensor representing the query embedding. 
            If the tokenizer or base model is not available, use a hash - based fallback.
        """
        if hasattr(self, 'tokenizer') and self.tokenizer and self.base_model:
            tokens = self.tokenizer.encode(query, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                embeddings = self.base_model.embed_tokens(tokens)
                # Use mean pooling for query embedding
                query_embedding = embeddings.mean(dim=1)
                return query_embedding
        else:
            # Fallback: use simple hash - based encoding
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
        """
        Extract memory keys from retrieved memories.

        Args:
            memories: List of memory dictionaries.

        Returns:
            List of memory keys (strings).
        """
        keys = []
        for memory in memories:
            if "content" in memory:
                content = str(memory["content"])
                # Extract key phrases (simplified)
                key_phrases = content.split()[:10]  # First 10 words as key
                keys.append(" ".join(key_phrases))
            else:
                keys.append("memory_entry")
        return keys

    def _extract_memory_values(self, memories: List[Dict[str, Any]]) -> List[str]:
        """
        Extract memory values from retrieved memories.

        Args:
            memories: List of memory dictionaries.

        Returns:
            List of memory values (strings).
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
        """
        Prepare enhanced input for multi - step CoT reasoning.

        Args:
            context: Current context dictionary.
            memory_context: Memory context dictionary.
            memory_keys: List of memory keys.
            memory_values: List of memory values.
            query_embedding: Query embedding tensor.

        Returns:
            A dictionary containing the enhanced reasoning input.
        """
        return {
            "context": context,
            "memory_context": memory_context,
            "memory_keys": memory_keys,
            "memory_values": memory_values,
            "query_embedding": query_embedding,
            "agent_state": self.state.value,
            "timestamp": str(uuid.uuid4())[:8]  # Simple timestamp
        }

    def _decode_enhanced_action_params(self, param_embedding: torch.Tensor, 
                                     action_type: str, 
                                     difficulty_level: int,
                                     confidence: float) -> Dict[str, Any]:
        """
        Decode enhanced action parameters considering difficulty and confidence.

        Args:
            param_embedding: Action parameter embedding tensor.
            action_type: Type of the action.
            difficulty_level: Difficulty level of the action.
            confidence: Confidence of the action.

        Returns:
            A dictionary containing the decoded enhanced action parameters.
        """
        params = {
            "embedding": param_embedding.detach().cpu().numpy().tolist(),
            "action_type": action_type,
            "difficulty_level": difficulty_level,
            "confidence": confidence,
            "timestamp": str(uuid.uuid4())[:8]
        }
        
        # Add action - specific parameters
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
        """
        Generate a detailed reasoning trace for transparency.

        Args:
            context: Current context dictionary.
            memory_summary: Memory summary dictionary.
            action_type: Type of the selected action.
            confidence: Confidence of the action.
            difficulty_level: Difficulty level of the action.
            relevant_memories: List of relevant memories.

        Returns:
            A string representing the reasoning trace.
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
        """
        Summarize memory for state synchronization.

        Returns:
            A dictionary containing the counts of observations, actions, and reflections.
        """
        return {
            "observations": len(self.memory.observations),
            "actions": len(self.memory.actions),
            "reflections": len(self.memory.reflections)
        }

    def detect_objects(self, image_input: Union[str, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        Detect objects in an image and return their coordinates.

        Args:
            image_input: Image path string, tensor, or numpy array.

        Returns:
            A dictionary containing detected objects with coordinates, image size, and number of objects.
            If an error occurs or no objects are detected, return an empty result.
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
            
            # Process bounding boxes and coordinates
            if "boxes" in results and "labels" in results:
                boxes = results["boxes"].cpu().numpy()
                labels = results["labels"].cpu().numpy()
                scores = results.get("scores", torch.ones(len(boxes))).cpu().numpy()
                
                # Convert to image coordinates
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
            
            # Get image dimensions
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
        """
        Get the coordinates of detected objects or a specific target object.

        Args:
            image_input: Image to analyze.
            target_object: Optional target object name to filter results.

        Returns:
            A list of [x, y] coordinates for the detected objects.
        """
        detection_results = self.detect_objects(image_input)
        
        coordinates = []
        for obj in detection_results.get("objects", []):
            if target_object is None or target_object.lower() in obj["class"].lower():
                coordinates.append(obj["coordinates"])
        
        return coordinates

    def point_to_object(self, image_input: Union[str, torch.Tensor, np.ndarray], 
                       object_description: str) -> Dict[str, Any]:
        """
        Point to a specific object in the image based on the description.

        Args:
            image_input: Image to analyze.
            object_description: Description of the object to find.

        Returns:
            A dictionary with object information and pointing coordinates. 
            If the object is not found, indicate failure.
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
        """
        Enable or disable the coordinate detection functionality.

        Args:
            enabled: Boolean indicating whether to enable coordinate detection, defaults to True.
        """
        self._coordinate_detection_enabled = enabled
        if hasattr(self.vision_encoder, 'enable_detection'):
            self.vision_encoder.enable_detection(enabled)

