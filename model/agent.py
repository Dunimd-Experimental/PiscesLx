#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces L1.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import uuid
import torch
import asyncio
import torch.nn as nn
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Callable

from .modeling_aurora import PiscesModel
from .multimodal import VisionEncoder, AudioEncoder
from .reasoner import PiscesReasoner, TreeSearchReasoner

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentAction:
    action_type: str
    parameters: Dict[str, Any]
    confidence: float = 1.0
    reasoning: str = ""

@dataclass
class AgentObservation:
    modality: str  # "text", "image", "audio", "tool_result"
    content: Any
    metadata: Dict[str, Any]


@dataclass
class AgentMemory:
    observations: List[AgentObservation]
    actions: List[AgentAction]
    reflections: List[str]
    
    def add_observation(self, observation: AgentObservation):
        self.observations.append(observation)
    
    def add_action(self, action: AgentAction):
        self.actions.append(action)
    
    def add_reflection(self, reflection: str):
        self.reflections.append(reflection)
    
    def get_recent_context(self, k: int = 5) -> Dict[str, List]:
        return {
            "recent_observations": self.observations[-k:],
            "recent_actions": self.actions[-k:],
            "recent_reflections": self.reflections[-k:]
        }

class MCPMessageType(Enum):
    """MCP message types for agent communication"""
    OBSERVATION = "observation"
    ACTION = "action"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE_UPDATE = "state_update"
    CAPABILITY_REGISTER = "capability_register"
    HEARTBEAT = "heartbeat"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"

@dataclass
class MCPMessage:
    """Standardized MCP message format"""
    message_type: str
    agent_id: str
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: str = ""
    priority: str = "normal"

class MCPProtocol:
    """MCP protocol implementation for agent communication"""
    
    @staticmethod
    def create_message(
        message_type: MCPMessageType,
        agent_id: str,
        payload: Dict[str, Any],
        correlation_id: str = ""
    ) -> MCPMessage:
        return MCPMessage(
            message_type=message_type.value,
            agent_id=agent_id,
            payload=payload,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id or str(uuid.uuid4())
        )
    
    @staticmethod
    def serialize(message: MCPMessage) -> str:
        return json.dumps(asdict(message))
    
    @staticmethod
    def deserialize(data: str) -> MCPMessage:
        return MCPMessage(**json.loads(data))

class MCPToolRegistry:
    """MCP-compatible tool registry using protocol communication"""
    
    def __init__(self, agent_id: str, message_handler: Callable):
        self.agent_id = agent_id
        self.message_handler = message_handler
        self.capabilities: Dict[str, Dict[str, Any]] = {}
    
    async def register_capability(self, name: str, description: str, 
                                  parameters: Dict[str, Any], 
                                  handler: Callable):
        """Register a capability via MCP protocol"""
        self.capabilities[name] = {
            "description": description,
            "parameters": parameters,
            "handler": handler
        }
        
        message = MCPProtocol.create_message(
            MCPMessageType.CAPABILITY_REGISTER,
            self.agent_id,
            {
                "capability": name,
                "description": description,
                "parameters": parameters
            }
        )
        await self.message_handler(message)
    
    async def discover_capabilities(self) -> List[str]:
        """Discover available capabilities via MCP"""
        message = MCPProtocol.create_message(
            MCPMessageType.SYNC_REQUEST,
            self.agent_id,
            {"type": "capability_discovery"}
        )
        response = await self.message_handler(message)
        return response.payload.get("capabilities", [])

class ToolRegistry:
    """Registry for managing available tools - deprecated in favor of MCP"""
    
    def __init__(self):
        self.tools: Dict[str, callable] = {}
    
    def register(self, name: str, func: callable):
        self.tools[name] = func
    
    def get_tool(self, name: str) -> Optional[callable]:
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

class PiscesAgent(nn.Module):
    """
    Native Pisces L1 Agent with integrated reasoning, perception, and action capabilities.
    
    Features:
    1. Unified multimodal perception (text, image, audio)
    2. Advanced reasoning with CoT and self-reflection
    3. Tool use and environment interaction
    4. Persistent memory and context management
    5. End-to-end trainable architecture
    """
    
    def __init__(self, cfg, tokenizer=None, model=None, agent_id: str = None):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        import weakref
        self._model_ref = None  # weak reference placeholder
        self.agent_id = agent_id or str(uuid.uuid4())
        
        # Use provided model or create new one
        import weakref
        if model is not None:
            # Store a weak reference (callable) to avoid registering model as a submodule.
            self._base_model_ref = weakref.ref(model)
            self._model_ref = self._base_model_ref
        else:
            # Stand-alone agent: no linked PiscesModel to avoid cycles.
            self._base_model_ref = None
            self._model_ref = None
            self._reasoner = PiscesReasoner(cfg)
            self._vision_encoder = VisionEncoder(cfg)
            self._audio_encoder = AudioEncoder(cfg)
        
        self.tree_reasoner = TreeSearchReasoner(None, tokenizer) if tokenizer else None

        # Expose base_model via a property for compatibility
    @property
    def base_model(self):
        return self._base_model_ref() if self._base_model_ref else None

    @property
    def reasoner(self):
        if self._base_model_ref:
            return self._base_model_ref().reasoner
        return self._reasoner

    @property
    def vision_encoder(self):
        if self._base_model_ref:
            return self._base_model_ref().vision
        return self._vision_encoder

    @property
    def audio_encoder(self):
        if self._base_model_ref:
            return self._base_model_ref().audio
        return self._audio_encoder
        
        # MCP Agent infrastructure
        self.memory = AgentMemory([], [], [])
        self.mcp_tools = MCPToolRegistry(self.agent_id, self._handle_mcp_message)
        self.state = AgentState.IDLE
        self.mcp_peers: Dict[str, Dict[str, Any]] = {}
        self.mcp_capabilities: Dict[str, Dict[str, Any]] = {}
        
        # Action prediction heads (now MCP-aware)
        self.action_type_head = nn.Linear(cfg.hidden_size, 10)  # 10 action types
        self.action_param_head = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.confidence_head = nn.Linear(cfg.hidden_size, 1)
        
        # MCP message handlers
        self.mcp_handlers = {
            MCPMessageType.OBSERVATION.value: self._handle_observation,
            MCPMessageType.ACTION.value: self._handle_action,
            MCPMessageType.TOOL_CALL.value: self._handle_tool_call,
            MCPMessageType.TOOL_RESULT.value: self._handle_tool_result,
            MCPMessageType.CAPABILITY_REGISTER.value: self._handle_capability_register,
            MCPMessageType.SYNC_REQUEST.value: self._handle_sync_request,
            MCPMessageType.SYNC_RESPONSE.value: self._handle_sync_response,
        }
        
        # Special tokens for agent control (MCP-compatible)
        self.start_agent_id = None
        self.end_agent_id = None
        self.tool_call_id = None
        self.tool_result_id = None
        
    def register_tool(self, name: str, func: callable):
        """Register a new tool for the agent to use - deprecated, use MCP registration"""
        import warnings
        warnings.warn("register_tool is deprecated. Use register_capability via MCP", DeprecationWarning)
        
    async def register_capability(self, name: str, description: str, 
                                  parameters: Dict[str, Any], handler: Callable):
        """Register a capability via MCP protocol"""
        await self.mcp_tools.register_capability(name, description, parameters, handler)
    
    async def _handle_mcp_message(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming MCP messages"""
        handler = self.mcp_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        else:
            return MCPProtocol.create_message(
                MCPMessageType.STATE_UPDATE,
                self.agent_id,
                {"error": f"Unknown message type: {message.message_type}"}
            )
    
    async def _handle_observation(self, message: MCPMessage) -> MCPMessage:
        """Handle MCP observation messages"""
        observation_data = message.payload
        observation = AgentObservation(
            modality=observation_data["modality"],
            content=observation_data["content"],
            metadata=observation_data.get("metadata", {})
        )
        
        self.memory.add_observation(observation)
        obs_embedding = self.process_observation(observation)
        
        return MCPProtocol.create_message(
            MCPMessageType.OBSERVATION,
            self.agent_id,
            {
                "status": "processed",
                "embedding_shape": list(obs_embedding.shape) if torch.is_tensor(obs_embedding) else None,
                "observation_id": str(uuid.uuid4())
            }
        )
    
    async def _handle_action(self, message: MCPMessage) -> MCPMessage:
        """Handle MCP action messages"""
        action_data = message.payload
        context = {
            "observation": action_data.get("observation"),
            "available_tools": list(self.mcp_capabilities.keys())
        }
        
        action = await self.mcp_plan_action(context)
        self.memory.add_action(action)
        
        return MCPProtocol.create_message(
            MCPMessageType.ACTION,
            self.agent_id,
            {
                "action": asdict(action),
                "agent_state": self.state.value
            }
        )
    
    async def _handle_tool_call(self, message: MCPMessage) -> MCPMessage:
        """Handle MCP tool call messages"""
        tool_data = message.payload
        tool_name = tool_data["tool_name"]
        parameters = tool_data["parameters"]
        
        if tool_name in self.mcp_capabilities:
            handler = self.mcp_capabilities[tool_name]["handler"]
            result = await handler(**parameters)
            
            return MCPProtocol.create_message(
                MCPMessageType.TOOL_RESULT,
                self.agent_id,
                {
                    "tool_name": tool_name,
                    "result": result,
                    "success": True
                }
            )
        else:
            return MCPProtocol.create_message(
                MCPMessageType.TOOL_RESULT,
                self.agent_id,
                {
                    "tool_name": tool_name,
                    "error": f"Tool {tool_name} not found",
                    "success": False
                }
            )
    
    async def _handle_tool_result(self, message: MCPMessage) -> MCPMessage:
        """Handle MCP tool result messages"""
        result_data = message.payload
        
        observation = AgentObservation(
            modality="tool_result",
            content=result_data,
            metadata={"source": message.agent_id}
        )
        
        self.memory.add_observation(observation)
        
        return MCPProtocol.create_message(
            MCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "tool_result_processed", "result_id": str(uuid.uuid4())}
        )
    
    async def _handle_capability_register(self, message: MCPMessage) -> MCPMessage:
        """Handle capability registration from other agents"""
        capability_data = message.payload
        capability_name = capability_data["capability"]
        
        if message.agent_id != self.agent_id:
            self.mcp_capabilities[f"{message.agent_id}.{capability_name}"] = capability_data
        
        return MCPProtocol.create_message(
            MCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "capability_registered", "capability": capability_name}
        )
    
    async def _handle_sync_request(self, message: MCPMessage) -> MCPMessage:
        """Handle synchronization requests"""
        sync_type = message.payload.get("type")
        
        if sync_type == "capability_discovery":
            return MCPProtocol.create_message(
                MCPMessageType.SYNC_RESPONSE,
                self.agent_id,
                {
                    "type": "capabilities",
                    "capabilities": list(self.mcp_capabilities.keys())
                }
            )
        elif sync_type == "state_sync":
            return MCPProtocol.create_message(
                MCPMessageType.SYNC_RESPONSE,
                self.agent_id,
                {
                    "type": "state",
                    "state": self.state.value,
                    "memory_summary": self._summarize_memory()
                }
            )
        
        return MCPProtocol.create_message(
            MCPMessageType.SYNC_RESPONSE,
            self.agent_id,
            {"error": "Unknown sync type"}
        )
    
    async def _handle_sync_response(self, message: MCPMessage) -> MCPMessage:
        """Handle synchronization responses"""
        response_data = message.payload
        
        if response_data.get("type") == "capabilities":
            self.mcp_peers[message.agent_id] = {
                "capabilities": response_data.get("capabilities", [])
            }
        
        return MCPProtocol.create_message(
            MCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "sync_completed", "peer_id": message.agent_id}
        )
    
    def process_observation(self, observation: AgentObservation) -> torch.Tensor:
        """Process multimodal observations into unified representation"""
        if observation.modality == "text":
            if self.tokenizer:
                tokens = self.tokenizer.encode(str(observation.content), return_tensors="pt")
                return self.base_model.embed_tokens(tokens)
            else:
                return torch.zeros(1, 1, self.cfg.hidden_size)
        
        elif observation.modality == "image":
            if isinstance(observation.content, str):  # image path
                image_tensor = self.vision_encoder.process_image(observation.content)
                if image_tensor is not None:
                    image_tensor = image_tensor.unsqueeze(0)
                    return self.vision_encoder(image_tensor)
            elif torch.is_tensor(observation.content):
                return self.vision_encoder(observation.content)
        
        elif observation.modality == "audio":
            if isinstance(observation.content, str):  # audio path
                audio_tensor = self.audio_encoder.process_audio(observation.content)
                if audio_tensor is not None:
                    return self.audio_encoder(audio_tensor)
            elif torch.is_tensor(observation.content):
                return self.audio_encoder(observation.content)
        
        elif observation.modality == "tool_result":
            # Convert tool results to embedding
            result_str = json.dumps(observation.content)
            if self.tokenizer:
                tokens = self.tokenizer.encode(result_str, return_tensors="pt")
                return self.base_model.embed_tokens(tokens)
        
        # Fallback to zero tensor
        return torch.zeros(1, 1, self.cfg.hidden_size)
    
    def plan_action(self, context: Dict[str, Any]) -> AgentAction:
        """Generate action based on current context and reasoning"""
        # Get recent context from memory
        memory_context = self.memory.get_recent_context(k=3)
        
        # Combine current observation with memory
        combined_input = self._prepare_reasoning_input(context, memory_context)
        
        # Use reasoner for deep thinking
        with torch.no_grad():
            reasoning_output = self.reasoner(combined_input)
            
            # Predict action type and parameters
            action_logits = self.action_type_head(reasoning_output["thinking_logits"][:, -1])
            action_type_idx = torch.argmax(action_logits, dim=-1).item()
            
            action_types = [
                "respond", "use_tool", "ask_clarification", "reflect", 
                "search_memory", "plan_next", "wait", "verify", 
                "correct_action", "explore"
            ]
            action_type = action_types[action_type_idx]
            
            # Predict confidence
            confidence = torch.sigmoid(self.confidence_head(reasoning_output["thinking_logits"][:, -1])).item()
            
            # Generate action parameters
            param_embedding = self.action_param_head(reasoning_output["thinking_logits"][:, -1])
            action_params = self._decode_action_params(param_embedding, action_type)
            
            return AgentAction(
                action_type=action_type,
                parameters=action_params,
                confidence=confidence,
                reasoning=reasoning_output.get("reasoning", "")
            )
    
    def execute_action(self, action: AgentAction) -> Any:
        """Execute the planned action"""
        self.state = AgentState.ACTING
        
        try:
            if action.action_type == "respond":
                return self._generate_response(action.parameters)
            
            elif action.action_type == "use_tool":
                tool_name = action.parameters.get("tool_name")
                tool_args = action.parameters.get("tool_args", {})
                
                tool_func = self.tools.get_tool(tool_name)
                if tool_func:
                    result = tool_func(**tool_args)
                    
                    # Store tool result as observation
                    tool_observation = AgentObservation(
                        modality="tool_result",
                        content=result,
                        metadata={"tool": tool_name, "args": tool_args}
                    )
                    self.memory.add_observation(tool_observation)
                    return result
                else:
                    return {"error": f"Tool {tool_name} not found"}
            
            elif action.action_type == "reflect":
                return self._perform_reflection(action.parameters)
            
            else:
                return {"status": f"Action {action.action_type} executed", "params": action.parameters}
        
        except Exception as e:
            self.state = AgentState.ERROR
            return {"error": str(e)}
        
        finally:
            self.state = AgentState.IDLE
    
    async def step(self, observation: AgentObservation) -> AgentAction:
        """
        Single agent step via MCP protocol: observe -> think -> act
        
        Args:
            observation: Current observation from environment
            
        Returns:
            AgentAction: The action taken by the agent
        """
        # Update state
        self.state = AgentState.THINKING
        
        # Store observation in memory
        self.memory.add_observation(observation)
        
        # Process observation via MCP
        obs_message = MCPProtocol.create_message(
            MCPMessageType.OBSERVATION,
            self.agent_id,
            {
                "modality": observation.modality,
                "content": observation.content,
                "metadata": observation.metadata
            }
        )
        
        await self._handle_observation(obs_message)
        
        # Plan action via MCP
        action_message = MCPProtocol.create_message(
            MCPMessageType.ACTION,
            self.agent_id,
            {
                "observation": asdict(observation),
                "available_tools": list(self.mcp_capabilities.keys())
            }
        )
        
        response = await self._handle_action(action_message)
        action_data = response.payload["action"]
        
        action = AgentAction(
            action_type=action_data["action_type"],
            parameters=action_data["parameters"],
            confidence=action_data.get("confidence", 1.0),
            reasoning=action_data.get("reasoning", "")
        )
        
        # Store action in memory
        self.memory.add_action(action)
        
        # Execute action
        result = await self.execute_action_async(action)
        
        # Add result to action parameters for memory
        action.parameters["result"] = result
        
        return action
    
    async def run(self, input_ids=None, images=None, audio=None, docs=None, task=None, max_steps=10, **kwargs) -> Dict[str, Any]:
        """
        Run a complete task with the agent supporting multimodal inputs via MCP protocol
        
        Args:
            input_ids: Text input as token ids
            images: Image input tensor or path
            audio: Audio input tensor or path
            docs: Document inputs
            task: Task description string (fallback)
            max_steps: Maximum number of steps to take
            
        Returns:
            Dict containing task results and agent history
        """
        self.state = AgentState.RUNNING
        
        try:
            # Initialize task with multimodal observations
            observations = []
            
            if input_ids is not None:
                text_content = self.tokenizer.decode(input_ids) if self.tokenizer else str(input_ids)
                observations.append(AgentObservation(
                    modality="text",
                    content=text_content,
                    metadata={"type": "text_input", "input_ids": input_ids}
                ))
            
            if images is not None:
                observations.append(AgentObservation(
                    modality="image",
                    content=images,
                    metadata={"type": "image_input"}
                ))
            
            if audio is not None:
                observations.append(AgentObservation(
                    modality="audio",
                    content=audio,
                    metadata={"type": "audio_input"}
                ))
            
            if docs is not None:
                observations.append(AgentObservation(
                    modality="text",
                    content=str(docs),
                    metadata={"type": "document_input"}
                ))
            
            # Fallback to task description if no multimodal inputs
            if not observations and task is not None:
                observations.append(AgentObservation(
                    modality="text",
                    content=task,
                    metadata={"type": "task_init"}
                ))
            
            # Add all observations to memory
            for obs in observations:
                self.memory.add_observation(obs)
            
            results = []
            
            for step_num in range(max_steps):
                if self.state == AgentState.COMPLETED:
                    break
                
                # Use last observation for step
                current_obs = observations[-1] if observations and step_num == 0 else results[-1].get("observation")
                action = await self.step(current_obs)
                
                # Execute action via MCP
                result = await self.execute_action_async(action)
                
                step_result = {
                    "step": step_num + 1,
                    "action": action,
                    "state": self.state.value,
                    "observation": current_obs,
                    "result": result
                }
                results.append(step_result)
                
                # Check if task is complete
                if await self._is_task_complete_async(result):
                    self.state = AgentState.COMPLETED
                    break
            
            return {
                "task_description": task or "Multimodal task",
                "multimodal_inputs": {
                    "text": input_ids is not None,
                    "image": images is not None,
                    "audio": audio is not None,
                    "docs": docs is not None
                },
                "steps": results,
                "final_state": self.state.value,
                "memory_summary": self._summarize_memory()
            }
            
        except Exception as e:
            self.state = AgentState.ERROR
            return {
                "status": "error",
                "error": str(e),
                "memory": self.memory
            }
    
    async def mcp_plan_action(self, context: Dict[str, Any]) -> AgentAction:
        """Plan next action via MCP protocol"""
        # Get recent context from memory
        memory_context = self.memory.get_recent_context(k=3)
        
        # Use MCP capabilities instead of local tools
        available_tools = list(self.mcp_capabilities.keys())
        
        # Combine current observation with memory
        combined_input = self._prepare_reasoning_input(context, memory_context)
        
        # Use reasoner for deep thinking
        with torch.no_grad():
            reasoning_output = self.reasoner(combined_input)
            
            # Predict action type and parameters
            action_logits = self.action_type_head(reasoning_output["thinking_logits"][:, -1])
            action_type_idx = torch.argmax(action_logits, dim=-1).item()
            
            action_types = [
                "respond", "use_tool", "ask_clarification", "reflect", 
                "search_memory", "plan_next", "wait", "verify", 
                "correct_action", "explore"
            ]
            action_type = action_types[action_type_idx]
            
            # Map to MCP capability if available
            if action_type == "use_tool" and available_tools:
                # Select best matching capability
                tool_name = context.get("tool_name", available_tools[0])
                action_type = tool_name
            
            # Predict confidence
            confidence = torch.sigmoid(self.confidence_head(reasoning_output["thinking_logits"][:, -1])).item()
            
            # Generate action parameters
            param_embedding = self.action_param_head(reasoning_output["thinking_logits"][:, -1])
            action_params = self._decode_action_params(param_embedding, action_type)
            
            return AgentAction(
                action_type=action_type,
                parameters=action_params,
                confidence=confidence,
                reasoning=reasoning_output.get("reasoning", "")
            )
    
    async def execute_action_async(self, action: AgentAction) -> Any:
        """Execute action via MCP protocol"""
        self.state = AgentState.ACTING
        
        try:
            if action.action_type in self.mcp_capabilities:
                handler_info = self.mcp_capabilities[action.action_type]
                handler = handler_info["handler"]
                
                # Execute via MCP
                result = await handler(**action.parameters)
                
                # Store tool result as observation
                tool_observation = AgentObservation(
                    modality="tool_result",
                    content=result,
                    metadata={"tool": action.action_type, "args": action.parameters}
                )
                self.memory.add_observation(tool_observation)
                
                return result
            
            elif action.action_type == "respond":
                return await self._generate_response_async(action.parameters)
            
            elif action.action_type == "reflect":
                return await self._perform_reflection_async(action.parameters)
            
            else:
                return {"status": f"Action {action.action_type} executed", "params": action.parameters}
        
        except Exception as e:
            self.state = AgentState.ERROR
            return {"error": str(e)}
        
        finally:
            self.state = AgentState.IDLE
    
    async def _is_task_complete_async(self, result: Any) -> bool:
        """Async check if task is complete"""
        if isinstance(result, dict) and "status" in result:
            return result.get("status") == "complete"
        
        # Simple heuristic: if result contains success indicators
        success_keywords = ["success", "complete", "done", "finished"]
        result_str = str(result).lower()
        return any(keyword in result_str for keyword in success_keywords)
    
    async def _generate_response_async(self, parameters: Dict[str, Any]) -> str:
        """Generate text response using base model asynchronously"""
        # Placeholder for async response generation
        return "Response generated by PiscesAgent via MCP"
    
    async def _perform_reflection_async(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform self-reflection and error correction asynchronously"""
        recent_actions = self.memory.actions[-5:] if self.memory.actions else []
        reflection = f"Reflecting on {len(recent_actions)} recent actions"
        
        self.memory.add_reflection(reflection)
        
        return {
            "reflection": reflection,
            "actions_reviewed": len(recent_actions),
            "confidence_improved": True
        }
    
    def _prepare_reasoning_input(self, context: Dict[str, Any], memory_context: Dict[str, List]) -> torch.Tensor:
        """Prepare input for the reasoner module"""
        # This is a simplified version - would need proper implementation
        # based on actual model architecture
        obs_embedding = context.get("observation_embedding", torch.zeros(1, 1, self.cfg.hidden_size))
        return obs_embedding
    
    def _decode_action_params(self, param_embedding: torch.Tensor, action_type: str) -> Dict[str, Any]:
        """Decode action parameters from embedding"""
        # Simplified parameter decoding
        return {
            "embedding": param_embedding.detach().cpu().numpy().tolist(),
            "decoded_from": action_type
        }
    
    def _generate_response(self, parameters: Dict[str, Any]) -> str:
        """Generate text response using base model"""
        # Placeholder for response generation
        return "Response generated by PiscesAgent"
    
    def _perform_reflection(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform self-reflection and error correction"""
        recent_actions = self.memory.actions[-5:] if self.memory.actions else []
        reflection = f"Reflecting on {len(recent_actions)} recent actions"
        
        self.memory.add_reflection(reflection)
        
        return {
            "reflection": reflection,
            "actions_reviewed": len(recent_actions),
            "confidence_improved": True
        }
    
    def _summarize_memory(self) -> Dict[str, Any]:
        """Create a summary of the agent's memory"""
        return {
            "total_observations": len(self.memory.observations),
            "total_actions": len(self.memory.actions),
            "total_reflections": len(self.memory.reflections),
            "recent_observations": len(self.memory.observations[-5:]),
            "recent_actions": len(self.memory.actions[-5:]),
            "mcp_capabilities": len(self.mcp_capabilities),
            "mcp_peers": len(self.mcp_peers)
        }

    async def discover_capabilities(self, peer_id: str = None) -> Dict[str, Any]:
        """Discover capabilities from MCP peers"""
        if peer_id:
            # Discover from specific peer
            sync_msg = MCPProtocol.create_message(
                MCPMessageType.SYNC_REQUEST,
                self.agent_id,
                {"type": "capability_discovery", "target": peer_id}
            )
            return await self._handle_sync_request(sync_msg)
        else:
            # Broadcast discovery
            discovered = {}
            for peer in self.mcp_peers:
                try:
                    capabilities = await self.discover_capabilities(peer)
                    discovered[peer] = capabilities
                except Exception as e:
                    discovered[peer] = {"error": str(e)}
            return discovered

    async def sync_state(self, peer_id: str = None) -> Dict[str, Any]:
        """Sync state with MCP peers"""
        if peer_id:
            sync_msg = MCPProtocol.create_message(
                MCPMessageType.SYNC_REQUEST,
                self.agent_id,
                {"type": "state_sync", "target": peer_id}
            )
            return await self._handle_sync_request(sync_msg)
        else:
            # Sync with all peers
            synced = {}
            for peer in self.mcp_peers:
                try:
                    state = await self.sync_state(peer)
                    synced[peer] = state
                except Exception as e:
                    synced[peer] = {"error": str(e)}
            return synced

    async def connect_to_mcp_hub(self, hub_endpoint: str):
        """Connect to MCP hub for centralized coordination"""
        # This would connect to a central MCP hub
        # For now, it's a placeholder for hub integration
        self.mcp_peers["hub"] = {
            "endpoint": hub_endpoint,
            "capabilities": ["coordination", "discovery", "monitoring"]
        }
        
        # Register self with hub
        register_msg = MCPProtocol.create_message(
            MCPMessageType.CAPABILITY_REGISTER,
            self.agent_id,
            {
                "capabilities": list(self.mcp_capabilities.keys()),
                "agent_type": "PiscesAgent",
                "version": "2.0.0"
            }
        )
        
        return await self._handle_capability_register(register_msg)
    
    def reset(self):
        """Reset agent state and memory"""
        self.memory = AgentMemory([], [], [])
        self.state = AgentState.IDLE
    
    def save_state(self, filepath: str):
        """Save agent state to file"""
        state_dict = {
            "memory": {
                "observations": [(obs.modality, str(obs.content), obs.metadata) for obs in self.memory.observations],
                "actions": [(act.action_type, act.parameters, act.confidence, act.reasoning) for act in self.memory.actions],
                "reflections": self.memory.reflections
            },
            "tools": self.tools.list_tools(),
            "state": self.state.value
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load agent state from file"""
        try:
            with open(filepath, 'r') as f:
                state_dict = json.load(f)
            
            # Note: This is simplified - would need proper deserialization
            self.reset()
            
        except Exception as e:
            print(f"Error loading agent state: {e}")
    
    def forward(self, inputs, **kwargs):
        """Forward pass for training"""
        # Integrate with base model for end-to-end training
        base_output = self.base_model(inputs, **kwargs)
        
        # Add agent-specific outputs
        hidden_states = base_output.last_hidden_state if hasattr(base_output, 'last_hidden_state') else base_output
        
        agent_outputs = {
            "base_output": base_output,
            "action_logits": self.action_type_head(hidden_states),
            "confidence_logits": self.confidence_head(hidden_states),
            "reasoning_output": self.reasoner(hidden_states)
        }
        
        return agent_outputs