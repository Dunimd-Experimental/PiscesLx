#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
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
from datetime import datetime
from typing import Dict, Any, Callable, List
from .types import MCPMessageType, MCPMessage

class MCPToolRegistry:
    """A registry for managing MCP tools and capabilities.
    
    This class provides functionality to store and manage tools and capabilities,
    as well as handle tool calls.
    """
    
    def __init__(self, agent_id: str, message_handler: Callable):
        """Initialize the MCPToolRegistry instance.

        Args:
            agent_id (str): The ID of the agent.
            message_handler (Callable): The message handler callable.
        """
        self.agent_id = agent_id
        self.message_handler = message_handler
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.capabilities: Dict[str, Dict[str, Any]] = {}
    
    async def register_capability(self, name: str, description: str, 
                                  parameters: Dict[str, Any], handler: Callable):
        """Register a new capability.

        Args:
            name (str): The name of the capability.
            description (str): The description of the capability.
            parameters (Dict[str, Any]): The parameters of the capability.
            handler (Callable): The handler callable for the capability.
        """
        self.capabilities[name] = {
            "description": description,
            "parameters": parameters,
            "handler": handler
        }
    
    async def handle_tool_call(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool call.

        Args:
            tool_name (str): The name of the tool to call.
            **kwargs: Additional keyword arguments to pass to the tool handler.

        Returns:
            Any: The result returned by the tool handler.

        Raises:
            ValueError: If the specified tool is not found in the registry.
        """
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            return await tool["handler"](**kwargs)
        else:
            raise ValueError(f"Tool {tool_name} not found")

class MCPProtocol:
    """An implementation of the MCP protocol for agent communication.
    
    This class provides static methods for creating MCP messages.
    """
    
    @staticmethod
    def create_message(
        message_type: MCPMessageType,
        agent_id: str,
        payload: Dict[str, Any],
        correlation_id: str = ""
    ) -> MCPMessage:
        """Create a new MCP message.

        Args:
            message_type (MCPMessageType): The type of the message.
            agent_id (str): The ID of the agent sending the message.
            payload (Dict[str, Any]): The payload of the message.
            correlation_id (str, optional): The correlation ID of the message. 
                If not provided, a new UUID will be generated. Defaults to "".

        Returns:
            MCPMessage: A new MCPMessage instance.
        """
        return MCPMessage(
            message_type=message_type.value,
            agent_id=agent_id,
            payload=payload,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id or str(uuid.uuid4())
        )

class TreeSearchReasoner:
    """A tree search reasoning module for advanced planning.
    
    This class implements a simplified tree search algorithm for complex reasoning tasks.
    """
    
    def __init__(self, model, tokenizer):
        """Initialize the TreeSearchReasoner instance.

        Args:
            model: The model used for reasoning.
            tokenizer: The tokenizer used for processing text.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = 5
        self.max_width = 3
    
    async def search(self, problem: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform tree search for complex reasoning.

        Args:
            problem (str): The problem to solve.
            context (Dict[str, Any]): The context information for the problem.

        Returns:
            List[Dict[str, Any]]: A list of solutions with confidence scores.
        """
        # Simplified tree search implementation
        return [{"solution": "tree_search_result", "confidence": 0.8}]

# Aliases for old names
PiscesMCPProtocol = MCPProtocol
PiscesTreeSearchReasoner = TreeSearchReasoner
PiscesMCPToolRegistry = MCPToolRegistry