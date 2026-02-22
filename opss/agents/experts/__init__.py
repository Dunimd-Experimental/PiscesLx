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

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ..base import (
    POPSSBaseAgent,
    POPSSAgentConfig,
    POPSSAgentContext,
    POPSSAgentResult,
    POPSSAgentThought,
    POPSSAgentState,
    POPSSAgentCapability,
)
from ..mcp_bridge import POPSSMCPBridge, POPSSMCPBridgeMixin

@dataclass
class POPSSCodeExpertConfig:
    language: str = "python"
    enable_test_generation: bool = True
    enable_documentation: bool = True
    code_style: str = "pep8"
    max_code_length: int = 10000

class POPSSCodeExpert(POPSSBaseAgent, POPSSMCPBridgeMixin):
    def __init__(self, config: POPSSAgentConfig, mcp_bridge: Optional[POPSSMCPBridge] = None):
        config.capabilities.add(POPSSAgentCapability.CODE_GENERATION)
        config.capabilities.add(POPSSAgentCapability.CODE_EXECUTION)
        
        super().__init__(config)
        POPSSMCPBridgeMixin.__init__(self, mcp_bridge=mcp_bridge)
        
        self.code_config = getattr(config, 'code_config', POPSSCodeExpertConfig())
    
    async def think(self, context: POPSSAgentContext) -> List[POPSSAgentThought]:
        thoughts = []
        
        thought1 = POPSSAgentThought(
            thought_id=f"thought_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent_id,
            context_id=context.context_id,
            thought_type="analysis",
            content=f"Analyzing code generation request: {context.user_request[:100]}...",
            confidence=0.9,
            reasoning="Need to understand the requirements and determine the best approach",
        )
        thoughts.append(thought1)
        
        if self._mcp_bridge:
            thought2 = POPSSAgentThought(
                thought_id=f"thought_{uuid.uuid4().hex[:8]}",
                agent_id=self.agent_id,
                context_id=context.context_id,
                thought_type="tool_selection",
                content="Selecting appropriate code generation tools",
                confidence=0.85,
                reasoning="Need to use MCP tools for code execution and validation",
            )
            thoughts.append(thought2)
        
        return thoughts
    
    async def act(self, thought: POPSSAgentThought, context: POPSSAgentContext) -> Dict[str, Any]:
        if thought.thought_type == "analysis":
            return await self._generate_code(context)
        elif thought.thought_type == "tool_selection":
            return await self._validate_and_execute(context)
        else:
            return await self._generate_code(context)
    
    async def _generate_code(self, context: POPSSAgentContext) -> Dict[str, Any]:
        code = f"# Generated code for: {context.user_request[:50]}\n\n"
        code += "def main():\n"
        code += f"    # TODO: Implement {context.user_request[:100]}\n"
        code += "    pass\n\n"
        code += "if __name__ == \"__main__\":\n"
        code += "    main()\n"
        
        return {
            "action": "code_generation",
            "code": code,
            "language": self.code_config.language if hasattr(self.code_config, 'language') else "python",
        }
    
    async def _validate_and_execute(self, context: POPSSAgentContext) -> Dict[str, Any]:
        if not self._mcp_bridge:
            return {"action": "validation_skipped", "reason": "No MCP bridge configured"}
        
        try:
            tools = self._mcp_bridge.get_available_tools()
            tool_names = [t.tool_name for t in tools]
            
            return {
                "action": "tools_identified",
                "available_tools": tool_names[:5],
                "count": len(tool_names),
            }
        except Exception as e:
            return {"action": "tool_error", "error": str(e)}
    
    async def observe(self, action_result: Dict[str, Any], context: POPSSAgentContext) -> bool:
        if action_result.get("action") == "code_generation":
            return True
        
        if action_result.get("action") == "validation_skipped":
            return True
        
        if action_result.get("action") == "tools_identified":
            return True
        
        return False
    
    async def need_tool(self, context: POPSSAgentContext) -> bool:
        return True
    
    async def select_tool(self, context: POPSSAgentContext) -> Optional[str]:
        if not self._mcp_bridge:
            return None
        
        task_lower = context.user_request.lower()
        
        if "test" in task_lower:
            tools = self._mcp_bridge.search_tools("test")
            if tools:
                return tools[0].tool_name
        
        tools = self._mcp_bridge.search_tools("code")
        if tools:
            return tools[0].tool_name
        
        all_tools = self._mcp_bridge.get_available_tools()
        if all_tools:
            return all_tools[0].tool_name
        
        return None


class POPSSSearchExpert(POPSSBaseAgent, POPSSMCPBridgeMixin):
    def __init__(self, config: POPSSAgentConfig, mcp_bridge: Optional[POPSSMCPBridge] = None):
        config.capabilities.add(POPSSAgentCapability.WEB_SEARCH)
        config.capabilities.add(POPSSAgentCapability.RESEARCH)
        
        super().__init__(config)
        POPSSMCPBridgeMixin.__init__(self, mcp_bridge=mcp_bridge)
    
    async def think(self, context: POPSSAgentContext) -> List[POPSSAgentThought]:
        return [
            POPSSAgentThought(
                thought_id=f"thought_{uuid.uuid4().hex[:8]}",
                agent_id=self.agent_id,
                context_id=context.context_id,
                thought_type="query_analysis",
                content=f"Analyzing search query: {context.user_request[:100]}...",
                confidence=0.95,
                reasoning="Need to understand what information is being sought",
            ),
            POPSSAgentThought(
                thought_id=f"thought_{uuid.uuid4().hex[:8]}",
                agent_id=self.agent_id,
                context_id=context.context_id,
                thought_type="source_selection",
                content="Selecting appropriate information sources",
                confidence=0.8,
                reasoning="Need to identify best sources for the query",
            ),
        ]
    
    async def act(self, thought: POPSSAgentThought, context: POPSSAgentContext) -> Dict[str, Any]:
        if thought.thought_type == "query_analysis":
            return {
                "action": "query_parsed",
                "query": context.user_request,
                "entities": [],
                "keywords": context.user_request.split()[:5],
            }
        elif thought.thought_type == "source_selection":
            sources = ["web_search", "document_processor", "fetch"]
            return {
                "action": "sources_selected",
                "sources": sources,
            }
        else:
            return {"action": "search_completed", "result": "Search completed"}
    
    async def observe(self, action_result: Dict[str, Any], context: POPSSAgentContext) -> bool:
        return action_result.get("action") in ["query_parsed", "sources_selected"]
    
    async def need_tool(self, context: POPSSAgentContext) -> bool:
        return True
    
    async def select_tool(self, context: POPSSAgentContext) -> Optional[str]:
        if not self._mcp_bridge:
            return None
        
        tools = self._mcp_bridge.search_tools("search")
        if tools:
            return tools[0].tool_name
        
        return None


class POPSSFileExpert(POPSSBaseAgent, POPSSMCPBridgeMixin):
    def __init__(self, config: POPSSAgentConfig, mcp_bridge: Optional[POPSSMCPBridge] = None):
        config.capabilities.add(POPSSAgentCapability.FILE_OPERATIONS)
        
        super().__init__(config)
        POPSSMCPBridgeMixin.__init__(self, mcp_bridge=mcp_bridge)
    
    async def think(self, context: POPSSAgentContext) -> List[POPSSAgentThought]:
        return [
            POPSSAgentThought(
                thought_id=f"thought_{uuid.uuid4().hex[:8]}",
                agent_id=self.agent_id,
                context_id=context.context_id,
                thought_type="operation_analysis",
                content=f"Analyzing file operation request: {context.user_request[:100]}...",
                confidence=0.9,
                reasoning="Need to determine the type of file operation",
            ),
        ]
    
    async def act(self, thought: POPSSAgentThought, context: POPSSAgentContext) -> Dict[str, Any]:
        return {
            "action": "file_operation_planned",
            "operation": "read",
            "path": context.user_request,
        }
    
    async def observe(self, action_result: Dict[str, Any], context: POPSSAgentContext) -> bool:
        return True
    
    async def need_tool(self, context: POPSSAgentContext) -> bool:
        return True
    
    async def select_tool(self, context: POPSSAgentContext) -> Optional[str]:
        return None


class POPSSAnalysisExpert(POPSSBaseAgent, POPSSMCPBridgeMixin):
    def __init__(self, config: POPSSAgentConfig, mcp_bridge: Optional[POPSSMCPBridge] = None):
        config.capabilities.add(POPSSAgentCapability.DATA_ANALYSIS)
        config.capabilities.add(POPSSAgentCapability.REASONING)
        
        super().__init__(config)
        POPSSMCPBridgeMixin.__init__(self, mcp_bridge=mcp_bridge)
    
    async def think(self, context: POPSSAgentContext) -> List[POPSSAgentThought]:
        return [
            POPSSAgentThought(
                thought_id=f"thought_{uuid.uuid4().hex[:8]}",
                agent_id=self.agent_id,
                context_id=context.context_id,
                thought_type="data_analysis",
                content=f"Analyzing data: {context.user_request[:100]}...",
                confidence=0.9,
                reasoning="Need to understand the data and analysis requirements",
            ),
        ]
    
    async def act(self, thought: POPSSAgentThought, context: POPSSAgentContext) -> Dict[str, Any]:
        return {
            "action": "analysis_completed",
            "result": f"Analysis of: {context.user_request[:50]}",
            "findings": [],
        }
    
    async def observe(self, action_result: Dict[str, Any], context: POPSSAgentContext) -> bool:
        return True
    
    async def need_tool(self, context: POPSSAgentContext) -> bool:
        return False
    
    async def select_tool(self, context: POPSSAgentContext) -> Optional[str]:
        return None


class POPSSCreativeExpert(POPSSBaseAgent):
    def __init__(self, config: POPSSAgentConfig):
        config.capabilities.add(POPSSAgentCapability.CREATIVE_WRITING)
        config.capabilities.add(POPSSAgentCapability.TEXT_GENERATION)
        
        super().__init__(config)
    
    async def think(self, context: POPSSAgentContext) -> List[POPSSAgentThought]:
        return [
            POPSSAgentThought(
                thought_id=f"thought_{uuid.uuid4().hex[:8]}",
                agent_id=self.agent_id,
                context_id=context.context_id,
                thought_type="creative_planning",
                content="Planning creative content generation",
                confidence=0.9,
                reasoning="Need to understand the creative requirements",
            ),
        ]
    
    async def act(self, thought: POPSSAgentThought, context: POPSSAgentContext) -> Dict[str, Any]:
        return {
            "action": "creative_content_generated",
            "content": f"Creative content for: {context.user_request[:50]}",
            "style": "general",
        }
    
    async def observe(self, action_result: Dict[str, Any], context: POPSSAgentContext) -> bool:
        return True
    
    async def need_tool(self, context: POPSSAgentContext) -> bool:
        return False
    
    async def select_tool(self, context: POPSSAgentContext) -> Optional[str]:
        return None


class POPSSResearchExpert(POPSSBaseAgent, POPSSMCPBridgeMixin):
    def __init__(self, config: POPSSAgentConfig, mcp_bridge: Optional[POPSSMCPBridge] = None):
        config.capabilities.add(POPSSAgentCapability.RESEARCH)
        config.capabilities.add(POPSSAgentCapability.WEB_SEARCH)
        config.capabilities.add(POPSSAgentCapability.SUMMARIZATION)
        
        super().__init__(config)
        POPSSMCPBridgeMixin.__init__(self, mcp_bridge=mcp_bridge)
    
    async def think(self, context: POPSSAgentContext) -> List[POPSSAgentThought]:
        return [
            POPSSAgentThought(
                thought_id=f"thought_{uuid.uuid4().hex[:8]}",
                agent_id=self.agent_id,
                context_id=context.context_id,
                thought_type="research_planning",
                content="Planning research approach",
                confidence=0.9,
                reasoning="Need to design research strategy",
            ),
        ]
    
    async def act(self, thought: POPSSAgentThought, context: POPSSAgentContext) -> Dict[str, Any]:
        return {
            "action": "research_completed",
            "summary": f"Research findings for: {context.user_request[:50]}",
            "sources": [],
        }
    
    async def observe(self, action_result: Dict[str, Any], context: POPSSAgentContext) -> bool:
        return True
    
    async def need_tool(self, context: POPSSAgentContext) -> bool:
        return True
    
    async def select_tool(self, context: POPSSAgentContext) -> Optional[str]:
        if not self._mcp_bridge:
            return None
        
        tools = self._mcp_bridge.search_tools("research")
        if tools:
            return tools[0].tool_name
        
        return None


class POPSSToolExpert(POPSSBaseAgent, POPSSMCPBridgeMixin):
    def __init__(self, config: POPSSAgentConfig, mcp_bridge: Optional[POPSSMCPBridge] = None):
        config.capabilities.add(POPSSAgentCapability.TOOL_USE)
        
        super().__init__(config)
        POPSSMCPBridgeMixin.__init__(self, mcp_bridge=mcp_bridge)
    
    async def think(self, context: POPSSAgentContext) -> List[POPSSAgentThought]:
        return [
            POPSSAgentThought(
                thought_id=f"thought_{uuid.uuid4().hex[:8]}",
                agent_id=self.agent_id,
                context_id=context.context_id,
                thought_type="tool_discovery",
                content="Discovering available tools",
                confidence=0.9,
                reasoning="Need to identify appropriate tools for the task",
            ),
        ]
    
    async def act(self, thought: POPSSAgentThought, context: POPSSAgentContext) -> Dict[str, Any]:
        if not self._mcp_bridge:
            return {"action": "tool_execution", "result": "No tools available"}
        
        tools = self._mcp_bridge.get_available_tools()
        return {
            "action": "tool_execution",
            "tool_count": len(tools),
            "tools": [t.tool_name for t in tools[:5]],
        }
    
    async def observe(self, action_result: Dict[str, Any], context: POPSSAgentContext) -> bool:
        return True
    
    async def need_tool(self, context: POPSSAgentContext) -> bool:
        return True
    
    async def select_tool(self, context: POPSSAgentContext) -> Optional[str]:
        if not self._mcp_bridge:
            return None
        
        tools = self._mcp_bridge.get_available_tools()
        if tools:
            return tools[0].tool_name
        
        return None
