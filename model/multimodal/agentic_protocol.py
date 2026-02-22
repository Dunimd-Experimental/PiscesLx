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
Agentic Protocol Handler for PiscesL1 Model-Agent Communication.

This module provides the core protocol handling for model-agent interactions,
supporting the unified XML syntax defined in xml_utils.py.

Protocol Flow:
    Model Output: "...<agentic><ag>code_expert</ag><tool><tn>execute</tn><tp>{}</tp></tool></agentic>..."
                  ↓
    xml_parser.extract_agentic_calls()
                  ↓
    YvAgenticProtocol.process_model_output()
                  ↓
    Execute via Agent Orchestrator or Tool Executor
                  ↓
    Return result wrapped in <result> XML tags

Key Components:
    - YvAgenticProtocol: Main protocol handler
    - YvAgenticContext: Execution context container
    - YvAgenticResult: Standardized result container

See Also:
    - model.multimodal.xml_utils: XML parsing utilities
    - opss.agents: External expert agents
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .xml_utils import (
    YvMCPXMLParser,
    YvMCPXMLGenerator,
    YvParsedAgenticCall,
    YvAgenticToolCall,
    YvAgenticAgentCall,
    YvAgenticGroup,
)


class YvAgenticExecutionMode(Enum):
    SINGLE = "single"
    SWARM = "swarm"
    SEQUENTIAL = "sequential"


@dataclass
class YvAgenticContext:
    execution_id: str
    session_id: str
    mode: str
    start_time: float
    agent_calls: List[YvParsedAgenticCall]
    metadata: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class YvAgenticResult:
    success: bool
    output: Any
    execution_time: float
    execution_mode: str
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)


class YvAgenticProtocol:
    def __init__(self):
        self.parser = YvMCPXMLParser()
        self.generator = YvMCPXMLGenerator()
        self.contexts: Dict[str, YvAgenticContext] = {}
    
    def process_model_output(
        self,
        model_output: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Optional[YvAgenticContext]]:
        agentic_calls = self.parser.extract_agentic_calls(model_output)
        
        if not agentic_calls:
            return model_output, None
        
        execution_id = str(uuid.uuid4())[:8]
        session_id = session_id or str(uuid.uuid4())[:8]
        start_time = time.time()
        
        if len(agentic_calls) == 1:
            mode = agentic_calls[0].mode
        else:
            mode = "sequential"
        
        context = YvAgenticContext(
            execution_id=execution_id,
            session_id=session_id,
            mode=mode,
            start_time=start_time,
            agent_calls=agentic_calls,
            metadata=metadata or {}
        )
        
        self.contexts[execution_id] = context
        
        return model_output, context
    
    def get_pending_tool_calls(
        self,
        context: YvAgenticContext
    ) -> List[Tuple[YvAgenticAgentCall, List[YvAgenticToolCall]]]:
        pending = []
        
        for call in context.agent_calls:
            if call.agent_group:
                for agent_name in call.agent_group.agents:
                    agent_call = YvAgenticAgentCall(
                        agent_name=agent_name,
                        raw_match="",
                        start_pos=0,
                        end_pos=0
                    )
                    pending.append((agent_call, call.tools))
            else:
                for agent in call.agents:
                    pending.append((agent, call.tools))
        
        return pending
    
    def create_result_wrapper(
        self,
        result: YvAgenticResult
    ) -> str:
        return self.generator.generate_result_xml(
            success=result.success,
            result=result.output,
            agent_name=result.agent_name,
            error_message=result.error
        )
    
    def create_result_wrapper_with_metadata(
        self,
        result: YvAgenticResult
    ) -> str:
        result_xml = self.create_result_wrapper(result)
        
        if result.metadata:
            metadata_json = json.dumps(result.metadata)
            escaped_metadata = self.parser.escape_xml(metadata_json)
            return f'<agent_result execution_id="{context.execution_id if hasattr(result, "metadata") else ""}" mode="{result.execution_mode}">{result_xml}<metadata>{escaped_metadata}</metadata></agent_result>'
        
        return result_xml
    
    def merge_results(
        self,
        results: List[YvAgenticResult]
    ) -> YvAgenticResult:
        if not results:
            return YvAgenticResult(
                success=False,
                output=None,
                execution_time=0.0,
                execution_mode="merge",
                error="No results to merge"
            )
        
        all_success = all(r.success for r in results)
        
        merged_output = {
            "results": [r.output for r in results],
            "count": len(results),
            "all_successful": all_success
        }
        
        total_time = sum(r.execution_time for r in results)
        
        agent_names = set(r.agent_name for r in results if r.agent_name)
        tool_names = set(r.tool_name for r in results if r.tool_name)
        
        all_intermediate = []
        for r in results:
            all_intermediate.extend(r.intermediate_results)
        
        return YvAgenticResult(
            success=all_success,
            output=merged_output,
            execution_time=total_time,
            execution_mode="merged",
            metadata={
                "agent_names": list(agent_names),
                "tool_names": list(tool_names),
                "intermediate_count": len(all_intermediate)
            },
            intermediate_results=all_intermediate
        )
    
    def extract_tools_from_call(
        self,
        call: YvParsedAgenticCall
    ) -> List[Dict[str, Any]]:
        tools = []
        
        for tool in call.tools:
            tools.append({
                "name": tool.tool_name,
                "parameters": tool.parameters,
                "priority": tool.priority,
                "fallback": tool.fallback
            })
        
        return tools
    
    def get_agent_names(
        self,
        call: YvParsedAgenticCall
    ) -> List[str]:
        if call.agent_group:
            return call.agent_group.agents
        
        return [agent.agent_name for agent in call.agents]
    
    def validate_call(
        self,
        call: YvParsedAgenticCall
    ) -> Tuple[bool, Optional[str]]:
        if not call.agents and not call.agent_group:
            return False, "No agent or agent group specified"
        
        return True, None
    
    def get_context(
        self,
        execution_id: str
    ) -> Optional[YvAgenticContext]:
        return self.contexts.get(execution_id)
    
    def remove_context(
        self,
        execution_id: str
    ) -> Optional[YvAgenticContext]:
        return self.contexts.pop(execution_id, None)
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "active_contexts": len(self.contexts),
            "context_ids": list(self.contexts.keys())
        }


protocol = YvAgenticProtocol()
