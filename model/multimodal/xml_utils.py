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
Unified XML utilities for MCP and Agent system.

This module provides XML parsing, escaping, and manipulation utilities
for model-agent communication with support for:
- Agent calls: <agentic><ag>agent_name</ag>...</agentic>
- Tool calls: <agentic><ag>agent</ag><tool><tn>tool_name</tn><tp>params</tp></tool></agentic>
- Multi-agent collaboration: <agentic><ag_group><ag>agent1</ag><ag>agent2</ag></ag_group>...</agentic>

Syntax Examples:
    Basic agent call:
        <agentic><ag>code_expert</ag></agentic>
    
    Agent with tool:
        <agentic>
            <ag>code_expert</ag>
            <tool><tn>execute</tn><tp>{"code": "print(1+1)", "language": "python"}</tp></tool>
        </agentic>
    
    Multi-agent swarm:
        <agentic mode="swarm">
            <ag_group>
                <ag>research_expert</ag>
                <ag>code_expert</ag>
                <ag>review_expert</ag>
            </ag_group>
            <tool><tn>search</tn><tp>{"query": "quicksort algorithm"}</tp></tool>
        </agentic>
"""

import re
import html
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class YvXMLMode(Enum):
    SINGLE = "single"
    SWARM = "swarm"
    SEQUENTIAL = "sequential"


@dataclass
class YvAgenticToolCall:
    tool_name: str
    parameters: Dict[str, Any]
    raw_match: str
    start_pos: int
    end_pos: int
    priority: Optional[str] = None
    fallback: Optional[str] = None


@dataclass
class YvAgenticAgentCall:
    agent_name: str
    raw_match: str
    start_pos: int
    end_pos: int
    timeout: Optional[str] = None
    context: Optional[str] = None


@dataclass
class YvAgenticGroup:
    agents: List[str]
    mode: str
    raw_match: str
    start_pos: int
    end_pos: int


@dataclass
class YvParsedAgenticCall:
    raw_match: str
    start_pos: int
    end_pos: int
    mode: str = "single"
    agents: List[YvAgenticAgentCall] = field(default_factory=list)
    tools: List[YvAgenticToolCall] = field(default_factory=list)
    agent_group: Optional[YvAgenticGroup] = None
    context_refs: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


class YvMCPXMLParser:
    def __init__(self):
        self._setup_patterns()
    
    def _setup_patterns(self):
        self.agent_pattern = re.compile(
            r'<agentic(?:\s+mode="([^"]*)")?(?:\s+timeout="([^"]*)")?>(.*?)</agentic>',
            re.DOTALL
        )
        
        self.agent_tag_pattern = re.compile(
            r'<ag(?:\s+context="([^"]*)")?>(.*?)</ag>',
            re.DOTALL
        )
        
        self.tool_pattern = re.compile(
            r'<tool(?:\s+priority="([^"]*)")?(?:\s+fallback="([^"]*)")?>(.*?)</tool>',
            re.DOTALL
        )
        
        self.tool_name_pattern = re.compile(r'<tn>(.*?)</tn>')
        
        self.tool_param_pattern = re.compile(r'<tp>(.*?)</tp>')
        
        self.agent_group_pattern = re.compile(
            r'<ag_group>(.*?)</ag_group>',
            re.DOTALL
        )
        
        self.context_ref_pattern = re.compile(
            r'<context_ref>(.*?)</context_ref>',
            re.DOTALL
        )
        
        self.xml_tag_pattern = re.compile(r'<[^>]+>')
        
        self.param_pattern = re.compile(r'<ap(\d+)>(.+?)</ap\1>', re.DOTALL)
        
        self.legacy_agent_pattern = re.compile(
            r'<agentic><an>(.+?)</an>(.*?)</agentic>',
            re.DOTALL
        )
    
    def extract_agentic_calls(self, text: str) -> List[PiscesLxParsedAgenticCall]:
        agentic_calls = []
        
        for match in self.agent_pattern.finditer(text):
            mode = match.group(1) or "single"
            timeout = match.group(2)
            content = match.group(3)
            
            call = PiscesLxParsedAgenticCall(
                raw_match=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
                mode=mode
            )
            
            for agent_match in self.agent_tag_pattern.finditer(content):
                agent_name = agent_match.group(2).strip()
                agent_context = agent_match.group(1)
                
                call.agents.append(PiscesLxAgenticAgentCall(
                    agent_name=agent_name,
                    raw_match=agent_match.group(0),
                    start_pos=match.start() + agent_match.start(),
                    end_pos=match.start() + agent_match.end(),
                    timeout=timeout,
                    context=agent_context
                ))
            
            for tool_match in self.tool_pattern.finditer(content):
                tool_content = tool_match.group(3)
                priority = tool_match.group(1)
                fallback = tool_match.group(2)
                
                tn_match = self.tool_name_pattern.search(tool_content)
                tp_match = self.tool_param_pattern.search(tool_content)
                
                if tn_match:
                    tool_name = tn_match.group(1).strip()
                    param_text = tp_match.group(1).strip() if tp_match else "{}"
                    
                    try:
                        parameters = json.loads(param_text)
                    except json.JSONDecodeError:
                        parameters = {"raw": param_text}
                    
                    call.tools.append(PiscesLxAgenticToolCall(
                        tool_name=tool_name,
                        parameters=parameters,
                        raw_match=tool_match.group(0),
                        start_pos=match.start() + tool_match.start(),
                        end_pos=match.start() + tool_match.end(),
                        priority=priority,
                        fallback=fallback
                    ))
            
            ag_group_match = self.agent_group_pattern.search(content)
            if ag_group_match:
                group_content = ag_group_match.group(1)
                group_agents = []
                for agent_match in self.agent_tag_pattern.finditer(group_content):
                    group_agents.append(agent_match.group(2).strip())
                
                call.agent_group = PiscesLxAgenticGroup(
                    agents=group_agents,
                    mode=mode,
                    raw_match=ag_group_match.group(0),
                    start_pos=match.start() + ag_group_match.start(),
                    end_pos=match.start() + ag_group_match.end()
                )
                call.agents = []
            
            context_ref_match = self.context_ref_pattern.search(content)
            if context_ref_match:
                call.context_refs.append(context_ref_match.group(1))
            
            if not call.tools and not call.agent_group:
                try:
                    parameters = json.loads(content.strip())
                    call.parameters = parameters
                except json.JSONDecodeError:
                    pass
            
            agentic_calls.append(call)
        
        if not agentic_calls:
            for match in self.legacy_agent_pattern.finditer(text):
                tool_name = match.group(1).strip()
                params_text = match.group(2).strip()
                
                parameters = {}
                for param_match in self.param_pattern.finditer(params_text):
                    param_index = param_match.group(1)
                    param_value = param_match.group(2).strip()
                    try:
                        parameters[f"ap{param_index}"] = json.loads(param_value)
                    except json.JSONDecodeError:
                        parameters[f"ap{param_index}"] = param_value
                
                legacy_call = PiscesLxParsedAgenticCall(
                    raw_match=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                legacy_call.agents = [PiscesLxAgenticAgentCall(
                    agent_name=tool_name,
                    raw_match=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end()
                )]
                legacy_call.tools = [PiscesLxAgenticToolCall(
                    tool_name=tool_name,
                    parameters=parameters,
                    raw_match=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end()
                )]
                agentic_calls.append(legacy_call)
        
        return agentic_calls
    
    def extract_legacy_calls(self, text: str) -> List[Dict[str, Any]]:
        calls = []
        for match in self.legacy_agent_pattern.finditer(text):
            tool_name = match.group(1).strip()
            params_text = match.group(2).strip()
            
            parameters = {}
            for param_match in self.param_pattern.finditer(params_text):
                param_index = param_match.group(1)
                param_value = param_match.group(2).strip()
                try:
                    parameters[f"ap{param_index}"] = json.loads(param_value)
                except json.JSONDecodeError:
                    parameters[f"ap{param_index}"] = param_value
            
            calls.append({
                "tool_name": tool_name,
                "parameters": parameters,
                "raw_match": match.group(0),
                "start_pos": match.start(),
                "end_pos": match.end()
            })
        
        return calls
    
    def remove_agentic_tags(self, text: str, placeholder: str = "") -> str:
        return self.agent_pattern.sub(placeholder, text)
    
    def escape_xml(self, text: str) -> str:
        return html.escape(str(text), quote=True)
    
    def unescape_xml(self, text: str) -> str:
        return html.unescape(text)
    
    def strip_all_xml_tags(self, text: str) -> str:
        return self.xml_tag_pattern.sub('', text)
    
    def validate_xml_structure(self, text: str) -> Tuple[bool, Optional[str]]:
        try:
            if '<agentic>' in text and '</agentic>' in text:
                for match in self.agent_pattern.finditer(text):
                    content = match.group(3)
                    has_agent = self.agent_tag_pattern.search(content) or self.agent_group_pattern.search(content)
                    has_legacy = '<an>' in content and '</an>' in content
                    
                    if not has_agent and not has_legacy:
                        return False, "Missing agent tag <ag> or <an>"
                    
                    if has_agent:
                        for tool_match in self.tool_pattern.finditer(content):
                            tool_content = tool_match.group(3)
                            if not self.tool_name_pattern.search(tool_content):
                                return False, "Missing tool name tag <tn> in <tool>"
                    
                    if has_legacy:
                        if '<an>' not in text or '</an>' not in text:
                            return False, "Missing tool name tag <an>"
                
                agent_matches = list(self.agent_pattern.finditer(text))
                for match in agent_matches:
                    content = match.group(3)
                    agent_match = self.agent_tag_pattern.search(content)
                    if agent_match:
                        agent_name = agent_match.group(2).strip()
                        if not agent_name:
                            return False, "Empty agent name in agentic call"
            
            return True, None
        
        except Exception as e:
            return False, f"XML validation error: {str(e)}"
    
    def parse_tool_parameters(self, param_text: str) -> Dict[str, Any]:
        try:
            return json.loads(param_text)
        except json.JSONDecodeError:
            return {"raw": param_text}


class YvMCPXMLGenerator:
    def __init__(self):
        self.parser = YvMCPXMLParser()
    
    def generate_agentic_call(
        self,
        agent_name: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        mode: str = "single",
        timeout: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        escaped_agent = self.parser.escape_xml(agent_name)
        
        agent_attrs = ""
        if timeout:
            agent_attrs = f' timeout="{timeout}"'
        if context:
            agent_attrs += f' context="{context}"'
        
        agent_xml = f"<ag{agent_attrs}>{escaped_agent}</ag>"
        
        tools_xml = ""
        if tools:
            for tool in tools:
                tn = self.parser.escape_xml(tool.get("name", ""))
                tp = self.parser.escape_xml(json.dumps(tool.get("params", {})))
                
                priority = tool.get("priority")
                fallback = tool.get("fallback")
                
                attrs = ""
                if priority:
                    attrs += f' priority="{priority}"'
                if fallback:
                    attrs += f' fallback="{fallback}"'
                
                tools_xml += f"<tool{attrs}><tn>{tn}</tn><tp>{tp}</tp></tool>"
        
        attrs = ""
        if mode and mode != "single":
            attrs = f' mode="{mode}"'
        if timeout:
            attrs += f' timeout="{timeout}"'
        
        return f'<agentic{attrs}>{agent_xml}{tools_xml}</agentic>'
    
    def generate_swarm_call(
        self,
        agent_names: List[str],
        tools: Optional[List[Dict[str, Any]]] = None,
        mode: str = "swarm"
    ) -> str:
        agents_xml = ""
        for name in agent_names:
            escaped = self.parser.escape_xml(name)
            agents_xml += f"<ag>{escaped}</ag>"
        
        ag_group_xml = f"<ag_group>{agents_xml}</ag_group>"
        
        tools_xml = ""
        if tools:
            for tool in tools:
                tn = self.parser.escape_xml(tool.get("name", ""))
                tp = self.parser.escape_xml(json.dumps(tool.get("params", {})))
                tools_xml += f"<tool><tn>{tn}</tn><tp>{tp}</tp></tool>"
        
        return f'<agentic mode="{mode}">{ag_group_xml}{tools_xml}</agentic>'
    
    def generate_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        priority: Optional[str] = None,
        fallback: Optional[str] = None
    ) -> str:
        escaped_tool = self.parser.escape_xml(tool_name)
        escaped_params = self.parser.escape_xml(json.dumps(parameters))
        
        attrs = ""
        if priority:
            attrs += f' priority="{priority}"'
        if fallback:
            attrs += f' fallback="{fallback}"'
        
        return f'<tool{attrs}><tn>{escaped_tool}</tn><tp>{escaped_params}</tp></tool>'
    
    def generate_result_xml(
        self,
        success: bool,
        result: Any,
        agent_name: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> str:
        if success:
            escaped_result = self.parser.escape_xml(json.dumps(result) if isinstance(result, (dict, list)) else str(result))
            if agent_name:
                return f'<result success="true" agent="{agent_name}"><data>{escaped_result}</data></result>'
            return f'<result success="true"><data>{escaped_result}</data></result>'
        else:
            escaped_error = self.parser.escape_xml(error_message or "Unknown error")
            if agent_name:
                return f'<result success="false" agent="{agent_name}"><error>{escaped_error}</error></result>'
            return f'<result success="false"><error>{escaped_error}</error></result>'


xml_parser = YvMCPXMLParser()
xml_generator = YvMCPXMLGenerator()
