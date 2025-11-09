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

"""
Unified XML utilities for MCP system.

This module provides XML parsing, escaping, and manipulation utilities
that can be used across the entire MCP system for consistent XML handling.
"""

import re
import html
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PiscesLxCoreMCPAgenticCall:
    """Represents a parsed agentic call from XML content."""
    tool_name: str
    parameters: Dict[str, Any]
    raw_match: str
    start_pos: int
    end_pos: int


class PiscesLxCoreMCPXMLParser:
    """Unified XML parser for MCP agent calls and content."""
    
    def __init__(self):
        """Initialize the XML parser with predefined patterns."""
        # Pattern for <agentic><an>tool_name</an><ap1>param1</ap1>...</agentic>
        self.agent_pattern = re.compile(r'<agentic><an>(.+?)</an>(.*?)</agentic>', re.DOTALL)
        # Pattern for individual parameters ap1, ap2, ap3, etc.
        self.param_pattern = re.compile(r'<ap(\d+)>(.+?)</ap\1>', re.DOTALL)
        # Pattern for any XML tags
        self.xml_tag_pattern = re.compile(r'<[^>]+>')
    
    def extract_agentic_calls(self, text: str) -> List[PiscesLxCoreMCPAgenticCall]:
        """
        Extract all <agentic> tags from text content.
        
        Args:
            text: Text containing agent tags
            
        Returns:
            List of parsed agentic calls
        """
        agent_calls = []
        
        for match in self.agent_pattern.finditer(text):
            tool_name = match.group(1).strip()
            params_text = match.group(2).strip()
            
            # Extract parameters
            parameters = {}
            for param_match in self.param_pattern.finditer(params_text):
                param_index = param_match.group(1)
                param_value = param_match.group(2).strip()
                parameters[f"ap{param_index}"] = param_value
            
            agent_call = PiscesLxCoreMCPAgenticCall(
                tool_name=tool_name,
                parameters=parameters,
                raw_match=match.group(0),
                start_pos=match.start(),
                end_pos=match.end()
            )
            agent_calls.append(agent_call)
        
        return agent_calls
    
    def remove_agentic_tags(self, text: str, placeholder: str = "") -> str:
        """
        Remove all <agentic> tags from text.
        
        Args:
            text: Text containing agent tags
            placeholder: Text to replace agent tags with
            
        Returns:
            Text with agent tags removed/replaced
        """
        return self.agent_pattern.sub(placeholder, text)
    
    def escape_xml(self, text: str) -> str:
        """
        Escape XML special characters in text.
        
        Args:
            text: Text to escape
            
        Returns:
            XML-escaped text
        """
        return html.escape(str(text), quote=True)
    
    def unescape_xml(self, text: str) -> str:
        """
        Unescape XML special characters in text.
        
        Args:
            text: XML-escaped text
            
        Returns:
            Unescaped text
        """
        return html.unescape(text)
    
    def strip_all_xml_tags(self, text: str) -> str:
        """
        Remove all XML tags from text.
        
        Args:
            text: Text containing XML tags
            
        Returns:
            Text with all XML tags removed
        """
        return self.xml_tag_pattern.sub('', text)
    
    def validate_xml_structure(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Basic validation of XML structure for agentic calls.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for basic XML structure
            if '<agentic>' in text and '</agentic>' in text:
                # Check for tool name
                if '<an>' not in text or '</an>' not in text:
                    return False, "Missing tool name tag <an>"
                
                # Validate tag nesting
                agent_matches = list(self.agent_pattern.finditer(text))
                for match in agent_matches:
                    full_match = match.group(0)
                    # Check if tool name is present and not empty
                    tool_name = match.group(1).strip()
                    if not tool_name:
                        return False, "Empty tool name in agentic call"
                    
                    # Check parameter tags are properly closed
                    params_text = match.group(2)
                    param_matches = list(self.param_pattern.finditer(params_text))
                    for param_match in param_matches:
                        param_content = param_match.group(2)
                        if not param_content.strip():
                            return False, f"Empty parameter ap{param_match.group(1)}"
            
            return True, None
            
        except Exception as e:
            return False, f"XML validation error: {str(e)}"


class PiscesLxCoreMCPXMLGenerator:
    """Unified XML generator for MCP system."""
    
    def __init__(self):
        """Initialize the XML generator."""
        self.parser = PiscesLxCoreMCPXMLParser()
    
    def generate_agentic_call(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        Generate an agentic call XML string.
        
        Args:
            tool_name: Name of the tool
            parameters: Dictionary of parameters
            
        Returns:
            XML string representing the agentic call
        """
        # Escape tool name
        escaped_tool = self.parser.escape_xml(tool_name)
        
        # Build parameters
        param_parts = []
        for key, value in parameters.items():
            if key.startswith('ap') and key[2:].isdigit():
                # Use the parameter index from the key
                param_index = key[2:]
                escaped_value = self.parser.escape_xml(str(value))
                param_parts.append(f'<ap{param_index}>{escaped_value}</ap{param_index}>')
            else:
                # Auto-assign parameter indices for non-standard keys
                param_index = len(param_parts) + 1
                escaped_value = self.parser.escape_xml(str(value))
                param_parts.append(f'<ap{param_index}>{escaped_value}</ap{param_index}>')
        
        params_xml = ''.join(param_parts)
        return f'<agentic><an>{escaped_tool}</an>{params_xml}</agentic>'
    
    def generate_result_xml(self, success: bool, result: Any, error_message: Optional[str] = None) -> str:
        """
        Generate a result XML string.
        
        Args:
            success: Whether the operation was successful
            result: Result data
            error_message: Error message if not successful
            
        Returns:
            XML string representing the result
        """
        if success:
            escaped_result = self.parser.escape_xml(str(result))
            return f'<result><success>true</success><data>{escaped_result}</data></result>'
        else:
            escaped_error = self.parser.escape_xml(error_message or "Unknown error")
            return f'<result><success>false</success><error>{escaped_error}</error></result>'


# Global instances for convenience
xml_parser = PiscesLxCoreMCPXMLParser()
xml_generator = PiscesLxCoreMCPXMLGenerator()