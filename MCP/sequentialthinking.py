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

import json
import logging
from datetime import datetime
from .simple_mcp import register_tool
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SequentialThinkingTool:
    """
    A tool for dynamic problem-solving through structured thinking.
    It supports revision of thoughts and branching of thought processes.
    """
    
    def __init__(self):
        """
        Initialize the SequentialThinkingTool instance.
        Sets up the tool's name, description, and initializes thought history and branches.
        """
        self.name = "sequentialthinking"
        self.description = "Dynamic problem-solving through structured thinking with revision and branching support"
        self.thought_history = []
        self.branches = {}
        
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema for thought data.
        
        Returns:
            Dict[str, Any]: A dictionary representing the JSON schema for thought data.
        """
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your current thinking step. This can include regular analytical steps, revisions of previous thoughts, questions about previous decisions, realizations about needing more analysis, changes in approach, hypothesis generation, or hypothesis verification."
                },
                "thought_number": {
                    "type": "number",
                    "description": "Current number in sequence (can go beyond initial total if needed)"
                },
                "total_thoughts": {
                    "type": "number",
                    "description": "Current estimate of thoughts needed (can be adjusted up/down)"
                },
                "next_thought_needed": {
                    "type": "boolean",
                    "description": "True if you need more thinking, even if at what seemed like the end"
                },
                "is_revision": {
                    "type": "boolean",
                    "description": "A boolean indicating if this thought revises previous thinking",
                    "default": False
                },
                "revises_thought": {
                    "type": "number",
                    "description": "If is_revision is true, which thought number is being reconsidered"
                },
                "branch_from_thought": {
                    "type": "number",
                    "description": "If branching, which thought number is the branching point"
                },
                "branch_id": {
                    "type": "string",
                    "description": "Identifier for the current branch (if any)"
                },
                "needs_more_thoughts": {
                    "type": "boolean",
                    "description": "If reaching end but realizing more thoughts needed"
                }
            },
            "required": ["thought", "thought_number", "total_thoughts", "next_thought_needed"]
        }
    
    def _validate_thought_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the thought data against the required fields and constraints.
        
        Args:
            data (Dict[str, Any]): The raw thought data to be validated.
            
        Returns:
            Dict[str, Any]: The validated and formatted thought data.
            
        Raises:
            ValueError: If any required field is missing or if the values violate constraints.
        """
        required_fields = ["thought", "thought_number", "total_thoughts", "next_thought_needed"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        thought = str(data["thought"])
        thought_number = int(data["thought_number"])
        total_thoughts = int(data["total_thoughts"])
        next_thought_needed = bool(data["next_thought_needed"])
        
        if thought_number <= 0:
            raise ValueError("thought_number must be positive")
        
        if total_thoughts <= 0:
            raise ValueError("total_thoughts must be positive")
        
        # Adjust total_thoughts if thought_number exceeds it
        if thought_number > total_thoughts:
            total_thoughts = thought_number
        
        validated_data = {
            "thought": thought,
            "thoughtNumber": thought_number,
            "totalThoughts": total_thoughts,
            "nextThoughtNeeded": next_thought_needed,
            "created": datetime.now().isoformat()
        }
        
        # Optional fields
        optional_fields = ["is_revision", "revises_thought", "branch_from_thought", "branch_id", "needs_more_thoughts"]
        for field in optional_fields:
            if field in data and data[field] is not None:
                key_mapping = {
                    "is_revision": "isRevision",
                    "revises_thought": "revisesThought",
                    "branch_from_thought": "branchFromThought",
                    "branch_id": "branchId",
                    "needs_more_thoughts": "needsMoreThoughts"
                }
                validated_data[key_mapping.get(field, field)] = data[field]
        
        return validated_data
    
    def _format_thought(self, thought_data: Dict[str, Any]) -> str:
        """
        Format a single thought for display purposes.
        
        Args:
            thought_data (Dict[str, Any]): The validated thought data to be formatted.
            
        Returns:
            str: A formatted string representing the thought in a box-like structure.
        """
        thought = thought_data["thought"]
        thought_num = thought_data["thoughtNumber"]
        total_thoughts = thought_data["totalThoughts"]
        
        prefix = "💭 Thought"
        context = ""
        
        if thought_data.get("isRevision"):
            prefix = "🔄 Revision"
            revises = thought_data.get("revisesThought")
            context = f" (revising thought {revises})"
        elif thought_data.get("branchFromThought"):
            prefix = "🌿 Branch"
            branch_id = thought_data.get("branchId")
            branch_from = thought_data.get("branchFromThought")
            context = f" (from thought {branch_from}, ID: {branch_id})"
        
        header = f"{prefix} {thought_num}/{total_thoughts}{context}"
        
        # Create a simple box format
        border = "─" * max(len(header) + 2, 40)
        formatted = f"┌{border}┐\n"
        formatted += f"│ {header.ljust(len(border))} │\n"
        formatted += f"├{border}┤\n"
        formatted += f"│ {thought.ljust(len(border))} │\n"
        formatted += f"└{border}┘"
        
        return formatted
    
    def _process_thought(self, thought_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a thought, validate it, add it to the history, and handle branching.
        
        Args:
            thought_data (Dict[str, Any]): The raw thought data to be processed.
            
        Returns:
            Dict[str, Any]: A dictionary containing the processing result, 
                            including success status and relevant data.
        """
        try:
            validated_data = self._validate_thought_data(thought_data)
            
            # Add to history
            self.thought_history.append(validated_data)
            
            # Handle branching
            branch_id = validated_data.get("branchId")
            branch_from = validated_data.get("branchFromThought")
            
            if branch_from and branch_id:
                if branch_id not in self.branches:
                    self.branches[branch_id] = []
                self.branches[branch_id].append(validated_data)
            
            # Log the formatted thought
            formatted_thought = self._format_thought(validated_data)
            logger.info("\n" + formatted_thought)
            
            return {
                "success": True,
                "data": {
                    "thoughtNumber": validated_data["thoughtNumber"],
                    "totalThoughts": validated_data["totalThoughts"],
                    "nextThoughtNeeded": validated_data["nextThoughtNeeded"],
                    "branches": list(self.branches.keys()),
                    "thoughtHistoryLength": len(self.thought_history),
                    "isRevision": validated_data.get("isRevision", False),
                    "revisesThought": validated_data.get("revisesThought"),
                    "branchId": validated_data.get("branchId"),
                    "created": validated_data["created"]
                }
            }
            
        except ValueError as e:
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def _get_thought_history(self) -> Dict[str, Any]:
        """
        Get the complete thought history and branch information.
        
        Returns:
            Dict[str, Any]: A dictionary containing the success status and thought history data.
        """
        return {
            "success": True,
            "data": {
                "thoughts": self.thought_history,
                "branches": self.branches,
                "total_thoughts": len(self.thought_history),
                "active_branches": len(self.branches)
            }
        }
    
    def _clear_history(self) -> Dict[str, Any]:
        """
        Clear the thought history and all branches.
        
        Returns:
            Dict[str, Any]: A dictionary containing the success status and confirmation message.
        """
        self.thought_history.clear()
        self.branches.clear()
        
        return {
            "success": True,
            "data": {
                "message": "Thought history cleared",
                "total_thoughts": 0,
                "branches": 0
            }
        }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute a sequential thinking operation based on the specified operation type.
        
        Args:
            **kwargs: Keyword arguments containing the operation type and thought data.
                      The 'operation' key specifies the operation type, defaulting to 'process_thought'.
            
        Returns:
            Dict[str, Any]: A dictionary containing the result of the operation, 
                            including success status and relevant data.
        """
        # Check for special operations
        operation = kwargs.pop("operation", "process_thought")
        
        if operation == "get_history":
            return self._get_thought_history()
        elif operation == "clear":
            return self._clear_history()
        elif operation == "process_thought":
            return self._process_thought(kwargs)
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }

# Register the tool
thinking_tool = SequentialThinkingTool()
register_tool(
    thinking_tool.name,
    thinking_tool.description,
    thinking_tool.get_schema(),
    thinking_tool.execute
)