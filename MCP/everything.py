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

import os
import json
import time
import random
import asyncio
from pathlib import Path
from datetime import datetime
from .simple_mcp import register_tool
from typing import Dict, Any, List, Optional

class EverythingTool:
    """
    A comprehensive example tool that provides multiple utilities.
    This tool supports various operations such as echo, math calculations, 
    long-running tasks, and more.
    """
    
    def __init__(self):
        """
        Initialize the EverythingTool instance.
        Sets up basic attributes including the tool name, description, 
        resource counter, and subscription set.
        """
        self.name = "everything"
        self.description = "Comprehensive example tool with echo, math, operations, and more utilities"
        self.resource_counter = 1
        self.subscriptions = set()
        
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema definition for the tool's input parameters.
        
        Returns:
            Dict[str, Any]: A dictionary containing the JSON schema for the tool's input.
        """
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation to perform",
                    "enum": [
                        "echo", "add", "long_operation", "print_env", "sample_llm",
                        "get_image", "annotated_message", "get_resource_ref", "elicit",
                        "get_resource_links", "structured_content"
                    ]
                },
                "message": {
                    "type": "string",
                    "description": "Message to echo (for echo operation)"
                },
                "a": {
                    "type": "number",
                    "description": "First number (for add operation)"
                },
                "b": {
                    "type": "number",
                    "description": "Second number (for add operation)"
                },
                "duration": {
                    "type": "number",
                    "description": "Duration in seconds (for long_operation)",
                    "default": 10
                },
                "steps": {
                    "type": "number",
                    "description": "Number of steps (for long_operation)",
                    "default": 5
                },
                "prompt": {
                    "type": "string",
                    "description": "Prompt for LLM (for sample_llm)"
                },
                "max_tokens": {
                    "type": "number",
                    "description": "Max tokens for LLM (for sample_llm)",
                    "default": 100
                },
                "message_type": {
                    "type": "string",
                    "description": "Type of annotated message",
                    "enum": ["error", "success", "debug"]
                },
                "include_image": {
                    "type": "boolean",
                    "description": "Include image in annotated message",
                    "default": False
                },
                "resource_id": {
                    "type": "number",
                    "description": "Resource ID (1-100) for resource reference",
                    "minimum": 1,
                    "maximum": 100
                },
                "count": {
                    "type": "number",
                    "description": "Number of resource links to return (1-10)",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10
                },
                "location": {
                    "type": "string",
                    "description": "City name or zip code (for structured_content)"
                }
            },
            "required": ["operation"]
        }
    
    def _echo(self, message: str) -> Dict[str, Any]:
        """
        Echo a message back with additional metadata.
        
        Args:
            message (str): The message to be echoed.
            
        Returns:
            Dict[str, Any]: A dictionary containing the success status, echoed message, 
                           its length, and a timestamp.
        """
        return {
            "success": True,
            "data": {
                "message": message,
                "length": len(message),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _add(self, a: float, b: float) -> Dict[str, Any]:
        """
        Add two numbers and return the result.
        
        Args:
            a (float): The first number to add.
            b (float): The second number to add.
            
        Returns:
            Dict[str, Any]: A dictionary containing the success status, input numbers, 
                           their sum, and the operation type.
        """
        return {
            "success": True,
            "data": {
                "a": a,
                "b": b,
                "sum": a + b,
                "operation": "addition"
            }
        }
    
    def _long_operation(self, duration: int = 10, steps: int = 5) -> Dict[str, Any]:
        """
        Simulate a long-running operation by sleeping in steps and recording progress.
        
        Args:
            duration (int, optional): Total duration of the operation in seconds. Defaults to 10.
            steps (int, optional): Number of steps to divide the operation into. Defaults to 5.
            
        Returns:
            Dict[str, Any]: A dictionary containing the success status, operation details, 
                           and progress results.
        """
        step_duration = duration / steps
        results = []
        
        for i in range(steps):
            step_num = i + 1
            progress = (step_num / steps) * 100
            
            # Simulate work by sleeping for the calculated step duration
            time.sleep(step_duration)
            
            results.append({
                "step": step_num,
                "progress": f"{progress:.1f}%",
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "success": True,
            "data": {
                "duration": duration,
                "steps": steps,
                "results": results,
                "completed": True
            }
        }
    
    def _print_env(self) -> Dict[str, Any]:
        """
        Print environment variables with sensitive information redacted.
        
        Returns:
            Dict[str, Any]: A dictionary containing the success status, filtered environment variables, 
                           their count, hostname, and username.
        """
        env_vars = dict(os.environ)
        
        # Define sensitive keywords to filter sensitive environment variables
        sensitive_keys = {'password', 'secret', 'token', 'key', 'auth'}
        filtered_env = {}
        
        for key, value in env_vars.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                filtered_env[key] = "[REDACTED]"
            else:
                filtered_env[key] = value
        
        return {
            "success": True,
            "data": {
                "environment": filtered_env,
                "count": len(filtered_env),
                "hostname": os.getenv('COMPUTERNAME', 'unknown'),
                "user": os.getenv('USERNAME', 'unknown')
            }
        }
    
    def _sample_llm(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """
        Simulate a response from a large language model (LLM).
        
        Args:
            prompt (str): The input prompt for the simulated LLM.
            max_tokens (int, optional): Maximum number of tokens for the response. Defaults to 100.
            
        Returns:
            Dict[str, Any]: A dictionary containing the success status, prompt, max tokens, 
                           simulated response, tokens used, and model name.
        """
        responses = [
            "That's a great question! Let me think about that...",
            "Based on my analysis, here's what I found:",
            "This is an interesting topic. Here's my perspective:",
            "Let me break this down for you:",
            "From what I understand, the key points are:"
        ]
        
        # Generate a mock response using a random starting phrase and the input prompt
        response_start = random.choice(responses)
        mock_content = f"{response_start} {prompt[:50]}... This would be the LLM's detailed response based on your prompt. The response is limited to {max_tokens} tokens as requested."
        
        return {
            "success": True,
            "data": {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "response": mock_content,
                "tokens_used": min(max_tokens, len(mock_content.split())),
                "model": "mock-llm-v1"
            }
        }
    
    def _get_image(self) -> Dict[str, Any]:
        """
        Get a tiny 1x1 pixel PNG image in base64 format.
        
        Returns:
            Dict[str, Any]: A dictionary containing the success status, image data, 
                           format, dimensions, and size.
        """
        # Base64-encoded 1x1 pixel PNG image
        tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        return {
            "success": True,
            "data": {
                "image_data": tiny_png,
                "format": "png",
                "width": 1,
                "height": 1,
                "size_bytes": len(tiny_png)
            }
        }
    
    def _annotated_message(self, message_type: str = "success", include_image: bool = False) -> Dict[str, Any]:
        """
        Generate an annotated message with different types (success, error, debug).
        
        Args:
            message_type (str, optional): Type of the message. Defaults to "success".
            include_image (bool, optional): Whether to include an image in the message. Defaults to False.
            
        Returns:
            Dict[str, Any]: A dictionary containing the success status and the annotated message data.
        """
        messages = {
            "success": {
                "title": "Operation Successful",
                "content": "The operation completed successfully without any issues.",
                "icon": "✅"
            },
            "error": {
                "title": "Operation Failed",
                "content": "There was an error while processing the operation.",
                "icon": "❌"
            },
            "debug": {
                "title": "Debug Information",
                "content": "Here is some debug information for troubleshooting.",
                "icon": "🐛"
            }
        }
        
        message = messages.get(message_type, messages["success"])
        
        result = {
            "title": message["title"],
            "content": message["content"],
            "type": message_type,
            "icon": message["icon"],
            "timestamp": datetime.now().isoformat()
        }
        
        if include_image:
            result["image"] = "sample_image_data_here"
        
        return {
            "success": True,
            "data": result
        }
    
    def _get_resource_ref(self, resource_id: int) -> Dict[str, Any]:
        """
        Get a reference to a resource based on the given resource ID.
        
        Args:
            resource_id (int): The ID of the resource (must be between 1 and 100).
        Returns:
            Dict[str, Any]: A dictionary containing the success status and resource reference data, 
                           or an error message if the ID is invalid.
        """
        if not 1 <= resource_id <= 100:
            return {
                "success": False,
                "error": "Resource ID must be between 1 and 100"
            }
        
        return {
            "success": True,
            "data": {
                "resource_id": resource_id,
                "uri": f"resource://example/{resource_id}",
                "title": f"Example Resource {resource_id}",
                "description": f"This is example resource number {resource_id}",
                "created": datetime.now().isoformat(),
                "content": f"Content for resource {resource_id}"
            }
        }
    
    def _elicit(self) -> Dict[str, Any]:
        """
        Start an elicitation process by providing a set of questions.
        
        Returns:
            Dict[str, Any]: A dictionary containing the success status, process information, 
                           questions, total question count, and guidance.
        """
        questions = [
            "What specific problem are you trying to solve?",
            "What is the current state of your system?",
            "What are the key requirements?",
            "What constraints do you have?",
            "What is your expected outcome?"
        ]
        
        return {
            "success": True,
            "data": {
                "process": "elicitation_started",
                "questions": questions,
                "total_questions": len(questions),
                "guidance": "Please answer these questions to help clarify your needs"
            }
        }
    
    def _get_resource_links(self, count: int = 3) -> Dict[str, Any]:
        """
        Get sample resource links.
        
        Args:
            count (int, optional): Number of resource links to return (1-10). Defaults to 3.
            
        Returns:
            Dict[str, Any]: A dictionary containing the success status and a list of resource links.
        """
        links = []
        for i in range(1, count + 1):
            links.append({
                "title": f"Resource {i}",
                "uri": f"resource://example/{i}",
                "description": f"Description for resource {i}",
                "type": "example"
            })
        
        return {
            "success": True,
            "data": {
                "links": links,
                "count": len(links)
            }
        }
    
    def _structured_content(self, location: str) -> Dict[str, Any]:
        """
        Get structured weather-like content based on the given location.
        
        Args:
            location (str): The city name or zip code to get weather data for.
            
        Returns:
            Dict[str, Any]: A dictionary containing the success status and structured weather data.
        """
        # Mock weather conditions
        weather_conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy"]
        
        # Use a simple hash to ensure consistent weather for the same location
        location_hash = sum(ord(c) for c in location) % len(weather_conditions)
        condition = weather_conditions[location_hash]
        
        # Generate mock temperature and humidity data
        temperature = random.randint(-10, 35)
        humidity = random.randint(20, 90)
        
        return {
            "success": True,
            "data": {
                "location": location,
                "temperature": temperature,
                "conditions": condition,
                "humidity": humidity,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the requested operation with the provided keyword arguments.
        
        Args:
            operation (str): The name of the operation to execute.
            **kwargs: Additional keyword arguments required by the operation.
            
        Returns:
            Dict[str, Any]: A dictionary containing the success status and operation result, 
                           or an error message if the operation is unknown or fails.
        """
        operations = {
            "echo": lambda: self._echo(kwargs.get("message", "Hello World")),
            "add": lambda: self._add(kwargs.get("a", 0), kwargs.get("b", 0)),
            "long_operation": lambda: self._long_operation(
                kwargs.get("duration", 10),
                kwargs.get("steps", 5)
            ),
            "print_env": self._print_env,
            "sample_llm": lambda: self._sample_llm(
                kwargs.get("prompt", "Hello"),
                kwargs.get("max_tokens", 100)
            ),
            "get_image": self._get_image,
            "annotated_message": lambda: self._annotated_message(
                kwargs.get("message_type", "success"),
                kwargs.get("include_image", False)
            ),
            "get_resource_ref": lambda: self._get_resource_ref(
                kwargs.get("resource_id", 1)
            ),
            "elicit": self._elicit,
            "get_resource_links": lambda: self._get_resource_links(
                kwargs.get("count", 3)
            ),
            "structured_content": lambda: self._structured_content(
                kwargs.get("location", "New York")
            )
        }
        
        if operation not in operations:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }
        
        try:
            return operations[operation]()
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Register the EverythingTool with the MCP system
everything_tool = EverythingTool()
register_tool(
    everything_tool.name,
    everything_tool.description,
    everything_tool.get_schema(),
    everything_tool.execute
)