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
from MCP import mcp
from pathlib import Path
from typing import Dict, Any, List, Optional

@mcp.tool()
def store_memory(key: str, value: Any, memory_file: str = "memory.json") -> Dict[str, Any]:
    """Store a value in memory with a given key."""
    try:
        memory_path = Path(memory_file)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing memory
        memory_data = {}
        if memory_path.exists():
            try:
                with open(memory_path, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
            except json.JSONDecodeError:
                memory_data = {}
        
        # Store the value
        memory_data[key] = value
        
        # Save to file
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "key": key,
            "message": f"Successfully stored value for key: {key}",
            "memory_file": str(memory_path.absolute())
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def retrieve_memory(key: str, memory_file: str = "memory.json") -> Dict[str, Any]:
    """Retrieve a value from memory by key."""
    try:
        memory_path = Path(memory_file)
        
        if not memory_path.exists():
            return {
                "success": False,
                "error": f"Memory file not found: {memory_file}",
                "key": key
            }
        
        # Load memory
        with open(memory_path, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        if key not in memory_data:
            return {
                "success": False,
                "error": f"Key not found: {key}",
                "available_keys": list(memory_data.keys())
            }
        
        return {
            "success": True,
            "key": key,
            "value": memory_data[key],
            "memory_file": str(memory_path.absolute())
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def list_memory(memory_file: str = "memory.json") -> Dict[str, Any]:
    """List all keys stored in memory."""
    try:
        memory_path = Path(memory_file)
        
        if not memory_path.exists():
            return {
                "success": True,
                "keys": [],
                "count": 0,
                "message": "Memory file does not exist yet",
                "memory_file": str(memory_path.absolute())
            }
        
        # Load memory
        with open(memory_path, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        keys = list(memory_data.keys())
        
        return {
            "success": True,
            "keys": keys,
            "count": len(keys),
            "memory_file": str(memory_path.absolute())
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def delete_memory(key: str, memory_file: str = "memory.json") -> Dict[str, Any]:
    """Delete a key from memory."""
    try:
        memory_path = Path(memory_file)
        
        if not memory_path.exists():
            return {
                "success": False,
                "error": f"Memory file not found: {memory_file}",
                "key": key
            }
        
        # Load memory
        with open(memory_path, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        if key not in memory_data:
            return {
                "success": False,
                "error": f"Key not found: {key}",
                "available_keys": list(memory_data.keys())
            }
        
        # Delete the key
        del memory_data[key]
        
        # Save back to file
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "key": key,
            "message": f"Successfully deleted key: {key}",
            "memory_file": str(memory_path.absolute())
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def clear_memory(memory_file: str = "memory.json") -> Dict[str, Any]:
    """Clear all memory by deleting the memory file."""
    try:
        memory_path = Path(memory_file)
        
        if not memory_path.exists():
            return {
                "success": True,
                "message": "Memory file does not exist",
                "memory_file": str(memory_path.absolute())
            }
        
        # Delete the memory file
        memory_path.unlink()
        
        return {
            "success": True,
            "message": "Memory cleared successfully",
            "memory_file": str(memory_path.absolute())
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def append_to_memory(key: str, value: Any, memory_file: str = "memory.json") -> Dict[str, Any]:
    """Append a value to an existing memory key (for lists)."""
    try:
        memory_path = Path(memory_file)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing memory
        memory_data = {}
        if memory_path.exists():
            try:
                with open(memory_path, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
            except json.JSONDecodeError:
                memory_data = {}
        
        # Get existing value or create new list
        existing_value = memory_data.get(key, [])
        if not isinstance(existing_value, list):
            existing_value = [existing_value]
        
        # Append new value
        existing_value.append(value)
        memory_data[key] = existing_value
        
        # Save to file
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "key": key,
            "message": f"Successfully appended value to key: {key}",
            "current_length": len(existing_value),
            "memory_file": str(memory_path.absolute())
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def get_memory_stats(memory_file: str = "memory.json") -> Dict[str, Any]:
    """Get statistics about the memory file."""
    try:
        memory_path = Path(memory_file)
        
        if not memory_path.exists():
            return {
                "success": True,
                "exists": False,
                "size_bytes": 0,
                "key_count": 0,
                "memory_file": str(memory_path.absolute())
            }
        
        # Load memory
        with open(memory_path, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        file_size = memory_path.stat().st_size
        
        return {
            "success": True,
            "exists": True,
            "size_bytes": file_size,
            "key_count": len(memory_data),
            "keys": list(memory_data.keys()),
            "memory_file": str(memory_path.absolute())
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }