#!/usr/bin/env/python3

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
import uuid
from MCP import mcp
from datetime import datetime
from typing import Dict, Any, List, Optional

class ThoughtNode:
    """Represents a single thought in the thinking sequence."""
    
    def __init__(self, content: str, parent_id: str = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.parent_id = parent_id
        self.timestamp = datetime.now().isoformat()
        self.children = []

class SequentialThinking:
    """Manages sequential thinking sessions."""
    
    def __init__(self):
        self.sessions = {}

    def create_session(self, initial_thought: str) -> str:
        """Create a new thinking session with an initial thought."""
        session_id = str(uuid.uuid4())
        root_thought = ThoughtNode(initial_thought)
        self.sessions[session_id] = {
            'root': root_thought,
            'thoughts': {root_thought.id: root_thought},
            'current_focus': root_thought.id,
            'created_at': datetime.now().isoformat()
        }
        return session_id

    def add_thought(self, session_id: str, content: str, parent_id: str = None) -> Dict[str, Any]:
        """Add a new thought to a session."""
        if session_id not in self.sessions:
            return {"success": False, "error": "Session not found"}
        
        session = self.sessions[session_id]
        
        if parent_id is None:
            parent_id = session['current_focus']
        
        if parent_id not in session['thoughts']:
            return {"success": False, "error": "Parent thought not found"}
        
        new_thought = ThoughtNode(content, parent_id)
        session['thoughts'][new_thought.id] = new_thought
        session['thoughts'][parent_id].children.append(new_thought.id)
        session['current_focus'] = new_thought.id
        
        return {
            "success": True,
            "thought_id": new_thought.id,
            "parent_id": parent_id,
            "timestamp": new_thought.timestamp
        }

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get a complete session with all thoughts."""
        if session_id not in self.sessions:
            return {"success": False, "error": "Session not found"}
        
        session = self.sessions[session_id]
        thoughts_data = {}
        
        for thought_id, thought in session['thoughts'].items():
            thoughts_data[thought_id] = {
                "id": thought.id,
                "content": thought.content,
                "parent_id": thought.parent_id,
                "timestamp": thought.timestamp,
                "children": thought.children
            }
        
        return {
            "success": True,
            "session_id": session_id,
            "created_at": session['created_at'],
            "current_focus": session['current_focus'],
            "thoughts": thoughts_data
        }

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a thinking session."""
        if session_id not in self.sessions:
            return {"success": False, "error": "Session not found"}
        
        del self.sessions[session_id]
        return {"success": True}

# Global instance
thinking_manager = SequentialThinking()

@mcp.tool()
def create_thinking_session(initial_thought: str) -> Dict[str, Any]:
    """Create a new sequential thinking session."""
    try:
        session_id = thinking_manager.create_session(initial_thought)
        return {
            "success": True,
            "session_id": session_id,
            "initial_thought": initial_thought
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def add_sequential_thought(session_id: str, content: str, parent_id: str = None) -> Dict[str, Any]:
    """Add a thought to an existing sequential thinking session."""
    try:
        return thinking_manager.add_thought(session_id, content, parent_id)
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def get_thinking_session(session_id: str) -> Dict[str, Any]:
    """Get a complete sequential thinking session with all thoughts."""
    try:
        return thinking_manager.get_session(session_id)
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def list_thinking_sessions() -> Dict[str, Any]:
    """List all active sequential thinking sessions."""
    try:
        sessions = thinking_manager.list_sessions()
        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def delete_thinking_session(session_id: str) -> Dict[str, Any]:
    """Delete a sequential thinking session."""
    try:
        return thinking_manager.delete_session(session_id)
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def get_thought_path(session_id: str, thought_id: str) -> Dict[str, Any]:
    """Get the path from root to a specific thought."""
    try:
        if session_id not in thinking_manager.sessions:
            return {"success": False, "error": "Session not found"}
        
        session = thinking_manager.sessions[session_id]
        if thought_id not in session['thoughts']:
            return {"success": False, "error": "Thought not found"}
        
        path = []
        current_id = thought_id
        
        while current_id:
            thought = session['thoughts'][current_id]
            path.insert(0, {
                "id": thought.id,
                "content": thought.content,
                "timestamp": thought.timestamp
            })
            current_id = thought.parent_id
        
        return {
            "success": True,
            "path": path,
            "depth": len(path)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def get_thinking_summary(session_id: str) -> Dict[str, Any]:
    """Get a summary of a thinking session."""
    try:
        if session_id not in thinking_manager.sessions:
            return {"success": False, "error": "Session not found"}
        
        session = thinking_manager.sessions[session_id]
        thoughts = session['thoughts']
        
        total_thoughts = len(thoughts)
        root_thought = session['root']
        
        # Count leaf thoughts (thoughts with no children)
        leaf_thoughts = [t for t in thoughts.values() if not t.children]
        
        # Calculate depth
        max_depth = 0
        for thought in thoughts.values():
            depth = 0
            current = thought
            while current.parent_id:
                depth += 1
                current = thoughts[current.parent_id]
            max_depth = max(max_depth, depth)
        
        return {
            "success": True,
            "session_id": session_id,
            "total_thoughts": total_thoughts,
            "leaf_thoughts": len(leaf_thoughts),
            "max_depth": max_depth,
            "root_content": root_thought.content,
            "created_at": session['created_at'],
            "current_focus": session['current_focus']
        }
    except Exception as e:
        return {"success": False, "error": str(e)}