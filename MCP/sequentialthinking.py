#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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
    """Manages sequential thinking sessions with memory limits."""
    
    def __init__(self):
        self.sessions = {}
        # Security limits for server environment
        self.max_sessions = 50  # Maximum concurrent sessions
        self.max_thoughts_per_session = 100  # Maximum thoughts per session
        self.max_content_length = 5000  # Maximum characters per thought
        self.max_total_memory = 1000  # Maximum total thoughts across all sessions
        self.session_timeout = 3600  # 1 hour timeout in seconds
    
    def _check_memory_limits(self) -> bool:
        """Check if memory limits are exceeded."""
        total_thoughts = sum(len(session['thoughts']) for session in self.sessions.values())
        return total_thoughts >= self.max_total_memory
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions based on timeout."""
        import time
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            created_time = datetime.fromisoformat(session['created_at']).timestamp()
            if current_time - created_time > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    def _enforce_session_limits(self):
        """Enforce session count limits by removing oldest sessions."""
        if len(self.sessions) > self.max_sessions:
            # Sort by creation time and remove oldest
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1]['created_at']
            )
            # Remove oldest sessions
            for session_id, _ in sorted_sessions[:-self.max_sessions]:
                del self.sessions[session_id]

    def create_session(self, initial_thought: str) -> str:
        """Create a new thinking session with an initial thought and memory checks."""
        # Clean up expired sessions first
        self._cleanup_expired_sessions()
        
        # Check memory limits
        if self._check_memory_limits():
            raise MemoryError("Maximum memory limit reached. Please delete some sessions.")
        
        # Enforce session count limits
        self._enforce_session_limits()
        
        # Validate content length
        if len(initial_thought) > self.max_content_length:
            raise ValueError(f"Initial thought too long (max: {self.max_content_length} characters)")
        
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
        """Add a new thought to a session with memory checks."""
        if session_id not in self.sessions:
            return {"success": False, "error": "Session not found"}
        
        session = self.sessions[session_id]
        
        # Check session-level thought limits
        if len(session['thoughts']) >= self.max_thoughts_per_session:
            return {"success": False, "error": f"Session full (max: {self.max_thoughts_per_session} thoughts)"}
        
        # Validate content length
        if len(content) > self.max_content_length:
            return {"success": False, "error": f"Thought too long (max: {self.max_content_length} characters)"}
        
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
            "timestamp": new_thought.timestamp,
            "session_thoughts": len(session['thoughts']),
            "limit_warning": len(session['thoughts']) >= self.max_thoughts_per_session * 0.8
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
    """Create a new sequential thinking session with security limits.
    
    Security limits:
    - Max 50 concurrent sessions
    - Max 100 thoughts per session
    - Max 5000 characters per thought
    - 1-hour session timeout
    - Max 1000 total thoughts across all sessions
    
    Args:
        initial_thought: The starting thought for the session
        
    Returns:
        Dict[str, Any]: Result with session_id or error
        
    Raises:
        MemoryError: If memory limits exceeded
        ValueError: If thought too long
    """
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
    """Add a thought to a session with security limits.
    
    Security limits:
    - Max 100 thoughts per session
    - Max 5000 characters per thought
    - Returns warning when session reaches 80% capacity
    
    Args:
        session_id: The session ID
        content: The thought content
        parent_id: Optional parent thought ID
        
    Returns:
        Dict[str, Any]: Result with success status and details
    """
    try:
        return thinking_manager.add_thought(session_id, content, parent_id)
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def get_thinking_session(session_id: str) -> Dict[str, Any]:
    """Get a complete sequential thinking session with all thoughts."""
    try:
        session = thinking_manager.get_session(session_id)
        if session["success"]:
            total_thoughts = len(session["thoughts"])
            memory_usage = {
                "total_thoughts": total_thoughts,
                "max_thoughts": thinking_manager.max_thoughts_per_session,
                "usage_percentage": (total_thoughts / thinking_manager.max_thoughts_per_session) * 100,
                "session_limit_warning": total_thoughts >= thinking_manager.max_thoughts_per_session * 0.8
            }
            session["memory_usage"] = memory_usage
        return session
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

@mcp.tool()
def get_thinking_statistics() -> Dict[str, Any]:
    """Get overall thinking system statistics and memory usage.
    
    Returns:
        Dict[str, Any]: System statistics including memory usage
    """
    total_sessions = len(thinking_manager.sessions)
    total_thoughts = sum(len(session['thoughts']) for session in thinking_manager.sessions.values())
    
    return {
        'total_sessions': total_sessions,
        'total_thoughts': total_thoughts,
        'max_sessions': thinking_manager.max_sessions,
        'max_thoughts_per_session': thinking_manager.max_thoughts_per_session,
        'max_total_thoughts': thinking_manager.max_total_memory,
        'session_usage_percentage': (total_sessions / thinking_manager.max_sessions) * 100,
        'memory_usage_percentage': (total_thoughts / thinking_manager.max_total_memory) * 100,
        'memory_limit_warning': total_thoughts >= thinking_manager.max_total_memory * 0.8,
        'session_limit_warning': total_sessions >= thinking_manager.max_sessions * 0.8
    }