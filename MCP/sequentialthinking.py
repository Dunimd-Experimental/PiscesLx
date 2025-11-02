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

import json
import uuid
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.mcp import PiscesLxCoreMCPPlaza

mcp = PiscesLxCoreMCPPlaza()

class ThoughtNode:
    """Represents a node in a sequential thinking process.
    
    Each node contains content, metadata, and relationships to other nodes.
    """
    
    def __init__(self, content: str, parent_id: Optional[str] = None):
        """Initialize a ThoughtNode instance.
        
        Args:
            content: The textual content of the thought
            parent_id: Optional identifier of the parent thought
        """
        self.id = str(uuid.uuid4())
        self.content = content
        self.parent_id = parent_id
        self.timestamp = datetime.now().isoformat()
        self.children = []

class SequentialThinking:
    """Manages sequential thinking processes with resource constraints.
    
    Provides session-based management of thought sequences with memory,
    time, and content size limitations to ensure stable operation.
    """
    
    def __init__(self):
        """Initialize the SequentialThinking manager with default limits."""
        self.sessions = {}
        self.max_sessions = 50           # Maximum concurrent sessions
        self.max_thoughts_per_session = 100   # Maximum thoughts per session
        self.max_content_length = 5000   # Maximum characters per thought
        self.max_total_memory = 1000     # Maximum total thoughts across all sessions
        self.session_timeout = 3600      # Session timeout in seconds (1 hour)
    
    def _check_memory_limits(self) -> bool:
        """Check if the total memory limit has been exceeded.
        
        Returns:
            bool: True if memory limit is exceeded, False otherwise
        """
        total_thoughts = sum(len(session['thoughts']) for session in self.sessions.values())
        return total_thoughts >= self.max_total_memory
    
    def _cleanup_expired_sessions(self) -> None:
        """Remove sessions that have exceeded their timeout period."""
        import time
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            created_time = datetime.fromisoformat(session['created_at']).timestamp()
            if current_time - created_time > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    def _enforce_session_limits(self) -> None:
        """Enforce maximum session count by removing oldest sessions."""
        if len(self.sessions) > self.max_sessions:
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1]['created_at']
            )
            for session_id, _ in sorted_sessions[:-self.max_sessions]:
                del self.sessions[session_id]

    def create_session(self, initial_thought: str) -> str:
        """Create a new thinking session with an initial thought.
        
        Args:
            initial_thought: The starting thought content
            
        Returns:
            str: The unique identifier for the new session
            
        Raises:
            MemoryError: If system memory limits are exceeded
            ValueError: If initial thought exceeds content length limits
        """
        self._cleanup_expired_sessions()
        
        if self._check_memory_limits():
            raise MemoryError("Maximum memory limit reached. Please delete some sessions.")
        
        self._enforce_session_limits()
        
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

    def add_thought(self, session_id: str, content: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a new thought to an existing session.
        
        Args:
            session_id: Identifier of the target session
            content: Content of the new thought
            parent_id: Optional identifier of the parent thought
            
        Returns:
            Dict[str, Any]: Operation result with thought details or error information
        """
        if session_id not in self.sessions:
            return {"success": False, "error": "Session not found"}
        
        session = self.sessions[session_id]
        
        if len(session['thoughts']) >= self.max_thoughts_per_session:
            return {"success": False, "error": f"Session full (max: {self.max_thoughts_per_session} thoughts)"}
        
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
        """Retrieve a complete session with all its thoughts.
        
        Args:
            session_id: Identifier of the session to retrieve
            
        Returns:
            Dict[str, Any]: Session data or error information
        """
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
        """Get a list of all active session identifiers.
        
        Returns:
            List[str]: List of session identifiers
        """
        return list(self.sessions.keys())

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session and all its associated thoughts.
        
        Args:
            session_id: Identifier of the session to delete
            
        Returns:
            Dict[str, Any]: Operation result
        """
        if session_id not in self.sessions:
            return {"success": False, "error": "Session not found"}
        
        del self.sessions[session_id]
        return {"success": True}

# Global instance for managing thinking sessions
thinking_manager = SequentialThinking()

@mcp.tool()
def create_thinking_session(initial_thought: str) -> Dict[str, Any]:
    """Create a new sequential thinking session with security limits.
    
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
def add_sequential_thought(session_id: str, content: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
    """Add a thought to a session with security limits.
    
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
    """Get a complete sequential thinking session with all thoughts.
    
    Args:
        session_id: Identifier of the session to retrieve
        
    Returns:
        Dict[str, Any]: Session data with memory usage statistics
    """
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
    """List all active sequential thinking sessions.
    
    Returns:
        Dict[str, Any]: List of session identifiers with count
    """
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
    """Delete a sequential thinking session.
    
    Args:
        session_id: Identifier of the session to delete
        
    Returns:
        Dict[str, Any]: Operation result
    """
    try:
        return thinking_manager.delete_session(session_id)
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def get_thought_path(session_id: str, thought_id: str) -> Dict[str, Any]:
    """Get the path from root to a specific thought.
    
    Args:
        session_id: Identifier of the session
        thought_id: Identifier of the target thought
        
    Returns:
        Dict[str, Any]: Path information or error
    """
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
    """Get a summary of a thinking session.
    
    Args:
        session_id: Identifier of the session to summarize
        
    Returns:
        Dict[str, Any]: Session summary or error
    """
    try:
        if session_id not in thinking_manager.sessions:
            return {"success": False, "error": "Session not found"}
        
        session = thinking_manager.sessions[session_id]
        thoughts = session['thoughts']
        
        total_thoughts = len(thoughts)
        root_thought = session['root']
        
        # Count leaf thoughts (thoughts with no children)
        leaf_thoughts = [t for t in thoughts.values() if not t.children]
        
        # Calculate maximum depth
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
