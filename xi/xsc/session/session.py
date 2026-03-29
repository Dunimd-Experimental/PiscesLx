#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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
Session management for Xi Message Controller.

This module provides session management for xsc server, handling:
- Session creation and validation
- Token generation and verification
- Client authentication
"""

import secrets
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any


@dataclass
class XmcSession:
    """
    Session data for Xi Message Controller.
    
    Attributes:
        session_id: Unique session identifier
        token: Authentication token
        client: Client identifier
        version: Client version
        created_at: Session creation timestamp
        last_active: Last activity timestamp
        capabilities: List of allowed capabilities
        metadata: Additional session metadata
    """
    session_id: str
    token: str
    client: str
    version: str
    created_at: float
    last_active: float
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class XmcSessionManager:
    """
    Session manager for Xi Message Controller.
    
    Handles session creation, validation, and cleanup.
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize the session manager.
        
        Args:
            secret_key: Optional secret key for token generation
        """
        self._sessions: Dict[str, XmcSession] = {}
        self._tokens: Dict[str, str] = {}
        self._secret_key = secret_key or secrets.token_hex(32)
    
    def _generate_token(self, session_id: str) -> str:
        """
        Generate a secure token for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Generated token string
        """
        timestamp = str(time.time())
        data = f"{session_id}:{timestamp}:{self._secret_key}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    def create_session(
        self,
        client: str,
        version: str,
        auth: Optional[Dict[str, Any]] = None
    ) -> XmcSession:
        """
        Create a new session.
        
        Args:
            client: Client identifier
            version: Client version
            auth: Optional authentication data
            
        Returns:
            Created XmcSession
        """
        session_id = f"xi-session-{secrets.token_hex(16)}"
        token = self._generate_token(session_id)
        
        session = XmcSession(
            session_id=session_id,
            token=token,
            client=client,
            version=version,
            created_at=time.time(),
            last_active=time.time(),
            capabilities=["train", "inference", "monitor", "model"]
        )
        
        self._sessions[session_id] = session
        self._tokens[token] = session_id
        
        return session
    
    def validate_token(self, token: str) -> Optional[XmcSession]:
        """
        Validate a token and return the associated session.
        
        Args:
            token: Token to validate
            
        Returns:
            XmcSession if valid, None otherwise
        """
        session_id = self._tokens.get(token)
        if not session_id:
            return None
        
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        if time.time() - session.last_active > 3600:
            self.remove_session(session_id)
            return None
        
        session.last_active = time.time()
        return session
    
    def validate_session(self, session_id: str, token: str) -> bool:
        """
        Validate a session by ID and token.
        
        Args:
            session_id: Session identifier
            token: Session token
            
        Returns:
            True if session is valid
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        if session.token != token:
            return False
        if time.time() - session.last_active > 3600:
            self.remove_session(session_id)
            return False
        
        session.last_active = time.time()
        return True
    
    def remove_session(self, session_id: str) -> bool:
        """
        Remove a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was removed
        """
        session = self._sessions.pop(session_id, None)
        if session:
            self._tokens.pop(session.token, None)
            return True
        return False
    
    def list_sessions(self) -> List[XmcSession]:
        """
        List all active sessions.
        
        Returns:
            List of active sessions
        """
        return list(self._sessions.values())
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions removed
        """
        expired = []
        now = time.time()
        for session_id, session in self._sessions.items():
            if now - session.last_active > 3600:
                expired.append(session_id)
        
        for session_id in expired:
            self.remove_session(session_id)
        
        return len(expired)
