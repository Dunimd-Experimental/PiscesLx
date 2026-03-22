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
from datetime import datetime, timedelta


@dataclass
class XmcSession:
    session_id: str
    token: str
    client: str
    version: str
    created_at: float
    last_active: float
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class XmcSessionManager:
    def __init__(self, secret_key: Optional[str] = None):
        self._sessions: Dict[str, XmcSession] = {}
        self._tokens: Dict[str, str] = {}
        self._secret_key = secret_key or secrets.token_hex(32)

    def _generate_token(self, session_id: str) -> str:
        timestamp = str(time.time())
        data = f"{session_id}:{timestamp}:{self._secret_key}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def create_session(
        self,
        client: str,
        version: str,
        auth: Optional[Dict[str, Any]] = None
    ) -> XmcSession:
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
        session = self._sessions.pop(session_id, None)
        if session:
            self._tokens.pop(session.token, None)
            return True
        return False

    def list_sessions(self) -> List[XmcSession]:
        return list(self._sessions.values())

    def cleanup_expired(self) -> int:
        expired = []
        now = time.time()
        for session_id, session in self._sessions.items():
            if now - session.last_active > 3600:
                expired.append(session_id)

        for session_id in expired:
            self.remove_session(session_id)

        return len(expired)


@dataclass
class XmcNotification:
    id: str
    type: str
    title: str
    message: str
    time: datetime
    read: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class XmcNotificationManager:
    def __init__(self, max_count: int = 1000, retention_days: int = 30):
        self._notifications: Dict[str, XmcNotification] = {}
        self._max_count = max_count
        self._retention_days = retention_days
        self._notification_order: List[str] = []

    def create_notification(
        self,
        notification_type: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> XmcNotification:
        notification_id = f"notif-{secrets.token_hex(8)}"
        
        notification = XmcNotification(
            id=notification_id,
            type=notification_type,
            title=title,
            message=message,
            time=datetime.now(),
            read=False,
            metadata=metadata or {},
        )
        
        self._notifications[notification_id] = notification
        self._notification_order.insert(0, notification_id)
        
        if len(self._notifications) > self._max_count:
            self._prune_old_notifications()
        
        return notification

    def get_notification(self, notification_id: str) -> Optional[XmcNotification]:
        return self._notifications.get(notification_id)

    def list_notifications(
        self,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[XmcNotification]:
        notifications = []
        
        for nid in self._notification_order[:limit]:
            notification = self._notifications.get(nid)
            if notification:
                if unread_only and notification.read:
                    continue
                notifications.append(notification)
        
        return notifications

    def mark_read(self, notification_id: str) -> bool:
        notification = self._notifications.get(notification_id)
        if notification:
            notification.read = True
            return True
        return False

    def delete_notification(self, notification_id: str) -> bool:
        if notification_id in self._notifications:
            del self._notifications[notification_id]
            if notification_id in self._notification_order:
                self._notification_order.remove(notification_id)
            return True
        return False

    def clear_all(self) -> int:
        count = len(self._notifications)
        self._notifications.clear()
        self._notification_order.clear()
        return count

    def _prune_old_notifications(self) -> int:
        cutoff = datetime.now() - timedelta(days=self._retention_days)
        pruned = []
        
        for nid, notification in self._notifications.items():
            if notification.time < cutoff:
                pruned.append(nid)
        
        for nid in pruned:
            self.delete_notification(nid)
        
        if len(self._notifications) > self._max_count:
            excess = len(self._notifications) - self._max_count
            for nid in self._notification_order[-excess:]:
                self.delete_notification(nid)
        
        return len(pruned)

    def get_unread_count(self) -> int:
        return sum(1 for n in self._notifications.values() if not n.read)
