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
Notification management for Xi Message Controller.

This module provides notification management for xsc server, handling:
- Notification creation and storage
- Notification retrieval and filtering
- Notification cleanup
"""

import secrets
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta


@dataclass
class XmcNotification:
    """
    Notification data for Xi Message Controller.
    
    Attributes:
        id: Unique notification identifier
        type: Notification type (info, warning, error, success)
        title: Notification title
        message: Notification message
        time: Notification timestamp
        read: Whether notification has been read
        metadata: Additional notification metadata
    """
    id: str
    type: str
    title: str
    message: str
    time: datetime
    read: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class XmcNotificationManager:
    """
    Notification manager for Xi Message Controller.
    
    Handles notification creation, retrieval, and cleanup.
    """
    
    def __init__(self, max_count: int = 1000, retention_days: int = 30):
        """
        Initialize the notification manager.
        
        Args:
            max_count: Maximum number of notifications to store
            retention_days: Days to retain notifications
        """
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
        """
        Create a new notification.
        
        Args:
            notification_type: Type of notification (info, warning, error, success)
            title: Notification title
            message: Notification message
            metadata: Optional additional metadata
            
        Returns:
            Created XmcNotification
        """
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
        """
        Get a notification by ID.
        
        Args:
            notification_id: Notification identifier
            
        Returns:
            XmcNotification if found, None otherwise
        """
        return self._notifications.get(notification_id)
    
    def list_notifications(
        self,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[XmcNotification]:
        """
        List notifications.
        
        Args:
            unread_only: If True, only return unread notifications
            limit: Maximum number of notifications to return
            
        Returns:
            List of notifications
        """
        notifications = []
        
        for nid in self._notification_order[:limit]:
            notification = self._notifications.get(nid)
            if notification:
                if unread_only and notification.read:
                    continue
                notifications.append(notification)
        
        return notifications
    
    def mark_read(self, notification_id: str) -> bool:
        """
        Mark a notification as read.
        
        Args:
            notification_id: Notification identifier
            
        Returns:
            True if notification was marked as read
        """
        notification = self._notifications.get(notification_id)
        if notification:
            notification.read = True
            return True
        return False
    
    def delete_notification(self, notification_id: str) -> bool:
        """
        Delete a notification.
        
        Args:
            notification_id: Notification identifier
            
        Returns:
            True if notification was deleted
        """
        if notification_id in self._notifications:
            del self._notifications[notification_id]
            if notification_id in self._notification_order:
                self._notification_order.remove(notification_id)
            return True
        return False
    
    def clear_all(self) -> int:
        """
        Clear all notifications.
        
        Returns:
            Number of notifications cleared
        """
        count = len(self._notifications)
        self._notifications.clear()
        self._notification_order.clear()
        return count
    
    def _prune_old_notifications(self) -> int:
        """
        Prune old notifications based on retention policy.
        
        Returns:
            Number of notifications pruned
        """
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
        """
        Get count of unread notifications.
        
        Returns:
            Number of unread notifications
        """
        return sum(1 for n in self._notifications.values() if not n.read)
