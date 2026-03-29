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
Notification routes.
"""

from typing import Dict, Any

from fastapi import FastAPI

from ...session import XmcNotificationManager
from ...core.dc import XiLogger


def setup_notification_routes(app: FastAPI, notification_manager: XmcNotificationManager, logger: XiLogger, request_count: Dict[str, int]) -> None:
    """
    Setup notification routes.
    
    Args:
        app: FastAPI application
        notification_manager: Notification manager instance
        logger: XiLogger instance
        request_count: Mutable request count reference
    """
    @app.get("/v1/notifications")
    async def list_notifications():
        request_count["value"] = request_count.get("value", 0) + 1
        try:
            notifications = notification_manager.list_notifications()
            result = [
                {
                    "id": n.id,
                    "type": n.type,
                    "title": n.title,
                    "message": n.message,
                    "time": n.time.isoformat(),
                    "read": n.read,
                }
                for n in notifications
            ]
            return {"notifications": result, "total": len(result)}
        except Exception as e:
            logger.error(f"Failed to list notifications: {e}", event="xi.notifications.error")
            return {"error": str(e)}
    
    @app.post("/v1/notifications")
    async def create_notification(request: dict):
        request_count["value"] = request_count.get("value", 0) + 1
        try:
            notification = notification_manager.create_notification(
                notification_type=request.get("type", "info"),
                title=request.get("title", ""),
                message=request.get("message", ""),
                metadata=request.get("metadata"),
            )
            return {
                "id": notification.id,
                "type": notification.type,
                "title": notification.title,
                "message": notification.message,
                "time": notification.time.isoformat(),
                "read": notification.read,
            }
        except Exception as e:
            logger.error(f"Failed to create notification: {e}", event="xi.notification.create_error")
            return {"error": str(e)}
    
    @app.post("/v1/notifications/{notification_id}/read")
    async def mark_notification_read(notification_id: str):
        request_count["value"] = request_count.get("value", 0) + 1
        try:
            success = notification_manager.mark_read(notification_id)
            return {"success": success}
        except Exception as e:
            logger.error(f"Failed to mark notification read: {e}", event="xi.notification.read_error")
            return {"error": str(e)}
    
    @app.delete("/v1/notifications/{notification_id}")
    async def delete_notification(notification_id: str):
        request_count["value"] = request_count.get("value", 0) + 1
        try:
            success = notification_manager.delete_notification(notification_id)
            return {"success": success}
        except Exception as e:
            logger.error(f"Failed to delete notification: {e}", event="xi.notification.delete_error")
            return {"error": str(e)}
