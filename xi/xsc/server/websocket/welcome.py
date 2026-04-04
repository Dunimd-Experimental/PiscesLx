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
WebSocket handler for Welcome screen operations.

This module handles WebSocket communication for the Welcome screen.
Business logic is delegated to the service layer.
"""

import asyncio
from typing import Dict, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect

from ...core.dc import XiLogger
from ..service.welcome import (
    get_welcome_validator,
    get_welcome_setup,
    get_welcome_agreement,
)


_welcome_ws_handler: Optional["PiscesL1WelcomeWebSocket"] = None


class PiscesL1WelcomeWebSocket:
    """
    WebSocket handler for Welcome screen with bidirectional communication.
    
    This class only handles WebSocket communication.
    All business logic is delegated to the service layer:
    - PiscesL1WelcomeValidator: Configuration validation
    - PiscesL1WelcomeSetup: Environment setup
    - PiscesL1WelcomeAgreement: Agreement handling
    
    Features:
    - Connection limit per client (max 1 connection per client)
    - Automatic cleanup of stale connections
    """

    MAX_CONNECTIONS = 10

    def __init__(self, logger: XiLogger):
        self.logger = logger
        self.active_connections: Set[WebSocket] = set()
        self._connection_tasks: Dict[WebSocket, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket) -> None:
        if len(self.active_connections) >= self.MAX_CONNECTIONS:
            old_conn = next(iter(self.active_connections), None)
            if old_conn:
                await self.disconnect(old_conn)
        
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(
            f"Welcome WebSocket connected: {len(self.active_connections)} total",
            event="xi.welcome_ws.connect"
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)
        
        task = self._connection_tasks.pop(websocket, None)
        if task and not task.done():
            task.cancel()
        
        try:
            await websocket.close()
        except Exception:
            pass
        
        self.logger.info(
            f"Welcome WebSocket disconnected: {len(self.active_connections)} remaining",
            event="xi.welcome_ws.disconnect"
        )

    async def handle_message(
        self,
        websocket: WebSocket,
        data: Dict[str, Any]
    ) -> None:
        msg_type = data.get("type")

        if msg_type == "get_agreement":
            await self._handle_get_agreement(websocket)
        elif msg_type == "validate_config":
            await self._handle_validate_config(websocket)
        elif msg_type == "setup_environment":
            await self._handle_setup_environment(websocket)
        elif msg_type == "complete_first_launch":
            await self._handle_complete_first_launch(websocket)
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            })

    async def _handle_get_agreement(self, websocket: WebSocket) -> None:
        """Handle get_agreement request using service layer."""
        try:
            agreement_service = get_welcome_agreement()
            agreement_info = await asyncio.to_thread(agreement_service.get_agreement)
            
            await websocket.send_json({
                "type": "agreement",
                "content": agreement_info.content,
                "version": agreement_info.version,
                "exists": agreement_info.exists
            })
        except Exception as e:
            self.logger.error(
                f"Failed to get agreement: {e}",
                event="xi.welcome_ws.agreement_error"
            )
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to read agreement: {str(e)}"
            })

    async def _handle_validate_config(self, websocket: WebSocket) -> None:
        """Handle validate_config request using service layer."""
        try:
            validator = get_welcome_validator()
            
            all_valid = True
            async for result in validator.validate_all():
                await websocket.send_json({
                    "type": "checking",
                    "step": result.step,
                    "message": f"Checking {validator.get_step_label(result.step)}..."
                })
                
                await websocket.send_json({
                    "type": "result",
                    "step": result.step,
                    "valid": result.valid,
                    "error": result.error,
                    "data": result.data,
                    "warnings": result.warnings
                })
                
                if not result.valid:
                    all_valid = False
            
            await websocket.send_json({
                "type": "done",
                "valid": all_valid
            })
            
        except Exception as e:
            self.logger.error(
                f"Config validation error: {e}",
                event="xi.welcome_ws.validate_error"
            )
            await websocket.send_json({
                "type": "error",
                "message": f"Validation failed: {str(e)}"
            })

    async def _handle_setup_environment(self, websocket: WebSocket) -> None:
        """Handle setup_environment request using service layer."""
        try:
            setup = get_welcome_setup()
            
            all_success = True
            async for result in setup.setup_all():
                await websocket.send_json({
                    "type": "checking",
                    "step": result.step,
                    "message": f"Setting up {setup.get_step_label(result.step)}..."
                })
                
                await websocket.send_json({
                    "type": "result",
                    "step": result.step,
                    "valid": result.success,
                    "error": result.error,
                    "data": result.data,
                    "warnings": result.warnings
                })
                
                if not result.success:
                    all_success = False
            
            await websocket.send_json({
                "type": "done",
                "valid": all_success
            })
            
        except Exception as e:
            self.logger.error(
                f"Environment setup error: {e}",
                event="xi.welcome_ws.setup_error"
            )
            await websocket.send_json({
                "type": "error",
                "message": f"Setup failed: {str(e)}"
            })

    async def _handle_complete_first_launch(self, websocket: WebSocket) -> None:
        """Handle complete_first_launch request using service layer."""
        try:
            agreement_service = get_welcome_agreement()
            success = await asyncio.to_thread(agreement_service.complete_first_launch)
            
            await websocket.send_json({
                "type": "first_launch_completed",
                "success": success
            })
        except Exception as e:
            self.logger.error(
                f"Failed to complete first launch: {e}",
                event="xi.welcome_ws.complete_error"
            )
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to complete first launch: {str(e)}"
            })


def get_welcome_ws_handler(logger: XiLogger) -> PiscesL1WelcomeWebSocket:
    """Get or create the global welcome WebSocket handler instance."""
    global _welcome_ws_handler
    if _welcome_ws_handler is None:
        _welcome_ws_handler = PiscesL1WelcomeWebSocket(logger)
    return _welcome_ws_handler
