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
Handshake route for client authentication.
"""

from datetime import datetime
from fastapi import FastAPI

from ...session import XmcSessionManager


def setup_handshake_route(app: FastAPI, session_manager: XmcSessionManager, port: int, start_time: datetime, logger) -> None:
    """
    Setup handshake route for client authentication.
    
    Args:
        app: FastAPI application
        session_manager: Session manager instance
        port: Server port
        start_time: Server start time
        logger: XiLogger instance
    """
    @app.post("/handshake")
    async def handshake(request: dict):
        client = request.get("client", "unknown")
        version = request.get("version", "1.0.0")
        auth = request.get("auth", {})
        
        logger.info(
            f"Handshake request from {client} v{version}",
            event="xsc.handshake.request"
        )
        
        if version != "1.0.0":
            return {
                "type": "handshake_error",
                "error": "Unsupported version",
                "supported_versions": ["1.0.0"]
            }, 400
        
        session = session_manager.create_session(
            client=client,
            version=version,
            auth=auth
        )
        
        logger.info(
            f"Handshake success: {session.session_id}",
            event="xsc.handshake.success"
        )
        
        return {
            "type": "handshake_ack",
            "session_id": session.session_id,
            "token": session.token,
            "capabilities": session.capabilities,
            "endpoints": {
                "ws": f"ws://127.0.0.1:{port}/ws",
                "api": f"http://127.0.0.1:{port}/api"
            },
            "server_info": {
                "version": "1.0.0",
                "uptime": (datetime.now() - start_time).total_seconds()
            }
        }
