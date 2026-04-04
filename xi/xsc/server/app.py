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
FastAPI Server Application for Xi Studio Backend.

This module provides the main FastAPI server that exposes REST and WebSocket
endpoints for the Xi Studio frontend.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..core.dc import XiLogger
from ..core.types import XiCommand, XiRunStatus
from ..executor import XiExecutor
from ..session import XmcSessionManager, XmcNotificationManager
from ..config import get_xi_config
from ..config.loader import find_project_root

from .router import setup_routes
from .websocket import setup_websockets
from .middleware import setup_middleware
from .hardware import collect_system_stats


class XiServer:
    """
    Main FastAPI server for Xi Studio.
    
    This class provides:
    - REST API endpoints for training/inference management
    - WebSocket endpoints for real-time streaming
    - Session and notification management
    - System monitoring
    """
    
    def __init__(self, port: int = 3140, root_dir: Optional[str] = None):
        """
        Initialize the Xi server.
        
        Args:
            port: Server port (default: 3140)
            root_dir: Working directory for command execution
        """
        self.port = port
        if root_dir:
            self.root_dir = Path(root_dir)
        else:
            self.root_dir = find_project_root()
        self.logger = XiLogger("Xi.Server", enable_file=True)
        self.session_manager = XmcSessionManager()
        self.notification_manager = XmcNotificationManager()
        self.executor = XiExecutor(str(self.root_dir))
        self._start_time = datetime.now()
        self._request_count = {"value": 0}
        
        self.app = FastAPI(
            title="Xi Studio API",
            description="Backend API for Xi Studio",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        setup_middleware(self.app, self.session_manager)
        setup_routes(
            self.app,
            self.executor,
            self.session_manager,
            self.notification_manager,
            self.root_dir,
            self.port,
            self.logger,
            self
        )
        setup_websockets(
            self.app,
            self.executor,
            self.logger,
            str(self.root_dir),
            self._start_time,
            self._request_count
        )
    
    def run(self, host: str = "127.0.0.1") -> None:
        """
        Start the server.
        
        Args:
            host: Server host (default: 127.0.0.1)
        """
        self.logger.info(
            f"Starting Xi server on {host}:{self.port}",
            event="xi.server.start"
        )
        
        uvicorn.run(
            self.app,
            host=host,
            port=self.port,
            log_level="info",
            ws_ping_interval=20,
            ws_ping_timeout=20,
            timeout_keep_alive=5,
            limit_concurrency=100,
            limit_max_requests=1000,
        )


_server_instance: Optional[XiServer] = None


def get_app() -> FastAPI:
    """
    Get the FastAPI application instance.
    
    Returns:
        FastAPI application
    """
    global _server_instance
    if _server_instance is None:
        _server_instance = XiServer()
    return _server_instance.app


app = get_app()
