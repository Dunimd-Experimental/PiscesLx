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
Server router module initialization.
"""

from .health import setup_health_routes
from .handshake import setup_handshake_route
from .runs import setup_runs_routes
from .models import setup_models_routes
from .inference import setup_inference_routes
from .filesystem import setup_filesystem_routes
from .xi import setup_xi_routes
from .notifications import setup_notification_routes
from .tools import setup_tools_routes


def setup_routes(app, executor, session_manager, notification_manager, root_dir, port, logger, server):
    """
    Setup all routes for the Xi server.
    
    Args:
        app: FastAPI application
        executor: XiExecutor instance
        session_manager: XmcSessionManager instance
        notification_manager: XmcNotificationManager instance
        root_dir: Working directory
        port: Server port
        logger: XiLogger instance
        server: XiServer instance
    """
    request_count = server._request_count
    
    setup_health_routes(app, server._start_time, datetime, request_count)
    setup_handshake_route(app, session_manager, port, server._start_time, logger)
    setup_runs_routes(app, root_dir, executor, logger, request_count)
    setup_models_routes(app, logger, request_count)
    setup_inference_routes(app, logger, request_count)
    setup_filesystem_routes(app, root_dir, logger, request_count)
    setup_xi_routes(app, root_dir, logger, request_count)
    setup_notification_routes(app, notification_manager, logger, request_count)
    setup_tools_routes(app, logger, request_count)


from datetime import datetime

__all__ = [
    "setup_routes",
    "setup_health_routes",
    "setup_handshake_route",
    "setup_runs_routes",
    "setup_models_routes",
    "setup_inference_routes",
    "setup_filesystem_routes",
    "setup_xi_routes",
    "setup_notification_routes",
    "setup_tools_routes",
]
