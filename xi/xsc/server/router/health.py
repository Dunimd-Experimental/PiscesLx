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
Health check and system statistics routes.
"""

from datetime import datetime
from fastapi import FastAPI

from ...core.types import XiSystemStats
from ..hardware import collect_system_stats


def setup_health_routes(app: FastAPI, stats_start_time, datetime, request_count: int) -> None:
    """
    Setup health check and stats routes.
    
    Args:
        app: FastAPI application
        stats_start_time: Server start time
        request_count: Mutable request count reference
    """
    @app.get("/healthz")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    @app.get("/stats")
    async def get_stats():
        nonlocal request_count
        request_count += 1
        stats = await collect_system_stats(stats_start_time, request_count)
        from dataclasses import asdict
        return asdict(stats)
