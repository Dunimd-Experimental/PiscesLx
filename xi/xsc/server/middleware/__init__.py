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
Middleware configuration for Xi Server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ...session import XmcSessionManager


def setup_middleware(app: FastAPI, session_manager: XmcSessionManager) -> None:
    """
    Setup middleware for the FastAPI application.
    
    Args:
        app: FastAPI application
        session_manager: Session manager instance
    """
    @app.middleware("http")
    async def check_session(request, call_next):
        # Allow WebSocket upgrade requests before any other checks
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)
        
        public_paths = {
            "/docs", "/redoc", "/openapi.json", "/healthz", "/handshake",
            "/v1/xi/validate-config", "/v1/xi/setup-environment", "/v1/xi/first-launch",
            "/v1/xi/complete-first-launch",
            "/v1/runs", "/stats", "/v1/models", "/v1/xi/config", "/v1/xi/paths",
            "/v1/xi/commands", "/v1/notifications", "/v1/tools/list", "/v1/tools/execute"
        }
        
        # Allow WebSocket connections
        if request.url.path.startswith("/ws/"):
            return await call_next(request)
        
        if request.url.path in public_paths or request.url.path.startswith("/docs") or request.url.path.startswith("/redoc"):
            return await call_next(request)
        
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            session = session_manager.validate_token(token)
            if session:
                return await call_next(request)
        
        accept_header = request.headers.get("accept", "")
        is_browser = "text/html" in accept_header and "application/json" not in accept_header
        if is_browser:
            from starlette.responses import HTMLResponse
            return HTMLResponse(
                content=_get_access_denied_html(),
                status_code=403
            )
        
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=403,
            content={"error": "Access denied"}
        )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _get_access_denied_html() -> str:
    """Get HTML content for access denied page."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Access Denied</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #ffffff;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }
        .card {
            max-width: 400px;
            width: 100%;
            background: #ffffff;
            border: 1px solid #e5e5e5;
            border-radius: 12px;
            padding: 32px;
            text-align: center;
        }
        .icon-wrap {
            width: 56px;
            height: 56px;
            margin: 0 auto 20px;
            border-radius: 50%;
            background: #fef2f2;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .icon-wrap svg { width: 28px; height: 28px; color: #ef4444; }
        .title {
            font-size: 18px;
            font-weight: 600;
            color: #171717;
            margin-bottom: 8px;
        }
        .desc {
            font-size: 14px;
            color: #737373;
            line-height: 1.5;
            margin-bottom: 20px;
        }
        .info {
            background: #f5f5f5;
            border-radius: 8px;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .info svg { width: 18px; height: 18px; color: #a3a3a3; flex-shrink: 0; }
        .info-text { text-align: left; }
        .info-label { font-size: 12px; color: #737373; }
        .info-value { font-family: monospace; font-size: 12px; color: #171717; }
    </style>
</head>
<body>
    <div class="card">
        <div class="icon-wrap">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
        </div>
        <h1 class="title">Access Denied</h1>
        <p class="desc">Direct browser access is not permitted.<br>Please use Xi Studio frontend.</p>
        <div class="info">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
            </svg>
            <div class="info-text">
                <div class="info-label">Server</div>
                <div class="info-value">127.0.0.1:3140</div>
            </div>
        </div>
    </div>
</body>
</html>'''
