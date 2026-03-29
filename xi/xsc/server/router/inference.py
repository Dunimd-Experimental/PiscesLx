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
Inference proxy routes.
"""

import os
from typing import Dict, Any

import httpx
from fastapi import FastAPI

from ...core.dc import XiLogger


def setup_inference_routes(app: FastAPI, logger: XiLogger, request_count: Dict[str, int]) -> None:
    """
    Setup inference proxy routes.
    
    Args:
        app: FastAPI application
        logger: XiLogger instance
        request_count: Mutable request count reference
    """
    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        request_count["value"] = request_count.get("value", 0) + 1
        return await _proxy_inference_request("/v1/chat/completions", request)
    
    @app.post("/v1/embeddings")
    async def create_embeddings(request: dict):
        request_count["value"] = request_count.get("value", 0) + 1
        return await _proxy_inference_request("/v1/embeddings", request)
    
    @app.post("/v1/images/generations")
    async def generate_images(request: dict):
        request_count["value"] = request_count.get("value", 0) + 1
        return await _proxy_inference_request("/v1/images/generations", request)
    
    async def _proxy_inference_request(endpoint: str, request: dict) -> dict:
        inference_url = os.environ.get("PISCESLX_INFERENCE_URL", "http://127.0.0.1:8000")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{inference_url}{endpoint}",
                    json=request
                )
                return response.json()
            except Exception as e:
                logger.error(f"Inference proxy error: {e}", event="xi.proxy.error")
                return {"error": str(e)}
