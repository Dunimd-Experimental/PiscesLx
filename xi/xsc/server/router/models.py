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
Model listing routes.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI


def setup_models_routes(app: FastAPI, root_dir: Path, request_count: Dict[str, int]) -> None:
    """
    Setup model listing routes.
    
    Args:
        app: FastAPI application
        root_dir: Working directory
        request_count: Mutable request count reference
    """
    @app.get("/v1/models")
    async def list_models():
        request_count["value"] = request_count.get("value", 0) + 1
        models = []
        
        config_dir = root_dir / "configs"
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                model_id = config_file.stem
                models.append({
                    "id": f"piscesl1-{model_id.lower()}",
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "piscesl1"
                })
        
        if not models:
            default_sizes = ["0.5B", "1B", "7B", "14B", "72B", "671B", "1T"]
            for size in default_sizes:
                models.append({
                    "id": f"piscesl1-{size.lower()}",
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "piscesl1"
                })
        
        return {"data": models, "object": "list"}
