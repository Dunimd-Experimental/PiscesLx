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
Xi configuration routes.

HTTP routes for Xi configuration operations.
Business logic is delegated to the service layer.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from ...core.dc import XiLogger
from ...config import get_xi_config, get_config_loader
from ..service.welcome import (
    get_welcome_validator,
    get_welcome_setup,
    get_welcome_agreement,
)


def setup_xi_routes(app: FastAPI, root_dir: Path, logger: XiLogger, request_count: Dict[str, int]) -> None:
    """
    Setup Xi API routes.
    
    Args:
        app: FastAPI application
        root_dir: Working directory
        logger: XiLogger instance
        request_count: Mutable request count reference
    """
    
    @app.get("/v1/xi/first-launch")
    async def check_first_launch():
        request_count["value"] = request_count.get("value", 0) + 1
        
        try:
            agreement_service = get_welcome_agreement(root_dir)
            loop = asyncio.get_event_loop()
            is_first = await loop.run_in_executor(None, agreement_service.is_first_launch)
            
            return {"is_first_launch": is_first}
        except Exception as e:
            logger.error(f"Failed to check first launch: {e}", event="xi.first_launch.error")
            return {"is_first_launch": False}

    @app.get("/v1/xi/agreement")
    async def get_agreement():
        request_count["value"] = request_count.get("value", 0) + 1
        
        agreement_service = get_welcome_agreement(root_dir)
        loop = asyncio.get_event_loop()
        agreement_info = await loop.run_in_executor(None, agreement_service.get_agreement)
        
        return {"agreement": agreement_info.content, "version": agreement_info.version}

    @app.post("/v1/xi/complete-first-launch")
    async def complete_first_launch():
        request_count["value"] = request_count.get("value", 0) + 1
        
        agreement_service = get_welcome_agreement(root_dir)
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, agreement_service.complete_first_launch)
        
        if success:
            return {"success": True}
        else:
            return {"success": False, "error": "Failed to update configuration"}
    
    @app.get("/v1/xi/validate-config")
    async def validate_xi_config():
        async def generate_validation_events():
            validator = get_welcome_validator(root_dir)
            
            async for result in validator.validate_all():
                yield f"data: {json.dumps({'event': 'checking', 'step': result.step, 'message': f'Checking {validator.get_step_label(result.step)}...'})}\n\n"
                
                yield f"data: {json.dumps({'event': 'result', 'step': result.step, 'valid': result.valid, 'error': result.error, 'data': result.data, 'warnings': result.warnings})}\n\n"
        
        return StreamingResponse(
            generate_validation_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    @app.get("/v1/xi/setup-environment")
    async def setup_environment():
        async def generate_setup_events():
            setup = get_welcome_setup(root_dir)
            
            async for result in setup.setup_all():
                yield f"data: {json.dumps({'event': 'checking', 'step': result.step, 'message': f'Setting up {setup.get_step_label(result.step)}...'})}\n\n"
                
                yield f"data: {json.dumps({'event': 'result', 'step': result.step, 'valid': result.success, 'error': result.error, 'data': result.data, 'warnings': result.warnings})}\n\n"
        
        return StreamingResponse(
            generate_setup_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    @app.get("/v1/xi/config")
    async def get_config():
        request_count["value"] = request_count.get("value", 0) + 1
        try:
            config = get_xi_config()
            return config.to_dict()
        except Exception as e:
            logger.error(f"Failed to get Xi config: {e}", event="xi.config.error")
            return {"error": str(e)}
    
    @app.get("/v1/xi/paths")
    async def get_paths():
        request_count["value"] = request_count.get("value", 0) + 1
        try:
            config = get_xi_config()
            resolved = config.get_resolved_paths()
            return {
                "root": str(config.project_root),
                "models": str(resolved.get("models", config.paths.models)),
                "checkpoints": str(resolved.get("checkpoints", config.paths.checkpoints)),
                "data": str(resolved.get("data", config.paths.data)),
                "outputs": str(resolved.get("outputs", config.paths.outputs)),
                "logs": str(resolved.get("logs", config.paths.logs)),
                "cache": str(resolved.get("cache", config.paths.cache)),
                "temp": str(resolved.get("temp", config.paths.temp)),
                "configs": str(resolved.get("configs", config.paths.configs)),
            }
        except Exception as e:
            logger.error(f"Failed to get Xi paths: {e}", event="xi.paths.error")
            return {"error": str(e)}
    
    @app.get("/v1/xi/commands")
    async def get_commands():
        request_count["value"] = request_count.get("value", 0) + 1
        try:
            config = get_xi_config()
            commands = {}
            for name, cmd in config.commands.items():
                commands[name] = {
                    "executable": cmd.executable,
                    "script": cmd.script,
                    "args": cmd.args,
                    "env": cmd.env,
                    "cwd": cmd.cwd,
                    "timeout": cmd.timeout,
                    "background": cmd.background,
                    "defaults": cmd.defaults,
                }
            return {"commands": commands, "total": len(commands)}
        except Exception as e:
            logger.error(f"Failed to get Xi commands: {e}", event="xi.commands.error")
            return {"error": str(e)}
    
    @app.get("/v1/xi/commands/{command_name}")
    async def get_command(command_name: str):
        request_count["value"] = request_count.get("value", 0) + 1
        try:
            config = get_xi_config()
            cmd = config.commands.get(command_name)
            if not cmd:
                return {"error": f"Command '{command_name}' not found"}
            
            result = {
                "name": command_name,
                "executable": cmd.executable,
                "script": cmd.script,
                "args": cmd.args,
                "env": cmd.env,
                "cwd": cmd.cwd,
                "timeout": cmd.timeout,
                "background": cmd.background,
                "defaults": cmd.defaults,
            }
            
            if cmd.schema:
                result["schema"] = {
                    "description": cmd.schema.description,
                    "parameters": [
                        {
                            "name": p.name,
                            "type": p.type,
                            "description": p.description,
                            "required": p.required,
                            "default": p.default,
                            "options": p.options,
                        }
                        for p in cmd.schema.parameters
                    ],
                }
            
            return result
        except Exception as e:
            logger.error(f"Failed to get Xi command: {e}", event="xi.command.error")
            return {"error": str(e)}
    
    @app.get("/v1/xi/commands/{command_name}/schema")
    async def get_command_schema(command_name: str):
        request_count["value"] = request_count.get("value", 0) + 1
        try:
            config = get_xi_config()
            cmd = config.commands.get(command_name)
            if not cmd:
                return {
                    "command": command_name,
                    "available": False,
                    "unavailable_reason": f"Command '{command_name}' not found",
                }
            
            if not cmd.schema:
                return {
                    "command": command_name,
                    "available": False,
                    "unavailable_reason": f"No schema defined for command '{command_name}'",
                }
            
            tabs = [{"name": t.name, "label": t.label, "available": t.available} for t in cmd.schema.tabs]
            parameters = [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "available": p.available,
                    "tab": p.tab,
                }
                for p in cmd.schema.parameters
            ]
            
            return {
                "command": command_name,
                "description": cmd.schema.description,
                "available": cmd.schema.available,
                "tabs": tabs,
                "parameters": parameters,
            }
        except Exception as e:
            logger.error(f"Failed to get command schema: {e}", event="xi.schema.error")
            return {
                "command": command_name,
                "available": False,
                "unavailable_reason": str(e),
            }
