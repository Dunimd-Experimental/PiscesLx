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
"""

import os
import re
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from ...core.dc import XiLogger
from ...config import get_xi_config


def setup_xi_routes(app: FastAPI, root_dir: Path, logger: XiLogger, request_count: Dict[str, int]) -> None:
    """
    Setup Xi configuration routes.
    
    Args:
        app: FastAPI application
        root_dir: Working directory
        logger: XiLogger instance
        request_count: Mutable request count reference
    """
    @app.get("/v1/xi/first-launch")
    async def check_first_launch():
        request_count["value"] = request_count.get("value", 0) + 1
        import tomli
        
        xi_toml_path = root_dir / ".xi" / "xi.toml"
        
        if not xi_toml_path.exists():
            return {"is_first_launch": True}
        
        try:
            with open(xi_toml_path, "rb") as f:
                config = tomli.load(f)
            
            first_launch = config.get("project", {}).get("first_launch", True)
            return {"is_first_launch": bool(first_launch)}
        except Exception:
            return {"is_first_launch": True}
    
    @app.post("/v1/xi/complete-first-launch")
    async def complete_first_launch():
        request_count["value"] = request_count.get("value", 0) + 1

        import tomli

        xi_toml_path = root_dir / ".xi" / "xi.toml"

        if not xi_toml_path.exists():
            return {"success": False, "error": "xi.toml not found"}

        try:
            with open(xi_toml_path, "r", encoding="utf-8") as f:
                content = f.read()

            new_content = re.sub(r'first_launch\s*=\s*true', 'first_launch = false', content, flags=re.IGNORECASE)

            with open(xi_toml_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(new_content)

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @app.get("/v1/xi/validate-config")
    async def validate_xi_config():
        async def generate_validation_events():
            all_valid = True
            xi_dir = root_dir / ".xi"
            xi_toml_path = xi_dir / "xi.toml"
            
            yield f"data: {json.dumps({'event': 'checking', 'step': 'xi_toml_syntax', 'message': 'Checking main configuration syntax'})}\n\n"
            
            import tomli
            if not xi_toml_path.exists():
                all_valid = False
                yield f"data: {json.dumps({'event': 'result', 'step': 'xi_toml_syntax', 'valid': False, 'error': 'xi.toml not found'})}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'valid': False})}\n\n"
                return
            
            try:
                with open(xi_toml_path, "rb") as f:
                    tomli.load(f)
                yield f"data: {json.dumps({'event': 'result', 'step': 'xi_toml_syntax', 'valid': True, 'error': None})}\n\n"
            except tomli.TOMLDecodeError as e:
                all_valid = False
                error_msg = f"Syntax error at line {e.lineno}: {e.msg}" if hasattr(e, 'lineno') else str(e)
                yield f"data: {json.dumps({'event': 'result', 'step': 'xi_toml_syntax', 'valid': False, 'error': error_msg})}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'valid': False})}\n\n"
                return
            
            yield f"data: {json.dumps({'event': 'checking', 'step': 'project_info', 'message': 'Checking project information'})}\n\n"
            
            try:
                config = get_xi_config()
                
                if not config.project.name:
                    all_valid = False
                    yield f"data: {json.dumps({'event': 'result', 'step': 'project_info', 'valid': False, 'error': 'Project name not configured'})}\n\n"
                else:
                    yield f"data: {json.dumps({'event': 'result', 'step': 'project_info', 'valid': True, 'error': None, 'data': {'name': config.project.name, 'version': config.project.version, 'backend': config.project.backend}})}\n\n"
            except Exception as e:
                all_valid = False
                yield f"data: {json.dumps({'event': 'result', 'step': 'project_info', 'valid': False, 'error': str(e)})}\n\n"
            
            yield f"data: {json.dumps({'event': 'done', 'valid': all_valid})}\n\n"
        
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
            all_valid = True
            
            yield f"data: {json.dumps({'event': 'checking', 'step': 'venv_create', 'message': 'Checking virtual environment'})}\n\n"
            
            try:
                config = get_xi_config()
                
                venv_enabled = config.environment.virtualenv and config.environment.virtualenv.enabled
                venv_path = root_dir / config.environment.virtualenv.path if venv_enabled else None
                
                if venv_enabled and venv_path:
                    if not venv_path.exists():
                        yield f"data: {json.dumps({'event': 'checking', 'step': 'venv_create', 'message': 'Creating virtual environment...'})}\n\n"
                        
                        create_result = subprocess.run(
                            ["python", "-m", "venv", str(venv_path)],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        
                        if create_result.returncode != 0:
                            all_valid = False
                            yield f"data: {json.dumps({'event': 'result', 'step': 'venv_create', 'valid': False, 'error': f'Failed to create venv: {create_result.stderr}'})}\n\n"
                        else:
                            yield f"data: {json.dumps({'event': 'result', 'step': 'venv_create', 'valid': True, 'error': None, 'data': {'path': str(venv_path)}})}\n\n"
                    else:
                        yield f"data: {json.dumps({'event': 'result', 'step': 'venv_create', 'valid': True, 'error': None, 'data': {'path': str(venv_path), 'exists': True}})}\n\n"
                else:
                    yield f"data: {json.dumps({'event': 'result', 'step': 'venv_create', 'valid': True, 'error': None, 'data': {'message': 'Virtual environment not configured'}})}\n\n"
            
            except Exception as e:
                all_valid = False
                yield f"data: {json.dumps({'event': 'result', 'step': 'venv_create', 'valid': False, 'error': str(e)})}\n\n"
            
            yield f"data: {json.dumps({'event': 'done', 'valid': all_valid})}\n\n"
        
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
