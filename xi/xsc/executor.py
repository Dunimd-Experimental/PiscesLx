#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
Command Executor for Xi Studio Backend Server.

This module provides the command execution layer that bridges the Xi
frontend to manage.py commands. It handles subprocess management,
output streaming, and process lifecycle control.

The executor uses asyncio subprocess for non-blocking command execution
and supports real-time output streaming via queues.
"""

import os
import sys
import asyncio
import signal
import re
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator, List
from datetime import datetime

from .dc import XiLogger, XiLogLevel
from .types import (
    XiCommand,
    XiRequest,
    XiResponse,
    XiRunStatus,
    XiLogEntry,
)
from .config import XiValueMapping


class XiExecutor:
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.logger = XiLogger(
            "Xi.Executor",
            enable_file=True
        )
        self.active_processes: Dict[str, asyncio.subprocess.Process] = {}
        self.output_queues: Dict[str, asyncio.Queue] = {}
        self._process_status: Dict[str, XiRunStatus] = {}

    def _build_argv(self, request: XiRequest) -> list:
        argv = [sys.executable, "manage.py", request.command.value]

        for key, value in request.args.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    argv.append(f"--{key}")
            elif isinstance(value, list):
                for item in value:
                    argv.extend([f"--{key}", str(item)])
            else:
                argv.extend([f"--{key}", str(value)])

        if request.run_id:
            argv.extend(["--run_id", request.run_id])
        if request.run_name:
            argv.extend(["--run_name", request.run_name])

        return argv
    
    def build_argv_from_schema(
        self,
        command_name: str,
        parameters: Dict[str, Any],
        schema_params: List[Any]
    ) -> List[str]:
        """
        Build command line arguments from schema and parameter values.
        
        This method uses the value_mapping configuration from the schema
        to properly format each parameter value for the command line.
        
        Args:
            command_name: Name of the command (e.g., "train")
            parameters: Dictionary of parameter values from frontend
            schema_params: List of parameter schema objects with value_mapping
            
        Returns:
            List of command line argument strings
        """
        from .loader import get_xi_config
        
        config = get_xi_config()
        cmd_config = config.commands.get(command_name)
        
        argv = [sys.executable, "manage.py", command_name]
        
        if cmd_config:
            for default_key, default_value in cmd_config.defaults.items():
                if default_key not in parameters:
                    parameters[default_key] = default_value
        
        for schema_param in schema_params:
            param_name = schema_param.name
            value = parameters.get(param_name)
            
            if value is None:
                continue
            
            mapping = schema_param.value_mapping
            if mapping:
                arg_str = self._apply_value_mapping(param_name, value, mapping)
                if arg_str:
                    if isinstance(arg_str, list):
                        argv.extend(arg_str)
                    else:
                        argv.append(arg_str)
            else:
                default_mapping = XiValueMapping()
                arg_str = self._apply_value_mapping(param_name, value, default_mapping)
                if arg_str:
                    if isinstance(arg_str, list):
                        argv.extend(arg_str)
                    else:
                        argv.append(arg_str)
        
        return argv
    
    def _apply_value_mapping(
        self,
        name: str,
        value: Any,
        mapping: XiValueMapping
    ) -> Optional[str | List[str]]:
        """
        Apply value mapping to convert a parameter value to command line argument(s).
        
        Args:
            name: Parameter name
            value: Parameter value
            mapping: Value mapping configuration
            
        Returns:
            Command line argument string(s) or None if should be skipped
        """
        if mapping.skip_if:
            if self._evaluate_skip_condition(value, mapping.skip_if):
                return None
        
        if value is None or value == "":
            if mapping.default_if_empty is not None:
                value = mapping.default_if_empty
            else:
                return None
        
        transformed_value = self._transform_value(value, mapping.transform)
        
        if mapping.template:
            return self._apply_template(mapping.template, name, transformed_value, mapping)
        
        if mapping.arg_format:
            return mapping.arg_format.format(name=name, value=transformed_value)
        
        prefix = mapping.arg_prefix or "--"
        separator = mapping.arg_separator or " "
        
        if isinstance(transformed_value, bool):
            if transformed_value:
                return f"{prefix}{name}"
            return None
        
        if isinstance(transformed_value, list):
            if mapping.wrap_value:
                transformed_value = [f'"{v}"' for v in transformed_value]
            
            if mapping.join_with:
                joined = mapping.join_with.join(str(v) for v in transformed_value)
                if mapping.wrap_value:
                    joined = f'"{joined}"'
                return f"{prefix}{name}{separator}{joined}"
            else:
                result = []
                for v in transformed_value:
                    result.append(f"{prefix}{name}{separator}{v}")
                return result
        
        str_value = str(transformed_value)
        if mapping.wrap_value:
            str_value = f'"{str_value}"'
        
        return f"{prefix}{name}{separator}{str_value}"
    
    def _transform_value(self, value: Any, transform: str) -> Any:
        """Apply transformation to value."""
        if not transform:
            return value
        
        if transform == "lowercase":
            return str(value).lower()
        elif transform == "uppercase":
            return str(value).upper()
        elif transform == "str":
            return str(value)
        elif transform == "int":
            return int(value)
        elif transform == "float":
            return float(value)
        elif transform == "json":
            import json
            return json.dumps(value)
        elif transform == "path":
            return str(value).replace("/", os.sep).replace("\\", os.sep)
        
        return value
    
    def _evaluate_skip_condition(self, value: Any, condition: str) -> bool:
        """Evaluate whether to skip this argument based on condition."""
        if condition == "empty":
            return value is None or value == "" or value == []
        elif condition == "false":
            return value is False
        elif condition == "true":
            return value is True
        elif condition == "null":
            return value is None
        elif condition == "zero":
            return value == 0
        elif condition.startswith("=="):
            expected = condition[2:].strip().strip('"\'')
            return str(value) == expected
        elif condition.startswith("!="):
            expected = condition[2:].strip().strip('"\'')
            return str(value) != expected
        
        return False
    
    def _apply_template(
        self,
        template: str,
        name: str,
        value: Any,
        mapping: XiValueMapping
    ) -> str:
        """Apply a template for complex argument generation."""
        result = template.replace("{name}", name)
        result = result.replace("{value}", str(value))
        result = result.replace("{prefix}", mapping.arg_prefix or "--")
        result = result.replace("{separator}", mapping.arg_separator or " ")
        
        if isinstance(value, list):
            result = result.replace("{joined}", mapping.join_with.join(str(v) for v in value))
        
        return result

    async def execute(
        self,
        request: XiRequest
    ) -> XiResponse:
        run_id = request.run_id or self._generate_run_id(request.command)

        try:
            argv = self._build_argv(request)

            self.logger.info(
                f"Executing command: {' '.join(argv)}",
                event="xi.executor.execute"
            )

            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self.root_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )

            self.active_processes[run_id] = process
            self._process_status[run_id] = XiRunStatus.RUNNING
            self.output_queues[run_id] = asyncio.Queue()

            asyncio.create_task(self._stream_output(run_id, process))

            if not request.background:
                await process.wait()
                self._cleanup_process(run_id)

                if process.returncode == 0:
                    return XiResponse(
                        success=True,
                        run_id=run_id,
                        message=f"Command {request.command.value} completed successfully"
                    )
                else:
                    return XiResponse(
                        success=False,
                        run_id=run_id,
                        error=f"Command failed with exit code {process.returncode}"
                    )

            return XiResponse(
                success=True,
                run_id=run_id,
                message=f"Command {request.command.value} started in background"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to execute command: {e}",
                event="xi.executor.error"
            )
            return XiResponse(
                success=False,
                error=str(e)
            )

    async def _stream_output(
        self,
        run_id: str,
        process: asyncio.subprocess.Process
    ) -> None:
        queue = self.output_queues.get(run_id)

        async def read_stream(stream, source: str):
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break

                    text = line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        entry = XiLogEntry(
                            timestamp=datetime.now().isoformat(),
                            level=XiLogLevel.INFO.value,
                            message=text,
                            source=source,
                            run_id=run_id
                        )
                        if queue:
                            await queue.put(entry)
            except Exception as e:
                self.logger.error(
                    f"Error streaming {source}: {e}",
                    event="xi.executor.stream_error"
                )

        await asyncio.gather(
            read_stream(process.stdout, "stdout"),
            read_stream(process.stderr, "stderr")
        )

        await process.wait()

        if process.returncode == 0:
            self._process_status[run_id] = XiRunStatus.COMPLETED
        else:
            self._process_status[run_id] = XiRunStatus.FAILED

        self._cleanup_process(run_id)

    def _generate_run_id(self, command: XiCommand) -> str:
        from opss.run import POPSSRunIdFactory
        factory = POPSSRunIdFactory(prefix=command.value)
        return factory.new_id()

    def _cleanup_process(self, run_id: str) -> None:
        if run_id in self.active_processes:
            del self.active_processes[run_id]

    async def control(
        self,
        run_id: str,
        action: str
    ) -> XiResponse:
        process = self.active_processes.get(run_id)
        if not process:
            return XiResponse(
                success=False,
                run_id=run_id,
                error=f"No active process found for run_id: {run_id}"
            )

        previous_status = self._process_status.get(run_id)

        try:
            if action == "pause":
                if sys.platform == "win32":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(signal.SIGSTOP)
                self._process_status[run_id] = XiRunStatus.PAUSED
                return XiResponse(
                    success=True,
                    run_id=run_id,
                    message=f"Process {run_id} paused",
                    data={"previous_status": previous_status.value if previous_status else None}
                )

            elif action == "resume":
                if sys.platform == "win32":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(signal.SIGCONT)
                self._process_status[run_id] = XiRunStatus.RUNNING
                return XiResponse(
                    success=True,
                    run_id=run_id,
                    message=f"Process {run_id} resumed",
                    data={"previous_status": previous_status.value if previous_status else None}
                )

            elif action == "cancel":
                process.terminate()
                self._process_status[run_id] = XiRunStatus.CANCELLED
                return XiResponse(
                    success=True,
                    run_id=run_id,
                    message=f"Process {run_id} cancelled",
                    data={"previous_status": previous_status.value if previous_status else None}
                )

            elif action == "kill":
                process.kill()
                self._process_status[run_id] = XiRunStatus.CANCELLED
                return XiResponse(
                    success=True,
                    run_id=run_id,
                    message=f"Process {run_id} killed",
                    data={"previous_status": previous_status.value if previous_status else None}
                )

            else:
                return XiResponse(
                    success=False,
                    run_id=run_id,
                    error=f"Unknown action: {action}"
                )

        except Exception as e:
            self.logger.error(
                f"Failed to control process {run_id}: {e}",
                event="xi.executor.control_error"
            )
            return XiResponse(
                success=False,
                run_id=run_id,
                error=str(e)
            )

    async def get_output_stream(
        self,
        run_id: str
    ) -> AsyncGenerator[XiLogEntry, None]:
        queue = self.output_queues.get(run_id)
        if not queue:
            return

        while True:
            try:
                entry = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield entry
            except asyncio.TimeoutError:
                if run_id not in self.active_processes:
                    break
                continue

    def get_status(self, run_id: str) -> Optional[XiRunStatus]:
        return self._process_status.get(run_id)

    def list_active_runs(self) -> Dict[str, XiRunStatus]:
        return dict(self._process_status)
