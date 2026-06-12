#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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

import json
import subprocess
from typing import Any

from encre.tools.base import build_tool


def _build_command(target: str, action: str, config_file: str, project_name: str) -> list[str]:
    if target == "docker":
        if action == "build":
            cmd = ["docker", "build"]
            if config_file:
                cmd.extend(["-f", config_file])
            if project_name:
                cmd.extend(["-t", project_name, "."])
            return cmd
        elif action == "deploy":
            return ["docker", "push", project_name] if project_name else ["docker", "stack", "deploy"]
        elif action == "rollback":
            return ["docker", "service", "rollback", project_name] if project_name else ["docker", "rollback"]
        elif action == "status":
            return ["docker", "ps"]

    elif target == "kubernetes":
        if action == "build":
            return ["kubectl", "apply", "-f", config_file] if config_file else ["kubectl", "apply", "-f", "."]
        elif action == "deploy":
            return ["kubectl", "apply", "-f", config_file] if config_file else ["kubectl", "apply", "-f", "."]
        elif action == "rollback":
            return ["kubectl", "rollout", "undo", f"deployment/{project_name}"] if project_name else ["kubectl", "rollout", "undo", "deployment/"]
        elif action == "status":
            base = ["kubectl", "get", "pods"]
            if project_name:
                base.extend(["-l", f"app={project_name}"])
            return base

    elif target == "cloud_run":
        if action == "deploy":
            cmd = ["gcloud", "run", "deploy"]
            if project_name:
                cmd.append(project_name)
            cmd.extend(["--source", "."])
            return cmd
        elif action == "status":
            return ["gcloud", "run", "services", "list"]

    elif target == "vercel":
        if action == "deploy":
            cmd = ["vercel", "deploy"]
            if config_file:
                cmd.extend(["--local-config", config_file])
            return cmd
        elif action == "status":
            return ["vercel", "list"]

    elif target == "netlify":
        if action == "deploy":
            cmd = ["netlify", "deploy"]
            if config_file:
                cmd.extend(["--config", config_file])
            return cmd
        elif action == "status":
            return ["netlify", "status"]

    return ["echo", f"Unknown target/action: {target}/{action}"]


async def _deploy_execute(**kwargs: Any) -> str:
    target = kwargs.get("target", "docker")
    action = kwargs.get("action", "build")
    config_file = kwargs.get("config_file", "")
    project_name = kwargs.get("project_name", "")

    cmd = _build_command(target, action, config_file, project_name)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if result.returncode != 0:
            output += f"\nDeploy command exited with code {result.returncode}"
        return json.dumps({
            "target": target,
            "action": action,
            "command": " ".join(cmd),
            "output": output,
        }, indent=2)
    except subprocess.TimeoutExpired:
        return json.dumps({
            "target": target,
            "action": action,
            "command": " ".join(cmd),
            "error": "Command timed out after 600 seconds",
        })
    except FileNotFoundError:
        return json.dumps({
            "target": target,
            "action": action,
            "command": " ".join(cmd),
            "error": "Required CLI tool not found in PATH",
        })
    except Exception as e:
        return f"Error during deploy: {e}"


EncreDeployTool = build_tool(
    name="deploy",
    description="Deploy applications to cloud platforms and container registries",
    input_schema={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["docker", "kubernetes", "cloud_run", "vercel", "netlify"],
                "description": "Deployment target platform",
            },
            "action": {
                "type": "string",
                "enum": ["build", "deploy", "rollback", "status"],
                "description": "Action to perform",
            },
            "config_file": {
                "type": "string",
                "description": "Path to configuration file (Dockerfile, k8s manifest, etc.)",
            },
            "project_name": {
                "type": "string",
                "description": "Project or application name",
            },
        },
        "required": ["target", "action"],
    },
    execute=_deploy_execute,
    intents=["coding"],
)
