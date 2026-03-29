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
Argument building utilities for Xi Executor.

This module provides functions to build command line arguments
from schema definitions and parameter values.
"""

import os
import sys
from typing import Optional, Dict, Any, List

from ..core.types import XiCommand
from ..config import XiValueMapping


def build_argv(request, root_dir) -> List[str]:
    """
    Build command line arguments from XiRequest.
    
    Args:
        request: XiRequest object
        root_dir: Working directory path
        
    Returns:
        List of command line argument strings
    """
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
    command_name: str,
    parameters: Dict[str, Any],
    schema_params: List[Any],
    config_loader=None
) -> List[str]:
    """
    Build command line arguments from schema and parameter values.
    
    This method uses the value_mapping configuration from the schema
    to properly format each parameter value for the command line.
    
    Args:
        command_name: Name of the command (e.g., "train")
        parameters: Dictionary of parameter values from frontend
        schema_params: List of parameter schema objects with value_mapping
        config_loader: Optional config loader instance
        
    Returns:
        List of command line argument strings
    """
    argv = [sys.executable, "manage.py", command_name]
    
    if config_loader:
        config = config_loader.get_config()
        cmd_config = config.commands.get(command_name)
        
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
            arg_str = apply_value_mapping(param_name, value, mapping)
            if arg_str:
                if isinstance(arg_str, list):
                    argv.extend(arg_str)
                else:
                    argv.append(arg_str)
        else:
            default_mapping = XiValueMapping()
            arg_str = apply_value_mapping(param_name, value, default_mapping)
            if arg_str:
                if isinstance(arg_str, list):
                    argv.extend(arg_str)
                else:
                    argv.append(arg_str)
    
    return argv


def apply_value_mapping(
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
        if evaluate_skip_condition(value, mapping.skip_if):
            return None
    
    if value is None or value == "":
        if mapping.default_if_empty is not None:
            value = mapping.default_if_empty
        else:
            return None
    
    transformed_value = transform_value(value, mapping.transform)
    
    if mapping.template:
        return apply_template(mapping.template, name, transformed_value, mapping)
    
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


def transform_value(value: Any, transform: str) -> Any:
    """
    Apply transformation to value.
    
    Args:
        value: Value to transform
        transform: Transformation type
        
    Returns:
        Transformed value
    """
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


def evaluate_skip_condition(value: Any, condition: str) -> bool:
    """
    Evaluate whether to skip this argument based on condition.
    
    Args:
        value: Value to check
        condition: Condition string
        
    Returns:
        True if should skip
    """
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


def apply_template(
    template: str,
    name: str,
    value: Any,
    mapping: XiValueMapping
) -> str:
    """
    Apply a template for complex argument generation.
    
    Args:
        template: Template string
        name: Parameter name
        value: Parameter value
        mapping: Value mapping configuration
        
    Returns:
        Formatted argument string
    """
    result = template.replace("{name}", name)
    result = result.replace("{value}", str(value))
    result = result.replace("{prefix}", mapping.arg_prefix or "--")
    result = result.replace("{separator}", mapping.arg_separator or " ")
    
    if isinstance(value, list):
        result = result.replace("{joined}", mapping.join_with.join(str(v) for v in value))
    
    return result
