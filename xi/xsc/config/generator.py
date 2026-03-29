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
Default Configuration Generator for Xi Studio.

This module provides functions to generate default .xi/ directory
structure with configuration files, command definitions, and schema.
"""

from pathlib import Path
from typing import Optional

from ..core.dc import XiLogger
from .defaults import (
    DEFAULT_XI_TOML,
    DEFAULT_TRAIN_TOML,
    DEFAULT_INFERENCE_TOML,
    DEFAULT_BENCHMARK_TOML,
    DEFAULT_DOWNLOAD_TOML,
    DEFAULT_SCHEMA_JSON,
)


class XiConfigGenerator:
    """
    Generator for default Xi configuration structure.
    
    This class creates the .xi/ directory with all necessary
    configuration files and command definitions.
    """
    
    CONFIG_DIR = ".xi"
    COMMANDS_DIR = "commands"
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the generator.
        
        Args:
            project_root: Path to project root. If None, uses current directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.config_dir = self.project_root / self.CONFIG_DIR
        self.commands_dir = self.config_dir / self.COMMANDS_DIR
        self.logger = XiLogger("Xi.ConfigGenerator", enable_file=True)
    
    def generate(self, force: bool = False) -> bool:
        """
        Generate the complete .xi/ structure.
        
        Args:
            force: If True, overwrite existing files
            
        Returns:
            True if generation was successful
        """
        try:
            self._create_directories()
            self._create_main_config(force)
            self._create_commands(force)
            self._create_schema(force)
            
            self.logger.info(
                f"Generated .xi/ configuration at {self.config_dir}",
                event="xi.config.generated"
            )
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to generate configuration: {e}",
                event="xi.config.generate_error"
            )
            return False
    
    def _create_directories(self) -> None:
        """Create .xi/ and .xi/commands/ directories."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.commands_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(
            f"Created directories: {self.config_dir}",
            event="xi.config.dirs_created"
        )
    
    def _create_main_config(self, force: bool) -> None:
        """
        Create main xi.toml configuration file.
        
        Args:
            force: If True, overwrite existing file
        """
        config_file = self.config_dir / "xi.toml"
        
        if config_file.exists() and not force:
            self.logger.debug(
                f"Config file exists, skipping: {config_file}",
                event="xi.config.exists"
            )
            return
        
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(DEFAULT_XI_TOML)
        
        self.logger.info(
            f"Created main config: {config_file}",
            event="xi.config.main_created"
        )
    
    def _create_commands(self, force: bool) -> None:
        """
        Create command definition TOML files.
        
        Args:
            force: If True, overwrite existing files
        """
        commands = {
            "train.toml": DEFAULT_TRAIN_TOML,
            "inference.toml": DEFAULT_INFERENCE_TOML,
            "benchmark.toml": DEFAULT_BENCHMARK_TOML,
            "download.toml": DEFAULT_DOWNLOAD_TOML,
        }
        
        for filename, content in commands.items():
            cmd_file = self.commands_dir / filename
            
            if cmd_file.exists() and not force:
                continue
            
            with open(cmd_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            self.logger.debug(
                f"Created command file: {cmd_file}",
                event="xi.config.command_created"
            )
    
    def _create_schema(self, force: bool) -> None:
        """
        Create JSON schema file for validation.
        
        Args:
            force: If True, overwrite existing file
        """
        schema_file = self.config_dir / "schema.json"
        
        if schema_file.exists() and not force:
            return
        
        with open(schema_file, "w", encoding="utf-8") as f:
            f.write(DEFAULT_SCHEMA_JSON)
        
        self.logger.debug(
            f"Created schema file: {schema_file}",
            event="xi.config.schema_created"
        )
    
    def exists(self) -> bool:
        """
        Check if .xi/ configuration directory exists.
        
        Returns:
            True if .xi/ directory exists with xi.toml
        """
        return (self.config_dir / "xi.toml").exists()
    
    def get_config_path(self) -> Path:
        """Get path to main configuration file."""
        return self.config_dir / "xi.toml"
    
    def get_commands_path(self) -> Path:
        """Get path to commands directory."""
        return self.commands_dir


def create_default_config(project_root: Optional[Path] = None, force: bool = False) -> bool:
    """
    Create default .xi/ configuration structure.
    
    This is a convenience function that creates a XiConfigGenerator
    and generates the configuration.
    
    Args:
        project_root: Path to project root
        force: If True, overwrite existing files
        
    Returns:
        True if generation was successful
    """
    generator = XiConfigGenerator(project_root)
    return generator.generate(force)


def ensure_config_exists(project_root: Optional[Path] = None) -> Path:
    """
    Ensure .xi/ configuration exists, creating it if necessary.
    
    Args:
        project_root: Path to project root
        
    Returns:
        Path to the .xi/ directory
    """
    generator = XiConfigGenerator(project_root)
    
    if not generator.exists():
        generator.generate(force=False)
    
    return generator.config_dir
