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

from __future__ import annotations

"""
Path Management and Runtime Configuration Module for PiscesLx.

This module provides centralized path management for the PiscesLx framework,
handling all filesystem operations related to cache directories, working
directories, and runtime configuration loading. It establishes a consistent
directory structure for storing model artifacts, checkpoints, logs, and
temporary files.

Directory Structure:
    The PiscesLx framework uses a standardized directory hierarchy under
    the home directory (default: .pisceslx/):
    
    .pisceslx/
    ├── cache/              # General cache storage
    │   ├── datasets/       # Downloaded dataset cache
    │   ├── models/         # Pre-trained model cache
    │   └── tokens/         # Tokenizer cache
    ├── checkpoints/        # Training checkpoints
    ├── logs/               # Log files
    ├── runs/               # Training/inference run artifacts
    └── tmp/                # Temporary files

Key Responsibilities:
    1. Runtime Configuration Loading:
       - Load environment-based configuration with PISCESLX_ prefix
       - Graceful fallback to defaults on configuration errors
       - Support for environment variable overrides
    
    2. Home Directory Management:
       - Resolve and create the PiscesLx home directory
       - Ensure directory exists before any operations
       - Cross-platform path handling (Windows/Linux/macOS)
    
    3. Cache Directory Management:
       - Create and manage named cache directories
       - Support for different cache types (datasets, models, etc.)
       - Automatic directory creation on access
    
    4. Working Directory Management:
       - Provide isolated working directories for different operations
       - Support for training runs, inference sessions, etc.

Thread Safety:
    All functions in this module are thread-safe. Directory creation
    operations use atomic checks to prevent race conditions.

Environment Variables:
    The following environment variables can be used to override defaults:
    
    - PISCESLX_HOME: Override the home directory location
    - PISCESLX_CACHE_DIR: Override the cache directory location
    - PISCESLX_LOG_LEVEL: Set the logging verbosity level

Usage Examples:
    Basic usage:
        from utils.paths import PiscesLxPaths
        
        home = get_pisceslx_home_dir()  # Creates .pisceslx/ if needed
        cache = PiscesLxPaths.get_cache_dir("datasets")  # Creates .pisceslx/datasets/
    
    Runtime configuration:
        from utils.paths import PiscesLxPaths
        
        config = load_runtime_configuration()
        log_level = config.get("log_level", "INFO")
    
    Working directory for training:
        from utils.paths import PiscesLxPaths
        
        work_dir = PiscesLxPaths.get_work_dir("runs/experiment_001")

Integration Points:
    - tools/data/download/: Uses PiscesLxPaths.get_cache_dir("datasets") for dataset storage
    - opss/train/checkpoint.py: Uses PiscesLxPaths.get_work_dir("checkpoints") for saves
    - tools/monitor/: Uses PiscesLxPaths.get_cache_dir("logs") for log storage
    - manage.py: Initializes paths at startup
"""

import os
from pathlib import Path
from typing import Optional

from utils.dc import PiscesLxConfiguration, PiscesLxFilesystem, PiscesLxLogger

_LOG = PiscesLxLogger("PiscesLx.Core.Paths")

#: Environment override for the artifact root. Documented since forever, implemented
#: only now: it was declared in this module's own docstring while
#: resolve_pisceslx_home() returned the literal default, so every artifact -- checkpoints,
#: caches, logs -- was pinned next to the repository no matter what was configured.
ENV_HOME = "PISCESLX_HOME"
#: Default artifact root, relative to the current working directory.
DEFAULT_PISCESLX_HOME = ".pisceslx"


class PiscesLxPaths:
    """Path and runtime-configuration accessors for PiscesL1.

    Methods rather than free functions: standard 6.2 forbids exposing module-level
    functions, so this class is the single entry point for the former path helpers.
    """

    @staticmethod
    def load_runtime_configuration() -> PiscesLxConfiguration:
        """
        Load runtime configuration from environment variables.

        This function creates a configuration object and populates it with
        values from environment variables prefixed with "PISCESLX_". This
        allows for runtime customization without modifying code files.

        The function gracefully handles configuration loading errors by
        logging a warning and returning an empty configuration object,
        ensuring the application can continue with default settings.

        Environment Variable Mapping:
            PISCESLX_LOG_LEVEL -> log_level
            PISCESLX_CACHE_DIR -> cache_dir
            PISCESLX_HOME -> home_dir
            PISCESLX_MAX_WORKERS -> max_workers

        Args:
            None

        Returns:
            PiscesLxConfiguration: A configuration object populated with
                environment variable values. If loading fails, returns an
                empty configuration object with default values.

        Raises:
            No exceptions are raised; errors are logged and handled gracefully.

        Example:
            >>> config = PiscesLxPaths.load_runtime_configuration()
            >>> log_level = config.get("log_level", "INFO")
            >>> print(f"Log level: {log_level}")
            Log level: INFO
        """
        cfg = PiscesLxConfiguration()
        try:
            cfg.load_from_env(prefix="PISCESLX_")
        except Exception as e:
            _LOG.warning("runtime_env_load_failed", error=str(e))
        return cfg

    @staticmethod
    def resolve_pisceslx_home() -> str:
        """
        Resolve the PiscesLx home directory path.

        This function returns the path to the PiscesLx home directory where
        all framework artifacts are stored. The default location is ".pisceslx"
        in the current working directory, but this can be overridden via
        environment variables or configuration.

        The home directory serves as the root for:
        - Cache directories (models, datasets, tokens)
        - Checkpoint storage
        - Log files
        - Temporary files
        - Run artifacts

        Args:
            None

        Returns:
            str: The path to the PiscesLx home directory. This is a relative
                path by default (".pisceslx"), but can be absolute if
                configured via environment variables.

        Note:
            This function only returns the path; it does not create the
            directory. Use PiscesLxPaths.get_pisceslx_home_dir() to ensure the directory
            exists.

        Example:
            >>> home = PiscesLxPaths.resolve_pisceslx_home()
            >>> print(home)
            .pisceslx

            >>> import os
            >>> os.environ["PISCESLX_HOME"] = "<volume>/pisceslx-artifacts"
            >>> PiscesLxPaths.resolve_pisceslx_home()
            '<volume>/pisceslx-artifacts'
        """
        override = os.environ.get(ENV_HOME, "").strip()
        return override if override else DEFAULT_PISCESLX_HOME

    @staticmethod
    def get_pisceslx_home_dir() -> str:
        """
        Get the PiscesLx home directory, creating it if necessary.

        This function ensures the PiscesLx home directory exists and returns
        its path. It combines PiscesLxPaths.resolve_pisceslx_home() with directory creation
        to provide a guaranteed-valid directory path.

        The function uses PiscesLxFilesystem for directory operations, which
        handles cross-platform path normalization and permission checks.

        Args:
            None

        Returns:
            str: The absolute or relative path to the PiscesLx home directory.
                The directory is guaranteed to exist when this function returns.

        Raises:
            OSError: If directory creation fails due to permissions or
                filesystem errors.

        Example:
            >>> home = PiscesLxPaths.get_pisceslx_home_dir()
            >>> print(f"Home directory: {home}")
            Home directory: .pisceslx

            >>> import os
            >>> os.path.exists(home)
            True
        """
        home = PiscesLxPaths.resolve_pisceslx_home()
        fs = PiscesLxFilesystem()
        fs.mkdir(home)
        return home

    @staticmethod
    def get_cache_dir(name: str) -> str:
        """
        Get or create a named cache directory under the PiscesLx home.

        This function creates and returns a subdirectory within the PiscesLx
        home directory, suitable for storing cached data such as downloaded
        datasets, pre-trained models, or tokenizer files.

        Common cache directory names:
            - "datasets": Downloaded and preprocessed datasets
            - "models": Pre-trained model weights
            - "tokens": Tokenizer files and vocabulary
            - "logs": Log files and monitoring data
            - "tmp": Temporary files (may be cleaned periodically)

        The function ensures both the home directory and the named cache
        directory exist before returning the path.

        Args:
            name (str): The name of the cache subdirectory. This will be
                created as a subdirectory under the PiscesLx home directory.
                Examples: "datasets", "models", "checkpoints"

        Returns:
            str: The path to the named cache directory. The directory is
                guaranteed to exist when this function returns.

        Raises:
            OSError: If directory creation fails due to permissions or
                filesystem errors.
            ValueError: If name is empty or contains invalid path characters.

        Example:
            >>> cache_dir = PiscesLxPaths.get_cache_dir("datasets")
            >>> print(f"PiscesLxDataset cache: {cache_dir}")
            PiscesLxDataset cache: .pisceslx/datasets

            >>> model_cache = PiscesLxPaths.get_cache_dir("models")
            >>> print(f"Model cache: {model_cache}")
            Model cache: .pisceslx/models
        """
        root = PiscesLxPaths.get_pisceslx_home_dir()
        fs = PiscesLxFilesystem()
        p = str(Path(root) / str(name))
        fs.mkdir(p)
        return p

    @staticmethod
    def get_work_dir(name: str) -> str:
        """
        Get or create a working directory for a specific operation.

        This function is an alias for PiscesLxPaths.get_cache_dir(), providing semantic
        clarity when the directory is used for active work rather than
        caching. Working directories are typically used for:

        - Training run artifacts (checkpoints, logs, configs)
        - Inference session outputs
        - Benchmark results
        - Experiment tracking data

        The function ensures the directory exists and returns its path,
        making it ready for immediate use.

        Args:
            name (str): The name of the working directory. This can include
                subdirectories using path separators.
                Examples: "runs/exp001", "inference/session_1"

        Returns:
            str: The path to the working directory. The directory is
                guaranteed to exist when this function returns.

        Raises:
            OSError: If directory creation fails due to permissions or
                filesystem errors.

        Example:
            >>> work_dir = PiscesLxPaths.get_work_dir("runs/experiment_001")
            >>> print(f"Working directory: {work_dir}")
            Working directory: .pisceslx/runs/experiment_001

            >>> checkpoint_dir = PiscesLxPaths.get_work_dir("checkpoints/epoch_10")
            >>> print(f"Checkpoint directory: {checkpoint_dir}")
            Checkpoint directory: .pisceslx/checkpoints/epoch_10
        """
        return PiscesLxPaths.get_cache_dir(name)

    @staticmethod
    @staticmethod
    def resolve_output_dir(value: Optional[str], fallback: str = "ckpt") -> str:
        """Resolve a configured artifact directory against the (possibly moved) home root.

        A configured path is used as written when it is absolute. A relative one that
        already begins with the default home (``.pisceslx/ckpt``) is re-rooted under the
        configured home instead of being duplicated into ``<home>/.pisceslx/ckpt``; any
        other relative path stays relative to the working directory, which is what the
        config schema promises.

        Args:
            value: The configured ``output_dir`` (may be empty or None).
            fallback: Sub-directory of the home used when nothing is configured.

        Returns:
            str: The directory the trainer should write to.
        """
        text = str(value or "").strip()
        home = PiscesLxPaths.get_pisceslx_home_dir()
        if not text:
            return str(Path(home) / fallback)
        candidate = Path(text)
        if candidate.is_absolute():
            return text
        default_parts = Path(DEFAULT_PISCESLX_HOME).parts
        if candidate.parts[:len(default_parts)] == default_parts:
            return str(Path(home) / Path(*candidate.parts[len(default_parts):] or "."))
        return text

    def get_checkpoint_dir(name: str) -> str:
        """Get or create a checkpoint directory.

        Args:
            name: The checkpoint group name (e.g., "evolution", "training").

        Returns:
            str: The checkpoint directory path.
        """
        return PiscesLxPaths.get_cache_dir(f"checkpoints/{name}")

    @staticmethod
    def get_log_file(module_name: str) -> str:
        """
        Get the log file path for a module.

        Args:
            module_name: The module name (e.g., "model.core.attention")

        Returns:
            str: The log file path (e.g., ".pisceslx/logs/model/core.log")
        """
        parts = module_name.split(".")
        if len(parts) >= 2:
            log_dir = PiscesLxPaths.get_cache_dir(f"logs/{parts[0]}/{parts[1]}")
        elif len(parts) == 1:
            log_dir = PiscesLxPaths.get_cache_dir(f"logs/{parts[0]}")
        else:
            log_dir = PiscesLxPaths.get_cache_dir("logs")

        return str(Path(log_dir) / "output.log")

    @staticmethod
    def get_settings_file() -> str:
        """
        Get the settings.yaml file path.

        This function returns the path to the global settings configuration file
        used for storing framework-wide settings like developer mode toggle.

        The settings file is located at:
            .pisceslx/settings/settings.yaml

        Returns:
            str: The absolute path to the settings.yaml file.
                The parent directory is guaranteed to exist.

        Example:
            >>> settings_path = PiscesLxPaths.get_settings_file()
            >>> print(settings_path)
            .pisceslx/settings/settings.yaml
        """
        settings_dir = PiscesLxPaths.get_cache_dir("settings")
        return str(Path(settings_dir) / "settings.yaml")

