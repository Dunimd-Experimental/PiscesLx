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
PiscesL1 Publishing Tools

This module provides a complete publishing pipeline for PiscesL1 models,
including model export, Docker image building, and registry publishing.

The publishing tools follow a modular architecture:
    - PiscesLxPublishConfig: Configuration management
    - PiscesLxPublishExporter: Model export functionality
    - PiscesLxPublishDockerBuilder: Docker image building
    - PiscesLxPublishRegistry: Registry publishing
    - PiscesLxPublishOrchestrator: Pipeline orchestration
    - PiscesLxToolsPublish: Main entry point

Usage Examples:
    Python API:
        >>> from tools.publish.runner import PiscesLxToolsPublish
        >>> publish_tool = PiscesLxToolsPublish({
        ...     "model_size": "7B",
        ...     "model_path": "./checkpoints/7B"
        ... })
        >>> results = publish_tool.run()

    Command Line:
        $ python -m tools.publish.cli publish --model-size 7B --model-path ./checkpoints/7B
        $ python -m tools.publish.cli export --model-path ./checkpoints/7B
        $ python -m tools.publish.cli build --model-size 7B --image-name myrepo/piscesl1-7b
        $ python -m tools.publish.cli push --image myrepo/piscesl1-7b:v1.0.0

Architecture:
    tools.publish
    ├── config.py          # PiscesLxPublishConfig, PiscesLxPublishModelExportConfig,
    │                       # PiscesLxPublishDockerConfig, PiscesLxPublishRegistryConfig,
    │                       # PiscesLxPublishChecksumConfig, PiscesLxPublishMetadataConfig
    ├── exporter.py        # PiscesLxPublishExporter
    ├── docker_builder.py  # PiscesLxPublishDockerBuilder
    ├── registry.py        # PiscesLxPublishRegistry
    ├── orchestrator.py    # PiscesLxPublishOrchestrator
    ├── runner.py          # PiscesLxToolsPublish, PiscesLxPublishRunner
    └── cli.py             # Command-line interface

Modules:
    config: Configuration classes for all publishing stages.
    exporter: Model weight, config, and tokenizer export.
    docker_builder: Docker image building with templates.
    registry: Container registry publishing with authentication.
    orchestrator: Pipeline stage coordination and execution.
    runner: Main entry point with unified interface.
    cli: Command-line interface with subcommands.

Example Pipeline:
    1. Export: Convert checkpoint to distributable format
       >>> exporter = PiscesLxPublishExporter(config)
       >>> exporter.export()

    2. Build: Create Docker image with inference engine
       >>> builder = PiscesLxPublishDockerBuilder(config)
       >>> builder.build()

    3. Publish: Push to container registry
       >>> publisher = PiscesLxPublishRegistry(config)
       >>> publisher.publish()

Or use the orchestrator for the full pipeline:
    >>> orchestrator = PiscesLxPublishOrchestrator(config)
    >>> results = orchestrator.run()

Environment Variables:
    DOCKER_REGISTRY_USERNAME: Username for Docker registry
    DOCKER_REGISTRY_PASSWORD: Password for Docker registry
    DOCKER_REGISTRY_URL: Default registry URL

Supported Registries:
    - Docker Hub (docker.io)
    - GitHub Container Registry (ghcr.io)
    - NVIDIA NGC (nvcr.io)
    - Azure Container Registry (azurecr.io)
    - Google Container Registry (gcr.io)
"""

from tools.publish.runner import PiscesLxToolsPublish
from tools.publish.runner import PiscesLxPublishRunner
from tools.publish.orchestrator import PiscesLxPublishOrchestrator
from tools.publish.config import PiscesLxPublishConfig
from tools.publish.exporter import PiscesLxPublishExporter
from tools.publish.docker_builder import PiscesLxPublishDockerBuilder
from tools.publish.registry import PiscesLxPublishRegistry

__all__ = [
    "PiscesLxToolsPublish",
    "PiscesLxPublishRunner",
    "PiscesLxPublishOrchestrator",
    "PiscesLxPublishConfig",
    "PiscesLxPublishExporter",
    "PiscesLxPublishDockerBuilder",
    "PiscesLxPublishRegistry",
]


def get_version() -> str:
    """Get the version of the publishing tools.

    Returns:
        Version string from configs.version.
    """
    from configs.version import VERSION
    return VERSION
