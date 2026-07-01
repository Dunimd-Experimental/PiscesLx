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
Publish Configuration Management System

Centralized configuration management for the PiscesLx model publishing pipeline.
This module provides a hierarchical configuration system using dataclasses,
enabling type-safe, validated, and serializable configuration objects.

Configuration Hierarchy:
    PiscesLxPublishConfig (root)
    ├── ModelExportConfig: Model export settings
    ├── DockerBuildConfig: Docker build settings
    ├── RegistryPublishConfig: Image registry publish settings
    └── ChecksumConfig: File checksum verification settings

Design Principles:
    - Type Safety: All configurations use Python dataclasses with type hints
    - Default Values: Sensible defaults for all parameters
    - Serialization: Full support for JSON/YAML serialization
    - Validation: Built-in validation through __post_init__ methods
    - Extensibility: Easy to extend with new configuration categories

Features:
    - Model Export: Support for safetensors and pytorch formats
    - Docker Build: Customizable base images and build arguments
    - Registry Publish: Multi-registry support (Docker Hub, GHCR, NGC, etc.)
    - Checksum: SHA256 verification for published artifacts

Usage Examples:
    Basic Configuration:
        >>> from tools.publish.config import PiscesLxPublishConfig
        >>> config = PiscesLxPublishConfig(
        ...     model_size="7B",
        ...     model_name="PiscesLx",
        ...     model_path="./checkpoints/7B"
        ... )

    Export Only:
        >>> from tools.publish.config import PiscesLxPublishConfig, ExportAction
        >>> config = PiscesLxPublishConfig(
        ...     model_path="./checkpoints/7B",
        ...     action=ExportAction.EXPORT_ONLY
        ... )

    Full Publish:
        >>> config = PiscesLxPublishConfig(
        ...     model_path="./checkpoints/7B",
        ...     docker=PiscesLxPublishDockerConfig(
        ...         image_name="piscesl1/piscesl1-7b",
        ...         image_tag="v1.0.0"
        ...     ),
        ...     registry=PiscesLxPublishRegistryConfig(
        ...         registry_url="docker.io",
        ...         repository="piscesl1"
        ...     )
        ... )

Integration:
    - PiscesLxPublishOrchestrator: Uses PiscesLxPublishConfig for pipeline
    - PiscesLxPublishExporter: Reads export configuration
    - PiscesLxPublishDockerBuilder: Uses docker configuration
    - PiscesLxPublishRegistry: Uses registry configuration

Supported Registries:
    - docker.io: Docker Hub
    - ghcr.io: GitHub Container Registry
    - nvcr.io: NVIDIA NGC
    - azurecr.io: Azure Container Registry
    - gcr.io: Google Container Registry
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class ExportAction(Enum):
    """Export action types"""
    EXPORT_ONLY = "export_only"
    BUILD_ONLY = "build_only"
    PUBLISH_ONLY = "publish_only"
    ALL = "all"


class ExportFormat(Enum):
    """Model export format types"""
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"


@dataclass
class PiscesLxPublishModelExportConfig:
    """Model export configuration.

    Defines parameters for exporting model weights, configurations,
    tokenizer, and generation settings.

    Attributes:
        checkpoint_path: Path to the model checkpoint file.
        output_dir: Directory to save exported artifacts.
        export_format: Export format (safetensors or pytorch).
        include_config: Whether to include model config.
        include_tokenizer: Whether to include tokenizer files.
        include_generation_config: Whether to include generation config.
        quantization: Quantization format (None, int8, int4, fp8).
        compression: Compression type (None, gzip, zip).

    Example:
        >>> export_config = PiscesLxPublishModelExportConfig(
        ...     checkpoint_path="./checkpoints/7B/model.pt",
        ...     output_dir="./publish/7B",
        ...     export_format=ExportFormat.SAFETENSORS,
        ...     quantization="int8"
        ... )
    """
    checkpoint_path: str = ""
    output_dir: str = "./publish"
    export_format: str = "safetensors"
    include_config: bool = True
    include_tokenizer: bool = True
    include_generation_config: bool = True
    quantization: Optional[str] = None
    compression: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.export_format not in [e.value for e in ExportFormat]:
            raise ValueError(f"Invalid export format: {self.export_format}")
        if self.quantization not in [None, "int8", "int4", "fp8"]:
            raise ValueError(f"Invalid quantization: {self.quantization}")


@dataclass
class PiscesLxPublishDockerConfig:
    """Docker build configuration.

    Defines parameters for building Docker images containing the model
    and inference engine.

    Attributes:
        base_image: Base Docker image with CUDA support.
        image_name: Name of the Docker image.
        image_tag: Tag for the Docker image.
        dockerfile_template: Template for Dockerfile generation.
        build_args: Additional build arguments for Docker.
        context: Build context (path or URL).
        use_cache: Whether to use Docker cache during build.
        platform: Target platform (linux/amd64, linux/arm64).

    Example:
        >>> docker_config = PiscesLxPublishDockerConfig(
        ...     base_image="nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04",
        ...     image_name="piscesl1/piscesl1-7b",
        ...     image_tag="v1.0.0",
        ...     platform="linux/amd64"
        ... )
    """
    base_image: str = "nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04"
    image_name: str = "piscesl1/piscesl1"
    image_tag: str = "latest"
    dockerfile_template: str = "default"
    build_args: Dict[str, str] = field(default_factory=dict)
    context: str = "."
    use_cache: bool = True
    platform: str = "linux/amd64"

    @property
    def full_image_name(self) -> str:
        """Get full image name with tag."""
        return f"{self.image_name}:{self.image_tag}"

    @property
    def full_image_with_registry(self) -> str:
        """Get full image name with registry."""
        return f"{self.image_name}:{self.image_tag}"


@dataclass
class PiscesLxPublishRegistryConfig:
    """Registry publish configuration.

    Defines parameters for publishing Docker images to container registries.

    Attributes:
        registry_url: URL of the container registry.
        repository: Repository name within the registry.
        image_tag: Tag for the published image.
        make_latest: Whether to also tag as 'latest'.
        credentials: Registry authentication credentials.
        retry_count: Number of retries for publish operations.
        timeout: Timeout in seconds for network operations.

    Supported Registries:
        - docker.io: Docker Hub (default)
        - ghcr.io: GitHub Container Registry
        - nvcr.io: NVIDIA NGC
        - azurecr.io: Azure Container Registry
        - gcr.io: Google Container Registry

    Example:
        >>> registry_config = PiscesLxPublishRegistryConfig(
        ...     registry_url="docker.io",
        ...     repository="piscesl1/piscesl1-7b",
        ...     image_tag="v1.0.0",
        ...     make_latest=True
        ... )
    """
    registry_url: str = "docker.io"
    repository: str = "piscesl1"
    image_tag: str = "latest"
    make_latest: bool = True
    credentials: Optional[Dict[str, str]] = None
    retry_count: int = 3
    timeout: int = 300

    @property
    def full_image_url(self) -> str:
        """Get full image URL for pushing."""
        return f"{self.registry_url}/{self.repository}:{self.image_tag}"

    @property
    def latest_image_url(self) -> str:
        """Get latest tag image URL."""
        return f"{self.registry_url}/{self.repository}:latest"


@dataclass
class PiscesLxPublishChecksumConfig:
    """Checksum verification configuration.

    Defines parameters for file checksum generation and verification.

    Attributes:
        algorithm: Hash algorithm (sha256, sha512, md5).
        include_patterns: File patterns to include in checksum.
        exclude_patterns: File patterns to exclude from checksum.
        generate_manifest: Whether to generate checksum manifest.
        verify_before_publish: Whether to verify checksums before publishing.

    Example:
        >>> checksum_config = PiscesLxPublishChecksumConfig(
        ...     algorithm="sha256",
        ...     generate_manifest=True,
        ...     verify_before_publish=True
        ... )
    """
    algorithm: str = "sha256"
    include_patterns: List[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: List[str] = field(default_factory=lambda: [".git/*", "*.tmp"])
    generate_manifest: bool = True
    verify_before_publish: bool = True


@dataclass
class PiscesLxPublishMetadataConfig:
    """Metadata configuration for model cards.

    Defines metadata to be included in the model card and package manifest.

    Attributes:
        model_name: Name of the model.
        model_version: Version string.
        architecture: Model architecture.
        parameters: Number of parameters.
        context_length: Maximum context length.
        modalities: Supported modalities.
        training_tokens: Number of training tokens.
        training_hardware: Hardware used for training.
        license: License for the model.
        author: Author information.
        description: Model description.
        citation: Citation information.
        homepage: Project homepage URL.
        repository: Repository URL.

    Example:
        >>> metadata = PiscesLxPublishMetadataConfig(
        ...     model_name="PiscesLx-7B",
        ...     model_version="1.0.0",
        ...     architecture="Yv",
        ...     parameters="7B",
        ...     context_length=10485760,
        ...     modalities=["text", "image", "audio", "video", "document", "agentic"],
        ...     license="Apache-2.0"
        ... )
    """
    model_name: str = "PiscesLx"
    model_version: str = "1.0.0"
    architecture: str = "Yv"
    parameters: str = "7B"
    context_length: int = 10485760
    modalities: List[str] = field(default_factory=lambda: ["text", "image", "audio", "video", "document", "agentic"])
    training_tokens: str = ""
    training_hardware: str = ""
    license: str = "Apache-2.0"
    author: str = "Dunimd Team"
    description: str = ""
    citation: str = ""
    homepage: str = ""
    repository: str = ""

    def to_model_card(self) -> Dict[str, Any]:
        """Convert to model card format."""
        return {
            "model_name": self.model_name,
            "version": self.model_version,
            "architecture": self.architecture,
            "parameters": self.parameters,
            "context_length": self.context_length,
            "modalities": self.modalities,
            "capabilities": {
                "text_generation": "text" in self.modalities,
                "image_generation": "image" in self.modalities,
                "audio_generation": "audio" in self.modalities,
                "video_generation": "video" in self.modalities,
                "tool_calling": "agentic" in self.modalities,
                "multimodal": len(self.modalities) > 1,
            },
            "training": {
                "tokens": self.training_tokens,
                "hardware": self.training_hardware,
            },
            "license": self.license,
            "author": self.author,
            "description": self.description,
            "citation": self.citation,
            "homepage": self.homepage,
            "repository": self.repository,
        }


@dataclass
class PiscesLxPublishConfig:
    """Root configuration for PiscesLx publishing pipeline.

    This is the main configuration class that contains all sub-configurations
    for the publishing workflow including export, docker build, registry publish,
    and metadata.

    Attributes:
        model_size: Model size identifier (e.g., "7B", "70B", "671B").
        model_name: Human-readable model name.
        action: Publishing action to perform.
        model_path: Path to the model checkpoint or directory.
        output_dir: Base output directory for artifacts.
        export: Model export configuration.
        docker: Docker build configuration.
        registry: Registry publish configuration.
        checksum: Checksum verification configuration.
        metadata: Model metadata configuration.

    Configuration Hierarchy:
        PiscesLxPublishConfig
        ├── PiscesLxPublishModelExportConfig
        ├── PiscesLxPublishDockerConfig
        ├── PiscesLxPublishRegistryConfig
        ├── PiscesLxPublishChecksumConfig
        └── PiscesLxPublishMetadataConfig

    Usage Examples:
        Minimal Configuration:
            >>> config = PiscesLxPublishConfig(
            ...     model_size="7B",
            ...     model_path="./checkpoints/7B"
            ... )

        Full Configuration:
            >>> config = PiscesLxPublishConfig(
            ...     model_size="7B",
            ...     model_name="PiscesLx-7B",
            ...     model_path="./checkpoints/7B",
            ...     output_dir="./publish",
            ...     action=ExportAction.ALL,
            ...     export=PiscesLxPublishModelExportConfig(
            ...         checkpoint_path="./checkpoints/7B/model.pt",
            ...         export_format="safetensors"
            ...     ),
            ...     docker=PiscesLxPublishDockerConfig(
            ...         image_name="piscesl1/piscesl1-7b",
            ...         image_tag="v1.0.0"
            ...     ),
            ...     registry=PiscesLxPublishRegistryConfig(
            ...         registry_url="docker.io"
            ...     ),
            ...     metadata=PiscesLxPublishMetadataConfig(
            ...         license="Apache-2.0",
            ...         description="PiscesLx 7B model"
            ...     )
            ... )

    CLI Usage:
        Command line interface automatically populates this config
        from CLI arguments. See tools.publish.cli for details.
    """
    model_size: str = "7B"
    model_name: str = "PiscesLx"
    action: str = "all"
    model_path: str = ""
    output_dir: str = "./publish"
    inference_engine_path: str = "./opss/infer"
    server_path: str = "./tools/infer/server.py"

    export: PiscesLxPublishModelExportConfig = field(default_factory=PiscesLxPublishModelExportConfig)
    docker: PiscesLxPublishDockerConfig = field(default_factory=PiscesLxPublishDockerConfig)
    registry: PiscesLxPublishRegistryConfig = field(default_factory=PiscesLxPublishRegistryConfig)
    checksum: PiscesLxPublishChecksumConfig = field(default_factory=PiscesLxPublishChecksumConfig)
    metadata: PiscesLxPublishMetadataConfig = field(default_factory=PiscesLxPublishMetadataConfig)

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        if not self.model_path:
            self.model_path = f"./checkpoints/{self.model_size}"

        self.export.checkpoint_path = self.export.checkpoint_path or self.model_path
        self.export.output_dir = self.export.output_dir or self.output_dir

        if not self.docker.image_name:
            self.docker.image_name = f"piscesl1/piscesl1-{self.model_size.lower()}"
        if not self.registry.repository:
            self.registry.repository = self.docker.image_name

        self.metadata.model_name = self.metadata.model_name or f"{self.model_name}-{self.model_size}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "model_size": self.model_size,
            "model_name": self.model_name,
            "action": self.action,
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "export": {
                "checkpoint_path": self.export.checkpoint_path,
                "output_dir": self.export.output_dir,
                "export_format": self.export.export_format,
                "include_config": self.export.include_config,
                "include_tokenizer": self.export.include_tokenizer,
                "include_generation_config": self.export.include_generation_config,
                "quantization": self.export.quantization,
                "compression": self.export.compression,
            },
            "docker": {
                "base_image": self.docker.base_image,
                "image_name": self.docker.image_name,
                "image_tag": self.docker.image_tag,
                "platform": self.docker.platform,
                "use_cache": self.docker.use_cache,
            },
            "registry": {
                "registry_url": self.registry.registry_url,
                "repository": self.registry.repository,
                "image_tag": self.registry.image_tag,
                "make_latest": self.registry.make_latest,
            },
            "metadata": self.metadata.to_model_card(),
        }

    def save_to_json(self, path: str) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save the JSON file.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PiscesLxPublishConfig':
        """Create configuration from dictionary.

        Args:
            data: Dictionary containing configuration data.

        Returns:
            PiscesLxPublishConfig instance.
        """
        export_data = data.get('export', {})
        docker_data = data.get('docker', {})
        registry_data = data.get('registry', {})
        metadata_data = data.get('metadata', {})

        export_config = PiscesLxPublishModelExportConfig(**export_data) if export_data else PiscesLxPublishModelExportConfig()
        docker_config = PiscesLxPublishDockerConfig(**docker_data) if docker_data else PiscesLxPublishDockerConfig()
        registry_config = PiscesLxPublishRegistryConfig(**registry_data) if registry_data else PiscesLxPublishRegistryConfig()

        metadata_config = PiscesLxPublishMetadataConfig(
            model_name=metadata_data.get('model_name', data.get('model_name', 'PiscesLx')),
            model_version=metadata_data.get('version', '1.0.0'),
            architecture=metadata_data.get('architecture', 'Yv'),
            parameters=metadata_data.get('parameters', data.get('model_size', '7B')),
            context_length=metadata_data.get('context_length', 10485760),
            license=metadata_data.get('license', 'Apache-2.0'),
        )

        return cls(
            model_size=data.get('model_size', '7B'),
            model_name=data.get('model_name', 'PiscesLx'),
            action=data.get('action', 'all'),
            model_path=data.get('model_path', ''),
            output_dir=data.get('output_dir', './publish'),
            export=export_config,
            docker=docker_config,
            registry=registry_config,
            metadata=metadata_config,
        )

    @classmethod
    def load_from_json(cls, path: str) -> 'PiscesLxPublishConfig':
        """Load configuration from JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            PiscesLxPublishConfig instance.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)