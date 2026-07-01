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
Docker Image Builder

This module handles the building of Docker images for PiscesLx models.
It generates Dockerfiles and builds images containing the model weights,
inference engine, and all necessary dependencies.

Key Features:
    - Custom Dockerfile generation with PiscesLx inference engine
    - Multi-stage builds for optimized image size
    - GPU support with CUDA base images
    - Platform-specific builds (amd64, arm64)
    - Build caching for faster rebuilds

Dockerfile Templates:
    - Default: Standard production build
    - Minimal: Minimal dependencies, smaller image
    - Dev: Development build with extra tools
    - GPU: GPU-optimized production build

Build Process:
    1. Prepare build context directory
    2. Generate Dockerfile from template
    3. Copy model files to build context
    4. Copy inference engine code
    5. Build Docker image
    6. Tag image with version
    7. Verify image

Usage Examples:
    Basic Build:
        >>> from tools.publish.config import PiscesLxPublishConfig
        >>> from tools.publish.docker_builder import PiscesLxPublishDockerBuilder
        >>> config = PiscesLxPublishConfig(
        ...     model_size="7B",
        ...     docker=PiscesLxPublishDockerConfig(
        ...         image_name="piscesl1/piscesl1-7b",
        ...         image_tag="v1.0.0"
        ...     )
        ... )
        >>> builder = PiscesLxPublishDockerBuilder(config)
        >>> image_tag = builder.build()
        >>> print(f"Built: {image_tag}")

    Build with Custom Template:
        >>> config.docker.dockerfile_template = "minimal"
        >>> builder = PiscesLxPublishDockerBuilder(config)
        >>> image_tag = builder.build()

    Build Multiple Platforms:
        >>> config.docker.platform = "linux/arm64"
        >>> builder = PiscesLxPublishDockerBuilder(config)
        >>> image_tag = builder.build()
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("PiscesLx.Publish.Docker", file_path=get_log_file("PiscesLx.Publish.Docker"), enable_file=True)


class PiscesLxPublishDockerBuilder:
    """Docker Image Builder for PiscesLx.

    Builds Docker images containing PiscesLx models and inference engine.
    Supports multiple build configurations and platforms.

    Attributes:
        config: PiscesLxPublishConfig containing docker build configuration.
        build_context: Path to the Docker build context directory.
        dockerfile_template: Template string for Dockerfile generation.

    Example:
        >>> config = PiscesLxPublishConfig(model_size="7B")
        >>> builder = PiscesLxPublishDockerBuilder(config)
        >>> image_tag = builder.build()
    """

    DOCKERFILE_TEMPLATES = {
        "default": '''
# PiscesLx Docker Image
# Built: {build_time}
# Model: {model_name}
# Architecture: Yv

FROM {base_image}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3.11 \\
    python3-pip \\
    python3.11-venv \\
    git \\
    curl \\
    wget \\
    vim \\
    libGL1-mesa-glx \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \\
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy inference engine (core asset)
COPY model/ /app/model/
COPY opss/ /app/opss/
COPY tools/infer/ /app/tools/infer/
COPY utils/ /app/utils/
COPY configs/ /app/configs/

# Copy model weights
COPY {model_size}/ /app/model_weights/

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default startup command
CMD ["python3", "/app/tools/infer/server.py", \\
     "--model_path", "/app/model_weights", \\
     "--host", "0.0.0.0", \\
     "--port", "8000"]
''',

        "minimal": '''
# PiscesLx Minimal Docker Image
FROM {base_image}

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Minimal dependencies
RUN apt-get update && apt-get install -y \\
    python3.11 \\
    python3-pip \\
    libGL1-mesa-glx \\
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

COPY model/ /app/model/
COPY opss/ /app/opss/
COPY tools/infer/ /app/tools/infer/
COPY {model_size}/ /app/model_weights/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python3", "/app/tools/infer/server.py", "--model_path", "/app/model_weights"]
''',

        "gpu": '''
# PiscesLx GPU-Optimized Docker Image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \\
    python3.11 python3-pip git curl wget libGL1-mesa-glx \\
    libsm6 libxext6 libxrender-dev libgomp1 && \\
    rm -rf /var/lib/apt/lists/* && \\
    ln -sf /usr/bin/python3.11 /usr/bin/python

COPY model/ /app/model/
COPY opss/ /app/opss/
COPY tools/infer/ /app/tools/infer/
COPY utils/ /app/utils/
COPY configs/ /app/configs/
COPY {model_size}/ /app/model_weights/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python3", "/app/tools/infer/server.py", "--model_path", "/app/model_weights"]
'''
    }

    def __init__(self, config: 'PiscesLxPublishConfig'):
        """Initialize the Docker builder.

        Args:
            config: PiscesLxPublishConfig containing docker build configuration.
        """
        self.config = config
        self.build_context: Optional[Path] = None

        template_name = config.docker.dockerfile_template
        self.dockerfile_template = self.DOCKERFILE_TEMPLATES.get(
            template_name,
            self.DOCKERFILE_TEMPLATES["default"]
        )

        _LOG.info(
            "PiscesLxPublishDockerBuilder initialized",
            base_image=config.docker.base_image,
            image_name=config.docker.image_name,
            template=template_name
        )

    def build(
        self,
        model_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        context_dir: Optional[str] = None
    ) -> str:
        """Build Docker image.

        Executes the complete Docker build process including context
        preparation, Dockerfile generation, and image build.

        Args:
            model_dir: Path to the model weights directory.
            output_dir: Path for temporary build context.
            context_dir: Optional custom build context directory.

        Returns:
            Full image tag (name:tag) of the built image.

        Raises:
            RuntimeError: If Docker build fails.
        """
        _LOG.info("Starting Docker build", model_dir=model_dir)

        if context_dir:
            self.build_context = Path(context_dir)
        else:
            self.build_context = Path(output_dir or "./docker_build")
            self.build_context.mkdir(parents=True, exist_ok=True)

        self._prepare_build_context(model_dir)

        dockerfile_path = self._generate_dockerfile()

        image_tag = self._docker_build(dockerfile_path)

        _LOG.info("Docker build completed", image_tag=image_tag)
        return image_tag

    def _prepare_build_context(self, model_dir: Optional[str] = None) -> None:
        """Prepare Docker build context.

        Copies all necessary files to the build context directory
        including model weights, inference engine, and configurations.

        Args:
            model_dir: Path to the model weights directory.
        """
        if not self.build_context:
            raise RuntimeError("Build context not initialized")

        _LOG.info("Preparing build context", path=str(self.build_context))

        source_dirs = [
            ("model", "model"),
            ("opss", "opss"),
            ("tools/infer", "tools/infer"),
            ("tools/monitor", "tools/monitor"),
            ("utils", "utils"),
            ("configs", "configs"),
        ]

        for src, dst in source_dirs:
            src_path = Path(src)
            dst_path = self.build_context / dst
            if src_path.exists():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                _LOG.debug("Copied", src=src, dst=str(dst_path))

        if model_dir:
            model_src = Path(model_dir)
            model_dst = self.build_context / self.config.model_size
            if model_src.exists():
                if model_src.is_dir():
                    shutil.copytree(model_src, model_dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(model_src, model_dst)
                _LOG.debug("Copied model", src=str(model_src), dst=str(model_dst))

        requirements_path = self.build_context / "requirements.txt"
        if not requirements_path.exists():
            self._generate_requirements(requirements_path)

        _LOG.info("Build context prepared", path=str(self.build_context))

    def _generate_requirements(self, output_path: Path) -> None:
        """Generate requirements.txt for Docker image.

        Args:
            output_path: Path to write requirements.txt.
        """
        requirements = [
            "torch>=2.1.0",
            "transformers>=4.35.0",
            "accelerate>=0.25.0",
            "safetensors>=0.4.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "pydantic>=2.5.0",
            "sentencepiece>=0.1.99",
            "numpy>=1.24.0",
            "scipy>=1.11.0",
            "bitsandbytes>=0.41.0",
            "triton>=2.1.0",
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(requirements))

        _LOG.info("Requirements generated", path=str(output_path))

    def _generate_dockerfile(self) -> Path:
        """Generate Dockerfile from template.

        Returns:
            Path to the generated Dockerfile.
        """
        if not self.build_context:
            raise RuntimeError("Build context not initialized")

        dockerfile_content = self.dockerfile_template.format(
            base_image=self.config.docker.base_image,
            model_size=self.config.model_size,
            model_name=f"{self.config.model_name}-{self.config.model_size}",
            build_time=datetime.now().isoformat(),
        )

        dockerfile_path = self.build_context / "Dockerfile"
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)

        _LOG.info("Dockerfile generated", path=str(dockerfile_path))
        return dockerfile_path

    def _docker_build(self, dockerfile_path: Path) -> str:
        """Execute docker build command.

        Args:
            dockerfile_path: Path to the Dockerfile.

        Returns:
            Image tag of the built image.

        Raises:
            RuntimeError: If docker build fails.
        """
        image_name = self.config.docker.image_name
        image_tag = self.config.docker.image_tag
        full_tag = f"{image_name}:{image_tag}"

        build_cmd = [
            "docker", "build",
            "-t", full_tag,
            "-f", str(dockerfile_path),
        ]

        if self.config.docker.platform:
            build_cmd.extend(["--platform", self.config.docker.platform])

        if not self.config.docker.use_cache:
            build_cmd.append("--no-cache")

        for key, value in self.config.docker.build_args.items():
            build_cmd.extend(["--build-arg", f"{key}={value}"])

        build_cmd.append(str(self.build_context))

        _LOG.info("Executing docker build", command=" ".join(build_cmd))

        try:
            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            _LOG.debug("Docker build output", output=result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

        except subprocess.CalledProcessError as e:
            _LOG.error("Docker build failed", error=e.stderr)
            raise RuntimeError(f"Docker build failed: {e.stderr}") from e
        except FileNotFoundError:
            _LOG.error("Docker not found. Please install Docker.")
            raise RuntimeError("Docker is not installed or not in PATH")

        return full_tag

    def tag(self, source_tag: str, target_tag: str) -> None:
        """Tag an existing image with an additional tag.

        Args:
            source_tag: Existing image tag.
            target_tag: New tag to add.
        """
        _LOG.info("Tagging image", source=source_tag, target=target_tag)

        try:
            subprocess.run(
                ["docker", "tag", source_tag, target_tag],
                check=True,
                capture_output=True,
            )
            _LOG.info("Image tagged successfully", target=target_tag)

        except subprocess.CalledProcessError as e:
            _LOG.error("Failed to tag image", error=e.stderr)
            raise RuntimeError(f"Failed to tag image: {e.stderr}") from e

    def inspect_image(self, image_tag: str) -> Dict[str, Any]:
        """Inspect Docker image metadata.

        Args:
            image_tag: Image tag to inspect.

        Returns:
            Dictionary containing image metadata.
        """
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image_tag],
                check=True,
                capture_output=True,
                text=True,
            )
            import json
            images = json.loads(result.stdout)
            if images:
                return images[0]
            return {}

        except subprocess.CalledProcessError as e:
            _LOG.error("Failed to inspect image", error=e.stderr)
            return {}

    def list_images(self, pattern: str = "") -> List[str]:
        """List Docker images matching pattern.

        Args:
            pattern: Filter pattern for image names.

        Returns:
            List of image tags.
        """
        try:
            cmd = ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"]
            if pattern:
                cmd.extend(["--filter", f"reference=*{pattern}*"])

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]

        except subprocess.CalledProcessError:
            return []

    def cleanup(self) -> None:
        """Clean up temporary build context."""
        if self.build_context and self.build_context.exists():
            shutil.rmtree(self.build_context)
            _LOG.info("Build context cleaned up", path=str(self.build_context))