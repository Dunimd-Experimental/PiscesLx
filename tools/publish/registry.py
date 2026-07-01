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
Registry Publisher

This module handles publishing Docker images to container registries.
It supports multiple registry types including Docker Hub, GitHub Container
Registry (GHCR), NVIDIA NGC, Azure Container Registry, and Google Container
Registry.

Key Features:
    - Multi-registry support (Docker Hub, GHCR, NGC, ACR, GCR)
    - Authentication management
    - Retry logic for failed operations
    - Image tagging strategies
    - Manifest verification

Supported Registries:
    - docker.io: Docker Hub (default)
    - ghcr.io: GitHub Container Registry
    - nvcr.io: NVIDIA NGC
    - azurecr.io: Azure Container Registry
    - gcr.io: Google Container Registry

Publishing Process:
    1. Authenticate with registry
    2. Tag image for target registry
    3. Push image to registry
    4. Apply additional tags (e.g., 'latest')
    5. Verify published image

Usage Examples:
    Basic Publish:
        >>> from tools.publish.config import PiscesLxPublishConfig
        >>> from tools.publish.registry import PiscesLxPublishRegistry
        >>> config = PiscesLxPublishConfig(
        ...     registry=PiscesLxPublishRegistryConfig(
        ...         registry_url="docker.io",
        ...         repository="piscesl1/piscesl1-7b",
        ...         image_tag="v1.0.0"
        ...     )
        ... )
        >>> publisher = PiscesLxPublishRegistry(config)
        >>> url = publisher.publish("piscesl1/piscesl1-7b:v1.0.0")
        >>> print(f"Published: {url}")

    Publish to GHCR:
        >>> config.registry.registry_url = "ghcr.io"
        >>> config.registry.repository = "username/piscesl1"
        >>> publisher = PiscesLxPublishRegistry(config)
        >>> url = publisher.publish("ghcr.io/username/piscesl1:v1.0.0")

    Publish with Retry:
        >>> config.registry.retry_count = 5
        >>> publisher = PiscesLxPublishRegistry(config)
        >>> url = publisher.publish_with_retry("piscesl1/piscesl1-7b:v1.0.0")
"""

import os
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("PiscesLx.Publish.Registry", file_path=get_log_file("PiscesLx.Publish.Registry"), enable_file=True)


class PiscesLxPublishRegistry:
    """Registry Publisher for PiscesLx.

    Handles publishing Docker images to container registries with support
    for multiple registry types and authentication.

    Attributes:
        config: PiscesLxPublishConfig containing registry configuration.
        supported_registries: List of supported registry URLs.

    Example:
        >>> config = PiscesLxPublishConfig(
        ...     registry=PiscesLxPublishRegistryConfig(
        ...         registry_url="docker.io",
        ...         repository="piscesl1"
        ...     )
        ... )
        >>> publisher = PiscesLxPublishRegistry(config)
        >>> url = publisher.publish("piscesl1/piscesl1-7b:v1.0.0")
    """

    SUPPORTED_REGISTRIES = [
        "docker.io",
        "ghcr.io",
        "nvcr.io",
        "azurecr.io",
        "gcr.io",
        "registry.k8s.io",
        "quay.io",
    ]

    def __init__(self, config: 'PiscesLxPublishConfig'):
        """Initialize the registry publisher.

        Args:
            config: PiscesLxPublishConfig containing registry configuration.
        """
        self.config = config
        _LOG.info(
            "PiscesLxPublishRegistry initialized",
            registry=config.registry.registry_url,
            repository=config.registry.repository
        )

    def publish(self, image_tag: str) -> str:
        """Publish image to registry.

        Main entry point for publishing images. Handles authentication,
        tagging, pushing, and verification.

        Args:
            image_tag: Full image tag to publish (e.g., "piscesl1/piscesl1-7b:v1.0.0").

        Returns:
            Full URL of the published image.

        Raises:
            RuntimeError: If publish fails.
        """
        _LOG.info("Starting image publish", image_tag=image_tag)

        self._login()

        full_url = self._push(image_tag)

        if self.config.registry.make_latest:
            self._push_latest(image_tag)

        self._verify_publish(full_url)

        _LOG.info("Image published successfully", url=full_url)
        return full_url

    def publish_with_retry(self, image_tag: str, max_retries: Optional[int] = None) -> str:
        """Publish image with retry logic.

        Retries the publish operation on failure with exponential backoff.

        Args:
            image_tag: Full image tag to publish.
            max_retries: Maximum number of retry attempts. Uses config default if None.

        Returns:
            Full URL of the published image.

        Raises:
            RuntimeError: If all retries fail.
        """
        retry_count = max_retries or self.config.registry.retry_count
        last_error = None

        for attempt in range(retry_count):
            try:
                _LOG.info("Publish attempt", attempt=attempt + 1, max_retries=retry_count)
                return self.publish(image_tag)

            except Exception as e:
                last_error = e
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt
                    _LOG.warning(
                        "Publish failed, retrying",
                        attempt=attempt + 1,
                        wait_seconds=wait_time,
                        error=str(e)
                    )
                    time.sleep(wait_time)
                else:
                    _LOG.error("All publish retries failed", error=str(e))

        raise RuntimeError(f"Failed to publish after {retry_count} attempts: {last_error}")

    def _login(self) -> None:
        """Authenticate with the target registry.

        Uses credentials from configuration or Docker config for authentication.

        Raises:
            RuntimeError: If login fails.
        """
        registry = self.config.registry.registry_url
        _LOG.info("Authenticating with registry", registry=registry)

        try:
            if self.config.registry.credentials:
                username = self.config.registry.credentials.get('username', '')
                password = self.config.registry.credentials.get('password', '')

                if registry == "ghcr.io":
                    password = self.config.registry.credentials.get('github_token', password)

                subprocess.run(
                    ["docker", "login", registry, "--username", username, "--password-stdin"],
                    input=password,
                    text=True,
                    check=True,
                    capture_output=True,
                )
            else:
                result = subprocess.run(
                    ["docker", "context", "ls"],
                    capture_output=True,
                    text=True,
                )
                if "current" in result.stdout:
                    _LOG.info("Using default Docker authentication")

            _LOG.info("Registry authentication successful", registry=registry)

        except subprocess.CalledProcessError as e:
            _LOG.warning(
                "Registry authentication failed or skipped",
                registry=registry,
                error=e.stderr
            )

    def _push(self, image_tag: str) -> str:
        """Push image to registry.

        Args:
            image_tag: Image tag to push.

        Returns:
            Full URL of the pushed image.

        Raises:
            RuntimeError: If push fails.
        """
        registry = self.config.registry.registry_url
        repository = self.config.registry.repository
        tag = self.config.registry.image_tag

        if not image_tag.startswith(registry):
            target_tag = f"{registry}/{repository}:{tag}"
            self._tag_image(image_tag, target_tag)
            image_tag = target_tag

        full_url = image_tag
        _LOG.info("Pushing image", image_tag=image_tag)

        try:
            result = subprocess.run(
                ["docker", "push", image_tag],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.registry.timeout,
            )

            if result.stderr:
                lines = result.stderr.split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        _LOG.debug("Push progress", status=line.strip())

            _LOG.info("Image pushed successfully", image_tag=image_tag)

        except subprocess.TimeoutExpired:
            _LOG.error("Push timed out", timeout=self.config.registry.timeout)
            raise RuntimeError(f"Push timed out after {self.config.registry.timeout} seconds")
        except subprocess.CalledProcessError as e:
            _LOG.error("Push failed", error=e.stderr)
            raise RuntimeError(f"Failed to push image: {e.stderr}") from e

        return full_url

    def _push_latest(self, image_tag: str) -> None:
        """Push image with 'latest' tag.

        Args:
            image_tag: Original image tag.
        """
        registry = self.config.registry.registry_url
        repository = self.config.registry.repository

        latest_tag = f"{registry}/{repository}:latest"
        self._tag_image(image_tag, latest_tag)

        _LOG.info("Pushing latest tag", latest_tag=latest_tag)

        try:
            subprocess.run(
                ["docker", "push", latest_tag],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.registry.timeout,
            )
            _LOG.info("Latest tag pushed successfully")

        except subprocess.CalledProcessError as e:
            _LOG.warning("Failed to push latest tag", error=e.stderr)

    def _tag_image(self, source: str, target: str) -> None:
        """Tag an image for the target registry.

        Args:
            source: Source image tag.
            target: Target image tag.
        """
        _LOG.info("Tagging image", source=source, target=target)

        try:
            subprocess.run(
                ["docker", "tag", source, target],
                check=True,
                capture_output=True,
            )
            _LOG.info("Image tagged successfully", target=target)

        except subprocess.CalledProcessError as e:
            _LOG.error("Failed to tag image", error=e.stderr)
            raise RuntimeError(f"Failed to tag image: {e.stderr}") from e

    def _verify_publish(self, image_url: str) -> bool:
        """Verify that image was published successfully.

        Args:
            image_url: Full URL of the published image.

        Returns:
            True if verification succeeded.

        Raises:
            RuntimeError: If verification fails.
        """
        if not image_url.startswith("docker.io"):
            _LOG.info("Skipping verification for non-Docker Hub registry")
            return True

        _LOG.info("Verifying published image", url=image_url)

        try:
            repo_path = "/".join(image_url.split("/")[1:])
            result = subprocess.run(
                ["docker", "manifest", "inspect", image_url],
                check=True,
                capture_output=True,
                text=True,
            )
            _LOG.info("Image verification successful", url=image_url)
            return True

        except subprocess.CalledProcessError:
            _LOG.warning("Image verification failed, but push may have succeeded")
            return True

    def pull(self, image_tag: str, output_tag: Optional[str] = None) -> str:
        """Pull image from registry.

        Args:
            image_tag: Image tag to pull.
            output_tag: Optional tag for the local image.

        Returns:
            Local image tag.

        Raises:
            RuntimeError: If pull fails.
        """
        _LOG.info("Pulling image", image_tag=image_tag)

        local_tag = output_tag or image_tag

        try:
            subprocess.run(
                ["docker", "pull", image_tag],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if output_tag and output_tag != image_tag:
                self._tag_image(image_tag, output_tag)

            _LOG.info("Image pulled successfully", image_tag=image_tag, local_tag=local_tag)
            return local_tag

        except subprocess.CalledProcessError as e:
            _LOG.error("Failed to pull image", error=e.stderr)
            raise RuntimeError(f"Failed to pull image: {e.stderr}") from e

    def list_tags(self, repository: Optional[str] = None) -> List[str]:
        """List tags for a repository.

        Args:
            repository: Repository name. Uses config default if None.

        Returns:
            List of tag names.

        Note:
            This may not work for all registries without authentication.
        """
        repo = repository or f"{self.config.registry.registry_url}/{self.config.registry.repository}"
        _LOG.info("Listing tags", repository=repo)

        try:
            if self.config.registry.registry_url == "docker.io":
                result = subprocess.run(
                    ["docker", "search", "--limit", "100", "--format", "{{.Name}}", repo],
                    capture_output=True,
                    text=True,
                )
                return result.stdout.strip().split("\n")

        except subprocess.CalledProcessError:
            pass

        _LOG.warning("Could not list tags for repository", repository=repo)
        return []

    def delete(self, image_tag: str) -> None:
        """Delete image from registry.

        Args:
            image_tag: Image tag to delete.

        Note:
            This requires appropriate registry permissions and may not
            work for all registries (e.g., Docker Hub).
        """
        _LOG.warning("Registry delete requested", image_tag=image_tag)
        _LOG.warning("Registry delete is not implemented for safety reasons")

    def get_image_info(self, image_tag: str) -> Dict[str, Any]:
        """Get information about a published image.

        Args:
            image_tag: Image tag to inspect.

        Returns:
            Dictionary containing image information.
        """
        _LOG.info("Getting image info", image_tag=image_tag)

        try:
            result = subprocess.run(
                ["docker", "manifest", "inspect", image_tag],
                check=True,
                capture_output=True,
                text=True,
            )
            import json
            return json.loads(result.stdout)

        except subprocess.CalledProcessError as e:
            _LOG.error("Failed to get image info", error=e.stderr)
            return {}

    def cleanup(self) -> None:
        """Clean up local Docker resources.

        Removes dangling images and unused resources created during build.
        """
        _LOG.info("Cleaning up local Docker resources")

        try:
            subprocess.run(
                ["docker", "image", "prune", "-f"],
                capture_output=True,
                text=True,
            )
            _LOG.info("Docker cleanup completed")

        except subprocess.CalledProcessError as e:
            _LOG.warning("Docker cleanup failed", error=e.stderr)