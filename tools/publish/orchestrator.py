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
Publish Orchestrator

This module serves as the main orchestrator for the PiscesLx publishing pipeline.
It coordinates the export, docker build, and registry publish components to
execute a complete publishing workflow.

Key Features:
    - Pipeline orchestration with progress tracking
    - Step-by-step execution with validation
    - Error handling and recovery
    - Result aggregation and reporting
    - Integration with all publishing components

Pipeline Stages:
    1. Export: Model weights, config, tokenizer
    2. Build: Docker image with inference engine
    3. Publish: Push to registry with verification

Orchestration Flow:
    PiscesLxPublishOrchestrator
    ├── PiscesLxPublishExporter (export stage)
    ├── PiscesLxPublishDockerBuilder (build stage)
    └── PiscesLxPublishRegistry (publish stage)

Usage Examples:
    Full Pipeline:
        >>> from tools.publish.config import PiscesLxPublishConfig
        >>> from tools.publish.orchestrator import PiscesLxPublishOrchestrator
        >>> config = PiscesLxPublishConfig(
        ...     model_size="7B",
        ...     model_path="./checkpoints/7B"
        ... )
        >>> orchestrator = PiscesLxPublishOrchestrator(config)
        >>> results = orchestrator.run()
        >>> print(results)

    Export Only:
        >>> orchestrator = PiscesLxPublishOrchestrator(config)
        >>> results = orchestrator.run_export()

    Build Only:
        >>> orchestrator = PiscesLxPublishOrchestrator(config)
        >>> results = orchestrator.run_build()

    Publish Only:
        >>> orchestrator = PiscesLxPublishOrchestrator(config)
        >>> results = orchestrator.run_publish()
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from .config import PiscesLxPublishConfig, ExportAction
from .exporter import PiscesLxPublishExporter
from .docker_builder import PiscesLxPublishDockerBuilder
from .registry import PiscesLxPublishRegistry

_LOG = PiscesLxLogger("PiscesLx.Publish.Orchestrator", file_path=get_log_file("PiscesLx.Publish.Orchestrator"), enable_file=True)


class PiscesLxPublishOrchestrator:
    """Main orchestrator for PiscesLx publishing pipeline.

    Coordinates all stages of the publishing workflow including export,
    docker build, and registry publish.

    Attributes:
        config: PiscesLxPublishConfig containing all pipeline configuration.
        exporter: PiscesLxPublishExporter instance for model export.
        docker_builder: PiscesLxPublishDockerBuilder instance for image build.
        registry_publisher: PiscesLxPublishRegistry instance for publishing.
        results: Dictionary storing results from pipeline execution.

    Example:
        >>> config = PiscesLxPublishConfig(model_size="7B")
        >>> orchestrator = PiscesLxPublishOrchestrator(config)
        >>> results = orchestrator.run()
        >>> # results = {'export': {...}, 'docker': {...}, 'registry': {...}}
    """

    def __init__(self, config: PiscesLxPublishConfig):
        """Initialize the orchestrator.

        Args:
            config: PiscesLxPublishConfig containing all pipeline configuration.
        """
        self.config = config
        self.exporter = PiscesLxPublishExporter(config)
        self.docker_builder = PiscesLxPublishDockerBuilder(config)
        self.registry_publisher = PiscesLxPublishRegistry(config)
        self.results: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        _LOG.info(
            "PiscesLxPublishOrchestrator initialized",
            model_size=config.model_size,
            action=config.action
        )

    def run(self) -> Dict[str, Any]:
        """Execute the complete publishing pipeline.

        Runs all stages based on configuration:
        - Export: Always runs
        - Build: Runs if action includes BUILD or ALL
        - Publish: Runs if action includes PUBLISH or ALL

        Returns:
            Dictionary containing results from all executed stages.

        Raises:
            RuntimeError: If any pipeline stage fails.
        """
        self.start_time = datetime.now()
        _LOG.info("Starting publishing pipeline", model_size=self.config.model_size)

        self.results = {
            "start_time": self.start_time.isoformat(),
            "model_size": self.config.model_size,
            "model_name": self.config.model_name,
            "stages": {},
            "status": "running",
        }

        try:
            self._run_export_stage()
            self._run_build_stage()
            self._run_publish_stage()

            self.results["status"] = "success"
            self.end_time = datetime.now()
            self.results["end_time"] = self.end_time.isoformat()
            self.results["duration_seconds"] = (self.end_time - self.start_time).total_seconds()

            _LOG.info(
                "Publishing pipeline completed successfully",
                duration=self.results["duration_seconds"]
            )

        except Exception as e:
            self.results["status"] = "failed"
            self.results["error"] = str(e)
            self.end_time = datetime.now()
            self.results["end_time"] = self.end_time.isoformat()
            self.results["duration_seconds"] = (self.end_time - self.start_time).total_seconds()

            _LOG.error("Publishing pipeline failed", error=str(e))
            raise

        return self.results

    def _run_export_stage(self) -> None:
        """Execute the export stage.

        Exports model weights, configuration, tokenizer, and generation settings.

        Raises:
            RuntimeError: If export fails.
        """
        stage_name = "export"
        _LOG.info("Stage 1: Export", stage=stage_name)

        try:
            export_results = self.exporter.export()
            self.results["stages"][stage_name] = {
                "status": "success",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "output": export_results,
            }

            verify_success, errors = self.exporter.verify_export()
            if not verify_success:
                _LOG.warning("Export verification found issues", errors=errors)

            summary = self.exporter.get_export_summary()
            self.results["stages"][stage_name]["summary"] = summary

            _LOG.info(
                "Export stage completed",
                stage=stage_name,
                files=summary.get("file_count", 0),
                size_mb=summary.get("total_size_mb", 0)
            )

        except Exception as e:
            self.results["stages"][stage_name] = {
                "status": "failed",
                "error": str(e),
            }
            _LOG.error("Export stage failed", error=str(e))
            raise RuntimeError(f"Export failed: {e}") from e

    def _run_build_stage(self) -> None:
        """Execute the docker build stage.

        Builds Docker image containing model and inference engine.

        Raises:
            RuntimeError: If build fails.
        """
        stage_name = "docker_build"
        action = self.config.action

        if action not in [ExportAction.ALL.value, ExportAction.BUILD_ONLY.value]:
            _LOG.info("Skipping build stage", action=action)
            return

        _LOG.info("Stage 2: Docker Build", stage=stage_name)

        try:
            export_output = self.results.get("stages", {}).get("export", {}).get("output", {})
            model_dir = export_output.get("weights")

            start_time = datetime.now()
            image_tag = self.docker_builder.build(
                model_dir=model_dir,
                output_dir=self.config.export.output_dir,
            )
            end_time = datetime.now()

            image_info = self.docker_builder.inspect_image(image_tag)

            self.results["stages"][stage_name] = {
                "status": "success",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "image_tag": image_tag,
                "image_info": {
                    "id": image_info.get("Id", ""),
                    "size": image_info.get("Size", 0),
                },
            }

            _LOG.info("Docker build stage completed", stage=stage_name, image_tag=image_tag)

        except Exception as e:
            self.results["stages"][stage_name] = {
                "status": "failed",
                "error": str(e),
            }
            _LOG.error("Docker build stage failed", error=str(e))
            raise RuntimeError(f"Docker build failed: {e}") from e

    def _run_publish_stage(self) -> None:
        """Execute the registry publish stage.

        Publishes Docker image to the configured registry.

        Raises:
            RuntimeError: If publish fails.
        """
        stage_name = "registry_publish"
        action = self.config.action

        if action not in [ExportAction.ALL.value, ExportAction.PUBLISH_ONLY.value]:
            _LOG.info("Skipping publish stage", action=action)
            return

        if not self.config.registry.registry_url:
            _LOG.info("Skipping publish stage", reason="no registry configured")
            return

        _LOG.info("Stage 3: Registry Publish", stage=stage_name)

        try:
            build_output = self.results.get("stages", {}).get("docker_build", {})
            image_tag = build_output.get("image_tag")

            if not image_tag:
                image_tag = f"{self.config.docker.image_name}:{self.config.docker.image_tag}"

            start_time = datetime.now()
            image_url = self.registry_publisher.publish(image_tag)
            end_time = datetime.now()

            self.results["stages"][stage_name] = {
                "status": "success",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "image_tag": image_tag,
                "image_url": image_url,
            }

            _LOG.info("Registry publish stage completed", stage=stage_name, url=image_url)

        except Exception as e:
            self.results["stages"][stage_name] = {
                "status": "failed",
                "error": str(e),
            }
            _LOG.error("Registry publish stage failed", error=str(e))
            raise RuntimeError(f"Registry publish failed: {e}") from e

    def run_export(self) -> Dict[str, Any]:
        """Execute only the export stage.

        Returns:
            Export stage results.
        """
        _LOG.info("Running export stage only")
        self.start_time = datetime.now()

        try:
            self._run_export_stage()
            self.end_time = datetime.now()
            self.results["duration_seconds"] = (self.end_time - self.start_time).total_seconds()
            return self.results["stages"]["export"]

        except Exception as e:
            self.results["stages"]["export"] = {"status": "failed", "error": str(e)}
            raise

    def run_build(self) -> Dict[str, Any]:
        """Execute only the docker build stage.

        Returns:
            Build stage results.
        """
        _LOG.info("Running docker build stage only")

        if "stages" not in self.results or "export" not in self.results.get("stages", {}):
            _run_export_stage()

        try:
            self._run_build_stage()
            return self.results["stages"]["docker_build"]

        except Exception as e:
            self.results["stages"]["docker_build"] = {"status": "failed", "error": str(e)}
            raise

    def run_publish(self) -> Dict[str, Any]:
        """Execute only the registry publish stage.

        Returns:
            Publish stage results.
        """
        _LOG.info("Running registry publish stage only")

        if "stages" not in self.results or "docker_build" not in self.results.get("stages", {}):
            self.run_build()

        try:
            self._run_publish_stage()
            return self.results["stages"]["registry_publish"]

        except Exception as e:
            self.results["stages"]["registry_publish"] = {"status": "failed", "error": str(e)}
            raise

    def run_validate(self) -> Dict[str, Any]:
        """Execute only the validation stage.

        Returns:
            Validation results with errors list.
        """
        _LOG.info("Running validation only")
        errors = self.validate()

        return {
            "action": "validate",
            "valid": len(errors) == 0,
            "errors": errors,
            "config": {
                "model_size": self.config.model_size,
                "model_path": self.config.model_path,
                "output_dir": self.config.output_dir,
                "registry_url": self.config.registry.registry_url,
            }
        }

    def run_info(self) -> Dict[str, Any]:
        """Execute only the info display.

        Returns:
            Configuration information.
        """
        _LOG.info("Displaying configuration info")

        return {
            "action": "info",
            "model_size": self.config.model_size,
            "model_name": self.config.model_name,
            "model_path": self.config.model_path,
            "output_dir": self.config.output_dir,
            "export": {
                "checkpoint_path": self.config.export.checkpoint_path,
                "export_format": self.config.export.export_format,
                "output_dir": self.config.export.output_dir,
            },
            "docker": {
                "image_name": self.config.docker.image_name,
                "image_tag": self.config.docker.image_tag,
                "template": self.config.docker.template,
                "base_image": self.config.docker.base_image,
            },
            "registry": {
                "registry_url": self.config.registry.registry_url,
                "repository": self.config.registry.repository,
                "image_tag": self.config.registry.image_tag,
            },
            "metadata": {
                "name": self.config.metadata.name,
                "version": self.config.metadata.version,
                "description": self.config.metadata.description,
                "author": self.config.metadata.author,
                "license": self.config.metadata.license,
            }
        }

    def run_list(self) -> Dict[str, Any]:
        """Execute only the list operation.

        Returns:
            Available templates and registries.
        """
        _LOG.info("Listing available templates")

        return {
            "action": "list",
            "templates": {
                "default": "Standard image with all dependencies",
                "minimal": "Minimal image with just inference engine",
                "gpu": "GPU-optimized image with CUDA support",
            },
            "registries": {
                "docker.io": "Docker Hub",
                "ghcr.io": "GitHub Container Registry",
                "nvcr.io": "NVIDIA NGC",
                "azurecr.io": "Azure Container Registry",
                "gcr.io": "Google Container Registry",
            },
            "export_formats": {
                "safetensors": "SafeTensor format (recommended, secure)",
                "pytorch": "PyTorch format",
            },
            "model_sizes": ["0.5B", "1B", "7B", "14B", "72B", "671B", "1T"],
        }

    def validate(self) -> List[str]:
        """Validate pipeline configuration.

        Checks if all required configuration is present and valid.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []

        if not self.config.model_path and not self.config.export.checkpoint_path:
            errors.append("Model path not specified")

        if not self.config.model_size:
            errors.append("Model size not specified")

        export_path = Path(self.config.export.output_dir)
        if not export_path.parent.exists():
            errors.append(f"Output directory parent does not exist: {export_path.parent}")

        if self.config.registry.registry_url:
            if self.config.registry.registry_url not in self.registry_publisher.SUPPORTED_REGISTRIES:
                errors.append(f"Unsupported registry: {self.config.registry.registry_url}")

        if errors:
            _LOG.warning("Validation found issues", errors=errors)
        else:
            _LOG.info("Validation passed")

        return errors

    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution results.

        Returns:
            Dictionary containing summary information.
        """
        if not self.results:
            return {"status": "not_run"}

        summary = {
            "status": self.results.get("status", "unknown"),
            "model_size": self.results.get("model_size"),
            "model_name": self.results.get("model_name"),
            "duration_seconds": self.results.get("duration_seconds"),
            "stages": {},
        }

        for stage_name, stage_result in self.results.get("stages", {}).items():
            summary["stages"][stage_name] = {
                "status": stage_result.get("status"),
                "duration": None,
            }

            if "start_time" in stage_result and "end_time" in stage_result:
                try:
                    start = datetime.fromisoformat(stage_result["start_time"])
                    end = datetime.fromisoformat(stage_result["end_time"])
                    summary["stages"][stage_name]["duration"] = (end - start).total_seconds()
                except Exception:
                    pass

        return summary

    def save_results(self, path: str) -> None:
        """Save pipeline results to JSON file.

        Args:
            path: Path to save the results file.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        _LOG.info("Results saved", path=path)

    def print_summary(self) -> None:
        """Print a human-readable summary of the pipeline execution."""
        summary = self.get_results_summary()

        print("\n" + "=" * 60)
        print("PiscesLx Publishing Pipeline Summary")
        print("=" * 60)
        print(f"Model: {summary.get('model_name', 'N/A')} ({summary.get('model_size', 'N/A')})")
        print(f"Status: {summary.get('status', 'N/A').upper()}")
        print(f"Duration: {summary.get('duration_seconds', 0):.2f} seconds")
        print("-" * 60)
        print("Stages:")

        for stage_name, stage_info in summary.get("stages", {}).items():
            status = stage_info.get("status", "unknown").upper()
            duration = stage_info.get("duration")
            duration_str = f" ({duration:.2f}s)" if duration else ""
            print(f"  {stage_name}: {status}{duration_str}")

        print("=" * 60 + "\n")