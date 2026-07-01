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
Publish Runner

This module provides the main entry point for the PiscesLx publishing tool.
It handles configuration loading, orchestrator initialization, and pipeline execution.

Key Features:
    - Unified runner for all publishing operations
    - Configuration from multiple sources (CLI, JSON, dict)
    - Progress tracking and reporting
    - Error handling and recovery
    - Results aggregation and export

Usage Examples:
    From CLI:
        $ python -m tools.publish.runner --model-size 7B --model-path ./checkpoints/7B

    From Python:
        >>> from tools.publish.runner import PiscesLxToolsPublish
        >>> publish_tool = PiscesLxToolsPublish({"model_size": "7B", "model_path": "./checkpoints/7B"})
        >>> results = publish_tool.run()

    Programmatic:
        >>> from tools.publish.config import PiscesLxPublishConfig
        >>> config = PiscesLxPublishConfig(model_size="7B")
        >>> runner = PiscesLxToolsPublish(config)
        >>> runner.run()
"""

import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Optional
from datetime import datetime

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from .config import PiscesLxPublishConfig, ExportAction
from .orchestrator import PiscesLxPublishOrchestrator

_LOG = None


def _get_logger():
    """Get or create logger for runner."""
    global _LOG
    if _LOG is None:
        _LOG = PiscesLxLogger(
            "PiscesLx.Publish",
            file_path=get_log_file("PiscesLx.Publish"),
            enable_file=True
        )
    return _LOG


class PiscesLxToolsPublish:
    """Main runner for PiscesLx publishing tool.

    This class serves as the unified entry point for all publishing operations.
    It handles configuration management, orchestrator coordination, and
    results reporting.

    Attributes:
        config: PiscesLxPublishConfig instance.
        orchestrator: PiscesLxPublishOrchestrator instance.
        log: Logger instance.

    Example:
        >>> # From dictionary
        >>> publish_tool = PiscesLxToolsPublish({
        ...     "model_size": "7B",
        ...     "model_path": "./checkpoints/7B"
        ... })
        >>> results = publish_tool.run()

        >>> # From config file
        >>> publish_tool = PiscesLxToolsPublish("./publish_config.json")
        >>> results = publish_tool.run()

        >>> # From config object
        >>> config = PiscesLxPublishConfig(model_size="7B")
        >>> publish_tool = PiscesLxToolsPublish(config)
        >>> results = publish_tool.run()

        >>> # From manage.py args namespace
        >>> from argparse import Namespace
        >>> args = Namespace(publish_model_size='7B', publish_action='full', ...)
        >>> publish_tool = PiscesLxToolsPublish(args)
        >>> results = publish_tool.run()
    """

    def __init__(self, config: Optional[Union[str, Dict[str, Any], PiscesLxPublishConfig]] = None):
        """Initialize the publishing tool.

        Args:
            config: Configuration source. Can be:
                - str: Path to JSON config file
                - dict: Configuration dictionary
                - PiscesLxPublishConfig: Configuration object
                - Namespace: argparse Namespace from manage.py
                - None: Create empty config

        Raises:
            ValueError: If configuration is invalid.
        """
        self.log = _get_logger()
        self.config = self._load_config(config)
        self.orchestrator = PiscesLxPublishOrchestrator(self.config)
        self._args = None

        self.log.info(
            "PiscesLxToolsPublish initialized",
            model_size=self.config.model_size,
            action=self.config.action
        )

    def _load_config(
        self,
        config: Optional[Union[str, Dict[str, Any], PiscesLxPublishConfig]]
    ) -> PiscesLxPublishConfig:
        """Load and validate configuration.

        Args:
            config: Configuration source.

        Returns:
            PiscesLxPublishConfig instance.

        Raises:
            ValueError: If configuration is invalid.
        """
        if config is None:
            return PiscesLxPublishConfig()

        if isinstance(config, PiscesLxPublishConfig):
            return config

        if hasattr(config, 'publish_action'):
            self._args = config
            return self._build_config_from_args(config)

        if isinstance(config, str):
            config_path = Path(config)
            if config_path.exists() and config_path.suffix == '.json':
                self.log.info("Loading config from file", path=str(config_path))
                return PiscesLxPublishConfig.load_from_json(str(config_path))
            elif config_path.exists() and config_path.is_dir():
                self.log.info("Loading config from directory", path=str(config_path))
                config_file = config_path / "publish_config.json"
                if config_file.exists():
                    return PiscesLxPublishConfig.load_from_json(str(config_file))
                model_size = config_path.name
                return PiscesLxPublishConfig(model_size=model_size, model_path=str(config_path))

        if isinstance(config, dict):
            self.log.info("Loading config from dictionary")
            return PiscesLxPublishConfig.from_dict(config)

        raise ValueError(f"Invalid configuration type: {type(config)}")

    def _build_config_from_args(self, args) -> PiscesLxPublishConfig:
        """Build configuration from manage.py args namespace.

        Args:
            args: argparse Namespace from manage.py.

        Returns:
            PiscesLxPublishConfig instance.
        """
        config_dict = {}

        action_map = {
            'full': 'all',
            'export': 'export_only',
            'build': 'build_only',
            'push': 'publish_only',
            'validate': 'validate_only',
            'info': 'info_only',
            'list': 'list_only',
        }
        config_dict['action'] = action_map.get(getattr(args, 'publish_action', 'full'), 'all')

        if getattr(args, 'publish_model_size', None):
            config_dict['model_size'] = args.publish_model_size
        elif getattr(args, 'model_size', None):
            config_dict['model_size'] = args.model_size

        if getattr(args, 'publish_model_path', None):
            config_dict['model_path'] = args.publish_model_path
        elif getattr(args, 'ckpt', None):
            config_dict['model_path'] = args.ckpt

        if getattr(args, 'publish_output_dir', None):
            config_dict['output_dir'] = args.publish_output_dir

        if getattr(args, 'publish_template', None):
            config_dict['docker'] = config_dict.get('docker', {})
            config_dict['docker']['template'] = args.publish_template

        if getattr(args, 'publish_base_image', None):
            config_dict['docker'] = config_dict.get('docker', {})
            config_dict['docker']['base_image'] = args.publish_base_image

        if getattr(args, 'publish_port', None):
            config_dict['docker'] = config_dict.get('docker', {})
            config_dict['docker']['port'] = args.publish_port

        health_check = getattr(args, 'publish_health_check', True)
        if getattr(args, 'publish_no_health_check', False):
            health_check = False
        config_dict['docker'] = config_dict.get('docker', {})
        config_dict['docker']['health_check'] = health_check

        if getattr(args, 'publish_env_vars', None):
            config_dict['docker'] = config_dict.get('docker', {})
            env_vars = {}
            for pair in args.publish_env_vars.split(','):
                if '=' in pair:
                    key, val = pair.split('=', 1)
                    env_vars[key.strip()] = val.strip()
            config_dict['docker']['env_vars'] = env_vars

        if getattr(args, 'publish_export_format', None):
            config_dict['export'] = config_dict.get('export', {})
            config_dict['export']['format'] = args.publish_export_format

        if getattr(args, 'publish_export_quantize', False):
            config_dict['export'] = config_dict.get('export', {})
            config_dict['export']['quantize'] = True
            config_dict['export']['quant_bits'] = getattr(args, 'publish_quant_bits', 4)

        checksum = getattr(args, 'publish_generate_checksum', True)
        if getattr(args, 'publish_no_checksum', False):
            checksum = False
        config_dict['checksum'] = config_dict.get('checksum', {})
        config_dict['checksum']['enabled'] = checksum
        if getattr(args, 'publish_checksum_algorithms', None):
            config_dict['checksum']['algorithms'] = args.publish_checksum_algorithms.split(',')

        if getattr(args, 'publish_registry', None):
            config_dict['registry'] = config_dict.get('registry', {})
            config_dict['registry']['registry_url'] = args.publish_registry

        if getattr(args, 'publish_registry_namespace', None):
            config_dict['registry'] = config_dict.get('registry', {})
            config_dict['registry']['namespace'] = args.publish_registry_namespace

        if getattr(args, 'publish_registry_repo', None):
            config_dict['registry'] = config_dict.get('registry', {})
            config_dict['registry']['repository'] = args.publish_registry_repo

        if getattr(args, 'publish_registry_tag', None):
            config_dict['registry'] = config_dict.get('registry', {})
            config_dict['registry']['image_tag'] = args.publish_registry_tag

        if getattr(args, 'publish_username', None):
            config_dict['registry'] = config_dict.get('registry', {})
            config_dict['registry']['username'] = args.publish_username

        if getattr(args, 'publish_password', None):
            config_dict['registry'] = config_dict.get('registry', {})
            config_dict['registry']['password'] = args.publish_password

        if getattr(args, 'publish_retry_count', None):
            config_dict['registry'] = config_dict.get('registry', {})
            config_dict['registry']['retry_count'] = args.publish_retry_count

        if getattr(args, 'publish_retry_delay', None):
            config_dict['registry'] = config_dict.get('registry', {})
            config_dict['registry']['retry_delay'] = args.publish_retry_delay

        if getattr(args, 'publish_metadata_name', None):
            config_dict['metadata'] = config_dict.get('metadata', {})
            config_dict['metadata']['name'] = args.publish_metadata_name

        if getattr(args, 'publish_metadata_version', None):
            config_dict['metadata'] = config_dict.get('metadata', {})
            config_dict['metadata']['version'] = args.publish_metadata_version

        if getattr(args, 'publish_metadata_description', None):
            config_dict['metadata'] = config_dict.get('metadata', {})
            config_dict['metadata']['description'] = args.publish_metadata_description

        if getattr(args, 'publish_metadata_author', None):
            config_dict['metadata'] = config_dict.get('metadata', {})
            config_dict['metadata']['author'] = args.publish_metadata_author

        if getattr(args, 'publish_metadata_license', None):
            config_dict['metadata'] = config_dict.get('metadata', {})
            config_dict['metadata']['license'] = args.publish_metadata_license

        if getattr(args, 'publish_config', None):
            config_path = Path(args.publish_config)
            if config_path.exists():
                if config_path.suffix == '.json':
                    return PiscesLxPublishConfig.load_from_json(str(config_path))
                elif config_path.suffix in ['.yaml', '.yml']:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        yaml_data = yaml.safe_load(f)
                    return PiscesLxPublishConfig.from_dict(yaml_data)

        self.log.info("Building config from manage.py args", extra=config_dict)
        return PiscesLxPublishConfig.from_dict(config_dict)

    def run(self) -> Dict[str, Any]:
        """Execute the publishing pipeline.

        Runs the complete publishing pipeline based on configuration:
        - Export: Always runs
        - Build: Runs if action is BUILD_ONLY or ALL
        - Publish: Runs if action is PUBLISH_ONLY or ALL

        Returns:
            Dictionary containing pipeline results.

        Raises:
            RuntimeError: If pipeline execution fails.
        """
        self.log.info(
            "Starting PiscesLx publish",
            model_size=self.config.model_size,
            model_path=self.config.model_path,
            action=self.config.action
        )

        validation_errors = self.orchestrator.validate()
        if validation_errors:
            error_msg = f"Configuration validation failed: {', '.join(validation_errors)}"
            self.log.error(error_msg)
            raise ValueError(error_msg)

        try:
            results = self.orchestrator.run()
            self.log.info("PiscesLx publish completed successfully")
            return results

        except Exception as e:
            self.log.error("PiscesLx publish failed", error=str(e))
            raise

    def run_export(self) -> Dict[str, Any]:
        """Execute only the export stage.

        Returns:
            Export stage results.
        """
        self.log.info("Running export only")
        return self.orchestrator.run_export()

    def run_build(self) -> Dict[str, Any]:
        """Execute only the docker build stage.

        Returns:
            Build stage results.
        """
        self.log.info("Running docker build only")
        return self.orchestrator.run_build()

    def run_publish(self) -> Dict[str, Any]:
        """Execute only the registry publish stage.

        Returns:
            Publish stage results.
        """
        self.log.info("Running registry publish only")
        return self.orchestrator.run_publish()

    def run_validate(self) -> Dict[str, Any]:
        """Execute only the validation stage.

        Returns:
            Validation stage results.
        """
        self.log.info("Running validation only")
        return self.orchestrator.run_validate()

    def run_info(self) -> Dict[str, Any]:
        """Execute only the info display.

        Returns:
            Info stage results.
        """
        self.log.info("Displaying info only")
        return self.orchestrator.run_info()

    def run_list(self) -> Dict[str, Any]:
        """Execute only the list operation.

        Returns:
            List stage results.
        """
        self.log.info("Listing available options")
        return self.orchestrator.run_list()

    def get_results(self) -> Dict[str, Any]:
        """Get pipeline execution results.

        Returns:
            Results dictionary from the last run.
        """
        return self.orchestrator.results

    def save_results(self, path: Optional[str] = None) -> str:
        """Save pipeline results to file.

        Args:
            path: Output path. If None, uses default path.

        Returns:
            Path to saved results file.
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"./publish_results_{timestamp}.json"

        self.orchestrator.save_results(path)
        return path

    def print_summary(self) -> None:
        """Print human-readable summary of the last run."""
        self.orchestrator.print_summary()


class PiscesLxPublishRunner:
    """Compatibility alias for PiscesLxToolsPublish.

    This class provides backward compatibility with any code that
    might reference the old runner name.
    """

    def __new__(cls, config: Union[str, Dict[str, Any], PiscesLxPublishConfig]):
        """Create PiscesLxToolsPublish instance.

        Args:
            config: Configuration source.

        Returns:
            PiscesLxToolsPublish instance.
        """
        return PiscesLxToolsPublish(config)


def main():
    """Main entry point for the publish tool.

    Can be run directly with:
        python -m tools.publish.runner [options]

    Or via CLI:
        python -m tools.publish.cli [options]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="PiscesLx Model Publishing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                # Full pipeline (export + build + publish)
                python -m tools.publish.runner --model-size 7B --model-path ./checkpoints/7B

                # Export only
                python -m tools.publish.runner --action export_only --model-size 7B --model-path ./checkpoints/7B

                # Build only
                python -m tools.publish.runner --action build_only --model-size 7B

                # From config file
                python -m tools.publish.runner --config ./publish_config.json

                # Custom output
                python -m tools.publish.runner --model-size 7B --output-dir ./my_publish
        """
    )

    parser.add_argument(
        "--model-size", "-s",
        default="7B",
        help="Model size (e.g., 7B, 70B, 671B). Default: 7B"
    )
    parser.add_argument(
        "--model-path", "-p",
        default="",
        help="Path to model checkpoint or directory"
    )
    parser.add_argument(
        "--model-name", "-n",
        default="PiscesLx",
        help="Model name. Default: PiscesLx"
    )
    parser.add_argument(
        "--action", "-a",
        choices=["all", "export_only", "build_only", "publish_only"],
        default="all",
        help="Publishing action. Default: all"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./publish",
        help="Output directory. Default: ./publish"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--image-name",
        help="Docker image name"
    )
    parser.add_argument(
        "--image-tag",
        default="latest",
        help="Docker image tag. Default: latest"
    )
    parser.add_argument(
        "--registry",
        default="docker.io",
        help="Container registry. Default: docker.io"
    )
    parser.add_argument(
        "--repository",
        help="Repository name (e.g., username/piscesl1-7b)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--save-results",
        help="Path to save results JSON"
    )

    args = parser.parse_args()

    if args.config:
        publish_tool = PiscesLxToolsPublish(args.config)
    else:
        config_dict = {
            "model_size": args.model_size,
            "model_name": args.model_name,
            "model_path": args.model_path,
            "action": args.action,
            "output_dir": args.output_dir,
        }

        if args.image_name:
            config_dict["docker"] = {
                "image_name": args.image_name,
                "image_tag": args.image_tag,
            }

        if args.registry:
            config_dict["registry"] = {
                "registry_url": args.registry,
                "repository": args.repository or args.image_name or f"piscesl1/piscesl1-{args.model_size.lower()}",
                "image_tag": args.image_tag,
            }

        publish_tool = PiscesLxToolsPublish(config_dict)

    try:
        results = publish_tool.run()

        publish_tool.print_summary()

        if args.save_results:
            output_path = publish_tool.save_results(args.save_results)
            print(f"\nResults saved to: {output_path}")

        sys.exit(0)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()