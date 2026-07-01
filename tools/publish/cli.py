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
Publish CLI

Command-line interface for the PiscesLx publishing tool.
Provides user-friendly commands for model export, Docker image building,
and registry publishing.

Commands:
    publish     Execute complete publishing pipeline
    export      Export model only
    build       Build Docker image only
    push        Push image to registry only
    validate    Validate configuration
    info        Show image information

Usage:
    python -m tools.publish.cli publish [options]
    python -m tools.publish.cli export [options]
    python -m tools.publish.cli build [options]
    python -m tools.publish.cli push [options]
    python -m tools.publish.cli validate [options]
    python -m tools.publish.cli info --image <image_tag>

Examples:
    # Full pipeline
    $ python -m tools.publish.cli publish --model-size 7B --model-path ./checkpoints/7B

    # Export only
    $ python -m tools.publish.cli export --model-path ./checkpoints/7B --output ./publish

    # Build only
    $ python -m tools.publish.cli build --model-size 7B --image-name myrepo/piscesl1-7b

    # Push to registry
    $ python -m tools.publish.cli push --image myrepo/piscesl1-7b:v1.0.0 --registry docker.io

    # Validate configuration
    $ python -m tools.publish.cli validate --model-size 7B --model-path ./checkpoints/7B

    # Show image info
    $ python -m tools.publish.cli info --image piscesl1/piscesl1-7b:v1.0.0
"""

import sys
import argparse
from typing import Optional, List

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("PiscesLx.Publish.CLI", file_path=get_log_file("PiscesLx.Publish.CLI"), enable_file=True)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tools.publish.cli",
        description="PiscesLx Model Publishing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            For more information, see:
                https://github.com/piscesl1/piscesl1#publishing

            Examples:
                # Full pipeline
                python -m tools.publish.cli publish --model-size 7B

                # Export only
                python -m tools.publish.cli export --model-path ./checkpoints/7B

                # Build only
                python -m tools.publish.cli build --model-size 7B

                # Push to registry
                python -m tools.publish.cli push --image myrepo/piscesl1:v1.0.0
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--model-size", "-s",
        default="7B",
        help="Model size (7B, 70B, 671B, etc.). Default: 7B"
    )
    common_parser.add_argument(
        "--model-name", "-n",
        default="PiscesLx",
        help="Model name. Default: PiscesLx"
    )
    common_parser.add_argument(
        "--model-path", "-p",
        default="",
        help="Path to model checkpoint or directory"
    )
    common_parser.add_argument(
        "--output-dir", "-o",
        default="./publish",
        help="Output directory. Default: ./publish"
    )
    common_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    publish_parser = subparsers.add_parser(
        "publish",
        parents=[common_parser],
        help="Execute complete publishing pipeline (export + build + push)"
    )
    publish_parser.add_argument(
        "--image-name",
        help="Docker image name (e.g., docker.io/username/piscesl1)"
    )
    publish_parser.add_argument(
        "--image-tag",
        default="latest",
        help="Docker image tag. Default: latest"
    )
    publish_parser.add_argument(
        "--registry",
        default="docker.io",
        help="Container registry. Default: docker.io"
    )
    publish_parser.add_argument(
        "--repository",
        help="Repository name (overrides image-name)"
    )
    publish_parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to registry"
    )
    publish_parser.add_argument(
        "--config",
        help="Path to configuration JSON file"
    )
    publish_parser.add_argument(
        "--save-results",
        help="Path to save results JSON"
    )

    export_parser = subparsers.add_parser(
        "export",
        parents=[common_parser],
        help="Export model to publishable format"
    )
    export_parser.add_argument(
        "--format",
        choices=["safetensors", "pytorch"],
        default="safetensors",
        help="Export format. Default: safetensors"
    )
    export_parser.add_argument(
        "--quantization",
        choices=["int8", "int4", "fp8"],
        help="Apply quantization during export"
    )
    export_parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip tokenizer export"
    )

    build_parser = subparsers.add_parser(
        "build",
        parents=[common_parser],
        help="Build Docker image"
    )
    build_parser.add_argument(
        "--image-name",
        required=True,
        help="Docker image name (required)"
    )
    build_parser.add_argument(
        "--image-tag",
        default="latest",
        help="Docker image tag. Default: latest"
    )
    build_parser.add_argument(
        "--base-image",
        default="nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04",
        help="Base Docker image. Default: nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04"
    )
    build_parser.add_argument(
        "--template",
        choices=["default", "minimal", "gpu"],
        default="default",
        help="Dockerfile template. Default: default"
    )
    build_parser.add_argument(
        "--platform",
        default="linux/amd64",
        help="Target platform. Default: linux/amd64"
    )
    build_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Docker build cache"
    )

    push_parser = subparsers.add_parser(
        "push",
        help="Push image to registry"
    )
    push_parser.add_argument(
        "--image",
        required=True,
        help="Image tag to push (e.g., myrepo/piscesl1:v1.0.0)"
    )
    push_parser.add_argument(
        "--registry",
        default="docker.io",
        help="Registry URL. Default: docker.io"
    )
    push_parser.add_argument(
        "--repository",
        help="Repository name (overrides image path)"
    )
    push_parser.add_argument(
        "--tag",
        default="latest",
        help="Tag for the pushed image. Default: latest"
    )
    push_parser.add_argument(
        "--make-latest",
        action="store_true",
        help="Also tag as 'latest'"
    )
    push_parser.add_argument(
        "--username",
        help="Registry username"
    )
    push_parser.add_argument(
        "--password",
        help="Registry password"
    )
    push_parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Number of retries. Default: 3"
    )

    validate_parser = subparsers.add_parser(
        "validate",
        parents=[common_parser],
        help="Validate configuration"
    )

    info_parser = subparsers.add_parser(
        "info",
        help="Show image information"
    )
    info_parser.add_argument(
        "--image",
        required=True,
        help="Image tag to inspect"
    )

    list_parser = subparsers.add_parser(
        "list",
        help="List published images"
    )
    list_parser.add_argument(
        "--pattern",
        default="",
        help="Filter pattern for images"
    )
    list_parser.add_argument(
        "--registry",
        default="docker.io",
        help="Registry to list from"
    )

    return parser


def cmd_publish(args: argparse.Namespace) -> int:
    """Execute publish command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    from tools.publish.runner import PiscesLxToolsPublish
    from tools.publish.config import PiscesLxPublishConfig, ExportAction

    _LOG.info("Executing publish command", model_size=args.model_size)

    config = PiscesLxPublishConfig(
        model_size=args.model_size,
        model_name=args.model_name,
        model_path=args.model_path or f"./checkpoints/{args.model_size}",
        output_dir=args.output_dir,
        action=ExportAction.ALL.value if not args.no_push else ExportAction.BUILD_ONLY.value,
    )

    if args.config:
        config = PiscesLxPublishConfig.load_from_json(args.config)
    else:
        if args.image_name:
            config.docker.image_name = args.image_name
        if args.image_tag:
            config.docker.image_tag = args.image_tag
        if args.registry:
            config.registry.registry_url = args.registry
        if args.repository:
            config.registry.repository = args.repository

    try:
        publish_tool = PiscesLxToolsPublish(config)
        results = publish_tool.run()
        publish_tool.print_summary()

        if args.save_results:
            output_path = publish_tool.save_results(args.save_results)
            print(f"\nResults saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_export(args: argparse.Namespace) -> int:
    """Execute export command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    from tools.publish.runner import PiscesLxToolsPublish
    from tools.publish.config import PiscesLxPublishConfig, ExportAction

    _LOG.info("Executing export command", model_path=args.model_path)

    config = PiscesLxPublishConfig(
        model_size=args.model_size,
        model_name=args.model_name,
        model_path=args.model_path or f"./checkpoints/{args.model_size}",
        output_dir=args.output_dir,
        action=ExportAction.EXPORT_ONLY.value,
    )

    config.export.export_format = args.format
    if args.quantization:
        config.export.quantization = args.quantization
    config.export.include_tokenizer = not args.no_tokenizer

    try:
        publish_tool = PiscesLxToolsPublish(config)
        results = publish_tool.run_export()

        print("\nExport completed successfully!")
        print(f"Output directory: {args.output_dir}")
        print("\nExported files:")
        for key, path in results.get("output", {}).items():
            print(f"  {key}: {path}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_build(args: argparse.Namespace) -> int:
    """Execute build command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    from tools.publish.runner import PiscesLxToolsPublish
    from tools.publish.config import PiscesLxPublishConfig, ExportAction

    _LOG.info("Executing build command", image_name=args.image_name)

    config = PiscesLxPublishConfig(
        model_size=args.model_size,
        model_name=args.model_name,
        model_path=args.model_path or f"./checkpoints/{args.model_size}",
        output_dir=args.output_dir,
        action=ExportAction.BUILD_ONLY.value,
    )

    config.docker.image_name = args.image_name
    config.docker.image_tag = args.image_tag
    config.docker.base_image = args.base_image
    config.docker.dockerfile_template = args.template
    config.docker.platform = args.platform
    config.docker.use_cache = not args.no_cache

    try:
        publish_tool = PiscesLxToolsPublish(config)
        results = publish_tool.run_build()

        print("\nDocker build completed successfully!")
        print(f"Image tag: {args.image_name}:{args.image_tag}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_push(args: argparse.Namespace) -> int:
    """Execute push command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    from tools.publish.config import PiscesLxPublishConfig
    from tools.publish.registry import PiscesLxPublishRegistry

    _LOG.info("Executing push command", image=args.image)

    config = PiscesLxPublishConfig()
    config.registry.registry_url = args.registry
    config.registry.retry_count = args.retry
    config.registry.make_latest = args.make_latest

    if args.username:
        config.registry.credentials = {
            "username": args.username,
            "password": args.password or "",
        }

    publisher = PiscesLxPublishRegistry(config)

    try:
        if args.repository:
            image_name = f"{args.registry}/{args.repository}:{args.tag}"
            publisher._tag_image(args.image, image_name)
            url = publisher.publish(image_name)
        else:
            url = publisher.publish_with_retry(args.image, max_retries=args.retry)

        print(f"\nImage pushed successfully!")
        print(f"URL: {url}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute validate command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    from tools.publish.runner import PiscesLxToolsPublish
    from tools.publish.config import PiscesLxPublishConfig

    _LOG.info("Executing validate command")

    config = PiscesLxPublishConfig(
        model_size=args.model_size,
        model_name=args.model_name,
        model_path=args.model_path or f"./checkpoints/{args.model_size}",
        output_dir=args.output_dir,
    )

    publish_tool = PiscesLxToolsPublish(config)
    errors = publish_tool.orchestrator.validate()

    if errors:
        print("\nValidation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("\nValidation PASSED")
        print(f"Model size: {args.model_size}")
        print(f"Model path: {args.model_path or f'./checkpoints/{args.model_size}'}")
        print(f"Output dir: {args.output_dir}")
        return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Execute info command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    from tools.publish.config import PiscesLxPublishConfig
    from tools.publish.registry import PiscesLxPublishRegistry

    _LOG.info("Executing info command", image=args.image)

    config = PiscesLxPublishConfig()
    publisher = PiscesLxPublishRegistry(config)

    try:
        info = publisher.get_image_info(args.image)

        if not info:
            print(f"Could not retrieve info for image: {args.image}")
            return 1

        print(f"\nImage: {args.image}")
        print("-" * 50)

        if "config" in info:
            config_data = info["config"]
            print(f"Architecture: {config_data.get('Architecture', 'N/A')}")
            print(f"OS: {config_data.get('Os', 'N/A')}")
            print(f"Created: {config_data.get('Created', 'N/A')}")

        if "rootfs" in info:
            layers = info["rootfs"].get("type", "layers")
            print(f"Rootfs layers: {layers}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """Execute list command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    from tools.publish.config import PiscesLxPublishConfig
    from tools.publish.registry import PiscesLxPublishRegistry

    _LOG.info("Executing list command", pattern=args.pattern)

    config = PiscesLxPublishConfig()
    config.registry.registry_url = args.registry

    publisher = PiscesLxPublishRegistry(config)

    try:
        tags = publisher.list_tags(args.pattern)

        if tags:
            print(f"\nImages in {args.registry}:")
            for tag in tags:
                print(f"  {tag}")
        else:
            print(f"\nNo images found matching pattern: {args.pattern or '*'}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments. Uses sys.argv if None.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "publish": cmd_publish,
        "export": cmd_export,
        "build": cmd_build,
        "push": cmd_push,
        "validate": cmd_validate,
        "info": cmd_info,
        "list": cmd_list,
    }

    command_func = commands.get(args.command)
    if command_func:
        return command_func(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())