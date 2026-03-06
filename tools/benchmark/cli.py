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

import sys
import json
import argparse
from pathlib import Path

from utils.dc import PiscesLxLogger

from .config import MODALITY_DATASETS
from .runner import PiscesLxToolsBenchmarkRunner, PiscesLxToolsBenchmarkComparer

from utils.paths import get_log_file

_LOG = PiscesLxLogger("PiscesLx.Tools.Benchmark", file_path=get_log_file("PiscesLx.Tools.Benchmark"), enable_file=True)


def main():
    parser = argparse.ArgumentParser(
        description="PiscesLx Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                # Single model benchmark
                python -m tools.benchmark.cli --model-path /path/to/model --datasets mmlu ceval

                # Compare multiple models
                python -m tools.benchmark.cli --compare /path/to/model1 /path/to/model2 --datasets mmlu

                # Custom configuration
                python -m tools.benchmark.cli --model-path /path/to/model --batch-size 16 --max-length 4096
        """,
    )

    # Model configuration
    parser.add_argument("--model-path", help="Path to model")
    parser.add_argument("--model-name", help="Model name (defaults to directory name)")

    # Evaluation configuration
    parser.add_argument("--datasets", nargs="+", default=["mmlu", "ceval"], help="Datasets to evaluate")
    parser.add_argument("--metrics", nargs="+", default=["accuracy"], help="Metrics to compute")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")

    # System configuration
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--use-cache", action="store_true", default=True, help="Use caching")
    parser.add_argument("--save-predictions", action="store_true", default=True, help="Save predictions")

    # Debug and logging
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable caching")
    parser.add_argument("--no-save-predictions", dest="save_predictions", action="store_false", help="Don't save predictions")

    # Service evaluation & modality options
    parser.add_argument("--modality", choices=["text", "image", "audio", "video", "doc"], help="Modality preset for datasets")
    parser.add_argument("--eval-type", choices=["LOCAL", "SERVICE"], default="LOCAL", help="Evaluation type")
    parser.add_argument("--api-url", help="Model service URL for SERVICE eval_type")
    parser.add_argument("--generation-config", help="JSON string for generation_config passed to EvalScope")
    parser.add_argument("--eval-batch-size", type=int, help="Concurrent batch size for EvalScope service evaluation")
    parser.add_argument("--timeout", type=int, help="Timeout in milliseconds for EvalScope evaluation")

    # Model comparison
    parser.add_argument("--compare", nargs="+", help="Compare multiple models")

    args = parser.parse_args()

    if args.debug:
        __LOG.set_level("DEBUG")

    parsed_generation_config = None
    if args.generation_config:
        try:
            parsed_generation_config = json.loads(args.generation_config)
        except Exception as e:
            __LOG.error("Invalid --generation-config JSON", event="benchmark.arg_error", error=str(e))
            sys.exit(2)

    try:
        if args.compare:
            model_configs = []
            for model_path in args.compare:
                model_name = args.model_name or Path(model_path).name
                model_configs.append({
                    "model_path": model_path,
                    "model_name": model_name,
                    "datasets": (MODALITY_DATASETS.get(args.modality) if args.modality else args.datasets),
                    "metrics": args.metrics,
                    "output_dir": args.output_dir,
                    "device": args.device,
                    "debug": args.debug,
                    "batch_size": args.batch_size,
                    "max_length": args.max_length,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "use_cache": args.use_cache,
                    "save_predictions": args.save_predictions,
                    "modality": (args.modality or "text"),
                    "eval_type": args.eval_type,
                    "api_url": args.api_url,
                    "generation_config": parsed_generation_config,
                    "eval_batch_size": args.eval_batch_size,
                    "timeout": args.timeout,
                })
            results = PiscesLxToolsBenchmarkComparer.compare(model_configs)
            print(json.dumps(results["comparison_summary"], indent=2))
        else:
            if not args.model_path:
                parser.error("--model-path is required when not using --compare")
            results = PiscesLxToolsBenchmarkRunner.run(
                model_path=args.model_path,
                model_name=args.model_name,
                datasets=(MODALITY_DATASETS.get(args.modality) if args.modality else args.datasets),
                metrics=args.metrics,
                batch_size=args.batch_size,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                output_dir=args.output_dir,
                device=args.device,
                debug=args.debug,
                use_cache=args.use_cache,
                save_predictions=args.save_predictions,
                modality=(args.modality or "text"),
                eval_type=args.eval_type,
                api_url=args.api_url,
                generation_config=parsed_generation_config,
                eval_batch_size=args.eval_batch_size,
                timeout=args.timeout,
            )
            print(json.dumps(results["summary"], indent=2))
    except Exception as e:
        __LOG.error(
            "Benchmark execution failed",
            event="benchmark.execution.failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        sys.exit(1)
