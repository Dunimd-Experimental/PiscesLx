#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import json
from typing import Dict, Any, Union, List

from dataclasses import asdict

from utils.log.core import PiscesLxCoreLog
from utils.config.loader import load_config_from_file
from utils.device.manager import PiscesLxCoreDeviceManager

try:
    from evalscope import run_task
    from evalscope.summarizer import Summarizer
except ImportError:
    # Fallback for when evalscope is not available
    run_task = None
    Summarizer = None

from .config import PiscesLxToolsBenchmarkConfig
from .builders import PiscesLxToolsTaskConfigBuilder
from .result import PiscesLxToolsResultManager, PiscesLxToolsComparisonManager

_logger = PiscesLxCoreLog("pisceslx.tools.benchmark")


class PiscesLxToolsBenchmark:
    """PiscesL1 Benchmark runner with EvalScope integration"""

    def __init__(self, config: Union[str, Dict, PiscesLxToolsBenchmarkConfig]):
        self.config = self._load_config(config)
        self.logger = _logger.bind(benchmark_model=self.config.model_name)

        # Setup device
        self.device_manager = PiscesLxCoreDeviceManager()
        if self.config.device == "auto":
            self.config.device = self.device_manager.get_optimal_device()

        # Initialize managers
        self.result_manager = PiscesLxToolsResultManager(self.config.output_dir)
        self.comparison_manager = PiscesLxToolsComparisonManager()

        self.logger.info(
            "Initialized PiscesL1Benchmark",
            event="benchmark.initialized",
            model_path=self.config.model_path,
            device=self.config.device,
            datasets=self.config.datasets,
            metrics=self.config.metrics,
        )

    def _load_config(self, config: Union[str, Dict, PiscesLxToolsBenchmarkConfig]) -> PiscesLxToolsBenchmarkConfig:
        if isinstance(config, str):
            config_data = load_config_from_file(config)
            return PiscesLxToolsBenchmarkConfig(**config_data)
        elif isinstance(config, dict):
            return PiscesLxToolsBenchmarkConfig(**config)
        elif isinstance(config, PiscesLxToolsBenchmarkConfig):
            return config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

    def run_benchmark(self) -> Dict[str, Any]:
        """Run benchmark using EvalScope"""
        self.logger.info(
            "Starting benchmark evaluation",
            event="benchmark.start",
            model_path=self.config.model_path,
            datasets=self.config.datasets,
            metrics=self.config.metrics,
        )
        try:
            task_config = PiscesLxToolsTaskConfigBuilder.build(self.config)
            if run_task is None:
                raise ImportError("evalscope is not available")
            results = run_task(task_config)
            if Summarizer is None:
                raise ImportError("evalscope.summarizer is not available")
            summarizer = Summarizer()
            summary = summarizer.summarize(results)
            self.result_manager.save_results(results, summary, self.config)
            self.logger.success(
                "Benchmark completed successfully",
                event="benchmark.completed",
                output_dir=self.config.output_dir,
                results_count=len(results),
            )
            return {"results": results, "summary": summary, "config": asdict(self.config)}
        except Exception as e:
            self.logger.error(
                "Benchmark failed",
                event="benchmark.failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def compare_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.logger.info(
            "Starting model comparison",
            event="benchmark.comparison.start",
            model_count=len(model_configs),
        )
        comparison_results: Dict[str, Any] = {}
        original_config = self.config
        try:
            for i, model_config in enumerate(model_configs):
                model_name = model_config.get("name", f"model_{i+1}")
                self.logger.info(
                    f"Evaluating model {i+1}/{len(model_configs)}",
                    event="benchmark.model.evaluation.start",
                    model_name=model_name,
                )
                merged_config = {**asdict(original_config), **model_config}
                self.config = PiscesLxToolsBenchmarkConfig(**merged_config)
                result = self.run_benchmark()
                comparison_results[model_name] = result
                self.logger.success(
                    f"Model {model_name} evaluation completed",
                    event="benchmark.model.evaluation.completed",
                    model_name=model_name,
                )
            comparison_summary = self.comparison_manager.generate_comparison_report(comparison_results)
            self.logger.success(
                "Model comparison completed",
                event="benchmark.comparison.completed",
                models=list(comparison_results.keys()),
            )
            return {"comparison_results": comparison_results, "comparison_summary": comparison_summary}
        finally:
            self.config = original_config


class PiscesLxToolsBenchmarkConfig:
    """PiscesLxTools Benchmark Configuration Factory"""
    
    @staticmethod
    def create(config: Union[str, Dict]) -> PiscesLxToolsBenchmarkConfig:
        """Create benchmark configuration from file or dict"""
        return PiscesLxToolsBenchmarkConfig.create(config)


class PiscesLxToolsBenchmarkRunner:
    """PiscesLxTools Single Benchmark Runner"""
    
    @staticmethod
    def run(config: Union[str, Dict, PiscesLxToolsBenchmarkConfig]) -> Dict[str, Any]:
        """Run benchmark with configuration"""
        runner = PiscesLxToolsBenchmark(config)
        return runner.run_benchmark()


class PiscesLxToolsBenchmarkComparer:
    """PiscesLxTools Multiple Models Benchmark Comparer"""
    
    @staticmethod
    def compare(results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare benchmark results"""
        manager = PiscesLxToolsComparisonManager()
        return manager.generate_comparison_report(results)
