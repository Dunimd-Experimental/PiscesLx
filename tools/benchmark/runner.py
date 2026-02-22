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

import json
from typing import Dict, Any, Union, List

from dataclasses import asdict

from utils.dc import PiscesLxLogger, PiscesLxConfiguration, PiscesLxSystemMonitor

from evalscope import run_task
from evalscope.summarizer import Summarizer

from .config import PiscesLxToolsBenchmarkConfig
from .builders import PiscesLxToolsTaskConfigBuilder
from .result import PiscesLxToolsResultManager, PiscesLxToolsComparisonManager

from utils.paths import get_log_file

_LOG = PiscesLxLogger("PiscesLx.Tools.Benchmark", file_path=get_log_file("PiscesLx.Tools.Benchmark"), enable_file=True)


class PiscesLxToolsBenchmark:
    """PiscesL1 Benchmark runner with EvalScope integration"""

    def __init__(self, config: Union[str, Dict, PiscesLxToolsBenchmarkConfig]):
        self.config = self._load_config(config)
        self.logger = PiscesLxLogger("PiscesLx.Tools.Benchmark", file_path=get_log_file("PiscesLx.Tools.Benchmark"), enable_file=True)

        # Setup device
        self._system_monitor = PiscesLxSystemMonitor()
        if self.config.device == "auto":
            devices = self._system_monitor.list_devices()
            gpu_devices = [d for d in devices if d.device_type.name == "GPU"]
            self.config.device = "cuda" if gpu_devices else "cpu"

        # Initialize managers
        self.result_manager = PiscesLxToolsResultManager(self.config.output_dir)
        self.comparison_manager = PiscesLxToolsComparisonManager()

        self.logger.info(
            "Initialized PiscesL1Benchmark",
            model_path=self.config.model_path,
            device=self.config.device,
            datasets=self.config.datasets,
            metrics=self.config.metrics,
        )

    def _load_config(self, config: Union[str, Dict, PiscesLxToolsBenchmarkConfig]) -> PiscesLxToolsBenchmarkConfig:
        if isinstance(config, str):
            cfg = PiscesLxConfiguration()
            cfg.load_from_file(config)
            return PiscesLxToolsBenchmarkConfig(**cfg.to_dict())
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
            model_path=self.config.model_path,
            datasets=self.config.datasets,
            metrics=self.config.metrics,
        )
        try:
            task_config = PiscesLxToolsTaskConfigBuilder.build(self.config)
            results = run_task(task_config)
            summarizer = Summarizer()
            summary = summarizer.summarize(results)
            self.result_manager.save_results(results, summary, self.config)
            self.logger.info(
                "Benchmark completed successfully",
                output_dir=self.config.output_dir,
                results_count=len(results),
            )
            return {"results": results, "summary": summary, "config": asdict(self.config)}
        except Exception as e:
            self.logger.error(
                "Benchmark failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def compare_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.logger.info(
            "Starting model comparison",
            model_count=len(model_configs),
        )
        comparison_results: Dict[str, Any] = {}
        original_config = self.config
        try:
            for i, model_config in enumerate(model_configs):
                model_name = model_config.get("name", f"model_{i+1}")
                self.logger.info(
                    f"Evaluating model {i+1}/{len(model_configs)}",
                    model_name=model_name,
                )
                merged_config = {**asdict(original_config), **model_config}
                self.config = PiscesLxToolsBenchmarkConfig(**merged_config)
                result = self.run_benchmark()
                comparison_results[model_name] = result
                self.logger.info(
                    f"Model {model_name} evaluation completed",
                    model_name=model_name,
                )
            comparison_summary = self.comparison_manager.generate_comparison_report(comparison_results)
            self.logger.info(
                "Model comparison completed",
                models=list(comparison_results.keys()),
            )
            return {"comparison_results": comparison_results, "comparison_summary": comparison_summary}
        finally:
            self.config = original_config


class PiscesLxToolsBenchmarkConfigFactory:
    """PiscesLxTools Benchmark Configuration Factory"""
    
    @staticmethod
    def create(config: Union[str, Dict]) -> PiscesLxToolsBenchmarkConfig:
        """Create benchmark configuration from file or dict"""
        if isinstance(config, str):
            cfg = PiscesLxConfiguration()
            cfg.load_from_file(config)
            return PiscesLxToolsBenchmarkConfig(**cfg.to_dict())
        elif isinstance(config, dict):
            return PiscesLxToolsBenchmarkConfig(**config)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")


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
    def compare(model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple models by running benchmark for each and then generating comparison report"""
        comparison_results: Dict[str, Any] = {}
        
        # Run benchmark for each model
        for i, model_config in enumerate(model_configs):
            model_name = model_config.get("model_name", f"model_{i+1}")
            result = PiscesLxToolsBenchmarkRunner.run(model_config)
            comparison_results[model_name] = result
        
        # Generate comparison report
        manager = PiscesLxToolsComparisonManager()
        comparison_summary = manager.generate_comparison_report(comparison_results)
        
        return {
            "comparison_results": comparison_results,
            "comparison_summary": comparison_summary
        }
