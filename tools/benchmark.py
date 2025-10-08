#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import datetime
import argparse
from pathlib import Path
from utils.log.core import PiscesLxCoreLog
from evalscope.models import ModelConfig
from dataclasses import dataclass, asdict
from evalscope import TaskConfig, run_task
from evalscope.metrics import MetricConfig
from evalscope.summarizer import Summarizer
from evalscope.datasets import DatasetConfig
from typing import Dict, List, Optional, Any, Union
from utils.config.loader import load_config_from_file
from utils.device.manager import PiscesLxCoreDeviceManager

# Initialize logger
_logger = PiscesLxCoreLog("pisceslx.tools.benchmark")

@dataclass
class BenchmarkConfig:
    """Benchmark configuration data class"""
    model_path: str
    model_name: Optional[str] = None
    datasets: List[str] = None
    metrics: List[str] = None
    batch_size: int = 8
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"
    output_dir: str = "benchmark_results"
    use_cache: bool = True
    save_predictions: bool = True
    debug: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and defaults"""
        if self.datasets is None:
            self.datasets = ["mmlu", "ceval", "gsm8k", "arc", "hellaswag"]
        if self.metrics is None:
            self.metrics = ["accuracy", "f1", "precision", "recall"]
        if self.model_name is None:
            self.model_name = Path(self.model_path).name


class ModelConfigBuilder:
    """Builder for model configurations"""
    
    @staticmethod
    def build(config: BenchmarkConfig) -> ModelConfig:
        """Build model configuration for EvalScope"""
        return ModelConfig(
            model_id=config.model_path,
            model_name=config.model_name,
            device=config.device,
            max_length=config.max_length,
            temperature=config.temperature,
            top_p=config.top_p,
            batch_size=config.batch_size,
            use_cache=config.use_cache
        )


class DatasetConfigBuilder:
    """Builder for dataset configurations"""
    
    @staticmethod
    def build(config: BenchmarkConfig) -> List[DatasetConfig]:
        """Build dataset configurations for EvalScope"""
        dataset_configs = []
        
        for dataset_name in config.datasets:
            dataset_config = DatasetConfig(
                dataset_id=dataset_name,
                subset="default",
                split="test",
                limit=None,  # Use full dataset
                cache_dir=os.path.join(config.output_dir, "cache", dataset_name)
            )
            dataset_configs.append(dataset_config)
        
        return dataset_configs


class MetricConfigBuilder:
    """Builder for metric configurations"""
    
    @staticmethod
    def build(config: BenchmarkConfig) -> List[MetricConfig]:
        """Build metric configurations for EvalScope"""
        metric_configs = []
        
        for metric_name in config.metrics:
            metric_config = MetricConfig(
                metric_id=metric_name,
                params={}
            )
            metric_configs.append(metric_config)
        
        return metric_configs


class TaskConfigBuilder:
    """Builder for task configurations"""
    
    @staticmethod
    def build(config: BenchmarkConfig) -> TaskConfig:
        """Build task configuration for EvalScope"""
        model_config = ModelConfigBuilder.build(config)
        dataset_configs = DatasetConfigBuilder.build(config)
        metric_configs = MetricConfigBuilder.build(config)
        
        return TaskConfig(
            model_config=model_config,
            dataset_configs=dataset_configs,
            metric_configs=metric_configs,
            output_dir=config.output_dir,
            save_predictions=config.save_predictions,
            debug=config.debug
        )


class ResultManager:
    """Manager for benchmark results"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_results(self, results: Dict[str, Any], summary: Dict[str, Any], config: BenchmarkConfig):
        """Save benchmark results to files"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, f"benchmark_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save configuration
        config_file = os.path.join(self.output_dir, f"benchmark_config_{timestamp}.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
        
        _logger.success(
            "Benchmark results saved successfully",
            event="benchmark.save_results",
            output_dir=self.output_dir,
            timestamp=timestamp,
            files=[results_file, summary_file, config_file]
        )


class ComparisonManager:
    """Manager for model comparisons"""
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison report from multiple model results"""
        report = {
            "models": list(results.keys()),
            "metrics": {},
            "rankings": {}
        }
        
        # Collect metrics for each model
        for model_name, result in results.items():
            summary = result.get("summary", {})
            for metric, value in summary.items():
                if metric not in report["metrics"]:
                    report["metrics"][metric] = {}
                report["metrics"][metric][model_name] = value
        
        # Generate rankings
        for metric, model_scores in report["metrics"].items():
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            report["rankings"][metric] = sorted_models
        
        return report


class PiscesL1Benchmark:
    """PiscesL1 Benchmark runner with EvalScope integration"""
    
    def __init__(self, config: Union[str, Dict, BenchmarkConfig]):
        """Initialize benchmark runner"""
        # Load configuration
        self.config = self._load_config(config)
        self.logger = _logger.bind(benchmark_model=self.config.model_name)
        
        # Setup device
        self.device_manager = PiscesLxCoreDeviceManager()
        if self.config.device == "auto":
            self.config.device = self.device_manager.get_optimal_device()
        
        # Initialize managers
        self.result_manager = ResultManager(self.config.output_dir)
        self.comparison_manager = ComparisonManager()
        
        self.logger.info(
            "Initialized PiscesL1Benchmark",
            event="benchmark.initialized",
            model_path=self.config.model_path,
            device=self.config.device,
            datasets=self.config.datasets,
            metrics=self.config.metrics
        )
    
    def _load_config(self, config: Union[str, Dict, BenchmarkConfig]) -> BenchmarkConfig:
        """Load and validate configuration"""
        if isinstance(config, str):
            config_data = load_config_from_file(config)
            return BenchmarkConfig(**config_data)
        elif isinstance(config, dict):
            return BenchmarkConfig(**config)
        elif isinstance(config, BenchmarkConfig):
            return config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run benchmark evaluation"""
        self.logger.info(
            "Starting benchmark evaluation",
            event="benchmark.start",
            datasets=self.config.datasets,
            metrics=self.config.metrics
        )
        
        try:
            # Create task configuration
            task_config = TaskConfigBuilder.build(self.config)
            
            # Run evaluation
            results = run_task(task_config)
            
            # Generate summary report
            summarizer = Summarizer()
            summary = summarizer.summarize(results)
            
            # Save results
            self.result_manager.save_results(results, summary, self.config)
            
            self.logger.success(
                "Benchmark completed successfully",
                event="benchmark.completed",
                summary=summary
            )
            
            return {
                "results": results,
                "summary": summary,
                "config": asdict(self.config)
            }
            
        except Exception as e:
            self.logger.error(
                "Benchmark failed",
                event="benchmark.failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    def compare_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple models"""
        self.logger.info(
            "Starting model comparison",
            event="benchmark.comparison.start",
            model_count=len(model_configs)
        )
        
        comparison_results = {}
        original_config = self.config
        
        try:
            for i, model_config in enumerate(model_configs):
                model_name = model_config.get("name", f"model_{i+1}")
                self.logger.info(
                    f"Evaluating model {i+1}/{len(model_configs)}",
                    event="benchmark.model.evaluation.start",
                    model_name=model_name
                )
                
                # Update config for current model
                merged_config = {**asdict(original_config), **model_config}
                self.config = BenchmarkConfig(**merged_config)
                
                # Run benchmark
                result = self.run_benchmark()
                comparison_results[model_name] = result
                
                self.logger.success(
                    f"Model {model_name} evaluation completed",
                    event="benchmark.model.evaluation.completed",
                    model_name=model_name
                )
            
            # Generate comparison report
            comparison_summary = self.comparison_manager.generate_comparison_report(comparison_results)
            
            self.logger.success(
                "Model comparison completed",
                event="benchmark.comparison.completed",
                models=list(comparison_results.keys())
            )
            
            return {
                "comparison_results": comparison_results,
                "comparison_summary": comparison_summary
            }
            
        finally:
            # Restore original config
            self.config = original_config


# Factory functions
def create_benchmark_config(model_path: str, **kwargs) -> BenchmarkConfig:
    """Create benchmark configuration"""
    return BenchmarkConfig(model_path=model_path, **kwargs)


def run_single_benchmark(model_path: str, datasets: List[str] = None, **kwargs) -> Dict[str, Any]:
    """Run single model benchmark"""
    config = create_benchmark_config(model_path, datasets=datasets, **kwargs)
    benchmark = PiscesL1Benchmark(config)
    return benchmark.run_benchmark()


def compare_multiple_models(model_configs: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """Compare multiple models"""
    if not model_configs:
        raise ValueError("At least one model configuration is required")
    
    # Use first model config as base config
    base_config = model_configs[0]
    config = create_benchmark_config(**{**base_config, **kwargs})
    benchmark = PiscesL1Benchmark(config)
    
    return benchmark.compare_models(model_configs)


# CLI interface
def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="PiscesL1 Benchmark with EvalScope",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model benchmark
  python -m tools.benchmark --model-path /path/to/model --datasets mmlu ceval
  
  # Compare multiple models
  python -m tools.benchmark --compare /path/to/model1 /path/to/model2 --datasets mmlu
  
  # Custom configuration
  python -m tools.benchmark --model-path /path/to/model --batch-size 16 --max-length 4096
        """
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
    
    # Model comparison
    parser.add_argument("--compare", nargs="+", help="Compare multiple models")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        _logger.set_level("DEBUG")
    
    try:
        if args.compare:
            # Compare multiple models
            model_configs = []
            for model_path in args.compare:
                model_name = args.model_name or Path(model_path).name
                model_configs.append({
                    "model_path": model_path,
                    "model_name": model_name,
                    "datasets": args.datasets,
                    "metrics": args.metrics,
                    "output_dir": args.output_dir,
                    "device": args.device,
                    "debug": args.debug,
                    "batch_size": args.batch_size,
                    "max_length": args.max_length,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "use_cache": args.use_cache,
                    "save_predictions": args.save_predictions
                })
            
            results = compare_multiple_models(model_configs)
            print(json.dumps(results["comparison_summary"], indent=2))
        else:
            # Single model benchmark
            if not args.model_path:
                parser.error("--model-path is required when not using --compare")
            
            results = run_single_benchmark(
                model_path=args.model_path,
                model_name=args.model_name,
                datasets=args.datasets,
                metrics=args.metrics,
                batch_size=args.batch_size,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                output_dir=args.output_dir,
                device=args.device,
                debug=args.debug,
                use_cache=args.use_cache,
                save_predictions=args.save_predictions
            )
            
            print(json.dumps(results["summary"], indent=2))
            
    except Exception as e:
        _logger.error(
            "Benchmark execution failed",
            event="benchmark.execution.failed",
            error=str(e),
            error_type=type(e).__name__
        )
        sys.exit(1)


if __name__ == "__main__":
    main()