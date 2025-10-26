import json
from typing import Dict, Any, Union, List

from dataclasses import asdict

from utils.log.core import PiscesLxCoreLog
from utils.config.loader import load_config_from_file
from utils.device.manager import PiscesLxCoreDeviceManager

from evalscope.summarizer import Summarizer
from evalscope import run_task

from .config import BenchmarkConfig
from .builders import TaskConfigBuilder
from .result import ResultManager, ComparisonManager

_logger = PiscesLxCoreLog("pisceslx.tools.benchmark")


class PiscesL1Benchmark:
    """PiscesL1 Benchmark runner with EvalScope integration"""

    def __init__(self, config: Union[str, Dict, BenchmarkConfig]):
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
            metrics=self.config.metrics,
        )

    def _load_config(self, config: Union[str, Dict, BenchmarkConfig]) -> BenchmarkConfig:
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
        self.logger.info(
            "Starting benchmark evaluation",
            event="benchmark.start",
            datasets=self.config.datasets,
            metrics=self.config.metrics,
        )
        try:
            task_config = TaskConfigBuilder.build(self.config)
            results = run_task(task_config)
            summarizer = Summarizer()
            summary = summarizer.summarize(results)
            self.result_manager.save_results(results, summary, self.config)
            self.logger.success(
                "Benchmark completed successfully",
                event="benchmark.completed",
                summary=summary,
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
                self.config = BenchmarkConfig(**merged_config)
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


# Factory functions

def create_benchmark_config(model_path: str, **kwargs) -> BenchmarkConfig:
    return BenchmarkConfig(model_path=model_path, **kwargs)


def run_single_benchmark(model_path: str, datasets: List[str] = None, **kwargs) -> Dict[str, Any]:
    config = create_benchmark_config(model_path, datasets=datasets, **kwargs)
    benchmark = PiscesL1Benchmark(config)
    return benchmark.run_benchmark()


def compare_multiple_models(model_configs: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    if not model_configs:
        raise ValueError("At least one model configuration is required")
    base_config = model_configs[0]
    config = create_benchmark_config(**{**base_config, **kwargs})
    benchmark = PiscesL1Benchmark(config)
    return benchmark.compare_models(model_configs)
