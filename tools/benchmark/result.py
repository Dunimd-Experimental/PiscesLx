import os
import json
import datetime
from typing import Dict, Any
from dataclasses import asdict

from utils.log.core import PiscesLxCoreLog

_logger = PiscesLxCoreLog("pisceslx.tools.benchmark")

from .config import PiscesLxToolsBenchmarkConfig


class PiscesLxToolsResultManager:
    """Manager for benchmark results"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_results(self, results: Dict[str, Any], summary: Dict[str, Any], config: PiscesLxToolsBenchmarkConfig):
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
            files=[results_file, summary_file, config_file],
        )


class PiscesLxToolsComparisonManager:
    """Manager for model comparisons"""

    def generate_comparison_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        report = {
            "models": list(results.keys()),
            "metrics": {},
            "rankings": {},
        }
        for model_name, result in results.items():
            summary = result.get("summary", {})
            for metric, value in summary.items():
                if metric not in report["metrics"]:
                    report["metrics"][metric] = {}
                report["metrics"][metric][model_name] = value

        for metric, model_scores in report["metrics"].items():
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            report["rankings"][metric] = sorted_models

        return report
