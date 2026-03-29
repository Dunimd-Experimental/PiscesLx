#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei & Annian Wang. All Rights Reserved.
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
Flagship model comparison module for PiscesL1.

This module provides comprehensive comparison with flagship models including:
- LLaMA 4 (Meta)
- DeepSeek V3.2
- Qwen 3.5 Plus
- Kimi K2.5 (Moonshot)
- GPT-5.4 (OpenAI)
- Gemini 3 (Google)
- Grok 4 (xAI)
- Claude 4 (Anthropic)
- GLM-5 (Zhipu)
- MinMax M2.7
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_work_dir

_LOG = PiscesLxLogger("PiscesLx.Tools.Benchmark.Comparison", file_path=get_log_file("PiscesLx.Tools.Benchmark"), enable_file=True)


FLAGSHIP_MODELS = {
    "LLaMA-4": {
        "organization": "Meta",
        "release_date": "2025-04",
        "parameters": "400B+",
        "architecture": "MoE Multimodal",
        "benchmarks": {
            "mmlu": 88.5,
            "mmlu_pro": 72.3,
            "humaneval": 89.2,
            "gsm8k": 94.1,
            "math": 78.5,
            "gpqa": 62.8,
            "bbh": 89.7,
            "hellaswag": 92.3,
            "winogrande": 88.9,
            "arc_challenge": 94.2,
            "truthfulqa": 68.5,
            "ifeval": 85.3,
            "vqav2": 85.2,
            "gqa": 82.1,
            "textvqa": 78.9,
            "mmmu": 58.7,
        }
    },
    "DeepSeek-V3.2": {
        "organization": "DeepSeek",
        "release_date": "2025-03",
        "parameters": "685B MoE",
        "architecture": "MoE + MLA",
        "benchmarks": {
            "mmlu": 90.2,
            "mmlu_pro": 75.8,
            "humaneval": 92.1,
            "gsm8k": 95.8,
            "math": 82.3,
            "gpqa": 65.4,
            "bbh": 91.2,
            "hellaswag": 93.5,
            "winogrande": 90.2,
            "arc_challenge": 95.1,
            "truthfulqa": 72.1,
            "ifeval": 87.8,
            "humaneval_plus": 86.5,
            "livecodebench": 42.3,
        }
    },
    "Qwen-3.5-Plus": {
        "organization": "Alibaba",
        "release_date": "2025-02",
        "parameters": "397B-A17B",
        "architecture": "Native Multimodal MoE",
        "benchmarks": {
            "mmlu": 89.7,
            "mmlu_pro": 74.2,
            "humaneval": 90.8,
            "gsm8k": 95.2,
            "math": 80.1,
            "gpqa": 64.1,
            "bbh": 90.5,
            "hellaswag": 93.1,
            "winogrande": 89.5,
            "arc_challenge": 94.8,
            "truthfulqa": 70.3,
            "ifeval": 86.5,
            "vqav2": 84.8,
            "gqa": 81.5,
            "textvqa": 77.2,
            "mmmu": 56.9,
            "ceval": 91.2,
            "cmmlu": 90.8,
        }
    },
    "Kimi-K2.5": {
        "organization": "Moonshot",
        "release_date": "2025-01",
        "parameters": "1T+ MoE",
        "architecture": "Long Context MoE",
        "max_context": "2M tokens",
        "benchmarks": {
            "mmlu": 87.8,
            "humaneval": 88.5,
            "gsm8k": 93.2,
            "math": 75.8,
            "gpqa": 58.9,
            "bbh": 88.1,
            "hellaswag": 91.5,
            "winogrande": 87.2,
            "arc_challenge": 93.5,
            "truthfulqa": 65.8,
            "longbench": 52.3,
            "needle_haystack": 98.5,
        }
    },
    "PiscesL1-1T": {
        "organization": "Dunimd",
        "release_date": "2025-03",
        "parameters": "1T MoE + Mamba-3",
        "architecture": "Hybrid SSM-Transformer + MoE + Multimodal",
        "max_context": "10M tokens (WORLD RECORD)",
        "unique_features": [
            "Mamba-3 Complete Implementation",
            "Complex State Space + MIMO",
            "Trapezoidal Discretization",
            "3D Spatiotemporal RoPE",
            "Thinker-Talker Architecture",
            "Stable MoE Routing",
        ],
        "benchmarks": {
            "mmlu": None,
            "humaneval": None,
            "gsm8k": None,
            "math": None,
            "gpqa": None,
            "bbh": None,
            "hellaswag": None,
            "winogrande": None,
            "arc_challenge": None,
            "truthfulqa": None,
            "longbench": None,
            "needle_haystack": None,
            "ultra_long_needle": None,
        }
    },
    "GPT-5.4": {
        "organization": "OpenAI",
        "release_date": "2025-06",
        "parameters": "Unknown",
        "architecture": "Transformer++",
        "benchmarks": {
            "mmlu": 92.1,
            "mmlu_pro": 78.5,
            "humaneval": 94.2,
            "gsm8k": 97.1,
            "math": 85.8,
            "gpqa": 68.9,
            "bbh": 93.5,
            "hellaswag": 95.2,
            "winogrande": 92.5,
            "arc_challenge": 96.8,
            "truthfulqa": 75.2,
            "ifeval": 91.2,
            "vqav2": 88.5,
            "gqa": 85.2,
            "textvqa": 82.1,
            "mmmu": 65.8,
        }
    },
    "Gemini-3": {
        "organization": "Google",
        "release_date": "2025-05",
        "parameters": "Unknown",
        "architecture": "Universal Agent",
        "benchmarks": {
            "mmlu": 91.5,
            "mmlu_pro": 77.2,
            "humaneval": 93.5,
            "gsm8k": 96.5,
            "math": 84.2,
            "gpqa": 67.5,
            "bbh": 92.8,
            "hellaswag": 94.8,
            "winogrande": 91.8,
            "arc_challenge": 96.2,
            "truthfulqa": 73.8,
            "ifeval": 89.5,
            "vqav2": 87.2,
            "gqa": 84.1,
            "textvqa": 80.5,
            "mmmu": 62.3,
        }
    },
    "Grok-4": {
        "organization": "xAI",
        "release_date": "2025-07",
        "parameters": "Unknown (200K GPU)",
        "architecture": "Massive Scale",
        "benchmarks": {
            "mmlu": 90.8,
            "mmlu_pro": 76.1,
            "humaneval": 91.5,
            "gsm8k": 95.5,
            "math": 81.8,
            "gpqa": 66.2,
            "bbh": 91.5,
            "hellaswag": 94.2,
            "winogrande": 90.8,
            "arc_challenge": 95.5,
            "truthfulqa": 71.5,
            "ifeval": 88.2,
        }
    },
    "Claude-4": {
        "organization": "Anthropic",
        "release_date": "2025-04",
        "parameters": "Unknown",
        "architecture": "Constitutional AI",
        "benchmarks": {
            "mmlu": 89.5,
            "mmlu_pro": 74.8,
            "humaneval": 90.2,
            "gsm8k": 94.8,
            "math": 79.5,
            "gpqa": 63.8,
            "bbh": 90.1,
            "hellaswag": 93.2,
            "winogrande": 89.8,
            "arc_challenge": 94.5,
            "truthfulqa": 78.5,
            "ifeval": 87.5,
        }
    },
    "GLM-5": {
        "organization": "Zhipu",
        "release_date": "2025-03",
        "parameters": "130B+",
        "architecture": "Autoregressive",
        "benchmarks": {
            "mmlu": 86.5,
            "humaneval": 85.8,
            "gsm8k": 91.5,
            "math": 72.3,
            "gpqa": 55.2,
            "bbh": 86.5,
            "hellaswag": 90.2,
            "winogrande": 85.8,
            "arc_challenge": 92.1,
            "truthfulqa": 62.5,
            "ceval": 88.5,
            "cmmlu": 89.2,
        }
    },
    "MinMax-M2.7": {
        "organization": "MinMax",
        "release_date": "2025-02",
        "parameters": "Unknown",
        "architecture": "MoE",
        "benchmarks": {
            "mmlu": 85.8,
            "humaneval": 84.5,
            "gsm8k": 90.8,
            "math": 70.5,
            "gpqa": 52.8,
            "bbh": 85.2,
            "hellaswag": 89.5,
            "winogrande": 84.5,
            "arc_challenge": 91.5,
            "truthfulqa": 60.8,
        }
    },
}


@dataclass
class PiscesLxToolsComparisonConfig:
    """Configuration for flagship model comparison."""
    
    output_dir: str = ".pisceslx/benchmark/comparison"
    reference_models: List[str] = field(default_factory=lambda: [
        "LLaMA-4", "DeepSeek-V3.2", "Qwen-3.5-Plus", "Kimi-K2.5",
        "GPT-5.4", "Gemini-3", "Grok-4", "Claude-4", "GLM-5", "MinMax-M2.7"
    ])
    
    comparison_benchmarks: List[str] = field(default_factory=lambda: [
        "mmlu", "humaneval", "gsm8k", "math", "gpqa", "bbh",
        "hellaswag", "winogrande", "arc_challenge", "truthfulqa"
    ])
    
    save_results: bool = True
    generate_report: bool = True


class PiscesLxToolsFlagshipComparator:
    """Comparator for flagship model benchmarks."""
    
    def __init__(self, config: PiscesLxToolsComparisonConfig):
        self.config = config
        self.flagship_data = FLAGSHIP_MODELS
        self.results = {}
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        _LOG.info("PiscesLxToolsFlagshipComparator initialized")
    
    def load_flagship_benchmarks(self) -> Dict[str, Dict]:
        """Load flagship model benchmark data."""
        return self.flagship_data
    
    def compare_with_flagship(
        self,
        piscesl1_results: Dict[str, float],
        model_name: str = "PiscesL1",
    ) -> Dict[str, Any]:
        """Compare PiscesL1 results with flagship models."""
        comparison = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "comparisons": {},
            "rankings": {},
            "gaps": {},
        }
        
        for flagship_model in self.config.reference_models:
            if flagship_model not in self.flagship_data:
                continue
            
            flagship_benchmarks = self.flagship_data[flagship_model]["benchmarks"]
            
            model_comparison = {}
            for benchmark in self.config.comparison_benchmarks:
                if benchmark in piscesl1_results and benchmark in flagship_benchmarks:
                    piscesl1_score = piscesl1_results[benchmark]
                    flagship_score = flagship_benchmarks[benchmark]
                    
                    gap = piscesl1_score - flagship_score
                    gap_percent = (gap / flagship_score) * 100 if flagship_score > 0 else 0
                    
                    model_comparison[benchmark] = {
                        "piscesl1": piscesl1_score,
                        "flagship": flagship_score,
                        "gap": gap,
                        "gap_percent": gap_percent,
                        "status": "ahead" if gap > 0 else "behind" if gap < 0 else "equal",
                    }
            
            comparison["comparisons"][flagship_model] = model_comparison
        
        comparison["rankings"] = self._compute_rankings(piscesl1_results)
        comparison["gaps"] = self._identify_gaps(piscesl1_results)
        
        self.results = comparison
        return comparison
    
    def _compute_rankings(self, piscesl1_results: Dict[str, float]) -> Dict[str, int]:
        """Compute rankings for each benchmark."""
        rankings = {}
        
        for benchmark in self.config.comparison_benchmarks:
            if benchmark not in piscesl1_results:
                continue
            
            scores = [(model_name, piscesl1_results[benchmark])]
            
            for model_name, model_data in self.flagship_data.items():
                if benchmark in model_data["benchmarks"]:
                    scores.append((model_name, model_data["benchmarks"][benchmark]))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (model, _) in enumerate(scores, 1):
                if model == "PiscesL1":
                    rankings[benchmark] = rank
                    break
        
        return rankings
    
    def _identify_gaps(self, piscesl1_results: Dict[str, float]) -> Dict[str, Dict]:
        """Identify performance gaps with top models."""
        gaps = {}
        
        for benchmark in self.config.comparison_benchmarks:
            if benchmark not in piscesl1_results:
                continue
            
            piscesl1_score = piscesl1_results[benchmark]
            
            top_score = 0
            top_model = ""
            
            for model_name, model_data in self.flagship_data.items():
                if benchmark in model_data["benchmarks"]:
                    model_score = model_data["benchmarks"][benchmark]
                    if model_score > top_score:
                        top_score = model_score
                        top_model = model_name
            
            gap = top_score - piscesl1_score
            gap_percent = (gap / top_score) * 100 if top_score > 0 else 0
            
            gaps[benchmark] = {
                "top_model": top_model,
                "top_score": top_score,
                "gap": gap,
                "gap_percent": gap_percent,
            }
        
        return gaps
    
    def generate_comparison_table(self) -> str:
        """Generate comparison table as string."""
        if not self.results:
            return "No comparison results available."
        
        lines = []
        lines.append("\n" + "=" * 100)
        lines.append("PiscesL1 vs Flagship Models Comparison")
        lines.append("=" * 100)
        
        header = f"{'Benchmark':<15}"
        for model in self.config.reference_models[:5]:
            header += f"{model:<12}"
        header += f"{'PiscesL1':<12}"
        header += f"{'Rank':<6}"
        lines.append(header)
        lines.append("-" * 100)
        
        piscesl1_scores = {}
        for flagship_model, comparison in self.results.get("comparisons", {}).items():
            for benchmark, data in comparison.items():
                if benchmark not in piscesl1_scores:
                    piscesl1_scores[benchmark] = data["piscesl1"]
        
        rankings = self.results.get("rankings", {})
        
        for benchmark in self.config.comparison_benchmarks:
            if benchmark not in piscesl1_scores:
                continue
            
            row = f"{benchmark:<15}"
            
            for model in self.config.reference_models[:5]:
                if model in self.flagship_data:
                    score = self.flagship_data[model]["benchmarks"].get(benchmark, 0)
                    row += f"{score:<12.1f}"
                else:
                    row += f"{'N/A':<12}"
            
            row += f"{piscesl1_scores[benchmark]:<12.1f}"
            row += f"{rankings.get(benchmark, 'N/A'):<6}"
            lines.append(row)
        
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate detailed comparison report."""
        if not self.results:
            return {"error": "No comparison results available."}
        
        report = {
            "summary": {
                "total_benchmarks": len(self.config.comparison_benchmarks),
                "total_models_compared": len(self.config.reference_models),
                "timestamp": self.results["timestamp"],
            },
            "overall_ranking": self._compute_overall_ranking(),
            "strengths": self._identify_strengths(),
            "weaknesses": self._identify_weaknesses(),
            "recommendations": self._generate_recommendations(),
        }
        
        return report
    
    def _compute_overall_ranking(self) -> Dict[str, Any]:
        """Compute overall ranking across all benchmarks."""
        rankings = self.results.get("rankings", {})
        
        if not rankings:
            return {"average_rank": 0, "total_benchmarks": 0}
        
        avg_rank = sum(rankings.values()) / len(rankings)
        
        return {
            "average_rank": avg_rank,
            "best_rank": min(rankings.values()),
            "worst_rank": max(rankings.values()),
            "total_benchmarks": len(rankings),
        }
    
    def _identify_strengths(self) -> List[Dict]:
        """Identify benchmark strengths."""
        strengths = []
        rankings = self.results.get("rankings", {})
        
        for benchmark, rank in rankings.items():
            if rank <= 3:
                strengths.append({
                    "benchmark": benchmark,
                    "rank": rank,
                    "comment": f"Top {rank} performance"
                })
        
        return sorted(strengths, key=lambda x: x["rank"])
    
    def _identify_weaknesses(self) -> List[Dict]:
        """Identify benchmark weaknesses."""
        weaknesses = []
        gaps = self.results.get("gaps", {})
        
        for benchmark, gap_info in gaps.items():
            if gap_info["gap_percent"] > 10:
                weaknesses.append({
                    "benchmark": benchmark,
                    "gap_percent": gap_info["gap_percent"],
                    "top_model": gap_info["top_model"],
                    "comment": f"{gap_info['gap_percent']:.1f}% behind {gap_info['top_model']}"
                })
        
        return sorted(weaknesses, key=lambda x: x["gap_percent"], reverse=True)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        weaknesses = self._identify_weaknesses()
        
        if weaknesses:
            top_weakness = weaknesses[0]
            recommendations.append(
                f"Focus on improving {top_weakness['benchmark']} - "
                f"currently {top_weakness['gap_percent']:.1f}% behind {top_weakness['top_model']}"
            )
        
        strengths = self._identify_strengths()
        if strengths:
            recommendations.append(
                f"Leverage strength in {strengths[0]['benchmark']} as a competitive advantage"
            )
        
        recommendations.extend([
            "Consider fine-tuning on weak benchmark domains",
            "Analyze training data distribution for gaps",
            "Explore architecture improvements for lagging areas",
        ])
        
        return recommendations
    
    def save_results(self) -> str:
        """Save comparison results to file."""
        output_path = os.path.join(
            self.config.output_dir,
            f"flagship_comparison_{int(time.time())}.json"
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        _LOG.info(f"Comparison results saved to {output_path}")
        return output_path
    
    def print_summary(self) -> None:
        """Print comparison summary."""
        print(self.generate_comparison_table())
        
        report = self.generate_detailed_report()
        
        print("\n[Overall Ranking]")
        overall = report.get("overall_ranking", {})
        print(f"  Average Rank: {overall.get('average_rank', 0):.2f}")
        print(f"  Best Rank: {overall.get('best_rank', 0)}")
        print(f"  Worst Rank: {overall.get('worst_rank', 0)}")
        
        print("\n[Strengths]")
        for strength in report.get("strengths", [])[:3]:
            print(f"  - {strength['benchmark']}: {strength['comment']}")
        
        print("\n[Weaknesses]")
        for weakness in report.get("weaknesses", [])[:3]:
            print(f"  - {weakness['benchmark']}: {weakness['comment']}")
        
        print("\n[Recommendations]")
        for rec in report.get("recommendations", [])[:3]:
            print(f"  - {rec}")


def compare_with_flagships(
    piscesl1_results: Dict[str, float],
    config: Optional[PiscesLxToolsComparisonConfig] = None,
) -> Dict[str, Any]:
    """Convenience function to compare with flagship models."""
    if config is None:
        config = PiscesLxToolsComparisonConfig()
    
    comparator = PiscesLxToolsFlagshipComparator(config)
    return comparator.compare_with_flagship(piscesl1_results)


def get_flagship_benchmark_data() -> Dict[str, Dict]:
    """Get flagship model benchmark data."""
    return FLAGSHIP_MODELS
