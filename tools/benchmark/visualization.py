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

from __future__ import annotations

"""
Visualization module for PiscesLx benchmark results.

This module provides comprehensive visualization capabilities including:
- Radar charts for multi-dimensional comparison
- Bar charts for benchmark comparison
- Heatmaps for performance matrices
- Line charts for trend analysis
- HTML report generation
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_work_dir

_LOG = PiscesLxLogger("PiscesLx.Tools.Benchmark.Visualization", file_path=get_log_file("PiscesLx.Tools.Benchmark"), enable_file=True)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    _LOG.warning("Matplotlib not available. Visualization features will be limited.")


@dataclass
class PiscesLxToolsVisualizationConfig:
    """Configuration for benchmark visualization."""
    
    output_dir: str = ".pisceslx/benchmark/visualization"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 150
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    
    title_fontsize: int = 16
    label_fontsize: int = 12
    legend_fontsize: int = 10
    
    save_format: str = "png"
    save_html: bool = True


class PiscesLxToolsBenchmarkVisualizer:
    """Visualizer for benchmark results."""
    
    def __init__(self, config: PiscesLxToolsVisualizationConfig):
        self.config = config
        
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(self.config.style)
            except Exception:
                pass
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        _LOG.info("PiscesLxToolsBenchmarkVisualizer initialized")
    
    def plot_radar_chart(
        self,
        scores: Dict[str, float],
        model_name: str = "PiscesLx",
        title: str = "Benchmark Performance Radar",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot radar chart for multi-dimensional comparison."""
        if not MATPLOTLIB_AVAILABLE:
            _LOG.warning("Matplotlib not available for radar chart")
            return None
        
        categories = list(scores.keys())
        values = list(scores.values())
        
        if not categories:
            return None
        
        num_vars = len(categories)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        values += values[:1]
        
        fig, ax = plt.subplots(figsize=self.config.figure_size, subplot_kw=dict(polar=True))
        
        ax.plot(angles, values, 'o-', linewidth=2, color=self.config.color_palette[0])
        ax.fill(angles, values, alpha=0.25, color=self.config.color_palette[0])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=self.config.label_fontsize)
        
        ax.set_ylim(0, 1)
        
        ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)
        
        if save_path is None:
            save_path = os.path.join(
                self.config.output_dir,
                f"radar_chart_{model_name}_{int(time.time())}.{self.config.save_format}"
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        _LOG.info(f"Radar chart saved to {save_path}")
        return save_path
    
    def plot_bar_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Benchmark Comparison",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot bar chart for benchmark comparison."""
        if not MATPLOTLIB_AVAILABLE:
            _LOG.warning("Matplotlib not available for bar chart")
            return None
        
        models = list(results.keys())
        benchmarks = set()
        for model_results in results.values():
            benchmarks.update(model_results.keys())
        benchmarks = sorted(list(benchmarks))
        
        if not benchmarks or not models:
            return None
        
        x = np.arange(len(benchmarks))
        width = 0.8 / len(models)
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        for i, model in enumerate(models):
            values = [results[model].get(b, 0) for b in benchmarks]
            ax.bar(
                x + i * width,
                values,
                width,
                label=model,
                color=self.config.color_palette[i % len(self.config.color_palette)]
            )
        
        ax.set_xlabel('Benchmarks', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Score', fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=self.config.label_fontsize)
        ax.legend(fontsize=self.config.legend_fontsize)
        ax.set_ylim(0, 1)
        
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(
                self.config.output_dir,
                f"bar_comparison_{int(time.time())}.{self.config.save_format}"
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        _LOG.info(f"Bar comparison chart saved to {save_path}")
        return save_path
    
    def plot_heatmap(
        self,
        matrix: List[List[float]],
        row_labels: List[str],
        col_labels: List[str],
        title: str = "Performance Heatmap",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot heatmap for performance matrix."""
        if not MATPLOTLIB_AVAILABLE:
            _LOG.warning("Matplotlib not available for heatmap")
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, fontsize=self.config.label_fontsize)
        ax.set_yticklabels(row_labels, fontsize=self.config.label_fontsize)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(
                    j, i, f"{matrix[i][j]:.2f}",
                    ha="center", va="center", color="black", fontsize=8
                )
        
        ax.set_title(title, fontsize=self.config.title_fontsize)
        
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Score", rotation=-90, va="bottom", fontsize=self.config.label_fontsize)
        
        if save_path is None:
            save_path = os.path.join(
                self.config.output_dir,
                f"heatmap_{int(time.time())}.{self.config.save_format}"
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        _LOG.info(f"Heatmap saved to {save_path}")
        return save_path
    
    def plot_line_trend(
        self,
        data: Dict[str, List[Tuple[float, float]]],
        title: str = "Performance Trend",
        xlabel: str = "Context Length",
        ylabel: str = "Accuracy",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot line chart for trend analysis."""
        if not MATPLOTLIB_AVAILABLE:
            _LOG.warning("Matplotlib not available for line chart")
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        for i, (label, points) in enumerate(data.items()):
            x_values = [p[0] for p in points]
            y_values = [p[1] for p in points]
            ax.plot(
                x_values, y_values,
                'o-',
                label=label,
                color=self.config.color_palette[i % len(self.config.color_palette)]
            )
        
        ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.legend(fontsize=self.config.legend_fontsize)
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(
                self.config.output_dir,
                f"line_trend_{int(time.time())}.{self.config.save_format}"
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        _LOG.info(f"Line trend chart saved to {save_path}")
        return save_path
    
    def plot_needle_haystack_heatmap(
        self,
        results: List[Dict],
        title: str = "Needle in Haystack Performance",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot heatmap for Needle in Haystack results."""
        if not MATPLOTLIB_AVAILABLE:
            _LOG.warning("Matplotlib not available for heatmap")
            return None
        
        context_lengths = sorted(set(r["context_length"] for r in results))
        positions = sorted(set(r["position"] for r in results))
        
        matrix = np.zeros((len(context_lengths), len(positions)))
        
        for r in results:
            i = context_lengths.index(r["context_length"])
            j = positions.index(r["position"])
            matrix[i][j] = 1.0 if r["correct"] else 0.0
        
        row_labels = [str(l) for l in context_lengths]
        col_labels = [f"{p:.1f}" for p in positions]
        
        return self.plot_heatmap(
            matrix.tolist(),
            row_labels,
            col_labels,
            title,
            save_path
        )


class PiscesLxToolsReportGenerator:
    """Generator for benchmark reports."""
    
    def __init__(self, config: PiscesLxToolsVisualizationConfig):
        self.config = config
        self.visualizer = PiscesLxToolsBenchmarkVisualizer(config)
        
        _LOG.info("PiscesLxToolsReportGenerator initialized")
    
    def generate_markdown_report(
        self,
        results: Dict[str, Any],
        model_name: str = "PiscesLx",
        save_path: Optional[str] = None,
    ) -> str:
        """Generate Markdown report."""
        report = []
        
        report.append(f"# {model_name} Benchmark Report\n")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("\n## Summary\n")
        
        if "summary" in results:
            summary = results["summary"]
            report.append("\n| Benchmark | Score |\n")
            report.append("|-----------|-------|\n")
            for benchmark, score in summary.items():
                if benchmark not in ["overall_score", "average"]:
                    report.append(f"| {benchmark} | {score:.4f} |\n")
            if "overall_score" in summary:
                report.append(f"| **Overall** | **{summary['overall_score']:.4f}** |\n")
            elif "average" in summary:
                report.append(f"| **Average** | **{summary['average']:.4f}** |\n")
        
        if "mmlu" in results:
            report.append("\n## MMLU Results\n")
            mmlu = results["mmlu"]
            if "accuracy_by_subject" in mmlu:
                report.append("\n| Subject | Accuracy |\n")
                report.append("|---------|----------|\n")
                for subject, acc in sorted(mmlu["accuracy_by_subject"].items()):
                    report.append(f"| {subject} | {acc:.4f} |\n")
        
        if "humaneval" in results:
            report.append("\n## HumanEval Results\n")
            humaneval = results["humaneval"]
            report.append(f"\n- **Pass Rate**: {humaneval.get('pass_rate', 0):.4f}\n")
            report.append(f"- **Correct**: {humaneval.get('correct', 0)}\n")
            report.append(f"- **Total**: {humaneval.get('total', 0)}\n")
        
        if "gsm8k" in results:
            report.append("\n## GSM8K Results\n")
            gsm8k = results["gsm8k"]
            report.append(f"\n- **Accuracy**: {gsm8k.get('accuracy', 0):.4f}\n")
            report.append(f"- **Correct**: {gsm8k.get('correct', 0)}\n")
            report.append(f"- **Total**: {gsm8k.get('total', 0)}\n")
        
        report_content = "".join(report)
        
        if save_path is None:
            save_path = os.path.join(
                self.config.output_dir,
                f"benchmark_report_{model_name}_{int(time.time())}.md"
            )
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        _LOG.info(f"Markdown report saved to {save_path}")
        return save_path
    
    def generate_html_report(
        self,
        results: Dict[str, Any],
        model_name: str = "PiscesLx",
        charts: Optional[Dict[str, str]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate HTML report."""
        html_parts = []
        
        html_parts.append(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_name} Benchmark Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #1f77b4;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        .summary-card {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .score {{
            font-size: 2em;
            font-weight: bold;
            color: #1f77b4;
        }}
        .chart-container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>{model_name} Benchmark Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
""")
        
        if "summary" in results:
            summary = results["summary"]
            overall = summary.get("overall_score", summary.get("average", 0))
            
            html_parts.append(f"""
    <div class="summary-card">
        <h2>Overall Score</h2>
        <p class="score">{overall:.4f}</p>
    </div>
""")
            
            html_parts.append("""
    <h2>Benchmark Summary</h2>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Score</th>
        </tr>
""")
            for benchmark, score in summary.items():
                if benchmark not in ["overall_score", "average"]:
                    html_parts.append(f"""
        <tr>
            <td>{benchmark}</td>
            <td>{score:.4f}</td>
        </tr>
""")
            html_parts.append("    </table>\n")
        
        if charts:
            html_parts.append("""
    <h2>Visualizations</h2>
""")
            for chart_name, chart_path in charts.items():
                if os.path.exists(chart_path):
                    html_parts.append(f"""
    <div class="chart-container">
        <h3>{chart_name}</h3>
        <img src="{os.path.basename(chart_path)}" alt="{chart_name}">
    </div>
""")
        
        if "mmlu" in results and "accuracy_by_subject" in results["mmlu"]:
            html_parts.append("""
    <h2>MMLU Results by Subject</h2>
    <table>
        <tr>
            <th>Subject</th>
            <th>Accuracy</th>
        </tr>
""")
            for subject, acc in sorted(results["mmlu"]["accuracy_by_subject"].items()):
                html_parts.append(f"""
        <tr>
            <td>{subject}</td>
            <td>{acc:.4f}</td>
        </tr>
""")
            html_parts.append("    </table>\n")
        
        html_parts.append("""
</body>
</html>
""")
        
        html_content = "".join(html_parts)
        
        if save_path is None:
            save_path = os.path.join(
                self.config.output_dir,
                f"benchmark_report_{model_name}_{int(time.time())}.html"
            )
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        _LOG.info(f"HTML report saved to {save_path}")
        return save_path
    
    def generate_latex_table(
        self,
        results: Dict[str, Dict[str, float]],
        caption: str = "Benchmark Results",
        save_path: Optional[str] = None,
    ) -> str:
        """Generate LaTeX table."""
        models = list(results.keys())
        benchmarks = set()
        for model_results in results.values():
            benchmarks.update(model_results.keys())
        benchmarks = sorted(list(benchmarks))
        
        latex_parts = []
        
        latex_parts.append("\\begin{table}[htbp]\n")
        latex_parts.append("\\centering\n")
        latex_parts.append(f"\\caption{{{caption}}}\n")
        latex_parts.append("\\begin{tabular}{l" + "c" * len(models) + "}\n")
        latex_parts.append("\\toprule\n")
        latex_parts.append("Benchmark & " + " & ".join(models) + " \\\\\n")
        latex_parts.append("\\midrule\n")
        
        for benchmark in benchmarks:
            row = [benchmark]
            for model in models:
                score = results[model].get(benchmark, 0)
                row.append(f"{score:.4f}")
            latex_parts.append(" & ".join(row) + " \\\\\n")
        
        latex_parts.append("\\bottomrule\n")
        latex_parts.append("\\end{tabular}\n")
        latex_parts.append("\\end{table}\n")
        
        latex_content = "".join(latex_parts)
        
        if save_path is None:
            save_path = os.path.join(
                self.config.output_dir,
                f"benchmark_table_{int(time.time())}.tex"
            )
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        _LOG.info(f"LaTeX table saved to {save_path}")
        return save_path


def create_visualizer(
    config: Optional[PiscesLxToolsVisualizationConfig] = None
) -> PiscesLxToolsBenchmarkVisualizer:
    """Factory function to create visualizer."""
    if config is None:
        config = PiscesLxToolsVisualizationConfig()
    return PiscesLxToolsBenchmarkVisualizer(config)


def create_report_generator(
    config: Optional[PiscesLxToolsVisualizationConfig] = None
) -> PiscesLxToolsReportGenerator:
    """Factory function to create report generator."""
    if config is None:
        config = PiscesLxToolsVisualizationConfig()
    return PiscesLxToolsReportGenerator(config)
