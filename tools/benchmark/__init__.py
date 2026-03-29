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
PiscesL1 Benchmark Module - Flagship-level evaluation suite.

This module provides comprehensive benchmark evaluation including:
- Text benchmarks (MMLU, HumanEval, GSM8K, etc.)
- Multimodal benchmarks (Vision, Audio, Video)
- Agent benchmarks (AgentBench, WebShop, etc.)
- Long context benchmarks (LongBench, Needle in Haystack)
- Performance metrics (Latency, Throughput, Memory)
- Flagship model comparison (LLaMA-4, DeepSeek, Qwen, etc.)
- Visualization and report generation
"""

from .config import PiscesLxToolsBenchmarkConfig, MODALITY_DATASETS
from .builders import (
    PiscesLxToolsModelConfigBuilder,
    PiscesLxToolsDatasetConfigBuilder,
    PiscesLxToolsMetricConfigBuilder,
    PiscesLxToolsTaskConfigBuilder
)
from .result import PiscesLxToolsResultManager, PiscesLxToolsComparisonManager
from .runner import PiscesLxToolsBenchmark, PiscesLxToolsBenchmarkRunner, PiscesLxToolsBenchmarkComparer
from .benchmark import PiscesL1BenchmarkConfig, PiscesL1BenchmarkEvaluator, create_benchmark_evaluator

from .multimodal import (
    PiscesLxToolsMultimodalConfig,
    PiscesLxToolsVisionEvaluator,
    PiscesLxToolsAudioEvaluator,
    PiscesLxToolsVideoEvaluator,
    PiscesLxToolsMultimodalBenchmarkRunner,
    create_multimodal_evaluator,
)

from .agent import (
    PiscesLxToolsAgentConfig,
    PiscesLxToolsAgentEvaluator,
    PiscesLxToolsAgentMetrics,
    PiscesLxToolsAgentBenchmarkRunner,
    create_agent_evaluator,
)

from .long_context import (
    PiscesLxToolsLongContextConfig,
    PiscesLxToolsLongContextEvaluator,
    PiscesLxToolsLongContextMetrics,
    PiscesLxToolsLongContextBenchmarkRunner,
    create_long_context_evaluator,
)

from .visualization import (
    PiscesLxToolsVisualizationConfig,
    PiscesLxToolsBenchmarkVisualizer,
    PiscesLxToolsReportGenerator,
    create_visualizer,
    create_report_generator,
)

from .comparison import (
    PiscesLxToolsComparisonConfig,
    PiscesLxToolsFlagshipComparator,
    FLAGSHIP_MODELS,
    compare_with_flagships,
    get_flagship_benchmark_data,
)

from .performance import (
    PiscesLxToolsPerformanceConfig,
    PiscesLxToolsPerformanceEvaluator,
    PiscesLxToolsPerformanceMetrics,
    PiscesLxToolsPerformanceBenchmarkRunner,
    create_performance_evaluator,
)

__all__ = [
    "PiscesLxToolsBenchmarkConfig",
    "MODALITY_DATASETS",
    "PiscesLxToolsModelConfigBuilder",
    "PiscesLxToolsDatasetConfigBuilder",
    "PiscesLxToolsMetricConfigBuilder",
    "PiscesLxToolsTaskConfigBuilder",
    "PiscesLxToolsResultManager",
    "PiscesLxToolsComparisonManager",
    "PiscesLxToolsBenchmark",
    "PiscesLxToolsBenchmarkRunner",
    "PiscesLxToolsBenchmarkComparer",
    "PiscesL1BenchmarkConfig",
    "PiscesL1BenchmarkEvaluator",
    "create_benchmark_evaluator",
    "PiscesLxToolsMultimodalConfig",
    "PiscesLxToolsVisionEvaluator",
    "PiscesLxToolsAudioEvaluator",
    "PiscesLxToolsVideoEvaluator",
    "PiscesLxToolsMultimodalBenchmarkRunner",
    "create_multimodal_evaluator",
    "PiscesLxToolsAgentConfig",
    "PiscesLxToolsAgentEvaluator",
    "PiscesLxToolsAgentMetrics",
    "PiscesLxToolsAgentBenchmarkRunner",
    "create_agent_evaluator",
    "PiscesLxToolsLongContextConfig",
    "PiscesLxToolsLongContextEvaluator",
    "PiscesLxToolsLongContextMetrics",
    "PiscesLxToolsLongContextBenchmarkRunner",
    "create_long_context_evaluator",
    "PiscesLxToolsVisualizationConfig",
    "PiscesLxToolsBenchmarkVisualizer",
    "PiscesLxToolsReportGenerator",
    "create_visualizer",
    "create_report_generator",
    "PiscesLxToolsComparisonConfig",
    "PiscesLxToolsFlagshipComparator",
    "FLAGSHIP_MODELS",
    "compare_with_flagships",
    "get_flagship_benchmark_data",
    "PiscesLxToolsPerformanceConfig",
    "PiscesLxToolsPerformanceEvaluator",
    "PiscesLxToolsPerformanceMetrics",
    "PiscesLxToolsPerformanceBenchmarkRunner",
    "create_performance_evaluator",
]
