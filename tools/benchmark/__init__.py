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
PiscesLx Benchmark Module - Flagship-level evaluation suite.

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
from .benchmark import (
    PiscesLxBenchmarkSuite,
    PiscesLxBenchmarkConfig,
    PiscesLxBenchmarkEvaluator,
)

from .multimodal import (
    PiscesLxToolsMultimodalBenchmarkRunner,
    PiscesLxToolsMultimodalConfig,
    PiscesLxToolsVisionEvaluator,
    PiscesLxToolsAudioEvaluator,
    PiscesLxToolsVideoEvaluator,
)

from .agent import (
    PiscesLxToolsAgentBenchmarkRunner,
    PiscesLxToolsAgentConfig,
    PiscesLxToolsAgentEvaluator,
    PiscesLxToolsAgentMetrics,
)

from .long_context import (
    PiscesLxToolsLongContextBenchmarkRunner,
    PiscesLxToolsLongContextConfig,
    PiscesLxToolsLongContextEvaluator,
    PiscesLxToolsLongContextMetrics,
)

from .visualization import (
    PiscesLxToolsReportGenerator,
    PiscesLxToolsBenchmarkVisualizer,
    PiscesLxToolsVisualizationConfig,
)

# Visual comparison

from .comparison import (
    PiscesLxToolsFlagshipComparator,
    PiscesLxToolsComparisonConfig,
    FLAGSHIP_MODELS,
)

# Standard benchmark interfaces

from .benchmark import (
    PiscesLxBenchmarkResult,
    PiscesLxBaseBenchmark,
    PiscesLxMMLUBenchmark,
    PiscesLxHumanEvalBenchmark,
    PiscesLxAIMEBenchmark,
    PiscesLxSWEBenchBenchmark,
    PiscesLxBenchmarkSuiteConfig,
    PiscesLxBenchmarkSuite,
)

from .performance import (
    PiscesLxToolsPerformanceBenchmarkRunner,
    PiscesLxToolsPerformanceConfig,
    PiscesLxToolsPerformanceEvaluator,
    PiscesLxToolsPerformanceMetrics,
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
    "PiscesLxBenchmarkConfig",
    "PiscesLxBenchmarkEvaluator",
    "PiscesLxToolsMultimodalConfig",
    "PiscesLxToolsVisionEvaluator",
    "PiscesLxToolsAudioEvaluator",
    "PiscesLxToolsVideoEvaluator",
    "PiscesLxToolsMultimodalBenchmarkRunner",
    "PiscesLxToolsAgentConfig",
    "PiscesLxToolsAgentEvaluator",
    "PiscesLxToolsAgentMetrics",
    "PiscesLxToolsAgentBenchmarkRunner",
    "PiscesLxToolsLongContextConfig",
    "PiscesLxToolsLongContextEvaluator",
    "PiscesLxToolsLongContextMetrics",
    "PiscesLxToolsLongContextBenchmarkRunner",
    "PiscesLxToolsVisualizationConfig",
    "PiscesLxToolsBenchmarkVisualizer",
    "PiscesLxToolsReportGenerator",
    "PiscesLxToolsComparisonConfig",
    "PiscesLxToolsFlagshipComparator",
    "FLAGSHIP_MODELS",
    "PiscesLxToolsPerformanceConfig",
    "PiscesLxToolsPerformanceEvaluator",
    "PiscesLxToolsPerformanceMetrics",
    "PiscesLxToolsPerformanceBenchmarkRunner",
    "PiscesLxBenchmarkResult",
    "PiscesLxBaseBenchmark",
    "PiscesLxMMLUBenchmark",
    "PiscesLxHumanEvalBenchmark",
    "PiscesLxAIMEBenchmark",
    "PiscesLxSWEBenchBenchmark",
    "PiscesLxBenchmarkSuiteConfig",
    "PiscesLxBenchmarkSuite",
]
