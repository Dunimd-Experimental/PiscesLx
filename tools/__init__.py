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

import importlib

_LAZY_EXPORTS = {
    # Benchmark
    'PiscesLxToolsBenchmark': 'tools.benchmark.runner',
    # Data download
    'PiscesLxToolsDataDatasetDownload': 'tools.data.download.runner',
    'PiscesLxToolsDataSourceRouter': 'tools.data.download.sources',
    'PiscesLxToolsDataDownloadConfig': 'tools.data.download.config',
    'PiscesLxToolsDataDownloadCache': 'tools.data.download.caches',
    'PiscesLxToolsDataConfigLoader': 'tools.data.download.config',
    'DatasetItem': 'tools.data.download.config',
    # Infer
    'PiscesLxInferOrchestrator': 'tools.infer.orchestrator',
    'PiscesLxToolsInferRunner': 'tools.infer.runner',
    'PiscesLxToolsInferConfig': 'tools.infer.config',
    'PiscesLxToolsInferImpl': 'tools.infer.impl',
    # Train
    'PiscesLxTrainOrchestrator': 'tools.train.orchestrator',
    'PiscesLxToolsTrainRunner': 'tools.train.runner',
    'PiscesLxToolsTrainConfig': 'tools.train.config',
    'PiscesLxToolsTrainImpl': 'tools.train.impl',
    'PiscesLxToolsProfiler': 'tools.train.profiler',
    'PiscesLxToolsQuantExporter': 'tools.train.quant_export',
    'PiscesLxToolsPreferenceTrainer': 'tools.train.pref_align',
    # Monitor
    'PiscesLxToolsMonitorOrchestrator': 'tools.monitor.orchestrator',
    'PiscesLxToolsMonitorRunner': 'tools.monitor.runner',
    'PiscesLxToolsMonitorImpl': 'tools.monitor.impl',
    'PiscesLxToolsMonitorConfig': 'tools.monitor.config',
}

__all__ = list(_LAZY_EXPORTS.keys())

def __getattr__(name):
    modpath = _LAZY_EXPORTS.get(name)
    if not modpath:
        raise AttributeError(f"module 'tools' has no attribute '{name}'")
    mod = importlib.import_module(modpath)
    try:
        return getattr(mod, name)
    except AttributeError:
        # Some modules export classes under different names; try best-effort fallbacks
        raise
