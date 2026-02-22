#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
import json
from .config import MODALITY_DATASETS
from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from typing import List, Dict, Any, Optional
from .runner import PiscesLxToolsBenchmarkRunner, PiscesLxToolsBenchmark

class PiscesLxToolsBenchmarkOrchestrator:
    """Orchestrator to integrate benchmark with manage.py command.

    It adapts manage.py arguments to the new tools.benchmark package APIs.
    """

    def __init__(self, args):
        self.args = args
        self.logger = PiscesLxLogger("PiscesLx.Tools.Benchmark", file_path=get_log_file("PiscesLx.Tools.Benchmark"), enable_file=True)

    def _parse_generation_config(self) -> Optional[Dict[str, Any]]:
        # Allow JSON via env var PISCES_BENCHMARK_GENERATION_CONFIG
        cfg = os.getenv("PISCES_BENCHMARK_GENERATION_CONFIG", "").strip()
        if not cfg:
            return None
        try:
            return json.loads(cfg)
        except Exception as e:
            self.logger.error("Invalid PISCES_BENCHMARK_GENERATION_CONFIG", error=str(e))
            return None
