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

from typing import Any, Optional
from utils import RIGHT, ERROR
from utils.hooks import PiscesLxCoreHookBus

# Reuse the training profiler to avoid duplication
try:
    from tools.train.profiler import PiscesLxToolsProfiler
except Exception:
    import time
    from typing import Dict, Optional
    
    class PiscesLxToolsProfiler:  # fallback with basic timing
        """Fallback profiler with basic timing functionality when training profiler is unavailable."""
        
        def __init__(self):
            self._phase_timers: Dict[str, float] = {}
            self._phase_results: Dict[str, float] = {}
            self._active = False
        
        def start(self, phase_name: str = "infer", **kwargs) -> None:
            """Start profiling for a specific phase with optional metadata."""
            self._active = True
            self._phase_timers[phase_name] = time.perf_counter()
            from utils import DEBUG
            print(f"{DEBUG} Profiler started for phase: {phase_name}")
        
        def stop(self, phase_name: str = "infer", **kwargs) -> Optional[float]:
            """Stop profiling and return elapsed time for the specified phase."""
            if not self._active or phase_name not in self._phase_timers:
                from utils import DEBUG
                print(f"{DEBUG} Profiler stop called for inactive phase: {phase_name}")
                return None
            
            elapsed = time.perf_counter() - self._phase_timers[phase_name]
            self._phase_results[phase_name] = elapsed
            self._active = False
            
            from utils import DEBUG
            print(f"{DEBUG} Profiler stopped for phase: {phase_name}, elapsed: {elapsed:.3f}s")
            return elapsed

try:
    from .config import PiscesLxToolsInferConfig
except Exception:
    class PiscesLxToolsInferConfig:  # minimal facade
        def __init__(self, data: dict):
            self.data = data
        @classmethod
        def from_args(cls, args: Any) -> "PiscesLxToolsInferConfig":
            d = {}
            if getattr(args, 'infer_mode', None):
                d.setdefault('infer', {})
                d['infer']['mode'] = args.infer_mode
            return cls(d)
        def get(self, key: str, default: Optional[Any] = None) -> Any:
            cur = self.data
            for part in key.split('.'):
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur

class PiscesLxToolsInferOrchestrator:
    """Orchestrates inference workflows for Pisces L1.

    Modes (phase-1):
    - standard: native Pisces inference (behavior preserved)
    - vllm: high-performance inference via VLLM (auto-fallback if unavailable)
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        self.hooks = PiscesLxCoreHookBus()
        self.profiler = PiscesLxToolsProfiler()
        self.cfg = PiscesLxToolsInferConfig.from_args(args)

    def run(self, args: Any) -> None:
        mode = self.cfg.get('infer.mode', default=(getattr(args, 'infer_mode', None) or 'standard'))
        RIGHT(f"Infer orchestrator mode: {mode}")
        if mode in ('standard', 'vllm'):
            self.run_standard_infer()
        else:
            ERROR(f"Unknown infer.mode: {mode}")
            raise SystemExit(1)

    def run_standard_infer(self) -> None:
        """Run native/vLLM inference via the class-based runner."""
        from .runner import PiscesLxToolsInferRunner
        runner = PiscesLxToolsInferRunner(self.args, hooks=self.hooks, profiler=self.profiler, cfg=self.cfg)
        runner.infer()
