#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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

from typing import Any
from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
logger = PiscesLxCoreLog("pisceslx.tools.infer.runner")
from . import impl as _impl
from .impl import PiscesLxToolsInferImpl

class PiscesLxToolsInferRunner:
    """Encapsulates the complete inference workflow (legacy-compatible).

    This runner mirrors the style of the training runner: it delegates to
    the implementation module while providing a place to wire hooks, profiler,
    and configuration context.
    
    Enhanced with utils device management, caching, and observability.
    """

    def _init_utils_components(self, cache_manager=None, device_manager=None, observability=None):
        """Initialize utils-enhanced components for inference optimization."""
        # Initialize device facade for device management
        self.device_facade = device_manager or PiscesLxCoreDeviceFacade(self.args)
        
        # Initialize cache manager for model caching
        self.cache_manager = cache_manager
        
        # Initialize observability for performance monitoring
        self.observability = observability
        
        logger.info("PiscesLxToolsInferRunner utils components initialized with device management")

    def __init__(self, args: Any, hooks=None, profiler=None, cfg=None, cache_manager=None, device_manager=None, observability=None) -> None:
        """Initialize the inference runner with enhanced device management.

        Args:
            args: Parsed CLI arguments (e.g., argparse.Namespace)
            hooks: Hook bus instance for lifecycle events
            profiler: Profiler instance
            cfg: Inference configuration facade (optional)
            cache_manager: Cache manager instance (optional)
            device_manager: Device manager instance (optional)
            observability: Observability facade (optional)
        """
        self.args = args
        # Initialize utils-enhanced components
        self._init_utils_components(cache_manager, device_manager, observability)
        
        # Instantiate class-based facade for unified style
        self._impl = PiscesLxToolsInferImpl()
        try:
            self._impl.set_context(hooks=hooks, profiler=profiler, cfg=cfg)
        except Exception:
            # Best-effort context wiring; do not block inference
            pass
        # Keep a reference to the module for compatibility (not used in primary path)
        self._impl_module = _impl

    def infer(self) -> None:
        """Start inference using the implementation module."""
        logger.success("Starting inference via PiscesLxToolsInferRunner")
        try:
            # Delegate to class facade to run inference
            self._impl.infer(self.args)
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
