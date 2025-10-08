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

from typing import Any
from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
logger = PiscesLxCoreLog("pisceslx.data.download")
from . import impl as _impl
from .impl import PiscesLxToolsMonitorImpl

class PiscesLxToolsMonitorRunner:
    """Encapsulates the complete monitoring workflow (legacy-compatible).

    This runner mirrors the style of the training runner: it delegates to
    the implementation module while providing a place to wire hooks, profiler,
    and configuration context.
    """

    def __init__(self, args: Any, hooks=None, profiler=None, cfg=None) -> None:
        """Initialize the monitoring runner.

        Args:
            args: Parsed CLI arguments (e.g., argparse.Namespace)
            hooks: Hook bus instance for lifecycle events
            profiler: Profiler instance
            cfg: Monitoring configuration facade (optional)
        """
        self.args = args
        # Instantiate class-based facade for unified style
        self._impl = PiscesLxToolsMonitorImpl()
        try:
            self._impl.set_context(hooks=hooks, profiler=profiler, cfg=cfg)
        except Exception:
            # Best-effort context wiring; do not block monitoring
            pass
        # Keep a reference to the module for compatibility (not used in primary path)
        self._impl_module = _impl

    def monitor(self) -> None:
        """Start monitoring using the implementation module."""
        logger.success("Starting monitoring via PiscesLxToolsMonitorRunner")
        try:
            # Delegate to class facade to run monitoring
            self._impl.monitor(self.args)
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            raise