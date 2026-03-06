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

import os
import time
import psutil
import platform
from datetime import datetime
from typing import Dict, Any, Optional, List

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from .config import PiscesLxToolsMonitorConfig
from .context_utils import PiscesLxMonitorCache, PiscesLxMonitorContextManager
from .stats_collector import PiscesLxMonitorStatsCollector
from .data_manager import PiscesLxMonitorDataManager
from .alert_manager import PiscesLxMonitorAlertManager
from .display_utils import PiscesLxToolsMonitorDisplay

_LOG = PiscesLxLogger("PiscesLx.Tools.Monitor", file_path=get_log_file("PiscesLx.Tools.Monitor"), enable_file=True)


class PiscesLxToolsMonitorOrchestrator:
    """Main orchestrator for system monitoring."""
    
    def __init__(self, args=None):
        self.config = PiscesLxToolsMonitorConfig(args)
        self.cache = PiscesLxMonitorCache()
        self.collector = PiscesLxMonitorStatsCollector()
        self.data_manager = PiscesLxMonitorDataManager(self.config.buffer_size)
        self.alert_manager = PiscesLxMonitorAlertManager(self.config, self.cache)
        self.display = PiscesLxToolsMonitorDisplay()
    
    def run(self, args):
        _LOG.info("Starting PiscesLx System Monitor...")
        
        if not self.collector.gpu_enabled:
            _LOG.error("NVIDIA GPU not detected or pynvml not available")
        
        time.sleep(2)
        
        last_net_io = psutil.net_io_counters()
        last_disk_io = psutil.disk_io_counters()
        last_time = time.time()
        
        psutil.cpu_percent(percpu=True)
        psutil.cpu_percent(percpu=False)
        
        try:
            while True:
                stats = self.collector.collect()
                self.data_manager.add(stats)
                
                alerts = self.alert_manager.check(stats)
                
                last_net_io, last_disk_io = self.display.render(
                    stats, last_net_io, last_disk_io, last_time, alerts
                )
                last_time = time.time()
                
                time.sleep(self.config.update_interval)
        except KeyboardInterrupt:
            _LOG.info("Monitor stopped by user")
        except Exception as e:
            _LOG.error(f"Monitor error: {e}")
        finally:
            self.collector.shutdown()
