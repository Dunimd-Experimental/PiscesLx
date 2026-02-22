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

import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


class PiscesLxMonitorDataManager:
    """Data manager for monitoring data with simple buffering."""

    def __init__(self, cache_manager=None, logger=None, log_interval: int = 60, buffer_size: int = 60):
        """Initialize the data manager."""
        self.cache_manager = cache_manager
        self.logger = logger
        self.log_interval = log_interval
        self.buffer_size = buffer_size
        self.data_buffer = []
        self.last_recorded_averages = {}
        self.last_log_time = time.time()

    def add_sample(self, data: Dict[str, Any]) -> None:
        """Add a data sample to the buffer."""
        self.data_buffer.append({
            "timestamp": time.time(),
            "data": data
        })
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)

    def add(self, data: Dict[str, Any]) -> None:
        """Add a data sample - alias for add_sample."""
        self.add_sample(data)

    def get_recent_samples(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent samples from buffer."""
        return self.data_buffer[-count:]

    def calculate_averages(self) -> Dict[str, Any]:
        """Calculate averages from buffered data."""
        if not self.data_buffer:
            return {}
        return {}

    def clear_buffer(self) -> None:
        """Clear the data buffer."""
        self.data_buffer.clear()
