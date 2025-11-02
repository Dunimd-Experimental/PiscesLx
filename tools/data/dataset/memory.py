#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import gc
import torch
import psutil
import threading
from queue import Queue, Empty
from typing import Dict, Optional, List

class PiscesLxToolsDataMemoryMonitor:
    """Monitor memory usage during dataset processing."""

    def __init__(self):
        """Initialize the memory monitor."""
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss
        self.peak_memory = self.initial_memory

    def check_memory(self) -> float:
        """
        Check current memory usage.
        
        Returns:
            float: Current memory usage in MB.
        """
        current_memory = self.process.memory_info().rss
        self.peak_memory = max(self.peak_memory, current_memory)
        return current_memory / 1024 / 1024

    def get_peak_memory(self) -> float:
        """
        Get peak memory usage.
        
        Returns:
            float: Peak memory usage in MB.
        """
        return self.peak_memory / 1024 / 1024

    def get_memory_stats(self) -> dict:
        """
        Get comprehensive memory statistics.
        
        Returns:
            dict: Memory statistics including current and peak usage.
        """
        current_mb = self.check_memory()
        peak_mb = self.get_peak_memory()
        return {
            "current_mb": current_mb,
            "peak_mb": peak_mb,
            "increase_mb": peak_mb - (self.initial_memory / 1024 / 1024)
        }

class PiscesLxToolsDataStreamingDataBuffer:
    """A class for buffering streaming data using a thread-safe queue."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize the streaming data buffer.
        
        Args:
            max_size (int): Maximum number of items to buffer.
        """
        self.max_size = max_size
        self.buffer = Queue(maxsize=max_size)
        self._stop_event = threading.Event()

    def put(self, item: Any) -> bool:
        """
        Put an item into the buffer.
        
        Args:
            item (Any): The item to add to the buffer.
            
        Returns:
            bool: True if the item was added successfully, False if the buffer is full.
        """
        try:
            self.buffer.put(item, block=False)
            return True
        except queue.Full:
            return False

    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Get an item from the buffer.
        
        Args:
            timeout (Optional[float]): Timeout in seconds.
            
        Returns:
            Optional[Any]: The item from the buffer, or None if timeout or stopped.
        """
        try:
            return self.buffer.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        """Stop the buffer."""
        self._stop_event.set()

    def is_stopped(self) -> bool:
        """
        Check if the buffer is stopped.
        
        Returns:
            bool: True if the buffer is stopped.
        """
        return self._stop_event.is_set()

    def size(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            int: Number of items in the buffer.
        """
        return self.buffer.qsize()

    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.
        
        Returns:
            bool: True if the buffer is empty.
        """
        return self.buffer.empty()

    def is_full(self) -> bool:
        """
        Check if the buffer is full.
        
        Returns:
            bool: True if the buffer is full.
        """
        return self.buffer.full()
