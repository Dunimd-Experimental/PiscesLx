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

import gc
import torch
import psutil
import threading
from queue import Queue, Empty
from typing import Dict, Optional, List

class MemoryMonitor:
    """A class for monitoring memory usage of the current process, system, and GPUs, 
    and performing memory cleanup operations when necessary."""

    def __init__(self, threshold_gb: float = 8.0):
        """Initialize the MemoryMonitor instance.

        Args:
            threshold_gb (float, optional): Memory threshold in gigabytes. 
                Garbage collection will be triggered if the process memory exceeds this value. Defaults to 8.0.
        """
        self.threshold_gb = threshold_gb
        self.alerts = 0

    def check_memory(self) -> Dict[str, float]:
        """Check the memory usage of the current process, system, and GPUs.

        Returns:
            Dict[str, float]: A dictionary containing memory usage information:
                - "process_memory_gb": Resident set size of the process in gigabytes.
                - "system_available_gb": Available system memory in gigabytes.
                - "system_used_percent": Percentage of used system memory.
                - "gpu_memory": A dictionary with memory information for each available GPU, 
                                including allocated, cached, and total memory.
        """
        # Get the current process
        process = psutil.Process()
        # Get the memory information of the current process
        memory_info = process.memory_info()
        # Get the system virtual memory information
        system_memory = psutil.virtual_memory()
        gpu_memory = {}
        try:
            if torch.cuda.is_available():
                # Iterate over all available GPUs
                for i in range(torch.cuda.device_count()):
                    gpu_memory[f"gpu_{i}"] = {
                        "allocated": torch.cuda.memory_allocated(i) / 1024**3,
                        "cached": torch.cuda.memory_reserved(i) / 1024**3,
                        "total": torch.cuda.get_device_properties(i).total_memory / 1024**3,
                    }
        except Exception:
            # Silently handle exceptions when getting GPU memory information
            pass
        return {
            "process_memory_gb": memory_info.rss / 1024**3,
            "system_available_gb": system_memory.available / 1024**3,
            "system_used_percent": system_memory.percent,
            "gpu_memory": gpu_memory,
        }

    def should_gc(self) -> bool:
        """Determine if garbage collection should be performed based on the process memory usage.

        Increments the alert counter if the process memory exceeds the predefined threshold.

        Returns:
            bool: True if the process memory exceeds the threshold, False otherwise.
        """
        mem = self.check_memory()
        if mem["process_memory_gb"] > self.threshold_gb:
            self.alerts += 1
            return True
        return False

    def cleanup(self):
        """Perform garbage collection and clean up the GPU cache if available.

        Silently handles exceptions that occur during the cleanup process.
        """
        try:
            # Perform garbage collection
            gc.collect()
            if torch.cuda.is_available():
                # Empty the GPU cache
                torch.cuda.empty_cache()
                # Synchronize the current stream on the current device
                torch.cuda.synchronize()
        except Exception:
            # Silently handle exceptions during cleanup
            pass

class StreamingDataBuffer:
    """A class for buffering streaming data using a thread-safe queue."""

    def __init__(self, buffer_size: int = 1000):
        """Initialize the StreamingDataBuffer instance.

        Args:
            buffer_size (int, optional): Maximum size of the buffer queue. Defaults to 1000.
        """
        self.buffer: "Queue[List[dict]]" = Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()

    def add_batch(self, batch_data: "List[dict]"):
        """Add a batch of data to the buffer if the stop event is not set.

        Args:
            batch_data (List[dict]): A batch of data to be added to the buffer.
        """
        if self._stop_event.is_set():
            return
        try:
            # Put the batch data into the buffer with a timeout of 1 second
            self.buffer.put(batch_data, timeout=1.0)
        except Exception:
            # Silently handle exceptions when adding data to the buffer
            pass

    def get_batch(self, timeout: float = 5.0) -> Optional["List[dict]"]:
        """Retrieve a batch of data from the buffer with a specified timeout.

        Args:
            timeout (float, optional): Maximum time in seconds to wait for data. Defaults to 5.0.

        Returns:
            Optional[List[dict]]: A batch of data if available within the timeout, None otherwise.
        """
        try:
            return self.buffer.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        """Set the stop event to prevent further data from being added to the buffer."""
        self._stop_event.set()