#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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
from typing import Dict

class PiscesLxToolsProfiler:
    """Minimal phase profiler to record timing and simple counters.

    Designed to be lightweight; can be extended later without breaking API.
    """

    def __init__(self):
        # Dictionary storing the start time of each phase. Key is phase name, value is start timestamp.
        self._phases: Dict[str, float] = {}
        # Dictionary storing the accumulated elapsed time of each phase. Key is phase name, value is total elapsed time.
        self._elapsed: Dict[str, float] = {}

    def start_phase(self, name: str) -> None:
        """Start recording the time for a specific phase.

        Records the current time as the start time for the given phase name.

        Args:
            name (str): The name of the phase to start timing.
        """
        self._phases[name] = time.perf_counter()

    def end_phase(self, name: str) -> None:
        """End the timing for a specific phase and accumulate the elapsed time.

        Retrieves the start time of the given phase, calculates the elapsed time,
        and adds it to the total elapsed time for that phase.

        Args:
            name (str): The name of the phase to end timing.
        """
        st = self._phases.pop(name, None)
        if st is not None:
            self._elapsed[name] = self._elapsed.get(name, 0.0) + (time.perf_counter() - st)

    def snapshot_metrics(self) -> Dict[str, float]:
        """Get a snapshot of the accumulated elapsed times for all phases.

        Returns a shallow copy of the internal elapsed time dictionary to prevent
        external modification of the original data.

        Returns:
            Dict[str, float]: A dictionary containing the phase names as keys and
                            their accumulated elapsed times as values.
        """
        # Return a shallow copy to avoid external mutation
        return dict(self._elapsed)
