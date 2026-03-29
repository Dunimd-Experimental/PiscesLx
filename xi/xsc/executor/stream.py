#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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

"""
Output streaming utilities for Xi Executor.

This module provides output streaming functionality for
real-time command output delivery.
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Optional

from ..core.types import XiLogEntry, XiRunStatus
from ..core.dc import XiLogger


async def stream_output(
    run_id: str,
    process,
    output_queue: asyncio.Queue,
    logger: Optional[XiLogger] = None
) -> None:
    """
    Stream output from process stdout and stderr.
    
    Args:
        run_id: Run identifier
        process: asyncio subprocess
        output_queue: Queue for output entries
        logger: Optional logger instance
    """
    async def read_stream(stream, source: str):
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                
                text = line.decode('utf-8', errors='replace').rstrip()
                if text:
                    entry = XiLogEntry(
                        timestamp=datetime.now().isoformat(),
                        level="info",
                        message=text,
                        source=source,
                        run_id=run_id
                    )
                    if output_queue:
                        await output_queue.put(entry)
        except Exception as e:
            if logger:
                logger.error(
                    f"Error streaming {source}: {e}",
                    event="xi.executor.stream_error"
                )
    
    await asyncio.gather(
        read_stream(process.stdout, "stdout"),
        read_stream(process.stderr, "stderr")
    )


async def get_output_stream(
    run_id: str,
    output_queue: asyncio.Queue,
    active_processes: dict
) -> AsyncGenerator[XiLogEntry, None]:
    """
    Get output stream for a run.
    
    Args:
        run_id: Run identifier
        output_queue: Queue for output entries
        active_processes: Dictionary of active processes
        
    Yields:
        XiLogEntry objects
    """
    while True:
        try:
            entry = await asyncio.wait_for(output_queue.get(), timeout=1.0)
            yield entry
        except asyncio.TimeoutError:
            if run_id not in active_processes:
                break
            continue
