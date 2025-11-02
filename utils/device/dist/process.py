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

import os
import time
import torch
from typing import Optional
import torch.distributed as dist
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("PiscesLx.Utils.Device.Dist.Process")

class PiscesLxCoreProcessGroupManager:
    """Manages the process group for distributed training in PiscesLx Core."""
    _initialized: bool = False
    _backend_used: Optional[str] = None

    @classmethod
    def init(cls, backend: Optional[str] = None, timeout_seconds: int = 1800) -> None:
        """Initialize the distributed process group.

        Args:
            backend (Optional[str]): Distributed backend to use. If None, defaults to 'nccl' 
                if CUDA is available, otherwise 'gloo'.
            timeout_seconds (int): Timeout in seconds for process group initialization. Defaults to 1800.

        Returns:
            None
        """
        if cls._initialized:
            return
        
        if dist is None or torch is None:
            cls._initialized = False
            return

        # Retrieve world size and local rank from environment
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        
        # Skip initialization in single-process scenarios
        if world_size <= 1 or local_rank < 0:
            cls._initialized = False
            return

        # Set default master address and port for distributed training
        os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))

        # Auto-select backend if not specified
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        # Verify distributed module availability
        if not dist.is_available():
            cls._initialized = False
            return

        # Early return if process group is already active
        if dist.is_initialized():
            cls._initialized = True
            return

        # Configure initialization parameters
        init_kwargs = {"backend": backend}
        
        # Add timeout if timedelta is supported
        try:
            from datetime import timedelta as _dt_timedelta
            init_kwargs["timeout"] = _dt_timedelta(seconds=timeout_seconds)
        except Exception:
            try:
                if hasattr(torch, 'timedelta'):
                    init_kwargs["timeout"] = torch.timedelta(seconds=timeout_seconds)  # type: ignore[attr-defined]
            except Exception as e:
                logger.debug("Timedelta initialization failed, continuing without timeout: %s", e)

        # Attempt initialization with primary backend and fallback to 'gloo'
        backends_to_try = [backend]
        if backend != "gloo":
            backends_to_try.append("gloo")

        last_err: Optional[BaseException] = None
        for be in backends_to_try:
            try:
                init_kwargs["backend"] = be
                dist.init_process_group(**init_kwargs)
                cls._backend_used = be
                # Configure CUDA device if available
                if torch.cuda.is_available():
                    try:
                        device_count = torch.cuda.device_count()
                        if 0 <= local_rank < device_count:
                            torch.cuda.set_device(local_rank)
                    except Exception as e:
                        logger.debug("Failed to set CUDA device for local_rank %s: %s", local_rank, e)
                cls._initialized = True
                break
            except BaseException as e:
                last_err = e
                time.sleep(0.5)
                continue

        if not cls._initialized and last_err is not None:
            cls._backend_used = None
            cls._initialized = False

    @classmethod
    def finalize(cls) -> None:
        """Shut down the distributed process group and reset internal state.

        Returns:
            None
        """
        if dist is not None and getattr(dist, "is_initialized", lambda: False)():
            try:
                dist.destroy_process_group()
            except Exception as e:
                logger.debug("Failed to destroy process group: %s", e)
        cls._initialized = False
        cls._backend_used = None
