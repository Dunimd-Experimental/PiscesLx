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

import os
import sys
from types import ModuleType
from utils import RIGHT, ERROR
from . import impl as _impl
from .impl import PiscesLxToolsTrainImpl

class PiscesLxToolsTrainRunner:
    """
    Encapsulates the complete training workflow (legacy-compatible).
    This class provides a wrapper for the legacy training implementation,
    allowing for a smooth transition while maintaining backward compatibility.
    """

    def __init__(self, args, hooks=None, profiler=None, cfg=None):
        """
        Initialize the training runner.

        Args:
            args: Training arguments.
            hooks (optional): Training hooks. Defaults to None.
            profiler (optional): Profiler instance. Defaults to None.
            cfg (optional): Configuration object. Defaults to None.
        """
        self.args = args
        # Instantiate class-based facade for unified style
        self._impl = PiscesLxToolsTrainImpl()
        try:
            self._impl.set_context(hooks=hooks, profiler=profiler, cfg=cfg)
        except Exception:
            # Best-effort context wiring; do not block training
            pass
        # Keep a reference to the module for helper delegations (no behavior change)
        self._impl_module = _impl

    def train(self) -> None:
        """
        Entry point to start training using the legacy implementation for now.
        Validates the training arguments before executing the training implementation.

        Raises:
            Exception: If the training arguments are invalid.
        """
        RIGHT("Starting training via PiscesLxToolsTrainRunner (delegating to legacy _train_impl)")
        # Validate arguments before running the implementation to preserve legacy behavior
        if hasattr(self._impl, "validate_args"):
            try:
                self.args = self._impl.validate_args(self.args)
            except Exception as e:
                ERROR(f"Invalid training arguments: {e}")
                raise
        # Delegate to class facade to run training
        self._impl.train(self.args)

    def setup_distributed_training(self):
        """
        Set up distributed training.
        Delegates the task to the implementation module.

        Returns:
            The result of the implementation's setup_distributed_training function.
        """
        # Delegate to class facade (behavior preserved)
        return self._impl.setup_distributed_training()

    def create_dataloader(self, *a, **kw):
        """
        Create a data loader.
        Delegates the task to the implementation module with the provided arguments.

        Args:
            *a: Positional arguments to pass to the implementation's create_dataloader function.
            **kw: Keyword arguments to pass to the implementation's create_dataloader function.

        Returns:
            The data loader created by the implementation module.
        """
        # Delegate to class facade (behavior preserved)
        return self._impl.create_dataloader(*a, **kw)

    def collate_fn(self, batch):
        """
        Collate a batch of data.
        Delegates the task to the implementation module.

        Args:
            batch: A batch of data to be collated.

        Returns:
            The collated batch of data.
        """
        # Delegate to class facade (behavior preserved)
        return self._impl.collate_fn(batch)
