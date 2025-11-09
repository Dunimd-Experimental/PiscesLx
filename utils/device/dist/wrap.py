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

import torch
from typing import Any
import torch.distributed as dist
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("PiscesLx.Core.Device.Dist.Wrap")

class PiscesLxCoreModelParallelizer:
    """A utility class for wrapping models with parallelization for training and inference."""

    @staticmethod
    def wrap_for_train(model: Any, plan: Any) -> Any:
        """Wrap the model for training with appropriate parallelization.

        If torch is not available, the original model is returned. 
        The function first checks if parallelization is disabled. If so, the model is moved to the specified device.
        It prefers using DistributedDataParallel when distributed training is initialized.
        If that fails, it falls back to DataParallel for multi-GPU single-process training.

        Args:
            model (Any): The model to be wrapped.
            plan (Any): The plan containing device and parallelization settings, 
                        which may include 'disable_parallel', 'device', and 'local_rank' attributes.

        Returns:
            Any: The wrapped model or the original model if no wrapping is applied.
        """
        if torch is None:
            return model
        
        # Check if parallelization is explicitly disabled
        if getattr(plan, 'disable_parallel', False):
            device = plan.device if isinstance(plan.device, str) else str(plan.device)
            dev = torch.device(device)
            return model.to(dev)

        # Move model to the target device
        device = plan.device if isinstance(plan.device, str) else str(plan.device)
        dev = torch.device(device)
        model = model.to(dev)

        # Attempt to use DistributedDataParallel if distributed training is initialized
        try:
            if dist is not None and dist.is_available() and dist.is_initialized():
                local_rank = int(getattr(plan, 'local_rank', 0))
                kwargs = {
                    'find_unused_parameters': True,
                    'broadcast_buffers': False,
                }
                if dev.type == 'cuda':
                    kwargs.update({
                        'device_ids': [local_rank],
                        'output_device': local_rank,
                    })
                model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
                return model
        except Exception as e:
            # Log failure of DistributedDataParallel and proceed to fallback
            if hasattr(model, 'module') or hasattr(model, 'device'):
                logger.debug(
                    "DistributedDataParallel failed, falling back to DataParallel: %s", 
                    e, 
                    exc_info=True
                )

        # Fallback to DataParallel for multi-GPU setups
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and dev.type == 'cuda':
            model = torch.nn.DataParallel(model)
        return model

    @staticmethod
    def wrap_for_infer(model: Any, plan: Any) -> Any:
        """Wrap the model for inference with appropriate parallelization.

        If torch is not available, the original model is returned.
        The function first checks if parallelization is disabled. If so, the model is moved to the specified device.
        It applies DataParallel for single-process multi-GPU inference.

        Args:
            model (Any): The model to be wrapped.
            plan (Any): The plan containing device and parallelization settings,
                        which may include 'disable_parallel' and 'device' attributes.

        Returns:
            Any: The wrapped model or the original model if no wrapping is applied.
        """
        if torch is None:
            return model
        
        # Check if parallelization is explicitly disabled
        if getattr(plan, 'disable_parallel', False):
            device = plan.device if isinstance(plan.device, str) else str(plan.device)
            dev = torch.device(device)
            return model.to(dev)

        # Move model to the target device
        device = plan.device if isinstance(plan.device, str) else str(plan.device)
        dev = torch.device(device)
        model = model.to(dev)

        # Apply DataParallel for multi-GPU inference
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and dev.type == 'cuda':
            model = torch.nn.DataParallel(model)
        return model
