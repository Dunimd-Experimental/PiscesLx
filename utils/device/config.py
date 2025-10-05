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

from typing import Any, Optional

class PiscesLxCoreDeviceConfig:
    """A lightweight facade for device configuration, maintaining parity with inference and training configurations.

    Provides unified management for device setup, GPU selection, memory optimization, 
    and distributed training settings.
    """

    def __init__(self, data: dict = None) -> None:
        """Initialize the configuration object with default or provided data.

        Args:
            data (dict, optional): Configuration data dictionary. Defaults to None.
        """
        if data is None:
            data = {
                'device': {
                    'type': 'auto',
                    'gpu_ids': None,
                    'memory_fraction': 0.8,
                },
                'training': {
                    'batch_size': 'auto',
                    'mixed_precision': True,
                    'gradient_accumulation_steps': 1,
                },
                'distributed': {
                    'enabled': False,
                    'world_size': 1,
                    'rank': 0,
                    'local_rank': 0,
                    'master_addr': 'localhost',
                    'master_port': '29500',
                    'backend': 'nccl',
                    'init_method': 'env://',
                }
            }
        self.data = data

    @classmethod
    def from_args(cls, args: Any) -> "PiscesLxCoreDeviceConfig":
        """Create a configuration object from command-line arguments.

        Args:
            args (Any): Command line arguments object.

        Returns:
            PiscesLxCoreDeviceConfig: New instance of the configuration object.
        """
        d: dict = {}

        # Extract device-related arguments
        if getattr(args, "device", None):
            d.setdefault("device", {})
            d["device"]["type"] = args.device

        if getattr(args, "gpu_ids", None):
            d.setdefault("device", {})
            d["device"]["gpu_ids"] = args.gpu_ids

        if getattr(args, "memory_efficient", None) is not None:
            d.setdefault("device", {})
            d["device"]["memory_efficient"] = args.memory_efficient

        # Extract training-related arguments
        if getattr(args, "batch_size", None):
            d.setdefault("training", {})
            d["training"]["batch_size"] = args.batch_size

        # Extract distributed training-related arguments
        if getattr(args, "distributed", None) is not None:
            d.setdefault("distributed", {})
            d["distributed"]["enabled"] = args.distributed

        if getattr(args, "world_size", None):
            d.setdefault("distributed", {})
            d["distributed"]["world_size"] = args.world_size

        if getattr(args, "rank", None) is not None:
            d.setdefault("distributed", {})
            d["distributed"]["rank"] = args.rank

        return cls(d)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieve value using dot-separated key path from configuration data.

        Args:
            key (str): Dot-separated key path to retrieve.
            default (Optional[Any], optional): Default value if key is not found. Defaults to None.

        Returns:
            Any: Value at the specified key path or default value.
        """
        cur: Any = self.data
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot-separated key notation.

        Args:
            key (str): Configuration key using dot notation (e.g., 'device.type').
            value (Any): Value to assign to the key.
        """
        keys = key.split(".")
        target = self.data

        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        # Set the final key-value pair
        target[keys[-1]] = value

    def dump_effective(self) -> dict:
        """Return a shallow copy of the current configuration data.

        Returns:
            dict: Copy of the configuration data.
        """
        return dict(self.data)

    def get_device_config(self) -> dict:
        """Get device-specific configuration parameters.

        Returns:
            dict: Dictionary containing device type, GPU IDs, and memory settings.
        """
        return {
            "type": self.get("device.type", "auto"),
            "gpu_ids": self.get("device.gpu_ids", None),
            "memory_efficient": self.get("device.memory_efficient", True),
        }

    def get_distributed_config(self) -> dict:
        """Get distributed training configuration parameters.

        Returns:
            dict: Dictionary containing distributed training settings.
        """
        return {
            "enabled": self.get("distributed.enabled", False),
            "world_size": self.get("distributed.world_size", 1),
            "rank": self.get("distributed.rank", 0),
        }

    def get_training_config(self) -> dict:
        """Get training-specific configuration parameters.

        Returns:
            dict: Dictionary containing training settings like batch size and precision.
        """
        return {
            "batch_size": self.get("training.batch_size", None),
            "mixed_precision": self.get("training.mixed_precision", True),
        }