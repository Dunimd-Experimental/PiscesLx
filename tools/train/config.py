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

import json
from typing import Any, Optional
from dataclasses import dataclass
from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
logger = PiscesLxCoreLog("pisceslx.tools.train")

@dataclass
class PiscesLxToolsTrainConfig:
    """Lightweight typed view for accessing train-time configuration.

    It merges CLI args with optional JSON config files, preferring CLI.
    Uses utils.config.manager.PiscesLxCoreConfigManager for enhanced configuration management.
    """

    data: dict
    _config_manager: PiscesLxCoreConfigManager = None

    def __post_init__(self):
        """Initialize config manager after dataclass creation."""
        if self._config_manager is None:
            self._config_manager = PiscesLxCoreConfigManager(self.data)

    @classmethod
    def from_args(cls, args: Any) -> "PiscesLxToolsTrainConfig":
        """Creates a PiscesLxToolsTrainConfig instance from command-line arguments.

        Args:
            args: Command-line arguments object, typically an argparse.Namespace.

        Returns:
            A new PiscesLxToolsTrainConfig instance populated with configuration data.
        """
        cfg_dict = {}
        # Load external JSON configuration file if the 'config' argument is provided and is a JSON file
        path = getattr(args, "config", None)
        if path and path.endswith(".json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg_dict.update(json.load(f))
            except Exception:
                pass

        # Overlay CLI arguments related to quantization into the configuration dictionary
        if getattr(args, "ckpt", None):
            cfg_dict.setdefault("quant", {})
            cfg_dict["quant"]["ckpt"] = args.ckpt
        if getattr(args, "save", None):
            cfg_dict.setdefault("quant", {})
            cfg_dict["quant"]["save"] = args.save
        if getattr(args, "quant_bits", None) is not None:
            cfg_dict.setdefault("quant", {})
            cfg_dict["quant"]["bits"] = args.quant_bits

        # Infer the training mode based on legacy command-line flags to maintain user experience
        if bool(getattr(args, "rlhf", False)):
            cfg_dict.setdefault("train", {})
            cfg_dict["train"]["mode"] = "preference"
        else:
            if bool(getattr(args, "quant", False)) or (
                bool(getattr(args, "ckpt", None)) and bool(getattr(args, "save", None))
            ):
                cfg_dict.setdefault("train", {})
                cfg_dict["train"]["mode"] = "quant_export"

        return cls(cfg_dict)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieves a value from the configuration using a dot-separated path.

        Args:
            key: Dot-separated path to the configuration value (e.g., "train.mode").
            default: Value to return if the key is not found. Defaults to None.

        Returns:
            The configuration value at the specified path, or the default value if not found.
        """
        # Use config manager for enhanced path-based retrieval
        if self._config_manager:
            return self._config_manager.get(key, default)
        
        # Fallback to manual path traversal
        cur = self.data
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def dump_effective(self) -> dict:
        """Returns a copy of the effective configuration data.

        Returns:
            A dictionary containing the effective configuration data.
        """
        return dict(self.data)
    
    def validate(self) -> bool:
        """Validate configuration using utils config manager.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        if self._config_manager:
            return self._config_manager.validate()
        return True
