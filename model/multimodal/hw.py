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

"""Hardware detection utilities for Ruchbah multimodal agents.

The helper classes in this module infer hardware capabilities such as device
type, gradient capacities, and checkpoint segmentation hints. They expose
configuration dictionaries that downstream components can consult to adapt
their workloads.
"""

from typing import Dict

class RuchbahHardwareAdaptiveConfig:
    """Hardware adapter that derives configuration tiers from detected devices.

    Attributes:
        device_info (Dict[str, Dict]): Raw device detection results including tier
            recommendations.
        adaptive_config (Dict[str, Dict]): Profiled configuration generated from
            the recommended tier.
    """

    def __init__(self):
        """Detect hardware information and build the adaptive configuration."""
        self.device_info = self._detect_hardware()
        self.adaptive_config = self._profile(self.device_info["recommended_config"])

    def _detect_hardware(self) -> Dict:
        """Detect the available CUDA hardware and determine the recommended configuration tier.

        Tries to detect the number of CUDA devices and their total memory. Based on these values,
        it assigns a recommended configuration tier. If any exception occurs, it defaults to 
        minimal configuration.

        Returns:
            Dict: A dictionary containing the device count, total memory in GB, and the recommended 
                  configuration tier.
        """
        try:
            import torch
            # Get the number of available CUDA devices, default to 0 if not available
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            total_mem = 0
            if device_count > 0:
                # Calculate the total memory of all CUDA devices in GB
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    total_mem += props.total_memory // (1024**3)
            # Determine the recommended configuration tier based on device count and total memory
            if device_count >= 4 and total_mem >= 320:
                tier = "maximum"
            elif device_count >= 2 and total_mem >= 80:
                tier = "high" if total_mem >= 160 else "medium"
            elif device_count >= 1 and total_mem >= 40:
                tier = "conservative"
            else:
                tier = "minimal"
            return {
                "device_count": device_count,
                "total_memory_gb": total_mem,
                "recommended_config": tier,
            }
        except Exception:
            # Return minimal configuration if hardware detection fails
            return {"device_count": 0, "total_memory_gb": 0, "recommended_config": "minimal"}

    def _profile(self, tier: str) -> Dict:
        """
        Get the adaptive configuration based on the given tier.

        Args:
            tier (str): The configuration tier, e.g., "minimal", "conservative", etc.

        Returns:
            Dict: A dictionary containing the adaptive configuration parameters. If the tier is 
                  not found, it defaults to the "minimal" configuration.
        """
        profiles = {
            "minimal":       {"max_layers": 2, "max_heads": 4,  "max_lstm_layers": 1, "gradient_clip_norm": 0.5, "use_mixed_precision": True,  "checkpoint_segments": 4},
            "conservative":  {"max_layers": 3, "max_heads": 8,  "max_lstm_layers": 1, "gradient_clip_norm": 1.0, "use_mixed_precision": True,  "checkpoint_segments": 2},
            "medium":        {"max_layers": 4, "max_heads": 12, "max_lstm_layers": 2, "gradient_clip_norm": 1.5, "use_mixed_precision": False, "checkpoint_segments": 1},
            "high":          {"max_layers": 6, "max_heads": 16, "max_lstm_layers": 3, "gradient_clip_norm": 2.0, "use_mixed_precision": False, "checkpoint_segments": 1},
            "maximum":       {"max_layers": 8, "max_heads": 24, "max_lstm_layers": 4, "gradient_clip_norm": 3.0, "use_mixed_precision": False, "checkpoint_segments": 1},
        }
        return profiles.get(tier, profiles["minimal"])

    def get_gradient_config(self) -> Dict:
        """
        Get the gradient-related configuration parameters.

        Returns:
            Dict: A dictionary containing gradient-related configuration parameters including 
                  max_grad_norm, use_mixed_precision, and checkpoint_segments.
        """
        return {
            "max_grad_norm": self.adaptive_config["gradient_clip_norm"],
            "use_mixed_precision": self.adaptive_config["use_mixed_precision"],
            "checkpoint_segments": self.adaptive_config["checkpoint_segments"],
        }
