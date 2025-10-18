#!/usr/bin/env/python3

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

from typing import Any, Optional

class PiscesLxToolsInferConfig:
    """Lightweight facade for inference configuration (parity with train config).

    For now it normalizes CLI inputs into a namespaced dictionary and exposes
    key inference optimization flags (attention backend, paged KV, speculative).
    Later, we can extend it to merge configs from JSON (inference_config).
    """

    def __init__(self, data: dict) -> None:
        """Initialize the configuration object.

        Args:
            data (dict): The configuration data.
        """
        self.data = data

    @classmethod
    def from_args(cls, args: Any) -> "PiscesLxToolsInferConfig":
        """Create a config object from CLI args (no JSON merge yet)."""
        d: dict = {}
        d.setdefault("infer", {})
        # mode
        if getattr(args, "infer_mode", None):
            d["infer"]["mode"] = args.infer_mode
        # attention backend flags
        d["infer"]["enable_sdpa"] = bool(getattr(args, "enable_sdpa", True))
        d["infer"]["enable_flash_attention"] = bool(getattr(args, "enable_flash_attention", True))
        # paged KV
        d["infer"]["enable_paged_kv"] = bool(getattr(args, "enable_paged_kv", False))
        d["infer"]["kv_page_size"] = int(getattr(args, "kv_page_size", 512))
        d["infer"]["kv_soft_cap_factor"] = float(getattr(args, "kv_soft_cap_factor", 1.5))
        # speculative decoding
        d["infer"]["speculative_enable"] = bool(getattr(args, "speculative", False))
        d["infer"]["spec_draft_length"] = int(getattr(args, "spec_draft_length", getattr(args, "spec_gamma", 4)))
        d["infer"]["spec_num_candidates"] = int(getattr(args, "spec_num_candidates", 4))
        return cls(d)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Perform dot-path retrieval from the underlying config dict.

        Args:
            key (str): The dot-separated key path to retrieve.
            default (Optional[Any], optional): The default value to return if the key is not found. Defaults to None.

        Returns:
            Any: The value at the specified key path, or the default value if not found.
        """
        cur: Any = self.data
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def dump_effective(self) -> dict:
        """Return a shallow copy of the effective configuration."""
        return dict(self.data)
