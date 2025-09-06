#!/usr/bin/env/python3

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
import json
from pathlib import Path
from typing import Dict, Any, Optional
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "MCP" / "MCP.json"
DEFAULT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "MCP"

def read_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Read MCP tools configuration (JSON). Returns empty structure on error."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        if not cfg_path.exists():
            return {"version": "1.0", "tools": {}, "meta": {}}
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Basic normalization
        data.setdefault("version", "1.0")
        data.setdefault("tools", {})
        data.setdefault("meta", {})
        return data
    except Exception:
        return {"version": "1.0", "tools": {}, "meta": {}}

def write_config(data: Dict[str, Any], path: Optional[str] = None) -> None:
    """Write MCP tools configuration (JSON)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def status(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = read_config(config_path)
    return {
        "tool_count": len(cfg.get("tools", {})),
        "keys": list(cfg.get("tools", {}).keys())[:10],
    }
