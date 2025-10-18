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

from typing import Dict, Any

def decision(score: float, config: Dict[str, Any]) -> bool:
    """
    统一阈值判断：读取 configs['detection']['threshold']，默认0.95
    """
    try:
        thresh = float(config.get("detection", {}).get("threshold", 0.95))
    except Exception:
        thresh = 0.95
    return score >= thresh

def pack_result(modality: str, score: float, extra: Dict[str, Any] = None) -> Dict[str, Any]:
    out = {
        "modality": modality,
        "score": score
    }
    if extra:
        out["extra"] = extra
    return out