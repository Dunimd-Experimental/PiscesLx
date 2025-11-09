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
from typing import Optional
from .content import PiscesLxCoreWatermarkManager

_DEF_MODEL_ID = os.getenv("PISCESLX_WM_MODEL_ID", "PiscesL1")

def get_watermark_manager_from_env() -> Optional[PiscesLxCoreWatermarkManager]:
    """Create a PiscesLxCoreWatermarkManager if enabled by env toggles.
    Env:
      PISCESLX_WM_ENABLE=1      enable content watermarking pipeline-wide
      PISCESLX_WM_MODEL_ID=...  model id used in watermark payload
    Returns None if disabled.
    """
    enable = os.getenv("PISCESLX_WM_ENABLE", "0").strip() in ("1", "true", "TRUE", "on", "ON")
    if not enable:
        return None
    try:
        return PiscesLxCoreWatermarkManager(model_id=_DEF_MODEL_ID)
    except Exception:
        return None
