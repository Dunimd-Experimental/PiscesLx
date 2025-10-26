#!/usr/bin/env python3

from typing import Optional
import os
from .content import WatermarkManager

_DEF_MODEL_ID = os.getenv("PISCES_WM_MODEL_ID", "PiscesL1-1.5B")


def get_watermark_manager_from_env() -> Optional[WatermarkManager]:
    """Create a WatermarkManager if enabled by env toggles.
    Env:
      PISCES_WM_ENABLE=1      enable content watermarking pipeline-wide
      PISCES_WM_MODEL_ID=...  model id used in watermark payload
    Returns None if disabled.
    """
    enable = os.getenv("PISCES_WM_ENABLE", "0").strip() in ("1", "true", "TRUE", "on", "ON")
    if not enable:
        return None
    try:
        return WatermarkManager(model_id=_DEF_MODEL_ID)
    except Exception:
        return None
