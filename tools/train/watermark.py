#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

from typing import Any, Dict, Optional

from utils.opsc.base import PiscesLxTransformOperator
from utils.opsc.interface import PiscesLxOperatorConfig
from utils.opsc.registry import PiscesLxOperatorRegistrar
from utils.dc import PiscesLxLogger


from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Train", file_path=get_log_file("PiscesLx.Tools.Train"), enable_file=True)


@PiscesLxOperatorRegistrar()
class TrainingWatermarkIntegrationOperator(PiscesLxTransformOperator):
    def __init__(
        self,
        config: Optional[PiscesLxOperatorConfig] = None,
        enabled: bool = True,
        jurisdiction: str = "GLOBAL",
    ):
        super().__init__(config)
        if config is not None:
            params = getattr(config, "parameters", {}) or {}
            enabled = bool(params.get("enabled", enabled))
            jurisdiction = str(params.get("jurisdiction", jurisdiction))
        self.enabled = enabled
        self.jurisdiction = jurisdiction
        self._content_wm = None

        if self.enabled:
            try:
                from opss.watermark.content_watermark_operator import create_content_watermark_operator

                self._content_wm = create_content_watermark_operator()
            except Exception as e:
                _LOG.warning("training_watermark_operator_unavailable", error=str(e))
                self._content_wm = None

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        return isinstance(inputs, dict)

    def transform(self, data: Any) -> Any:
        if not self.enabled or not isinstance(data, dict):
            return data

        payload = data.get("watermark_payload") or data.get("payload")
        if payload is None:
            return data

        text = data.get("text") or data.get("output_text")
        if isinstance(text, str) and self._content_wm is not None and isinstance(payload, dict) and payload:
            try:
                wm_text = self._content_wm.embed_text(text, payload, metadata={"jurisdiction": self.jurisdiction})
                data["watermarked_text"] = wm_text
            except Exception as e:
                data["watermark_error"] = str(e)

        meta = data.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            data["metadata"] = meta
        meta.setdefault("jurisdiction", self.jurisdiction)
        meta.setdefault("watermark_enabled", self.enabled)
        return data


@PiscesLxOperatorRegistrar()
class TrainingPipelineWatermarkOperator(PiscesLxTransformOperator):
    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None, enabled: bool = True):
        super().__init__(config)
        if config is not None:
            params = getattr(config, "parameters", {}) or {}
            enabled = bool(params.get("enabled", enabled))
        self.enabled = enabled
        self._integrator = TrainingWatermarkIntegrationOperator(
            PiscesLxOperatorConfig(parameters=getattr(config, "parameters", {}) or {}) if config is not None else None,
            enabled=enabled,
        )

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        return isinstance(inputs, dict)

    def transform(self, data: Any) -> Any:
        if not self.enabled or not isinstance(data, dict):
            return data
        return self._integrator.transform(data)

