#!/usr/bin/env/python3
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

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from utils.dc import PiscesLxLogger
from configs.version import VERSION
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus


_LOG = PiscesLxLogger(__name__)


class POPSSNativeInferenceOperator(PiscesLxOperatorInterface):
    def __init__(self):
        super().__init__()
        self.name = "infer.native_ruchbah"
        self.version = VERSION
        self.type = "inference"

    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        try:
            ckpt_path = str(inputs.get("ckpt") or "").strip()
            prompt = str(inputs.get("prompt") or "").strip()
            if not ckpt_path:
                raise ValueError("ckpt is required")
            if not prompt:
                raise ValueError("prompt is required")
            if not os.path.isfile(ckpt_path):
                raise ValueError(f"ckpt must be a local file: {ckpt_path}")

            model_size = str(inputs.get("model_size") or "0.5B").strip() or "0.5B"
            seq_len = int(inputs.get("seq_len") or 512)

            generation = inputs.get("generation") or {}
            max_new_tokens = int(generation.get("max_new_tokens", 256))
            temperature = float(generation.get("temperature", 0.7))
            top_p = float(generation.get("top_p", 0.9))
            top_k = int(generation.get("top_k", 50))
            use_speculative = bool(generation.get("use_speculative", False))
            mode = str(generation.get("mode") or ("thinking" if use_speculative else "auto"))

            from model import YvConfig, YvModel
            from model.tokenizer import YvTokenizer

            cfg_path = self._resolve_model_cfg_path(model_size=model_size, config_path=inputs.get("model_config"))
            cfg = YvConfig.from_json(str(cfg_path))
            model = YvModel(cfg)

            raw = torch.load(ckpt_path, map_location="cpu")
            if isinstance(raw, dict):
                state = raw.get("model_state_dict") or raw.get("model") or raw.get("state_dict") or raw
            else:
                state = raw
            if not isinstance(state, dict):
                raise ValueError("checkpoint does not contain a state_dict")
            model.load_state_dict(state, strict=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            tokenizer = YvTokenizer()
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            out_ids, stats = model.generate(
                input_ids=input_ids,
                max_length=int(input_ids.shape[1] + max_new_tokens),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_speculative=use_speculative,
                mode=mode,
                seq_len=seq_len,
            )
            text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)

            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={"text": text, "stats": stats or {}, "backend": "native_ruchbah"},
                metadata={"version": self.version, "model_size": model_size, "model_config": str(cfg_path)},
                execution_time=0.0,
            )
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                metadata={"version": self.version, "error_type": type(e).__name__},
                execution_time=0.0,
            )

    def _resolve_model_cfg_path(self, *, model_size: str, config_path: Optional[str]) -> Path:
        cand = []
        if config_path:
            cand.append(Path(str(config_path)))
        cand.append(Path("configs") / "model" / f"{model_size}.json")
        cand.append(Path("configs") / f"{model_size}.json")
        p = next((x for x in cand if x.exists()), None)
        if p is None:
            raise FileNotFoundError(f"model config not found for model_size={model_size}")
        return p

