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

import torch, os
from utils.log import RIGHT
from model import PiscesModel, PiscesConfig
from transformers import BitsAndBytesConfig

def quantize(checkpoint, save_path, bits=8):
    # Validate arguments
    _validate_quantize_args(checkpoint, save_path, bits)
    cfg = PiscesConfig.from_json("configs/0.5B.json")
    model = PiscesModel(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])
    if bits == 8:
        import bitsandbytes as bnb
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight = bnb.nn.Params8bit(m.weight)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    RIGHT("Quantized model saved to", save_path)


def _validate_quantize_args(checkpoint: str, save_path: str, bits: int):
    """Validate and normalize arguments for quantize()."""
    if not checkpoint or not os.path.exists(checkpoint):
        raise ValueError(f"checkpoint not found: {checkpoint}")
    if not save_path:
        raise ValueError("save_path is required")
    try:
        b = int(bits)
    except Exception:
        raise ValueError("bits must be integer")
    if b not in (4, 8):
        raise ValueError("bits must be one of {4, 8}")