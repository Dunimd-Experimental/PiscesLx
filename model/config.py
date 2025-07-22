#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import json
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PiscesConfig:
    """Pisces L1 model configuration"""
    model_type: str = "pisces_l1"
    vocab_size: int = 100_352
    hidden_size: int = 2048
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 4
    moe_num_experts: int = 64
    moe_top_k: int = 2
    intermediate_size: int = 5632
    max_position_embeddings: int = 8192
    rope_theta: float = 1e6
    dropout: float = 0.0
    image_res: int = 224
    image_patch: int = 14
    mm_tokens: int = 256
    audio_tokens: int = 512
    task_classes: int = 256
    eval_dims: int = 7
    rope_scaling: Optional[Dict[str, Any]] = {"type": "yarn", "factor": 32, "original_max_position_embeddings": 32768}
    
    @classmethod
    def from_json(cls, path):
        """Load configuration from JSON file"""
        return cls(**json.load(open(path)))