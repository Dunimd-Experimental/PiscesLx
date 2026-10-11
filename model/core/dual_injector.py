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

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .subconscious import YvSubconsciousSystem


class YvDualInjector(nn.Module):
    def __init__(self, cfg, device=None, dtype=None):
        super().__init__()
        self.use_subconscious = getattr(cfg, 'use_subconscious', False)
        self.subconscious = YvSubconsciousSystem(cfg, device, dtype) if self.use_subconscious else None

    def inject(
        self,
        h: torch.Tensor,
        layer_idx: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        film_params = None
        if attention_mask is not None and attention_mask.dim() == 2 and h.dim() >= 2 \
                and attention_mask.shape[:2] == h.shape[:2]:
            # Re-zero padded slots at every layer boundary. Nothing here takes a mask, and an
            # attention output at a padded position is a weighted sum over real keys, so pad
            # slots silently regain content from real tokens and feed it into the next layer
            # (measured: layer-0 pad slots exactly 0, layer-1 pad slots 0.49 and varying with
            # pad content; docs/audit A54). Keeping them zero is what makes the stack
            # padding-content independent.
            h = h * attention_mask.unsqueeze(-1).to(h.dtype)
        if self.subconscious is not None:
            film_params = self.subconscious.get_film_params(
                h, layer_idx if layer_idx is not None else 0, attention_mask)
        return h, None, film_params

    def get_trainable_params(self) -> Dict[str, nn.Parameter]:
        params = {}
        if self.subconscious is not None:
            for name, p in self.subconscious.named_parameters():
                if p.requires_grad:
                    params[f"subconscious.{name}"] = p
        return params

    def clear_cache(self):
        if self.subconscious is not None:
            self.subconscious.clear_cache()
