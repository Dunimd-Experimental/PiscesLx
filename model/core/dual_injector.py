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
from .knowledge_experts import YvKnowledgeExpertPool


class YvDualInjector(nn.Module):
    def __init__(self, cfg, device=None, dtype=None):
        super().__init__()
        self.use_subconscious = getattr(cfg, 'use_subconscious', False)
        self.use_knowledge_experts = getattr(cfg, 'use_knowledge_experts', False)
        self.subconscious = YvSubconsciousSystem(cfg, device, dtype) if self.use_subconscious else None
        if self.use_knowledge_experts and self.use_subconscious:
            self.knowledge_pool = YvKnowledgeExpertPool(cfg, device, dtype)
            self.subconscious.set_knowledge_pool(self.knowledge_pool)
        else:
            self.knowledge_pool = None

    def inject(
        self,
        h: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        film_params = None
        if self.subconscious is not None:
            film_params = self.subconscious.get_film_params(h, layer_idx if layer_idx is not None else 0)
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
