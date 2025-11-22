#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import torch
import math
import random
from typing import Tuple, Iterable

def _select_parameter_slices(model: torch.nn.Module, top_k_layers: int = 8) -> Iterable[torch.nn.Parameter]:
    """Select a stable subset of Linear/Conv weights to host watermark.
    This selection should be deterministic across runs (based on parameter names order).
    """
    candidates = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name.lower() for k in ["weight"]) and (p.dim() in (2, 4)):
            candidates.append((name, p))
    candidates.sort(key=lambda x: x[0])
    return [p for _, p in candidates[:top_k_layers]]

def _codebook(owner_id: str, seed: int, dim: int) -> torch.Tensor:
    rnd = random.Random(hash((owner_id, seed)))
    vec = torch.tensor([1.0 if rnd.random() > 0.5 else -1.0 for _ in range(dim)], dtype=torch.float32)
    return vec / math.sqrt(dim)

def watermarked_regularizer(param: torch.Tensor, owner_id: str, seed: int, strength: float = 1e-5) -> torch.Tensor:
    """Compute a tiny regularization loss to align a flattened parameter with codebook.
    L = strength * (dot(w_flat_norm, code))^2 to maximize correlation magnitude.
    """
    w = param.view(-1)
    if w.numel() < 256:
        return w.new_tensor(0.0)
    code = _codebook(owner_id, seed, w.numel()).to(w.device)
    w_norm = torch.nn.functional.normalize(w, dim=0)
    corr = (w_norm * code).sum()
    return strength * (-corr)  # maximizing corr -> minimize negative corr

def verify_weights(model: torch.nn.Module, owner_id: str, seed: int, top_k_layers: int = 8) -> Tuple[float, bool]:
    """Aggregate correlation score over selected layers and return pass/fail.
    Returns (score, passed). A simple threshold can be tuned offline.
    """
    params = _select_parameter_slices(model, top_k_layers=top_k_layers)
    if not params:
        return 0.0, False
    scores = []
    for p in params:
        w = p.detach().view(-1)
        if w.numel() < 256:
            continue
        code = _codebook(owner_id, seed, w.numel()).to(w.device)
        w_norm = torch.nn.functional.normalize(w, dim=0)
        corr = (w_norm * code).sum().item()
        scores.append(corr)
    if not scores:
        return 0.0, False
    avg_score = float(sum(scores) / len(scores))
    # naive threshold; to be calibrated with experiments
    passed = avg_score > 0.02
    return avg_score, passed
