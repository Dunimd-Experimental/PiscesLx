#!/usr/bin/env python3

# Weight-level watermarking utilities (skeleton)
# Methods: Uchida/DeepSigns-style correlation embedding and verification.
# NOTE: Integrate during training as an additional regularization term with very small strength.

from typing import Tuple, Iterable
import torch
import math
import random


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
