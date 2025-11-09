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

import math
from typing import Tuple, List
import torch

def _dct_matrix(n: int, device=None, dtype=None) -> torch.Tensor:
    """Create an NxN DCT-II transform matrix."""
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    alpha = torch.ones(n, device=device, dtype=dtype)
    alpha[0] = 1.0 / math.sqrt(2.0)
    C = math.sqrt(2.0 / n) * alpha.unsqueeze(1) * torch.cos((math.pi * (2 * k + 1) * i) / (2.0 * n))
    return C

def block_dct2(x: torch.Tensor, block: int) -> torch.Tensor:
    """Apply 2D DCT (type-II) in non-overlapping blocks on a single-channel image tensor [H, W]."""
    H, W = x.shape
    assert H % block == 0 and W % block == 0
    C = _dct_matrix(block, device=x.device, dtype=x.dtype)
    CT = C.t()
    x_blocks = x.unfold(0, block, block).unfold(1, block, block)  # [H//b, W//b, b, b]
    x_blocks = x_blocks.contiguous()
    Hn, Wn = x_blocks.shape[:2]
    y = torch.zeros_like(x)
    for by in range(Hn):
        for bx in range(Wn):
            bmat = x_blocks[by, bx]
            yb = C @ bmat @ CT
            y[by * block:(by + 1) * block, bx * block:(bx + 1) * block] = yb
    return y

def block_idct2(X: torch.Tensor, block: int) -> torch.Tensor:
    """Inverse of block_dct2 for a single-channel [H, W]."""
    H, W = X.shape
    assert H % block == 0 and W % block == 0
    C = _dct_matrix(block, device=X.device, dtype=X.dtype)
    CT = C.t()
    X_blocks = X.unfold(0, block, block).unfold(1, block, block)
    X_blocks = X_blocks.contiguous()
    Hn, Wn = X_blocks.shape[:2]
    y = torch.zeros_like(X)
    for by in range(Hn):
        for bx in range(Wn):
            B = X_blocks[by, bx]
            yb = CT @ B @ C
            y[by * block:(by + 1) * block, bx * block:(bx + 1) * block] = yb
    return y

def midband_coordinates(block: int, band: str = "mid") -> List[Tuple[int, int]]:
    """Return a list of (u,v) coords considered mid-band for a DCT block."""
    coords: List[Tuple[int, int]] = []
    if band == "low":
        r0, r1 = 0, max(1, block // 3)
    elif band == "high":
        r0, r1 = max(1, block // 2), block
    else:  # mid
        r0, r1 = block // 3, (2 * block) // 3
    for u in range(r0, r1):
        for v in range(r0, r1):
            if not (u == 0 and v == 0):
                coords.append((u, v))
    return coords

def enforce_pair_relation(a: torch.Tensor, b: torch.Tensor, bit: int, delta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enforce |a| > |b| if bit=1 else |b| > |a| by minimal multiplicative scaling."""
    mag_a = torch.abs(a)
    mag_b = torch.abs(b)
    if bit == 1:
        if mag_a <= mag_b + delta:
            scale = (mag_b + delta + 1e-8) / (mag_a + 1e-8)
            a = a * scale
    else:
        if mag_b <= mag_a + delta:
            scale = (mag_a + delta + 1e-8) / (mag_b + 1e-8)
            b = b * scale
    return a, b

def embed_bits_in_dct(channel: torch.Tensor, bits: str, block: int, band: str, strength: float, seed: int) -> torch.Tensor:
    """Embed bits into a single-channel image via block DCT pairwise embedding."""
    H, W = channel.shape
    C = block_dct2(channel, block)
    coords = midband_coordinates(block, band)
    # deterministic RNG
    g = torch.Generator(device=channel.device)
    g.manual_seed(seed)

    Hn, Wn = H // block, W // block
    bit_idx = 0
    for by in range(Hn):
        for bx in range(Wn):
            if bit_idx >= len(bits):
                break
            # pick two coords per bit (deterministic)
            if len(coords) < 2:
                continue
            sel = torch.randperm(len(coords), generator=g, device=channel.device)[:2]
            (u1, v1), (u2, v2) = coords[sel[0].item()], coords[sel[1].item()]
            y0, x0 = by * block, bx * block
            a = C[y0 + u1, x0 + v1]
            b = C[y0 + u2, x0 + v2]
            delta = strength * (0.1 + 0.9 * torch.rand((), generator=g, device=channel.device))
            bit = 1 if bits[bit_idx] == '1' else 0
            a2, b2 = enforce_pair_relation(a, b, bit, float(delta))
            C[y0 + u1, x0 + v1] = a2
            C[y0 + u2, x0 + v2] = b2
            bit_idx += 1
        if bit_idx >= len(bits):
            break
    out = block_idct2(C, block)
    return out

def extract_bits_from_dct(channel: torch.Tensor, n_bits: int, block: int, band: str, seed: int) -> str:
    """Extract up to n_bits by reading the enforced pair relations in block DCT domain."""
    H, W = channel.shape
    C = block_dct2(channel, block)
    coords = midband_coordinates(block, band)
    g = torch.Generator(device=channel.device)
    g.manual_seed(seed)

    Hn, Wn = H // block, W // block
    bits: List[str] = []
    for by in range(Hn):
        for bx in range(Wn):
            if len(bits) >= n_bits:
                break
            if len(coords) < 2:
                continue
            sel = torch.randperm(len(coords), generator=g, device=channel.device)[:2]
            (u1, v1), (u2, v2) = coords[sel[0].item()], coords[sel[1].item()]
            y0, x0 = by * block, bx * block
            a = C[y0 + u1, x0 + v1]
            b = C[y0 + u2, x0 + v2]
            bits.append('1' if torch.abs(a) > torch.abs(b) else '0')
        if len(bits) >= n_bits:
            break
    return ''.join(bits)
