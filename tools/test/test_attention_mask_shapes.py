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

"""
Attention-mask shape regression for the long-context lane.

Guards a defect that was found while integrating the Subconscious Block Field:
``YvTokenSparseAttention`` raised a 2-D mask's rank with an ``unsqueeze(1)`` loop,
which re-interprets the trunk's ``[T, T]`` causal mask as ``[T, 1, 1, T]`` and broadcasts
the *sequence* axis into *batch*. The visible symptom was a reshape error far away from
the cause; the dangerous symptom is the case where the broadcast divides evenly and
silently produces wrong attention scores.
"""

from typing import Callable, List, Tuple

import torch

from model.config import YvConfig
from model.core.token_sparse_attn import YvTokenSparseAttention

__all__ = ["PiscesLxAttentionMaskShapeTests"]


class PiscesLxAttentionMaskShapeTests:
    """Every mask rank the trunk can hand over must return [B, T, hidden]."""

    BATCH = 2
    SEQ = 16
    HIDDEN = 320
    HEADS = 10

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []
        self.module = YvTokenSparseAttention(
            YvConfig(hidden_size=self.HIDDEN, n_head=self.HEADS),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.states = torch.randn(self.BATCH, self.SEQ, self.HIDDEN)

    def record(self, name: str, passed: bool, detail: str) -> None:
        self.results.append((name, passed, detail))

    def masks(self) -> List[Tuple[str, Callable[[], torch.Tensor]]]:
        t, b = self.SEQ, self.BATCH
        return [
            ("2D causal [T,T]", lambda: torch.zeros(t, t)),
            ("2D padding [B,T]", lambda: torch.ones(b, t)),
            ("3D per-sample causal [B,T,T]", lambda: torch.zeros(b, t, t)),
            ("3D padding [B,1,T]", lambda: torch.ones(b, 1, t)),
            ("4D [B,1,T,T]", lambda: torch.zeros(b, 1, t, t)),
            ("4D [1,1,T,T]", lambda: torch.zeros(1, 1, t, t)),
        ]

    def check_accepted_shapes(self) -> None:
        """Valid ranks must produce [B, T, hidden] — never [T, T, hidden]."""
        for name, build in self.masks():
            try:
                out = self.module(self.states, build())
            except Exception as error:
                self.record(f"accept::{name}", False,
                            f"raised {type(error).__name__}: {str(error)[:70]}")
                continue
            expected = (self.BATCH, self.SEQ, self.HIDDEN)
            self.record(f"accept::{name}", tuple(out.shape) == expected,
                        f"got {tuple(out.shape)}, want {expected}")
            if tuple(out.shape) == expected:
                self.record(f"finite::{name}", bool(torch.isfinite(out).all()), "all finite")

    def check_broadcast_is_not_silent(self) -> None:
        """A mask that would broadcast a sequence axis into batch must be rejected."""
        hostile = torch.zeros(self.SEQ, 1, 1, self.SEQ)  # leading dim is T, not B or 1
        rejected = False
        try:
            out = self.module(self.states, hostile)
        except ValueError:
            rejected = True
        except Exception:
            rejected = True
        self.record("reject::sequence-as-batch mask", rejected,
                    "rejected instead of silently broadcasting" if rejected
                    else "ACCEPTED - silent broadcast is possible again")

    def check_causal_mask_actually_masks(self) -> None:
        """The mask must change the result, proving it is applied rather than ignored."""
        unmasked = self.module(self.states, None)
        causal = torch.full((self.SEQ, self.SEQ), -1e4)
        causal.triu_(1)
        masked = self.module(self.states, causal)
        differs = float((unmasked - masked).norm()) > 1e-6
        self.record("effect::causal mask applied", differs,
                    f"|delta| = {float((unmasked - masked).norm()):.4e}")

    def run(self) -> int:
        """Execute all checks; non-zero exit on any failure."""
        self.check_accepted_shapes()
        self.check_broadcast_is_not_silent()
        self.check_causal_mask_actually_masks()
        failed = [name for name, ok, _ in self.results if not ok]
        for name, ok, detail in self.results:
            print(f"[{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")
        print(f"\n{len(self.results) - len(failed)}/{len(self.results)} checks passed")
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(PiscesLxAttentionMaskShapeTests().run())
