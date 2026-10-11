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

"""Shape and objective regression for the multi-path reasoning core (§6.4).

The path-pruning branch indexed the fused attention output ``[batch, seq, hidden]``
with *head* indices, so the sequence axis was addressed by numbers below the head
count and the next stage died with an index error on ordinary inputs. The collapse
then multiplied a ``[batch, heads]`` weight tensor against a ``[streams, batch, seq,
hidden]`` stack. These checks run every branch of the core forward, assert the
thinking logits keep ``[batch, seq, vocab]``, that the stream collapse is a convex
combination of the retained paths, and that the memory-bounded cross-entropy equals
the plain mean cross-entropy.
"""

import torch
import torch.nn.functional as F

from model.config import YvConfig
from model.reasoning.reasoner.multipath_core import YvMultiPathReasoningEngine

HIDDEN = 64
HEADS = 8
VOCAB = 256


class PiscesLxReasonerShapeTest:
    """Assert each reasoning branch returns [batch, seq, vocab] and finite losses."""

    RESULTS = []

    def __init__(self) -> None:
        self.results: list = []

    def record(self, name: str, ok: bool, detail: str) -> None:
        self.results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")

    def _engine(self, **overrides) -> YvMultiPathReasoningEngine:
        cfg = YvConfig(hidden_size=HIDDEN, vocab_size=VOCAB, n_head=4, n_layer=2)
        for key, value in overrides.items():
            setattr(cfg, key, value)
        torch.manual_seed(0)
        engine = YvMultiPathReasoningEngine(cfg)
        if engine.reasoning_heads != HEADS:
            raise AssertionError(
                f"hidden_size {HIDDEN} should resolve {HEADS} reasoning paths, "
                f"got {engine.reasoning_heads}")
        return engine

    def _produced(self, out) -> torch.Tensor:
        """Read the collapsed thinking logits from an engine result."""
        return out.get("thinking_logits") if isinstance(out, dict) else None

    def check_pruning_branch_shapes(self) -> None:
        """Head-subspace pruning must return the input geometry, not a shuffled one."""
        engine = self._engine()
        engine.eval()
        states = torch.randn(2, 40, HIDDEN)
        out = engine.forward(hidden_states=states, input_ids=torch.randint(0, VOCAB, (2, 40)))
        produced = self._produced(out)
        expected = (2, 40, VOCAB)
        self.record("pruning.shape_preserved",
                    produced is not None and tuple(produced.shape) == expected,
                    f"got {None if produced is None else tuple(produced.shape)}, want {expected}")

    def check_stream_collapse_is_convex(self) -> None:
        """Path weights must gate streams, not rescale them: non-negative, rows sum to 1."""
        engine = self._engine()
        path_weights = F.softmax(torch.randn(4, engine.reasoning_heads), dim=-1)
        uncertainty = torch.rand(4, 1, 1)
        streams = len([n for n in engine.reasoning_streams if n != "reflection"])
        weights, rule = engine._stream_weights(path_weights, uncertainty, streams)
        row_sums = weights.sum(dim=-1)
        self.record("collapse.convex_partition",
                    tuple(weights.shape) == (4, streams)
                    and bool((weights >= 0).all())
                    and bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
                             and rule == "path_partition"),
                    f"shape={tuple(weights.shape)} sums={[round(float(v), 4) for v in row_sums]}")
        # Without path scores the collapse must fall back to a confidence-scaled uniform
        # mixture, which is a contraction: row sums stay within [0, 1].
        fallback, fallback_rule = engine._stream_weights(None, uncertainty, streams)
        sums = fallback.sum(dim=-1)
        self.record("collapse.fallback_bounded",
                    bool((fallback >= 0).all()) and bool((sums <= 1.0 + 1e-6).all())
                    and fallback_rule == "uniform_confidence_scaled",
                    f"max row sum={float(sums.max()):.4f} rule={fallback_rule}")

    def check_chunked_ce_matches_plain_mean(self) -> None:
        """The memory-bounded CE must equal the plain mean CE over the same tokens."""
        engine = self._engine()
        seq, hidden = 96, HIDDEN
        logits = torch.randn(2, seq, VOCAB)
        labels = torch.randint(0, VOCAB, (2, seq))
        labels[:, :8] = -100
        uncertainty = torch.rand(2, 1, 1)
        streams = {name: torch.randn(2, seq, hidden)
                   for name in ("hypothesis", "evidence", "conclusion")}
        total = engine._compute_reasoning_loss(streams, logits, uncertainty, labels, None, None)
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB)
        shift_labels = labels[:, 1:].reshape(-1)
        reference = F.cross_entropy(shift_logits[shift_labels != -100],
                                   shift_labels[shift_labels != -100])
        expected = reference + float(engine.cfg.reasoning_uncertainty_weight) * uncertainty.mean()
        self.record("reasoning.ce_chunk_equivalence",
                    bool(torch.allclose(total, expected, atol=1e-5)),
                    f"chunked={float(total):.6f} plain={float(expected):.6f}")

    def check_long_sequence_branch(self) -> None:
        """The LSH chunked branch must also preserve the [batch, seq, vocab] geometry."""
        engine = self._engine()
        engine.eval()
        seq = int(engine.cfg.reasoning_linear_attention_tokens) + 40
        out = engine.forward(hidden_states=torch.randn(1, seq, HIDDEN),
                             input_ids=torch.randint(0, VOCAB, (1, seq)))
        produced = self._produced(out)
        self.record("lsh_branch.shape_preserved",
                    produced is not None and tuple(produced.shape) == (1, seq, VOCAB),
                    f"got {None if produced is None else tuple(produced.shape)}, want {(1, seq, VOCAB)}")

    def check_lsh_hash_is_load_bearing(self) -> None:
        """Changing only the hash rows must change the output, else the LSH is decorative."""
        engine = self._engine()
        engine.eval()
        states = torch.randn(1, 300, HIDDEN)
        with torch.no_grad():
            baseline = engine._lsh_chunked_attention(states)
        lsh = engine.chunked_attention['lsh_proj']
        saved = lsh.weight.detach().clone()
        try:
            with torch.no_grad():
                lsh.weight.copy_(torch.randn_like(saved))
                reshaped_buckets = engine._lsh_chunked_attention(states)
        finally:
            with torch.no_grad():
                lsh.weight.copy_(saved)
        moved = float((baseline - reshaped_buckets).abs().max())
        finite = bool(torch.isfinite(baseline).all()) and bool(torch.isfinite(reshaped_buckets).all())
        self.record("lsh.hash_changes_grouping", finite and moved > 0.0,
                    f"max|delta on rehash|={moved:.4e}, finite={finite}")
        with torch.no_grad():
            restored = engine._lsh_chunked_attention(states)
        self.record("lsh.reversible",
                    bool(torch.allclose(restored, baseline, atol=1e-6)),
                    f"max|delta after restore|={float((restored - baseline).abs().max()):.2e}")

    def check_training_step_produces_a_loss(self) -> None:
        """The core must return a real objective, not a stand-in constant."""
        engine = self._engine()
        engine.train()
        seq = 40
        labels = torch.randint(0, VOCAB, (2, seq))
        out = engine.forward(hidden_states=torch.randn(2, seq, HIDDEN),
                             input_ids=labels, labels=labels)
        loss = out.get("loss") if isinstance(out, dict) else None
        ok = (loss is not None and torch.is_tensor(loss)
              and bool(torch.isfinite(loss)) and float(loss.detach()) > 0.0)
        self.record("reasoning.loss_is_real", ok,
                    "none" if loss is None else f"{float(loss.detach()):.4f}")
        if ok:
            loss.backward()
            graded = [name for name, param in engine.named_parameters()
                      if param.grad is not None and float(param.grad.abs().sum()) > 0.0]
            self.record("reasoning.gradient_reaches_core", len(graded) > 0,
                        f"{len(graded)} parameters received gradient")

    def run(self) -> int:
        for check in (self.check_pruning_branch_shapes,
                      self.check_stream_collapse_is_convex,
                      self.check_chunked_ce_matches_plain_mean,
                      self.check_long_sequence_branch,
                      self.check_lsh_hash_is_load_bearing,
                      self.check_training_step_produces_a_loss):
            try:
                check()
            except Exception as exc:
                self.record(check.__name__, False, f"{type(exc).__name__}: {exc}")
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"{passed}/{len(self.results)} checks passed")
        return 0 if passed == len(self.results) else 1


if __name__ == "__main__":
    raise SystemExit(PiscesLxReasonerShapeTest().run())
