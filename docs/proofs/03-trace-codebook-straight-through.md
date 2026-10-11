# Trace-codebook alignment by a straight-through selection gate

**Code:** `model/core/block_field.py::YvSubconsciousBlockField._straight_through_gate`,
applied in `read_and_modulate`; scores from
`model/core/block_addresser.py::YvResonantTraceAddresser.score_sequence`.
**Executable proof:** `tools/test/test_sbf_addressing_gradient.py`
(run by `python manage.py test --stage 10`).
**Defect closed:** A21, `docs/audit/00-audit-report.md` §48.

---

## 0. The problem, stated precisely

Notation. Trunk states `h ∈ R^{B×T×d}`. The addresser holds a trace codebook
`R ∈ R^{N×k}` (`traces`) and a query projection `W` (`query_proj`), and scores

```
s(h_t) = (W h_t / sqrt(k)) R^T  ∈ R^N ,      c_t = argmin-free dwell-smoothed argmax_n s(h_t)_n
```

The selected block is then read and injected:

```
Δ_t = α(γ_t ⊙ h_t + β_t)         (block c_t applied to h_t)
h'_t = h_t + Δ_t
```

`c_t` is an **integer index**. Every downstream use of the selection goes through
`c_t`, hence through `index_select`/`index_copy` on an integer tensor, whose Jacobian w.r.t.
`s` is zero almost everywhere. Therefore

```
∂L/∂R = 0     and     ∂L/∂W = 0
```

for any differentiable `L(h')`: the discrete choice severs the only path from the objective
back to the codebook. Measured consequence (A21): over a full training run the addressing
module's 2 tensors / 177,152 elements all reported `grad is None` and `traces` kept its
`randn·0.02` initialisation exactly — a "learnable" addresser that never learns, i.e. the
查 side of compute-search separation addressing the pool by a frozen random projection.

Assumptions used below. (A1) `score_sequence` is differentiable in `(W, R)` — it is an
`einsum` of two linear maps. (A2) The block read is differentiable in `h`, and `Δ` is
computable for the selected block alone (that is exactly what the pool guarantees).
(A3) The injection has the additive form `h' = h + g·Δ` with `g` the selection gate — true by
construction of `read_and_modulate`.

---

## 1. Proposition 1 — value neutrality (the fix costs the forward pass nothing)

Define, with `p = softmax(s)` computed in FP32 and `p_c_t := p_{t,c_t}`,

```
g_t = 1 + ( p_c_t − stopgrad(p_c_t) ).
```

**Claim.** `g_t ≡ 1` as a value, so `h'_t = h_t + Δ_t` is bit-identical to the ungated forward,
in both training and inference; and in inference (`not self.training` or
`torch.is_grad_enabled() is False` or `not scores.requires_grad`) the gate is not built at
all, so no `O(B·T·N)` softmax is paid.

**Proof.** `stopgrad(v) = v` for values, hence `p_c_t − stopgrad(p_c_t) = 0` exactly and
`g_t = 1 + 0 = 1`. ∎ (The only numerical requirement is that the value `1.0` be representable
in the working precision; it is exact in FP16/FP32/BF16. The gate is therefore cast to
`Δ`'s dtype rather than `Δ` being promoted, so `1.0 · Δ = Δ` bit-for-bit.)

**Measured.** `p1.gate_value_is_one`: `max|g−1| = 0.000e+00` over all 128 positions.
`p1.forward_bitwise_equal`: `max|h'_grad − h'_no_grad| = 0.000e+00` — the modulated tensor is
bitwise equal, so the P1 throughput and retrieval-latency metrics are unaffected by this fix.

---

## 2. Proposition 2 — the gradient is the pathwise gradient of soft selection

**Claim.** For any differentiable objective `L(h')`, the gradient the gate sends into the
scores is

```
∂L/∂s_{t,n} = ( ⟨∂L/∂h'_t , Δ_t⟩ ) · p_c_t ( 1[n = c_t] − p_n )            (1)
```

i.e. precisely the gradient obtained by *relaxing* the selection to its softmax
(`h'_t = h_t + Σ_n p_n Δ_t^{(n)}`) and then evaluating that relaxed model **at the hard point**
`p = onehot(c)`, without ever reading the other `N−1` blocks.

**Proof.** Forward value `g_t = 1` (Prop. 1), so the upstream gradient through the gate is
`∂L/∂g_t = ⟨∂L/∂h'_t, Δ_t⟩` by (A3) and the chain rule on `h'_t = h_t + g_t Δ_t`.
By definition of `g_t`, `∂L/∂p_c_t = ∂L/∂g_t` (the `stopgrad` term contributes 0 to the
gradient by construction, and the additive `1` is constant). The softmax Jacobian is
`∂p_c/∂s_n = p_c(1[n=c] − p_n)` — substituting gives (1). ∎

**Equivalence with the relaxation.** The fully soft model has
`∂L/∂s_n = Σ_m ⟨∂L/∂h', Δ^{(m)}⟩ p_m(1[n=m] − p_n)`. At `p = onehot(c)` all `Δ^{(m≠c)}`
terms vanish because `p_m = 0`, leaving exactly (1). So the gate is the **exact** pathwise
(REINFORCE-free) gradient of the soft-selection model at the hard point, not an approximation
of it — the only approximation is that it evaluates the relaxation at the hard point instead of
at `p` itself, which is the standard straight-through choice and keeps the forward hard.

**Necessity.** Why not read all `N` blocks and differentiate the true soft model? A block read
costs ≈1.1e9 FLOPs/token at full scale (`docs/design/00-subconscious-block-field.md` §3.2,
`docs/proofs/00-sbf-theorems.md` T1). Soft selection over `N = 628` blocks multiplies that by
628 → ≈6.9e11 FLOPs/token, ≈500× the entire 7B trunk's forward cost, forfeiting the 1500 tok/s
P1 target by three orders of magnitude. The gate needs only the block that was **already**
read: cost `O(B·T·N)` for the softmax, i.e. the addressing cost Theorem 1 already declares
negligible against block application. Hence the mechanism is not merely "an estimator" but the
only member of this family that meets the retrieval budget.

**Measured.** `p2.gradient_identity`: the real gate's `∂L/∂W` against the reference
implementation of (1) — `max|diff| = 1.54e-09` against `max|ref| = 1.49e-03` (relative 1.0e-6,
the FP32 round-off of the two different reduction orders).

---

## 3. Proposition 3 — bounded credit, gradient stability

**Claim.** With `κ = α(γ_max + c)` (Theorem 5 of `00-sbf-theorems.md`), for every token

```
|∂L/∂g_t| = |⟨∂L/∂h'_t, Δ_t⟩| ≤ ‖∂L/∂h'_t‖ · κ ‖h_t‖                        (2)
```

and the induced codebook gradient satisfies `‖∂L/∂R‖ ≤ (2) · max_n ∂`-structure bound, i.e. the
alignment signal is Lipschitz in the trunk gradient and cannot explode independently of it.

**Proof.** (2) is Cauchy–Schwarz together with Theorem 5's `‖Δ_t‖ ≤ κ‖h_t‖`, which holds for
*any* block including a wrongly addressed one (premise A1: `γ` is a bounded `tanh`, `β` is
renormalised onto a sphere of radius `c‖h‖`). `‖∂L/∂R‖ = ‖∂L/∂s‖·‖W h‖/√k` and each entry of
`∂L/∂s` is bounded by (2) times `p_c ≤ 1` and the softmax row-sum `≤ 1`, giving the stated
bound. ∎

Consequence: the gate cannot destabilise training. Its magnitude is controlled by the same
quantities that bound the injection itself, and `κ` is configuration (`block_inject_alpha`,
`block_gamma_max`, `block_beta_bound`), not a free multiplier tuned per run.

**Measured.** `p3.credit_bounded`: worst realised `‖Δ‖/‖h‖ = 0.02606 ≤ κ = 0.07500`, with
`‖∂L/∂traces‖ = 7.28e-06 > 0` — the bound is met *and* non-vacuous.

---

## 4. Proposition 4 — the signal is data-driven, not self-referential

The rejected alternative, and why it is wrong, is part of the proof (AGENTS.md §8.4).

**Candidate A (considered and rejected): alignment CE on the selector's own choice**
`L = −mean_t log softmax(s)_c_t` with `c_t = argmax_n s_t`. Since the target is produced by the
same `s`, `∇_s L = p − onehot(c)` is a *pure sharpening* field: it always increases the margin
of the current winner regardless of whether that block helped. Its fixed points are any
configuration with an infinite margin, so it (i) admits the degenerate solution "one block owns
every query" and (ii) drives `‖R‖ → ∞` with no reference to `L_LM`. It is a self-consistency
drift, i.e. winner-take-all: precisely the failure mode that would leave the pool's other
blocks never trained and violate "被激活的知识块参与训练". Measured symptom while it was wired:
`aux_loss` jumped 2.27 → 26.17 (a term competing with, not serving, the objective) while
`traces.grad` stayed `None` — the bucket it was added to is reported, not optimised.

**Candidate B (adopted): the gate's credit.** Its sign is determined by
`⟨∂L/∂h'_t, Δ_t⟩` — the *inner product between the block's contribution and the direction that
reduces the language-modelling loss*. A block that helps the LM raises its own selection
probability; a block that hurts lowers it. Unlike Candidate A it has no data-independent fixed
point, and unlike a policy-gradient/score-function estimator it needs no baseline and no
variance from sampling, because the forward is deterministic and the estimator is pathwise
(Prop. 2). Its stationary condition on the codebook is `⟨∂L/∂h', Δ^{(n)}⟩` equal across the
blocks that share mass — the natural "assign each query region to the block that helps it
most" criterion, which is what 查算分离 is for.

**Measured.** `p4.traces_update`: after 6 SGD steps on a random-target objective,
`max|traces − traces_0| = 1.45e-04 > 0` (A21's symptom was exactly `0`), and
`p4.query_proj_trains`: `‖∂L/∂W‖ = 3.65e-06 > 0`.
`p4.hard_path_is_dead` is the negative control: with the same objective fed from a detached
hard read, `traces.grad is None` — so the passing checks above are attributable to the gate and
nothing else.

---

## 5. Scope and what remains unproven

* The gate trains the codebook **for the blocks it actually reads**; a block never addressed
  receives no alignment gradient. Reachability of cold blocks is the staleness term's job
  (`block_staleness_gain`/`block_staleness_cap_tokens`, Theorem 2's machinery) and is checked
  separately in `test_sbf_mechanism.py`.
* Convergence of the joint (codebook, pool contents) process is **not** claimed: Prop. 2 gives
  the exact gradient of a well-defined relaxation, not a proof that its fixed point is optimal.
  Empirical validation is the held-out CE comparison in `tools/dev/verify_checkpoint.py` once a
  run with the gate trained is compared against the A21-era weight.
* Pool contents remain trainable independently (Theorem 6: only resident blocks get gradient),
  which is why the gate's `Δ` may be treated as differentiable *and* the same derivation holds
  when the block parameters are held fixed.
