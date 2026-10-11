# 证明 02 · Ink 方差的对数栅格编码与步进信赖域

对应实现：
- `opss/optim/ink/quantizer.py` — `PiscesLxInkBlockQuantizer._variance_log_grid`,
  `quantize_variance_log`, `dequantize_variance_log`
- `opss/optim/ink/optimizer.py` — `PiscesLxInkOptimizer.step`（分母与 `MAX_COORDINATE_STEP`）
- `opss/optim/ink/sparse.py` — `PiscesLxInkSparseSelector.select`, `_effective_ratio_for`,
  `_structured_mask`, `_magnitude_mask`

可执行证明（纳入验收命令，不靠人工跑）：
```
PYTHONPATH=. python tools/test/test_ink_quantizer_contract.py     # 6/6
PYTHONPATH=. python tools/test/test_ink_optimizer_contract.py     # 4/4
```

---

## 记号与前提

设某参数张量展平为 `x ∈ R^n`，按宽度 `B`（`variance_block_size`）分块；块内分量记 `x_i`。
二阶矩状态 `v ≥ 0`（逐项），由 `v_t = β₂ v_{t-1} + (1-β₂) ĝ²` 递推，`0 ≤ β₂ < 1`，故 `v` 非负。
量化级数 `L = 2^bits`（bits ∈ {4, 8}），可用码字 `1 … L-1`，码字 `0` 专用于真零。

**显式前提**
(P1) 每块内正分量有限且非空；否则该块所有码字为 0（见引理 3）。
(P2) `v` 在块内的动态范围 `R = v_max / v_min⁺` 有限（`v_min⁺` 为最小正分量）。
(P3) 一阶矩 `m` 与 `v` 来自同一 `ĝ` 序列（稀疏掩码对两者一致施加）。

---

## 引理 1（构造良定）

对任意满足 (P1) 的块，`_variance_log_grid` 输出的 `(c, ℓ, Δ)` 满足
`ℓ = ln v_min⁺`，`Δ = (ln v_max − ln v_min⁺)/(L−2) ≥ 0`，且 `c_i ∈ {0} ∪ [1, L−1]` 为整数。

*证明。* 行最大 `v_max = max_j x_j`；`v_min⁺` 由 `torch.where(x > 0, x, +inf)` 的 `min` 取得，
故为最小正分量。`Δ` 是两对数之差除以区间数 `L−2`；因 `v_max ≥ v_min⁺ > 0`，对数单调得 `Δ ≥ 0`。
正分量的码字 `c_i = round(1 + (ln x_i − ℓ)/Δ)` 被 `clamp(1, L−1)` 限制在整数码区间；
`x_i = 0` 由 `torch.where` 直接赋 0。∎

## 定理 1（相对误差界；正分量不塌缩）

在 (P1)(P2) 下，重构 `v̂_i = exp(ℓ + (c_i − 1)Δ)`（`c_i ≥ 1`）满足

1. `v_i > 0 ⇒ v̂_i > 0`；
2. `|ln v̂_i − ln v_i| ≤ Δ/2`，等价地 `e^{−Δ/2} ≤ v̂_i / v_i ≤ e^{Δ/2}`。

*证明。* 令 `y_i = (ln v_i − ℓ)/Δ ∈ [0, L−2]`（由 `ℓ, Δ` 定义与对数单调性）。
`c_i − 1 = round(y_i)`，故 `|round(y_i) − y_i| ≤ 1/2`。于是
`ln v̂_i − ln v_i = ℓ + (c_i − 1)Δ − (ℓ + y_i Δ) = Δ(round(y_i) − y_i)`，取绝对值 ≤ `Δ/2`。
指数函数严格正且单调，得 1、2。边界可达性：`y_i = k + 1/2` 时等号成立，故界为紧（非松弛）。∎

**数值核验。** `test_ink_quantizer_contract.py::quantiser.variance_log_grid` 在跨约 6 个数量级
（`LOG_SPREAD_SIGMA = 3.0`）的方差上实测比值区间
`[0.9219, 1.0847]` ⊂ `[e^{−Δ/2}, e^{Δ/2}] = [0.9213, 1.0854]`（8 bit），4 bit 同样成立；
且 `collapsed = inflated = 0`。

## 定理 2（对称 absmax 编码必然塌缩，故不可用于方差）

设块采用对称绝对值编码：`scale = v_max/(L̃/2)`（`L̃ = 2^bits` 个对称码字），重构前向映射
`c = round(x/scale)`。则

`v_i ≤ scale/2 = v_max/L̃ ⇒ c_i = 0 ⇒ v̂_i = 0`。

即动态范围超过 `log₂ L̃` 比特的块中，处于低端一半量级区间的正分量**全部**映射为 0。
对 `L = 16`（4 bit），`v_max/v_i > 8` 即可触发；对 `L = 256`，`v_max/v_i > 128`。

*证明。* `round` 到最近整数：`|v_i/scale| < 1/2` 时 `c = 0`，重构 `0 · scale = 0`。条件
`v_i < v_max/L̃` 与之等价。∎

**推论（A16 的机制）。** Adam 步进 `u_i = m̂_i/(√(v̂_i/bc₂) + ε)`。若 `v̂_i = 0` 而 `m̂_i ≠ 0`，
分母退化为 `ε`，`|u_i| = |m̂_i|/ε` 无上界，单步位移以 `1/ε` 倍尺度放大——实测该比例的中位数由
`1.0·lr` 变为 `1.34×10⁷·lr`，并在第 15 步发散为 NaN。定理 1 的 1) 排除了这条路径：
`v̂_i > 0` 使分母与 `√v_i` 同阶，`|u_i|` 回到有界的符号步尺度。

**数值核验。** `quantiser.log_grid_beats_signed_int4`：同一数据上有符号 int4 使
`3539/3584`（98.7%）的正分量塌缩为 0，对数栅格 int4 塌缩数为 `0`。

## 定理 3（信赖域给出每坐标位移上界）

设 `lr > 0`，`update_i = clamp(m̂_i/(√(v̂_i/bc₂) + ε), −1, 1)`，参数更新
`θ_i ← θ_i − lr·update_i`（权重衰减在定理 4 单独处理）。则对每次被选中的坐标

`|Δθ_i| ≤ lr`。

*证明。* `|update_i| ≤ 1` 由 `clamp` 直接给出；`|Δθ_i| = lr·|update_i| ≤ lr`。∎

**为何这不是限制表达能力的近似。** 对被选坐标且状态已收敛的平稳梯度，
`m̂_i → E[g]`，`v̂_i → E[g²]`，由 Jensen `E[g]² ≤ E[g²]` 得
`|m̂_i|/√v̂_i ≤ 1` 以概率 1（零方差时取等号）；故截断只作用于 `m̂_i² > v̂_i` 的暂态
——即梯度刚突破自身历史 RMS 的时刻——而这正是量化误差最可能造成灾难性位移的时刻。
被舍去的量至多为 `(1/√n)·(|m̂_i|/√v̂_i − 1)` 的谱范数贡献，方向不变。
实测：`ink.trust_region_bounds_every_step` 在 200 步、梯度尺度按 `10^{k mod 3}` 反复突变的
激励下，单步最大位移 = `1.0000 × lr`，从未越界；`ink.first_step_is_adam_sign_step` 给出
`median|Δ|/lr = 1.0000`、下降符号一致率 `1.0000`（Adam 理论值均为 1）。

## 定理 4（解耦权重衰减的独立性）

实现采用 `θ ← θ − lr·λ·θ − lr·update`，等价于 `θ ← (1 − lr·λ)θ − lr·update`。
映射 `θ ↦ (1−lr·λ)θ − lr·update(θ)` 在 `|update| ≤ 1`（定理 3）下是
`(1 − lr·λ)`-Lipschitz 的仿射收缩叠加有界扰动，故对 `lr·λ ∈ (0, 1)` 迭代有界：
`‖θ_t‖ ≤ (1 − lr·λ)^t ‖θ_0‖ + lr·Σ_{k<t}(1−lr·λ)^k ≤ (1−lr·λ)^t ‖θ_0‖ + 1/λ`。
即权重范数上界不依赖梯度幅度，衰减与更新可分离证明——这正是把它写成两次 `add_`
而非在梯度上加 `λθ` 的原因（后者会污染 `v̂` 的统计量，与定理 1 的界耦合）。

## 定理 5（声明稀疏率成为唯一权威）

设选择器按块粒度选中 `k = max(1, round(N_b · ρ))` 个块（`N_b` 为块数），
`ρ = min(1, max(ρ_min, r_eff · f))`，其中 `r_eff` 为 warmup 调度的当前密度，
`f = clamp(ρ_target/observed, 0.5, 2.0)` 为逐参数比例控制器。

1. **固定点**：若某参数的实测选择密度 `observed = ρ_target`，则 `f = 1`，
   `ρ = r_eff`；当 `step ≥ warmup_steps` 时 `r_eff = ρ_target`，即声明值是不动点。
2. **收敛方向**：`observed < ρ_target ⇒ f > 1 ⇒ ρ` 上升，反之下降；故控制律把
   `observed` 推向 `ρ_target`，且 `f` 的 `[0.5, 2.0]` 界保证单步密度至多翻倍/减半，
   不会因一步噪声整体开关一个参数。
3. **覆盖性下界**：`k ≥ 1` 保证每步至少更新一个块（`ρ_min` 的 `MIN_SELECTION_RATIO` 兜底），
   故不存在"整步空转"。

*证明。* 1)、2) 由 `f` 的定义与 `clamp` 单调性直接得出；3) 由 `max(1, ·)`。∎

**数值核验。** `ink.declared_sparsity_is_authoritative`：声明 `0.01` 得 32 个被更新坐标
（块宽 32 的整数倍，目标 41 的块对齐结果），声明 `0.1` 得 416（目标 410），
warmup 生效时第 0 步为全密度 `4096/4096`。修复前该三值恒为 `384/4096`（9.375%），
因为 `_structured_sparse_projection` 读取的是从未被任何配置写入的
`gradient_compression_ratio = 0.1`。

## 引理 6（复杂度与额外存储）

分块数为 `⌈n/B⌉`。栅格拟合每块需 `max`、`min`、逐元素 `log/round/clamp` 与 `exp`，
故时间 `Θ(n)`（与原 absmax 同阶，多出每元素一次 `log` 与一次 `exp`），
空间 `O(n)` 码字 + 每块 2 个 fp32。有效比特率
`bits + 64/B`（`B = 256` 时 8-bit 为 `8.25` bit，4-bit 为 `4.25` bit），
与 absmax 方案逐位相同——**精度提升不以内存为代价**。

## 定理 7（真实目标下降，非代理指标）

在固定批量最小二乘目标 `J(θ) = mean_i ((Xθ)_i − y_i)²` 上，以 `sparse_ratio = 0.5`、
`lr = 5×10⁻²` 迭代 200 步，实测 `J` 由 `7.8563×10¹` 降至 `2.3047×10⁰`，
相对下降 `97.1% ≥ 50%`（阈值 `DESCENT_IMPROVEMENT_MIN`），且权重保持有限。

该命题的作用是把前面各定理与"优化器真的在学习"绑定：定理 1–4 保证单步良定，
引理 5 保证声明密度被遵守，而只有 `J` 单调下降才排除
"步长正确但方向错误"这一类失败——此前 Ink 恰属此类（`descent_agreement = 0.553`，
等同随机符号）。

---

## 与既有方法的对比

| 方案 | 方差表示 | 正分量塌缩 | 相对误差 | 额外内存 |
|---|---|---|---|---|
| 符号 absmax（修复前） | 线性，含负码 | 是（4-bit 实测 98.7%） | 无上界（分母→ε） | 0 |
| 8-bit AdamW（Dettmers 2021） | 线性 + 偏移 | 部分（低端截断） | 绝对界，非相对界 | 0 |
| 本方案（分块对数栅格） | `log v` 均匀 | 否（定理 1） | `e^{±Δ/2}`，紧 | `64/B` bit/元素 |

## 结论↔代码↔测试对应

| 结论 | 代码 | 测试 |
|---|---|---|
| 定理 1 | `quantizer.py:_variance_log_grid/dequantize_variance_log` | `quantiser.variance_log_grid` |
| 定理 2 | `quantizer.py:quantize_int4`（保留给动量） | `quantiser.log_grid_beats_signed_int4` |
| 定理 3 | `optimizer.py:MAX_COORDINATE_STEP`, `step()` | `ink.trust_region_bounds_every_step` |
| 引理 1/3 | 同上（`code 0` 语义） | `quantiser.zeros_stay_zero` |
| 定理 5 | `sparse.py:select/_effective_ratio_for/_structured_mask` | `ink.declared_sparsity_is_authoritative` |
| 定理 7 | 整个 `step()` | `ink.reduces_a_real_loss` |
