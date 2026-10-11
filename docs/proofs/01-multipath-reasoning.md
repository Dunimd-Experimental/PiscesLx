# 多路推理核心数学证明集（路径剪切 · 流塌缩 · LSH 分块注意 · 分块交叉熵）

> 配套实现：`model/reasoning/reasoner/multipath_core.py`（`YvMultiPathReasoningEngine`）
> 遵循 `AGENTS.md` §8：每项结论给出**正确性 / 有效性 / 必要性**，每步标注依据
> （定义 / 引理 / 定理 / 已知结论 / 算术），并与代码位置、机检测试一一对应。
> 近似之处显式声明其位置、误差界与可接受理由；不声称未证明的性质。

### 记号表

| 记号 | 含义 | 来源 |
|------|------|------|
| `B, T, d` | 批大小、序列长度、隐维 | `hidden_states` 形状 |
| `m` | 推理路径数 = 注意力头数 | `cfg.reasoning_heads`，经 `_resolve_head_count` 校正 |
| `h` | 单路径子空间宽 = `d/m` | 整除性见引理 0 |
| `ρ, k` | 保留比例、保留路径数 `k = clamp(round(ρm), 2, m)` | `cfg.reasoning_path_keep_fraction` |
| `S` | 推理流数（hypothesis / evidence / conclusion）= 3 | `reasoning_streams` |
| `α ∈ R^{B×m}` | 归一化保留路径权重 | `forward` 剪切块 |
| `β ∈ R^{B×S}` | 流塌缩权重 | `_stream_weights` |
| `c, L` | LSH 分块宽、哈希位数 | `_lsh_chunked_attention` 形参默认 |
| `V` | 词表大小 | `cfg.vocab_size` |
| `n` | 参与 CE 的有效 token 数 | `_compute_reasoning_loss` |

**显式假设**
- **H1（能量有界）** 每个路径子空间能量 `E_j = ‖y_j‖² ≤ E_max`，`j = 1..m`。
- **H2（门控值域）** 事实校验器与不确定性头以 `Sigmoid` 结尾，故
  `v ∈ (0,1)`、`u ∈ (0,1)`（实现：`fact_verifier`、`uncertainty_head` 末层）。
- **A1（LSH 近邻同桶）** 对查询 `q`，落在其窗口外的键的稠密注意力质量 `≤ ε`。
- **A2（整除性）** `m | d`；由 `_resolve_head_count` 在构造期强制成立。

---

## 引理 0（几何良定义：头子空间与稠密输出同形）

**命题**：`path_states = raw.reshape(B, T, m, h)` 与 `nn.MultiheadAttention` 的内部分头一致，
且门控重组合后的张量仍为 `[B, T, d]`。

**证明**：`_resolve_head_count`（`multipath_core.py:460`）返回不超过请求值且整除 `d` 的最大因子，
故 `d = m·h`（依据：A2 + 带余除法）。`nn.MultiheadAttention` 要求 `embed_dim % num_heads == 0`
（PyTorch 文档定义），其输出 `raw` 的第 `d` 维按头顺序拼接，因此 `reshape(B,T,m,h)` 的第 `j` 片
恰为第 `j` 头的输出子空间（依据：该模块的 out_proj 之前按头分块拼接的定义）。逐元素门控
`path_states ⊙ (α 的广播)` 不改变形状，`reshape(B,T,d)` 为其逆视图（依据：行主序视图可逆）。∎

> **为何必要**：修复前的实现以**头序号**索引 `[B,T,d]` 输出（等价于沿 batch 轴索引），
> 在 `T ≥ 1` 的普通输入上即越界；本引理锁定"路径 = 头子空间"这一唯一正确解释。

---

## 定理 1（路径剪切 = 有界对角门控；复杂度线性；信息损失界）

**命题 1.1（非扩张）** 设 `α_{ij} ≥ 0`，`Σ_j α_{ij} = 1`，`ŷ` 为剪切后融合态，则
`‖ŷ_{bi}‖ ≤ max_j ‖y_{bj}‖ · √m`，且逐子空间有 `‖ŷ_{bj}‖ = α_{bj}‖y_{bj}‖ ≤ ‖y_{bj}‖`。

**命题 1.2（复杂度）** 剪枝条代价 `O(B·m·log m + B·T·d)`，即关于 `T` 线性。

**命题 1.3（被剪质量界）** 记被剪掉的权重质量 `q = Σ_{j∉K} α_j`，则
`‖y − ŷ‖² ≤ q² · E_max`，且 `q ≤ (m − k)·α_{(k+1)}`，其中 `α_{(k+1)}` 为第 `k+1` 大的保留权重。

**证明**：
- 1.1：`α_{bj} ≤ Σ_j α_{bj} = 1`（依据：`softmax` 非负归一，实现 `multipath_core.py:606`），
  故每子空间缩比 ≤ 1；`‖ŷ_b‖² = Σ_j α_{bj}²‖y_{bj}‖² ≤ Σ_j ‖y_{bj}‖² ≤ m·max_j‖y_{bj}‖²`
  （依据：柯西不等式与 `m` 项求和）。∎
- 1.2：`torch.topk` 在长度 `m` 的轴上代价 `O(B·m·log m)`（依据：选择算法标准界），
  `scatter_` 与逐元素乘各 `O(B·m)`、`O(B·T·d)`，reshape 为视图 `O(1)`（依据：引理 0）。∎
- 1.3：`‖y−ŷ‖² = Σ_{j∉K} α_{bj}²‖y_{bj}‖² ≤ E_max Σ_{j∉K} α_{bj}² ≤ E_max (Σ_{j∉K} α_{bj})² = q²E_max`
  （依据：H1 + 非负数平方和 ≤ 和的平方）。排序性给出 `α_{(j)} ≤ α_{(k+1)}` 对 `j > k`，
  故 `q = Σ_{j>k} α_{(j)} ≤ (m−k)·α_{(k+1)}`。∎

**有效性**：`k = m` 时 `K` 为全集，剪切退化为纯归一化门控，不丢任何子空间（`q = 0`），
即表达能力不弱于未剪切的稠密注意力（依据：命题 1.3 取 `q=0`）。

**近似声明**：本设计**保留方向、按重要性缩放幅值**，不声称与稠密输出恒等；
幅值界由 1.1 给出，损失界由 1.3 给出，二者随 `k/m` 单调收紧。

---

## 定理 2（流塌缩为凸组合：几何保持、非扩张、梯度稳定）

**命题 2.1（凸性）** 由 `_stream_weights`（`multipath_core.py:497`）产生的 `β` 满足
`β_{bs} ≥ 0` 且 `Σ_s β_{bs} = 1`。

**命题 2.2（形状保持）** `_efficient_path_collapse`（`multipath_core.py:666`）输出 `[B, T, d]`。

**命题 2.3（非扩张与梯度界）** 塌缩算子 `C(y^{(1)},…,y^{(S)}) = (Σ_s β_s y^{(s)}) ⊙ v`，
其中 `v ∈ (0,1]` 为逐位置校验置信度，则
`‖C‖ ≤ max_s ‖y^{(s)}‖`；其对 `y^{(s)}` 的 Jacobian 谱范数 `≤ 1`。

**证明**：
- 2.1：`_head_group_indices`（`multipath_core.py:480`）把 `{0,…,m−1}` 映到 `{0,…,S−1}`，
  每个 `j` 恰属一组（依据：定义 `s = ⌊jS/m⌋` 为函数）；`scatter_add_` 在该划分上求和，
  故 `Σ_s β_{bs} = Σ_j α_{bj} = 1`，非负性由 `α ≥ 0` 保持。∎
- 2.2：`stack` 得 `[S,B,T,d]`，乘 `βᵀ.view(S,B,1,1)` 沿 `S` 求和消去该轴（依据：广播规则，
  `S` 与 `B` 由 `transpose(0,1)` 对齐），故结果 `[B,T,d]`。∎
- 2.3：三角不等式 + 凸性：`‖Σ_s β_s y^{(s)}_t‖ ≤ Σ_s β_s‖y^{(s)}_t‖ ≤ max_s‖y^{(s)}_t‖`；
  再乘 `v_t ≤ 1`（依据：H2）不增大。Jacobian 关于各 `y^{(s)}` 为对角乘子 `β_s v_t ≤ 1`，
  分块对角矩阵的谱范数等于最大块范数（已知结论），故 `≤ 1`。∎

**推论（梯度稳定性）** 塌缩不放大反传梯度范数；校验门控引入的附加项含 `∂v/∂(·)`，
其因子为 `Sigmoid` 导数 `≤ 1/4`（已知结论），故整段乘法链的 Lipschitz 常数仍为 `O(1)`。

**无路径分数的分支**：当走 LSH 长序列路径而无 `α` 时，`β = (1/S)·(1−ū)`，
`ū ∈ (0,1)` ⇒ 行和 `1−ū ∈ (0,1)`，塌缩为**压缩映射**（依据：均匀分布 + H2），
范数界与 2.3 同样成立，且不声称归一（诚实：此时不保证行和为 1，只保证 ≤ 1）。

---

## 定理 3（LSH 分桶分块注意：线性复杂度与窗口近似误差界）

**命题 3.1（复杂度）** `_lsh_chunked_attention`（`multipath_core.py:809`）单样本代价
`O(T·log T + T·c·d)`，关于 `T` 线性（`c` 固定）；相对稠密注意 `O(T²·d)` 的加速比为 `T/(2c)`。

**命题 3.2（逐位置误差界）** 设查询 `i` 的稠密注意力权重为 `a_i`、窗口内重归一化后为 `â_i`，
并满足 A1（窗口外质量 `≤ ε`），则输出差
`‖z_i^dense − z_i^lsh‖ ≤ 2ε · max_j ‖v_j‖`。

**证明**：
- 3.1：投影三次 `O(T·d²)`（对固定 `d` 视为 `O(T)`）、哈希 `O(T·d·L)`、
  逐样本 `argsort` `O(T log T)`（依据：比较排序下界可达）、
  分块注意 `(T/c)` 个块 × 每块 `c × 2c` 得分 × `d` = `O(T·c·d)`（依据：实现把 `q/k/v`
   reshape 为 `[B, ⌈T/c⌉, c, d]` 并与宽度 `2c` 的窗口相乘），
  gather/scatter 各 `O(T·d)`。求和得结论。∎
- 3.2：`z_i^dense = Σ_j a_{ij} v_j`，`z_i^lsh = Σ_{j∈W(i)} â_{ij} v_j`。
  两个概率向量在 `W(i)` 之外总差 `≤ ε`，窗口内 `Σ_j |a_{ij} − â_{ij}| ≤ 2ε`
  （依据：`ℓ1` 距离与尾部质量关系，`â` 为 `a` 在 `W` 上的重归一化）；
  由对偶 `|Σ_j w_j v_j| ≤ (max_j‖v_j‖)·‖w‖_1` 得界。∎

**分桶的意义（必要性）**：若不作桶排序而仅按原序分块，窗口 `W(i)` 与语义无关，
A1 不成立且 `ε` 无界——这正是修复前的实现。现实现使 `W(i)` 由哈希桶决定，
依据 sign-projection LSH 的碰撞概率 `P[同桶] = 1 − θ(u,v)/π`（Indyk–Motwani 1998；
Charikar 2002），角距离小的键以概率 `1 − θ/π` 同桶；再并入"前一桶块"窗口，
把跨界质量纳入，进一步减小 `ε`（参照 Reformer, Kitaev et al., ICLR 2020）。

**近似位置声明**：唯一近似即 A1 的 `ε`；其代价在命题 3.2 中显式给定，
且 `c` 与 `L` 为可调超参（`c=64, L=8` 为默认），`T ≤ c` 时窗口覆盖全序列，`ε = 0`（无近似）。

---

## 引理 4（分块交叉熵与整体均值交叉熵恒等，非近似）

**命题**：`_compute_reasoning_loss`（`multipath_core.py:883`，分块段 `:915-929`）满足

```
Σ_r n_r L_r / n  =  (1/n) Σ_{i∈V} ℓ_i   =   整体均值 CE
```

其中 `V` 为有效 token 集、`|V| = n`，`{V_r}` 是其按 `c = REASONING_CE_TOKEN_CHUNK` 的一个划分，
`n_r = |V_r|`，`L_r = (1/n_r) Σ_{i∈V_r} ℓ_i`。

**证明**：`Σ_r n_r L_r = Σ_r Σ_{i∈V_r} ℓ_i = Σ_{i∈V} ℓ_i`（依据：划分的不交并 + 有限和重排），
两边除以 `n = Σ_r n_r` 即得。∎

**内存收益（算术）**：`n = B·T` 时全量实现需 `O(n·V)` fp32 中间量；
分块实现峰值 `O(c·V)`。取 `B=1, T=512, V=154885`：
`1·512·154885·4 B = 3.17 GB` → `1024·154885·4 B = 0.63 GB`（依据：`c = 1024`）。

**机检**：`tools/test/test_reasoner_shapes.py::check_chunked_ce_matches_plain_mean`
以 `atol=1e-5` 断言恒等式（含 `-100` 忽略位），非抽样。

---

## 引理 5（目标函数有界，满足标准收敛分析的前提校验）

**命题**：推理总目标为四项之和，各项有界：
CE 项 `≥ 0`（`−log p` 定义）；流 CE 项权重 `w_s = cfg.reasoning_stream_loss_weight`；
不确定性项 `w_u · ū ∈ (0, w_u)`（依据：H2）；一致性项 `w_c · MSE(v, 1) ∈ [0, w_c]`
（依据：`v ∈ (0,1)` ⇒ `|v − 1| < 1`）。故 `0 ≤ L_reason < ∞`。

**说明**：由引理 5 + 定理 2 的 Lipschitz 界可得该子模块梯度范数有界，
这是标准 Adam/AdamW 收敛分析所需的**有界梯度**条件在实现中成立的经验证前提；
本文不声称给出新的收敛定理，亦不声称全模型收敛性。

---

## 对比证明（相对既有方法的数学优势）

| 机制 | 成本 | 可微性 | 与本实现比较 |
|------|------|--------|--------------|
| Self-Consistency（Wang et al., ICLR 2023）：采样 `N` 条链 + 多数投票 | `N ×` 前向；投票 `O(N)` | 投票为不可微离散算子，路径分数**无梯度** | 本实现单次前向（定理 1.2），`α` 全程可微（定理 2.3），成本 `×1` |
| Switch / MoE 路由（Fedus 2021、Shazeer 2016） | 每 token 载入 `k` 个专家参数 | 可微（需负载均衡辅助损失） | 本实现路由**同一注意力输出的头子空间**，不引入专家参数、容量零增长（引理 0） |
| 稠密多头注意（Vaswani 2017） | `O(T² d)` | 可微 | 长序列走定理 3 的 `O(T log T + T c d)`，误差界 3.2 |
| Reformer LSH（Kitaev 2020） | `O(T log T)` | 可微 | 同阶；本实现额外并入前一块窗口，减小 `ε`（定理 3 论证） |

---

## 结论 ↔ 实现 ↔ 机检 对应表

| 结论 | 实现位置 | 机检 |
|------|----------|------|
| 引理 0 整除性 | `_resolve_head_count:460` | `check_pruning_branch_shapes`（构造期断言 `m=8`） |
| 定理 1 剪切线性/几何 | `forward:595-610` | `pruning.shape_preserved` |
| 定理 2.1 凸划分 | `_stream_weights:518-520` | `collapse.convex_partition` |
| 定理 2.2 形状保持 | `_efficient_path_collapse:693-700` | `pruning/key/lsh_branch.shape_preserved` |
| 定理 2.3 压缩分支 | `_stream_weights:527-529` | `collapse.fallback_bounded` |
| 定理 3 分桶生效 | `_lsh_chunked_attention:852-877` | `lsh.hash_changes_grouping`、`lsh.reversible` |
| 引理 4 CE 恒等 | `_compute_reasoning_loss:915-929` | `reasoning.ce_chunk_equivalence` |
| 引理 5 目标有界且真实 | `_compute_reasoning_loss:931-953` | `reasoning.loss_is_real`、`reasoning.gradient_reaches_core` |

全部机检由 `python manage.py test --stage 10` 复现（不接受人工声明）。

---

## 变更与缺陷修复留痕（2026-10-11）

1. **头序号索引稠密输出**（越界崩溃）→ 头子空间门控（定理 1，引理 0 为其前提）。
2. **塌缩以 `[B, heads]` 乘 `[S,B,T,d]`**（广播错位）→ 头划分到流的凸组合（定理 2）。
3. **校验置信 `[B,T,1]` 误当作 `[B,1]` 并 `unsqueeze`**（第二次错位）→ 逐位置门控。
4. **`_get_key_positions` 沿序列轴抽稀**：破坏与 `labels` 的对齐，且仅省一次线性投影
   （稠密注意力本身为 `O(T²)`）→ 移除，流在完整序列上运行。
5. **CE 被计算两次**（`_compute_reasoning_loss` 一次、`forward` 分块一次）
   → 单一目标函数（引理 4/5）。
6. **LSH 哈希与排序结果被丢弃**（分桶机制名存实亡）→ 桶排序分块注意（定理 3）。
7. **`chunk_q/k/v` 二次投影**（`forward` 与 `_lsh_chunked_attention` 各投影一次）
   → 由注意力内部单次投影，接口收敛为 `states`。
8. **未使用的 `linear_attention` 模块与 `YvDeepMultiPathReasoningEngine` 重复引擎**
   （携带同一缺陷且无处引用）→ 删除，保持单一实现来源。
9. **`depth_controller` 定义后从未参与决策**（死参数）→ 深度由控制器请求、复杂度估计封顶
   （`forward:555-563`，长序列另受 `LONG_SEQUENCE_ABSTRACTION_CAP` 约束）。
