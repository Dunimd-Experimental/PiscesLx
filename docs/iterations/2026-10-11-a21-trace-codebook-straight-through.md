# 2026-10-11 · A21：寻址码本从不学习 → 直通门（straight-through gate）

关联：`docs/audit/00-audit-report.md` §48（登记）、`docs/proofs/03-trace-codebook-straight-through.md`（证明）、
`tools/test/test_sbf_addressing_gradient.py`（机检，已进入 `python manage.py test --stage 10`）。

---

## 假设与第一次尝试（已证伪，如实留痕）

**假设 H1**：给码本加一个可微辅助目标即可——把选择器自己选中的块当标签，
`L_align = CE(softmax(s), c)`，折进 `total_aux_loss` 就会被优化。

**实测否证 H1（两点，都是硬数据）**：

1. `loss=18.2193 ≠ lm(12.1347) + aux`，且 `aux_loss` 由 2.27 抬到 **26.174**，而
   `block_field.addresser.traces.grad` 与 `query_proj.weight.grad` **仍然是 None**。
   查 `model/core/model.py` 得组合点是 `loss = lm_loss + reasoner_loss`（再 `+ mtp_loss_weight·mtp`），
   **`total_aux_loss` 只被报告、从不进 loss**。MoE 负载均衡之所以还训练，是因为
   `tools/train/core.py:2549` 走的是另一条收集通路（`PiscesLxMoEGradientOperator` 的
   `total_auxiliary_loss`），不是模型输出里那个 `aux_loss` 字段。
   → 把项park 在 `aux_loss` 里等于什么都没做，且让日志上的 aux 变成一个没人优化的数。
2. 即便接通了，`CE(softmax(s), argmax s)` 的梯度是 `p − onehot(c)`，**与块是否有用无关**：
   它只放大当前赢家（winner-take-all），固定点是"一个块吃掉所有 query"，
   会直接破坏"被激活的知识块参与训练"。这不是创新，是会制造 A5 那类退化。

**结论**：H1 撤销。要的不是"再加一个 loss"，而是把**被选中块对目标的实际贡献**这条
因果通路接回打分。

## 设计（最终形态）

`read_and_modulate` 里给注入加一个直通门 `g = 1 + (p_c − stopgrad(p_c))`，`p = softmax(s)`：

- **值恒等于 1**（证明 P1）→ 前向逐位不变，推理不付任何代价；推理模式下门根本不构造。
- **梯度是软选择在硬点处的路径级梯度**（证明 P2，`docs/proofs/03` §2 有推导）：
  `∂L/∂s_{t,n} = ⟨∂L/∂h'_t, Δ_t⟩ · p_c(1[n=c] − p_n)`。
  也就是说码本被推向"它的调制真的降低了 LM loss"的块，方向由数据决定，不是自指。
- **必要性（证明 P2 的对比项）**：真软选择要读全部 N=628 块 ≈ 6.9e11 FLOPs/token，
  比 7B 主干前向贵 ~500×，1500 tok/s 直接报废；门只需要**已经读过**的那一块。
- **有界性（P3）**：`|∂L/∂g| ≤ ‖∂L/∂h'‖·κ‖h‖`，κ = α(γ_max+c) 来自 Theorem 5，
  是配置量而非新调参——不引入新超参（§9 禁冗余）。

改动文件：`model/core/block_field.py`（门 + 用后向安全的诊断）、
新增 `tools/test/test_sbf_addressing_gradient.py`。
撤销的半成品：`block_addresser.alignment_loss`、`model.py` 的 `sbf_alignment` 接线、
`model/config.py::sbf_alignment_weight` 全部删除——不留新旧并存。

## 结果（机检）

`python -m tools.test.test_sbf_addressing_gradient` → **8/8**：

| 检查 | 实测 |
|------|------|
| p1.gate_value_is_one | `max|g−1| = 0.000e+00`（128 个位置） |
| p1.forward_bitwise_equal | `max|h'_grad − h'_no_grad| = 0.000e+00` |
| p2.gradient_identity | `max|diff| = 1.543e-09` vs `max|ref| = 1.489e-03`（相对 1.0e-6） |
| p3.credit_bounded | `worst |dh|/|h| = 0.026062 ≤ κ = 0.075000`，`|dL/dtraces| = 7.28e-06 > 0` |
| p4.traces_update | 6 步 SGD 后 `max|traces − traces_0| = 1.445e-04 > 0`（A21 的症状正是 0） |
| p4.query_proj_trains | `|dL/d(query_proj.weight)| = 3.65e-06 > 0` |
| p4.no_gate_at_inference | 推理路径不构造概率张量 |
| p4.hard_path_is_dead | 阴性对照：同一目标喂给 detach 的硬读取 → `traces.grad is None` |

阴性对照是这一组断言的关键：它证明上面通过的检查**归因于门本身**，不是别处的接线变化。

`python manage.py test --stage 10` → **12/12 suites**（新套件已被自动发现并计入验收）。

## 分析

A21 的根因不是"缺一个 loss"，是**离散选择的梯度断裂**，而 `model.py` 的
`total_aux_loss` 又是"只报告不优化"的桶，两者叠加会让人误以为"加了项却没效果=机制不行"。
教训两条：接任何辅助项之前先确认它接的是**被 backward 的那个标量**；
诊断量的语义要和它的真实通路一致（否则日志会撒谎）。

## 下一步

1. 端到端机检：跑 60 步真实训练（`PISCESLX_HOME=/e/pisceslx/a21`，隔离工件根），
   用 `tools/dev/verify_checkpoint.py` 断言 `block_field.addresser.traces` 出现在
   **被更新张量集合**里——这是 §48 里我自己写的验收口径，必须用它验收。
2. 顺带登记的新缺陷（本次读到）：`model/core/model.py` 的 `total_aux_loss` 报告了层内 MoE
   辅助项，但真正的优化通路在 `tools/train/core.py` 的 MoE 算子里，两处口径重复且
   `aux_loss` 字段对使用者有误导性 → 需要单一事实来源（A22）。
3. `ensure_all_modules`（`model/core/model.py:1500`）**全仓无人调用**；
   实际只 materialise 了 `TRAINING_COMPONENTS = (block_field, long_context, mtp_heads)`。
   rca/crv/vericot/seer/task_head/eval_head/reasoner/agentic 若在 optimizer 建立后才首用，
   它们的参数就永不训练——这与"首份权重里 91 张量从未更新"高度吻合，是下一条要查的主线。

---

## 追加状态（同日 07:19，交接用，含一条未验证改动）

端到端验证**尚未取得证据**，机制级结论不受影响。三次启动的实测账目：

1. `--config configs/train/pretrain_cpu.yaml` → `No model.txt found`，**exit 0**。
   → 缺陷 A24：致命配置错误不返回非零码，CI 会把失败读成成功。
2. 同旗标 + `--dataset data/corpus_smoke` → 被设备契约拒绝
   （`mixed_precision=bf16`、`flash_attention` on cpu）。
   判据不是我猜的：日志里 `Host memory budget declared` 一次都没出现、
   我新加的传播分支也没打 `Stage config declares`，而 `Applied top-level config from model config file` 出现了
   —— 说明 **`manage.py train --config` 读的是模型配置文件，stage 文件的 `training:` 段
   （`memory_budget_gb/seed/max_steps/mixed_precision/flash_attention`）被静默忽略**。
   → 缺陷 A23：旗标语义混淆 + 静默半忽略，正是 §42 契约要消灭的"声明存在、实际不接线"。
3. 改用 `--train_config` 的运行正在跑（`E:\pisceslx\a21\run30b.log`，`python -u`，工件根隔离）。

**未验证改动的处置规则**（不要留死代码）：`tools/train/core.py:286` 的
"stage 声明值传播到 model config 之后再 enforce" 只在 `run30b.log` 出现
`Stage config declares ...` 且随后出现 `Host tier resolved` 时成立；
若这两个日志行都不出现，该分支从不触发，必须撤销，改为在配置加载层收口（单一事实来源）。

**A21 的关闭条件不变**：`tools/dev/verify_checkpoint.py` 的被更新张量集合里必须出现
`block_field.addresser.traces`。在那之前 A21 记"机制已证、端到端未验"。

## A25 取证（闸门失效的确切位置，供下一条施工直接接手）

`run30b.log` 的因果链是完整的：

```
27: [INFO] Host memory budget declared: 3.00 GB (free commit 0.36 GB)
40: Traceback ...
68: RuntimeError: DefaultCPUAllocator: not enough memory: you tried to allocate 388213760 bytes
    (self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, ...))
```

声明 3.0 GB、实测可用提交电荷 **0.36 GB**，闸门**没有拒绝建模**，于是 388MB 的原生分配直接把进程打死。
`utils/host_memory.py:199-205` 的判定式本身是对的：

```
limit = min(available=0.36, budget=3.0) = 0.36
if limit < max(required_gb, MINIMUM_USABLE_COMMIT_GB): raise     # 0.36 < 3.0 → 必然应该抛
```

所以失效点**不在这个比较**，而在**它根本没被调用**：`tools/train/core.py:695-702` 的
`reserve_host_memory_for_build()` 在 `require_headroom(...)` 之前有一段 `if ...: return`
早退分支（读取时的第 700 行 `return`），`required_gb` 取的是 `self._memory_budget.budget_gb`。
下一步施工：(1) 打印/断言该早退条件在 `budget_gb=3.0` 且存在模型待建时**不得**成立；
(2) 加机检——`PiscesLxHostMemoryBudget` 在"声明预算 > 可用电荷"的组合下必须抛
`PiscesLxHostMemoryBudgetError`，并且训练器在**任何 `nn.*` 构造之前**必须经过一次
`require_headroom`（可用一条 AST/顺序断言写成 `tools/check` 或 `tools/test` 的契约）。
同类既往证据：A20 的教训是"机制生效要看它自己发出的日志行"，A25 是它的强化版——
**日志行出现了（declared），但强制动作没发生**，所以判据不能停在"有这行日志"，
必须断言"拒绝路径在低电荷下真的被走到"。


