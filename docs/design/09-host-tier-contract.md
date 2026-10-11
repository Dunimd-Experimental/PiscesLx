# 主机层契约 · CPU 与 T4 的深度适配（设计，2026-10-11）

## 0. 为什么要单独立这一层

本轮在同一台 16GB 无 GPU 主机上实测到的事实，逐条都推翻了"配置里打开关即可"的假设：

| 实测事实 | 出处 |
|---|---|
| 建模期预分配 1000 块分页 KV 池 ≈ **8.4 GB**，与模型规模无关（0.01B 也照撞），且几何取默认 32 层/32 头/128 维 | `model/core/model.py`、`model/core/cache.py`（docs/audit §30） |
| 400 步 0.01B 真实峰值 **2.51 GB**（20 步测量只得 2.07 GB，因为不含检查点写入） | docs/audit §39 |
| 白天可用提交电荷在 **3.41–5.08 GB** 之间浮动 | `utils/host_memory` |
| CPU 吞吐 **9–14 tok/s**，而 P1 要求 T4 上 ≥1500 tok/s —— 差两个数量级 | 训练日志 |
| `F.scaled_dot_product_attention` 的 `scale` 只接受 Python float，把可学习标量读出计算图 | `model/core/attention.py`（§39） |
| `sparsity_reg_weight = 1e-6` 等魔数写死在代码里，未过配置 | `model/core/hybrid.py:1235` |
| `memory_budget_gb` / `seed` 曾声明但从未落到配置对象（手写键清单遗漏） | `tools/train/config.py`（§38） |

结论：**"在什么设备上跑"不是一个开关，而是一组互相约束的能力与预算**。
现状是三十几个独立布尔各自 `getattr`，没有单一契约，所以"声明与实际不符"会不断复发。
本设计把它们收成一个校验过的层对象。

## 1. 契约对象

```
configs/host/cpu.yaml        # 主机层（本机）
configs/host/t4.yaml         # 加速器层（阶段 7）
model/host_tier.py: YvHostTier / YvHostTierResolver   # 已实现（§42）
```

字段分四组，**每组都必须被显式声明**（默认值一律禁止，避免 §38 类事故）：

1. **预算**：`host_memory_ceiling_gb`、`checkpoint_retention`、`peak_measurement`（`rss|peak_commit`）、
   `abort_policy`（`checkpoint_then_abort|raise`）。
2. **数值与内核**：`compute_dtype`、`state_dtype`、`attention_backend`
   （`sdpa_math|sdpa_flash|eager`）、`ssm_scan`（`chunked_parallel|sequential`）、
   `kernel_target`（`none|sm_75|sm_90`）。
3. **状态压缩**：`optimizer`（`ink|adamw`）、`momentum_bits`、`variance_codec`
   （`log_grid|absmax`）、`variance_bits`、`sparse_ratio`、`sparse_warmup_steps`、
   `trust_region_cap`。
4. **知识域（查算分离）**：`block_residency`（`in_process|mmap_host_backing`）、
   `resident_blocks`、`transfer`（`none|async_stream|prefetch_ladder`）、
   `film_injection`（必须为 `on`，见 §3）。

## 2. 两层的取值与理由

### CPU 主机层（`configs/host/cpu.yaml`）

- `attention_backend: sdpa_math`。Turing/FlashAttention 路径不适用；
  `flash_attention: false` 已在主机配置声明，但必须由层对象统一校验，
  不允许"配置说 flash=true 而设备没有"（现在只 `_LOG.warning`）。
- `ssm_scan: sequential` + **强制把 `YvFlashSSM` 的 B/C 宽度与 `d_model` 解耦**
  （已修：`d_state>d_model` 时旧代码静默切出空张量，§30）。
- `optimizer: ink`，`variance_codec: log_grid`（证明 02 定理 1），
  `trust_region_cap: 1.0`（定理 3）。CPU 上 8-bit 状态是**容量**手段而非速度手段：
  23M 参数的 fp32 Adam 状态在主机预算内可承受，但对 0.5B 层不可。
- `block_residency: mmap_host_backing`、`resident_blocks: 4`、`transfer: none`。
  主机层不追 P1 的 ≤5ms/token 调取指标，但**必须走同一份代码路径**，
  否则夜间 T4 上跑的是一条从未被 CPU 测过的分支（现在 `block_field` 的
  注入曾被 `**kwargs` 吞掉，就是这种分叉的代价）。
- `peak_measurement: peak_commit`（Windows `GetProcessMemoryInfo` 的
  `PeakPagefileUsage`），因为崩溃来自提交电荷耗尽而非工作集；
  `abort_policy: checkpoint_then_abort`（本轮 step 270 丢进度换来的规则）。

### T4 加速器层（`configs/host/t4.yaml`）

- `compute_dtype: fp16`。T4（sm_75）有 FP16 Tensor Core、**无原生 BF16 加速**，
  所以 `mixed_precision: bf16` 在这层必须被校验拒绝而不是静默降速。
- `kernel_target: sm_75` + 分块 selective-scan、多 CUDA stream、梯度检查点。
- `attention_backend: sdpa_flash` 的 Turing 替代方案（P1 要求 Prefill ≥3000 tok/s）。
- `transfer: prefetch_ladder`、`block_residency: mmap_host_backing`：
  314B 知识域（628×0.5B）不可能常驻 16GB 显存，寻址/预取阶梯是这一层的核心工作，
  也是阶段 6/7 尚未实现的部分。
- `optimizer: ink` + `sparse_ratio: 0.01`，
  目标显存峰值 ≤15GB、单步 ≤0.5s。

## 3. 两层的**共同不变量**（禁止按设备分叉的东西）

设备可以不同，以下四条不可以，否则"P1 在 T4 达标"不代表模型机制成立：

1. **`film_injection` 必须为 on 且可机检**。查算分离的注入被块类型静默丢弃时，
   整条"知识块参与训练"的断言就假了（实测：11.56M 参数算完即弃，§32）。
   验收：训练后 `verify_checkpoint` 的未更新张量集合中**不得包含**知识域参数。
2. **可学习标量必须在图上**。`NO-DETACHED-SCALAR-PARAM`（第 14 条禁令）在两设备上都跑。
3. **声明即生效**。配置装载走字段级通用应用（`_apply_declared_fields`，§38），
   每个机制生效都必须有它自己发出的日志行作为证据（不看结果数值反推）。
4. **确定性**。`seed` 属层字段；同种子两次运行的 `model_state_dict` 必须逐位相同
   （已验：567/567，§39），否则任何 P1/P2 指标都只是抽样而非结果。

## 4. 落地顺序（每步都带机检证据，不产出"只写了文档"的东西）

1. ~~`model/config.py::YvHostTier`~~ → **已实现为 `model/host_tier.py`**（`YvHostTier` +
   `YvHostTierResolver.resolve/contradictions/enforce`），强制点接在
   `PiscesLxTrainingOperator.__init__`（`_setup_mixed_precision` 之前，设备已解析完毕），
   并打印 `Host tier resolved` 证据行。设计取舍：**校验既有的 `mixed_precision` /
   `flash_attention` / `vram_*` 字段，不新开一套同名开关**，以免出现第二条轨（§9）。
   仍待做：`configs/host/*.yaml` 显式档位文件。
2. 校验器：`tools/check/checks_host_tier.py` —— 声明与设备能力矛盾即失败
   （如 `attention_backend: sdpa_flash` 且无 CUDA、`mixed_precision: bf16` 且 sm_75）。
3. `sparsity_reg_weight` 等魔数迁入层字段（`NO-MAGIC-NUMBER` 计数应下降，作为进度证据）。
4. CPU 层：`abort_policy: checkpoint_then_abort` 与 `peak_measurement: peak_commit` 固化
   （本轮已实现，待补测试）。
5. T4 层：主机后备场 + 异步预取阶梯（阶段 6 未完成项），随后才谈 P1 速度指标。

## 5. 与提示词的关系

GPU 那半边**不是**额外要求：阶段 7 与 P1（T4 ≥1500 tok/s、单步 ≤0.5s、显存 ≤15GB、
1M 输入 ≤15min/≤14GB）就是它。CPU 这半边确实没写，但它是当前唯一能运行的设备，
且它的三条产物——内存闸门、确定性、"声明即生效"——是 P0"真实、可证明、是一个整体"
在实践上的落点。因此本设计不新增功能，只是把已有的分散开关收成一个可校验契约。
