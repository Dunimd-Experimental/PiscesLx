# 推导 00 · 主机层内存预算与优化器状态体积

对应代码：`model/core/cache.py`（分页 KV 池）、`model/core/model.py::YvModel.__init__`（缓存配置构造）、
`opss/optim/ink/*`（状态压缩）、`utils/host_memory.py`（计量与闸门）。
所有数字要么是本机实测，要么由下列公式从声明量推出；**不允许引用未验证的量**。

## 1. 分页 KV 池的体积（构造期 8.4 GB 的来源）

单块字节数（`YvCacheBlock.__init__` 分配 key/value 两张张量）：

    B_block = 2 · n_heads · block_size · head_dim · bytes(dtype)

默认几何（`YvCacheConfig` 的 dataclass 默认值）：`n_heads = 32`、`block_size = 512`、
`head_dim = 128`、`dtype = float16 (2 B)`：

    B_block = 2 · 32 · 512 · 128 · 2 B = 8 388 608 B = 8 MiB

预分配块数（旧实现，`model/core/cache.py`）：

    max_blocks = (max_cache_size · n_layers) / block_size + 100
               = (8192 · 32) / 512 + 100 = 512 + 100 = 612
    n_pool     = min(1000, max_blocks) = 612

    **构造期提交 = 612 · 8 MiB = 4896 MiB ≈ 4.78 GiB**（key+value 合计）

实测的失败点是" allocate 4194304 bytes(=4 MiB，单张量) 失败"，与上式的单张量 4 MiB 一致；
进程当时已把可用提交电荷从 4.76 GB 吃到耗尽，量级与 4.78 GiB 的推算吻合。

要点：`n_layers/n_heads/head_dim` 来自 **`YvCacheConfig` 的默认值**而非模型本身，
因为 `model.py` 传的是普通 dict，而 `YvUnifiedCacheManager` 只保留属于 dataclass 字段的键
（`{k: v for k, v in config.items() if k in YvCacheConfig.__dataclass_fields__}`）。
所以**每一档模型**（0.01B 到 1T）在构造期都付出同一笔 4.78 GiB —— 与规模无关。
这不是"16 GB 主机的宿命"，是实现缺陷：修复后 0.01B 的建模 + 前向 + 反传 + 优化器状态
峰值实测 **2.51 GB**。

## 2. 优化器状态的体积（Ink 的意义）

参数量 N 的 fp32 Adam 状态字节数：

    S_adam = N · (4 B 权重 + 4 B 梯度 + 4 B m + 4 B v) = 16 N

Ink（8-bit 动量 + 8-bit 对数栅格方差 + 稀疏梯度选择率 ρ）：

    S_ink = N · (4 + 4 + 1 + 1) B + 分块参数
          = 10 N + N/B_size · 8 B        （每块 log_lo/log_step 两个 fp32）

取 `variance_block_size = 256`：`N/256 · 8 = 0.03125 N` 字节，可忽略，故

    压缩比 = S_adam / S_ink = 16 / 10.03 ≈ 1.60×（状态部分 8/10 → 动量+方差从 8 B 降到 2.03 B，即 3.94×）

对 0.01B 档（N = 23.30 M）：`S_adam = 372.8 MB`，`S_ink ≈ 233.3 MB`，省 **139.5 MB**。
主机层这点差别不是关键；到 0.5B 档（N = 0.5 G）就是 **8.0 GB vs 5.0 GB**，
在 15 GB 显存预算内是决定性的。

对数栅格的额外代价：每元素一次 `log` 与一次 `exp`，相对 absmax 编码是常数因子 overhead，
不改变 `Θ(N)` 复杂度（证明 02 引理 6）。

## 3. 白天内存闸的取值推导

闸门语义（`utils/host_memory.py`）：建模前要求 **剩余提交电荷 ≥ 声明预算**，
每步要求 **本进程峰值私有提交 ≤ 声明预算**。于是预算必须同时满足：

    budget ≥ 实测峰值            （否则正常训练被自己拒掉）
    budget ≤ 运行时的可用提交电荷  （否则建模请求被拒）

本机实测：
- 0.01B 一次 400 步（含检查点写入）进程峰值 = **2.51 GB**；
- 白天可用提交电荷区间 = **3.41 – 5.08 GB**（他机在用）。

可行区间为 `2.51 ≤ budget ≤ 3.41`，取 **3.0 GB**：留 0.49 GB 的峰值余量，
同时低于最坏情况的可用量。此前取 5.0（> 3.41）导致每次建模都被拒；
取 2.5（< 2.51）导致跑到第 270 步被中止并丢弃进度。两次错误都不是硬件限制，
而是**用低精度测量给硬约束定参**——20 步、无检查点写入的峰值只有 2.07 GB。

对 0.1B 档的夜间预算可同式外推：权重体积约为 0.01B 的 6.8 倍（158.45 M / 23.30 M），
若构造期缺陷保持已修状态，则

    peak(0.1B) ≈ 0.53 GB(torch 基线) + 6.8 · (2.51 − 0.53) GB ≈ 14.0 GB

超出本机 16 GB 的实际情况，故 **0.1B 必须等 GPU/T4 主机**；
这条外推在夜间实测前只是估计，不作为达标依据。

## 4. 检查点体积与保留

实测一份 `final_model.pt`（N = 23.30 M）的**解压后**体积构成：

    model_state_dict    567 张量  318.1 MiB   （> 4N = 88.9 MiB：含 SBF 常驻块、缓存等缓冲）
    momentum_codes      503 张量   22.2 MiB   （1 B/参数 = N，8-bit 动量）
    variance_codes      503 张量   22.2 MiB   （1 B/参数，8-bit 对数栅格方差）
    分块参数(3 组)      1509 张量    0.9 MiB   （scales + log_lo + log_step，每块 2 个 fp32）

即 **Ink 的压缩状态确实进了检查点**，且量级与 §2 的 `2N + N/B_size·64/8` 公式一致：
实测 `2 · 22.2 MiB = 44.4 MiB ≈ 2N`（N = 23.30 M → 46.6 MB，差值来自 503 个参数中
部分是 4-bit 打包/零长张量）。磁盘上文件是 221.9 MiB < 318.1 + 45 MiB，
因为 `torch.save` 的 zip 压缩；**不能拿文件大小当状态的证据**——
我第一次测时用顶层 `numel()` 求和，得到 `optimizer_state_dict = 0.0 MiB` 的错误结论，
差点对内写下"Ink 状态未持久化、断点续训不可用"。嵌套结构必须递归求和。
保留 `CHECKPOINT_RETENTION = 3` 意味着稳态磁盘占用上界 `3 · 221.9 MB ≈ 0.65 GB/档`；
A20（跨运行从不裁剪）修复后该上界才成立。

## 5. 每步参数覆盖率：`num_update_blocks` 的真实代价

`PiscesLxInkOptimizer.step()` 把当前有梯度的参数排成列表，按 `_num_blocks` 切窗，
**每一步只更新其中一个窗口**：

    m_k = len(params_with_grad)
    window = max(1, m_k / num_update_blocks)
    每步被更新的参数量 = window  →  覆盖率 = 1 / num_update_blocks

于是某参数在 `T` 个全局步里的**自身更新次数**约为

    t_param ≈ T / num_update_blocks

实测（`final_model.pt`，`global_step = 281`，`num_update_blocks` 当时无处声明、取默认 4）：
`ink.step` 里单参数的最大更新次数 = **76**。校验该式：`281 / 4 = 70.25 ≈ 76`
（差值来自不同步里"有梯度的参数集合"大小变化，窗口边界随之移动）——公式与实测量一致。

**这为什么直接污染对小步数运行的解读**：Adam 的偏置校正用的是**参数自身的** `t`：

    bc1 = 1 - β1^t ,  bc2 = 1 - β2^t

`num_update_blocks = 4` 时，全局 281 步对应 `t = 70`，即每个参数其实只经历了 70 次更新。
所以"400 步跑完 CE 还高于 uniform"里，有 **4 倍的有效步数缺失**，不能全归因于数据量或学习率。
反过来说：先前两次数值（首跑 11.2703、复跑 11.6965）是在 `t ≈ 100` 的有效性下测得的；
把覆盖率改成 1（`num_update_blocks = 1`）后，同样 400 步就是 400 次/参数的真实更新。
这条改动**必须用一次真实运行验证**，在拿到新的留出 CE 之前不算改进，只算假设。

**内存视角的取舍**：分窗本意是降低单步峰值（一步只动 1/4 参数 ⇒ 只反传/写回那部分状态），
但主机层实测瓶颈在**建模与检查点写入**，不在参数更新本身（§3：0.01B 峰值 2.51 GB，
其中状态仅 45 MiB）。所以对 `cpu` 档，1 是正确默认；
只有在状态写回成为瓶颈的大档上才值得 >1，而那属于 T4 层要实测的事。
