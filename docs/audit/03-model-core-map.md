# model/ 承重地图（逐层：可达 / 内部使用 / 不可达 / 重复 / 静默退化）

> 来源：对 `model/core/attention.py`（6163 行）、`blocks.py`（3580）、`norms.py`（1042）的
> **全文通读**（两个子代理各自逐行读完，回传带 `文件:行` 的证据）；`block_field.py`/`block_addresser.py`/
> `knowledge_experts.py`/`subconscious.py`/`dual_injector.py`/部分 `model.py` 由主线通读。
> 状态：**进行中**——`cache.py`/`hybrid.py`/`mamba3.py`/`embedding.py`/`model/layer.py·gate.py·expert.py`
> 以及 `model.py` 全文尚未入账（见 §5 待读清单）。
> 规模事实：`model/` 全树 **94 文件 / 69,810 行**；`getattr(cfg|self.cfg, '键', 默认值)` 形式的配置读取
> 共 **288 个不同键**（232 + 56），每一个都是潜在的静默错配点。

---

## 1. 注意力层（attention.py）

| 类 | 算法 | 判定 | 关键证据 |
|---|---|---|---|
| YvAttention (5186) | MLA+Duo+Hydra+LCA+DSA+Ring 融合 | **可达（唯一入口）** | 构造点 `blocks.py:1089,1229,1350,1580`、`hybrid.py:1076,1184` |
| YvAttentionSink (506) | 4 个可学 sink token | 可达 `5313` | `use_attention_sink` **不在 config.py 里** → 走默认值 |
| YvQKNormalizer (628) | RMS QK-norm | 可达 `5309` | `use_qk_norm`（config.py:405） |
| YvHydraHeadAttention (5794) | FA/LA 头分割 | 可达 | `blocks.py:1094,1234,1585`，`use_hydra_head`（config.py:353=True） |
| YvLatentCondensedAttention (5985) | 潜空间 KV 压缩 | 可达 `5317` | `use_lca`（config.py:362）；另有一份**死实例** `blocks.py:1110`（`self.lca` 从未调用） |
| YvAttentionMaskFactory (6140) | 缓存因果掩码 | 仅本文件内部使用 | 1285,1309,2144,2720,3295,4624,4885 |
| YvALiBi (320) | ALiBi 斜率+偏置 | 配置可开（5305，config.py:851=False）但**功能上死** | 见 A28 |
| YvAttentionBackend (179) / YvAttentionConfig (251) | 10 成员 str 枚举 / 33 字段数据类 | **不可达** | 全仓无 `.成员` 访问、无 `YvAttentionConfig(` 构造；`attention_backend` 不在 config.py |
| YvLinearAttention (748)、YvCirculantAttention (1198)、YvSlidingWindowAttention (1482)、YvSparseAttention (1680)、YvPagedAttention (1903)、YvFlashAttention (2165)、YvLocalGlobalAttention (2489)、YvMultiQueryAttention (3145)、YvDynamicH2OAttention (3320)、YvH2OAttention (3822)、YvHISAAttention (4268)、YvMixtureBlockAttention (4771) | 各一篇论文的名字 | **全部不可达** | 无任何构造点；`model/core/__init__.py:152-156` 只导出前三个类，其余 19 个连导出都没有 |

**结论**：22 个类里 **12 个不可达 ≈ 4000 行死重**。且**不存在任何分派表**——`YvAttention` 自己在 5201-5203
写明"无算法变体分支"，选择靠 `getattr(cfg, 布尔/整数)` 决定挂哪些子模块，外加**运行期硬编码长度阈值**：
LCA `kv_len>4096`(5660)、DSA `>1024`(5719)、Ring `>ring_attention_threshold`(5750，config.py:698=32768)、
对比增强 `>100000`(5776)。`attention_type`（config.py:481 默认 `"eg_mla"`）与 `YvAttentionType`（123-154，11 成员）
**没有任何消费者读取**，config.py:1088 只把枚举解包成字符串。

## 2. Block / Norm / MoE 层（blocks.py + norms.py + layer/gate/expert）

**主干分派真相**：`model.py:1120-1160` 由 `backbone_allow_legacy_blocks`（config.py:330 = **False**，
且没有任何 `configs/model/*.yaml` 打开它）决定——**永远走 `YvHybridBlock`**（1132）。只有它为 True 时才建
`YvTransformerBlock`（1154），其内再按 `use_parallel`→`YvParallelBlock`(1520/1777)、`use_deepnorm`→
`YvDeepNormBlock`(1522/1840) 分支。**可构造的 block 只有 3 种：hybrid / standard / deepnorm**；
`YvBlockType`(208，6 成员) 枚举不可达（1506 读进一个从不使用的属性），
`CROSS_ATTENTION/ADAPTIVE/MIXTURE_OF_DEPTHS` 这些枚举值**永不可选**。

不可达清单（blocks.py/norms.py）：`YvBlockConfig`(242)、`YvSwiGLU`(419)/`YvGeGLU`(473)（活 MLP 是
`YvSwiGLUExpert`，expert.py:484）、`YvLoRA`(524)/`YvDoRA`(578)（1515-1516 读了旗标、无 config 键、无构造点）、
`YvCrossAttention`(838)→`YvCrossAttentionBlock`(1327) **整条 enc-dec 路径死**、`YvExpertChoiceMLP`(921)
（与 layer.py:120 的 router 重复）、`YvManifoldConstraint`(2549)（Poincaré/球面/Stiefel/Grassmann 全套指数映射）、
`YvMHCTransformer`(3366)/`YvMHCLayerReplacement`(3459)/`YvMHCLoss`(3529) **整棵 mHC 树死**、
norms 侧 `YvLayerNorm`(211)/`YvAdaptiveLayerNorm`(241)/`YvGroupNorm`(276)/`YvDeepNorm`(699)/
`YvParallelResidualNorm`(801) 仅导出或"imported-but-unused"（blocks.py:196、hybrid.py:188）。
`YvNorms._residual_alpha_for_depth`(1024) **从未被调用**。

**MoE 实况**：`YvDeepSeekMoELayer`(layer.py:963 ← `YvDynamicMoELayer`:519)，构造点
`blocks.py:1123,1247,1358,1598` + `hybrid.py:1102,1209`；专家 = `YvExpert` SwiGLU ModuleList(layer.py:665-685) + 共享专家(687)；
top-k = softmax→topk→重 softmax(gate.py:387,403-404)，前 `random_to_gradient_steps` 步随机路由(400)，k 动态(379)；
aux = load_balance + z + budget + sparsity(gate.py:434-448)→layer.py:905/959→blocks.py:2309→`model.py:2223-2265`→`"aux_loss"`(2383,2526)；
专家今天**全部可训练**（`PURE_LOOKUP` 的 `requires_grad=False`(expert.py:270-272) 只在 `compute_mode="compute_split"` 生效，
而 blocks 从不传该参数 → layer.py:646-648）。

**关键**：`_build_router`(layer.py:754-777) **硬编码 `YvMoEGate`**，无论配置如何 ⇒
`YvFineGrainedRouter`(gate.py:262)、`YvStableMoEGate`(831)、`hicl_router`、`online_router` **全部未被构造**。

## 3. 缺陷登记（本轮新增，编号接续 §48/A21）

- **A26 查侧被做成外挂文件库**：`knowledge_experts.py::YvKnowledgeExpertPool` 从
  `knowledge_expert_path="./knowledge_experts"`（config.py:1026）按 `expert_{idx}.pt` 读文件、
  读完立即 `freeze()`（`requires_grad_(False)`）、整个 forward 包在 `torch.no_grad()` 里、
  文件缺失只 warning 一句 "using uninitialized expert" 继续跑。**这与"314B 是模型自身权重、被激活块参与训练"
  直接冲突，且与 SBF 是两套并存的池**；`use_knowledge_experts=True` 在两档 tier 被显式打开
  （config.py:1547、1578），而 7B 档写的是 `compute_params_total=7000000000, lookup_params_total=0`。
- **A28 开了 ALiBi = 没有位置编码**：`alibi_bias` 在 5747 算完后再未被使用，同时 5289 在 `use_alibi` 为真时**跳过 RoPE**。
- **A29 DSA 打分器永不学习**：5722 的 `with torch.no_grad():` 把重要性计算**和 topk 一起**包住，
  `dsa_importance_scorer`(5339) 拿不到梯度，尽管 `dsa_use_dynamic=True`。
- **A30 旗标被无视**：`use_duo_attention=False`（config.py:722）仍无条件执行 `_apply_duo_tiling`(5741)；
  `use_mla`、`use_flash_attention`(config.py:492) 同样不被读取即生效/失效。
- **A31 名字冒充机制**：`YvCirculantAttention._fft_attention`(1270) 与 `_fft_causal_attention`(1296)
  都只是调 `F.scaled_dot_product_attention`，**没有 FFT/循环矩阵数学**；`fft_len`(1403) 算了传进去不用；
  后者从未被调用。而 `use_circulant_attention: bool = True`（config.py:992）。
- **A32 Ring 在线 softmax 数学错误（>32K 为活路径，默认开）**：`chunk_max` 跨 ring 步不重标定(2973-2978)，
  与自身 docstring(2928-2930) 矛盾；因果掩码只在 `ring_step==0` 施加(2966)，非分布式模式复用同一份 K/V
  (2898-2900) ⇒ **能看见未来**；`_standard_ring_attention`(3129) 把各自独立的 softmax 直接相加、不重归一化，
  `chunk_normalizer`(3098) 闲置。
- **A33 sink 只在训练期生效**：5623、5772 —— 它存在的理由（推理期稳定性）从不发生。
- **A34 掩码静默丢弃**：5425-5435、5926-5929、6108-6110 在形状不符时 `attn_mask=None, is_causal=True`，
  padding 掩码可以无警告地消失。
- **A36 配置键错配族**（同类可批量机检）：`use_parallel`(1507) vs 配置里 `use_parallel_attention_mlp`(397) ⇒ parallel 分支死；
  `residual_dropout`(1364,1438,hybrid 1105/1230) vs 配置 `residual_dropout_p`(392) ⇒ **dropout 恒被钉在默认 0.1**；
  `swiglu_clamp`(1623) 读了不用（活的 SwiGLU 不 clamp）；`moe_aux_loss_free`(324) 从未被读，gate 照样发 aux。
- **A37 router 工厂硬编码**：见 §2，四个 router/gate 实现从未被构造。
- **A38 Phi balancing 双重计入**：实例化两次（layer.py:652-659 与 blocks.py:1127,1253,1709），**求和两次**
  （blocks.py:1194,1318,2316）。
- **A39 恒 0 的约束损失**：`get_constraint_loss` 返回常量 `0.0`（3233、3363）⇒ `YvMHCLoss` 的惩罚项
  （3571-3573）是看起来生效的 no-op。
- **A40 attn-res 历史被 detach**：2511、2540 ⇒ `attn_res_*_proj` 参数饥饿。
- **A41 四套残差数学并存**：`YvUnifiedResidual` 加性含 mHC(1456)、`forward_deepnorm`(1468) **替换**流、
  hybrid 的 `residual + dropout(scale*out)`(1338-41/1350-51)、`YvDeepNorm`(797)、`YvParallelResidualNorm` 门控(900)。
- **A42 重复 norm**：`YvLayerNorm(use_rms)` vs `YvRMSNorm` vs `YvGroupNorm` 的 RMS 分支(306-311) vs 裸 `nn.LayerNorm`(1675,3294-3296)。
- **A43 任务感知 RoPE 永远收不到任务嵌入**：`enable_task_aware`(norms 541-545) 从不被喂 `task_embedding`，
  因为所有 `self.rope(x, len)` 调用都只传两个位置参数（attention.py:5685-5716）⇒ 相关参数饥饿。
- **A44 同一旗标默认值自相矛盾**：`use_mhc` 在 1441 为 True、1517 默认 False、config.py:645 为 True。
- **A45 硬编码阈值族**（违反 §4.7，可机检）：`hidden_size>=2048` 自动切 SSM(1800)、`seq_len>8192`(2299)、
  gate floor 0.01(2301)、`0.05` 分布式混合(2289)、`sqrt(original_max)`(norms 583)、`1e6`(559/565)、
  `min(128, head_dim//4)`(model.py:1105)、`kv_len>100000` 的 0.15/0.25 对比缩放(5776-5780)、
  HISA 混合 0.5/0.35/0.15(4707-4709)、`scale=32`/`original_max_position_embeddings=4096`(5294-5295)。
- **A46 PQCache 非确定性静默量化**：`numel()>1e7` 才触发、k-means 质心用 `torch.randperm`、整体在 `no_grad` 下
  （2066-2104）⇒ 同种子不同形状给出不同数值行为，破坏确定性口径。
- **A47 死路径必崩**：`self.scale` 在 3760、4193 被使用，但两个 H2O 的 `__init__`（3450-3507、3932-3953）
  从未赋值 ⇒ `seq_len>streaming_window` 即 `AttributeError`；`YvDynamicH2OAttention` 的 `self.compressed_memory`(3655)
  未注册为 buffer/module。
- **A48 未证实但高度可疑**：`q[..., -d:].copy_(q_r)`(5710-5711) 只在推理分支；cache 侧
  `k[:, :, :self.n_kv_head]`(2437,3076) 是在已 `repeat_interleave` 的张量上切**头轴**——正确性存疑，
  子代理标注为"不确定/很可能错"，**不作为结论**，需用数值对照实验证实后再定性。

## 4. 对"创新"的直接含义

1. **必须先减后加**（§9/§14）：注意力与 block 层的不可达实现约 4000+ 行（还不算 mHC 树、enc-dec、
   四个 router、五个 norm）。在它们还在树里的时候再加新机制，就是指令书明令禁止的"架构堆砌冗余"。
2. **"7B 打 400B"的杠杆当前被 A26 掐断**：查侧走外挂文件 + freeze + no_grad ⇒ 314B 永远不会成为"自身权重"。
   这一条不清掉，查算分离的 P0 与 "MMLU +8" 无从谈起。
3. **可机检化的系统性收口**（下一件落地物，纯 CPU、不依赖训练）：
   * **CONFIG-KEY-MISMATCH**：AST 扫 `getattr(<cfg 变量>, 'k', default)`，`k` 必须是被检类（`YvConfig`
     或传入的数据类）的已声明字段；违规即 error。直接抓住 A36 全族。
   * **CONFIG-KEY-UNREAD**：声明了却全仓无人读的字段（`attention_type`、`use_flash_attention`、
     `moe_aux_loss_free`、`use_h2o_attention` 等）也要报，否则"配置即事实"是假的。
   * **UNREACHABLE-CORE-CLASS**：`model/**` 里既无构造点又不在分派表内的类 → 计数并进 CI，逼出删除清单。
   * **NO-DETACHED-CREDIT-LOSS**：对"打分器/路由/门控"类参数外层包 `torch.no_grad()` 的模式报警
     （A29/A40/A46 同一形状）。

## 5. 待读清单（尚未入账，勿在这些文件上下任何结论）

`model/core/cache.py`(2610)、`hybrid.py`(1356)、`mamba3.py`(1864)、`embedding.py`(1282)、`model.py`(3041 全文)、
`mamba_upgrades.py`、`long_context.py`、`csa_hca.py`、`plt_trie.py`、`mr_rope.py`、`duo_attention.py`、`eg_mla.py`、
`token_sparse_attn.py`、`predictive_delta.py`、`mhc_lite.py`、`memory_attention.py`、`ewc.py`、`layer_route.py`、
`subconscious.py`、`dual_injector.py`；以及 **`model/layer.py`、`model/gate.py`、`model/expert.py`**
（本轮从引用中发现的承重文件，位置待 `ls` 确认——我此前只统计了 `model/core/*.py`，是遗漏）。

## 6. 已落地：两条配置机检规则 + A25 修复（2026-10-11，实测）

**两条规则已进 CI**（`tools/check/checks_config.py`，注册在 `runner.py` 的 `_CHECKS`，
规则号在 `base.py`：`CONFIG_KEY_UNDECLARED` / `CONFIG_FIELD_DEAD`）：

- **`NO-UNDECLARED-CONFIG-READ`（ERROR）**：`getattr(<config>, '<key>', default)` 里 `<key>` 不被任何
  配置 schema 声明 → 报错。schema 的**定义**是"`@dataclass` 装饰的类、或类名以
  `*Config|*Settings|*Spec|*Args|*Options` 结尾的类的类级标注赋值"；`__init__` 里现赋的实例属性
  **故意不算声明**，否则"默认值读取自己造出属性"会把这个缺陷本身藏起来。
- **`NO-DEAD-CONFIG-FIELD`（WARNING）**：`YvConfig` 声明的字段全仓无人读。

**实测产出**（`python -m tools.check`，347 文件 / 16 规则 / errors 4118 / warnings 1）：

| 规则 | 命中 | 代表性键 |
|------|------|---------|
| NO-UNDECLARED-CONFIG-READ | **81** | `moe_dropout`、`moe_routing_temperature`、`max_gpu_experts`、`kv_cache_max_size`、`max_rca_rounds`、`rca_convergence_threshold`、`mamba3_use_mimo`、`ssm_state_dim`、`lyx_*`、`mhc_streams`… |
| NO-DEAD-CONFIG-FIELD | **1 条 / 115 个字段** | `activation_variance_weight`、`circulant_threshold`、`dynamic_h2o_*`、`chinchilla_*`… |

**误报核查（已做，结果：无假阳性）**：`residual_dropout` / `use_parallel` 虽然 `YvConfig` 不含，
但 `YvBlockConfig`（`model/core/blocks.py:303/308`）确实声明了它们，故规则**未**报——规则以
"任一 schema 声明即算已声明"为口径，宁可漏（跨类错配）也不误报。81 条命中抽查全部为真：
`kv_cache_max_size` 全仓无任何类级声明（与 §40 "传进去的键被缓存管理器丢弃" 同一根因）。

**A25 已修**（真因不是 `reserve_host_memory_for_build` 里的早退，而是**调用时机**）：
`tools/train/core.py` 的两处 `self.model = model_class(**model_kwargs)`（cache 路径 813、主路径 856）
都在 `_materialize_model_components()`（其中才调 `reserve_host_memory_for_build()`，857）**之前**执行。
即"构造"这一步本身就是原生分配，而闸门在构造之后才检查 → 0.36GB 可用提交电荷撞上 3.00GB 声明预算时
必然死在 `DefaultCPUAllocator: not enough memory`，闸门只来得及报告它本该拦下的那次崩溃。
修法：在 `Step 2a` 之前、任何 `model_class(**kwargs)` 之前**单点**插入 `self.reserve_host_memory_for_build()`，
两条构造路径同时被覆盖。`_materialize_model_components` 里的那次保留为第二道闸。
