# embedding / MoE 核心 / 长上下文簇 承重地图（A63–A84）

> 来源：子代理全文通读 `model/core/embedding.py`(1282)、`model/moe/layer.py`、`model/moe/gate.py`、
> `model/moe/expert.py`、`long_context.py`、`csa_hca.py`、`plt_trie.py`、`mr_rope.py`、`duo_attention.py`、
> `eg_mla.py`、`predictive_delta.py`、`mhc_lite.py`、`memory_attention.py`、`ewc.py`、`layer_route.py`，
> 并核对 `model/core/model.py`、`blocks.py`、`hybrid.py`、`cache.py` 的调用点。它是只读代理，故由主线落盘。
> 接续 `docs/audit/03`（A26–A48）、`docs/audit/04`（A49–A62）。

## 0. 一句话结论

**主干的"嵌入层"整份文件没被用过**（真嵌入是 `model.py:745` 的裸 `nn.Embedding`），
**MoE 的全部辅助损失都不进被优化的标量**，
**长上下文簇里 8 个文件永不被构造、2 个文件无条件构建却从不被调用**。
`model/` 里 **≈5,000 行是死重**（2,683 行整文件级 + ~393 行死 router + ~1,036 行死 gate +
732 行 `online_router.py` + ~120 行 `YvHiClRouter`），外加三组"注册了但没有任何梯度"的参数集。

## 1. embedding.py：整个模块不可达

| 类 | 行 | 判定 |
|---|---|---|
| YvEmbeddingConfig / YvSinusoidalPositionEmbedding / YvLearnedPositionEmbedding / YvModalityEmbedding / YvTokenEmbedding / YvUnifiedEmbedding / YvMultimodalEmbeddingProjector | 138 / 245 / 378 / 527 / 700 / 857 / 1115 | **全部不可达**（全仓零构造点，只有 `model/core/__init__.py:208-215`、`model/__init__.py:183-190` 的 re-export） |
| YvAdaptiveEmbeddingScaling | 166 | 仅本文件内部使用（1282→807，而 807 的宿主本身不可达） |

- **A63 地雷**：一旦被构建，`_build_cache(10485760)`(330，由 `YvEmbeddingConfig.max_position_embeddings=10485760`(155) 驱动)
  分配 2×10.5M×hidden 的 fp32 缓冲（hidden=2048 时 **≈86 GB**）；`YvLearnedPositionEmbedding:460` 同样建 10.5M×hidden 表。
- **A64 模态索引空间不自洽**：`MODALITY_IDS`(584-591) 有 6 项，`cfg.n_modalities=5`(config.py:960)，
  `gate.py:246` 的 `expert_modal_affinity` 形状按 5 建 ⇒ `modal_id≥5` 时 `gate.py:366` IndexError。
  且 `YvMultimodalEmbeddingProjector` 收了 `doc_dim/agentic_dim`(1183-1184) 从不使用，`forward` 映射表(1271-1276)
  里**没有 document/agentic** ⇒ 与它自己的 docstring(1127-1128) 承诺的模态冲突（这条正是"模态=4"口径要扫干净的残留形态）。
- **A65 词表绑定名不副实**：`tie_word_embeddings=True`(config.py:869) 在 `model.py:926-928` 生效，**但默认仍然另行分配多份 vocab×hidden**：
  `mtp_heads` 先建 4 份 `Linear(hidden, vocab)` 再重绑(1433-1440)、投机解码默认开
  （`model.py:1320/1333` → `speculative.py:698→747`）另加**未绑定**的 `nn.Embedding`(335) 与 `lm_head`(348)、
  `YvModelForMaskedLM:2996` 又是一份未绑头（该类全仓未实例化）。⇒ "tied 开启"时至少 4 份 vocab×hidden 存在。
- `resize_token_embeddings`(1077) 无人调用，**且它不重绑 `lm_head`** ⇒ 一旦被调用会让词表与输出头脱钩（活的 resize 是 `model.py:1568`）。

## 2. MoE 核心：只有一个 router 活着，其余全是装饰

`_build_router`(`layer.py:754-777`) **无任何分支**，无条件构造 `YvMoEGate`(`gate.py:103`)；其 docstring(760-766) 声称按配置选择，是假的。

| 路由/门 | 行 | 判定 |
|---|---|---|
| YvMoEGate | gate.py:103 | 唯一可达 |
| YvExpertChoiceRouter / YvFineGrainedRouter | layer.py:124 / 262 | 不可达 ⇒ `moe_fine_grained=True`(config.py:322) 与 `moe_num_sub_experts=4`(323) 在 628-629 存下来后**无处使用** |
| YvStableMoEGate | gate.py:831 | **构造即孤儿**：`model.py:774` 建它（`use_path_moe=True` config.py:336），只交给 `YvTransformerBlock`(1159)，而默认主干 `YvHybridBlock`(1132) **不收 gate** ⇒ 参数被注册、永不前向、`requires_grad` 从未被使用；其 loss 恒 0.0(1144/1153/1231) |
| YvMoELayer / YvExpertOrientedRouter / YvUltraSparseGate | gate.py:1392 / 1234 / 2023 | 不可达（逻辑被 228-241、251-261、390-395 内联复制） |
| YvPathMoEGate | gate.py:1740 | 模型树内不可达（只有测试）；`layer.py:112` 导入它却不用 |
| YvPhiBalancing | gate.py:1825 | **A66**：`layer.py:653` 构造（`use_phi_balancing=True` config.py:344）但 **`forward` 从无调用** ⇒ `expert_bias` 从不生效、`routing_ema` 从不更新、`get_loss()`(1901) 恒 0.0；而 `blocks.py:1127/1253/1709` 调的是 **不存在的方法** `get_regularization_loss()` ⇒ 一旦走 legacy 主干即 `AttributeError`(2316) |
| YvOnlineClusterRouter / YvHiClRouter | online_router.py:161 / hicl_router.py:89 | 不可达（732 行 + ~120 行）；`use_hicl_router=True`(config.py:981) 无人读 |

**路由机制实况**：token-choice top-k(`gate.py:398-404`)，**不是** expert-choice；
前 `moe_random_to_gradient_steps=2000` 步**均匀随机路由**、权重 1/k(174,399-401)；
动态 top-k(478-507) 可把声明的 `moe_top_k=8` 悄悄降到 **3**，因为 `dynamic_top_k_max=3` 是构造函数硬编码默认(152)、**config 里没有这个键**；
容量是软的（训练期乘性重标定 509-538 + Python 循环再分配 728-828，带 `.item()` 同步）；
共享专家混合 `F.softmax(self.shared_expert_weight)`(859) 之后 `_apply_skip_residual`(741-753) 又加回
**上一次前向**的 detached 共享输出（`SKIP_RESIDUAL_WEIGHT=0.1`(739)，叠加在 943 的 `+shared_output` 上）⇒ **共享通路被双重计入**。
`compute_mode`/`PURE_LOOKUP` 只在 `moe_compute_mode=="compute_split"`(646-648) 生效，默认 `"default"`(config.py:325)
⇒ `expert.py:270-272` 的 `requires_grad=False` **实践中从不触发**。
`noise_std` 被下限 0.2 顶掉(159) ⇒ `moe_noise_std=0.1`(config.py:310) 与 `layer.py:774` 的默认被静默忽略。

- **A67（最重要的一条）所有 MoE 辅助损失都不进目标**：`gate.py:434-448`（load-balance + z-loss + SoftMoE budget + sparsity）
  → `layer.py:1357` φ-loss → `1388` diversity → `hybrid.py:1345-1348` SSM/attention 稀疏 → `model.py:2346/2358` 累积
  → 只作为返回值里的 `"aux_loss"` 键(2526)。**全仓 grep：零读取者。** 被 backward 的标量是
  `loss = lm_loss + reasoner_loss + mtp·w`(2468/2489)。`tools/train/core.py:2528-2549` 反传的是
  `PiscesLxMoEGradientOperator` 自己重算的 `total_auxiliary_loss`(`opss/train/moe_gradient.py:871`)，且该算子是 opt-in(core.py:429)。
  ⇒ **P1 的"MoE 负载 CV ≤ 0.3"当前没有任何梯度在优化它**；这也解释 §48/A21 里"项加进 aux 却无梯度"的根因不是个例。
- **A68 稀疏门是安慰剂**：`_should_compute_aux_head`(282-295) 收下 `sparse_gate` 却从不读它
  ⇒ 4 个可学 `sparse_gate_*` 参数(270-273) 不门控任何东西；头只被 warmup 计数跳过。"GSA 式可学路由"不存在。
- **A69 `gate.py:274` 把 `gate_sparsity_threshold` 硬编码成 0.01**，无视 config.py:621 的真实字段（且该字段在 gate.py 里从未被读）。

## 3. 长上下文簇

| 文件 | 判定 | 关键证据 |
|---|---|---|
| `long_context.py` (222) | `YvOOMBContext` **可达**；**A70 `YvREFORM` 是安慰剂** | `use_oomb_context=True`(config.py:766)/`use_reform=True`(771)，建于 `model.py:1401/1396`；OOMB 真被用(2236)，而 REFORM 的 `compress_kv/gather_important/recompute_attention` **零调用点**——`model.py:1218-1222` 用一个 norm 启发式**伪造**了它的效果，模块本身从未触及；`_init_long_context_stack`(1162-1196) 自己也从未被调用 |
| `csa_hca.py` (467) | **A71 整文件死** | `use_csa_attention=True`(663) 无人读；`YvHybridAttention` 零构造 |
| `plt_trie.py` (350) | **A72 无条件构建、永不使用** | `use_plt_trie`(cache.py:345) 无人读 ⇒ `cache.py:1813` 恒建；三个包装方法(2275/2301/2319) 零外部调用 ⇒ trie 永远为空；还把 50000 词表先验烧进 180/242/301 |
| `mr_rope.py` (145) | **A73 死且被重复实现** | `use_mr_rope=True`(983) 无人读，`YvMrRoPERotaryEmbedding` 零构造；`norms.py:529/624` 另写了一份 `_mr_radices_for`，由 `attention.py:5297` 喂 |
| `duo_attention.py` (250) | **A74 死** | `use_duo_attention=False`(722) 无人读；`attention.py:5238-5239` 只取它的 ratio/buffer 键去跑自己的内联版；`143` 用 `getattr(self,...,8192)` 读一个从未被设置的属性 |
| `eg_mla.py` (237) | **A75 死** | `use_eg_mla=True`(718) 与 `attention_type="eg_mla"`(481) **都无人读**（只在 `model.py:149` 的 docstring 里出现）；全仓无 `eg_mla` 导入 |
| `predictive_delta.py` (508) | **A76 恒开 + 尺寸错 + 永不训练** | `use_predictive_delta`(cache.py:340) 无人读 ⇒ `cache.py:1802` 恒建，且 `hidden_size` 走 `getattr(config,'hidden_size',4096)`（`YvCacheConfig` 无此字段、`model.py:726-739` 也不传）⇒  predictor 恒为 `Linear(4096→512)`；喂真实 `[B,S,H]`(cache.py:1870/model.py:2343) 时 `hidden_size≠4096` 即崩（**不确定**：只在 OOMB 分块分支观察到）；`254-260` 在 `no_grad` 下算预测再 `detach`(296) ⇒ 这批 `n_layer×Linear` **从不接收梯度**；`_enable_delta` 硬置 True(151) |
| `mhc_lite.py` (84) | **A77 不可达 + 一开就崩** | `use_mhc_lite=True` 会在 `model.py:1408` 构建，随后 `model.py:1215` 读 `num_streams`，而类里存的是 `n_groups`(mhc_lite.py:55) ⇒ AttributeError；同一行还对 4-D 张量 `.expand(-1,n,-1)`；`q_proj/k_proj=Linear(1,1)`(58-59) 闲置 |
| `memory_attention.py` (51) | **A78 三个类都是空壳** | `__init__(cfg)` 丢弃 cfg，`forward` 返回 `None`，连 re-export 都没有 |
| `ewc.py` (167) | **A79 死** | `use_ewc=True`(740) 无人读，`YvEWC` 零构造 |
| `layer_route.py` (104) | **A80 仅 legacy 可达 + 孤儿参数** | `use_layer_route=True`(982) 被 `backbone_allow_legacy_blocks=False`(330, model.py:1120/1154) 挡死；建于 `blocks.py:1549`、`forward` 在 legacy 内有用(2205,2328-2330)，但 **`lora_adapt`(98) 零调用** ⇒ 4 组 `lora_q/k/v/o`(83-86) 分配了不被任何梯度使用 |

## 4. 其余静默退化（可直接转成机检）

- **A81 读了但 config 里根本没有的键**（默认值被永久钉死）：`moe_dropout`(layer.py:682,693)、
  `max_gpu_experts`(707 — config 里其实叫 `vram_max_experts_on_gpu`)、`moe_routing_temperature`/
  `moe_temperature_min`/`moe_temperature_max`(gate.py:177-179 → 3.0/1.5/5.0)、
  `diversity_ortho_weight`/`diversity_entropy_weight`/`diversity_variance_weight`(layer.py:1263-1265)、
  `layer_route_lora_rank`/`_lora_scale`/`_gate_reg`(blocks.py:1551-1553)、`residual_dropout`(hybrid.py:1230 vs config.py:392 `residual_dropout_p`)。
  ⇒ 已由 `tools/check/checks_config.py::NO-UNDECLARED-CONFIG-READ` 覆盖。
- **A82 声明却无人读的旗标**：`moe_aux_loss_free`(324)、`use_csa_attention`(663)、`use_eg_mla`(718)、
  `use_duo_attention`(722)、`use_ewc`(740)、`use_mr_rope`(983)、`use_hicl_router`(981)、`attention_type`(481)、
  `use_plt_trie`/`use_predictive_delta`。⇒ 由 `NO-DEAD-CONFIG-FIELD` 覆盖（WARNING 级，逐条清零）。
- **A83 状态泄漏与非确定性**：`YvMoEGate._path_moe_cache`(263-267,409-424) 的"4 层共享一段路由"是**每个 gate 实例自己的**，
  而每个 `YvDeepSeekMoELayer` 自有 gate ⇒ 默认主干下 PathMoE 从不跨层共享；
  `YvDynamicMoELayer._layer_count`(618-619) 不跨实例重置 ⇒ `is_first_layer` 每进程只触发一次；
  `_skip_buffer` 跨批次存活(750)；`_compute_aux_loss_free` 在计数填满前恒返回新 `torch.tensor(0.0)`(475-476)。
- **A84 硬编码应当来自配置的量**：`noise_std` 下限 0.2(159)、`dynamic_top_k_min/max=1/3`(152)、
  `load_balance_threshold=0.15`(153)、`bias_update_rate/freq=0.05/10`(188-189)、温度阈值 50/10/0.1/0.05(658-667)、
  `SKIP_RESIDUAL_WEIGHT=0.1`(layer.py:739)、`expert_embed_dim=max(64, hidden//64)`(gate.py:229)、
  `*0.01` aux 标量(layer.py:483/502, gate.py:440/573)、`8192` 序列截断(gate.py:300)、`128` 流式锚点行(duo 146-151)。
- 只读代理未覆盖 `tools/test/`，故"是否被测试调用"一律按未验证处理；标注"不确定"的两处（A76 的崩溃条件、
  `cache_config` 是否由外部注入器提供）**不作为结论**。
