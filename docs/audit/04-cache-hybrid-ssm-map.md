# cache.py / hybrid.py / mamba3.py 承重地图（A49–A62）

> 来源：子代理对 `model/core/cache.py`(2610)、`hybrid.py`(1356)、`mamba3.py`(1864)、
> `mamba_upgrades.py`(211)、`token_sparse_attn.py`(174) 的**全文通读**，含 `model.py`/`blocks.py`/
> `config.py`/`predictive_delta.py` 的调用点核对。它是只读代理，故由主线落盘。
> 未验证项在 §6 明示，不当结论用。接续 `docs/audit/03-model-core-map.md` 的编号。

## 1. 缓存层：五个子管理器全被绕开

| 类 | 判定 | 证据 |
|---|---|---|
| YvUnifiedCacheManager (1760) | **可达，唯一活路径** | 只用到 `get_kv_cache`/`update_kv_cache`/`compute_pending_prediction`（`blocks.py:2227,2343`、`hybrid.py:1252,1263`）；真实实现是 `kv_cache` dict-of-blocks + `_concat_recent` |
| YvPagedCacheManager (535) | 构造了但**被绕开** | `allocate_sequence/append/read` 全仓零调用者 |
| YvSSMCacheManager (871) | 被绕开 | `get_ssm_state/update_ssm_state`(2240,2243) 无调用者 |
| YvSpeculativeCacheManager (1040) | 被绕开 | 包装器 2246-2259 无调用者 |
| YvMultimodalCacheManager (1248) | 被绕开 | 包装器 2261/2264 无调用者 |
| YvCacheCompressor (1404) | 6 个选择算法里 5 个死 | 仅 `compress_importance_based` 经 `compress_cache`(2273) 可达，而 2273 无调用者 |
| YvCacheType (173) / YvKVCacheRing (2433) | **不可达** | 无分派表；Ring 连 re-export 都没有，且名不副实（只增不换） |
| YvLynxKVQuantizer (2480) | **不可达 + 到了就崩** | 由 `flagship_level>=0.5`(2185) 门控，而该字段在 `YvCacheConfig` 上不存在 ⇒ 恒 0.0；且调用是 `quantize(k,v)`(2187) 而签名是 `quantize(tensor)`(2507) ⇒ TypeError |
| YvSTARKVAdapter (2561) | 不可达 | `flagship_level>=0.6`(1841) 同上恒假 ⇒ `is_enabled()` 恒 False |

- **A49 H2O 缓存链断在注入点**：`get/set_h2o_cache` 在 `attention.py:4045/4124` 有钩子，但
  `hybrid.py:1245` 的 `set_cache_manager` **零调用者**（`model.py:795` 直接赋属性）⇒ 钩子恒空转。
- **A50 配置键不存在即回落硬编码**：`YvCacheConfig` 由 `getattr(cfg,'kv_cache_max_size',2048)` 构造，
  而 config.py 里没有 `kv_cache_max_size`（有 `max_cache_size=8192`，从未被读）⇒ 所有档位都用字面量 2048；
  `cache.py:1800,1833` 从 `YvCacheConfig` 上 `getattr(config,'hidden_size',4096)` ⇒ **每一层的 predictor LoRA 都按 4096 宽建**；
  `cfg.cache_config` 这个键在 config.py 也不存在。
- **A51 67M 参数游离在 parameters() 之外**：`YvPredictiveDeltaCoder` 的 `num_layers × Linear(hidden→kv_lora_rank)`
  无条件构建（`use_predictive_delta` 从未被读，cache.py:340），且挂在**非 nn.Module** 的管理器上（1760）
  ⇒ 不进 `model.parameters()`，永不训练；又用字面量 `device='cpu'`(predictive_delta.py:178-183)，构建即 ≈268MB。

## 2. 主干与 SSM：唯一可达的块类型是 YvHybridBlock

`model.py:1132` 由 `backbone_allow_legacy_blocks`（config.py:330 = **False**）决定 ⇒ **每层都是 hybrid**；
`use_mamba3`(517)/`mamba3_layers`(518) 只影响永不执行的 legacy 分支(1139-1152)。
不可达：`YvJambaBlock`(1046)、`YvProgressiveHybridGate`(563)、`YvAdaptiveRouter`(741)、
`YvHierarchicalFusion`(890)、`YvHybridConfig`(239)、`YvHybridMode`(195)——在 hybrid.py 自己内部都零引用。
`YvMamba3Stack/Encoder/Decoder`、`YvParallelScan`(868)、`YvChunkedParallelScan`(928)、`YvHierarchicalSSM`(1288) 仅 re-export。

- **A52 配置无法选择任何 SSM 变体**：`mamba3_complex_state`、`mamba3_trapezoidal`、`mamba3_mimo`、
  `mamba3_use_v_kernel`、`mamba3_use_ss_duality`、`mamba3_use_adaptive_dt`、`mamba3_use_flash_ssm`、
  `mamba3_use_gated`、`mamba3_use_bidirectional`、`mamba3_chunk_size`、`mamba3_gate_mode/init/temperature`、
  `mamba3_use_fast_path`、`mamba3_conv_bias/proj_bias`、`mamba3_layer_norm_eps`、`mamba3_dropout`、
  `use_sparse_ssm`、`sparse_ssm_ratio`、`hybrid_total_steps`、`sequence_threshold` 在 config.py 之外**无消费者**；
  `hybrid.py:1186-1192` 只把 26 个字段中的 5 个传给 `YvMamba3Config`，而 `YvMamba3Block`(1551-1563,1584-1629)
  **无条件构建并运行全部子 SSM** ⇒ 即便键名正确也是惰性的。`use_trapezoidal` 的模块(1553)建了不调用。
- **A53（最严重）SSM 的输入不依赖输入**：`mamba3.py:904,905,967,968,1033,1034,1082,1137,1280,1332,1665`
  在 forward 内部用 `torch.randn` 生成 `B/C/delta` ⇒ 这些路径**与 token 无关、不可训练、破坏评测可复现性**。
  这些正是"唯一可达块类型"里的 SSM 分支。
- **A54 padding 污染状态**：`YvMamba3Block.forward` 忽略 `attention_mask`(1571)。
- **A55 三套互相矛盾的 d_state**：`blocks.py:1804` `max(64, n_kv_head*16)`（且 1800 硬编码 `hidden_size>=2048` 才走 SSM）、
  `cache.py:928` `head_dim*2`、`hybrid.py:1196` 默认 16，而 config 声明 `mamba3_d_state=128`；
  `mamba3.py:1462-1463` 的 `BC[..., :d_state]` 只被一个**空洞检查**(1457) 保护。`YvSelectiveScan` 默认 `d_state=16`
  被 Bidirectional/Gated/MultiHead/SSDuality/Hierarchical 静默继承(1012,1013,1063,1116,1266,1303)。
- **A56 `A_log` 梯度被丢**：`YvAdaptiveDiscretization` 返回 `(dt, A)`(1181)，而 1598 只取 `dt` ⇒ `A_log` 饥饿。
- **A57 `residual_scale_ssm` 从未进入 forward**（`hybrid.py:1215`）⇒ 梯度恒 None。
  **这条直接解释了"60 个标量/1,499 元素从未被更新"里的一部分**（见 §33/§48 的遗留项）。
- **A58 `_chunked_forward` 跨块不携带状态**（1638,1653 分配 `h` 却从不延续）⇒ 与 `_associative_scan`(811) 数学不同却可被选中；
  `_flash_scan` 的 `chunk_size=64` 外循环(1428-1431) 是无效果切分，无内存收益。

## 3. 同一机制多份实现（§9 冗余）

| 机制 | 份数 | 真实差别 |
|---|---|---|
| 选择性 SSM 递推 | 6（hybrid 545-557；mamba3 778-790、1431-1439、893、957；mamba_upgrades 196-207） | 状态布局 `[b,d_inner,s]` vs `[b,s,d]`；mamba3 clamp `delta∈[1e-6,10]`、`A∈[-10,-1e-6]`、输出 `±1e4`(846-860)，hybrid 不 clamp；GatedDelta 插入时间塌缩全局门 `x.mean(dim=1)`(184) 是真不同 |
| Attention↔SSM 融合 | 4（Progressive 723-729、Unified 1031-1042、Hierarchical 956-967、Jamba 1151-1155） | **前两份数学完全相同、连硬编码 `[[0.7,0.3]]/[[0.3,0.7]]` 都一样**(646/1007)；只有一份活 |
| "重要性"打分 | 4（cache 1512-1530、1587-1625、2159-2172、1979-2020） | decay/100 vs decay/50+recency vs `Σk²/√d`+softmax vs 跨头均值 0.5/0.5 |
| 缓存存储 | 4（kv_cache 1896、paged 1818、h2o 1798、generation 1796） | **效果上无实质差别**，其中三份是死重 |
| KV 量化 | 3（fp8-look 2192-2198、`_quantize_tensor` 2209、Lynx 2480） | 互斥条件是 `torch.cuda.is_available()`(2189) ⇒ `quantization_bits` 只在 CPU 生效，**同一份权重在不同机器上数值不同** |

## 4. 读了不生效的旗标

`cache_quantization`(只进统计 2378，量化 1922 无条件)、`use_h2o_attention`(从未读，H2O 恒开)、
`enable_compression`/`enable_cache_compression`/`cache_compression_ratio`(`self.compression_ratio` 1470 设了不用)、
`use_predictive_delta`/`use_plt_trie`(从未读 ⇒ 两者恒建)、`enable_offload`/`offload_threshold`(**没有任何 offload 代码**)、
`YvCacheConfig.max_seq_len`(331)、`cache_type`(338，无分派)、`quantization_bits`(2204，CUDA 上不可达)、
`mamba3_sequence_threshold`(读进 model.py:460 的死局部变量)。

**A59 STAR-KV 用最新基重构旧块**：`_projections` 每次 `compress` 覆盖(2594)，而 `_concat_recent` 用它解**更早**的块(1972) ⇒ 重构静默错误。

**A60 到就崩的接口不一致**：`_evict_block` 空池抛错(670,693)；`read()` 抛 `KeyError/RuntimeError`(818,822) 而契约说返回 None；
`get_kv_cache` 抛错(1882) 而 `hybrid.py:1253` 明确处理 None；`YvGatedDeltaNetwork.forward` 对 3 维 `state` 做
`'bsdm,bsm->bdm'`(mamba_upgrades 188/205) 且 `deltaA` 用 `d_model` 广播对 `d_state`(191) ⇒ RuntimeError；
`YvSparseSSM._prune_weights` 在**构造期**把 `param.data` 清零(79) 不可逆，且 `torch.quantile` 在 >16.7M 元素上失败。

**A61 每步设备同步**：`hybrid.py:1280` `ssm_gate.item()` 逐层逐步同步；`_forward_ssm` 返回的 loss 被 1346 覆盖。

**A62 token_sparse_attn 名不副实**：154-158 只对前 `head_dim//4` 维打分，且**照样物化它声称要避免的 `[B,H,T,T]`**；
`cluster_centroids`(66) 是死 buffer。

## 5. 内存算术（构造期即分配）

`cache.py:437-438` 每块 `2·n_heads·block_size·head_dim·itemsize`；`605` 上限含硬编码 `+100`；
分页池已改为**按需增长**（616-636，即我此前修的那条，已确认无残留的 eager-1000 模式）；
`mamba3.py:811-812` `_associative_scan` 物化 `[B,L,d_state,d_inner]`（L>4096 才走，阈值 854 硬编码）
⇒ `d_state=128, d_inner=4096, L=8192, B=1` 约 **2.1 GB**；`hybrid.py:634,998` + `YvMamba3Block:1552-1563`
每层约 10 个 `d_inner²` 矩阵全部构建且全部前向。

## 6. 未验证 / 存疑（不作为结论）

* 子代理按指示排除了 `tools/test/`，故 `allocate_paged_sequence`/`get_ssm_state` 是否被测试调用未查。
* `cfg.cache_config` 是否由某个配置文件注入器（而非 `model/config.py`）提供，未验证。
* `YvSparseSSM` 与 `YvFlashSSM` 在真实训练里的存活比例未做运行验证（白天主机不允许跑）。
