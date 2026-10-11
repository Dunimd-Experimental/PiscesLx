# 提示词 → 工件 核对清单（完成度审计，2026-10-11）

用途：把 `docs/prompts/**` 与 `AGENTS.md` 的每一条硬要求，对到磁盘上可复核的工件与命令。
状态只有三种：**已达标（附证据）**、**部分（缺口写明）**、**未开始 / 受阻（写明所缺资源）**。
本清单的存在本身就意味着：**目标未完成** —— 只要有一行不是"已达标"，不得宣布完成。

## 1. P0 底线（AGENTS §10、铁律 3/4）

| 要求 | 证据 | 状态 |
|---|---|---|
| 交付真实可用权重 | `E:/pisceslx-coverage/ckpt/final_model.pt`（证据 `verify_evidence.txt`）：strict 载入 `missing=[]/unexpected=[]`、478/567 被更新、留出 next-token CE **9.5699 vs uniform 11.9293**（低 2.36 nats）→ `verdict=true` | **部分**：仅 0.01B 主机可行性档（281 个优化器步、约 4 万 token 语料），PPL≈1.43e4（verify 报告 14,327.52，uniform 基线约 1.52e5） 远不足以"执行任务"；0.1B/0.5B/7B 无任何权重；本轮增益未做单因子消融（覆盖率 vs `attention.scale` 入图叠加，docs/audit §43） |
| 权重上 HF + ModelScope | **实现已落地**（§45）：`tools/publish/hub.py` 的 `PiscesLxHubConfig` + `PiscesLxModelHubPublisher.plan/publish`，CLI 子命令 `python -m tools.publish.cli hub --checkpoint … --model_config …`；默认 dry-run（`--publish` 才真上传），凭据只从 `HUGGINGFACE_TOKEN`/`MODELSCOPE_API_TOKEN` 环境变量读且永不回显，卡片含本地绝对路径即拒发，缺 SDK 直接报错不手搓 HTTP。实测 dry-run 列出 4 文件 / 252,910,748 B / 双仓目标。契约测试 `test_hub_publish_contract.py` 6/6，其中一条把"源码备份不含权重"变成可复核断言 | **部分**：实现与离线验证完成；**真实上传仍等账号 token**，且属外部可见动作需你点头 |
| 源码上 Gitee + GitHub（不含权重） | `.gitignore` 已覆盖 `.pisceslx/` 与 `*.pt`；未创建远端 | **受阻**：等仓库；且 §13 禁止推主仓库，只能推私用仓库 |
| 真实训练（非随机/非半成品） | `docs/iterations/2026-10-11-host-tier-training-link.md`；同种子两次运行权重 **567/567 逐位相同**（`utils/seed.PiscesLxDeterminism`） | 已达标（主机层范围） |
| 无 mock/伪造 | `python -m tools.check` 14 条规则；`NO-STUB-NOR-TODO` 6 项在册待清 | **部分** |
| MoE 非 Dense（全家族） | 12 份 `configs/model/*.yaml`（0.01B…1T）均含 `moe_num_experts/top_k`；`manage.py test --stage 10` 内 `test_moe_routing_contract` 8/8 | 已达标（配置与契约层面；1T 从未实跑） |
| 查算分离为自身权重 | 池内块权重与 MoE 专家实测在训练（`knowledge` 1,048,576 元素有梯度；`experts` 30/30 有梯度）；FiLM 注入接通后 `router/state_evolver/film_generators` 取得梯度，死元素 32.36M→21.41M；`test_sbf_mechanism` 17/17、`test_sbf_constants` | **未达标**：A21 — 寻址码本 `block_addresser.traces`（177,152 元素）`grad is None`，"调哪一块"从不学习（§48）；314B 池的磁盘后备/异步传输未实现 |
| 模态恰为 4 种 | `model/multimodal/types.py::YvModality.CORE = (TEXT, IMAGE, AUDIO, VIDEO)`；代码中 `YvModality.DOCUMENT/AGENT` 命中数 **0**；`docs/design` 有说明 | 已达标 |
| 编码规范 | `tools/check` 14 条：命名 / 顶层裸函数 / dmsc 日志 / 版权头逐字 / 硬编码 / 包导出真实性 等 | **部分**：NO-MAGIC-NUMBER 3854、NO-SILENT-EXCEPT 182、NO-STUB 6 待清 |
| 证明正确且与实现一一对应 | `docs/proofs/00-sbf-theorems.md`、`01-multipath-reasoning.md`、`02-ink-variance-log-grid.md`（后者含"结论↔代码↔测试"对应表，且 8/8 套件执行） | **部分**：并非每个创新都有证明 |
| 留痕目录齐全 | `docs/derivations/` **为空**（0 文件）；`docs/research/` 仅 2 文件 | **未达标**：§0 强制推导落盘 |

## 2. P1 核心指标（§10）

全部为**未验证**，且都在用户尚未提供的 GPU 主机上才可能测得：

| 指标 | 目标 | 现状 |
|---|---|---|
| T4 训练吞吐 | ≥1500 tok/s (FP16) | 主机 CPU 实测 **9–25 tok/s**；T4 未跑 |
| 单步耗时 | ≤0.5 s | CPU 约 2.3 s/步；T4 未测 |
| 显存峰值 | ≤15 GB | 无 GPU；CPU 侧进程峰值 2.51 GB（闸门 `memory_budget_gb: 3.0` 强制） |
| 10000 步无 NaN | — | 未跑；主机层已加 `non-finite loss` 即中止的保护 |
| 验证 Loss / PPL | ≤2.3 | 主机层 11.70；差距巨大，需真规模训练 |
| MoE 负载 CV | ≤0.3 | 未测 |
| 查算分离使 MMLU +8 | — | 未测（无评测运行） |
| 推理 tok/s、投机、Prefill、TTFT、1M 上下文 | 见 §10 | 未测；阶段 7 引擎未做 |
| 知识块调取 ≤5 ms/token | — | 未测（内存常驻小池可测，磁盘后备未实现） |

## 3. 十阶段执行总序（§12）

- 阶段 0 调研：`docs/research/00-fermi-feasibility.md` + `fermi_raw_numbers.txt` → **部分**（旗舰模型技术档案不完整）
- 阶段 1 审计：`docs/audit/`（60 个文件，含 §13–§40 的缺陷登记与复现命令）→ **已达标**
- 阶段 2 清理：EnTA/Encre 物理删除、水印与 MCP 服务层移至 `third_party/`（实测原路径全部不存在）→ **已达标**；`opss/train/multitask.py` 第二套训练系统待合并（A4，§36）→ **部分**
- 阶段 3 规范整改 + 检查脚本：14 条规则可跑（`python -m tools.check`，含 `--json` 工件）→ **部分**（残留计数见上）
- 阶段 4 数据集自建：`tools/data/{download,clean,dedup,augment,dataset}`、`data/corpus/` 5 份语料（WikiText-2/103、GSM8K）→ **部分**（无四模态语料、无注册表清单）
- 阶段 5 算法合并旗舰化：推理多路径重写 + 证明 01；4 份 fusion 实现合并**未做** → **部分**
- 阶段 6 架构创新：SBF 常量推导 `test_sbf_constants`、机制断言 `test_sbf_mechanism` 17/17、本轮修通 FiLM 注入 → **部分**（磁盘 mmap 宿主后备场 + 预取阶梯未实现）
- 阶段 7 T4 适配：仅有设计（`docs/design/09-host-tier-contract.md`）→ **未开始**
- 阶段 8 全家族更新：12 档配置齐备但除 0.01B/0.1B 外无实跑 → **部分**
- 阶段 9 真实训练与权重交付：主机层首份可复现权重已过 uniform；规模化与双平台上传**未做** → **进行中**
- 阶段 10 文档与最终交付：**未开始**

## 4. 四条铁律的对账

1. 先查资料：`docs/research/`（2 份）→ 部分，覆盖不足。
2. 真实可证明是一个整体：`tools/check` 全绿前不许声明；本轮把"声明即生效"变成机制
   （`_apply_declared_fields` + 每个机制须有自己的日志行），并新增
   `NO-DETACHED-SCALAR-PARAM`、`test_film_contract` 两条防复发的机检 → 进行中。
3. 交付真实权重：主机层一份可复现；旗舰规模 0 份 → **未完成**。
4. 诚实优先：本清单 + §30–§40 的更正记录（含我三次越界操作与两次错误定性）→ 已执行。

## 5. 距离"完成"还差什么（按依赖顺序）

1. **GPU/T4 主机**（用户提供）→ 才能验 P1 全部速度/显存/稳定性指标。
2. **仓库与 HF/ModelScope 账号**（用户提供）→ 才能做源码备份与权重双上传；注意主仓库禁推。
3. **阶段 7 引擎实现**（`host_tier` 契约 + `checks_host_tier.py` + sm_75 内核 + 磁盘后备知识域）。
4. **阶段 4 数据**：四模态语料与带推理 span 的标注数据（当前 19.7M 的 `thinking_head`
   因**无标注**而不获梯度，属数据缺口而非代码断裂）。
5. **规模化训练**：0.1B → 0.5B → 7B，逐档产出权重并双上传。
6. **技术债**：NO-MAGIC-NUMBER 3854 / NO-SILENT-EXCEPT 182 / NO-STUB 6；A4/A5 双轨合并；
   剩余 62 个无梯度标量。
7. **留痕补齐**：`docs/derivations/` 空目录、`docs/research/` 仅 2 份。
