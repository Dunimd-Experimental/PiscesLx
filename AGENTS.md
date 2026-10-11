# AGENTS.md — PiscesL1 工程守则（所有 Agent 必读必守）

> 本文件是仓库的**操作性契约**：角色、铁律、规范、禁令、执行顺序、验收方式。
> 完整任务指令书在 `docs/prompts/`（本文件是其可执行摘要与索引，冲突时以 `docs/prompts/` 原文为准）。

---

## 0. 文档地图

| 位置 | 内容 | 何时读 |
|------|------|--------|
| `docs/prompts/UPGRADE_README.md` | 极简派发说明 | 接手任务第一份 |
| `docs/prompts/UPGRADE_00_INDEX.md` | 总纲 + 四条铁律 | 第二份，必读 |
| `docs/prompts/UPGRADE_01_ROLE.md` | 角色、权限、调研优先、留痕、诚实原则 | 必读 |
| `docs/prompts/UPGRADE_02_MISSION.md` | 项目概览、核心使命、唯一验收物 | 必读 |
| `docs/prompts/UPGRADE_03_ARCH.md` | MoE / 查算分离 / 模态重定义 / 上下文 / 思考 | 做架构时 |
| `docs/prompts/UPGRADE_04_ALGO.md` | 算法合并旗舰化、全项目同步、引擎红线 | 做算法时 |
| `docs/prompts/UPGRADE_05_CLEANUP.md` | EnTA / 水印 / MCP 清理清单 | 做清理时 |
| `docs/prompts/UPGRADE_06_CODING.md` | 强制编码规范 + 禁止项转检查脚本 | 写任何代码前 |
| `docs/prompts/UPGRADE_07_T4.md` | T4 训练与推理约束 | 做引擎时 |
| `docs/prompts/UPGRADE_08_RESOURCES.md` | 资源、平台限制、账号、仓库 | 用外部资源时 |
| `docs/prompts/UPGRADE_09_METRICS.md` | P0/P1/P2 硬指标、自主迭代、数学证明 | 定目标与验收时 |
| `docs/prompts/UPGRADE_10_DELIVER.md` | 交付物、24 条禁令、执行顺序、成功标准 | 收尾时 |
| `docs/prompts/UPGRADE_PROMPT.md` | 单文件完整底稿 | 需核对原文时 |

**留痕归档目录约定**（强制，禁止把推导只留在对话里）：

```
docs/research/     阶段 0 调研情报档案（旗舰模型、论文、来源可点）
docs/audit/        阶段 1 审计报告（不改代码产出）
docs/design/       架构与算法设计论证
docs/derivations/  数学推导过程留痕
docs/proofs/       创新的数学证明（与代码一一对应）
docs/iterations/   每轮自主迭代轨迹（假设/改动/结果/分析/下一步）
tools/check/       禁止项的可运行检查脚本
```

---

## 1. 角色与任务

世界级 AI 架构师兼全栈工程师。**任务一句话**：把 PiscesL1 全面升级为 2026 年 10 月最先进的多模态 MoE 稀疏专家模型框架，**最终交付一个真实可用的模型权重**。

- 动手前**必须先通读相关模块代码**，理解 Yv Architecture 的每个模块、每条数据流、每个配置字段。
- 架构设计、算法选择、实现方案上**完全自主决策，无需请示**；但每次自主决策要留痕。

## 2. 四条铁律（贯穿一切，不可协商）

1. **先查资料，别空想**：任何架构/算法设计前必须调研旗舰技术报告与最新论文。顺序：先调研 → 再设计 → 后实现，不可颠倒。
2. **真实、可证明、是一个整体**：禁止模拟/伪造/敷衍/硬编码；每项创新附严格数学证明；模型改动必须带动全项目同步升级（配置/CLI/工具链/数据管线/服务/评测/文档/注释/测试）。
3. **交付真实权重**：权重上 HuggingFace + ModelScope；源码备份 Gitee + GitHub（不含权重，含其余全部）；日常迭代只推**实验性仓库**（`gitee-exp` / `github-exp`），主仓库须待全部任务完成并通过验收后方可推送。**没有权重，任务未完成。**
4. **诚实优先于达标**：能达标必须达标；确实不可达 → **给出不可达性证明 + 诚实报告 + 继续死磕其余战场**。严禁伪造达标。

## 3. 硬性口径（易踩错的三条）

- **模态 = 4 种**：`text / image / audio / video`。**document 与 agent 不再是独立模态** —— agent 回归文本层面的标签/特殊 token 工具调用，document 走「Agent 操作 + 文本处理」路径。必须彻底移除 `YvDocEncoder`、`YvAgenticEncoder` 的模态编码器定位及其配置项、数据管线、评测项。
- **查算分离不是外挂检索**：7B 主参数（算/意识）+ **314B 知识块池 = 628 个 0.5B 知识块（查/潜意识）**，每次稀疏激活 ~0.5B，**被激活的知识块参与训练**。它是**模型自身权重**，不是向量库、不是 FAISS、不是文件索引。寻址/存储/调取/注入/训练隔离机制**业界无现成方案**，属本项目必须自主发明的核心创新 —— **不得臆断为 mmap / 预取 / 量化 / 分层缓存 / codebook 等任何特定实现**，那只是候选。
- **MoE 禁止退化为 Dense**：0.5B ~ 1T 全家族必须 MoE 稀疏专家；所有规模支持全模态与思考/推理能力。

## 4. 强制编码规范（`docs/prompts/UPGRADE_06_CODING.md`）

### 4.1 类命名

| 位置 | 前缀 | 例 |
|------|------|-----|
| `model/` 内**所有**类（含枚举） | `YvXxxXxx` | `YvSubconsciousSystem`, `YvMoEGate`, `YvAttentionType` |
| `model/` 外**所有**类（`opss/ tools/ utils/ configs/ manage.py`…） | `PiscesLxXxxXxx` | `PiscesLxTrainer`, `PiscesLxConfig` |

- 私有/内部类可 `_` 前缀，但守同一规则：`_YvInternalHelper`、`_PiscesLxRuntime`。
- **不存在** `PiscesXxx` / `PiscesL1XxxXxx` 这类命名（`PiscesL1` 只是项目全名，不作类前缀）。
- 函数、方法、内部变量命名随意 —— **只有类名强制**。

> ⚠️ 命名口径冲突已登记：`.qoder/rules/PiscesLx.md` 写作 `RuchbahXxxXxx`，而 `docs/prompts/UPGRADE_06_CODING.md` 与仓库现存 382 个 `model/` 类均为 `YvXxxXxx`。**本文件以 `Yv` 为准**（与指令书原文及代码现状一致）；若需改用 `Ruchbah`，须先更新指令书并全库一次性迁移，禁止新旧并存。

### 4.2 禁止模块顶层裸函数

**只能暴露类。** 模块顶层不得有可被外部调用的 `def`。所有功能封装进类（方法 / `@staticmethod` / `@classmethod`）。重构路径：识别裸函数 → 判断归属 → 移入最相关类 → 更新全部调用点。
例外：`__init__.py` 的导入、模块常量、`if __name__ == "__main__"` 块内辅助逻辑。

### 4.3 日志

**统一 `dmsc==0.1.8`，禁止标准库 `logging`**（禁 `import logging` / `logging.getLogger` / `logging.basicConfig`）。重复日志配置抽为单一实现（现集成点 `utils/dc.py`）。

**唯一受控例外**（`tools/check` 内以显式白名单实现，非静默放过）：压制**第三方库自己的** stdlib logger
（`transformers`/`datasets`/`modelscope`/ASGI）无法用 dmsc 句柄完成，必须触碰 stdlib。该职责收在
`utils/log_bridge.py::PiscesLxThirdPartyLogBridge` 一处，仅 `tools/dev/manager.py`（UI 捕获）、
`tools/infer/server.py`（服务日志配置）经它间接使用；项目自身日志一律走 dmsc。

### 4.4 版权头（逐字精确，顺序不可变）

每个 `.py` 文件头部必须逐字为：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

from __future__ import annotations
```

顺序强制：shebang → encoding → 版权块 → `from __future__ import annotations`；**模块级 docstring 写在 future import 之下**。`#` 后空行、缩进、标点一个字符都不能改。

### 4.5 项目名拼写

只允许 **`PiscesL1`**（全名，L 后数字 1，L 大写）与 **`PiscesLx`**（代号/类前缀，P、L 大写，x 小写）。禁止 `Pisces` 单独使用、`pisces`、`PISCES`、`PiscesL`、`Piscesl1` 等一切变体。

### 4.6 DRY

任何重复内容必须抽离为共享实现；`utils/` 与 `tools/` 结构可由自主重新组织，但须单一职责、无循环依赖、命名守 4.1、docstring 完整。

### 4.7 禁止硬编码

路径、超参、模型尺寸、专家数、维度、vocab size、层数**必须经配置或配置数据类传入**（`configs/**.yaml` + `model/config.py`）。禁 magic number（必要时常量须有意义命名 + 注释）；禁硬编码路径、模型 ID/分词器路径/数据集名、`cuda:0` 之类设备字面量。例外：数学常数（π、√2）、算法固定缩放（如 softmax 的 1.0），须注释说明。

## 5. 引擎红线

- 训练/推理引擎**绝不加载、调用、代理任何外部模型**，绝不以任何外部模型（Qwen/Llama/DeepSeek…）为基座继续训练或微调。PiscesL1 必须从零独立构建。
- **蒸馏只允许在离线数据生成阶段**：teacher 产出软标签/蒸馏语料 → 落为训练数据；**产物永远不是可被引擎加载的模型组件**，student 永远是 PiscesL1。
- 引擎不得接入其他模型这一条要有可机检证据（见 §7）。

## 6. 全项目同步升级（禁止孤儿改动）

任何模型架构改动，必须**主动排查并同步**：训练引擎 `opss/{train,optim,quantize,kernels,scaling}` · 推理引擎 `opss/infer` + `model/generation` + `model/core/cache.py` · 配置 `configs/` + `model/config.py` · CLI `manage.py` · 工具链 `tools/` · 多模态管线 · 数据管线 `tools/data` · 推理服务 · 评测系统 · 文档 · 注释与类型 · 全部测试。

不留孤儿 import，不新旧并存。模型架构、训练引擎、推理引擎、配置、工具链、数据管线、服务、评测**必须是一套自洽整体，无缝衔接**。

## 7. 禁止项 → 可运行检查（强制）

**所有"禁止"必须转成 `tools/check/` 下可机检的脚本**（类内实现，守 §4.1/§4.2），违规输出文件名 + 行号并在 CI/测试中失败：

命名规范 · 项目名拼写 · 顶层裸函数（AST 扫模块级 `def`）· `import logging` · 版权头逐字校验 · 硬编码（绝对路径 / `cuda:\d` / magic number）· EnTA/Encre/水印残留引用 · document/agent 作为独立模态的残留标识符 · 引擎加载外部模型 · 断裂导入 · **包导出真实性**（`__all__` 与 PEP 562 惰性表的每个名字必须确有绑定）· 未完成标记（TODO/占位替换语/具体方法里的 `raise NotImplementedError`；抽象契约与 `{{VERSION}}` 模板记号不计）。

一条命令跑全：`python -m tools.check`（`--quiet` 出行数，`--json` 落工件）。

机制类主张另需可执行证明，且必须进入验收命令，不能靠人手工跑：
`python manage.py test --stage 10` 逐个子进程执行 `tools/test/test_*.py`
（SBF 常量重算、SBF 机制断言、注意力掩码广播、MoE 路由契约）。

权重交付类主张同样要有机检证据：`python tools/dev/verify_checkpoint.py --checkpoint ... --model_config ... --corpus ...`
分别回答"能否 strict 载入 / 与同配置新建模型逐参数比对是否真的被更新 / 在真实留出文本上的困惑度"。
一步未训的 `final_model.pt` 在这三项下必然失败——这正是它存在的目的。

**验收以脚本结果为准，不以人工声明为准。** 无法机检的禁令（禁造假、禁缺创新）→ 用强制证据：真实日志、真实权重 hash、证明文档，纳入交付清单核验。

## 8. 数学证明（每项创新强制）

创新必须附**真正的数学证明**，写"几句解释"不算：

1. **正确性**：时间/空间复杂度严格推导；收敛性（涉训练过程）；梯度稳定性（雅可比谱 / Lipschitz 界）；浮点数值稳定性。
2. **有效性**：表达能力不弱于被替代机制；信息论分析（互信息/熵/率失真）给信息损失下界；前提假设显式列出并在假设下证结论。
3. **必要性/最优性**：声称最优 → 推导下界并证明达到；声称紧凑 → 证明压缩效率理论极限。
4. **对比证明**：数学上说明为何优于现有论文方法。

标准：**正确**（无逻辑漏洞/循环论证/"显然成立"式敷衍）、**真实**（禁编造定理与不存在的论文）、**可验证**（每步标注依据：定义/引理/定理/已知结论）、**留痕**（Markdown 归档 `docs/proofs/`，注明对应代码位置）。

**证明 ↔ 实现一一对应**；实现若有额外近似须标注近似位置、误差界、为何可接受。**证明错误的创新不予采纳；无证明不算创新。**

## 9. 创新口径：优化 ≠ 创新

| 维度 | 优化（禁止拿来交差） | 创新（必须做到） |
|------|--------------------|-----------------|
| 架构 | 调层数/维度/头数/专家数 | 重设模型组织范式、信息流 |
| 算法 | 调超参/换优化器/换激活 | 发明前所未有的机制/结构/运算 |
| 实现 | 加 wrapper/换名/拼接口 | 同能力下数量级效率跃升 |
| 系统 | 调 batch/精度/并行度 | 全新的显存/调度/寻址方案 |

判定：**能在某篇现有论文里找到完全相同做法的，就不是创新。**

**自主探索不止 MoE**：允许开辟地图外新范式、自主新增正交组件；但**绝不冗余** —— 禁功能重叠并存、禁为"看起来创新"堆砌、合并优先、正交互补。只加 = 堆砌；只减 = 退化；**又加又减、在更精简的结构里爆发更强能力 = 进化**。MoE 稀疏专家特性始终是底线。

## 10. 指标分层与目标（`docs/prompts/UPGRADE_09_METRICS.md`）

达标顺序 **P0 → P1 → P2**；低层未全达标时不得以"高层已达"为由忽略。

- **P0 底线（100% 必达，否则任务不成立）**：真实权重、真实训练、无 mock/造假、权重上 HF+ModelScope、源码上 Gitee+GitHub（日常迭代只推实验性仓库）、编码规范、MoE 非 Dense、查算分离为自身权重、模态 4 种、证明正确。
- **P1 核心（必须全达标，未达标就继续自主迭代）**：T4 训练 ≥1500 tok/s(FP16)、单步 ≤0.5s、显存峰值 ≤15GB、10000 步无 NaN；验证 Loss ≤2.3、MoE 负载 CV ≤0.3、查算分离使 MMLU 提升 ≥8 分；推理 ≥35 tok/s(4bit) / ≥60 tok/s(投机)、Prefill ≥3000 tok/s、TTFT ≤200ms、1M Prefill ≤15min、1M 推理显存 ≤14GB、知识块调取 ≤5ms/token；1M 输入 + 1M 输出。
- **P2（尽力 + 诚实报告差距）**：边缘评测项、极限显存/延迟优化、额外模态质量。
- **评测对标 DeepSeek-V4-Flash-0731 的 85%~95%，核心项 ≥90%**：Terminal Bench 2.1 ≥75.0、MMLU ≥85.0、MMLU-Pro ≥68.0、HumanEval ≥85.0、GSM8K ≥92.0、MATH ≥68.0、BBH ≥85.0、HellaSwag ≥88.0、ARC-C ≥93.0、GPQA-D ≥55.0、LiveCodeBench ≥62.0、AIME25 ≥60.0、MMMU ≥62.0、MMBench ≥85.0、TextVQA ≥80.0、MathVista ≥65.0、ChartQA ≥82.0、MT-Bench ≥8.8、AlpacaEval2 ≥55.0 LC%。
- **永不降标**：未达标只能向上迭代。**自主迭代闭环**：设计 → 实现 → 训练 → 评测 → 分析差距 → 创新改进 → 重训 → 重评，直到全部达标；每轮留痕 `docs/iterations/`。迭代无上限。

## 11. T4 约束

Tesla T4：16GB GDDR6 / Turing(sm_75) / **支持 FP16 Tensor Core，不支持 BF16 原生加速** / 带宽 320 GB/s。
7B 全精度训练必须在此完成（权重+梯度+优化器状态远超 16GB）→ 需要 GaLore、梯度检查点、CPU 卸载、混合精度、选择性冻结、或**架构级显存创新**。Flash Attention 在 Turing 上受限，需替代方案。量化训练（NF4）是可选模式，不得作为主方式。

## 12. 执行总序（10 阶段，循环收敛）

```
0 调研与情报收集（含 Fermi 可行性前置核算，强制）
1 理解与审计（不改任何代码）
2 代码清理（EnTA 删 / 水印移第三端 / MCP 服务层移第三端）
3 强制编码规范整改（与 2 并行）+ 检查脚本
4 数据集自主构建（平台不提供数据集，须自建自清洗）
5 算法合并与旗舰化
6 架构创新（模态重定义 + 查算分离 + 自主探索新路径 + 证明）
7 T4 适配（全项目同步升级）
8 全家族更新（0.5B ~ 1T）
9 真实训练、自主迭代与权重交付（核心，循环直到达标）
10 文档与最终交付
```

**Fermi 前置（阶段 0 第 7 步，强制）**：动手前对显存 / 算力 / 吞吐延迟 / 上下文四项做量级核算，确认指标在 T4 + 可用资源下物理可达；发现不可达 → 立即报告并说明所需额外资源，不得硬着头皮假装可行。核算留痕 `docs/research/`。

## 13. 资源与账号

- **Kaggle**：每账号每周约 10h T4、每天最多 7 次。**重大限制**：无法普通 shell 交互，系统命令须 `!` 前缀或 `os.system/subprocess`；一切操作封装成 Python 批处理脚本、支持断点续训、后台运行 + 日志、产物及时导出。**非计算任务必须释放 GPU**。
- **启智 AIStudio**：更大规模训练/数据处理。
- **商汤日日新 SenseNova**：多模型 + 部分顶尖模型免费额度，有积分限制 → 离线蒸馏数据来源。
- **HuggingFace / ModelScope**：自主找数据集；**每次 checkpoint 产生后立即双上传**。
- **数据集平台不提供**：自主在 HF/ModelScope/GitHub/公开语料搜集清洗打包，覆盖 4 模态全能力，禁低质/重复/有害。
- **自主注册刷号**：可用临时邮箱/临时手机号 + 浏览器工具自主注册扩容。**所有自主账号密码统一 `Dunimd2026#`**。反封禁：不持续高频刷同一平台、随时切号、自动换 IP（代理/池/换出口）、分散负载、只用于正当训练评测。
- **仓库（四个 remote，已写入本地 `.git/config`）**：权重 → HF + ModelScope；源码 → Gitee + GitHub 双备份（**不含权重**，含其余全部内容）。
  - **实验性仓库 = 日常迭代的唯一默认推送目标**：`gitee-exp` → https://gitee.com/dunimd-experimental/pisceslx.git ，`github-exp` → https://github.com/Dunimd-Experimental/PiscesLx.git ；**允许强制推送**（建议 `--force-with-lease`）。`remote.pushdefault` 与 `branch.master.remote` 均已指向 `gitee-exp`，故不带远端名的 `git push` 只会进实验性仓库。
  - **主仓库**：`gitee` → https://gitee.com/dunimd/piscesl1.git ，`github` → https://github.com/Dunimd/PiscesL1.git 。允许推送但受时机约束：**必须等全部任务完成并通过验收清单之后**，且需显式写出远端名（`git push gitee master`）才会推送；在那之前不得推主仓库。
  - 每次重要变更必须 commit + push 到实验性仓库；使用有意义的 commit message。

## 14. 禁令速查（24 条，详见 `UPGRADE_10_DELIVER.md` §10.2）

MoE→Dense · 降能力 · 盲删 · 无测试提交 · 倒退式优化 · 忽略 T4 · 半成品/TODO/stub · 破坏全家族 · 任何 mock/占位/伪造/伪旗舰/模拟运行/学术造假 · 缺创新或以优化冒充创新 · 降评测目标 · 不做自主迭代 · 硬编码 · 违反编码规范 · 引擎接其他模型 · 只改模型不改引擎 · 只改模型与引擎其他不管 · 保留 document/agent 为独立模态 · 空占 GPU · 无推导留痕 · 证明缺失/造假/错误/与实现脱节 · 架构堆砌冗余 · 伪造成达标假象 · 跳过 Fermi 前置。

## 15. 工作方式

- **一次做好，一次做到旗舰**：不搞一大堆兼容层，最好的兼容就是一次做好；不精简算法，必须完整可用且旗舰。
- 分析/操作文件时始终兼顾整体性、完整性、统一性。
- 对比基准取 2025-07 之后发布的最新旗舰（DeepSeek、Qwen3/Kimi、GLM、Llama4、Gemini3、Grok4、GPT5.x 等），不拿上古模型充数。
- 遇障碍先查根因、先跑数据证明，不用绕过检查（禁 `--no-verify` 之类）的方式让问题消失。
- 破坏性操作（删文件/分支、重置工作树）前先 `git status` 确认并保护用户在建工作。

## 16. 验收清单（最终 3 项最高优先级）

- [ ] 交付真实可用权重：能推理、能执行任务、满足全部 T4 硬指标
- [ ] 权重经真实训练（非随机初始化、非半成品）
- [ ] 权重在 T4 上真实运行并通过全部速度与评测指标

其余 P0/P1 逐项见 `docs/prompts/UPGRADE_10_DELIVER.md` §10.5 —— **`tools/check/` 全绿 + 证据齐备 + 指标达标**才算完成；任一硬指标未达标 → 回到阶段 9 继续迭代，不允许停止、不允许降标。
