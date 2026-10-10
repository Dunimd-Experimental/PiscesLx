# 阶段 1 · 代码库审计报告（理解与审计，不改代码阶段的产出）

> 对应 `docs/prompts/UPGRADE_10_DELIVER.md` 阶段 1。
> **本报告所有数字由 `tools/check/` 实测产生**，原始工件：
> `docs/audit/conformance-baseline.json`（清理前）、`docs/audit/conformance-after-cleanup.json`（清理后）。
> 复核命令：`python -m tools.check --json docs/audit/x.json`

---

## 1. 仓库规模与结构

| 项 | 值 |
|----|----|
| Python 文件（全仓） | 454 → 清理后 **320**（模型层范围）+ 第三端 140 |
| `model/` 类总数 | 382 个 `Yv*`（命名基线已高度统一） |
| 版权头合规率（清理前） | **453 / 454**（仅 1 个文件缺失） |
| 顶层入口 | `manage.py`（2085 行，**单个 `main()` 占 1672 行**） |
| 依赖 | `requirements.txt` 锁定 `dmsc==0.1.8`（本机已实测可安装、可导入） |

目录职责：`model/{core,moe,multimodal,reasoning,generation,channels,tokenizer,utils}` ·
`opss/{train,infer,optim,quantize,kernels,scaling,knowledge,concurrency,am,run,validation}` ·
`tools/{benchmark,cache,data,dev,infer,monitor,publish,test,train}` · `configs/{model,train}` · `utils/`。

## 2. 阶段 2 清理：执行结果（可核对）

### 2.1 物理删除（EnTA / Encre）

| 目标 | 状态 | 机检证据 |
|------|------|----------|
| `model/agentic/enta/` | 已删 | `CLEAN-ENTA-ENCRE = 0` |
| `enta/`（Python + Rust native 后端） | 已删 | 同上 |
| `.encre/` | 已删 | 同上 |
| `configs/agentevolver/` | 已删 | 同上 |
| `configs/teachers.yaml` | 已删 | 同上 |
| `opss/agents/` | 已删 | `CLEAN-MCP-SERVICE-LAYER = 2`（仅存文档性引用，待清） |
| `YvConfig` 中 `encre_*` / `enta` 字段与 `YvEntaConfig` | 已删（−177 行） | `model/config.py` 内 EnTA 标识符归零 |
| `manage.py --enta / --enta_config` | 已删 | CLI 实测 `python manage.py help` 退出码 0 |

**安全措施**：删除前发现 `enta/` 下有**用户未提交的在建工作**
（`training.py`、`tools/builtin/train_*.py` 等 7 个新增 + 6 个修改），
已先落本地快照提交 `1172d5c`，再执行删除 —— 全部可恢复，未丢一行。

### 2.2 移至第三端

| 组件 | 新位置 |
|------|--------|
| `opss/watermark/` | `third_party/watermark/watermark/` |
| `tools/wmc/` | `third_party/watermark/wmc/` |
| `configs/watermark.yaml` | `third_party/watermark/watermark.yaml` |
| `tools/train/watermark.py`、`tools/infer/watermark.py` | `third_party/watermark/tools/` |
| `opss/mcp/` | `third_party/mcp_service/mcp/` |
| `model/multimodal/server.py` | `third_party/mcp_service/mcp_generation_server.py` |

`tools/check/` 的扫描范围已排除 `third_party/` —— 第三端不参与模型层规范验收，
但完整保留其代码（未精简，符合「不得精简算法」）。

### 2.3 模型层工具调用能力解耦（本轮最大工程项）

`model/multimodal/mcp.py` 原本 `from opss.mcp import 10 个 POPSS* 符号`，即**模型层直接依赖服务层**，
与「服务层移至第三端」互斥。处理方式不是打兼容层，而是**把协议原语下沉到模型层**：

- 新增 `model/multimodal/mcp_protocol.py`（12 个 `Yv*` 导出）：
  `YvMCPMessageType / YvMCPToolType / YvExecutionMode / YvMCPMessage / YvMCPToolMetadata /
  YvMCPExecutionResult / YvAgenticAction / YvAgenticObservation / YvMCPProtocol /
  YvCoreMCPToolRegistry / YvMCPExecutionManager / YvMCPUnifiedToolExecutor / YvMCPService`
- `mcp.py`、`agentic.py` 改接模型层协议，`POPSS*` 符号在模型层出现次数 **0**。
- **功能实测通过**（非编译通过，是真跑通）：注册→执行→返回 `{'a':1}`、未注册工具返回
  `unknown tool 'nope'`、执行计数与 `schemas()` 均正确。

> 实测中抓到并修复了一个**我自己引入的真实缺陷**：执行器默认值写成 `registry or YvCoreMCPToolRegistry()`，
> 而 `YvCoreMCPToolRegistry` 定义了 `__len__`，空注册表被判为假值 → 执行器悄悄换用**另一个**注册表，
> 导致已注册工具报「unknown tool」。改为显式 `if registry is not None`。
> **教训**：`or` 兜底遇 `__len__` 类容器必错。此类问题编译期不可见，只有功能测试能抓。

同时按「禁止冗余组件」（`UPGRADE_09_METRICS.md` 9.8.2）删除了
`agentic.py` 里**只赋值从不调用**的 `self.tree_reasoner`，而不是为它凭空造一个类来交差。

### 2.4 历史腐坏修复（清理前就存在的断裂）

| 断裂引用 | 位置 | 修复 |
|----------|------|------|
| `from model.modeling import YvModel` | `opss/quantize/engine.py`、`tools/benchmark/benchmark.py` | → `model.core.model` |
| `from utils.error import PiscesLxCoreValidationError` | `opss/validation/__init__.py` | → `from utils import …` |
| `from .server import YvMCPGenerationServer` | `model/multimodal/__init__.py` | 随服务层移除 |
| `from .enta import YvEntaTrainer` | `model/agentic/__init__.py`、`model/__init__.py` | 随 EnTA 删除 |

**新增可机检**：`tools/check` 加了 `PiscesLxDanglingImportCheck`（`NO-DANGLING-IMPORT`），
解析**绝对 + 相对**两类一级包导入并对照磁盘存在性。它当场抓出了 `.server` 这类
「相对导入指向已删模块」的隐性断裂 —— 这正是它必须成为常驻检查的理由。
当前实测：**悬空 import = 0**。

### 2.5 运行验证

```
python -c "import model, model.multimodal.mcp, model.multimodal.agentic"  → OK（CPU，dmsc 0.1.8 实装）
python manage.py help                                                     → 退出码 0
全库 py_compile / ast.parse                                               → 语法错误 0
```

## 3. 规范符合性：清理前 → 清理后（同一套检查）

| 规则 | 清理前 | 清理后 | 说明 |
|------|-------:|-------:|------|
| `NO-DANGLING-IMPORT` | — | **0** | 新增检查，断裂引用已清零 |
| `CLEAN-ENTA-ENCRE` | 41 | **0** | EnTA 除根完成 |
| `CLEAN-WATERMARK` | 191 | 69 | 余量集中在 `tools/{train,infer}/orchestrator.py`、`tools/help.py`、`model/multimodal/generator.py`、`configs/__init__.py` |
| `CLEAN-MCP-SERVICE-LAYER` | 13 | 2 | 余 2 处为文档性提及 |
| `NAMING-FORBIDDEN-PREFIX` | 1 | 1 | `model/tokenizer/tokenizer.py` 的 `PiscesLx160KTokenizer`（`model/` 内禁止 `Pisces*` 前缀） |
| `NAMING-MODEL-YV` | 35 | 23 | 主要在 `react_agentic.py`(9)、`enhanced_fusion.py`(6)、`recursive_depth.py`(4)、`expert_init.py`(2) |
| `NAMING-OUTSIDE-PISCESLX` | 746 | 436 | 减量来自删除 `opss/agents`、`enta/`；余量在 `opss/`(大量 `POPSS*` 类)、`tools/` |
| `EXPOSE-CLASSES-ONLY` | 359 | 137 | 顶层裸函数；`tools/setup.py`(10)、`opss/run/cli.py`(9)、`tools/publish/cli.py`(9)、`utils/paths.py`(8) |
| `HEADER-VERBATIM` | 460→**69**（修提取器 bug 后） | 60 | 见 §5 的自查 |
| `LOGGING-DMSC-ONLY` | 32 | 32 | 12 个文件，`model/generation/speculative.py`(6)、`tools/dev/manager.py`(5) 等 |
| `NO-STUB-NOR-TODO` | 77 | 41 | TODO/NotImplementedError |
| `MODALITY-4-NO-DOCUMENT` | 47 | 46 | **阶段 6 待办**：`doc.py`、`enhanced_fusion.py`、`generator.py`、`core/model.py`、`core/embedding.py` |
| `MODALITY-4-NO-AGENT` | 19 | 19 | **阶段 6 待办**：`agentic_encoder.py`、`model/agentic/__init__.py`、`multimodal/__init__.py` |
| `ENGINE-NO-EXTERNAL-MODEL` | 25 | 22 | **红线**：`model/tokenizer/tokenizer.py`(5)、`reasoner/multipath_*`(6)、`opss/train/distill.py`(3) |
| `NO-MAGIC-NUMBER` | 4427 | 3895 | 工作量最大项 |
| `NO-HARDCODED-DEVICE` | 11 | 11 | `tools/train/core.py`(4)、`tools/publish/*`(4) |
| `NO-HARDCODED-MODEL-ID` | 29 | 5 | |
| `NO-HARDCODED-PATH` | 12 | 1 | |
| **合计 error** | **6119** | **4788** | −1331（−21.8%），且检查项从 10 增到 11 |

## 4. 算法与冗余审计（阶段 5/6 的输入）

### 4.1 已实现的 2026 旗舰算法（`UPGRADE_04_ALGO.md` 4.2 清单，需在保留中升级）

`mHC / mHC-lite`、`Kimi AttnRes`、`PathMoE`、`Phi-Balancing`、`HydraHead`、`LCA`、`DSpark`、
`MoVE`、`SparseCut`、`CSA/HCA`、`MrRoPE`、`HiCL Router`、`LayerRoute`、`CoMeT`、`Engram`、
`Seirênes`、`SAGE`、`SEAL`、`A-Evolve`。

### 4.2 识别出的功能重叠（合并去重候选，阶段 5 处理）

| 重叠簇 | 涉及实现 | 判定 |
|--------|----------|------|
| **融合机制 ×4** | `multimodal/{fusion, enhanced_fusion, rca_fusion, sync_fusion}` | 违背 9.8.2「禁止功能重叠」→ 合并为**单一旗舰融合**（`UPGRADE_03_ARCH.md` 3.3.2「统一融合机制」明文要求） |
| **注意力变体 ×N** | `core/{attention, duo_attention, memory_attention, csa_hca, eg_mla}` | 需给出正交性论证，重叠者合并 |
| **路由策略 ×5** | `moe/{gate, hicl_router, online_router, layer_route, path_moe}` | 主路由唯一化，其余作为其正交分量或并入 |
| **推理系统 ×6** | `reasoning/reasoner/{cot_memory, multipath_core, multipath_system, recursive_depth, unified, enhanced_system}` | `unified` 应为唯一出口 |
| **量化目录 ×2** | `opss/{quantize, quantization}` | 目录级重复，必须合一 |
| **死组件** | `agentic.py` 的 `tree_reasoner` | **本轮已删**（只赋值不调用＝堆砌） |

### 4.3 查算分离现状（阶段 6 的起点）

- `model/core/subconscious.py`（`YvSubconsciousSystem`）：16 codebooks × 131K entries 乘积量化 +
  FiLM 调制注入 —— **「潜意识」思想雏形已在，但它是「码本量化」路线，不是块级知识池**。
- `model/core/knowledge_experts.py`：知识专家池，`knowledge_expert_path` 指向**磁盘外部路径**
  （`model/config.py` 默认 `./knowledge_experts`）→ 与 3.2 的正确含义（**模型自身权重**）冲突，
  **必须重做**，不能沿用。
- `model/core/dual_injector.py`：注入通路，可复用为「唤起后进入激活路径」的机制底座。
- `Memory Separation`（FAISS 外部检索 + N-gram 地址哈希）：**方向性错误**，属外挂检索而非查算分离，
  按 `UPGRADE_02_MISSION.md` 的警示需重新评估改造。

> **留白确认**：寻址 / 存储 628 块 / 调取 / 注入 / 训练隔离 五项机制**在仓库中无正确实现先例**，
> 属必须自主发明项。设计论证与数学证明见 `docs/design/`、`docs/proofs/`。

## 5. 自查与诚实记录（工具本身也会错）

审计工具自身在测量过程中暴露 4 个缺陷，**全部已修**，且这些缺陷若未修会使报告失真：

1. **版权头提取器漏掉 shebang 行** → 460/460 文件被误报为违规（真实值 69）。
   若直接采信，会得出「全库版权头不规范」的**错误结论**。已修：提取块首行包含 `#!`。
2. **检查器把自己内嵌的探测正则当作违规**（`CLEAN-WATERMARK` 假阳性 8、`CLEAN-ENTA-ENCRE` 7）
   → 增加 `tools/check/` 自我豁免。未修则「残留清零」永远不可能达成。
3. **AST 类名误用**（`ast.Keyword`、已移除的 `ast.Index`）、**`__slots__` 漏 `_paths`**、
   遗留 `del` 死代码 → 分别导致运行期异常。
4. **规则 ID 未按位置区分**（`NAMING-OUTSIDE-PISCESLX` 未落地、`coverage()` 报未覆盖）→ 修正，
   `coverage()` 现在会主动暴露「声明了但没有可运行检查」的规则。

同时如实记录：`tools/infer/core.py` 的批量删除中我**误删了 3 行正常逻辑**
（`_apply_ink_quantization()` 调用、`YvAgentic` 构造、`_setup_moe_operator()` 调用），
已在同轮以功能测试驱动复原。此类批量编辑的教训是：**删除必须逐锚点断言 + 事后功能复验**，
不能只做语法检查。

## 6. 全项目影响面清单（阶段 7「同步升级」的必查表）

改动模型架构时**必须同步**的落点（已按现存文件核实）：

| 层 | 落点 |
|----|------|
| 配置 | `model/config.py`(YvConfig ~300 字段)、`configs/model/*.yaml`(0.5B–1T)、`configs/train/*`、`configs/dataset.yaml`、`configs/settings.yaml`、`configs/version.py` |
| CLI | `manage.py`（含 `_build_train_argv_from_args` 的 argv 映射表） |
| 训练引擎 | `opss/train/*`、`opss/optim/*`、`opss/{quantize,quantization}`、`tools/train/{core,orchestrator,config}.py`、`opss/subconscious_trainer.py` |
| 推理引擎 | `opss/infer/*`、`model/generation/*`、`model/core/cache.py`、`tools/infer/{core,orchestrator,config,server}.py` |
| 数据管线 | `tools/data/{datamodule,download,distributed,streaming}` |
| 评测 | `tools/benchmark/*`、`tools/test/*` |
| 服务 | `tools/infer/server.py`、`opss/mcp`(已移出) |
| 文档 | `README.md`、`README.zh.md`、`CONTRIBUTING.md`、`SECURITY.md`、`docs/**` |

## 7. 下一步（阶段 3 剩余 → 4 → 5 → 6）

1. **红线优先**：`ENGINE-NO-EXTERNAL-MODEL` 22 处 —— 引擎内 tokenizer / 外部模型名解耦（P0）。
2. 模态重定义：删 `doc.py` + `agentic_encoder.py` 的模态定位，统一 4 模态管线（P0）。
3. 水印残留 69 处彻底清出模型层（P0）。
4. `import logging` 32 处 → dmsc；版权头 60 文件；顶层裸函数 137；命名 459（`model/` 内 24 + 外 436，
   含 `POPSS*` → `PiscesLx*` 的统一）（P0）。
5. magic number 3895 —— 量最大，需按「移入 `YvConfig`/YAML」的批处理策略推进，并让
   `tools/check` 在每一批后重新计数以防回潮。
6. 知识块池寻址/调取机制设计 + 数学证明（阶段 6 核心，见 `docs/design/`、`docs/proofs/`）。
