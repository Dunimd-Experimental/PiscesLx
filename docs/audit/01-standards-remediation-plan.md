# 编码规范存量整改计划（阶段 3 未完成项）

> 配套：`docs/audit/00-audit-report.md`（初始审计）、`AGENTS.md` §4/§7、
> 实时数据：`docs/audit/conformance-revert.json`（`python -m tools.check --json <path>` 复跑）
>
> 当前清零 **15 / 19** 条规则。剩 4 项，本文给出剩余量、已完成手法、**已踩过的坑**与安全的下一步做法。

## 1. 剩余量

| 规则 | 剩余 | 分布 |
|------|-----:|------|
| `NO-MAGIC-NUMBER` | 3884 | model 1839 · tools 1169 · opss 712 · utils 153 · manage 11 |
| `EXPOSE-CLASSES-ONLY` | 128 | 63 个文件；48 个文件已有类可收容，15 个纯函数模块 |
| `NO-STUB-NOR-TODO` | 42 | TODO/FIXME/`NotImplementedError`/stub |
| `NO-HARDCODED-PATH` | 1 | `tools/setup.py` CUDA 探测路径 |

已清零（15 项）：EnTA、水印、MCP 服务层、引擎外部模型、悬空 import、document 模态、agent 模态、
类命名（model 内 / 外 / 禁止前缀）、版权头、stdlib logging、项目名拼写、硬编码设备、硬编码模型 ID。

## 2. `EXPOSE-CLASSES-ONLY`：安全做法与一次真实事故

### 2.1 已验证可行的样板：`utils/paths.py` → `PiscesLxPaths`

8 个函数折为 `@staticmethod`，**127 个调用文件**同步，全项目导入通过。要点（缺一不可）：

1. **不留模块级壳函数**。留壳＝规范禁止的兼容层。
2. **三种导入形态都要覆盖**（首轮只覆盖第 1 种，直接导致 `model`、`tools.infer`、
   `model.core.model` 导入失败）：
   - 顶层绝对：`from utils.paths import get_log_file`
   - 函数内缩进惰性导入：`    from utils.paths import get_log_file`
   - 相对导入：`from ..paths import get_log_file`
3. **绝不改写定义处**：替换需前置 `(?<!def )` 守卫。
4. **lookahead 替换不要补左括号**：`(?=\s*\()` 之后替换串若写成 `Cls.name(`，
   原括号保留 → `name((`，直接语法坏掉。替换串只能是 `Cls.name`。
5. **每次写入前 `ast.parse`**：不通过就整文件跳过。这条守卫在本轮至少 3 次阻止了半成品落盘。
6. 函数被搬走后，**模块自身 docstring 里的旧用法示例也要改**，否则文档在教一个不存在的调用形式。

### 2.2 一次真实事故（已完整回退，仓库干净）

尝试把上述手法**泛化成工具并一次跑 6 个文件**时造成连带破坏：传播阶段对
**目标模块的所有函数名**做了无差别的调用点改写，其中包含 `main`、`_log`、
`logger_info` 这类通用名。结果 `manage.py` 里对**它自己的** `main()` 的调用被改成了
`PiscesLxCli.main()` —— 不是格式问题，是会改变行为的破坏；工具脚本自身也被改写坏。

已用 `git checkout` 逐文件回退到事故前提交，删除该工具，并复验：11/11 关键模块导入通过、
`grep` 全仓无残留 `PiscesLxCli.main` / `PiscesLxSetup._log` 形式、规则计数回到
4050 errors（= 事故前状态）。

### 2.3 因此，剩余 128 个函数的正确工序

**核心修正：传播必须按 import 限定，而不是按名字满仓搜。**

对每个待迁文件依次做，且**一个文件一次提交**：

1. 取该文件的顶层函数集合 `F`，与其中**名字足够具体**的子集 `F_safe`
   （判定标准：`name` 在仓库中出现的文件里，**导入过本模块**的比例；
   只保留 100% 来自本模块的名字。`main`/`_log`/`logger_info`/`setup` 这类必须
   改为**显式逐个文件核对**，不可批量）。
2. 选择收容类：文件内已有类 → 并入方法最多的那个；无类 → 生成 `PiscesLx<模块名>`
   （`model/` 下必须生成 `Yv<模块名>`）。
3. 并入已有类时，**只插方法体，不重复 `class X:` 头**；删除函数要逐个删
   （整段 `first..last` 切片会连带删掉夹在函数之间的类，导致 anchor 丢失）。
4. 改写调用点：先筛出 `import` 了该模块的文件，只在这些文件里替换；其余文件若仍出现
   同名调用，**保持原样并单独报告**（它们调用的是自己的同名函数）。
5. 每步 `ast.parse` 守卫；随后 `python -c "import <每个受影响包>"` +
   `python -m tools.check` + `for t in tools.test.test_*: python -m $t`。
6. 计数验收：`EXPOSE-CLASSES-ONLY` 应恰好减少本次迁移的函数数；不降反升即为改错。

建议顺序（先易后难、先叶子后入口）：
`utils/vram_controller.py`(7) → `opss/opd.py`(4) → `opss/knowledge/run_build.py`(5) →
`model/core/norms.py`(4，注意类名前缀须 `Yv`) → `model/moe/mega_kernel.py`(2) →
`model/utils/{quantization,rope}.py`(2+2) → `opss/run/cli.py`(9) →
`tools/publish/cli.py`(9，**含 `main`，逐个核对**) → `tools/setup.py`(10，同上) →
最后 `manage.py`(6)。

## 3. `NO-MAGIC-NUMBER` 3884：按「是否属于模型配置」分层

不可用统一脚本暴力处理——多数需要判断"这是配置项还是算法常数"。分三档推进，
**每档之后重跑检查器以防回潮**：

1. **模型结构/超参（必须先做，~1839 中的主干部分）**：维度、层数、头数、专家数、
   `top_k`、温度、比例、窗口长度 → 一律进 `YvConfig` + `configs/model/*.yaml`，
   引擎代码只从 `cfg` 取。这一档直接关系"零硬编码"验收与全家族（0.5B–1T）一致性。
   样板：SBF 的 14 个参数已全部走配置（`block_count` / `block_memory_rows` /
   `block_rank` / `block_trace_dim` / `block_pool_slots` / `block_gamma_max` /
   `block_beta_bound` / `block_inject_alpha` / `block_hysteresis` /
   `block_staleness_gain` / `block_staleness_cap_tokens` / `block_min_dwell_tokens`），
   可作为其余部分的改写范式。
2. **半经验数值（阈值、缩放、epsilon、位宽）**：进配置并在注释写明来源；
   若由公式导出（如 `block_min_dwell_tokens = ceil(transfer_ms / budget_ms)`、
   `tau_staleness` 见 `docs/proofs/00-sbf-theorems.md` 定理 6）则写公式，不写常量。
3. **合法常数**：数学常数、softmax/位运算固定因子、`0/1/-1/2` 类结构性下标——
   检查器已放行；剩余的补一行"为何固定"。

推进方式：按目录批量（`model/core` → `model/moe` → `model/multimodal` → `model/reasoning`
→ `opss` → `tools` → `utils`），每批 ≤ 一个目录，改完即跑检查 + 回归。

## 4. `NO-STUB-NOR-TODO` 42

`NotImplementedError` 与 `TODO/FIXME` 属"半成品"禁令（§10.2-7）。逐条判断：
能实现则实现；已不需要则**连同其调用路径一起删除**（不留占位）；
`tools/test`、`docs` 内的降为 warning，不计入验收。

## 5. `NO-HARDCODED-PATH` 1

`tools/setup.py` 的 CUDA 安装目录探测。它同时是该文件的顶层函数之一，
**与 §2.3 的类化改造合并处理**：折进类之后由
`PISCESLX_CUDA_PREFIX` 环境变量驱动，默认值退化为可配置候选列表，
不写死绝对路径（此前单独修补失败，正是因为绕不开它还是裸函数）。

## 6. 不可机检的 P0 项 → 用证据验收（§6.8 后半）

| 禁令 | 强制证据 |
|------|----------|
| 禁造假 | 真实训练日志、checkpoint 的真实 sha256、评测原始输出目录 |
| 禁缺创新 | `docs/design/` + `docs/proofs/` + 可运行回归（现有：SBF 17 项机制断言、掩码 14 项、常量 11 项） |
| 禁伪达标 | 未达标项须附不可达性论证：已交付 `docs/research/00-fermi-feasibility.md` 命题 1/3/4/5 |
| 权重交付 | HF + ModelScope 上传回执；Gitee + GitHub 源码提交（不含权重） |

## 7. 复跑命令

```bash
python -m tools.check --json docs/audit/now.json          # 全部 19 条规则
python -m tools.test.test_sbf_mechanism                   # 17 项（SBF 机制/定理）
python -m tools.test.test_attention_mask_shapes           # 14 项（掩码广播缺陷防复活）
python -m tools.test.test_sbf_constants                   # 11 项（证明常量复算）
```
