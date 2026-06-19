<div align="center">

# EnTA

[English](README.md) | 简体中文

[文档](https://dunimd.github.io/enta) | [更新日志](docs/CHANGELOG.md) | [安全](docs/SECURITY.md) | [贡献指南](docs/CONTRIBUTING.md) | [行为准则](docs/CODE_OF_CONDUCT.md)

<a href="https://github.com/mf2023/EnTA" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-enta-181717?style=flat-square&logo=github"/>
</a>
<a href="https://gitee.com/dunimd/enta" target="_blank">
    <img alt="Gitee" src="https://img.shields.io/badge/Gitee-enta-C71D23?style=flat-square&logo=gitee"/>
</a>

<a href="https://x.com/Dunimd2025" target="_blank">
    <img alt="X" src="https://img.shields.io/badge/X-Dunimd-000000?style=flat-square&logo=x"/>
</a>
<a href="https://space.bilibili.com/3493284091529457" target="_blank">
    <img alt="BiliBili" src="https://img.shields.io/badge/BiliBili-Dunimd-00A1D6?style=flat-square&logo=bilibili"/>
</a>
<a href="https://huggingface.co/dunimd" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-Dunimd-FFD21E?style=flat-square&logo=huggingface"/>
</a>
<a href="https://modelscope.cn/organization/dunimd" target="_blank">
    <img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-Dunimd-1E6CFF?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTcuMDA2IDBDMy4xNDIgMCAwIDMuMTQyIDAgNy4wMDZTMy4xNDIgMTQuMDEyIDcuMDA2IDE0LjAxMkMxMC44NyAxNC4wMTIgMTQuMDEyIDEwLjg3IDE0LjAxMiA3LjAwNkMxNC4wMTIgMy4xNDIgMTAuODcgMCA3LjAwNiAwWiIgZmlsbD0iIzFFNkNGRiIvPgo8L3N2Zz4K"/>
</a>

<a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square"/>
</a>

**EnTA** — AI Agent 项目，以一个仓库统一发布 Python Agent 框架、Rust 原生核心以及 EnTA Desktop 三部分。后端无关、多提供商支持，具备流式工具调用、安全/权限引擎、持久化记忆、多智能体编排、桌面 UI、CLI 运行器、WebSocket 服务器，以及 18 个外部聊天平台适配器。

</div>

<h2 align="center">🏗️ 核心架构</h2>

### 多语言架构
EnTA 采用多语言架构，包含三个主要组件，使每一层都能使用最适合其领域的语言：

<div align="center">

| 语言 | 组件 | 用途 |
|:---------|:----------|:--------|
| **Python** | `backend/enta/` | AI Agent 框架核心 — Agent 循环、31 个 LLM 后端、36 个内置工具、18 个平台适配器、安全引擎、记忆、技能、Swarm、钩子、LSP、MCP |
| **Rust** | `native/crates/enta-core` | 原生高性能库 — 文件 I/O、正则搜索、glob、diff、沙箱、分词器、BM25 索引器、Landlock、SIMD 搜索、LSP 协议、语义相似度 |
| **Rust** | `native/crates/enta-py` | PyO3 绑定，将 Rust 核心暴露给 Python 作为 `enta._native` |
| **TypeScript** | `desktop/` | EnTA Desktop — 基于 Electron 的桌面应用，React UI、Monaco Editor、xterm.js 终端、与 Python 后端的 IPC 桥接 |

</div>

EnTA 以单一仓库形式发布，三个组件共用同一条发版线。它不是独立库，也不是可单独分发的包 —— 请克隆整个仓库以使用 EnTA。

### 仓库结构

```
d:\enta/
├── pyproject.toml             # Python 构建与依赖（构建 backend/enta/）
├── build.py                   # 一键构建脚本（Rust + Python + Desktop）
├── gen.py                     # 图标/资源生成器
├── package-lock.json          # 根 node 锁文件（与 desktop 同步）
├── LICENSE                    # Apache 2.0 许可证
├── README.md / README.zh.md   # 本文档
│
├── backend/                   # Python Agent 框架
│   ├── enta/                 #   `enta` Python 包
│   │   ├── __init__.py        #   公开 API 表面
│   │   ├── agent.py           #   EncreAgent — 公开 Agent 类
│   │   ├── loop.py            #   EncreLoop — 执行循环
│   │   ├── session.py         #   EncreSession — 对话状态
│   │   ├── safety.py          #   EncreSafetyEngine — 6 种权限模式
│   │   ├── autosafety.py      #   EncreAutoSafetyClassifier
│   │   ├── config.py          #   配置管理
│   │   ├── crypto.py          #   AES-GCM 加密辅助函数
│   │   ├── ssrf.py            #   EncreSSRFGuard（DNS + CIDR 黑名单）
│   │   ├── ratelimit.py       #   EncreRateLimiter
│   │   ├── rollback.py        #   基于 Git 的回滚（EncreRollbackGit）
│   │   ├── recovery.py        #   错误恢复引擎
│   │   ├── scheduler.py       #   EncreScheduler（cron 风格任务）
│   │   ├── goal.py            #   Goal 运行器与评估循环
│   │   ├── telemetry.py       #   EncreTelemetry（turn / tool / retry）
│   │   ├── native.py          #   Python 包装 enta._native
│   │   ├── _native.pyi        #   Rust 扩展的类型存根
│   │   ├── backend.py         #   create_backend() 工厂
│   │   ├── dangerous_commands.txt  # Bash 安全模式
│   │   ├── backends/          #   31 个 LLM 提供商适配器
│   │   ├── adapters/          #   18 个平台适配器（聊天平台）
│   │   ├── agents/            #   内置子 Agent 定义
│   │   ├── channels/          #   传输层：websocket、terminal、HTTP、slash
│   │   ├── tools/             #   36 个内置工具 + 工具注册表 + MCP
│   │   ├── hooks/             #   EncreHookSystem
│   │   ├── memdir/            #   持久化记忆系统（frontmatter）
│   │   ├── skills/            #   技能注册表 + 11 个内置技能
│   │   ├── swarm/             #   多智能体系统（teammate/mailbox/...）
│   │   ├── task/              #   任务管理器与执行器
│   │   ├── server/            #   WebSocket 服务器 + 管理 HTTP API
│   │   ├── gateway/           #   Gateway 客户端/服务器协议
│   │   ├── compact/           #   上下文压缩（9 种策略）
│   │   ├── lsp/               #   LSP 客户端 + 多语言服务器管理
│   │   ├── codebase/          #   代码索引器（BM25 + 依赖图）
│   │   ├── computer/          #   桌面与浏览器自动化
│   │   ├── evolution/         #   元认知、反射、策略优化
│   │   ├── feedback/          #   错误修正学习器
│   │   ├── notebook/          #   交互式 Python 内核会话
│   │   ├── plugins/           #   插件注册表与清单类型
│   │   ├── profile/           #   用户画像推断
│   │   ├── soul/              #   灵魂系统文件（人格/记忆）
│   │   ├── spec/              #   规范文档引擎
│   │   ├── prompts/           #   提示词块、技能、安全、目标
│   │   ├── sandbox/           #   Docker 容器沙箱
│   │   ├── search/            #   基于 MCP 的 Web 搜索
│   │   ├── learning/          #   技能生成与整合
│   │   ├── rules/             #   规则加载器
│   │   ├── thinking/          #   思考配置解析
│   │   ├── iclaw/             #   iClaw CLI（自动化运行器）
│   │   ├── git/               #   Git 仓库与 diff 工具
│   │   └── utils/             #   ID 生成、Token 计数、类型
│   └── tests/                 # Pytest 测试套件（由 pyproject.toml 配置）
│       ├── conftest.py
│       ├── test_agent.py
│       ├── test_backends.py
│       ├── test_backend_*.py
│       ├── test_safety.py
│       ├── test_session.py
│       ├── test_loop.py
│       ├── test_native.py
│       └── ...                #   40+ 个测试模块
│
├── native/                    # Rust 工作区
│   ├── Cargo.toml             #   工作区根
│   └── crates/
│       ├── enta-core/        #   原生核心（包名：`enta`）
│       │   ├── Cargo.toml
│       │   └── src/
│       │       ├── lib.rs
│       │       ├── fs.rs          # 原生读/写
│       │       ├── search.rs      # 正则搜索、glob
│       │       ├── simd_search.rs # SIMD 加速模式匹配
│       │       ├── diff.rs        # Unified diff + apply
│       │       ├── shell.rs       # 沙箱化 Shell 执行
│       │       ├── sandbox.rs     # 沙箱结果类型
│       │       ├── landlock.rs    # Linux Landlock 强制
│       │       ├── tokenizer.rs   # 启发式 Token 计数器
│       │       ├── embedding.rs   # Cosine / Jaccard 相似度
│       │       ├── indexer.rs     # BM25 代码搜索索引
│       │       └── lsp_proto.rs   # LSP JSON-RPC 解析/构造
│       └── enta-py/          # PyO3 绑定 → `enta._native`
│           ├── Cargo.toml
│           └── src/lib.rs
│
├── desktop/                   # Electron 桌面应用
│   ├── main.ts                #   Electron 主进程
│   ├── preload.ts             #   上下文桥接（IPC）
│   ├── build.js               #   esbuild 配置
│   ├── package.json           #   Node.js 依赖与脚本
│   ├── electron-builder.yml   #   NSIS / pkg / deb 打包
│   ├── tsconfig.json          #   TypeScript 配置（主进程）
│   ├── fetch_icons.js         #   图标获取脚本
│   └── renderer/              #   前端（React 19）
│       ├── index.html
│       ├── styles.css
│       ├── design-system.css
│       ├── xterm.css
│       ├── bundle.js          #   esbuild 产物
│       ├── tsconfig.json
│       ├── assets/            #   Logo 与图标
│       ├── src/               #   TypeScript 源码
│       │   ├── app.ts
│       │   ├── chat.ts
│       │   ├── session.ts
│       │   ├── session_inner.ts
│       │   ├── settings.ts
│       │   ├── state.ts
│       │   ├── stream.ts
│       │   ├── ws.ts
│       │   ├── search.ts
│       │   ├── i18n.ts
│       │   ├── iclaw.ts
│       │   ├── agents.ts
│       │   ├── automation.ts
│       │   ├── tools.ts
│       │   ├── files.ts
│       │   ├── workspace.ts
│       │   ├── viewmanager.ts
│       │   ├── permissions.ts
│       │   ├── dialog.ts
│       │   ├── notifications.ts
│       │   ├── slash_commands.ts
│       │   ├── icons.ts
│       │   ├── crypto.ts       #   渲染端 AES 辅助
│       │   ├── easter-egg.ts
│       │   ├── transition-helper.ts
│       │   ├── types.ts
│       │   ├── global.d.ts
│       │   └── locales/        #   en.ts / zh.ts
│       └── vs/                 #   Monaco Editor（已打包）
│
├── docs/                      # 项目文档
│   ├── CHANGELOG.md
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTENT_GUIDELINES.md
│   ├── CONTRIBUTING.md
│   ├── DATA_PROCESSING_RULES.md
│   ├── MINORS_PRIVACY.md
│   ├── PLAN.md
│   ├── PRIVACY.md
│   ├── PRIVACY_CN.md
│   ├── SECURITY.md
│   ├── TERMS.md
│   ├── TERMS_CN.md
│   ├── THANKS.md
│   ├── THANKS_CN.md
│   └── USER_AGREEMENT.md
│
└── .github/
    └── workflows/
        └── build-binary.yml   # CI：构建发布二进制
```

<h2 align="center">🚀 关键特性</h2>

#### 🧠 多提供商 LLM 后端（31 个）
- **第一方**：OpenAI、Anthropic Claude、DeepSeek、Google Gemini、Groq、Ollama、Local（HuggingFace transformers）
- **中文与区域**：阿里（Qwen / DashScope）、腾讯、小米 MiMo、Kimi（Moonshot）、GLM（智谱 / Z.ai）、Minimax
- **聚合/路由**：OpenRouter、AI Gateway、OpenCode、Kilocode、Arcee、Novita
- **云与企业**：AWS Bedrock、GitHub Copilot
- **自托管与兼容**：LM Studio、OpenAI Compatible、OpenAI SSE
- **元后端**：Failover（多后端健康切换）、Router（按成本/任务路由）、Retry（透明重试）
- **目录**：MCP Catalog、静态模型目录（`enta.backends.catalog`）
- 统一流式接口、自动思考块解析、OpenAI 兼容的 Chat Completions

#### 🤖 内置子 Agent（9 个）
- **通用模式**：`coder`、`researcher`、`critic`
- **工作区模式**：`architect`、`planner`
- **计划/规范模式**：`spec-writer`
- **iClaw（自动）模式**：`monitor`、`executor`、`scheduler`
- 每个子 Agent 由 `enta/agents/builtin.py` 中的 `SubAgentConfig` 定义，提示词模板位于 `enta/prompts/skills/`。

#### 🛠️ 内置工具（36 个）
- **文件**：`file_read`、`file_write`、`file_edit`、`apply_patch`、`find_tool`、`glob`、`grep`、`pdf`、`spreadsheet`
- **Shell 与执行**：`bash`、`bash_io`（后台、输出、停止、列表）、`docker`、`deploy`、`agent`（派生子 Agent）
- **Web**：`web_fetch`、`web_search`（基于 MCP）、`browser`（Playwright 自动化）
- **开发**：`lsp`（LSP 诊断/Hover）、`notebook`（IPython 内核）、`database`
- **任务系统**：`task_create`、`task_get`、`task_list`、`task_update`、`task_output`、`task_stop`
- **调度与记忆**：`cron_create`、`cron_delete`、`cron_list`、`todo`、`memory`
- **桌面与媒体**：`desktop`（pyautogui / mss / uiautomation）、`image`（Pillow / OCR）
- **集成**：`git`、`rest_client`
- **外部协议**：MCP 客户端（`enta.tools.mcp` + `MCPManager`），支持 Model Context Protocol 服务器

#### 🌐 平台适配器（18 个）
聊天平台适配器将外部消息（Telegram、Discord、Slack …）归一化为统一通道。具体包括：
- Discord、Slack、Telegram、**钉钉**、Email（IMAP+SMTP）、**飞书 / Lark**
- **企业微信**（WeCom）、**微信**（Weixin）、WhatsApp、Signal、SMS
- **Matrix**、**MS Graph**（Microsoft 365）、**Home Assistant**、**QQ Bot**、**BlueBubbles**（iMessage）
- **腾讯元宝**（Yuanbao）、**Webhook**（通用入站 Webhook）
- 所有适配器均实现 `BaseAdapter`，并通过 `enta.adapters.manager` 注册。

#### 🔒 安全与权限引擎
- **6 种权限模式**：`bypass`（无检查）、`dont_ask`（自动允许）、`accept_edits`（自动允许编辑）、`plan`（先规划）、`auto`（启发式）、`default`（请求确认）
- **Bash 命令分析器**（`enta.safety.analyze_bash_command`、`dangerous_commands.txt`）含危险模式检测
- **SSRF 防护**（`enta.ssrf.EncreSSRFGuard`）结合 DNS 解析与 CIDR 黑名单
- **Docker 容器沙箱**（`enta.sandbox.container.EncreContainerSandbox`），带 seccomp 风格的隔离
- **Rust 级沙箱**（`enta-core/sandbox.rs`）结合 Linux **Landlock** 强制（`landlock.rs`）
- **自动安全分类器**（`enta.autosafety.EncreAutoSafetyClassifier`）— 基于 AI 的权限决策
- **限流器**（`enta.ratelimit.EncreRateLimiter`）用于后端与工具调用

#### 💾 持久化记忆（`enta.memdir`）
- 以 frontmatter 解析的 Markdown 文件存储于 `memdir/` 目录
- `MEMORY.md` 入口索引，带 **老化/新鲜度追踪**（`memdir/age.py`、`memdir/manifest.py`）
- 基于 NumPy 相似度的 **语义搜索**（`memdir/semantic.py`）
- Working memory 与整合动作
- 目标/记忆提示词模板位于 `prompts/memdir/`

#### 🎯 技能系统（11 个内置）
- `debug`、`loop`、`batch`、`verify`、`stuck`、`code_review`、`refactor`、`gen_test`、`web_research`、`data_viz`、`write_docs`
- **基于优先级的覆盖**：managed > user > project > bundled
- **动态系统提示词生成**（`enta.prompts.system.EncrePromptBuilder`）
- 技能生成器与整合器（`enta.learning`）支持运行时创建技能

#### 🤝 多智能体编排
- **Swarm**：`EncreTeammate`、`EncreSwarmManager`（并发受限）、`EncreMailbox`（异步邮箱）
- **Orchestrator**：`EncreOrchestrator`、`EncreBlackboard`（共享状态）、`EncreConsensus`（提案/投票）、`AgentRole` / `RoleRegistry`
- **任务规划器**：`EncreTaskPlanner` 生成 `TaskTree` / `TaskNode`
- **任务系统**：`enta.task` 提供类型化 CRUD 与 bash / agent / workflow 执行器

#### 🧱 Rust 原生核心（`native/crates/enta-core`）
- **快速文件 I/O**（`fs.rs`）— 支持 offset/limit 的原生读/写
- **正则搜索与 glob**（`search.rs`）
- **SIMD 加速模式匹配**（`simd_search.rs`）— `simd_contains`、`simd_find_all`、`simd_memmem`
- **Unified diff**（`diff.rs`）— `compute_diff` / `apply_diff`
- **沙箱化 Shell 执行**（`shell.rs` + `sandbox.rs`），结合 Landlock
- **Linux Landlock**（`landlock.rs`）— 只读文件系统、禁止网络、完整沙箱
- **启发式 Token 计数器**（`tokenizer.rs`）— 英文 / CJK / 数字 / 代码 启发式
- **语义相似度**（`embedding.rs`）— Cosine 与 Jaccard 文本相似度
- **BM25 索引器**（`indexer.rs`）— `Bm25Index` 暴露给 Python
- **LSP 协议**（`lsp_proto.rs`）— JSON-RPC 2.0 解析器、诊断、Content-Length 辅助
- 可选 feature：`embedding`（Candle + tokenizers）、`simd`（`wide`）、`landlock`（仅 Linux）

#### 🖥️ EnTA Desktop（Electron + React 19）
- 全功能 AI 聊天 UI，Markdown 渲染（`markdown-it`）、代码高亮（`highlight.js`）、文件附件
- **多会话**聊天，支持分支与搜索
- **设置**：模型配置、网关、Agent、MCP 服务器、技能、规则、记忆、代码索引、Agent
- **iClaw 模式** — 自动化运行器，支持批量操作
- **内嵌终端**（`xterm.js` + `node-pty`），带工作目录浏览
- **代码编辑器** — Monaco Editor，内置 16+ 种语言的 worker（TS、JS、Python、Rust、Go、Java、C/C++/C#、PHP、Ruby、Swift、Kotlin、SQL 等）
- **i18n** — `en.ts` / `zh.ts` 语言包，运行时切换
- **系统托盘** + 通知 + 语音输入
- **加密浏览器 Cookie 存储** — AES-256-GCM（密钥文件位于用户数据目录）
- **Git 集成** — 状态 / diff，5 秒缓存 + 进行中 diff 跟踪
- **多目标打包**：NSIS（Windows x64）、`pkg`（macOS x64 + arm64）、`deb` / `rpm`（Linux x64）

#### 🛠️ 开发者工具链
- **上下文压缩**（`enta.compact`）— 9 种策略：Always、Auto、TokenBudget、BudgetReduction、Semantic、Snip、MicroCompact、ContextCollapse、MultiStagePipeline
- **语义压缩** — `SemanticToolOutputCompactor`、`ContextPartitioner` 与分层 Context
- **LSP 客户端** — 16+ 种语言服务器在 `PATH` 上自动发现（Python、TypeScript、JavaScript、Rust、Go、Java、C#、C++、PHP、Ruby、Swift、Kotlin、CSS、HTML、JSON、YAML …）
- **代码库索引器** — BM25 搜索 + 依赖图（`enta.codebase`）
- **进化** — 元认知、反射循环、策略优化、错误/学习器
- **反馈学习器** — 基于 Jaccard 相似度的错误修正
- **交互式 Notebook** — IPython 内核会话
- **插件系统** — `EncrePlugin` / `PluginManifest` / `PluginRegistry`
- **画像推断** + **灵魂系统** 用于用户人格
- **规范引擎** — `SpecDocument` / `SpecSection` / `EncreSpecEngine`
- **目标系统** + **调度器** + **遥测**
- **WebSocket 服务器** + 管理 HTTP API + 会话管理器
- **CLI 运行器** — `python -m enta.iclaw`（iClaw 模式）
- **Gateway 协议** — `GatewayServer` / `GatewayClient`

<h2 align="center">🔧 开发环境搭建</h2>

### 前置条件
- **Python**：3.11+（`enta` 包要求 `>=3.11`）
- **Node.js**：18+（Electron 42，桌面应用推荐 20+）
- **Rust**：1.65+（stable，可选 — 仅在需要重新编译原生扩展时使用）
- **平台**：Linux（x64、arm64）、macOS（x64、arm64）、Windows（x64）

### 克隆与安装

```bash
# 1. 克隆仓库
git clone https://github.com/mf2023/EnTA.git
cd enta

# 2. （推荐）一键构建 — Rust 扩展 + Python wheel + Desktop 打包
python build.py

# 3. --- Python Agent 框架 ---

# 以可编辑模式安装核心依赖（使用 pyproject.toml）
pip install -e .

# 安装开发依赖（pytest、ruff、mypy、pre-commit）
pip install -e ".[dev]"

# 按需安装可选后端/适配器
pip install -e ".[anthropic]"       # Anthropic Claude 后端
pip install -e ".[ollama]"          # Ollama 本地后端
pip install -e ".[discord]"         # Discord 适配器
pip install -e ".[slack]"           # Slack 适配器
pip install -e ".[telegram]"        # Telegram 适配器
pip install -e ".[dingtalk]"        # 钉钉适配器
pip install -e ".[email]"           # 邮件适配器（IMAP + SMTP）
pip install -e ".[local]"           # 本地模型后端（PyTorch + transformers）
pip install -e ".[aws]"             # AWS Bedrock 后端（boto3）
pip install -e ".[aiohttp]"         # aiohttp HTTP 工具
pip install -e ".[native]"          # 预编译的原生扩展（enta-native）

# 一键安装所有 extras（anthropic, ollama, native, aiohttp, discord, slack,
# telegram, dingtalk, email, local, aws）
pip install -e ".[all]"

# 4. --- 桌面应用 ---

cd desktop
npm install
cd ..

# Playwright 浏览器（用于 `browser` 工具）
playwright install
```

### 构建命令

```bash
# 一键：构建 Rust 扩展、把 _native 复制到 backend/enta/、
#       安装 Python 包、并打包桌面渲染器
python build.py

# 或手动构建各组件：

# Python：构建 wheel
pip install build
python -m build

# Rust：构建原生扩展（release）
cd native
cargo build --release -p enta-py
# 编译产物 _native.pyd / _native.so 会被复制到 backend/enta/
# （build.py 脚本会自动完成此步骤）

# 桌面：打包 TypeScript 并启动 Electron
cd desktop
npm run build         # esbuild → dist/main.js, dist/preload.js, renderer/bundle.js
npm start             # build + 启动 Electron
npm run dist          # build + electron-builder（NSIS / pkg / deb / rpm）
cd ..
```

### 运行测试与代码质量

```bash
# Python 测试（pytest 在 pyproject.toml 中配置：testpaths = ["backend/tests"]）
pytest -v
pytest backend/tests/test_agent.py -v
pytest backend/tests/test_backends.py -v

# Lint 与类型检查
ruff check .
mypy backend/enta

# 桌面类型检查（主进程 + 渲染端）
cd desktop
npm run typecheck
```

<h2 align="center">⚡ 快速开始</h2>

### 基础 Agent（Python API）

```python
import asyncio
from enta import EncreAgent

async def main():
    agent = EncreAgent(
        backend="openai",
        model="gpt-4o",
        api_key="...",
    )

    async for event in agent.run("写一个 Python 脚本，递归列出所有 .py 文件"):
        if event.type == "text":
            print(event.content, end="")
        elif event.type == "tool_result":
            print(f"\n[工具: {event.tool_name}] → {event.result[:100]}...")
        elif event.type == "finish":
            print(f"\n\n完成。原因: {event.reason}")

asyncio.run(main())
```

### 带工具权限/安全

```python
from enta import EncreAgent

agent = EncreAgent(
    backend="anthropic",
    model="claude-sonnet-4-20250514",
    api_key="...",
    tool_permission_mode="auto",   # 可选: bypass / dont_ask / accept_edits / plan / auto / default
)
```

### 多后端切换

```python
from enta import EncreAgent

for provider, model, key in [
    ("openai", "gpt-4o", OPENAI_KEY),
    ("anthropic", "claude-sonnet-4-20250514", ANTHROPIC_KEY),
    ("google", "gemini-2.5-pro", GOOGLE_KEY),
]:
    agent = EncreAgent(backend=provider, model=model, api_key=key)
    async for event in agent.run("用一句话解释 Rust 的所有权"):
        if event.type == "text":
            print(event.content, end="")
    print("\n" + "=" * 40)
```

### CLI：iClaw 模式

```bash
# iClaw 是一个长生命周期自动化运行器（enta/iclaw/__main__.py）
python -m enta.iclaw --help
```

### MCP 服务器

```python
from enta.tools.mcp_manager import MCPManager, bootstrap_mcp_servers

# 从用户默认配置路径引导 MCP 服务器
manager = MCPManager()
await bootstrap_mcp_servers(manager)
```

<h2 align="center">🖥️ 桌面应用</h2>

EnTA Desktop 是一个基于 Electron + React 19 的 AI 聊天应用：

- **聊天**：Markdown 渲染（`markdown-it`）、代码高亮（`highlight.js`）、文件附件
- **会话**：多并发会话、分支、模糊搜索（`fuse.js`）
- **设置**：模型配置、网关、Agent、MCP 服务器、技能、规则、记忆、代码索引、Agent
- **iClaw 模式**：自动化运行器，支持批量操作
- **终端**：内嵌终端（`xterm.js` + `node-pty`）
- **编辑器**：Monaco Editor，16+ 种语言 worker（TypeScript、JavaScript、Python、Rust、Go、Java、C/C++/C#、PHP、Ruby、Swift、Kotlin、SQL …）
- **i18n**：英文（`en.ts`）与中文（`zh.ts`）本地化，运行时切换
- **Git 集成**：状态 / diff，5 秒缓存 + 进行中 diff 跟踪
- **系统托盘 + 通知**
- **加密浏览器 Cookie 存储**（AES-256-GCM，密钥位于用户数据目录）

**启动：**

```bash
cd desktop
npm start
```

**打包分发：**

```bash
cd desktop
npm run dist
# → Windows: NSIS 安装包（x64）
# → macOS:   .pkg（x64 + arm64）
# → Linux:   .deb 与 .rpm（x64）
```

<h2 align="center">🔧 配置</h2>

### 配置文件示例

```yaml
# config.yaml
backend: "openai"
model: "gpt-4o"
api_key: "${OPENAI_API_KEY}"

safety:
  tool_permission_mode: "auto"   # bypass / dont_ask / accept_edits / plan / auto / default

memory:
  enabled: true
  max_tokens: 4096

compact:
  strategy: "auto"               # enta.compact 9 种策略之一
  threshold_tokens: 32000

tools:
  enabled:
    - file_read
    - file_write
    - bash
    - web_fetch
    - web_search
  disabled:
    - browser
    - notebook
```

### 配置来源（优先级从低到高）

1. 内置默认值（`enta.config.EncreConfig`）
2. 配置文件（YAML、TOML）
3. 环境变量（前缀 `ENTA_`）
4. 创建 Agent 时的程序化配置（传入 `EncreAgent(...)`）

<h2 align="center">❓ 常见问题</h2>

**问：EnTA 支持多少个 LLM 后端？**
答：**31** 个后端：OpenAI、Anthropic、DeepSeek、Google Gemini、Groq、Ollama、Local（HuggingFace）、阿里（Qwen / DashScope）、腾讯、小米、Kimi、GLM、Minimax、Arcee、Novita、OpenRouter、AWS Bedrock、GitHub Copilot、LM Studio、OpenAI Compatible、OpenAI SSE、AI Gateway、Kilocode、OpenCode，以及元后端 Failover、Router、Retry，外加 MCP Catalog 与静态模型目录 `enta.backends.catalog`。

**问：EnTA 内置多少个工具？**
答：**36** 个内置工具，位于 `enta/tools/builtin/`，并配以可插拔的工具注册表与 MCP 客户端，可接入外部工具。

**问：如何添加自定义工具？**
答：继承 `EncreTool`（`enta/tools/base.py`），实现 `execute(**kwargs) -> str` 与格式化方法，然后使用 `ToolRegistry` 注册（或通过 `enta/tools/discovery.py` 的发现机制）。

**问：安全/权限模式有哪些？**
答：**6** 种模式：`bypass`（无检查）、`dont_ask`（自动允许）、`accept_edits`（自动允许编辑）、`plan`（先规划）、`auto`（启发式）、`default`（请求确认），定义于 `enta.utils.types.PermissionMode`。

**问：这是一个可以通过 `pip install` 安装的 PyPI 库吗？**
答：不是。EnTA 以单一仓库形式发布，包含 Python、Rust 和 Electron/TypeScript 代码；三个组件共用同一条发版线。如需使用，请克隆仓库并按照开发环境搭建指南操作，然后通过 `pip install -e .` 从源码安装 Python 包。

**问：记忆系统如何工作？**
答：记忆以 frontmatter 解析的 `.md` 文件存储在 `memdir/` 目录中，由 `MEMORY.md` 索引（`enta.memdir`）。系统追踪新鲜度/老化程度，自动修剪过期条目，并提供基于 NumPy 相似度的语义搜索。

**问：EnTA 是否支持多 Agent 工作流？**
答：支持。**Swarm** 系统（`enta.swarm`）提供队友、并发受限的 Swarm 管理器、异步邮箱、Orchestrator、Blackboard 与共识协议。**Task** 系统（`enta.task`）提供类型化 CRUD 与 bash / agent / workflow 执行器。`enta/agents/builtin.py` 中还定义了 **9 个内置子 Agent**（coder、researcher、critic、architect、planner、spec-writer、monitor、executor、scheduler）。

**问：Rust 在 EnTA 中扮演什么角色？**
答：性能关键的操作 — 文件 I/O、正则搜索、glob、unified diff、SIMD 模式匹配、BM25 索引、沙箱 / Landlock、Token 计数、语义相似度、LSP JSON-RPC 解析 — 都由 Rust 实现（`native/crates/enta-core`），并通过 PyO3 暴露给 Python（`enta._native`）。通过 `python build.py` 或 `cargo build --release -p enta-py` 即可构建。

**问：如何构建桌面应用进行分发？**
答：在 `desktop/` 目录中运行 `npm run dist`。electron-builder 会生成 NSIS 安装包（Windows x64）、`.pkg`（macOS x64 + arm64）、`.deb` 与 `.rpm`（Linux x64）。产物输出至 `desktop/release/`。

**问：聊天平台适配器位于哪里？**
答：位于 `enta/adapters/`，共 18 个适配器，均实现 `BaseAdapter`，包括 Discord、Slack、Telegram、钉钉、飞书、企业微信、微信、WhatsApp、Signal、SMS、Email、Matrix、MS Graph、Home Assistant、QQ Bot、BlueBubbles、腾讯元宝以及通用 Webhook 适配器。

**问：如何将 EnTA 作为服务器运行？**
答：`enta.server` 暴露了 WebSocket 服务器与管理 HTTP API。可使用 `EncreServer` / `run_server`（`enta.server.app`，由 `enta/__init__.py` 惰性加载）以及 `enta.server.session_manager` 中的 `SessionManager`。

<h2 align="center">🌏 社区与许可</h2>

- 欢迎提交 Issue 和 PR！
- GitHub：https://github.com/mf2023/EnTA
- Gitee：https://gitee.com/dunimd/enta
- GitCode：https://gitcode.com/dunimd/enta

<div align="center">

## 📄 许可证与开源协议

### 🏛️ 项目许可证

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache License 2.0">
  </a>
</p>

本项目采用 **Apache License 2.0**。详见 [LICENSE](LICENSE) 文件。

### 📋 依赖许可协议

下表汇总了 **Python 框架**、**Rust 核心** 与 **Electron 桌面应用** 实际使用的依赖包。版本号与精确的许可文本不在此处固定，请以各上游项目为准。

<div align="center">

| 📦 包名 | 📜 协议 | 📦 包名 | 📜 协议 |
|:-----------|:-----------|:-----------|:-----------|
| httpx | BSD-3-Clause | pydantic | MIT |
| beautifulsoup4 | MIT | markdownify | MIT |
| lxml | BSD-3-Clause | tomli | MIT |
| tomli-w | MIT | pyyaml | MIT |
| cryptography | Apache-2.0 / BSD | zero-api-key-web-search | Apache-2.0 |
| pathspec | MPL-2.0 | websockets | BSD-3-Clause |
| Pillow | Historical | playwright | Apache-2.0 |
| tiktoken | MIT | numpy | BSD-3-Clause |
| loguru | MIT | openai | Apache-2.0 |
| anthropic | MIT | google-generativeai | Apache-2.0 |
| ollama | MIT | groq | MIT |
| aiohttp | Apache-2.0 | discord.py | MIT |
| slack_bolt | MIT | slack_sdk | MIT |
| python-telegram-bot | GPL-3.0 | dingtalk-stream | MIT |
| aioimaplib | BSD-3-Clause | aiosmtplib | MIT |
| torch | BSD-3-Clause | transformers | Apache-2.0 |
| boto3 | Apache-2.0 | pytest | MIT |
| pytest-asyncio | Apache-2.0 | ruff | MIT |
| mypy | MIT | pre-commit | MIT |
| mss | MIT | openpyxl | MIT |
| pdfplumber | MIT | pyautogui | BSD-3-Clause |
| pypdf | BSD-3-Clause | PyPDF2 | BSD-3-Clause |
| pytesseract | Apache-2.0 | uiautomation | MIT |
| watchfiles | MIT | | |
| serde | MIT/Apache-2.0 | serde_json | MIT/Apache-2.0 |
| regex | MIT/Apache-2.0 | walkdir | MIT/Apache-2.0 |
| similar | MIT | glob | MIT/Apache-2.0 |
| tempfile | MIT/Apache-2.0 | candle-core | Apache-2.0 |
| tokenizers | Apache-2.0 | wide | MIT/Apache-2.0 |
| pyo3 | MIT/Apache-2.0 | | |
| electron | MIT | electron-builder | MIT |
| esbuild | MIT | typescript | Apache-2.0 |
| @xterm/xterm | MIT | @xterm/addon-fit | MIT |
| @xterm/addon-webgl | MIT | node-pty | MIT |
| markdown-it | MIT | highlight.js | BSD-3-Clause |
| fuse.js | Apache-2.0 | monaco-editor | MIT |
| react | MIT | react-dom | MIT |
| simple-icons | CC0-1.0 | | |

</div>

</div>
