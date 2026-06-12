<div align="center">

# Encre

English | [简体中文](README.zh.md)

[Documentation](https://dunimd.github.io/encre) | [Changelog](docs/CHANGELOG.md) | [Security](docs/SECURITY.md) | [Contributing](docs/CONTRIBUTING.md) | [Code of Conduct](docs/CODE_OF_CONDUCT.md)

<a href="https://github.com/mf2023/Encre" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-encre-181717?style=flat-square&logo=github"/>
</a>
<a href="https://gitee.com/dunimd/encre" target="_blank">
    <img alt="Gitee" src="https://img.shields.io/badge/Gitee-encre-C71D23?style=flat-square&logo=gitee"/>
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

**Encre** — A Python agent framework, a Rust native core, and an Electron desktop app, all developed together in one repository. Backend-agnostic and multi-provider, with streaming tool calls, a permission/safety engine, persistent memory, multi-agent orchestration, a desktop UI, a CLI runner, a WebSocket server, and 18 platform adapters.

</div>

<h2 align="center">🏗️ Core Architecture</h2>

### Multi-Language Architecture
Encre uses a multi-language architecture with three major components, letting each layer use the most appropriate language for its domain:

<div align="center">

| Language | Component | Purpose |
|:---------|:----------|:--------|
| **Python** | `backend/encre/` | AI Agent framework core — agent loop, 31 LLM backends, 36 built-in tools, 18 adapters, safety engine, memory, skills, swarm, hooks, LSP, MCP |
| **Rust** | `native/crates/encre-core` | Native high-performance library — file I/O, regex search, glob, diff, sandbox, tokenizer, BM25 indexer, Landlock, SIMD search, LSP protocol, semantic embeddings |
| **Rust** | `native/crates/encre-py` | PyO3 bindings exposing the Rust core to Python as `encre._native` |
| **TypeScript** | `desktop/` | Encre Desktop — Electron-based desktop app with React UI, Monaco Editor, xterm.js terminal, IPC bridge to the Python backend |

</div>

Encre is shipped as a single repository with three components that share the same release cycle. There is no standalone library or separately installable package — clone the whole repository to use Encre.

### Repository Structure

```
d:\encre/
├── pyproject.toml             # Python build & dependencies (builds backend/encre/)
├── build.py                   # One-shot build script (Rust + Python + Desktop)
├── gen.py                     # Icon / asset generator
├── package-lock.json          # Root node lockfile (mirrors desktop)
├── LICENSE                    # Apache 2.0 license
├── README.md / README.zh.md   # This document
│
├── backend/                   # Python agent framework
│   ├── encre/                 #   The `encre` Python package
│   │   ├── __init__.py        #   Public API surface
│   │   ├── agent.py           #   EncreAgent — public Agent class
│   │   ├── loop.py            #   EncreLoop — execution loop
│   │   ├── session.py         #   EncreSession — conversation state
│   │   ├── safety.py          #   EncreSafetyEngine — 6 permission modes
│   │   ├── autosafety.py      #   EncreAutoSafetyClassifier
│   │   ├── config.py          #   Configuration management
│   │   ├── crypto.py          #   AES-GCM encryption helpers
│   │   ├── ssrf.py            #   EncreSSRFGuard (DNS + CIDR blocklists)
│   │   ├── ratelimit.py       #   EncreRateLimiter
│   │   ├── rollback.py        #   Git-based rollback (EncreRollbackGit)
│   │   ├── recovery.py        #   Error recovery engine
│   │   ├── scheduler.py       #   EncreScheduler (cron-style jobs)
│   │   ├── goal.py            #   Goal runner + evaluator loop
│   │   ├── telemetry.py       #   EncreTelemetry (turns, tools, retries)
│   │   ├── native.py          #   Python wrapper over encre._native
│   │   ├── _native.pyi        #   Type stubs for the Rust extension
│   │   ├── backend.py         #   create_backend() factory
│   │   ├── dangerous_commands.txt  # Bash safety patterns
│   │   ├── backends/          #   31 LLM provider adapters
│   │   ├── adapters/          #   18 platform adapters (chat platforms)
│   │   ├── agents/            #   Built-in sub-agent definitions
│   │   ├── channels/          #   Transport: websocket, terminal, HTTP, slash
│   │   ├── tools/             #   36 built-in tools + tool registry + MCP
│   │   ├── hooks/             #   EncreHookSystem
│   │   ├── memdir/            #   Persistent memory system (frontmatter)
│   │   ├── skills/            #   Skill registry + 11 bundled skills
│   │   ├── swarm/             #   Multi-agent system (teammate/mailbox/...)
│   │   ├── task/              #   Task manager & executor
│   │   ├── server/            #   WebSocket server + admin HTTP API
│   │   ├── gateway/           #   Gateway client/server protocol
│   │   ├── compact/           #   Context compaction (9 strategies)
│   │   ├── lsp/               #   LSP client + multi-server manager
│   │   ├── codebase/          #   Code indexer (BM25 + dependency graph)
│   │   ├── computer/          #   Desktop & browser automation
│   │   ├── evolution/         #   Meta-cognition, reflex, strategy optimizer
│   │   ├── feedback/          #   Error-correction learner
│   │   ├── notebook/          #   Interactive Python kernel session
│   │   ├── plugins/           #   Plugin registry & manifest types
│   │   ├── profile/           #   User profile inference
│   │   ├── soul/              #   Soul system files (persona/memory)
│   │   ├── spec/              #   Spec document engine
│   │   ├── prompts/           #   Prompt blocks, skills, safety, goals
│   │   ├── sandbox/           #   Docker container sandbox
│   │   ├── search/            #   MCP-backed web search
│   │   ├── learning/          #   Skill generation + consolidation
│   │   ├── rules/             #   Rules loader
│   │   ├── thinking/          #   Thinking config resolution
│   │   ├── iclaw/             #   iClaw CLI (automation runner)
│   │   ├── git/               #   Git repository & diff utilities
│   │   └── utils/             #   ID generation, token counting, types
│   └── tests/                 # Pytest suite (configured via pyproject.toml)
│       ├── conftest.py
│       ├── test_agent.py
│       ├── test_backends.py
│       ├── test_backend_*.py
│       ├── test_safety.py
│       ├── test_session.py
│       ├── test_loop.py
│       ├── test_native.py
│       └── ...                #   40+ test modules
│
├── native/                    # Rust workspace
│   ├── Cargo.toml             #   Workspace root
│   └── crates/
│       ├── encre-core/        #   Native core (package name: `encre`)
│       │   ├── Cargo.toml
│       │   └── src/
│       │       ├── lib.rs
│       │       ├── fs.rs          # Native read/write
│       │       ├── search.rs      # Regex search, glob
│       │       ├── simd_search.rs # SIMD-accelerated matching
│       │       ├── diff.rs        # Unified diff + apply
│       │       ├── shell.rs       # Sandboxed shell execution
│       │       ├── sandbox.rs     # Sandbox result types
│       │       ├── landlock.rs    # Linux Landlock enforcement
│       │       ├── tokenizer.rs   # Heuristic token counter
│       │       ├── embedding.rs   # Cosine / Jaccard similarity
│       │       ├── indexer.rs     # BM25 code search index
│       │       └── lsp_proto.rs   # LSP JSON-RPC parser/builder
│       └── encre-py/          # PyO3 bindings → `encre._native`
│           ├── Cargo.toml
│           └── src/lib.rs
│
├── desktop/                   # Electron desktop application
│   ├── main.ts                #   Electron main process
│   ├── preload.ts             #   Context bridge (IPC)
│   ├── build.js               #   esbuild configuration
│   ├── package.json           #   Node.js dependencies & scripts
│   ├── electron-builder.yml   #   NSIS / pkg / deb packaging
│   ├── tsconfig.json          #   TypeScript config (main process)
│   ├── fetch_icons.js         #   Icon fetcher
│   └── renderer/              #   Frontend (React 19)
│       ├── index.html
│       ├── styles.css
│       ├── design-system.css
│       ├── xterm.css
│       ├── bundle.js          #   esbuild output
│       ├── tsconfig.json
│       ├── assets/            #   Logos & icons
│       ├── src/               #   TypeScript sources
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
│       │   ├── crypto.ts       #   Renderer-side AES helpers
│       │   ├── easter-egg.ts
│       │   ├── transition-helper.ts
│       │   ├── types.ts
│       │   ├── global.d.ts
│       │   └── locales/        #   en.ts / zh.ts
│       └── vs/                 #   Monaco Editor (bundled)
│
├── docs/                      # Project documentation
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
        └── build-binary.yml   # CI: builds release binaries
```

<h2 align="center">🚀 Key Features</h2>

#### 🧠 Multi-Provider LLM Backends (31)
- **First-party**: OpenAI, Anthropic Claude, DeepSeek, Google Gemini, Groq, Ollama, Local (HuggingFace transformers)
- **Chinese & regional**: Alibaba (Qwen / DashScope), Tencent, Xiaomi MiMo, Kimi (Moonshot), GLM (Zhipu / Z.ai), MiniMax
- **Aggregators / routers**: OpenRouter, AI Gateway, OpenCode, Kilocode, Arcee, Novita
- **Cloud & enterprise**: AWS Bedrock, GitHub Copilot
- **Self-hosted & compatible**: LM Studio, OpenAI Compatible, OpenAI SSE
- **Meta-backends**: Failover (multi-backend health failover), Router (cost / task routing), Retry (transparent retry)
- **Catalog**: MCP Catalog, native model catalog (`encre.backends.catalog`)
- Unified streaming interface, automatic thinking-block resolution, OpenAI-compatible chat completions

#### 🤖 Built-in Sub-Agents (9)
- **General mode**: `coder`, `researcher`, `critic`
- **Workspace mode**: `architect`, `planner`
- **Plan / Spec mode**: `spec-writer`
- **iClaw (auto) mode**: `monitor`, `executor`, `scheduler`
- Each sub-agent is defined as a `SubAgentConfig` and loaded from `encre/agents/builtin.py` with prompt templates under `encre/prompts/skills/`.

#### 🛠️ Built-in Tools (36)
- **File**: `file_read`, `file_write`, `file_edit`, `apply_patch`, `find_tool`, `glob`, `grep`, `pdf`, `spreadsheet`
- **Shell & execution**: `bash`, `bash_io` (background, output, kill, list), `docker`, `deploy`, `agent` (spawn a sub-agent)
- **Web**: `web_fetch`, `web_search` (MCP-backed), `browser` (Playwright automation)
- **Development**: `lsp` (LSP diagnostics/hover), `notebook` (IPython kernel), `database`
- **Task system**: `task_create`, `task_get`, `task_list`, `task_update`, `task_output`, `task_stop`
- **Scheduling & memory**: `cron_create`, `cron_delete`, `cron_list`, `todo`, `memory`
- **Desktop & media**: `desktop` (pyautogui / mss / uiautomation), `image` (Pillow / OCR)
- **Integrations**: `git`, `rest_client`
- **External protocol**: MCP client (`encre.tools.mcp` + `MCPManager`) for Model Context Protocol servers

#### 🌐 Platform Adapters (18)
Chat-platform adapters turn external messages (Telegram, Discord, Slack, …) into a normalized channel. Concretely:
- Discord, Slack, Telegram, **DingTalk**, Email (IMAP+SMTP), **Feishu / Lark**
- **WeCom** (企业微信), **Weixin** (微信), **WhatsApp**, **Signal**, **SMS**
- **Matrix**, **MS Graph** (Microsoft 365), **Home Assistant**, **QQ Bot**, **BlueBubbles** (iMessage)
- **Yuanbao** (腾讯元宝), **Webhook** (generic incoming webhook)
- Each adapter implements `BaseAdapter` and is registered through `encre.adapters.manager`.

#### 🔒 Safety & Permission Engine
- **6 permission modes**: `bypass` (no checks), `dont_ask` (auto-allow), `accept_edits` (auto-allow edits), `plan` (plan first), `auto` (heuristic), `default` (ask for confirmation)
- **Bash command analyzer** with dangerous-pattern detection (`encre.safety.analyze_bash_command`, `dangerous_commands.txt`)
- **SSRF guard** combining DNS resolution with CIDR blocklists (`encre.ssrf.EncreSSRFGuard`)
- **Docker container sandbox** (`encre.sandbox.container.EncreContainerSandbox`) with seccomp-style isolation
- **Rust-level sandbox** (`encre-core/sandbox.rs`) with Linux **Landlock** enforcement (`landlock.rs`)
- **Auto safety classifier** (`encre.autosafety.EncreAutoSafetyClassifier`) — AI-based permission decisions
- **Rate limiter** (`encre.ratelimit.EncreRateLimiter`) for backends and tool calls

#### 💾 Persistent Memory (`encre.memdir`)
- Frontmatter-parsed Markdown files stored in a `memdir/` directory
- `MEMORY.md` entrypoint index with **aging / freshness tracking** (`memdir/age.py`, `memdir/manifest.py`)
- **Semantic search** powered by NumPy similarity (`memdir/semantic.py`)
- Working memory + consolidation actions
- Goal/memory prompt templates under `prompts/memdir/`

#### 🎯 Skill System (11 bundled)
- `debug`, `loop`, `batch`, `verify`, `stuck`, `code_review`, `refactor`, `gen_test`, `web_research`, `data_viz`, `write_docs`
- **Priority-based override**: managed > user > project > bundled
- **Dynamic system-prompt generation** via `encre.prompts.system.EncrePromptBuilder`
- Skill generator + consolidator (`encre.learning`) for runtime skill creation

#### 🤝 Multi-Agent Orchestration
- **Swarm**: `EncreTeammate`, `EncreSwarmManager` (concurrency-limited), `EncreMailbox` (async mailbox)
- **Orchestrator**: `EncreOrchestrator`, `EncreBlackboard` (shared state), `EncreConsensus` (proposal/vote), `AgentRole` / `RoleRegistry`
- **Task planner**: `EncreTaskPlanner` produces a `TaskTree` of `TaskNode`s
- **Task system**: typed CRUD with bash / agent / workflow executors (`encre.task`)

#### 🧱 Native Rust Core (`native/crates/encre-core`)
- **Fast file I/O** (`fs.rs`) — native read/write with offset/limit
- **Regex search & glob** (`search.rs`)
- **SIMD-accelerated matching** (`simd_search.rs`) — `simd_contains`, `simd_find_all`, `simd_memmem`
- **Unified diff** (`diff.rs`) — `compute_diff` / `apply_diff`
- **Sandboxed shell execution** (`shell.rs` + `sandbox.rs`) with Landlock
- **Linux Landlock** (`landlock.rs`) — read-only filesystem, no-network, full sandbox
- **Heuristic token counter** (`tokenizer.rs`) — English / CJK / digit / code heuristics
- **Semantic similarity** (`embedding.rs`) — cosine and Jaccard text similarity
- **BM25 indexer** (`indexer.rs`) — `Bm25Index` exposed to Python
- **LSP protocol** (`lsp_proto.rs`) — JSON-RPC 2.0 parser, diagnostics, content-length helpers
- Optional features: `embedding` (Candle + tokenizers), `simd` (`wide`), `landlock` (Linux only)

#### 🖥️ Encre Desktop (Electron + React 19)
- Full-featured AI chat UI with Markdown rendering (`markdown-it`), code highlighting (`highlight.js`), and file attachments
- **Multi-session** chat with branching and search
- **Settings**: model config, gateway, agent, MCP servers, skills, rules, memory, code index, agents
- **iClaw mode** — automation runner with batch operations
- **Embedded terminal** (`xterm.js` + `node-pty`) with a working directory browser
- **Code editor** — Monaco Editor (16+ language workers bundled, including TS, JS, Python, Rust, Go, Java, C/C++/C#, PHP, Ruby, Swift, Kotlin, SQL, etc.)
- **i18n** — `en.ts` / `zh.ts` locales, runtime language switching
- **System tray** + notifications + voice input support
- **Encrypted browser cookie store** — AES-256-GCM (key file in user data dir)
- **Git integration** — status / diff with 5-second cache and in-flight diff tracking
- **Multi-target packaging**: NSIS (Windows x64), `pkg` (macOS x64 + arm64), `deb` / `rpm` (Linux x64)

#### 🛠️ Developer Toolchain
- **Context compaction** (`encre.compact`) — 9 strategies: Always, Auto, TokenBudget, BudgetReduction, Semantic, Snip, MicroCompact, ContextCollapse, MultiStagePipeline
- **Semantic compact** — `SemanticToolOutputCompactor`, `ContextPartitioner` with tiered context
- **LSP client** — 16+ language servers auto-discovered on `PATH` (Python, TypeScript, JavaScript, Rust, Go, Java, C#, C++, PHP, Ruby, Swift, Kotlin, CSS, HTML, JSON, YAML, …)
- **Codebase indexer** — BM25 search + dependency graph (`encre.codebase`)
- **Evolution** — meta-cognition, reflex loop, strategy optimizer, error/learner
- **Feedback learner** — Jaccard similarity-based error correction
- **Interactive notebook** — IPython kernel session
- **Plugin system** — `EncrePlugin` / `PluginManifest` / `PluginRegistry`
- **Profile inference** + **Soul system** for user persona
- **Spec engine** — `SpecDocument` / `SpecSection` / `EncreSpecEngine`
- **Goal system** + **Scheduler** + **Telemetry**
- **WebSocket server** + admin HTTP API + session manager
- **CLI runner** — `python -m encre.iclaw` (iClaw mode)
- **Gateway protocol** — `GatewayServer` / `GatewayClient`

<h2 align="center">🔧 Development Setup</h2>

### Prerequisites
- **Python**: 3.11+ (the `encre` package requires `>=3.11`)
- **Node.js**: 18+ (Electron 42, recommended 20+ for the desktop app)
- **Rust**: 1.65+ (stable, optional — only needed if you want to rebuild the native extension)
- **Platforms**: Linux (x64, arm64), macOS (x64, arm64), Windows (x64)

### Clone & Setup

```bash
# 1. Clone the repository
git clone https://github.com/mf2023/Encre.git
cd encre

# 2. (Recommended) one-shot build — Rust extension + Python wheel + Desktop bundle
python build.py

# 3. --- Python Agent Framework ---

# Install core dependencies in editable mode (uses pyproject.toml)
pip install -e .

# Install with development dependencies (pytest, ruff, mypy, pre-commit)
pip install -e ".[dev]"

# Install optional backends / adapters as needed
pip install -e ".[anthropic]"       # Anthropic Claude backend
pip install -e ".[ollama]"          # Ollama local backend
pip install -e ".[discord]"         # Discord adapter
pip install -e ".[slack]"           # Slack adapter
pip install -e ".[telegram]"        # Telegram adapter
pip install -e ".[dingtalk]"        # DingTalk adapter
pip install -e ".[email]"           # Email adapter (IMAP + SMTP)
pip install -e ".[local]"           # Local model backend (PyTorch + transformers)
pip install -e ".[aws]"             # AWS Bedrock backend (boto3)
pip install -e ".[aiohttp]"         # aiohttp HTTP utilities
pip install -e ".[native]"          # Pre-built native extension (encre-native)

# Install every extra (anthropic, ollama, native, aiohttp, discord, slack,
# telegram, dingtalk, email, local, aws) in one go
pip install -e ".[all]"

# 4. --- Desktop Application ---

cd desktop
npm install
cd ..

# Playwright browsers (for the `browser` tool)
playwright install
```

### Build Commands

```bash
# One-shot: builds the Rust extension, copies _native into backend/encre/,
#            installs the Python package, and bundles the desktop renderer.
python build.py

# Or build each component by hand:

# Python: build a wheel
pip install build
python -m build

# Rust: build the native extension (release)
cd native
cargo build --release -p encre-py
# The compiled _native.pyd / _native.so is then copied into backend/encre/
# (the build.py script does this automatically)

# Desktop: bundle TypeScript and launch Electron
cd desktop
npm run build         # esbuild → dist/main.js, dist/preload.js, renderer/bundle.js
npm start             # build + launch Electron
npm run dist          # build + electron-builder (NSIS / pkg / deb / rpm)
cd ..
```

### Run Tests & Quality

```bash
# Python tests (pytest is configured via pyproject.toml: testpaths = ["backend/tests"])
pytest -v
pytest backend/tests/test_agent.py -v
pytest backend/tests/test_backends.py -v

# Lint & type check
ruff check .
mypy backend/encre

# Desktop type check (main process + renderer)
cd desktop
npm run typecheck
```

<h2 align="center">⚡ Quick Start</h2>

### Basic Agent (Python API)

```python
import asyncio
from encre import EncreAgent

async def main():
    agent = EncreAgent(
        backend="openai",
        model="gpt-4o",
        api_key="...",
    )

    async for event in agent.run("Write a Python script that recursively lists all .py files"):
        if event.type == "text":
            print(event.content, end="")
        elif event.type == "tool_result":
            print(f"\n[Tool: {event.tool_name}] → {event.result[:100]}...")
        elif event.type == "finish":
            print(f"\n\nDone. Reason: {event.reason}")

asyncio.run(main())
```

### With Tool Permission / Safety

```python
from encre import EncreAgent

agent = EncreAgent(
    backend="anthropic",
    model="claude-sonnet-4-20250514",
    api_key="...",
    tool_permission_mode="auto",   # one of: bypass / dont_ask / accept_edits / plan / auto / default
)
```

### Multi-Provider Switch

```python
from encre import EncreAgent

for provider, model, key in [
    ("openai", "gpt-4o", OPENAI_KEY),
    ("anthropic", "claude-sonnet-4-20250514", ANTHROPIC_KEY),
    ("google", "gemini-2.5-pro", GOOGLE_KEY),
]:
    agent = EncreAgent(backend=provider, model=model, api_key=key)
    async for event in agent.run("Explain Rust ownership in one sentence"):
        if event.type == "text":
            print(event.content, end="")
    print("\n" + "=" * 40)
```

### CLI: iClaw Mode

```bash
# iClaw is a long-running automation runner (encre/iclaw/__main__.py)
python -m encre.iclaw --help
```

### MCP Servers

```python
from encre.tools.mcp_manager import MCPManager, bootstrap_mcp_servers

# Bootstrap MCP servers from the user's default config path
manager = MCPManager()
await bootstrap_mcp_servers(manager)
```

<h2 align="center">🖥️ Desktop Application</h2>

Encre Desktop is an Electron + React 19 AI chat application:

- **Chat**: Markdown rendering (`markdown-it`), syntax highlighting (`highlight.js`), file attachments
- **Sessions**: Multiple concurrent sessions, branching, fuzzy search (`fuse.js`)
- **Settings**: Model config, gateway, agent, MCP servers, skills, rules, memory, code index, agents
- **iClaw Mode**: Automation runner with batch operations
- **Terminal**: Embedded terminal (`xterm.js` + `node-pty`)
- **Editor**: Monaco Editor with 16+ language workers (TypeScript, JavaScript, Python, Rust, Go, Java, C/C++/C#, PHP, Ruby, Swift, Kotlin, SQL, …)
- **i18n**: English (`en.ts`) and Chinese (`zh.ts`) localization, runtime switching
- **Git integration**: status / diff with 5-second cache, in-flight diff tracking
- **System tray + notifications**
- **Encrypted browser cookie store** (AES-256-GCM, key in user data dir)

**Launch:**

```bash
cd desktop
npm start
```

**Package for distribution:**

```bash
cd desktop
npm run dist
# → Windows: NSIS installer (x64)
# → macOS:   .pkg (x64 + arm64)
# → Linux:   .deb and .rpm (x64)
```

<h2 align="center">🔧 Configuration</h2>

### Configuration Example

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
  strategy: "auto"               # one of the 9 encre.compact strategies
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

### Configuration Sources (priority low → high)

1. Built-in defaults (`encre.config.EncreConfig`)
2. Configuration files (YAML, TOML)
3. Environment variables (prefixed with `ENCRE_`)
4. Programmatic configuration passed to `EncreAgent(...)`

<h2 align="center">❓ Frequently Asked Questions</h2>

**Q: How many LLM backends does Encre support?**
A: **31** backends: OpenAI, Anthropic, DeepSeek, Google Gemini, Groq, Ollama, Local (HuggingFace), Alibaba (Qwen / DashScope), Tencent, Xiaomi, Kimi, GLM, MiniMax, Arcee, Novita, OpenRouter, AWS Bedrock, GitHub Copilot, LM Studio, OpenAI Compatible, OpenAI SSE, AI Gateway, Kilocode, OpenCode, plus the meta-backends Failover, Router, Retry, and the MCP Catalog and the static `encre.backends.catalog` provider catalog.

**Q: How many built-in tools ship with Encre?**
A: **36** built-in tools under `encre/tools/builtin/`, plus a fully pluggable tool registry and an MCP client for external tools.

**Q: How do I add a custom tool?**
A: Subclass `EncreTool` (defined in `encre/tools/base.py`), implement `execute(**kwargs) -> str` and the formatting methods, then register the tool with the `ToolRegistry` (or use the discovery mechanism in `encre/tools/discovery.py`).

**Q: What are the permission / safety modes?**
A: **6** modes: `bypass` (no checks), `dont_ask` (auto-allow), `accept_edits` (auto-allow edits), `plan` (plan first), `auto` (heuristic), `default` (ask for confirmation). They are surfaced through `encre.utils.types.PermissionMode`.

**Q: Is this a PyPI library I can `pip install`?**
A: No. Encre is shipped as a single repository containing Python, Rust, and Electron/TypeScript code; the three components share a single release cycle. To use it, clone the repository and follow the Development Setup guide. The Python package can then be installed from source via `pip install -e .`.

**Q: How does the memory system work?**
A: Memory is stored as frontmatter-parsed `.md` files in a `memdir/` directory with a `MEMORY.md` index (`encre.memdir`). The system tracks freshness/age, automatically prunes aged entries, and exposes semantic search backed by NumPy similarity.

**Q: Does Encre support multi-agent workflows?**
A: Yes. The **Swarm** system (`encre.swarm`) provides teammates, a concurrency-limited swarm manager, an async mailbox, an orchestrator, a blackboard, and a consensus protocol. The **Task** system (`encre.task`) provides typed CRUD with bash / agent / workflow executors. There are also **9 built-in sub-agents** defined in `encre/agents/builtin.py` (coder, researcher, critic, architect, planner, spec-writer, monitor, executor, scheduler).

**Q: How does Encre use Rust?**
A: Performance-critical operations — file I/O, regex search, glob, unified diff, SIMD pattern matching, BM25 indexing, sandbox / Landlock, token counting, semantic similarity, and the LSP JSON-RPC parser — are implemented in Rust (`native/crates/encre-core`) and exposed to Python through PyO3 (`encre._native`). The extension is built with `python build.py` or `cargo build --release -p encre-py`.

**Q: How do I build the desktop app for distribution?**
A: Run `npm run dist` in the `desktop/` directory. `electron-builder` produces NSIS installers (Windows x64), `.pkg` packages (macOS x64 + arm64), and `.deb` / `.rpm` packages (Linux x64). Output goes to `desktop/release/`.

**Q: Where do the chat-platform adapters live?**
A: Under `encre/adapters/` — 18 adapters implementing `BaseAdapter`, including Discord, Slack, Telegram, DingTalk, Feishu, WeCom, Weixin, WhatsApp, Signal, SMS, Email, Matrix, MS Graph, Home Assistant, QQ Bot, BlueBubbles, Yuanbao, and a generic Webhook adapter.

**Q: How do I run Encre as a server?**
A: `encre.server` exposes a WebSocket server and an admin HTTP API. Use `EncreServer` / `run_server` from `encre.server.app` (lazy-loaded from `encre/__init__.py`) and the `SessionManager` from `encre.server.session_manager`.

<h2 align="center">🌏 Community & License</h2>

- Issues and PRs are welcome!
- GitHub: https://github.com/mf2023/Encre
- Gitee: https://gitee.com/dunimd/encre
- GitCode: https://gitcode.com/dunimd/encre

<div align="center">

## 📄 License & Open Source Agreements

### 🏛️ Project License

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache License 2.0">
  </a>
</p>

This project uses **Apache License 2.0**. See [LICENSE](LICENSE) for the full text.

### 📋 Dependency Licenses

The dependency license list below reflects the packages used by the **Python framework**, the **Rust core**, and the **Electron desktop app**. Versions and exact license texts are not pinned here; consult each upstream project for the authoritative license.

<div align="center">

| 📦 Package | 📜 License | 📦 Package | 📜 License |
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
