# 05 · 代码清理

## 5.1 物理删除：EnTA / Encre 训练系统

**原因**：项目不再使用 Agent 驱动训练（EnTA = Encre Train Agent）。训练由标准训练流程完成，不由 Agent 驱动。

**删除的文件/目录**：
- `model/agentic/enta/` — EnTA Trainer
- `enta/` — 根目录 Agent 后端（Python + Rust native）
- `.encre/` — Encre 配置和文档
- `configs/agentevolver/` — Agent evolver 配置
- `configs/teachers.yaml` — EnTA 教师模型配置

**清理的引用**：
- `manage.py` 中的 `--enta`, `--enta_config` 参数
- `configs/train/` 中所有引用 EnTA 的配置
- `model/config.py`（YvConfig）中的 YvEntaConfig 及相关字段
- 所有 import EnTA 模块的代码
- `opss/agents/` — Agent 基础设施（orchestrator, swarm, registry 等）

**注意**：删除后确保所有依赖被正确清理，不留下断裂的 import。

## 5.2 移至第三端：水印系统

**原因**：水印不属于训练和推理流程，应由独立第三端系统处理。

**移除的文件/目录**：
- `configs/watermark.yaml` — 水印配置
- `opss/watermark/` — 水印操作器（DCT, content, weight, compliance, audit, orchestrator, protocol）
- `tools/wmc/` — 水印检测工具

**清理的引用**：
- `manage.py` 中的 `watermark` 命令
- 所有代码中对水印模块的 import
- YvConfig 中的水印相关配置字段

**处理方式**：将水印相关代码打包为独立第三端模块（项目外的独立仓库或独立目录），从模型代码中彻底移除。

## 5.3 审查并处理：MCP / Server 基础设施

以下组件**不属于模型本身**，属于服务层基础设施：
- `opss/mcp/` — MCP 服务器基础设施（core, execution, monitor, registry, sandbox, session 等）
- `model/multimodal/server.py` — MCP 生成服务器（服务层，非模型层）

**需要保留的**（模型级工具调用能力——注意：回归为文本层面的工具调用，不是"agent 模态"）：
- `model/multimodal/mcp.py` — MCP 工具注册和协议适配
- `model/multimodal/agentic.py` — 模型级工具调用能力
- `model/multimodal/agentic_protocol.py` — 工具调用协议适配（标签/特殊 token 形式）
- `model/multimodal/react_agentic.py` — ReAct 框架
- `model/multimodal/seer_executor.py` — 工具执行
- `model/multimodal/tool_executor.py` — 工具执行
- `model/multimodal/enhanced_agentic.py` — 增强工具调用能力

**需要移除/改造的**（按 3.3.1 模态重定义）：
- `model/multimodal/agentic_encoder.py` — 作为"agent 模态编码器"的定位应移除（agent 是文本，不是模态）
- `model/multimodal/doc*.py` / `YvDocEncoder` — document 模态编码器应移除（文档走 Agent/文本能力）

**处理方式**：将服务层 MCP 基础设施移至第三端或独立服务模块，保留模型级工具调用能力，移除 document/agent 模态编码器。清理 `manage.py` 中的 `mcp` 命令。

## 5.4 全面审查其他非模型组件

除上述外，**自主审查整个代码库**，识别其他不属于"模型架构 + 训练引擎 + 推理引擎"的组件并处理：
- 任何与模型训练/推理无关的功能模块
- 任何只服务于已删除系统（EnTA, 水印）的辅助代码
- 任何冗余的配置文件和工具脚本
- `manage.py` 中失效的命令

**处理方式**：物理删除或移至第三端，确保核心代码库干净。
