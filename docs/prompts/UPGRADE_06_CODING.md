# 06 · 强制编码规范（不可协商）

以下规范**强制性**，适用于每一个文件。违反任何一条即视为不合格交付。

## 6.1 类命名规范

| 位置 | 类命名前缀 | 示例 |
|------|-----------|------|
| `model/` 目录内所有类 | **`YvXxxXxx`** | `YvSubconsciousSystem`, `YvMoEGate`, `YvUnifiedReasoner` |
| `model/` 目录外所有类 | **`PiscesLxXxxXxx`** | `PiscesLxTrainer`, `PiscesLxConfig` |

- `model/` 内每个类必须以 `Yv` 开头，`YvXxxXxx` 格式（驼峰、至少两段语义）
- `model/` 外每个类必须以 `PiscesLx` 开头，`PiscesLxXxxXxx` 格式
- 私有/内部类可用 `_` 前缀，但仍遵守前缀规则，如 `_YvInternalHelper`、`_PiscesLxRuntime`
- 枚举类同样遵守：`model/` 内 `Yv...`（如 `YvAttentionType`），`model/` 外 `PiscesLx...`

## 6.2 禁止直接暴露函数

**禁止在模块顶层直接暴露函数。所有功能必须以类的形式暴露。**

- 不允许在模块顶层定义可被外部调用的裸函数（如 `def compute_attention(...)`）
- **所有功能必须封装在类中**（作为方法、`@staticmethod`、`@classmethod`）
- 未暴露的、散落在模块顶层的函数，必须**移入对应的类**中暴露
- **函数命名随意**，但**类命名必须严格遵守 6.1**
- 例外：`__init__.py` 的导入、模块常量定义不受限；`if __name__ == "__main__"` 块中的辅助逻辑可接受

**重构原则**：识别顶层裸函数 → 判断归属 → 移入最相关的类（静态/实例方法）→ 更新所有调用点。

## 6.3 日志系统

**所有日志必须使用 `dmsc`（版本 0.1.8），禁止使用 Python 标准库 `logging`。**

- **禁止 `import logging`**，禁止 `logging.getLogger(...)`、`logging.basicConfig(...)` 等
- **统一使用 dmsc**：所有日志通过 dmsc 接口产生
- **重复的日志配置必须抽离**为单一的、统一的实现

**当前状态**：代码库中已存在 `utils/dc.py` 的 dmsc 集成（`PiscesLxCoreLogLevel` 等）。你需要：找出所有 `import logging` 和 logging 用法 → 统一替换为 dmsc → 抽离重复日志逻辑 → 确保 dmsc 版本 0.1.8。

## 6.4 代码抽离与复用

**任何重复内容都必须抽离（DRY），统一使用。**

- 重复的代码块、配置、工具函数必须抽离为共享实现
- **`utils/` 和 `tools/` 由你自主决定如何组织**：审查现有结构，重新组织、合并、抽离，确保无重复
- 抽离时确保：单一职责 / 无循环依赖 / 命名符合 6.1（utils/tools 在 `model/` 外用 `PiscesLx...`）/ 完整 docstring

## 6.5 文件头版权注释（逐字精确）

**每个 `.py` 文件的文件头必须逐字包含以下内容，顺序不可变更：**

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

**严格规则：**
1. **这个版权头是唯一允许的版本**，不允许任何变体、改写或自定义
2. **`from __future__ import annotations` 必须紧跟在版权信息之下**（版权头与它**绑定**）
3. **顺序强制**：shebang → encoding → 版权注释块 → `from __future__ import annotations`
4. **若有模块级 docstring**：必须写在 `from __future__ import annotations` **之下**
   - 正确顺序：shebang → encoding → 版权 → future import → docstring → import 语句 → 代码
5. 版权注释中**每个字符都不可更改**，包括 `#` 后的空行、缩进、标点

## 6.6 项目名称拼写

**只允许**：
- ✅ **`PiscesL1`**（全名，L 后的 1 是数字，必须大写 L）
- ✅ **`PiscesLx`**（项目代号，P 和 L 大写，x 小写）

**严格禁止**：❌ `Pisces`（单独使用）、`pisces`（全小写）、`PISCES`、`PiscesL`、`Piscesl1` 等任何变体

**语境选择**：指代整个项目/框架全名时用 `PiscesL1`；指代项目代号（类名前缀、命名空间）时用 `PiscesLx`。

## 6.7 规范执行要求

- 规范**强制性、不可协商**
- 在**所有新增代码**和**所有修改的既有代码**中严格遵守
- **全面审查并修正既有代码**中的违规（`import logging`、顶层裸函数、命名不符等）
- 编写自动化检查脚本（作为测试的一部分）验证规范符合性

## 6.8 把"禁止"变成可运行的检查（强制）

**所有的"禁止事项"都必须转化为可自动运行的检查脚本/测试**，不能只停留在文字要求。这是为了让"如实违规"立刻变成"可检测失败"，而不是靠自觉。

**要求**：
- 编写 `tools/check/` 下的自动化脚本（类内实现，遵守 6.1/6.2），对以下内容做**可机检**：
  - 命名规范（`model/` 内 `Yv*`、`model/` 外 `PiscesLx*`）
  - 项目名拼写（扫描 `Pisces`/`pisces` 等违规变体）
  - 顶层裸函数检测（AST 扫描模块级 `def`）
  - 禁止 `import logging`
  - 版权头逐字校验（含 future import 紧跟其后）
  - 硬编码扫描（可疑的绝对路径、`cuda:\d` 字面量、magic number）
  - 残留引用扫描（已删除的 EnTA/Encre/水印/doc 模态/agent 模态标识符）
  - `document`/`agent` 作为独立模态的残留标识符
- 每个检查项必须能**输出违规文件与行号**，并在 CI/测试中失败
- **验收时直接运行这些脚本**，以脚本结果为准，而不是人工声明
- 对于**无法机检**的禁止项（如"禁止造假""禁止缺创新"），必须设计**替代证据要求**（如必须提交真实日志、真实权重 hash、证明文档），并在交付物清单中强制核验

> 原则：**能被脚本抓住的，就不要靠自觉；不能机检的，就用证据强制。**
