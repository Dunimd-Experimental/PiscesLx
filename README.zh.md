# ⚠️ 合规提示

根据中国《生成式人工智能服务管理暂行办法》等相关规定，PiscesL1 未提供模型备案服务；若在中国境内向公众提供任何形式的服务，请自行完成备案。

---

<div align="center">

# PiscesL1

简体中文 | [English](README.md)

<a href="https://space.bilibili.com/3493284091529457" target="_blank">
    <img alt="BiliBili" src="https://img.shields.io/badge/BiliBili-PiscesL1-00A1D6?style=flat-square&logo=bilibili"/>
</a>
<a href="https://gitee.com/dunimd" target="_blank">
    <img alt="Gitee" src="https://img.shields.io/badge/Gitee-Dunimd-C71D23?style=flat-square&logo=gitee"/>
</a>
<a href="https://github.com/mf2023/piscesl1" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-PiscesL1-181717?style=flat-square&logo=github"/>
</a>
<a href="https://huggingface.co/dunimd" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-dunimd-FFD21E?style=flat-square&logo=huggingface"/>
</a>
<a href="https://modelscope.cn/organization/dunimd" target="_blank">
    <img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-dunimd-1E6CFF?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTcuMDA2IDBDMy4xNDIgMCAwIDMuMTQyIDAgNy4wMDZTMy4xNDIgMTQuMDEyIDcuMDA2IDE0LjAxMkMxMC44NyAxNC4wMTIgMTQuMDEyIDEwLjg3IDE0LjAxMiA3LjAwNkMxNC4wMTIgMy4xNDIgMTAuODcgMCA3LjAwNiAwWiIgZmlsbD0iIzFFNkNGRiIvPgo8L3N2Zz4K"/>
</a>

</div>

采用 Arctic 架构的轻量级多模态混合专家模型（MoE），支持文本、图像、音频、视频、文档与智能体理解。PiscesL1（PiscesLx 系列，Dunimd 项目组）面向研究与实用，可在单张 RTX 4090 上运行，体系可扩展至 1T 参数规模。

## ❄️ Arctic 架构创新

### 🧠 多路径推理引擎（PiscesReasoner）
- 层次推理链（HRC）：多层抽象
- 并行假设思维：最多 8 路并行与动态选择
- 动态事实验证：实时真值校验与一致性评分
- 元认知反思：对推理过程进行不确定性量化
- 推理特殊 tokens：`<|start_hypothesis|>`, `<|start_evidence|>`, `<|start_conclusion|>`, `<|hypothesis_split|>`, `<|hypothesis_merge|>`

### 🔧 MoE 专家系统
- 8 专家 Top-2 路由与 StableMoEGate
- LSTM 负载预测：动态容量与专家分配
- 梯度检查点友好：固定形状模式
- 稳定门控：噪声注入与容量控制

### 🌐 五模态编码系统
- VisionEncoder：NaViT 原生分辨率（最高 1024px）
- VideoEncoder：时序视觉、帧级注意力
- AudioEncoder：高级音频特征提取
- DocEncoder：文档结构理解（LayoutLMv3）
- AgentEncoder：智能体行为建模（观察 / 行动 / 反思）

### ⚛️ 高级多模态融合
- DynamicModalFusion：统一 token 级多模态融合
- 原生跨模态注意力（无需张量网络）
- 硬件自适应配置
- 质量感知门控

### 📏 超长上下文
- YaRN RoPE：支持 10M+ token，动态 NTK 缩放
- H2O 注意力：面向大模型的流式注意力
- 动态位置编码与长因子缩放
- 内存友好滑窗与压缩

### 🤖 原生智能体系统
- PiscesAgent：原生多模态智能体，支持 MCP 协议
- 工具集成与环境交互
- 持久化记忆与经验累积
- 多智能体通信与分布式推理

### 🎯 高级优化（K-FAC 增强）
- K-FAC 二阶近似
- 自适应梯度裁剪
- 自然梯度下降
- 相比完整二阶方法显著节省内存

---

## 🚀 特性

- Arctic 架构与多路径推理
- 五模态统一理解与跨模态融合
- 8 专家 Top-2 路由与负载预测
- 10M+ 上下文窗口
- 多位量化（2/4/8）
- 0.5B–1.5B 在 ~14.6GB 显存可训练（QLoRA + Checkpoint）
- 原生 MCP 智能体集成
- 一条命令全流程：`python manage.py`

---

## 🛠️ 安装与环境

- Python：推荐 3.11
- CUDA：11.8+（GPU 训练与推理）
- 依赖：详见 `requirements.txt`

### 快速设置
```bash
git clone https://gitee.com/dunimd/piscesl1.git
# 或
git clone https://github.com/mf2023/piscesl1.git
cd piscesl1
python manage.py setup
```

---

## ⚡ 命令行使用

所有命令通过：
```bash
python manage.py <command>
```
查看帮助：
```bash
python manage.py help
```

### 主要命令
| 命令       | 描述                                                             |
|------------|------------------------------------------------------------------|
| setup      | 环境设置与依赖安装                                               |
| update     | 从远程仓库拉取最新代码                                           |
| version    | 显示当前版本与更新摘要                                           |
| changelog  | 显示版本历史（支持 --all / --version X.X.XXXX）                  |
| train      | 训练模型（支持量化 / LoRA / RLHF）                               |
| infer      | 模型推理（支持 MCP 集成与推测式解码）                            |
| check      | 检查 GPU 与依赖项                                                |
| monitor    | 系统监控（GPU/CPU/内存）                                         |
| download   | 下载数据集                                                       |
| dataset    | 数据集管理与转换                                                 |
| cache      | 缓存维护（stats / clear-dataset / clear-downloads / clear-all）  |
| benchmark  | 模型评测与基准测试                                               |
| mcp        | MCP 工具管理（status / warmup / refresh-cache）                  |
| watermark  | 水印检测（文本/文件，支持批量与 JSON 输出）                      |
| help       | 显示帮助信息                                                     |

#### 示例
```bash
# 基础
python manage.py version
python manage.py changelog --all
python manage.py changelog --version 1.0.0150

# 数据集
python manage.py download --max_samples 50000

# 训练
python manage.py train --model_size 0.5B --dataset Chinese2
python manage.py train --model_size 1.5B --dataset Chinese2 --resume_ckpt runs/last.pt --reset_lr
python manage.py train --model_size 7B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora
python manage.py train --model_size 7B --dataset Chinese2 --rlhf --rlhf_dataset dunimd/human_feedback --rlhf_lr 1e-5

# 推理
python manage.py infer --ckpt ckpt/latest.pt --prompt "你好，PiscesL1!"
python manage.py infer --ckpt ckpt/model.pt --prompt "Hi" --speculative --draft_model ckpt/draft.pt --spec_gamma 4

# 评测
python manage.py benchmark --list
python manage.py benchmark --info mmlu
python manage.py benchmark --benchmark mmlu --config configs/0.5B.json --seq_len 4096 --model ckpt/model.pt
python manage.py benchmark --perf --config configs/0.5B.json --selftest

# MCP
python manage.py mcp --mcp_action status
python manage.py mcp --mcp_action warmup
python manage.py mcp --mcp_action refresh-cache

# 缓存
python manage.py cache --cache_action stats
python manage.py cache --cache_action clear-dataset
python manage.py cache --cache_action clear-downloads
python manage.py cache --cache_action clear-all
```

---

## 🧠 模型架构与配置

核心组件位于 `model/` 与 `model/multimodal/`，默认模型配置位于 `configs/model/*.json`。

| 模型大小 | 层数 | 隐藏层 | 注意力头数 | KV 头数 | MoE 专家数 | 上下文 | 默认量化 |
|----------|------|--------|------------|---------|------------|--------|----------|
| 0.5B     | 16   | 640    | 10         | 5       | 6          | 256K   | 关闭（可选） |
| 1.5B     | 16   | 896    | 14         | 7       | 6          | 256K   | 关闭（可选） |
| 7B       | 28   | 3584   | 32         | 8       | 8          | 1M     | 开启（LoRA 默认） |
| 32B      | 64   | 5120   | 40         | 8       | 8          | 1M     | 开启（LoRA 默认） |
| 64B      | 80   | 6656   | 52         | 8       | 8          | 10M    | 开启（LoRA 默认） |
| 70B      | 80   | 8192   | 64         | 8       | 8          | 10M    | 开启（LoRA 默认） |
| 128B     | 120  | 10240  | 80         | 8       | 8          | 10M    | 开启（LoRA 默认） |
| 314B     | 160  | 12288  | 96         | 12      | 16         | 10M    | 开启（LoRA 默认） |
| 671B     | 200  | 16384  | 128        | 16       | 32         | 10M    | 开启（LoRA 默认） |
| 1T       | 240  | 20480  | 160        | 20      | 64         | 10M    | 开启（LoRA 默认） |

说明：
- 量化默认由 `configs/model/*.json` 决定；可用训练参数覆盖：`--force_quant --quant_bits {2,4,8}`、`--force_lora`。

### 量化示例
```bash
# 2 位量化（实验）
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 2 --force_lora

# 4 位量化（常用）
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora

# 8 位量化（更稳定）
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 8 --force_lora
```

---

## 📦 数据集

数据集由 `configs/dataset.json` 配置并通过：
```bash
python manage.py download
```
下载。默认优先来源：ModelScope → HuggingFace（不可访问时自动镜像）。

示例（完整列表请见 `configs/dataset.json`）：
- 中文：Chinese1、Chinese2、Chinese3、Chinese4、Chinese5、Chinese6
- 英文：English1、English2、English3、English4
- 数学：Math1、Math2、Math4、Math5
- 代码：Code1、Code2、Code4
- Web：Web1、Web2、Web3
- 音频：Audio1、Audio2、Audio3
- 图像：Image1、Image2
- 文档/视觉：VQAv2、FinQA、DocVQ1A、Exam、SG1、Chat1、Publaynet1、Medical1、Financial1

---

## 🏆 ~14.6GB 显存上的训练

支持在 ~14.6GB 显存上训练 0.5B–1.5B 模型（量化 + LoRA + 内存优化）。

- 多位量化：2/4/8
- LoRA：约 0.024% 参数可训练（1.5B）
- 梯度检查点
- K-FAC 二阶增强
- 自适应梯度裁剪

示例：
```bash
# 4 位量化 + LoRA
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora

# 8 位量化 + LoRA
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 8 --force_lora
```

---

## ⚡ 快速开始
```bash
# 1. 环境
python manage.py setup

# 2. 更新（可选）
python.manage.py update

# 3. 下载默认数据集
python manage.py download

# 4. 训练 0.5B
python manage.py train --model_size 0.5B

# 5. 推理
python manage.py infer --ckpt ckpt/latest.pt --prompt "用简单的话解释机器学习"
```

## 🤖 MCP 原生智能体支持（Beta）
MCP 支持位于 `model/mcp/` 与 tools/mcp。
```bash
python manage.py mcp --mcp_action status
python manage.py mcp --mcp_action warmup
python manage.py mcp --mcp_action refresh-cache
```

---

## ❓ 常见问题（FAQ）
- 如何查看可用命令？`python manage.py help`
- 如何添加新数据集？编辑 `configs/dataset.json` 并运行 `python manage.py download`。自定义数据集建议 JSONL（text）或 Parquet（input_ids/labels）。
- 显存不足怎么办？用更小模型、降低序列长度，或启用 4 位量化（`--force_quant --quant_bits 4`，通常配合 `--force_lora`）。
- 如何恢复训练？`--resume_ckpt path/to/ckpt.pt`（可选 `--reset_lr`）
- 仅用 CPU？可使用 `--device cpu`（性能较慢）。
- 如何进行评测？`python manage.py benchmark ...`，配合 `--config`、`--seq_len`、`--model` 等参数。

---

## 📄 许可证
本项目采用 Apache License 2.0，详见 [LICENSE](LICENSE)。

---

## 🌏 社区与引用
- 欢迎提交 Issues 与 PR！
- Gitee: https://gitee.com/dunimd/piscesl1.git
- GitHub: https://github.com/mf2023/piscesl1.git
- ModelScope: https://www.modelscope.cn/models/mfchina2024/PiscesL1

<h3 align="center">以直觉航行数据之深邃 · 以共情赋予智能形态</h3>