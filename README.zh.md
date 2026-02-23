<div align="center">

# ⚠️ 合规提示

**根据各国相关法律法规（包括但不限于中国《生成式人工智能服务管理暂行办法》、欧盟《人工智能法案》、美国《AI风险管理框架》、日本《AI指导原则》等），开发者或使用者需自行承担合规责任，未履行相关义务可能导致服务被叫停、面临监管处罚或承担法律责任。**

---

# PiscesL1

[English](README.md) | 简体中文 

<a href="https://space.bilibili.com/3493284091529457" target="_blank">
    <img alt="BiliBili" src="https://img.shields.io/badge/BiliBili-Dunimd-00A1D6?style=flat-square&logo=bilibili"/>
</a>
<a href="https://x.com/Dunimd2025" target="_blank">
    <img alt="X" src="https://img.shields.io/badge/X-Dunimd-000000?style=flat-square&logo=x"/>
</a>

<a href="https://gitee.com/dunimd" target="_blank">
    <img alt="Gitee" src="https://img.shields.io/badge/Gitee-Dunimd-C71D23?style=flat-square&logo=gitee"/>
</a>
<a href="https://github.com/mf2023/piscesl1" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-PiscesL1-181717?style=flat-square&logo=github"/>
</a>
<a href="https://huggingface.co/dunimd" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-Dunimd-FFD21E?style=flat-square&logo=huggingface"/>
</a>
<a href="https://modelscope.cn/organization/dunimd" target="_blank">
    <img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-Dunimd-1E6CFF?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTcuMDA2IDBDMy4xNDIgMCAwIDMuMTQyIDAgNy4wMDZTMy4xNDIgMTQuMDEyIDcuMDA2IDE0LjAxMkMxMC44NyAxNC4wMTIgMTQuMDEyIDEwLjg3IDE0LjAxMiA3LjAwNkMxNC4wMTIgMy4xNDIgMTAuODcgMCA3LjAwNiAwWiIgZmlsbD0iIzFFNkNGRiIvPgo8L3N2Zz4K"/>
</a>

采用**Yv架构**的高性能多模态混合专家模型（MoE），支持文本、图像、音频、视频、文档与智能体理解。PiscesL1（PiscesLx 系列，Dunimd团队）面向研究与实用，可在单张 RTX 4090 上运行，体系可扩展至 1T 参数规模。

</div>

<h2 align="center">Yv架构</h2>

### 🧠 YvUnifiedReasoner - 统一推理系统

YvUnifiedReasoner 实现了智能路由的统一推理框架，在链式思维（CoT）与多路径推理引擎之间动态切换：

- **YvCoTMemoryReasoner**：记忆增强的链式思维推理器，支持自适应深度控制（1-3层）、早期停止机制、以及错误分析与自纠错
- **YvMultiPathReasoningEngine**：多路径推理引擎，支持最多8路假设流并行探索，配合动态事实验证与元认知不确定性评分
- **智能路由**：基于问题复杂度与序列长度自动选择最优推理路径
- **控制符支持**：`<|start_hypothesis|>`, `<|start_evidence|>`, `<|start_conclusion|>`, `<|hypothesis_split|>`, `<|hypothesis_merge|>` 等控制符帮助外部工具精确追踪模型思维路径

### 🔧 Yv MoE Scaling - DeepSeek-V3风格MoE

完整的DeepSeek-V3风格混合专家系统实现：

- **YvStableMoEGate**：带LSTM负载预测器的稳定门控，支持6-64专家的Top-K路由
- **Fine-grained Expert Segmentation**：细粒度专家分割，每个"专家"是多个子专家的组合，实现更灵活的路由
- **Shared Expert Isolation**：共享专家隔离，部分专家始终激活处理所有token
- **Auxiliary Loss-free Load Balancing**：无辅助损失的负载均衡，避免传统辅助损失对模型质量的影响
- **UltraMem TDQKR Optimization**：Tucker分解查询键检索优化，将路由复杂度从O(N)降至O(√N)
- **Dynamic Device Migration**：动态专家迁移，支持大规模专家池的高效内存管理

### 🌐 Multimodal Perception Stack - 多模态感知栈

六模态统一感知架构：

- **YvVisionEncoder**：NaViT风格patch编码，支持原生分辨率（最高2048px）、patch packing
- **YvVideoEncoder**：帧级注意力编码，支持3D RoPE时空位置编码
- **YvAudioEncoder**：音频频谱编码，支持流式音频处理
- **YvDocEncoder**：LayoutLMv3风格文档编码，支持布局感知结构推理
- **YvAgenticEncoder**：智能体状态编码，支持动作空间与状态表示
- **YvCrossModalAttention**：跨模态注意力，实现模态间的深度交互

### ⚛️ YvDynamicModalFusion - 动态模态融合

Token级多模态融合系统：

- **Cross-Modal Attention**：跨模态注意力实现模态间信息交互
- **Modality-Aware Position Embeddings**：模态感知位置嵌入
- **Quality-Weighted Gating**：质量加权门控，根据融合质量动态调整权重
- **YvEnhancedModalFusion**：增强融合模块，支持对比跨模态对齐与在线自适应权重
- **多种融合策略**：支持在文本序列前插入融合token、拼接3D特征或输出压缩摘要

### 📏 Ultra-Long Context Fabric - 超长上下文架构

业界领先的10M+ token上下文支持：

- **YaRN RoPE + Dynamic NTK Scaling**：动态NTK缩放的YaRN位置编码，支持10M+ token外推
- **H2O Heavy-Hitter Oracle Attention**：重击者预言机注意力，保留重要token实现超长上下文
- **Streaming Attention**：流式注意力，支持无限长度生成
- **Sliding Window Attention**：滑动窗口注意力，局部注意力与全局token结合
- **Linear Attention**：O(n)复杂度的线性注意力，支持ELU/Performer/Softmax特征映射
- **Paged Attention**：分页注意力，高效的KV缓存管理与共享
- **Ring Attention**：环形注意力，分布式超长上下文处理
- **Attention Sinks**：注意力汇，保证流式推理的稳定性

### 🔥 Hybrid Attention-SSM - 混合注意力-状态空间模型

业界前沿的混合架构实现：

- **Mamba-3 Integration**：完整的Mamba-3 SSM集成，支持梯形离散化、复数状态、MIMO结构
- **YvSelectiveSSM**：选择性状态空间模型，输入依赖的状态转移
- **Progressive Gating**：渐进式门控，从纯注意力平滑过渡到混合模式，保证训练稳定性
- **Adaptive Routing**：自适应路由，基于序列特征动态选择注意力或SSM
- **Jamba-style Interleaved Architecture**：Jamba风格交错架构，注意力层与SSM层交替

### 🎯 Advanced Attention Mechanisms - 高级注意力机制

完整的注意力机制实现：

- **Flash Attention 2/3**：GPU优化的高效注意力，支持Ampere+与Hopper+架构
- **Multi-Head Latent Attention (MLA)**：DeepSeek风格的KV压缩，大幅降低KV缓存
- **Grouped Query Attention (GQA)**：分组查询注意力，平衡质量与效率
- **ALiBi Position Encoding**：线性偏置位置编码，无需位置嵌入
- **QK Normalization**：查询键归一化，提升大模型训练稳定性

### 🚀 Training Envelope & Optimization - 训练与优化

完整的训练优化套件：

- **GaLore Optimization**：低秩梯度投影优化，支持自适应秩调整、多模态模块优化
- **K-FAC Enhanced Gradient Clipping**：K-FAC增强的梯度裁剪，支持层协调
- **Multi-bit Quantization (2/4/8-bit)**：多位量化支持，极限省显存
- **LoRA/QLoRA**：低秩适配微调，支持所有线性层
- **Speculative Decoding**：推测式解码，2-3x推理加速
- **Multi-Token Prediction (MTP)**：多token预测，提升生成质量
- **Smart Gradient Accumulation**：智能梯度累积，自适应内存管理
- **Multi-task Learning**：多任务学习支持，自适应任务权重

#### 参考配置
核心组件位于 `model/` 与 `model/multimodal/`，默认超参存放在 `configs/model/*.json`。

| Model Size | Layers | Hidden | Heads | KV Heads | MoE Experts | Top-K | Context | MLA Rank |
|------------|--------|--------|-------|----------|-------------|-------|---------|----------|
| 0.5B       | 16     | 640    | 10    | 5        | 6           | 2     | 256K    | 256      |
| 1.5B       | 16     | 896    | 14    | 7        | 6           | 2     | 256K    | 256      |
| 7B         | 28     | 3584   | 32    | 8        | 8           | 2     | 1M      | 512      |
| 32B        | 64     | 5120   | 40    | 8        | 8           | 2     | 1M      | 512      |
| 64B        | 80     | 6656   | 52    | 8        | 8           | 2     | 10M     | 1024     |
| 70B        | 80     | 8192   | 64    | 8        | 8           | 2     | 10M     | 1024     |
| 128B       | 120    | 10240  | 80    | 8        | 8           | 2     | 10M     | 1536     |
| 314B       | 160    | 12288  | 96    | 12       | 16          | 4     | 10M     | 2048     |
| 671B       | 200    | 16384  | 128   | 16       | 32          | 6     | 10M     | 2048     |
| 1T         | 240    | 20480  | 160   | 20       | 64          | 8     | 10M     | 2560     |

说明：量化缺省值继承各自配置文件，可在训练命令中通过 `--force_quant --quant_bits {2,4,8}`、`--force_lora` 直接覆盖。

```bash
# 2 位量化（实验，极限省显存）
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 2 --force_lora

# 4 位量化（均衡）
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora

# 8 位量化（稳定）
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 8 --force_lora
```

---

<h2 align="center">🛠️ 安装与环境</h2>

- Python：推荐 3.11+
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

<h2 align="center">⚡ 快速开始</h2>

### 基础环境设置
```bash
# 1. 克隆仓库
git clone https://gitee.com/dunimd/piscesl1.git
# 或
git clone https://github.com/mf2023/piscesl1.git
cd piscesl1

# 2. 环境设置
python manage.py setup

# 3. 下载默认数据集
python manage.py download
```

### 核心命令
所有命令通过：
```bash
python manage.py <command>
```
查看帮助：
```bash
python manage.py help
```

| 命令       | 描述                                                             |
|------------|------------------------------------------------------------------|
| setup      | 环境设置与依赖安装                                               |
| train      | 训练模型（支持量化 / LoRA / RLHF / GaLore）                      |
| serve      | 启动 OpenAI 兼容后端推理服务                                     |
| check      | 检查 GPU 与依赖项                                                |
| monitor    | 系统监控（GPU/CPU/内存）                                         |
| download   | 下载数据集                                                       |
| benchmark  | 模型评测与基准测试                                               |
| mcp        | MCP 工具管理（status / warmup / refresh-cache） |
| watermark  | 水印检测（文本/文件/图像/音频/视频/模型权重）                    |
| action     | 后台进程管理（提交/状态/控制）                                   |
| help       | 显示帮助信息                                                     |

### 快速体验
```bash
# 训练 0.5B 模型
python manage.py train --model_size 0.5B

# 启动后端服务
python manage.py serve --model_size 7B --port 8000
```

### API 使用示例
```bash
# Chat Completion
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "pisceslx-7b", "messages": [{"role": "user", "content": "你好，请介绍一下自己"}]}'

# 流式响应
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "pisceslx-7b", "messages": [...], "stream": true}'

# Embedding 生成
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "pisceslx-7b", "input": "你好世界"}'
```

### 常用示例
```bash
# 数据集管理
python manage.py download --max_samples 50000

# 训练示例
python manage.py train --model_size 0.5B --dataset Chinese2
python manage.py train --model_size 1B --dataset Chinese2 --resume_ckpt runs/last.pt --reset_lr
python manage.py train --model_size 7B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora
python manage.py train --model_size 7B --dataset Chinese2 --rlhf --rlhf_dataset dunimd/human_feedback --rlhf_lr 1e-5

# 后端服务
python manage.py serve --model_size 7B --port 8000
python manage.py serve --model_size 14B --host 0.0.0.0 --port 8080 --workers 4
python manage.py serve --model_size 72B

# 评测示例
python manage.py benchmark --list
python manage.py benchmark --info mmlu
python manage.py benchmark --benchmark mmlu --config configs/0.5B.json --seq_len 4096 --model ckpt/model.pt
python manage.py benchmark --perf --config configs/0.5B.json --selftest

# MCP 工具
python manage.py mcp --mcp_action status
python manage.py mcp --mcp_action warmup
python manage.py mcp --mcp_action refresh-cache

# 水印检测
python manage.py watermark --text "检测文本水印"
python manage.py watermark --file document.txt
python manage.py watermark --image-file image.png
python manage.py watermark --audio-file audio.wav
python manage.py watermark --video-file video.mp4
python manage.py watermark --model-file model.pt
python manage.py watermark --weights-verify --ckpt model.pt

# 后台进程管理
python manage.py action submit train configs/train.json
python manage.py action submit train configs/train.json --gpu_count 2 --priority high
python manage.py action submit serve configs/serve.json
python manage.py action status
python manage.py action logs <run_id>
python manage.py action control <run_id> pause
python manage.py action control <run_id> resume
python manage.py action control <run_id> stop
python manage.py action list
python manage.py action list --running

# GPU 资源管理
python manage.py action gpu list
python manage.py action gpu status
python manage.py action gpu status --gpu_id 0
python manage.py action gpu release --task_id <run_id>

# 任务队列管理
python manage.py action queue list
python manage.py action queue stats
python manage.py action queue clear --priority low

# 系统资源
python manage.py action resources status
python manage.py action resources utilization

# 任务恢复
python manage.py action recover <run_id>
python manage.py action recover <run_id> --checkpoint runs/<run_id>/ckpt.pt
```

---

<h2 align="center">📦 数据集</h2>

数据集由 `configs/dataset.yaml` 配置并通过：
```bash
python manage.py download
```
- 下载的默认优先来源：ModelScope → HuggingFace（不可访问时自动镜像）。

- 完整列表请见 `configs/dataset.yaml`

---

<h2 align="center">❓ 常见问题（FAQ）</h2>

- 如何查看可用命令？`python manage.py help`
- 如何添加新数据集？编辑 `configs/dataset.yaml` 并运行 `python manage.py download`。自定义数据集建议 JSONL（text）或 Parquet（input_ids/labels）。
- 显存不足怎么办？用更小模型、降低序列长度，或启用 4 位量化（`--force_quant --quant_bits 4`，通常配合 `--force_lora`）。
- 如何恢复训练？`--resume_ckpt path/to/ckpt.pt`（可选 `--reset_lr`）
- 仅用 CPU？可使用 `--device cpu`（性能较慢）。
- 如何进行评测？`python manage.py benchmark ...`，配合 `--config`、`--seq_len`、`--model` 等参数。

---

<h2 align="center">🌏 社区与引用</h2>

- 欢迎提交 Issues 与 PR！
- Gitee: https://gitee.com/dunimd/piscesl1.git
- GitHub: https://github.com/mf2023/piscesl1.git
- ModelScope: https://www.modelscope.cn/models/mfchina2024/PiscesL1

---

<div align="center">

## 📄 许可证与开源协议

### 🏛️ 项目许可证

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache License 2.0">
  </a>
</p>

本项目采用 **Apache License 2.0** 开源协议，详见 [LICENSE](LICENSE) 文件。

### 📋 依赖包开源协议

本项目依赖的开源包及其协议信息如下：

<div align="center">

| 📦 包名 | 📜 开源协议 | 📦 包名 | 📜 开源协议 |
|:-------|:-----------|:-------|:-----------|
| torch | BSD-style | torchvision | BSD-style |
| torchaudio | BSD-style | torch-directml | MIT |
| transformers | Apache 2.0 | tokenizers | Apache 2.0 |
| huggingface-hub | Apache 2.0 | modelscope | Apache 2.0 |
| numpy | BSD 3-Clause | addict | MIT |
| accelerate | Apache 2.0 | einops | MIT |
| timm | Apache 2.0 | pytorch-lightning | Apache 2.0 |
| pillow | HPND | PyMuPDF | AGPL 3.0 |
| python-docx | MIT | python-pptx | MIT |
| bitsandbytes | MIT | peft | Apache 2.0 |
| wheel | MIT | xformers | BSD 3-Clause |
| trl | Apache 2.0 | nvidia-ml-py3 | BSD 3-Clause |
| fastapi | MIT | uvicorn | BSD 3-Clause |
| python-multipart | Apache 2.0 | pydantic | MIT |
| pandas | BSD 3-Clause | gradio | Apache 2.0 |
| ijson | BSD 3-Clause | pyarrow | Apache 2.0 |
| tqdm | MIT | jsonlines | MIT |
| streamlit | Apache 2.0 | PyYAML | MIT |
| GitPython | BSD 3-Clause | opencv-python | MIT |
| numpy | BSD 3-Clause | addict | MIT |
| openai | Apache 2.0 | requests | Apache 2.0 |
| beautifulsoup4 | MIT | psutil | BSD 3-Clause |
| pytz | MIT | pywin32 | PSF |
| duckduckgo-search | MIT | plotly | MIT |
| evalscope | Apache 2.0 |  |  |
| safetensors | Apache 2.0 | deepspeed | Apache 2.0 |
| aiofiles | Apache 2.0 |  |  |
| pathlib2 | MIT | textual | MIT |
| dmsc | Apache 2.0 |  |  |

</div>

</div>
