<div align="center">

[![Encre Agent](https://img.shields.io/badge/🚀_Encre_Agent—通用型智能体旗舰-brightgreen?style=for-the-badge&logo=git)](https://gitee.com/dunimd/encre.git)

</div>

<div align="center">

# ⚖️ 法律声明

**遵守各国AI监管法规是使用者的法定义务。**

根据相关法律法规（包括但不限于中国《生成式人工智能服务管理暂行办法》、欧盟《人工智能法案》、美国《AI风险管理框架》等），使用者需自行履行合规义务。未合规使用可能导致服务终止、行政处罚或法律追责，相关风险由使用者自行承担。

**本项目采用 Apache 2.0 许可证，允许商业使用。**

---

# PiscesL1

[English](README.md) | 简体中文

[安全](SECURITY.md) | [贡献](CONTRIBUTING.md) | [行为准则](CODE_OF_CONDUCT.md)

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
    <img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-Dunimd-1E6CFF?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbmFtZT0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik03LjAwNiAwQzMuMTQyIDAgMCAzLjE0MiAwIDcuMDA2UzMuMTQyIDE0LjAxMiA3LjAwNiAxNC4wMTJDMTAuODcgMTQuMDEyIDE0LjAxMiAxMC44NyAxNC4wMTIgNy4wMDZDMTQuMDEyIDMuMTQyIDEwLjg3IDAgNy4wMDYgMFoiIGZpbGw9IiMxRTZDRkYiLz48L3N2Zz4K"/>
</a>
<a href="https://gitee.com/dunimd/encre.git" target="_blank">
    <img alt="Encre Agent" src="https://img.shields.io/badge/Encre_Agent-Dunimd-7B68EE?style=flat-square&logo=git"/>
</a>

采用 **Yv架构** 的高性能多模态混合专家模型（MoE），支持文本、图像、音频、视频、文档与智能体理解。PiscesL1（PiscesLx 系列，Dunimd团队）面向研究与实用，可在单张 RTX 4090 上运行，体系可扩展至 1T 参数规模。

</div>

<h2 align="center">Yv架构</h2>

### 🧠 YvUnifiedReasoner - 统一推理系统

YvUnifiedReasoner 实现了智能路由的统一推理框架，在链式思维（CoT）与多路径推理引擎之间动态切换：

- **YvCoTMemoryReasoner**：记忆增强的链式思维推理器，支持自适应深度控制（1-3层）、早期停止机制、以及错误分析与自纠错
- **YvMultiPathReasoningEngine**：多路径推理引擎，支持最多8路假设流并行探索，配合动态事实验证与元认知不确定性评分
- **智能路由**：根据问题复杂度和序列长度自动选择最优推理路径
- **控制Token**：`<|start_hypothesis|>`、`<|start_evidence|>`、`<|start_conclusion|>`、`<|hypothesis_split|>`、`<|hypothesis_merge|>` 使外部工具能精确追踪模型的推理路径

### 🔧 Yv MoE Scaling - 混合专家系统

混合专家（MoE）实现：

- **YvStableMoEGate**：带LSTM负载预测的稳定门控，支持6-64专家的Top-K路由
- **细粒度专家分割**：每个"专家"由多个子专家组合而成，路由更灵活
- **共享专家隔离**：始终激活的共享专家，处理所有token
- **无辅助损失负载均衡**：无需影响模型质量的辅助损失即可实现负载均衡
- **UltraMem TDQKR 优化**：Tucker分解查询-键检索优化，路由复杂度从O(N)降至O(√N)
- **动态设备迁移**：为大型专家池高效管理内存的动态专家迁移

### 🌐 多模态感知栈

六模态统一感知架构：

- **YvVisionEncoder**：NaViT风格的patch编码，支持原生分辨率（最高2048px）和patch打包
- **YvVideoEncoder**：帧级注意力编码，3D RoPE时空位置编码
- **YvAudioEncoder**：音频频谱编码，支持流式音频处理
- **YvDocEncoder**：LayoutLMv3风格文档编码，支持布局感知的结构推理
- **YvAgenticEncoder**：智能体状态编码，包含动作空间和状态表示
- **YvCrossModalAttention**：跨模态注意力，支持模态间深度交互

### ⚛️ YvDynamicModalFusion - 动态模态融合

Token级多模态融合系统：

- **跨模态注意力**：模态间信息交换的跨模态注意力
- **模态感知位置编码**：模态感知的位置嵌入
- **质量加权门控**：根据融合质量动态调整权重的质量加权门控
- **YvEnhancedModalFusion**：增强融合模块，包含对比跨模态对齐和在线自适应权重
- **多融合策略**：支持在文本序列前插入融合token、拼接3D特征或输出压缩摘要

### 📏 超长上下文结构

行业领先的 10M+ token 上下文支持：

- **YaRN RoPE + 动态 NTK 缩放**：YaRN 位置编码配合动态 NTK 缩放，支持 10M+ token 外推
- **H2O Heavy-Hitter Oracle Attention**：保留重要 token 的超长上下文注意力
- **流式注意力**：无限长度生成的流式注意力
- **滑动窗口注意力**：局部注意力与全局 token 结合的滑动窗口注意力
- **线性注意力**：O(n) 复杂度的线性注意力，支持 ELU/Performer/Softmax 特征映射
- **分页注意力**：高效 KV 缓存管理和共享的分页注意力
- **环形注意力**：分布式超长上下文处理的环形注意力
- **注意力汇点**：保障流式推理稳定性的注意力汇点

### 🔥 混合注意力-SSM

业界前沿的混合架构实现：

- **Mamba-3 集成**：完整的 Mamba-3 SSM 集成，支持梯形离散化、复状态和 MIMO 结构
- **YvSelectiveSSM**：选择性状态空间模型，具有输入相关的状态转换
- **渐进式门控**：从纯注意力到混合模式的平滑过渡门控，保障训练稳定性
- **自适应路由**：根据序列特征动态选择注意力或 SSM 的自适应路由
- **Jamba 风格交错架构**：注意力和 SSM 层交替的 Jamba 风格架构

### 🎯 先进注意力机制

完整的注意力机制实现：

- **Flash Attention 2/3**：GPU 优化高效注意力，支持 Ampere+ 和 Hopper+ 架构
- **多头潜在注意力（MLA）**：低秩 KV 压缩，大幅减少 KV 缓存
- **分组查询注意力（GQA）**：平衡质量和效率的分组查询注意力
- **ALiBi 位置编码**：无需位置嵌入的线性偏置注意力
- **QK 归一化**：改进大模型训练稳定性的查询-键归一化

### 🚀 训练优化套件

完整的训练优化工具集：

- **GaLore 优化**：低秩梯度投影优化，支持自适应秩调整和多模态模块优化
- **K-FAC 增强梯度裁剪**：K-FAC 增强梯度裁剪，支持层间协调
- **多比特量化（2/4/8-bit）**：极致内存节省的多比特量化支持
- **LoRA/QLoRA**：支持所有线性层的低秩适配微调
- **推测解码**：2-3倍推理加速的推测解码
- **多Token预测（MTP）**：提升生成质量的多Token预测
- **智能梯度累积**：自适应内存管理的智能梯度累积
- **多任务学习**：自适应任务权重的多任务学习支持

#### 参考配置
核心组件位于 `model/` 和 `model/multimodal/`，默认超参数存储在 `configs/model/*.json` 中。

| 模型大小 | 层数 | 隐藏维度 | 注意力头 | KV头 | MoE专家 | Top-K | 上下文 | MLA秩 |
|---------|------|---------|---------|------|---------|-------|--------|-------|
| 0.5B    | 16   | 640     | 10      | 5    | 6       | 2     | 256K   | 256   |
| 1.5B    | 16   | 896     | 14      | 7    | 6       | 2     | 256K   | 256   |
| 7B      | 28   | 3584    | 32      | 8    | 8       | 2     | 1M     | 512   |
| 32B     | 64   | 5120    | 40      | 8    | 8       | 2     | 1M     | 512   |
| 64B     | 80   | 6656    | 52      | 8    | 8       | 2     | 10M    | 1024  |
| 70B     | 80   | 8192    | 64      | 8    | 8       | 2     | 10M    | 1024  |
| 128B    | 120  | 10240   | 80      | 8    | 8       | 2     | 10M    | 1536  |
| 314B    | 160  | 12288   | 96      | 12   | 16      | 4     | 10M    | 2048  |
| 671B    | 200  | 16384   | 128     | 16   | 32      | 6     | 10M    | 2048  |
| 1T      | 240  | 20480   | 160     | 20   | 64      | 8     | 10M    | 2560  |

注意：默认量化值继承自相应的配置文件，可在训练命令中通过 `--force_quant --quant_bits {2,4,8}`、`--force_lora` 直接覆盖。

```bash
# 2比特量化（实验性，极致内存节省）
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 2 --force_lora

# 4比特量化（均衡模式）
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora

# 8比特量化（稳定模式）
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 8 --force_lora
```

---

<h2 align="center">🛠️ 安装与环境</h2>

- Python: 推荐 3.11+
- CUDA: 11.8+ (用于 GPU 训练和推理)
- 依赖: 参见 `requirements.txt`

### 快速安装
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

# 2. 环境安装
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

| 命令      | 描述                                           |
|-----------|------------------------------------------------|
| setup     | 环境安装与依赖配置                             |
| enta      | **EnTA 自主训练循环**（LLM驱动，多教师）       |
| train     | 模型训练（支持量化/LoRA/RLHF/GaLore）         |
| serve     | 启动 OpenAI 兼容的后端推理服务                 |
| test      | 项目健康检查（8阶段验证）                      |
| monitor   | 系统监控（GPU/CPU/内存）                       |
| download  | 下载数据集                                     |
| benchmark | 模型评估与基准测试                             |
| mcp       | MCP工具管理（status/warmup/refresh-cache）     |
| watermark | 水印检测（文本/文件/图像/音频/视频/模型权重） |
| action    | 后台进程管理（提交/状态/控制）                 |
| dev       | 开发者模式（vim风格命令接口）                  |
| cache     | .pisceslx 目录缓存管理                         |
| publish   | 将模型打包为 Docker 镜像发布                   |
| help      | 查看帮助信息                                   |

### EnTA 训练
```bash
# 列出已配置的教师模型
python manage.py enta --list_models

# 干跑验证管线
python manage.py enta --dry_run

# 完整训练（指定教师模型和学生检查点）
python manage.py enta --teacher deepseek-r1 --model_path ./ckpt/7B.pt

# 启用圆桌讨论和潜意识训练
python manage.py enta --teacher deepseek-r1 \
  --aux_teachers "deepseek-v3.2,qwen3.6,agens-2.0-flash" \
  --model_path ./ckpt/7B.pt

# 从高级阶段开始
python manage.py enta --enta_stage advanced --max_tasks 10000

# 关闭潜意识（仅训练7B核心）
python manage.py enta --no_subconscious
```

### 快速体验
```bash
# 训练 0.5B 模型
python manage.py train --model_size 0.5B

# 启动后端服务
python manage.py serve --model_size 7B --port 8000
```

### API使用示例
```bash
# 聊天补全
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "pisceslx-7b", "messages": [{"role": "user", "content": "你好，介绍一下自己"}]}'

# 流式响应
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "pisceslx-7b", "messages": [...], "stream": true}'

# 生成 Embedding
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "pisceslx-7b", "input": "你好世界"}'
```

### 常见示例
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

# 基准测试示例
python manage.py benchmark --list
python manage.py benchmark --info mmlu
python manage.py benchmark --benchmark mmlu --config configs/0.5B.json --seq_len 4096 --model ckpt/model.pt
python manage.py benchmark --perf --config configs/0.5B.json --selftest

# MCP工具
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

# 开发者模式（vim风格训练命令界面）
python manage.py dev enable    # 启用开发者模式
python manage.py dev disable   # 禁用开发者模式
python manage.py dev status    # 查看开发者模式状态

# .pisceslx 目录缓存管理
python manage.py cache         # 查看缓存状态
python manage.py cache clean   # 清理所有缓存（保留 settings/）

# 将模型打包为 Docker 镜像并发布
python manage.py publish --publish_action full --publish_model_size 7B --publish_registry docker.io
python manage.py publish --publish_action full --publish_model_size 7B --publish_model_path ./ckpt/7B.pt
python manage.py publish --publish_action export --publish_model_size 7B --publish_output_dir ./export/
python manage.py publish --publish_action build --publish_model_size 7B --publish_template gpu
python manage.py publish --publish_action push --publish_registry ghcr.io --publish_registry_namespace myuser
python manage.py publish --publish_action validate --publish_model_size 7B
python manage.py publish --publish_action info --publish_model_size 7B
python manage.py publish --publish_action list
```

---

<h2 align="center">📦 数据集</h2>

数据集通过 `configs/dataset.yaml` 配置，通过以下命令下载：
```bash
python manage.py download
```
- 默认下载优先顺序：ModelScope → HuggingFace（无法访问时自动切换镜像）
- 完整列表参见 `configs/dataset.yaml`

---

<h2 align="center">❓ 常见问题 (FAQ)</h2>

- 如何查看可用命令？`python manage.py help`
- 如何添加新数据集？编辑 `configs/dataset.yaml` 并运行 `python manage.py download`。自定义数据集推荐 JSONL（文本）或 Parquet（input_ids/labels）格式。
- GPU 显存不足？使用更小模型、缩短序列长度，或启用 4 比特量化（`--force_quant --quant_bits 4`，通常配合 `--force_lora`）。
- 如何恢复训练？`--resume_ckpt path/to/ckpt.pt`（可选 `--reset_lr`）
- 只有 CPU？可使用 `--device cpu`（性能较慢）。
- 如何评估模型？`python manage.py benchmark ...`，配合 `--config`、`--seq_len`、`--model` 等参数。
- EnTA 如何工作？EnTA 是一个 LLM 驱动的自主训练智能体。详见 [EnTA 架构](#-encre-train-agent-enta)。

---

<h2 align="center">🌏 社区与引用</h2>

- 欢迎提交 Issues 和 PRs！
- Gitee: https://gitee.com/dunimd/piscesl1.git
- GitHub: https://github.com/mf2023/piscesl1.git
- ModelScope: https://www.modelscope.cn/models/mfchina2024/PiscesL1

---

<h2 align="center">📚 学术引用</h2>

本项目实现了以下学术论文中的算法。我们衷心感谢作者们的贡献。

### 注意力机制

| 算法 | 论文 | 作者 | 会议 | 年份 | 代码 |
|-----|------|------|------|------|------|
| ALiBi | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation | Press et al. | ICLR | 2022 | [attention.py](model/core/attention.py#L346-L348) |
| Attention Sink | Efficient Streaming Language Models with Attention Sinks | Xiao et al. | ICLR | 2024 | [attention.py](model/core/attention.py#L533-L535) |
| QK Normalization | Query-Key Normalization for Transformers | Henry et al. | ICLR | 2020 | [attention.py](model/core/attention.py#L656-L657) |
| Linear Attention | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention | Katharopoulos et al. | ICML | 2020 | [attention.py](model/core/attention.py#L787-L789) |
| S4 | Efficiently Modeling Long Sequences with Structured State Spaces | Gu et al. | ICLR | 2022 | [attention.py](model/core/attention.py#L1073-L1075) |
| Longformer | Longformer: The Long-Document Transformer | Beltagy et al. | - | 2020 | [attention.py](model/core/attention.py#L1225-L1226) |
| BigBird | Big Bird: Transformers for Longer Sequences | Zaheer et al. | NeurIPS | 2020 | [attention.py](model/core/attention.py#L1437-L1438) |
| Ring Attention | Ring Attention with Blockwise Transformers for Near-Infinite Context | Liu et al. | ICLR | 2024 | [attention.py](model/core/attention.py#L2479-L2481) |
| MQA | Fast Transformer Decoding: One Write-Head is All You Need | Shazeer | - | 2019 | [attention.py](model/core/attention.py#L2831-L2832) |
| H2O | H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models | Zhang et al. | ICLR | 2024 | [attention.py](model/core/attention.py#L3026-L3028) |
| LongRoPE | LongRoPE: Extending LLM Context Window Beyond 2M Tokens | Ding et al. | ICML | 2024 | [attention.py](model/core/attention.py#L4152-L4154) |
| PagedAttention | Efficient Memory Management for Large Language Model Serving with PagedAttention | Kwon et al. | SOSP | 2023 | [attention.py](model/core/attention.py#L1654-L1656) |
| Flash Attention | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao et al. | NeurIPS | 2022 | [attention.py](model/core/attention.py#L1910-L1914) |
| Flash Attention 2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | Dao | - | 2023 | [attention.py](model/core/attention.py#L1913-L1914) |
| Flash Attention 3 | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Blockwise Parallelism | Dao et al. | - | 2024 | [flash_attention.py](opss/infer/flash_attention.py#L37-L38) |
| CoPE | Context-aware Position Encoding for Better Length Extrapolation | Yang et al. | arXiv | 2024 | [attention.py](model/core/attention.py#L4197-L4199) |

### 位置编码

| 算法 | 论文 | 作者 | 会议 | 年份 | 代码 |
|-----|------|------|------|------|------|
| Sinusoidal PE | Attention Is All You Need | Vaswani et al. | NeurIPS | 2017 | [embedding.py](model/core/embedding.py#L283-L284) |
| RoPE | RoFormer: Enhanced Transformer with Rotary Position Embedding | Su et al. | - | 2021 | [norms.py](model/core/norms.py#L548-L687) |
| YaRN | YaRN: Efficient Context Window Extension of Large Language Models | Peng et al. | - | 2023 | [norms.py](model/core/norms.py#L689-L841) |

### 归一化与激活

| 算法 | 论文 | 作者 | 会议 | 年份 | 代码 |
|-----|------|------|------|------|------|
| RMSNorm | Root Mean Square Layer Normalization | Zhang & Sennrich | NeurIPS | 2019 | [norms.py](model/core/norms.py#L192-L193) |
| Adaptive LayerNorm | Scalable Diffusion Models with Transformers (DiT) | Peebles & Xie | ICCV | 2023 | [norms.py](model/core/norms.py#L387-L388) |
| LayerScale | Going deeper with Image Transformers | Touvron et al. | ICCV | 2021 | [blocks.py](model/core/blocks.py#L349-L350) |
| SwiGLU | GLU Variants Improve Transformer | Shazeer | - | 2020 | [blocks.py](model/core/blocks.py#L402-L440) |
| GeGLU | GLU Variants Improve Transformer | Shazeer | - | 2020 | [blocks.py](model/core/blocks.py#L453-L490) |
| Group Normalization | Group Normalization | Wu & He | ECCV | 2018 | [norms.py](model/core/norms.py#L497-L498) |

### 状态空间模型

| 算法 | 论文 | 作者 | 会议 | 年份 | 代码 |
|-----|------|------|------|------|------|
| Mamba | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu & Dao | arXiv | 2023 | [blocks.py](model/core/blocks.py#L1434-L1437) |
| Mamba-2 | Mamba-2: Transforming Transformers | Dao et al. | arXiv | 2024 | [blocks.py](model/core/blocks.py#L1437) |

### 混合专家

| 算法 | 论文 | 作者 | 会议 | 年份 | 代码 |
|-----|------|------|------|------|------|
| UltraMem TDQKR | UltraMem | ByteDance | ICLR | 2025 | [layer.py](model/moe/layer.py#L216) |
| DeepSeekMoE | DeepSeek-V3 Technical Report | DeepSeek Team | - | 2024 | [expert.py](model/moe/expert.py#L50), [layer.py](model/moe/layer.py#L390) |

### 推理优化

| 算法 | 论文 | 作者 | 会议 | 年份 | 代码 |
|-----|------|------|------|------|------|
| Speculative Decoding | Fast Inference from Transformers via Speculative Decoding | Leviathan et al. | ICML | 2023 | [cache.py](model/core/cache.py#L1013-L1015) |
| BLIP-2 | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders | Li et al. | ICML | 2023 | [cache.py](model/core/cache.py#L1217-L1219) |

### 训练优化

| 算法 | 论文 | 作者 | 会议 | 年份 | 代码 |
|-----|------|------|------|------|------|
| K-FAC | Optimizing Neural Networks with Kronecker-factored Approximate Curvature | Martens & Grosse | ICML | 2015 | [kfac.py](opss/train/kfac.py#L65-L66) |
| GaLore | GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | Zhao et al. | arXiv | 2024 | [galore.py](opss/optim/galore.py#L35-L37) |

### 对齐与强化学习

| 算法 | 论文 | 作者 | 会议 | 年份 | 代码 |
|-----|------|------|------|------|------|
| DPO | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafailov et al. | NeurIPS | 2023 | [dpo.py](opss/train/dpo.py#L22-L29) |
| GRPO | DeepSeek R1 Technical Report | DeepSeek Team | arXiv | 2024 | [grpo.py](opss/train/grpo.py#L33-L34) |
| RLVR | DeepSeek R1 Technical Report / OpenAI o1 | DeepSeek / OpenAI | arXiv | 2024/2025 | [rlvr.py](opss/train/rlvr.py#L35-L36) |

### 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{piscesl1,
  author = {Wenze Wei, Dunimd Team},
  title = {PiscesL1: A High-Performance Multimodal Mixture-of-Experts Model with Autonomous Training Agent},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/mf2023/piscesl1}
}
```

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

本项目使用的开源包及其协议信息：

<div align="center">

| 📦 包名 | 📜 协议 | 📦 包名 | 📜 协议 |
|:--------|:--------|:--------|:--------|
| torch | BSD-style | torchvision | BSD-style |
| torchaudio | BSD-style | torch-directml | MIT |
| transformers | Apache 2.0 | tokenizers | Apache 2.0 |
| huggingface-hub | Apache 2.0 | modelscope | Apache 2.0 |
| numpy | BSD 3-Clause | scipy | BSD 3-Clause |
| scikit-learn | BSD 3-Clause | addict | MIT |
| accelerate | Apache 2.0 | einops | MIT |
| timm | Apache 2.0 | pytorch-lightning | Apache 2.0 |
| pillow | HPND | PyMuPDF | AGPL 3.0 |
| bitsandbytes | MIT | peft | Apache 2.0 |
| flash-attn | BSD 3-Clause | triton | MIT |
| deepspeed | Apache 2.0 | datasets | Apache 2.0 |
| wandb | MIT | tensorboard | Apache 2.0 |
| docker | Apache 2.0 | | |

</div>

</div>

---

<div align="center">

**✅ 连接已建立。**

</div>
