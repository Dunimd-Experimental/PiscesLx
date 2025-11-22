<div align="center">

# ⚠️ 合规提示

**根据各国相关法律法规（包括但不限于中国《生成式人工智能服务管理暂行办法》、欧盟《人工智能法案》、美国《AI风险管理框架》、日本《AI指导原则》等），开发者或使用者需自行承担合规责任，未履行相关义务可能导致服务被叫停、面临监管处罚或承担法律责任。**

---


# PiscesL1

[English](README.md) | 简体中文 

<a href="https://space.bilibili.com/3493284091529457" target="_blank">
    <img alt="BiliBili" src="https://img.shields.io/badge/BiliBili-Dunimd-00A1D6?style=flat-square&logo=bilibili"/>
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

采用 Arctic 架构的轻量级多模态混合专家模型（MoE），支持文本、图像、音频、视频、文档与智能体理解。PiscesL1（PiscesLx 系列，Dunimd团队）面向研究与实用，可在单张 RTX 4090 上运行，体系可扩展至 1T 参数规模。

</div>

<h2 align="center">❄️ Arctic 架构</h2>

### 🧠 ArcticUnifiedReasoner
ArcticUnifiedReasoner 统一调度分层推理链（HRC），可以同时运行最多 8 路假设流，配合动态事实验证与元认知不确定性评分，为推理过程提供自检反馈。`<|start_hypothesis|>`, `<|start_evidence|>`, `<|start_conclusion|>`, `<|hypothesis_split|>`, `<|hypothesis_merge|>` 等控制符帮助外部工具精确追踪模型思维路径。

### 🔧 Arctic MoE Scaling
ArcticStableMoEGate 及其 LSTM 负载预测器负责 8 专家 Top-2 路由，集成装载噪声、固定形状执行与容量感知门控，既保证大模型训练稳定，也让小模型保持一致接口，方便在不同算力之间切换。

### 🌐 Multimodal Perception Stack
视觉、视频、音频、文档与智能体输入由 ArcticVisionEncoder（最高 1024px 的 NaViT 风格 patch）、ArcticVideoEncoder（带 3D RoPE 的帧级注意力）、ArcticAudioEncoder、ArcticDocEncoder（LayoutLMv3 风格结构推理）和 ArcticAgenticEncoder 统一规整，主干网络因此能够直接消费跨模态特征而无需额外的 token 化逻辑。

### ⚛️ ArcticDynamicModalFusion
ArcticDynamicModalFusion 通过跨模态注意力、模态感知位置嵌入与质量加权门控实现 token 级融合，可根据场景选择在文本序列前插入融合 token、拼接 3D 特征或输出压缩摘要，训练、推理、MCP 工具对齐均共用这一逻辑。

### 📏 Ultra-Long Context Fabric
借助 YaRN RoPE + 动态 NTK 缩放、H2O 流式注意力、滑窗与压缩策略，Arctic 架构单序列可扩展至 10M+ token，并兼容推测式解码及高速 KV 缓存分段，适合长文档与多轮 Agent 工作负载。

### 🤖 ArcticAgentic Runtime
ArcticAgentic 是原生 MCP 感知的智能体运行时：可嵌入环境观测、维护持久记忆、调度工具调用并协调多智能体对话。由于与主干共享编码器与融合层，智能体轨迹可以直接通过 `python manage.py` 工作流训练或复盘。

### 🎯 Training Envelope & Optimization
所有模型尺寸共用同一训练外壳：K-FAC 增强的梯度裁剪、多位量化（2/4/8）、LoRA/QLoRA 以及检查点流水线，让 0.5B–1.5B 规模在 ~14.6 GB GPU 上可行，同时保留一条直达 1T 参数的升级路径。一条命令（`python manage.py train|infer|benchmark`）即可覆盖全生命周期。

#### 参考配置
核心组件位于 `model/` 与 `model/multimodal/`，默认超参存放在 `configs/model/*.json`。

| Model Size | Layers | Hidden | Heads | KV Heads | MoE Experts | Context | Quantization (default) |
|------------|--------|--------|-------|----------|-------------|---------|------------------------|
| 0.5B       | 16     | 640    | 10    | 5        | 6           | 256K    | Off (optional)         |
| 1.5B       | 16     | 896    | 14    | 7        | 6           | 256K    | Off (optional)         |
| 7B         | 28     | 3584   | 32    | 8        | 8           | 1M      | On (LoRA default)      |
| 32B        | 64     | 5120   | 40    | 8        | 8           | 1M      | On (LoRA default)      |
| 64B        | 80     | 6656   | 52    | 8        | 8           | 10M     | On (LoRA default)      |
| 70B        | 80     | 8192   | 64    | 8        | 8           | 10M     | On (LoRA default)      |
| 128B       | 120    | 10240  | 80    | 8        | 8           | 10M     | On (LoRA default)      |
| 314B       | 160    | 12288  | 96    | 12       | 16          | 10M     | On (LoRA default)      |
| 671B       | 200    | 16384  | 128   | 16       | 32          | 10M     | On (LoRA default)      |
| 1T         | 240    | 20480  | 160   | 20       | 64          | 10M     | On (LoRA default)      |

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

# 3. 更新（可选）
python manage.py update

# 4. 下载默认数据集
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

### 快速体验
```bash
# 训练 0.5B 模型
python manage.py train --model_size 0.5B

# 推理测试
python manage.py infer --ckpt ckpt/latest.pt --prompt "用简单的话解释机器学习"
```

### 常用示例
```bash
# 基础操作
python manage.py version
python manage.py changelog --all
python manage.py changelog --version 1.0.0150

# 数据集管理
python manage.py download --max_samples 50000

# 训练示例
python manage.py train --model_size 0.5B --dataset Chinese2
python manage.py train --model_size 1.5B --dataset Chinese2 --resume_ckpt runs/last.pt --reset_lr
python manage.py train --model_size 7B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora
python manage.py train --model_size 7B --dataset Chinese2 --rlhf --rlhf_dataset dunimd/human_feedback --rlhf_lr 1e-5

# 推理示例
python manage.py infer --ckpt ckpt/latest.pt --prompt "你好，PiscesL1!"
python manage.py infer --ckpt ckpt/model.pt --prompt "Hi" --speculative --draft_model ckpt/draft.pt --spec_gamma 4

# 评测示例
python manage.py benchmark --list
python manage.py benchmark --info mmlu
python manage.py benchmark --benchmark mmlu --config configs/0.5B.json --seq_len 4096 --model ckpt/model.pt
python manage.py benchmark --perf --config configs/0.5B.json --selftest

# MCP 工具
python manage.py mcp --mcp_action status
python manage.py mcp --mcp_action warmup
python manage.py mcp --mcp_action refresh-cache

# 缓存管理
python manage.py cache --cache_action stats
python manage.py cache --cache_action clear-dataset
python manage.py cache --cache_action clear-downloads
python manage.py cache --cache_action clear-all
```

---

<h2 align="center">📦 数据集</h2>

数据集由 `configs/dataset.json` 配置并通过：
```bash
python manage.py download
```
- 下载的默认优先来源：ModelScope → HuggingFace（不可访问时自动镜像）。

- 完整列表请见 `configs/dataset.json`

---

<h2 align="center">❓ 常见问题（FAQ）</h2>

- 如何查看可用命令？`python manage.py help`
- 如何添加新数据集？编辑 `configs/dataset.json` 并运行 `python manage.py download`。自定义数据集建议 JSONL（text）或 Parquet（input_ids/labels）。
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
| transformers | Apache 2.0 | tokenizers | Apache 2.0 |
| datasets | Apache 2.0 | huggingface-hub | Apache 2.0 |
| modelscope | Apache 2.0 | opencv-python | MIT |
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
| GitPython | BSD 3-Clause | mcp[cli] | MIT |
| openai | Apache 2.0 | requests | Apache 2.0 |
| beautifulsoup4 | MIT | psutil | BSD 3-Clause |
| pytz | MIT | pywin32 | PSF |
| duckduckgo-search | MIT | plotly | MIT |
| safetensors | Apache 2.0 | torch-directml | MIT |
| torch-audio | BSD-style | deepspeed | Apache 2.0 |
| mpi4py | BSD 3-Clause | evalscope | Apache 2.0 |
| fastmcp | MIT | aiofiles | Apache 2.0 |
| pathlib2 | MIT |  |  |

</div>