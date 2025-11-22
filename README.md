<div align="center">

# ⚠️ Compliance Notice

**In accordance with relevant laws and regulations of various countries (including but not limited to China's "Interim Measures for Generative AI Service Management", EU's "Artificial Intelligence Act", US's "AI Risk Management Framework", and Japan's "AI Guidelines"), developers or users bear their own compliance responsibilities. Failure to fulfill related obligations may result in service suspension, regulatory penalties, or legal liabilities.**

---


# PiscesL1

English | [简体中文](README.zh.md)

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

A lightweight multimodal Mixture-of-Experts (MoE) model featuring the Arctic architecture, supporting text, image, audio, video, document, and agent understanding. PiscesL1 (PiscesLx series, Dunimd Team) is designed for research and practical applications, capable of running on a single RTX 4090 GPU with scalable architecture up to 1T parameters.

</div>

<h2 align="center">❄️ Arctic Architecture</h2>

### 🧠 ArcticUnifiedReasoner
ArcticUnifiedReasoner unified scheduling Hierarchical Reasoning Chains (HRC) can simultaneously run up to 8 hypothesis streams, combined with dynamic fact verification and metacognitive uncertainty scoring to provide self-checking feedback for the reasoning process. Control tokens like `<|start_hypothesis|>`, `<|start_evidence|>`, `<|start_conclusion|>`, `<|hypothesis_split|>`, `<|hypothesis_merge|>` help external tools precisely track the model's thinking path.

### 🔧 Arctic MoE Scaling
ArcticStableMoEGate with its LSTM load predictor handles 8-expert Top-2 routing, integrating load noise, fixed-shape execution, and capacity-aware gating. This ensures stable training for large models while maintaining consistent interfaces for small models, facilitating switching between different computational resources.

### 🌐 Multimodal Perception Stack
Visual, video, audio, document, and agent inputs are unified by ArcticVisionEncoder (NaViT-style patches up to 1024px), ArcticVideoEncoder (frame-level attention with 3D RoPE), ArcticAudioEncoder, ArcticDocEncoder (LayoutLMv3-style structural reasoning), and ArcticAgenticEncoder. This allows the backbone network to directly consume cross-modal features without additional tokenization logic.

### ⚛️ ArcticDynamicModalFusion
ArcticDynamicModalFusion implements token-level fusion through cross-modal attention, modality-aware position embeddings, and quality-weighted gating. It can choose to insert fusion tokens before text sequences, concatenate 3D features, or output compressed summaries based on scenarios. Training, inference, and MCP tool alignment all share this logic.

### 📏 Ultra-Long Context Fabric
With YaRN RoPE + dynamic NTK scaling, H2O streaming attention, sliding window, and compression strategies, the Arctic architecture can scale single sequences to 10M+ tokens. It also supports speculative decoding and high-speed KV cache segmentation, suitable for long documents and multi-round Agent workloads.

### 🤖 ArcticAgentic Runtime
ArcticAgentic is an MCP-native agent runtime that can embed environment observation, maintain persistent memory, schedule tool calls, and coordinate multi-agent conversations. Since it shares encoders and fusion layers with the backbone, agent trajectories can be directly trained or reviewed through the `python manage.py` workflow.

### 🎯 Training Envelope & Optimization
All model sizes share the same training envelope: K-FAC enhanced gradient clipping, multi-bit quantization (2/4/8), LoRA/QLoRA, and checkpoint pipeline. This makes 0.5B–1.5B scale feasible on ~14.6 GB GPU while retaining an upgrade path to 1T parameters. One command (`python manage.py train|infer|benchmark`) covers the entire lifecycle.

#### Reference Configuration
Core components are located in `model/` and `model/multimodal/`, with default hyperparameters stored in `configs/model/*.json`.

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

Note: Default quantization values inherit from their respective config files and can be directly overridden in training commands via `--force_quant --quant_bits {2,4,8}`, `--force_lora`.

```bash
# 2-bit quantization (experimental, extreme memory saving)
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 2 --force_lora

# 4-bit quantization (balanced)
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora

# 8-bit quantization (stable)
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 8 --force_lora
```

---

<h2 align="center">🛠️ Installation & Environment</h2>

- Python: Recommended 3.11
- CUDA: 11.8+ (for GPU training and inference)
- Dependencies: See `requirements.txt`

### Quick Setup
```bash
git clone https://gitee.com/dunimd/piscesl1.git
# or
git clone https://github.com/mf2023/piscesl1.git
cd piscesl1
python manage.py setup
```

---

<h2 align="center">⚡ Quick Start</h2>

### Basic Environment Setup
```bash
# 1. Clone repository
git clone https://gitee.com/dunimd/piscesl1.git
# or
git clone https://github.com/mf2023/piscesl1.git
cd piscesl1

# 2. Environment setup
python manage.py setup

# 3. Update (optional)
python manage.py update

# 4. Download default dataset
python manage.py download
```

### Core Commands
All commands through:
```bash
python manage.py <command>
```
View help:
```bash
python manage.py help
```

| Command   | Description                                                         |
|-----------|---------------------------------------------------------------------|
| setup     | Environment setup and dependency installation                      |
| update    | Pull latest code from remote repository                            |
| version   | Show current version and update summary                             |
| changelog | Show version history (support --all / --version X.X.XXXX)         |
| train     | Train model (support quantization / LoRA / RLHF)                   |
| infer     | Model inference (support MCP integration & speculative decoding)   |
| check     | Check GPU and dependencies                                         |
| monitor   | System monitoring (GPU/CPU/memory)                                 |
| download  | Download dataset                                                  |
| dataset   | Dataset management and conversion                                  |
| cache     | Cache maintenance (stats / clear-dataset / clear-downloads / clear-all) |
| benchmark | Model evaluation and benchmarking                                  |
| mcp       | MCP tool management (status / warmup / refresh-cache)              |
| watermark | Watermark detection (text/file, support batch & JSON output)       |
| help      | Show help information                                              |

### Quick Experience
```bash
# Train 0.5B model
python manage.py train --model_size 0.5B

# Inference test
python manage.py infer --ckpt ckpt/latest.pt --prompt "Explain machine learning in simple terms"
```

### Common Examples
```bash
# Basic operations
python manage.py version
python manage.py changelog --all
python manage.py changelog --version 1.0.0150

# Dataset management
python manage.py download --max_samples 50000

# Training examples
python manage.py train --model_size 0.5B --dataset Chinese2
python manage.py train --model_size 1.5B --dataset Chinese2 --resume_ckpt runs/last.pt --reset_lr
python manage.py train --model_size 7B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora
python manage.py train --model_size 7B --dataset Chinese2 --rlhf --rlhf_dataset dunimd/human_feedback --rlhf_lr 1e-5

# Inference examples
python manage.py infer --ckpt ckpt/latest.pt --prompt "Hello, PiscesL1!"
python manage.py infer --ckpt ckpt/model.pt --prompt "Hi" --speculative --draft_model ckpt/draft.pt --spec_gamma 4

# Benchmark examples
python manage.py benchmark --list
python manage.py benchmark --info mmlu
python manage.py benchmark --benchmark mmlu --config configs/0.5B.json --seq_len 4096 --model ckpt/model.pt
python manage.py benchmark --perf --config configs/0.5B.json --selftest

# MCP tools
python manage.py mcp --mcp_action status
python manage.py mcp --mcp_action warmup
python manage.py mcp --mcp_action refresh-cache

# Cache management
python manage.py cache --cache_action stats
python manage.py cache --cache_action clear-dataset
python manage.py cache --cache_action clear-downloads
python manage.py cache --cache_action clear-all
```

---

<h2 align="center">📦 Dataset</h2>

Dataset is configured by `configs/dataset.json` and downloaded through:
```bash
python manage.py download
```
- Default download priority: ModelScope → HuggingFace (automatic mirroring when inaccessible)

- Complete list see `configs/dataset.json`

---

<h2 align="center">❓ Frequently Asked Questions (FAQ)</h2>

- How to view available commands? `python manage.py help`
- How to add new dataset? Edit `configs/dataset.json` and run `python manage.py download`. Custom dataset recommend JSONL (text) or Parquet (input_ids/labels).
- Insufficient GPU memory? Use smaller model, reduce sequence length, or enable 4-bit quantization (`--force_quant --quant_bits 4`, usually with `--force_lora`).
- How to resume training? `--resume_ckpt path/to/ckpt.pt` (optional `--reset_lr`)
- CPU only? Can use `--device cpu` (slower performance).
- How to perform evaluation? `python manage.py benchmark ...`, with `--config`, `--seq_len`, `--model` and other parameters.

---

<h2 align="center">🌏 Community & Citation</h2>

- Welcome to submit Issues and PRs!
- Gitee: https://gitee.com/dunimd/piscesl1.git
- GitHub: https://github.com/mf2023/piscesl1.git
- ModelScope: https://www.modelscope.cn/models/mfchina2024/PiscesL1

---

<div align="center">

## 📄 License & Open Source Agreements

### 🏛️ Project License

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache License 2.0">
  </a>
</p>

This project uses **Apache License 2.0** open source agreement, see [LICENSE](LICENSE) file.

### 📋 Dependency Package Open Source Agreements

Open source packages and their agreement information used by this project:

<div align="center">

| 📦 Package | 📜 License | 📦 Package | 📜 License |
|:-----------|:-----------|:-----------|:-----------|
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