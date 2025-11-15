# ⚠️ Compliance Notice

**Developers and users are solely responsible for compliance with applicable laws and regulations in their respective jurisdictions, including but not limited to China's "Interim Measures for the Management of Generative Artificial Intelligence Services", the EU's "Artificial Intelligence Act", the US "AI Risk Management Framework", and Japan's "AI Guidelines"; failure to comply may result in service suspension, regulatory penalties, or legal liability.**

---

<div align="center">

# PiscesL1

English | [简体中文](README.zh.md)

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

A next-generation lightweight multimodal Mixture-of-Experts (MoE) model with Arctic Architecture, supporting text, image, audio, video, document, and agent understanding. Designed for both research and practical applications, PiscesL1 (PiscesLx Series by Dunimd Project Group) can run on a single RTX 4090 and scale up to 1T parameters with modular innovations.

## ❄️ Arctic Architecture Innovations

### 🧠 Multi-Path Reasoning Engine (ArcticUnifiedReasoner)
- Hierarchical Reasoning Chains (HRC): multi-layer abstraction
- Parallel hypothesis thinking: up to 8 concurrent streams with dynamic selection
- Dynamic fact verification: real-time truth checking and consistency scoring
- Meta-cognitive reflection: uncertainty quantification over reasoning
- Reasoning special tokens: `<|start_hypothesis|>`, `<|start_evidence|>`, `<|start_conclusion|>`, `<|hypothesis_split|>`, `<|hypothesis_merge|>`

### 🔧 MoE Expert System
- 8 experts Top-2 routing with StableMoEGate
- LSTM load prediction: dynamic capacity and allocation
- Gradient-checkpoint friendly fixed-shape mode
- Stable gates: noise injection and capacity control

### 🌐 Multimodal Encoding System
- ArcticVisionEncoder: NaViT native resolution (up to 1024px)
- ArcticVideoEncoder: temporal visual understanding, frame-level attention  
- ArcticAudioEncoder: advanced audio feature extraction
- ArcticDocEncoder: document structure understanding (LayoutLMv3)
- ArcticAgenticEncoder: agent behavior modeling (compatibility wrapper)

### ⚛️ Advanced Multimodal Fusion
- DynamicModalFusion: unified token-level multimodal fusion
- Native cross-modal attention (no tensor network requirement)
- Hardware-adaptive configuration
- Quality-aware fusion gates

### 📏 Ultra-Long Context
- YaRN RoPE: 10M+ token support with dynamic NTK scaling
- H2O attention: streaming attention for large models
- Dynamic position encoding with long-factor scaling
- Memory-efficient sliding window and compression

### 🤖 Advanced Agent System
- PiscesAgent: native multimodal agent with MCP protocol support
- Tool integration and environment interaction
- Persistent memory and experience accumulation
- Multi-agent communication for distributed reasoning

### 🎯 Advanced Optimization (K-FAC Enhanced)
- Second-order optimization via K-FAC diagonal Fisher approximation
- Adaptive gradient clipping
- Natural gradient descent
- Large memory savings vs full second-order methods

---

## 🚀 Features

- Arctic architecture with multi-path reasoning
- Unified multimodal understanding and cross-modal fusion
- 8-expert Top-2 routing and load prediction
- 10M+ context window
- Multi-bit quantization (2/4/8)
- 0.5B–1.5B training feasible on ~14.6GB GPUs (QLoRA + checkpointing)
- Native MCP agent integration
- One-command workflow via `python manage.py`

---

## 🛠️ Installation & Environment

- Python: 3.11 recommended
- CUDA: 11.8+ (for GPU training/inference)
- Dependencies: see `requirements.txt`

### Quick Setup
```bash
git clone https://gitee.com/dunimd/piscesl1.git
# or
git clone https://github.com/mf2023/piscesl1.git
cd piscesl1
python manage.py setup
```

---

## ⚡ Command Line Usage

All commands are managed via:
```bash
python manage.py <command>
```
For help:
```bash
python manage.py help
```

### Main Commands
| Command   | Description                                                            |
|-----------|------------------------------------------------------------------------|
| setup     | Environment setup and dependency installation                          |
| update    | Pull latest code from remote repository                                |
| version   | Show current version and changelog summary                             |
| changelog | Show version history (--all or --version X.X.XXXX)                     |
| train     | Train the model (supports quant/LoRA/RLHF via flags)                   |
| infer     | Run inference (supports MCP integration and speculative decoding)      |
| check     | Check GPU and dependencies                                             |
| monitor   | System monitor (GPU/CPU/memory)                                        |
| download  | Download datasets                                                      |
| dataset   | Dataset management and conversion                                      |
| cache     | Cache maintenance (stats / clear-dataset / clear-downloads / clear-all)|
| benchmark | Model evaluation and benchmarking                                      |
| mcp       | MCP tool management (status / warmup / refresh-cache)                  |
| watermark | Watermark detection (text/file, batch, verbose, JSON)                  |
| help      | Show help message                                                      |

#### Examples
```bash
# Basics
python manage.py version
python manage.py changelog --all
python manage.py changelog --version 1.0.0150

# Dataset
python manage.py download --max_samples 50000

# Training
python manage.py train --model_size 0.5B --dataset Chinese2
python manage.py train --model_size 1.5B --dataset Chinese2 --resume_ckpt runs/last.pt --reset_lr
python manage.py train --model_size 7B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora
python manage.py train --model_size 7B --dataset Chinese2 --rlhf --rlhf_dataset dunimd/human_feedback --rlhf_lr 1e-5

# Inference
python manage.py infer --ckpt ckpt/latest.pt --prompt "Hello, Pisces!"
python manage.py infer --ckpt ckpt/model.pt --prompt "Hi" --speculative --draft_model ckpt/draft.pt --spec_gamma 4

# Benchmark
python manage.py benchmark --list
python manage.py benchmark --info mmlu
python manage.py benchmark --benchmark mmlu --config configs/0.5B.json --seq_len 4096 --model ckpt/model.pt
python manage.py benchmark --perf --config configs/0.5B.json --selftest

# MCP
python manage.py mcp --mcp_action status
python manage.py mcp --mcp_action warmup
python manage.py mcp --mcp_action refresh-cache

# Cache
python manage.py cache --cache_action stats
python manage.py cache --cache_action clear-dataset
python manage.py cache --cache_action clear-downloads
python manage.py cache --cache_action clear-all
```

---

## 🧠 Model Architecture & Configurations

Core components are implemented under `model/` and `model/multimodal/`. Default model configs are in `configs/model/*.json`.

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

Notes:
- Quantization defaults follow `configs/model/*.json`. Use training flags to override: `--force_quant --quant_bits {2,4,8}`, `--force_lora`.

### Quantization Examples
```bash
# 2-bit (experimental, max memory saving)
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 2 --force_lora

# 4-bit (balanced)
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora

# 8-bit (stable)
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 8 --force_lora
```

---

## 📦 Datasets

Datasets are configured in `configs/dataset.json` and downloaded via:
```bash
python manage.py download
```
Default source preference: ModelScope -> HuggingFace (with auto mirror if needed).

Examples (see `configs/dataset.json` for full list):
- Chinese: Chinese1, Chinese2, Chinese3, Chinese4, Chinese5, Chinese6
- English: English1, English2, English3, English4
- Math: Math1, Math2, Math4, Math5
- Code: Code1, Code2, Code4
- Web: Web1, Web2, Web3
- Audio: Audio1, Audio2, Audio3
- Image: Image1, Image2
- Document/Visual: VQAv2, FinQA, DocVQ1A, Exam, SG1, Chat1, Publaynet1, Medical1, Financial1

---

## 🏆 Training on ~14.6GB GPU

PiscesL1 supports training 0.5B–1.5B models on ~14.6GB GPUs using quantization, LoRA, and memory optimizations.

- Multi-bit quantization: 2/4/8-bit
- LoRA: ~0.024% trainable params for 1.5B
- Gradient checkpointing
- K-FAC enhanced second-order methods
- Adaptive gradient clipping

Examples:
```bash
# 4-bit + LoRA
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora

# 8-bit + LoRA
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 8 --force_lora
```

---

## ⚡ Quick Start
```bash
# 1. Environment setup
python manage.py setup

# 2. Pull latest code (optional)
python manage.py update

# 3. Download default datasets
python manage.py download

# 4. Train a small model (0.5B)
python manage.py train --model_size 0.5B

# 5. Run inference
python manage.py infer --ckpt ckpt/latest.pt --prompt "Explain machine learning in simple terms"
```

## 🤖 MCP Native Agent Support [Beta]
MCP support is implemented under `model/mcp/` and tools/mcp.
```bash
python manage.py mcp --mcp_action status
python manage.py mcp --mcp_action warmup
python manage.py mcp --mcp_action refresh-cache
```

---

## ❓ FAQ
- How do I see all commands? `python manage.py help`
- How to add a dataset? Edit `configs/dataset.json` and run `python manage.py download`. For custom datasets, prefer JSONL with `text` or Parquet with `input_ids`/`labels`.
- OOM? Use a smaller model, reduce sequence length, or enable 4-bit quantization (`--force_quant --quant_bits 4`, often with `--force_lora`).
- Resume training? `--resume_ckpt path/to/ckpt.pt` (optionally `--reset_lr`)
- CPU-only? `--device cpu` works for some features (slower).
- Benchmarking? `python manage.py benchmark ...` with `--config`, `--seq_len`, `--model` etc.

---

## 📄 License
This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE).

---

## 🌏 Community & Citation
- Issues & PRs welcome!
- Gitee: https://gitee.com/dunimd/piscesl1.git
- GitHub: https://github.com/mf2023/piscesl1.git
- ModelScope: https://www.modelscope.cn/models/mfchina2024/PiscesL1

<h3 align="center">Where intuition navigates the depths of data · And empathy gives form to intelligence</h3>