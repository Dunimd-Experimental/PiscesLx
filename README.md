<div align="center">

[![Encre Agent](https://img.shields.io/badge/Encre_Agent—General_Purpose_AI_Agent-181717?style=for-the-badge&logo=github)](https://github.com/mf2023/Encre)

</div>

<div align="center">

# ⚖️ Legal Disclaimer

**Compliance with AI regulations is the user's legal obligation.**

Under applicable laws and regulations (including but not limited to China's "Interim Measures for the Management of Generative AI Services", EU "AI Act", US "AI Risk Management Framework", etc.), users are responsible for fulfilling compliance obligations. Non-compliant use may result in service termination, administrative penalties, or legal liability. Users assume all related risks.

**This project is licensed under Apache 2.0, permitting commercial use.**

---

# PiscesL1

English | [简体中文](README.zh.md)

[Security](SECURITY.md) | [Contributing](CONTRIBUTING.md) | [Code of Conduct](CODE_OF_CONDUCT.md)

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
    <img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-Dunimd-1E6CFF?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1zbmFtZT0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik03LjAwNiAwQzMuMTQyIDAgMCAzLjE0MiAwIDcuMDA2UzMuMTQyIDE0LjAxMiA3LjAwNiAxNC4wMTJDMTAuODcgMTQuMDEyIDE0LjAxMiAxMC44NyAxNC4wMTIgNy4wMDZDMTQuMDEyIDMuMTQyIDEwLjg3IDAgNy4wMDYgMFoiIGZpbGw9IiMxRTZDRkYiLz48L3N2Zz4K"/>
</a>
<a href="https://github.com/mf2023/Encre" target="_blank">
    <img alt="Encre Agent" src="https://img.shields.io/badge/Encre_Agent-181717?style=flat-square&logo=github"/>
</a>

A multimodal Mixture-of-Experts (MoE) model framework featuring the **Yv Architecture**, supporting text, image, audio, video, document, and agent modalities. PiscesL1 (PiscesLx series, Dunimd Team) provides a configurable architecture from 0.5B to 1T parameters. The codebase serves as a reference architecture design with configurable hyperparameters — trained model weights are not included in this repository.

</div>

<h2 align="center">Yv Architecture</h2>

### 🧠 YvUnifiedReasoner - Unified Reasoning System

YvUnifiedReasoner implements a routing framework that selects between Chain-of-Thought (CoT) and Multi-Path reasoning:

- **YvCoTMemoryReasoner**: Memory-augmented chain-of-thought reasoner with configurable depth (1-3 layers), early stopping, and self-correction
- **YvMultiPathReasoningEngine**: Multi-path reasoning supporting up to 8 parallel hypothesis streams with fact verification
- **Complexity-Based Routing**: Selection of reasoning path based on input complexity and sequence length
- **Control Tokens**: `<|start_hypothesis|>`, `<|start_evidence|>`, `<|start_conclusion|>`, `<|hypothesis_split|>`, `<|hypothesis_merge|>`

### 🔧 Yv MoE - Mixture-of-Experts

Mixture-of-Experts implementation with fine-grained routing:

- **YvStableMoEGate**: Gating with LSTM load predictor, configurable expert count and top-K routing
- **Fine-grained Expert Segmentation**: Each expert combines multiple sub-experts for finer routing granularity
- **Shared Expert Isolation**: Dedicated experts that process all tokens for domain-general knowledge
- **Auxiliary Loss-free Load Balancing**: Load balancing without auxiliary losses, using expert-level buffers
- **Cognitive Density Optimization**: Expert specialization via routing entropy, mutual information, and diversity losses

### 🧠 Subconscious System — Compute/Lookup Split

Architecture separating active computation from static knowledge lookup:

- **Product-Quantized Knowledge Field**: 16 codebooks × 131K entries, 131K¹⁶ virtual combinations (~0.27B params)
- **Dynamic Navigation Head**: 0.23B head routes queries through the codebook space
- **FiLM Modulation**: Retrieved knowledge modulates core layer activations via scale/shift
- **0.5B Overhead**: Combined head + field adds 0.5B to a 7B core, enabling 314B-equivalent knowledge capacity
- **Parallel to Context**: Knowledge runs parallel to the 1M context window, not increasing sequence length

### 🧠 Memory Separation — External Knowledge Store

Optional external knowledge retrieval via FAISS index:

- **Deterministic O(1) Lookup**: N-gram address hashing for constant-time retrieval
- **IVF-PQ Index**: Approximate nearest neighbor search for large knowledge bases
- **Cross-Attention Injection**: Retrieved knowledge integrated via gated cross-attention
- **Offline Knowledge Builder**: Independent 0.5B encoder for building the store from corpora
- **mmap-backed**: Memory-mapped storage for efficient disk-to-GPU transfer

### 🌐 Multimodal Perception Stack

Six-modality processing architecture:

- **YvVisionEncoder**: Patch encoding with native resolution support (up to 2048px) and patch packing
- **YvVideoEncoder**: Frame-level encoding with 3D spatio-temporal position encoding via RoPE
- **YvAudioEncoder**: Audio spectrum encoding with streaming processing support
- **YvDocEncoder**: Document image encoding with layout-aware structural processing
- **YvAgenticEncoder**: Agent state encoding for action space and state representation
- **YvCrossModalAttention**: Cross-modal attention for inter-modal interaction

### ⚛️ YvDynamicModalFusion - Dynamic Modal Fusion

Token-level multimodal fusion:

- **Cross-Modal Attention**: Exchange of information between modalities
- **Modality-Aware Position Embeddings**: Position encoding that accounts for modality type
- **Quality-Weighted Gating**: Dynamic weighting based on fusion quality scores
- **Recurrent Modal Refiner**: Iterative refinement loop for cross-modal consistency
- **Multiple Fusion Strategies**: Fusion tokens prepended to text, 3D feature concatenation, or compressed summary output

### 📏 Long Context Processing

Context extension mechanisms up to 4M tokens:

- **YaRN RoPE + Dynamic NTK Scaling**: Position encoding extrapolation for extended sequences
- **H2O Heavy-Hitter Oracle Attention**: Token retention based on attention score accumulation
- **Streaming Attention**: Generation beyond trained context length by maintaining a local window
- **Sliding Window Attention**: Local attention with configurable window size
- **Linear Attention**: O(n) complexity attention via kernel feature maps (ELU/Performer)
- **Paged Attention**: Block-based KV cache management for efficient memory use
- **Ring Attention**: Distributed processing for extended contexts
- **OOMB**: Out-of-Order Memory Banking for billion-token training contexts
- **REFORM**: Compression-gather-recompute strategy for memory-efficient long context

### 🔄 Hybrid Attention-SSM

Hybrid attention and state space model architecture:

- **Mamba-3 Integration**: SSM with discretization, complex states, and MIMO structure
- **YvSelectiveSSM**: State Space Model with input-dependent state transitions
- **Progressive Gating**: Gradual transition from attention to hybrid mode during training
- **Adaptive Routing**: Per-token selection between attention or SSM based on sequence features
- **Jamba-style Interleaved**: Alternating attention and SSM layers

### 🎯 Attention Mechanisms

Multiple attention implementations:

- **Flash Attention 2/3**: GPU-optimized efficient attention (Ampere+ and Hopper+)
- **Multi-Head Latent Attention (MLA)**: Low-rank KV compression via latent space
- **Embedding-Gated MLA (EG-MLA)**: MLA with embedding-based gating for improved expressiveness
- **Grouped Query Attention (GQA)**: Shared KV heads across query groups
- **ALiBi Position Encoding**: Extrapolation via linear bias without position embeddings
- **H2O Attention**: Heavy-Hitter Oracle for cache compression
- **DuoAttention**: Separate retrieval and streaming attention heads
- **Circulant Attention**: FFT-based approximation for long sequences
- **QK Normalization**: Query-Key normalization for training stability

### 🚀 Training Pipeline & Optimization

Training support:

- **EnTA (Encre Train Agent)**: Automated multi-teacher distillation pipeline (Rust-native backend)
- **GaLore**: Gradient low-rank projection for memory-efficient training
- **Multi-bit Quantization (FP4/INT4/INT8)**: Quantization for memory reduction
- **LoRA/QLoRA**: Low-rank adaptation for fine-tuning
- **Speculative Decoding**: Draft-verify for inference acceleration
- **Multi-Token Prediction (MTP)**: Predicting multiple future tokens per position
- **DAPO**: Decoupled clipping policy optimization for reinforcement learning
- **TTT-E2E**: Test-time training for adaptation during inference
- **MTP (Multi-Token Prediction)**: Auxiliary multi-token prediction heads
- **Ink Optimizer**: Unified optimizer with INT8/INT4 compression and sparse gradients
- **Smart Gradient Accumulation**: Adaptive memory management
- **Multi-task Learning**: Support for multiple task heads with adaptive weighting

#### Reference Configuration
Core components are located in `model/` and `model/multimodal/`, with hyperparameters in `configs/model/*.yaml`.

| Model Size | Layers | Hidden | Heads | KV Heads | MoE Experts | Top-K | Context | MLA Rank |
|------------|--------|--------|-------|----------|-------------|-------|---------|----------|
| 0.5B       | 16     | 640    | 10    | 5        | 6           | 2     | 256K    | 256      |
| 1.5B       | 16     | 896    | 14    | 7        | 6           | 2     | 256K    | 256      |
| 7B         | 28     | 3584   | 32    | 8        | 128         | 8     | 1M      | 512      |
| 32B        | 64     | 5120   | 40    | 8        | 128         | 8     | 1M      | 512      |
| 64B        | 80     | 6656   | 52    | 8        | 128         | 8     | 1M      | 1024     |
| 70B        | 80     | 8192   | 64    | 8        | 128         | 8     | 2M      | 1024     |
| 128B       | 120    | 10240  | 80    | 8        | 128         | 8     | 2M      | 1536     |
| 314B       | 160    | 12288  | 96    | 12       | 256         | 8     | 4M      | 2048     |
| 671B       | 200    | 16384  | 128   | 16       | 256         | 8     | 4M      | 2048     |
| 1T         | 240    | 20480  | 160   | 20       | 512         | 8     | 4M      | 2560     |

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

- Python: Recommended 3.11+
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

# 3. Download default dataset
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
| setup     | Environment setup and dependency installation                       |
| enta      | **EnTA autonomous training loop** (LLM-driven, multi-teacher)      |
| train     | Train model (support quantization / LoRA / RLHF / GaLore)          |
| serve     | Start OpenAI-compatible backend inference service                  |
| test      | Project health check (8-stage validation)                          |
| monitor   | System monitoring (GPU/CPU/memory)                                  |
| download  | Download dataset                                                   |
| benchmark | Model evaluation and benchmarking                                   |
| mcp       | MCP tool management (status / warmup / refresh-cache)              |
| watermark | Watermark detection (text/file/image/audio/video/model weights)   |
| action    | Background process management (submit/status/control)               |
| dev       | Developer mode for training (vim-style command interface)           |
| cache     | Cache management for .pisceslx directory                           |
| publish   | Package and publish models as Docker images                        |
| help      | Show help information                                             |

### EnTA Training
```bash
# List configured teacher models
python manage.py enta --list_models

# Dry run (validate pipeline)
python manage.py enta --dry_run

# Full training with teacher model and student checkpoint
python manage.py enta --teacher deepseek-r1 --model_path ./ckpt/7B.pt

# With round-table and subconscious training
python manage.py enta --teacher deepseek-r1 \
  --aux_teachers "deepseek-v3.2,qwen3.6,agens-2.0-flash" \
  --model_path ./ckpt/7B.pt

# Start from advanced stage
python manage.py enta --enta_stage advanced --max_tasks 10000

# Disable subconscious (train 7B core only)
python manage.py enta --no_subconscious
```

### Quick Experience
```bash
# Train 0.5B model
python manage.py train --model_size 0.5B

# Start backend service
python manage.py serve --model_size 7B --port 8000
```

### API Usage Examples
```bash
# Chat Completion
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "pisceslx-7b", "messages": [{"role": "user", "content": "Hello, introduce yourself"}]}'

# Streaming Response
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "pisceslx-7b", "messages": [...], "stream": true}'

# Embedding Generation
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "pisceslx-7b", "input": "Hello world"}'
```

### Common Examples
```bash
# Dataset management
python manage.py download --max_samples 50000

# Training examples
python manage.py train --model_size 0.5B --dataset Chinese2
python manage.py train --model_size 1B --dataset Chinese2 --resume_ckpt runs/last.pt --reset_lr
python manage.py train --model_size 7B --dataset Chinese2 --force_quant --quant_bits 4 --force_lora
python manage.py train --model_size 7B --dataset Chinese2 --rlhf --rlhf_dataset dunimd/human_feedback --rlhf_lr 1e-5

# Backend service
python manage.py serve --model_size 7B --port 8000
python manage.py serve --model_size 14B --host 0.0.0.0 --port 8080 --workers 4
python manage.py serve --model_size 72B

# Benchmark examples
python manage.py benchmark --list
python manage.py benchmark --info mmlu
python manage.py benchmark --benchmark mmlu --config configs/0.5B.json --seq_len 4096 --model ckpt/model.pt
python manage.py benchmark --perf --config configs/0.5B.json --selftest

# MCP tools
python manage.py mcp --mcp_action status
python manage.py mcp --mcp_action warmup
python manage.py mcp --mcp_action refresh-cache

# Watermark detection
python manage.py watermark --text "Detect text watermark"
python manage.py watermark --file document.txt
python manage.py watermark --image-file image.png
python manage.py watermark --audio-file audio.wav
python manage.py watermark --video-file video.mp4
python manage.py watermark --model-file model.pt
python manage.py watermark --weights-verify --ckpt model.pt

# Background process management
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

# GPU resource management
python manage.py action gpu list
python manage.py action gpu status
python manage.py action gpu status --gpu_id 0
python manage.py action gpu release --task_id <run_id>

# Task queue management
python manage.py action queue list
python manage.py action queue stats
python manage.py action queue clear --priority low

# System resources
python manage.py action resources status
python manage.py action resources utilization

# Task recovery
python manage.py action recover <run_id>
python manage.py action recover <run_id> --checkpoint runs/<run_id>/ckpt.pt

# Developer mode (vim-style command interface for training)
python manage.py dev enable    # Enable developer mode
python manage.py dev disable   # Disable developer mode
python manage.py dev status    # Check developer mode status

# Cache management for .pisceslx directory
python manage.py cache         # Show cache status
python manage.py cache clean   # Clean all cache (settings/ protected)

# Publish models as Docker images with inference engine
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

<h2 align="center">📦 Dataset</h2>

Dataset is configured by `configs/dataset.yaml` and downloaded through:
```bash
python manage.py download
```
- Default download priority: ModelScope → HuggingFace (automatic mirroring when inaccessible)
- Complete list see `configs/dataset.yaml`

---

<h2 align="center">❓ Frequently Asked Questions (FAQ)</h2>

- How to view available commands? `python manage.py help`
- How to add new dataset? Edit `configs/dataset.yaml` and run `python manage.py download`. Custom dataset recommend JSONL (text) or Parquet (input_ids/labels).
- Insufficient GPU memory? Use smaller model, reduce sequence length, or enable 4-bit quantization (`--force_quant --quant_bits 4`, usually with `--force_lora`).
- How to resume training? `--resume_ckpt path/to/ckpt.pt` (optional `--reset_lr`)
- CPU only? Can use `--device cpu` (slower performance).
- How to perform evaluation? `python manage.py benchmark ...`, with `--config`, `--seq_len`, `--model` and other parameters.
- How does EnTA work? EnTA is an LLM-driven agent that orchestrates multi-teacher distillation. See [Training Pipeline](#-training-pipeline--optimization).
- Does this repository include trained model weights? No. This repository provides the architecture, training pipeline, and inference code. Model weights must be trained separately using the EnTA pipeline.

---

<h2 align="center">🌏 Community & Citation</h2>

- Welcome to submit Issues and PRs!
- Gitee: https://gitee.com/dunimd/piscesl1.git
- GitHub: https://github.com/mf2023/piscesl1.git
- ModelScope: https://www.modelscope.cn/models/mfchina2024/PiscesL1

---

<h2 align="center">📚 Academic References</h2>

This project implements algorithms from the following academic papers. We sincerely thank the authors for their contributions.

### Attention Mechanisms

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
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

### Position Encoding

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| Sinusoidal PE | Attention Is All You Need | Vaswani et al. | NeurIPS | 2017 | [embedding.py](model/core/embedding.py#L283-L284) |
| RoPE | RoFormer: Enhanced Transformer with Rotary Position Embedding | Su et al. | - | 2021 | [norms.py](model/core/norms.py#L548-L687) |
| YaRN | YaRN: Efficient Context Window Extension of Large Language Models | Peng et al. | - | 2023 | [norms.py](model/core/norms.py#L689-L841) |

### Normalization & Activation

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| RMSNorm | Root Mean Square Layer Normalization | Zhang & Sennrich | NeurIPS | 2019 | [norms.py](model/core/norms.py#L192-L193) |
| Adaptive LayerNorm | Scalable Diffusion Models with Transformers (DiT) | Peebles & Xie | ICCV | 2023 | [norms.py](model/core/norms.py#L387-L388) |
| LayerScale | Going deeper with Image Transformers | Touvron et al. | ICCV | 2021 | [blocks.py](model/core/blocks.py#L349-L350) |
| SwiGLU | GLU Variants Improve Transformer | Shazeer | - | 2020 | [blocks.py](model/core/blocks.py#L402-L440) |
| GeGLU | GLU Variants Improve Transformer | Shazeer | - | 2020 | [blocks.py](model/core/blocks.py#L453-L490) |
| Group Normalization | Group Normalization | Wu & He | ECCV | 2018 | [norms.py](model/core/norms.py#L497-L498) |

### State Space Models

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| Mamba | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu & Dao | arXiv | 2023 | [blocks.py](model/core/blocks.py#L1434-L1437) |
| Mamba-2 | Mamba-2: Transforming Transformers | Dao et al. | arXiv | 2024 | [blocks.py](model/core/blocks.py#L1437) |

### Mixture of Experts

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| UltraMem TDQKR | UltraMem | ByteDance | ICLR | 2025 | [layer.py](model/moe/layer.py#L216) |
| DeepSeekMoE | DeepSeek-V3 Technical Report | DeepSeek Team | - | 2024 | [expert.py](model/moe/expert.py#L50), [layer.py](model/moe/layer.py#L390) |

### Inference Optimization

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| Speculative Decoding | Fast Inference from Transformers via Speculative Decoding | Leviathan et al. | ICML | 2023 | [cache.py](model/core/cache.py#L1013-L1015) |
| BLIP-2 | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models | Li et al. | ICML | 2023 | [cache.py](model/core/cache.py#L1217-L1219) |

### Training Optimization

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| K-FAC | Optimizing Neural Networks with Kronecker-factored Approximate Curvature | Martens & Grosse | ICML | 2015 | [kfac.py](opss/train/kfac.py#L65-L66) |
| K-FAC for Conv | A Kronecker-factored Approximate Fisher Matrix for Convolution Layers | Grosse & Martens | ICML | 2016 | [kfac.py](opss/train/kfac.py#L67-L68) |
| Multi-Task Uncertainty | Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics | Kendall et al. | CVPR | 2018 | [multitask_uncertainty.py](opss/train/multitask_uncertainty.py#L51-L53) |
| SGDR | SGDR: Stochastic Gradient Descent with Warm Restarts | Loshchilov & Hutter | arXiv | 2016 | [modality_scheduler.py](opss/train/modality_scheduler.py#L49-L51) |
| Chinchilla Scaling | Training Compute-Optimal Large Language Models | Hoffmann et al. | - | 2022 | [scaling/__init__.py](opss/scaling/__init__.py#L25) |

### Alignment & Reinforcement Learning

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| DPO | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafailov et al. | NeurIPS | 2023 | [dpo.py](opss/train/dpo.py#L22-L29) |
| GRPO | DeepSeek R1 Technical Report | DeepSeek Team | arXiv | 2024 | [grpo.py](opss/train/grpo.py#L33-L34) |
| RLVR | DeepSeek R1 Technical Report / OpenAI o1 | DeepSeek / OpenAI | arXiv | 2024/2025 | [rlvr.py](opss/train/rlvr.py#L35-L36) |
| TPO | Test-Time Preference Optimization: On-the-fly Alignment via Iterative Textual Feedback | - | arXiv | 2025 | [tpo.py](opss/infer/tpo.py#L33-L35) |

### Reasoning & Agentic

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| ReAct | ReAct: Synergizing Reasoning and Acting in Language Models | Yao et al. | ICLR | 2023 | [react_agentic.py](model/multimodal/react_agentic.py#L31-L32) |
| Chain-of-Thought | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Wei et al. | NeurIPS | 2022 | [react_agentic.py](model/multimodal/react_agentic.py#L33) |

### Optimizers

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| GaLore | GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | Zhao et al. | arXiv | 2024 | [galore.py](opss/optim/galore.py#L35-L37) |
| ROOT | ROOT: Robust Orthogonalized Optimizer for Neural Network Training | Huawei Noah's Ark Lab | arXiv | 2024 | [root.py](opss/optim/root.py#L33-L34) |
| FP4 Training | Optimizing Large Language Model Training Using FP4 Quantization | - | arXiv | 2025 | [fp4.py](opss/optim/fp4.py#L35-L36) |

### Quantization

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| GPTQ | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Frantar et al. | ICLR | 2023 | [orchestrator.py](tools/train/orchestrator.py#L840) |

### Citation

If you use this project in your research, please cite:

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
| accelerate | Apache 2.0 | addict | MIT |
| aiofiles | Apache 2.0 | audioread | MIT |
| av | BSD 3-Clause | beautifulsoup4 | MIT |
| bert-score | MIT | bitsandbytes | MIT |
| causal-conv1d | Apache 2.0 | datasets | Apache 2.0 |
| decord | Apache 2.0 | deepspeed | Apache 2.0 |
| dmsc | Apache 2.0 | docker | Apache 2.0 |
| duckduckgo-search | MIT | einops | MIT |
| evalscope | Apache 2.0 | fastapi | MIT |
| flash-attn | BSD 3-Clause | GitPython | BSD 3-Clause |
| gradio | Apache 2.0 | httpx | BSD 3-Clause |
| huggingface-hub | Apache 2.0 | hydra-core | MIT |
| ijson | BSD 3-Clause | imageio | BSD 3-Clause |
| imageio-ffmpeg | BSD 3-Clause | jsonlines | MIT |
| kagglehub | Apache 2.0 | librosa | ISC |
| lm-eval | MIT | mamba-ssm | Apache 2.0 |
| mlflow | Apache 2.0 | modelscope | Apache 2.0 |
| numpy | BSD 3-Clause | nvidia-ml-py3 | BSD 3-Clause |
| ocrmypdf | MPL 2.0 | omegaconf | BSD 3-Clause |
| openai | Apache 2.0 | opencv-python | MIT |
| pandas | BSD 3-Clause | pathlib2 | MIT |
| pdf2image | MIT | pdfplumber | MIT |
| peft | Apache 2.0 | pillow | HPND |
| plotly | MIT | psutil | BSD 3-Clause |
| pyarrow | Apache 2.0 | pydantic | MIT |
| pydub | MIT | PyMuPDF | AGPL 3.0 |
| python-docx | MIT | python-multipart | Apache 2.0 |
| python-pptx | MIT | pytorch-lightning | Apache 2.0 |
| pytz | MIT | pywin32 | PSF |
| PyYAML | MIT | requests | Apache 2.0 |
| rich | MIT | rouge-score | Apache 2.0 |
| sacrebleu | Apache 2.0 | safetensors | Apache 2.0 |
| scikit-learn | BSD 3-Clause | scipy | BSD 3-Clause |
| soundfile | BSD 3-Clause | streamlit | Apache 2.0 |
| tensorboard | Apache 2.0 | textual | MIT |
| timm | Apache 2.0 | tokenizers | Apache 2.0 |
| torch | BSD-style | torch-directml | MIT |
| torchaudio | BSD-style | torchvision | BSD-style |
| tqdm | MIT | transformers | Apache 2.0 |
| triton | MIT | trl | Apache 2.0 |
| uvicorn | BSD 3-Clause | wandb | MIT |
| wheel | MIT | windows-curses | BSD 3-Clause |
| wrapt | BSD 3-Clause | xformers | BSD 3-Clause |

</div>

</div>
