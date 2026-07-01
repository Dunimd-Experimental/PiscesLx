<div align="center">

[![Encre Agent](https://img.shields.io/badge/Encre_Agent—General_Purpose_AI_Agent-181717?style=for-the-badge&logo=github)](https://github.com/mf2023/Encre)

</div>

<div align="center">

# ⚖️ Legal Disclaimer

**Compliance with AI regulations is the user's legal obligation.**

Under applicable laws and regulations (including but not limited to China's "Interim Measures for the Management of Generative AI Services", EU "AI Act", US "AI Risk Management Framework", etc.), users are responsible for fulfilling compliance obligations. Non-compliant use may result in service termination, administrative penalties, or legal liability. Users assume all related risks.

**This project is licensed under Apache 2.0, permitting commercial use.**

---

<img src="assets/svg/PiscesLx.svg" width="128" height="72">

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
- Does this repository include trained model weights? No. This repository provides the architecture and inference code. Model weights must be trained separately.

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
| QK Normalization | Query-Key Normalization for Transformers | Henry et al. | EMNLP (Findings) | 2020 | [attention.py](model/core/attention.py#L656-L657) |
| Linear Attention | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention | Katharopoulos et al. | ICML | 2020 | [attention.py](model/core/attention.py#L787-L789) |
| S4 | Efficiently Modeling Long Sequences with Structured State Spaces | Gu et al. | ICLR | 2022 | [attention.py](model/core/attention.py#L1073-L1075) |
| Longformer | Longformer: The Long-Document Transformer | Beltagy et al. | ICLR | 2020 | [attention.py](model/core/attention.py#L1225-L1226) |
| BigBird | Big Bird: Transformers for Longer Sequences | Zaheer et al. | NeurIPS | 2020 | [attention.py](model/core/attention.py#L1437-L1438) |
| Sliding Window Attn | Mistral 7B | Jiang et al. | arXiv | 2023 | [attention.py](model/core/attention.py#L1455) |
| Ring Attention | Ring Attention with Blockwise Transformers for Near-Infinite Context | Liu et al. | ICLR | 2024 | [attention.py](model/core/attention.py#L2479-L2481) |
| MQA | Fast Transformer Decoding: One Write-Head is All You Need | Shazeer | arXiv | 2019 | [attention.py](model/core/attention.py#L2831-L2832) |
| H2O | H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models | Zhang et al. | NeurIPS | 2023 | [attention.py](model/core/attention.py#L3026-L3028) |
| LongRoPE | LongRoPE: Extending LLM Context Window Beyond 2M Tokens | Ding et al. | ICML | 2024 | [attention.py](model/core/attention.py#L4152-L4154) |
| MLA | DeepSeek-V2 (Multi-head Latent Attention) | DeepSeek-AI | arXiv | 2024 | [attention.py](model/core/attention.py#L5118) |
| EG-MLA | Embedding-Gated MLA (Yv Architecture original) | Dunimd Team | - | 2026 | [eg_mla.py](model/core/eg_mla.py#L50) |
| DuoAttention | DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads | Xiao et al. | arXiv | 2024 | [duo_attention.py](model/core/duo_attention.py#L42) |
| PagedAttention | Efficient Memory Management for Large Language Model Serving with PagedAttention | Kwon et al. | SOSP | 2023 | [attention.py](model/core/attention.py#L1654-L1656) |
| Flash Attention | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao et al. | NeurIPS | 2022 | [attention.py](model/core/attention.py#L1910-L1914) |
| Flash Attention 2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | Dao | arXiv | 2023 | [attention.py](model/core/attention.py#L1913-L1914) |
| Flash Attention 3 | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Blockwise Parallelism | Shah et al. | arXiv | 2024 | [flash_attention.py](opss/infer/flash_attention.py#L37-L38) |
| CoPE | CAPE: Context-Adaptive Positional Encoding for Length Extrapolation | Zheng et al. | arXiv | 2024 | [attention.py](model/core/attention.py#L4197-L4199) |
| Tactic Sparse Attn | Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs | Zhu et al. | ICLR | 2026 | [token_sparse_attn.py](model/core/token_sparse_attn.py#L55) |
| CSA/HCA | Compressed Sparse Attention / Heavily Compressed Attention (DeepSeek-V4) | DeepSeek-AI | arXiv | 2026 | [csa_hca.py](model/core/csa_hca.py#L133) |

### Position Encoding

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| Sinusoidal PE | Attention Is All You Need | Vaswani et al. | NeurIPS | 2017 | [embedding.py](model/core/embedding.py#L283-L284) |
| RoPE | RoFormer: Enhanced Transformer with Rotary Position Embedding | Su et al. | arXiv | 2021 | [norms.py](model/core/norms.py#L548-L687) |
| YaRN | YaRN: Efficient Context Window Extension of Large Language Models | Peng et al. | arXiv | 2023 | [norms.py](model/core/norms.py#L689-L841) |
| MrRoPE | MrRoPE: Mixed-Radix Rotary Position Embedding | Tian et al. | arXiv | 2026 | [mr_rope.py](model/core/mr_rope.py#L27) |

### Normalization & Activation

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| RMSNorm | Root Mean Square Layer Normalization | Zhang & Sennrich | NeurIPS | 2019 | [norms.py](model/core/norms.py#L192-L193) |
| Adaptive LayerNorm | Scalable Diffusion Models with Transformers (DiT) | Peebles & Xie | ICCV | 2023 | [norms.py](model/core/norms.py#L387-L388) |
| LayerScale | Going deeper with Image Transformers | Touvron et al. | ICCV | 2021 | [blocks.py](model/core/blocks.py#L349-L350) |
| SwiGLU | GLU Variants Improve Transformer | Shazeer | arXiv | 2020 | [blocks.py](model/core/blocks.py#L402-L440) |
| GeGLU | GLU Variants Improve Transformer | Shazeer | arXiv | 2020 | [blocks.py](model/core/blocks.py#L453-L490) |
| Group Normalization | Group Normalization | Wu & He | ECCV | 2018 | [norms.py](model/core/norms.py#L497-L498) |

### State Space Models

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| Mamba | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu & Dao | arXiv | 2023 | [mamba3.py](model/core/mamba3.py#L679) |
| Mamba-2 | Mamba-2: State Space Duality | Dao & Gu | arXiv | 2024 | [mamba3.py](model/core/mamba3.py#L1168) |
| Jamba | Jamba: A Hybrid Transformer-Mamba Language Model | Lieber et al. | arXiv | 2024 | [hybrid.py](model/core/hybrid.py#L961) |

### Mixture of Experts

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| MoE Routing | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | Shazeer et al. | ICLR | 2017 | [gate.py](model/moe/gate.py#L125) |
| Expert Choice | Mixture-of-Experts with Expert Choice Routing | Zhou et al. | NeurIPS | 2022 | [layer.py](model/moe/layer.py#L118) |
| DeepSeekMoE | DeepSeek-V2 / DeepSeek-V3 Technical Report | DeepSeek-AI | arXiv | 2024 | [layer.py](model/moe/layer.py#L832) |
| Fine-Grained MoE | DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model | DeepSeek-AI | arXiv | 2024 | [layer.py](model/moe/layer.py#L222) |
| PathNet | PathNet: Evolution Channels Gradient Descent in Super Neural Networks | Fernando et al. | arXiv | 2017 | [gate.py](model/moe/gate.py#L1739) |
| Phi Balancing | $\phi$-Balancing for Mixture-of-Experts Training | Chen et al. | arXiv | 2026 | [gate.py](model/moe/gate.py#L1811) |
| UltraMem TDQKR | Ultra-Sparse Memory Network (UltraMem) | ByteDance Seed | ICLR | 2025 | [layer.py](model/moe/layer.py#L216) |
| HiCl Router | HiCL: Hippocampal-Inspired Continual Learning | Kapoor et al. | AAAI | 2026 | [hicl_router.py](model/moe/hicl_router.py#L43) |
| Graph-of-Tokens | Improving Routing in Sparse Mixture of Experts with Graph of Tokens | Nguyen et al. | arXiv | 2025 | [graph_of_tokens.py](model/moe/graph_of_tokens.py#L27) |
| RoMA | Routing Manifold Alignment Improves Generalization of MoE LLMs | Li et al. | ICLR | 2026 | [graph_of_tokens.py](model/moe/graph_of_tokens.py#L132) |
| PathMoE | Path-Constrained Mixture-of-Experts | Gu et al. | arXiv | 2026 | [path_moe.py](model/moe/path_moe.py) |
| Info Bottleneck | The Information Bottleneck Method | Tishby et al. | arXiv | 2000 | [diversity.py](model/moe/diversity.py#L204) |
| Contrastive Div. | SimCLR: A Simple Framework for Contrastive Learning | Chen et al. | ICML | 2020 | [diversity.py](model/moe/diversity.py#L531) |

### Inference Optimization

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| Speculative Decoding | Fast Inference from Transformers via Speculative Decoding | Leviathan et al. | ICML | 2023 | [speculative.py](model/generation/speculative.py#L582) |
| Medusa | Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads | Cai et al. | arXiv | 2024 | [speculative.py](model/generation/speculative.py#L373) |
| D-Spark | DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation | DeepSeek-AI / PKU | arXiv | 2026 | [speculative.py](model/generation/speculative.py#L1385) |
| BLIP-2 | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models | Li et al. | ICML | 2023 | [cache.py](model/core/cache.py#L1217-L1219) |

### Training Optimization

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| LoRA | LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | ICLR | 2022 | [blocks.py](model/core/blocks.py#L521) |
| DoRA | DoRA: Weight-Decomposed Low-Rank Adaptation | Liu et al. | ICML | 2024 | [blocks.py](model/core/blocks.py#L575) |
| K-FAC | Optimizing Neural Networks with Kronecker-factored Approximate Curvature | Martens & Grosse | ICML | 2015 | [kfac.py](opss/train/kfac.py#L65-L66) |
| K-FAC for Conv | A Kronecker-factored Approximate Fisher Matrix for Convolution Layers | Grosse & Martens | ICML | 2016 | [kfac.py](opss/train/kfac.py#L67-L68) |
| GaLore | GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | Zhao et al. | ICML | 2024 | [galore.py](opss/optim/galore.py#L103) |
| Mixture-of-Depths | Mixture-of-Depths: Dynamically Allocating Compute in Transformer Networks | Raposo et al. | arXiv | 2024 | [blocks.py](model/core/blocks.py#L724) |
| Adaptive Computation Time | Adaptive Computation Time for Recurrent Neural Networks | Graves | arXiv | 2016 | [blocks.py](model/core/blocks.py#L640) |
| EWC | Overcoming Catastrophic Forgetting | Kirkpatrick et al. | PNAS | 2017 | [ewc.py](model/core/ewc.py#L41) |
| DeepNet | DeepNet: Scaling Transformers to 1,000 Layers | Wang et al. | arXiv | 2022 | [norms.py](model/core/norms.py#L1193) |
| Knowledge Distillation | Distilling the Knowledge in a Neural Network | Hinton et al. | NeurIPS Workshop | 2015 | [distill.py](opss/train/distill.py#L796) |
| Multi-Task Uncertainty | Multi-Task Learning Using Uncertainty to Weigh Losses | Kendall et al. | CVPR | 2018 | [multitask_uncertainty.py](opss/train/multitask_uncertainty.py#L51-L53) |
| SGDR | SGDR: Stochastic Gradient Descent with Warm Restarts | Loshchilov & Hutter | arXiv | 2016 | [modality_scheduler.py](opss/train/modality_scheduler.py#L49-L51) |
| Chinchilla Scaling | Training Compute-Optimal Large Language Models | Hoffmann et al. | NeurIPS | 2022 | [scaling/__init__.py](opss/scaling/__init__.py#L25) |

### Alignment & Reinforcement Learning

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| DPO | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafailov et al. | NeurIPS | 2023 | [dpo.py](opss/train/dpo.py#L534) |
| GRPO | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI | arXiv | 2025 | [grpo.py](opss/train/grpo.py#L249) |
| RLVR | DeepSeek-R1 / OpenAI o1 | DeepSeek / OpenAI | arXiv | 2025 | [rlvr.py](opss/train/rlvr.py#L129) |
| DAPO | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | Yu et al. (ByteDance/Tsinghua) | arXiv | 2025 | [dapo.py](opss/train/dapo.py#L38) |
| TPO | Test-Time Preference Optimization: On-the-fly Alignment via Iterative Textual Feedback | Li et al. | ICML | 2025 | [tpo.py](opss/infer/tpo.py#L33-L35) |

### Reasoning & Agentic

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| Chain-of-Thought | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Wei et al. | NeurIPS | 2022 | [cot_memory.py](model/reasoning/reasoner/cot_memory.py#L94) |
| Self-Consistency | Self-Consistency Improves Chain of Thought Reasoning in Language Models | Wang et al. | ICLR | 2023 | [multipath_core.py](model/reasoning/reasoner/multipath_core.py#L109) |
| Tree-of-Thoughts | Tree of Thoughts: Deliberate Problem Solving with Large Language Models | Yao et al. | NeurIPS | 2023 | [recursive_depth.py](model/reasoning/reasoner/recursive_depth.py#L1146) |
| ReAct | ReAct: Synergizing Reasoning and Acting in Language Models | Yao et al. | ICLR | 2023 | [react_agentic.py](model/multimodal/react_agentic.py#L31-L32) |
| VeriCoT | VeriCoT: Neuro-symbolic Chain-of-Thought Validation via Logical Consistency Checks | Feng et al. | arXiv | 2025 | [vericot.py](model/reasoning/vericot.py#L35) |
| TTT-E2E | End-to-End Test-Time Training for Long Context | Tandon et al. | arXiv | 2025 | [ttt_e2e.py](model/reasoning/ttt_e2e.py#L41) |

### Optimizers

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| GaLore | GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | Zhao et al. | ICML | 2024 | [galore.py](opss/optim/galore.py#L103) |
| ROOT | ROOT: Robust Orthogonalized Optimizer for Neural Network Training | Huawei Noah's Ark Lab | arXiv | 2025 | [root.py](opss/optim/root.py#L125) |
| FP4 Training | Optimizing Large Language Model Training Using FP4 Quantization | Wang et al. | arXiv | 2025 | [fp4.py](opss/optim/fp4.py#L151) |
| Muon | Muon Optimizer | - | - | 2025 | [muon.py](utils/muon.py) |

### Quantization

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| GPTQ | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Frantar et al. | ICLR | 2023 | [methods.py](opss/quantize/methods.py#L118) |
| AWQ | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | Lin et al. | MLSys | 2024 | [methods.py](opss/quantize/methods.py#L320) |
| SmoothQuant | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | Xiao et al. | ICML | 2023 | [methods.py](opss/quantize/methods.py#L556) |

### Multimodal

| Algorithm | Paper | Authors | Venue | Year | Code |
|-----------|-------|---------|-------|------|------|
| ViT | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | Dosovitskiy et al. | ICLR | 2021 | [vision.py](model/multimodal/vision.py#L403) |
| SigLIP | Sigmoid Loss for Language Image Pre-Training | Zhai et al. | ICCV | 2023 | [vision.py](model/multimodal/vision.py) |
| LayoutLMv3 | LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking | Huang et al. | ACM Multimedia | 2022 | [doc.py](model/multimodal/doc.py#L85) |
| HiFi-GAN | HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | Kong et al. | NeurIPS | 2020 | [audio.py](model/multimodal/audio.py#L543) |
| FiLM | FiLM: Visual Reasoning with a General Conditioning Layer | Perez et al. | NeurIPS | 2017 | [dual_injector.py](model/core/dual_injector.py#L55) |
| SAGE | SAGE: Multi-Agent Self-Evolution for LLM Reasoning | Peng et al. | arXiv | 2026 | [seer_executor.py](model/multimodal/seer_executor.py#L36) |
| CoMeT | CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling | Zhao et al. | ACL | 2026 | [comet.py](model/core/comet.py#L145) |
| Seirênes | Seirênes: Adversarial Self-Play with Evolving Distractions for LLM Reasoning | Zhang et al. | arXiv | 2026 | [seirenes.py](model/core/seirenes.py) |
| mHC-lite | mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations | Yang & Gao | arXiv | 2026 | [mhc_lite.py](model/core/mhc_lite.py#L41) |
| Engram | Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models | Cheng et al. (DeepSeek) | arXiv | 2026 | [memory_router.py](model/core/memory_router.py#L34) |
| LayerRoute | LayerRoute: Input-Conditioned Adaptive Layer Skipping via LoRA Fine-Tuning | Sikdar | arXiv | 2026 | [layer_route.py](model/core/layer_route.py#L27) |
| HydraHead | HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization | Tan et al. (Alibaba) | arXiv | 2026 | [attention.py](model/core/attention.py#L6195-L6196) |
| LCA | Latent-Condensed Transformer for Efficient Long Context Modeling | You et al. | ACL | 2026 | [attention.py](model/core/attention.py#L6345-L6346) |
| Circulant Attn | Vision Transformers are Circulant Attention Learners | Han et al. (Tsinghua) | AAAI | 2026 | [attention.py](model/core/attention.py#L1213) |
| SparDA | SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference | Fu et al. (NVIDIA) | arXiv | 2026 | [rca_fusion.py](model/multimodal/rca_fusion.py) |
| SEER | Self-Guided Function Calling in LLMs via Stepwise Experience Recall | Cui et al. | EMNLP | 2025 | [seer_executor.py](model/multimodal/seer_executor.py) |
| SEAL | Self-Adapting Language Models (SEAL) | Zweiger et al. (MIT) | arXiv | 2025 | [self_evolution.py](model/reasoning/self_evolution.py#L40) |
| A-Evolve | Position: Agentic Evolution is the Path to Evolving LLMs | Lin et al. | arXiv | 2026 | [self_evolution.py](model/reasoning/self_evolution.py#L150) |

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
