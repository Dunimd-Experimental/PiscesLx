# Pisces L1

A 0.5B multimodal MoE model supporting text/image/audio/document, can run inference on single RTX 4090.

## Features

- **Multimodal**: Text, image, audio, and document understanding
- **MoE Architecture**: Mixture of Experts for efficient scaling
- **Lightweight**: 0.5B parameters, suitable for consumer GPUs
- **Modern**: RMSNorm, RoPE, Grouped-Query Attention

## Quick Start

### GPU Check

Before training, check your GPU status:
```bash
python manage.py check
```

### Installation & Usage (One Command)

```bash
# 1. Setup environment and install all dependencies (auto venv)
python manage.py setup

# 2. Start training (auto-downloads dataset)
python manage.py download
python manage.py train

# 3. Inference
python manage.py infer --ckpt ckpt/pisces_tiny_stories_epoch1.pt --prompt "Hello"
```

**Note**: No need to run any shell or PowerShell scripts. All environment setup and dependency installation are handled by `python run.py setup`.

## Model Architecture

- **Base Model**: 0.5B parameters, 24 layers
- **MoE**: 64 experts, top-2 routing
- **Attention**: 16 heads, 4 KV heads (GQA)
- **Context**: 8K tokens
- **Multimodal**: CLIP (vision), AST (audio), LayoutLMv3 (documents)

## Datasets

- **TinyStories**: Text generation
- **COCO Few-shot**: Image-text few-shot learning
- **AudioSet Captions**: Audio caption generation
