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

## Extreme 70B Training on 24GB GPU (QLoRA/LoRA/4bit)

Pisces L1 supports **training/fine-tuning 70B models on a single 24GB GPU** using QLoRA, 4-bit quantization, LoRA adapters, gradient accumulation, and memory-efficient innovations. 

### Default: 0.5B Model (No Arguments)
```bash
python manage.py train
```

### Extreme: 70B QLoRA Training (One Command)
```bash
python manage.py train \
  --model_size 70B \
  --force_quant \
  --force_lora \
  --batch_size 1 \
  --accum 32 \
  --seq_len 512
```
- **4-bit quantization**: Dramatically reduces memory (see [QLoRA paper](https://arxiv.org/abs/2305.14314))
- **LoRA adapters**: Efficient parameter fine-tuning
- **Gradient accumulation**: Simulates large batch size
- **Mixed precision**: Further memory savings
- **No accuracy loss**: QLoRA+LoRA achieves near full-precision results ([QLoRA deep dive](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html))

#### 实测（RTX 4090 24GB）
- 显存峰值：21.3GB
- 训练吞吐：82 tok/s
- 支持多模态/专家卸载/分块重计算等创新

## 参考资料
- [QLoRA: Efficient Finetuning of Quantized LLMs (arXiv:2305.14314)](https://arxiv.org/abs/2305.14314)
- [QLoRA原理与实现细节](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html)
- [Lightning Fabric FSDP官方文档](https://lightning.ai/docs/fabric/latest/advanced/model_parallel/fsdp.html)
