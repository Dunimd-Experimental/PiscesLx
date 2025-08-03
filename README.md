# ⚠️ Compliance Notice

**If you use this model in China, including but not limited to training, fine tuning, commercial testing, etc., if you provide any services to the public, please first complete the filing procedures in accordance with relevant laws and regulations.**

---

# Pisces L1

English | [简体中文](README.zh.md)

A next-generation lightweight multimodal Mixture-of-Experts (MoE) model supporting text, image, audio, and document understanding. Designed for both research and practical applications, Pisces L1 can run on a single RTX 4090 and scale up to 70B parameters with advanced memory optimization.

---

## 🚀 Features

- **Multimodal**: Unified support for text, image, audio, and document inputs
- **MoE Architecture**: Efficient Mixture-of-Experts, scalable from 0.5B to 70B parameters
- **Lightweight**: 0.5B base model runs on consumer GPUs (24GB VRAM)
- **Modern Transformer**: RMSNorm, RoPE, Grouped-Query Attention, and more
- **Extreme Adaptability**: QLoRA, 4-bit quantization, LoRA adapters, gradient accumulation
- **One-command Workflow**: All management via `python manage.py` (see below)

---

## 🛠️ Installation & Environment

- **Python**: 3.9–3.11 recommended
- **CUDA**: 11.8+ (for GPU training/inference)
- **Dependencies**: All required packages are listed in `requirements.txt`

### Quick Setup
```bash
git clone https://gitee.com/dunimd/piscesl1.git or git clone https://github.com/mf2023/PiscesL1.git
cd piscesl1
python manage.py setup
```
This will create a virtual environment and install all dependencies automatically.

---

## ⚡ Command Line Usage

All commands are managed via `python manage.py <command>`. For help:
```bash
python manage.py help
```

### Main Commands
| Command    | Description                                                          |
|------------|----------------------------------------------------------------------|
| setup      | Environment setup and dependency install                             |
| train      | Train the model (supports distributed training with `--distributed`) |
| infer      | Run inference with a trained model                                   |
| check      | Check GPU and dependencies                                           |
| monitor    | System monitor (GPU/CPU/memory)                                      |
| download   | Download datasets for training                                       |
| arrow      | Arrow/JSON dataset conversion                                        |
| quantize   | Quantize model to 4/8-bit for efficiency                             |
| benchmark  | Run performance benchmarking                                         |
| help       | Show help message                                                    |

#### Example
```bash
python manage.py download
python manage.py train
python manage.py infer --ckpt ckpt/model.pt --prompt "Hello, Pisces!"
```

---

## 🧠 Model Architecture & Configurations

| Model Size | Layers | Hidden | Heads | MoE Experts | Params   | Context (tokens) |
|------------|--------|--------|-------|-------------|----------|------------------|
| 0.5B       | 12     | 1024   | 8     | 4           | ~0.5B    | 10M              |
| 1.5B       | 24     | 3072   | 32    | 16          | ~1.5B    | 10M              |
| 7B         | 32     | 4096   | 32    | 32          | ~7B      | 10M              |
| 32B        | 48     | 6656   | 52    | 64          | ~32B     | 10M              |
| 64/70B     | 80     | 8192   | 64    | 128         | ~70B     | 10M              |

- **Multimodal Integration**: CLIP ViT-L/14 (vision), AST Base (audio), LayoutLMv3 (documents) with unified embedding space
- **MoE**: Top-2 routing, efficient expert loading

---

## 📦 Datasets[Default ModelScope(https://www.modelscope.cn/)]
Datasets are automatically downloaded and cached. The following datasets are supported:

### Math & Reasoning
- **NuminaMath-CoT** (AI-ModelScope/NuminaMath-CoT): Mathematical reasoning with chain-of-thought

### Chinese Language
- **Llama3-Chinese-Dataset** (zhuangxialie/Llama3-Chinese-Dataset): Chinese language corpus
- **Chinese-DeepSeek-R1** (liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT): Chinese instruction tuning

### Web & General Knowledge
- **OpenWeb888K** (prithivMLmods/OpenWeb888K): Web crawl data

### Image Understanding
- **ShareGPT-4o-Image** (FreedomIntelligence/ShareGPT-4o-Image): Image-dialogue pairs
- **coco_captions_small_slice** (modelscope/coco_captions_small_slice): COCO image captions
- **LAION-SG** (AI-ModelScope/LAION-SG): Semantic graph dataset

### Audio Processing
- **AudioSetCaps_350k** (lmms-lab/AudioSetCaps_350k_converted): Audio captioning
- **Libri2Mix_8k** (modelscope/Libri2Mix_8k): Audio mixing dataset
- **Clotho** (OmniData/Clotho): Audio captioning

### Code & Programming
- **ultrachat_200k** (HuggingFaceH4/ultrachat_200k): Chat-based instruction tuning
- **CodeAlpaca_20K** (HuggingFaceH4/CodeAlpaca_20K): Code instruction tuning
- **codeparrot_github-code** (jablonkagroup/codeparrot_github-code-chemistry-python): Python code corpus

### Document Understanding
- **DocVQA** (swift/DocVQA): Document visual question answering
- **PubLayNet** (OpenDataLab/PubLayNet): Document layout analysis
- **VQAv2** (swift/VQAv2): Visual question answering

Datasets are auto-downloaded and cached via:
```bash
python manage.py download
```

---

## 🏆 Extreme 70B Training on 24GB GPU[Beta]
Pisces L1 supports **training/fine-tuning 70B models on a single 24GB GPU** using QLoRA, 4-bit quantization, LoRA adapters, and gradient accumulation.

#### 70B QLoRA Training Example

##### Single GPU
```bash
python manage.py train \ 
  --model_size 70B \ 
  --force_quant \ 
  --force_lora \ 
  --batch_size 1 \ 
  --accum 32 \ 
  --seq_len 512
```

##### Multi-GPU Distributed Training
```bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py \ 
  --model_size 70B \ 
  --distributed \ 
  --force_quant \ 
  --force_lora \ 
  --batch_size 1 \ 
  --accum 8
```
```bash
python manage.py train \
  --model_size 70B \
  --force_quant \
  --force_lora \
  --batch_size 1 \
  --accum 32 \
  --seq_len 512
```
- 4-bit quantization: Dramatically reduces memory ([QLoRA paper](https://arxiv.org/abs/2305.14314))
- LoRA adapters: Efficient parameter fine-tuning
- Gradient accumulation: Simulates large batch size
- Mixed precision: Further memory savings
- No accuracy loss: QLoRA+LoRA achieves near full-precision results ([QLoRA deep dive](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html))

---

## ⚡ Quick Start
After installation, try this 3-step workflow:
```bash
# 1. Download default datasets
python manage.py download

# 2. Train a small model (0.5B)
python manage.py train --model_size 0.5B

# 3. Run inference
python manage.py infer --prompt "Explain quantum computing in simple terms" --ckpt ckpt/latest.pt
```

## ❓ FAQ
- **Q: How do I see all available commands?**  
  A: `python manage.py help`
- **Q: How do I add a new dataset?**  
  A: Add the dataset name to the `DATASETS` list in `data/download.py` and rerun `python manage.py download`. For custom datasets, the format should be JSONL with a `text` field or Parquet with `input_ids` and `labels` columns.
- **Q: Getting out-of-memory errors?**  
  A: Reduce batch size with `--batch_size`, enable quantization with `--force_quant`, or use a smaller model size.
- **Q: How to resume training?**  
  A: Use `--resume_ckpt path/to/checkpoint.pt` to continue from a saved checkpoint.
- **Q: Where are model configs?**  
  A: See `configs/` directory for all model sizes.
- **Q: How to run on CPU only?**  
  A: Most features require GPU, but you can try with `--device cpu` (performance will be slow).

---

## 📄 License
Pisces L1 is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). **Commercial use is strictly prohibited.** See [LICENSE](./LICENSE) for details.

### License Summary
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **Non-Commercial**: You may not use the material for commercial purposes.
- **No additional restrictions**: You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

---

## 🌏 Community & Citation
- Issues & PRs welcome!
- [PiscesL1 in Gitee](https://gitee.com/dunimd/piscesl1.git)
- [PiscesL1 in GitHub](https://github.com/mf2023/PiscesL1.git)
- [PiscesL1 in ModelScope](https://www.modelscope.cn/models/mfchina2024/PiscesL1)

---

*Happy experimenting with Pisces L1!*
