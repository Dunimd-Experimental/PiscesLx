# Pisces L1

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
| Command    | Description                                 |
|------------|---------------------------------------------|
| setup      | Environment setup and dependency install    |
| train      | Train the model                             |
| infer      | Run inference with a trained model          |
| check      | Check GPU and dependencies                  |
| monitor    | System monitor (GPU/CPU/memory)             |
| download   | Download datasets for training              |
| arrow      | Arrow/JSON dataset conversion               |
| help       | Show help message                           |

#### Example
```bash
python manage.py download
python manage.py train
python manage.py infer --ckpt ckpt/model.pt --prompt "Hello, Pisces!"
```

---

## 🧠 Model Architecture & Configurations

| Model Size | Layers | Hidden | Heads | MoE Experts | Params   | Context |
|------------|--------|--------|-------|-------------|----------|---------|
| 0.5B       | 12     | 1024   | 8     | 4           | ~0.5B    | 2K      |
| 1.5B       | 24     | 3072   | 32    | 16          | ~1.5B    | 4K      |
| 7B         | 32     | 4096   | 32    | 32          | ~7B      | 8K      |
| 32B        | 48     | 6656   | 52    | 64          | ~32B     | 16K     |
| 64/70B     | 80     | 8192   | 64    | 128         | ~70B     | 32K     |

- **Multimodal**: CLIP (vision), AST (audio), LayoutLMv3 (documents)
- **MoE**: Top-2 routing, efficient expert loading

---

## 📦 Datasets
- **TinyStories**: Text generation
- **COCO Few-shot**: Image-text few-shot learning
- **AudioSet Captions**: Audio caption generation

Datasets are auto-downloaded and cached via:
```bash
python manage.py download
```

---

## 🏆 Extreme 70B Training on 24GB GPU
Pisces L1 supports **training/fine-tuning 70B models on a single 24GB GPU** using QLoRA, 4-bit quantization, LoRA adapters, and gradient accumulation.

#### 70B QLoRA Training Example
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

## ❓ FAQ
- **Q: How do I see all available commands?**  
  A: `python manage.py help`
- **Q: How do I add a new dataset?**  
  A: Add its name to `data_cache/model.txt` and rerun `python manage.py download`.
- **Q: Where are model configs?**  
  A: See `configs/` directory for all model sizes.
- **Q: How to run on CPU only?**  
  A: Most features require GPU, but you can try with `--device cpu` (performance will be slow).

---

## 📄 License
Pisces L1 is released under the GNU Affero General Public License v3.0. See [LICENSE](./LICENSE) for details.

---

## 🌏 Community & Citation
- Issues & PRs welcome!
- For academic use, please cite the project if it helps your research.

---

*Happy experimenting with Pisces L1!*
