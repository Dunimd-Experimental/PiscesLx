# ⚠️ Compliance Notice

**If you use this model in China, including but not limited to training, fine tuning, commercial testing, etc., if you provide any services to the public, please first complete the filing procedures in accordance with relevant laws and regulations.**

---

# Pisces L1

English | [简体中文](README.zh.md)

A next-generation lightweight multimodal Mixture-of-Experts (MoE) model supporting text, image, audio, and document understanding. Designed for both research and practical applications, Pisces L1 can run on a single RTX 4090 and scale up to 314B parameters with advanced memory optimization.

---

## 🚀 Features

- **Multimodal**: Unified support for text, image, audio, video, document, and agent behavior inputs
- **MoE Architecture**: Efficient Mixture-of-Experts, scalable from 0.5B to 314B parameters
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
| source     | Activate virtual environment                                         |
| update       | Pull latest code from remote repository                              |
| train      | Train the model                                                      |
| infer      | Run inference with a trained model                                   |
| check      | Check GPU and dependencies                                           |
| monitor    | System monitor (GPU/CPU/memory)                                      |
| download   | Download datasets for training                                       |
| dataset    | Arrow/JSON dataset conversion                                        |
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
| 0.5B       | 10     | 896    | 8     | 4           | ~0.5B    | 256K            |
| 1.5B       | 14     | 1536   | 24    | 8           | ~1.5B    | 256K            |
| 7B         | 28     | 3584   | 32    | 16          | ~7B      | 1M              |
| 32B        | 40     | 5120   | 64    | 32          | ~32B     | 1M              |
| 64/70B     | 48     | 8192   | 64    | 64          | ~70B     | 10M             |
| 128B       | 120    | 10240  | 80    | 64          | ~128B    | 10M             |
| 314B       | 160    | 12288  | 96    | 16          | ~314B    | 10M             |

- **Multimodal Integration**: CLIP ViT-L/14 (vision), AST Base (audio), Video Encoder (temporal), LayoutLMv3 (documents) with unified embedding space
- **MoE**: Top-2 routing, efficient expert loading

---

## 📦 Datasets[Default ModelScope(https://www.modelscope.cn/)]
Datasets are automatically downloaded and cached. The following datasets are supported:

### Chinese Language
- **Chinese1** (baicai003/Llama3-Chinese-dataset): Chinese language corpus
- **Chinese2** (liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT): Chinese instruction tuning
- **Chinese3** (AI-ModelScope/OpenOrca-Chinese): Chinese instruction data
- **Chinese4** (AI-ModelScope/ultrachat_200k): Chinese dialogue data

### English Language
- **English1** (YorickHe/CoT): Chain-of-thought reasoning data
- **English2** (DAMO_ConvAI/EnDoc2BotDialogue): English dialogue dataset
- **English3** (Intelligent-Internet/wikipedia_en): Wikipedia English corpus

### Math & Reasoning
- **Math1** (swift/MetaMathQA): Mathematical reasoning dataset
- **Math2** (AI-MO/NuminaMath-CoT): Mathematical reasoning with chain-of-thought
- **Math3** (AI-ModelScope/NuminaMath-CoT): Mathematical problem solving
- **Math4** (xpengx/EleutherAI-proof-pile-2): Mathematical proof data
- **Math5** (tastelikefeet/competition_math): Competition mathematics

### Code & Programming
- **Code1** (HuggingFaceH4/CodeAlpaca_20K): Code instruction tuning
- **Code2** (jablonkagroup/codeparrot_github-code-chemistry-python): Python code corpus
- **Code3** (jablonkagroup/codeparrot_github-code-chemistry-python): Additional Python code
- **Code4** (codefuse-ai/CodeExercise-Python-27k): Python exercise dataset

### Web & General Knowledge
- **Web1** (AI-ModelScope/webvid-10M): Web video data
- **Web2** (prithivMLmods/OpenWeb888K): Web crawl data
- **Web3** (OmniData/Pile-OpenWebText2): Web text corpus

### Audio Processing
- **Audio1** (OmniData/Clotho): Audio captioning dataset
- **Audio2** (modelscope/Libri2Mix_8k): Audio mixing dataset
- **Audio3** (lmms-lab/AudioSetCaps_350k_converted): Audio captioning

### Image Understanding
- **Image1** (modelscope/coco_captions_small_slice): COCO image captions
- **Image2** (FreedomIntelligence/ShareGPT-4o-Image): Image-dialogue pairs

### Document & Visual Understanding
- **VQAv2** (swift/VQAv2): Visual question answering
- **FinQA** (OmniData/FinQA): Financial question answering
- **DocVQA** (swift/DocVQA): Document visual question answering
- **Exam** (modelscope/ceval-exam): Chinese exam questions
- **SG1** (AI-ModelScope/LAION-SG): Semantic graph dataset
- **Chat1** (HuggingFaceH4/ultrachat_200k): Chat-based instruction tuning
- **PubLayNet1** (OpenDataLab/PubLayNet): Document layout analysis
- **Medical1** (krisfu/delicate_medical_r1_data): Medical instruction data
- **Financial1** (BJQW14B/bs_challenge_financial_14b_dataset): Financial dataset

### Agent & Behavior Understanding
- **Agent1** (AI-ModelScope/agent-instruct): Agent instruction tuning dataset
- **Agent2** (OmniData/agent-dialogue): Multi-turn agent conversation data
- **Agent3** (swift/agent-reasoning): Agent reasoning and planning tasks
- **Agent4** (HuggingFaceH4/agent-tool-use): Tool-using agent behavior dataset
- **Agent5** (modelscope/agent-environment): Agent-environment interaction data

Datasets are auto-downloaded and cached via:
```bash
python manage.py download
```

---

## 🏆 Training 1.5B Model on 24GB GPU [Beta]
Pisces L1 supports **training/fine-tuning 1.5B models on a single 24GB GPU** using QLoRA, 4-bit quantization, LoRA adapters, and gradient accumulation.

#### 1.5B QLoRA Training Example

##### Single GPU
```bash
python manage.py train --model_size 1.5B --resume_ckpt checkpoint.pt --reset_lr
```

##### Resume Training
- 4-bit quantization: Dramatically reduces memory ([QLoRA paper](https://arxiv.org/abs/2305.14314))
- LoRA adapters: Efficient parameter fine-tuning
- Gradient accumulation: Simulates large batch size
- Mixed precision: Further memory savings
- No accuracy loss: QLoRA+LoRA achieves near full-precision results ([QLoRA deep dive](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html))

---

## ⚡ Quick Start
After installation, try this 6-step workflow:
```bash
# 1. Environment setup
python manage.py setup

# 2. Activate environment
python manage.py source

# 3. Pull latest code (optional)
python manage.py pull

# 4. Download default datasets
python manage.py download

# 5. Train a small model (0.5B)
python manage.py train --model_size 0.5B

# 6. Run inference
python manage.py infer --prompt "Explain quantum computing in simple terms" --ckpt ckpt/latest.pt
```

## 🤖 MCP Native Agent Support [Beta]
Pisces L1 now includes **native MCP (Multi-Agent Communication Protocol)** support, enabling seamless agent-to-agent communication and distributed task execution.

### MCP Features
- **Native Integration**: Built-in MCP protocol support in `model/agent.py`
- **Async Communication**: All agent methods support async/await
- **Capability Discovery**: Dynamic registration and discovery of agent capabilities
- **Multimodal MCP**: Full support for text, image, audio via MCP protocol
- **Zero Dependencies**: No additional libraries required

### Quick MCP Usage
```python
import asyncio
from model.agent import PiscesAgent

async def main():
    # Create MCP-native agent
    agent = PiscesAgent(agent_id="agent_001")
    
    # Register capabilities via MCP
    async def web_search(query: str):
        return {"results": ["result1", "result2"]}
    
    await agent.register_capability(
        name="web_search",
        description="Search the web",
        parameters={"query": str},
        handler=web_search
    )
    
    # Run via MCP protocol
    result = await agent.run(task="Search AI news")

asyncio.run(main())
```

### MCP Commands
```bash
# MCP agent CLI
python manage.py agent --mcp-mode

# Discover peer capabilities
python manage.py agent --discover-peers

# Sync with MCP hub
python manage.py agent --connect-hub http://localhost:8080
```

## 🎯 Model Evaluation & Benchmarking
Pisces L1 includes comprehensive benchmarking support for 26 standardized evaluation benchmarks covering Chinese, English, mathematics, coding, and reasoning tasks.

### Available Benchmarks

#### Core Benchmarks
| Benchmark | Language | Focus Areas | Questions |
|-----------|----------|-------------|-----------|
| **MMLU** | English | 57 disciplines (STEM/Humanities/Social Sciences) | Multi-choice |
| **C-Eval** | Chinese | 52 Chinese disciplines | Multi-choice |
| **C-Eval Hard** | Chinese | College/postgraduate entrance exams | Hard |
| **SuperCLUE** | Chinese | General, reasoning, agent, hard tasks | Mixed |
| **SuperBench** | Chinese/English | Semantics, alignment, code, agent, safety | 32 tasks |
| **OpenCompass 2.0** | Chinese/English | Language, knowledge, reasoning, math, code, agent | 15k questions |

#### Code & Programming
| Benchmark | Language | Focus | Problems |
|-----------|----------|--------|----------|
| **HumanEval** | English | Python function completion | 164 |
| **MBPP** | English | Basic Python programming | 974 |
| **LiveCodeBench v5** | English | Real-time programming contests | Continuous |
| **MBPP-Plus** | English | Advanced programming | 974 |
| **DS-1000** | English | Data science code | 1000 |
| **CRUXEval** | English | Code execution & reasoning | 800 |

#### Mathematics & Reasoning
| Benchmark | Language | Focus | Problems |
|-----------|----------|--------|----------|
| **GSM8K** | English | Elementary math word problems | 8500 |
| **AIME 2024-2025** | English | High school math competition | 15 |
| **CMATH** | Chinese | Chinese math (elementary to high school) | 5800 |
| **BBH** | English | 23 advanced reasoning tasks | 6500 |
| **DROP** | English | Reading comprehension + numerical reasoning | 96k |

#### Chinese Language
| Benchmark | Language | Focus | Problems |
|-----------|----------|--------|----------|
| **CMMLU** | Chinese | 67 Chinese disciplines | 11.7k |
| **AGI-Eval** | Chinese/English | College entrance, postgraduate, law, CPA | 8.1k |

#### Evaluation & Safety
| Benchmark | Language | Focus | Problems |
|-----------|----------|--------|----------|
| **HellaSwag** | English | Commonsense reasoning | Sentence completion |
| **ARC-Challenge** | English | Scientific reasoning | Multi-choice |
| **MT-Bench** | Multi-turn | 8-turn conversation evaluation | 80 |
| **IFEval** | English | Instruction following | 541 |
| **TruthfulQA** | English | Factuality & hallucination | 817 |
| **SafetyBench** | Chinese/English | Safety alignment | 11k |
| **Chatbot Arena** | Multi-turn | Human blind testing (Elo) | Real-time |

### Usage Examples

#### List all benchmarks
```bash
python manage.py benchmark --list
```

#### Get benchmark details
```bash
python manage.py benchmark --info mmlu
```

#### Run performance benchmark
```bash
python manage.py benchmark --perf --config configs/7B.json --seq_len 4096
```

#### Run specific benchmark
```bash
python manage.py benchmark --benchmark mmlu --config configs/7B.json
```

## ❓ FAQ
- **Q: How do I see all available commands?**  
  A: `python manage.py help`
- **Q: How do I add a new dataset?**  
  A: Add the dataset name to the `DATASETS` list in `data/download.py` and rerun `python manage.py download`. For custom datasets, the format should be JSONL with a `text` field or Parquet with `input_ids` and `labels` columns.
- **Q: Getting out-of-memory errors?**  
  A: Use a smaller model size, reduce sequence length, or enable 4-bit quantization via `python manage.py quantize` before training.
- **Q: How to resume training?**  
  A: Use `--resume_ckpt path/to/checkpoint.pt` to continue from a saved checkpoint.
- **Q: Where are model configs?**  
  A: See `configs/` directory for all model sizes.
- **Q: How to run on CPU only?**  
  A: Most features require GPU, but you can try with `--device cpu` (performance will be slow).
- **Q: How to run model evaluation?**  
  A: Use `python tools/benchmark.py` with available benchmarks. See the Model Evaluation section above.

---

## 📄 License
This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

### License Summary
- **Commercial Use**: ✅ Permitted
- **Modification**: ✅ Permitted  
- **Distribution**: ✅ Permitted
- **Attribution**: ✅ Required
- **Patent Grant**: ✅ Included

---

## 🌏 Community & Citation
- Issues & PRs welcome!
- [PiscesL1 in Gitee](https://gitee.com/dunimd/piscesl1.git)
- [PiscesL1 in GitHub](https://github.com/mf2023/PiscesL1.git)
- [PiscesL1 in ModelScope](https://www.modelscope.cn/models/mfchina2024/PiscesL1)

<h3 align="center">Where intuition navigates the depths of data</h3>
<h3 align="center">And empathy gives form to intelligence</h3>

![summary](./icons/PD.png)