# ⚠️ Compliance Notice

**If you use this model in China, including but not limited to training, fine tuning, commercial testing, etc., if you provide any services to the public, please first complete the filing procedures in accordance with relevant laws and regulations.**

---

# Pisces L1

English | [简体中文](README.zh.md)

A next-generation lightweight multimodal Mixture-of-Experts (MoE) model with **Arctic Architecture**, supporting text, image, audio, video, document, and agent understanding. Designed for both research and practical applications, Pisces L1 (PiscesLx Series by Dunimd Project Group) can run on a single RTX 4090 and scale up to 314B parameters with breakthrough innovations.

## ❄️ Arctic Architecture Innovations

### 🧠 Quantum Reasoning Engine (PiscesReasoner)
- **Hierarchical Reasoning Chains (HRC)**: Multi-layer abstraction processing
- **Quantum Superposition Thinking**: 8 parallel hypothesis streams with quantum collapse
- **Dynamic Fact Verification**: Real-time truth checking and consistency scoring  
- **Meta-cognitive Reflection**: Self-awareness of reasoning process with uncertainty quantification
- **Quantum Special Tokens**: `<start_hypothesis>`, `<start_evidence>`, `<start_conclusion>`, `<quantum_merge>`

### 🔧 MoE Expert System
- **8 Expert Top-2 Routing**: Intelligent load balancing with StableMoEGate
- **LSTM Load Prediction**: Dynamic capacity adjustment and expert allocation
- **Gradient Checkpoint Compatible**: Fixed shape mode for memory efficiency
- **Stable Expert Gates**: Advanced routing with noise injection and capacity control

### 🌐 5-Modal Encoding System
- **VisionEncoder**: NaViT native resolution support (up to 1024px)
- **VideoEncoder**: Temporal visual understanding with frame-level attention
- **AudioEncoder**: Advanced audio feature extraction and processing
- **DocEncoder**: Document structure understanding with LayoutLMv3 integration
- **AgentEncoder**: Comprehensive agent behavior modeling (observations, actions, reflections)

### ⚛️ Quantum Entangled Fusion
- **DynamicModalFusion**: Advanced cross-modal attention with quantum entanglement
- **Tensor Network Compression**: 6-pair correlation networks for modal interactions
- **Hardware Adaptive Configuration**: Intelligent hardware detection and optimization
- **Quality-aware Fusion Gates**: Adaptive fusion based on content quality

### 📏 Ultra-Long Context System
- **YaRN RoPE**: 10M+ token support with dynamic NTK scaling
- **H2O Attention**: Streaming attention for 128B+ parameter models
- **Dynamic Position Encoding**: Adaptive position encoding with long factor scaling
- **Memory-efficient Context**: Sliding window and compression techniques

### 🤖 Advanced Agent System
- **PiscesAgent**: Native multimodal agent with MCP protocol support
- **Tool Integration**: Built-in tool use capabilities and environment interaction
- **Persistent Memory**: Context management and experience accumulation
- **MCP Communication**: Multi-Agent Communication Protocol for distributed reasoning

### 🎯 K-FAC Optimization
- **Second-order Optimization**: Diagonal Fisher matrix approximation
- **Memory-efficient Implementation**: 99%+ memory reduction vs full K-FAC
- **Curvature-aware Training**: Natural gradient computation for faster convergence
- **Adaptive Gradient Clipping**: Dynamic threshold adjustment based on training history

---

## 🚀 Features

- **Arctic Architecture**: Revolutionary multimodal architecture with quantum-inspired reasoning
- **Quantum Reasoning Engine**: Beyond traditional Chain-of-Thought with hierarchical abstraction
- **5-Modal Understanding**: Unified processing of text, image, audio, video, document, and agent inputs
- **MoE Expert System**: 8 experts with intelligent Top-2 routing and load prediction
- **Ultra-Long Context**: Up to 10M+ tokens with YaRN RoPE and H2O attention
- **Advanced Quantization**: 2-bit, 4-bit, 8-bit quantization options with gradient stability
- **Memory Optimization**: Runs 0.5B model on 14.58GB GPU with QLoRA + gradient checkpointing
- **K-FAC Optimization**: Second-order optimization with diagonal Fisher matrix approximation
- **Native Agent Support**: Built-in MCP protocol and tool integration
- **One-command Workflow**: Complete management via `python manage.py`

---

## 🛠️ Installation & Environment

- **Python**: 3.11 recommended
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
| update     | Pull latest code from remote repository                              |
| version    | Show current version and changelog                                   |
| changelog  | Show version history (--all for all, --version X.X.XXXX for specific) |
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
python manage.py version          # Show current version
python manage.py changelog --all    # Show all versions
python manage.py changelog --version 1.0.0150  # Show specific version
python manage.py train
python manage.py infer --ckpt ckpt/model.pt --prompt "Hello, Pisces!"
```

---

## 🧠 Model Architecture & Configurations

### Arctic Architecture Components
- **Core Transformer**: RMSNorm, YaRN RoPE, Grouped-Query Attention
- **Quantum Reasoning**: 4-layer hierarchical abstraction with meta-cognitive reflection
- **Multimodal Fusion**: Quantum entangled cross-modal attention with tensor networks
- **MoE System**: Dynamic expert routing with LSTM load prediction
- **Memory Optimization**: Gradient checkpointing, mixed precision, K-FAC optimization

| Model Size | Layers | Hidden | Heads | KV Heads | MoE Experts | Params (Actual) | Context | Quantization |
|------------|--------|--------|-------|----------|-------------|-----------------|---------|-------------|
| 0.5B       | 16     | 640    | 10    | 5        | 6           | 0.5B            | 256K    | 2/4/8-bit   |
| 1.5B       | 16     | 896    | 14    | 7        | 6           | 1.5B            | 256K    | 2/4/8-bit   |
| 7B         | 28     | 3584   | 32    | 8        | 8           | 7B              | 1M      | 2/4/8-bit   |
| 32B        | 64     | 5120   | 40    | 8        | 8           | 32B             | 1M      | 2/4/8-bit   |
| 64B        | 80     | 6656   | 52    | 8        | 8           | 64B             | 10M     | 2/4/8-bit   |
| 70B        | 80     | 8192   | 64    | 8        | 8           | 70B             | 10M     | 2/4/8-bit   |
| 128B       | 120    | 10240  | 80    | 8        | 8           | 128B            | 10M     | 2/4/8-bit   |
| 314B       | 160    | 12288  | 96    | 12       | 16          | 314B            | 10M     | 2/4/8-bit   |

### Parameter Breakdown (0.5B Configuration)
- **Core Transformer**: ~500M parameters
- **Multimodal Encoders**: Optimized and balanced
  - VisionEncoder: ~120M, VideoEncoder: ~150M, AudioEncoder: ~80M
  - DocEncoder: ~80M, AgentEncoder: ~70M
- **Quantum Reasoning Engine**: Integrated within core parameters
- **Modal Fusion System**: Lightweight design for efficiency

### Quantization Options
```bash
# 2-bit quantization (experimental, maximum memory saving)
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 2

# 4-bit quantization (default, balanced performance)
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 4

# 8-bit quantization (stable, minimal quality loss)
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 8
```

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

## 🏆 Advanced Training on 14.58GB GPU
Pisces L1 Arctic architecture supports **training 1.5B parameter models on 14.58GB GPU** using advanced quantization, LoRA, and memory optimization techniques.

### Memory Optimization Strategies
- **Multi-bit Quantization**: 2-bit (experimental), 4-bit (default), 8-bit (stable)
- **LoRA Adaptation**: Only 0.024% parameters trainable (360K out of 1.5B)
- **Gradient Checkpointing**: Reduces activation memory by 50%+
- **K-FAC Optimization**: Diagonal Fisher matrix approximation
- **Adaptive Gradient Clipping**: Handles gradient explosion automatically

### Training Examples

#### 1.5B Model with 4-bit Quantization
```bash
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --force_lora
```

#### 1.5B Model with 8-bit for Stability
```bash
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 8 --force_lora
```

#### Memory Usage Comparison
| Configuration | Memory Usage | Trainable % | Gradient Stability |
|---------------|--------------|-------------|--------------------|
| Full Precision | >40GB | 100% | Stable |
| 8-bit + LoRA | ~18GB | 0.024% | Very Stable |
| 4-bit + LoRA | ~14.5GB | 0.024% | Manageable |
| 2-bit + LoRA | ~11GB | 0.024% | Experimental |

### Training Performance
- **Loss Convergence**: 35.38 → 31.62 in 140 steps (10.6% improvement)
- **Gradient Clipping**: Automatic handling of 280K+ gradient norms
- **Memory Efficiency**: Stable training on 14.58GB GPU
- **Speed**: Proportional to model complexity (1.5B slower than 0.5B as expected)

---

## ⚡ Quick Start
After installation, try this 6-step workflow:
```bash
# 1. Environment setup
python manage.py setup

# 2. Activate environment
python manage.py source

# 3. Pull latest code (optional)
python manage.py update

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

<h3 align="center">Where intuition navigates the depths of data And empathy gives form to intelligence</h3>

![summary](./icons/PD.png)