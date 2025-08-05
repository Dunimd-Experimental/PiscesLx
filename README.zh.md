# ⚠️ 合规提示

**如果你在中国境内使用本模型，包括但不限于训练、微调、商用测试等，凡是向社会公众提供任何服务的，请先按照相关法律法规完成备案手续。**

---


# Pisces L1

简体中文 | [English](README.md)

下一代轻量级多模态混合专家模型(Mixture-of-Experts, MoE)，支持文本、图像、音频和文档理解。Pisces L1专为研究设计，可在单个RTX 4090上运行，并通过高级内存优化扩展至70B参数。

---

## 🚀 特性

- **多模态**: 统一支持文本、图像、音频和文档输入
- **MoE架构**: 高效的混合专家模型，参数从0.5B到70B可扩展
- **轻量级**: 0.5B基础模型可在消费级GPU(24GB VRAM)上运行
- **现代Transformer**: RMSNorm、RoPE、分组查询注意力等
- **极致适应性**: QLoRA、4位量化、LoRA适配器、梯度累积
- **一键式工作流**: 所有管理通过`python manage.py`完成(见下文)

---

## 🛠️ 安装与环境

- **Python**: 推荐3.9–3.11
- **CUDA**: 11.8+ (用于GPU训练/推理)
- **依赖项**: 所有必需的包都列在`requirements.txt`中

### 快速设置
```bash
git clone https://gitee.com/dunimd/piscesl1.git or git clone https://github.com/mf2023/PiscesL1.git
cd piscesl1
python manage.py setup
```
这将自动创建虚拟环境并安装所有依赖项。

---

## ⚡ 命令行使用

所有命令通过`python manage.py <command>`管理。如需帮助：
```bash
python manage.py help
```

### 主要命令
| 命令         | 描述                                          |
|-------------|----------------------------------------------|
| setup       | 环境设置和依赖安装                              |
| source      | 激活虚拟环境                                   |
| pull        | 从远程仓库拉取最新代码                           |
| train       | 训练模型                                       |
| infer       | 使用训练好的模型进行推理                         |
| check       | 检查GPU和依赖项                                |
| monitor     | 系统监控(GPU/CPU/内存)                         |
| download    | 下载训练数据集                                 |
| arrow       | Arrow/JSON数据集转换                           |
| quantize    | 将模型量化为4/8位以提高效率                      |
| benchmark   | 运行性能基准测试                                |
| help        | 显示帮助信息                                   |

#### 示例
```bash
python manage.py download
python manage.py train
python manage.py infer --ckpt ckpt/model.pt --prompt "你好，Pisces!"
```

---

## 🧠 模型架构与配置

| 模型大小 | 层数 | 隐藏层大小 | 注意力头数 | MoE专家数 | 参数规模 | 上下文长度( tokens) |
|---------|------|-----------|-----------|----------|---------|-------------------|
| 0.5B    | 12   | 1024      | 8         | 4        | ~0.5B   | 1M                |
| 1.5B    | 24   | 3072      | 32        | 16       | ~1.5B   | 1M                |
| 7B      | 32   | 4096      | 32        | 32       | ~7B     | 1M                |
| 32B     | 48   | 6656      | 52        | 64       | ~32B    | 10M               |
| 64/70B  | 80   | 8192      | 64        | 128      | ~70B    | 10M               |

- **多模态集成**: CLIP ViT-L/14(视觉)、AST Base(音频)、LayoutLMv3(文档)，具有统一的嵌入空间
- **MoE**: Top-2路由，高效专家加载

---

## 📦 数据集[默认魔塔社区(https://www.modelscope.cn/)]
数据集会自动下载并缓存。支持以下数据集：

### 中文语言
- **Chinese1** (baicai003/Llama3-Chinese-dataset): 中文语料库
- **Chinese2** (liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT): 中文指令微调
- **Chinese3** (AI-ModelScope/OpenOrca-Chinese): 中文指令数据
- **Chinese4** (AI-ModelScope/ultrachat_200k): 中文对话数据

### 英文语言
- **English1** (YorickHe/CoT): 思维链推理数据
- **English2** (DAMO_ConvAI/EnDoc2BotDialogue): 英文对话数据集
- **English3** (Intelligent-Internet/wikipedia_en): 维基百科英文语料库

### 数学与推理
- **Math1** (swift/MetaMathQA): 数学推理数据集
- **Math2** (AI-MO/NuminaMath-CoT): 带思维链的数学推理
- **Math3** (AI-ModelScope/NuminaMath-CoT): 数学问题解决
- **Math4** (xpengx/EleutherAI-proof-pile-2): 数学证明数据
- **Math5** (tastelikefeet/competition_math): 竞赛数学

### 代码与编程
- **Code1** (HuggingFaceH4/CodeAlpaca_20K): 代码指令微调
- **Code2** (jablonkagroup/codeparrot_github-code-chemistry-python): Python代码语料库
- **Code3** (jablonkagroup/codeparrot_github-code-chemistry-python): 额外Python代码
- **Code4** (codefuse-ai/CodeExercise-Python-27k): Python练习数据集

### 网络与通用知识
- **Web1** (AI-ModelScope/webvid-10M): 网络视频数据
- **Web2** (prithivMLmods/OpenWeb888K): 网络爬取数据
- **Web3** (OmniData/Pile-OpenWebText2): 网络文本语料库

### 音频处理
- **Audio1** (OmniData/Clotho): 音频标题生成数据集
- **Audio2** (modelscope/Libri2Mix_8k): 音频混合数据集
- **Audio3** (lmms-lab/AudioSetCaps_350k_converted): 音频标题生成

### 图像理解
- **Image1** (modelscope/coco_captions_small_slice): COCO图像标题
- **Image2** (FreedomIntelligence/ShareGPT-4o-Image): 图像对话对

### 文档与视觉理解
- **VQAv2** (swift/VQAv2): 视觉问答
- **FinQA** (OmniData/FinQA): 金融问答
- **DocVQA** (swift/DocVQA): 文档视觉问答
- **Exam** (modelscope/ceval-exam): 中文考试题
- **SG1** (AI-ModelScope/LAION-SG): 语义图数据集
- **Chat1** (HuggingFaceH4/ultrachat_200k): 基于对话的指令微调
- **PubLayNet1** (OpenDataLab/PubLayNet): 文档布局分析
- **Medical1** (krisfu/delicate_medical_r1_data): 医疗指令数据
- **Financial1** (BJQW14B/bs_challenge_financial_14b_dataset): 金融数据集

通过以下命令自动下载和缓存数据集：
```bash
python manage.py download
```

---

## 🏆 在24GB GPU上训练7B模型 [Beta]
Pisces L1支持**在单个24GB GPU上训练/微调7B模型**，使用QLoRA、4位量化、LoRA适配器和梯度累积技术。

#### 7B QLoRA训练示例

##### 单GPU训练
```bash
python manage.py train --model_size 7B --resume_ckpt latest.pt
```

##### 继续训练
```bash
python manage.py train --model_size 70B --resume_ckpt checkpoint.pt --reset_lr
```
- 4位量化：显著减少内存占用([QLoRA论文](https://arxiv.org/abs/2305.14314))
- LoRA适配器：高效参数微调
- 梯度累积：模拟大批次训练
- 混合精度：进一步节省内存
- 无精度损失：QLoRA+LoRA达到接近全精度的结果([QLoRA深度解析](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html))

---

## ⚡ 快速开始

安装完成后，试试这个6步工作流程：
```bash
# 1. 环境设置
python manage.py setup

# 2. 激活环境
python manage.py source

# 3. 拉取最新代码(可选)
python manage.py pull

# 4. 下载默认数据集
python manage.py download

# 5. 训练小模型(0.5B)
python manage.py train --model_size 0.5B

# 6. 运行推理
python manage.py infer --prompt "用简单的话解释量子计算" --ckpt ckpt/latest.pt
```

## 🤖 MCP原生智能体支持 [Beta]
Pisces L1现已包含**原生MCP(多智能体通信协议)**支持，实现智能体间无缝通信和分布式任务执行。

### MCP特性
- **原生集成**: 在`model/agent.py`中内置MCP协议支持
- **异步通信**: 所有智能体方法支持async/await
- **能力发现**: 动态注册和发现智能体能力
- **多模态MCP**: 通过MCP协议全面支持文本、图像、音频
- **零依赖**: 无需额外库

### MCP快速使用
```python
import asyncio
from model.agent import PiscesAgent

async def main():
    # 创建MCP原生智能体
    agent = PiscesAgent(agent_id="agent_001")
    
    # 通过MCP注册能力
    async def web_search(query: str):
        return {"results": ["结果1", "结果2"]}
    
    await agent.register_capability(
        name="web_search",
        description="搜索网络",
        parameters={"query": str},
        handler=web_search
    )
    
    # 通过MCP协议运行任务
    result = await agent.run(task="搜索AI最新进展")

asyncio.run(main())
```

### MCP命令
```bash
# MCP智能体CLI
python manage.py agent --mcp-mode

# 发现对等能力
python manage.py agent --discover-peers

# 与MCP中心同步
python manage.py agent --connect-hub http://localhost:8080
```

## 🎯 模型评测与基准测试
Pisces L1包含对26个标准化评测基准的全面支持，涵盖中文、英文、数学、编程和推理任务。

### 可用基准测试

#### 核心基准
| 基准测试 | 语言 | 关注领域 | 题目数量 |
|----------|------|----------|----------|
| **MMLU** | 英文 | 57个学科(STEM/人文/社会科学) | 多选题 |
| **C-Eval** | 中文 | 52个中文学科 | 多选题 |
| **C-Eval Hard** | 中文 | 高考/研究生入学考试 | 高难度 |
| **SuperCLUE** | 中文 | 通用、推理、智能体、高难度任务 | 混合 |
| **SuperBench** | 中文/英文 | 语义、对齐、代码、智能体、安全 | 32个任务 |
| **OpenCompass 2.0** | 中文/英文 | 语言、知识、推理、数学、代码、智能体 | 1.5万题 |

#### 代码与编程
| 基准测试 | 语言 | 关注重点 | 题目数量 |
|----------|------|----------|----------|
| **HumanEval** | 英文 | Python函数补全 | 164 |
| **MBPP** | 英文 | 基础Python编程 | 974 |
| **LiveCodeBench v5** | 英文 | 实时编程竞赛 | 持续更新 |
| **MBPP-Plus** | 英文 | 高级编程 | 974 |
| **DS-1000** | 英文 | 数据科学代码 | 1000 |
| **CRUXEval** | 英文 | 代码执行与推理 | 800 |

#### 数学与推理
| 基准测试 | 语言 | 关注重点 | 题目数量 |
|----------|------|----------|----------|
| **GSM8K** | 英文 | 小学数学文字题 | 8500 |
| **AIME 2024-2025** | 英文 | 高中数学竞赛 | 15 |
| **CMATH** | 中文 | 中文数学(小学到高中) | 5800 |
| **BBH** | 英文 | 23个高级推理任务 | 6500 |
| **DROP** | 英文 | 阅读理解+数值推理 | 9.6万 |

#### 中文语言
| 基准测试 | 语言 | 关注重点 | 题目数量 |
|----------|------|----------|----------|
| **CMMLU** | 中文 | 67个中文学科 | 1.17万 |
| **AGI-Eval** | 中文/英文 | 高考、研究生、法律、CPA | 8100 |

#### 评测与安全
| 基准测试 | 语言 | 关注重点 | 题目数量 |
|----------|------|----------|----------|
| **HellaSwag** | 英文 | 常识推理 | 句子补全 |
| **ARC-Challenge** | 英文 | 科学推理 | 多选题 |
| **MT-Bench** | 多轮对话 | 8轮对话评估 | 80 |
| **IFEval** | 英文 | 指令遵循 | 541 |
| **TruthfulQA** | 英文 | 事实性与幻觉 | 817 |
| **SafetyBench** | 中文/英文 | 安全对齐 | 1.1万 |
| **Chatbot Arena** | 多轮对话 | 人工盲测(Elo评分) | 实时 |

### 使用示例

#### 列出所有基准测试
```bash
python manage.py benchmark --list
```

#### 获取基准测试详情
```bash
python manage.py benchmark --info mmlu
```

#### 运行性能基准测试
```bash
python manage.py benchmark --perf --config configs/7B.json --seq_len 4096
```

#### 运行特定基准测试
```bash
python manage.py benchmark --benchmark mmlu --config configs/7B.json
```

## ❓ 常见问题
- **问：如何查看所有可用命令？**
  答：`python manage.py help`
- **问：如何添加新数据集？**
  答：将数据集名称添加到`data/download.py`的`DATASETS`列表中，并重新运行`python manage.py download`。对于自定义数据集，格式应为带`text`字段的JSONL或带`input_ids`和`labels`列的Parquet。
- **问：遇到内存不足错误？**
  答：使用更小的模型尺寸，减少序列长度，或通过`python manage.py quantize`在训练前启用4位量化。
- **问：如何恢复训练？**
  答：使用`--resume_ckpt path/to/checkpoint.pt`从保存的检查点继续训练。
- **问：模型配置文件在哪里？**
  答：参见`configs/`目录下的所有模型尺寸配置。
- **问：如何仅在CPU上运行？**
  答：大多数功能需要GPU，但可以尝试使用`--device cpu`(性能会很慢)。
- **问：如何运行模型评测？**
  答：使用`python tools/benchmark.py`配合可用基准测试。参见上方的模型评测部分。

---

## 📄 许可证
Pisces L1采用知识共享署名-非商业性使用 4.0 国际许可协议 (CC BY-NC 4.0)。**明确禁止商用。** 详见[LICENSE](./LICENSE)。

### 许可证概要
- **署名**：您必须给出适当的署名，提供指向许可证的链接，并标明是否（对原始作品）作了修改。
- **非商业性使用**：您不得将本作品用于商业目的。
- **无附加限制**：您不得适用法律术语或者技术措施从而限制其他人做许可协议允许的事情。

---

## 🌏 社区与引用
- 欢迎提交issues和PR！
- [PiscesL1 in Gitee](https://gitee.com/dunimd/piscesl1.git)
- [PiscesL1 in GitHub](https://github.com/mf2023/PiscesL1.git)
- [PiscesL1 in ModelScope](https://www.modelscope.cn/models/mfchina2024/PiscesL1)

---

*祝您使用Pisces L1实验愉快！*