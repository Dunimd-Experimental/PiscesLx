# ⚠️ 合规提示

**如果你在中国境内使用本模型，包括但不限于训练、微调、商用测试等，凡是向社会公众提供任何服务的，请先按照相关法律法规完成手续。**

---

# Pisces L1

简体中文 | [English](README.md)

采用**Arctic架构**的下一代轻量级多模态混合专家模型(MoE)，支持文本、图像、音频、视频、文档和智能体理解。Pisces L1 (PiscesLx系列，Dunimd项目组)专为研究和实际应用而设计，可在单个RTX 4090上运行，具备突破性创新，扩展至314B参数。

## ⭐ Arctic架构创新

### 🧠 量子推理引擎 (PiscesReasoner)
- **层次推理链(HRC)**: 多层抽象处理，超越传统思维链
- **量子超位置思维**: 8个并行假设流与量子坍缩机制
- **动态事实验证**: 实时真实性检查和一致性评分
- **元认知反思**: 推理过程的自我感知与不确定性量化
- **量子特殊tokens**: `<start_hypothesis>`, `<start_evidence>`, `<start_conclusion>`, `<quantum_merge>`

### 🔧 MoE专家系统
- **8专家Top-2路由**: 智能负载均衡与StableMoEGate
- **LSTM负载预测**: 动态容量调整和专家分配
- **梯度检查点兼容**: 固定形状模式，内存高效
- **稳定专家门控**: 高级路由，噪声注入与容量控制

### 🌐 5模态编码系统
- **VisionEncoder**: NaViT原生分辨率支持(高达1024px)
- **VideoEncoder**: 时序视觉理解，帧级注意力
- **AudioEncoder**: 高级音频特征提取与处理
- **DocEncoder**: 文档结构理解，LayoutLMv3集成
- **AgentEncoder**: 全面智能体行为建模(观察、行动、反思)

### ⚛️ 量子纠缠融合
- **DynamicModalFusion**: 高级跨模态注意力与量子纠缠
- **张量网络压缩**: 6对相关性网络实现模态交互
- **硬件自适应配置**: 智能硬件检测与优化
- **质量感知融合门**: 基于内容质量的自适应融合

### 📏 超长上下文系统
- **YaRN RoPE**: 支持10M+ tokens，动态NTK缩放
- **H2O注意力**: 流式注意力，支持128B+参数模型
- **动态位置编码**: 自适应位置编码，长因子缩放
- **内存高效上下文**: 滑动窗口与压缩技术

### 🤖 高级智能体系统
- **PiscesAgent**: 原生多模态智能体，MCP协议支持
- **工具集成**: 内置工具使用能力与环境交互
- **持久化内存**: 上下文管理与经验积累
- **MCP通信**: 多智能体通信协议，分布式推理

### 🎯 K-FAC优化
- **二阶优化**: 对角Fisher矩阵近似
- **内存高效实现**: 比完整K-FAC减少99%+内存
- **曲率感知训练**: 自然梯度计算，加速收敛
- **自适应梯度裁剪**: 基于训练历史的动态阈值调整

---

## 🚀 特性

- **Arctic架构**: 革命性多模态架构，量子启发推理
- **量子推理引擎**: 超越传统思维链，层次化抽象处理
- **5模态理解**: 统一处理文本、图像、音频、视频、文档和智能体输入
- **MoE专家系统**: 8专家智能Top-2路由和负载预测
- **超长上下文**: 支持10M+ tokens，YaRN RoPE和H2O注意力
- **高级量化**: 2位、4位、8位量化选项，梯度稳定性优化
- **内存优化**: 14.58GB GPU上运行0.5B模型，QLoRA+梯度检查点
- **K-FAC优化**: 二阶优化，对角Fisher矩阵近似
- **原生智能体支持**: 内置MCP协议和工具集成
- **一键式工作流**: 通过`python manage.py`完整管理

---

## 🛠️ 安装与环境

- **Python**: 3.11推荐
- **CUDA**: 11.8+ (用于GPU训练/推理)
- **依赖项**: 所有必需包列在`requirements.txt`中

### 快速设置
```bash
git clone https://gitee.com/dunimd/piscesl1.git or git clone https://github.com/mf2023/piscesl1.git
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
| 命令       | 描述                                                          |
|------------|---------------------------------------------------------------|
| setup      | 环境设置和依赖安装                                            |
| source     | 激活虚拟环境                                                  |
| update     | 从远程仓库拉取最新代码                                        |
| version    | 显示当前版本和更新日志                                        |
| changelog  | 显示版本历史(--all显示所有，--version X.X.XXXX显示特定版本)   |
| rlhf       | 来自人类反馈的强化学习训练                                     |
| watermark  | 检测是否为PiscesLx系列模型生成                                |
| train      | 训练模型                                                      |
| infer      | 使用训练好的模型进行推理                                      |
| check      | 检查GPU和依赖项                                               |
| monitor    | 系统监控(GPU/CPU/内存)                                        |
| download   | 下载训练数据集                                                |
| dataset    | 数据集管理工具                                                |
| quantize   | 将模型量化为4/8位以提高效率                                   |
| benchmark  | 运行性能基准测试                                               |
| help       | 显示帮助信息                                                  |

#### 示例
```bash
python manage.py download
python manage.py version          # 显示当前版本
python manage.py changelog --all    # 显示所有版本
python manage.py changelog --version 1.0.0150  # 显示特定版本
python manage.py train
python manage.py infer --ckpt ckpt/model.pt --prompt "你好，Pisces!"
```

---

## 🧠 模型架构与配置

### Arctic架构组件
- **核心Transformer**: RMSNorm、YaRN RoPE、分组查询注意力
- **量子推理**: 4层层次化抽象，元认知反思
- **多模态融合**: 量子纠缠跨模态注意力，张量网络
- **MoE系统**: 动态专家路由，LSTM负载预测
- **内存优化**: 梯度检查点、混合精度、K-FAC优化

| 模型大小 | 层数 | 隐藏层 | 注意力头数 | KV头数 | MoE专家数 | 参数(实际) | 上下文 | 量化 |
|----------|------|--------|------------|--------|-----------|------------|---------|-------------|
| 0.5B     | 16   | 640    | 10         | 5      | 6         | 0.5B       | 256K    | 2/4/8-bit   |
| 1.5B     | 16   | 896    | 14         | 7      | 6         | 1.5B       | 256K    | 2/4/8-bit   |
| 7B       | 28   | 3584   | 32         | 8      | 8         | 7B         | 1M      | 2/4/8-bit   |
| 32B      | 64   | 5120   | 40         | 8      | 8         | 32B        | 1M      | 2/4/8-bit   |
| 64B      | 80   | 6656   | 52         | 8      | 8         | 64B        | 10M     | 2/4/8-bit   |
| 70B      | 80   | 8192   | 64         | 8      | 8         | 70B        | 10M     | 2/4/8-bit   |
| 128B     | 120  | 10240  | 80         | 8      | 8         | 128B       | 10M     | 2/4/8-bit   |
| 314B     | 160  | 12288  | 96         | 12     | 16        | 314B       | 10M     | 2/4/8-bit   |

### 参数分解 (0.5B配置)
- **核心Transformer**: ~500M参数
- **多模态编码器**: 优化平衡设计
  - VisionEncoder: ~120M, VideoEncoder: ~150M, AudioEncoder: ~80M
  - DocEncoder: ~80M, AgentEncoder: ~70M
- **量子推理引擎**: 集成在核心参数中
- **模态融合系统**: 轻量级设计，提高效率

### 量化选项
```bash
# 2位量化（实验性，最大内存节省）
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 2

# 4位量化（默认，均衡性能）
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 4

# 8位量化（稳定，最小质量损失）
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 8
```

---

## 📦 数据集[默认魔塔社区(https://www.modelscope.cn/)]
数据集自动下载并缓存。支持以下数据集：

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
- **Chat1** (HuggingFaceH4/ultrachat_200k): 对话指令微调
- **PubLayNet1** (OpenDataLab/PubLayNet): 文档布局分析
- **Medical1** (krisfu/delicate_medical_r1_data): 医疗指令数据
- **Financial1** (BJQW14B/bs_challenge_financial_14b_dataset): 金融数据集

### 智能体与行为理解
- **Agent1** (AI-ModelScope/agent-instruct): 智能体指令微调数据集
- **Agent2** (OmniData/agent-dialogue): 多轮智能体对话数据
- **Agent3** (swift/agent-reasoning): 智能体推理与规划任务
- **Agent4** (HuggingFaceH4/agent-tool-use): 工具使用智能体行为数据集
- **Agent5** (modelscope/agent-environment): 智能体环境交互数据

数据集通过以下命令自动下载并缓存：
```bash
python manage.py download
```

---

## 🏆 14.58GB GPU上的高级训练
Pisces L1 Arctic架构支持**在14.58GB GPU上训练1.5B参数模型**，使用高级量化、LoRA和内存优化技术。

### 内存优化策略
- **多位量化**: 2位（实验性）、4位（默认）、8位（稳定）
- **LoRA适配**: 仅0.024%参数可训练（360K / 1.5B）
- **梯度检查点**: 减少激活内存50%+
- **K-FAC优化**: 对角Fisher矩阵近似
- **自适应梯度裁剪**: 自动处理梯度爆炸

### 训练示例

#### 1.5B模型，4位量化
```bash
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --force_lora
```

#### 1.5B模型，8位稳定性
```bash
python manage.py train --model_size 1.5B --dataset Chinese2 --force_quant --quant_bits 8 --force_lora
```

#### 内存使用对比
| 配置 | 内存使用 | 可训练% | 梯度稳定性 |
|---------------|--------------|-------------|--------------------|
| 全精度 | >40GB | 100% | 稳定 |
| 8位 + LoRA | ~18GB | 0.024% | 非常稳定 |
| 4位 + LoRA | ~14.5GB | 0.024% | 可管理 |
| 2位 + LoRA | ~11GB | 0.024% | 实验性 |

### 训练性能
- **损失收敛**: 35.38 → 31.62（140步），10.6%改善
- **梯度裁剪**: 自动处理280K+梯度范数
- **内存效率**: 14.58GB GPU上稳定训练
- **速度**: 与模型复杂度成正比（1.5B比0.5B慢，符合预期）

---

## ⚡ 快速开始

安装完成后，试试这个6步工作流程：
```bash
# 1. 环境设置
python manage.py setup

# 2. 激活环境
python manage.py source

# 3. 拉取最新代码(可选)
python manage.py update

# 4. 下载默认数据集
python manage.py download

# 5. 训练小模型(0.5B)
python manage.py train --model_size 0.5B

# 6. 运行推理
python manage.py infer --prompt "用简单的话解释量子计算" --ckpt ckpt/latest.pt
```

## 🤖 MCP原生智能体支持 [Beta]
Pisces L1现已包含**原生MCP(Model Context Protocol)**支持，实现智能体间无缝通信和分布式任务执行。

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
| 基准测试 | 语言 | 关注领域 | 题型 |
|----------|------|----------|------|
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
  答：使用`python manage.py benchmark`配合可用基准测试。参见上方的模型评测部分。

---

## 📄 许可证
本项目采用 **Apache许可证2.0** 授权 - 详见[LICENSE](LICENSE)文件。

### 许可证摘要
- **商业使用**: ✅ 允许
- **修改**: ✅ 允许
- **分发**: ✅ 允许
- **署名**: ✅ 需要
- **专利授权**: ✅ 包含

---

## 🌏 社区与引用
- 欢迎提交issues和PR！
- [PiscesL1 in Gitee](https://gitee.com/dunimd/piscesl1.git)
- [PiscesL1 in GitHub](https://github.com/mf2023/piscesl1.git)
- [PiscesL1 in ModelScope](https://www.modelscope.cn/models/mfchina2024/PiscesL1)

<h3 align="center">以直觉航行数据之深邃 以共情赋予智能形态</h3>

![summary](./icons/PD.png)