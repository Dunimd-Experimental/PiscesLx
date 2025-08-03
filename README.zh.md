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
| train       | 训练模型(支持`--distributed`分布式训练)          |
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
| 0.5B    | 12   | 1024      | 8         | 4        | ~0.5B   | 10M               |
| 1.5B    | 24   | 3072      | 32        | 16       | ~1.5B   | 10M               |
| 7B      | 32   | 4096      | 32        | 32       | ~7B     | 10M               |
| 32B     | 48   | 6656      | 52        | 64       | ~32B    | 10M               |
| 64/70B  | 80   | 8192      | 64        | 128      | ~70B    | 10M               |

- **多模态集成**: CLIP ViT-L/14(视觉)、AST Base(音频)、LayoutLMv3(文档)，具有统一的嵌入空间
- **MoE**: Top-2路由，高效专家加载

---

## 📦 数据集[默认魔塔社区(https://www.modelscope.cn/)]
数据集会自动下载并缓存。支持以下数据集：

### 数学与推理
- **NuminaMath-CoT** (AI-ModelScope/NuminaMath-CoT): 带思维链的数学推理数据集

### 中文语言
- **Llama3-Chinese-Dataset** (zhuangxialie/Llama3-Chinese-Dataset): 中文语料库
- **Chinese-DeepSeek-R1** (liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT): 中文指令微调数据

### 网络与通用知识
- **OpenWeb888K** (prithivMLmods/OpenWeb888K): 网络爬取数据

### 图像理解
- **ShareGPT-4o-Image** (FreedomIntelligence/ShareGPT-4o-Image): 图像对话对
- **coco_captions_small_slice** (modelscope/coco_captions_small_slice): COCO图像标题
- **LAION-SG** (AI-ModelScope/LAION-SG): 语义图数据集

### 音频处理
- **AudioSetCaps_350k** (lmms-lab/AudioSetCaps_350k_converted): 音频标题生成
- **Libri2Mix_8k** (modelscope/Libri2Mix_8k): 音频混合数据集
- **Clotho** (OmniData/Clotho): 音频标题生成

### 代码与编程
- **ultrachat_200k** (HuggingFaceH4/ultrachat_200k): 基于对话的指令微调
- **CodeAlpaca_20K** (HuggingFaceH4/CodeAlpaca_20K): 代码指令微调
- **codeparrot_github-code** (jablonkagroup/codeparrot_github-code-chemistry-python): Python代码语料库

### 文档理解
- **DocVQA** (swift/DocVQA): 文档视觉问答
- **PubLayNet** (OpenDataLab/PubLayNet): 文档布局分析
- **VQAv2** (swift/VQAv2): 视觉问答

通过以下命令自动下载和缓存数据集：
```bash
python manage.py download
```

---

## 🏆 在24GB GPU上训练70B模型[Beta]
Pisces L1支持**在单个24GB GPU上训练/微调70B模型**，使用QLoRA、4位量化、LoRA适配器和梯度累积技术。

#### 70B QLoRA训练示例

##### 单GPU
```bash
python manage.py train \
  --model_size 70B \
  --force_quant \
  --force_lora \
  --batch_size 1 \
  --accum 32 \
  --seq_len 512
```

##### 多GPU分布式训练
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
- 4位量化：显著减少内存占用([QLoRA论文](https://arxiv.org/abs/2305.14314))
- LoRA适配器：高效参数微调
- 梯度累积：模拟大批次训练
- 混合精度：进一步节省内存
- 无精度损失：QLoRA+LoRA达到接近全精度的结果([QLoRA深度解析](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html))

---

## ⚡ 快速开始
安装完成后，尝试以下三步工作流：
```bash
# 1. 下载默认数据集
python manage.py download

# 2. 训练小型模型(0.5B)
python manage.py train --model_size 0.5B

# 3. 运行推理
python manage.py infer --prompt "用简单的语言解释量子计算" --ckpt ckpt/latest.pt
```

## ❓ 常见问题
- **问：如何查看所有可用命令？**
  答：`python manage.py help`
- **问：如何添加新数据集？**
  答：将数据集名称添加到`data/download.py`的`DATASETS`列表中，并重新运行`python manage.py download`。对于自定义数据集，格式应为带`text`字段的JSONL或带`input_ids`和`labels`列的Parquet。
- **问：遇到内存不足错误？**
  答：使用`--batch_size`减少批次大小，使用`--force_quant`启用量化，或使用更小的模型尺寸。
- **问：如何恢复训练？**
  答：使用`--resume_ckpt path/to/checkpoint.pt`从保存的检查点继续训练。
- **问：模型配置文件在哪里？**
  答：参见`configs/`目录下的所有模型尺寸配置。
- **问：如何仅在CPU上运行？**
  答：大多数功能需要GPU，但可以尝试使用`--device cpu`(性能会很慢)。

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