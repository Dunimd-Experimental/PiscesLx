# 02 · 项目概览、核心使命、最终交付物

## 项目概览

PiscesL1 是采用 **Yv Architecture** 的多模态 MoE 模型框架，由 Dunimd Team (Wenze Wei) 开发。

- **模型类型**：多模态 MoE 稀疏专家模型
- **支持模态（升级后）**：text, image, audio, video（共 4 种）。**document 和 agent 不再是独立模态**（见 `UPGRADE_03_ARCH.md` 3.3）
- **可配置规模**：0.5B / 1.5B / 7B / 32B / 64B / 70B / 128B / 314B / 671B / 1T
- **核心特性**：查算分离（Subconscious System）、混合注意力-SSM、多路径推理、长上下文、EnTA 训练代理（待删除）

### 关键目录结构

```
model/
├── core/           # attention, blocks, model, mamba3, hybrid, subconscious, knowledge_experts, dual_injector, ...
├── moe/            # gate, layer, expert, diversity, hicl_router, graph_of_tokens, path_moe, mega_kernel, online_router, ...
├── multimodal/     # vision, audio, video, doc, agentic 编码器 + fusion + mcp + server + generator + ...
├── reasoning/      # reasoner/(cot_memory, multipath_core, multipath_system, recursive_depth, unified, enhanced_system), self_evolution, ttt_e2e, vericot, verification
├── agentic/enta/   # EnTA 训练代理（待物理删除）
├── generation/     # speculative decoding
├── channels/       # 通道
├── tokenizer/      # 分词器
├── config.py       # YvConfig 配置数据类（~300+ 字段，1289+ 行）
└── utils/
configs/
├── model/          # 各规模 YAML（0.5B ~ 1T）
├── train/          # 训练配置
├── agentevolver/   # Agent 配置（待删除）
├── dataset.yaml, settings.yaml, teachers.yaml（待删除）, version.py, watermark.yaml（待移除）
opss/               # train, infer, optim, quantize, scaling, kernels, knowledge, concurrency, agents（待删除）, mcp（待审查）, watermark（待移除）
tools/              # benchmark, cache, data, dev, infer, monitor, publish, test, train, wmc（待移除）
manage.py           # 统一 CLI 入口
```

### 核心架构组件

1. **YvUnifiedReasoner**：CoT + Multi-Path 推理路由
2. **Yv MoE**：稳定门控（LSTM 负载预测器）、细粒度专家分割、共享专家隔离
3. **Subconscious System（查算分离）**：16 codebooks × 131K entries 乘积量化知识场，FiLM 调制注入——"潜意识"思想雏形：知识块权重内嵌于模型、稀疏激活、按需唤起
4. **Memory Separation**：FAISS 外部知识检索，N-gram 地址哈希——**注意：旧的"外部检索"实现，与查算分离的正确含义（模型自身的知识块池，非外部库）不一致，需重新评估改造**
5. **多模态感知栈**：YvVisionEncoder (SigLIP), YvVideoEncoder (3D RoPE), YvAudioEncoder (HiFi-GAN), YvDocEncoder (LayoutLMv3), YvAgenticEncoder——**注意：doc 与 agentic 编码器需按 3.3 节移除/合并**
6. **YvDynamicModalFusion**：token 级融合，质量加权门控
7. **长上下文**：YaRN+NTK, H2O, Streaming, Sliding Window, Linear Attention, Paged Attention, Ring Attention, OOMB
8. **Hybrid Attention-SSM**：Mamba-3, YvSelectiveSSM, 渐进门控

> **⚠️ 关于查算分离的已知误读澄清（务必阅读）**
>
> 此前文档/讨论中对"查算分离"出现过两处**错误表述**，Agent 不得沿用：
> 1. ❌ "314B 完全冻结、从不训练。" → ✅ 正确：314B 由 628 个 0.5B 知识块组成，**每次仅稀疏激活约 0.5B**，**被激活的知识块参与训练**（可理解为先训练/复用这些 0.5B 块，再与主参数联合工作）。
> 2. ❌ "314B 从磁盘/CPU 慢速加载（如 5ms/token 的 mmap 调取）。" → ✅ 正确：**调取机制是本项目必须自主发明的核心创新，业界无现成参考，本项目也尚无实现先例。不得臆断为任何特定实现（内存加载、预取、量化、分层缓存、codebook 等均只是候选，不是规定）。**
>
> 由这两处误读会推导出伪矛盾（如"未训练的知识块里没有知识"），该矛盾**不存在**。请以 `UPGRADE_03_ARCH.md` 3.2 节为准。

## 核心使命

**一句话概括**：将 PiscesL1 全面升级为 2026 年 10 月最先进的多模态 MoE 稀疏专家模型框架，统一所有模态，合并去重并旗舰化全部算法，实现 7B 主参数 + 314B 知识块池（628 × 0.5B，稀疏激活约 0.5B）的查算分离架构，支持 1M 输入 + 1M 输出上下文，并能在 NVIDIA T4（16GB）上进行全精度快速训练和完美推理。

### 使命分解

1. **模态重定义与统一**：模态体系从 6 种（text/image/audio/video/document/agent）修正为 **4 种（text/image/audio/video）**——document 与 agent 不再作为独立模态（agent 本质是文本），并统一这 4 种模态的表示和处理
2. **算法旗舰化**：合并重复算法，确保每个保留的算法都是 2026 年旗舰级
3. **架构创新**：不仅是算法优化，可进行架构级创新
4. **查算分离**：7B 主参数（MoE 稀疏专家，激活计算）+ 314B 知识块池（628 × 0.5B，稀疏激活约 0.5B，被激活的块参与训练）
5. **T4 适配**：7B 全精度快速训练 + 完美推理，全部在 T4 上完成
6. **全尺寸服务**：架构服务于 0.5B ~ 1T 全家族，7B 只是当前最大训练目标
7. **代码清理**：删除 EnTA/Encre 系统，移除水印至第三端，清理不属于模型的组件

## 最终交付物（唯一的验收物）

**最终必须交付一个真实可用的模型权重（checkpoint）。**

- 必须**真正能够推理**——不是空壳、不是随机初始化、不是未训练的半成品
- 必须**真正能够执行任务**——实际完成推理、多模态理解、Agent 操作等真实任务
- 必须**满足全部 T4 约束**——能在 T4 上真实运行推理，达到 `UPGRADE_09_METRICS.md` 规定的全部硬性指标（吞吐、延迟、评测分数）
- 必须**经过真实训练**——训练过程真实发生，评测结果真实可复现
- **没有交付这个权重，任务就不算完成，不允许停止。**

> 所有架构设计、算法优化、代码清理、工程实现，最终都要收敛到一个真实的、可运行的、达标的大模型权重。文档和代码是过程，**权重才是终点**。达不到要求，不许停。
