#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Dict, Any, Tuple
import torch

"""
受控采样水印（lexical control）：
- 在采样阶段对词表分桶（基于seed/key），对“签名桶”提升概率，形成稳定统计指纹
- 提供统计检测（Z分数/LLR）显著性评分
- 部署时在生成阶段调用 apply_watermark_logits()（本模块提供可插拔接口）
"""

def _bucket_index(token_id: int, vocab_size: int, seed: int, num_buckets: int = 2) -> int:
    # 简单哈希分桶：稳定可复现
    return ((token_id * 1315423911 + seed) % (10**9 + 7)) % num_buckets

def apply_watermark_logits(logits: torch.Tensor, vocab_size: int, seed: int, boost: float = 0.15) -> torch.Tensor:
    """
    输入：
    - logits: [vocab] 未归一化对数概率
    - vocab_size: 词表大小
    - seed: 会话/载荷派生的整数种子
    - boost: 对签名桶的提升幅度（对数域）
    输出：调整后的 logits
    """
    if logits.shape[0] != vocab_size:
        vocab_size = logits.shape[0]
    # 生成桶掩码：签名桶=1，非签名桶=0
    idxs = torch.arange(vocab_size, device=logits.device)
    buckets = torch.tensor([_bucket_index(int(i.item()), vocab_size, seed) for i in idxs], device=logits.device)
    signature_mask = (buckets == 1).float()
    # 调整：对签名桶加boost
    adjusted = logits + boost * signature_mask
    return adjusted

def detection_score(token_ids: torch.Tensor, vocab_size: int, seed: int) -> float:
    """
    对生成序列做简单统计检测：
    - 统计落入签名桶的比例，与随机假设做Z分数比较
    返回：分数（越高越可能带水印）
    """
    if token_ids.numel() == 0:
        return 0.0
    buckets = torch.tensor([_bucket_index(int(t.item()), vocab_size, seed) for t in token_ids])
    sig_hits = (buckets == 1).sum().item()
    n = token_ids.numel()
    p0 = 0.5  # 随机假设
    # Z = (x - np0) / sqrt(np0(1-p0))
    denom = (n * p0 * (1 - p0)) ** 0.5 if n > 0 else 1.0
    z = (sig_hits - n * p0) / (denom if denom > 1e-8 else 1.0)
    # 把Z分数映射到[0,1]：sigmoid
    return 1.0 / (1.0 + math.exp(-z))