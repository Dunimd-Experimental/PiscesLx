#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""Authoritative model catalog.

Every entry describes one provider:
  - ``id``       : the backend_type string accepted by ``create_backend``
  - ``label``    : human-readable provider name
  - ``base_url`` : official documented endpoint
  - ``docs``     : provider documentation URL (so users can verify)
  - ``models``   : list of {id, label, context} entries — *context* is the
                   model's documented max INPUT context window in tokens
  - ``allow_custom``: whether arbitrary model IDs are accepted in addition
                      to the curated list. True for nearly every provider —
                      most expose models that change weekly.

Curated model lists are intentionally conservative: only models the provider
actively documents on their own pricing or API reference page. Users who
need a cutting-edge SKU can pick "Custom" and type the ID themselves.

To extend: add to ``PROVIDERS`` below and re-export through
``encre/backends/__init__.py`` and ``encre/__init__.py``.
"""

from typing import Any


# Default per-turn output token budgets, keyed by provider id.
# Providers cap differently from input context; these match documented limits.
DEFAULT_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "anthropic": 128000,
    "openai": 128000,
    "deepseek": 32768,
    "google": 8192,
    "groq": 32768,
    "ollama": 8192,
    "lmstudio": 8192,
    "bedrock": 8192,
    "openai_compatible": 8192,
    "openrouter": 128000,
    "novita": 8192,
    "aigateway": 8192,
    "glm": 65536,
    "kimi": 65536,
    "arcee": 8192,
    "gmi": 8192,
    "minimax": 65536,
    "alibaba": 65536,
    "kilocode": 128000,
    "xiaomi": 8192,
    "tencent": 65536,
    "huggingface": 8192,
    "opencode-zen": 65536,
    "opencode-go": 8192,
    "github-copilot": 65536,
    "failover": 8192,
    "router": 8192,
    "local": 4096,
}


PROVIDERS: list[dict[str, Any]] = [
    # ── Anthropic ────────────────────────────────────────────────────────
    {
        "id": "anthropic",
        "label": "Anthropic",
        "base_url": "https://api.anthropic.com",
        "docs": "https://docs.anthropic.com/en/docs/about-claude/models",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "claude-opus-4-7", "label": "Claude Opus 4.7", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5", "context": 200000, "modalities": ["text", "image"]},
        ],
    },
    # ── OpenAI ───────────────────────────────────────────────────────────
    {
        "id": "openai",
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "docs": "https://platform.openai.com/docs/models",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "gpt-5.5", "label": "GPT-5.5", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "gpt-5.4", "label": "GPT-5.4", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "gpt-5.4-mini", "label": "GPT-5.4 mini", "context": 400000, "modalities": ["text", "image"]},
            {"id": "gpt-4.1", "label": "GPT-4.1", "context": 1047576, "modalities": ["text", "image"]},
            {"id": "gpt-4.1-mini", "label": "GPT-4.1 mini", "context": 1047576, "modalities": ["text", "image"]},
            {"id": "gpt-4.1-nano", "label": "GPT-4.1 nano", "context": 1047576, "modalities": ["text", "image"]},
            {"id": "gpt-4o", "label": "GPT-4o", "context": 128000, "modalities": ["text", "image", "audio"]},
            {"id": "gpt-4o-mini", "label": "GPT-4o mini", "context": 128000, "modalities": ["text", "image", "audio"]},
            {"id": "o3", "label": "o3", "context": 200000, "modalities": ["text"]},
            {"id": "o3-mini", "label": "o3-mini", "context": 200000, "modalities": ["text"]},
            {"id": "o4-mini", "label": "o4-mini", "context": 200000, "modalities": ["text"]},
        ],
    },
    # ── DeepSeek ─────────────────────────────────────────────────────────
    {
        "id": "deepseek",
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "docs": "https://api-docs.deepseek.com/quick_start/pricing",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "deepseek-v4-flash", "label": "DeepSeek-V4-Flash", "context": 1000000, "modalities": ["text"]},
            {"id": "deepseek-v4-pro", "label": "DeepSeek-V4-Pro", "context": 1000000, "modalities": ["text"]},
            {"id": "deepseek-chat", "label": "DeepSeek-V3.2 (chat, deprecated)", "context": 128000, "modalities": ["text"]},
            {"id": "deepseek-reasoner", "label": "DeepSeek-R1 (reasoner, deprecated)", "context": 128000, "modalities": ["text"]},
        ],
    },
    # ── Google Gemini ────────────────────────────────────────────────────
    {
        "id": "google",
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com",
        "docs": "https://ai.google.dev/gemini-api/docs/models",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash", "context": 1048576, "modalities": ["text", "image", "audio", "video"]},
            {"id": "gemini-3.1-pro", "label": "Gemini 3.1 Pro", "context": 2097152, "modalities": ["text", "image", "audio", "video"]},
            {"id": "gemini-3-flash", "label": "Gemini 3 Flash", "context": 1048576, "modalities": ["text", "image", "audio", "video"]},
            {"id": "gemini-2.5-pro", "label": "Gemini 2.5 Pro", "context": 2097152, "modalities": ["text", "image", "audio", "video"]},
            {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash", "context": 1048576, "modalities": ["text", "image", "audio", "video"]},
            {"id": "gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash-Lite", "context": 1048576, "modalities": ["text", "image", "audio", "video"]},
        ],
    },
    # ── Groq ─────────────────────────────────────────────────────────────
    {
        "id": "groq",
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "docs": "https://console.groq.com/docs/models",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B Versatile", "context": 131072, "modalities": ["text"]},
            {"id": "llama-3.1-8b-instant", "label": "Llama 3.1 8B Instant", "context": 131072, "modalities": ["text"]},
            {"id": "openai/gpt-oss-120b", "label": "GPT-OSS 120B", "context": 131072, "modalities": ["text"]},
            {"id": "openai/gpt-oss-20b", "label": "GPT-OSS 20B", "context": 131072, "modalities": ["text"]},
            {"id": "groq/compound", "label": "Groq Compound", "context": 131072, "modalities": ["text"]},
            {"id": "groq/compound-mini", "label": "Groq Compound Mini", "context": 131072, "modalities": ["text"]},
            {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "label": "Llama 4 Scout 17B", "context": 131072, "modalities": ["text", "image"]},
            {"id": "qwen/qwen3-32b", "label": "Qwen3-32B", "context": 131072, "modalities": ["text"]},
        ],
    },
    # ── Ollama (local) ───────────────────────────────────────────────────
    {
        "id": "ollama",
        "label": "Ollama (local)",
        "base_url": "http://localhost:11434",
        "docs": "https://ollama.com/library",
        "allow_custom": True,
        "auth": "none",
        "models": [
            {"id": "llama3.3", "label": "Llama 3.3 70B", "context": 131072, "modalities": ["text"]},
            {"id": "llama3.2", "label": "Llama 3.2", "context": 131072, "modalities": ["text"]},
            {"id": "llama3.1", "label": "Llama 3.1", "context": 131072, "modalities": ["text"]},
            {"id": "qwen2.5", "label": "Qwen 2.5", "context": 131072, "modalities": ["text"]},
            {"id": "qwen2.5-coder", "label": "Qwen 2.5 Coder", "context": 131072, "modalities": ["text"]},
            {"id": "deepseek-r1", "label": "DeepSeek R1", "context": 131072, "modalities": ["text"]},
            {"id": "deepseek-v3", "label": "DeepSeek V3", "context": 131072, "modalities": ["text"]},
            {"id": "mistral", "label": "Mistral 7B", "context": 32768, "modalities": ["text"]},
            {"id": "mixtral", "label": "Mixtral 8x7B", "context": 32768, "modalities": ["text"]},
            {"id": "phi4", "label": "Phi-4 14B", "context": 16384, "modalities": ["text"]},
            {"id": "gemma2", "label": "Gemma 2", "context": 8192, "modalities": ["text"]},
        ],
    },
    # ── LM Studio (local) ────────────────────────────────────────────────
    {
        "id": "lmstudio",
        "label": "LM Studio (local)",
        "base_url": "http://localhost:1234/v1",
        "docs": "https://lmstudio.ai/docs/api",
        "allow_custom": True,
        "auth": "none",
        "models": [],  # always user-supplied; LM Studio exposes whatever you've loaded
    },
    # ── Local transformers ───────────────────────────────────────────────
    {
        "id": "local",
        "label": "Local (transformers)",
        "base_url": "",
        "docs": "https://huggingface.co/docs/transformers",
        "allow_custom": True,
        "auth": "none",
        "models": [],
    },
    # ── AWS Bedrock ──────────────────────────────────────────────────────
    {
        "id": "bedrock",
        "label": "AWS Bedrock",
        "base_url": "",
        "docs": "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html",
        "allow_custom": True,
        "auth": "aws_iam",
        "models": [
            {"id": "anthropic.claude-opus-4-7", "label": "Claude Opus 4.7 (Bedrock)", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "anthropic.claude-sonnet-4-6", "label": "Claude Sonnet 4.6 (Bedrock)", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "anthropic.claude-haiku-4-5-20251001-v1:0", "label": "Claude Haiku 4.5 (Bedrock)", "context": 200000, "modalities": ["text", "image"]},
            {"id": "us.meta.llama3-3-70b-instruct-v1:0", "label": "Llama 3.3 70B Instruct", "context": 128000, "modalities": ["text"]},
            {"id": "amazon.nova-pro-v1:0", "label": "Amazon Nova Pro", "context": 300000, "modalities": ["text", "image", "video"]},
            {"id": "amazon.nova-lite-v1:0", "label": "Amazon Nova Lite", "context": 300000, "modalities": ["text", "image", "video"]},
            {"id": "amazon.nova-micro-v1:0", "label": "Amazon Nova Micro", "context": 128000, "modalities": ["text"]},
        ],
    },
    # ── Generic OpenAI-compatible ─────────────────────────────────────────
    {
        "id": "openai_compatible",
        "label": "OpenAI-Compatible (custom)",
        "base_url": "",
        "docs": "",
        "allow_custom": True,
        "auth": "api_key",
        "models": [],
    },
    # ── Aggregators ──────────────────────────────────────────────────────
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "docs": "https://openrouter.ai/models",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "openrouter/auto", "label": "Auto (router)", "context": 200000, "modalities": ["text", "image"]},
            {"id": "anthropic/claude-opus-4.7", "label": "Claude Opus 4.7", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "anthropic/claude-sonnet-4.6", "label": "Claude Sonnet 4.6", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "openai/gpt-5.5", "label": "GPT-5.5", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "openai/gpt-5.4", "label": "GPT-5.4", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "openai/gpt-4.1", "label": "GPT-4.1", "context": 1047576, "modalities": ["text", "image"]},
            {"id": "google/gemini-2.5-pro", "label": "Gemini 2.5 Pro", "context": 2097152, "modalities": ["text", "image", "audio", "video"]},
            {"id": "google/gemini-3.5-flash", "label": "Gemini 3.5 Flash", "context": 1048576, "modalities": ["text", "image", "audio", "video"]},
            {"id": "deepseek/deepseek-v4-flash", "label": "DeepSeek V4 Flash", "context": 1000000, "modalities": ["text"]},
            {"id": "deepseek/deepseek-v4-pro", "label": "DeepSeek V4 Pro", "context": 1000000, "modalities": ["text"]},
            {"id": "meta-llama/llama-3.3-70b-instruct", "label": "Llama 3.3 70B Instruct", "context": 128000, "modalities": ["text"]},
            {"id": "x-ai/grok-4", "label": "xAI Grok 4", "context": 256000, "modalities": ["text", "image"]},
        ],
    },
    {
        "id": "novita",
        "label": "Novita AI",
        "base_url": "https://api.novita.ai/v3/openai",
        "docs": "https://novita.ai/docs/api-reference/llms-llm-api",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "deepseek/deepseek-v4-flash", "label": "DeepSeek V4 Flash", "context": 1000000, "modalities": ["text"]},
            {"id": "deepseek/deepseek-v4-pro", "label": "DeepSeek V4 Pro", "context": 1000000, "modalities": ["text"]},
            {"id": "meta-llama/llama-3.3-70b-instruct", "label": "Llama 3.3 70B Instruct", "context": 131072, "modalities": ["text"]},
            {"id": "qwen/qwen-2.5-72b-instruct", "label": "Qwen 2.5 72B Instruct", "context": 131072, "modalities": ["text"]},
        ],
    },
    {
        "id": "aigateway",
        "label": "Cloudflare AI Gateway",
        "base_url": "https://gateway.ai.cloudflare.com/v1",
        "docs": "https://developers.cloudflare.com/ai-gateway/",
        "allow_custom": True,
        "auth": "api_key",
        "models": [],
    },
    {
        "id": "huggingface",
        "label": "Hugging Face Inference",
        "base_url": "https://api-inference.huggingface.co/v1",
        "docs": "https://huggingface.co/docs/api-inference",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "meta-llama/Llama-3.3-70B-Instruct", "label": "Llama 3.3 70B Instruct", "context": 131072, "modalities": ["text"]},
            {"id": "Qwen/Qwen2.5-72B-Instruct", "label": "Qwen 2.5 72B Instruct", "context": 131072, "modalities": ["text"]},
            {"id": "deepseek-ai/DeepSeek-V3", "label": "DeepSeek V3", "context": 131072, "modalities": ["text"]},
            {"id": "deepseek-ai/DeepSeek-R1", "label": "DeepSeek R1", "context": 131072, "modalities": ["text"]},
        ],
    },
    {
        "id": "github-copilot",
        "label": "GitHub Copilot",
        "base_url": "https://api.githubcopilot.com",
        "docs": "https://docs.github.com/en/copilot",
        "allow_custom": True,
        "auth": "github_oauth",
        "models": [
            {"id": "gpt-4o", "label": "GPT-4o", "context": 128000, "modalities": ["text", "image", "audio"]},
            {"id": "gpt-4o-mini", "label": "GPT-4o mini", "context": 128000, "modalities": ["text", "image", "audio"]},
            {"id": "o3-mini", "label": "o3-mini", "context": 200000, "modalities": ["text"]},
            {"id": "claude-3.5-sonnet", "label": "Claude 3.5 Sonnet", "context": 200000, "modalities": ["text", "image"]},
            {"id": "gemini-2.0-flash-001", "label": "Gemini 2.0 Flash", "context": 1048576, "modalities": ["text", "image", "audio", "video"]},
        ],
    },
    # ── 国内服务商 ──────────────────────────────────────────────────────
    {
        "id": "glm",
        "label": "智谱 GLM (BigModel)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "docs": "https://docs.bigmodel.cn/cn/guide/models/text",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "glm-5.1", "label": "GLM-5.1", "context": 200000, "modalities": ["text"]},
            {"id": "glm-5", "label": "GLM-5", "context": 200000, "modalities": ["text"]},
            {"id": "glm-5-turbo", "label": "GLM-5-Turbo", "context": 200000, "modalities": ["text"]},
            {"id": "glm-4.7", "label": "GLM-4.7", "context": 200000, "modalities": ["text"]},
            {"id": "glm-4.6", "label": "GLM-4.6", "context": 200000, "modalities": ["text"]},
            {"id": "glm-4.7-flash", "label": "GLM-4.7-Flash (free)", "context": 200000, "modalities": ["text"]},
        ],
    },
    {
        "id": "kimi",
        "label": "Kimi (Moonshot)",
        "base_url": "https://api.moonshot.cn/v1",
        "docs": "https://platform.moonshot.cn/docs/intro",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "kimi-k2.6", "label": "Kimi K2.6", "context": 262144, "modalities": ["text", "image"]},
            {"id": "kimi-k2.5", "label": "Kimi K2.5", "context": 262144, "modalities": ["text", "image"]},
            {"id": "kimi-k2-0905-preview", "label": "Kimi K2 (preview)", "context": 262144, "modalities": ["text", "image"]},
            {"id": "kimi-k2-thinking", "label": "Kimi K2 Thinking", "context": 262144, "modalities": ["text"]},
        ],
    },
    {
        "id": "minimax",
        "label": "MiniMax",
        "base_url": "https://api.minimaxi.com/v1",
        "docs": "https://platform.minimaxi.com/document/Models",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "MiniMax-M2.7", "label": "MiniMax-M2.7", "context": 204800, "modalities": ["text"]},
            {"id": "MiniMax-M2.7-highspeed", "label": "MiniMax-M2.7-HighSpeed", "context": 204800, "modalities": ["text"]},
            {"id": "MiniMax-M2.5", "label": "MiniMax-M2.5", "context": 204800, "modalities": ["text"]},
            {"id": "MiniMax-M2.1", "label": "MiniMax-M2.1", "context": 204800, "modalities": ["text"]},
        ],
    },
    {
        "id": "alibaba",
        "label": "通义千问 (DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "docs": "https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "qwen3.6-max-preview", "label": "Qwen3.6-Max", "context": 256000, "modalities": ["text"]},
            {"id": "qwen3.6-plus", "label": "Qwen3.6-Plus", "context": 256000, "modalities": ["text"]},
            {"id": "qwen3.6-flash", "label": "Qwen3.6-Flash", "context": 256000, "modalities": ["text"]},
            {"id": "qwen-max", "label": "Qwen-Max", "context": 32768, "modalities": ["text"]},
            {"id": "qwen-plus", "label": "Qwen-Plus", "context": 131072, "modalities": ["text"]},
            {"id": "qwen-turbo", "label": "Qwen-Turbo", "context": 131072, "modalities": ["text"]},
            {"id": "qwen-long", "label": "Qwen-Long", "context": 10000000, "modalities": ["text"]},
        ],
    },
    {
        "id": "tencent",
        "label": "腾讯混元",
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "docs": "https://cloud.tencent.com/document/product/1729/104753",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "hy3-preview", "label": "Hy3 Preview", "context": 131072, "modalities": ["text"]},
            {"id": "hunyuan-t1", "label": "Hunyuan-T1", "context": 131072, "modalities": ["text"]},
            {"id": "hunyuan-turbo-s", "label": "Hunyuan-TurboS", "context": 131072, "modalities": ["text"]},
            {"id": "hunyuan-2.0-instruct", "label": "Tencent HY 2.0 Instruct", "context": 131072, "modalities": ["text"]},
            {"id": "hunyuan-2.0-think", "label": "Tencent HY 2.0 Think", "context": 131072, "modalities": ["text"]},
            {"id": "hunyuan-lite", "label": "Hunyuan-Lite (free)", "context": 262144, "modalities": ["text"]},
        ],
    },
    {
        "id": "xiaomi",
        "label": "Xiaomi MiMo",
        "base_url": "https://api.xiaomimimo.com/v1",
        "docs": "https://huggingface.co/XiaomiMiMo",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "mimo-7b-rl", "label": "MiMo 7B RL", "context": 32768, "modalities": ["text"]},
            {"id": "mimo-7b-base", "label": "MiMo 7B Base", "context": 32768, "modalities": ["text"]},
        ],
    },
    # ── Coding / agent specialists ───────────────────────────────────────
    {
        "id": "arcee",
        "label": "Arcee AI",
        "base_url": "https://api.arcee.ai/v1",
        "docs": "https://www.arcee.ai/",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "virtuoso-large", "label": "Virtuoso Large", "context": 128000, "modalities": ["text"]},
            {"id": "virtuoso-medium-v2", "label": "Virtuoso Medium v2", "context": 128000, "modalities": ["text"]},
            {"id": "caller-large", "label": "Caller Large", "context": 32000, "modalities": ["text"]},
            {"id": "spotlight", "label": "Spotlight (vision)", "context": 32000, "modalities": ["text", "image"]},
            {"id": "maestro-reasoning", "label": "Maestro Reasoning", "context": 64000, "modalities": ["text"]},
        ],
    },
    {
        "id": "gmi",
        "label": "GMI Cloud",
        "base_url": "https://api.gmi-serving.com/v1",
        "docs": "https://docs.gmicloud.ai/",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "deepseek-ai/DeepSeek-V3", "label": "DeepSeek V3", "context": 128000, "modalities": ["text"]},
            {"id": "deepseek-ai/DeepSeek-R1", "label": "DeepSeek R1", "context": 128000, "modalities": ["text"]},
            {"id": "meta-llama/Llama-3.3-70B-Instruct", "label": "Llama 3.3 70B Instruct", "context": 131072, "modalities": ["text"]},
        ],
    },
    {
        "id": "kilocode",
        "label": "Kilo Code",
        "base_url": "https://kilocode.ai/api/openrouter",
        "docs": "https://kilocode.ai/docs/",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "anthropic/claude-sonnet-4.6", "label": "Claude Sonnet 4.6", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "anthropic/claude-opus-4.7", "label": "Claude Opus 4.7", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "openai/gpt-5.5", "label": "GPT-5.5", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "openai/gpt-5.4", "label": "GPT-5.4", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "google/gemini-2.5-pro", "label": "Gemini 2.5 Pro", "context": 2097152, "modalities": ["text", "image", "audio", "video"]},
            {"id": "google/gemini-3.5-flash", "label": "Gemini 3.5 Flash", "context": 1048576, "modalities": ["text", "image", "audio", "video"]},
        ],
    },
    {
        "id": "opencode-zen",
        "label": "OpenCode Zen",
        "base_url": "https://opencode.ai/zen/v1",
        "docs": "https://opencode.ai/docs/zen/",
        "allow_custom": True,
        "auth": "api_key",
        "models": [
            {"id": "grok-code", "label": "Grok Code", "context": 256000, "modalities": ["text"]},
            {"id": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6", "context": 1000000, "modalities": ["text", "image"]},
            {"id": "qwen3-coder", "label": "Qwen3 Coder", "context": 256000, "modalities": ["text"]},
            {"id": "kimi-k2.6", "label": "Kimi K2.6", "context": 262144, "modalities": ["text", "image"]},
        ],
    },
    {
        "id": "opencode-go",
        "label": "OpenCode Go",
        "base_url": "https://opencode.ai/go/v1",
        "docs": "https://opencode.ai/docs/",
        "allow_custom": True,
        "auth": "api_key",
        "models": [],
    },
    # ── Meta backends (failover / router) ────────────────────────────────
    {
        "id": "failover",
        "label": "Failover (composite)",
        "base_url": "",
        "docs": "",
        "allow_custom": True,
        "auth": "none",
        "models": [],
    },
    {
        "id": "router",
        "label": "Router (composite)",
        "base_url": "",
        "docs": "",
        "allow_custom": True,
        "auth": "none",
        "models": [],
    },
]


# Helper: lookup by id ───────────────────────────────────────────────────

def get_provider(provider_id: str) -> dict[str, Any] | None:
    """Return the catalog entry for ``provider_id`` or None if unknown."""
    for p in PROVIDERS:
        if p["id"] == provider_id:
            return p
    return None


def get_model(provider_id: str, model_id: str) -> dict[str, Any] | None:
    """Return the model entry, or None if the model isn't in the catalog."""
    provider = get_provider(provider_id)
    if not provider:
        return None
    for m in provider.get("models", []):
        if m["id"] == model_id:
            return m
    return None


def default_output_tokens(provider_id: str) -> int:
    """Return a safe default for the per-turn output token budget."""
    return DEFAULT_MAX_OUTPUT_TOKENS.get(provider_id, 8192)


def catalog_payload() -> dict[str, Any]:
    """Serializable snapshot used by the frontend model form."""
    return {
        "providers": PROVIDERS,
        "default_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
    }


__all__ = [
    "PROVIDERS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "get_provider",
    "get_model",
    "default_output_tokens",
    "catalog_payload",
]
