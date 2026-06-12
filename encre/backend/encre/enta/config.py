#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EnTA Backend Configuration — entirely environment-driven.

Every teacher model is configured through environment variables.
No hardcoded API keys, URLs, or model names in the code.

Convention:
    ENTA_{MODEL_NAME}_BACKEND_TYPE   — backend type: openai, anthropic, deepseek, etc.
    ENTA_{MODEL_NAME}_API_KEY        — API key
    ENTA_{MODEL_NAME}_BASE_URL       — base URL (optional, defaults per backend)
    ENTA_{MODEL_NAME}_MODEL          — model name for API calls

    Where {MODEL_NAME} is any identifier with underscores.
    The model name used in EnTA is the lowercase version.

Example for a model called "deepseek-r1":
    ENTA_DEEPSEEK_R1_BACKEND_TYPE=deepseek
    ENTA_DEEPSEEK_R1_API_KEY=sk-xxx
    ENTA_DEEPSEEK_R1_BASE_URL=https://api.deepseek.com/v1
    ENTA_DEEPSEEK_R1_MODEL=deepseek-reasoner

    → EnTA refers to it as "deepseek-r1"

Example for "my-custom-model":
    ENTA_MY_CUSTOM_MODEL_BACKEND_TYPE=openai
    ENTA_MY_CUSTOM_MODEL_API_KEY=sk-xxx
    ENTA_MY_CUSTOM_MODEL_BASE_URL=https://my-api.com/v1
    ENTA_MY_CUSTOM_MODEL_MODEL=my-model-name

    → EnTA refers to it as "my-custom-model"
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

# Pattern: ENTA_{NAME}_FIELD → value
_ENV_PATTERN = re.compile(r"^ENTA_([A-Z0-9_]+)_(BACKEND_TYPE|API_KEY|BASE_URL|MODEL)$")


def _discover_models_from_env() -> Dict[str, Dict[str, str]]:
    """Scan environment variables for ENTA_* pattern and build config dict.

    Returns:
        Dict mapping model names (lowercase, hyphenated) to their config dicts.
    """
    models: Dict[str, Dict[str, str]] = {}

    for env_key, env_val in os.environ.items():
        match = _ENV_PATTERN.match(env_key)
        if not match:
            continue

        raw_name = match.group(1)      # e.g. "DEEPSEEK_R1"
        field = match.group(2)         # e.g. "API_KEY"

        # Normalize name: lowercase, underscores → hyphens
        model_name = raw_name.lower().replace("_", "-")

        if model_name not in models:
            models[model_name] = {}

        models[model_name][field.lower()] = env_val

    return models


# Auto-discovered models from environment
_MODEL_CONFIG = _discover_models_from_env()


def get_model_backend_config(model_name: str) -> Dict[str, str]:
    """Get backend config for a model name, resolved from env.

    Args:
        model_name: e.g. "deepseek-r1", "my-custom-model".

    Returns:
        Dict with backend_type, api_key, base_url, model.

    Raises:
        KeyError: If model_name is not configured in environment.
    """
    config = _MODEL_CONFIG.get(model_name)
    if config is None:
        available = list(_MODEL_CONFIG.keys())
        raise KeyError(
            f"Model '{model_name}' not configured in environment. "
            f"Set ENTA_{model_name.upper().replace('-', '_')}_BACKEND_TYPE=... "
            f"and ENTA_{model_name.upper().replace('-', '_')}_API_KEY=...\n"
            f"Currently configured: {available if available else '(none)'}"
        )

    required = {"backend_type", "api_key"}
    missing = required - set(config.keys())
    if missing:
        env_prefix = model_name.upper().replace("-", "_")
        env_vars = [f"ENTA_{env_prefix}_{k.upper()}" for k in missing]
        raise KeyError(
            f"Model '{model_name}' missing required env vars: {', '.join(env_vars)}"
        )

    return config


def create_backend_for_model(model_name: str) -> Any:
    """Create an Encre backend instance from environment configuration.

    Args:
        model_name: e.g. "deepseek-r1".

    Returns:
        Initialized backend instance.
    """
    from encre.backend import create_backend

    cfg = get_model_backend_config(model_name)
    return create_backend(
        cfg["backend_type"],
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url", ""),
        model=cfg.get("model", ""),
    )


def list_configured_models() -> Dict[str, str]:
    """List all models found in environment with their backend type.

    Returns:
        Dict of model_name → backend_type.
    """
    return {
        name: cfg.get("backend_type", "unknown")
        for name, cfg in _MODEL_CONFIG.items()
    }
