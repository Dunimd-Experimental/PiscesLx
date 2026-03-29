#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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

"""
Default configuration templates for Xi Studio.
"""

DEFAULT_XI_TOML = '''# Xi Studio Configuration
# This file configures paths, API settings, and UI preferences

[project]
name = "piscesl1"
version = "1.0.0"
backend = "piscesl1"
description = "PiscesL1 LLM Training Project"
author = ""

[paths]
root = "."
models = ".pisceslx/models"
checkpoints = ".pisceslx/checkpoints"
data = ".pisceslx/data"
outputs = ".pisceslx/outputs"
logs = ".pisceslx/logs"
cache = ".pisceslx/cache"
temp = ".pisceslx/temp"
configs = "configs"

[api]
host = "127.0.0.1"
port = 3140
cors_origins = ["http://localhost:3000"]
timeout = 120
max_workers = 4

[ui]
theme = "system"
language = "en"
sidebar_collapsed = false

[notifications]
enabled = true
retention_days = 30
max_count = 1000
sound = false
'''

DEFAULT_TRAIN_TOML = '''# Training Command Configuration

[train]
executable = "python"
script = "manage.py"
args = ["train"]
env = { CUDA_VISIBLE_DEVICES = "0" }
cwd = "${paths.root}"
timeout = 86400
background = true

[train.defaults]
epochs = 10
batch_size = 32
learning_rate = 0.001
warmup_steps = 100
save_steps = 500
eval_steps = 500
'''

DEFAULT_INFERENCE_TOML = '''# Inference Command Configuration

[inference]
executable = "python"
script = "manage.py"
args = ["serve"]
env = {}
cwd = "${paths.root}"
timeout = 60
background = true

[inference.defaults]
host = "127.0.0.1"
port = 8000
max_length = 2048
temperature = 0.7
top_p = 0.9
'''

DEFAULT_BENCHMARK_TOML = '''# Benchmark Command Configuration

[benchmark]
executable = "python"
script = "manage.py"
args = ["benchmark"]
env = {}
cwd = "${paths.root}"
timeout = 3600
background = true

[benchmark.defaults]
batch_sizes = [1, 2, 4, 8, 16]
seq_lengths = [128, 256, 512, 1024, 2048]
warmup_runs = 3
measure_runs = 10
'''

DEFAULT_DOWNLOAD_TOML = '''# Download Command Configuration

[download]
executable = "python"
script = "manage.py"
args = ["download"]
env = {}
cwd = "${paths.root}"
timeout = 86400
background = true

[download.defaults]
source = "huggingface"
force = false
verify = true
'''

DEFAULT_SCHEMA_JSON = '''{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Xi Configuration Schema",
  "description": "Schema for Xi Studio configuration file",
  "type": "object",
  "required": ["project"],
  "properties": {
    "project": {
      "type": "object",
      "required": ["name", "backend"],
      "properties": {
        "name": {
          "type": "string",
          "description": "Project name"
        },
        "version": {
          "type": "string",
          "default": "1.0.0"
        },
        "backend": {
          "type": "string",
          "enum": ["piscesl1", "custom"],
          "default": "piscesl1"
        },
        "description": {
          "type": "string"
        },
        "author": {
          "type": "string"
        }
      }
    },
    "paths": {
      "type": "object",
      "properties": {
        "root": {
          "type": "string",
          "default": "."
        },
        "models": {
          "type": "string"
        },
        "checkpoints": {
          "type": "string"
        },
        "data": {
          "type": "string"
        },
        "outputs": {
          "type": "string"
        },
        "logs": {
          "type": "string"
        },
        "cache": {
          "type": "string"
        },
        "temp": {
          "type": "string"
        },
        "configs": {
          "type": "string"
        }
      }
    },
    "api": {
      "type": "object",
      "properties": {
        "host": {
          "type": "string",
          "default": "127.0.0.1"
        },
        "port": {
          "type": "integer",
          "default": 3140
        },
        "cors_origins": {
          "type": "array",
          "items": { "type": "string" }
        },
        "timeout": {
          "type": "integer",
          "default": 120
        },
        "max_workers": {
          "type": "integer",
          "default": 4
        }
      }
    },
    "ui": {
      "type": "object",
      "properties": {
        "title": {
          "type": "string",
          "default": "Xi Studio"
        },
        "logo": {
          "type": "string"
        },
        "theme": {
          "type": "string",
          "enum": ["light", "dark", "system"],
          "default": "system"
        },
        "language": {
          "type": "string",
          "default": "en"
        },
        "sidebar_collapsed": {
          "type": "boolean",
          "default": false
        }
      }
    },
    "notifications": {
      "type": "object",
      "properties": {
        "enabled": {
          "type": "boolean",
          "default": true
        },
        "retention_days": {
          "type": "integer",
          "default": 30
        },
        "max_count": {
          "type": "integer",
          "default": 1000
        },
        "sound": {
          "type": "boolean",
          "default": false
        }
      }
    }
  }
}
'''

__all__ = [
    "DEFAULT_XI_TOML",
    "DEFAULT_TRAIN_TOML",
    "DEFAULT_INFERENCE_TOML",
    "DEFAULT_BENCHMARK_TOML",
    "DEFAULT_DOWNLOAD_TOML",
    "DEFAULT_SCHEMA_JSON",
]
