#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def help():
    print("Pisces L1 Management Tool Help\n")
    print("Available commands:")
    print("  setup      - Environment setup and dependency installation")
    print("  source     - Activate virtual environment")
    print("  update     - Pull latest code from remote repository")
    print("  version    - Show current version and changelog")
    print("  changelog  - Show version history (--all for all, --version X.X.XXXX for specific)")
    print("  train      - Train the model")
    print("  infer      - Run inference with integrated MCP server and tool support")
    print("  check      - Check GPU and dependencies")
    print("  monitor    - System monitor (GPU/CPU/memory)")
    print("  download   - Download datasets for training")
    print("  dataset    - Arrow/JSON dataset conversion")
    print("  quantize   - Model quantization (8-bit or 4-bit)")
    print("  benchmark  - Model evaluation & benchmarking (26 standardized benchmarks)")
    print("  mcp        - MCP server management (start/stop/status/test)")
    print("  rlhf       - RLHF (Reinforcement Learning from Human Feedback) training")
    print("  help       - Show this help message")
    print("\nExample usage:")
    print("  python manage.py setup")
    print("  python manage.py update")
    print("  python manage.py version")
    print("  python manage.py changelog --all")
    print("  python manage.py changelog --version 1.0.0150")
    print("  python manage.py download")
    print("  python manage.py train --model_size 0.5B --dataset Chinese2")
    print("  python manage.py infer --ckpt ckpt/model.pt --prompt 'Hello' --image path/to/image.jpg")
    print("  python manage.py quantize --ckpt model.pt --save quantized.pt --bits 8")
    print("  python manage.py benchmark --list")
    print("  python manage.py benchmark --info mmlu")
    print("  python manage.py benchmark --benchmark mmlu --config configs/7B.json")
    print("  python manage.py benchmark --perf --config configs/7B.json --seq_len 4096")
    print("\nMCP Server examples:")
    print("  python manage.py mcp --mcp_action start")
    print("  python manage.py mcp --mcp_action status")
    print("  python manage.py mcp --mcp_action test")
    print("  python manage.py mcp --mcp_action stop")
    print("\nInference with MCP (Agent calls via <agent> tags):")
    print("  python manage.py infer --prompt 'Search for latest AI news <agent><an>web_search</an><ap1>latest AI news</ap1><ap2>5</ap2></agent>'")
    print("  python manage.py infer --prompt 'Calculate result: <agent><an>calculator</an><ap1>2 + 3 * 4</ap1></agent>'")
    print("  python manage.py infer --prompt 'Analyze this image <agent><an>image_analysis</an><ap1>path/to/image.jpg</ap1><ap2>description</ap2></agent>' --image path/to/image.jpg")
    print("\nRLHF examples:")
    print("  python manage.py rlhf --model_size 1.5B --rlhf_dataset dunimd/human_feedback")