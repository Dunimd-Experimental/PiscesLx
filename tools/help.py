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
    print("  train      - Train the model")
    print("  infer      - Run inference with a trained model")
    print("  check      - Check GPU and dependencies")
    print("  monitor    - System monitor (GPU/CPU/memory)")
    print("  download   - Download datasets for training")
    print("  arrow      - Arrow/JSON dataset conversion")
    print("  quantize   - Model quantization (8-bit or 4-bit)")
    print("  benchmark       Model evaluation & benchmarking (26 standardized benchmarks)")
    print("  agent           Native agent interface for reasoning and tool use")
    print("  help            Show this help message")
    print("\nExample usage:")
    print("  python manage.py train")
    print("  python manage.py infer --ckpt ckpt/model.pt --prompt 'Hello'")
    print("  python manage.py quantize --ckpt model.pt --save quantized.pt --bits 8")
    print("  python manage.py benchmark --list")
    print("  python manage.py benchmark --info mmlu")
    print("  python manage.py benchmark --benchmark mmlu --config configs/7B.json")
    print("  python manage.py benchmark --perf --config configs/7B.json --seq_len 4096")
    print("  python manage.py help")
    print("\nAgent examples:")
    print("  python manage.py agent --interactive --config configs/7B.json")
    print("  python manage.py agent --task 'Calculate 15*23 and save to file' --config configs/7B.json --max-steps 3")
    print("  python manage.py agent --interactive --config configs/7B.json --model path/to/model.pth")