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

import os
import sys
import json
import argparse
from pathlib import Path
from utils.log import RIGHT, DEBUG, ERROR
from configs.version import VERSION

# Get the root directory of the project
ROOT = os.path.abspath(os.path.dirname(__file__))

# List of available commands
COMMANDS = [
    'setup',      # Environment setup
    'source',     # Activate virtual environment
    'update',       # Pull latest code from remote
    'train',      # Train model
    'infer',      # Inference
    'check',      # Check GPU/deps
    'monitor',    # System monitor
    'download',   # Download datasets
    'quantize',   # Model quantization
    'benchmark',  # Model evaluation & benchmarking
    'agent',      # Native agent interface
    'rlhf',       # RLHF training
    'help',       # Show help for commands
    'dataset',    # Dataset management tool
]

def main():
    """
    Main function to handle command-line arguments and execute corresponding commands.
    This function parses the command-line arguments and calls the appropriate function
    based on the specified command.
    """
    parser = argparse.ArgumentParser(description="Pisces L1 Management Tool (manage.py)")
    parser.add_argument('command', nargs='?', choices=COMMANDS, help="Command to execute")
    parser.add_argument('--ckpt', default='', help='Checkpoint file (for infer)')
    parser.add_argument('--prompt', default='Hello, please introduce yourself', help='Prompt (for infer)')
    parser.add_argument('--image', default='', help='Image path (for infer)')
    parser.add_argument('--max_samples', type=int, default=50000, help='Max samples per dataset (for download)')
    parser.add_argument('--speculative', action='store_true', help='Enable speculative decoding for faster inference')
    parser.add_argument('--draft_model', type=str, default='', help='Path to draft model for speculative decoding')
    parser.add_argument('--spec_gamma', type=int, default=4, help='Number of speculative tokens per step (default: 4)')
    parser.add_argument('--save', default='', help='[quantize] Output path for quantized model')
    parser.add_argument('--bits', type=int, default=8, help='[quantize] Quantization bits (4 or 8)')
    parser.add_argument('--config', default='configs/0.5B.json', help='[benchmark] Model config path')
    parser.add_argument('--seq_len', type=int, default=512, help='[benchmark] Sequence length for benchmark')
    parser.add_argument('--list', action='store_true', help='[benchmark] List all available benchmarks')
    parser.add_argument('--info', type=str, help='[benchmark] Show detailed info about a benchmark')
    parser.add_argument('--benchmark', type=str, help='[benchmark] Run specific benchmark evaluation')
    parser.add_argument('--model', type=str, help='[benchmark] Model checkpoint path')
    parser.add_argument('--perf', action='store_true', help='[benchmark] Run performance benchmark')
    parser.add_argument('--model_size', default='0.5B', type=str, help='Model size, e.g. 0.5B, 1.5B, 7B, 70B, 128B')
    parser.add_argument('--resume_ckpt', default='', type=str, help='Path to checkpoint to resume training')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate after resuming checkpoint')
    parser.add_argument('--quant', action='store_true', help='Force enable quantization (4-bit)')
    parser.add_argument('--no_quant', action='store_true', help='Force disable quantization')
    parser.add_argument('--task', default='', help='[agent] Task to execute')
    parser.add_argument('--interactive', action='store_true', help='[agent] Interactive agent mode')
    parser.add_argument('--max-steps', type=int, default=5, help='[agent] Maximum agent steps')
    parser.add_argument('--rlhf', action='store_true', help='Enable RLHF (Reinforcement Learning from Human Feedback)')
    parser.add_argument('--rlhf_dataset', type=str, default='dunimd/human_feedback', help='RLHF human feedback dataset')
    parser.add_argument('--rlhf_lr', type=float, default=1e-5, help='RLHF learning rate')
    parser.add_argument('--rlhf_batch_size', type=int, default=4, help='RLHF batch size')
    parser.add_argument('--rlhf_mini_batch_size', type=int, default=1, help='RLHF mini-batch size')
    parser.add_argument('--rlhf_accum_steps', type=int, default=4, help='RLHF gradient accumulation steps')
    parser.add_argument('--rlhf_epochs', type=int, default=3, help='RLHF training epochs')
    parser.add_argument('--rlhf_max_samples', type=int, default=1000, help='RLHF maximum number of samples')
    parser.add_argument('--rlhf_max_length', type=int, default=512, help='RLHF maximum sequence length')
    args = parser.parse_args()
    
    RIGHT("Versione del modello PiscesL1: " + VERSION)
    
    if args.command is None or args.command == 'help':
        from tools.help import help
        help()
    elif args.command == 'train':
        from tools.train import train
        train(args)
    elif args.command == 'infer':
        from tools.infer import infer
        infer(args)
    elif args.command == 'check':
        from tools.check import check
        check(args)
    elif args.command == 'monitor':
        from tools.monitor import monitor
        monitor()
    elif args.command == 'download':
        from data.download import download_datasets, optimize_datasets
        download_datasets(args.max_samples)
        optimize_datasets(max_keep=5000)
    elif args.command == 'dataset':
        from tools.dataset import dataset
        dataset(args)
    elif args.command == 'setup':
        from tools.setup import setup
        setup(args)
    elif args.command == 'source':
        from tools.source import source
        source()
    elif args.command == 'update':
        from tools.update import update
        update()
    elif args.command == 'quantize':
        from tools.quantize import quantize
        if not args.ckpt or not args.save:
            ERROR("quantize requires --ckpt and --save arguments")
            sys.exit(1)
        quantize(args.ckpt, args.save, args.bits)
    elif args.command == 'benchmark':
        from tools.benchmark import list_benchmarks, benchmark_info, performance_benchmark, run_benchmark
        
        if args.list:
            list_benchmarks()
        elif args.info:
            benchmark_info(args.info)
        elif args.perf:
            performance_benchmark(args.config, args.seq_len, args.model)
        elif args.benchmark:
            run_benchmark(args.benchmark, args.model, args.config)
        else:
            performance_benchmark(args.config, args.seq_len)
    elif args.command == 'rlhf':
        from tools.rlhf import rlhf_train

        if not hasattr(args, 'model_path') or not args.model_path:
            args.model_path = f"configs/{args.model_size}"
        
        rlhf_train(args)
        RIGHT("RLHF training completed!")
    else:
        ERROR(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()