#!/usr/bin/env/python3

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
from configs.version import VERSION
from utils import RIGHT, DEBUG, ERROR

ROOT = os.path.abspath(os.path.dirname(__file__))

# List of available commands with their purposes
COMMANDS = [
    'setup',      # Set up the environment
    'source',     # Activate the virtual environment
    'update',     # Pull the latest code from the remote repository
    'version',    # Display the current version and changelog
    'changelog',  # Show the version history and specific version changelog
    'train',      # Train the model
    'infer',      # Perform inference with MCP integration
    'check',      # Check GPU and dependencies
    'monitor',    # Monitor the system
    'download',   # Download datasets
    'quantize',   # Quantize the model
    'benchmark',  # Evaluate and benchmark the model
    'mcp',        # Perform MCP server operations
    'rlhf',       # Conduct RLHF training
    'help',       # Show help information for commands
    'dataset',    # Manage datasets
    'watermark',  # Detect and manage watermarks
    'cache',      # Maintain the cache
]

def main():
    """
    Main function to handle command-line arguments and execute corresponding commands.
    This function parses the command-line arguments and calls the appropriate function
    based on the specified command.
    
    Args:
        None
    
    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Pisces L1 Management Tool (manage.py)")
    parser.add_argument('command', nargs='?', choices=COMMANDS, help="Command to execute")
    parser.add_argument('--ckpt', default='', help='Checkpoint file (for inference)')
    parser.add_argument('--prompt', default='Hello, please introduce yourself', help='Prompt (for inference)')
    parser.add_argument('--image', default='', help='Image path (for inference)')
    parser.add_argument('--max_samples', type=int, default=50000, help='Maximum number of samples per dataset (for download)')
    parser.add_argument('--speculative', action='store_true', help='Enable speculative decoding for faster inference')
    parser.add_argument('--draft_model', type=str, default='', help='Path to the draft model for speculative decoding')
    parser.add_argument('--spec_gamma', type=int, default=4, help='Number of speculative tokens per step (default: 4)')
    parser.add_argument('--save', default='', help='Output path for the quantized model (for quantization)')
    parser.add_argument('--bits', type=int, default=8, help='Quantization bits (4 or 8) (for quantization)')
    parser.add_argument('--config', default='configs/0.5B.json', help='Model configuration path (for benchmarking)')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length for benchmarking (for benchmarking)')
    parser.add_argument('--list', action='store_true', help='List all available benchmarks (for benchmarking)')
    parser.add_argument('--info', type=str, help='Show detailed information about a benchmark (for benchmarking)')
    parser.add_argument('--benchmark', type=str, help='Run a specific benchmark evaluation (for benchmarking)')
    parser.add_argument('--model', type=str, help='Model checkpoint path (for benchmarking)')
    parser.add_argument('--perf', action='store_true', help='Run a performance benchmark (for benchmarking)')
    parser.add_argument('--model_size', default='0.5B', type=str, help='Model size, e.g., 0.5B, 1.5B, 7B, 70B, 128B')
    parser.add_argument('--dataset', default='Chinese2', type=str, help='Dataset name for training')
    parser.add_argument('--resume_ckpt', default='', type=str, help='Path to the checkpoint to resume training')
    parser.add_argument('--reset_lr', action='store_true', help='Reset the learning rate after resuming from a checkpoint')
    parser.add_argument('--quant', action='store_true', help='Force enable 4-bit quantization')
    parser.add_argument('--no_quant', action='store_true', help='Force disable quantization')
    parser.add_argument('--force_quant', action='store_true', help='Override the configuration to force enable quantization')
    parser.add_argument('--force_lora', action='store_true', help='Override the configuration to force enable LoRA')
    parser.add_argument('--quant_bits', type=int, choices=[2, 4, 8], default=4, help='Quantization bits: 2, 4, or 8 (default: 4)')
    parser.add_argument('--rlhf', action='store_true', help='Enable RLHF (Reinforcement Learning from Human Feedback)')
    parser.add_argument('--rlhf_dataset', type=str, default='dunimd/human_feedback', help='RLHF human feedback dataset')
    parser.add_argument('--rlhf_lr', type=float, default=1e-5, help='RLHF learning rate')
    parser.add_argument('--rlhf_batch_size', type=int, default=4, help='RLHF batch size')
    parser.add_argument('--rlhf_mini_batch_size', type=int, default=1, help='RLHF mini-batch size')
    parser.add_argument('--rlhf_accum_steps', type=int, default=4, help='RLHF gradient accumulation steps')
    parser.add_argument('--rlhf_epochs', type=int, default=3, help='RLHF training epochs')
    parser.add_argument('--rlhf_max_samples', type=int, default=1000, help='RLHF maximum number of samples')
    parser.add_argument('--rlhf_max_length', type=int, default=512, help='RLHF maximum sequence length')
    parser.add_argument('--mcp_host', type=str, default='localhost', help='MCP server host (for MCP operations)')
    parser.add_argument('--mcp_port', type=int, default=8080, help='MCP server port (for MCP operations)')
    parser.add_argument('--mcp_action', type=str, choices=['status', 'warmup', 'refresh-cache'], default='status', help='MCP action to perform (for MCP operations)')
    parser.add_argument('--text', type=str, help='Text content to check for watermark (for watermark detection)')
    parser.add_argument('--file', type=str, help='File path to check for watermark (for watermark detection)')
    parser.add_argument('--batch', action='store_true', help='Enable batch mode for directory processing (for watermark detection)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output (for watermark detection)')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format (for watermark detection)')
    parser.add_argument('--cache_action', type=str, choices=['stats', 'clear-all', 'clear-dataset', 'clear-downloads'], default='stats', help='Cache action to perform: stats, clear-all, clear-dataset, or clear-downloads (for cache maintenance)')

    args, unknown = parser.parse_known_args()
    
    # Display version information if not running the version or changelog command
    if args.command not in ['version', 'changelog']:
        RIGHT("PiscesL1 Model Version: " + VERSION)
    
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
    elif args.command == 'version':
        from tools.version import show_version
        show_version()
    elif args.command == 'changelog':
        from tools.changelog import show_changelog, parse_changelog_args
        
        # Parse changelog-specific arguments from unknown arguments
        changelog_parser = parse_changelog_args()
        
        try:
            changelog_args = changelog_parser.parse_args(unknown)
            show_changelog(changelog_args)
        except SystemExit:
            # argparse calls sys.exit() on help or error
            pass
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
    elif args.command == 'mcp':
        from tools.mcp import status as mcp_status, background_refresh, read_config, write_config, discover_tools, merge_to_config
        if args.mcp_action == 'status':
            RIGHT("MCP config status")
            s = mcp_status()
            print(json.dumps(s, indent=2, ensure_ascii=False))
        elif args.mcp_action == 'warmup':
            RIGHT("Starting MCP background discovery (non-blocking)...")
            background_refresh()
            RIGHT("MCP background discovery started")
        elif args.mcp_action == 'refresh-cache':
            RIGHT("Refreshing MCP tools cache (blocking)...")
            cfg = read_config()
            discovered = discover_tools()
            merged = merge_to_config(cfg, discovered)
            write_config(merged)
            RIGHT("MCP tools cache refreshed")
    elif args.command == 'rlhf':
        from tools.rlhf import rlhf_train

        if not hasattr(args, 'model_path') or not args.model_path:
            args.model_path = f"configs/{args.model_size}"
        
        rlhf_train(args)
        RIGHT("RLHF training completed!")
    elif args.command == 'watermark':
        from tools.watermark_check import detect_watermark, batch_detect
        
        if args.file:
            if args.batch:
                result = batch_detect(args.file, args.verbose)
            else:
                with open(args.file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                result = detect_watermark(content, args.verbose)
        elif args.text:
            result = detect_watermark(args.text, args.verbose)
        else:
            ERROR("Watermark command requires --text or --file argument")
            sys.exit(1)
            
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == 'cache':
        from tools.cache import cache_stats, clear_all_cache, clear_dataset_cache, clear_downloads_cache
        if args.cache_action == 'stats':
            s = cache_stats()
            print(json.dumps(s, ensure_ascii=False, indent=2))
        elif args.cache_action == 'clear-all':
            clear_all_cache()
        elif args.cache_action == 'clear-dataset':
            clear_dataset_cache()
        elif args.cache_action == 'clear-downloads':
            clear_downloads_cache()

    else:
        ERROR(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()