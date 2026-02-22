#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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
PiscesL1 Management Tool - Command-Line Interface Entry Point.

This module serves as the primary entry point for the PiscesL1 Large Language
Model framework's command-line interface. It provides a unified interface for
all framework operations including training, backend inference service, 
benchmarking, monitoring, and system management.

Architecture Overview:
    The management tool follows a command-dispatch pattern where:
    
    1. Command Parsing: argparse processes CLI arguments into a namespace object
    2. Command Dispatch: The main() function routes to appropriate handlers
    3. Orchestrator Execution: Each command delegates to a dedicated orchestrator
    4. Result Reporting: Results are formatted and returned to the user

Available Commands:
    **Core Commands:**
        - setup: Initialize environment and install dependencies
        - help: Display comprehensive usage documentation
    
    **Training Commands:**
        - train: Train models with various configurations
          * Standard training with configurable model sizes
          * Quantization-aware training (2/4/8-bit)
          * RLHF (Reinforcement Learning from Human Feedback)
          * LoRA fine-tuning support
          * Checkpoint resumption with learning rate reset
    
    **Backend Service Commands:**
        - serve: Start OpenAI-compatible backend inference service
          * Multi-size model support (0.5B / 1B / 7B / 14B / 72B / 671B / 1T)
          * OpenAI-compatible API endpoints
          * Agent XML pattern interception (<ag>, <swarm>, <orchestrate>, <tool>)
          * Streaming response support (SSE)
          * Embedding generation
          * Image/video generation endpoints
          * OPSS integration (MCP Plaza, Swarm, Orchestrator)
    
    **Background Management:**
        - action: Manage background training/inference processes
          * Submit and track runs across terminals
          * Real-time status monitoring
          * Process control (pause/resume/cancel)
          * Log retrieval and management
    
    **Dataset Commands:**
        - download: Download and prepare training datasets
    
    **Benchmark Commands:**
        - benchmark: Evaluate model performance
          * Standard benchmarks (MMLU, etc.)
          * Performance benchmarking
          * Self-test mode
    
    **Monitor Commands:**
        - check: Validate GPU and system dependencies
        - monitor: Real-time system monitoring
    
    **MCP Commands:**
        - mcp: Model Context Protocol tool management
    
    **Watermark Commands:**
        - watermark: Detect and verify watermarks
          * Text watermark detection
          * Image/audio/video watermark detection
          * Model weight verification
    
    **Cache Commands:**
        - cache: Manage framework cache directories

Configuration System:
    The tool supports multiple configuration layers:
    
    1. Command-line arguments (highest priority)
    2. Configuration files (JSON/YAML)
    3. Default values (lowest priority)
    
    Configuration files can be specified via --train_config and --serve_config
    options for training and serving respectively.

Error Handling:
    All commands implement comprehensive error handling:
    - Invalid arguments are caught and reported with usage hints
    - Missing dependencies are detected with installation instructions
    - Runtime errors are logged with context for debugging
    - Exit codes indicate success (0) or failure (1)

Logging:
    The tool uses structured logging via PiscesLxLogger:
    - All operations are logged with timestamps and context
    - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    - Logs are written to both console and file

Usage Examples:
    Basic setup:
        $ python manage.py setup
    
    Training a model:
        $ python manage.py train --model_size 7B --dataset Chinese2
    
    Starting backend service:
        $ python manage.py serve --model_size 7B --port 8000
    
    Running benchmarks:
        $ python manage.py benchmark --benchmark mmlu --model model.pt

Integration Points:
    - tools/: All tool modules for command implementation
    - opss/: Operations modules for advanced functionality
    - model/: Model definitions and configurations
    - configs/: Configuration files for models and datasets
    - utils/: Utility functions and logging

Thread Safety:
    The management tool is designed for single-threaded command execution.
    Background processes are managed separately via the action command.

Exit Codes:
    - 0: Success
    - 1: Error (invalid arguments, runtime failure, etc.)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from configs.version import CVERSION


def _import_yaml():
    global yaml
    import yaml
    return yaml


ROOT = os.path.abspath(os.path.dirname(__file__))


def _get_logger():
    """
    Get or create the logger instance for the management tool.
    
    This function lazily initializes the logger to avoid import issues
    during early startup. The logger is used for structured logging
    throughout the management tool.
    
    Args:
        None
    
    Returns:
        PiscesLxLogger: A configured logger instance for the 'pisceslx.manage'
            namespace.
    
    Note:
        The logger is created on first call and cached for subsequent calls.
        This lazy initialization prevents import errors when the utils module
        is not yet available.
    """
    from utils.dc import PiscesLxLogger
    from utils.paths import get_log_file
    return PiscesLxLogger("PiscesLx.Manage", file_path=get_log_file("PiscesLx.Manage"), enable_file=True)


COMMANDS = [
    'setup',
    'train',
    'serve',
    'check',
    'monitor',
    'download',
    'benchmark',
    'mcp',
    'help',
    'watermark',
    'action',
]


def main():
    """
    Main entry point for the PiscesL1 management tool.
    
    This function serves as the primary dispatcher for all CLI commands. It
    parses command-line arguments, validates the requested command, and
    delegates execution to the appropriate handler module.
    
    The function follows this execution flow:
    
    1. **Argument Parsing**: Uses argparse to parse all CLI arguments into
       a namespace object. Unknown arguments are captured for pass-through
       to sub-commands (e.g., action command).
    
    2. **Command Dispatch**: Routes to the appropriate handler based on
       the command name:
       
       - help/setup/check: Direct function calls
       - train/serve/monitor/benchmark: Orchestrator pattern
       - action: CLI sub-command pattern
       - download: Tool class pattern
       - mcp/watermark/cache: Specialized handlers
    
    3. **Error Handling**: Catches and logs errors, returning appropriate
       exit codes for scripting integration.
    
    Command Handlers:
        Each command is handled by a dedicated module:
        
        - setup: tools/setup.py - Environment initialization
        - train: tools/train/orchestrator.py - Training pipeline
        - serve: tools/infer/server.py - OpenAI-compatible backend service
        - check: tools/check.py - System validation
        - monitor: tools/monitor/orchestrator.py - System monitoring
        - download: tools/data/download/ - Dataset management
        - benchmark: tools/benchmark/orchestrator.py - Evaluation
        - mcp: opss/mcp/mcps/ - MCP tool management
        - watermark: tools/wmc/ - Watermark detection
        - cache: tools/cache/ - Cache management
        - action: opss/run/ - Background process management
    
    Args:
        None (reads from sys.argv)
    
    Returns:
        None: This function exits the process with an appropriate exit code.
            Exit code 0 indicates success, non-zero indicates failure.
    
    Side Effects:
        - Parses sys.argv for command and options
        - May create/modify files (training checkpoints, logs)
        - May start background processes (backend service)
        - May modify system state (setup, cache operations)
    
    Raises:
        SystemExit: On invalid command or execution failure.
            Exit code 1 indicates an error condition.
    
    Example:
        Command-line usage:
            $ python manage.py train --model_size 7B --dataset Chinese2
            $ python manage.py serve --model_size 7B --port 8000
            $ python manage.py benchmark --benchmark mmlu
        
        Programmatic usage (not recommended):
            >>> import sys
            >>> sys.argv = ['manage.py', 'check']
            >>> main()
    
    Note:
        This function is designed to be called once at program startup.
        Multiple calls may have unexpected behavior due to global state.
    """
    parser = argparse.ArgumentParser(description="PiscesL1 Management Tool (manage.py)")
    parser.add_argument('command', nargs='?', choices=COMMANDS, help="Command to execute")
    parser.add_argument('--ckpt', default='', help='Checkpoint file (for benchmarking)')
    parser.add_argument('--max_samples', type=int, default=50000, help='Maximum number of samples per dataset (for download)')
    parser.add_argument('--config', default='configs/0.5B.yaml', help='Model configuration path (for benchmarking)')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length for benchmarking (for benchmarking)')
    parser.add_argument('--list', action='store_true', help='List all available benchmarks (for benchmarking)')
    parser.add_argument('--info', type=str, help='Show detailed information about a benchmark (for benchmarking)')
    parser.add_argument('--benchmark', type=str, help='Run a specific benchmark evaluation (for benchmarking)')
    parser.add_argument('--model', type=str, help='Model checkpoint path (for benchmarking)')
    parser.add_argument('--perf', action='store_true', help='Run a performance benchmark (for benchmarking)')
    parser.add_argument('--selftest', action='store_true', help='Run built-in benchmark tests (for benchmarking)')
    parser.add_argument('--model_size', default='0.5B', type=str, help='Model size: 0.5B, 1B, 7B, 14B, 72B, 671B, 1T')
    parser.add_argument('--dataset', default='', type=str, help='Dataset name for training')
    parser.add_argument('--resume_ckpt', default=None, type=str, help='Path to the checkpoint to resume training')
    parser.add_argument('--reset_lr', action='store_true', help='Reset the learning rate after resuming from a checkpoint')
    parser.add_argument('--quant', action='store_true', help='Force enable 4-bit quantization')
    parser.add_argument('--no_quant', action='store_true', help='Force disable quantization')
    parser.add_argument('--force_quant', action='store_true', help='Override the configuration to force enable quantization')
    parser.add_argument('--force_lora', action='store_true', help='Override the configuration to force enable LoRA')
    parser.add_argument('--quant_bits', type=int, choices=[2, 4, 8], default=None, help='Quantization bits: 2, 4, or 8')
    parser.add_argument('--rlhf', action='store_true', help='Enable RLHF (Reinforcement Learning from Human Feedback)')
    parser.add_argument('--rlhf_dataset', type=str, default=None, help='RLHF human feedback dataset')
    parser.add_argument('--rlhf_lr', type=float, default=None, help='RLHF learning rate')
    parser.add_argument('--rlhf_batch_size', type=int, default=None, help='RLHF batch size')
    parser.add_argument('--rlhf_mini_batch_size', type=int, default=None, help='RLHF mini-batch size')
    parser.add_argument('--rlhf_accum_steps', type=int, default=None, help='RLHF gradient accumulation steps')
    parser.add_argument('--rlhf_epochs', type=int, default=None, help='RLHF training epochs')
    parser.add_argument('--rlhf_max_samples', type=int, default=None, help='RLHF maximum number of samples')
    parser.add_argument('--rlhf_max_length', type=int, default=None, help='RLHF maximum sequence length')
    parser.add_argument('--mcp_host', type=str, default='localhost', help='MCP server host (for MCP operations)')
    parser.add_argument('--mcp_port', type=int, default=8080, help='MCP server port (for MCP operations)')
    parser.add_argument('--mcp_action', type=str, choices=['status', 'warmup', 'refresh-cache'], default='status', help='MCP action to perform (for MCP operations)')
    parser.add_argument('--text', type=str, help='Text content to check for watermark (for watermark detection)')
    parser.add_argument('--file', type=str, help='File path to check for watermark (for watermark detection)')
    parser.add_argument('--image-file', type=str, help='Image file to check for watermark (for watermark detection)')
    parser.add_argument('--audio-file', type=str, help='Audio file to check for watermark (for watermark detection)')
    parser.add_argument('--video-file', type=str, help='Video file to check for watermark (for watermark detection)')
    parser.add_argument('--model-file', type=str, help='Model checkpoint file to verify watermark (for watermark detection)')
    parser.add_argument('--batch', action='store_true', help='Enable batch mode for directory processing (for watermark detection)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output (for watermark detection)')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format (for watermark detection)')
    parser.add_argument('--jurisdiction', type=str, default='GLOBAL', choices=['CN', 'EU', 'US', 'UK', 'JP', 'KR', 'GLOBAL'], help='Jurisdiction for compliance checking (for watermark detection)')
    parser.add_argument('--expected-owner', type=str, default=None, help='Expected owner ID for model weight verification (for watermark detection)')
    parser.add_argument('--frame-sample-rate', type=int, default=30, help='Frame sample rate for video watermark detection (for watermark detection)')
    parser.add_argument('--max-frames', type=int, default=300, help='Maximum frames to process for video watermark detection (for watermark detection)')
    parser.add_argument('--weights-verify', action='store_true', help='Verify weight-level watermark (for watermark)')
    parser.add_argument('--monitor_mode', type=str, choices=['standard'], help='Monitor mode (for system/tools observability)')
    parser.add_argument('--update_interval', type=float, help='Monitor screen update interval seconds')
    parser.add_argument('--log_interval', type=float, help='Monitor log aggregation interval seconds')
    parser.add_argument('--train_mode', type=str, choices=['standard', 'quant_export', 'preference'], default='standard', help='Training mode')
    parser.add_argument('--train_config', type=str, default=None, help='Training config file path (JSON/YAML)')
    parser.add_argument('--dry_run', action='store_true', help='Resolve configs and exit without running')
    parser.add_argument('--model_path', type=str, default=None, help='Model path for preference training')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID for cross-terminal tracking/control (optional)')
    parser.add_argument('--run_name', type=str, default=None, help='Run name (optional)')
    parser.add_argument('--run_dir', type=str, default=None, help='Run directory override (optional)')
    parser.add_argument('--control_interval', type=float, default=0.5, help='Control polling interval seconds (run mode)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Backend service host')
    parser.add_argument('--port', type=int, default=8000, help='Backend service port')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes for backend service')
    parser.add_argument('--max_concurrency', type=int, default=2, help='Max concurrent requests for backend service')
    parser.add_argument('--request_timeout', type=float, default=120.0, help='Request timeout seconds for backend service')
    parser.add_argument('--serve_config', type=str, default=None, help='Backend service config file path (JSON/YAML)')
    parser.add_argument('--enable_opss', action='store_true', help='Enable OPSS integration (MCP Plaza, Swarm, Orchestrator)')
    parser.add_argument('--enable_agent_intercept', action='store_true', help='Enable agent XML pattern interception')
    parser.add_argument('--api_key', type=str, default=None, help='API key for backend service authentication')
    parser.add_argument('--cors_origins', type=str, default='*', help='CORS allowed origins (comma-separated)')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    
    args, unknown = parser.parse_known_args()
    
    if args.command is None or args.command == 'help':
        from tools.help import help
        help()
    elif args.command == 'train':
        from tools.train.orchestrator import PiscesLxTrainOrchestrator
        orchestrator = PiscesLxTrainOrchestrator(args)
        orchestrator.run(args)
    elif args.command == 'serve':
        from tools.infer.server import PiscesLxBackendServer
        server = PiscesLxBackendServer(args)
        server.run()
    elif args.command == 'action':
        from opss.run import POPSSRunCLI
        cli = POPSSRunCLI()
        argv = []
        try:
            argv = list(sys.argv[2:])
        except Exception:
            argv = list(unknown or [])
        sys.exit(int(cli.run(args, argv)))
    elif args.command == 'check':
        from tools.check import check
        check(args)
    elif args.command == 'monitor':
        from tools.monitor.orchestrator import PiscesLxToolsMonitorOrchestrator
        orchestrator = PiscesLxToolsMonitorOrchestrator(args)
        orchestrator.run(args)
    elif args.command == 'download':
        from tools.data.download import PiscesLxToolsDataDatasetDownload
        tool = PiscesLxToolsDataDatasetDownload()
        tool.download(args.max_samples)
        tool.optimize(max_keep=5000)
    elif args.command == 'setup':
        from tools.setup import setup
        setup(args)
    elif args.command == 'benchmark':
        from tools.benchmark.orchestrator import PiscesLxToolsBenchmarkOrchestrator
        orchestrator = PiscesLxToolsBenchmarkOrchestrator(args)
        if args.selftest:
            orchestrator.run_tests(args)
        else:
            orchestrator.run(args)
    elif args.command == 'mcp':
        from opss.mcp.mcps import POPSSToolRegistry
        if args.mcp_action == 'status':
            _get_logger().info("MCP config status", event="manage.right")
            registry = POPSSToolRegistry.get_instance()
            tools = registry.list_tools()
            print(json.dumps({
                "tool_count": len(tools),
                "tools": tools[:10]
            }, indent=2, ensure_ascii=False))
        elif args.mcp_action == 'warmup':
            _get_logger().info("Starting MCP background discovery...", event="manage.right")
            _get_logger().info("MCP tools registered")
        elif args.mcp_action == 'refresh-cache':
            _get_logger().info("Refreshing MCP tools cache...")
            registry = POPSSToolRegistry.get_instance()
            tools = registry.list_tools()
            _get_logger().info(f"MCP tools refreshed: {len(tools)} tools registered")
    
    elif args.command == 'watermark':
        from tools.wmc import (
            detect_watermark, 
            batch_detect,
            detect_image_watermark, 
            detect_audio_watermark,
            detect_video_watermark,
            detect_model_watermark
        )
        
        if args.weights_verify:
            try:
                from opss.watermark import POPSSWatermarkWeightOperator
                from model import YvModel, YvConfig
                if not args.ckpt:
                    _get_logger().error("--ckpt is required for --weights-verify")
                    sys.exit(1)
                owner_id = 'piscesl1'
                seed = 2025
                threshold = 0.02
                try:
                    from configs.version import VERSION
                    with open('configs/watermark.yaml', 'r', encoding='utf-8') as wf:
                        wm_cfg = _import_yaml().safe_load(wf) or {}
                        # Replace {{VERSION}} placeholder with actual version
                        if "watermark_system" in wm_cfg and wm_cfg["watermark_system"].get("version") == "{{VERSION}}":
                            wm_cfg["watermark_system"]["version"] = VERSION
                        if "watermark_system" in wm_cfg:
                            ws = wm_cfg["watermark_system"]
                            weight_cfg = ws.get("weight", {})
                            owner_id = str(weight_cfg.get("owner_id", owner_id))
                            seed = int(weight_cfg.get("seed", seed))
                            threshold = float(weight_cfg.get("verify_threshold", threshold))
                except Exception:
                    pass
                model_size = getattr(args, 'model_size', '0.5B').upper()
                cfg_path = f"configs/model/{model_size}.yaml"
                if not os.path.exists(cfg_path):
                    _LOG.error(f"Model config not found: {cfg_path}")
                    sys.exit(1)
                cfg = YvConfig.from_yaml(cfg_path)
                model = YvModel(cfg)
                load_ckpt(model, None, 0, args.ckpt)
                score, passed = verify_weights(model, owner_id, seed)
                out = {
                    "owner_id": owner_id,
                    "seed": seed,
                    "threshold": threshold,
                    "score": float(score),
                    "passed": bool(passed),
                    "ckpt": args.ckpt,
                    "model_size": model_size,
                }
                print(json.dumps(out, ensure_ascii=False, indent=2))
            except Exception as e:
                _LOG.error(f"weights-verify failed: {e}", event="manage.error")
                sys.exit(1)
        else:
            result = None
            jurisdiction = getattr(args, 'jurisdiction', 'GLOBAL')
            verbose = args.verbose
            
            if args.video_file:
                result = detect_video_watermark(
                    args.video_file, 
                    verbose=verbose,
                    jurisdiction=jurisdiction,
                    frame_sample_rate=getattr(args, 'frame_sample_rate', 30),
                    max_frames=getattr(args, 'max_frames', 300)
                )
            elif args.model_file:
                result = detect_model_watermark(
                    args.model_file,
                    verbose=verbose,
                    expected_owner=getattr(args, 'expected_owner', None),
                    jurisdiction=jurisdiction
                )
            elif args.image_file:
                result = detect_image_watermark(
                    args.image_file, 
                    verbose=verbose,
                    jurisdiction=jurisdiction
                )
            elif args.audio_file:
                result = detect_audio_watermark(
                    args.audio_file, 
                    verbose=verbose,
                    jurisdiction=jurisdiction
                )
            elif args.file:
                if args.batch:
                    result = batch_detect(args.file, verbose=verbose, jurisdiction=jurisdiction)
                else:
                    with open(args.file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    result = detect_watermark(content, verbose=verbose, jurisdiction=jurisdiction)
            elif args.text:
                result = detect_watermark(args.text, verbose=verbose, jurisdiction=jurisdiction)
            else:
                _LOG.error("Watermark command requires one of: --text, --file, --image-file, --audio-file, --video-file, --model-file, or --weights-verify with --ckpt")
                sys.exit(1)
            
            if result is not None and args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        _get_logger().error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
