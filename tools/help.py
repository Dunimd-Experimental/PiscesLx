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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
Command-Line Interface Help Module for PiscesL1.

This module provides comprehensive help documentation for the PiscesL1
command-line interface. It displays detailed usage information for all
available commands, options, and examples to guide users in operating
the framework effectively.

Key Responsibilities:
    1. Command Documentation:
       - List all available CLI commands
       - Document command-specific options and flags
       - Provide default values and parameter types
    
    2. Usage Examples:
       - Common use case demonstrations
       - Command combination examples
       - Advanced feature usage patterns
    
    3. Integration Documentation:
       - OpenAI-compatible API usage
       - Agent XML pattern syntax
       - Watermark detection usage
       - Background run management

Command Categories:
    The PiscesL1 CLI is organized into the following command categories:
    
    **Core Commands:**
        - setup: Environment setup and dependency installation
        - help: Display help information
    
    **Training Commands:**
        - train: Model training with various configurations
        - Supports quantization, LoRA, RLHF, and checkpoint resumption
    
    **Backend Service Commands:**
        - serve: Start OpenAI-compatible backend inference service
        - Multi-size model support (0.5B / 1B / 7B / 14B / 72B / 671B / 1T)
        - Agent XML pattern interception
    
    **Background Management:**
        - action: Manage background training/inference processes
        - Cross-terminal tracking and control
    
    **Dataset Commands:**
        - download: Download and prepare training datasets
    
    **Benchmark Commands:**
        - benchmark: Model evaluation and performance testing
    
    **Monitor Commands:**
        - check: GPU and dependency validation
        - monitor: Real-time system monitoring
    
    **MCP Commands:**
        - mcp: Model Context Protocol tool management
    
    **Watermark Commands:**
        - watermark: Detect and verify watermarks in various media

Output Format:
    The help text is formatted for terminal display with:
    - Clear section headers
    - Indented option lists
    - Type annotations (str, int, float, path)
    - Default values in parentheses
    - Practical examples with comments

Usage Examples:
    Display help:
        $ python manage.py help
        
        Output:
        PiscesL1 - Command Line Interface
        
        Usage:
          python manage.py [command] [options]
        
        Core Commands:
          setup       Environment setup and dependency installation
          help        Show this help message
        ...
    
    Programmatic usage:
        from tools.help import help
        help()

Integration Points:
    - manage.py: Called via "python manage.py help" command
    - ArgumentParser: Provides context for CLI argument parsing
    - All tool modules: Documents their CLI interfaces

Extensibility:
    When adding new commands or options, update the help() function
    to include:
    1. Command name and description in the appropriate section
    2. All available options with types and defaults
    3. Practical usage examples

Note:
    The help text is intentionally comprehensive to serve as the
    primary reference for CLI usage. Keep it synchronized with
    actual command implementations in manage.py.
"""


def help():
    """
    Display comprehensive help information for the PiscesL1 CLI.
    
    This function prints detailed documentation for all available commands,
    their options, default values, and usage examples. It serves as the
    primary reference for operating the PiscesL1 framework from the
    command line.
    
    The help output is organized into logical sections:
    
    1. **Core Commands**: setup, help
    2. **Training**: train with extensive options
    3. **Backend Service**: serve for OpenAI-compatible API
    4. **Background Management**: action for process control
    5. **Dataset**: download for data preparation
    6. **Benchmark**: benchmark for evaluation
    7. **Monitor**: check and monitor for system status
    8. **MCP**: mcp for tool management
    9. **Watermark**: watermark for detection and verification
    10. **Examples**: Practical usage demonstrations
    
    Args:
        None
    
    Returns:
        None: This function prints to stdout and returns nothing.
    
    Side Effects:
        - Prints comprehensive help text to stdout
        - No modifications to system state
    
    Example:
        Command-line usage:
            $ python manage.py help
            
        Programmatic usage:
            >>> from tools.help import help
            >>> help()
            PiscesL1 - Command Line Interface
            ...
    
    Note:
        The help text should be kept synchronized with the actual
        command implementations in manage.py. When adding new features,
        update both the implementation and this help text.
    """
    print("PiscesL1 - Command Line Interface")
    print("")
    print("Usage:")
    print("  python manage.py [command] [options]")
    print("")
    print("Core Commands:")
    print("  setup       Environment setup and dependency installation")
    print("  help        Show this help message")
    print("")
    print("Training:")
    print("  train       Train the model")
    print("    --model_size <str>            Model size: 0.5B, 1B, 7B, 14B, 72B, 671B, 1T (default: 0.5B)")
    print("    --dataset <str>               Training dataset name (e.g. Chinese2)")
    print("    --train_mode <str>            Training mode: standard, quant_export, preference (default: standard)")
    print("    --train_config <path>         Training config file (JSON/YAML)")
    print("    --resume_ckpt <path>          Resume from checkpoint")
    print("    --reset_lr                    Reset learning rate after resume")
    print("    --quant                       Force enable 4-bit quantization")
    print("    --no_quant                    Force disable quantization")
    print("    --force_quant                 Override config to enable quantization")
    print("    --force_lora                  Override config to enable LoRA")
    print("    --quant_bits {2,4,8}          Quantization bits (default: 4)")
    print("    --rlhf                        Enable RLHF training")
    print("    --rlhf_dataset <str>          RLHF dataset (default: dunimd/human_feedback)")
    print("    --rlhf_lr <float>             RLHF learning rate (default: 1e-5)")
    print("    --rlhf_batch_size <int>       RLHF batch size (default: 4)")
    print("    --rlhf_mini_batch_size <int>  RLHF mini-batch size (default: 1)")
    print("    --rlhf_accum_steps <int>      RLHF gradient accumulation steps (default: 4)")
    print("    --rlhf_epochs <int>           RLHF training epochs (default: 3)")
    print("    --rlhf_max_samples <int>      RLHF max samples (default: 1000)")
    print("    --rlhf_max_length <int>       RLHF max sequence length (default: 512)")
    print("    --run_id <str>                Run ID for cross-terminal tracking")
    print("    --run_name <str>              Run name")
    print("    --run_dir <path>              Run directory override")
    print("    --control_interval <float>    Control polling interval seconds (default: 0.5)")
    print("    --dry_run                     Resolve configs and exit without running")
    print("")
    print("Backend Service (OpenAI-compatible API):")
    print("  serve       Start backend inference service")
    print("    --model_size <str>            Model size: 0.5B, 1B, 7B, 14B, 72B, 671B, 1T (default: 0.5B)")
    print("    --host <str>                  Service host (default: 127.0.0.1)")
    print("    --port <int>                  Service port (default: 8000)")
    print("    --workers <int>               Number of worker processes (default: 1)")
    print("    --max_concurrency <int>       Max concurrent requests (default: 2)")
    print("    --request_timeout <float>     Request timeout seconds (default: 120.0)")
    print("    --serve_config <path>         Service config file (JSON/YAML)")
    print("    --disable_opss                Disable OPSS integration (enabled by default)")
    print("    --disable_agent_intercept     Disable agent XML pattern interception (enabled by default)")
    print("    --api_key <str>               API key for authentication")
    print("    --cors_origins <str>          CORS allowed origins (default: *)")
    print("    --log_level <str>             Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)")
    print("")
    print("  API Endpoints (OpenAI-compatible):")
    print("    POST /v1/chat/completions     Chat completion (streaming supported)")
    print("    POST /v1/embeddings           Text embeddings")
    print("    POST /v1/images/generations   Image generation")
    print("    POST /v1/videos/generations   Video generation")
    print("    POST /v1/agents/execute       Agent execution")
    print("    POST /v1/tools/execute        Tool execution")
    print("    GET  /v1/models               List available models")
    print("    GET  /health                  Health check")
    print("")
    print("  Agent XML Patterns (enabled by default):")
    print("    <ag name=\"agent_name\">prompt</ag>           Single agent execution")
    print("    <swarm>...</swarm>                           Multi-agent swarm")
    print("    <orchestrate>...</orchestrate>               Agent orchestration")
    print("    <tool name=\"tool_name\">input</tool>         Tool execution")
    print("")
    print("Background Run Management:")
    print("  action       Manage background runs (training/services)")
    print("    submit <type> <config>        Submit a new run (type: train/serve)")
    print("      --gpu_count <int>           Number of GPUs to allocate (default: 1)")
    print("      --gpu_memory <int>          Minimum GPU memory in MB (default: 0=auto)")
    print("      --priority <str>            Task priority: high, normal, low (default: normal)")
    print("      --daemon                    Run in background (default)")
    print("      --foreground                Run in foreground mode")
    print("    serve <config>                Start service in background")
    print("    status [run_id]               Show run status (all if no run_id)")
    print("    attach <run_id>               Attach to a running process")
    print("    logs <run_id>                 Show run logs")
    print("    control <run_id> <action>     Send control command (pause/resume/cancel/save_ckpt_now/kill/stop)")
    print("    worker <run_id>               Run as worker process (internal use)")
    print("    list                          List all runs")
    print("    kill <run_id>                 Kill a running process")
    print("    clean [--dry-run]             Clean up old runs")
    print("    gpu <action>                  GPU resource management")
    print("      list                        List all GPUs with status")
    print("      status [--gpu_id <id>]      Show GPU utilization details")
    print("      release --task_id <id>      Release GPUs allocated to task")
    print("    queue <action>                Task queue management")
    print("      list                        List queued tasks")
    print("      clear [--priority <level>]  Clear tasks from queue")
    print("      stats                       Show queue statistics")
    print("    resources <action>            System resource management")
    print("      status                      Show system resource status")
    print("      utilization                 Show resource utilization")
    print("    recover <run_id>              Recover crashed task from checkpoint")
    print("      --checkpoint <path>         Specific checkpoint to recover from")
    print("      --max_restarts <int>        Maximum restart attempts (default: 3)")
    print("")
    print("Dataset:")
    print("  download    Download datasets for training")
    print("    --max_samples <int>           Max samples per dataset (default: 50000)")
    print("")
    print("Benchmark:")
    print("  benchmark   Model evaluation and benchmarking")
    print("    --list                       List all benchmarks")
    print("    --info <name>                Show benchmark details")
    print("    --benchmark <name>           Run specific benchmark")
    print("    --config <path>              Model config path (default: configs/0.5B.yaml)")
    print("    --seq_len <int>              Sequence length (default: 512)")
    print("    --model <path>               Model checkpoint path")
    print("    --perf                       Run performance benchmark")
    print("    --selftest                   Run built-in tests")
    print("")
    print("Monitor:")
    print("  check       Check GPU and dependencies")
    print("  monitor     System monitor (GPU/CPU/memory)")
    print("    --monitor_mode <str>         Monitor mode: standard (default: standard)")
    print("    --update_interval <float>    Screen update interval seconds")
    print("    --log_interval <float>       Log aggregation interval seconds")
    print("")
    print("MCP:")
    print("  mcp         MCP tool management")
    print("    --mcp_action <str>           Action: status, warmup, refresh-cache (default: status)")
    print("    --mcp_host <str>             MCP server host (default: localhost)")
    print("    --mcp_port <int>             MCP server port (default: 8080)")
    print("")
    print("Watermark:")
    print("  watermark   Watermark detection for text, files, or model weights")
    print("    --text <str>                 Text to check for watermark")
    print("    --file <path>                File or directory to check")
    print("    --image-file <path>          Image file to check")
    print("    --audio-file <path>          Audio file to check")
    print("    --batch                      Enable directory batch mode")
    print("    --verbose, -v                Verbose output")
    print("    --json                       Output JSON format")
    print("    --weights-verify             Verify weight-level watermark (requires --ckpt)")
    print("    --ckpt <path>                Model checkpoint for weight verification")
    print("    --model_size <str>           Model size for weight verification (default: 0.5B)")
    print("")
    print("Examples:")
    print("  # Core")
    print("  python manage.py setup")
    print("  python manage.py help")
    print("")
    print("  # Training")
    print("  python manage.py train --model_size 0.5B --dataset Chinese2")
    print("  python manage.py train --model_size 7B --dataset Chinese2 --quant --quant_bits 4 --force_lora")
    print("  python manage.py train --model_size 1B --dataset Chinese2 --resume_ckpt runs/last.pt --reset_lr")
    print("  python manage.py train --model_size 7B --dataset Chinese2 --rlhf --rlhf_dataset dunimd/human_feedback")
    print("  python manage.py train --model_size 0.5B --dataset Chinese2 --run_id train_001 --run_name my_training")
    print("")
    print("  # Backend Service")
    print("  python manage.py serve --model_size 7B --port 8000")
    print("  python manage.py serve --model_size 14B --host 0.0.0.0 --port 8080 --workers 4")
    print("  python manage.py serve --model_size 72B")
    print("  python manage.py serve --model_size 7B --api_key sk-xxxxx --cors_origins 'http://localhost:3000'")
    print("")
    print("  # API Usage (curl examples)")
    print("  curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"pisceslx-7b\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'")
    print("  curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"pisceslx-7b\", \"messages\": [...], \"stream\": true}'")
    print("  curl http://localhost:8000/v1/embeddings -H 'Content-Type: application/json' -d '{\"model\": \"pisceslx-7b\", \"input\": \"Hello world\"}'")
    print("")
    print("  # Agent XML Patterns (in chat messages)")
    print("  curl http://localhost:8000/v1/chat/completions -d '{\"messages\": [{\"content\": \"<ag name=\\\"web_search\\\">latest AI news</ag>\"}]}'")
    print("  curl http://localhost:8000/v1/chat/completions -d '{\"messages\": [{\"content\": \"<swarm><agent>agent1</agent><agent>agent2</agent></swarm>\"}]}'")
    print("")
    print("  # Background Run Management")
    print("  python manage.py action submit train configs/train.json")
    print("  python manage.py action submit train configs/train.json --gpu_count 2 --priority high")
    print("  python manage.py action submit serve configs/serve.json")
    print("  python manage.py action status")
    print("  python manage.py action status train_001")
    print("  python manage.py action logs train_001")
    print("  python manage.py action control train_001 pause")
    print("  python manage.py action control train_001 resume")
    print("  python manage.py action control train_001 cancel")
    print("  python manage.py action control train_001 save_ckpt_now")
    print("  python manage.py action control train_001 stop")
    print("  python manage.py action list")
    print("  python manage.py action list --running")
    print("  python manage.py action clean --dry-run")
    print("  python manage.py action gpu list")
    print("  python manage.py action gpu status")
    print("  python manage.py action gpu status --gpu_id 0")
    print("  python manage.py action gpu release --task_id train_001")
    print("  python manage.py action queue list")
    print("  python manage.py action queue stats")
    print("  python manage.py action queue clear --priority low")
    print("  python manage.py action resources status")
    print("  python manage.py action resources utilization")
    print("  python manage.py action recover train_001")
    print("  python manage.py action recover train_001 --checkpoint runs/train_001/ckpt.pt")
    print("")
    print("  # Dataset")
    print("  python manage.py download --max_samples 50000")
    print("")
    print("  # Benchmark")
    print("  python manage.py benchmark --list")
    print("  python manage.py benchmark --info mmlu")
    print("  python manage.py benchmark --benchmark mmlu --config configs/0.5B.yaml --model ckpt/model.pt")
    print("  python manage.py benchmark --perf --config configs/0.5B.yaml --selftest")
    print("")
    print("  # Monitor")
    print("  python manage.py check")
    print("  python manage.py monitor --monitor_mode standard --update_interval 1.0")
    print("")
    print("  # MCP")
    print("  python manage.py mcp --mcp_action status")
    print("  python manage.py mcp --mcp_action warmup")
    print("  python manage.py mcp --mcp_action refresh-cache")
    print("")
    print("  # Watermark")
    print("  python manage.py watermark --text 'Check if this text contains a watermark'")
    print("  python manage.py watermark --file ./outputs/ --batch --verbose")
    print("  python manage.py watermark --text 'Test text' --json")
    print("  python manage.py watermark --weights-verify --ckpt ckpt/model.pt --model_size 0.5B")
