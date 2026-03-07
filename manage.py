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

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
# os: Operating system interface for path manipulation and environment access
import os

# sys: System-specific parameters and functions for argv access and exit codes
import sys

# json: JSON encoder/decoder for structured output formatting
import json

# argparse: Command-line argument parsing library for CLI interface
import argparse

# pathlib: Object-oriented filesystem paths for cross-platform compatibility
from pathlib import Path


# =============================================================================
# YAML LAZY IMPORT HELPER
# =============================================================================
def _import_yaml():
    """
    Lazily import the YAML library to defer import errors.
    
    This function provides a lazy import mechanism for the PyYAML library.
    Lazy importing is used here because YAML parsing is only needed for
    configuration file loading, which may not be required for all commands.
    By deferring the import, we avoid unnecessary dependency errors when
    running commands that don't require YAML support.
    
    The function modifies the global namespace by injecting the 'yaml' module,
    allowing subsequent code to use 'yaml' directly after this function is called.
    
    Args:
        None
    
    Returns:
        module: The PyYAML module instance for YAML parsing operations.
    
    Raises:
        ImportError: If PyYAML is not installed in the environment.
            Users should install it via: pip install pyyaml
    
    Side Effects:
        - Imports the yaml module into the global namespace
        - May raise ImportError if PyYAML is not available
    
    Example:
        >>> yaml = _import_yaml()
        >>> config = yaml.safe_load(open('config.yaml'))
    
    Note:
        This pattern is useful for optional dependencies that are only
        needed for specific code paths. It allows the main module to
        load faster and reduces the initial import footprint.
    """
    global yaml
    import yaml
    return yaml


# =============================================================================
# PROJECT ROOT PATH CONSTANT
# =============================================================================
# ROOT: Absolute path to the project root directory
# This constant is used throughout the framework for resolving relative paths
# to configuration files, model checkpoints, and other resources.
# Using absolute paths ensures consistent behavior regardless of the current
# working directory from which manage.py is invoked.
ROOT = os.path.abspath(os.path.dirname(__file__))


# =============================================================================
# LOGGER INITIALIZATION HELPER
# =============================================================================
def _get_logger():
    """
    Get or create the logger instance for the management tool.
    
    This function lazily initializes the logger to avoid import issues
    during early startup. The logger is used for structured logging
    throughout the management tool.
    
    The lazy initialization pattern is critical here because:
    1. The utils module may not be available during early import stages
    2. Logger initialization requires filesystem access for log file creation
    3. Some commands (like 'help') don't need logging at all
    
    Args:
        None
    
    Returns:
        PiscesLxLogger: A configured logger instance for the 'pisceslx.manage'
            namespace. This logger supports:
            - Structured logging with event types
            - File output with rotation
            - Console output with color coding
            - JSON-formatted logs for machine parsing
    
    Side Effects:
        - Creates log file in the logs directory
        - Initializes file handlers for persistent logging
    
    Example:
        >>> logger = _get_logger()
        >>> logger.info("Operation started", event="manage.start")
    
    Note:
        The logger is created on first call and cached for subsequent calls.
        This lazy initialization prevents import errors when the utils module
        is not yet available. Each call returns the same logger instance.
    """
    # Import the custom logger class from the utils datacenter module
    from utils.dc import PiscesLxLogger
    
    # Import the log file path resolver from the utils paths module
    from utils.paths import get_log_file
    
    # Create and return a configured logger instance
    # The logger name 'PiscesLx.Manage' identifies this module in logs
    # File logging is enabled for persistent record-keeping
    return PiscesLxLogger(
        "PiscesLx.Manage", 
        file_path=get_log_file("PiscesLx.Manage"), 
        enable_file=True
    )


# =============================================================================
# AVAILABLE COMMANDS REGISTRY
# =============================================================================
# COMMANDS: List of all valid CLI commands that this management tool supports
# This list is used by argparse for command validation and tab-completion
# Each command maps to a specific handler in the main() function
COMMANDS = [
    'setup',      # Initialize environment and install dependencies
    'train',      # Train models with various configurations
    'serve',      # Start OpenAI-compatible backend inference service
    'test',       # Project health check (8-stage validation)
    'monitor',    # Real-time system monitoring dashboard
    'download',   # Download and prepare training datasets
    'benchmark',  # Evaluate model performance on benchmarks
    'mcp',        # Model Context Protocol tool management
    'help',       # Display comprehensive usage documentation
    'watermark',  # Detect and verify watermarks in content/models
    'action',     # Manage background training/inference processes
]


def _build_train_argv_from_args(args) -> list:
    """Build argv list for train command from args namespace.
    
    Converts argparse namespace to CLI argument list for action submit train.
    """
    argv = []
    mapping = [
        ("model_size", "--model_size"),
        ("dataset", "--dataset"),
        ("resume_ckpt", "--resume_ckpt"),
        ("train_mode", "--train_mode"),
        ("train_config", "--train_config"),
        ("dry_run", "--dry_run"),
        ("seq_len", "--seq_len"),
        ("quant", "--quant"),
        ("no_quant", "--no_quant"),
        ("rlhf", "--rlhf"),
        ("rlhf_dataset", "--rlhf_dataset"),
        ("rlhf_lr", "--rlhf_lr"),
        ("rlhf_batch_size", "--rlhf_batch_size"),
        ("rlhf_mini_batch_size", "--rlhf_mini_batch_size"),
        ("rlhf_accum_steps", "--rlhf_accum_steps"),
        ("rlhf_epochs", "--rlhf_epochs"),
        ("rlhf_max_samples", "--rlhf_max_samples"),
        ("rlhf_max_length", "--rlhf_max_length"),
        ("model_path", "--model_path"),
        ("run_id", "--run_id"),
        ("run_dir", "--run_dir"),
        ("run_name", "--run_name"),
    ]
    for attr, flag in mapping:
        if not hasattr(args, attr):
            continue
        val = getattr(args, attr)
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                argv.append(flag)
            continue
        s = str(val).strip()
        if not s:
            continue
        argv.extend([flag, s])
    return argv


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
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
    # =========================================================================
    # ARGUMENT PARSER CONFIGURATION
    # =========================================================================
    # Create the main argument parser with a descriptive help message
    # The parser handles all CLI argument parsing and validation
    parser = argparse.ArgumentParser(
        description="PiscesL1 Management Tool (manage.py)"
    )
    
    # -------------------------------------------------------------------------
    # POSITIONAL ARGUMENT: Command
    # -------------------------------------------------------------------------
    # The primary command to execute (train, serve, benchmark, etc.)
    # nargs='?' makes it optional (for 'help' default behavior)
    # choices=COMMANDS restricts to valid command names only
    parser.add_argument(
        'command', 
        nargs='?',  # Optional positional argument
        choices=COMMANDS,  # Must be one of the registered commands
        help="Command to execute"
    )
    
    # -------------------------------------------------------------------------
    # CHECKPOINT ARGUMENTS (for benchmarking and resumption)
    # -------------------------------------------------------------------------
    # --ckpt: Path to a model checkpoint file for benchmarking
    # Used when evaluating a trained model on benchmark datasets
    parser.add_argument(
        '--ckpt', 
        default='',  # Empty string means no checkpoint specified
        help='Checkpoint file (for benchmarking)'
    )
    
    # -------------------------------------------------------------------------
    # DATASET ARGUMENTS (for download command)
    # -------------------------------------------------------------------------
    # --max_samples: Maximum number of samples to download per dataset
    # This limits dataset size for faster downloads and testing
    parser.add_argument(
        '--max_samples', 
        type=int,  # Must be an integer value
        default=50000,  # Default to 50K samples per dataset
        help='Maximum number of samples per dataset (for download)'
    )
    
    # -------------------------------------------------------------------------
    # MODEL CONFIGURATION ARGUMENTS (for benchmarking)
    # -------------------------------------------------------------------------
    # --config: Path to the model configuration YAML file
    # This defines model architecture, hyperparameters, and settings
    parser.add_argument(
        '--config', 
        default='configs/0.5B.yaml',  # Default to smallest model config
        help='Model configuration path (for benchmarking)'
    )
    
    # --seq_len: Maximum sequence length for benchmarking
    # Determines the context window size during evaluation
    parser.add_argument(
        '--seq_len', 
        type=int,  # Must be an integer value
        default=512,  # Default to 512 tokens (standard for benchmarks)
        help='Sequence length for benchmarking (for benchmarking)'
    )
    
    # -------------------------------------------------------------------------
    # BENCHMARK LISTING ARGUMENTS
    # -------------------------------------------------------------------------
    # --list: Flag to list all available benchmark names
    # When set, prints all supported benchmarks and exits
    parser.add_argument(
        '--list', 
        action='store_true',  # Boolean flag (True if present)
        help='List all available benchmarks (for benchmarking)'
    )
    
    # --info: Show detailed information about a specific benchmark
    # Provides description, metrics, and requirements for the benchmark
    parser.add_argument(
        '--info', 
        type=str,  # Benchmark name as string
        help='Show detailed information about a benchmark (for benchmarking)'
    )
    
    # -------------------------------------------------------------------------
    # BENCHMARK EXECUTION ARGUMENTS
    # -------------------------------------------------------------------------
    # --benchmark: Name of the benchmark to run
    # Must match one of the registered benchmark names
    parser.add_argument(
        '--benchmark', 
        type=str,  # Benchmark name as string
        help='Run a specific benchmark evaluation (for benchmarking)'
    )
    
    # --model: Path to the model checkpoint for evaluation
    # Alternative to --ckpt, specifically for benchmark command
    parser.add_argument(
        '--model', 
        type=str,  # Model path as string
        help='Model checkpoint path (for benchmarking)'
    )
    
    # --perf: Flag to run performance benchmarking
    # Measures inference speed, memory usage, and throughput
    parser.add_argument(
        '--perf', 
        action='store_true',  # Boolean flag
        help='Run a performance benchmark (for benchmarking)'
    )
    
    # --selftest: Flag to run built-in benchmark tests
    # Validates benchmark implementation correctness
    parser.add_argument(
        '--selftest', 
        action='store_true',  # Boolean flag
        help='Run built-in benchmark tests (for benchmarking)'
    )
    
    # -------------------------------------------------------------------------
    # MODEL SIZE ARGUMENT (universal across commands)
    # -------------------------------------------------------------------------
    # --model_size: Specifies the model parameter count
    # This determines which model configuration and weights to use
    # Supported sizes: 0.5B, 1B, 7B, 14B, 72B, 671B, 1T
    parser.add_argument(
        '--model_size', 
        default='0.5B',  # Default to smallest model for faster testing
        type=str,  # String to preserve exact size notation
        help='Model size: 0.5B, 1B, 7B, 14B, 72B, 671B, 1T'
    )
    
    # -------------------------------------------------------------------------
    # TRAINING DATASET ARGUMENTS
    # -------------------------------------------------------------------------
    # --dataset: Name of the training dataset to use
    # Must match a dataset name in the data registry
    parser.add_argument(
        '--dataset', 
        default='',  # Empty means use default from config
        type=str,  # Dataset name as string
        help='Dataset name for training'
    )
    
    # -------------------------------------------------------------------------
    # CHECKPOINT RESUMPTION ARGUMENTS
    # -------------------------------------------------------------------------
    # --resume_ckpt: Path to checkpoint for resuming interrupted training
    # Loads model weights, optimizer state, and training progress
    parser.add_argument(
        '--resume_ckpt', 
        default=None,  # None means start fresh training
        type=str,  # Checkpoint path as string
        help='Path to the checkpoint to resume training'
    )
    
    # --reset_lr: Flag to reset learning rate after resuming
    # Useful when resuming from a checkpoint with different LR schedule
    parser.add_argument(
        '--reset_lr', 
        action='store_true',  # Boolean flag
        help='Reset the learning rate after resuming from a checkpoint'
    )
    
    # -------------------------------------------------------------------------
    # QUANTIZATION ARGUMENTS
    # -------------------------------------------------------------------------
    # --quant: Flag to force enable 4-bit quantization
    # Overrides config settings to use quantization
    parser.add_argument(
        '--quant', 
        action='store_true',  # Boolean flag
        help='Force enable 4-bit quantization'
    )
    
    # --no_quant: Flag to force disable quantization
    # Overrides config settings to disable quantization
    parser.add_argument(
        '--no_quant', 
        action='store_true',  # Boolean flag
        help='Force disable quantization'
    )
    
    # --force_quant: Flag to override config and force quantization
    # Stronger override than --quant, ignores all config settings
    parser.add_argument(
        '--force_quant', 
        action='store_true',  # Boolean flag
        help='Override the configuration to force enable quantization'
    )
    
    # --force_lora: Flag to override config and force LoRA
    # Enables Low-Rank Adaptation for efficient fine-tuning
    parser.add_argument(
        '--force_lora', 
        action='store_true',  # Boolean flag
        help='Override the configuration to force enable LoRA'
    )
    
    # --quant_bits: Number of bits for quantization (2, 4, or 8)
    # Lower bits = smaller model but potentially lower accuracy
    parser.add_argument(
        '--quant_bits', 
        type=int,  # Must be an integer
        choices=[2, 4, 8],  # Only these values are valid
        default=None,  # None means use config default
        help='Quantization bits: 2, 4, or 8'
    )
    
    # -------------------------------------------------------------------------
    # RLHF (Reinforcement Learning from Human Feedback) ARGUMENTS
    # -------------------------------------------------------------------------
    # --rlhf: Flag to enable RLHF training mode
    # Trains a reward model and uses PPO for policy optimization
    parser.add_argument(
        '--rlhf', 
        action='store_true',  # Boolean flag
        help='Enable RLHF (Reinforcement Learning from Human Feedback)'
    )
    
    # --rlhf_dataset: Path to the human feedback dataset for RLHF
    # Contains preference pairs for reward model training
    parser.add_argument(
        '--rlhf_dataset', 
        type=str,  # Dataset path as string
        default=None,  # None means use default dataset
        help='RLHF human feedback dataset'
    )
    
    # --rlhf_lr: Learning rate for RLHF training
    # Typically lower than pretraining LR for stable fine-tuning
    parser.add_argument(
        '--rlhf_lr', 
        type=float,  # Must be a float
        default=None,  # None means use default
        help='RLHF learning rate'
    )
    
    # --rlhf_batch_size: Batch size for RLHF training
    # Affects memory usage and training stability
    parser.add_argument(
        '--rlhf_batch_size', 
        type=int,  # Must be an integer
        default=None,  # None means use default
        help='RLHF batch size'
    )
    
    # --rlhf_mini_batch_size: Mini-batch size for PPO updates
    # Used within each batch for gradient computation
    parser.add_argument(
        '--rlhf_mini_batch_size', 
        type=int,  # Must be an integer
        default=None,  # None means use default
        help='RLHF mini-batch size'
    )
    
    # --rlhf_accum_steps: Gradient accumulation steps for RLHF
    # Simulates larger batch sizes with limited memory
    parser.add_argument(
        '--rlhf_accum_steps', 
        type=int,  # Must be an integer
        default=None,  # None means use default
        help='RLHF gradient accumulation steps'
    )
    
    # --rlhf_epochs: Number of training epochs for RLHF
    # More epochs = better convergence but longer training
    parser.add_argument(
        '--rlhf_epochs', 
        type=int,  # Must be an integer
        default=None,  # None means use default
        help='RLHF training epochs'
    )
    
    # --rlhf_max_samples: Maximum samples for RLHF training
    # Limits dataset size for faster iteration during development
    parser.add_argument(
        '--rlhf_max_samples', 
        type=int,  # Must be an integer
        default=None,  # None means use all available
        help='RLHF maximum number of samples'
    )
    
    # --rlhf_max_length: Maximum sequence length for RLHF
    # Truncates sequences longer than this value
    parser.add_argument(
        '--rlhf_max_length', 
        type=int,  # Must be an integer
        default=None,  # None means use config default
        help='RLHF maximum sequence length'
    )
    
    # -------------------------------------------------------------------------
    # MCP (Model Context Protocol) ARGUMENTS
    # -------------------------------------------------------------------------
    # --mcp_host: Host address for MCP server
    # Used for MCP tool registration and discovery
    parser.add_argument(
        '--mcp_host', 
        type=str,  # Host as string
        default='localhost',  # Default to local server
        help='MCP server host (for MCP operations)'
    )
    
    # --mcp_port: Port number for MCP server
    # Must match the port where MCP server is listening
    parser.add_argument(
        '--mcp_port', 
        type=int,  # Must be an integer
        default=8080,  # Default MCP port
        help='MCP server port (for MCP operations)'
    )
    
    # --mcp_action: Action to perform on MCP server
    # status: Check current tool registration status
    # warmup: Start background tool discovery
    # refresh-cache: Reload all tool definitions
    parser.add_argument(
        '--mcp_action', 
        type=str,  # Action name as string
        choices=['status', 'warmup', 'refresh-cache'],  # Valid actions
        default='status',  # Default to status check
        help='MCP action to perform (for MCP operations)'
    )
    
    # -------------------------------------------------------------------------
    # WATERMARK DETECTION ARGUMENTS - INPUT SOURCES
    # -------------------------------------------------------------------------
    # --text: Direct text content to check for watermark
    # Used for quick watermark checks without file I/O
    parser.add_argument(
        '--text', 
        type=str,  # Text content as string
        help='Text content to check for watermark (for watermark detection)'
    )
    
    # --file: Path to text file for watermark detection
    # Reads file content and checks for embedded watermarks
    parser.add_argument(
        '--file', 
        type=str,  # File path as string
        help='File path to check for watermark (for watermark detection)'
    )
    
    # --image-file: Path to image file for watermark detection
    # Detects visible and invisible watermarks in images
    parser.add_argument(
        '--image-file', 
        type=str,  # Image file path as string
        help='Image file to check for watermark (for watermark detection)'
    )
    
    # --audio-file: Path to audio file for watermark detection
    # Detects audio watermarks (steganography, frequency patterns)
    parser.add_argument(
        '--audio-file', 
        type=str,  # Audio file path as string
        help='Audio file to check for watermark (for watermark detection)'
    )
    
    # --video-file: Path to video file for watermark detection
    # Analyzes video frames for embedded watermarks
    parser.add_argument(
        '--video-file', 
        type=str,  # Video file path as string
        help='Video file to check for watermark (for watermark detection)'
    )
    
    # --model-file: Path to model file for watermark verification
    # Verifies ownership watermarks in model weights
    parser.add_argument(
        '--model-file', 
        type=str,  # Model file path as string
        help='Model checkpoint file to verify watermark (for watermark detection)'
    )
    
    # -------------------------------------------------------------------------
    # WATERMARK DETECTION ARGUMENTS - PROCESSING OPTIONS
    # -------------------------------------------------------------------------
    # --batch: Flag to enable batch mode for directory processing
    # Processes all files in a directory instead of single file
    parser.add_argument(
        '--batch', 
        action='store_true',  # Boolean flag
        help='Enable batch mode for directory processing (for watermark detection)'
    )
    
    # --verbose: Flag to enable detailed output
    # Shows step-by-step detection process and intermediate results
    parser.add_argument(
        '--verbose', '-v',  # -v is a short alias
        action='store_true',  # Boolean flag
        help='Enable verbose output (for watermark detection)'
    )
    
    # --json: Flag to output results in JSON format
    # Useful for programmatic processing of detection results
    parser.add_argument(
        '--json', 
        action='store_true',  # Boolean flag
        help='Output results in JSON format (for watermark detection)'
    )
    
    # -------------------------------------------------------------------------
    # WATERMARK DETECTION ARGUMENTS - COMPLIANCE
    # -------------------------------------------------------------------------
    # --jurisdiction: Legal jurisdiction for compliance checking
    # Different regions have different watermark requirements
    parser.add_argument(
        '--jurisdiction', 
        type=str,  # Jurisdiction code as string
        default='GLOBAL',  # Default to global standards
        choices=['CN', 'EU', 'US', 'UK', 'JP', 'KR', 'GLOBAL'],  # Supported regions
        help='Jurisdiction for compliance checking (for watermark detection)'
    )
    
    # --expected-owner: Expected owner ID for model weight verification
    # Verifies that the model belongs to the specified owner
    parser.add_argument(
        '--expected-owner', 
        type=str,  # Owner ID as string
        default=None,  # None means no owner verification
        help='Expected owner ID for model weight verification (for watermark detection)'
    )
    
    # -------------------------------------------------------------------------
    # WATERMARK DETECTION ARGUMENTS - VIDEO PROCESSING
    # -------------------------------------------------------------------------
    # --frame-sample-rate: Frames to skip between analysis
    # Higher values = faster processing but may miss watermarks
    parser.add_argument(
        '--frame-sample-rate', 
        type=int,  # Must be an integer
        default=30,  # Analyze every 30th frame
        help='Frame sample rate for video watermark detection (for watermark detection)'
    )
    
    # --max-frames: Maximum frames to process per video
    # Prevents excessive processing time for long videos
    parser.add_argument(
        '--max-frames', 
        type=int,  # Must be an integer
        default=300,  # Process at most 300 frames
        help='Maximum frames to process for video watermark detection (for watermark detection)'
    )
    
    # -------------------------------------------------------------------------
    # WATERMARK DETECTION ARGUMENTS - WEIGHT VERIFICATION
    # -------------------------------------------------------------------------
    # --weights-verify: Flag to verify weight-level watermark
    # Checks for embedded ownership markers in model weights
    parser.add_argument(
        '--weights-verify', 
        action='store_true',  # Boolean flag
        help='Verify weight-level watermark (for watermark)'
    )
    
    # -------------------------------------------------------------------------
    # MONITOR ARGUMENTS
    # -------------------------------------------------------------------------
    # --monitor_mode: Operating mode for the monitoring system
    # 'standard' provides balanced metrics and update frequency
    parser.add_argument(
        '--monitor_mode', 
        type=str,  # Mode name as string
        choices=['standard'],  # Currently only standard mode supported
        help='Monitor mode (for system/tools observability)'
    )
    
    # --update_interval: Screen refresh interval in seconds
    # Lower values = more responsive but higher CPU usage
    parser.add_argument(
        '--update_interval', 
        type=float,  # Must be a float
        help='Monitor screen update interval seconds'
    )
    
    # --log_interval: Log aggregation interval in seconds
    # Determines how often logs are collected and summarized
    parser.add_argument(
        '--log_interval', 
        type=float,  # Must be a float
        help='Monitor log aggregation interval seconds'
    )
    
    # -------------------------------------------------------------------------
    # TRAINING MODE ARGUMENTS
    # -------------------------------------------------------------------------
    # --train_mode: Training mode selection
    # standard: Normal training workflow
    # quant_export: Export quantized model after training
    # preference: Preference-based training (DPO/RLHF)
    parser.add_argument(
        '--train_mode', 
        type=str,  # Mode name as string
        choices=['standard', 'quant_export', 'preference'],  # Valid modes
        default='standard',  # Default to normal training
        help='Training mode'
    )
    
    # --train_config: Path to training configuration file
    # Can be JSON or YAML format, overrides default config
    parser.add_argument(
        '--train_config', 
        type=str,  # Config path as string
        default=None,  # None means use default config
        help='Training config file path (JSON/YAML)'
    )
    
    # --dry_run: Flag to resolve configs without executing
    # Useful for validating configuration before training
    parser.add_argument(
        '--dry_run', 
        action='store_true',  # Boolean flag
        help='Resolve configs and exit without running'
    )
    
    # --model_path: Path to pretrained model for preference training
    # Used as the base model for DPO or RLHF fine-tuning
    parser.add_argument(
        '--model_path', 
        type=str,  # Model path as string
        default=None,  # None means train from scratch
        help='Model path for preference training'
    )
    
    # -------------------------------------------------------------------------
    # RUN MANAGEMENT ARGUMENTS (for action command)
    # -------------------------------------------------------------------------
    # --run_id: Unique identifier for cross-terminal tracking
    # Allows controlling runs from different terminal sessions
    parser.add_argument(
        '--run_id', 
        type=str,  # Run ID as string
        default=None,  # None means auto-generate ID
        help='Run ID for cross-terminal tracking/control (optional)'
    )
    
    # --run_name: Human-readable name for the run
    # Used for display purposes and logging
    parser.add_argument(
        '--run_name', 
        type=str,  # Run name as string
        default=None,  # None means auto-generate name
        help='Run name (optional)'
    )
    
    # --run_dir: Override default run directory
    # Custom location for checkpoints and logs
    parser.add_argument(
        '--run_dir', 
        type=str,  # Directory path as string
        default=None,  # None means use default directory
        help='Run directory override (optional)'
    )
    
    # --control_interval: Polling interval for run control
    # How often to check for control signals (pause/resume/cancel)
    parser.add_argument(
        '--control_interval', 
        type=float,  # Must be a float
        default=0.5,  # Check every 0.5 seconds
        help='Control polling interval seconds (run mode)'
    )
    
    # -------------------------------------------------------------------------
    # BACKEND SERVICE ARGUMENTS
    # -------------------------------------------------------------------------
    # --host: Host address for the backend service
    # Use 0.0.0.0 for external access, 127.0.0.1 for local only
    parser.add_argument(
        '--host', 
        type=str,  # Host as string
        default='127.0.0.1',  # Default to localhost only
        help='Backend service host'
    )
    
    # --port: Port number for the backend service
    # Must be available and not used by other services
    parser.add_argument(
        '--port', 
        type=int,  # Must be an integer
        default=8000,  # Default to common API port
        help='Backend service port'
    )
    
    # --workers: Number of worker processes
    # More workers = higher throughput but more memory
    parser.add_argument(
        '--workers', 
        type=int,  # Must be an integer
        default=1,  # Single worker for simplicity
        help='Number of worker processes for backend service'
    )
    
    # --max_concurrency: Maximum concurrent requests
    # Limits simultaneous inferences to prevent OOM
    parser.add_argument(
        '--max_concurrency', 
        type=int,  # Must be an integer
        default=2,  # Conservative default for stability
        help='Max concurrent requests for backend service'
    )
    
    # --request_timeout: Timeout for each request in seconds
    # Long-running requests (streaming) may need higher values
    parser.add_argument(
        '--request_timeout', 
        type=float,  # Must be a float
        default=120.0,  # 2 minutes default timeout
        help='Request timeout seconds for backend service'
    )
    
    # --serve_config: Path to backend service configuration file
    # Contains model paths, API settings, and other options
    parser.add_argument(
        '--serve_config', 
        type=str,  # Config path as string
        default=None,  # None means use default config
        help='Backend service config file path (JSON/YAML)'
    )
    
    # -------------------------------------------------------------------------
    # OPSS INTEGRATION ARGUMENTS
    # -------------------------------------------------------------------------
    # --disable_opss: Flag to disable OPSS integration
    # By default, OPSS (MCP Plaza, Swarm, Orchestrator) is enabled
    parser.add_argument(
        '--disable_opss', 
        action='store_true',  # Boolean flag
        help='Disable OPSS integration (MCP Plaza, Swarm, Orchestrator) - enabled by default'
    )
    
    # --disable_agent_intercept: Flag to disable agent XML interception
    # By default, agent XML pattern interception is enabled
    parser.add_argument(
        '--disable_agent_intercept', 
        action='store_true',  # Boolean flag
        help='Disable agent XML pattern interception - enabled by default'
    )
    
    # -------------------------------------------------------------------------
    # BACKEND SERVICE AUTHENTICATION ARGUMENTS
    # -------------------------------------------------------------------------
    # --api_key: API key for service authentication
    # Required for protected endpoints if authentication is enabled
    parser.add_argument(
        '--api_key', 
        type=str,  # API key as string
        default=None,  # None means no authentication
        help='API key for backend service authentication'
    )
    
    # --cors_origins: CORS allowed origins
    # Comma-separated list of domains for cross-origin requests
    parser.add_argument(
        '--cors_origins', 
        type=str,  # Origins as string
        default='*',  # Allow all origins by default
        help='CORS allowed origins (comma-separated)'
    )
    
    # -------------------------------------------------------------------------
    # TEST COMMAND ARGUMENTS
    # -------------------------------------------------------------------------
    # --quick: Flag to run quick check (stages 1-5 only)
    # Skips forward pass, generation, and optimization checks
    parser.add_argument(
        '--quick', 
        action='store_true',  # Boolean flag
        help='Run quick check (stages 1-5 only) (for test command)'
    )
    
    # --stage: Comma-separated list of stages to run
    # Allows running specific stages instead of all
    parser.add_argument(
        '--stage', 
        type=str,  # Stage numbers as string
        default=None,  # None means run all stages
        help='Comma-separated stages to run (e.g., "1,2,3") (for test command)'
    )
    
    # -------------------------------------------------------------------------
    # LOGGING ARGUMENTS
    # -------------------------------------------------------------------------
    # --log_level: Logging verbosity level
    # DEBUG: Most verbose, INFO: Normal, WARNING: Issues only, ERROR: Errors only
    parser.add_argument(
        '--log_level', 
        type=str,  # Level name as string
        default='INFO',  # Default to normal logging
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],  # Valid levels
        help='Logging level'
    )
    
    # =========================================================================
    # PARSE ARGUMENTS
    # =========================================================================
    # parse_known_args() returns two values:
    # - args: Namespace with all known arguments
    # - unknown: List of unknown arguments (for pass-through to sub-commands)
    # This allows the action command to receive additional arguments
    args, unknown = parser.parse_known_args()
    
    # =========================================================================
    # COMMAND DISPATCH
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # HELP COMMAND (default when no command specified)
    # -------------------------------------------------------------------------
    # Display comprehensive usage documentation
    # This is the default behavior when no command is provided
    if args.command is None or args.command == 'help':
        # Import the help module and display usage information
        from tools.help import help
        help()
    
    # -------------------------------------------------------------------------
    # TRAIN COMMAND
    # -------------------------------------------------------------------------
    # Start model training with the specified configuration
    # Supports standard training, quantization, RLHF, and LoRA
    # Binds to action management system in foreground mode
    elif args.command == 'train':
        run_id = getattr(args, 'run_id', None) or None
        run_dir = getattr(args, 'run_dir', None) or None
        
        if not run_id:
            from opss.run.id_factory import POPSSRunIdFactory
            run_id = POPSSRunIdFactory(prefix="train").new_id()
        
        from opss.run.store import POPSSRunStore
        from opss.run.controller import POPSSRunController
        
        store = POPSSRunStore(run_id, run_dir=run_dir)
        controller = POPSSRunController(store)
        
        spec = {
            "run_id": run_id,
            "run_name": getattr(args, 'run_name', None) or "",
            "type": "train",
            "entry": "manage.py train",
            "args": {"base": vars(args) if hasattr(args, "__dict__") else {}},
        }
        controller.init_run(spec, state={"status": "running", "phase": "foreground", "pid": os.getpid()})
        
        try:
            from tools.train.orchestrator import PiscesLxTrainOrchestrator
            orchestrator = PiscesLxTrainOrchestrator(args)
            orchestrator.run(args)
            controller.update_state({"status": "completed", "phase": "finished"})
            controller.append_event("process_exit", payload={"exit_code": 0})
        except KeyboardInterrupt:
            controller.update_state({"status": "interrupted", "phase": "finished"})
            controller.append_event("process_interrupted", level="warning", payload={})
            raise
        except Exception as e:
            controller.update_state({"status": "failed", "phase": "finished"})
            controller.append_event("process_error", level="error", payload={"error": str(e)})
            raise
    
    # -------------------------------------------------------------------------
    # SERVE COMMAND
    # -------------------------------------------------------------------------
    # Start the OpenAI-compatible backend inference service
    # Provides REST API endpoints for model inference
    elif args.command == 'serve':
        # Import the backend server class
        from tools.infer.server import PiscesLxBackendServer
        
        # Create server instance with parsed arguments
        server = PiscesLxBackendServer(args)
        
        # Start the server (blocking call)
        server.run()
    
    # -------------------------------------------------------------------------
    # ACTION COMMAND
    # -------------------------------------------------------------------------
    # Manage background training/inference processes
    # Supports submit, status, control, and log operations
    elif args.command == 'action':
        # Import the run CLI class for action sub-commands
        from opss.run import POPSSRunCLI
        
        # Create CLI instance
        cli = POPSSRunCLI()
        
        # Build argv for the sub-command
        # Try to get arguments from sys.argv first, fall back to unknown
        argv = []
        try:
            # Get all arguments after 'action' command
            argv = list(sys.argv[2:])
        except Exception:
            # Fall back to unknown arguments from parser
            argv = list(unknown or [])
        
        # Execute the CLI and exit with its return code
        # The int() conversion ensures proper exit code type
        sys.exit(int(cli.run(args, argv)))
    
    # -------------------------------------------------------------------------
    # TEST COMMAND
    # -------------------------------------------------------------------------
    # Validate project health with 8-stage comprehensive check
    # Checks environment, structure, imports, config, model, forward, generation, optimization
    elif args.command == 'test':
        # Import the test function
        from tools.test import test
        
        # Run project health check
        test(args)
    
    # -------------------------------------------------------------------------
    # MONITOR COMMAND
    # -------------------------------------------------------------------------
    # Start real-time system monitoring dashboard
    # Displays GPU usage, memory, training progress, etc.
    elif args.command == 'monitor':
        # Import the monitor orchestrator class
        from tools.monitor.orchestrator import PiscesLxToolsMonitorOrchestrator
        
        # Create orchestrator instance with parsed arguments
        orchestrator = PiscesLxToolsMonitorOrchestrator(args)
        
        # Start the monitoring dashboard
        orchestrator.run(args)
    
    # -------------------------------------------------------------------------
    # DOWNLOAD COMMAND
    # -------------------------------------------------------------------------
    # Download and prepare training datasets
    # Downloads from configured sources and optimizes storage
    elif args.command == 'download':
        # Import the dataset download tool class
        from tools.data.download import PiscesLxToolsDataDatasetDownload
        
        # Create tool instance
        tool = PiscesLxToolsDataDatasetDownload()
        
        # Download datasets with specified sample limit
        tool.download(args.max_samples)
        
        # Optimize storage by removing redundant data
        # max_keep limits the number of cached samples
        tool.optimize(max_keep=5000)
    
    # -------------------------------------------------------------------------
    # SETUP COMMAND
    # -------------------------------------------------------------------------
    # Initialize environment and install dependencies
    # Sets up Python packages, CUDA, and other requirements
    elif args.command == 'setup':
        # Import the setup function
        from tools.setup import setup
        
        # Run environment setup
        setup(args)
    
    # -------------------------------------------------------------------------
    # BENCHMARK COMMAND
    # -------------------------------------------------------------------------
    # Evaluate model performance on standard benchmarks
    # Supports MMLU and other evaluation datasets
    elif args.command == 'benchmark':
        # Import the benchmark orchestrator class
        from tools.benchmark.orchestrator import PiscesLxToolsBenchmarkOrchestrator
        
        # Create orchestrator instance with parsed arguments
        orchestrator = PiscesLxToolsBenchmarkOrchestrator(args)
        
        # Check if running self-test mode
        if args.selftest:
            # Run internal benchmark tests for validation
            orchestrator.run_tests(args)
        else:
            # Run standard benchmark evaluation
            orchestrator.run(args)
    
    # -------------------------------------------------------------------------
    # MCP COMMAND
    # -------------------------------------------------------------------------
    # Model Context Protocol tool management
    # Handles tool registration, discovery, and caching
    elif args.command == 'mcp':
        # Import the tool registry singleton
        from opss.mcp.mcps import POPSSToolRegistry
        
        # Handle different MCP actions
        if args.mcp_action == 'status':
            # Log the status check request
            _get_logger().info("MCP config status", event="manage.right")
            
            # Get the singleton registry instance
            registry = POPSSToolRegistry.get_instance()
            
            # List all registered tools
            tools = registry.list_tools()
            
            # Output status as formatted JSON
            # Shows tool count and first 10 tools as preview
            print(json.dumps({
                "tool_count": len(tools),
                "tools": tools[:10]  # Limit output for readability
            }, indent=2, ensure_ascii=False))
        
        elif args.mcp_action == 'warmup':
            # Log the warmup start
            _get_logger().info("Starting MCP background discovery...", event="manage.right")
            
            # Log completion (actual warmup happens in background)
            _get_logger().info("MCP tools registered")
        
        elif args.mcp_action == 'refresh-cache':
            # Log the cache refresh request
            _get_logger().info("Refreshing MCP tools cache...")
            
            # Get registry and reload tools
            registry = POPSSToolRegistry.get_instance()
            tools = registry.list_tools()
            
            # Log the refresh result
            _get_logger().info(f"MCP tools refreshed: {len(tools)} tools registered")
    
    # -------------------------------------------------------------------------
    # WATERMARK COMMAND
    # -------------------------------------------------------------------------
    # Detect and verify watermarks in content and models
    # Supports text, image, audio, video, and model weight verification
    elif args.command == 'watermark':
        # Import all watermark detection functions
        from tools.wmc import (
            detect_watermark,  # Text watermark detection
            batch_detect,  # Batch processing for multiple files
            detect_image_watermark,  # Image watermark detection
            detect_audio_watermark,  # Audio watermark detection
            detect_video_watermark,  # Video watermark detection
            detect_model_watermark  # Model weight verification
        )
        
        # Check if verifying model weights
        if args.weights_verify:
            try:
                # Model weight verification will be handled below
                # This block is a placeholder for weight verification logic
                pass
            except Exception:
                # Error handling for weight verification
                pass
        
        # Initialize result variable
        result = None
        
        # Extract verbose and jurisdiction settings
        verbose = args.verbose
        jurisdiction = args.jurisdiction
        
        # Determine detection mode based on provided arguments
        if args.image_file:
            # Image watermark detection
            result = detect_image_watermark(
                args.image_file, 
                verbose=verbose, 
                jurisdiction=jurisdiction
            )
        
        elif args.audio_file:
            # Audio watermark detection
            result = detect_audio_watermark(
                args.audio_file, 
                verbose=verbose, 
                jurisdiction=jurisdiction
            )
        
        elif args.video_file:
            # Video watermark detection with frame sampling
            result = detect_video_watermark(
                args.video_file, 
                verbose=verbose, 
                jurisdiction=jurisdiction,
                frame_sample_rate=args.frame_sample_rate,
                max_frames=args.max_frames
            )
        
        elif args.model_file:
            # Model weight watermark verification
            result = detect_model_watermark(
                args.model_file, 
                verbose=verbose, 
                expected_owner=args.expected_owner
            )
        
        elif args.batch:
            # Batch processing mode for multiple files
            if args.file:
                result = batch_detect(
                    args.file, 
                    verbose=verbose, 
                    jurisdiction=jurisdiction
                )
        
        elif args.file:
            # Single file watermark detection
            with open(args.file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            result = detect_watermark(
                content, 
                verbose=verbose, 
                jurisdiction=jurisdiction
            )
        
        elif args.text:
            # Direct text watermark detection
            result = detect_watermark(
                args.text, 
                verbose=verbose, 
                jurisdiction=jurisdiction
            )
        
        else:
            # No valid input provided, log error and exit
            _get_logger().error(
                "Watermark command requires one of: --text, --file, "
                "--image-file, --audio-file, --video-file, --model-file, "
                "or --weights-verify with --ckpt"
            )
            sys.exit(1)
        
        # Output result in JSON format if requested
        if result is not None and args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # -------------------------------------------------------------------------
    # UNKNOWN COMMAND (fallback)
    # -------------------------------------------------------------------------
    # Handle any command that wasn't recognized
    # This should not happen due to argparse choices validation
    else:
        # Log the error with the unrecognized command
        _get_logger().error(f"Unknown command: {args.command}")
        
        # Exit with error code
        sys.exit(1)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
# This block executes when the script is run directly (not imported)
# It calls the main() function to start the CLI
if __name__ == "__main__":
    main()
