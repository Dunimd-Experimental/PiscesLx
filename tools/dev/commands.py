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
PiscesLx Developer Mode Commands.

This module implements the command registry and executor for the developer mode,
providing a vim-style command interface for training debugging.

Available Commands:
    Memory Commands:
        :mem [module]     - Show memory details for specific module or all
        :mem-gpu          - Show GPU memory allocation
        :mem-cpu          - Show CPU memory usage
    
    Model Commands:
        :layer <n>        - Show layer information (parameters, memory)
        :layers           - List all layers with parameter counts
        :grad             - Show gradient statistics
        :grad-norm        - Show gradient norm history
    
    Training Control:
        :pause            - Pause training
        :resume           - Resume training
        :save [name]      - Save checkpoint with optional name
        :lr <value>       - Adjust learning rate dynamically
        :batch <size>     - Adjust batch size (if supported)
    
    Configuration:
        :config           - Show current training configuration
        :config-model     - Show model configuration
        :config-data      - Show data configuration
    
    Monitoring:
        :watch <var>      - Watch a variable value
        :watch-clear      - Clear watched variables
        :profile [type]   - Performance profiling (gpu/cpu/memory)
    
    Intervention:
        :inject <target>  - Force injection into model
        :freeze <layer>   - Freeze a specific layer
        :unfreeze <layer> - Unfreeze a specific layer
        :nan-check        - Check for NaN values in model
    
    Help:
        :help             - Show all available commands
        :help <cmd>       - Show help for specific command
        :q                - Close overlay / return to main view

Architecture:
    The command system follows a registry pattern where each command is
    registered with a handler function. Commands are parsed and executed
    by the PiscesLxDevModeCommands class.
"""

import os
import sys
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from utils.paths import get_log_file
from utils.dc import PiscesLxLogger


_LOG = PiscesLxLogger("PiscesLx.Tools.Dev", file_path=get_log_file("PiscesLx.Tools.Dev"), enable_file=True)


class PiscesLxDevModeCommands:
    """
    Command registry and executor for developer mode.
    
    This class manages command registration, parsing, and execution.
    It provides a vim-style command interface for training debugging.
    
    Attributes:
        _commands: Dictionary mapping command names to handlers
        _aliases: Dictionary mapping command aliases to canonical names
        _watch_list: List of variables being watched
        _manager: Reference to the PiscesLxDevModeManager instance
    
    Example:
        >>> commands = PiscesLxDevModeCommands(manager)
        >>> result = commands.execute(":mem attention", trainer)
        >>> print(result)
        "Memory for attention: 4.2GB"
    """
    
    def __init__(self, manager: Any):
        """
        Initialize the command registry.
        
        Args:
            manager: The PiscesLxDevModeManager instance
        """
        self._manager = manager
        self._commands: Dict[str, Callable] = {}
        self._aliases: Dict[str, str] = {}
        self._watch_list: List[str] = []
        self._grad_norm_history: deque = deque(maxlen=100)
        
        self._register_builtin_commands()
        _LOG.info("PiscesLxDevModeCommands initialized")
    
    def _register_builtin_commands(self) -> None:
        """Register all built-in commands."""
        self._commands['mem'] = self._cmd_mem
        self._commands['mem-gpu'] = self._cmd_mem_gpu
        self._commands['mem-cpu'] = self._cmd_mem_cpu
        self._commands['layer'] = self._cmd_layer
        self._commands['layers'] = self._cmd_layers
        self._commands['grad'] = self._cmd_grad
        self._commands['grad-norm'] = self._cmd_grad_norm
        self._commands['pause'] = self._cmd_pause
        self._commands['resume'] = self._cmd_resume
        self._commands['save'] = self._cmd_save
        self._commands['lr'] = self._cmd_lr
        self._commands['batch'] = self._cmd_batch
        self._commands['config'] = self._cmd_config
        self._commands['config-model'] = self._cmd_config_model
        self._commands['config-data'] = self._cmd_config_data
        self._commands['watch'] = self._cmd_watch
        self._commands['watch-clear'] = self._cmd_watch_clear
        self._commands['profile'] = self._cmd_profile
        self._commands['inject'] = self._cmd_inject
        self._commands['freeze'] = self._cmd_freeze
        self._commands['unfreeze'] = self._cmd_unfreeze
        self._commands['nan-check'] = self._cmd_nan_check
        self._commands['help'] = self._cmd_help
        self._commands['h'] = self._cmd_help
        self._commands['q'] = self._cmd_quit
        self._commands['quit'] = self._cmd_quit
        
        self._aliases['m'] = 'mem'
        self._aliases['l'] = 'layer'
        self._aliases['s'] = 'save'
        self._aliases['c'] = 'config'
        self._aliases['w'] = 'watch'
        self._aliases['p'] = 'profile'
    
    def execute(self, command_str: str, trainer: Optional[Any] = None) -> Tuple[str, bool]:
        """
        Execute a command string.
        
        This method parses the command string and executes the corresponding
        handler function, returning the result for display.
        
        Args:
            command_str: The command string (e.g., ":mem attention")
            trainer: Optional trainer instance for context
        
        Returns:
            Tuple[str, bool]: (result_text, is_overlay)
                - result_text: The text to display
                - is_overlay: Whether to show as overlay (True) or inline (False)
        """
        command_str = command_str.strip()
        if not command_str:
            return "", False
        
        if command_str.startswith(':'):
            command_str = command_str[1:]
        
        parts = command_str.split(None, 1)
        if not parts:
            return "", False
        
        cmd_name = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""
        
        cmd_name = self._aliases.get(cmd_name, cmd_name)
        
        if cmd_name not in self._commands:
            return f"Unknown command: {cmd_name}. Type :help for available commands.", True
        
        try:
            handler = self._commands[cmd_name]
            result = handler(args_str, trainer)
            
            if isinstance(result, tuple):
                return result
            return str(result), True
        except Exception as e:
            _LOG.error("Command execution failed", command=cmd_name, error=str(e))
            return f"Error executing command '{cmd_name}': {str(e)}", True
    
    def _get_model(self, trainer: Any) -> Optional[Any]:
        """Get the model from trainer."""
        if trainer is None:
            return None
        return getattr(trainer, 'model', None)
    
    def _cmd_mem(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """
        Show memory details.
        
        Usage:
            :mem           - Show all memory info
            :mem attention - Show memory for attention layers
            :mem embed     - Show memory for embedding layers
        """
        module_name = args.strip().lower() if args else None
        
        model = self._get_model(trainer)
        if model is None:
            return "No model attached", True
        
        lines = ["[MEMORY DETAILS]", ""]
        
        if module_name:
            lines.append(f"Module: {module_name}")
            lines.append("-" * 40)
            module_mem = self._get_module_memory(model, module_name)
            for name, mem in module_mem.items():
                lines.append(f"  {name}: {mem:.2f} GB")
        else:
            total_params = sum(p.numel() for p in model.parameters())
            total_mem = total_params * 4 / (1024**3)
            
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
                gpu_max = torch.cuda.max_memory_allocated() / (1024**3)
                
                lines.append(f"GPU Allocated: {gpu_allocated:.2f} GB")
                lines.append(f"GPU Reserved:  {gpu_reserved:.2f} GB")
                lines.append(f"GPU Peak:      {gpu_max:.2f} GB")
                lines.append("")
            
            lines.append(f"Model Parameters: {total_params:,}")
            lines.append(f"Model Memory (FP32): {total_mem:.2f} GB")
            lines.append("")
            lines.append("Module Breakdown:")
            lines.append("-" * 40)
            
            module_mems = self._get_all_module_memory(model)
            for name, mem in sorted(module_mems.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"  {name}: {mem:.2f} GB")
        
        lines.append("")
        lines.append("Press 'q' to return")
        
        return "\n".join(lines), True
    
    def _get_module_memory(self, model: Any, module_name: str) -> Dict[str, float]:
        """Get memory usage for a specific module type."""
        result = {}
        total = 0.0
        
        for name, module in model.named_modules():
            if module_name in name.lower() or module_name in type(module).__name__.lower():
                mem = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024**3)
                result[name] = mem
                total += mem
        
        if not result:
            result[f"total_{module_name}"] = 0.0
        else:
            result[f"total"] = total
        
        return result
    
    def _get_all_module_memory(self, model: Any) -> Dict[str, float]:
        """Get memory usage grouped by module type."""
        result = {}
        
        for name, module in model.named_modules():
            if not name:
                continue
            
            module_type = type(module).__name__
            mem = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024**3)
            
            if module_type not in result:
                result[module_type] = 0.0
            result[module_type] += mem
        
        return result
    
    def _cmd_mem_gpu(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Show GPU memory allocation details."""
        if not torch.cuda.is_available():
            return "CUDA not available", True
        
        lines = ["[GPU MEMORY DETAILS]", ""]
        
        for i in range(torch.cuda.device_count()):
            lines.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            lines.append(f"  Allocated: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
            lines.append(f"  Reserved:  {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
            lines.append(f"  Peak:      {torch.cuda.max_memory_allocated(i) / (1024**3):.2f} GB")
            lines.append("")
        
        lines.append("Press 'q' to return")
        return "\n".join(lines), True
    
    def _cmd_mem_cpu(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Show CPU memory usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            
            lines = ["[CPU MEMORY DETAILS]", ""]
            lines.append(f"Total:     {mem.total / (1024**3):.2f} GB")
            lines.append(f"Available: {mem.available / (1024**3):.2f} GB")
            lines.append(f"Used:      {mem.used / (1024**3):.2f} GB")
            lines.append(f"Percent:   {mem.percent}%")
            lines.append("")
            lines.append("Press 'q' to return")
            return "\n".join(lines), True
        except ImportError:
            return "psutil not available for CPU memory info", True
    
    def _cmd_layer(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Show specific layer information."""
        if not args.strip():
            return "Usage: :layer <layer_index>", True
        
        try:
            layer_idx = int(args.strip())
        except ValueError:
            return "Invalid layer index. Use: :layer <number>", True
        
        model = self._get_model(trainer)
        if model is None:
            return "No model attached", True
        
        lines = [f"[LAYER {layer_idx} DETAILS]", ""]
        
        layer_found = False
        for name, module in model.named_modules():
            if f".{layer_idx}." in name or name.endswith(f".{layer_idx}"):
                layer_found = True
                lines.append(f"Name: {name}")
                lines.append(f"Type: {type(module).__name__}")
                
                params = sum(p.numel() for p in module.parameters())
                mem = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024**3)
                
                lines.append(f"Parameters: {params:,}")
                lines.append(f"Memory: {mem:.4f} GB")
                lines.append("")
                
                for pname, param in module.named_parameters():
                    lines.append(f"  {pname}: {param.shape} ({param.dtype})")
        
        if not layer_found:
            lines.append(f"Layer {layer_idx} not found")
        
        lines.append("")
        lines.append("Press 'q' to return")
        return "\n".join(lines), True
    
    def _cmd_layers(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """List all layers with parameter counts."""
        model = self._get_model(trainer)
        if model is None:
            return "No model attached", True
        
        lines = ["[LAYER SUMMARY]", ""]
        
        layer_info = []
        for name, module in model.named_modules():
            if not name:
                continue
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                layer_info.append((name, type(module).__name__, params))
        
        layer_info.sort(key=lambda x: -x[2])
        
        for name, mtype, params in layer_info[:20]:
            lines.append(f"{name[:40]:<40} {mtype:<20} {params:>12,}")
        
        if len(layer_info) > 20:
            lines.append(f"... and {len(layer_info) - 20} more layers")
        
        lines.append("")
        lines.append("Press 'q' to return")
        return "\n".join(lines), True
    
    def _cmd_grad(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Show gradient statistics."""
        model = self._get_model(trainer)
        if model is None:
            return "No model attached", True
        
        lines = ["[GRADIENT STATISTICS]", ""]
        
        total_norm = 0.0
        param_count = 0
        nan_count = 0
        inf_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                
                if torch.isnan(param.grad).any():
                    nan_count += 1
                if torch.isinf(param.grad).any():
                    inf_count += 1
        
        total_norm = total_norm ** 0.5
        
        lines.append(f"Total Gradient Norm: {total_norm:.6f}")
        lines.append(f"Parameters with Gradients: {param_count}")
        lines.append(f"NaN Gradients: {nan_count}")
        lines.append(f"Inf Gradients: {inf_count}")
        
        if self._grad_norm_history:
            lines.append("")
            lines.append("Recent Gradient Norms:")
            for i, norm in enumerate(list(self._grad_norm_history)[-10:]):
                lines.append(f"  Step -{len(self._grad_norm_history) - i}: {norm:.6f}")
        
        lines.append("")
        lines.append("Press 'q' to return")
        return "\n".join(lines), True
    
    def _cmd_grad_norm(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Show gradient norm history."""
        lines = ["[GRADIENT NORM HISTORY]", ""]
        
        if not self._grad_norm_history:
            lines.append("No gradient norm history available")
        else:
            for i, norm in enumerate(list(self._grad_norm_history)):
                lines.append(f"Step {i + 1}: {norm:.6f}")
        
        lines.append("")
        lines.append("Press 'q' to return")
        return "\n".join(lines), True
    
    def _cmd_pause(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Pause training."""
        self._manager.pause()
        return "Training paused. Use :resume to continue.", False
    
    def _cmd_resume(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Resume training."""
        self._manager.resume()
        return "Training resumed.", False
    
    def _cmd_save(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Save checkpoint."""
        if trainer is None:
            return "No trainer attached", True
        
        name = args.strip() or f"dev_save_{int(time.time())}"
        
        try:
            output_dir = getattr(trainer.config, 'output_dir', '.pisceslx/ckpt')
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{name}.pt")
            
            if hasattr(trainer, 'save_checkpoint'):
                trainer.save_checkpoint(filepath)
                return f"Checkpoint saved to {filepath}", False
            else:
                return "Trainer does not support save_checkpoint", True
        except Exception as e:
            return f"Failed to save checkpoint: {str(e)}", True
    
    def _cmd_lr(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Adjust learning rate."""
        if not args.strip():
            if trainer is None:
                return "No trainer attached", True
            
            optimizer = getattr(trainer, 'optimizer', None)
            if optimizer is None:
                return "No optimizer available", True
            
            current_lr = optimizer.param_groups[0]['lr']
            return f"Current learning rate: {current_lr:.2e}", True
        
        try:
            new_lr = float(args.strip())
        except ValueError:
            return "Invalid learning rate value", True
        
        if trainer is None:
            return "No trainer attached", True
        
        optimizer = getattr(trainer, 'optimizer', None)
        if optimizer is None:
            return "No optimizer available", True
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        return f"Learning rate set to {new_lr:.2e}", False
    
    def _cmd_batch(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Show or adjust batch size."""
        if trainer is None:
            return "No trainer attached", True
        
        config = getattr(trainer, 'config', None)
        if config is None:
            return "No config available", True
        
        data_config = getattr(config, 'data', None)
        if data_config is None:
            return "No data config available", True
        
        current_batch = getattr(data_config, 'batch_size', 'unknown')
        
        if not args.strip():
            return f"Current batch size: {current_batch}", True
        
        try:
            new_batch = int(args.strip())
            if new_batch < 1:
                return "Batch size must be positive", True
            data_config.batch_size = new_batch
            return f"Batch size set to {new_batch} (takes effect on next dataloader creation)", False
        except ValueError:
            return "Invalid batch size value", True
    
    def _cmd_config(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Show current training configuration."""
        if trainer is None:
            return "No trainer attached", True
        
        config = getattr(trainer, 'config', None)
        if config is None:
            return "No config available", True
        
        lines = ["[TRAINING CONFIGURATION]", ""]
        
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
        
        for key, value in sorted(config_dict.items()):
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in sorted(value.items()):
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        
        lines.append("")
        lines.append("Press 'q' to return")
        return "\n".join(lines), True
    
    def _cmd_config_model(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Show model configuration."""
        model = self._get_model(trainer)
        if model is None:
            return "No model attached", True
        
        lines = ["[MODEL CONFIGURATION]", ""]
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        lines.append(f"Total Parameters: {total_params:,}")
        lines.append(f"Trainable Parameters: {trainable_params:,}")
        lines.append(f"Frozen Parameters: {total_params - trainable_params:,}")
        lines.append("")
        
        lines.append("Model Structure:")
        lines.append(str(model)[:500])
        if len(str(model)) > 500:
            lines.append("... (truncated)")
        
        lines.append("")
        lines.append("Press 'q' to return")
        return "\n".join(lines), True
    
    def _cmd_config_data(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Show data configuration."""
        if trainer is None:
            return "No trainer attached", True
        
        config = getattr(trainer, 'config', None)
        if config is None:
            return "No config available", True
        
        data_config = getattr(config, 'data', None)
        if data_config is None:
            return "No data config available", True
        
        lines = ["[DATA CONFIGURATION]", ""]
        
        if hasattr(data_config, 'to_dict'):
            data_dict = data_config.to_dict()
        else:
            data_dict = {k: v for k, v in vars(data_config).items() if not k.startswith('_')}
        
        for key, value in sorted(data_dict.items()):
            lines.append(f"{key}: {value}")
        
        lines.append("")
        lines.append("Press 'q' to return")
        return "\n".join(lines), True
    
    def _cmd_watch(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Watch a variable."""
        var_name = args.strip()
        if not var_name:
            if not self._watch_list:
                return "No variables being watched. Use :watch <var_name>", True
            
            lines = ["[WATCHED VARIABLES]", ""]
            for var in self._watch_list:
                value = self._get_variable_value(var, trainer)
                lines.append(f"{var}: {value}")
            lines.append("")
            lines.append("Press 'q' to return")
            return "\n".join(lines), True
        
        if var_name not in self._watch_list:
            self._watch_list.append(var_name)
        
        value = self._get_variable_value(var_name, trainer)
        return f"Watching {var_name}: {value}", False
    
    def _get_variable_value(self, var_name: str, trainer: Any) -> str:
        """Get the value of a watched variable."""
        if trainer is None:
            return "N/A (no trainer)"
        
        parts = var_name.split('.')
        obj = trainer
        
        try:
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif part.isdigit() and isinstance(obj, (list, tuple)):
                    obj = obj[int(part)]
                else:
                    return f"Variable not found: {var_name}"
            
            if isinstance(obj, torch.Tensor):
                return f"Tensor(shape={list(obj.shape)}, dtype={obj.dtype})"
            return str(obj)[:100]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _cmd_watch_clear(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Clear watched variables."""
        self._watch_list.clear()
        return "Watch list cleared", False
    
    def _cmd_profile(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Performance profiling."""
        profile_type = args.strip().lower() or 'gpu'
        
        lines = [f"[PERFORMANCE PROFILE - {profile_type.upper()}]", ""]
        
        if profile_type in ('gpu', 'cuda'):
            if not torch.cuda.is_available():
                return "CUDA not available", True
            
            lines.append(f"Device: {torch.cuda.get_device_name(0)}")
            lines.append(f"CUDA Version: {torch.version.cuda}")
            lines.append("")
            lines.append("Memory:")
            lines.append(f"  Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
            lines.append(f"  Reserved:  {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
            lines.append("")
            
            try:
                props = torch.cuda.get_device_properties(0)
                lines.append("Device Properties:")
                lines.append(f"  Compute Capability: {props.major}.{props.minor}")
                lines.append(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
                lines.append(f"  Multiprocessors: {props.multi_processor_count}")
            except Exception:
                pass
        
        elif profile_type in ('cpu', 'system'):
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                
                lines.append(f"CPU Usage: {cpu_percent}%")
                lines.append(f"Memory Usage: {mem.percent}%")
                lines.append(f"Available Memory: {mem.available / (1024**3):.2f} GB")
            except ImportError:
                lines.append("psutil not available")
        
        elif profile_type == 'memory':
            if torch.cuda.is_available():
                lines.append("GPU Memory Summary:")
                lines.append(str(torch.cuda.memory_summary()))
            else:
                lines.append("CUDA not available")
        
        lines.append("")
        lines.append("Press 'q' to return")
        return "\n".join(lines), True
    
    def _cmd_inject(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Force injection into model."""
        target = args.strip()
        if not target:
            return "Usage: :inject <target> [value]", True
        
        parts = target.split(None, 1)
        param_path = parts[0]
        value_str = parts[1] if len(parts) > 1 else None
        
        model = self._get_model(trainer)
        if model is None:
            return "No model attached", True
        
        try:
            parts_path = param_path.split('.')
            obj = model
            
            for part in parts_path[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return f"Path not found: {param_path}", True
            
            final_attr = parts_path[-1]
            if not hasattr(obj, final_attr):
                return f"Attribute not found: {final_attr}", True
            
            param = getattr(obj, final_attr)
            
            if value_str == 'zeros':
                if isinstance(param, torch.nn.Parameter):
                    param.data.zero_()
                    return f"Set {param_path} to zeros", False
            elif value_str == 'freeze':
                param.requires_grad = False
                return f"Frozen {param_path}", False
            elif value_str == 'unfreeze':
                param.requires_grad = True
                return f"Unfrozen {param_path}", False
            
            return f"Current value of {param_path}: shape={param.shape if hasattr(param, 'shape') else 'N/A'}", True
        except Exception as e:
            return f"Error: {str(e)}", True
    
    def _cmd_freeze(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Freeze a layer or parameter."""
        target = args.strip()
        if not target:
            return "Usage: :freeze <layer_name>", True
        
        model = self._get_model(trainer)
        if model is None:
            return "No model attached", True
        
        frozen_count = 0
        for name, param in model.named_parameters():
            if target in name:
                param.requires_grad = False
                frozen_count += 1
        
        if frozen_count > 0:
            return f"Frozen {frozen_count} parameters matching '{target}'", False
        return f"No parameters found matching '{target}'", True
    
    def _cmd_unfreeze(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Unfreeze a layer or parameter."""
        target = args.strip()
        if not target:
            return "Usage: :unfreeze <layer_name>", True
        
        model = self._get_model(trainer)
        if model is None:
            return "No model attached", True
        
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if target in name:
                param.requires_grad = True
                unfrozen_count += 1
        
        if unfrozen_count > 0:
            return f"Unfrozen {unfrozen_count} parameters matching '{target}'", False
        return f"No parameters found matching '{target}'", True
    
    def _cmd_nan_check(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Check for NaN values in model."""
        model = self._get_model(trainer)
        if model is None:
            return "No model attached", True
        
        lines = ["[NaN CHECK]", ""]
        
        nan_params = []
        nan_grads = []
        inf_params = []
        inf_grads = []
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)
            
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
                if torch.isinf(param.grad).any():
                    inf_grads.append(name)
        
        if nan_params:
            lines.append(f"NaN in parameters ({len(nan_params)}):")
            for name in nan_params[:10]:
                lines.append(f"  {name}")
        else:
            lines.append("No NaN in parameters")
        
        if inf_params:
            lines.append(f"Inf in parameters ({len(inf_params)}):")
            for name in inf_params[:10]:
                lines.append(f"  {name}")
        else:
            lines.append("No Inf in parameters")
        
        if nan_grads:
            lines.append(f"NaN in gradients ({len(nan_grads)}):")
            for name in nan_grads[:10]:
                lines.append(f"  {name}")
        else:
            lines.append("No NaN in gradients")
        
        if inf_grads:
            lines.append(f"Inf in gradients ({len(inf_grads)}):")
            for name in inf_grads[:10]:
                lines.append(f"  {name}")
        else:
            lines.append("No Inf in gradients")
        
        lines.append("")
        lines.append("Press 'q' to return")
        return "\n".join(lines), True
    
    def _cmd_help(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Show help for commands."""
        specific_cmd = args.strip().lower() if args else None
        
        if specific_cmd:
            specific_cmd = self._aliases.get(specific_cmd, specific_cmd)
            help_texts = {
                'mem': ':mem [module] - Show memory details\n  Examples:\n    :mem          - Show all memory info\n    :mem attention - Show memory for attention layers',
                'layer': ':layer <n> - Show layer information\n  Example:\n    :layer 5 - Show details for layer 5',
                'grad': ':grad - Show gradient statistics\n  Displays total norm, NaN/Inf counts, and recent history',
                'pause': ':pause - Pause training\n  Use :resume to continue',
                'resume': ':resume - Resume paused training',
                'save': ':save [name] - Save checkpoint\n  Example:\n    :save my_checkpoint',
                'lr': ':lr [value] - Show or set learning rate\n  Examples:\n    :lr           - Show current LR\n    :lr 1e-5      - Set LR to 1e-5',
                'config': ':config - Show current training configuration',
                'watch': ':watch <var> - Watch a variable\n  Example:\n    :watch optimizer.param_groups.0.lr',
                'profile': ':profile [type] - Performance profiling\n  Types: gpu, cpu, memory\n  Example:\n    :profile gpu',
                'freeze': ':freeze <layer> - Freeze parameters matching pattern\n  Example:\n    :freeze layer.5',
            }
            
            if specific_cmd in help_texts:
                return help_texts[specific_cmd], True
            return f"No help available for: {specific_cmd}", True
        
        lines = ["[AVAILABLE COMMANDS]", ""]
        lines.append("Memory:")
        lines.append("  :mem [module]     - Show memory details")
        lines.append("  :mem-gpu          - Show GPU memory")
        lines.append("  :mem-cpu          - Show CPU memory")
        lines.append("")
        lines.append("Model:")
        lines.append("  :layer <n>        - Show layer information")
        lines.append("  :layers           - List all layers")
        lines.append("  :grad             - Show gradient statistics")
        lines.append("  :grad-norm        - Show gradient norm history")
        lines.append("")
        lines.append("Training Control:")
        lines.append("  :pause            - Pause training")
        lines.append("  :resume           - Resume training")
        lines.append("  :save [name]      - Save checkpoint")
        lines.append("  :lr [value]       - Show/set learning rate")
        lines.append("  :batch [size]     - Show/set batch size")
        lines.append("")
        lines.append("Configuration:")
        lines.append("  :config           - Show training config")
        lines.append("  :config-model     - Show model config")
        lines.append("  :config-data      - Show data config")
        lines.append("")
        lines.append("Monitoring:")
        lines.append("  :watch <var>      - Watch variable")
        lines.append("  :watch-clear      - Clear watch list")
        lines.append("  :profile [type]   - Performance profiling")
        lines.append("")
        lines.append("Intervention:")
        lines.append("  :inject <target>  - Force injection")
        lines.append("  :freeze <layer>   - Freeze layer")
        lines.append("  :unfreeze <layer> - Unfreeze layer")
        lines.append("  :nan-check        - Check for NaN values")
        lines.append("")
        lines.append("Other:")
        lines.append("  :help             - Show this help")
        lines.append("  :q                - Close overlay")
        lines.append("")
        lines.append("Press 'q' to return")
        return "\n".join(lines), True
    
    def _cmd_quit(self, args: str, trainer: Any) -> Tuple[str, bool]:
        """Close overlay."""
        return "", False
    
    def record_grad_norm(self, norm: float) -> None:
        """Record a gradient norm for history tracking."""
        self._grad_norm_history.append(norm)
