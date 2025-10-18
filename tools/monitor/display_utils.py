#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import os
import platform
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from utils import PiscesLxCoreLog


class PiscesLxToolsMonitorDisplayUtils:
    """Display utilities for system monitoring - internal utilities."""
    
    @staticmethod
    def get_progress_bar(percent: float, length: int = 30) -> str:
        """Create a progress bar string."""
        filled = int(length * percent / 100)
        bar = '�? * filled + '�? * (length - filled)
        return f"[{bar}] {percent:.1f}%"

    @staticmethod
    def bytes_to_human(n: int) -> str:
        """Convert bytes to human-readable format."""
        symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
        prefix = {}
        for i, s in enumerate(symbols):
            prefix[s] = 1 << (i + 1) * 10
        
        for s in reversed(symbols):
            if n >= prefix[s]:
                value = float(n) / prefix[s]
                return f'{value:.2f}{s}B'
        return f"{n}B"


class PiscesLxToolsMonitorDisplay:
    """Display utilities for system monitoring."""
    
    def __init__(self):
        """Initialize display utilities."""
        pass
    
    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def format_header(self) -> str:
        """Format the monitor header."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
�?                                   PiscesL1 System Monitor                                      �?
�?                                         {current_time}                                        �?
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
    
    def format_cpu_section(self, stats: Dict[str, Any]) -> str:
        """Format CPU statistics section."""
        if not stats or 'cpu' not in stats:
            return ""
        
        cpu_info = stats['cpu']
        cpu_percent_total = stats.get('cpu_percent_total', 0)
        cpu_freq = stats.get('cpu_freq', [])
        
        lines = ["�?🖥�? CPU Status:                                                                                �?]
        lines.append(f"�?   Total Usage: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(cpu_percent_total)}                                         �?)
        
        # Per-core usage
        if 'cpu_percent_per_core' in stats:
            per_core = stats['cpu_percent_per_core']
            if per_core:
                core_info = "�?   Per Core: "
                for i, usage in enumerate(per_core[:8]):  # Show first 8 cores
                    core_info += f"C{i}:{usage:.0f}% "
                lines.append(core_info.ljust(98) + "�?)
        
        # CPU frequency
        if cpu_freq and len(cpu_freq) > 0:
            avg_freq = sum(freq.current for freq in cpu_freq if hasattr(freq, 'current')) / len(cpu_freq)
            max_freq = max(freq.max for freq in cpu_freq if hasattr(freq, 'max'))
            lines.append(f"�?   Frequency: {avg_freq/1000:.2f} GHz (max: {max_freq/1000:.2f} GHz)" + " " * (57 - len(f"{avg_freq/1000:.2f} GHz (max: {max_freq/1000:.2f} GHz)")) + "�?)
        
        lines.append("�? + " " * 98 + "�?)
        return "\n".join(lines)
    
    def format_memory_section(self, stats: Dict[str, Any]) -> str:
        """Format memory statistics section."""
        if not stats or 'memory' not in stats:
            return ""
        
        memory = stats['memory']
        lines = ["�?💾 Memory Status:                                                                               �?]
        lines.append(f"�?   Usage: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(memory.get('percent', 0))}                                            �?)
        lines.append(f"�?   Total: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(memory.get('total', 0))} | Used: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(memory.get('used', 0))} | Free: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(memory.get('free', 0))}" + " " * (23 - len(f"Total: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(memory.get('total', 0))} | Used: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(memory.get('used', 0))} | Free: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(memory.get('free', 0))}")) + "�?)
        
        # Swap information
        if 'swap' in stats and stats['swap']:
            swap = stats['swap']
            if swap.get('total', 0) > 0:
                lines.append(f"�?   Swap: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(swap.get('percent', 0))} | Total: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(swap.get('total', 0))} | Used: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(swap.get('used', 0))}" + " " * (15 - len(f"Swap: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(swap.get('percent', 0))} | Total: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(swap.get('total', 0))} | Used: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(swap.get('used', 0))}")) + "�?)
        
        lines.append("�? + " " * 98 + "�?)
        return "\n".join(lines)
    
    def format_gpu_section(self, stats: Dict[str, Any]) -> str:
        """Format GPU statistics section."""
        if not stats or 'gpu' not in stats or not stats['gpu']:
            return ""
        
        gpu_list = stats['gpu']
        lines = ["�?🎮 GPU Status:                                                                                  �?]
        
        for i, gpu in enumerate(gpu_list):
            name = gpu.get('name', 'Unknown')
            util = gpu.get('util', 0)
            mem_percent = gpu.get('mem_percent', 0)
            mem_used = gpu.get('mem_used', 0)
            mem_total = gpu.get('mem_total', 0)
            
            lines.append(f"�?   GPU {i}: {name[:20]}" + " " * (20 - len(name[:20])) + f"| Util: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(util)} | Mem: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(mem_percent)} �?)
            lines.append(f"�?         Memory: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(mem_used)} / {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(mem_total)}" + " " * (60 - len(f"Memory: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(mem_used)} / {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(mem_total)}")) + "�?)
        
        lines.append("�? + " " * 98 + "�?)
        return "\n".join(lines)
    
    def format_disk_section(self, stats: Dict[str, Any]) -> str:
        """Format disk statistics section."""
        if not stats or 'disk' not in stats:
            return ""
        
        disk = stats['disk']
        lines = ["�?💽 Disk Status:                                                                                 �?]
        lines.append(f"�?   Usage: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(disk.get('percent', 0))}                                           �?)
        lines.append(f"�?   Total: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk.get('total', 0))} | Used: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk.get('used', 0))} | Free: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk.get('free', 0))}" + " " * (22 - len(f"Total: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk.get('total', 0))} | Used: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk.get('used', 0))} | Free: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk.get('free', 0))}")) + "�?)
        
        # Disk I/O if available
        if 'disk_io' in stats and stats['disk_io']:
            disk_io = stats['disk_io']
            lines.append(f"�?   I/O: Read: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk_io.get('read_bytes', 0))} | Write: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk_io.get('write_bytes', 0))}" + " " * (39 - len(f"I/O: Read: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk_io.get('read_bytes', 0))} | Write: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk_io.get('write_bytes', 0))}")) + "�?)
        
        lines.append("�? + " " * 98 + "�?)
        return "\n".join(lines)
    
    def format_network_section(self, stats: Dict[str, Any]) -> str:
        """Format network statistics section."""
        if not stats or 'network' not in stats:
            return ""
        
        network = stats['network']
        lines = ["�?🌐 Network Status:                                                                               �?]
        lines.append(f"�?   Sent: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(network.get('bytes_sent', 0))} | Received: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(network.get('bytes_recv', 0))}" + " " * (35 - len(f"Sent: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(network.get('bytes_sent', 0))} | Received: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(network.get('bytes_recv', 0))}")) + "�?)
        lines.append(f"�?   Packets: Sent: {network.get('packets_sent', 0):,} | Received: {network.get('packets_recv', 0):,}" + " " * (33 - len(f"Packets: Sent: {network.get('packets_sent', 0):,} | Received: {network.get('packets_recv', 0):,}")) + "�?)
        
        # Network errors if significant
        if network.get('errin', 0) > 0 or network.get('errout', 0) > 0:
            lines.append(f"�?   Errors: In: {network.get('errin', 0):,} | Out: {network.get('errout', 0):,}" + " " * (44 - len(f"Errors: In: {network.get('errin', 0):,} | Out: {network.get('errout', 0):,}")) + "�?)
        
        lines.append("�? + " " * 98 + "�?)
        return "\n".join(lines)
    
    def format_io_rates(self, stats: Dict[str, Any], last_net_io: Optional[Any], 
                       last_disk_io: Optional[Any], time_delta: float) -> str:
        """Format I/O rate calculations."""
        lines = []
        
        if time_delta > 0:
            # Network I/O rates
            if 'network' in stats and last_net_io:
                current_net = stats['network']
                net_sent_rate = (current_net.get('bytes_sent', 0) - last_net_io.bytes_sent) / time_delta
                net_recv_rate = (current_net.get('bytes_recv', 0) - last_net_io.bytes_recv) / time_delta
                
                if net_sent_rate > 0 or net_recv_rate > 0:
                    lines.append("�?📊 Network I/O Rates:                                                                             �?)
                    lines.append(f"�?   Upload: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(int(net_sent_rate))}/s | Download: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(int(net_recv_rate))}/s" + " " * (37 - len(f"Upload: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(int(net_sent_rate))}/s | Download: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(int(net_recv_rate))}/s")) + "�?)
            
            # Disk I/O rates
            if 'disk_io' in stats and last_disk_io and stats['disk_io']:
                current_disk = stats['disk_io']
                disk_read_rate = (current_disk.get('read_bytes', 0) - last_disk_io.read_bytes) / time_delta
                disk_write_rate = (current_disk.get('write_bytes', 0) - last_disk_io.write_bytes) / time_delta
                
                if disk_read_rate > 0 or disk_write_rate > 0:
                    if not lines:  # Add header if not already added
                        lines.append("�?📊 I/O Rates:                                                                                    �?)
                    lines.append(f"�?   Disk Read: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(int(disk_read_rate))}/s | Write: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(int(disk_write_rate))}/s" + " " * (32 - len(f"Disk Read: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(int(disk_read_rate))}/s | Write: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(int(disk_write_rate))}/s")) + "�?)
        
        if lines:
            lines.append("�? + " " * 98 + "�?)
        
        return "\n".join(lines)
    
    def format_footer(self) -> str:
        """Format the monitor footer."""
        return """
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
    
    def format_full_display(self, stats: Dict[str, Any], last_net_io: Optional[Any], 
                           last_disk_io: Optional[Any], time_delta: float) -> str:
        """Format complete monitoring display."""
        sections = [
            self.format_header(),
            self.format_cpu_section(stats),
            self.format_memory_section(stats),
            self.format_gpu_section(stats),
            self.format_disk_section(stats),
            self.format_network_section(stats),
            self.format_io_rates(stats, last_net_io, last_disk_io, time_delta),
            self.format_footer()
        ]
        
        return "\n".join(sections)
