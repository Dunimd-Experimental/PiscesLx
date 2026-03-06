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

import os
import platform
from datetime import datetime
from typing import Dict, Any, Optional, List


class PiscesLxToolsMonitorDisplayUtils:
    """Display utilities for system monitoring - internal utilities."""
    
    @staticmethod
    def get_progress_bar(percent: float, length: int = 30) -> str:
        """Create a progress bar string."""
        if percent is None:
            percent = 0
        filled = int(length * percent / 100)
        bar = '█' * filled + '░' * (length - filled)
        return f"[{bar}] {percent:.1f}%"

    @staticmethod
    def bytes_to_human(n: int) -> str:
        """Convert bytes to human-readable format."""
        if n is None:
            return "0B"
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
        pass
    
    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def render(self, stats: Dict[str, Any], last_net_io, last_disk_io, last_time: float, alerts: List[str] = None):
        """Render the monitoring display."""
        import time
        import psutil
        
        self.clear_screen()
        
        print(f"--- PiscesLx System Monitor --- (Press Ctrl+C to exit)")
        print(f"System: {platform.system()} {platform.release()} | Time: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)
        
        print("--- CPU ---")
        cpu_total = stats.get('cpu_percent_total', 0) or 0
        print(f"Total: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(cpu_total)}")
        per_core = stats.get('cpu_percent_per_core', []) or []
        for i, percent in enumerate(per_core[:8]):
            print(f"  Core {i}: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(percent or 0, length=20)}")
        
        print("\n--- Memory ---")
        mem = stats.get('memory', {}) or {}
        print(f"RAM: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(mem.get('percent', 0) or 0)} "
              f"({PiscesLxToolsMonitorDisplayUtils.bytes_to_human(mem.get('used', 0))} / "
              f"{PiscesLxToolsMonitorDisplayUtils.bytes_to_human(mem.get('total', 0))})")
        swap = stats.get('swap', {}) or {}
        if swap.get('total', 0) > 0:
            print(f"Swap: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(swap.get('percent', 0) or 0)}")
        
        gpu_list = stats.get('gpu', []) or []
        if gpu_list:
            print("\n--- GPU ---")
            for i, gpu in enumerate(gpu_list):
                name = gpu.get('name', 'Unknown') or 'Unknown'
                print(f"GPU {i} ({name[:20]}):")
                print(f"  Util: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(gpu.get('util', 0) or 0)}")
                print(f"  Mem:  {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(gpu.get('mem_percent', 0) or 0)}")
        
        current_time = time.time()
        elapsed = (current_time - last_time) if last_time else 1
        if elapsed == 0:
            elapsed = 1
        
        print("\n--- Disk ---")
        current_disk_io = psutil.disk_io_counters()
        if last_disk_io and current_disk_io:
            read_speed = (current_disk_io.read_bytes - last_disk_io.read_bytes) / elapsed
            write_speed = (current_disk_io.write_bytes - last_disk_io.write_bytes) / elapsed
            print(f"I/O: Read {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(int(read_speed))}/s | "
                  f"Write {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(int(write_speed))}/s")
        for disk in (stats.get('disk_usage', []) or [])[:3]:
            print(f"  {disk.get('mountpoint', '?')}: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(disk.get('percent', 0) or 0)}")
        
        print("\n--- Network ---")
        current_net_io = psutil.net_io_counters()
        if last_net_io and current_net_io:
            up_speed = (current_net_io.bytes_sent - last_net_io.bytes_sent) / elapsed
            down_speed = (current_net_io.bytes_recv - last_net_io.bytes_recv) / elapsed
            print(f"Upload: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(int(up_speed))}/s | "
                  f"Download: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(int(down_speed))}/s")
        
        if alerts:
            print("\n--- Alerts ---")
            for alert in alerts[:5]:
                print(f"  ⚠ {alert}")
        
        return current_net_io, current_disk_io
    
    def format_header(self) -> str:
        """Format the monitor header."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
+{'='*100}+
|{'PiscesL1 System Monitor':^100}|
|{current_time:^100}|
+{'='*100}+
"""
    
    def format_cpu_section(self, stats: Dict[str, Any]) -> str:
        """Format CPU statistics section."""
        if not stats or 'cpu' not in stats:
            return ""
        
        cpu_info = stats['cpu']
        cpu_percent_total = stats.get('cpu_percent_total', 0) or 0
        cpu_freq = stats.get('cpu_freq', [])
        
        lines = ["| CPU Status:", "-" * 100]
        lines.append(f"|   Total Usage: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(cpu_percent_total)}")
        
        if 'cpu_percent_per_core' in stats:
            per_core = stats['cpu_percent_per_core'] or []
            if per_core:
                core_info = "|   Per Core: "
                for i, usage in enumerate(per_core[:8]):
                    core_info += f"C{i}:{(usage or 0):.0f}% "
                lines.append(core_info)
        
        if cpu_freq and len(cpu_freq) > 0:
            freqs = [freq.current for freq in cpu_freq if hasattr(freq, 'current')]
            max_freqs = [freq.max for freq in cpu_freq if hasattr(freq, 'max')]
            if freqs and max_freqs:
                avg_freq = sum(freqs) / len(freqs)
                max_freq = max(max_freqs)
                lines.append(f"|   Frequency: {avg_freq/1000:.2f} GHz (max: {max_freq/1000:.2f} GHz)")
        
        lines.append("-" * 100)
        return "\n".join(lines)
    
    def format_memory_section(self, stats: Dict[str, Any]) -> str:
        """Format memory statistics section."""
        if not stats or 'memory' not in stats:
            return ""
        
        memory = stats['memory'] or {}
        lines = ["| Memory Status:", "-" * 100]
        lines.append(f"|   Usage: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(memory.get('percent', 0) or 0)}")
        lines.append(f"|   Total: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(memory.get('total', 0))} | "
                    f"Used: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(memory.get('used', 0))} | "
                    f"Free: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(memory.get('free', 0))}")
        
        if 'swap' in stats and stats['swap']:
            swap = stats['swap'] or {}
            if swap.get('total', 0) > 0:
                lines.append(f"|   Swap: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(swap.get('percent', 0) or 0)} | "
                           f"Total: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(swap.get('total', 0))} | "
                           f"Used: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(swap.get('used', 0))}")
        
        lines.append("-" * 100)
        return "\n".join(lines)
    
    def format_gpu_section(self, stats: Dict[str, Any]) -> str:
        """Format GPU statistics section."""
        if not stats or 'gpu' not in stats or not stats['gpu']:
            return ""
        
        gpu_list = stats['gpu'] or []
        lines = ["| GPU Status:", "-" * 100]
        
        for i, gpu in enumerate(gpu_list):
            name = gpu.get('name', 'Unknown') or 'Unknown'
            util = gpu.get('util', 0) or 0
            mem_percent = gpu.get('mem_percent', 0) or 0
            mem_used = gpu.get('mem_used', 0) or 0
            mem_total = gpu.get('mem_total', 0) or 0
            
            lines.append(f"|   GPU {i}: {name[:30]}")
            lines.append(f"|     Util: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(util)} | "
                        f"Mem: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(mem_percent)}")
            lines.append(f"|     Memory: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(mem_used)} / "
                        f"{PiscesLxToolsMonitorDisplayUtils.bytes_to_human(mem_total)}")
        
        lines.append("-" * 100)
        return "\n".join(lines)
    
    def format_disk_section(self, stats: Dict[str, Any]) -> str:
        """Format disk statistics section."""
        if not stats or 'disk' not in stats:
            return ""
        
        disk = stats.get('disk', {}) or {}
        lines = ["| Disk Status:", "-" * 100]
        lines.append(f"|   Usage: {PiscesLxToolsMonitorDisplayUtils.get_progress_bar(disk.get('percent', 0) or 0)}")
        lines.append(f"|   Total: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk.get('total', 0))} | "
                    f"Used: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk.get('used', 0))} | "
                    f"Free: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk.get('free', 0))}")
        
        if 'disk_io' in stats and stats['disk_io']:
            disk_io = stats['disk_io'] or {}
            lines.append(f"|   I/O: Read: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk_io.get('read_bytes', 0))} | "
                        f"Write: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(disk_io.get('write_bytes', 0))}")
        
        lines.append("-" * 100)
        return "\n".join(lines)
    
    def format_network_section(self, stats: Dict[str, Any]) -> str:
        """Format network statistics section."""
        if not stats or 'network' not in stats:
            return ""
        
        network = stats['network'] or {}
        lines = ["| Network Status:", "-" * 100]
        lines.append(f"|   Sent: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(network.get('bytes_sent', 0))} | "
                    f"Received: {PiscesLxToolsMonitorDisplayUtils.bytes_to_human(network.get('bytes_recv', 0))}")
        lines.append(f"|   Packets: Sent: {network.get('packets_sent', 0):,} | "
                    f"Received: {network.get('packets_recv', 0):,}")
        
        if network.get('errin', 0) > 0 or network.get('errout', 0) > 0:
            lines.append(f"|   Errors: In: {network.get('errin', 0):,} | Out: {network.get('errout', 0):,}")
        
        lines.append("-" * 100)
        return "\n".join(lines)
    
    def format_footer(self) -> str:
        """Format the monitor footer."""
        return f"+{'='*100}+\n"
    
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
            self.format_footer()
        ]
        
        return "\n".join(sections)
