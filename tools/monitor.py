#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import time
import psutil
import platform
from utils.log import RIGHT, ERROR
from utils.progress import get_progress_bar

UPDATE_INTERVAL = 1

# --- Helper Functions ---
def bytes_to_human(n):
    """Converts bytes to a human-readable format (KB, MB, GB, etc.)."""
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return f'{value:.2f}{s}B'
    return f"{n}B"

# --- GPU Monitoring Setup ---
GPU_ENABLED = False
gpu_count = 0
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    if gpu_count > 0:
        GPU_ENABLED = True
except Exception:
    GPU_ENABLED = False

# --- Main Monitoring Logic ---
def get_system_stats():
    """Gathers all system statistics."""
    stats = {}

    # CPU
    stats['cpu_percent_total'] = psutil.cpu_percent(percpu=False)
    stats['cpu_percent_per_core'] = psutil.cpu_percent(percpu=True)
    try:
        # psutil.cpu_freq() might not be available on all systems or require root
        stats['cpu_freq'] = psutil.cpu_freq(percpu=True)
    except Exception:
        stats['cpu_freq'] = []

    # Memory
    mem = psutil.virtual_memory()
    stats['mem_total'] = mem.total
    stats['mem_used'] = mem.used
    stats['mem_percent'] = mem.percent
    
    # Swap Memory
    swap = psutil.swap_memory()
    stats['swap_total'] = swap.total
    stats['swap_used'] = swap.used
    stats['swap_percent'] = swap.percent

    # GPU
    if GPU_ENABLED:
        gpu_stats = []
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_stats.append({
                'name': pynvml.nvmlDeviceGetName(handle).decode('utf-8') if isinstance(pynvml.nvmlDeviceGetName(handle), bytes) else pynvml.nvmlDeviceGetName(handle),
                'util': util.gpu,
                'mem_total': mem_info.total,
                'mem_used': mem_info.used,
                'mem_percent': (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0
            })
        stats['gpu'] = gpu_stats

    # Disks
    disk_partitions = psutil.disk_partitions()
    disk_usage_stats = []
    for partition in disk_partitions:
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_usage_stats.append({
                'device': partition.device,
                'mountpoint': partition.mountpoint,
                'total': usage.total,
                'used': usage.used,
                'percent': usage.percent
            })
        except (PermissionError, FileNotFoundError):
            # Can't access the drive or it's a special device
            continue
    stats['disk_usage'] = disk_usage_stats

    return stats

def display_stats(stats, last_net_io, last_disk_io, last_time):
    """Clears the screen and displays the formatted statistics."""
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"--- Pisces L1 System Monitor --- (Press Ctrl+C to exit)")
    print(f"System: {platform.system()} {platform.release()} | Update Interval: {UPDATE_INTERVAL}s")
    print("-" * 60)

    # CPU Info
    print("--- CPU ---")
    print(f"Total Usage: {get_progress_bar(stats['cpu_percent_total'])}")
    for i, (percent, freq) in enumerate(zip(stats['cpu_percent_per_core'], stats.get('cpu_freq', []))):
        freq_str = f"{freq.current:.0f}MHz" if freq else "N/A"
        print(f"  Core {i:<2}: {get_progress_bar(percent, length=20)} @ {freq_str}")
    
    # Memory Info
    print("\n--- Memory ---")
    mem_total_h = bytes_to_human(stats['mem_total'])
    mem_used_h = bytes_to_human(stats['mem_used'])
    print(f"RAM  : {get_progress_bar(stats['mem_percent'])} ({mem_used_h} / {mem_total_h})")
    if stats['swap_total'] > 0:
        swap_total_h = bytes_to_human(stats['swap_total'])
        swap_used_h = bytes_to_human(stats['swap_used'])
        print(f"Swap : {get_progress_bar(stats['swap_percent'])} ({swap_used_h} / {swap_total_h})")

    # GPU Info
    if GPU_ENABLED and 'gpu' in stats:
        print("\n--- GPU ---")
        for i, gpu in enumerate(stats['gpu']):
            gpu_mem_total_h = bytes_to_human(gpu['mem_total'])
            gpu_mem_used_h = bytes_to_human(gpu['mem_used'])
            print(f"GPU {i} ({gpu['name']}):")
            print(f"  Usage : {get_progress_bar(gpu['util'])}")
            print(f"  Memory: {get_progress_bar(gpu['mem_percent'])} ({gpu_mem_used_h} / {gpu_mem_total_h})")

    # Time delta for speed calculation
    current_time = time.time()
    elapsed_time = current_time - last_time
    if elapsed_time == 0:
        elapsed_time = 1 # Avoid division by zero on the first run

    # Disk Info
    current_disk_io = psutil.disk_io_counters()
    print("\n--- Disk ---")
    if last_disk_io:
        read_speed = (current_disk_io.read_bytes - last_disk_io.read_bytes) / elapsed_time
        write_speed = (current_disk_io.write_bytes - last_disk_io.write_bytes) / elapsed_time
        print(f"I/O  : Read: {bytes_to_human(read_speed)}/s | Write: {bytes_to_human(write_speed)}/s")
    
    for disk in stats['disk_usage']:
        disk_total_h = bytes_to_human(disk['total'])
        disk_used_h = bytes_to_human(disk['used'])
        print(f"  {disk['mountpoint']:<10} {get_progress_bar(disk['percent'])} ({disk_used_h} / {disk_total_h})")

    # Network Info
    current_net_io = psutil.net_io_counters()
    print("\n--- Network ---")
    if last_net_io:
        sent_speed = (current_net_io.bytes_sent - last_net_io.bytes_sent) / elapsed_time
        recv_speed = (current_net_io.bytes_recv - last_net_io.bytes_recv) / elapsed_time
        print(f"Upload   : {bytes_to_human(sent_speed)}/s (Total: {bytes_to_human(current_net_io.bytes_sent)})")
        print(f"Download : {bytes_to_human(recv_speed)}/s (Total: {bytes_to_human(current_net_io.bytes_recv)})")

    return current_net_io, current_disk_io

def monitor():
    RIGHT("Starting Pisces L1 System Monitor...")
    if not GPU_ENABLED:
        ERROR("NVIDIA GPU not detected or 'pynvml' library failed to load.")
        ERROR("For GPU monitoring, ensure you have an NVIDIA GPU and run: pip install pynvml")
    time.sleep(2)
    # Main function to run the monitor.
    last_net_io = psutil.net_io_counters()
    last_disk_io = psutil.disk_io_counters()
    psutil.cpu_percent(percpu=True)
    psutil.cpu_percent(percpu=False)
    last_time = time.time()
    try:
        while True:
            stats = get_system_stats()
            last_net_io, last_disk_io = display_stats(stats, last_net_io, last_disk_io, last_time)
            last_time = time.time()
            time.sleep(UPDATE_INTERVAL)
    except KeyboardInterrupt:
        RIGHT("\n\nMonitor stopped. Goodbye!")
    except Exception as e:
        ERROR(f"\nAn error occurred: {e}")
    finally:
        if GPU_ENABLED:
            pynvml.nvmlShutdown()
