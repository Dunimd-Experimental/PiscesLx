#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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
import time
import psutil
import platform
from utils.log import RIGHT, ERROR
from utils.progress import get_progress_bar

# Define the update interval for system monitoring (in seconds)
UPDATE_INTERVAL = 1

# --- Helper Functions ---
def bytes_to_human(n):
    """
    Converts bytes to a human-readable format (KB, MB, GB, etc.).

    Args:
        n (int): The number of bytes to convert.

    Returns:
        str: A string representing the bytes in a human-readable format.
    """
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
# Flag indicating whether GPU monitoring is enabled
GPU_ENABLED = False
# Number of detected GPUs
gpu_count = 0
try:
    # Try to import the pynvml library
    import pynvml
    # Initialize the pynvml library
    pynvml.nvmlInit()
    # Get the number of GPUs
    gpu_count = pynvml.nvmlDeviceGetCount()
    if gpu_count > 0:
        GPU_ENABLED = True
except Exception:
    # If an error occurs, disable GPU monitoring
    GPU_ENABLED = False

# --- Main Monitoring Logic ---
def get_system_stats():
    """
    Gathers all system statistics.

    Returns:
        dict: A dictionary containing various system statistics.
    """
    stats = {}

    # CPU Statistics
    # Get the total CPU usage percentage
    stats['cpu_percent_total'] = psutil.cpu_percent(percpu=False)
    # Get the CPU usage percentage for each core
    stats['cpu_percent_per_core'] = psutil.cpu_percent(percpu=True)
    try:
        # psutil.cpu_freq() might not be available on all systems or require root
        # Get the CPU frequency for each core
        stats['cpu_freq'] = psutil.cpu_freq(percpu=True)
    except Exception:
        # If an error occurs, set an empty list
        stats['cpu_freq'] = []

    # Memory Statistics
    # Get virtual memory information
    mem = psutil.virtual_memory()
    stats['mem_total'] = mem.total
    stats['mem_used'] = mem.used
    stats['mem_percent'] = mem.percent
    
    # Swap Memory Statistics
    # Get swap memory information
    swap = psutil.swap_memory()
    stats['swap_total'] = swap.total
    stats['swap_used'] = swap.used
    stats['swap_percent'] = swap.percent

    # GPU Statistics
    if GPU_ENABLED:
        gpu_stats = []
        for i in range(gpu_count):
            # Get the handle of the i-th GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # Get the GPU utilization rate
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            # Get the GPU memory information
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_stats.append({
                'name': pynvml.nvmlDeviceGetName(handle).decode('utf-8') if isinstance(pynvml.nvmlDeviceGetName(handle), bytes) else pynvml.nvmlDeviceGetName(handle),
                'util': util.gpu,
                'mem_total': mem_info.total,
                'mem_used': mem_info.used,
                'mem_percent': (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0
            })
        stats['gpu'] = gpu_stats

    # Disk Statistics
    # Get all disk partitions
    disk_partitions = psutil.disk_partitions()
    disk_usage_stats = []
    for partition in disk_partitions:
        try:
            # Get the disk usage information for the partition
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
    """
    Clears the screen and displays the formatted statistics.

    Args:
        stats (dict): A dictionary containing system statistics.
        last_net_io (psutil._common.snetio): Last network I/O counters.
        last_disk_io (psutil._common.sdiskio): Last disk I/O counters.
        last_time (float): Last monitoring time.

    Returns:
        tuple: Current network I/O counters and disk I/O counters.
    """
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
        elapsed_time = 1  # Avoid division by zero on the first run

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
    """
    Main function to run the system monitor.
    Continuously collects and displays system statistics until interrupted.
    """
    RIGHT("Starting Pisces L1 System Monitor...")
    if not GPU_ENABLED:
        ERROR("NVIDIA GPU not detected or 'pynvml' library failed to load.")
        ERROR("For GPU monitoring, ensure you have an NVIDIA GPU and run: pip install pynvml")
    time.sleep(2)
    # Initialize the last network I/O counters
    last_net_io = psutil.net_io_counters()
    # Initialize the last disk I/O counters
    last_disk_io = psutil.disk_io_counters()
    # Initialize CPU usage statistics
    psutil.cpu_percent(percpu=True)
    psutil.cpu_percent(percpu=False)
    # Initialize the last monitoring time
    last_time = time.time()
    try:
        while True:
            # Get system statistics
            stats = get_system_stats()
            # Display system statistics and update I/O counters
            last_net_io, last_disk_io = display_stats(stats, last_net_io, last_disk_io, last_time)
            # Update the last monitoring time
            last_time = time.time()
            # Sleep for the update interval
            time.sleep(UPDATE_INTERVAL)
    except KeyboardInterrupt:
        RIGHT("\n\nMonitor stopped. Goodbye!")
    except Exception as e:
        ERROR(f"\nAn error occurred: {e}")
    finally:
        if GPU_ENABLED:
            # Shutdown the pynvml library
            pynvml.nvmlShutdown()