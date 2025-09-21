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
import time
import psutil
import platform
from utils import RIGHT, ERROR, get_progress_bar

UPDATE_INTERVAL = 1

def bytes_to_human(n):
    """
    Converts bytes to a human-readable format (KB, MB, GB, etc.).

    Args:
        n (int): The number of bytes to convert.

    Returns:
        str: A string representing the bytes in a human-readable format.
    """
    # Symbols for different byte units
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    # Dictionary to store prefix values for each symbol
    prefix = {}
    for i, s in enumerate(symbols):
        # Calculate the prefix value for each symbol
        prefix[s] = 1 << (i + 1) * 10
    # Iterate through symbols in reverse order
    for s in reversed(symbols):
        if n >= prefix[s]:
            # Convert bytes to the corresponding unit
            value = float(n) / prefix[s]
            return f'{value:.2f}{s}B'
    return f"{n}B"

# Flag indicating whether GPU monitoring is enabled
GPU_ENABLED = False
# Number of detected GPUs
gpu_count = 0
try:
    # Import the pynvml library for NVIDIA GPU monitoring
    import pynvml
    # Initialize the pynvml library
    pynvml.nvmlInit()
    # Get the count of available NVIDIA GPUs
    gpu_count = pynvml.nvmlDeviceGetCount()
    if gpu_count > 0:
        GPU_ENABLED = True
except Exception:
    # Disable GPU monitoring if an error occurs
    GPU_ENABLED = False

def get_system_stats():
    """
    Gathers all system statistics, including CPU, memory, GPU, disk, etc.

    Returns:
        dict: A dictionary containing various system statistics.
    """
    # Initialize an empty dictionary to store system statistics
    stats = {}

    # Get CPU usage statistics
    stats['cpu_percent_total'] = psutil.cpu_percent(percpu=False)
    stats['cpu_percent_per_core'] = psutil.cpu_percent(percpu=True)
    try:
        # Get CPU frequency statistics for each core
        stats['cpu_freq'] = psutil.cpu_freq(percpu=True)
    except Exception:
        # Set an empty list if CPU frequency information is unavailable
        stats['cpu_freq'] = []

    # Get virtual memory statistics
    mem = psutil.virtual_memory()
    stats['mem_total'] = mem.total
    stats['mem_used'] = mem.used
    stats['mem_percent'] = mem.percent
    
    # Get swap memory statistics
    swap = psutil.swap_memory()
    stats['swap_total'] = swap.total
    stats['swap_used'] = swap.used
    stats['swap_percent'] = swap.percent

    # Get GPU statistics if GPU monitoring is enabled
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

    # Get disk usage statistics
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
            # Skip partitions that cannot be accessed
            continue
    stats['disk_usage'] = disk_usage_stats

    return stats

def display_stats(stats, last_net_io, last_disk_io, last_time):
    """
    Clears the screen and displays the formatted system statistics.

    Args:
        stats (dict): A dictionary containing system statistics.
        last_net_io (psutil._common.snetio): Last network I/O counters.
        last_disk_io (psutil._common.sdiskio): Last disk I/O counters.
        last_time (float): Last monitoring time.

    Returns:
        tuple: Current network I/O counters and disk I/O counters.
    """
    # Clear the terminal screen
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"--- Pisces L1 System Monitor --- (Press Ctrl+C to exit)")
    print(f"System: {platform.system()} {platform.release()} | Update Interval: {UPDATE_INTERVAL}s")
    print("-" * 60)

    # Display CPU information
    print("--- CPU ---")
    print(f"Total Usage: {get_progress_bar(stats['cpu_percent_total'])}")
    for i, (percent, freq) in enumerate(zip(stats['cpu_percent_per_core'], stats.get('cpu_freq', []))):
        freq_str = f"{freq.current:.0f}MHz" if freq else "N/A"
        print(f"  Core {i:<2}: {get_progress_bar(percent, length=20)} @ {freq_str}")
    
    # Display memory information
    print("\n--- Memory ---")
    mem_total_h = bytes_to_human(stats['mem_total'])
    mem_used_h = bytes_to_human(stats['mem_used'])
    print(f"RAM  : {get_progress_bar(stats['mem_percent'])} ({mem_used_h} / {mem_total_h})")
    if stats['swap_total'] > 0:
        swap_total_h = bytes_to_human(stats['swap_total'])
        swap_used_h = bytes_to_human(stats['swap_used'])
        print(f"Swap : {get_progress_bar(stats['swap_percent'])} ({swap_used_h} / {swap_total_h})")

    # Display GPU information if available
    if GPU_ENABLED and 'gpu' in stats:
        print("\n--- GPU ---")
        for i, gpu in enumerate(stats['gpu']):
            gpu_mem_total_h = bytes_to_human(gpu['mem_total'])
            gpu_mem_used_h = bytes_to_human(gpu['mem_used'])
            print(f"GPU {i} ({gpu['name']}):")
            print(f"  Usage : {get_progress_bar(gpu['util'])}")
            print(f"  Memory: {get_progress_bar(gpu['mem_percent'])} ({gpu_mem_used_h} / {gpu_mem_total_h})")

    # Calculate the elapsed time for speed calculation
    current_time = time.time()
    elapsed_time = current_time - last_time
    if elapsed_time == 0:
        # Avoid division by zero on the first run
        elapsed_time = 1

    # Display disk I/O information
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

    # Display network I/O information
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
            # Collect system statistics
            stats = get_system_stats()
            # Display system statistics and update I/O counters
            last_net_io, last_disk_io = display_stats(stats, last_net_io, last_disk_io, last_time)
            # Update the last monitoring time
            last_time = time.time()
            # Wait for the specified update interval
            time.sleep(UPDATE_INTERVAL)
    except KeyboardInterrupt:
        RIGHT("\n\nMonitor stopped. Goodbye!")
    except Exception as e:
        ERROR(f"\nAn error occurred: {e}")
    finally:
        if GPU_ENABLED:
            # Shutdown the pynvml library to release resources
            pynvml.nvmlShutdown()