#!/usr/bin/env python3

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

import sys
import time
from typing import Optional

class ProgressBar:
    """
    A unified progress bar utility class based on the implementation in monitor.py.
    """
    
    def __init__(self, total: int, desc: str = "", length: int = 30, 
                 fill_char: str = '█', empty_char: str = '-',
                 show_eta: bool = True, file=None):
        """
        Initialize the progress bar.

        Args:
            total (int): The total number of items to process.
            desc (str, optional): The description of the progress bar. Defaults to "".
            length (int, optional): The length of the progress bar. Defaults to 30.
            fill_char (str, optional): The character used to represent completed progress. Defaults to '█'.
            empty_char (str, optional): The character used to represent incomplete progress. Defaults to '-'.
            show_eta (bool, optional): Whether to show the estimated time of arrival (ETA). Defaults to True.
            file: The file object to write the progress bar to. Defaults to sys.stdout.
        """
        self.total = total
        self.desc = desc
        self.length = length
        self.fill_char = fill_char
        self.empty_char = empty_char
        self.show_eta = show_eta
        self.file = file or sys.stdout
        self.current = 0
        self.start_time = time.time()
        self.last_print_time = 0
        
    def _format_time(self, seconds: float) -> str:
        """
        Format the time in seconds into a human-readable string.

        Args:
            seconds (float): The time in seconds.

        Returns:
            str: The formatted time string.
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _get_bar(self, percent: float) -> str:
        """
        Generate the progress bar string.

        Args:
            percent (float): The progress percentage.

        Returns:
            str: The progress bar string.
        """
        filled_length = int(self.length * percent / 100)
        bar = self.fill_char * filled_length + self.empty_char * (self.length - filled_length)
        return f'|{bar}|'
    
    def update(self, n: int = 1):
        """
        Update the progress bar by a specified number of items.

        Args:
            n (int, optional): The number of items to update. Defaults to 1.
        """
        self.current += n
        self._display()
    
    def set_description(self, desc: str):
        """
        Set the description of the progress bar.

        Args:
            desc (str): The new description.
        """
        self.desc = desc
    
    def _display(self):
        """
        Display or update the progress bar on the screen.
        Skip if total is 0 to avoid division by zero.
        Prevent overly frequent updates.
        """
        if self.total == 0:
            return
            
        percent = min(100.0, (self.current / self.total) * 100)
        bar = self._get_bar(percent)
        
        elapsed = time.time() - self.start_time
        
        if self.show_eta and self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f" ETA: {self._format_time(eta)}"
        else:
            eta_str = ""
        
        desc_str = f"{self.desc}: " if self.desc else ""
        progress_str = f"{self.current}/{self.total}"
        
        line = f"\r{desc_str}{bar} {percent:5.1f}% {progress_str}{eta_str}"
        
        # Prevent overly frequent updates
        if time.time() - self.last_print_time > 0.1:
            print(line, end="", flush=True, file=self.file)
            self.last_print_time = time.time()
    
    def close(self):
        """
        Complete the progress bar display.
        If the progress has reached 100%, print the final state with elapsed time.
        Otherwise, print a newline to end the progress bar.
        """
        if self.current >= self.total:
            percent = 100.0
            bar = self._get_bar(percent)
            elapsed = time.time() - self.start_time
            
            desc_str = f"{self.desc}: " if self.desc else ""
            progress_str = f"{self.total}/{self.total}"
            
            print(f"\r{desc_str}{bar} 100.0% {progress_str} [{self._format_time(elapsed)}]", 
                  file=self.file)
        else:
            print(file=self.file)
    
    def __enter__(self):
        """
        Support the context manager protocol.
        Returns the instance itself when entering the context.

        Returns:
            ProgressBar: The current instance.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Support the context manager protocol.
        Close the progress bar when exiting the context.

        Args:
            exc_type: The exception type.
            exc_val: The exception value.
            exc_tb: The exception traceback.
        """
        self.close()


def progress_bar(iterable, desc: str = "", total: Optional[int] = None, 
                length: int = 30, file=None):
    """
    Wrap an iterable with a progress bar.

    If the length of the iterable can be determined, use a full progress bar.
    Otherwise, use a simple counter to show the processing progress.

    Args:
        iterable: The iterable object to wrap.
        desc (str, optional): The description of the progress. Defaults to "".
        total (Optional[int], optional): The total number of items. If None, try to get it from the iterable. Defaults to None.
        length (int, optional): The length of the progress bar. Defaults to 30.
        file: The file object to write the progress information to. Defaults to None.

    Yields:
        The items from the iterable.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    
    if total is None:
        # For iterables without a length, use a simple counter
        count = 0
        for item in iterable:
            yield item
            count += 1
            if count % 100 == 0:
                print(f"\r{desc}: {count} items processed", end="", flush=True, file=file)
        print(f"\r{desc}: {count} items processed")
    else:
        # Use a full progress bar
        pbar = ProgressBar(total, desc, length, file=file)
        for item in iterable:
            yield item
            pbar.update(1)
        pbar.close()


def get_simple_progress_bar(percent: float, length: int = 30) -> str:
    """
    Generate a simple progress bar string for compatibility with monitor.py.

    Args:
        percent (float): The progress percentage.
        length (int, optional): The length of the progress bar. Defaults to 30.

    Returns:
        str: The simple progress bar string.
    """
    try:
        percent = float(percent)
        if not 0 <= percent <= 100:
            percent = 0
    except (ValueError, TypeError):
        percent = 0
        
    filled_length = int(length * percent / 100)
    bar = '█' * filled_length + '-' * (length - filled_length)
    return f'|{bar}| {percent:5.1f}%'


# Function for backward compatibility with monitor.py
get_progress_bar = get_simple_progress_bar