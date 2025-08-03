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

import sys
import time
from typing import Optional

class ProgressBar:
    """统一进度条工具类，基于monitor.py的实现"""
    
    def __init__(self, total: int, desc: str = "", length: int = 30, 
                 fill_char: str = '█', empty_char: str = '-',
                 show_eta: bool = True, file=None):
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
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _get_bar(self, percent: float) -> str:
        """获取进度条字符串"""
        filled_length = int(self.length * percent / 100)
        bar = self.fill_char * filled_length + self.empty_char * (self.length - filled_length)
        return f'|{bar}|'
    
    def update(self, n: int = 1):
        """更新进度"""
        self.current += n
        self._display()
    
    def set_description(self, desc: str):
        """设置描述"""
        self.desc = desc
    
    def _display(self):
        """显示进度条"""
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
        
        # 防止过于频繁的更新
        if time.time() - self.last_print_time > 0.1:
            print(line, end="", flush=True, file=self.file)
            self.last_print_time = time.time()
    
    def close(self):
        """完成进度条"""
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
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def progress_bar(iterable, desc: str = "", total: Optional[int] = None, 
                length: int = 30, file=None):
    """迭代器包装器，自动显示进度条"""
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    
    if total is None:
        # 对于没有长度的迭代器，使用简单的计数器
        count = 0
        for item in iterable:
            yield item
            count += 1
            if count % 100 == 0:
                print(f"\r{desc}: {count} items processed", end="", flush=True, file=file)
        print(f"\r{desc}: {count} items processed")
    else:
        # 使用完整的进度条
        pbar = ProgressBar(total, desc, length, file=file)
        for item in iterable:
            yield item
            pbar.update(1)
        pbar.close()


def get_simple_progress_bar(percent: float, length: int = 30) -> str:
    """获取简单的进度条字符串，兼容monitor.py"""
    try:
        percent = float(percent)
        if not 0 <= percent <= 100:
            percent = 0
    except (ValueError, TypeError):
        percent = 0
        
    filled_length = int(length * percent / 100)
    bar = '█' * filled_length + '-' * (length - filled_length)
    return f'|{bar}| {percent:5.1f}%'


# 向后兼容monitor.py的函数
get_progress_bar = get_simple_progress_bar