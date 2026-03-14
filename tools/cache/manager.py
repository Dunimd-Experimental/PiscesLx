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
PiscesLx Cache Manager.

This module implements the cache management functionality for the .pisceslx
directory, providing status display and cleaning capabilities.

The cache manager protects the settings/ directory from any deletion,
ensuring user configurations are never accidentally removed.

Commands:
    python manage.py cache           # Show cache status
    python manage.py cache clean     # Clean all cache

Example:
    >>> from tools.cache.manager import PiscesLxCacheManager
    >>> manager = PiscesLxCacheManager()
    >>> manager.status()
    Cache Status:
      cache/     500 MB    120 files
      logs/      1.2 GB    45 files
      settings/  PROTECTED
    Total: 1.7 GB
    >>> manager.clean()
    Cleaning cache...
      Removed cache/    500 MB
      Removed logs/     1.2 GB
      Protected: settings/
    Done. Freed 1.7 GB
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.paths import get_cache_dir, get_log_file
from utils.dc import PiscesLxLogger


_LOG = PiscesLxLogger("PiscesLx.Tools.Cache", file_path=get_log_file("PiscesLx.Tools.Cache"), enable_file=True)


class PiscesLxCacheManager:
    """
    Cache manager for the .pisceslx directory.
    
    This class provides functionality to view cache status and clean
    cache directories while protecting the settings/ directory.
    
    Attributes:
        PROTECTED_DIRS: Set of directory names that cannot be deleted
        _cache_root: Path to the .pisceslx directory
    
    Example:
        >>> manager = PiscesLxCacheManager()
        >>> manager.status()
        >>> manager.clean()
    """
    
    PROTECTED_DIRS = frozenset({'settings'})
    
    def __init__(self):
        """
        Initialize the cache manager.
        """
        self._cache_root = self._get_cache_root()
        _LOG.debug("PiscesLxCacheManager initialized", cache_root=self._cache_root)
    
    def _get_cache_root(self) -> str:
        """
        Get the path to the .pisceslx directory.
        
        Returns:
            str: Path to .pisceslx directory
        """
        home = Path.home()
        cache_root = home / '.pisceslx'
        return str(cache_root)
    
    def _get_directories(self) -> List[str]:
        """
        Get all directories in .pisceslx.
        
        Returns:
            List[str]: List of directory names
        """
        if not os.path.exists(self._cache_root):
            return []
        
        dirs = []
        for item in os.listdir(self._cache_root):
            item_path = os.path.join(self._cache_root, item)
            if os.path.isdir(item_path):
                dirs.append(item)
        
        return sorted(dirs)
    
    def _get_directory_info(self, dir_name: str) -> Tuple[int, int]:
        """
        Get size and file count for a directory.
        
        Args:
            dir_name: Name of the directory
        
        Returns:
            Tuple[int, int]: (total_size_bytes, file_count)
        """
        dir_path = os.path.join(self._cache_root, dir_name)
        
        if not os.path.exists(dir_path):
            return (0, 0)
        
        total_size = 0
        file_count = 0
        
        try:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                    except OSError:
                        pass
        except OSError:
            pass
        
        return (total_size, file_count)
    
    def _format_size(self, size_bytes: int) -> str:
        """
        Format bytes to human readable string.
        
        Args:
            size_bytes: Size in bytes
        
        Returns:
            str: Formatted size string
        """
        if size_bytes == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        size = float(size_bytes)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.1f} {units[unit_index]}"
    
    def _is_protected(self, dir_name: str) -> bool:
        """
        Check if a directory is protected.
        
        Args:
            dir_name: Name of the directory
        
        Returns:
            bool: True if protected, False otherwise
        """
        return dir_name in self.PROTECTED_DIRS
    
    def status(self) -> None:
        """
        Display cache status.
        
        Prints a formatted table showing all directories in .pisceslx
        with their sizes and file counts. Protected directories are
        marked as PROTECTED.
        """
        print("Cache Status:")
        
        directories = self._get_directories()
        total_size = 0
        
        for dir_name in directories:
            if self._is_protected(dir_name):
                print(f"  {dir_name}/     PROTECTED")
            else:
                size, files = self._get_directory_info(dir_name)
                total_size += size
                size_str = self._format_size(size)
                print(f"  {dir_name}/     {size_str}    {files} files")
        
        print(f"Total: {self._format_size(total_size)}")
    
    def clean(self) -> None:
        """
        Clean all non-protected cache directories.
        
        Removes all files and subdirectories in non-protected directories
        while keeping the directory structure intact. Protected directories
        are never touched.
        """
        print("Cleaning cache...")
        
        directories = self._get_directories()
        total_freed = 0
        
        for dir_name in directories:
            if self._is_protected(dir_name):
                print(f"  Protected: {dir_name}/")
                continue
            
            dir_path = os.path.join(self._cache_root, dir_name)
            
            if not os.path.exists(dir_path):
                continue
            
            size, _ = self._get_directory_info(dir_name)
            
            try:
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except OSError as e:
                        _LOG.warning("Failed to remove item", path=item_path, error=str(e))
                
                total_freed += size
                print(f"  Removed {dir_name}/    {self._format_size(size)}")
            except OSError as e:
                _LOG.error("Failed to clean directory", directory=dir_name, error=str(e))
        
        print(f"Done. Freed {self._format_size(total_freed)}")
    
    def get_cache_root(self) -> str:
        """
        Get the cache root directory path.
        
        Returns:
            str: Path to .pisceslx directory
        """
        return self._cache_root
    
    def get_total_size(self) -> int:
        """
        Get total size of all non-protected directories.
        
        Returns:
            int: Total size in bytes
        """
        directories = self._get_directories()
        total_size = 0
        
        for dir_name in directories:
            if not self._is_protected(dir_name):
                size, _ = self._get_directory_info(dir_name)
                total_size += size
        
        return total_size
    
    def get_directory_list(self) -> List[Dict[str, any]]:
        """
        Get detailed information for all directories.
        
        Returns:
            List[Dict]: List of directory info dictionaries
        """
        directories = self._get_directories()
        result = []
        
        for dir_name in directories:
            info = {
                'name': dir_name,
                'protected': self._is_protected(dir_name)
            }
            
            if not info['protected']:
                size, files = self._get_directory_info(dir_name)
                info['size'] = size
                info['size_formatted'] = self._format_size(size)
                info['files'] = files
            
            result.append(info)
        
        return result
