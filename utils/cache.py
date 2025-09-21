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
import json
import yaml
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

class PiscesCache:
    """Pure cache system - only provides infrastructure, no business logic.
    
    This is a cache system that:
    - Only manages cache directories and files
    - Does NOT hardcode any business logic
    - Lets callers decide what to cache, where to cache, and how to cache
    - Provides flexible storage backends (JSON, YAML, pickle, raw)
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize cache system.
        
        Args:
            project_root: Project root directory. If None, uses current directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.base_cache_dir = self.project_root / ".pisceslx"
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track created directories for cleanup
        self._created_dirs: set = set()
    
    def get_cache_dir(self, *path_parts: str) -> Path:
        """Get or create a cache directory.
        
        Args:
            *path_parts: Path components relative to base cache dir
            
        Returns:
            Path to the cache directory
            
        Example:
            cache_dir = cache.get_cache_dir("logs", "2024")
            # Returns: /.pisceslx/cache/logs/2024/
        """
        cache_path = self.base_cache_dir.joinpath(*path_parts)
        cache_path.mkdir(parents=True, exist_ok=True)
        self._created_dirs.add(str(cache_path))
        return cache_path
    
    def get_cache_file(self, filename: str, *dir_parts: str) -> Path:
        """Get path for a cache file.
        
        Args:
            filename: Name of the cache file
            *dir_parts: Optional directory path components
            
        Returns:
            Path to the cache file (directory is created if needed)
            
        Example:
            cache_file = cache.get_cache_file("app.log", "logs", "2024")
            # Returns: /.pisceslx/cache/logs/2024/app.log
        """
        if dir_parts:
            cache_dir = self.get_cache_dir(*dir_parts)
            return cache_dir / filename
        else:
            return self.base_cache_dir / filename
    
    def save_json(self, data: Any, filename: str, *dir_parts: str) -> bool:
        """Save data as JSON cache file.
        
        Args:
            data: Data to cache (must be JSON serializable)
            filename: Cache file name
            *dir_parts: Optional directory path
            
        Returns:
            True if successful
            
        Example:
            cache.save_json({"status": "ok"}, "status.json", "app", "cache")
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False
    
    def load_json(self, filename: str, *dir_parts: str) -> Optional[Any]:
        """Load JSON cache file.
        
        Args:
            filename: Cache file name
            *dir_parts: Optional directory path
            
        Returns:
            Cached data or None if not found/invalid
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            if not cache_file.exists():
                return None
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def save_yaml(self, data: Any, filename: str, *dir_parts: str) -> bool:
        """Save data as YAML cache file.
        
        Args:
            data: Data to cache
            filename: Cache file name
            *dir_parts: Optional directory path
            
        Returns:
            True if successful
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            with open(cache_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
            return True
        except Exception:
            return False
    
    def load_yaml(self, filename: str, *dir_parts: str) -> Optional[Any]:
        """Load YAML cache file.
        
        Args:
            filename: Cache file name
            *dir_parts: Optional directory path
            
        Returns:
            Cached data or None if not found/invalid
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            if not cache_file.exists():
                return None
            with open(cache_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            return None
    
    def save_pickle(self, data: Any, filename: str, *dir_parts: str) -> bool:
        """Save data as pickle cache file.
        
        Args:
            data: Data to cache (can be any Python object)
            filename: Cache file name
            *dir_parts: Optional directory path
            
        Returns:
            True if successful
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception:
            return False
    
    def load_pickle(self, filename: str, *dir_parts: str) -> Optional[Any]:
        """Load pickle cache file.
        
        Args:
            filename: Cache file name
            *dir_parts: Optional directory path
            
        Returns:
            Cached data or None if not found/invalid
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            if not cache_file.exists():
                return None
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def save_text(self, text: str, filename: str, *dir_parts: str, encoding: str = 'utf-8') -> bool:
        """Save text as cache file.
        
        Args:
            text: Text content to cache
            filename: Cache file name
            *dir_parts: Optional directory path
            encoding: Text encoding
            
        Returns:
            True if successful
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            with open(cache_file, 'w', encoding=encoding) as f:
                f.write(text)
            return True
        except Exception:
            return False
    
    def load_text(self, filename: str, *dir_parts: str, encoding: str = 'utf-8') -> Optional[str]:
        """Load text cache file.
        
        Args:
            filename: Cache file name
            *dir_parts: Optional directory path
            encoding: Text encoding
            
        Returns:
            Cached text or None if not found
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            if not cache_file.exists():
                return None
            with open(cache_file, 'r', encoding=encoding) as f:
                return f.read()
        except Exception:
            return None
    
    def save_binary(self, data: bytes, filename: str, *dir_parts: str) -> bool:
        """Save binary data as cache file.
        
        Args:
            data: Binary data to cache
            filename: Cache file name
            *dir_parts: Optional directory path
            
        Returns:
            True if successful
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            with open(cache_file, 'wb') as f:
                f.write(data)
            return True
        except Exception:
            return False
    
    def load_binary(self, filename: str, *dir_parts: str) -> Optional[bytes]:
        """Load binary cache file.
        
        Args:
            filename: Cache file name
            *dir_parts: Optional directory path
            
        Returns:
            Cached binary data or None if not found
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            if not cache_file.exists():
                return None
            with open(cache_file, 'rb') as f:
                return f.read()
        except Exception:
            return None
    
    def exists(self, filename: str, *dir_parts: str) -> bool:
        """Check if cache file exists.
        
        Args:
            filename: Cache file name
            *dir_parts: Optional directory path
            
        Returns:
            True if cache file exists
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            return cache_file.exists()
        except Exception:
            return False
    
    def delete(self, filename: str, *dir_parts: str) -> bool:
        """Delete cache file.
        
        Args:
            filename: Cache file name
            *dir_parts: Optional directory path
            
        Returns:
            True if deleted or not exists, False on error
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            if cache_file.exists():
                cache_file.unlink()
            return True
        except Exception:
            return False
    
    def get_file_age(self, filename: str, *dir_parts: str) -> Optional[float]:
        """Get age of cache file in seconds.
        
        Args:
            filename: Cache file name
            *dir_parts: Optional directory path
            
        Returns:
            File age in seconds, or None if not exists
        """
        try:
            cache_file = self.get_cache_file(filename, *dir_parts)
            if not cache_file.exists():
                return None
            return time.time() - cache_file.stat().st_mtime
        except Exception:
            return None
    
    def cleanup_old_files(self, max_age_days: int = 30, *dir_parts: str) -> int:
        """Clean up old cache files in directory.
        
        Args:
            max_age_days: Maximum age in days
            *dir_parts: Directory path to clean
            
        Returns:
            Number of files cleaned up
        """
        try:
            if dir_parts:
                cache_dir = self.get_cache_dir(*dir_parts)
            else:
                cache_dir = self.base_cache_dir
                
            cleaned = 0
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 3600
            
            for cache_file in cache_dir.glob("*"):
                if cache_file.is_file():
                    file_age = current_time - cache_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        cache_file.unlink()
                        cleaned += 1
                        
            return cleaned
        except Exception:
            return 0
    
    def list_cache_files(self, *dir_parts: str, pattern: str = "*") -> List[str]:
        """List cache files in directory.
        
        Args:
            *dir_parts: Directory path
            pattern: File pattern to match
            
        Returns:
            List of relative file paths
        """
        try:
            if dir_parts:
                cache_dir = self.get_cache_dir(*dir_parts)
            else:
                cache_dir = self.base_cache_dir
                
            files = []
            for cache_file in cache_dir.glob(pattern):
                if cache_file.is_file():
                    # Return relative path from base cache dir
                    try:
                        relative_path = cache_file.relative_to(self.base_cache_dir)
                        files.append(str(relative_path))
                    except ValueError:
                        files.append(cache_file.name)
            return files
        except Exception:
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache system statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            total_files = 0
            total_size = 0
            
            for root, dirs, files in os.walk(self.base_cache_dir):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.is_file():
                        total_files += 1
                        total_size += file_path.stat().st_size
            
            return {
                "base_dir": str(self.base_cache_dir),
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "created_directories": list(self._created_dirs)
            }
        except Exception:
            return {"error": "Failed to get cache stats"}

    def save_settings(self, namespace: str, data: Dict[str, Any]) -> bool:
        """Atomically save settings to a YAML file (compatible with SettingsStore).

        Args:
            namespace: Settings namespace, e.g., "dataset"
            data: Any dictionary serializable to YAML

        Returns:
            True if successful, False otherwise
        """
        return self.save_yaml(data, f"{namespace}_settings.yaml", "settings")

    def load_settings(self, namespace: str) -> Dict[str, Any]:
        """Load settings from a YAML file (compatible with SettingsStore).

        Args:
            namespace: Settings namespace, e.g., "dataset"

        Returns:
            Settings dictionary, or an empty dictionary if the file does not exist or loading fails
        """
        result = self.load_yaml(f"{namespace}_settings.yaml", "settings")
        return result if result is not None else {}


# Global cache instance
_cache_instance = None

def get_cache(project_root: Optional[str] = None) -> PiscesCache:
    """Get global cache instance.
    
    Args:
        project_root: Optional project root directory
        
    Returns:
        PiscesCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = PiscesCache(project_root)
    return _cache_instance


# Convenience aliases for backward compatibility
def get_cache_manager(project_root: Optional[str] = None) -> PiscesCache:
    """Alias for get_cache() - for backward compatibility."""
    return get_cache(project_root)

def get_config_manager(project_root: Optional[str] = None) -> PiscesCache:
    """Alias for get_cache() - for backward compatibility."""
    return get_cache(project_root)