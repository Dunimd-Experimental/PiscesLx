#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import os
from pathlib import Path
from typing import Any, Optional
from utils.log.core import PiscesLxCoreLog
from utils.error import PiscesLxCoreError, PiscesLxCoreIOError, PiscesLxCoreFilesystemError

logger = PiscesLxCoreLog("PiscesLx.Core.Fs.Core")

class PiscesLxCoreFS:
    """Object-oriented filesystem helper with safe operations.

    - Handles project-root aware paths
    - Provides safe directory creation
    - Supports atomic write operations for text and bytes
    - Offers JSON read and write helpers
    """

    def __init__(self, project_root: Optional[Path] = None) -> None:
        """Initialize the filesystem helper.

        Args:
            project_root (Optional[Path]): The project root directory. 
                If None, it will be automatically detected.
        """
        self._project_root = project_root or self._detect_project_root()

    def project_root(self) -> Path:
        """Get the project root directory.

        Returns:
            Path: The project root path.
        """
        return self._project_root

    def _detect_project_root(self) -> Path:
        """Detect and return the project root directory.

        Returns:
            Path: The project root path obtained from the config loader.
        """
        from utils.config.loader import PiscesLxCoreConfigLoader
        loader = PiscesLxCoreConfigLoader()
        return loader._project_root

    def safe_mkdir(self, path: os.PathLike | str, mode: int = 0o755, exist_ok: bool = True) -> Path:
        """Create a directory safely with the specified permissions.

        Args:
            path (os.PathLike | str): The path of the directory to create.
            mode (int, optional): The permissions to set for the directory. Defaults to 0o755.
            exist_ok (bool, optional): If True, ignore "directory already exists" errors. Defaults to True.

        Returns:
            Path: The created directory path.

        Raises:
            PiscesLxCoreFilesystemError: If the directory creation fails.
        """
        p = Path(path)
        try:
            p.mkdir(parents=True, exist_ok=exist_ok)
            os.chmod(p, mode)
            return p
        except Exception as e:
            raise PiscesLxCoreFilesystemError("safe_mkdir failed", context={"path": str(p)}, cause=e)

    def ensure_parent_dir(self, path: os.PathLike | str, mode: int = 0o755) -> Path:
        """Ensure the parent directory of the given path exists.

        Args:
            path (os.PathLike | str): The path for which to ensure the parent directory exists.
            mode (int, optional): The permissions to set for the parent directory if created. Defaults to 0o755.

        Returns:
            Path: The parent directory path.
        """
        parent = Path(path).parent
        return self.safe_mkdir(parent, mode=mode, exist_ok=True)

    def atomic_write_text(self, path: os.PathLike | str, text: str, encoding: str = "utf-8") -> None:
        """Write text to a file atomically.

        Args:
            path (os.PathLike | str): The path of the file to write.
            text (str): The text to write.
            encoding (str, optional): The encoding to use for writing. Defaults to "utf-8".

        Raises:
            PiscesLxCoreIOError: If the write operation fails.
        """
        p = Path(path)
        dirp = self.ensure_parent_dir(p)
        tmp = None
        try:
            import tempfile
            fd, tmp = tempfile.mkstemp(prefix=".tmp", dir=str(dirp))
            with os.fdopen(fd, "w", encoding=encoding) as f:
                f.write(text)
            os.replace(tmp, p)
        except Exception as e:
            try:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
            except Exception as cleanup_e:
                logger.debug("TEMP_FILE_CLEANUP_FAILED", {"tmp_file": tmp, "error": str(cleanup_e)})
            raise PiscesLxCoreIOError("atomic_write_text failed", context={"path": str(p)}, cause=e)

    def atomic_write_bytes(self, path: os.PathLike | str, data: bytes) -> None:
        """Write bytes to a file atomically.

        Args:
            path (os.PathLike | str): The path of the file to write.
            data (bytes): The bytes to write.

        Raises:
            PiscesLxCoreIOError: If the write operation fails.
        """
        p = Path(path)
        dirp = self.ensure_parent_dir(p)
        tmp = None
        try:
            import tempfile
            fd, tmp = tempfile.mkstemp(prefix=".tmp", dir=str(dirp))
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
        except Exception as e:
            try:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
            except Exception as cleanup_e:
                logger.debug("TEMP_FILE_CLEANUP_FAILED", {"tmp_file": tmp, "error": str(cleanup_e)})
            raise PiscesLxCoreIOError("atomic_write_bytes failed", context={"path": str(p)}, cause=e)

    def read_json(self, path: os.PathLike | str, encoding: str = "utf-8") -> Any:
        """Read a JSON file and return its contents.

        Args:
            path (os.PathLike | str): The path of the JSON file to read.
            encoding (str, optional): The encoding to use for reading. Defaults to "utf-8".

        Returns:
            Any: The parsed JSON data.

        Raises:
            PiscesLxCoreIOError: If the file is not found or the read operation fails.
        """
        from json import load
        p = Path(path)
        try:
            with open(p, "r", encoding=encoding) as f:
                return load(f)
        except FileNotFoundError as e:
            raise PiscesLxCoreIOError("json not found", context={"path": str(p)}, cause=e)
        except Exception as e:
            raise PiscesLxCoreIOError("json read failed", context={"path": str(p)}, cause=e)

    def write_json(self, path: os.PathLike | str, obj: Any, encoding: str = "utf-8") -> None:
        """Write an object to a JSON file atomically.

        Args:
            path (os.PathLike | str): The path of the JSON file to write.
            obj (Any): The object to serialize to JSON.
            encoding (str, optional): The encoding to use for writing. Defaults to "utf-8".

        Raises:
            PiscesLxCoreIOError: If the write operation fails.
        """
        from json import dumps
        try:
            text = dumps(obj, ensure_ascii=False)
            self.atomic_write_text(path, text, encoding=encoding)
        except PiscesLxCoreError:
            raise
        except Exception as e:
            raise PiscesLxCoreIOError("json write failed", context={"path": str(path)}, cause=e)

    def app_dir(self) -> Path:
        """Get the application directory under the project root.

        Returns:
            Path: The path to the ".pisceslx" directory.
        """
        p = self.project_root() / ".pisceslx"
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    # Backward-compatible alias
    def get_pisceslx_root(self) -> Path:
        return self.app_dir()

    def logs_dir(self) -> Path:
        """Get the logs directory under the application directory.

        Returns:
            Path: The path to the logs directory.
        """
        p = self.app_dir() / "logs"
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    def cache_dir(self) -> Path:
        """Get the cache directory under the application directory.

        Returns:
            Path: The path to the cache directory.
        """
        p = self.app_dir() / "cache"
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    def reports_dir(self) -> Path:
        """Get the reports directory under the application directory.

        Returns:
            Path: The path to the reports directory.
        """
        p = self.app_dir() / "reports"
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    def observability_dir(self) -> Path:
        """Get the observability directory under the application directory."""
        p = self.app_dir() / "observability"
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    def temp_dir(self) -> Path:
        """Get the temp directory under the application directory."""
        p = self.app_dir() / "tmp"
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    def ensure_pisceslx_subdir(self, category: str, *parts: str) -> Path:
        """Ensure a subpath under .pisceslx/<category> exists and return it.

        Args:
            category: One of 'logs', 'cache', 'reports', 'observability', 'tmp'.
            parts: Additional path parts (e.g., filename).
        """
        base_map = {
            "logs": self.logs_dir,
            "cache": self.cache_dir,
            "reports": self.reports_dir,
            "observability": self.observability_dir,
            "tmp": self.temp_dir,
        }
        base_fn = base_map.get(category)
        if base_fn is None:
            base_fn = self.app_dir
        base = base_fn()
        p = base.joinpath(*parts) if parts else base
        try:
            # Ensure parent exists (for files)
            if p.suffix:
                self.ensure_parent_dir(p)
            else:
                p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    def normalize_under_category(self, category: str, path_or_name: str | os.PathLike) -> Path:
        """Force any provided path to be under .pisceslx/<category> with its basename.

        Examples:
            normalize_under_category('logs', 'logs/abc.log') -> .pisceslx/logs/abc.log
            normalize_under_category('cache', '/var/tmp/foo') -> .pisceslx/cache/foo
        """
        name = Path(path_or_name).name
        return self.ensure_pisceslx_subdir(category, name)
