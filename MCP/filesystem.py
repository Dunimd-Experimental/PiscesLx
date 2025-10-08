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
import os
import json
import shutil
import mimetypes
from MCP import mcp
from pathlib import Path
from typing import Dict, Any, List, Optional

class FilesystemTool:
    """
    A class that provides file system operations with security controls.
    """
    
    def __init__(self):
        """
        Initialize the FilesystemTool instance.
        Set the tool name, description, and allowed paths for file operations.
        """
        self.name = "filesystem"
        self.description = "File system operations with security controls and path validation"
        self.allowed_paths = self._get_allowed_paths()
        
    def _get_allowed_paths(self) -> List[str]:
        """
        Get the list of allowed paths for file operations.
        Allows current directory, user home directory, and common safe paths.

        Returns:
            List[str]: A list of allowed paths.
        """
        return [
            os.getcwd(),
            os.path.expanduser("~"),
            r"D:\piscesl1\tools\data",
            os.path.join(os.getcwd(), "workspace"),
            os.path.join(os.getcwd(), "files")
        ]
    
    def _validate_path(self, path: str) -> Path:
        """
        Validate and resolve the given path.
        Check if the resolved path is within the allowed directories.

        Args:
            path (str): The path to validate.

        Returns:
            Path: The resolved and validated path.

        Raises:
            ValueError: If the path is invalid or not within allowed directories.
        """
        try:
            resolved_path = Path(path).resolve()
            
            # Check if the resolved path is within any allowed directory
            allowed = False
            for allowed_path in self.allowed_paths:
                allowed_base = Path(allowed_path).resolve()
                try:
                    resolved_path.relative_to(allowed_base)
                    allowed = True
                    break
                except ValueError:
                    continue
            
            if not allowed:
                raise ValueError(f"Path {path} is not within allowed directories")
            
            return resolved_path
            
        except Exception as e:
            raise ValueError(f"Invalid path: {str(e)}")
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema for the file system operations.
        Defines the structure of the input parameters for each operation.

        Returns:
            Dict[str, Any]: A dictionary representing the operation schema.
        """
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "File system operation to perform",
                    "enum": [
                        "read_file", "write_file", "create_directory", "list_directory",
                        "delete_file", "delete_directory", "move", "copy", "get_info",
                        "search_files", "create_file", "append_file", "read_binary"
                    ]
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to file"
                },
                "binary_content": {
                    "type": "string",
                    "description": "Base64 encoded binary content"
                },
                "source_path": {
                    "type": "string",
                    "description": "Source path for move/copy operations"
                },
                "target_path": {
                    "type": "string",
                    "description": "Target path for move/copy operations"
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern for file search"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively",
                    "default": True
                },
                "create_parents": {
                    "type": "boolean",
                    "description": "Create parent directories if they don't exist",
                    "default": True
                }
            },
            "required": ["operation", "path"]
        }
    
    def _read_file(self, path: Path) -> Dict[str, Any]:
        """
        Read the content of a text file.

        Args:
            path (Path): The path to the file to read.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and file data.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "success": True,
                "data": {
                    "content": content,
                    "path": str(path),
                    "size": len(content),
                    "encoding": "utf-8",
                    "mime_type": mimetypes.guess_type(str(path))[0] or "text/plain"
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _read_binary(self, path: Path) -> Dict[str, Any]:
        """
        Read the content of a binary file and encode it in Base64.

        Args:
            path (Path): The path to the binary file to read.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and file data.
        """
        try:
            import base64
            with open(path, 'rb') as f:
                content = f.read()
            
            return {
                "success": True,
                "data": {
                    "content": base64.b64encode(content).decode('utf-8'),
                    "path": str(path),
                    "size": len(content),
                    "mime_type": mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _write_file(self, path: Path, content: str, create_parents: bool = True) -> Dict[str, Any]:
        """
        Write text content to a file.

        Args:
            path (Path): The path to the file to write.
            content (str): The content to write to the file.
            create_parents (bool, optional): Whether to create parent directories if they don't exist. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and file data.
        """
        try:
            if create_parents:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "data": {
                    "path": str(path),
                    "size": len(content),
                    "created": not path.exists(),
                    "mime_type": mimetypes.guess_type(str(path))[0] or "text/plain"
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_file(self, path: Path, create_parents: bool = True) -> Dict[str, Any]:
        """
        Create an empty file.

        Args:
            path (Path): The path to the file to create.
            create_parents (bool, optional): Whether to create parent directories if they don't exist. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and file data.
        """
        try:
            if create_parents:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            path.touch(exist_ok=False)
            
            return {
                "success": True,
                "data": {
                    "path": str(path),
                    "created": True,
                    "size": 0
                }
            }
        except FileExistsError:
            return {
                "success": False,
                "error": f"File already exists: {path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _append_file(self, path: Path, content: str) -> Dict[str, Any]:
        """
        Append text content to an existing file.

        Args:
            path (Path): The path to the file to append to.
            content (str): The content to append to the file.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and file data.
        """
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "data": {
                    "path": str(path),
                    "appended_size": len(content),
                    "total_size": path.stat().st_size
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_directory(self, path: Path, create_parents: bool = True) -> Dict[str, Any]:
        """
        Create a directory.

        Args:
            path (Path): The path to the directory to create.
            create_parents (bool, optional): Whether to create parent directories if they don't exist. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and directory data.
        """
        try:
            if create_parents:
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(exist_ok=False)
            
            return {
                "success": True,
                "data": {
                    "path": str(path),
                    "created": True,
                    "is_directory": True
                }
            }
        except FileExistsError:
            return {
                "success": False,
                "error": f"Directory already exists: {path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _list_directory(self, path: Path) -> Dict[str, Any]:
        """
        List the contents of a directory.

        Args:
            path (Path): The path to the directory to list.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and directory contents.
        """
        try:
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Directory does not exist: {path}"
                }
            
            if not path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {path}"
                }
            
            # Import datetime here as it was missing in the original code
            from datetime import datetime
            items = []
            for item in path.iterdir():
                stat = item.stat()
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "is_directory": item.is_dir(),
                    "is_file": item.is_file(),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "mime_type": mimetypes.guess_type(str(item))[0] or ("inode/directory" if item.is_dir() else "application/octet-stream")
                })
            
            return {
                "success": True,
                "data": {
                    "path": str(path),
                    "items": items,
                    "total_items": len(items),
                    "total_files": len([i for i in items if i["is_file"]]),
                    "total_directories": len([i for i in items if i["is_directory"]])
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_info(self, path: Path) -> Dict[str, Any]:
        """
        Get information about a file or directory.

        Args:
            path (Path): The path to the file or directory.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and file/directory information.
        """
        try:
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Path does not exist: {path}"
                }
            
            # Import datetime here as it was missing in the original code
            from datetime import datetime
            stat = path.stat()
            
            info = {
                "path": str(path),
                "name": path.name,
                "is_directory": path.is_dir(),
                "is_file": path.is_file(),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "mime_type": mimetypes.guess_type(str(path))[0] or ("inode/directory" if path.is_dir() else "application/octet-stream")
            }
            
            if path.is_file():
                info.update({
                    "extension": path.suffix,
                    "readable": os.access(path, os.R_OK),
                    "writable": os.access(path, os.W_OK),
                    "executable": os.access(path, os.X_OK)
                })
            
            return {
                "success": True,
                "data": info
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _delete_file(self, path: Path) -> Dict[str, Any]:
        """
        Delete a file.

        Args:
            path (Path): The path to the file to delete.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and deletion information.
        """
        try:
            if not path.exists():
                return {
                    "success": False,
                    "error": f"File does not exist: {path}"
                }
            
            if path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is a directory, use delete_directory: {path}"
                }
            
            path.unlink()
            
            return {
                "success": True,
                "data": {
                    "path": str(path),
                    "deleted": True,
                    "was_file": True
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _delete_directory(self, path: Path) -> Dict[str, Any]:
        """
        Delete a directory.

        Args:
            path (Path): The path to the directory to delete.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and deletion information.
        """
        try:
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Directory does not exist: {path}"
                }
            
            if not path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {path}"
                }
            
            import shutil
            shutil.rmtree(path)
            
            return {
                "success": True,
                "data": {
                    "path": str(path),
                    "deleted": True,
                    "was_directory": True
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _move(self, source_path: Path, target_path: Path) -> Dict[str, Any]:
        """
        Move a file or directory from source path to target path.

        Args:
            source_path (Path): The source path of the file or directory.
            target_path (Path): The target path to move to.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and move information.
        """
        try:
            if not source_path.exists():
                return {
                    "success": False,
                    "error": f"Source path does not exist: {source_path}"
                }
            
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(target_path))
            
            return {
                "success": True,
                "data": {
                    "source": str(source_path),
                    "target": str(target_path),
                    "moved": True
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _copy(self, source_path: Path, target_path: Path) -> Dict[str, Any]:
        """
        Copy a file or directory from source path to target path.

        Args:
            source_path (Path): The source path of the file or directory.
            target_path (Path): The target path to copy to.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and copy information.
        """
        try:
            if not source_path.exists():
                return {
                    "success": False,
                    "error": f"Source path does not exist: {source_path}"
                }
            
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            if source_path.is_dir():
                shutil.copytree(str(source_path), str(target_path))
            else:
                shutil.copy2(str(source_path), str(target_path))
            
            return {
                "success": True,
                "data": {
                    "source": str(source_path),
                    "target": str(target_path),
                    "copied": True,
                    "is_directory": source_path.is_dir()
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _search_files(self, path: Path, pattern: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Search for files matching a given pattern in a directory.

        Args:
            path (Path): The directory path to search in.
            pattern (str): The search pattern for files.
            recursive (bool, optional): Whether to search recursively. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result and search matches.
        """
        try:
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Path does not exist: {path}"
                }
            
            import fnmatch
            matches = []
            
            if recursive:
                for root, dirs, files in os.walk(path):
                    for filename in fnmatch.filter(files, pattern):
                        file_path = Path(root) / filename
                        matches.append(str(file_path))
            else:
                for item in path.iterdir():
                    if item.is_file() and fnmatch.fnmatch(item.name, pattern):
                        matches.append(str(item))
            
            return {
                "success": True,
                "data": {
                    "pattern": pattern,
                    "path": str(path),
                    "recursive": recursive,
                    "matches": matches,
                    "count": len(matches)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a file system operation based on the given operation type and parameters.

        Args:
            operation (str): The type of file system operation to perform.
            **kwargs: Additional parameters for the operation.

        Returns:
            Dict[str, Any]: A dictionary containing the operation result.
        """
        try:
            path = kwargs.get("path", "")
            if not path:
                return {
                    "success": False,
                    "error": "Path is required"
                }
            
            resolved_path = self._validate_path(path)
            
            operations = {
                "read_file": lambda: self._read_file(resolved_path),
                "read_binary": lambda: self._read_binary(resolved_path),
                "write_file": lambda: self._write_file(
                    resolved_path,
                    kwargs.get("content", ""),
                    kwargs.get("create_parents", True)
                ),
                "create_file": lambda: self._create_file(
                    resolved_path,
                    kwargs.get("create_parents", True)
                ),
                "append_file": lambda: self._append_file(
                    resolved_path,
                    kwargs.get("content", "")
                ),
                "create_directory": lambda: self._create_directory(
                    resolved_path,
                    kwargs.get("create_parents", True)
                ),
                "list_directory": lambda: self._list_directory(resolved_path),
                "get_info": lambda: self._get_info(resolved_path),
                "delete_file": lambda: self._delete_file(resolved_path),
                "delete_directory": lambda: self._delete_directory(resolved_path),
                "move": lambda: self._move(
                    self._validate_path(kwargs.get("source_path", "")),
                    self._validate_path(kwargs.get("target_path", ""))
                ),
                "copy": lambda: self._copy(
                    self._validate_path(kwargs.get("source_path", "")),
                    self._validate_path(kwargs.get("target_path", ""))
                ),
                "search_files": lambda: self._search_files(
                    resolved_path,
                    kwargs.get("pattern", "*"),
                    kwargs.get("recursive", True)
                )
            }
            
            if operation not in operations:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                }
            
            return operations[operation]()
            
        except ValueError as e:
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }




@mcp.tool()
def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """Read content from a file."""
    try:
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        
        if not path.is_file():
            return {"success": False, "error": f"Path is not a file: {file_path}"}
        
        content = path.read_text(encoding=encoding)
        
        return {
            "success": True,
            "content": content,
            "size": len(content),
            "file_path": str(path.absolute())
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def write_file(file_path: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """Write content to a file."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        path.write_text(content, encoding=encoding)
        
        return {
            "success": True,
            "file_path": str(path.absolute()),
            "size": len(content),
            "message": f"File written successfully: {file_path}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def list_directory(directory_path: str = ".") -> Dict[str, Any]:
    """List contents of a directory."""
    try:
        path = Path(directory_path)
        if not path.exists():
            return {"success": False, "error": f"Directory not found: {directory_path}"}
        
        if not path.is_dir():
            return {"success": False, "error": f"Path is not a directory: {directory_path}"}
        
        items = []
        for item in path.iterdir():
            items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0,
                "modified": item.stat().st_mtime
            })
        
        return {
            "success": True,
            "directory": str(path.absolute()),
            "items": items,
            "count": len(items)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def create_directory(directory_path: str, parents: bool = True) -> Dict[str, Any]:
    """Create a directory."""
    try:
        path = Path(directory_path)
        path.mkdir(parents=parents, exist_ok=True)
        
        return {
            "success": True,
            "directory": str(path.absolute()),
            "message": f"Directory created: {directory_path}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def delete_path(file_path: str) -> Dict[str, Any]:
    """Delete a file or directory."""
    try:
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"Path not found: {file_path}"}
        
        if path.is_dir():
            shutil.rmtree(path)
            message = f"Directory deleted: {file_path}"
        else:
            path.unlink()
            message = f"File deleted: {file_path}"
        
        return {
            "success": True,
            "message": message
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def file_info(file_path: str) -> Dict[str, Any]:
    """Get detailed information about a file or directory."""
    try:
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"Path not found: {file_path}"}
        
        stat = path.stat()
        
        return {
            "success": True,
            "path": str(path.absolute()),
            "name": path.name,
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
            "is_directory": path.is_dir(),
            "is_file": path.is_file(),
            "extension": path.suffix if path.is_file() else None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }