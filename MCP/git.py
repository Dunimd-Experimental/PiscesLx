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
import logging
import subprocess
from pathlib import Path
from .simple_mcp import register_tool
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class GitTool:
    """
    A tool for managing Git repositories.
    
    This class provides methods to perform various Git operations, 
    such as status checking, committing changes, and cloning repositories.
    """
    
    def __init__(self):
        """
        Initialize the GitTool instance.
        
        Sets the name and description of the tool.
        """
        self.name = "git"
        self.description = "Git repository operations and management"
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema for Git operation parameters.
        
        Returns:
            Dict[str, Any]: A dictionary defining the structure and constraints of the input parameters.
        """
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Git operation to perform",
                    "enum": [
                        "status", "log", "diff", "add", "commit", "branch", 
                        "checkout", "init", "clone", "push", "pull"
                    ]
                },
                "repo_path": {
                    "type": "string",
                    "description": "Path to the Git repository"
                },
                "message": {
                    "type": "string",
                    "description": "Commit message (for commit operation)"
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to add (for add operation)"
                },
                "branch_name": {
                    "type": "string",
                    "description": "Branch name (for branch/checkout operations)"
                },
                "remote_url": {
                    "type": "string",
                    "description": "Remote repository URL (for clone operation)"
                },
                "max_count": {
                    "type": "integer",
                    "description": "Maximum number of commits to show (for log operation)",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": ["operation", "repo_path"]
        }
    
    def _run_git_command(self, repo_path: str, *args) -> Dict[str, Any]:
        """
        Run a Git command and return the execution results.
        
        Args:
            repo_path (str): Path to the Git repository.
            *args: Additional arguments for the Git command.
            
        Returns:
            Dict[str, Any]: A dictionary containing the execution status and result data.
        """
        try:
            # Resolve the full path of the repository
            repo_path = Path(repo_path).expanduser().resolve()
            
            # Check if the repository path exists
            if not repo_path.exists():
                return {
                    "success": False,
                    "error": f"Repository path does not exist: {repo_path}"
                }
            
            # Check if the path is a Git repository
            git_dir = repo_path / '.git'
            if not git_dir.exists() and 'init' not in args:
                return {
                    "success": False,
                    "error": f"Not a git repository: {repo_path}"
                }
            
            # Construct the Git command
            cmd = ['git', '-C', str(repo_path)] + list(args)
            
            # Execute the Git command
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            # Check if the command execution failed
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr.strip() or result.stdout.strip()
                }
            
            return {
                "success": True,
                "data": {
                    "output": result.stdout.strip(),
                    "repo_path": str(repo_path)
                }
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Git command timed out"
            }
        except Exception as e:
            logger.error(f"Git command error: {str(e)}")
            return {
                "success": False,
                "error": f"Git error: {str(e)}"
            }
    
    def _parse_git_status(self, status_output: str) -> Dict[str, Any]:
        """
        Parse the output of the 'git status --porcelain' command.
        
        Args:
            status_output (str): The output of the 'git status --porcelain' command.
            
        Returns:
            Dict[str, Any]: A dictionary containing the parsed status information.
        """
        lines = status_output.strip().split('\n')
        
        staged = []
        unstaged = []
        untracked = []
        
        # Iterate through each line of the output to parse the status
        for line in lines[2:] if len(lines) > 2 else lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('??'):
                untracked.append(line[3:].strip())
            elif len(line) >= 3:
                index_status = line[0]
                worktree_status = line[1]
                filename = line[3:].strip()
                
                if index_status != ' ':
                    staged.append(filename)
                if worktree_status != ' ':
                    unstaged.append(filename)
        
        return {
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked,
            "clean": not any([staged, unstaged, untracked])
        }
    
    def execute(self, operation: str, repo_path: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a specified Git operation.
        
        Args:
            operation (str): The Git operation to perform.
            repo_path (str): Path to the Git repository.
            **kwargs: Additional parameters for the operation.
            
        Returns:
            Dict[str, Any]: A dictionary containing the execution status and result data.
        """
        if operation == "status":
            result = self._run_git_command(repo_path, 'status', '--porcelain')
            if result['success']:
                result['data']['parsed'] = self._parse_git_status(result['data']['output'])
            return result
            
        elif operation == "log":
            max_count = kwargs.get('max_count', 10)
            return self._run_git_command(
                repo_path, 'log', '--oneline', f'-{max_count}', '--decorate', '--graph'
            )
            
        elif operation == "diff":
            return self._run_git_command(repo_path, 'diff', '--cached')
            
        elif operation == "add":
            files = kwargs.get('files', [])
            if not files:
                return {
                    "success": False,
                    "error": "No files specified for add operation"
                }
            return self._run_git_command(repo_path, 'add', *files)
            
        elif operation == "commit":
            message = kwargs.get('message', '')
            if not message:
                return {
                    "success": False,
                    "error": "No commit message provided"
                }
            return self._run_git_command(repo_path, 'commit', '-m', message)
            
        elif operation == "branch":
            branch_name = kwargs.get('branch_name')
            if branch_name:
                return self._run_git_command(repo_path, 'checkout', '-b', branch_name)
            else:
                return self._run_git_command(repo_path, 'branch', '-a')
                
        elif operation == "checkout":
            branch_name = kwargs.get('branch_name', '')
            if not branch_name:
                return {
                    "success": False,
                    "error": "No branch name provided"
                }
            return self._run_git_command(repo_path, 'checkout', branch_name)
            
        elif operation == "init":
            repo_path = Path(kwargs.get('repo_path', repo_path)).expanduser().resolve()
            try:
                repo_path.mkdir(parents=True, exist_ok=True)
                result = self._run_git_command(str(repo_path), 'init')
                return result
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to initialize repository: {str(e)}"
                }
                
        elif operation == "clone":
            remote_url = kwargs.get('remote_url', '')
            if not remote_url:
                return {
                    "success": False,
                    "error": "No remote URL provided"
                }
            target_dir = kwargs.get('repo_path', '.')
            return self._run_git_command(target_dir, 'clone', remote_url, target_dir)
            
        elif operation == "push":
            return self._run_git_command(repo_path, 'push')
            
        elif operation == "pull":
            return self._run_git_command(repo_path, 'pull')
            
        else:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}"
            }

# Register the Git tool
git_tool = GitTool()
register_tool(git_tool.name, git_tool.description, git_tool.get_schema(), git_tool.execute)