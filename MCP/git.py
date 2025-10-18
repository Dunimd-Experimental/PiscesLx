#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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
import subprocess
from MCP import mcp
from typing import Dict, Any, List, Optional

class GitTool:
    """Git repository management and analysis tools."""
    
    def __init__(self, repo_path: str = None):
        self.repo_path = repo_path or os.getcwd()
    
    def _run_git_command(self, command: List[str], cwd: str = None) -> Dict[str, Any]:
        """Run a git command and return the result."""
        try:
            cwd = cwd or self.repo_path
            result = subprocess.run(
                ['git'] + command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout.strip(),
                    "stderr": result.stderr.strip()
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr.strip() or f"Git command failed with exit code {result.returncode}",
                    "exit_code": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Git command timed out"
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Git is not installed or not found in PATH"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def is_git_repo(self, path: str = None) -> bool:
        """Check if the given path is a git repository."""
        try:
            path = path or self.repo_path
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

# Global instance
git_tool = GitTool()

@mcp.tool()
def git_status(repo_path: str = None) -> Dict[str, Any]:
    """Get the current git status of a repository."""
    try:
        if repo_path is None:
            repo_path = os.getcwd()
        
        if not git_tool.is_git_repo(repo_path):
            return {
                "success": False,
                "error": f"Not a git repository: {repo_path}"
            }
        
        result = git_tool._run_git_command(['status', '--porcelain'], cwd=repo_path)
        
        if result["success"]:
            lines = result["output"].split('\n') if result["output"] else []
            status_info = []
            
            for line in lines:
                if line.strip():
                    status = line[:2]
                    filename = line[3:]
                    status_info.append({
                        "status": status,
                        "filename": filename,
                        "staged": status[0] != ' ',
                        "modified": status[1] != ' '
                    })
            
            return {
                "success": True,
                "repo_path": repo_path,
                "files": status_info,
                "clean": len(status_info) == 0,
                "count": len(status_info)
            }
        else:
            return result
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def git_log(repo_path: str = None, max_count: int = 10) -> Dict[str, Any]:
    """Get the git commit history."""
    try:
        if repo_path is None:
            repo_path = os.getcwd()
        
        if not git_tool.is_git_repo(repo_path):
            return {
                "success": False,
                "error": f"Not a git repository: {repo_path}"
            }
        
        result = git_tool._run_git_command([
            'log', 
            '--oneline', 
            '--max-count', str(max_count),
            '--format=format:%H|%s|%an|%ae|%ad'
        ], cwd=repo_path)
        
        if result["success"]:
            lines = result["output"].split('\n') if result["output"] else []
            commits = []
            
            for line in lines:
                if line.strip() and '|' in line:
                    parts = line.split('|', 4)
                    if len(parts) == 5:
                        commits.append({
                            "hash": parts[0],
                            "message": parts[1],
                            "author": parts[2],
                            "email": parts[3],
                            "date": parts[4]
                        })
            
            return {
                "success": True,
                "repo_path": repo_path,
                "commits": commits,
                "count": len(commits)
            }
        else:
            return result
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def git_branch(repo_path: str = None) -> Dict[str, Any]:
    """Get the list of branches in a repository."""
    try:
        if repo_path is None:
            repo_path = os.getcwd()
        
        if not git_tool.is_git_repo(repo_path):
            return {
                "success": False,
                "error": f"Not a git repository: {repo_path}"
            }
        
        result = git_tool._run_git_command(['branch', '-a'], cwd=repo_path)
        
        if result["success"]:
            lines = result["output"].split('\n') if result["output"] else []
            branches = []
            current_branch = None
            
            for line in lines:
                line = line.strip()
                if line:
                    is_current = line.startswith('*')
                    branch_name = line[2:] if is_current else line
                    
                    if is_current:
                        current_branch = branch_name
                    
                    branches.append({
                        "name": branch_name,
                        "is_current": is_current,
                        "is_remote": branch_name.startswith('remotes/')
                    })
            
            return {
                "success": True,
                "repo_path": repo_path,
                "branches": branches,
                "current_branch": current_branch,
                "count": len(branches)
            }
        else:
            return result
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def git_remote(repo_path: str = None) -> Dict[str, Any]:
    """Get the list of remote repositories."""
    try:
        if repo_path is None:
            repo_path = os.getcwd()
        
        if not git_tool.is_git_repo(repo_path):
            return {
                "success": False,
                "error": f"Not a git repository: {repo_path}"
            }
        
        result = git_tool._run_git_command(['remote', '-v'], cwd=repo_path)
        
        if result["success"]:
            lines = result["output"].split('\n') if result["output"] else []
            remotes = []
            
            for line in lines:
                if line.strip() and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        name = parts[0]
                        url_and_type = parts[1].split(' ')
                        if len(url_and_type) >= 2:
                            remotes.append({
                                "name": name,
                                "url": url_and_type[0],
                                "type": url_and_type[1].strip('()')
                            })
            
            return {
                "success": True,
                "repo_path": repo_path,
                "remotes": remotes,
                "count": len(remotes)
            }
        else:
            return result
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def git_diff(repo_path: str = None, staged: bool = False) -> Dict[str, Any]:
    """Show changes between commits, commit and working tree, etc."""
    try:
        if repo_path is None:
            repo_path = os.getcwd()
        
        if not git_tool.is_git_repo(repo_path):
            return {
                "success": False,
                "error": f"Not a git repository: {repo_path}"
            }
        
        cmd = ['diff', '--no-color']
        if staged:
            cmd.append('--staged')
        
        result = git_tool._run_git_command(cmd, cwd=repo_path)
        
        if result["success"]:
            return {
                "success": True,
                "repo_path": repo_path,
                "diff": result["output"],
                "staged": staged,
                "has_changes": bool(result["output"].strip())
            }
        else:
            return result
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }