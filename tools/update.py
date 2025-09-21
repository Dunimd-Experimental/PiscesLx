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
import sys
from pathlib import Path
from utils import display_update_log, RIGHT, DEBUG, ERROR

# Try to import GitPython. If unavailable, use subprocess as fallback.
try:
    from git import Repo, InvalidGitRepositoryError, GitCommandError
    GITPYTHON_AVAILABLE = True
except ImportError:
    GITPYTHON_AVAILABLE = False
    import subprocess

def update():
    """
    Pull the latest code from the remote repository.
    Use GitPython for better integration, and fall back to subprocess if GitPython is not available.
    
    After the update is successful, display the latest update log.
    """
    # Primary remote repository URL
    remote_url = 'https://gitee.com/dunimd/piscesl1.git'
    # Backup remote repository URL
    backup_url = 'https://github.com/mf2023/piscesl1.git'
    
    # Get the current working directory, which is assumed to be the project root
    project_root = Path().cwd()
    
    if GITPYTHON_AVAILABLE:
        _update_with_gitpython(project_root, remote_url, backup_url)
    else:
        DEBUG("GitPython not available, falling back to system git commands")
        _update_with_subprocess(remote_url)
    
    # Display the latest update log after a successful update
    display_update_log(project_root)

def _update_with_gitpython(project_root: Path, primary_url: str, backup_url: str):
    """
    Update the repository using GitPython with enhanced error handling and progress reporting.

    Args:
        project_root (Path): Path to the project root directory.
        primary_url (str): URL of the primary remote repository.
        backup_url (str): URL of the backup remote repository.
    """
    try:
        # Try to initialize or get an existing Git repository
        try:
            repo = Repo(project_root)
            DEBUG("Found existing git repository")
        except InvalidGitRepositoryError:
            ERROR("Current directory is not a git repository")
            DEBUG(f"To clone the repository: git clone {primary_url}")
            sys.exit(1)
        
        # Check if there are any remote repositories configured
        if not repo.remotes:
            DEBUG("No remotes found, adding origin...")
            repo.create_remote('origin', primary_url)
        
        # Get the 'origin' remote repository
        origin = repo.remote('origin')
        
        # Get the name of the current branch
        try:
            current_branch = repo.active_branch.name
            DEBUG(f"Current branch: {current_branch}")
        except TypeError:
            # Handle the detached HEAD state
            current_branch = 'master'
            DEBUG("In detached HEAD state, assuming master branch")
        
        # Save uncommitted changes if they exist
        if repo.is_dirty():
            DEBUG("Working directory has uncommitted changes, stashing...")
            repo.git.stash('save', 'Auto-stash before update')
        
        # Fetch the latest changes from the remote repository
        DEBUG(f"Fetching latest changes from {primary_url}...")
        try:
            origin.fetch()
        except GitCommandError as e:
            if "gitee.com" in str(e) and backup_url:
                DEBUG("Primary remote failed, trying backup repository...")
                try:
                    # Try using the backup URL
                    origin.set_url(backup_url)
                    origin.fetch()
                    DEBUG(f"Successfully connected to backup: {backup_url}")
                except GitCommandError:
                    origin.set_url(primary_url)  # Restore the original URL
                    raise
            else:
                raise
        
        # Get information about the latest commit
        try:
            latest_commit = origin.refs[current_branch].commit
            current_commit = repo.head.commit
            
            if latest_commit.hexsha == current_commit.hexsha:
                RIGHT("Repository is already up to date")
                return
            
            # List the commits that the local repository is behind
            commits_behind = list(repo.iter_commits(f'{current_commit.hexsha}..{latest_commit.hexsha}'))
            DEBUG(f"Found {len(commits_behind)} new commit(s)")
            
            # Display the last 3 new commits
            for commit in commits_behind[:3]:
                DEBUG(f"  - {commit.hexsha[:8]}: {commit.summary}")
            if len(commits_behind) > 3:
                DEBUG(f"  ... and {len(commits_behind) - 3} more commits")
        
        except Exception as e:
            DEBUG(f"Could not get commit info: {e}")
        
        # Reset the local repository to the latest version of the remote branch
        DEBUG("Updating to latest version...")
        repo.git.reset('--hard', f'origin/{current_branch}')
        
        # Clean up untracked files
        repo.git.clean('-fd')
        
        RIGHT("Code successfully updated to the latest version")
        
        # Check if there are stashed changes and prompt the user
        try:
            stashes = repo.git.stash('list').strip()
            if stashes and 'Auto-stash before update' in stashes:
                DEBUG("Previous changes were stashed. Run 'git stash pop' to restore them.")
        except:
            pass
            
    except GitCommandError as e:
        ERROR(f"Git operation failed: {e}")
        ERROR("This might be due to network issues or repository access problems")
        sys.exit(1)
    except Exception as e:
        ERROR(f"Unexpected error during update: {e}")
        sys.exit(1)

def _update_with_subprocess(remote_url: str):
    """
    Fallback method to update the repository by calling git commands via subprocess.

    Args:
        remote_url (str): URL of the remote repository.
    """
    try:
        DEBUG(f"Pulling latest code from {remote_url}...")
        subprocess.run(['git', 'fetch', '--all'], check=True)
        subprocess.run(['git', 'reset', '--hard', 'origin/master'], check=True)
        RIGHT("Code successfully updated to the latest version")
    except subprocess.CalledProcessError as e:
        ERROR(f"Failed to pull code: {e}")
        sys.exit(1)
    except FileNotFoundError:
        ERROR("Git command not found. Please install git or ensure it's in your PATH")
        sys.exit(1)