#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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

import subprocess
from dataclasses import dataclass, field


@dataclass
class GitState:
    in_repo: bool
    commit_hash: str = ""
    branch: str = ""
    remote_url: str = ""
    is_clean: bool = True
    changed_files: list[str] = field(default_factory=list)
    untracked_files: list[str] = field(default_factory=list)
    has_unpushed: bool = False
    worktree_count: int = 1


class EncreGitRepo:
    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self._git_dir = self._find_git_root()
        self._in_repo = self._git_dir is not None

    def is_in_repo(self) -> bool:
        return self._in_repo

    def get_state(self) -> GitState:
        if not self._in_repo:
            return GitState(in_repo=False)

        commit_hash = self._get_commit_hash()
        branch = self._get_branch()
        remote_url = self._get_remote_url()
        is_clean = self._is_clean()
        changed_files = self._get_changed_files()
        untracked_files = self._get_untracked_files()
        has_unpushed = self._has_unpushed_commits()
        worktree_count = self._get_worktree_count()

        return GitState(
            in_repo=True,
            commit_hash=commit_hash,
            branch=branch,
            remote_url=remote_url,
            is_clean=is_clean,
            changed_files=changed_files,
            untracked_files=untracked_files,
            has_unpushed=has_unpushed,
            worktree_count=worktree_count,
        )

    def get_diff(self, file_path: str | None = None) -> str:
        if not self._in_repo:
            return ""
        args = ["git", "diff", "HEAD", "--"]
        if file_path:
            args.append(file_path)
        return self._run_git(args)

    def get_diff_stats(self) -> dict[str, int]:
        if not self._in_repo:
            return {"files": 0, "insertions": 0, "deletions": 0}
        output = self._run_git(["git", "diff", "--numstat", "HEAD"])
        return self._parse_numstat(output)

    def get_changed_files(self) -> list[str]:
        if not self._in_repo:
            return []
        return self._get_changed_files()

    def stash_to_clean_state(self) -> str | None:
        if not self._in_repo:
            return None
        try:
            self._run_git(["git", "stash", "push", "-m", "encre-auto-stash"])
            return "encre-auto-stash"
        except Exception:
            return None

    def unstash(self) -> None:
        if not self._in_repo:
            return
        try:
            self._run_git(["git", "stash", "pop"])
        except Exception:
            pass

    def is_transient_state(self) -> bool:
        if not self._in_repo:
            return False
        assert self._git_dir is not None
        import os
        transient_dirs = ["MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "BISECT_START", "rebase-merge", "rebase-apply"]
        for name in transient_dirs:
            if os.path.exists(os.path.join(self._git_dir, name)):
                return True
        return False

    def has_unpushed_commits(self) -> bool:
        if not self._in_repo:
            return False
        return self._has_unpushed_commits()

    def get_commit_hash(self) -> str:
        if not self._in_repo:
            return ""
        return self._get_commit_hash()

    def get_branch(self) -> str:
        if not self._in_repo:
            return ""
        return self._get_branch()

    def _get_commit_hash(self) -> str:
        try:
            return self._run_git(["git", "rev-parse", "HEAD"]).strip()
        except Exception:
            return ""

    def _get_branch(self) -> str:
        try:
            branch = self._run_git(["git", "branch", "--show-current"]).strip()
            if not branch:
                branch = self._run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
            return branch
        except Exception:
            return ""

    def _get_remote_url(self) -> str:
        try:
            return self._run_git(["git", "remote", "get-url", "origin"]).strip()
        except Exception:
            return ""

    def _is_clean(self) -> bool:
        try:
            output = self._run_git(["git", "status", "--porcelain"])
            return output.strip() == ""
        except Exception:
            return True

    def _get_changed_files(self) -> list[str]:
        try:
            output = self._run_git(["git", "status", "--porcelain"])
            files: list[str] = []
            for line in output.strip().split("\n"):
                line = line.strip()
                if len(line) >= 3:
                    path = line[3:].strip()
                    if path and path not in files:
                        files.append(path)
            return files
        except Exception:
            return []

    def _get_untracked_files(self) -> list[str]:
        try:
            output = self._run_git(["git", "ls-files", "--others", "--exclude-standard"])
            return [f for f in output.strip().split("\n") if f]
        except Exception:
            return []

    def _has_unpushed_commits(self) -> bool:
        try:
            output = self._run_git(["git", "log", "@{u}.."]).strip()
            return bool(output)
        except Exception:
            return False

    def _get_worktree_count(self) -> int:
        try:
            output = self._run_git(["git", "worktree", "list"]).strip()
            if not output:
                return 1
            return len(output.split("\n"))
        except Exception:
            return 1

    @staticmethod
    def _parse_numstat(output: str) -> dict[str, int]:
        files = 0
        insertions = 0
        deletions = 0
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                files += 1
                try:
                    add = int(parts[0]) if parts[0] != "-" else 0
                    dlt = int(parts[1]) if parts[1] != "-" else 0
                    insertions += add
                    deletions += dlt
                except ValueError:
                    pass
        return {"files": files, "insertions": insertions, "deletions": deletions}

    def _run_git(self, args: list[str], timeout: float = 15.0) -> str:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=self.workspace,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout

    def _find_git_root(self) -> str | None:
        import os
        current = os.path.abspath(self.workspace)
        while True:
            git_path = os.path.join(current, ".git")
            if os.path.exists(git_path):
                return git_path
            parent = os.path.dirname(current)
            if parent == current:
                return None
            current = parent
