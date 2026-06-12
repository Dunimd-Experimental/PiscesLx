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

import base64
import copy
import json
import os
import pathlib
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from encre.config import EncreConfig
from encre.utils.idgen import BranchIDGenerator
from encre.utils.tokens import count_message_tokens, estimate_tokens


@dataclass
class SessionCheckpoint:
    checkpoint_id: str
    label: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    turn_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    plan_items: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = 0.0


@dataclass
class BranchMeta:
    id: str
    parent_branch_id: str | None = None
    fork_point_message_id: str | None = None
    created_at: float = 0.0
    messages_count: int = 0
    tokens: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0, "total": 0})
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    compactions: list[dict[str, Any]] = field(default_factory=list)


class EncreSession:
    SYSTEM_BUDGET_RATIO = 0.15
    HISTORY_BUDGET_RATIO = 0.50
    RESPONSE_BUDGET_RATIO = 0.35

    def __init__(self, config: EncreConfig) -> None:
        self.id: str = str(uuid.uuid4())
        self.config: EncreConfig = config
        self.messages: list[dict[str, Any]] = []
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.tool_call_count: int = 0
        self.turn_count: int = 0
        self.metadata: dict[str, Any] = {}
        self._checkpoints: OrderedDict[str, SessionCheckpoint] = OrderedDict()
        self._max_checkpoints: int = config.checkpoint_max_count
        self.artifacts: list[dict[str, Any]] = []
        self.plan_items: list[dict[str, Any]] = []
        self.active_branch_id: str = f"br_{1:04d}"
        self._branch_counter: int = 1
        self.branches: dict[str, BranchMeta] = {}
        self.branches[self.active_branch_id] = BranchMeta(id=self.active_branch_id, created_at=time.time())
        self._branch_last_seq: dict[str, int] = {self.active_branch_id: -1}
        self._branch_last_message_id: dict[str, str | None] = {self.active_branch_id: None}
        self._context_cache: dict[str, list[dict[str, Any]]] = {}
        self._messages_version: int = 0
        self._turn_partition_cache: tuple[int, list[list[dict[str, Any]]]] | None = None
        self._summary_cache: tuple[int, int, str] | None = None

    def mark_messages_dirty(self) -> None:
        self._messages_version += 1
        self._context_cache.clear()
        self._turn_partition_cache = None
        self._summary_cache = None

    def add_message(self, role: str, content: str | list[dict[str, Any]], **kwargs: Any) -> None:
        message: dict[str, Any] = {"role": role, "content": content, "created_at": int(time.time() * 1000)}
        message["branch_id"] = self.active_branch_id
        seq = self._get_next_seq(self.active_branch_id)
        message["seq_in_branch"] = seq
        message["id"] = BranchIDGenerator.message_id(self.id, self.active_branch_id, seq)
        message["parent_id"] = self._branch_last_message_id.get(self.active_branch_id)
        message.update(kwargs)
        self.messages.append(message)
        self._branch_last_seq[self.active_branch_id] = seq
        self._branch_last_message_id[self.active_branch_id] = message["id"]
        self.mark_messages_dirty()
        self.updated_at = time.time()

    def add_user_message_with_image(self, text: str, image_path: str) -> None:
        """Add a user message containing both text and an image."""
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if image_path.startswith(("http://", "https://")):
            content.append({
                "type": "image_url",
                "image_url": {"url": image_path},
            })
        else:
            mime = _guess_image_mime(image_path)
            try:
                with open(image_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            })
        self.add_message("user", content)

    def add_user_message_with_file(self, text: str, file_path: str) -> None:
        """Add a user message with an attached file's content."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception:
            return
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        filename = os.path.basename(file_path)
        content.append({
            "type": "text",
            "text": f"\n<attached_file filename=\"{filename}\">\n{file_content}\n</attached_file>",
        })
        self.add_message("user", content)

    def add_message_content(self, role: str, blocks: list[dict[str, Any]]) -> None:
        """Add a message with raw content blocks (for advanced multimodal use)."""
        self.add_message(role, blocks)

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        is_error: bool = False,
        sub_agent_messages: list[dict[str, Any]] | None = None,
        sub_agent_session_id: str | None = None,
    ) -> None:
        seq = self._get_next_seq(self.active_branch_id)
        msg: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
            "branch_id": self.active_branch_id,
            "seq_in_branch": seq,
            "id": BranchIDGenerator.message_id(self.id, self.active_branch_id, seq),
        }
        if sub_agent_messages is not None:
            msg["sub_agent_messages"] = sub_agent_messages
        if sub_agent_session_id is not None:
            msg["sub_agent_session_id"] = sub_agent_session_id
        msg["parent_id"] = self._branch_last_message_id.get(self.active_branch_id)
        self.messages.append(msg)
        self._branch_last_seq[self.active_branch_id] = seq
        self._branch_last_message_id[self.active_branch_id] = msg["id"]
        self.mark_messages_dirty()
        self.tool_call_count += 1
        self.updated_at = time.time()

    def add_artifact(self, file_path: str, tool_name: str = "file_write", diff_text: str = "") -> dict[str, Any]:
        import os
        from pathlib import Path
        name = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        try:
            size = os.path.getsize(file_path)
        except OSError:
            size = 0
        entry: dict[str, Any] = {
            "path": file_path,
            "name": name,
            "ext": ext,
            "size": size,
            "tool": tool_name,
            "created_at": time.time(),
            "diff_text": diff_text,
            "branch_id": self.active_branch_id,
        }
        existing = {a["path"] for a in self.artifacts}
        if file_path not in existing:
            self.artifacts.append(entry)
            if self.active_branch_id in self.branches:
                self.branches[self.active_branch_id].artifacts.append(entry)
        return entry

    def rebuild_artifacts_from_messages(self) -> None:
        """Scan self.messages and rebuild artifacts from tool call data.

        Parses file paths from tool call arguments (reliable JSON), extracts
        diffs from corresponding tool results, and sets self.artifacts and
        per-branch artifacts.  Called after loading session from disk so that
        artifacts are always authoritative regardless of streaming bugs.
        """
        _FILE_TOOL_NAMES = {"file_write", "file_edit", "write_file", "writeFile", "apply_patch"}
        import re as _re

        # Build tool_call_id -> result lookup
        tool_results: dict[str, str] = {}
        for msg in self.messages:
            if msg.get("role") == "tool":
                tcid = msg.get("tool_call_id", "")
                if tcid:
                    tool_results[tcid] = str(msg.get("content", ""))

        new_artifacts: list[dict[str, Any]] = []
        branch_artifacts: dict[str, list[dict[str, Any]]] = {}
        global_seen: set[str] = set()

        for msg in self.messages:
            if msg.get("role") != "assistant":
                continue
            branch_id = msg.get("branch_id", self.active_branch_id)
            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                continue

            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                if name not in _FILE_TOOL_NAMES:
                    continue

                # Parse arguments to get file_path
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        continue
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    continue

                file_path = ""
                if name == "apply_patch":
                    patches = args.get("patches", [])
                    if patches and isinstance(patches, list):
                        file_path = patches[0].get("new_path") or patches[0].get("old_path", "")
                else:
                    file_path = args.get("file_path", "") or args.get("path", "")

                if not file_path:
                    continue

                # Deduplicate globally
                if file_path in global_seen:
                    continue
                global_seen.add(file_path)

                # Get diff text from corresponding tool result
                tc_id = tc.get("id", "")
                result = tool_results.get(tc_id, "")
                diff_text = ""
                if result:
                    m = _re.search(r"```diff\n(.+?)\n```", result, _re.DOTALL)
                    if m:
                        diff_text = m.group(1).strip()

                from pathlib import Path
                name_from_path = os.path.basename(file_path)
                ext = os.path.splitext(file_path)[1].lower().lstrip(".")
                try:
                    size = os.path.getsize(file_path)
                except OSError:
                    size = 0

                entry: dict[str, Any] = {
                    "path": file_path,
                    "name": name_from_path,
                    "ext": ext,
                    "size": size,
                    "tool": name,
                    "created_at": time.time(),
                    "diff_text": diff_text,
                    "branch_id": branch_id,
                }

                new_artifacts.append(entry)
                branch_artifacts.setdefault(branch_id, []).append(entry)

        self.artifacts = new_artifacts
        for bid, arts in branch_artifacts.items():
            if bid in self.branches:
                self.branches[bid].artifacts = arts

    def _get_next_seq(self, branch_id: str) -> int:
        return self._branch_last_seq.get(branch_id, -1) + 1

    def get_branch_messages(self, branch_id: str) -> list[dict[str, Any]]:
        lineage: list[str] = []
        current = self.branches.get(branch_id)
        while current:
            lineage.insert(0, current.id)
            current = self.branches.get(current.parent_branch_id) if current.parent_branch_id else None

        if not lineage:
            if branch_id == f"br_{1:04d}":
                lineage = [branch_id]

        result: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for i, bid in enumerate(lineage):
            branch_meta = self.branches.get(bid) if i > 0 else None
            msgs = sorted(
                [m for m in self.messages if m.get("branch_id") == bid],
                key=lambda m: m.get("seq_in_branch", 0),
            )
            if i < len(lineage) - 1:
                next_meta = self.branches.get(lineage[i + 1])
                if next_meta and next_meta.fork_point_message_id:
                    truncated: list[dict[str, Any]] = []
                    for m in msgs:
                        if m.get("id", "") == next_meta.fork_point_message_id:
                            break
                        truncated.append(m)
                    msgs = truncated
            if branch_meta:
                if branch_meta.fork_point_message_id:
                    first_id = msgs[0].get("id", "") if msgs else ""
                    if first_id == branch_meta.fork_point_message_id:
                        msgs = msgs[1:]
            for m in msgs:
                mid = m.get("id", "")
                if mid and mid in seen_ids:
                    continue
                if mid:
                    seen_ids.add(mid)
                result.append(m)

        if branch_id == f"br_{1:04d}":
            legacy = sorted(
                [m for m in self.messages if m.get("branch_id") is None],
                key=lambda m: self.messages.index(m),
            )
            for m in legacy:
                mid = m.get("id", "")
                if mid and mid in seen_ids:
                    continue
                if mid:
                    seen_ids.add(mid)
                result.append(m)

        return result

    def create_branch(self, from_branch_id: str, fork_point_message_id: str) -> BranchMeta:
        self._branch_counter += 1
        new_id = f"br_{self._branch_counter:04d}"
        meta = BranchMeta(
            id=new_id,
            parent_branch_id=from_branch_id,
            fork_point_message_id=fork_point_message_id,
            created_at=time.time(),
        )
        self.branches[new_id] = meta
        return meta

    def retry_at_user_index(self, user_index: int) -> tuple[str, BranchMeta]:
        """Fork at the N-th user message (0-based, within the active branch context).

        Finds the user message by its index across messages visible in the
        current branch context (not all messages in the session), creates a
        new branch forking from that message, and switches to the new branch.
        Returns (user_message_text, new_branch).

        Using the active branch context ensures the index matches what the
        frontend computes from displayed messages.  Counting across ALL
        messages would include user messages from unrelated branches and
        shift the index, causing the fork point to point to the wrong message.
        """
        branch_msgs = self.get_branch_messages(self.active_branch_id)
        user_msgs = [m for m in branch_msgs if m.get("role") == "user"]
        if user_index < 0 or user_index >= len(user_msgs):
            raise ValueError(
                f"User message index {user_index} out of range "
                f"(total user messages in active branch: {len(user_msgs)})"
            )
        target = user_msgs[user_index]
        # Find the actual session message by matching content+role (ID may differ
        # across branches for the same logical message).  Fall back to ID match.
        msg_id = target.get("id", "")
        if msg_id:
            # Verify the target message exists in the session (it should).
            session_match = next((m for m in self.messages if m.get("id") == msg_id), None)
            if not session_match:
                raise ValueError(f"User message {msg_id} not found in session")
        else:
            raise ValueError(f"User message at index {user_index} has no ID")

        original_branch = target.get("branch_id", self.active_branch_id)
        branch = self.create_branch(original_branch, msg_id)
        self.switch_branch(branch.id)

        content = target.get("content", "")
        if isinstance(content, list):
            texts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = " ".join(texts)
        return str(content), branch

    def retry_assistant(self, message_id: str) -> tuple[str, BranchMeta]:
        target = None
        for m in self.messages:
            if m.get("id", "").endswith(f":M:{message_id}") or m.get("id") == message_id:
                target = m
                break
        if not target:
            raise ValueError(f"Message {message_id} not found")

        parent_id = target.get("parent_id")
        if not parent_id:
            raise ValueError("Cannot retry: no parent message")
        parent = None
        for m in self.messages:
            if m.get("id") == parent_id:
                parent = m
                break
        if not parent:
            raise ValueError("Parent message not found")

        original_branch = target.get("branch_id", self.active_branch_id)
        branch = self.create_branch(original_branch, parent_id)
        self.switch_branch(branch.id)
        return parent_id, branch

    def switch_branch(self, branch_id: str) -> None:
        if branch_id in self.branches:
            self.active_branch_id = branch_id

    def rollback_to(self, branch_id: str, message_id: str) -> list[str]:
        target_seq = None
        for m in self.messages:
            if (m.get("id", "").endswith(f":M:{message_id}") or m.get("id") == message_id) and m.get("branch_id") == branch_id:
                target_seq = m.get("seq_in_branch", 0)
                break
        if target_seq is None:
            return []

        removed: list[str] = []
        remaining: list[dict[str, Any]] = []
        for m in self.messages:
            if m.get("branch_id") == branch_id and isinstance(m.get("seq_in_branch"), int) and m["seq_in_branch"] > target_seq:
                removed.append(m.get("id", ""))
            else:
                remaining.append(m)

        self.messages = remaining

        # Remove child branches (those that fork from this branch or its descendants)
        descendant_ids = {branch_id}
        changed = True
        while changed:
            changed = False
            for bid, bmeta in list(self.branches.items()):
                if bmeta.parent_branch_id in descendant_ids:
                    if bid not in descendant_ids:
                        descendant_ids.add(bid)
                        changed = True
        for bid in descendant_ids:
            if bid != branch_id:
                self.branches.pop(bid, None)

        if branch_id in self.branches:
            self.active_branch_id = branch_id
            meta = self.branches[branch_id]
            meta.messages_count = sum(1 for m in self.messages if m.get("branch_id") == branch_id)
            branch_msgs = [m for m in self.messages if m.get("branch_id") == branch_id]
            total = self.count_messages_tokens(branch_msgs)
            meta.tokens = {"input": total, "output": 0, "total": total}
            meta.artifacts = [a for a in meta.artifacts if a.get("id", "") not in removed]
            meta.compactions = [c for c in meta.compactions if c.get("id", "") not in removed]

        # Recalculate turn_count and tool_call_count from remaining messages
        # so that is_max_turns_reached() reflects actual conversation state
        # after rollback rather than the stale pre-rollback counter.
        branch_msgs = [m for m in self.messages if m.get("branch_id") == branch_id]
        self.turn_count = sum(1 for m in branch_msgs if m.get("role") == "assistant")
        self.tool_call_count = sum(
            1 for m in branch_msgs
            if m.get("role") == "assistant" and m.get("tool_calls")
        )

        self.mark_messages_dirty()
        self.updated_at = time.time()
        return removed

    def checkpoint(self, label: str = "") -> str:
        if self._max_checkpoints <= 0:
            return ""
        cid = str(uuid.uuid4())[:12]
        meta = copy.deepcopy(self.metadata)
        meta["_branch_state"] = {
            "active_branch_id": self.active_branch_id,
            "_branch_counter": self._branch_counter,
            "branches": {k: v.__dict__ for k, v in self.branches.items()},
        }
        cp = SessionCheckpoint(
            checkpoint_id=cid,
            label=label,
            messages=copy.deepcopy(self.messages),
            tool_call_count=self.tool_call_count,
            turn_count=self.turn_count,
            metadata=meta,
            plan_items=copy.deepcopy(self.plan_items),
            artifacts=copy.deepcopy(self.artifacts),
            created_at=time.time(),
        )
        self._checkpoints[cid] = cp
        while len(self._checkpoints) > self._max_checkpoints:
            self._checkpoints.popitem(last=False)
        return cid

    def rollback(self, checkpoint_id: str) -> bool:
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None:
            return False
        self.messages = copy.deepcopy(cp.messages)
        self.tool_call_count = cp.tool_call_count
        self.turn_count = cp.turn_count
        self.metadata = copy.deepcopy(cp.metadata)
        self.plan_items = copy.deepcopy(cp.plan_items)
        self.artifacts = copy.deepcopy(cp.artifacts)
        branch_state = cp.metadata.get("_branch_state", {})
        if branch_state:
            self.active_branch_id = branch_state.get("active_branch_id", self.active_branch_id)
            self._branch_counter = branch_state.get("_branch_counter", self._branch_counter)
            branches_data = branch_state.get("branches", {})
            if branches_data:
                self.branches = {}
                for k, v in branches_data.items():
                    self.branches[k] = BranchMeta(**v)
        self.rebuild_runtime_caches()
        self.updated_at = time.time()
        return True

    def list_checkpoints(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for cp in self._checkpoints.values():
            result.append({
                "checkpoint_id": cp.checkpoint_id,
                "label": cp.label,
                "message_count": len(cp.messages),
                "tool_call_count": cp.tool_call_count,
                "turn_count": cp.turn_count,
                "plan_items_count": len(cp.plan_items),
                "artifacts_count": len(cp.artifacts),
                "created_at": cp.created_at,
            })
        return result

    def clear_checkpoints(self) -> None:
        self._checkpoints.clear()

    def is_expired(self) -> bool:
        age = time.time() - self.created_at
        max_age_seconds = self.config.session_max_age_hours * 3600
        return age > max_age_seconds

    def is_max_turns_reached(self) -> bool:
        if self.config.max_turns <= 0:
            return False
        return self.turn_count >= self.config.max_turns

    def get_context_messages(self) -> list[dict[str, Any]]:
        cached = self._context_cache.get(self.active_branch_id)
        if cached is not None:
            return cached
        msgs = self.get_branch_messages(self.active_branch_id)
        self._context_cache[self.active_branch_id] = msgs
        return msgs

    def get_summary(self) -> tuple[int, str]:
        if self._summary_cache is not None and self._summary_cache[0] == self._messages_version:
            return self._summary_cache[1], self._summary_cache[2]

        msg_count = 0
        preview = ""
        for m in self.messages:
            if m.get("role") != "user":
                continue
            msg_count += 1
            if preview:
                continue
            c = m.get("content", "")
            if isinstance(c, str) and c.strip():
                preview = c.strip()[:80]
                continue
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip():
                        preview = b["text"].strip()[:80]
                        break

        self._summary_cache = (self._messages_version, msg_count, preview)
        return msg_count, preview

    def replace_branch_messages(self, branch_id: str, new_messages: list[dict[str, Any]]) -> None:
        other = [m for m in self.messages if m.get("branch_id") != branch_id]
        self.messages = other + new_messages
        last_seq = -1
        last_id: str | None = None
        for m in new_messages:
            s = m.get("seq_in_branch", -1)
            if isinstance(s, int) and s >= last_seq:
                last_seq = s
                last_id = m.get("id")
        self._branch_last_seq[branch_id] = last_seq
        self._branch_last_message_id[branch_id] = last_id
        self.mark_messages_dirty()

    def rebuild_runtime_caches(self) -> None:
        self._branch_last_seq = {}
        self._branch_last_message_id = {}
        self.mark_messages_dirty()
        for branch_id in self.branches:
            self._branch_last_seq.setdefault(branch_id, -1)
            self._branch_last_message_id.setdefault(branch_id, None)
        for m in self.messages:
            branch_id = m.get("branch_id", self.active_branch_id)
            seq = m.get("seq_in_branch", -1)
            if branch_id not in self._branch_last_seq or (isinstance(seq, int) and seq >= self._branch_last_seq[branch_id]):
                self._branch_last_seq[branch_id] = seq if isinstance(seq, int) else self._branch_last_seq.get(branch_id, -1)
                self._branch_last_message_id[branch_id] = m.get("id")
        self._branch_last_seq.setdefault(self.active_branch_id, -1)
        self._branch_last_message_id.setdefault(self.active_branch_id, None)

    def clear_history(self) -> None:
        self.messages.clear()
        self.tool_call_count = 0
        self.turn_count = 0
        self.active_branch_id = f"br_{1:04d}"
        self._branch_counter = 1
        self.branches = {self.active_branch_id: BranchMeta(id=self.active_branch_id, created_at=time.time())}
        self._branch_last_seq = {self.active_branch_id: -1}
        self._branch_last_message_id = {self.active_branch_id: None}
        self.mark_messages_dirty()
        self.updated_at = time.time()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return estimate_tokens(text)

    @staticmethod
    def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
        return count_message_tokens(messages)

    def truncate_messages(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        system_budget = int(max_tokens * self.SYSTEM_BUDGET_RATIO)
        history_budget = int(max_tokens * self.HISTORY_BUDGET_RATIO)

        system_msgs: list[dict[str, Any]] = []
        history_msgs: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                history_msgs.append(msg)

        system_truncated = self._truncate_to_budget(system_msgs, system_budget)
        history_truncated = self._smart_truncate_history(history_msgs, history_budget)

        return system_truncated + history_truncated

    def _truncate_to_budget(
        self,
        messages: list[dict[str, Any]],
        budget: int,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        used = 0
        for msg in messages:
            tokens = self.count_messages_tokens([msg])
            if used + tokens > budget:
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 200:
                    truncated = content[:200] + "\n...[truncated]"
                    truncated_msg = dict(msg)
                    truncated_msg["content"] = truncated
                    result.append(truncated_msg)
                break
            result.append(msg)
            used += tokens
        return result

    def _smart_truncate_history(
        self,
        messages: list[dict[str, Any]],
        budget: int,
    ) -> list[dict[str, Any]]:
        total = self.count_messages_tokens(messages)
        if total <= budget:
            return messages

        preserved_pairs: list[int] = []
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "tool":
                for j in range(i - 1, -1, -1):
                    msg = messages[j]
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "assistant" and isinstance(content, str) and content:
                        preserved_pairs = [j, i]
                        break
                    elif role == "assistant" and msg.get("tool_calls"):
                        preserved_pairs = [j, i]
                        break
                if preserved_pairs:
                    continue

        preserved = 0
        for i in preserved_pairs:
            if i < len(messages):
                preserved += self.count_messages_tokens([messages[i]])

        preserved_indices: set[int] = set(preserved_pairs)
        compressed: list[dict[str, Any]] = []
        compressed.append({
            "role": "user",
            "content": "[Previous conversation context has been compressed to save tokens]",
        })

        used = self.count_messages_tokens(compressed)
        for i in range(len(messages) - 1, -1, -1):
            if i in preserved_indices:
                compressed.append(messages[i])
                used += self.count_messages_tokens([messages[i]])
                continue
            tokens = self.count_messages_tokens([messages[i]])
            if used + tokens <= budget:
                compressed.append(messages[i])
                used += tokens
            else:
                break

        compressed.reverse()
        return compressed

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tool_call_count": self.tool_call_count,
            "turn_count": self.turn_count,
            "metadata": self.metadata,
            "plan_items": self.plan_items,
            "artifacts": self.artifacts,
            "active_branch_id": self.active_branch_id,
            "_branch_counter": self._branch_counter,
            "branches": {k: v.__dict__ for k, v in self.branches.items()},
        }

    def to_meta_dict(self) -> dict[str, Any]:
        """Session metadata only — lightweight, no messages."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tool_call_count": self.tool_call_count,
            "turn_count": self.turn_count,
            "metadata": self.metadata,
            "plan_items": self.plan_items,
            "artifacts": self.artifacts,
            "active_branch_id": self.active_branch_id,
            "_branch_counter": self._branch_counter,
            "branches": {k: v.__dict__ for k, v in self.branches.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: EncreConfig) -> EncreSession:
        session = cls(config)
        session.id = data.get("id", session.id)
        session.messages = data.get("messages", [])
        session.created_at = data.get("created_at", session.created_at)
        session.updated_at = data.get("updated_at", session.updated_at)
        session.tool_call_count = data.get("tool_call_count", 0)
        session.turn_count = data.get("turn_count", 0)
        session.metadata = data.get("metadata", {})
        session.plan_items = data.get("plan_items", [])
        session.artifacts = data.get("artifacts", [])
        session.active_branch_id = data.get("active_branch_id", session.active_branch_id)
        session._branch_counter = data.get("_branch_counter", session._branch_counter)
        branches_raw = data.get("branches", {})
        session.branches = {}
        for k, v in branches_raw.items():
            if isinstance(v, BranchMeta):
                session.branches[k] = v
            elif isinstance(v, dict):
                session.branches[k] = BranchMeta(**v)
        if not session.branches:
            session.branches[session.active_branch_id] = BranchMeta(id=session.active_branch_id, created_at=session.created_at)
        session.rebuild_artifacts_from_messages()
        session.rebuild_runtime_caches()
        return session

    # ── legacy single-file I/O (backwards compat) ──────────────────────

    def save_to_json(self, filepath: str) -> None:
        import json
        from encre.crypto import encrypt
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        try:
            encrypted = encrypt(payload)
        except Exception:
            encrypted = payload
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(encrypted)

    @classmethod
    def load_from_json(cls, filepath: str, config: EncreConfig) -> EncreSession:
        import json
        from encre.crypto import decrypt
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            raise ValueError("Empty session file")
        if not raw.startswith("{"):
            try:
                raw = decrypt(raw)
            except Exception:
                pass
        data = json.loads(raw)
        return cls.from_dict(data, config)

    # ── directory‑based I/O (one turn = one file) ──────────────────────

    def save_to_dir(self, dirpath: str) -> None:
        """Write the full session into '{dirpath}/turn_NNNN.json' files.

        Messages are partitioned by turn: turn 0 = system + first user,
        turnout N = assistant + tool_calls + tool_results + guidance.
        """
        import json
        from encre.crypto import encrypt
        d = pathlib.Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)

        # Group messages by turn
        turns = self._partition_messages_into_turns()
        for idx, msgs in enumerate(turns):
            fpath = d / f"turn_{idx:04d}.json"
            payload = json.dumps(msgs, ensure_ascii=False, separators=(",", ":"))
            try:
                payload = encrypt(payload)
            except Exception:
                pass
            fpath.write_text(payload, encoding="utf-8")

        for p in d.iterdir():
            if not p.name.startswith("turn_") or p.suffix != ".json":
                continue
            try:
                idx = int(p.stem.split("_", 1)[1])
            except Exception:
                continue
            if idx >= len(turns):
                p.unlink(missing_ok=True)

        # Write meta
        self._save_meta_file(d)

    @classmethod
    def load_from_dir(cls, dirpath: str, config: EncreConfig) -> "EncreSession":
        """Load a session from a directory of turn files."""
        import json
        from encre.crypto import decrypt
        d = pathlib.Path(dirpath)
        if not d.is_dir():
            raise ValueError(f"Session directory not found: {dirpath}")

        # Read meta first
        meta_path = d / "meta.json"
        session = cls(config)
        if meta_path.exists():
            raw = meta_path.read_text(encoding="utf-8").strip()
            if raw and not raw.startswith("{"):
                try:
                    raw = decrypt(raw)
                except Exception:
                    pass
            try:
                meta = json.loads(raw)
                session.id = meta.get("id", session.id)
                session.created_at = meta.get("created_at", session.created_at)
                session.updated_at = meta.get("updated_at", session.updated_at)
                session.tool_call_count = meta.get("tool_call_count", 0)
                session.turn_count = meta.get("turn_count", 0)
                session.metadata = meta.get("metadata", {})
                session.plan_items = meta.get("plan_items", [])
                session.artifacts = meta.get("artifacts", [])
                session.active_branch_id = meta.get("active_branch_id", session.active_branch_id)
                session._branch_counter = meta.get("_branch_counter", session._branch_counter)
                branches_raw = meta.get("branches", {})
                session.branches = {}
                for k, v in branches_raw.items():
                    if isinstance(v, BranchMeta):
                        session.branches[k] = v
                    elif isinstance(v, dict):
                        session.branches[k] = BranchMeta(**v)
                if not session.branches:
                    session.branches[session.active_branch_id] = BranchMeta(id=session.active_branch_id, created_at=session.created_at)
            except json.JSONDecodeError:
                pass

        # Read turn files in order
        turn_files = sorted(
            [p for p in d.iterdir() if p.name.startswith("turn_") and p.suffix == ".json"],
            key=lambda p: p.name,
        )
        messages: list[dict[str, Any]] = []
        for fpath in turn_files:
            raw = fpath.read_text(encoding="utf-8").strip()
            if not raw:
                continue
            if not raw.startswith("["):
                try:
                    raw = decrypt(raw)
                except Exception:
                    pass
            try:
                turn_msgs = json.loads(raw)
                if isinstance(turn_msgs, list):
                    messages.extend(turn_msgs)
            except json.JSONDecodeError:
                continue

        session.messages = messages
        # Rebuild artifacts from authoritative message data rather than
        # relying on potentially-buggy streaming artifact creation.
        session.rebuild_artifacts_from_messages()
        session.rebuild_runtime_caches()
        return session

    @staticmethod
    def load_preview(dirpath: str) -> str | None:
        """Read only the first user message for preview (fast)."""
        import json
        from encre.crypto import decrypt
        d = pathlib.Path(dirpath)
        for fpath in sorted(d.iterdir()):
            if not fpath.name.startswith("turn_"):
                continue
            raw = fpath.read_text(encoding="utf-8").strip()
            if raw and not raw.startswith("["):
                try:
                    raw = decrypt(raw)
                except Exception:
                    pass
            try:
                msgs = json.loads(raw)
                if isinstance(msgs, list):
                    for m in msgs:
                        if m.get("role") == "user":
                            c = m.get("content", "")
                            if isinstance(c, str) and c.strip():
                                return c.strip()[:80]
                            elif isinstance(c, list):
                                for b in c:
                                    if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip():
                                        return b["text"].strip()[:80]
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def search_turns(dirpath: str, query_lower: str) -> list[dict[str, Any]]:
        """Search all turn files in a session directory for a query string."""
        import json
        from encre.crypto import decrypt
        d = pathlib.Path(dirpath)
        results: list[dict[str, Any]] = []
        for fpath in sorted(d.iterdir()):
            if not fpath.name.startswith("turn_"):
                continue
            raw = fpath.read_text(encoding="utf-8").strip()
            if raw and not raw.startswith("["):
                try:
                    raw = decrypt(raw)
                except Exception:
                    pass
            try:
                msgs = json.loads(raw)
                if isinstance(msgs, list):
                    for m in msgs:
                        role = m.get("role", "")
                        if role not in ("user", "assistant"):
                            continue
                        content = m.get("content", "")
                        text = ""
                        if isinstance(content, str):
                            text = content
                        elif isinstance(content, list):
                            text = " ".join(
                                b.get("text", "") for b in content
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        if query_lower in text.lower():
                            idx = text.lower().index(query_lower)
                            start = max(0, idx - 40)
                            end = min(len(text), idx + len(query_lower) + 80)
                            results.append({
                                "role": role,
                                "snippet": text[start:end].strip()[:120],
                            })
                            break
            except json.JSONDecodeError:
                continue
        return results

    @staticmethod
    def export_to_markdown(dirpath: str) -> str:
        """Export all turn files in a session directory as a single Markdown string."""
        import json
        from encre.crypto import decrypt
        d = pathlib.Path(dirpath)
        if not d.is_dir():
            raise ValueError(f"Session directory not found: {dirpath}")

        lines: list[str] = []
        lines.append("# Session Export\n")
        lines.append(f"_Session ID: {d.name}_\n")

        turn_files = sorted(
            [p for p in d.iterdir() if p.name.startswith("turn_") and p.suffix == ".json"],
            key=lambda p: p.name,
        )

        turn_num = 0
        for fpath in turn_files:
            raw = fpath.read_text(encoding="utf-8").strip()
            if not raw:
                continue
            if not raw.startswith("["):
                try:
                    raw = decrypt(raw)
                except Exception:
                    pass
            try:
                turn_msgs = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(turn_msgs, list):
                continue

            turn_num += 1
            lines.append(f"\n## Turn {turn_num}\n")

            for m in turn_msgs:
                role = m.get("role", "")
                content = m.get("content", "")

                if role == "system":
                    continue
                elif role == "user":
                    lines.append(f"### User\n\n{EncreSession._extract_text(content)}\n")
                elif role == "assistant":
                    reasoning = m.get("reasoning_content", "")
                    if reasoning:
                        lines.append(f"<details>\n<summary>Thinking</summary>\n\n{reasoning}\n\n</details>\n")
                    tool_calls = m.get("tool_calls")
                    if content:
                        lines.append(f"### Assistant\n\n{EncreSession._extract_text(content)}\n")
                    if tool_calls:
                        for tc in tool_calls:
                            func = tc.get("function", {})
                            tc_name = func.get("name", "unknown")
                            tc_args = func.get("arguments", "{}")
                            lines.append(f"#### 🔧 Tool: `{tc_name}`\n")
                            try:
                                args_parsed = json.loads(tc_args) if isinstance(tc_args, str) else tc_args
                                lines.append("```json\n" + json.dumps(args_parsed, indent=2, ensure_ascii=False) + "\n```\n")
                            except Exception:
                                lines.append(f"```\n{tc_args}\n```\n")
                elif role == "tool":
                    lines.append(f"#### 📋 Tool Result\n\n```\n{EncreSession._extract_text(content)}\n```\n")

        return "".join(lines)

    @staticmethod
    def _extract_text(content: str | list[dict[str, Any]]) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "text":
                        parts.append(b.get("text", ""))
                    elif b.get("type") == "image_url":
                        parts.append("[Image]")
            return "".join(parts)
        return str(content)

    @staticmethod
    def read_meta(dirpath: str) -> dict[str, Any] | None:
        """Read session meta from a turn directory."""
        import json
        from encre.crypto import decrypt
        mp = pathlib.Path(dirpath) / "meta.json"
        if not mp.exists():
            return None
        raw = mp.read_text(encoding="utf-8").strip()
        if raw and not raw.startswith("{"):
            try:
                raw = decrypt(raw)
            except Exception:
                pass
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _save_meta_file(self, dirpath: pathlib.Path) -> None:
        import json
        from encre.crypto import encrypt
        payload = json.dumps(self.to_meta_dict(), ensure_ascii=False, separators=(",", ":"))
        try:
            payload = encrypt(payload)
        except Exception:
            pass
        (dirpath / "meta.json").write_text(payload, encoding="utf-8")

    def _partition_messages_into_turns(self) -> list[list[dict[str, Any]]]:
        """Split self.messages into per‑turn chunks.

        Turn 0: system + first user message.
        Turn N: assistant + tool_results + user guidance messages.
        """
        if self._turn_partition_cache is not None and self._turn_partition_cache[0] == self._messages_version:
            return self._turn_partition_cache[1]
        if not self.messages:
            return []

        turns: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        saw_user = False

        for msg in self.messages:
            role = msg.get("role", "")
            if role == "system" and not saw_user:
                current.append(msg)
                continue
            if role == "user" and not saw_user:
                current.append(msg)
                saw_user = True
                continue
            if role == "assistant":
                # New assistant message → start a new turn
                if current and saw_user:
                    turns.append(current)
                    current = []
            current.append(msg)

        if current:
            turns.append(current)

        self._turn_partition_cache = (self._messages_version, turns)
        return turns


def _guess_image_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/png")
