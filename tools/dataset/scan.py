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

import ijson
import os, json
from collections import defaultdict
from .utils import natural_sort_key
from .loader import robust_json_load, parse_nested_strings

def collect_json_files(path_input):
    """
    Collect all JSON/JSONL data files in the directory, excluding management files.

    Args:
        path_input (str): The input path, which can be either a file or a directory.

    Returns:
        list: A list of paths to JSON/JSONL files.
    """
    json_files = []
    # Check if the input path is a file
    if os.path.isfile(path_input):
        # If it's a JSON or JSONL file, add it to the list
        if path_input.endswith((".json", ".jsonl")):
            json_files = [path_input]
    else:
        # Traverse the directory tree
        for root, dirs, filenames in os.walk(path_input):
            for f in filenames:
                # Check if the file is a JSON/JSONL file and not a management file
                if f.endswith((".json", ".jsonl")) and f not in ["dataset_info.json", "state.json", "dataset_dict.json"]:
                    full_path = os.path.join(root, f)
                    json_files.append(full_path)
    return json_files

def scan_fields(src_path):
    """
    Enhanced field scanning: Traverse the entire file in a streaming manner to ensure that fields appearing later can also be identified.
    Return info: {fields: {name: {missing, types, example}}, total}

    Meanwhile, load the data for subsequent display and editing.

    Args:
        src_path (str): Path to the JSON/JSONL file.

    Returns:
        tuple: (data: list[dict], info: dict)
    """
    # First, load the data (for subsequent editing).
    data = robust_json_load(src_path)

    # Use streaming scanning for full statistics to avoid missing fields due to only using the front samples.
    stats = defaultdict(lambda: {"cnt": 0, "type": set(), "ex": None})
    total = 0

    def _add_stat(key_path: str, v):
        s = stats[key_path]
        s["cnt"] += 1
        s["type"].add(type(v).__name__ if v is not None else "None")
        if s["ex"] is None:
            if isinstance(v, (list, dict)):
                try:
                    ln = len(v)
                except Exception:
                    ln = "?"
                s["ex"] = f"{type(v).__name__}({ln} items)"
            else:
                try:
                    s["ex"] = str(v)[:120]
                except Exception:
                    s["ex"] = "?"

    def _normalize_role(role_raw: str) -> str:
        r = (role_raw or "").strip().lower()
        if r in ("user", "human", "teacher"):
            return "human"
        if r in ("assistant", "bot", "gpt", "model"):
            return "assistant"
        if r in ("system", "sys"):
            return "system"
        return r or "unknown"

    def _maybe_chat_aggregate(it: dict):
        # Detect role/content pairs and aggregate into synthetic chat fields
        role_key = None
        content_key = None
        for rk in ("role", "from", "speaker", "author"):
            if rk in it:
                role_key = rk
                break
        for ck in ("content", "value", "text", "message", "parts"):
            if ck in it:
                content_key = ck
                break
        if role_key and content_key:
            role = _normalize_role(str(it.get(role_key, "")))
            content_val = it.get(content_key)
            # Flatten common variants
            if isinstance(content_val, list):
                try:
                    content_val = " ".join(x for x in content_val if isinstance(x, str))
                except Exception:
                    pass
            _add_stat(f"[chat].{role}", content_val)

        # Tools/functions (count presence)
        if any(k in it for k in ("tool_calls", "tool_call", "function_call")):
            _add_stat("[chat].tool_call", it.get("tool_calls") or it.get("tool_call") or it.get("function_call"))

    def _walk(obj, prefix: str = ""):
        # Collect stats for current object
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = k if not prefix else f"{prefix}.{k}"
                _add_stat(path, v)
                # Recurse
                if isinstance(v, dict):
                    _walk(v, path)
                elif isinstance(v, list):
                    _add_stat(path, v)
                    if v and all(isinstance(e, dict) for e in v):
                        # Chat-style detection inside lists of dicts
                        for it in v:
                            _maybe_chat_aggregate(it)
                            _walk(it, path + "[]")
                    else:
                        # Primitive list or mixed; still record elements' type examples
                        for e in v[:3]:  # sample a few
                            _add_stat(path + "[]", e)
        elif isinstance(obj, list):
            _add_stat(prefix or "[]", obj)
            for e in obj:
                _walk(e, (prefix or "[]") + "[]")

    def update_stats(rec: dict):
        """
        Update the statistics based on the given record. Recursively traverses nested structures
        and aggregates chat-style role/content messages into synthetic fields under [chat].

        Args:
            rec (dict): A dictionary representing a record.
        """
        nonlocal total
        if not isinstance(rec, dict):
            return
        total += 1
        _walk(rec)

    try:
        if src_path.endswith('.jsonl'):
            # Parse JSONL line by line, extract only keys and types without relying on front samples.
            with open(src_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # Parse nested JSON strings before updating stats
                        obj = parse_nested_strings(obj)
                        update_stats(obj)
                    except Exception:
                        # Try to fix common issues: trim trailing garbage like robust_json_load
                        try:
                            fixed_line = line.rstrip()
                            while fixed_line and fixed_line[-1] not in '}]")':
                                fixed_line = fixed_line[:-1]
                            if fixed_line and len(fixed_line) > 10:
                                obj = json.loads(fixed_line)
                                obj = parse_nested_strings(obj)
                                update_stats(obj)
                                continue
                        except Exception:
                            pass
                        # Skip bad lines to ensure the overall scanning is not interrupted.
                        continue
        else:
            # For standard JSON: Try streaming parsing as an array first; if it's a single object, be compatible.
            with open(src_path, 'r', encoding='utf-8') as f:
                try:
                    for obj in ijson.items(f, 'item'):
                        obj = parse_nested_strings(obj)
                        update_stats(obj)
                except Exception:
                    # If it's not an array structure, try to load it as a whole object.
                    f.seek(0)
                    try:
                        obj = json.load(f)
                        if isinstance(obj, dict):
                            obj = parse_nested_strings(obj)
                            update_stats(obj)
                        elif isinstance(obj, list):
                            for o in obj:
                                o = parse_nested_strings(o)
                                update_stats(o)
                    except Exception:
                        pass
    except Exception:
        # If streaming scanning fails, fall back to using the loaded data for statistics.
        for rec in data:
            update_stats(rec if isinstance(rec, dict) else {})

    # If no samples are obtained from streaming scanning but data has been loaded by robust_json_load, use the loaded data to supplement the statistics.
    if total == 0 and isinstance(data, list) and data:
        for rec in data:
            if isinstance(rec, dict):
                update_stats(rec)

    info = {
        "fields": {
            k: {
                "missing": max(total - v["cnt"], 0),
                "types": sorted(list(str(t) for t in v["type"])),
                "example": str(v["ex"]) if v["ex"] is not None else ""
            }
            for k, v in stats.items()
        },
        "total": total
    }
    return data, info

def sort_json_files(json_files, path_input, current_file=None):
    """
    Sort the list of JSON files using natural sorting.

    Args:
        json_files (list): A list of paths to JSON files.
        path_input (str): The base input path.
        current_file (str, optional): The current file path. Defaults to None.

    Returns:
        tuple: A tuple containing the sorted list of JSON files and the default index.
    """
    sorted_files = sorted(json_files, key=lambda x: natural_sort_key(os.path.relpath(x, path_input)))
    default_index = 0
    # If the current file exists in the sorted list, get its index.
    if current_file and current_file in sorted_files:
        default_index = sorted_files.index(current_file)
    return sorted_files, default_index
