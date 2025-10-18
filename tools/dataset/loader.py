#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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

import json, ast
import streamlit as st
from typing import Any

CHUNK_SIZE = 1000

def parse_nested_strings(value):
    """
    Recursively parse nested JSON objects within strings.

    Args:
        value (Any): The input value that may contain nested JSON strings.

    Returns:
        Any: The input value with nested JSON strings parsed into Python objects.
    """
    def _parse(obj: Any) -> Any:
        # If the object is a dictionary, recursively parse each value in the dictionary
        if isinstance(obj, dict):
            return {k: _parse(v) for k, v in obj.items()}
        # If the object is a list, recursively parse each element in the list
        elif isinstance(obj, list):
            return [_parse(e) for e in obj]
        # If the object is a string, attempt to parse it as JSON
        elif isinstance(obj, str):
            # Remove leading and trailing whitespace
            stripped_obj = obj.strip()
            # Check if the string might be a JSON object or array
            if stripped_obj and (stripped_obj.startswith('{') or stripped_obj.startswith('[')):
                try:
                    # Try to parse the string as JSON
                    return json.loads(stripped_obj)
                except json.JSONDecodeError:
                    try:
                        # Try to fix escaped quotes and parse as JSON
                        return json.loads(stripped_obj.replace('\\"', '"'))
                    except:
                        try:
                            # Try to evaluate the string as a Python literal
                            return ast.literal_eval(stripped_obj)
                        except:
                            # Return the original string if all parsing attempts fail
                            return obj
        return obj
    return _parse(value)

def robust_json_load(file_path, max_lines=None):
    """
    A versatile and robust JSON/JSONL loader with the following features:
    - Automatically identifies standard JSON (arrays/objects) and JSON Lines formats.
    - Automatically fixes common formatting errors (e.g., trailing garbage, escaping).
    - Supports recursive parsing of nested JSON strings.
    - Ensures field integrity by automatically skipping and warning about exceptions.
    - Returns a list of dictionaries for compatibility with downstream DataFrames.

    Args:
        file_path (str): The path to the JSON/JSONL file.
        max_lines (int, optional): The maximum number of records to load. Defaults to None.

    Returns:
        list: A list of dictionaries containing the parsed JSON data.
    """
    import json, ast
    # List to store the parsed JSON data
    data = []
    # Counter for the number of invalid lines encountered
    invalid_lines = 0

    def parse_nested(obj):
        """
        Recursively parse nested strings into dictionaries or lists.

        Args:
            obj (Any): The input object that may contain nested JSON strings.

        Returns:
            Any: The input object with nested JSON strings parsed into Python objects.
        """
        if isinstance(obj, dict):
            # Recursively parse each value in the dictionary
            return {k: parse_nested(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively parse each element in the list
            return [parse_nested(e) for e in obj]
        elif isinstance(obj, str):
            # Remove leading and trailing whitespace
            stripped_str = obj.strip()
            # Check if the string might be a JSON object or array
            if stripped_str and (stripped_str.startswith('{') or stripped_str.startswith('[')):
                try:
                    # Parse the string as JSON and recursively parse the result
                    return parse_nested(json.loads(stripped_str))
                except Exception:
                    try:
                        # Evaluate the string as a Python literal and recursively parse the result
                        return parse_nested(ast.literal_eval(stripped_str))
                    except Exception:
                        # Return the original string if parsing fails
                        return obj
            return obj
        else:
            # Return non-dict, non-list, and non-string objects as-is
            return obj

    try:
        # Open the file and read its content, removing leading and trailing whitespace
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # First attempt: Try to load the entire content as standard JSON (array/object)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                data = parsed
            elif isinstance(parsed, dict):
                data = [parsed]
            else:
                data = []
            # Recursively parse nested JSON strings in each dictionary record
            data = [parse_nested(rec) for rec in data if isinstance(rec, dict)]
            if max_lines and len(data) > max_lines:
                return data[:max_lines]
            return data
        except Exception:
            pass

        # Second attempt: Try to parse the content line by line (JSONL format)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        for line_num, line in enumerate(lines):
            if max_lines and len(data) >= max_lines:
                break
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    data.append(parse_nested(obj))
            except Exception:
                # Try to fix common escaping and truncation issues
                try:
                    # Remove trailing characters that don't close JSON structures
                    fixed_line = line.rstrip()
                    while fixed_line and fixed_line[-1] not in '}]")':
                        fixed_line = fixed_line[:-1]
                    if fixed_line and len(fixed_line) > 10:
                        obj = json.loads(fixed_line)
                        if isinstance(obj, dict):
                            data.append(parse_nested(obj))
                            st.warning(f"Fixed formatting issue on line {line_num + 1}")
                            continue
                except Exception:
                    pass
                invalid_lines += 1
                if invalid_lines <= 5:
                    st.warning(f"Skipped line {line_num + 1} due to formatting error")
        return data
    except Exception as e:
        st.error(f"JSON parsing failed: {str(e)}")
        return []
