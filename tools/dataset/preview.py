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

import streamlit as st
from .loader import parse_nested_strings

def get_all_fields(base_fields, new_fields, batch_fields):
    """
    Combine and deduplicate fields from three field lists.

    Args:
        base_fields (list): The base list of fields.
        new_fields (list): The list of new fields.
        batch_fields (list): The list of batch fields.

    Returns:
        list: A list containing all unique fields from the three input lists.
    """
    all_fields = []  # Store all unique fields
    seen = set()  # Store fields that have been seen to avoid duplicates
    for field_list in [base_fields, new_fields, batch_fields]:
        for field in field_list:
            if field and field not in seen:
                all_fields.append(field)
                seen.add(field)
    return all_fields

def process_preview_data(data, field_order, rules, rename_map, defaults=None):
    """
    Process preview data by filtering and renaming fields according to given rules.

    Args:
        data (list): A list of records, where each record is a dictionary.
        field_order (list): A list of fields in the desired order.
        rules (list): A list of fields to be filtered out.
        rename_map (dict): A dictionary mapping old field names to new ones.

    Returns:
        list: A list of processed records.
    """
    processed_data = []  # Store processed records
    defaults = defaults or {}
    for idx, rec in enumerate(data):
        if isinstance(rec, dict):
            rec = parse_nested_strings(rec)  # Parse nested strings in the record
            fil = {}  # Store the filtered and renamed record
            # Add fields in the specified order
            for ordered_field in field_order:
                if ordered_field in rec and ordered_field not in rules:
                    fil[ordered_field] = rec[ordered_field]
                elif ordered_field in rec and ordered_field in rename_map:
                    new_name = rename_map[ordered_field]
                    if new_name:
                        fil[new_name] = rec[ordered_field]
                elif ordered_field not in rec:
                    # Inject default value for missing fields when provided
                    target_name = rename_map.get(ordered_field, ordered_field)
                    if target_name and ordered_field not in rules:
                        if ordered_field in defaults:
                            fil[target_name] = defaults.get(ordered_field)
            # Add fields not in the specified order
            for k, v in rec.items():
                if k not in field_order:
                    fil[k] = v
            processed_data.append(fil)
    return processed_data
