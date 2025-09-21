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

import ast, json, os
import streamlit as st
from tools.dataset.i18n import t

def init_field_rules(info):
    """
    Initialize the field order and rules based on the provided field information.

    Args:
        info (dict): A dictionary containing field information, with a "fields" key mapping to another dictionary.

    Returns:
        dict: The field rules stored in the session state. If not present, returns an empty dictionary.
    """
    # Get the base fields from the input information
    base_fields = list(info["fields"].keys())
    # Retrieve the existing field order from session state; if not present, use an empty list
    existing_order = st.session_state.get("field_order") or []

    # If there's no existing order or no overlap with current fields, reset to base fields
    if not existing_order or not any(field in info["fields"] for field in existing_order):
        st.session_state["field_order"] = base_fields
    else:
        # Unify the field order: preserve existing order and append missing fields
        seen_fields = set()
        unified_order = []
        # Add existing fields in their original order
        for field in existing_order:
            if field not in seen_fields:
                unified_order.append(field)
                seen_fields.add(field)
        # Add missing fields from base fields
        for field in base_fields:
            if field not in seen_fields:
                unified_order.append(field)
                seen_fields.add(field)
        st.session_state["field_order"] = unified_order

    # Get the rules from session state; if not present, default to an empty dictionary
    rules = st.session_state.get("rules", {})
    return rules

def _parse_default(s):
    """
    Parse the input string representing a default value into the best-effort Python type.

    Args:
        s (str): The input string representing the default value.

    Returns:
        Any: The parsed value. Returns None if the input is empty or None.
             Otherwise, returns the parsed Python type or the original string if parsing fails.
    """
    if s is None or s == "":
        return None

    # Normalize common JSON literals
    stripped_input = s.strip()
    literal_mapping = {"null": None, "None": None, "true": True, "True": True, "false": False, "False": False}
    if stripped_input in literal_mapping:
        return literal_mapping[stripped_input]

    try:
        return ast.literal_eval(stripped_input)
    except Exception:
        return stripped_input

def add_new_field(rules, info):
    """
    Implement the logic for adding a new field via the Streamlit sidebar form.

    Args:
        rules (dict): The current field rules.
        info (dict): A dictionary containing field information.
    """
    # Display the title for the "Add New Field" section
    st.markdown(t("sidebar.add_new_field_title"))

    # Create two columns in the sidebar for field name and default value inputs
    col1, col2 = st.columns(2)
    with col1:
        # Input field for the new field name
        new_field_name = st.text_input(t("sidebar.field_name"), key="new_field_name", placeholder=t("ph.new_field_name"))
    with col2:
        # Input field for the default value of the new field
        new_field_value = st.text_input(t("sidebar.default_value"), key="new_field_value", placeholder=t("ph.default_value"))

    # Check if the "Add Field" button is clicked
    if st.button(t("sidebar.add_field_btn"), key="add_field"):
        if new_field_name and new_field_name not in rules:
            # Add the new field to the rules
            rules[new_field_name] = new_field_name

            # Initialize the "new_fields" in session state if not present
            if "new_fields" not in st.session_state:
                st.session_state["new_fields"] = {}

            # Parse the default value to the appropriate Python type
            parsed_default = _parse_default(new_field_value)

            # Store the new field and its default value in session state
            st.session_state["new_fields"][new_field_name] = parsed_default

            # Initialize the field order if not already set
            if "field_order" not in st.session_state:
                st.session_state["field_order"] = list(info["fields"].keys())

            # Add the new field to the field order
            st.session_state["field_order"].append(new_field_name)

            # Update the rules in session state
            st.session_state["rules"] = rules

            # Auto-apply the new field to loaded data and persist to file
            if st.session_state.get('loaded_data') and st.session_state.get('current_file'):
                try:
                    updated_data = []
                    for record in st.session_state.get('loaded_data', []):
                        if isinstance(record, dict):
                            if new_field_name not in record:
                                record = {**record, new_field_name: parsed_default}
                        updated_data.append(record)

                    source_path = st.session_state['current_file']
                    temp_path = source_path + '.tmp'

                    if source_path.endswith('.jsonl'):
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            for record in updated_data:
                                json.dump(record, f, ensure_ascii=False)
                                f.write('\n')
                    else:
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            json.dump(updated_data, f, ensure_ascii=False, indent=2)

                    os.replace(temp_path, source_path)
                    st.session_state.loaded_data = updated_data
                    st.success(t("sidebar.add_field_written"))
                except Exception as e:
                    st.error(t("sidebar.add_field_failed").format(err=str(e)))

            # Rerun the app to reflect the changes
            st.rerun()
        elif new_field_name in rules:
            # Show a warning if the field already exists
            st.sidebar.warning(t("sidebar.field_exists").format(name=new_field_name))
        else:
            # Show a warning if no field name is entered
            st.sidebar.warning(t("sidebar.enter_field_name"))

def manage_fields(info):
    """
    Implement interactions for renaming, sorting, and deleting fields through the Streamlit sidebar.

    Args:
        info (dict): A dictionary containing field information.
    """
    # Display the title for managing existing fields
    st.markdown(t("sidebar.manage_fields_title"))

    # Initialize the field order if not already set
    if "field_order" not in st.session_state:
        st.session_state["field_order"] = list(info["fields"].keys())

    # Create a copy of the field information
    all_field_info = dict(info["fields"])

    # Add new fields to the field information
    for field_name in st.session_state.get("new_fields", {}):
        if field_name not in all_field_info:
            default_val = st.session_state["new_fields"].get(field_name)
            type_name = type(default_val).__name__ if default_val is not None else "None"
            example = "" if default_val is None else str(default_val)[:80]
            all_field_info[field_name] = {"missing": 0, "types": [type_name], "example": example}

    # Get the rules from session state; if not present, default to an empty dictionary
    rules = st.session_state.get("rules", {})

    # Iterate through the field order
    for idx, field_name in enumerate(st.session_state["field_order"]):
        if field_name not in all_field_info:
            continue

        field_meta = all_field_info[field_name]

        # Create an expander for each field
        with st.expander(f"{field_name}", expanded=False):
            # Input field for renaming the field
            new_name = st.text_input(t("sidebar.rename"), value=field_name, key=f"n_{field_name}")
            rules[field_name] = new_name

            # Create three columns for action buttons
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if idx > 0:
                    # Button to move the field up
                    if st.button("↑", key=f"up_{field_name}", type="secondary", use_container_width=True):
                        # Swap the field with the one above
                        st.session_state["field_order"][idx], st.session_state["field_order"][idx-1] = st.session_state["field_order"][idx-1], st.session_state["field_order"][idx]

                        # Remember the field order for the current file
                        current_file = st.session_state.get('current_file')
                        if current_file:
                            if 'file_field_orders' not in st.session_state:
                                st.session_state['file_field_orders'] = {}
                            st.session_state['file_field_orders'][current_file] = list(st.session_state["field_order"])

                        # Auto-persist the reordered data to the file
                        if st.session_state.get('loaded_data') and current_file:
                            try:
                                field_order = st.session_state["field_order"]
                                updated_data = []
                                for record in st.session_state.get('loaded_data', []):
                                    if isinstance(record, dict):
                                        # Reorder keys according to the field order
                                        new_record = {k: record[k] for k in field_order if k in record}
                                        for k, v in record.items():
                                            if k not in new_record:
                                                new_record[k] = v
                                        updated_data.append(new_record)
                                    else:
                                        updated_data.append(record)

                                temp_path = current_file + '.tmp'
                                if current_file.endswith('.jsonl'):
                                    with open(temp_path, 'w', encoding='utf-8') as f:
                                        for record in updated_data:
                                            json.dump(record, f, ensure_ascii=False)
                                            f.write('\n')
                                else:
                                    with open(temp_path, 'w', encoding='utf-8') as f:
                                        json.dump(updated_data, f, ensure_ascii=False, indent=2)

                                os.replace(temp_path, current_file)
                                st.session_state.loaded_data = updated_data
                                st.success(t("sidebar.reorder_success"))
                            except Exception as e:
                                st.error(t("sidebar.reorder_failed").format(err=str(e)))

                        st.rerun()
                else:
                    # Disabled button if the field is already at the top
                    st.button("↑", key=f"up_{field_name}_disabled", disabled=True, use_container_width=True)

            with col2:
                if idx < len(st.session_state["field_order"]) - 1:
                    # Button to move the field down
                    if st.button("↓", key=f"down_{field_name}", type="secondary", use_container_width=True):
                        # Swap the field with the one below
                        st.session_state["field_order"][idx], st.session_state["field_order"][idx+1] = st.session_state["field_order"][idx+1], st.session_state["field_order"][idx]

                        # Remember the field order for the current file
                        current_file = st.session_state.get('current_file')
                        if current_file:
                            if 'file_field_orders' not in st.session_state:
                                st.session_state['file_field_orders'] = {}
                            st.session_state['file_field_orders'][current_file] = list(st.session_state["field_order"])

                        # Auto-persist the reordered data to the file
                        if st.session_state.get('loaded_data') and current_file:
                            try:
                                field_order = st.session_state["field_order"]
                                updated_data = []
                                for record in st.session_state.get('loaded_data', []):
                                    if isinstance(record, dict):
                                        # Reorder keys according to the field order
                                        new_record = {k: record[k] for k in field_order if k in record}
                                        for k, v in record.items():
                                            if k not in new_record:
                                                new_record[k] = v
                                        updated_data.append(new_record)
                                    else:
                                        updated_data.append(record)

                                temp_path = current_file + '.tmp'
                                if current_file.endswith('.jsonl'):
                                    with open(temp_path, 'w', encoding='utf-8') as f:
                                        for record in updated_data:
                                            json.dump(record, f, ensure_ascii=False)
                                            f.write('\n')
                                else:
                                    with open(temp_path, 'w', encoding='utf-8') as f:
                                        json.dump(updated_data, f, ensure_ascii=False, indent=2)

                                os.replace(temp_path, current_file)
                                st.session_state.loaded_data = updated_data
                                st.success(t("sidebar.reorder_success"))
                            except Exception as e:
                                st.error(t("sidebar.reorder_failed").format(err=str(e)))

                        st.rerun()
                else:
                    # Disabled button if the field is already at the bottom
                    st.button("↓", key=f"down_{field_name}_disabled", disabled=True, use_container_width=True)

            with col3:
                # Button to delete the field
                if st.button("🗑️", key=f"d_{field_name}", type="primary", use_container_width=True):
                    rules[field_name] = ""
                    st.session_state["field_order"].remove(field_name)
                    st.session_state["rules"] = rules

                    # Auto-remove the field from all records and persist the changes
                    if st.session_state.get('loaded_data') and st.session_state.get('current_file'):
                        try:
                            updated_data = []
                            for record in st.session_state.get('loaded_data', []):
                                if isinstance(record, dict) and field_name in record:
                                    record = {k: v for k, v in record.items() if k != field_name}
                                updated_data.append(record)

                            source_path = st.session_state['current_file']
                            temp_path = source_path + '.tmp'

                            if source_path.endswith('.jsonl'):
                                with open(temp_path, 'w', encoding='utf-8') as f:
                                    for record in updated_data:
                                        json.dump(record, f, ensure_ascii=False)
                                        f.write('\n')
                            else:
                                with open(temp_path, 'w', encoding='utf-8') as f:
                                    json.dump(updated_data, f, ensure_ascii=False, indent=2)

                            os.replace(temp_path, source_path)
                            st.session_state.loaded_data = updated_data
                            st.success(t("sidebar.delete_success").format(name=field_name))
                        except Exception as e:
                            st.error(t("sidebar.delete_failed").format(err=str(e)))

                    st.rerun()

    # Detect field renames, apply them to the data, and persist the changes
    try:
        prev_rules = st.session_state.get('_last_rules', {}) or {}
        rules = st.session_state.get('rules', {}) or {}
        renames = {}

        for old_name, new_name in rules.items():
            if not new_name:
                continue
            if old_name in prev_rules and prev_rules.get(old_name) != new_name:
                renames[old_name] = new_name

        if renames and st.session_state.get('loaded_data') and st.session_state.get('current_file'):
            updated_data = []
            for record in st.session_state.get('loaded_data', []):
                if isinstance(record, dict):
                    new_record = dict(record)
                    for old, new in renames.items():
                        if old in new_record and new not in new_record:
                            new_record[new] = new_record.pop(old)
                    updated_data.append(new_record)
                else:
                    updated_data.append(record)

            source_path = st.session_state['current_file']
            temp_path = source_path + '.tmp'

            if source_path.endswith('.jsonl'):
                with open(temp_path, 'w', encoding='utf-8') as f:
                    for record in updated_data:
                        json.dump(record, f, ensure_ascii=False)
                        f.write('\n')
            else:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_data, f, ensure_ascii=False, indent=2)

            os.replace(temp_path, source_path)
            st.session_state.loaded_data = updated_data

            # Update the field order entries
            for i, fname in enumerate(st.session_state.get('field_order', [])):
                if fname in renames:
                    st.session_state['field_order'][i] = renames[fname]

            st.success(t("sidebar.rename_applied"))

        # Take a snapshot of the current rules
        st.session_state['_last_rules'] = dict(rules)
    except Exception as e:
        st.error(t("sidebar.rename_persist_failed").format(err=str(e)))
