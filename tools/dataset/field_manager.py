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

import streamlit as st
import ast, json, os
from tools.dataset.i18n import t

def init_field_rules(info):
    """
    Initialize the field order and rules.

    Args:
        info (dict): A dictionary containing field information.

    Returns:
        dict: The field rules stored in the session state.
    """
    # Initialize or repair the field order for the current file
    base_fields = list(info["fields"].keys())
    # Existing order (may come from a previous file)
    fo = st.session_state.get("field_order") or []
    # If there's no overlap at all, or the order is empty, reset to the current fields
    if not fo or not any(f in info["fields"] for f in fo):
        st.session_state["field_order"] = base_fields
    else:
        # Unify the field order: preserve the existing order, then append all missing current fields
        seen = set()
        unified = []
        for k in fo:
            if k not in seen:
                unified.append(k)
                seen.add(k)
        for k in base_fields:
            if k not in seen:
                unified.append(k)
                seen.add(k)
        st.session_state["field_order"] = unified
    # Get the rules from the session state, default to an empty dictionary if not present
    rules = st.session_state.get("rules", {})
    return rules

def _parse_default(s):
    """
    Parse the default value from text to the best-effort Python type.

    Args:
        s (str): The input string representing the default value.

    Returns:
        Any: The parsed value. Returns None if the input is empty, 
             otherwise returns the parsed Python type or the original string.
    """
    if s is None or s == "":
        return None
    # Normalize common JSON literals
    sn = s.strip()
    mapping = {"null": None, "None": None, "true": True, "True": True, "false": False, "False": False}
    if sn in mapping:
        return mapping[sn]
    try:
        return ast.literal_eval(sn)
    except Exception:
        return s

def add_new_field(rules, info):
    """
    Implement the logic for adding a new field via the Streamlit sidebar form.

    Args:
        rules (dict): The current field rules.
        info (dict): A dictionary containing field information.
    """
    # Display the title for adding a new field
    st.sidebar.markdown(t("sidebar.add_new_field_title"))
    # Create two columns in the sidebar
    col1, col2 = st.sidebar.columns(2)
    with col1:
        # Input field for the new field name
        new_field_name = st.text_input(t("sidebar.field_name"), key="new_field_name", placeholder=t("ph.new_field_name"))
    with col2:
        # Input field for the default value of the new field
        new_field_value = st.text_input(t("sidebar.default_value"), key="new_field_value", placeholder=t("ph.default_value"))
    # Check if the add field button is clicked
    if st.sidebar.button(t("sidebar.add_field_btn"), key="add_field"):
        if new_field_name and new_field_name not in rules:
            # Add the new field to the rules
            rules[new_field_name] = new_field_name
            # Initialize the new_fields in the session state if it's not present
            if "new_fields" not in st.session_state:
                st.session_state["new_fields"] = {}
            # Parse the default value to the best-effort Python type
            parsed_default = _parse_default(new_field_value)
            # Store the new field and its default value in the session state
            st.session_state["new_fields"][new_field_name] = parsed_default
            # Initialize the field order if it hasn't been set yet
            if "field_order" not in st.session_state:
                st.session_state["field_order"] = list(info["fields"].keys())
            # Add the new field to the field order
            st.session_state["field_order"].append(new_field_name)
            # Update the rules in the session state
            st.session_state["rules"] = rules

            # Auto-apply the new field to the loaded data and persist it to the file
            try:
                if st.session_state.get('loaded_data') and st.session_state.get('current_file'):
                    updated = []
                    for rec in st.session_state.get('loaded_data', []):
                        if isinstance(rec, dict):
                            if new_field_name not in rec:
                                rec = {**rec, new_field_name: parsed_default}
                        updated.append(rec)
                    src_path = st.session_state['current_file']
                    temp_path = src_path + '.tmp'
                    if src_path.endswith('.jsonl'):
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            for rec in updated:
                                json.dump(rec, f, ensure_ascii=False)
                                f.write('\n')
                    else:
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            json.dump(updated, f, ensure_ascii=False, indent=2)
                    os.replace(temp_path, src_path)
                    st.session_state.loaded_data = updated
                    st.sidebar.success(t("sidebar.add_field_written"))
            except Exception as e:
                st.sidebar.error(t("sidebar.add_field_failed").format(err=str(e)))

            # Rerun the app to reflect the changes
            st.rerun()
        elif new_field_name in rules:
            # Show a warning if the field already exists
            st.sidebar.warning(t("sidebar.field_exists").format(name=new_field_name))
        else:
            # Show a warning if no field name is entered
            st.sidebar.warning(t("sidebar.enter_field_name"))

def manage_fields(info, modal_of):
    """
    Implement interactions for renaming, sorting, and deleting fields.

    Args:
        info (dict): A dictionary containing field information.
        modal_of (function): A function to get the modal information of a field.
    """
    # Display the title for managing existing fields
    st.sidebar.markdown(t("sidebar.manage_fields_title"))
    # Initialize the field order if it hasn't been set yet
    if "field_order" not in st.session_state:
        st.session_state["field_order"] = list(info["fields"].keys())
    # Create a copy of the field information
    all_field_info = dict(info["fields"])
    # Add new fields to the field information
    for field_name in st.session_state.get("new_fields", {}):
        if field_name not in all_field_info:
            default_val = st.session_state["new_fields"].get(field_name)
            tname = type(default_val).__name__ if default_val is not None else "None"
            example = "" if default_val is None else str(default_val)[:80]
            all_field_info[field_name] = {"missing": 0, "types": [tname], "example": example}
    # Get the rules from the session state, default to an empty dictionary if not present
    rules = st.session_state.get("rules", {})
    # Iterate through the field order
    for idx, name in enumerate(st.session_state["field_order"]):
        if name not in all_field_info:
            continue
        meta = all_field_info[name]
        mod = modal_of(name)
        # Create an expander for each field
        with st.sidebar.expander(f"{mod}:{name}", expanded=False):
            # Input field for renaming the field
            new = st.text_input(t("sidebar.rename"), value=name, key=f"n_{name}")
            rules[name] = new
            # Create three columns for the action buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if idx > 0:
                    # Button to move the field up
                    if st.button("↑", key=f"up_{name}", type="secondary", use_container_width=True):
                        st.session_state["field_order"][idx], st.session_state["field_order"][idx-1] = st.session_state["field_order"][idx-1], st.session_state["field_order"][idx]
                        # Remember the field order for each file in the session mapping
                        current_file = st.session_state.get('current_file')
                        if current_file:
                            if 'file_field_orders' not in st.session_state:
                                st.session_state['file_field_orders'] = {}
                            st.session_state['file_field_orders'][current_file] = list(st.session_state["field_order"])
                        # Auto-persist: reorder keys in all records and write back to the file
                        try:
                            if st.session_state.get('loaded_data') and current_file:
                                order = st.session_state["field_order"]
                                updated = []
                                for rec in st.session_state.get('loaded_data', []):
                                    if isinstance(rec, dict):
                                        # Place keys in the specified order first, then the rest in the original order
                                        new_rec = {k: rec[k] for k in order if k in rec}
                                        for k, v in rec.items():
                                            if k not in new_rec:
                                                new_rec[k] = v
                                        updated.append(new_rec)
                                    else:
                                        updated.append(rec)
                                temp_path = current_file + '.tmp'
                                if current_file.endswith('.jsonl'):
                                    with open(temp_path, 'w', encoding='utf-8') as f:
                                        for rec in updated:
                                            json.dump(rec, f, ensure_ascii=False)
                                            f.write('\n')
                                else:
                                    with open(temp_path, 'w', encoding='utf-8') as f:
                                        json.dump(updated, f, ensure_ascii=False, indent=2)
                                os.replace(temp_path, current_file)
                                st.session_state.loaded_data = updated
                                st.sidebar.success(t("sidebar.reorder_success"))
                        except Exception as e:
                            st.sidebar.error(t("sidebar.reorder_failed").format(err=str(e)))
                        st.rerun()
                else:
                    # Disabled button if the field is already at the top
                    st.button("↑", key=f"up_{name}_disabled", disabled=True, use_container_width=True)
            with col2:
                if idx < len(st.session_state["field_order"]) - 1:
                    # Button to move the field down
                    if st.button("↓", key=f"down_{name}", type="secondary", use_container_width=True):
                        st.session_state["field_order"][idx], st.session_state["field_order"][idx+1] = st.session_state["field_order"][idx+1], st.session_state["field_order"][idx]
                        current_file = st.session_state.get('current_file')
                        if current_file:
                            if 'file_field_orders' not in st.session_state:
                                st.session_state['file_field_orders'] = {}
                            st.session_state['file_field_orders'][current_file] = list(st.session_state["field_order"])
                        try:
                            if st.session_state.get('loaded_data') and current_file:
                                order = st.session_state["field_order"]
                                updated = []
                                for rec in st.session_state.get('loaded_data', []):
                                    if isinstance(rec, dict):
                                        new_rec = {k: rec[k] for k in order if k in rec}
                                        for k, v in rec.items():
                                            if k not in new_rec:
                                                new_rec[k] = v
                                        updated.append(new_rec)
                                    else:
                                        updated.append(rec)
                                temp_path = current_file + '.tmp'
                                if current_file.endswith('.jsonl'):
                                    with open(temp_path, 'w', encoding='utf-8') as f:
                                        for rec in updated:
                                            json.dump(rec, f, ensure_ascii=False)
                                            f.write('\n')
                                else:
                                    with open(temp_path, 'w', encoding='utf-8') as f:
                                        json.dump(updated, f, ensure_ascii=False, indent=2)
                                os.replace(temp_path, current_file)
                                st.session_state.loaded_data = updated
                                st.sidebar.success("✅ Successfully written back to file in new order")
                        except Exception as e:
                            st.sidebar.error(f"❌ Failed to write back in order: {str(e)}")
                        st.rerun()
                else:
                    # Disabled button if the field is already at the bottom
                    st.button("↓", key=f"down_{name}_disabled", disabled=True, use_container_width=True)
            with col3:
                # Button to delete the field
                if st.button("🗑️", key=f"d_{name}", type="primary", use_container_width=True):
                    rules[name] = ""
                    st.session_state["field_order"].remove(name)
                    st.session_state["rules"] = rules
                    # Auto-remove this field from all records and persist the changes
                    try:
                        if st.session_state.get('loaded_data') and st.session_state.get('current_file'):
                            updated = []
                            for rec in st.session_state.get('loaded_data', []):
                                if isinstance(rec, dict) and name in rec:
                                    rec = {k: v for k, v in rec.items() if k != name}
                                updated.append(rec)
                            src_path = st.session_state['current_file']
                            temp_path = src_path + '.tmp'
                            if src_path.endswith('.jsonl'):
                                with open(temp_path, 'w', encoding='utf-8') as f:
                                    for rec in updated:
                                        json.dump(rec, f, ensure_ascii=False)
                                        f.write('\n')
                            else:
                                with open(temp_path, 'w', encoding='utf-8') as f:
                                    json.dump(updated, f, ensure_ascii=False, indent=2)
                            os.replace(temp_path, src_path)
                            st.session_state.loaded_data = updated
                            st.sidebar.success(t("sidebar.delete_success").format(name=name))
                    except Exception as e:
                        st.sidebar.error(t("sidebar.delete_failed").format(err=str(e)))
                    st.rerun()

    # Detect renames, auto-apply them to the file, and persist the metadata
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
            updated = []
            for rec in st.session_state.get('loaded_data', []):
                if isinstance(rec, dict):
                    new_rec = dict(rec)
                    for old, new in renames.items():
                        if old in new_rec and new not in new_rec:
                            new_rec[new] = new_rec.pop(old)
                    updated.append(new_rec)
                else:
                    updated.append(rec)
            src_path = st.session_state['current_file']
            temp_path = src_path + '.tmp'
            if src_path.endswith('.jsonl'):
                with open(temp_path, 'w', encoding='utf-8') as f:
                    for rec in updated:
                        json.dump(rec, f, ensure_ascii=False)
                        f.write('\n')
            else:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(updated, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, src_path)
            st.session_state.loaded_data = updated
            # Update the field order entries
            for i, fname in enumerate(st.session_state.get('field_order', [])):
                if fname in renames:
                    st.session_state['field_order'][i] = renames[fname]
            st.sidebar.success(t("sidebar.rename_applied"))
        # Take a snapshot of the current rules
        st.session_state['_last_rules'] = dict(rules)
    except Exception as e:
        st.sidebar.error(t("sidebar.rename_persist_failed").format(err=str(e)))
