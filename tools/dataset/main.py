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

import tempfile
import pandas as pd
from data import TEXT_FIELD_KEYS
from typing import Dict, List, Any
from collections import defaultdict
from tools.dataset.arrow import arrow as arrow_to_json
from tools.dataset.utils import natural_sort_key, modal_of
from tools.dataset.preview import get_all_fields, process_preview_data
from tools.dataset.loader import parse_nested_strings, robust_json_load
from tools.dataset.scan import collect_json_files, scan_fields, sort_json_files
from tools.dataset.field_manager import init_field_rules, add_new_field, manage_fields
import streamlit as st, pyarrow as pa, pyarrow.json as paj, json, os, jsonlines, ast, ijson

# Define the chunk size for processing
CHUNK_SIZE = 2000

# Set the page configuration for the Streamlit app
st.set_page_config("PiscesData Control Center", layout="wide")

def dataset(args=None):
    """
    Encapsulate the main process into an interface that can be called externally, 
    facilitating binding in files like manage.py.
    
    Args:
        args: Command line arguments, can be None for standalone usage
    """
    # Initialize pagination state in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'page_size' not in st.session_state:
        st.session_state.page_size = 50
    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = []

    # Step 1: File selection
    # First row: path input
    path_input = st.text_input("Enter file path or directory path", "data_cache")
    # Second row: action buttons (rescan, convert) — tighten spacing
    try:
        c_rescan, c_arrow = st.columns([1, 1], gap="small")
    except TypeError:
        # Fallback for older Streamlit versions without 'gap' parameter
        c_rescan, c_arrow = st.columns([1, 1])
    with c_rescan:
        if st.button("🔄 Rescan Directory"):
            st.cache_data.clear()
            st.rerun()
    with c_arrow:
        if st.button("🧭 Convert Arrow→JSON"):
            if not os.path.exists(path_input):
                st.warning("⚠️ Path does not exist")
            else:
                log_box = st.expander("Conversion Log", expanded=True)
                def _progress_cb(level: str, message: str):
                    if level == "error":
                        log_box.error(message)
                    elif level == "success":
                        log_box.success(message)
                    elif level == "debug":
                        log_box.write(message)
                    else:
                        log_box.info(message)
                with st.spinner("Converting Arrow dataset under the selected path..."):
                    old_cwd = os.getcwd()
                    try:
                        os.chdir(path_input)
                        arrow_to_json(progress_cb=_progress_cb)
                        st.success("✅ Arrow→JSON conversion completed")
                    except Exception as e:
                        st.error(f"❌ Conversion failed: {str(e)}")
                    finally:
                        os.chdir(old_cwd)

    # Collect all JSON/JSONL files in the input path
    json_files = collect_json_files(path_input)
    if not json_files:
        st.warning(f"📁 No json/jsonl files found in {path_input}")
        st.info("Please place your json/jsonl files in the specified directory")
        st.stop()

    # File selection logic
    src_path = None
    if len(json_files) == 1:
        src_path = json_files[0]
        st.info(f"📄 Auto-selected file: {os.path.relpath(src_path, path_input)}")
        if st.session_state.get('current_file') != src_path:
            current_file = st.session_state.get('current_file')
            if current_file and 'field_order' in st.session_state:
                if 'file_field_orders' not in st.session_state:
                    st.session_state['file_field_orders'] = {}
                st.session_state['file_field_orders'][current_file] = st.session_state['field_order']
            for key in ['df', 'loaded_data', 'rules', 'new_fields', 'edited_df']:
                st.session_state.pop(key, None)
            st.session_state['current_file'] = src_path
            if 'file_field_orders' in st.session_state and src_path in st.session_state['file_field_orders']:
                st.session_state['field_order'] = st.session_state['file_field_orders'][src_path]
            st.rerun()
    else:
        sorted_files, default_index = sort_json_files(json_files, path_input, st.session_state.get('current_file'))
        selected_file = st.selectbox(
            "Please select the file to process (sorted A-Z):",
            sorted_files,
            format_func=lambda x: os.path.relpath(x, path_input),
            key="file_selector",
            index=default_index
        )
        if st.session_state.get('current_file') != selected_file:
            current_file = st.session_state.get('current_file')
            if current_file and 'field_order' in st.session_state:
                if 'file_field_orders' not in st.session_state:
                    st.session_state['file_field_orders'] = {}
                st.session_state['file_field_orders'][current_file] = st.session_state['field_order']
            for key in ['df', 'loaded_data', 'rules', 'new_fields', 'edited_df']:
                st.session_state.pop(key, None)
            st.session_state['current_file'] = selected_file
            if 'file_field_orders' in st.session_state and selected_file in st.session_state['file_field_orders']:
                st.session_state['field_order'] = st.session_state['file_field_orders'][selected_file]
            st.rerun()
        src_path = selected_file

    # Step 2: Scan fields
    with st.spinner("🔍 Scanning fields..."):
        try:
            data, info = scan_fields(src_path)
            st.session_state.loaded_data = data
        except Exception as e:
            st.error(f"❌ Failed to load data: {str(e)}")
            info = {"fields": {}, "total": 0}
    total = info["total"]
    st.success(f"Total {total} samples, {len(info['fields'])} fields")

    # Intelligent file size detection
    file_size_mb = os.path.getsize(src_path) / (1024 * 1024)
    if file_size_mb > 100:  # Larger than 100MB
        st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). It is recommended to reduce the number of preview entries.")
    st.divider()

    # Step 3: Sidebar: Field renaming/deletion
    st.sidebar.markdown("### Field Rules")
    rules = init_field_rules(info)
    add_new_field(rules, info)
    manage_fields(info, modal_of)

    # Step 4: Main page: Sample preview + Cell editing + Save + Export + Text replacement
    import re

    # Preview settings
    default_limit = 1000
    load_col1, load_col2 = st.columns([3, 1])
    # Consistent with 1.py: Checkbox and function input in two columns of the same row
    with load_col1:
        load_all = st.checkbox("🚀 Load All Data" if file_size_mb <= 100 else "🚀 Load All Data (Proceed with Caution)", value=False)
    with load_col2:
        # Avoid empty label warning: Provide a non-empty but hidden label
        func_input = st.text_input("Function Input", placeholder="f(x)", key="func_input", label_visibility="collapsed")
    preview_limit = st.number_input("Number of Preview Entries", min_value=10, max_value=100000, value=1000, step=100, disabled=load_all)

    # Data processing and DataFrame generation
    if st.session_state.loaded_data:
        data = st.session_state.loaded_data
        rules = st.session_state.get("rules", {})
        rename_map = {k: v for k, v in rules.items() if v and v != ""}
        base_fields = list(info["fields"].keys())
        new_fields = list(st.session_state.get("new_fields", {}).keys())
        batch_fields = list(st.session_state.get("batch_field_values", {}).keys())
        all_fields = get_all_fields(base_fields, new_fields, batch_fields)
        field_order = st.session_state.get("field_order", all_fields)
        preview_data = data if load_all else data[:preview_limit]
        # Merge defaults: batch values first, then user-defined new field defaults take precedence
        batch_defaults = st.session_state.get("batch_field_values", {}) or {}
        new_defaults = st.session_state.get("new_fields", {}) or {}
        defaults = {**batch_defaults, **new_defaults}
        processed_data = process_preview_data(preview_data, field_order, rules, rename_map, defaults=defaults)
        # Convert to DataFrame for editing
        clean_data = []
        for item in processed_data:
            if isinstance(item, dict):
                clean_item = {}
                for k, v in item.items():
                    if isinstance(v, (dict, list)):
                        clean_item[k] = str(v)
                    else:
                        clean_item[k] = v
                clean_data.append(clean_item)
            else:
                clean_data.append({"data": str(item)})
        df = pd.DataFrame(clean_data)
        st.session_state['df'] = df

    # Display data
    if st.session_state.get('df') is not None:
        df = st.session_state['df']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        base_key = hash(src_path) % 100000
        column_config = {}
        for col in df.columns:
            try:
                col_data = df[col]
                if pd.api.types.is_bool_dtype(col_data):
                    column_config[col] = st.column_config.CheckboxColumn(col)
                elif pd.api.types.is_numeric_dtype(col_data):
                    if not pd.api.types.is_bool_dtype(col_data):
                        column_config[col] = st.column_config.NumberColumn(col, format="%.2f")
                    else:
                        column_config[col] = st.column_config.CheckboxColumn(col)
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    column_config[col] = st.column_config.DateColumn(col)
                else:
                    column_config[col] = st.column_config.TextColumn(col)
            except Exception:
                column_config[col] = st.column_config.TextColumn(col)
        editor_key = f"data_editor_main_{base_key}"
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            key=editor_key,
            column_config=column_config
        )
        st.session_state.df = edited_df

    # Note: The auto-save logic for field addition/renaming/deletion has been moved to field_manager

    # Save modifications
    if st.button("💾 Save Changes", key="save_edited_data"):
        if st.session_state.get('current_file'):
            try:
                src_path = st.session_state['current_file']
                edited_df = st.session_state.get('df')
                if edited_df is not None:
                    save_data = edited_df.to_dict('records')
                else:
                    save_data = df.to_dict('records')
                if not save_data:
                    st.warning("⚠️ No data to save")
                    st.stop()
                if src_path.endswith('.jsonl'):
                    with open(src_path, 'w', encoding='utf-8') as f:
                        for rec in save_data:
                            json.dump(rec, f, ensure_ascii=False)
                            f.write('\n')
                else:
                    with open(src_path, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, ensure_ascii=False, indent=2)
                if 'field_order' in st.session_state:
                    if 'file_field_orders' not in st.session_state:
                        st.session_state['file_field_orders'] = {}
                    st.session_state['file_field_orders'][src_path] = st.session_state['field_order']
                st.session_state.loaded_data = save_data
                st.success("✅ Changes saved to the original file!")
                st.session_state.pop('df', None)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Save failed: {str(e)}")
        else:
            st.warning("⚠️ Please load a file first")

    st.divider()

    # Export options
    col1, col2, col3 = st.columns(3)
    with col1:
        dst_arrow = st.text_input("Output Arrow Path", src_path.replace(".json", ".arrow").replace(".jsonl", ".arrow"))
        if st.button("🚀 Generate Arrow"):
            if st.session_state.loaded_data:
                try:
                    table = pa.Table.from_pylist(st.session_state.loaded_data)
                    import pyarrow.feather as feather
                    feather.write_feather(table, dst_arrow, compression='zstd')
                    st.success(f"✅ Arrow export completed! Total {len(st.session_state.loaded_data)} samples → {dst_arrow}")
                except Exception as e:
                    st.error(f"❌ Arrow conversion error: {str(e)}")
            else:
                st.warning("⚠️ No data to export")

    with col2:
        dst_json = st.text_input("Output JSON Path", src_path.replace(".json","_clean.json").replace(".jsonl","_clean.json"))
        if st.button("📝 Generate JSON"):
            if st.session_state.loaded_data:
                try:
                    with open(dst_json, 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.loaded_data, f, ensure_ascii=False, indent=2)
                    st.success(f"✅ JSON export completed! Total {len(st.session_state.loaded_data)} samples → {dst_json}")
                except Exception as e:
                    st.error(f"❌ JSON conversion error: {str(e)}")
            else:
                st.warning("⚠️ No data to export")

    with col3:
        st.markdown("### 🔥 Replace Original File")
        confirm_replace = st.checkbox("⚠️ Confirm to replace the original file (This operation is irreversible)", value=False)
        if confirm_replace:
            if st.button("💾 Replace Original File Directly", type="primary"):
                try:
                    edited_df = st.session_state.get('df')
                    if edited_df is not None:
                        save_data = edited_df.to_dict('records')
                    else:
                        save_data = df.to_dict('records')
                    if not save_data:
                        st.warning("⚠️ No data to replace")
                        st.stop()
                    temp_path = src_path + ".tmp"
                    if src_path.endswith('.jsonl'):
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            for rec in save_data:
                                json.dump(rec, f, ensure_ascii=False)
                                f.write('\n')
                    else:
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            json.dump(save_data, f, ensure_ascii=False, indent=2)
                    os.replace(temp_path, src_path)
                    st.session_state.loaded_data = save_data
                    st.success(f"✅ Original file replaced! Total {len(save_data)} samples")
                    st.session_state.pop('df', None)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error replacing file: {str(e)}")
                    try:
                        os.remove(temp_path)
                    except:
                        pass

    # Text replacement function (sidebar)
    # Add a divider between "Field Rules" and "Text Content Replacement"
    st.sidebar.divider()
    st.sidebar.markdown("### Text Content Replacement")
    search_text = st.sidebar.text_input("Search Text", placeholder="Enter text to search...")
    replace_text = st.sidebar.text_input("Replace With", placeholder="Enter replacement content...")
    confirm_text_replace = st.sidebar.checkbox("Confirm Replacement", value=False)
    if st.sidebar.button("Apply Replacement") and confirm_text_replace and st.session_state.get('current_file'):
        if not search_text:
            st.sidebar.warning("Please enter search text")
        else:
            try:
                with open(st.session_state.current_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                try:
                    escaped_search = re.escape(search_text)
                    matches = len(re.findall(escaped_search, file_content))
                    if matches > 0:
                        new_content = re.sub(escaped_search, replace_text, file_content)
                        backup_file = st.session_state.current_file + '.backup'
                        with open(backup_file, 'w', encoding='utf-8') as f:
                            f.write(file_content)
                        with open(st.session_state.current_file, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        st.session_state.loaded_data = robust_json_load(st.session_state.current_file)
                        clean_data = []
                        for item in st.session_state.loaded_data:
                            if isinstance(item, dict):
                                clean_item = {}
                                for k, v in item.items():
                                    if isinstance(v, (dict, list)):
                                        clean_item[k] = str(v)
                                    else:
                                        clean_item[k] = v
                                clean_data.append(clean_item)
                            else:
                                clean_data.append({"data": str(item)})
                        df = pd.DataFrame(clean_data)
                        st.session_state['df'] = df
                        st.sidebar.success(f"✅ Text replacement completed! Total {matches} replacements")
                        st.sidebar.info(f"Backup created: {backup_file}")
                        st.rerun()
                    else:
                        st.sidebar.info("No matching text found")
                except re.error:
                    if search_text in file_content:
                        new_content = file_content.replace(search_text, replace_text)
                        backup_file = st.session_state.current_file + '.backup'
                        with open(backup_file, 'w', encoding='utf-8') as f:
                            f.write(file_content)
                        with open(st.session_state.current_file, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        st.session_state.loaded_data = robust_json_load(st.session_state.current_file)
                        clean_data = []
                        for item in st.session_state.loaded_data:
                            if isinstance(item, dict):
                                clean_item = {}
                                for k, v in item.items():
                                    if isinstance(v, (dict, list)):
                                        clean_item[k] = str(v)
                                    else:
                                        clean_item[k] = v
                                clean_data.append(clean_item)
                            else:
                                clean_data.append({"data": str(item)})
                        df = pd.DataFrame(clean_data)
                        st.session_state['df'] = df
                        count = file_content.count(search_text)
                        st.sidebar.success(f"✅ Text replacement completed! Total {count} replacements")
                        st.sidebar.info(f"Backup created: {backup_file}")
                        st.rerun()
                    else:
                        st.sidebar.info("No matching text found")
            except Exception as e:
                st.sidebar.error(f"Error in text replacement: {str(e)}")

    # Copyright information at the bottom of the main page
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:gray;'>© 2025 Wenze Wei · Pisces L1 / Dunimd Project Team. All Rights Reserved.</div>",
        unsafe_allow_html=True,
    )

if __name__ == '__main__':
    dataset()
