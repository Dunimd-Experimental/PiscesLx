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
from tools.dataset.func import eval_expr_safe
from tools.dataset.arrow import arrow as arrow_to_json
from tools.dataset.utils import natural_sort_key, modal_of
from tools.dataset.preview import get_all_fields, process_preview_data
from tools.dataset.loader import parse_nested_strings, robust_json_load
from tools.dataset.scan import collect_json_files, scan_fields, sort_json_files
from tools.dataset.field_manager import init_field_rules, add_new_field, manage_fields
import streamlit as st, pyarrow as pa, pyarrow.json as paj, json, os, jsonlines, ast, ijson, html, time, shutil
from tools.dataset.settings import get_settings, set_settings, AppSettings, SettingsStore, render_settings_page
from tools.dataset.i18n import t, set_lang

# Define the chunk size for processing
CHUNK_SIZE = 2000

# Apply language before setting page config so title is localized at first paint
try:
    set_lang(get_settings().language)
except Exception:
    pass
# Set the page configuration for the Streamlit app
st.set_page_config(t("app.title"), layout="wide")

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

    # Early: Settings subpage should fully replace main page
    # Load current settings and apply session override to drive Settings UI defaults
    _early_settings = get_settings()
    try:
        set_lang(_early_settings.language)
    except Exception:
        pass
    _ov = st.session_state.get("settings_override")
    if isinstance(_ov, dict):
        try:
            _early_settings.dev_mode = bool(_ov.get("dev_mode", _early_settings.dev_mode))
            _early_settings.func_preview_default = bool(_ov.get("func_preview_default", _early_settings.func_preview_default))
            _early_settings.remember_func_per_file = bool(_ov.get("remember_func_per_file", _early_settings.remember_func_per_file))
        except Exception:
            pass
    # Settings page toggle is handled via a sidebar button (see bottom). Remove top-row button for stability.
    if st.session_state.get("show_settings_page", False):
        render_settings_page(_early_settings)
        return

    # Step 1: File selection
    # First row: path input (use settings default or recent)
    _default_path = None
    try:
        _default_path = (
            st.session_state.get('last_path')
            if _early_settings.remember_recent_path and st.session_state.get('last_path')
            else _early_settings.default_open_path
        )
    except Exception:
        _default_path = _early_settings.default_open_path or "data_cache"
    path_input = st.text_input(t("input.path_label"), _default_path, placeholder=t("ph.path_input"))
    # remember recent path
    if path_input and path_input != st.session_state.get('last_path') and _early_settings.remember_recent_path:
        st.session_state['last_path'] = path_input
    # Second row: action buttons (rescan, convert) — tighten spacing
    try:
        c_rescan, c_arrow = st.columns([1, 1], gap="small")
    except TypeError:
        # Fallback for older Streamlit versions without 'gap' parameter
        c_rescan, c_arrow = st.columns([1, 1])
    with c_rescan:
        if st.button(t("btn.rescan")):
            st.cache_data.clear()
            st.rerun()
    with c_arrow:
        if st.button(t("btn.convert_arrow")):
            if not os.path.exists(path_input):
                st.warning(t("warn.path_missing"))
            else:
                log_box = st.expander(t("exp.convert_log"), expanded=True)
                def _progress_cb(level: str, message: str):
                    if level == "error":
                        log_box.error(message)
                    elif level == "success":
                        log_box.success(message)
                    elif level == "debug":
                        log_box.write(message)
                    else:
                        log_box.info(message)
                with st.spinner(t("spinner.converting_arrow")):
                    old_cwd = os.getcwd()
                    try:
                        os.chdir(path_input)
                        arrow_to_json(progress_cb=_progress_cb)
                        st.success(t("success.arrow_done"))
                    except Exception as e:
                        st.error(t("error.convert_failed").format(err=str(e)))
                    finally:
                        os.chdir(old_cwd)

    # Collect all JSON/JSONL files in the input path
    json_files = collect_json_files(path_input)
    if not json_files:
        st.warning(t("warn.no_files").format(path=path_input))
        st.info(t("info.place_files"))
        st.stop()

    # File selection logic
    src_path = None
    if len(json_files) == 1:
        src_path = json_files[0]
        st.info(t("info.auto_selected").format(file=os.path.relpath(src_path, path_input)))
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
            t("select.file_label"),
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

    # Derive a stable per-file UI key suffix
    base_key = hash(src_path) % 100000

    # Load settings once and react to changes (including manual YAML edits)
    settings = get_settings()
    # Allow session-level override (Apply without saving)
    _ov = st.session_state.get("settings_override")
    if isinstance(_ov, dict):
        try:
            settings.dev_mode = bool(_ov.get("dev_mode", settings.dev_mode))
            settings.func_preview_default = bool(_ov.get("func_preview_default", settings.func_preview_default))
            settings.remember_func_per_file = bool(_ov.get("remember_func_per_file", settings.remember_func_per_file))
        except Exception:
            pass
    settings_sig = (settings.dev_mode, settings.func_preview_default, settings.remember_func_per_file)
    prev_sig = st.session_state.get("settings_sig")
    if prev_sig != settings_sig:
        try:
            # Clear keys so new defaults and scoping take effect immediately
            for k in list(st.session_state.keys()):
                if k.startswith("func_input_") or k.startswith("func_preview_enabled_"):
                    st.session_state.pop(k, None)
            # Also clear possible globals
            st.session_state.pop("func_input_global", None)
            st.session_state.pop("func_preview_enabled_global", None)
        except Exception:
            pass
        st.session_state["settings_sig"] = settings_sig


    # Step 2: Scan fields
    with st.spinner(t("spinner.scan_fields")):
        try:
            data, info = scan_fields(src_path)
            st.session_state.loaded_data = data
        except Exception as e:
            st.error(t("error.load_failed").format(err=str(e)))
            info = {"fields": {}, "total": 0}
    total = info["total"]
    st.success(t("info.scan_summary").format(total=total, fields=len(info['fields'])))

    # Intelligent file size detection
    file_size_mb = os.path.getsize(src_path) / (1024 * 1024)
    if file_size_mb > 100:  # Larger than 100MB
        st.warning(t("warn.large_file").format(mb=file_size_mb))
    st.divider()

    # Step 3: Sidebar: Field renaming/deletion
    st.sidebar.markdown(t("sidebar.field_rules"))
    # Debug: show scanned fields (only in dev mode)
    if settings.dev_mode:
        with st.sidebar.expander(t("sidebar.scanned_fields"), expanded=False):
            try:
                st.write(t("dev.count").format(n=len(info['fields'])))
                preview_fields = list(info["fields"].items())[:100]
                for k, meta in preview_fields:
                    miss = meta.get("missing")
                    types = ",".join(meta.get("types", []))
                    st.write(f"- {k}  [{t('dev.missing')}={miss}, {t('dev.types')}={types}]")
            except Exception:
                pass
    rules = init_field_rules(info)
    add_new_field(rules, info)
    manage_fields(info, modal_of)

    # Step 4: Main page: Sample preview + Cell editing + Save + Export + Text replacement
    import re

    # Preview settings
    default_limit = 1000
    # Top row: Load-all (narrow) | Preview dropdown (wide) | Function input (medium)
    load_col1, load_col2, load_col3 = st.columns([1, 5, 2])
    with load_col1:
        load_all = st.checkbox(t("chk.load_all") if file_size_mb <= 100 else t("chk.load_all_caution"), value=False)
    with load_col2:
        # Middle wide container for f(x) preview (fixed height, scrollable)
        fx_preview_container = st.container()
    with load_col3:
        # Avoid empty label warning: Provide a non-empty but hidden label
        func_key = f"func_input_{base_key}" if settings.remember_func_per_file else "func_input_global"
        func_input = st.text_input(
            t("input.function"),
            placeholder="f(x)",
            key=func_key,
            label_visibility="collapsed",
        )
    preview_limit = st.number_input(t("input.preview_limit"), min_value=10, max_value=100000, value=1000, step=100, disabled=load_all)

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
        # Preview toggle for function result column now controlled by Settings only
        preview_enabled = settings.func_preview_default

        # Bind function box: evaluate expression per record and show results below in a fixed-height scrollable container (no table column)
        fx = st.session_state.get(func_key)
        if preview_enabled and isinstance(fx, str) and fx.strip():
            try:
                results: list[str] = []
                for i, rec in enumerate(preview_data):
                    r = rec if isinstance(rec, dict) else {"data": rec}
                    try:
                        val = eval_expr_safe(fx, r, i)
                    except Exception as e:
                        val = t("fx.per_record_error").format(err=str(e))
                    # Normalize to displayable string
                    try:
                        if isinstance(val, (list, dict)):
                            import json as _json
                            val = _json.dumps(val, ensure_ascii=False)
                        elif hasattr(val, 'isoformat') and callable(getattr(val, 'isoformat')):
                            val = str(val)
                        elif not isinstance(val, (str, int, float, bool)) and val is not None:
                            val = str(val)
                    except Exception:
                        try:
                            val = str(val)
                        except Exception:
                            val = t("fx.unrepresentable")
                    results.append(val)

                # Deduplicate while preserving order
                seen: set[str] = set()
                options: list[str] = []
                for s in results:
                    if s not in seen:
                        seen.add(s)
                        options.append(s)
                if not options:
                    options = [t("dev.no_result")]

                # Render single-line, horizontally scrollable container (no wrapping)
                with fx_preview_container:
                    # Join all unique results into one line with separators
                    content = "  |  ".join(str(x).replace("\n", " ") for x in options)
                    html_content = (
                        "<div style=\"height:40px; max-height:40px; overflow-x:auto; overflow-y:hidden;"
                        " border:1px solid #ddd; border-radius:6px; padding:0 8px;"
                        " font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; line-height:40px;"
                        " white-space: nowrap; word-break: keep-all; background-color: #fafafa;\""
                        f" title=\"{html.escape(content)}\">"
                        f"{html.escape(content)}"
                        "</div>"
                    )
                    st.markdown(html_content, unsafe_allow_html=True)
            except Exception as e:
                st.warning(t("fx.eval_error").format(err=str(e)))
        st.session_state['df'] = df

    # Display data
    if st.session_state.get('df') is not None:
        df = st.session_state['df']
        # Apply date_format (Plan A): format datetime-like columns as strings for display
        try:
            date_fmt = get_settings().date_format
            for col in list(df.columns):
                col_data = df[col]
                # Detect datetime-like content heuristically
                if pd.api.types.is_datetime64_any_dtype(col_data):
                    try:
                        df[col] = pd.to_datetime(col_data, errors='coerce').dt.strftime(date_fmt)
                    except Exception:
                        pass
                else:
                    # Try best-effort conversion for object columns that might be ISO date strings
                    if pd.api.types.is_object_dtype(col_data):
                        try:
                            parsed = pd.to_datetime(col_data, errors='coerce', utc=False)
                            if parsed.notna().any():
                                df[col] = parsed.dt.strftime(date_fmt)
                        except Exception:
                            pass
        except Exception:
            pass
        col1, col2 = st.columns(2)
        with col1:
            st.metric(t("metric.total_rows"), len(df))
        with col2:
            st.metric(t("metric.total_cols"), len(df.columns))
        # base_key already defined for this file's UI elements
        column_config = {}
        for col in df.columns:
            try:
                col_data = df[col]
                if pd.api.types.is_bool_dtype(col_data):
                    column_config[col] = st.column_config.CheckboxColumn(col)
                elif pd.api.types.is_numeric_dtype(col_data):
                    if not pd.api.types.is_bool_dtype(col_data):
                        column_config[col] = st.column_config.NumberColumn(col, format=get_settings().number_format)
                    else:
                        column_config[col] = st.column_config.CheckboxColumn(col)
                # Datetime columns have been formatted as strings above; treat as text for display consistency
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
    if st.button(t("btn.save_changes"), key="save_edited_data"):
        if st.session_state.get('current_file'):
            try:
                src_path = st.session_state['current_file']
                edited_df = st.session_state.get('df')
                if edited_df is not None:
                    save_data = edited_df.to_dict('records')
                else:
                    save_data = df.to_dict('records')
                if not save_data:
                    st.warning(t("warn.no_data_to_save"))
                    st.stop()
                backup_file = None
                settings_now = get_settings()
                try:
                    if settings_now.backup_before_save and os.path.exists(src_path):
                        backup_file = src_path + '.backup'
                        shutil.copyfile(src_path, backup_file)
                except Exception:
                    backup_file = None
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
                st.success(t("success.changes_saved"))
                st.session_state.pop('df', None)
                st.rerun()
            except Exception as e:
                try:
                    if get_settings().rollback_on_failure and 'backup_file' in locals() and backup_file and os.path.exists(backup_file):
                        shutil.copyfile(backup_file, src_path)
                except Exception:
                    pass
                st.error(t("error.save_failed").format(err=str(e)))
        else:
            st.warning(t("warn.load_file_first"))

    st.divider()

    # Export options
    col1, col2, col3 = st.columns(3)
    with col1:
        dst_arrow = st.text_input(t("label.output_arrow"), src_path.replace(".json", ".arrow").replace(".jsonl", ".arrow"))
        if st.button(t("btn.gen_arrow")):
            if st.session_state.loaded_data:
                try:
                    table = pa.Table.from_pylist(st.session_state.loaded_data)
                    import pyarrow.feather as feather
                    feather.write_feather(table, dst_arrow, compression='zstd')
                    st.success(t("success.arrow_export").format(n=len(st.session_state.loaded_data), path=dst_arrow))
                except Exception as e:
                    st.error(t("error.arrow_export").format(err=str(e)))
            else:
                st.warning(t("warn.no_data_to_export"))

    with col2:
        dst_json = st.text_input(t("label.output_json"), src_path.replace(".json","_clean.json").replace(".jsonl","_clean.json"))
        if st.button(t("btn.gen_json")):
            if st.session_state.loaded_data:
                try:
                    with open(dst_json, 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.loaded_data, f, ensure_ascii=False, indent=2)
                    st.success(t("success.json_export").format(n=len(st.session_state.loaded_data), path=dst_json))
                except Exception as e:
                    st.error(t("error.json_export").format(err=str(e)))
            else:
                st.warning(t("warn.no_data_to_export"))

    with col3:
        st.markdown(t("section.replace_original"))
        confirm_replace = st.checkbox(t("chk.confirm_replace_original"), value=False)
        if confirm_replace:
            if st.button(t("btn.replace_original"), type="primary"):
                try:
                    edited_df = st.session_state.get('df')
                    if edited_df is not None:
                        save_data = edited_df.to_dict('records')
                    else:
                        save_data = df.to_dict('records')
                    if not save_data:
                        st.warning(t("warn.no_data_to_export"))
                        st.stop()
                    temp_path = src_path + ".tmp"
                    backup_file = None
                    settings_now = get_settings()
                    try:
                        if settings_now.backup_before_save and os.path.exists(src_path):
                            backup_file = src_path + '.backup'
                            shutil.copyfile(src_path, backup_file)
                    except Exception:
                        backup_file = None
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
                    st.success(t("success.replace_original").format(n=len(save_data)))
                    st.session_state.pop('df', None)
                    st.rerun()
                except Exception as e:
                    try:
                        if get_settings().rollback_on_failure and 'backup_file' in locals() and backup_file and os.path.exists(backup_file):
                            shutil.copyfile(backup_file, src_path)
                    except Exception:
                        pass
                    st.error(t("error.replace_original").format(err=str(e)))
                    try:
                        os.remove(temp_path)
                    except:
                        pass

    # Text replacement function (sidebar)
    # Add a divider between "Field Rules" and "Text Content Replacement"
    st.sidebar.divider()
    st.sidebar.markdown(t("sidebar.text_replace"))
    search_text = st.sidebar.text_input(t("sidebar.search_text"), placeholder=t("ph.search_text"))
    replace_text = st.sidebar.text_input(t("sidebar.replace_with"), placeholder=t("ph.replace_with"))
    if st.sidebar.button(t("sidebar.apply_replace")) and st.session_state.get('current_file'):
        if not search_text:
            st.sidebar.warning(t("sidebar.search_text"))
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
                        st.sidebar.success(t("sidebar.replace_completed").format(n=matches))
                        st.sidebar.info(t("sidebar.backup_created").format(path=backup_file))
                        st.rerun()
                    else:
                        st.sidebar.info(t("sidebar.no_match"))
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
                        st.sidebar.success(t("sidebar.replace_completed").format(n=count))
                        st.sidebar.info(t("sidebar.backup_created").format(path=backup_file))
                        st.rerun()
                    else:
                        st.sidebar.info(t("sidebar.no_match"))
            except Exception as e:
                st.sidebar.error(t("sidebar.error").format(err=str(e)))

    # Copyright information at the bottom of the main page
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:gray;'>© 2025 Wenze Wei · Pisces L1 / Dunimd Project Team. All Rights Reserved.</div>",
        unsafe_allow_html=True,
    )

    # Simple autosave draft based on interval
    try:
        settings_now = get_settings()
        if settings_now.autosave_enabled and st.session_state.get('current_file') and st.session_state.get('df') is not None:
            now = time.time()
            last = st.session_state.get('last_autosave_ts', 0)
            if now - last >= max(10, int(settings_now.autosave_interval_sec)):
                src_path = st.session_state['current_file']
                draft_path = src_path + '.autosave'
                data_to_save = st.session_state['df'].to_dict('records')
                with open(draft_path, 'w', encoding='utf-8') as f:
                    # Save as JSON array draft regardless of original format to keep it simple
                    json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                st.session_state['last_autosave_ts'] = now
    except Exception:
        pass

    # Sidebar settings button at the very bottom (circular, no divider above)
    # Style only the last sidebar button to look circular
    st.sidebar.markdown(
        """
        <style>
        div[data-testid="stSidebar"] div.stButton:last-of-type > button {
            border-radius: 50% !important;
            width: 40px; height: 40px; padding: 0 !important;
            line-height: 40px; text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Toggle main-page settings view
    if st.sidebar.button("⚙", help=t("sidebar.settings_hint"), key="btn_settings_min"):
        st.session_state["show_settings_page"] = True
        st.rerun()

if __name__ == '__main__':
    dataset()
