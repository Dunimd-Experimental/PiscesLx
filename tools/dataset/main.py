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

import os
import sys
import html
import tempfile
import pandas as pd
from typing import Dict, List, Any
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from i18n import t, set_lang
from func import eval_expr_safe
from jsonl_creator import JSONLCreator
from arrow import arrow as arrow_to_json
from tools.dataset.utils import natural_sort_key
from preview import get_all_fields, process_preview_data
from loader import parse_nested_strings, robust_json_load
from scan import collect_json_files, scan_fields, sort_json_files
from field_manager import init_field_rules, add_new_field, manage_fields

# Import root utils functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import PiscesLxCoreConfigManagerFacade, get_cache_manager

import streamlit as st, pyarrow as pa, pyarrow.json as paj, json, jsonlines, ast, time, shutil

# Import data module from root
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
try:
    from tools.data import TEXT_FIELD_KEYS
except ImportError:
    TEXT_FIELD_KEYS = ["system", "user", "assistant", "instruction", "input", "output", "question", "answer"]

# Define project root for config manager
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
from template_page import render_template_page
from tools.dataset.settings import get_settings, set_settings, AppSettings, SettingsStore, render_settings_page

# Define the chunk size for processing
CHUNK_SIZE = 2000

# Initialize config manager for centralized file management
config_manager = PiscesLxCoreConfigManagerFacade(PROJECT_ROOT)
st.session_state.config_manager = config_manager

# Cache manager already imported above
cache_manager = get_cache_manager()
st.session_state.cache_manager = cache_manager

# Apply language before setting page config so title is localized at first paint
try:
    current_settings = get_settings()
    # Always apply current language from settings, regardless of session state
    set_lang(current_settings.language)
    # Track current language to detect changes
    st.session_state["_last_language"] = current_settings.language
except Exception:
    pass
# Set the page configuration for the Streamlit app
st.set_page_config(t("app.title"), layout="wide")

# Get project root for config manager
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

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
        # Always apply current language to ensure immediate effect
        set_lang(_early_settings.language)
        st.session_state["_last_language"] = _early_settings.language
    except Exception:
        pass
    
    # Apply settings override immediately
    _ov = st.session_state.get("settings_override")
    if isinstance(_ov, dict):
        try:
            _early_settings.dev_mode = bool(_ov.get("dev_mode", _early_settings.dev_mode))
            _early_settings.func_preview_default = bool(_ov.get("func_preview_default", _early_settings.func_preview_default))
            _early_settings.remember_func_per_file = bool(_ov.get("remember_func_per_file", _early_settings.remember_func_per_file))
            # Force refresh settings cache
            st.session_state["settings_refresh"] = True
        except Exception:
            pass
    # Settings page toggle is handled via a sidebar button (see bottom). Remove top-row button for stability.
    if st.session_state.get("show_settings_page", False):
        render_settings_page(_early_settings)
        return
    
    # Template page toggle
    if st.session_state.get("show_template_page", False):
        render_template_page()
        return

    # Sidebar: Collapsible dataset creation tools
    with st.sidebar.expander(f"### {t('sidebar.dataset_creator')}", expanded=_early_settings.expand_sidebar_panels_by_default):
        # Initialize default path
        _default_path = None
        try:
            # Use last path if configured to remember recent paths and last path exists
            _default_path = (
                st.session_state.get('last_path')
                if _early_settings.remember_recent_path and st.session_state.get('last_path')
                else _early_settings.default_open_path
            )
        except Exception:
            # Fallback to default open path or cache manager data directory
            if _early_settings.default_open_path:
                _default_path = _early_settings.default_open_path
            else:
                cache_manager = get_cache_manager()
                _default_path = cache_manager.get_or_create_cache_dir("data_cache")

        # Initialize new dataset name in session state if not present
        if "new_dataset_name" not in st.session_state:
            st.session_state["new_dataset_name"] = "new_dataset.json"
        # Create text input for new dataset name
        new_dataset_name = st.text_input(t("input.new_dataset_name"), st.session_state["new_dataset_name"], key="new_dataset_name_sidebar")

        # Create placeholder for error messages
        error_placeholder = st.empty()
        # Check if there's a file error message
        if st.session_state.get('file_error_msg'):
            current_time = time.time()
            elapsed = current_time - st.session_state.error_start_time
            # Display error message for 5 seconds
            if elapsed < 5:
                error_placeholder.error(st.session_state.file_error_msg)
                # Schedule auto-refresh if not already scheduled
                if not st.session_state.get('auto_refresh_scheduled'):
                    st.session_state.auto_refresh_scheduled = True
                    time.sleep(5)
                    del st.session_state.file_error_msg
                    del st.session_state.error_start_time
                    del st.session_state.auto_refresh_scheduled
                    st.rerun()
            else:
                # Clean up error-related session state variables
                del st.session_state.file_error_msg
                del st.session_state.error_start_time
                if st.session_state.get('auto_refresh_scheduled'):
                    del st.session_state.auto_refresh_scheduled

        # Create button to trigger dataset creation
        if st.button(f"📝 {t('btn.create_dataset')}", key="create_dataset_btn_sidebar", use_container_width=True):
            try:
                # Initialize JSONL creator with default path
                creator = JSONLCreator(_default_path)
                # Define standard dataset structure
                standard_data = [
                    {
                        "id": "sample_001",
                        "system": "You are a helpful assistant.",
                        "user": "Hello, how can I help you?",
                        "assistant": "I'm ready to assist you with any questions or tasks."
                    }
                ]
                # Construct output path
                output_path = os.path.join(_default_path, new_dataset_name)
                # Check if file already exists
                if os.path.exists(output_path):
                    st.session_state.file_error_msg = f"{t('error.file_exists')}: {new_dataset_name}"
                    st.session_state.error_start_time = time.time()
                    st.rerun()
                else:
                    # Create new dataset file
                    creator.create_jsonl_file(new_dataset_name, standard_data)
                    st.success(f"{t('success.dataset_created')}: {new_dataset_name}")
                    # Reset new dataset name input
                    st.session_state["new_dataset_name"] = "new_dataset.json"
                    # Perform a partial refresh
                    st.rerun()
            except Exception as e:
                # Display error message if dataset creation fails
                st.error(f"{t('error.create_failed')}: {str(e)}")

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
        # Fallback to default open path or cache manager data directory
        if _early_settings.default_open_path:
            _default_path = _early_settings.default_open_path
        else:
                from utils import get_cache_manager
                cache_manager = get_cache_manager()
                _default_path = cache_manager.get_or_create_cache_dir("data_cache")
    path_input = st.text_input(t("input.path_label"), _default_path, placeholder=t("ph.path_input"))
    # remember recent path
    if path_input and path_input != st.session_state.get('last_path') and _early_settings.remember_recent_path:
        st.session_state['last_path'] = path_input
    # Second row: action buttons (rescan, convert) �?tighten spacing
    try:
        c_rescan, c_arrow = st.columns([1, 1], gap="small")
    except TypeError:
        # Fallback for older Streamlit versions without 'gap' parameter
        c_rescan, c_arrow = st.columns([1, 1])
    with c_rescan:
        if st.button(t("btn.rescan")):
            try:
                st.cache_data.clear()
                if "file_cache" in st.session_state:
                    del st.session_state.file_cache
                scan_keys = [k for k in list(st.session_state.keys()) if k.startswith("scan_")]
                for key in scan_keys:
                    del st.session_state[key]
                st.rerun()
            except Exception as rescan_error:
                st.error(f"Re-scan failed: {str(rescan_error)}")
                st.info("Please refresh the page manually")
    with c_arrow:
        if st.button(t("btn.convert_arrow")):
            if not os.path.exists(path_input):
                st.warning(t("warn.path_missing"))
            else:
                log_box = st.expander(t("exp.convert_log"), expanded=True)
                progress_bar = st.progress(0)
                status_text = st.empty()
                conversion_log = []
                def _progress_cb(level: str, message: str):
                    if level == "error":
                        log_box.error(message)
                    elif level == "success":
                        log_box.success(message)
                        progress_bar.progress(1.0)
                        status_text.text("Conversion complete")
                    elif level == "debug":
                        log_box.write(message)
                    else:
                        log_box.info(message)
                        if "Converting" in message:
                            try:
                                parts = message.split(" ")
                                current = int(parts[1])
                                total = int(parts[3])
                                progress = current / max(total, 1)
                                progress_bar.progress(progress)
                                status_text.text(message)
                                conversion_log.append({"step": current, "total": total, "file": "unknown"})
                            except:
                                pass
                with st.spinner(t("spinner.converting_arrow")):
                    old_cwd = os.getcwd()
                    try:
                        os.chdir(path_input)
                        arrow_to_json(progress_cb=_progress_cb)
                        st.success(t("success.arrow_done"))
                        with st.expander("Conversion Details"):
                            st.json(conversion_log)
                        if "arrow_cache" in st.session_state:
                            del st.session_state.arrow_cache
                    except Exception as e:
                        st.error(t("error.convert_failed").format(err=str(e)))
                        if "memory" in str(e).lower():
                            st.warning("Memory issue detected. Try converting smaller batches.")
                        elif "permission" in str(e).lower():
                            st.warning("Permission denied. Check file access rights.")
                    finally:
                        os.chdir(old_cwd)

    # Perform advanced file collection with multi-format support and error recovery
    json_files = []
    try:
        # Collect JSON files using the dedicated function
        json_files = collect_json_files(path_input)
        # Remove duplicates and sort the file list
        json_files = sorted(set(json_files))
    except Exception as e:
        st.error(f"File collection failed: {str(e)}")
        
        # Use fallback pattern matching to discover files if the primary method fails
        try:
            import glob
            # Define file patterns for various data formats
            patterns = [
                "**/*.json", "**/*.jsonl", "**/*.ndjson",
                "**/*.csv", "**/*.tsv", "**/*.parquet"
            ]
            
            # Iterate through patterns to find matching files
            for pattern in patterns:
                files = glob.glob(os.path.join(path_input, pattern), recursive=True)
                json_files.extend(files)
            # Remove duplicates and sort the combined file list
            json_files = sorted(set(json_files))
        except Exception as fallback_error:
            st.error(f"Fallback file discovery failed: {str(fallback_error)}")
            json_files = []
    
    # Check if no files were found
    if not json_files:
        # Display a warning and information message in the main area
        st.warning(t("warn.no_files").format(path=path_input))
        st.info(t("info.place_files"))
        
        # Auto-create directory if missing
        if not os.path.exists(path_input):
            os.makedirs(path_input, exist_ok=True)

        # Stop further processing if no files are available
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

    # Advanced settings management with validation and recovery
    settings = get_settings()
    
    # Enhanced settings validation and repair
    try:
        # Validate settings integrity
        if not hasattr(settings, 'dev_mode'):
            settings.dev_mode = False
        if not hasattr(settings, 'func_preview_default'):
            settings.func_preview_default = True
        if not hasattr(settings, 'remember_func_per_file'):
            settings.remember_func_per_file = False
            
        # Allow session-level override with validation
        _ov = st.session_state.get("settings_override")
        if isinstance(_ov, dict):
            try:
                settings.dev_mode = bool(_ov.get("dev_mode", settings.dev_mode))
                settings.func_preview_default = bool(_ov.get("func_preview_default", settings.func_preview_default))
                settings.remember_func_per_file = bool(_ov.get("remember_func_per_file", settings.remember_func_per_file))
            except Exception as override_error:
                st.error(f"Settings override failed: {str(override_error)}")
                
    except Exception as settings_error:
        st.error(f"Settings system error: {str(settings_error)}")
        
        # Emergency settings reset
        try:
            default_settings = AppSettings()
            settings = default_settings
            st.warning("Settings reset to defaults due to corruption")
        except Exception as reset_error:
            st.error(f"Settings reset failed: {str(reset_error)}")
            settings = type('EmergencySettings', (), {
                'dev_mode': False,
                'func_preview_default': True,
                'remember_func_per_file': False
            })()
    
    settings_sig = (settings.dev_mode, settings.func_preview_default, settings.remember_func_per_file)
    prev_sig = st.session_state.get("settings_sig")
    if prev_sig != settings_sig:
        try:
            # Enhanced cache clearing with error handling
            keys_to_clear = [key for key in list(st.session_state.keys()) 
                           if key.startswith(("func_input_", "func_preview_enabled_"))]
            for key in keys_to_clear:
                try:
                    st.session_state.pop(key, None)
                except KeyError:
                    pass  # Already removed
            # Also clear possible globals
            st.session_state.pop("func_input_global", None)
            st.session_state.pop("func_preview_enabled_global", None)
        except Exception as clear_error:
            st.error(f"Cache clearing failed: {str(clear_error)}")
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
    file_size_mb = os.path.getsize(src_path) / (1024 * 1024) if src_path else 0
    if file_size_mb > 100:  # Larger than 100MB
        st.warning(t("warn.large_file").format(mb=file_size_mb))
    st.divider()

    # Developer diagnostics panel (only in Developer Mode), placed BELOW the Dataset Creator
    if settings.dev_mode:
        with st.sidebar.expander(f"{t('dev.panel_title')}{t('dev.debug_suffix')}", expanded=_early_settings.expand_sidebar_panels_by_default):
            try:
                if not src_path:
                    st.info(t('dev.not_loaded'))
                else:
                    # File and size
                    try:
                        rel_name = os.path.relpath(src_path, path_input) if path_input else os.path.basename(src_path)
                    except Exception:
                        rel_name = os.path.basename(src_path)
                    size_mb = (os.path.getsize(src_path) / (1024*1024)) if os.path.exists(src_path) else 0.0
                    st.caption(f"{t('dev.file')}: {rel_name}")
                    st.caption(f"{t('dev.size_mb')}: {size_mb:.2f}")

                    # Compute metrics from scanned data
                    sample = st.session_state.loaded_data if st.session_state.get('loaded_data') else []
                    n = len(sample)
                    # Avg fields/record
                    try:
                        if n > 0:
                            avg_fields = sum(len(r) if isinstance(r, dict) else 1 for r in sample[:min(n, 2000)]) / min(n, 2000)
                        else:
                            avg_fields = float(len(info.get('fields') or {}))
                    except Exception:
                        avg_fields = float(len(info.get('fields') or {}))

                    # Duplicate ratio (on a sample up to 2000)
                    try:
                        k = min(n, 2000)
                        if k > 0:
                            import json as _json
                            ser = [ _json.dumps(sample[i], ensure_ascii=False, sort_keys=True) if isinstance(sample[i], (dict, list)) else str(sample[i]) for i in range(k) ]
                            unique = len(set(ser))
                            dup_ratio = 0.0 if k == 0 else max(0.0, 1.0 - unique/float(k))
                        else:
                            dup_ratio = 0.0
                    except Exception:
                        dup_ratio = 0.0

                    # Missing ratio using scan info if available
                    try:
                        fields_meta = info.get('fields') or {}
                        if total and fields_meta:
                            missing_sum = 0
                            for _, meta in fields_meta.items():
                                missing_sum += int(meta.get('missing', 0))
                            denom = max(1, total * max(1, len(fields_meta)))
                            missing_ratio = min(1.0, max(0.0, missing_sum / float(denom)))
                        else:
                            missing_ratio = 0.0
                    except Exception:
                        missing_ratio = 0.0

                    # Noise/Quality heuristic
                    noise_score = min(100.0, max(0.0, 100.0 * (0.5 * dup_ratio + 0.5 * missing_ratio)))
                    quality_score = max(0.0, 100.0 - noise_score)

                    # Render compact metrics
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric(t('dev.records'), total)
                    with m2:
                        st.metric(t('dev.fields'), f"{avg_fields:.1f}")
                    with m3:
                        st.metric(t('dev.dup_ratio'), f"{dup_ratio*100:.1f}%")
                    with m4:
                        st.metric(t('dev.missing_ratio'), f"{missing_ratio*100:.1f}%")

                    q1, q2 = st.columns(2)
                    with q1:
                        st.metric(t('dev.noise_score'), f"{noise_score:.1f}")
                    with q2:
                        st.metric(t('dev.quality_score'), f"{quality_score:.1f}")

                    # Text field analysis (more precise diagnostics)
                    try:
                        # Determine candidate text fields
                        fields_meta = info.get('fields') or {}
                        candidate_text_fields = []
                        for fname, meta in fields_meta.items():
                            types = set(meta.get('types') or [])
                            if ('str' in types) or (fname.lower() in [k.lower() for k in (TEXT_FIELD_KEYS or [])]):
                                candidate_text_fields.append(fname)
                        # Limit to top few text-like fields for quick insight
                        candidate_text_fields = candidate_text_fields[:5]
                        if candidate_text_fields and n > 0:
                            st.markdown(f"**{t('dev.text_fields')}**: {', '.join(candidate_text_fields)}")
                            # Sample up to 2000 records
                            k = min(n, 2000)
                            import numpy as _np
                            for fname in candidate_text_fields:
                                vals = []
                                empty_count = 0
                                total_count = 0
                                for i in range(k):
                                    rec = sample[i]
                                    s = ''
                                    if isinstance(rec, dict):
                                        v = rec.get(fname)
                                        s = str(v) if v is not None else ''
                                    else:
                                        s = str(rec)
                                    total_count += 1
                                    if not s or s.strip() == '':
                                        empty_count += 1
                                    else:
                                        vals.append(len(s))
                                empty_ratio = (empty_count / max(1, total_count))
                                avg_len = float(_np.mean(vals)) if vals else 0.0
                                p95_len = float(_np.percentile(vals, 95)) if vals else 0.0
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.metric(t('dev.empty_text_ratio'), f"{empty_ratio*100:.1f}%", help=f"{fname}")
                                with c2:
                                    st.metric(t('dev.avg_len'), f"{avg_len:.1f}", help=f"{fname}")
                                with c3:
                                    st.metric(t('dev.p95_len'), f"{p95_len:.0f}", help=f"{fname}")
                    except Exception:
                        pass

                    # Top missing fields listing
                    try:
                        fields_meta = info.get('fields') or {}
                        if total and fields_meta:
                            miss_list = []
                            for fname, meta in fields_meta.items():
                                miss = int(meta.get('missing', 0))
                                miss_list.append((fname, miss / float(total)))
                            miss_list.sort(key=lambda x: x[1], reverse=True)
                            top_miss = miss_list[:5]
                            if top_miss:
                                st.markdown(f"**{t('dev.top_missing_fields')}**")
                                for fname, ratio in top_miss:
                                    st.caption(f"- {fname}: {ratio*100:.1f}%")
                    except Exception:
                        pass

                    # Note on sampling
                    try:
                        st.caption(t('dev.analysis_note'))
                    except Exception:
                        pass
            except Exception:
                pass

    # Step 3: Sidebar: Field renaming/deletion
    with st.sidebar.expander(t("sidebar.field_rules"), expanded=_early_settings.expand_sidebar_panels_by_default):
        # Debug: show scanned fields (only in dev mode)
        if settings.dev_mode:
            with st.expander(t("sidebar.scanned_fields"), expanded=False):
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
        manage_fields(info)

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
    # Dynamic preview limit based on total records to avoid confusing large defaults
    _max_limit = max(1, int(total))
    _default_limit = 1000 if _max_limit >= 1000 else _max_limit
    _step = max(1, min(100, _max_limit // 10 if _max_limit >= 10 else 1))
    preview_limit = st.number_input(
        t("input.preview_limit"),
        min_value=1,
        max_value=_max_limit,
        value=int(_default_limit),
        step=int(_step),
        disabled=load_all,
    )

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
        # Pagination: use preview_limit as page size when not loading all
        if load_all:
            preview_data = data
            # Reset to first page when switching to load_all for consistency
            st.session_state.current_page = 1
            st.session_state.page_size = int(preview_limit)
        else:
            n = len(data)
            page_size = int(preview_limit)
            st.session_state.page_size = page_size
            # total pages (integer ceil without math import)
            total_pages = max(1, (n + page_size - 1) // page_size)
            current_page = int(st.session_state.get('current_page', 1))
            # Clamp current page in case data size changed
            current_page = min(max(1, current_page), total_pages)
            # Pager controls (only if more than one page)
            effective_page_size = min(page_size, n) if n > 0 else 0
            if total_pages > 1:
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                try:
                    p1, p2, p3, p4, p5 = st.columns([1, 1, 3, 1, 1], gap="small")
                except TypeError:
                    p1, p2, p3, p4, p5 = st.columns([1, 1, 3, 1, 1])
                with p1:
                    if st.button(t("pager.first"), key="pager_first", disabled=(current_page == 1)):
                        current_page = 1
                with p2:
                    if st.button(t("pager.prev"), key="pager_prev", disabled=(current_page == 1)):
                        current_page = max(1, current_page - 1)
                with p4:
                    if st.button(t("pager.next"), key="pager_next", disabled=(current_page == total_pages)):
                        current_page = min(total_pages, current_page + 1)
                with p5:
                    if st.button(t("pager.last"), key="pager_last", disabled=(current_page == total_pages)):
                        current_page = total_pages
                with p3:
                    st.markdown(t("pager.page_info").format(
                        current=current_page,
                        total=total_pages,
                        count=n,
                        per_page=effective_page_size
                    ), unsafe_allow_html=True)
            st.session_state.current_page = current_page
            start = (current_page - 1) * page_size
            end = start + page_size
            preview_data = data[start:end]
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
        # Default: allow adding/deleting rows (dynamic rows) when supported by Streamlit
        num_rows_param = "dynamic"
        try:
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                key=editor_key,
                column_config=column_config,
                num_rows=num_rows_param,
            )
        except TypeError:
            # For older Streamlit versions without num_rows param, fall back gracefully
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                key=editor_key,
                column_config=column_config,
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
                        # 备份文件和原文件在同一目录
                        src_path_obj = Path(src_path)
                        backup_file = str(src_path_obj.parent / (src_path_obj.stem + '.backup'))
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
        dst_arrow = st.text_input(t("label.output_arrow"), src_path.replace(".json", ".arrow").replace(".jsonl", ".arrow") if src_path else "output.arrow")
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
        dst_json = st.text_input(t("label.output_json"), src_path.replace(".json","_clean.json").replace(".jsonl","_clean.json") if src_path else "output_clean.json")
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

    # Use col3 to create UI elements for JSONL file creation
    with col3:
        # Create a text input field for the output JSONL file path, 
        # with a default value generated by replacing the source file extension
        dst_jsonl = st.text_input(t("label.output_jsonl"), src_path.replace(".json","_new.jsonl").replace(".jsonl","_new.jsonl") if src_path else "output_new.jsonl")
        
        # Check if the "Create JSONL" button is clicked
        if st.button(t("btn.create_jsonl")):
            try:
                # Initialize a JSONLCreator instance
                creator = JSONLCreator()
                
                # If data is loaded, create a JSONL file from the loaded data
                if st.session_state.loaded_data:
                    creator.create_from_data(st.session_state.loaded_data, dst_jsonl)
                    st.success(t("success.jsonl_created").format(n=len(st.session_state.loaded_data), path=dst_jsonl))
                else:
                    # If no data is loaded, let the user select a template type
                    template_type = st.selectbox(t("select.template"), [t("template.multimodal"), t("template.text"), t("template.conversation")])
                    # Map the selected template display name to its internal name
                    template_map = {
                        t("template.multimodal"): "multimodal",
                        t("template.text"): "text", 
                        t("template.conversation"): "conversation"
                    }
                    # Create a JSONL file from the selected template
                    creator.create_from_template(template_map[template_type], dst_jsonl)
                    st.success(t("success.jsonl_template_created").format(path=dst_jsonl))
            except Exception as e:
                # Display an error message if the JSONL creation fails
                st.error(t("error.jsonl_create_failed").format(err=str(e)))

    with st.sidebar.expander(t("sidebar.text_replace"), expanded=_early_settings.expand_sidebar_panels_by_default):
        search_text = st.text_input(t("sidebar.search_text"), placeholder=t("ph.search_text"))
        replace_text = st.text_input(t("sidebar.replace_with"), placeholder=t("ph.replace_with"))
        if st.button(t("sidebar.apply_replace"), use_container_width=True) and st.session_state.get('current_file'):
            if not search_text:
                st.warning(t("sidebar.search_text"))
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
                            st.success(t("sidebar.replace_completed").format(n=matches))
                            st.info(t("sidebar.backup_created").format(path=backup_file))
                            st.rerun()
                        else:
                            st.info(t("sidebar.no_match"))
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
                            st.success(t("sidebar.replace_completed").format(n=count))
                            st.info(t("sidebar.backup_created").format(path=backup_file))
                            st.rerun()
                        else:
                            st.info(t("sidebar.no_match"))
                except Exception as e:
                    st.error(t("sidebar.error").format(err=str(e)))

    # Copyright information at the bottom of the main page
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:gray;'>© 2025 Wenze Wei · PiscesL1 / Dunimd Project Team. All Rights Reserved.</div>",
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

    # Fixed settings button at the bottom of sidebar using container
    with st.sidebar.container():
        st.sidebar.markdown(
            """
            <style>
            div[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] > div:last-child {
                position: fixed;
                bottom: 20px;
                left: 20px;
                z-index: 1000;
            }
            div[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] > div:last-child button {
                border-radius: 50% !important;
                width: 40px; height: 40px; padding: 0 !important;
                line-height: 40px; text-align: center;
                background-color: #1f77b4;
                color: white;
                border: none;
                cursor: pointer;
                margin-right: 5px;
            }
            div[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] > div:last-child button:hover {
                background-color: #0e5a8a;
            }
            div[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] > div:last-child {
                display: flex;
                gap: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        # Create a row for settings button only
        button_col = st.sidebar.columns(1)[0]
        
        with button_col:
            # Toggle main-page settings view
            if st.button("�?, help=t("sidebar.settings_hint"), key="btn_settings_min"):
                st.session_state["show_settings_page"] = True
                st.rerun()

if __name__ == '__main__':
    dataset()
