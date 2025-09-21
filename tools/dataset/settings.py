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

import os
import yaml
from i18n import t, set_lang
from utils import get_config_manager
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
from func_templates import FunctionTemplateManager

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from utils import get_cache_manager
cache_manager = get_cache_manager()
DATA_CACHE_DIR = cache_manager.get_cache_dir("data_cache")

# Use centralized config manager
_config_manager = get_config_manager(PROJECT_ROOT)

@dataclass
class AppSettings:
    """Top-level application settings.

    Attributes:
        dev_mode (bool): Developer mode switch. Defaults to False.
        func_preview_default (bool): Default state of function preview when opening a file. Defaults to True.
        remember_func_per_file (bool): Whether to remember function input for each file. Defaults to True.
        func_timeout_sec (int): Maximum execution time per function call in seconds. Defaults to 30.
        func_memory_limit_mb (int): Maximum memory usage per function call in megabytes. Defaults to 512.
        func_parallel_enabled (bool): Enable parallel processing for large datasets. Defaults to False.
        extra (Dict[str, Any]): Reserved field for future settings. Defaults to an empty dict.
        default_open_path (str): Default workspace path. Defaults to the data cache directory.
        remember_recent_path (bool): Whether to remember the recent workspace path. Defaults to True.
        language (str): Language setting. Options are 'zh' or 'en'. Defaults to 'zh'.
        number_format (str): Number display format. Defaults to '%.2f'.
        date_format (str): Date display format. Defaults to '%Y-%m-%d'.
        backup_before_save (bool): Whether to create a backup before saving. Defaults to True.
        autosave_enabled (bool): Whether to enable auto-save. Defaults to False.
        autosave_interval_sec (int): Auto-save interval in seconds. Defaults to 120.
        rollback_on_failure (bool): Whether to rollback on failure. Defaults to True.
        expand_panels_by_default (bool): Whether to expand all settings panels by default. Defaults to True.
        expand_sidebar_panels_by_default (bool): Whether to expand sidebar panels by default. Defaults to True.
        expand_template_panels_by_default (bool): Whether to expand function template panels by default. Defaults to True.
    """
    dev_mode: bool = False
    func_preview_default: bool = True
    remember_func_per_file: bool = True
    func_timeout_sec: int = 30
    func_memory_limit_mb: int = 512
    func_parallel_enabled: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)
    default_open_path: str = DATA_CACHE_DIR
    remember_recent_path: bool = True
    language: str = "zh"
    number_format: str = "%.2f"
    date_format: str = "%Y-%m-%d"
    backup_before_save: bool = True
    autosave_enabled: bool = False
    autosave_interval_sec: int = 120
    rollback_on_failure: bool = True
    expand_panels_by_default: bool = True
    expand_sidebar_panels_by_default: bool = True
    expand_template_panels_by_default: bool = True

class SettingsStore:
    """Abstract settings store.

    You can implement concrete backends later, e.g.:
    - In-memory via streamlit session_state
    - Local file (JSON/YAML)
    - Env/CLI overrides

    Attributes:
        namespace (str): Namespace for the settings. Defaults to "dataset".
        settings_path (str): Path to the settings YAML file.
    """

    def __init__(self, namespace: str = "dataset") -> None:
        """Initialize the SettingsStore instance.

        Args:
            namespace (str, optional): Namespace for the settings. Defaults to "dataset".
        """
        self.namespace = namespace
        self.settings_path = os.path.join(os.path.dirname(__file__), "settings.yaml")

    def load(self) -> AppSettings:
        """Load settings from YAML; fallback to defaults if not found or invalid.

        Returns:
            AppSettings: An instance of AppSettings with loaded or default values.
        """
        try:
            settings_data = _config_manager.load_settings("dataset")
            if settings_data:
                return AppSettings(
                    dev_mode=bool(settings_data.get('dev_mode', False)),
                    func_preview_default=bool(settings_data.get('func_preview_default', True)),
                    remember_func_per_file=bool(settings_data.get('remember_func_per_file', True)),
                    func_timeout_sec=int(settings_data.get('func_timeout_sec', 30)),
                    func_memory_limit_mb=int(settings_data.get('func_memory_limit_mb', 512)),
                    func_parallel_enabled=bool(settings_data.get('func_parallel_enabled', False)),
                    extra=dict(settings_data.get('extra') or {}),
                    default_open_path=str(settings_data.get('default_open_path', DATA_CACHE_DIR)),
                    remember_recent_path=bool(settings_data.get('remember_recent_path', True)),
                    language=str(settings_data.get('language', 'zh')),
                    number_format=str(settings_data.get('number_format', '%.2f')),
                    date_format=str(settings_data.get('date_format', '%Y-%m-%d')),
                    backup_before_save=bool(settings_data.get('backup_before_save', True)),
                    autosave_enabled=bool(settings_data.get('autosave_enabled', False)),
                    autosave_interval_sec=int(settings_data.get('autosave_interval_sec', 120)),
                    rollback_on_failure=bool(settings_data.get('rollback_on_failure', True)),
                    expand_panels_by_default=bool(settings_data.get('expand_panels_by_default', True)),
                    expand_sidebar_panels_by_default=bool(settings_data.get('expand_sidebar_panels_by_default', True)),
                    expand_template_panels_by_default=bool(settings_data.get('expand_template_panels_by_default', True)),
                )
        except Exception as e:
            import logging
            logging.warning(f"Failed to load settings from config: {e}")
        
        return AppSettings()

    def save(self, settings: AppSettings) -> None:
        """Persist settings to YAML atomically.

        Args:
            settings (AppSettings): An instance of AppSettings to be saved.
        """
        settings_data = asdict(settings)
        if 'default_open_path' in settings_data and hasattr(settings_data['default_open_path'], 'as_posix'):
            settings_data['default_open_path'] = str(settings_data['default_open_path'])
        _config_manager.save_settings("dataset", settings_data)

    def reset(self) -> None:
        """Reset settings to defaults by removing YAML file if exists."""
        try:
            if os.path.exists(self.settings_path):
                os.remove(self.settings_path)
        except Exception:
            pass


# Convenience helper function to get current settings
def get_settings(store: Optional[SettingsStore] = None) -> AppSettings:
    """Get the current settings.

    Args:
        store (Optional[SettingsStore], optional): An instance of SettingsStore. If None, a new instance will be created. Defaults to None.

    Returns:
        AppSettings: An instance of AppSettings with the current settings.
    """
    import streamlit as st
    store = store or SettingsStore()
    settings = store.load()
    
    _ov = st.session_state.get("settings_override")
    if isinstance(_ov, dict):
        try:
            for key, value in _ov.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
        except Exception:
            pass
    
    return settings

# Convenience helper function to set and save settings
def set_settings(settings: AppSettings, store: Optional[SettingsStore] = None) -> None:
    """Set and save the settings.

    Args:
        settings (AppSettings): An instance of AppSettings to be set.
        store (Optional[SettingsStore], optional): An instance of SettingsStore. If None, a new instance will be created. Defaults to None.
    """
    store = store or SettingsStore()
    store.save(settings)

# Render the Settings full-page subview
def render_settings_page(initial_settings: AppSettings) -> None:
    """Render the Settings full-page subview.

    - Top-left Back button to return
    - Tabs: General / Preview / Data / Developer
    - Actions: Apply (session only), Save (YAML), Reset (defaults), Import/Export YAML

    Args:
        initial_settings (AppSettings): An instance of AppSettings with the initial settings.
    """
    import streamlit as st

    try:
        set_lang(initial_settings.language)
        st.session_state.pop("_language_change_pending", None)
    except Exception:
        pass

    try:
        top_cols = st.columns([1, 100], gap="small")
    except TypeError:
        top_cols = st.columns([1, 100])
    with top_cols[0]:
        if st.button(t("btn.back"), key="btn_back_settings_main_top"):
            st.session_state["show_settings_page"] = False
            st.session_state.pop("settings_override", None)
            st.session_state.pop("settings_refresh", None)
            st.rerun()
    with top_cols[1]:
        st.markdown(
            f"""
            <div style="font-weight: 700; font-size: 24px; line-height: 38px; margin-left: 18px;">
                {t('title.settings')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    t_general, t_preview, t_cache, t_dev = st.tabs([
        t("tab.general"), t("tab.preview"), t("tab.cache"), t("tab.developer")
    ])

    def _apply_setting_change(key, value):
        """Apply a single setting change immediately.

        Args:
            key (str): The name of the setting to change.
            value: The new value of the setting.
        """
        try:
            new_settings = AppSettings(
                dev_mode=st.session_state.get("settings_dev_mode", initial_settings.dev_mode),
                func_preview_default=st.session_state.get("settings_func_preview_default", initial_settings.func_preview_default),
                remember_func_per_file=st.session_state.get("settings_remember_func_per_file", initial_settings.remember_func_per_file),
                func_timeout_sec=int(st.session_state.get("settings_func_timeout_sec", initial_settings.func_timeout_sec)),
                func_memory_limit_mb=int(st.session_state.get("settings_func_memory_limit_mb", initial_settings.func_memory_limit_mb)),
                func_parallel_enabled=st.session_state.get("settings_func_parallel_enabled", initial_settings.func_parallel_enabled),
                extra=initial_settings.extra,
                default_open_path=st.session_state.get("settings_default_open_path", initial_settings.default_open_path),
                remember_recent_path=st.session_state.get("settings_remember_recent_path", initial_settings.remember_recent_path),
                language=st.session_state.get("settings_language", initial_settings.language),
                number_format=st.session_state.get("settings_number_format", initial_settings.number_format),
                date_format=st.session_state.get("settings_date_format", initial_settings.date_format),
                backup_before_save=st.session_state.get("settings_backup_before_save", initial_settings.backup_before_save),
                autosave_enabled=st.session_state.get("settings_autosave_enabled", initial_settings.autosave_enabled),
                autosave_interval_sec=int(st.session_state.get("settings_autosave_interval_sec", initial_settings.autosave_interval_sec)),
                rollback_on_failure=st.session_state.get("settings_rollback_on_failure", initial_settings.rollback_on_failure),
                expand_panels_by_default=st.session_state.get("settings_expand_panels_by_default", initial_settings.expand_panels_by_default),
                expand_sidebar_panels_by_default=st.session_state.get("settings_expand_sidebar_panels_by_default", initial_settings.expand_sidebar_panels_by_default),
                expand_template_panels_by_default=st.session_state.get("settings_expand_template_panels_by_default", initial_settings.expand_template_panels_by_default),
            )
            
            set_settings(new_settings)
            
            for k in list(st.session_state.keys()):
                if k.startswith("func_input_") or k.startswith("func_preview_enabled_"):
                    st.session_state.pop(k, None)
            st.session_state.pop("func_input_global", None)
            st.session_state.pop("func_preview_enabled_global", None)
            
            st.session_state["settings_override"] = asdict(new_settings)
            st.session_state["settings_needs_rerun"] = True
            
            if key == "language" and new_settings.language != initial_settings.language:
                if not st.session_state.get("_language_change_pending", False):
                    st.session_state["_language_change_pending"] = True
                    st.session_state["_last_language"] = new_settings.language
                    set_lang(new_settings.language)
            
        except Exception as e:
            st.error(t("settings.autosave_error").format(err=str(e)))

    with t_general:
        with st.expander(t("cap.workspace_settings"), expanded=initial_settings.expand_panels_by_default):
            default_open_path = st.text_input(
                t("fld.default_open_path"),
                value=initial_settings.default_open_path,
                key="settings_default_open_path",
                help=t("help.default_open_path"),
                on_change=lambda: _apply_setting_change("default_open_path", st.session_state.settings_default_open_path)
            )
            remember_recent_path = st.checkbox(
                t("fld.remember_recent_path"),
                value=initial_settings.remember_recent_path,
                key="settings_remember_recent_path",
                help=t("help.remember_recent_path"),
                on_change=lambda: _apply_setting_change("remember_recent_path", st.session_state.settings_remember_recent_path)
            )

        with st.expander(t("cap.locale_settings"), expanded=initial_settings.expand_panels_by_default):
            language = st.selectbox(
                t("fld.language"),
                options=["zh", "en"],
                index=["zh", "en"].index(initial_settings.language if initial_settings.language in ["zh", "en"] else "zh"),
                key="settings_language",
                format_func=lambda v: {"zh": "中文（简体）", "en": "English"}.get(v, v),
                on_change=lambda: _apply_setting_change("language", st.session_state.settings_language)
            )
            number_format = st.text_input(
                t("fld.number_format"),
                value=initial_settings.number_format,
                key="settings_number_format",
                help=t("help.number_format"),
                on_change=lambda: _apply_setting_change("number_format", st.session_state.settings_number_format)
            )
            date_format = st.text_input(
                t("fld.date_format"),
                value=initial_settings.date_format,
                key="settings_date_format",
                help=t("help.date_format"),
                on_change=lambda: _apply_setting_change("date_format", st.session_state.settings_date_format)
            )

        with st.expander(t("cap.backup_settings"), expanded=initial_settings.expand_panels_by_default):
            backup_before_save = st.checkbox(
                t("fld.backup_before_save"),
                value=initial_settings.backup_before_save,
                key="settings_backup_before_save",
                help=t("help.backup_before_save"),
                on_change=lambda: _apply_setting_change("backup_before_save", st.session_state.settings_backup_before_save)
            )
            autosave_enabled = st.checkbox(
                t("fld.autosave_enabled"),
                value=initial_settings.autosave_enabled,
                key="settings_autosave_enabled",
                help=t("help.autosave_enabled"),
                on_change=lambda: _apply_setting_change("autosave_enabled", st.session_state.settings_autosave_enabled)
            )
            autosave_interval_sec = st.number_input(
                t("fld.autosave_interval_sec"),
                min_value=10,
                max_value=3600,
                value=int(initial_settings.autosave_interval_sec),
                step=10,
                key="settings_autosave_interval_sec",
                disabled=not st.session_state.get("settings_autosave_enabled", initial_settings.autosave_enabled),
                on_change=lambda: _apply_setting_change("autosave_interval_sec", st.session_state.settings_autosave_interval_sec)
            )
            rollback_on_failure = st.checkbox(
                t("fld.rollback_on_failure"),
                value=initial_settings.rollback_on_failure,
                key="settings_rollback_on_failure",
                help=t("help.rollback_on_failure"),
                on_change=lambda: _apply_setting_change("rollback_on_failure", st.session_state.settings_rollback_on_failure)
            )

        with st.expander(t("cap.ui_settings"), expanded=initial_settings.expand_panels_by_default):
            expand_panels_by_default = st.checkbox(
                t("fld.expand_panels_by_default"),
                value=initial_settings.expand_panels_by_default,
                key="settings_expand_panels_by_default",
                help=t("help.expand_panels_by_default"),
                on_change=lambda: _apply_setting_change("expand_panels_by_default", st.session_state.settings_expand_panels_by_default)
            )
            expand_sidebar_panels_by_default = st.checkbox(
                t("fld.expand_sidebar_panels_by_default"),
                value=initial_settings.expand_sidebar_panels_by_default,
                key="settings_expand_sidebar_panels_by_default",
                help=t("help.expand_sidebar_panels_by_default"),
                on_change=lambda: _apply_setting_change("expand_sidebar_panels_by_default", st.session_state.settings_expand_sidebar_panels_by_default)
            )
            expand_template_panels_by_default = st.checkbox(
                t("fld.expand_template_panels_by_default"),
                value=initial_settings.expand_template_panels_by_default,
                key="settings_expand_template_panels_by_default",
                help=t("help.expand_template_panels_by_default"),
                on_change=lambda: _apply_setting_change("expand_template_panels_by_default", st.session_state.settings_expand_template_panels_by_default)
            )

    with t_preview:
        with st.expander(t("cap.preview_settings"), expanded=initial_settings.expand_panels_by_default):
            func_preview_default = st.checkbox(
                t("fld.func_preview_default"),
                value=initial_settings.func_preview_default,
                key="settings_func_preview_default",
                help=t("help.func_preview_default"),
                on_change=lambda: _apply_setting_change("func_preview_default", st.session_state.settings_func_preview_default)
            )
        
        with st.expander(t("cap.function_execution"), expanded=initial_settings.expand_panels_by_default):
            func_timeout_sec = st.number_input(
                t("fld.func_timeout_sec"),
                min_value=1,
                max_value=300,
                value=int(initial_settings.func_timeout_sec),
                step=5,
                key="settings_func_timeout_sec",
                help=t("help.func_timeout_sec"),
                on_change=lambda: _apply_setting_change("func_timeout_sec", st.session_state.settings_func_timeout_sec)
            )
            
            func_memory_limit_mb = st.number_input(
                t("fld.func_memory_limit_mb"),
                min_value=64,
                max_value=4096,
                value=int(initial_settings.func_memory_limit_mb),
                step=64,
                key="settings_func_memory_limit_mb",
                help=t("help.func_memory_limit_mb"),
                on_change=lambda: _apply_setting_change("func_memory_limit_mb", st.session_state.settings_func_memory_limit_mb)
            )
            
            func_parallel_enabled = st.checkbox(
                t("fld.func_parallel_enabled"),
                value=initial_settings.func_parallel_enabled,
                key="settings_func_parallel_enabled",
                help=t("help.func_parallel_enabled"),
                on_change=lambda: _apply_setting_change("func_parallel_enabled", st.session_state.settings_func_parallel_enabled)
            )
            
            remember_func_per_file = st.checkbox(
                t("fld.remember_func_per_file"),
                value=initial_settings.remember_func_per_file,
                key="settings_remember_func_per_file",
                help=t("help.remember_func_per_file"),
                on_change=lambda: _apply_setting_change("remember_func_per_file", st.session_state.settings_remember_func_per_file)
            )
        
        if st.button(f"🔗 {t('title.template_library')}", key="function_templates_btn", help="Manage function templates", use_container_width=True):
            st.session_state.show_template_page = True
            st.session_state.show_settings_page = False
            st.session_state.came_from_settings = True
            st.rerun()

    with t_dev:
        with st.expander(t("cap.developer"), expanded=initial_settings.expand_panels_by_default):
            dev_mode = st.checkbox(
                t("fld.dev_mode"), 
                value=initial_settings.dev_mode, 
                key="settings_dev_mode",
                on_change=lambda: _apply_setting_change("dev_mode", st.session_state.settings_dev_mode)
            )

    with t_cache:
        with st.expander(t("cache.section_stats"), expanded=initial_settings.expand_panels_by_default):
            from utils import get_cache_manager
            import os
            from pathlib import Path
            cache_mgr = get_cache_manager()
            stats = cache_mgr.get_cache_stats()
            cache_dir = Path(str(stats.get("base_dir", "-")))
            cache_root = cache_dir.parent if cache_dir.name == 'cache' else cache_dir
            total_files = 0
            total_size_bytes = 0
            try:
                for root, dirs, files in os.walk(str(cache_root)):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        try:
                            total_files += 1
                            total_size_bytes += os.path.getsize(fpath)
                        except Exception:
                            pass
            except Exception:
                pass

            def _fmt_size(n):
                try:
                    n = float(n)
                    for unit in ["B","KB","MB","GB","TB"]:
                        if n < 1024 or unit == "TB":
                            return f"{n:.2f} {unit}"
                        n /= 1024
                except Exception:
                    return str(n)

            display_path = os.path.normpath(str(cache_root))
            st.markdown(
                f"<div style='color:#6b7280; font-size:12px'>{t('cache.base_dir')}: <code>{display_path}</code></div>",
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            col_a, col_b = st.columns([1,1])
            with col_a:
                st.markdown(f"<div style='text-align:center;font-weight:700;font-size:20px'>{t('cache.total_files')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center;font-size:30px;line-height:34px;color:#6b7280'>{total_files}</div>", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"<div style='text-align:center;font-weight:700;font-size:20px'>{t('cache.total_size')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center;font-size:30px;line-height:34px;color:#6b7280'>{_fmt_size(total_size_bytes)}</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            with st.expander(t('cache.dir_breakdown'), expanded=False):
                from collections import defaultdict
                candidate_cache_dir = os.path.join(str(cache_root), 'cache')
                base_path = candidate_cache_dir if os.path.isdir(candidate_cache_dir) else str(cache_root)
                size_by_dir = defaultdict(int)
                try:
                    for e in os.listdir(base_path):
                        p = os.path.join(base_path, e)
                        if os.path.isdir(p):
                            size_by_dir[e] += 0
                except Exception:
                    pass
                root_bucket = t('cache.root_files')
                size_by_dir[root_bucket] += 0
                for root, dirs, files in os.walk(base_path):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        try:
                            sz = os.path.getsize(fpath)
                        except Exception:
                            sz = 0
                        try:
                            rel = os.path.relpath(root, base_path)
                        except Exception:
                            rel = root
                        parts = rel.split(os.sep) if rel not in ('.', '') else []
                        key = parts[0] if parts else root_bucket
                        size_by_dir[key] += sz
                labels = list(size_by_dir.keys())
                values = list(size_by_dir.values())
                try:
                    import plotly.express as px
                    import pandas as pd
                    df_sizes = pd.DataFrame({'dir': labels, 'size': values})
                    fig = px.pie(df_sizes, values='size', names='dir', hole=0.0)
                    fig.update_layout(height=380, margin=dict(l=0,r=0,b=0,t=0))
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.warning('Plotly is required to display the pie chart. Please install plotly and refresh.')

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button(t("cache.refresh")):
                st.rerun()

        with st.expander(t("cache.section_clean"), expanded=initial_settings.expand_panels_by_default):
            from utils import get_cache_manager
            cache_mgr = get_cache_manager()
            import os, shutil

            st.warning(t("cache.caution_detailed"))

            def _dir_size(path: str) -> int:
                total = 0
                try:
                    for root, dirs, files in os.walk(path):
                        for f in files:
                            fp = os.path.join(root, f)
                            try:
                                total += os.path.getsize(fp)
                            except Exception:
                                pass
                except Exception:
                    pass
                return total

            def _fmt_size(n):
                try:
                    n = float(n)
                    for unit in ["B","KB","MB","GB","TB"]:
                        if n < 1024 or unit == "TB":
                            return f"{n:.2f} {unit}"
                        n /= 1024
                except Exception:
                    return str(n)

            base_root = os.path.normpath(str(cache_mgr.base_cache_dir))
            pisces_root = os.path.normpath(os.path.dirname(base_root))
            ds_dir = os.path.join(base_root, "dataset")

            with st.expander(t("cache.clean_dataset_title"), expanded=False):
                st.caption(f"{t('cache.paths')}: ")
                st.code(ds_dir)
                est_ds = _dir_size(ds_dir) if os.path.isdir(ds_dir) else 0
                st.caption(f"{t('cache.estimated_size')}: {_fmt_size(est_ds)}")
                ds_confirm = st.checkbox(t("cache.confirm_understand") + " (Dataset)", key="cache_confirm_ds")
                if st.button(t("cache.clear_dataset"), disabled=not ds_confirm):
                    try:
                        if os.path.exists(ds_dir):
                            shutil.rmtree(ds_dir, ignore_errors=True)
                        cache_mgr.get_cache_dir("dataset")
                        st.success(t("cache.clear_dataset_done"))
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            with st.expander(t("cache.clean_downloads_title"), expanded=False):
                st.caption(t("cache.clean_downloads_desc"))
                data_dir = os.path.join(base_root, "data_cache")
                datatemp_dir = os.path.join(base_root, "datatemp")
                st.code(os.path.normpath(data_dir))
                st.code(os.path.normpath(datatemp_dir))
                est_data = _dir_size(data_dir) if os.path.isdir(data_dir) else 0
                est_dt = _dir_size(datatemp_dir) if os.path.isdir(datatemp_dir) else 0
                st.caption(f"{t('cache.estimated_size')}: {_fmt_size(est_data + est_dt)}")
                dl_confirm = st.checkbox(t("cache.confirm_understand") + " (downloads)", key="cache_confirm_downloads")
                if st.button(t("cache.clear_downloads"), disabled=not dl_confirm):
                    try:
                        if os.path.isdir(data_dir):
                            shutil.rmtree(data_dir, ignore_errors=True)
                        if os.path.isdir(datatemp_dir):
                            shutil.rmtree(datatemp_dir, ignore_errors=True)
                        st.success(t("cache.clear_downloads_done"))
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            with st.expander(t("cache.clean_all_title"), expanded=False):
                st.caption(f"{t('cache.paths')}: ")
                st.code(pisces_root)
                est_all = _dir_size(pisces_root) if os.path.isdir(pisces_root) else 0
                st.caption(f"{t('cache.estimated_size')}: {_fmt_size(est_all)}")
                all_confirm = st.checkbox(t("cache.confirm_understand") + " (.pisceslx)", key="cache_confirm_all")
                if st.button(t("cache.clear_all"), disabled=not all_confirm):
                    try:
                        if os.path.isdir(pisces_root):
                            shutil.rmtree(pisces_root, ignore_errors=True)
                        os.makedirs(base_root, exist_ok=True)
                        st.success(t("cache.clear_all_done"))
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    if st.session_state.pop("settings_needs_rerun", False):
        st.rerun()

__all__ = [
    'AppSettings',
    'SettingsStore',
    'get_settings',
    'set_settings',
    'render_settings_page',
]
