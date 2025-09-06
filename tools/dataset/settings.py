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

from __future__ import annotations

import os
import yaml
from .i18n import t, set_lang
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict

# Project constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATA_CACHE_DIR = os.path.join(PROJECT_ROOT, "data_cache")

@dataclass
class AppSettings:
    """Top-level app settings (skeleton).

    Attributes:
        dev_mode (bool): Developer mode switch. Defaults to False.
        func_preview_default (bool): Default state of function preview when opening a file. Defaults to True.
        remember_func_per_file (bool): Whether to remember function input for each file. Defaults to True.
        extra (Dict[str, Any]): Reserved field for future settings. Defaults to an empty dict.
    """
    # Developer options
    dev_mode: bool = False

    # Function preview options
    func_preview_default: bool = True  # default on/off when opening a file
    remember_func_per_file: bool = True  # whether to remember func input per file

    # Reserved room for future settings
    extra: Dict[str, Any] = field(default_factory=dict)

    # Workspace & recent path
    default_open_path: str = DATA_CACHE_DIR
    remember_recent_path: bool = True

    # Locale
    language: str = "zh"  # zh / en
    number_format: str = "%.2f"  # used for NumberColumn format
    date_format: str = "%Y-%m-%d"  # display preference

    # Auto-save & backup
    backup_before_save: bool = True
    autosave_enabled: bool = False
    autosave_interval_sec: int = 120  # save draft on interactions at most once per interval
    rollback_on_failure: bool = True


# ---- Storage abstraction (skeleton) ----

class SettingsStore:
    """Abstract settings store (no-op skeleton).

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
        # settings.yaml lives alongside this module
        self.settings_path = os.path.join(os.path.dirname(__file__), "settings.yaml")

    def load(self) -> AppSettings:
        """Load settings from YAML; fallback to defaults if not found or invalid.

        Returns:
            AppSettings: An instance of AppSettings with loaded or default values.
        """
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                if not isinstance(data, dict):
                    # invalid file, regenerate defaults
                    defaults = AppSettings()
                    self.save(defaults)
                    return defaults
                return AppSettings(
                    dev_mode=bool(data.get('dev_mode', False)),
                    func_preview_default=bool(data.get('func_preview_default', True)),
                    remember_func_per_file=bool(data.get('remember_func_per_file', True)),
                    extra=dict(data.get('extra') or {}),
                    default_open_path=str(data.get('default_open_path', DATA_CACHE_DIR)),
                    remember_recent_path=bool(data.get('remember_recent_path', True)),
                    language=str(data.get('language', 'zh')),
                    number_format=str(data.get('number_format', '%.2f')),
                    date_format=str(data.get('date_format', '%Y-%m-%d')),
                    backup_before_save=bool(data.get('backup_before_save', True)),
                    autosave_enabled=bool(data.get('autosave_enabled', False)),
                    autosave_interval_sec=int(data.get('autosave_interval_sec', 120)),
                    rollback_on_failure=bool(data.get('rollback_on_failure', True)),
                )
            else:
                # Auto-generate a default settings.yaml on first load
                defaults = AppSettings()
                self.save(defaults)
                return defaults
        except Exception:
            # Fall through to defaults on any error
            pass
        return AppSettings()

    def save(self, settings: AppSettings) -> None:
        """Persist settings to YAML atomically.

        Args:
            settings (AppSettings): An instance of AppSettings to be saved.
        """
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        tmp_path = self.settings_path + ".tmp"
        data = asdict(settings)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        os.replace(tmp_path, self.settings_path)

    def reset(self) -> None:
        """Reset settings to defaults by removing YAML file if exists."""
        try:
            if os.path.exists(self.settings_path):
                os.remove(self.settings_path)
        except Exception:
            pass


# ---- Convenience helpers (skeleton) ----

def get_settings(store: Optional[SettingsStore] = None) -> AppSettings:
    """Get the current settings.

    Args:
        store (Optional[SettingsStore], optional): An instance of SettingsStore. If None, a new instance will be created. Defaults to None.

    Returns:
        AppSettings: An instance of AppSettings with the current settings.
    """
    store = store or SettingsStore()
    return store.load()

def set_settings(settings: AppSettings, store: Optional[SettingsStore] = None) -> None:
    """Set and save the settings.

    Args:
        settings (AppSettings): An instance of AppSettings to be set.
        store (Optional[SettingsStore], optional): An instance of SettingsStore. If None, a new instance will be created. Defaults to None.
    """
    store = store or SettingsStore()
    store.save(settings)


__all__ = [
    'AppSettings',
    'SettingsStore',
    'get_settings',
    'set_settings',
    'render_settings_page',
]


# ---- UI rendering (moved from main): Settings full-page subview ----

def render_settings_page(initial_settings: AppSettings) -> None:
    """Render the Settings full-page subview.

    - Top-left Back button to return
    - Tabs: General / Preview / Data / Developer
    - Actions: Apply (session only), Save (YAML), Reset (defaults), Import/Export YAML

    Args:
        initial_settings (AppSettings): An instance of AppSettings with the initial settings.
    """
    import streamlit as st  # local import to avoid circular deps

    # Apply current language for UI rendering
    try:
        set_lang(initial_settings.language)
    except Exception:
        pass

    # Top-left Back button + big bold title on the right (tight layout)
    try:
        top_cols = st.columns([1, 100], gap="small")
    except TypeError:
        top_cols = st.columns([1, 100])
    with top_cols[0]:
        if st.button(t("btn.back"), key="btn_back_settings_main_top"):
            st.session_state["show_settings_page"] = False
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

    # Tabs
    t_general, t_preview, t_data, t_dev = st.tabs([
        t("tab.general"), t("tab.preview"), t("tab.data"), t("tab.developer")
    ])

    with t_general:
        st.caption(t("cap.general"))
        # Workspace / recent path
        default_open_path = st.text_input(
            t("fld.default_open_path"),
            value=initial_settings.default_open_path,
            key="settings_default_open_path",
            help=t("help.default_open_path"),
        )
        remember_recent_path = st.checkbox(
            t("fld.remember_recent_path"),
            value=initial_settings.remember_recent_path,
            key="settings_remember_recent_path",
            help=t("help.remember_recent_path"),
        )

        st.divider()
        # Locale
        language = st.selectbox(
            t("fld.language"),
            options=["zh", "en"],
            index=["zh", "en"].index(initial_settings.language if initial_settings.language in ["zh", "en"] else "zh"),
            key="settings_language",
            format_func=lambda v: {"zh": "中文（简体）", "en": "English"}.get(v, v),
        )
        number_format = st.text_input(
            t("fld.number_format"),
            value=initial_settings.number_format,
            key="settings_number_format",
            help=t("help.number_format"),
        )
        date_format = st.text_input(
            t("fld.date_format"),
            value=initial_settings.date_format,
            key="settings_date_format",
            help=t("help.date_format"),
        )

        st.divider()
        # Auto-save & backup
        backup_before_save = st.checkbox(
            t("fld.backup_before_save"),
            value=initial_settings.backup_before_save,
            key="settings_backup_before_save",
            help=t("help.backup_before_save"),
        )
        autosave_enabled = st.checkbox(
            t("fld.autosave_enabled"),
            value=initial_settings.autosave_enabled,
            key="settings_autosave_enabled",
            help=t("help.autosave_enabled"),
        )
        autosave_interval_sec = st.number_input(
            t("fld.autosave_interval_sec"),
            min_value=10,
            max_value=3600,
            value=int(initial_settings.autosave_interval_sec),
            step=10,
            key="settings_autosave_interval_sec",
            disabled=not autosave_enabled,
        )
        rollback_on_failure = st.checkbox(
            t("fld.rollback_on_failure"),
            value=initial_settings.rollback_on_failure,
            key="settings_rollback_on_failure",
            help=t("help.rollback_on_failure"),
        )

        st.divider()
        # Existing option
        remember_func_per_file = st.checkbox(
            t("fld.remember_func_per_file"),
            value=initial_settings.remember_func_per_file,
            key="settings_remember_func_per_file",
            help=t("help.remember_func_per_file"),
        )

    with t_preview:
        st.caption(t("cap.preview"))
        func_preview_default = st.checkbox(
            t("fld.func_preview_default"),
            value=initial_settings.func_preview_default,
            key="settings_func_preview_default",
            help=t("help.func_preview_default"),
        )

    with t_data:
        st.caption(t("cap.data"))
        st.info(t("info.data_reserved"))

    with t_dev:
        st.caption(t("cap.developer"))
        dev_mode = st.checkbox(t("fld.dev_mode"), value=initial_settings.dev_mode, key="settings_dev_mode")

    st.divider()
    # Auto-save: persist settings immediately when any control changes
    try:
        new_settings = AppSettings(
            dev_mode=dev_mode,
            func_preview_default=func_preview_default,
            remember_func_per_file=remember_func_per_file,
            extra=initial_settings.extra,
            default_open_path=default_open_path,
            remember_recent_path=remember_recent_path,
            language=language,
            number_format=number_format,
            date_format=date_format,
            backup_before_save=backup_before_save,
            autosave_enabled=autosave_enabled,
            autosave_interval_sec=int(autosave_interval_sec if autosave_enabled else initial_settings.autosave_interval_sec),
            rollback_on_failure=rollback_on_failure,
        )
        if new_settings != initial_settings:
            # Persist only when autosave is enabled; otherwise keep changes session-only
            if new_settings.autosave_enabled:
                set_settings(new_settings)
            # Clear cached keys so new settings take effect immediately
            for k in list(st.session_state.keys()):
                if k.startswith("func_input_") or k.startswith("func_preview_enabled_"):
                    st.session_state.pop(k, None)
            st.session_state.pop("func_input_global", None)
            st.session_state.pop("func_preview_enabled_global", None)
            # If language changed, refresh UI to apply translations immediately
            if new_settings.language != initial_settings.language:
                st.rerun()
    except Exception as e:
        st.error(t("settings.autosave_error").format(err=str(e)))
