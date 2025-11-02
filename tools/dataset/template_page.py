#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from i18n import t
import streamlit as st
from settings import get_settings
from func_templates import FunctionTemplateManager
from func import (
    get, join_text, lower, upper, strip, replace,
    json_loads, json_dumps, to_int, to_float, safe_number,
    dict_lookup, date_parse, date_format,
)

def render_template_page():
    """Render the function template management page.
    
    This function initializes the session state, renders the top navigation,
    operation area, and different panels for template management, including
    template editor, version history, and template list.
    """
    # Initialize session state for template management
    if 'template_editor_open' not in st.session_state:
        st.session_state.template_editor_open = False
    if 'selected_template_id' not in st.session_state:
        st.session_state.selected_template_id = None
    if 'version_view_open' not in st.session_state:
        st.session_state.version_view_open = None
    if 'template_tester_open' not in st.session_state:
        st.session_state.template_tester_open = None
    
    # Initialize template manager
    template_manager = FunctionTemplateManager()
    
    # Render top navigation: back button + page title (aligned with Settings page)
    came_from_settings = st.session_state.get("came_from_settings", False)
    back_label = t("btn.back")  # Only show arrow to avoid vertical line breaks
    try:
        top_cols = st.columns([1, 100], gap="small")  # Align with Settings page
    except TypeError:
        top_cols = st.columns([1, 100])
    
    with top_cols[0]:
        if st.button(back_label, key="back_to_main"):
            st.session_state["show_template_page"] = False
            if came_from_settings:
                st.session_state["show_settings_page"] = True
                st.session_state.pop("came_from_settings", None)
            st.session_state.template_editor_open = False
            st.session_state.selected_template_id = None
            st.session_state.version_view_open = None
            st.rerun()
    
    with top_cols[1]:
        st.markdown(
            f"""
            <div style="font-weight: 700; font-size: 24px; line-height: 38px; margin-left: 18px;">{t('title.template_library')}</div>
            """,
            unsafe_allow_html=True,
        )
    
    # Read settings to control default expansion
    _app_settings = get_settings()
    _exp_def = bool(getattr(_app_settings, 'expand_template_panels_by_default', True))

    # Render top operation area in an expander
    with st.expander("🛠️ " + t("operation"), expanded=_exp_def):
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button(f"✨ {t('btn.create_template')}", type="primary", use_container_width=True):
                st.session_state.template_editor_open = True
                st.session_state.selected_template_id = None
                st.rerun()
        
        with col2:
            # Export all templates
            _templates_for_export = template_manager.list_templates()
            if _templates_for_export:
                export_data = template_manager.export_all_templates()
                st.download_button(
                    label=f"📥 {t('btn.export_all')}",
                    data=export_data,
                    file_name=f"templates_backup_{st.session_state.get('current_time', 'unknown')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col3:
            # Category filter
            categories = template_manager.get_categories()
            selected_category = st.selectbox(
                "",
                [t("option.all_categories")] + categories,
                key="category_filter",
                label_visibility="collapsed"
            )
    
    # Render optional expanders: editor, version history
    if st.session_state.template_editor_open:
        _ed_name = None
        if st.session_state.selected_template_id:
            _cur = FunctionTemplateManager().get_template(st.session_state.selected_template_id)
            _ed_name = (_cur or {}).get('name')
        title_text = (
            "✏️ " + t('hdr.edit_template') + (f" - {_ed_name}" if _ed_name else "")
            if st.session_state.selected_template_id
            else "✨ " + t('hdr.create_template')
        )
        with st.expander(title_text, expanded=_exp_def):
            _render_template_editor(template_manager)
    
    if st.session_state.version_view_open:
        with st.expander("📋 " + t('title.version_history'), expanded=_exp_def):
            _render_version_history(template_manager)
    # Template tester feature has been removed
    
    # Render template list in an expander
    _render_template_list(template_manager, selected_category, _exp_def)

def _render_template_editor(template_manager):
    """Render the template editor form.
    
    Args:
        template_manager (FunctionTemplateManager): An instance of FunctionTemplateManager
            for template management operations.
    """
    template_data = None
    if st.session_state.selected_template_id:
        template_data = template_manager.get_template(st.session_state.selected_template_id)
    
    # Render editor title
    st.subheader('✨ ' + t('hdr.create_template') if not template_data else '✏️ ' + t('hdr.edit_template'))
    if not template_data:
        st.caption(t('desc.build_templates'))
    else:
        st.caption(template_data.get('name', ''))
    
    with st.form("template_form"):
        # Two-column layout for basic information
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            name = st.text_input(
                "🏷️ " + t("fld.template_name"),
                value=template_data.get("name", "") if template_data else "",
                key="template_name",
                placeholder=t("ph.template_name")
            )
            category = st.text_input(
                "📂 " + t("fld.template_category"),
                value=template_data.get("category", "") if template_data else "",
                key="template_category",
                placeholder=t("ph.template_category")
            )
            tags = st.text_input(
                "🏷️ " + t("fld.template_tags"),
                value=", ".join(template_data.get("tags", [])) if template_data else "",
                key="template_tags",
                placeholder=t("ph.template_tags")
            )
        
        with right_col:
            description = st.text_area(
                "📝 " + t("fld.template_description"),
                value=template_data.get("description", "") if template_data else "",
                key="template_description",
                height=120,
                placeholder=t("ph.template_description")
            )
        
        # Add a divider above the code editor
        st.divider()
        function_code = st.text_area(
            "💻 " + t("fld.template_code"),
            value=template_data.get("function_code", "") if template_data else "",
            height=350,
            key="template_code",
            placeholder=t("ph.template_code")
        )
        
        # Add a divider above the form operation area
        st.divider()
        col1, col4 = st.columns([1, 1])
        
        with col1:
            if st.form_submit_button(
                ("💾 " + t("btn.save_template")) if not template_data else ("💾 " + t("btn.update_template")), 
                type="primary", 
                use_container_width=True
            ):
                if name and function_code:
                    tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
                    
                    if template_data:
                        success = template_manager.update_template(
                            st.session_state.selected_template_id,
                            name=name,
                            category=category,
                            description=description,
                            tags=tags_list,
                            function_code=function_code
                        )
                        if success:
                            st.success("✅ " + t("msg.template_updated"))
                        else:
                            st.error("❌ " + t("error.export_failed"))
                    else:
                        template_id = template_manager.save_template(
                            name=name,
                            category=category,
                            description=description,
                            tags=tags_list,
                            function_code=function_code
                        )
                        st.success("✅ " + t("msg.template_saved"))
                    
                    st.session_state.template_editor_open = False
                    st.rerun()
                else:
                    st.error("❌ " + t("error.name_code_required"))
        
        # Preview/test button has been removed
        
        with col4:
            if st.form_submit_button("❌ " + t("btn.cancel"), type="secondary", use_container_width=True):
                st.session_state.template_editor_open = False
                st.rerun()

def _render_version_history(template_manager):
    """Render the version history page.
    
    Args:
        template_manager (FunctionTemplateManager): An instance of FunctionTemplateManager
            for template management operations.
    """
    template_id = st.session_state.version_view_open
    versions = template_manager.list_versions(template_id)
    template = template_manager.get_template(template_id)
    
    # Render page title using native component
    st.subheader('📋 ' + t('title.version_history') + f" - {template.get('name', 'Template')}")
    
    if versions:
        # Show version statistics
        total_versions = len(versions)
        st.info(f"📊 v{total_versions}")
        
        for version in versions:
            # Render each version card
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
                
                with col1:
                    st.metric(label=t('label.version'), value=f"v{version.get('version')}")
                    st.caption(version.get('updated_at', '')[:10])
                
                with col2:
                    desc = version.get('description', '')
                    if len(desc) > 80:
                        desc = desc[:77] + "..."
                    st.write(desc or t('label.no_description'))
                    
                    # Render tags
                    tags = version.get('tags', [])
                    if tags:
                        tag_html = " ".join(
                            f"<span style='background: #f8f9fa; color: #6c757d; padding: 2px 8px; border-radius: 12px; font-size: 10px; margin-right: 5px;'>{tag}</span>"
                            for tag in tags
                        )
                        st.markdown(f"<div style='margin-top: 8px;'>{tag_html}</div>", unsafe_allow_html=True)
                
                with col3:
                    # Show code preview
                    code_preview = version.get('function_code', '')[:100]
                    if len(code_preview) > 100:
                        code_preview = code_preview[:97] + "..."
                    st.code(code_preview, language="python")
                
                with col4:
                    # Render rollback button
                    rollback_key = f"rollback_{version['id']}_{version['version']}"
                    if st.button("⏪ " + t("btn.rollback"), key=rollback_key, use_container_width=True):
                        if template_manager.rollback_to_version(template_id, version['version']):
                            st.success("✅ " + t("msg.rollback_success_inline").format(version=version['version']))
                            st.session_state.version_view_open = None
                            st.rerun()
                        else:
                            st.error("❌ " + t("error.export_failed"))
    else:
        # Render empty state
        st.info("📋 " + t('label.no_version_history'))
        st.caption(t('label.no_version_history_desc'))
    
    # Render back button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("❌ " + t("btn.close_history"), type="secondary", use_container_width=True):
            st.session_state.version_view_open = None
            st.rerun()

def _render_template_tester(template_manager):
    """Render the template tester view.
    
    Args:
        template_manager (FunctionTemplateManager): An instance of FunctionTemplateManager
            for template management operations.
    """
    template_id = st.session_state.template_tester_open
    tpl = template_manager.get_template(template_id)
    name = (tpl or {}).get('name', 'Template')

    # Render header in native style
    st.subheader("🧪 " + t('tester.title') + f" - {name}")
    st.divider()

    # Prepare environment for code execution
    code = (tpl or {}).get('function_code', '')
    env = {
        '__builtins__': {},
        # Helper functions
        'get': get, 'join_text': join_text, 'lower': lower, 'upper': upper, 'strip': strip, 'replace': replace,
        'json_loads': json_loads, 'json_dumps': json_dumps,
        'to_int': to_int, 'to_float': to_float, 'safe_number': safe_number,
        'dict_lookup': dict_lookup,
        'date_parse': date_parse, 'date_format': date_format,
    }
    ns = {}
    available_funcs = []
    try:
        exec(code, env, ns)
        for fn_name in ['transform', 'main', 'apply']:
            if callable(ns.get(fn_name)):
                available_funcs.append(fn_name)
    except Exception as e:
        st.error(t('error.exec_failed').format(err=str(e)))
        available_funcs = []

    # Layout: Source chooser | Function picker
    left, right = st.columns([2, 1])
    rec_to_use = None
    
    with left:
        st.subheader(t('tester.source_record'))
        loaded = st.session_state.get('loaded_data') or []
        use_loaded = False
        if loaded:
            st.caption(t('tester.use_loaded_data').format(n=min(len(loaded), 1000)))
            use_loaded = st.checkbox(t('tester.use_loaded_data').format(n=min(len(loaded), 1000)), value=True, key='tester_use_loaded')
        if use_loaded and loaded:
            idx = st.number_input(t('tester.record_index'), min_value=1, max_value=len(loaded), value=1, step=1)
            rec_to_use = loaded[int(idx) - 1]
        else:
            st.caption(t('tester.or'))
            sample_json = st.text_area(t('tester.input_json'), height=160, placeholder='{"text": "hello"}')
            if sample_json.strip():
                try:
                    rec_to_use = json.loads(sample_json)
                except Exception:
                    rec_to_use = None
                    st.warning('JSON parse error')
    
    with right:
        st.subheader(t('tester.pick_function'))
        if available_funcs:
            fn_name = st.selectbox(t('tester.pick_function'), options=available_funcs, key='tester_fn_pick')
        else:
            st.error(t('error.no_function_found'))
            fn_name = None

    st.divider()
    run = st.button('▶ ' + t('btn.run_test'), type='primary')
    if run:
        if not fn_name:
            st.stop()
        if rec_to_use is None:
            st.warning(t('warn.no_data_to_save'))
            st.stop()
        try:
            result = ns[fn_name](rec_to_use)
            st.write("")
            st.subheader(t('tester.result'))
            try:
                st.json(result)
            except Exception:
                st.code(str(result), language='python')
        except Exception as e:
            st.error(t('error.exec_failed').format(err=str(e)))

    st.divider()
    if st.button("❌ " + t("btn.close"), type="secondary"):
        st.session_state.template_tester_open = None
        st.rerun()

def _render_template_list(template_manager, selected_category, exp_def: bool):
    """Render the main template list.
    
    Args:
        template_manager (FunctionTemplateManager): An instance of FunctionTemplateManager
            for template management operations.
        selected_category (str): The selected category for filtering templates.
        exp_def (bool): Whether to expand the panels by default.
    """
    # Get filtered templates
    # Respect localized "All Categories" option
    category_filter = None if selected_category == t("option.all_categories") else selected_category
    templates = template_manager.list_templates(category_filter)
    
    if templates:
        # Put the whole list in an expander (including search box)
        with st.expander("📚 " + t('title.template_library'), expanded=exp_def):
            # Render search box
            search_q = st.text_input(t("label.search_templates"), key="template_search_box", placeholder=t("ph.search_templates2"))
            filtered = templates
            if search_q:
                q = search_q.strip().lower()
                def _hit(tpl):
                    name = str(tpl.get('name','')).lower()
                    desc = str(tpl.get('description','')).lower()
                    tags = ','.join(tpl.get('tags', [])).lower()
                    return (q in name) or (q in desc) or (q in tags)
                filtered = [tpl for tpl in templates if _hit(tpl)]

            total_templates = len(filtered)
            categories_count = len(template_manager.get_categories())
            st.caption(t('label.templates_in_categories').format(n=total_templates, k=categories_count))
            st.divider()
            
            # Render template grid
            for template in filtered:
                name = template.get('name', 'Unnamed Template')
                category = template.get('category', 'default')
                tags = ', '.join(template.get('tags', [])) if template.get('tags') else t('label.no_tags')
                # Chinese-labeled title for better readability
                header = f"{t('label.cn_name')}：{name} ｜ 📂 {t('label.cn_category')}：{category} ｜ 🏷️ {t('label.cn_tags')}：{tags}"
                with st.expander(header, expanded=exp_def):
                    # Two-column metrics at the top to avoid uneven heights
                    col2, col3 = st.columns([1, 1])
                    
                    with col2:
                        version = template.get('version', 1)
                        st.metric(label=t('label.version'), value=f"v{version}")
                    
                    with col3:
                        updated = template.get('updated_at', '')[:10]
                        st.metric(label=t('label.updated'), value=updated)

                    # Make description more prominent
                    if template.get('description'):
                        st.markdown("**📝 " + t('fld.template_description') + "**")
                        st.write(template.get('description'))

                    # Full code preview
                    with st.expander("📄 " + t("view_function_code"), expanded=False):
                        st.code(template.get('function_code', ''), language="python")

                    # Operation buttons
                    col1, col2, col3, col5 = st.columns([1, 1, 1, 1])
                    
                    with col1:
                        if st.button("✏️ " + t("btn.edit"), key=f"edit_{template['id']}", use_container_width=True):
                            st.session_state.template_editor_open = True
                            st.session_state.selected_template_id = template['id']
                            st.rerun()
                    
                    with col2:
                        if st.button("📋 " + t("btn.view_versions"), key=f"versions_{template['id']}", use_container_width=True):
                            st.session_state.version_view_open = template['id']
                            st.rerun()
                    
                    with col3:
                        export_data = template_manager.export_template(template['id'])
                        if export_data:
                            st.download_button(
                                label="📤 " + t("btn.export_template"),
                                data=export_data,
                                file_name=f"{template.get('name', 'template')}.json",
                                mime="application/json",
                                key=f"export_{template['id']}",
                                use_container_width=True
                            )
                    # Test button has been removed
                    
                    with col5:
                        if st.button("🗑️ " + t("btn.delete_template"), key=f"delete_{template['id']}", use_container_width=True):
                            if template_manager.delete_template(template['id']):
                                st.success("✅ " + t("msg.template_deleted"))
                                st.rerun()
                            else:
                                st.error("❌ " + t("error.delete_template_failed"))
    
    else:
        # Render empty state
        st.info("🎯 " + t('empty.no_templates'))
        st.caption(t('empty.start_building'))

__all__ = [
    'render_template_page',
    '_render_template_editor',
    '_render_version_history',
    '_render_template_tester',
    '_render_template_list',
]        