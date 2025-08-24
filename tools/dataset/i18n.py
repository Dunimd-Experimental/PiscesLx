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

from __future__ import annotations

from typing import Dict

# Simple i18n registry
_MESSAGES: Dict[str, Dict[str, str]] = {
    "en": {
        # Common
        "title.settings": "Settings",
        "app.title": "PiscesData Control Center",
        "btn.back": "←",
        # Tabs
        "tab.general": "General",
        "tab.preview": "Preview",
        "tab.data": "Data",
        "tab.developer": "Developer",
        # Captions
        "cap.general": "General preferences",
        "cap.preview": "Preview behavior",
        "cap.data": "Data options (reserved)",
        "cap.developer": "Developer options",
        # General fields
        "fld.default_open_path": "Default open path",
        "help.default_open_path": "The path shown by default when opening the tool.",
        "fld.remember_recent_path": "Remember recent path",
        "help.remember_recent_path": "Remember the last used path and use it next time.",
        "fld.language": "Language",
        "fld.number_format": "Number format",
        "help.number_format": "Format string for numbers, e.g. %.2f",
        "fld.date_format": "Date format",
        "help.date_format": "Display format for dates, e.g. %Y-%m-%d",
        "fld.backup_before_save": "Backup before save",
        "help.backup_before_save": "Create a .backup file before saving.",
        "fld.autosave_enabled": "Enable autosave (draft)",
        "help.autosave_enabled": "Periodically write a .autosave draft of current edits.",
        "fld.autosave_interval_sec": "Autosave interval (sec)",
        "fld.rollback_on_failure": "Rollback on failure",
        "help.rollback_on_failure": "If save fails, attempt to restore from backup.",
        "fld.remember_func_per_file": "Remember function per file",
        "help.remember_func_per_file": "Remember function input separately for each file.",
        # Preview
        "fld.func_preview_default": "Default enable f(x) preview",
        "help.func_preview_default": "Use as the default value for the preview switch of new files.",
        # Data
        "info.data_reserved": "No more data-related settings available yet.",
        # Dev
        "fld.dev_mode": "Developer Mode",

        # Main page
        "input.path_label": "Enter file path or directory path",
        "btn.rescan": "🔄 Rescan Directory",
        "btn.convert_arrow": "🧭 Convert Arrow→JSON",
        "warn.path_missing": "⚠️ Path does not exist",
        "exp.convert_log": "Conversion Log",
        "spinner.converting_arrow": "Converting Arrow dataset under the selected path...",
        "success.arrow_done": "✅ Arrow→JSON conversion completed",
        "error.convert_failed": "❌ Conversion failed: {err}",
        "warn.no_files": "📁 No json/jsonl files found in {path}",
        "info.place_files": "Please place your json/jsonl files in the specified directory",
        "select.file_label": "Please select the file to process (sorted A-Z):",
        "info.auto_selected": "📄 Auto-selected file: {file}",
        "spinner.scan_fields": "🔍 Scanning fields...",
        "error.load_failed": "❌ Failed to load data: {err}",
        "info.scan_summary": "✅ Total {total} samples, {fields} fields",
        "metric.total_rows": "Total Rows",
        "metric.total_cols": "Total Columns",
        "warn.large_file": "⚠️ Large file detected ({mb:.1f} MB). It is recommended to reduce the number of preview entries.",
        "chk.load_all": "🚀 Load All Data",
        "chk.load_all_caution": "🚀 Load All Data (Proceed with Caution)",
        "input.function": "Function Input",
        "input.preview_limit": "Number of Preview Entries",
        "sidebar.field_rules": "### Field Rules",
        "sidebar.scanned_fields": "Scanned Fields (debug)",
        "sidebar.text_replace": "### Text Content Replacement",
        "sidebar.search_text": "Search Text",
        "sidebar.replace_with": "Replace With",
        "sidebar.confirm_replace": "Confirm Replacement",
        "sidebar.apply_replace": "Apply Replacement",
        "sidebar.replace_completed": "✅ Text replacement completed! Total {n} replacements",
        "sidebar.backup_created": "Backup created: {path}",
        "sidebar.no_match": "No matching text found",
        "sidebar.error": "Error in text replacement: {err}",
        # Placeholders
        "ph.search_text": "Enter text to search...",
        "ph.replace_with": "Enter replacement content...",
        # FX Preview UI
        "fx.toggle_label": "Enable f(x) preview",
        "fx.toggle_help": "Show/hide function expression preview dropdown. Won't write back to file",
        "fx.eval_error": "Function eval error: {err}",
        "fx.per_record_error": "<ERR: {err}>",
        "fx.unrepresentable": "<UNREPRESENTABLE>",
        # Dev debug panel
        "dev.count": "Count: {n}",
        "dev.missing": "missing",
        "dev.types": "types",
        "dev.no_result": "<No result>",
        # Path input placeholder
        "ph.path_input": "Select or enter a data directory...",
        # Settings autosave error
        "settings.autosave_error": "Failed to auto-save settings: {err}",
        # Field manager (sidebar)
        "sidebar.add_new_field_title": "#### Add New Field",
        "sidebar.field_name": "Field Name",
        "sidebar.default_value": "Default Value",
        "sidebar.add_field_btn": "➕ Add Field",
        "sidebar.add_field_written": "✅ New field has been written to all records and saved",
        "sidebar.add_field_failed": "❌ Failed to write: {err}",
        "sidebar.field_exists": "Field '{name}' already exists",
        "sidebar.enter_field_name": "Please enter a field name",
        "ph.new_field_name": "Enter new field name",
        "ph.default_value": "Default value",
        "sidebar.manage_fields_title": "#### Manage Existing Fields",
        "sidebar.rename": "Rename",
        "sidebar.reorder_success": "✅ Successfully written back to file in new order",
        "sidebar.reorder_failed": "❌ Failed to write back in order: {err}",
        "sidebar.delete_success": "✅ Successfully deleted field and saved: {name}",
        "sidebar.delete_failed": "❌ Failed to delete and save: {err}",
        "sidebar.rename_applied": "✅ Renames have been automatically applied and saved",
        "sidebar.rename_persist_failed": "❌ Failed to persist renames: {err}",
        "btn.save_changes": "💾 Save Changes",
        "warn.no_data_to_save": "⚠️ No data to save",
        "success.changes_saved": "✅ Changes saved to the original file!",
        "error.save_failed": "❌ Save failed: {err}",
        "warn.load_file_first": "⚠️ Please load a file first",
        "label.output_arrow": "Output Arrow Path",
        "btn.gen_arrow": "🚀 Generate Arrow",
        "success.arrow_export": "✅ Arrow export completed! Total {n} samples → {path}",
        "error.arrow_export": "❌ Arrow conversion error: {err}",
        "label.output_json": "Output JSON Path",
        "btn.gen_json": "📝 Generate JSON",
        "success.json_export": "✅ JSON export completed! Total {n} samples → {path}",
        "error.json_export": "❌ JSON conversion error: {err}",
        "warn.no_data_to_export": "⚠️ No data to export",
        "section.replace_original": "### 🔥 Replace Original File",
        "chk.confirm_replace_original": "⚠️ Confirm to replace the original file (This operation is irreversible)",
        "btn.replace_original": "💾 Replace Original File Directly",
        "success.replace_original": "✅ Original file replaced! Total {n} samples",
        "error.replace_original": "❌ Error replacing file: {err}",
        "sidebar.settings_hint": "设置",
    },
    "zh": {
        # Common
        "title.settings": "设置",
        "app.title": "PiscesData 控制中心",
        "btn.back": "←",
        # Tabs
        "tab.general": "通用",
        "tab.preview": "预览",
        "tab.data": "数据",
        "tab.developer": "开发者",
        # Captions
        "cap.general": "通用偏好",
        "cap.preview": "预览行为",
        "cap.data": "数据选项（预留）",
        "cap.developer": "开发者选项",
        # General fields
        "fld.default_open_path": "默认打开路径",
        "help.default_open_path": "工具启动时默认展示的路径。",
        "fld.remember_recent_path": "记住最近路径",
        "help.remember_recent_path": "记住上次使用的路径并在下次默认使用。",
        "fld.language": "语言",
        "fld.number_format": "数字格式",
        "help.number_format": "数字显示格式，例如 %.2f",
        "fld.date_format": "日期格式",
        "help.date_format": "日期显示格式，例如 %Y-%m-%d",
        "fld.backup_before_save": "保存前备份",
        "help.backup_before_save": "保存前生成 .backup 备份文件。",
        "fld.autosave_enabled": "启用自动保存（草稿）",
        "help.autosave_enabled": "按间隔周期保存当前编辑为 .autosave 草稿。",
        "fld.autosave_interval_sec": "自动保存间隔（秒）",
        "fld.rollback_on_failure": "失败回滚",
        "help.rollback_on_failure": "保存失败时，尝试从备份回滚。",
        "fld.remember_func_per_file": "记住每个文件的函数输入",
        "help.remember_func_per_file": "为每个文件分别记忆函数输入和开关状态。",
        # Preview
        "fld.func_preview_default": "默认启用 f(x) 预览",
        "help.func_preview_default": "作为新文件预览开关的默认值。",
        # Data
        "info.data_reserved": "暂无更多数据类设置。",
        # Dev
        "fld.dev_mode": "开发者模式",

        # Main page
        "input.path_label": "输入文件或目录路径",
        "btn.rescan": "🔄 重新扫描目录",
        "btn.convert_arrow": "🧭 将 Arrow 转为 JSON",
        "warn.path_missing": "⚠️ 路径不存在",
        "exp.convert_log": "转换日志",
        "spinner.converting_arrow": "正在转换所选路径下的 Arrow 数据集...",
        "success.arrow_done": "✅ Arrow→JSON 转换完成",
        "error.convert_failed": "❌ 转换失败：{err}",
        "warn.no_files": "📁 在 {path} 未找到 json/jsonl 文件",
        "info.place_files": "请将 json/jsonl 文件放置到指定目录",
        "select.file_label": "请选择要处理的文件（A-Z 排序）：",
        "info.auto_selected": "📄 已自动选择文件：{file}",
        "spinner.scan_fields": "🔍 正在扫描字段...",
        "error.load_failed": "❌ 加载数据失败：{err}",
        "info.scan_summary": "✅ 共 {total} 条样本，{fields} 个字段",
        "metric.total_rows": "总行数",
        "metric.total_cols": "总列数",
        "warn.large_file": "⚠️ 检测到大文件（{mb:.1f} MB）。建议减少预览条目数。",
        "chk.load_all": "🚀 加载全部数据",
        "chk.load_all_caution": "🚀 加载全部数据（谨慎操作）",
        "input.function": "函数输入",
        "input.preview_limit": "预览条目数",
        "sidebar.field_rules": "### 字段规则",
        "sidebar.scanned_fields": "已扫描字段（调试）",
        "sidebar.text_replace": "### 文本内容替换",
        "sidebar.search_text": "搜索文本",
        "sidebar.replace_with": "替换为",
        "sidebar.confirm_replace": "确认替换",
        "sidebar.apply_replace": "应用替换",
        "sidebar.replace_completed": "✅ 文本替换完成！共 {n} 处",
        "sidebar.backup_created": "已创建备份：{path}",
        "sidebar.no_match": "未找到匹配文本",
        "sidebar.error": "文本替换出错：{err}",
        # Placeholders
        "ph.search_text": "输入要搜索的文本…",
        "ph.replace_with": "输入替换后的内容…",
        # FX Preview UI
        "fx.toggle_label": "启用 f(x) 预览",
        "fx.toggle_help": "显示/隐藏函数表达式预览下拉，不会写回文件",
        "fx.eval_error": "函数求值错误：{err}",
        "fx.per_record_error": "<错误: {err}>",
        "fx.unrepresentable": "<不可表示>",
        # Dev debug panel
        "dev.count": "数量：{n}",
        "dev.missing": "缺失",
        "dev.types": "类型",
        "dev.no_result": "<无结果>",
        # Path input placeholder
        "ph.path_input": "选择或输入数据目录…",
        # Settings autosave error
        "settings.autosave_error": "自动保存设置失败：{err}",
        # Field manager (sidebar)
        "sidebar.add_new_field_title": "#### 新增字段",
        "sidebar.field_name": "字段名",
        "sidebar.default_value": "默认值",
        "sidebar.add_field_btn": "➕ 添加字段",
        "sidebar.add_field_written": "✅ 新字段已写入所有记录并保存",
        "sidebar.add_field_failed": "❌ 写入失败：{err}",
        "sidebar.field_exists": "字段 '{name}' 已存在",
        "sidebar.enter_field_name": "请输入字段名",
        "ph.new_field_name": "输入新字段名",
        "ph.default_value": "默认值",
        "sidebar.manage_fields_title": "#### 管理已有字段",
        "sidebar.rename": "重命名",
        "sidebar.reorder_success": "✅ 已按新顺序回写到文件",
        "sidebar.reorder_failed": "❌ 按顺序回写失败：{err}",
        "sidebar.delete_success": "✅ 字段已删除并保存：{name}",
        "sidebar.delete_failed": "❌ 删除并保存失败：{err}",
        "sidebar.rename_applied": "✅ 重命名已自动应用并保存",
        "sidebar.rename_persist_failed": "❌ 持久化重命名失败：{err}",
        "btn.save_changes": "💾 保存修改",
        "warn.no_data_to_save": "⚠️ 无可保存的数据",
        "success.changes_saved": "✅ 变更已保存到原文件！",
        "error.save_failed": "❌ 保存失败：{err}",
        "warn.load_file_first": "⚠️ 请先加载文件",
        "label.output_arrow": "输出 Arrow 路径",
        "btn.gen_arrow": "🚀 生成 Arrow",
        "success.arrow_export": "✅ Arrow 导出完成！共 {n} 条 → {path}",
        "error.arrow_export": "❌ Arrow 转换错误：{err}",
        "label.output_json": "输出 JSON 路径",
        "btn.gen_json": "📝 生成 JSON",
        "success.json_export": "✅ JSON 导出完成！共 {n} 条 → {path}",
        "error.json_export": "❌ JSON 转换错误：{err}",
        "warn.no_data_to_export": "⚠️ 无可导出的数据",
        "section.replace_original": "### 🔥 直接替换原文件",
        "chk.confirm_replace_original": "⚠️ 确认直接替换原文件（该操作不可逆）",
        "btn.replace_original": "💾 直接替换原文件",
        "success.replace_original": "✅ 已替换原文件！共 {n} 条",
        "error.replace_original": "❌ 替换出错：{err}",
        "sidebar.settings_hint": "设置",
    },
}

_current_lang = "zh"

def set_lang(lang: str) -> None:
    global _current_lang
    _current_lang = lang if lang in _MESSAGES else "zh"


def t(key: str, *, lang: str | None = None) -> str:
    L = (lang or _current_lang)
    m = _MESSAGES.get(L) or _MESSAGES["zh"]
    return m.get(key) or _MESSAGES["en"].get(key) or key

__all__ = ["t", "set_lang"]
