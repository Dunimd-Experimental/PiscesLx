/**
 * Copyright © 2026 Wenze Wei. All Rights Reserved.
 *
 * This file is part of Xi.
 * The Xi project belongs to the Dunimd Team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

export type WidgetType =
  | "text"
  | "textarea"
  | "number"
  | "slider"
  | "toggle"
  | "select"
  | "multiselect"
  | "file"
  | "directory"
  | "color"
  | "date"
  | "time"
  | "datetime"
  | "code"
  | "markdown"
  | "keyvalue"
  | "list"
  | "custom";

export interface WidgetValidation {
  pattern?: string;
  message?: string;
  min_length?: number;
  max_length?: number;
}

export interface WidgetStyle {
  width?: "full" | "half" | "auto" | string;
  height?: number;
  placeholder?: string;
  prefix?: string;
  suffix?: string;
  class_name?: string;
}

export interface WidgetConfig {
  type: WidgetType;
  style?: WidgetStyle;
  validation?: WidgetValidation;
  props?: Record<string, unknown>;
  depends_on?: string[];
  show_if?: string;
  disabled_if?: string;
  custom_component?: string;
  custom_props?: Record<string, unknown>;
}

export interface ValueMapping {
  arg_format?: string;
  arg_prefix?: string;
  arg_separator?: string;
  skip_if?: string;
  transform?: "lowercase" | "uppercase" | "str" | "int" | "float" | "json" | "path" | "";
  default_if_empty?: string | number | boolean | null;
  join_with?: string;
  wrap_value?: boolean;
  template?: string;
}

export interface DynamicParameter {
  name: string;
  type: string;
  description: string;
  required: boolean;
  default: string | number | boolean | null;
  options?: string[];
  min?: number;
  max?: number;
  source?: string;
  source_type?: string;
  filter?: string;
  available?: boolean;
  unavailable_reason?: string;
  tab?: string;
  widget?: WidgetConfig;
  value_mapping?: ValueMapping;
}

export interface DynamicTab {
  name: string;
  label: string;
  available: boolean;
  unavailable_reason: string;
}

export interface DynamicCommandSchema {
  command: string;
  description: string;
  available: boolean;
  unavailable_reason: string;
  tabs: DynamicTab[];
  parameters: DynamicParameter[];
}

export interface DynamicOption {
  value: string;
  label: string;
}

export interface DynamicOptionsResponse {
  parameter: string;
  options: DynamicOption[];
  source?: string;
  message?: string;
}

export interface CommandExecuteRequest {
  command: string;
  parameters: Record<string, unknown>;
}

export interface CommandExecuteResponse {
  success: boolean;
  run_id?: string;
  command?: string[];
  error?: string;
}
