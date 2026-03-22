/**
 * Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
 *
 * This file is part of PiscesL1.
 * The PiscesL1 project belongs to the Dunimd Team.
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
 *
 * DISCLAIMER: Users must comply with applicable AI regulations.
 * Non-compliance may result in service termination or legal liability.
 */

"use client";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  MultiSelect,
  MultiSelectContent,
  MultiSelectItem,
  MultiSelectTrigger,
  MultiSelectValue,
} from "@/components/ui/multi-select";
import { Textarea } from "@/components/ui/textarea";
import type { DynamicParameter, DynamicOption, WidgetConfig } from "@/types/dynamic";
import { UnavailableField } from "@/components/ui/unavailable";
import { FolderOpen, FileText, Calendar, Clock, Palette } from "lucide-react";

interface DynamicWidgetProps {
  parameter: DynamicParameter;
  value: unknown;
  onChange: (name: string, value: unknown) => void;
  options?: DynamicOption[];
  disabled?: boolean;
  allValues?: Record<string, unknown>;
}

export function DynamicWidget({
  parameter,
  value,
  onChange,
  options = [],
  disabled = false,
  allValues = {},
}: DynamicWidgetProps) {
  const widget = parameter.widget;
  const widgetType = widget?.type || getDefaultWidgetType(parameter.type);
  const style = widget?.style || {};
  const validation = widget?.validation || {};

  if (parameter.available === false) {
    return <UnavailableField reason={parameter.unavailable_reason || "Not available"} />;
  }

  const isDisabled = disabled || shouldDisable(widget, allValues);
  const isHidden = shouldHide(widget, allValues);

  if (isHidden) {
    return null;
  }

  const widthClass = style.width === "half" ? "md:col-span-1" : "md:col-span-2";
  const placeholder = style.placeholder || parameter.description || "";

  switch (widgetType) {
    case "text":
      return (
        <Input
          type="text"
          value={String(value ?? parameter.default ?? "")}
          onChange={(e) => onChange(parameter.name, e.target.value)}
          placeholder={placeholder}
          disabled={isDisabled}
          className={style.class_name}
          minLength={validation.min_length}
          maxLength={validation.max_length}
        />
      );

    case "textarea":
      return (
        <Textarea
          value={String(value ?? parameter.default ?? "")}
          onChange={(e) => onChange(parameter.name, e.target.value)}
          placeholder={placeholder}
          disabled={isDisabled}
          className={cn("textarea--dynamic", style.class_name)}
          style={{ '--textarea-height': style.height ? `${style.height}px` : '80px' } as React.CSSProperties}
          minLength={validation.min_length}
          maxLength={validation.max_length}
        />
      );

    case "number":
      return (
        <Input
          type="number"
          value={Number(value ?? parameter.default ?? 0)}
          onChange={(e) => onChange(parameter.name, parseFloat(e.target.value) || 0)}
          placeholder={placeholder}
          disabled={isDisabled}
          className={style.class_name}
          min={parameter.min}
          max={parameter.max}
          step={parameter.type === "float" ? "0.0001" : "1"}
        />
      );

    case "slider":
      return (
        <div className="flex items-center gap-4">
          <Slider
            value={[Number(value ?? parameter.default ?? parameter.min ?? 0)]}
            onValueChange={([v]) => onChange(parameter.name, v)}
            min={parameter.min ?? 0}
            max={parameter.max ?? 100}
            step={1}
            disabled={isDisabled}
            className={style.class_name}
          />
          <span className="w-12 text-sm text-muted-foreground">
            {Number(value ?? parameter.default ?? 0)}
          </span>
        </div>
      );

    case "toggle":
      return (
        <div className="flex items-center gap-2">
          <Switch
            checked={Boolean(value ?? parameter.default ?? false)}
            onCheckedChange={(checked) => onChange(parameter.name, checked)}
            disabled={isDisabled}
          />
          <span className="text-sm text-muted-foreground">
            {Boolean(value ?? parameter.default ?? false) ? "Enabled" : "Disabled"}
          </span>
        </div>
      );

    case "select":
      const selectOptions = options.length > 0
        ? options
        : parameter.options?.map(opt => ({ value: opt, label: opt })) || [];
      
      const hasOptions = selectOptions.length > 0;
      const selectValue = String(value ?? parameter.default ?? "");
      const displayValue = hasOptions ? selectValue : "None";

      return (
        <Select
          value={displayValue}
          onValueChange={(v) => onChange(parameter.name, v)}
          disabled={isDisabled || !hasOptions}
        >
          <SelectTrigger className={!hasOptions ? "text-muted-foreground" : ""}>
            <SelectValue placeholder={hasOptions ? placeholder : "None"} />
          </SelectTrigger>
          <SelectContent>
            {hasOptions ? (
              selectOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))
            ) : (
              <SelectItem value="None" disabled className="text-muted-foreground">
                None
              </SelectItem>
            )}
          </SelectContent>
        </Select>
      );

    case "multiselect":
      const multiOptions = options.length > 0
        ? options
        : parameter.options?.map(opt => ({ value: opt, label: opt })) || [];
      
      const selectedValues = Array.isArray(value) ? value : 
        (value ? String(value).split(",") : []);

      return (
        <MultiSelect
          values={selectedValues}
          onValuesChange={(values) => onChange(parameter.name, values)}
          disabled={isDisabled}
        >
          <MultiSelectTrigger>
            <MultiSelectValue placeholder={placeholder} />
          </MultiSelectTrigger>
          <MultiSelectContent>
            {multiOptions.map((opt) => (
              <MultiSelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </MultiSelectItem>
            ))}
          </MultiSelectContent>
        </MultiSelect>
      );

    case "file":
      return (
        <div className="flex gap-2">
          <Input
            type="text"
            value={String(value ?? parameter.default ?? "")}
            onChange={(e) => onChange(parameter.name, e.target.value)}
            placeholder={placeholder}
            disabled={isDisabled}
            className={style.class_name}
          />
          <Button
            variant="secondary"
            size="icon"
            disabled={isDisabled}
            onClick={() => {
              const input = document.createElement("input");
              input.type = "file";
              input.onchange = (e) => {
                const file = (e.target as HTMLInputElement).files?.[0];
                if (file) {
                  onChange(parameter.name, file.name);
                }
              };
              input.click();
            }}
          >
            <FileText className="h-4 w-4" />
          </Button>
        </div>
      );

    case "directory":
      return (
        <div className="flex gap-2">
          <Input
            type="text"
            value={String(value ?? parameter.default ?? "")}
            onChange={(e) => onChange(parameter.name, e.target.value)}
            placeholder={placeholder}
            disabled={isDisabled}
            className={style.class_name}
          />
          <Button
            variant="secondary"
            size="icon"
            disabled={isDisabled}
          >
            <FolderOpen className="h-4 w-4" />
          </Button>
        </div>
      );

    case "color":
      return (
        <div className="flex gap-2 items-center">
          <Input
            type="color"
            value={String(value ?? parameter.default ?? "#000000")}
            onChange={(e) => onChange(parameter.name, e.target.value)}
            disabled={isDisabled}
            className="w-12 h-10 p-1 cursor-pointer"
          />
          <Input
            type="text"
            value={String(value ?? parameter.default ?? "")}
            onChange={(e) => onChange(parameter.name, e.target.value)}
            placeholder="#000000"
            disabled={isDisabled}
            className={style.class_name}
          />
        </div>
      );

    case "date":
      return (
        <div className="flex gap-2">
          <Input
            type="date"
            value={String(value ?? parameter.default ?? "")}
            onChange={(e) => onChange(parameter.name, e.target.value)}
            disabled={isDisabled}
            className={style.class_name}
          />
          <Calendar className="h-4 w-4 text-muted-foreground self-center" />
        </div>
      );

    case "time":
      return (
        <div className="flex gap-2">
          <Input
            type="time"
            value={String(value ?? parameter.default ?? "")}
            onChange={(e) => onChange(parameter.name, e.target.value)}
            disabled={isDisabled}
            className={style.class_name}
          />
          <Clock className="h-4 w-4 text-muted-foreground self-center" />
        </div>
      );

    case "datetime":
      return (
        <div className="flex gap-2">
          <Input
            type="datetime-local"
            value={String(value ?? parameter.default ?? "")}
            onChange={(e) => onChange(parameter.name, e.target.value)}
            disabled={isDisabled}
            className={style.class_name}
          />
        </div>
      );

    case "code":
      return (
        <Textarea
          value={String(value ?? parameter.default ?? "")}
          onChange={(e) => onChange(parameter.name, e.target.value)}
          placeholder={placeholder}
          disabled={isDisabled}
          className={`font-mono text-sm ${style.class_name || ""}`}
          style={{ height: style.height ? `${style.height}px` : "120px" }}
        />
      );

    case "markdown":
      return (
        <Textarea
          value={String(value ?? parameter.default ?? "")}
          onChange={(e) => onChange(parameter.name, e.target.value)}
          placeholder={placeholder}
          disabled={isDisabled}
          className={style.class_name}
          style={{ height: style.height ? `${style.height}px` : "120px" }}
        />
      );

    case "keyvalue":
      const kvValue = typeof value === "object" && value !== null ? value : {};
      return (
        <div className="space-y-2">
          {Object.entries(kvValue as Record<string, string>).map(([k, v]) => (
            <div key={k} className="flex gap-2">
              <Input
                type="text"
                value={k}
                onChange={(e) => {
                  const newKv = { ...(kvValue as Record<string, string>) };
                  delete newKv[k];
                  newKv[e.target.value] = v as string;
                  onChange(parameter.name, newKv);
                }}
                placeholder="Key"
                disabled={isDisabled}
              />
              <Input
                type="text"
                value={v as string}
                onChange={(e) => {
                  const newKv = { ...(kvValue as Record<string, string>) };
                  newKv[k] = e.target.value;
                  onChange(parameter.name, newKv);
                }}
                placeholder="Value"
                disabled={isDisabled}
              />
              <Button
                variant="secondary"
                size="icon"
                disabled={isDisabled}
                onClick={() => {
                  const newKv = { ...(kvValue as Record<string, string>) };
                  delete newKv[k];
                  onChange(parameter.name, newKv);
                }}
              >
                ×
              </Button>
            </div>
          ))}
          <Button
            variant="secondary"
            size="sm"
            disabled={isDisabled}
            onClick={() => {
              const newKv = { ...(kvValue as Record<string, string>) };
              newKv[""] = "";
              onChange(parameter.name, newKv);
            }}
          >
            Add Entry
          </Button>
        </div>
      );

    case "list":
      const listValue = Array.isArray(value) ? value : 
        (value ? String(value).split(",") : []);
      return (
        <div className="space-y-2">
          {listValue.map((item: string, index: number) => (
            <div key={index} className="flex gap-2">
              <Input
                type="text"
                value={item}
                onChange={(e) => {
                  const newList = [...listValue];
                  newList[index] = e.target.value;
                  onChange(parameter.name, newList);
                }}
                placeholder={`Item ${index + 1}`}
                disabled={isDisabled}
              />
              <Button
                variant="secondary"
                size="icon"
                disabled={isDisabled}
                onClick={() => {
                  const newList = listValue.filter((_, i) => i !== index);
                  onChange(parameter.name, newList);
                }}
              >
                ×
              </Button>
            </div>
          ))}
          <Button
            variant="secondary"
            size="sm"
            disabled={isDisabled}
            onClick={() => {
              onChange(parameter.name, [...listValue, ""]);
            }}
          >
            Add Item
          </Button>
        </div>
      );

    case "custom":
      return (
        <div className="p-4 border rounded-md bg-muted/50">
          <p className="text-sm text-muted-foreground">
            Custom component: {widget?.custom_component || "Not specified"}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Props: {JSON.stringify(widget?.custom_props || {})}
          </p>
        </div>
      );

    default:
      return (
        <Input
          type="text"
          value={String(value ?? parameter.default ?? "")}
          onChange={(e) => onChange(parameter.name, e.target.value)}
          placeholder={placeholder}
          disabled={isDisabled}
          className={style.class_name}
        />
      );
  }
}

function getDefaultWidgetType(paramType: string): string {
  const mapping: Record<string, string> = {
    string: "text",
    integer: "number",
    float: "number",
    boolean: "toggle",
    select: "select",
    path: "directory",
  };
  return mapping[paramType] || "text";
}

function shouldDisable(widget: WidgetConfig | undefined, allValues: Record<string, unknown>): boolean {
  if (!widget?.disabled_if) return false;
  return evaluateCondition(widget.disabled_if, allValues);
}

function shouldHide(widget: WidgetConfig | undefined, allValues: Record<string, unknown>): boolean {
  if (!widget?.show_if) return false;
  return !evaluateCondition(widget.show_if, allValues);
}

function evaluateCondition(condition: string, values: Record<string, unknown>): boolean {
  try {
    const parts = condition.split("==");
    if (parts.length === 2) {
      const paramName = parts[0].trim();
      const expectedValue = parts[1].trim().replace(/['"]/g, "");
      return String(values[paramName]) === expectedValue;
    }
    return false;
  } catch {
    return false;
  }
}

interface DynamicFieldProps {
  parameter: DynamicParameter;
  value: unknown;
  onChange: (name: string, value: unknown) => void;
  options?: DynamicOption[];
  disabled?: boolean;
  allValues?: Record<string, unknown>;
}

export function DynamicField({
  parameter,
  value,
  onChange,
  options,
  disabled,
  allValues,
}: DynamicFieldProps) {
  return (
    <div className="space-y-2">
      <Label className={parameter.required ? "after:content-['*'] after:text-red-500" : ""}>
        {parameter.name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
      </Label>
      <DynamicWidget
        parameter={parameter}
        value={value}
        onChange={onChange}
        options={options}
        disabled={disabled}
        allValues={allValues}
      />
      {parameter.description && (
        <p className="text-xs text-muted-foreground">{parameter.description}</p>
      )}
    </div>
  );
}
