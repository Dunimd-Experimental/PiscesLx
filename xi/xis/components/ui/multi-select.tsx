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

import * as React from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

interface MultiSelectProps {
  values: string[];
  onValuesChange: (values: string[]) => void;
  children: React.ReactNode;
  disabled?: boolean;
}

interface MultiSelectTriggerProps {
  children: React.ReactNode;
  className?: string;
}

interface MultiSelectContentProps {
  children: React.ReactNode;
}

interface MultiSelectItemProps {
  value: string;
  children: React.ReactNode;
}

interface MultiSelectValueProps {
  placeholder?: string;
}

const MultiSelectContext = React.createContext<{
  values: string[];
  onValuesChange: (values: string[]) => void;
  open: boolean;
  setOpen: (open: boolean) => void;
}>({
  values: [],
  onValuesChange: () => {},
  open: false,
  setOpen: () => {},
});

export function MultiSelect({ values, onValuesChange, children, disabled }: MultiSelectProps) {
  const [open, setOpen] = React.useState(false);

  return (
    <MultiSelectContext.Provider value={{ values, onValuesChange, open, setOpen }}>
      <div className="relative" data-disabled={disabled}>
        {children}
      </div>
    </MultiSelectContext.Provider>
  );
}

export function MultiSelectTrigger({ children, className }: MultiSelectTriggerProps) {
  const { open, setOpen, values } = React.useContext(MultiSelectContext);

  return (
    <button
      type="button"
      onClick={() => setOpen(!open)}
      className={cn(
        "flex min-h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
    >
      {children}
    </button>
  );
}

export function MultiSelectValue({ placeholder }: MultiSelectValueProps) {
  const { values } = React.useContext(MultiSelectContext);

  if (values.length === 0) {
    return <span className="text-muted-foreground">{placeholder}</span>;
  }

  return (
    <div className="flex flex-wrap gap-1">
      {values.map((value) => (
        <Badge key={value} variant="secondary" className="mr-1">
          {value}
        </Badge>
      ))}
    </div>
  );
}

export function MultiSelectContent({ children }: MultiSelectContentProps) {
  const { open, setOpen } = React.useContext(MultiSelectContext);

  if (!open) return null;

  return (
    <>
      <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
      <div className="absolute z-50 mt-1 w-full rounded-md border bg-popover p-1 text-popover-foreground shadow-md">
        <div className="max-h-60 overflow-auto">{children}</div>
      </div>
    </>
  );
}

export function MultiSelectItem({ value, children }: MultiSelectItemProps) {
  const { values, onValuesChange } = React.useContext(MultiSelectContext);
  const isSelected = values.includes(value);

  const handleClick = () => {
    if (isSelected) {
      onValuesChange(values.filter((v) => v !== value));
    } else {
      onValuesChange([...values, value]);
    }
  };

  return (
    <div
      onClick={handleClick}
      className={cn(
        "relative flex cursor-pointer select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none hover:bg-accent hover:text-accent-foreground",
        isSelected && "bg-accent"
      )}
    >
      <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
        {isSelected && <X className="h-3 w-3" />}
      </span>
      {children}
    </div>
  );
}
