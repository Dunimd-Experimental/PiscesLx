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

import { ReactNode, useState } from "react";
import { PanelLeftClose, PanelLeft } from "lucide-react";
import { cn } from "@/lib/utils";

interface SidebarPanelProps {
  title?: string;
  children: ReactNode;
  defaultCollapsed?: boolean;
  width?: number;
  collapsedWidth?: number;
  onCollapsedChange?: (collapsed: boolean) => void;
  showToggleButton?: boolean;
  toggleButtonPosition?: "header" | "sidebar";
  renderToggleButton?: (collapsed: boolean, onToggle: () => void) => ReactNode;
}

export function SidebarPanel({
  title,
  children,
  defaultCollapsed = false,
  width = 224,
  collapsedWidth = 0,
  onCollapsedChange,
  showToggleButton = true,
  toggleButtonPosition = "sidebar",
  renderToggleButton,
}: SidebarPanelProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  const handleToggle = () => {
    const newState = !collapsed;
    setCollapsed(newState);
    onCollapsedChange?.(newState);
  };

  const currentWidth = collapsed ? collapsedWidth : width;

  return (
    <div
      className="border-r border-border/50 flex flex-col bg-muted/20 transition-all duration-200 overflow-hidden"
      style={{ width: currentWidth }}
    >
      {!collapsed && (
        <>
          {title && (
            <div className="p-3 border-b border-border/50">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                {title}
              </span>
            </div>
          )}
          <div className="flex-1 overflow-y-auto p-2">
            {children}
          </div>
        </>
      )}

      {showToggleButton && toggleButtonPosition === "sidebar" && !collapsed && (
        <button
          onClick={handleToggle}
          className="p-2 border-t border-border/50 hover:bg-muted/50 transition-colors flex items-center justify-center"
          title="Collapse sidebar"
        >
          <PanelLeftClose className="h-4 w-4 text-muted-foreground" />
        </button>
      )}

      {collapsed && collapsedWidth > 0 && showToggleButton && toggleButtonPosition === "sidebar" && (
        <button
          onClick={handleToggle}
          className="p-2 hover:bg-muted/50 transition-colors flex items-center justify-center h-full"
          title="Expand sidebar"
        >
          <PanelLeft className="h-4 w-4 text-muted-foreground" />
        </button>
      )}
    </div>
  );
}

export function useSidebarCollapse(defaultCollapsed = false) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  const toggle = () => setCollapsed((prev) => !prev);
  const collapse = () => setCollapsed(true);
  const expand = () => setCollapsed(false);

  return {
    collapsed,
    toggle,
    collapse,
    expand,
    setCollapsed,
  };
}

export function SidebarToggleButton({
  collapsed,
  onToggle,
  className,
}: {
  collapsed: boolean;
  onToggle: () => void;
  className?: string;
}) {
  return (
    <button
      onClick={onToggle}
      className={cn(
        "p-1.5 rounded-md hover:bg-muted/50 transition-colors flex items-center justify-center",
        className
      )}
      title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
    >
      {collapsed ? (
        <PanelLeft className="h-4 w-4 text-muted-foreground" />
      ) : (
        <PanelLeftClose className="h-4 w-4 text-muted-foreground" />
      )}
    </button>
  );
}
