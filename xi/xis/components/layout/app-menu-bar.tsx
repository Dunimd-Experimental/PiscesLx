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

import { ReactNode, useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";

interface MenuItem {
  label?: string;
  onClick?: () => void;
  shortcut?: string;
  disabled?: boolean;
  divider?: boolean;
}

interface MenuGroup {
  label: string;
  items: MenuItem[];
}

interface AppMenuBarProps {
  groups: MenuGroup[];
  className?: string;
}

export function AppMenuBar({ groups, className }: AppMenuBarProps) {
  const [openGroup, setOpenGroup] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpenGroup(null);
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpenGroup(null);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const handleGroupClick = (label: string) => {
    setOpenGroup(openGroup === label ? null : label);
  };

  const handleItemClick = (item: MenuItem) => {
    if (item.disabled) return;
    item.onClick?.();
    setOpenGroup(null);
  };

  return (
    <div ref={menuRef} className={cn("flex items-center gap-0.5", className)}>
      {groups.map((group) => (
        <div key={group.label} className="relative">
          <button
            className={cn(
              "px-3 py-1 text-sm rounded transition-colors",
              openGroup === group.label
                ? "bg-muted/70"
                : "hover:bg-muted/50"
            )}
            onClick={() => handleGroupClick(group.label)}
          >
            {group.label}
          </button>

          {openGroup === group.label && (
            <div className="absolute top-full left-0 mt-1 min-w-[180px] bg-popover border border-border rounded-md shadow-lg py-1 z-50">
              {group.items.map((item, index) => (
                item.divider ? (
                  <div key={index} className="h-px bg-border my-1 mx-2" />
                ) : (
                  <button
                    key={index}
                    className={cn(
                      "w-full px-3 py-1.5 text-sm text-left flex items-center justify-between transition-colors",
                      item.disabled
                        ? "text-muted-foreground cursor-not-allowed"
                        : "hover:bg-muted/50"
                    )}
                    onClick={() => handleItemClick(item)}
                    disabled={item.disabled}
                  >
                    <span>{item.label}</span>
                    {item.shortcut && (
                      <span className="text-xs text-muted-foreground ml-4">
                        {item.shortcut}
                      </span>
                    )}
                  </button>
                )
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export type { MenuItem, MenuGroup };
