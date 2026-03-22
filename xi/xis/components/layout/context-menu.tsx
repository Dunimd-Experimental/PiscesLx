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

import { createContext, useContext, useState, useCallback, ReactNode, useEffect, useRef } from "react";

interface ContextMenuItem {
  id: string;
  label: string;
  icon?: ReactNode;
  shortcut?: string;
  disabled?: boolean;
  divider?: boolean;
  danger?: boolean;
  onClick?: () => void;
}

interface ContextMenuState {
  x: number;
  y: number;
  items: ContextMenuItem[];
  visible: boolean;
}

interface ContextMenuContextType {
  showContextMenu: (x: number, y: number, items: ContextMenuItem[]) => void;
  hideContextMenu: () => void;
}

const ContextMenuContext = createContext<ContextMenuContextType | null>(null);

export function useContextMenu() {
  const context = useContext(ContextMenuContext);
  if (!context) {
    throw new Error("useContextMenu must be used within a ContextMenuProvider");
  }
  return context;
}

interface ContextMenuProviderProps {
  children: ReactNode;
}

export function ContextMenuProvider({ children }: ContextMenuProviderProps) {
  const [menuState, setMenuState] = useState<ContextMenuState>({
    x: 0,
    y: 0,
    items: [],
    visible: false,
  });
  const menuRef = useRef<HTMLDivElement>(null);

  const showContextMenu = useCallback((x: number, y: number, items: ContextMenuItem[]) => {
    setMenuState({ x, y, items, visible: true });
  }, []);

  const hideContextMenu = useCallback(() => {
    setMenuState((prev) => ({ ...prev, visible: false }));
  }, []);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        hideContextMenu();
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        hideContextMenu();
      }
    };

    if (menuState.visible) {
      document.addEventListener("mousedown", handleClickOutside);
      document.addEventListener("keydown", handleKeyDown);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [menuState.visible, hideContextMenu]);

  const handleItemClick = useCallback((item: ContextMenuItem) => {
    if (item.disabled) return;
    hideContextMenu();
    item.onClick?.();
  }, [hideContextMenu]);

  return (
    <ContextMenuContext.Provider value={{ showContextMenu, hideContextMenu }}>
      {children}
      {menuState.visible && (
        <div
          ref={menuRef}
          className="context-menu"
          style={{
            '--context-menu-left': `${Math.min(menuState.x, window.innerWidth - 220)}px`,
            '--context-menu-top': `${Math.min(menuState.y, window.innerHeight - (menuState.items.length * 36 + 16))}px`,
          } as React.CSSProperties}
        >
          {menuState.items.map((item, index) => (
            item.divider ? (
              <div key={`divider-${index}`} className="my-1 h-px bg-border/50" />
            ) : (
              <button
                key={item.id}
                className={`
                  w-full flex items-center gap-3 px-3 py-2 text-sm text-left
                  transition-colors duration-100
                  ${item.disabled 
                    ? "opacity-50 cursor-not-allowed" 
                    : item.danger
                      ? "text-red-500 hover:bg-red-500/10"
                      : "hover:bg-muted/50"
                  }
                `}
                onClick={() => handleItemClick(item)}
                disabled={item.disabled}
              >
                {item.icon && <span className="w-4 h-4 flex items-center justify-center">{item.icon}</span>}
                <span className="flex-1">{item.label}</span>
                {item.shortcut && (
                  <span className="text-xs text-muted-foreground">{item.shortcut}</span>
                )}
              </button>
            )
          ))}
        </div>
      )}
    </ContextMenuContext.Provider>
  );
}
