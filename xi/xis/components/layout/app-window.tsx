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

import { useRef, useState, useCallback, ReactNode, MouseEvent, useEffect } from "react";
import { X, Minus, Plus, PanelLeft, PanelLeftClose } from "lucide-react";
import { cn } from "@/lib/utils";
import { AppMenuBar, MenuGroup } from "./app-menu-bar";

const HEADER_HEIGHT = 48;
const MIN_WIDTH = 400;
const MIN_HEIGHT = 300;
const RESIZE_BORDER_WIDTH = 6;

type ResizeDirection = "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw" | null;

interface AppWindowProps {
  appId: string;
  title?: string;
  icon?: ReactNode;
  children: ReactNode;
  defaultSize?: { width: number; height: number };
  onMinimize: () => void;
  onClose: () => void;
  savedPosition?: { x: number; y: number } | null;
  onPositionChange?: (position: { x: number; y: number }) => void;
  savedSize?: { width: number; height: number } | null;
  onSizeChange?: (size: { width: number; height: number }) => void;
  isMaximized?: boolean;
  onMaximize?: () => void;
  onRestore?: () => void;
  isFocused?: boolean;
  onFocus?: () => void;
  sidebarCollapsed?: boolean;
  onSidebarToggle?: () => void;
  menuGroups?: MenuGroup[];
}

export function AppWindow({
  appId,
  children,
  defaultSize = { width: 800, height: 600 },
  onMinimize,
  onClose,
  savedPosition,
  onPositionChange,
  savedSize,
  onSizeChange,
  isMaximized = false,
  onMaximize,
  onRestore,
  isFocused = false,
  onFocus,
  sidebarCollapsed,
  onSidebarToggle,
  menuGroups,
}: AppWindowProps) {
  const windowRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: -9999, y: -9999 });
  const [size, setSize] = useState(savedSize || defaultSize);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [resizeDirection, setResizeDirection] = useState<ResizeDirection>(null);
  const [isReady, setIsReady] = useState(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const resizeStart = useRef({ x: 0, y: 0, width: 0, height: 0, posX: 0, posY: 0, direction: null as ResizeDirection });
  const hasInitializedPosition = useRef(false);
  const preMaximizeState = useRef<{ position: { x: number; y: number }; size: { width: number; height: number } } | null>(null);

  useEffect(() => {
    if (hasInitializedPosition.current && savedPosition) return;

    const calculatePosition = () => {
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      let newPosition: { x: number; y: number };

      if (savedPosition) {
        newPosition = {
          x: Math.max(0, Math.min(savedPosition.x, viewportWidth - size.width)),
          y: Math.max(HEADER_HEIGHT, Math.min(savedPosition.y, viewportHeight - size.height)),
        };
      } else {
        const allWindows = document.querySelectorAll('.app-window');
        const OFFSET = 30;
        let offsetX = 0;
        let offsetY = 0;
        
        if (allWindows.length > 0) {
          offsetX = (allWindows.length - 1) * OFFSET;
          offsetY = (allWindows.length - 1) * OFFSET;
        }
        
        const centerX = (viewportWidth - size.width) / 2;
        const centerY = (viewportHeight - size.height) / 2;
        
        newPosition = {
          x: Math.max(0, Math.min(centerX + offsetX, viewportWidth - size.width - 20)),
          y: Math.max(HEADER_HEIGHT, Math.min(centerY + offsetY, viewportHeight - size.height - 20)),
        };
      }

      setPosition(newPosition);
      setIsReady(true);
      hasInitializedPosition.current = true;
    };

    const timer = setTimeout(calculatePosition, 0);
    return () => clearTimeout(timer);
  }, [savedPosition, size.width, size.height]);

  const handleMouseDown = useCallback((e: MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    if (target.closest(".window-controls") || target.closest(".window-control") || target.closest(".header-buttons")) return;
    e.preventDefault();
    
    if (onFocus) {
      onFocus();
    }
    
    const rect = windowRef.current?.getBoundingClientRect();
    if (!rect) return;

    if (isMaximized) {
      const relativeX = e.clientX - rect.left;
      const relativeWidth = rect.width;
      const newWidth = preMaximizeState.current?.size.width || defaultSize.width;
      const newX = Math.max(0, e.clientX - (relativeX / relativeWidth) * newWidth);
      const newY = Math.max(HEADER_HEIGHT, e.clientY - 20);
      
      preMaximizeState.current = {
        position: { x: newX, y: newY },
        size: preMaximizeState.current?.size || defaultSize,
      };
      
      setPosition({ x: newX, y: newY });
      setSize(preMaximizeState.current.size);
      if (onRestore) onRestore();
      
      dragOffset.current = {
        x: e.clientX - newX,
        y: e.clientY - newY,
      };
    } else {
      dragOffset.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
    }
    
    setIsDragging(true);
  }, [isMaximized, defaultSize, onRestore, onFocus]);

  const handleMouseMove = useCallback(
    (e: globalThis.MouseEvent) => {
      if (isDragging) {
        const newX = e.clientX - dragOffset.current.x;
        const newY = Math.max(HEADER_HEIGHT, e.clientY - dragOffset.current.y);
        setPosition({
          x: Math.max(0, newX),
          y: newY,
        });
      }
      if (isResizing && resizeStart.current.direction) {
        const dir = resizeStart.current.direction;
        const deltaX = e.clientX - resizeStart.current.x;
        const deltaY = e.clientY - resizeStart.current.y;
        
        let newWidth = resizeStart.current.width;
        let newHeight = resizeStart.current.height;
        let newX = resizeStart.current.posX;
        let newY = resizeStart.current.posY;

        if (dir.includes("e")) {
          newWidth = Math.max(MIN_WIDTH, resizeStart.current.width + deltaX);
        }
        if (dir.includes("w")) {
          const maxDeltaX = resizeStart.current.width - MIN_WIDTH;
          const actualDeltaX = Math.min(deltaX, maxDeltaX);
          newWidth = resizeStart.current.width - actualDeltaX;
          newX = resizeStart.current.posX + actualDeltaX;
        }
        if (dir.includes("s")) {
          newHeight = Math.max(MIN_HEIGHT, resizeStart.current.height + deltaY);
        }
        if (dir.includes("n")) {
          const maxDeltaY = resizeStart.current.height - MIN_HEIGHT;
          const actualDeltaY = Math.min(deltaY, maxDeltaY);
          newHeight = resizeStart.current.height - actualDeltaY;
          newY = Math.max(HEADER_HEIGHT, resizeStart.current.posY + actualDeltaY);
        }

        setSize({ width: newWidth, height: newHeight });
        setPosition({ x: newX, y: newY });
      }
    },
    [isDragging, isResizing]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    setIsResizing(false);
    setResizeDirection(null);
    if (onPositionChange && !isMaximized) {
      onPositionChange(position);
    }
    if (onSizeChange && !isMaximized) {
      onSizeChange(size);
    }
  }, [position, size, onPositionChange, onSizeChange, isMaximized]);

  const getCursorForDirection = (dir: ResizeDirection): string => {
    switch (dir) {
      case "n":
      case "s":
        return "ns-resize";
      case "e":
      case "w":
        return "ew-resize";
      case "ne":
      case "sw":
        return "nesw-resize";
      case "nw":
      case "se":
        return "nwse-resize";
      default:
        return "";
    }
  };

  useEffect(() => {
    if (isDragging || isResizing) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      window.addEventListener("mouseleave", handleMouseUp);
      document.body.style.userSelect = "none";
      document.body.style.cursor = isResizing ? getCursorForDirection(resizeStart.current.direction) : "move";
    } else {
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    }

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("mouseleave", handleMouseUp);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
  }, [isDragging, isResizing, handleMouseMove, handleMouseUp]);

  const handleToggleMaximize = useCallback(() => {
    if (isMaximized) {
      if (preMaximizeState.current && onRestore) {
        setPosition(preMaximizeState.current.position);
        setSize(preMaximizeState.current.size);
        onRestore();
      }
    } else {
      preMaximizeState.current = { position, size };
      if (onMaximize) {
        onMaximize();
      }
    }
  }, [isMaximized, position, size, onMaximize, onRestore]);

  const handleResizeStart = useCallback((direction: ResizeDirection) => (e: MouseEvent<HTMLDivElement>) => {
    if (isMaximized || !direction) return;
    e.preventDefault();
    e.stopPropagation();
    resizeStart.current = {
      x: e.clientX,
      y: e.clientY,
      width: size.width,
      height: size.height,
      posX: position.x,
      posY: position.y,
      direction,
    };
    setResizeDirection(direction);
    setIsResizing(true);
  }, [isMaximized, size, position]);

  const windowStyle = {
    '--window-left': isMaximized ? '0px' : `${position.x}px`,
    '--window-top': isMaximized ? `${HEADER_HEIGHT}px` : `${position.y}px`,
    '--window-width': isMaximized ? '100vw' : `${size.width}px`,
    '--window-height': isMaximized ? `calc(100vh - ${HEADER_HEIGHT}px)` : `${size.height}px`,
    '--window-opacity': isReady ? 1 : 0,
    '--window-pointer-events': isReady ? 'auto' : 'none',
  } as React.CSSProperties;

  return (
    <div
      ref={windowRef}
      className={cn(
        "app-window card-acrylic rounded-xl flex flex-col",
        isFocused ? "app-window--focused" : "app-window--normal",
        isMaximized && "app-window--maximized"
      )}
      style={windowStyle}
      onMouseDown={onFocus}
    >
      <div
        className={cn(
          "flex items-center px-4 py-2 border-b border-border/50 select-none relative",
          isMaximized ? "" : "cursor-move"
        )}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleToggleMaximize}
      >
        <div className="window-controls flex items-center gap-2 z-10">
          <button
            className="window-control window-control--close"
            onClick={onClose}
            title="Close"
          >
            <X className="window-control__icon" />
          </button>
          <button
            className="window-control window-control--minimize"
            onClick={onMinimize}
            title="Minimize"
          >
            <Minus className="window-control__icon" />
          </button>
          <button
            className="window-control window-control--maximize"
            onClick={handleToggleMaximize}
            title={isMaximized ? "Restore" : "Maximize"}
          >
            <Plus className="window-control__icon" />
          </button>
        </div>

        {onSidebarToggle && (
          <div className="header-buttons flex items-center z-10 ml-2">
            <button
              onClick={onSidebarToggle}
              className="p-1.5 rounded-md hover:bg-muted/50 transition-colors flex items-center justify-center"
              title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {sidebarCollapsed ? (
                <PanelLeft className="h-4 w-4 text-muted-foreground" />
              ) : (
                <PanelLeftClose className="h-4 w-4 text-muted-foreground" />
              )}
            </button>
          </div>
        )}

        {menuGroups && menuGroups.length > 0 && (
          <div className="header-buttons flex items-center z-10 ml-1">
            <AppMenuBar groups={menuGroups} />
          </div>
        )}
      </div>

      <div className="flex-1 overflow-hidden relative">
        {children}
      </div>

      {!isMaximized && (
        <>
          <div className="resize-handle resize-handle--n" onMouseDown={handleResizeStart("n")} />
          <div className="resize-handle resize-handle--s" onMouseDown={handleResizeStart("s")} />
          <div className="resize-handle resize-handle--e" onMouseDown={handleResizeStart("e")} />
          <div className="resize-handle resize-handle--w" onMouseDown={handleResizeStart("w")} />
          <div className="resize-handle resize-handle--ne" onMouseDown={handleResizeStart("ne")} />
          <div className="resize-handle resize-handle--nw" onMouseDown={handleResizeStart("nw")} />
          <div className="resize-handle resize-handle--se" onMouseDown={handleResizeStart("se")} />
          <div className="resize-handle resize-handle--sw" onMouseDown={handleResizeStart("sw")} />
        </>
      )}
    </div>
  );
}
