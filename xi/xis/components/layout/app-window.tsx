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
import { Button } from "@/components/ui/button";
import { Minimize2, Maximize2, X } from "lucide-react";
import { cn } from "@/lib/utils";

const HEADER_HEIGHT = 48;
const MIN_WIDTH = 400;
const MIN_HEIGHT = 300;

interface AppWindowProps {
  appId: string;
  title: string;
  icon: ReactNode;
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
}

export function AppWindow({
  appId,
  title,
  icon,
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
}: AppWindowProps) {
  const windowRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: -9999, y: -9999 });
  const [size, setSize] = useState(savedSize || defaultSize);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const resizeStart = useRef({ x: 0, y: 0, width: 0, height: 0 });
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
        newPosition = {
          x: Math.max(0, (viewportWidth - size.width) / 2),
          y: Math.max(HEADER_HEIGHT, (viewportHeight - size.height) / 2),
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
      if (isResizing) {
        const newWidth = Math.max(MIN_WIDTH, resizeStart.current.width + (e.clientX - resizeStart.current.x));
        const newHeight = Math.max(MIN_HEIGHT, resizeStart.current.height + (e.clientY - resizeStart.current.y));
        setSize({ width: newWidth, height: newHeight });
      }
    },
    [isDragging, isResizing]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    setIsResizing(false);
    if (onPositionChange && !isMaximized) {
      onPositionChange(position);
    }
    if (onSizeChange && !isMaximized) {
      onSizeChange(size);
    }
  }, [position, size, onPositionChange, onSizeChange, isMaximized]);

  useEffect(() => {
    if (isDragging || isResizing) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      window.addEventListener("mouseleave", handleMouseUp);
      document.body.style.userSelect = "none";
      document.body.style.cursor = isResizing ? "se-resize" : "move";
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

  const handleResizeStart = useCallback((e: MouseEvent<HTMLDivElement>) => {
    if (isMaximized) return;
    e.preventDefault();
    e.stopPropagation();
    resizeStart.current = {
      x: e.clientX,
      y: e.clientY,
      width: size.width,
      height: size.height,
    };
    setIsResizing(true);
  }, [isMaximized, size]);

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
          "flex items-center justify-between px-4 py-3 border-b border-border/50 select-none",
          isMaximized ? "" : "cursor-move"
        )}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleToggleMaximize}
      >
        <div className="flex items-center gap-2">
          {icon}
          <span className="font-semibold">{title}</span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="secondary"
            size="icon"
            className="h-8 w-8 bg-muted/50 hover:bg-muted"
            onClick={onMinimize}
          >
            <Minimize2 className="h-4 w-4" />
          </Button>
          <Button
            variant="secondary"
            size="icon"
            className="h-8 w-8 bg-muted/50 hover:bg-muted"
            onClick={handleToggleMaximize}
          >
            <Maximize2 className="h-4 w-4" />
          </Button>
          <Button
            variant="secondary"
            size="icon"
            className="h-8 w-8 bg-muted/50 hover:bg-muted"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-hidden relative">
        {children}
      </div>

      {!isMaximized && (
        <div
          className="resize-handle"
          onMouseDown={handleResizeStart}
          title="Resize"
        />
      )}
    </div>
  );
}
