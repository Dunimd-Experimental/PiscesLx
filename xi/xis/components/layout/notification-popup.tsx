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

import { useState, useEffect, useRef } from "react";
import { Bell, X, Check, AlertCircle, Info, AlertTriangle, Calendar, Sun, Cloud } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useNotifications } from "@/lib/stores/config-store";
import type { NotificationType, XiNotification } from "@/types/config";

const getNotificationIcon = (type: NotificationType) => {
  switch (type) {
    case "success":
      return <Check className="h-4 w-4 text-green-500" />;
    case "warning":
      return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
    case "error":
      return <AlertCircle className="h-4 w-4 text-red-500" />;
    default:
      return <Info className="h-4 w-4 text-blue-500" />;
  }
};

const formatTime = (timeStr: string) => {
  const date = new Date(timeStr);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);

  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return date.toLocaleDateString();
};

interface NotificationPopupProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLButtonElement>;
}

export function NotificationPopup({ isOpen, onClose, anchorRef }: NotificationPopupProps) {
  const [position, setPosition] = useState({ right: 8, top: 48 });
  const popupRef = useRef<HTMLDivElement>(null);

  const {
    notifications,
    unreadCount,
    fetchNotifications,
    markNotificationRead,
    deleteNotification,
    clearAllNotifications,
  } = useNotifications();

  useEffect(() => {
    if (isOpen) {
      fetchNotifications();
    }
  }, [isOpen, fetchNotifications]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        popupRef.current &&
        !popupRef.current.contains(event.target as Node) &&
        anchorRef.current &&
        !anchorRef.current.contains(event.target as Node)
      ) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen, onClose, anchorRef]);

  useEffect(() => {
    if (isOpen && anchorRef.current) {
      const rect = anchorRef.current.getBoundingClientRect();
      setPosition({
        right: window.innerWidth - rect.right,
        top: rect.bottom + 4,
      });
    }
  }, [isOpen, anchorRef]);

  const handleMarkAsRead = async (id: string) => {
    await markNotificationRead(id);
  };

  const handleClearNotification = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    await deleteNotification(id);
  };

  const handleClearAll = async () => {
    await clearAllNotifications();
  };

  if (!isOpen) return null;

  return (
    <div
      ref={popupRef}
      className="notification-popup acrylic shadow-2xl animate-fade-in overflow-hidden"
      style={{
        '--popup-right': `${position.right}px`,
        '--popup-top': `${position.top}px`,
      } as React.CSSProperties}
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Bell className="h-4 w-4" />
          <span className="font-semibold text-sm">Notifications</span>
          {unreadCount > 0 && (
            <span className="flex h-5 min-w-5 items-center justify-center rounded-full bg-primary px-1.5 text-xs font-medium text-primary-foreground">
              {unreadCount}
            </span>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 px-2 text-xs"
          onClick={handleClearAll}
          disabled={notifications.length === 0}
        >
          Clear all
        </Button>
      </div>

      <div className="max-h-80 overflow-y-auto">
        {notifications.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
            <Bell className="h-10 w-10 mb-2 opacity-50" />
            <p className="text-sm">No notifications</p>
          </div>
        ) : (
          <div className="divide-y divide-border/50">
            {notifications.map((notification: XiNotification) => (
              <div
                key={notification.id}
                className={cn(
                  "flex gap-3 p-3 transition-colors hover:bg-muted/50 cursor-pointer",
                  !notification.read && "bg-primary/5"
                )}
                onClick={() => handleMarkAsRead(notification.id)}
              >
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-muted">
                  {getNotificationIcon(notification.type)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-sm font-medium truncate">{notification.title}</p>
                    <button
                      onClick={(e) => handleClearNotification(e, notification.id)}
                      className="flex-shrink-0 p-1 rounded hover:bg-muted"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                  <p className="text-xs text-muted-foreground line-clamp-2 mt-0.5">
                    {notification.message}
                  </p>
                  <p className="text-xs text-muted-foreground/70 mt-1">
                    {formatTime(notification.time)}
                  </p>
                </div>
                {!notification.read && (
                  <div className="w-2 h-2 rounded-full bg-primary flex-shrink-0 mt-1.5" />
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="border-t border-border/50 p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">
              {new Date().toLocaleDateString("en-US", {
                weekday: "long",
                month: "long",
                day: "numeric",
              })}
            </span>
          </div>
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Sun className="h-3 w-3" />
            <span>24°C</span>
            <Cloud className="h-3 w-3 ml-1" />
          </div>
        </div>
      </div>
    </div>
  );
}
