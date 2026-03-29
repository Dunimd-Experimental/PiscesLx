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

import { useEffect, useState, useCallback, useRef } from "react";
import { ChevronLeft, ChevronRight, Wifi, WifiOff, Volume2, VolumeX, Bell } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSidebar } from "./sidebar-context";
import { NotificationPopup } from "./notification-popup";

export function Header() {
  const { collapsed, toggle } = useSidebar();
  const [time, setTime] = useState({ hours: "", minutes: "", dayPeriod: "" });
  const [date, setDate] = useState({ month: "", day: "", year: "" });
  const [isOnline, setIsOnline] = useState(true);
  const [isMuted, setIsMuted] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const notificationButtonRef = useRef<HTMLButtonElement>(null);

  const updateTime = useCallback(() => {
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes();

    setTime({
      hours: hours.toString().padStart(2, "0"),
      minutes: minutes.toString().padStart(2, "0"),
      dayPeriod: "",
    });

    setDate({
      month: (now.getMonth() + 1).toString().padStart(2, "0"),
      day: now.getDate().toString().padStart(2, "0"),
      year: now.getFullYear().toString(),
    });
  }, []);

  useEffect(() => {
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, [updateTime]);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    setIsOnline(navigator.onLine);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  return (
    <>
      <header className="header">
        <div className="header__left">
          <Button
            variant="secondary"
            size="icon"
            onClick={toggle}
            className="h-8 w-8 bg-muted/50 hover:bg-muted"
          >
            {collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>
        </div>

        <div className="header__right">
          <Button
            variant="secondary"
            size="icon"
            className="h-8 w-8 bg-muted/50 hover:bg-muted"
            onClick={() => setIsOnline(!isOnline)}
            title={isOnline ? "Network Connected" : "Network Disconnected"}
          >
            {isOnline ? (
              <Wifi className="h-4 w-4" />
            ) : (
              <WifiOff className="h-4 w-4" />
            )}
          </Button>

          <Button
            variant="secondary"
            size="icon"
            className="h-8 w-8 bg-muted/50 hover:bg-muted"
            onClick={() => setIsMuted(!isMuted)}
            title={isMuted ? "Muted" : "Volume"}
          >
            {isMuted ? (
              <VolumeX className="h-4 w-4" />
            ) : (
              <Volume2 className="h-4 w-4" />
            )}
          </Button>

          <Button
            ref={notificationButtonRef}
            variant="secondary"
            className="flex h-8 items-center justify-center gap-2 px-3 bg-muted/50 hover:bg-muted"
            onClick={() => setShowNotifications(!showNotifications)}
            title="Notifications"
          >
            <span className="text-sm font-medium leading-none">
              {time.hours}:{time.minutes}
            </span>
            <Bell className="h-4 w-4" />
          </Button>
        </div>
      </header>

      <NotificationPopup
        isOpen={showNotifications}
        onClose={() => setShowNotifications(false)}
        anchorRef={notificationButtonRef}
      />
    </>
  );
}
