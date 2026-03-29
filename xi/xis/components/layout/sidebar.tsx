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

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useSidebar } from "./sidebar-context";
import { useApps } from "./apps-context";
import { Monitor } from "lucide-react";
import { Button } from "@/components/ui/button";

import {
  LayoutDashboard,
  Brain,
  MessageSquare,
  Database,
  Cpu,
  Play,
  Folder,
} from "lucide-react";

const navigation = [
  { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { name: "Training", href: "/training", icon: Brain },
  { name: "Inference", href: "/inference", icon: MessageSquare },
  { name: "Data", href: "/data", icon: Database },
  { name: "Models", href: "/models", icon: Cpu },
  { name: "Runs", href: "/runs", icon: Play },
];

const apps = [
  { id: "monitor", name: "Monitor", icon: Monitor, color: "bg-blue-500" },
  { id: "explorer", name: "Explorer", icon: Folder, color: "bg-amber-500" },
];

export function Sidebar() {
  const pathname = usePathname();
  const { collapsed } = useSidebar();
  const { openApp, isAppRunning } = useApps();
  const [showApps, setShowApps] = useState(false);
  const appsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (appsRef.current && !appsRef.current.contains(event.target as Node)) {
        setShowApps(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <aside
      className={cn(
        "sidebar",
        collapsed ? "sidebar--collapsed" : "sidebar--expanded"
      )}
    >
      <nav className="sidebar__nav">
        {navigation.map((item) => {
          const isActive = pathname.startsWith(item.href);
          const Icon = item.icon;

          return (
            <Tooltip key={item.name} delayDuration={0}>
              <TooltipTrigger asChild>
                <Link
                  href={item.href}
                  className={cn(
                    "sidebar__item",
                    isActive && "sidebar__item--active",
                    collapsed && "sidebar__item--collapsed"
                  )}
                >
                  <Icon className="h-5 w-5 flex-shrink-0" />
                  {!collapsed && <span>{item.name}</span>}
                </Link>
              </TooltipTrigger>
              {collapsed && (
                <TooltipContent side="right" className="font-medium">
                  {item.name}
                </TooltipContent>
              )}
            </Tooltip>
          );
        })}
      </nav>

      <div className="sidebar__footer">
        <div className="relative" ref={appsRef}>
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
              <Button
                variant="secondary"
                className={cn(
                  "sidebar__apps-btn",
                  collapsed && "sidebar__apps-btn--collapsed"
                )}
                onClick={() => setShowApps(!showApps)}
              >
                <div className="grid grid-cols-2 gap-0.5">
                  <div className="h-1.5 w-1.5 rounded-sm bg-current" />
                  <div className="h-1.5 w-1.5 rounded-sm bg-current" />
                  <div className="h-1.5 w-1.5 rounded-sm bg-current" />
                  <div className="h-1.5 w-1.5 rounded-sm bg-current" />
                </div>
                {!collapsed && <span className="ml-3 text-sm">Apps</span>}
              </Button>
            </TooltipTrigger>
            {collapsed && (
              <TooltipContent side="right" className="font-medium">
                Apps
              </TooltipContent>
            )}
          </Tooltip>

          {showApps && (
            <div
              className="fixed z-50 h-64 w-80 rounded-xl acrylic p-4 shadow-xl apps-popup"
              style={{
                '--apps-popup-left': collapsed ? '72px' : '200px',
                '--apps-popup-bottom': '8px',
              } as React.CSSProperties}
            >
              <div className="sidebar__apps-grid">
                {apps.map((app) => {
                  const Icon = app.icon;
                  const isRunning = isAppRunning(app.id);

                  return (
                    <button
                      key={app.id}
                      onClick={() => {
                        setShowApps(false);
                        openApp(app.id);
                      }}
                      className="sidebar__app-item relative"
                    >
                      <div className={`sidebar__app-icon sidebar__app-icon--${app.id === 'monitor' ? 'blue' : 'amber'}`}>
                        <Icon className="h-6 w-6 text-white" />
                      </div>
                      {isRunning && (
                        <span className="sidebar__app-indicator" />
                      )}
                      <span className="sidebar__app-label">{app.name}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
