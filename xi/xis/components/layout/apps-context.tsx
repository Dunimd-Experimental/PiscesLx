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

import { createContext, useContext, useState, ReactNode } from "react";

type AppState = "closed" | "minimized" | "normal" | "maximized";

interface AppPosition {
  x: number;
  y: number;
}

interface App {
  id: string;
  name: string;
  state: AppState;
  position: AppPosition | null;
  size: { width: number; height: number } | null;
  hidden?: boolean;
  params?: Record<string, unknown>;
}

interface AppsContextType {
  apps: App[];
  focusedAppId: string | null;
  openApp: (id: string, hidden?: boolean) => void;
  openAppWithParams: (id: string, params: Record<string, unknown>) => void;
  closeApp: (id: string) => void;
  minimizeApp: (id: string) => void;
  maximizeApp: (id: string) => void;
  restoreApp: (id: string) => void;
  focusApp: (id: string) => void;
  isAppRunning: (id: string) => boolean;
  isAppMaximized: (id: string) => boolean;
  isAppFocused: (id: string) => boolean;
  updateAppPosition: (id: string, position: AppPosition) => void;
  getAppPosition: (id: string) => AppPosition | null;
  updateAppSize: (id: string, size: { width: number; height: number }) => void;
  getAppSize: (id: string) => { width: number; height: number } | null;
  getAppParams: (id: string) => Record<string, unknown> | null;
  clearAppParams: (id: string) => void;
}

const AppsContext = createContext<AppsContextType | undefined>(undefined);

export function AppsProvider({ children }: { children: ReactNode }) {
  const [apps, setApps] = useState<App[]>([
    { id: "monitor", name: "System Monitor", state: "closed", position: null, size: null },
    { id: "explorer", name: "File Explorer", state: "closed", position: null, size: null },
    { id: "new-run", name: "New Run", state: "closed", position: null, size: null, hidden: true },
  ]);
  const [focusedAppId, setFocusedAppId] = useState<string | null>(null);

  const openApp = (id: string, hidden?: boolean) => {
    setApps((prev) =>
      prev.map((app) => {
        if (app.id === id) {
          if (app.state === "closed") {
            return { ...app, state: "normal", position: null };
          }
          return { ...app, state: "normal" };
        }
        return app;
      })
    );
    setFocusedAppId(id);
  };

  const openAppWithParams = (id: string, params: Record<string, unknown>) => {
    setApps((prev) =>
      prev.map((app) => {
        if (app.id === id) {
          if (app.state === "closed") {
            return { ...app, state: "normal", position: null, params };
          }
          return { ...app, state: "normal", params };
        }
        return app;
      })
    );
    setFocusedAppId(id);
  };

  const closeApp = (id: string) => {
    setApps((prev) =>
      prev.map((app) =>
        app.id === id ? { ...app, state: "closed", position: null, size: null } : app
      )
    );
  };

  const minimizeApp = (id: string) => {
    setApps((prev) =>
      prev.map((app) =>
        app.id === id ? { ...app, state: "minimized" } : app
      )
    );
  };

  const maximizeApp = (id: string) => {
    setApps((prev) =>
      prev.map((app) =>
        app.id === id ? { ...app, state: "maximized" } : app
      )
    );
  };

  const restoreApp = (id: string) => {
    setApps((prev) =>
      prev.map((app) =>
        app.id === id ? { ...app, state: "normal" } : app
      )
    );
  };

  const isAppRunning = (id: string) => {
    const app = apps.find((a) => a.id === id);
    return app ? app.state !== "closed" : false;
  };

  const isAppMaximized = (id: string) => {
    const app = apps.find((a) => a.id === id);
    return app?.state === "maximized";
  };

  const updateAppPosition = (id: string, position: AppPosition) => {
    setApps((prev) =>
      prev.map((app) =>
        app.id === id ? { ...app, position } : app
      )
    );
  };

  const getAppPosition = (id: string) => {
    const app = apps.find((a) => a.id === id);
    return app?.position || null;
  };

  const updateAppSize = (id: string, size: { width: number; height: number }) => {
    setApps((prev) =>
      prev.map((app) =>
        app.id === id ? { ...app, size } : app
      )
    );
  };

  const getAppSize = (id: string) => {
    const app = apps.find((a) => a.id === id);
    return app?.size || null;
  };

  const focusApp = (id: string) => {
    setFocusedAppId(id);
  };

  const isAppFocused = (id: string) => {
    return focusedAppId === id;
  };

  const getAppParams = (id: string) => {
    const app = apps.find((a) => a.id === id);
    return app?.params || null;
  };

  const clearAppParams = (id: string) => {
    setApps((prev) =>
      prev.map((app) =>
        app.id === id ? { ...app, params: undefined } : app
      )
    );
  };

  return (
    <AppsContext.Provider
      value={{ 
        apps, 
        focusedAppId,
        openApp, 
        openAppWithParams,
        closeApp, 
        minimizeApp, 
        maximizeApp, 
        restoreApp, 
        focusApp,
        isAppRunning, 
        isAppMaximized,
        isAppFocused,
        updateAppPosition, 
        getAppPosition, 
        updateAppSize, 
        getAppSize,
        getAppParams,
        clearAppParams
      }}
    >
      {children}
    </AppsContext.Provider>
  );
}

export function useApps() {
  const context = useContext(AppsContext);
  if (context === undefined) {
    throw new Error("useApps must be used within an AppsProvider");
  }
  return context;
}
