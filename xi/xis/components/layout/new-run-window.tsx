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

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Play,
  Brain,
  MessageSquare,
  Download,
  Gauge,
  Package,
  CheckCircle,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { useApps } from "./apps-context";
import { AppWindow } from "./app-window";
import { useRunsStore } from "@/lib/stores";
import type { RunTypeInfo, RunTypeParameter } from "@/lib/api/runs-ws";
import { cn } from "@/lib/utils";
import { useSidebarCollapse } from "./sidebar-panel";
import { DynamicWidget } from "@/components/ui/dynamic-widget";
import type { DynamicParameter, WidgetType } from "@/types/dynamic";

interface NewRunWindowProps {
  state: "minimized" | "normal" | "maximized";
}

const iconMap: Record<string, React.ReactNode> = {
  Brain: <Brain className="h-5 w-5" />,
  MessageSquare: <MessageSquare className="h-5 w-5" />,
  Download: <Download className="h-5 w-5" />,
  Gauge: <Gauge className="h-5 w-5" />,
  Package: <Package className="h-5 w-5" />,
  Play: <Play className="h-5 w-5" />,
};

function convertToDynamicParameter(param: RunTypeParameter): DynamicParameter {
  return {
    name: param.name,
    type: param.type,
    description: param.description,
    required: param.required,
    default: param.default as string | number | boolean | null,
    options: param.options,
    min: param.min,
    max: param.max,
    source: param.source,
    source_type: param.source_type ?? undefined,
    filter: param.filter,
    available: param.available,
    unavailable_reason: param.unavailable_reason,
    tab: param.tab,
    widget: param.widget ? {
      type: param.widget.type as unknown as WidgetType,
      style: {
        width: param.widget.style.width as "full" | "half" | "auto" | undefined,
        placeholder: param.widget.style.placeholder,
      },
      props: param.widget.props,
    } : undefined,
  };
}

export function NewRunWindow({ state }: NewRunWindowProps) {
  const {
    minimizeApp,
    closeApp,
    maximizeApp,
    restoreApp,
    isAppMaximized,
    getAppPosition,
    updateAppPosition,
    getAppSize,
    updateAppSize,
    focusApp,
    isAppFocused,
    getAppParams,
    clearAppParams,
  } = useApps();
  const savedPosition = getAppPosition("new-run");
  const savedSize = getAppSize("new-run");
  const maximized = isAppMaximized("new-run");
  const focused = isAppFocused("new-run");

  const {
    runTypes,
    schema,
    schemaLoading,
    config,
    isLoading,
    isWsConnected,
    error,
    connectWebSocket,
    getSchema,
    updateConfigValue,
    createRun,
    resetSchema,
  } = useRunsStore();
  
  const [selectedRunType, setSelectedRunType] = useState<RunTypeInfo | null>(null);
  const [runName, setRunName] = useState("");
  const [activeTab, setActiveTab] = useState("basic");
  const [initialRunType, setInitialRunType] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [createResult, setCreateResult] = useState<{ success: boolean; message: string } | null>(null);
  const { collapsed, toggle: toggleSidebar } = useSidebarCollapse(false);

  useEffect(() => {
    if (!isWsConnected && !error) {
      connectWebSocket();
    }
  }, [isWsConnected, error, connectWebSocket]);

  useEffect(() => {
    const params = getAppParams("new-run");
    if (params?.runType && typeof params.runType === "string") {
      setInitialRunType(params.runType);
      clearAppParams("new-run");
    }
  }, [getAppParams, clearAppParams]);

  useEffect(() => {
    if (initialRunType && runTypes.length > 0 && !selectedRunType) {
      const targetRunType = runTypes.find(rt => rt.name === initialRunType);
      if (targetRunType) {
        setSelectedRunType(targetRunType);
        setRunName(`${targetRunType.label}-${Date.now()}`);
        setInitialRunType(null);
      }
    }
  }, [initialRunType, runTypes, selectedRunType]);

  useEffect(() => {
    if (selectedRunType) {
      getSchema(selectedRunType.name);
      setActiveTab("basic");
    }
  }, [selectedRunType, getSchema]);

  const handleClose = () => {
    setSelectedRunType(null);
    setRunName("");
    resetSchema();
    closeApp("new-run");
  };

  const handleSelectRunType = (runType: RunTypeInfo) => {
    setSelectedRunType(runType);
    setRunName(`${runType.label}-${Date.now()}`);
  };

  const handleCreateRun = async () => {
    if (!selectedRunType || !runName.trim()) {
      return;
    }

    setIsCreating(true);
    setCreateResult(null);

    try {
      const result = await createRun(selectedRunType.name, runName);
      
      if (result.success) {
        setCreateResult({ 
          success: true, 
          message: `Run "${runName}" created successfully! ID: ${result.run_id}` 
        });
        
        setTimeout(() => {
          closeApp("new-run");
          setSelectedRunType(null);
          setRunName("");
          resetSchema();
          setCreateResult(null);
        }, 1500);
      } else {
        setCreateResult({ 
          success: false, 
          message: result.error || "Failed to create run" 
        });
      }
    } catch (err) {
      setCreateResult({ 
        success: false, 
        message: String(err) 
      });
    } finally {
      setIsCreating(false);
    }
  };

  const handleConfigChange = (name: string, value: unknown) => {
    updateConfigValue(name, value);
  };

  if (state === "minimized") {
    return null;
  }

  const sidebarWidth = collapsed ? 0 : 200;

  const tabs = schema?.tabs || [];
  const parameters = schema?.parameters || [];
  const currentTabParams = parameters.filter(p => p.tab === activeTab);

  return (
    <AppWindow
      appId="new-run"
      defaultSize={{ width: 900, height: 600 }}
      onMinimize={() => minimizeApp("new-run")}
      onClose={handleClose}
      savedPosition={savedPosition}
      onPositionChange={(pos) => updateAppPosition("new-run", pos)}
      savedSize={savedSize}
      onSizeChange={(size) => updateAppSize("new-run", size)}
      isMaximized={maximized}
      onMaximize={() => maximizeApp("new-run")}
      onRestore={() => restoreApp("new-run")}
      isFocused={focused}
      onFocus={() => focusApp("new-run")}
      sidebarCollapsed={collapsed}
      onSidebarToggle={toggleSidebar}
    >
      <div className="flex h-full">
        <div
          className="border-r border-border/50 flex flex-col bg-muted/20 transition-all duration-200 overflow-hidden"
          style={{ width: sidebarWidth }}
        >
          {!collapsed && (
            <>
              <div className="p-3 border-b border-border/50">
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Run Type
                </span>
              </div>
              <div className="flex-1 overflow-y-auto p-2 space-y-1">
                {isLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <img src="/load.svg" alt="Loading" className="h-6 w-6" />
                  </div>
                ) : runTypes.length === 0 ? (
                  <div className="text-center text-muted-foreground text-sm py-8">
                    No types available
                  </div>
                ) : (
                  runTypes.map((runType) => (
                    <button
                      key={runType.name}
                      onClick={() => handleSelectRunType(runType)}
                      className={cn(
                        "w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-left transition-colors",
                        selectedRunType?.name === runType.name
                          ? "bg-primary/10 text-primary"
                          : "hover:bg-muted/50"
                      )}
                    >
                      <div
                        className="flex items-center justify-center w-8 h-8 rounded-md flex-shrink-0"
                        style={{ backgroundColor: `${runType.color}20` }}
                      >
                        <div style={{ color: runType.color }}>
                          {iconMap[runType.icon] || <Play className="h-5 w-5" />}
                        </div>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm truncate">{runType.label}</div>
                      </div>
                      {selectedRunType?.name === runType.name && (
                        <CheckCircle className="h-4 w-4 text-primary flex-shrink-0" />
                      )}
                    </button>
                  ))
                )}
              </div>
            </>
          )}
        </div>

        <div className="flex-1 flex flex-col min-w-0">
          {selectedRunType ? (
            <>
              <div className="p-4 border-b border-border/50 bg-muted/10">
                <div className="flex items-center gap-3">
                  <div
                    className="flex items-center justify-center w-10 h-10 rounded-lg flex-shrink-0"
                    style={{ backgroundColor: `${selectedRunType.color}20` }}
                  >
                    <div style={{ color: selectedRunType.color }}>
                      {iconMap[selectedRunType.icon] || <Play className="h-5 w-5" />}
                    </div>
                  </div>
                  <div className="min-w-0">
                    <h2 className="font-semibold">{selectedRunType.label}</h2>
                    <p className="text-xs text-muted-foreground truncate">
                      {selectedRunType.description}
                    </p>
                  </div>
                </div>
              </div>

              {tabs.length > 1 && (
                <div className="flex border-b border-border/50 bg-muted/5 overflow-x-auto">
                  {tabs.map((tab) => (
                    <button
                      key={tab.name}
                      onClick={() => setActiveTab(tab.name)}
                      disabled={!tab.available}
                      className={cn(
                        "px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors",
                        activeTab === tab.name
                          ? "text-primary border-b-2 border-primary"
                          : "text-muted-foreground hover:text-foreground",
                        !tab.available && "opacity-50 cursor-not-allowed"
                      )}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              )}

              <div className="flex-1 overflow-y-auto p-4">
                {schemaLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                  </div>
                ) : (
                  <div className="space-y-4">
                    {activeTab === "basic" && (
                      <div className="space-y-2">
                        <Label htmlFor="run-name">Run Name</Label>
                        <Input
                          id="run-name"
                          value={runName}
                          onChange={(e) => setRunName(e.target.value)}
                          placeholder="Enter a name for this run"
                        />
                      </div>
                    )}

                    {currentTabParams.length > 0 ? (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {currentTabParams.map((param) => (
                          <div
                            key={param.name}
                            className={cn(
                              "space-y-2",
                              param.widget?.style?.width === "full" && "md:col-span-2"
                            )}
                          >
                            <Label className={param.required ? "after:content-['*'] after:text-red-500" : ""}>
                              {param.name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
                            </Label>
                            <DynamicWidget
                              parameter={convertToDynamicParameter(param)}
                              value={config[param.name]}
                              onChange={handleConfigChange}
                              allValues={config}
                            />
                            {param.description && (
                              <p className="text-xs text-muted-foreground">{param.description}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center text-muted-foreground text-sm py-8">
                        No configuration options for this tab
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="p-4 border-t border-border/50 flex flex-col gap-3">
                {createResult && (
                  <div className={cn(
                    "flex items-center gap-2 p-3 rounded-lg text-sm",
                    createResult.success 
                      ? "bg-green-500/10 text-green-600 dark:text-green-400" 
                      : "bg-red-500/10 text-red-600 dark:text-red-400"
                  )}>
                    {createResult.success ? (
                      <CheckCircle className="h-4 w-4 flex-shrink-0" />
                    ) : (
                      <AlertCircle className="h-4 w-4 flex-shrink-0" />
                    )}
                    <span>{createResult.message}</span>
                  </div>
                )}
                <div className="flex justify-end gap-2">
                  <Button variant="outline" onClick={() => {
                    setSelectedRunType(null);
                    resetSchema();
                    setCreateResult(null);
                  }}>
                    Cancel
                  </Button>
                  <Button 
                    onClick={handleCreateRun} 
                    disabled={!runName.trim() || schemaLoading || isCreating}
                  >
                    {isCreating ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    {isCreating ? "Creating..." : "Create Run"}
                  </Button>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground">
              <Play className="h-12 w-12 mb-4 opacity-30" />
              <p className="text-sm">Select a run type from the left panel</p>
            </div>
          )}
        </div>
      </div>
    </AppWindow>
  );
}
