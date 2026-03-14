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

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Activity,
  Cpu,
  MemoryStick,
  Server,
  Network,
  Gauge,
  Zap,
  Thermometer,
  Battery,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface GpuInfo {
  vendor: string;
  name: string;
  utilization: number;
  memory_used_gb: number;
  memory_total_gb: number;
  temperature: number;
  power_draw: number;
}

function getGpuVendorColor(vendor: string): { bg: string; text: string; bar: string } {
  switch (vendor.toLowerCase()) {
    case "nvidia":
      return { bg: "bg-green-100 dark:bg-green-900/20", text: "text-green-600 dark:text-green-400", bar: "bg-green-500" };
    case "amd":
      return { bg: "bg-red-100 dark:bg-red-900/20", text: "text-red-600 dark:text-red-400", bar: "bg-red-500" };
    case "intel":
      return { bg: "bg-blue-100 dark:bg-blue-900/20", text: "text-blue-600 dark:text-blue-400", bar: "bg-blue-500" };
    default:
      return { bg: "bg-violet-100 dark:bg-violet-900/20", text: "text-violet-600 dark:text-violet-400", bar: "bg-violet-500" };
  }
}

function getGpuVendorLabel(vendor: string): string {
  switch (vendor.toLowerCase()) {
    case "nvidia":
      return "NVIDIA";
    case "amd":
      return "AMD";
    case "intel":
      return "Intel";
    default:
      return vendor.toUpperCase();
  }
}

export function MonitorPopup() {
  const [open, setOpen] = useState(false);

  const { data: stats } = useQuery({
    queryKey: ["stats"],
    queryFn: async () => {
      const result = await apiClient.getStats();
      return result as Record<string, unknown>;
    },
    refetchInterval: 2000,
  });

  const systemStats = stats as Record<string, unknown> | undefined;

  const cpuPercent = (systemStats?.cpu_percent as number) || 0;
  const memPercent = (systemStats?.memory_percent as number) || 0;
  const memUsed = (systemStats?.memory_used_gb as number) || 0;
  const memTotal = (systemStats?.memory_total_gb as number) || 0;
  const gpuCount = (systemStats?.gpu_count as number) || 0;
  const gpuVendors = (systemStats?.gpu_vendors as string[]) || [];
  const gpuNames = (systemStats?.gpu_names as string[]) || [];
  const gpuUtils = (systemStats?.gpu_utilization as number[]) || [];
  const gpuMemUsed = (systemStats?.gpu_memory_used as number[]) || [];
  const gpuMemTotal = (systemStats?.gpu_memory_total as number[]) || [];
  const gpuTemps = (systemStats?.gpu_temperatures as number[]) || [];
  const gpuPower = (systemStats?.gpu_power_draw as number[]) || [];
  const qps = ((systemStats?.qps as number) || 0).toFixed(2);
  const uptime = (systemStats?.uptime_seconds as number) || 0;

  const gpus: GpuInfo[] = [];
  for (let i = 0; i < gpuCount; i++) {
    gpus.push({
      vendor: gpuVendors[i] || "unknown",
      name: gpuNames[i] || `GPU ${i}`,
      utilization: gpuUtils[i] || 0,
      memory_used_gb: gpuMemUsed[i] || 0,
      memory_total_gb: gpuMemTotal[i] || 0,
      temperature: gpuTemps[i] || 0,
      power_draw: gpuPower[i] || 0,
    });
  }

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="relative h-8 w-8"
        >
          <Activity className="h-4 w-4" />
          <span className="absolute right-0 top-0 h-2 w-2 rounded-full bg-green-500" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl p-0 gap-0">
        <DialogHeader className="px-6 py-4 border-b">
          <DialogTitle className="flex items-center gap-2 text-base font-semibold">
            <Activity className="h-5 w-5 text-primary" />
            System Monitor
          </DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="w-full justify-start rounded-none border-b bg-transparent px-6 py-0 h-12">
            <TabsTrigger value="overview" className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none px-4 py-3">
              <Cpu className="h-4 w-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="performance" className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none px-4 py-3">
              <Activity className="h-4 w-4 mr-2" />
              Performance
            </TabsTrigger>
            <TabsTrigger value="gpu" className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none px-4 py-3">
              <Server className="h-4 w-4 mr-2" />
              GPU
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="p-6 mt-0">
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-lg border p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-md bg-emerald-100 dark:bg-emerald-900/20">
                      <Cpu className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                    </div>
                    <span className="text-sm font-medium text-muted-foreground">CPU Usage</span>
                  </div>
                  <span className="text-2xl font-bold">{cpuPercent.toFixed(0)}%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 transition-all duration-300 rounded-full"
                    style={{ width: `${cpuPercent}%` }}
                  />
                </div>
                <p className="mt-2 text-xs text-muted-foreground">{navigator.hardwareConcurrency || 8} cores available</p>
              </div>

              <div className="rounded-lg border p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-md bg-amber-100 dark:bg-amber-900/20">
                      <MemoryStick className="h-4 w-4 text-amber-600 dark:text-amber-400" />
                    </div>
                    <span className="text-sm font-medium text-muted-foreground">Memory</span>
                  </div>
                  <span className="text-2xl font-bold">{memPercent.toFixed(0)}%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-amber-500 transition-all duration-300 rounded-full"
                    style={{ width: `${memPercent}%` }}
                  />
                </div>
                <p className="mt-2 text-xs text-muted-foreground">{memUsed.toFixed(1)} / {memTotal.toFixed(1)} GB used</p>
              </div>
            </div>

            {gpuCount > 0 && (
              <div className="mt-4 rounded-lg border p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Server className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">GPU Overview</span>
                  <span className="text-sm text-muted-foreground">({gpuCount} detected)</span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  {gpus.map((gpu, idx) => {
                    const colors = getGpuVendorColor(gpu.vendor);
                    return (
                      <div key={idx} className="p-3 rounded-md bg-muted/50">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <div className={cn("p-1 rounded", colors.bg)}>
                              <Server className={cn("h-3 w-3", colors.text)} />
                            </div>
                            <span className="text-xs font-medium truncate max-w-[100px]">{gpu.name}</span>
                          </div>
                          <span className="font-semibold">{gpu.utilization.toFixed(0)}%</span>
                        </div>
                        <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                          <div
                            className={cn("h-full transition-all duration-300 rounded-full", colors.bar)}
                            style={{ width: `${gpu.utilization}%` }}
                          />
                        </div>
                        <div className="mt-1 flex justify-between text-xs text-muted-foreground">
                          <span>{gpu.memory_used_gb.toFixed(1)}/{gpu.memory_total_gb.toFixed(1)} GB</span>
                          {gpu.temperature > 0 && <span>{gpu.temperature.toFixed(0)}°C</span>}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="mt-4 grid grid-cols-3 gap-3">
              <div className="rounded-lg border p-3 text-center">
                <div className="flex items-center justify-center gap-1 text-muted-foreground mb-1">
                  <Zap className="h-3 w-3" />
                  <span className="text-xs">QPS</span>
                </div>
                <p className="text-xl font-semibold">{qps}</p>
              </div>
              <div className="rounded-lg border p-3 text-center">
                <div className="flex items-center justify-center gap-1 text-muted-foreground mb-1">
                  <Gauge className="h-3 w-3" />
                  <span className="text-xs">Uptime</span>
                </div>
                <p className="text-xl font-semibold">{formatUptime(uptime)}</p>
              </div>
              <div className="rounded-lg border p-3 text-center">
                <div className="flex items-center justify-center gap-1 text-muted-foreground mb-1">
                  <Network className="h-3 w-3" />
                  <span className="text-xs">GPUs</span>
                </div>
                <p className="text-xl font-semibold">{gpuCount}</p>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="performance" className="p-6 mt-0">
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <Activity className="h-12 w-12 mb-4 opacity-20" />
              <p className="text-sm">Performance metrics coming soon</p>
              <p className="text-xs mt-1">Historical data and charts will be available here</p>
            </div>
          </TabsContent>

          <TabsContent value="gpu" className="p-6 mt-0">
            {gpuCount > 0 ? (
              <div className="space-y-3">
                {gpus.map((gpu, idx) => {
                  const colors = getGpuVendorColor(gpu.vendor);
                  const memPercent = gpu.memory_total_gb > 0 ? (gpu.memory_used_gb / gpu.memory_total_gb) * 100 : 0;
                  
                  return (
                    <div key={idx} className="rounded-lg border p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <div className={cn("p-2 rounded-md", colors.bg)}>
                            <Server className={cn("h-4 w-4", colors.text)} />
                          </div>
                          <div>
                            <span className="font-medium">{gpu.name}</span>
                            <span className={cn("ml-2 text-xs px-1.5 py-0.5 rounded", colors.bg, colors.text)}>
                              {getGpuVendorLabel(gpu.vendor)}
                            </span>
                          </div>
                        </div>
                        <span className="text-2xl font-bold">{gpu.utilization.toFixed(0)}%</span>
                      </div>
                      
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-muted-foreground">GPU Utilization</span>
                            <span>{gpu.utilization.toFixed(0)}%</span>
                          </div>
                          <div className="h-2 bg-muted rounded-full overflow-hidden">
                            <div
                              className={cn("h-full transition-all duration-300 rounded-full", colors.bar)}
                              style={{ width: `${gpu.utilization}%` }}
                            />
                          </div>
                        </div>
                        
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-muted-foreground">Memory Usage</span>
                            <span>{gpu.memory_used_gb.toFixed(1)} / {gpu.memory_total_gb.toFixed(1)} GB ({memPercent.toFixed(0)}%)</span>
                          </div>
                          <div className="h-2 bg-muted rounded-full overflow-hidden">
                            <div
                              className={cn("h-full transition-all duration-300 rounded-full", colors.bar)}
                              style={{ width: `${memPercent}%` }}
                            />
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4 pt-2">
                          {gpu.temperature > 0 && (
                            <div className="flex items-center gap-2">
                              <Thermometer className="h-4 w-4 text-muted-foreground" />
                              <span className="text-sm text-muted-foreground">Temperature:</span>
                              <span className="text-sm font-medium">{gpu.temperature.toFixed(0)}°C</span>
                            </div>
                          )}
                          {gpu.power_draw > 0 && (
                            <div className="flex items-center gap-2">
                              <Battery className="h-4 w-4 text-muted-foreground" />
                              <span className="text-sm text-muted-foreground">Power:</span>
                              <span className="text-sm font-medium">{gpu.power_draw.toFixed(1)}W</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <Server className="h-12 w-12 mb-4 opacity-20" />
                <p className="text-sm">No GPU detected</p>
                <p className="text-xs mt-1">Connect a GPU to see detailed metrics</p>
              </div>
            )}
          </TabsContent>
        </Tabs>

        <div className="px-6 py-3 border-t bg-muted/30 text-xs text-muted-foreground flex items-center justify-between">
          <span>Auto-refresh: 2s</span>
          <span>PLx Studio Monitor</span>
        </div>
      </DialogContent>
    </Dialog>
  );
}
