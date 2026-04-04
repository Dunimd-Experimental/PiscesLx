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

import { useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Play,
  Pause,
  Square,
  Brain,
  MessageSquare,
  Download,
  Gauge,
  Trash2,
  Clock,
  CheckCircle,
  XCircle,
  Package,
} from "lucide-react";
import { useRunsStore } from "@/lib/stores";
import { useApps } from "@/components/layout/apps-context";
import type { RunInfo } from "@/types/training";

export default function RunsPage() {
  const { runs, isLoading, isWsConnected, connectWebSocket, controlRun } = useRunsStore();
  const { openApp } = useApps();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [hasAutoOpened, setHasAutoOpened] = useState(false);

  useEffect(() => {
    if (!isWsConnected) {
      connectWebSocket();
    }
  }, [isWsConnected, connectWebSocket]);

  useEffect(() => {
    if (!hasAutoOpened && searchParams.get("create") === "true") {
      setHasAutoOpened(true);
      router.replace("/runs");
      setTimeout(() => {
        openApp("new-run");
      }, 100);
    }
  }, [searchParams, hasAutoOpened, router, openApp]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running":
        return <img src="/load.svg" alt="Running" className="h-4 w-4" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-blue-500" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />;
      case "paused":
        return <Pause className="h-4 w-4 text-yellow-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "running":
        return "default";
      case "completed":
        return "secondary";
      case "failed":
        return "destructive";
      case "paused":
        return "outline";
      default:
        return "outline";
    }
  };

  const getCommandIcon = (command: string) => {
    switch (command) {
      case "train":
        return <Brain className="h-4 w-4" />;
      case "serve":
      case "inference":
        return <MessageSquare className="h-4 w-4" />;
      case "download":
        return <Download className="h-4 w-4" />;
      case "benchmark":
        return <Gauge className="h-4 w-4" />;
      case "install":
        return <Package className="h-4 w-4" />;
      default:
        return <Play className="h-4 w-4" />;
    }
  };

  const commandLabels: Record<string, string> = {
    train: "Training",
    serve: "Inference",
    inference: "Inference",
    download: "Download",
    benchmark: "Benchmark",
    install: "Install",
    monitor: "Monitor",
    test: "Test",
    dev: "Dev",
  };

  const stats = {
    total: runs.length,
    running: runs.filter((r: RunInfo) => r.status === "running").length,
    completed: runs.filter((r: RunInfo) => r.status === "completed").length,
    failed: runs.filter((r: RunInfo) => r.status === "failed").length,
    paused: runs.filter((r: RunInfo) => r.status === "paused").length,
  };

  return (
    <ScrollArea className="h-full">
      <div className="space-y-6 p-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold tracking-tight">Runs</h1>
          <Button variant="secondary" onClick={() => openApp("new-run")}>
            <Play className="mr-2 h-4 w-4" />
            New Run
          </Button>
        </div>

        <div className="grid gap-4 md:grid-cols-5">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Runs</CardTitle>
              <Play className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Running</CardTitle>
              <img src="/load.svg" alt="Running" className="h-4 w-4" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-500">{stats.running}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Completed</CardTitle>
              <CheckCircle className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-500">{stats.completed}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Paused</CardTitle>
              <Pause className="h-4 w-4 text-yellow-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-500">{stats.paused}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Failed</CardTitle>
              <XCircle className="h-4 w-4 text-red-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-500">{stats.failed}</div>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>All Runs</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-12">
                <img src="/load.svg" alt="Loading" className="h-12 w-12 mb-4" />
                <span className="text-muted-foreground">Loading runs...</span>
              </div>
            ) : !isWsConnected ? (
              <div className="flex flex-col items-center justify-center py-12">
                <img src="/load.svg" alt="Connecting" className="h-12 w-12 mb-4" />
                <span className="text-muted-foreground">Connecting to runs service...</span>
              </div>
            ) : runs.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Play className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground">No runs yet</p>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="grid grid-cols-12 gap-4 rounded-lg bg-muted/50 p-3 text-sm font-medium">
                  <div className="col-span-2">Status</div>
                  <div className="col-span-3">Run ID</div>
                  <div className="col-span-2">Command</div>
                  <div className="col-span-2">Phase</div>
                  <div className="col-span-2">Created</div>
                  <div className="col-span-1">Actions</div>
                </div>
                {runs.map((run: RunInfo) => (
                  <div
                    key={run.run_id}
                    className="grid grid-cols-12 gap-4 rounded-lg border p-3 hover:bg-muted/50 transition-colors items-center"
                  >
                    <div className="col-span-2">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(run.status)}
                        <Badge variant={getStatusBadge(run.status)} className="text-xs">
                          {run.status}
                        </Badge>
                      </div>
                    </div>
                    <div className="col-span-3">
                      <p className="font-mono text-sm truncate">{run.run_id}</p>
                    </div>
                    <div className="col-span-2">
                      <div className="flex items-center gap-2">
                        {getCommandIcon(run.command || "run")}
                        <span className="text-sm">
                          {commandLabels[run.command || "run"] || run.command || "Run"}
                        </span>
                      </div>
                    </div>
                    <div className="col-span-2">
                      <span className="text-sm text-muted-foreground">
                        {run.phase || "init"}
                      </span>
                    </div>
                    <div className="col-span-2">
                      <span className="text-xs text-muted-foreground">
                        {run.created_at ? new Date(run.created_at).toLocaleString() : "N/A"}
                      </span>
                    </div>
                    <div className="col-span-1">
                      <div className="flex gap-1">
                        {run.status === "running" && (
                          <>
                            <Button
                              variant="secondary"
                              size="icon"
                              className="h-8 w-8"
                              onClick={() => controlRun(run.run_id, "pause")}
                            >
                              <Pause className="h-3 w-3" />
                            </Button>
                            <Button
                              variant="secondary"
                              size="icon"
                              className="h-8 w-8"
                              onClick={() => controlRun(run.run_id, "cancel")}
                            >
                              <Square className="h-3 w-3" />
                            </Button>
                          </>
                        )}
                        {run.status === "paused" && (
                          <Button
                            variant="secondary"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => controlRun(run.run_id, "resume")}
                          >
                            <Play className="h-3 w-3" />
                          </Button>
                        )}
                        {run.status === "completed" && (
                          <Button variant="secondary" size="icon" className="h-8 w-8 text-red-500">
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  );
}
