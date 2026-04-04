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

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Activity,
  Brain,
  Cpu,
  MessageSquare,
  Play,
  Plus,
  Zap,
  Pause,
  Square,
  RotateCcw,
  Clock,
  Download,
  Gauge,
  Trash2,
} from "lucide-react";
import Link from "next/link";
import type { RunInfo } from "@/types/training";
import { useRunsStore } from "@/lib/stores";
import { useMonitorStore } from "@/lib/stores/monitor-store";
import { useEffect } from "react";

export default function DashboardPage() {
  const { runs, isLoading: runsLoading, connectWebSocket: connectRunsWs, controlRun } = useRunsStore();
  const { stats, connectWebSocket: connectMonitorWs } = useMonitorStore();

  useEffect(() => {
    connectRunsWs();
    connectMonitorWs();
  }, [connectRunsWs, connectMonitorWs]);

  const systemStats = stats;

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running":
        return <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />;
      case "completed":
        return <div className="h-2 w-2 rounded-full bg-blue-500" />;
      case "failed":
        return <div className="h-2 w-2 rounded-full bg-red-500" />;
      case "paused":
        return <div className="h-2 w-2 rounded-full bg-yellow-500" />;
      default:
        return <div className="h-2 w-2 rounded-full bg-gray-500" />;
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
        return <MessageSquare className="h-4 w-4" />;
      case "download":
        return <Download className="h-4 w-4" />;
      case "benchmark":
        return <Gauge className="h-4 w-4" />;
      default:
        return <Play className="h-4 w-4" />;
    }
  };

  const commandLabels: Record<string, string> = {
    train: "Training",
    serve: "Inference",
    download: "Download",
    benchmark: "Benchmark",
    monitor: "Monitor",
    test: "Test",
    dev: "Dev",
    action: "Action",
  };

  return (
    <ScrollArea className="h-full">
      <div className="page-container">
        <div className="page-grid page-grid--2col">
          <Card className="card--hover" asChild>
            <Link href="/training?create=true">
              <CardHeader>
                <div className="page-header">
                  <div className="flex items-center gap-3">
                    <div className="rounded-lg bg-primary/10 p-3">
                      <Brain className="h-8 w-8 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-xl">Training</CardTitle>
                    </div>
                  </div>
                  <Plus className="h-6 w-6 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="page-stats">
                  <div className="page-stats__item">
                    <Play className="h-4 w-4" />
                    <span>{runs.filter((r: RunInfo) => r.status === "running").length || 0} Active</span>
                  </div>
                  <div className="page-stats__item">
                    <Clock className="h-4 w-4" />
                    <span>{runs.length || 0} Total Runs</span>
                  </div>
                </div>
              </CardContent>
            </Link>
          </Card>

          <Card className="card--hover" asChild>
            <Link href="/inference">
              <CardHeader>
                <div className="page-header">
                  <div className="flex items-center gap-3">
                    <div className="rounded-lg bg-primary/10 p-3">
                      <MessageSquare className="h-8 w-8 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-xl">Inference</CardTitle>
                    </div>
                  </div>
                  <Zap className="h-6 w-6 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="page-stats">
                  <div className="page-stats__item">
                    <Activity className="h-4 w-4" />
                    <span>QPS: {((systemStats?.qps as number) || 0).toFixed(2)}</span>
                  </div>
                  <div className="page-stats__item">
                    <Cpu className="h-4 w-4" />
                    <span>{(systemStats?.gpu_count as number) || 0} GPUs</span>
                  </div>
                </div>
              </CardContent>
            </Link>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <div className="page-header">
              <CardTitle>All Runs</CardTitle>
              <Button variant="secondary" size="sm" asChild>
                <Link href="/runs">View Details</Link>
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {runsLoading ? (
              <div className="page-loading">
                <div className="page-loading__spinner" />
              </div>
            ) : runs.length === 0 ? (
              <div className="page-empty">
                <Play className="page-empty__icon h-12 w-12" />
                <p className="page-empty__title">No runs yet</p>
                <Button variant="secondary" asChild>
                  <Link href="/runs?create=true">
                    <Play className="mr-2 h-4 w-4" />
                    New Run
                  </Link>
                </Button>
              </div>
            ) : (
              <div className="page-list">
                <div className="page-list__header" style={{ gridTemplateColumns: 'repeat(12, 1fr)' }}>
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
                    className="page-list__item"
                    style={{ gridTemplateColumns: 'repeat(12, 1fr)' }}
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
                            <RotateCcw className="h-3 w-3" />
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
