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

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Play,
  Pause,
  Square,
  RotateCcw,
  Brain,
  MessageSquare,
  Download,
  Gauge,
  Trash2,
  Clock,
  CheckCircle,
  XCircle,
  Loader,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api";
import Link from "next/link";
import type { RunInfo } from "@/types/training";

export default function RunsPage() {
  const { data: runs, isLoading, refetch } = useQuery({
    queryKey: ["runs"],
    queryFn: () => apiClient.listRuns(),
    refetchInterval: 5000,
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running":
        return <Loader className="h-4 w-4 animate-spin text-green-500" />;
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
  };

  const stats = {
    total: runs?.total || 0,
    running: runs?.runs.filter((r: { status: string }) => r.status === "running").length || 0,
    completed: runs?.runs.filter((r: { status: string }) => r.status === "completed").length || 0,
    failed: runs?.runs.filter((r: { status: string }) => r.status === "failed").length || 0,
    paused: runs?.runs.filter((r: { status: string }) => r.status === "paused").length || 0,
  };

  return (
    <ScrollArea className="h-full">
      <div className="space-y-6 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Runs</h1>
            <p className="text-muted-foreground">
              Manage all training, inference, and background tasks
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => refetch()}>
              <RotateCcw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
            <Button asChild>
              <Link href="/training/new">
                <Play className="mr-2 h-4 w-4" />
                New Run
              </Link>
            </Button>
          </div>
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
              <Loader className="h-4 w-4 text-green-500" />
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
            <CardDescription>
              All tasks from training, inference, downloads, and benchmarks
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
              </div>
            ) : !runs?.runs || runs.runs.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Play className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground mb-2">No runs yet</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Start a new training, inference, or benchmark task
                </p>
                <Button variant="outline" asChild>
                  <Link href="/training/new">
                    <Play className="mr-2 h-4 w-4" />
                    New Run
                  </Link>
                </Button>
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
                {runs.runs.map((run: RunInfo) => (
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
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <Pause className="h-3 w-3" />
                            </Button>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <Square className="h-3 w-3" />
                            </Button>
                          </>
                        )}
                        {run.status === "paused" && (
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <RotateCcw className="h-3 w-3" />
                          </Button>
                        )}
                        {run.status === "completed" && (
                          <Button variant="ghost" size="icon" className="h-8 w-8 text-red-500">
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
