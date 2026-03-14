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
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Brain,
  Plus,
  Play,
  Pause,
  ChevronRight,
  Layers,
} from "lucide-react";
import { useTrainingStore } from "@/lib/stores";
import Link from "next/link";
import type { RunStatus } from "@/types";

const statusColors: Record<RunStatus, string> = {
  pending: "bg-yellow-500",
  running: "bg-green-500 animate-pulse",
  paused: "bg-orange-500",
  completed: "bg-blue-500",
  failed: "bg-red-500",
  cancelled: "bg-gray-500",
};

export default function TrainingPage() {
  const { runs, isLoading, fetchRuns, controlRun } = useTrainingStore();
  const [activeTab, setActiveTab] = useState<RunStatus | "all">("all");

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  const filteredRuns = activeTab === "all"
    ? runs
    : runs.filter((run) => run.status === activeTab);

  const handleControl = async (runId: string, action: "pause" | "resume" | "cancel" | "kill") => {
    await controlRun(runId, action);
  };

  return (
    <ScrollArea className="h-full">
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Training</h1>
              <p className="text-muted-foreground">
                Manage and monitor your training jobs
              </p>
            </div>
            <Button asChild>
              <Link href="/training/new">
                <Plus className="mr-2 h-4 w-4" />
                New Training
              </Link>
            </Button>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Runs</CardTitle>
                <Brain className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{runs.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active</CardTitle>
                <Play className="h-4 w-4 text-green-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {runs.filter((r) => r.status === "running").length}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Completed</CardTitle>
                <Layers className="h-4 w-4 text-blue-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {runs.filter((r) => r.status === "completed").length}
                </div>
              </CardContent>
            </Card>
          </div>

          <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as RunStatus | "all")}>
            <TabsList>
              <TabsTrigger value="all">All</TabsTrigger>
              <TabsTrigger value="running">Running</TabsTrigger>
              <TabsTrigger value="completed">Completed</TabsTrigger>
              <TabsTrigger value="failed">Failed</TabsTrigger>
            </TabsList>

            <TabsContent value={activeTab} className="mt-4">
              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
                </div>
              ) : filteredRuns.length === 0 ? (
                <Card>
                  <CardContent className="flex flex-col items-center justify-center py-12">
                    <Brain className="h-12 w-12 text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No training runs found</p>
                    <Button className="mt-4" asChild>
                      <Link href="/training/new">Start Training</Link>
                    </Button>
                  </CardContent>
                </Card>
              ) : (
                <div className="space-y-4">
                  {filteredRuns.map((run) => (
                    <Card key={run.run_id} className="card-hover">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4">
                            <div className={`h-3 w-3 rounded-full ${statusColors[run.status]}`} />
                            <div>
                              <p className="font-medium">{run.run_id}</p>
                              <p className="text-sm text-muted-foreground">
                                {run.phase}
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            {run.status === "running" && (
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => handleControl(run.run_id, "pause")}
                              >
                                <Pause className="h-4 w-4" />
                              </Button>
                            )}
                            {run.status === "paused" && (
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => handleControl(run.run_id, "resume")}
                              >
                                <Play className="h-4 w-4" />
                              </Button>
                            )}
                            <Button variant="ghost" size="icon" asChild>
                              <Link href={`/training/${run.run_id}`}>
                                <ChevronRight className="h-4 w-4" />
                              </Link>
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </ScrollArea>
  );
}
