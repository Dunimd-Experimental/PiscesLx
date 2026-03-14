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
import {
  Database,
  Upload,
  FileJson,
  FolderOpen,
  Search,
  MoreHorizontal,
} from "lucide-react";
import { MainLayout } from "@/components/layout";
import { useState } from "react";

export default function DataPage() {
  const [searchQuery, setSearchQuery] = useState("");

  const datasets = [
    { id: "ds-001", name: "SFT Training Data", type: "text", samples: 50000, size: "256 MB" },
    { id: "ds-002", name: "DPO Preference Data", type: "text", samples: 10000, size: "64 MB" },
    { id: "ds-003", name: "Image-Text Pairs", type: "image-text", samples: 25000, size: "1.2 GB" },
    { id: "ds-004", name: "Audio Transcriptions", type: "audio", samples: 5000, size: "512 MB" },
    { id: "ds-005", name: "Video Captions", type: "video", samples: 3000, size: "2.5 GB" },
  ];

  const filteredDatasets = datasets.filter(
    (ds) =>
      ds.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      ds.type.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <MainLayout>
      <ScrollArea className="h-full">
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Data Management</h1>
              <p className="text-muted-foreground">
                Upload and manage your training datasets
              </p>
            </div>
            <Button>
              <Upload className="mr-2 h-4 w-4" />
              Upload Dataset
            </Button>
          </div>

          <div className="grid gap-4 md:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Datasets</CardTitle>
                <Database className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{datasets.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Samples</CardTitle>
                <FileJson className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {datasets.reduce((acc, ds) => acc + ds.samples, 0).toLocaleString()}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Size</CardTitle>
                <FolderOpen className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">4.3 GB</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Data Types</CardTitle>
                <Database className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">5</div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Datasets</CardTitle>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    placeholder="Search datasets..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9 pr-4 py-1 text-sm rounded-md border bg-background"
                  />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {filteredDatasets.map((dataset) => (
                  <div
                    key={dataset.id}
                    className="flex items-center justify-between rounded-lg border p-4 hover:bg-muted/50 transition-colors"
                  >
                    <div className="flex items-center gap-4">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                        <Database className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium">{dataset.name}</p>
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          <span>{dataset.type}</span>
                          <span>{dataset.samples.toLocaleString()} samples</span>
                          <span>{dataset.size}</span>
                        </div>
                      </div>
                    </div>
                    <Button variant="ghost" size="icon">
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </ScrollArea>
    </MainLayout>
  );
}
