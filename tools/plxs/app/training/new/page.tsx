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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Play,
  Settings2,
  Database,
  Cpu,
  ArrowLeft,
} from "lucide-react";
import { useTrainingStore } from "@/lib/stores";
import Link from "next/link";
import type { ModelSize, TrainingStage } from "@/types";

const MODEL_SIZES: ModelSize[] = ["0.5B", "1.5B", "7B", "32B", "64B", "70B", "128B", "314B", "671B", "1T"];
const TRAINING_STAGES: TrainingStage[] = ["pretrain", "continued_pretrain", "sft", "alignment_dpo", "alignment_ppo", "alignment_orpo", "specialized"];

export default function NewTrainingPage() {
  const { config, setConfig, setStage } = useTrainingStore();
  const [activeTab, setActiveTab] = useState("basic");

  const handleStartTraining = async () => {
    console.log("Starting training with config:", config);
  };

  return (
    <ScrollArea className="h-full">
        <div className="space-y-6">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" asChild>
              <Link href="/training">
                <ArrowLeft className="h-4 w-4" />
              </Link>
            </Button>
            <div>
              <h1 className="text-3xl font-bold tracking-tight">New Training</h1>
              <p className="text-muted-foreground">
                Configure and start a new training job
              </p>
            </div>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="basic">Basic</TabsTrigger>
              <TabsTrigger value="optimizer">Optimizer</TabsTrigger>
              <TabsTrigger value="data">Data</TabsTrigger>
              <TabsTrigger value="advanced">Advanced</TabsTrigger>
            </TabsList>

            <TabsContent value="basic" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Cpu className="h-5 w-5" />
                    Model Configuration
                  </CardTitle>
                  <CardDescription>
                    Select model size and training stage
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Model Size</Label>
                      <Select
                        value={config.model_size}
                        onValueChange={(v) => setConfig({ model_size: v as ModelSize })}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select model size" />
                        </SelectTrigger>
                        <SelectContent>
                          {MODEL_SIZES.map((size) => (
                            <SelectItem key={size} value={size}>
                              {size}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Training Stage</Label>
                      <Select
                        value={config.stage || "pretrain"}
                        onValueChange={(v) => setStage(v as TrainingStage)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select stage" />
                        </SelectTrigger>
                        <SelectContent>
                          {TRAINING_STAGES.map((stage) => (
                            <SelectItem key={stage} value={stage}>
                              {stage}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Model Name</Label>
                    <Input
                      value={config.model_name}
                      onChange={(e) => setConfig({ model_name: e.target.value })}
                      placeholder="piscesl1-base"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Output Directory</Label>
                    <Input
                      value={config.output_dir}
                      onChange={(e) => setConfig({ output_dir: e.target.value })}
                      placeholder=".pisceslx/ckpt"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings2 className="h-5 w-5" />
                    Training Settings
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Max Steps</Label>
                      <Input
                        type="number"
                        value={config.max_steps}
                        onChange={(e) => setConfig({ max_steps: parseInt(e.target.value) })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Batch Size</Label>
                      <Input
                        type="number"
                        value={config.data?.batch_size}
                        onChange={(e) => setConfig({ data: { ...config.data!, batch_size: parseInt(e.target.value) } })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Sequence Length</Label>
                      <Input
                        type="number"
                        value={config.data?.sequence_length}
                        onChange={(e) => setConfig({ data: { ...config.data!, sequence_length: parseInt(e.target.value) } })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Mixed Precision</Label>
                      <Select
                        value={config.mixed_precision}
                        onValueChange={(v) => setConfig({ mixed_precision: v as "fp32" | "fp16" | "bf16" })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="fp32">FP32</SelectItem>
                          <SelectItem value="fp16">FP16</SelectItem>
                          <SelectItem value="bf16">BF16</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="optimizer" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Optimizer Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Learning Rate</Label>
                      <Input
                        type="number"
                        step="0.0001"
                        value={config.optimizer?.learning_rate}
                        onChange={(e) => setConfig({
                          optimizer: { ...config.optimizer!, learning_rate: parseFloat(e.target.value) }
                        })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Weight Decay</Label>
                      <Input
                        type="number"
                        step="0.01"
                        value={config.optimizer?.weight_decay}
                        onChange={(e) => setConfig({
                          optimizer: { ...config.optimizer!, weight_decay: parseFloat(e.target.value) }
                        })}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="data" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Database className="h-5 w-5" />
                    Dataset Configuration
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Dataset configuration will be available here.
                  </p>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="advanced" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Advanced Settings</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Advanced settings including GaLore, FP4, and distributed training.
                  </p>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          <div className="flex justify-end gap-4">
            <Button variant="outline" asChild>
              <Link href="/training">Cancel</Link>
            </Button>
            <Button onClick={handleStartTraining}>
              <Play className="mr-2 h-4 w-4" />
              Start Training
            </Button>
          </div>
        </div>
      </ScrollArea>
  );
}
