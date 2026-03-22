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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Play, Settings2, Cpu, ArrowLeft, Loader2, Zap, Brain, Wrench } from "lucide-react";
import { useTrainingStore } from "@/lib/stores";
import Link from "next/link";
import type { DynamicParameter, DynamicTab } from "@/types/dynamic";
import { DynamicField } from "@/components/ui/dynamic-widget";
import { UnavailableTab, UnavailableCommand } from "@/components/ui/unavailable";

export default function NewTrainingPage() {
  const { 
    config, 
    schema, 
    parameters, 
    tabs, 
    dynamicOptions, 
    isLoading, 
    fetchSchema, 
    setConfig,
    startTraining,
    error,
  } = useTrainingStore();
  const [activeTab, setActiveTab] = useState("basic");
  const [isStarting, setIsStarting] = useState(false);

  useEffect(() => {
    fetchSchema();
  }, [fetchSchema]);

  const handleStartTraining = async () => {
    setIsStarting(true);
    const result = await startTraining();
    if (result.success && result.run_id) {
      window.location.href = `/runs`;
    } else {
      console.error("Failed to start training:", result.error);
    }
    setIsStarting(false);
  };

  const handleParameterChange = (name: string, value: unknown) => {
    setConfig({ [name]: value });
  };

  const getParamsByTab = (tabName: string): DynamicParameter[] => {
    return parameters.filter(p => (p.tab || "basic") === tabName);
  };

  const renderParamGroup = (params: DynamicParameter[]) => {
    if (params.length === 0) {
      return (
        <p className="text-muted-foreground text-sm">
          No parameters in this section.
        </p>
      );
    }

    return (
      <div className="grid gap-4 md:grid-cols-2">
        {params.map((param) => {
          const isAvailable = param.available !== false;
          const options = dynamicOptions[param.name] || [];
          
          return (
            <div key={param.name} className={`space-y-2 ${!isAvailable ? "opacity-60" : ""}`}>
              <DynamicField
                parameter={param}
                value={config[param.name]}
                onChange={handleParameterChange}
                options={options}
                allValues={config}
              />
            </div>
          );
        })}
      </div>
    );
  };

  const getTabIcon = (tabName: string) => {
    switch (tabName) {
      case "basic": return <Cpu className="h-5 w-5" />;
      case "quant": return <Zap className="h-5 w-5" />;
      case "rlhf": return <Brain className="h-5 w-5" />;
      case "advanced": return <Wrench className="h-5 w-5" />;
      default: return <Settings2 className="h-5 w-5" />;
    }
  };

  const getTabDescription = (tabName: string) => {
    switch (tabName) {
      case "basic": return "Model size and training configuration";
      case "quant": return "Quantization and LoRA settings";
      case "rlhf": return "Reinforcement Learning from Human Feedback";
      case "advanced": return "Advanced run configuration";
      default: return "";
    }
  };

  const renderTabContent = (tab: DynamicTab) => {
    const isAvailable = tab.available !== false;
    const tabParams = getParamsByTab(tab.name);
    
    if (!isAvailable) {
      return <UnavailableTab reason={tab.unavailable_reason} />;
    }
    
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {getTabIcon(tab.name)}
            {tab.label || tab.name}
          </CardTitle>
          <CardDescription>{getTabDescription(tab.name)}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {renderParamGroup(tabParams)}
        </CardContent>
      </Card>
    );
  };

  const isCommandAvailable = schema?.available !== false;
  const unavailableReason = schema?.unavailable_reason || "";

  return (
    <ScrollArea className="h-full">
      <div className="space-y-6 p-6">
        <div className="flex items-center gap-4">
          <Button variant="secondary" size="icon" asChild>
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

        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            <span className="ml-2 text-muted-foreground">Loading configuration...</span>
          </div>
        )}

        {!isLoading && !isCommandAvailable && (
          <UnavailableCommand 
            command="training" 
            reason={unavailableReason} 
          />
        )}

        {!isLoading && isCommandAvailable && tabs.length > 0 && (
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList 
              className="tabs-list--grid"
              style={{ '--grid-columns': Math.min(tabs.length, 6) } as React.CSSProperties}
            >
              {tabs.map((tab) => (
                <TabsTrigger 
                  key={tab.name} 
                  value={tab.name}
                  disabled={tab.available === false}
                  className={tab.available === false ? "cursor-not-allowed" : ""}
                >
                  {tab.label || tab.name}
                </TabsTrigger>
              ))}
            </TabsList>

            {tabs.map((tab) => (
              <TabsContent key={tab.name} value={tab.name} className="mt-4 space-y-4">
                {renderTabContent(tab)}
              </TabsContent>
            ))}
          </Tabs>
        )}

        <div className="flex justify-end gap-4">
          <Button variant="secondary" asChild>
            <Link href="/training">Cancel</Link>
          </Button>
          {error && (
            <p className="text-sm text-destructive self-center">{error}</p>
          )}
          <Button 
            variant="secondary" 
            onClick={handleStartTraining} 
            disabled={isLoading || isStarting || !isCommandAvailable}
          >
            {isStarting ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Play className="mr-2 h-4 w-4" />
            )}
            Start Training
          </Button>
        </div>
      </div>
    </ScrollArea>
  );
}
