/**
 * Copyright © 2026 Wenze Wei. All Rights Reserved.
 *
 * This file is part of Xi.
 * The Xi project belongs to the Dunimd Team.
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
 */

import { create } from "zustand";
import type { RunInfo, DynamicParameter, DynamicOption, DynamicTab, DynamicCommandSchema } from "@/types";
import { apiClient } from "@/lib/api/client";

interface TrainingState {
  runs: RunInfo[];
  currentRun: RunInfo | null;
  config: Record<string, unknown>;
  schema: DynamicCommandSchema | null;
  parameters: DynamicParameter[];
  tabs: DynamicTab[];
  dynamicOptions: Record<string, DynamicOption[]>;
  isLoading: boolean;
  error: string | null;

  fetchRuns: () => Promise<void>;
  fetchSchema: () => Promise<void>;
  fetchParameterOptions: (parameterName: string) => Promise<void>;
  setConfig: (config: Record<string, unknown>) => void;
  controlRun: (runId: string, action: "pause" | "resume" | "cancel" | "kill") => Promise<void>;
  selectRun: (runId: string) => void;
  startTraining: (runName?: string) => Promise<{ success: boolean; run_id?: string; error?: string }>;
}

const DEFAULT_CONFIG: Record<string, unknown> = {
  model_size: "7B",
  train_mode: "standard",
  seq_len: 2048,
};

export const useTrainingStore = create<TrainingState>((set, get) => ({
  runs: [],
  currentRun: null,
  config: DEFAULT_CONFIG,
  schema: null,
  parameters: [],
  tabs: [],
  dynamicOptions: {},
  isLoading: false,
  error: null,

  fetchRuns: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await apiClient.listRuns();
      set({ runs: response.runs, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  fetchSchema: async () => {
    set({ isLoading: true, error: null });
    try {
      const schema = await apiClient.getCommandSchema("train");
      const parameters = schema.parameters || [];
      const tabs = schema.tabs || [];
      
      const defaultConfig: Record<string, unknown> = { ...DEFAULT_CONFIG };
      for (const param of parameters) {
        if (param.default !== null && param.default !== undefined) {
          defaultConfig[param.name] = param.default;
        }
      }
      
      set({ 
        schema: schema,
        parameters: parameters, 
        tabs: tabs,
        config: defaultConfig,
        isLoading: false 
      });
      
      for (const param of parameters) {
        if (param.source && param.source_type === "directory" && param.available !== false) {
          await get().fetchParameterOptions(param.name);
        }
      }
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  fetchParameterOptions: async (parameterName: string) => {
    try {
      const response = await apiClient.getParameterOptions("train", parameterName);
      set((state) => ({
        dynamicOptions: {
          ...state.dynamicOptions,
          [parameterName]: response.options || [],
        },
      }));
    } catch (error) {
      console.error(`Failed to fetch options for ${parameterName}:`, error);
    }
  },

  setConfig: (config) => {
    set((state) => ({
      config: { ...state.config, ...config },
    }));
  },

  controlRun: async (runId, action) => {
    set({ isLoading: true, error: null });
    try {
      await apiClient.controlRun(runId, { action });
      await get().fetchRuns();
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  selectRun: (runId) => {
    const run = get().runs.find((r) => r.run_id === runId);
    set({ currentRun: run || null });
  },

  startTraining: async (runName?: string) => {
    set({ isLoading: true, error: null });
    try {
      const config = get().config;
      const result = await apiClient.createRun({
        command: "train",
        args: config,
        run_name: runName,
        background: true,
      });
      
      if (result.success) {
        await get().fetchRuns();
        set({ isLoading: false });
        return { success: true, run_id: result.run_id };
      } else {
        set({ error: result.error || "Failed to start training", isLoading: false });
        return { success: false, error: result.error };
      }
    } catch (error) {
      const errorMsg = String(error);
      set({ error: errorMsg, isLoading: false });
      return { success: false, error: errorMsg };
    }
  },
}));
