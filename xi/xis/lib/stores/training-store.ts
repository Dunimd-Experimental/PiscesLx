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
import { PiscesL1TrainingWS, trainingWS, type ServerMessage, type RunStatus } from "@/lib/api/training-ws";

interface TrainingOutput {
  run_id: string;
  line: string;
  source: string;
  timestamp: string;
}

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
  ws: PiscesL1TrainingWS | null;
  isWsConnected: boolean;
  outputBuffers: Record<string, TrainingOutput[]>;

  connectWebSocket: () => Promise<void>;
  disconnectWebSocket: () => void;
  fetchRuns: () => Promise<void>;
  fetchSchema: () => Promise<void>;
  fetchParameterOptions: (parameterName: string) => Promise<void>;
  setConfig: (config: Record<string, unknown>) => void;
  controlRun: (runId: string, action: "pause" | "resume" | "cancel" | "kill") => Promise<void>;
  selectRun: (runId: string) => void;
  startTraining: (runName?: string) => Promise<{ success: boolean; run_id?: string; error?: string }>;
  getOutputBuffer: (runId: string) => TrainingOutput[];
  clearOutputBuffer: (runId: string) => void;
}

const DEFAULT_CONFIG: Record<string, unknown> = {
  model_size: "7B",
  train_mode: "standard",
  seq_len: 2048,
};

const MAX_OUTPUT_BUFFER_SIZE = 1000;

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
  ws: null,
  isWsConnected: false,
  outputBuffers: {},

  connectWebSocket: async () => {
    const existingWs = get().ws;
    console.log("[TrainingStore] connectWebSocket called, existingWs:", existingWs?.isConnected);
    if (existingWs && existingWs.isConnected) {
      console.log("[TrainingStore] WebSocket already connected, skipping");
      return;
    }

    const ws = new PiscesL1TrainingWS();
    console.log("[TrainingStore] Created new PiscesL1TrainingWS instance");

    ws.onConnect(() => {
      console.log("[TrainingStore] WebSocket onConnect callback fired");
      set({ isWsConnected: true, error: null });
      ws.getRuns();
    });

    ws.onDisconnect(() => {
      console.log("[TrainingStore] WebSocket onDisconnect callback fired");
      set({ isWsConnected: false });
    });

    ws.on("training_started", (msg: ServerMessage) => {
      if (msg.type === "training_started") {
        const newRun: RunInfo = {
          run_id: msg.run_id,
          run_dir: "",
          status: "running",
          phase: "",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          pid: null,
        };
        set((state) => ({
          runs: [...state.runs.filter(r => r.run_id !== msg.run_id), newRun],
          outputBuffers: { ...state.outputBuffers, [msg.run_id]: [] },
        }));
      }
    });

    ws.on("output", (msg: ServerMessage) => {
      if (msg.type === "output") {
        const output: TrainingOutput = {
          run_id: msg.run_id,
          line: msg.line,
          source: msg.source,
          timestamp: msg.timestamp,
        };
        set((state) => {
          const buffer = state.outputBuffers[msg.run_id] || [];
          const newBuffer = [...buffer, output];
          if (newBuffer.length > MAX_OUTPUT_BUFFER_SIZE) {
            newBuffer.splice(0, newBuffer.length - MAX_OUTPUT_BUFFER_SIZE);
          }
          return {
            outputBuffers: { ...state.outputBuffers, [msg.run_id]: newBuffer },
          };
        });
      }
    });

    ws.on("status", (msg: ServerMessage) => {
      if (msg.type === "status") {
        set((state) => ({
          runs: state.runs.map((r) =>
            r.run_id === msg.run_id
              ? { ...r, status: msg.status, phase: msg.phase ?? "" }
              : r
          ),
          currentRun:
            state.currentRun?.run_id === msg.run_id
              ? { ...state.currentRun, status: msg.status, phase: msg.phase ?? "" }
              : state.currentRun,
        }));
      }
    });

    ws.on("completed", (msg: ServerMessage) => {
      if (msg.type === "completed") {
        set((state) => ({
          runs: state.runs.map((r) =>
            r.run_id === msg.run_id
              ? { ...r, status: msg.status as RunStatus }
              : r
          ),
          currentRun:
            state.currentRun?.run_id === msg.run_id
              ? { ...state.currentRun, status: msg.status as RunStatus }
              : state.currentRun,
        }));
      }
    });

    ws.on("error", (msg: ServerMessage) => {
      if (msg.type === "error") {
        set({ error: msg.message });
      }
    });

    ws.on("runs_list", (msg: ServerMessage) => {
      if (msg.type === "runs_list") {
        set((state) => {
          const existingRunIds = new Set(state.runs.map(r => r.run_id));
          const newRuns: RunInfo[] = msg.runs
            .filter(r => !existingRunIds.has(r.run_id))
            .map(r => ({
              run_id: r.run_id,
              run_dir: r.run_dir || "",
              status: r.status,
              phase: r.phase || "",
              created_at: r.created_at || new Date().toISOString(),
              updated_at: r.updated_at || new Date().toISOString(),
              pid: r.pid ?? null,
            }));
          return {
            runs: [...state.runs, ...newRuns],
          };
        });
      }
    });

    ws.on("schema", (msg: ServerMessage) => {
      if (msg.type === "schema") {
        const parameters: DynamicParameter[] = (msg.parameters || []).map((p: unknown) => {
          const param = p as Record<string, unknown>;
          let defaultValue: string | number | boolean | null = null;
          if (
            param.default !== null &&
            param.default !== undefined &&
            (typeof param.default === "string" || typeof param.default === "number" || typeof param.default === "boolean")
          ) {
            defaultValue = param.default as string | number | boolean;
          }
          return {
            name: typeof param.name === "string" ? param.name : "",
            type: typeof param.type === "string" ? param.type : "string",
            description: typeof param.description === "string" ? param.description : "",
            required: param.required === true,
            default: defaultValue,
            options: Array.isArray(param.options) ? param.options as string[] : undefined,
            min: typeof param.min === "number" ? param.min : undefined,
            max: typeof param.max === "number" ? param.max : undefined,
            source: typeof param.source === "string" ? param.source : undefined,
            source_type: typeof param.source_type === "string" ? param.source_type : undefined,
            filter: typeof param.filter === "string" ? param.filter : undefined,
            available: param.available !== false,
            unavailable_reason: typeof param.unavailable_reason === "string" ? param.unavailable_reason : "",
            tab: typeof param.tab === "string" ? param.tab : "basic",
          };
        });

        const tabs: DynamicTab[] = (msg.tabs || []).map((t: unknown) => {
          const tab = t as Record<string, unknown>;
          return {
            name: typeof tab.name === "string" ? tab.name : "",
            label: typeof tab.label === "string" ? tab.label : (tab.name as string) || "",
            available: tab.available !== false,
            unavailable_reason: typeof tab.unavailable_reason === "string" ? tab.unavailable_reason : "",
            parameters: Array.isArray(tab.parameters) ? tab.parameters : [],
          };
        });

        const defaultConfig: Record<string, unknown> = { ...DEFAULT_CONFIG };
        for (const param of parameters) {
          if (param.default !== null && param.default !== undefined && typeof param.name === "string") {
            defaultConfig[param.name] = param.default;
          }
        }

        set({
          schema: {
            command: msg.command,
            description: msg.description,
            available: msg.available,
            unavailable_reason: msg.unavailable_reason,
            tabs: tabs,
            parameters: parameters,
          },
          parameters: parameters,
          tabs: tabs,
          config: defaultConfig,
        });

        // Fetch dynamic options for parameters with sources
        for (const param of parameters) {
          if (param.source && param.source_type === "directory" && param.available !== false) {
            get().fetchParameterOptions(param.name);
          }
        }
      }
    });

    try {
      console.log("[TrainingStore] Calling ws.connect()...");
      await ws.connect();
      console.log("[TrainingStore] ws.connect() resolved successfully");
      set({ ws });
      console.log("[TrainingStore] WebSocket instance set in store");
    } catch (error) {
      console.error("[TrainingStore] Failed to connect WebSocket:", error);
      set({ error: `Failed to connect WebSocket: ${error}` });
    }
  },

  disconnectWebSocket: () => {
    const ws = get().ws;
    if (ws) {
      ws.disconnect();
      set({ ws: null, isWsConnected: false });
    }
  },

  fetchRuns: async () => {
    set({ isLoading: true, error: null });
    const ws = get().ws;
    if (ws && ws.isConnected) {
      ws.getRuns();
      set({ isLoading: false });
    } else {
      set({ error: "WebSocket not connected", isLoading: false });
    }
  },

  fetchSchema: async () => {
    set({ isLoading: true, error: null });
    const ws = get().ws;
    if (ws && ws.isConnected) {
      ws.getSchema("train");
      // Schema will be received via WebSocket message handler
      set({ isLoading: false });
    } else {
      set({ error: "WebSocket not connected", isLoading: false });
    }
  },

  fetchParameterOptions: async (parameterName: string) => {
    // TODO: Implement WebSocket-based parameter options fetching
    console.warn(`fetchParameterOptions(${parameterName}): WebSocket not implemented, skipping`);
  },

  setConfig: (config) => {
    set((state) => ({
      config: { ...state.config, ...config },
    }));
  },

  controlRun: async (runId, action) => {
    const ws = get().ws;
    if (ws && ws.isConnected) {
      ws.control(runId, action);
    } else {
      console.error("WebSocket not connected, cannot control run");
      set({ error: "WebSocket not connected" });
    }
  },

  selectRun: (runId) => {
    const run = get().runs.find((r) => r.run_id === runId);
    set({ currentRun: run || null });
    if (runId) {
      const ws = get().ws;
      if (ws && ws.isConnected) {
        ws.subscribe(runId);
      }
    }
  },

  startTraining: async (runName?: string) => {
    const ws = get().ws;
    if (!ws || !ws.isConnected) {
      set({ error: "WebSocket not connected. Please wait and try again." });
      return { success: false, error: "WebSocket not connected" };
    }

    set({ isLoading: true, error: null });
    try {
      const config = get().config;
      ws.startTraining("train", config, runName);
      set({ isLoading: false });
      return { success: true };
    } catch (error) {
      const errorMsg = String(error);
      set({ error: errorMsg, isLoading: false });
      return { success: false, error: errorMsg };
    }
  },

  getOutputBuffer: (runId) => {
    return get().outputBuffers[runId] || [];
  },

  clearOutputBuffer: (runId) => {
    set((state) => {
      const newBuffers = { ...state.outputBuffers };
      delete newBuffers[runId];
      return { outputBuffers: newBuffers };
    });
  },
}));
