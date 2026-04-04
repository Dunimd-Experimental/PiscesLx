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
import type { RunInfo } from "@/types";
import { PiscesL1RunsWS, runsWS, type RunsServerMessage, type RunTypeInfo, type RunTypeSchema, type RunTypeParameter } from "@/lib/api/runs-ws";

interface RunsState {
  runs: RunInfo[];
  runTypes: RunTypeInfo[];
  schema: RunTypeSchema | null;
  schemaLoading: boolean;
  config: Record<string, unknown>;
  isLoading: boolean;
  error: string | null;
  ws: PiscesL1RunsWS | null;
  isWsConnected: boolean;
  lastCreatedRun: { run_id: string; run_type: string; name: string } | null;

  connectWebSocket: () => Promise<void>;
  disconnectWebSocket: () => void;
  controlRun: (runId: string, action: "pause" | "resume" | "cancel" | "kill") => void;
  getRunTypes: () => void;
  getSchema: (runType: string) => void;
  setConfig: (config: Record<string, unknown>) => void;
  updateConfigValue: (name: string, value: unknown) => void;
  createRun: (runType: string, name: string, runId?: string) => Promise<{ success: boolean; run_id?: string; error?: string }>;
  resetSchema: () => void;
  clearLastCreatedRun: () => void;
}

export const useRunsStore = create<RunsState>((set, get) => ({
  runs: [],
  runTypes: [],
  schema: null,
  schemaLoading: false,
  config: {},
  isLoading: false,
  error: null,
  ws: null,
  isWsConnected: false,
  lastCreatedRun: null,

  connectWebSocket: async () => {
    const existingWs = get().ws;
    if (existingWs && existingWs.isConnected) {
      return;
    }

    const ws = new PiscesL1RunsWS();

    ws.onConnect(() => {
      set({ isWsConnected: true, error: null });
      ws.getRuns();
      ws.getRunTypes();
    });

    ws.onDisconnect(() => {
      set({ isWsConnected: false });
    });

    ws.on("runs_list", (msg: RunsServerMessage) => {
      if (msg.type === "runs_list") {
        set({ runs: msg.runs, isLoading: false });
      }
    });

    ws.on("runs_update", (msg: RunsServerMessage) => {
      if (msg.type === "runs_update") {
        set({ runs: msg.runs });
      }
    });

    ws.on("run_update", (msg: RunsServerMessage) => {
      if (msg.type === "run_update") {
        set((state) => ({
          runs: state.runs.map((r) =>
            r.run_id === msg.run.run_id ? { ...r, ...msg.run } : r
          ),
        }));
      }
    });

    ws.on("run_types", (msg: RunsServerMessage) => {
      if (msg.type === "run_types") {
        set({ runTypes: msg.run_types.filter(rt => rt.enabled) });
      }
    });

    ws.on("schema", (msg: RunsServerMessage) => {
      if (msg.type === "schema") {
        const defaultConfig: Record<string, unknown> = {};
        msg.parameters.forEach((param: RunTypeParameter) => {
          if (param.default !== undefined && param.default !== null) {
            defaultConfig[param.name] = param.default;
          }
        });
        set({ schema: msg as RunTypeSchema, schemaLoading: false, config: defaultConfig });
      }
    });

    ws.on("run_created", (msg: RunsServerMessage) => {
      if (msg.type === "run_created") {
        set({ 
          schema: null, 
          config: {},
          lastCreatedRun: {
            run_id: msg.run_id,
            run_type: msg.run_type,
            name: msg.name,
          }
        });
      }
    });

    ws.on("error", (msg: RunsServerMessage) => {
      if (msg.type === "error") {
        set({ error: msg.message, schemaLoading: false });
      }
    });

    try {
      set({ isLoading: true });
      await ws.connect();
      set({ ws });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  disconnectWebSocket: () => {
    const ws = get().ws;
    if (ws) {
      ws.disconnect();
      set({ ws: null, isWsConnected: false });
    }
  },

  controlRun: (runId, action) => {
    const ws = get().ws;
    if (ws && ws.isConnected) {
      ws.control(runId, action);
    } else {
      set({ error: "WebSocket not connected" });
    }
  },

  getRunTypes: () => {
    const ws = get().ws;
    if (ws && ws.isConnected) {
      ws.getRunTypes();
    }
  },

  getSchema: (runType: string) => {
    const ws = get().ws;
    if (ws && ws.isConnected) {
      set({ schemaLoading: true, schema: null });
      ws.getSchema(runType);
    }
  },

  setConfig: (config: Record<string, unknown>) => {
    set({ config });
  },

  updateConfigValue: (name: string, value: unknown) => {
    set((state) => ({
      config: { ...state.config, [name]: value },
    }));
  },

  createRun: (runType: string, name: string, runId?: string): Promise<{ success: boolean; run_id?: string; error?: string }> => {
    return new Promise((resolve) => {
      const ws = get().ws;
      const config = get().config;
      
      if (!ws || !ws.isConnected) {
        resolve({ success: false, error: "WebSocket not connected" });
        return;
      }

      const handleCreated = (msg: RunsServerMessage) => {
        if (msg.type === "run_created") {
          ws.off("run_created", handleCreated);
          ws.off("error", handleError);
          resolve({ success: true, run_id: msg.run_id });
        }
      };

      const handleError = (msg: RunsServerMessage) => {
        if (msg.type === "error" && msg.run_type === runType) {
          ws.off("run_created", handleCreated);
          ws.off("error", handleError);
          resolve({ success: false, error: msg.message });
        }
      };

      ws.on("run_created", handleCreated);
      ws.on("error", handleError);

      ws.createRun(runType, name, config, runId);

      setTimeout(() => {
        ws.off("run_created", handleCreated);
        ws.off("error", handleError);
        resolve({ success: false, error: "Timeout waiting for response" });
      }, 30000);
    });
  },

  resetSchema: () => {
    set({ schema: null, config: {}, schemaLoading: false });
  },

  clearLastCreatedRun: () => {
    set({ lastCreatedRun: null });
  },
}));
