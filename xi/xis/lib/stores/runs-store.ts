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
import { PiscesL1RunsWS, runsWS, type RunsServerMessage } from "@/lib/api/runs-ws";

interface RunsState {
  runs: RunInfo[];
  isLoading: boolean;
  error: string | null;
  ws: PiscesL1RunsWS | null;
  isWsConnected: boolean;

  connectWebSocket: () => Promise<void>;
  disconnectWebSocket: () => void;
  controlRun: (runId: string, action: "pause" | "resume" | "cancel" | "kill") => void;
}

export const useRunsStore = create<RunsState>((set, get) => ({
  runs: [],
  isLoading: false,
  error: null,
  ws: null,
  isWsConnected: false,

  connectWebSocket: async () => {
    const existingWs = get().ws;
    if (existingWs && existingWs.isConnected) {
      return;
    }

    const ws = new PiscesL1RunsWS();

    ws.onConnect(() => {
      set({ isWsConnected: true, error: null });
      ws.getRuns();
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

    ws.on("error", (msg: RunsServerMessage) => {
      if (msg.type === "error") {
        set({ error: msg.message });
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
}));
