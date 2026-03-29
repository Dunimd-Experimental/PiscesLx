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
import type { SystemStats } from "@/types";
import { PiscesL1MonitorWS, monitorWS, type MonitorServerMessage } from "@/lib/api/monitor-ws";

interface MonitorState {
  stats: SystemStats | null;
  isLoading: boolean;
  error: string | null;
  ws: PiscesL1MonitorWS | null;
  isWsConnected: boolean;

  connectWebSocket: () => Promise<void>;
  disconnectWebSocket: () => void;
}

const initialStats: SystemStats = {
  cpu_percent: 0,
  memory_percent: 0,
  memory_used_gb: 0,
  memory_total_gb: 0,
  gpu_count: 0,
  gpu_utilization: [],
  gpu_memory_used: [],
  gpu_memory_total: [],
  gpu_vendors: [],
  gpu_names: [],
  gpu_temperatures: [],
  gpu_power_draw: [],
  uptime_seconds: 0,
  request_count: 0,
  qps: 0,
};

export const useMonitorStore = create<MonitorState>((set, get) => ({
  stats: initialStats,
  isLoading: false,
  error: null,
  ws: null,
  isWsConnected: false,

  connectWebSocket: async () => {
    const existingWs = get().ws;
    if (existingWs && existingWs.isConnected) {
      return;
    }

    const ws = new PiscesL1MonitorWS();

    ws.onConnect(() => {
      set({ isWsConnected: true, error: null });
      ws.getStats();
    });

    ws.onDisconnect(() => {
      set({ isWsConnected: false });
    });

    ws.on("stats", (msg: MonitorServerMessage) => {
      if (msg.type === "stats") {
        set({ stats: msg.data, isLoading: false });
      }
    });

    ws.on("error", (msg: MonitorServerMessage) => {
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
}));
