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
import type { SystemStats, MonitorStats, GPUStats, Alert } from "@/types";
import { apiClient } from "@/lib/api/client";

interface MonitorState {
  systemStats: SystemStats | null;
  monitorStats: MonitorStats | null;
  alerts: Alert[];
  isLoading: boolean;
  error: string | null;
  autoRefresh: boolean;
  refreshInterval: number;

  fetchStats: () => Promise<void>;
  addAlert: (alert: Alert) => void;
  clearAlerts: () => void;
  setAutoRefresh: (enabled: boolean) => void;
  setRefreshInterval: (ms: number) => void;
}

export const useMonitorStore = create<MonitorState>((set, get) => ({
  systemStats: null,
  monitorStats: null,
  alerts: [],
  isLoading: false,
  error: null,
  autoRefresh: true,
  refreshInterval: 2000,

  fetchStats: async () => {
    set({ isLoading: true, error: null });
    try {
      const stats = await apiClient.getStats();
      set({ systemStats: stats as unknown as SystemStats, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  addAlert: (alert) => {
    set((state) => ({ alerts: [...state.alerts, alert] }));
  },

  clearAlerts: () => {
    set({ alerts: [] });
  },

  setAutoRefresh: (enabled) => {
    set({ autoRefresh: enabled });
  },

  setRefreshInterval: (ms) => {
    set({ refreshInterval: ms });
  },
}));
