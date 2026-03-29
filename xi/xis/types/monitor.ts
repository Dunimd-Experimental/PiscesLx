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

export interface SystemStats {
  cpu_percent: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  gpu_count: number;
  gpu_utilization: number[];
  gpu_memory_used: number[];
  gpu_memory_total: number[];
  gpu_vendors: string[];
  gpu_names: string[];
  gpu_temperatures: number[];
  gpu_power_draw: number[];
  uptime_seconds: number;
  request_count: number;
  qps: number;
}

export interface GPUStats {
  id: number;
  name: string;
  utilization: number;
  memory_used: number;
  memory_total: number;
  temperature: number;
  power_draw: number;
  power_limit: number;
}

export interface MemoryStats {
  total: number;
  used: number;
  free: number;
  percent: number;
}

export interface CPUStats {
  percent_total: number;
  percent_per_core: number[];
  count: number;
}

export interface DiskStats {
  total: number;
  used: number;
  free: number;
  percent: number;
}

export interface MonitorStats {
  cpu: CPUStats;
  memory: MemoryStats;
  disk: DiskStats;
  gpu: GPUStats[];
  network?: {
    bytes_sent: number;
    bytes_recv: number;
  };
}

export interface Alert {
  id: string;
  type: "warning" | "error" | "info";
  message: string;
  timestamp: string;
  source: string;
  details?: Record<string, unknown>;
}
