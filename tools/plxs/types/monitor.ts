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

export interface SystemStats {
  uptime_seconds: number;
  request_count: number;
  success_count: number;
  error_count: number;
  qps: number;
  model_size: string;
  latency_p50_ms?: number;
  latency_p95_ms?: number;
  engine_status?: Record<string, unknown>;
  model_info?: Record<string, unknown>;
  run_status?: Record<string, unknown>;
  opss_status?: Record<string, unknown>;
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
