/**
 * Copyright © 2026 Wenze Wei. All Rights Reserved.
 *
 * This file is part of Xi.
 * The Xi project belongs to the Dunimd Team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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

import axios, { AxiosInstance, AxiosError } from "axios";
import type {
  ModelListResponse,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  EmbeddingRequest,
  EmbeddingResponse,
  ImageGenerationRequest,
  ImageGenerationResponse,
  AgentExecuteRequest,
  AgentExecuteResponse,
  ToolListResponse,
  ToolExecuteRequest,
  ToolExecuteResponse,
  RunListResponse,
  RunControlRequest,
  RunControlResponse,
  HandshakeResponse,
  CommandSchema,
  ParameterOptionsResponse,
} from "@/types";

export interface HandshakeState {
  sessionId: string | null;
  token: string | null;
  capabilities: string[];
  endpoints: {
    sse: string;
    api: string;
  };
  serverInfo: {
    version: string;
    uptime: number;
  };
  isConnected: boolean;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:3140";

class ApiClient {
  private client: AxiosInstance;
  private _handshakeState: HandshakeState = {
    sessionId: null,
    token: null,
    capabilities: [],
    endpoints: { sse: "", api: "" },
    serverInfo: { version: "", uptime: 0 },
    isConnected: false,
  };

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 120000,
      headers: {
        "Content-Type": "application/json",
      },
    });

    this.client.interceptors.request.use((config) => {
      if (this._handshakeState.token) {
        config.headers.set("Authorization", `Bearer ${this._handshakeState.token}`);
      }
      return config;
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401 || error.response?.status === 403) {
          this._handshakeState.isConnected = false;
        }
        return Promise.reject(error);
      }
    );
  }

  get handshakeState(): HandshakeState {
    return { ...this._handshakeState };
  }

  async handshake(clientVersion?: string): Promise<HandshakeResponse> {
    const response = await this.client.post("/handshake", {
      client_version: clientVersion,
      client_type: "web",
    });
    
    if (response.data.success) {
      this._handshakeState = {
        sessionId: response.data.session_id,
        token: null,
        capabilities: [],
        endpoints: {
          sse: API_BASE_URL,
          api: API_BASE_URL,
        },
        serverInfo: {
          version: response.data.server_version,
          uptime: 0,
        },
        isConnected: true,
      };
    }
    
    return response.data;
  }

  async getConfig(): Promise<Record<string, unknown>> {
    const response = await this.client.get("/config");
    return response.data;
  }

  async getCommands(): Promise<Record<string, CommandSchema>> {
    const response = await this.client.get("/config/commands");
    return response.data;
  }

  async executeCommand(request: {
    command: string;
    args?: Record<string, unknown>;
    run_id?: string;
    run_name?: string;
    background?: boolean;
  }): Promise<{ success: boolean; run_id?: string; message?: string; error?: string }> {
    const response = await this.client.post("/runs/execute", request);
    return response.data;
  }

  async controlRun(runId: string, request: RunControlRequest): Promise<RunControlResponse> {
    const response = await this.client.post(`/runs/${runId}/control`, request);
    return response.data;
  }

  async listRuns(): Promise<RunListResponse> {
    const response = await this.client.get("/runs");
    return response.data;
  }

  async getRunTypes(): Promise<{
    run_types: Array<{
      name: string;
      label: string;
      description: string;
      icon: string;
      color: string;
      enabled: boolean;
      order: number;
    }>;
    total: number;
  }> {
    const response = await this.client.get("/v1/runs/types");
    return response.data;
  }

  async getRunStatus(runId: string): Promise<{
    run_id: string;
    status: string;
    pid?: number;
  }> {
    const response = await this.client.get(`/runs/${runId}`);
    return response.data;
  }

  async getRunOutput(runId: string, lines?: number): Promise<{
    run_id: string;
    lines: Array<{
      timestamp: string;
      level: string;
      message: string;
      source: string;
    }>;
    total: number;
  }> {
    const response = await this.client.get(`/runs/${runId}/output`, {
      params: { lines },
    });
    return response.data;
  }

  async listModels(): Promise<ModelListResponse> {
    const response = await this.client.get("/models");
    return response.data;
  }

  async getModelInfo(modelName: string): Promise<Record<string, unknown>> {
    const response = await this.client.get(`/models/${modelName}`);
    return response.data;
  }

  async deleteModel(modelName: string): Promise<{ success: boolean; message: string }> {
    const response = await this.client.delete(`/models/${modelName}`);
    return response.data;
  }

  async listDirectory(path: string = "."): Promise<{
    path: string;
    files: Array<{
      name: string;
      path: string;
      type: string;
      size?: number;
      modified?: string;
    }>;
    total: number;
  }> {
    const response = await this.client.get("/fs/list", {
      params: { path },
    });
    return response.data;
  }

  async readFile(path: string): Promise<{ path: string; content: string }> {
    const response = await this.client.get("/fs/read", {
      params: { path },
    });
    return response.data;
  }

  async writeFile(path: string, content: string = ""): Promise<{ success: boolean; path: string }> {
    const response = await this.client.post("/fs/write", null, {
      params: { path, content },
    });
    return response.data;
  }

  async createDirectory(path: string): Promise<{ success: boolean; path: string }> {
    const response = await this.client.post("/fs/mkdir", null, {
      params: { path },
    });
    return response.data;
  }

  async deletePath(path: string): Promise<{ success: boolean; path: string }> {
    const response = await this.client.delete("/fs/delete", {
      params: { path },
    });
    return response.data;
  }

  async checkPathExists(path: string): Promise<{
    exists: boolean;
    is_file: boolean;
    is_dir: boolean;
  }> {
    const response = await this.client.get("/fs/exists", {
      params: { path },
    });
    return response.data;
  }

  async getSystemStats(): Promise<{
    cpu_percent: number;
    memory_percent: number;
    memory_used_gb: number;
    memory_total_gb: number;
    gpu_count: number;
    gpus: Array<{
      index: number;
      vendor: string;
      name: string;
      utilization: number;
      memory_used_gb: number;
      memory_total_gb: number;
      temperature: number;
    }>;
  }> {
    const response = await this.client.get("/system/stats");
    return response.data;
  }

  async healthCheck(): Promise<{
    status: string;
    version: string;
    connections: number;
    active_runs: number;
  }> {
    const response = await this.client.get("/health");
    return response.data;
  }

  async listNotifications(limit?: number, unreadOnly?: boolean): Promise<{
    notifications: Array<{
      id: string;
      type: string;
      title: string;
      message: string;
      timestamp: string;
      read: boolean;
    }>;
    total: number;
    unread_count: number;
  }> {
    const response = await this.client.get("/notifications", {
      params: { limit, unread_only: unreadOnly },
    });
    return response.data;
  }

  async createNotification(request: {
    type: string;
    title: string;
    message: string;
    metadata?: Record<string, unknown>;
  }): Promise<Record<string, unknown>> {
    const response = await this.client.post("/notifications", request);
    return response.data;
  }

  async markNotificationAsRead(notificationId: string): Promise<{ success: boolean; message: string }> {
    const response = await this.client.post(`/notifications/${notificationId}/read`);
    return response.data;
  }

  async markAllNotificationsAsRead(): Promise<{ success: boolean; message: string }> {
    const response = await this.client.post("/notifications/read-all");
    return response.data;
  }

  async deleteNotification(notificationId: string): Promise<{ success: boolean; message: string }> {
    const response = await this.client.delete(`/notifications/${notificationId}`);
    return response.data;
  }

  async clearAllNotifications(): Promise<{ success: boolean; message: string }> {
    const response = await this.client.delete("/notifications");
    return response.data;
  }

  getSSEUrl(): string {
    return `${API_BASE_URL}/sse/connect`;
  }

  getBaseUrl(): string {
    return API_BASE_URL;
  }

  getHandshakeState(): HandshakeState {
    return { ...this._handshakeState };
  }

  async getStats(): Promise<{
    cpu_percent: number;
    memory_percent: number;
    memory_used_gb: number;
    memory_total_gb: number;
    gpu_count: number;
    gpus: Array<{
      index: number;
      vendor: string;
      name: string;
      utilization: number;
      memory_used_gb: number;
      memory_total_gb: number;
      temperature: number;
    }>;
    qps?: number;
  }> {
    const response = await this.client.get("/system/stats");
    return response.data;
  }

  async getDrives(): Promise<{
    is_windows: boolean;
    drives: Array<{
      name: string;
      path: string;
      total: number;
      used: number;
      free: number;
    }>;
  }> {
    const response = await this.client.get("/fs/drives");
    return response.data;
  }

  async getDirectory(path: string): Promise<{
    path: string;
    items: Array<{
      name: string;
      path: string;
      is_dir: boolean;
      size: number;
      modified: string;
    }>;
    disk?: {
      total: number;
      used: number;
      free: number;
    } | null;
    is_windows: boolean;
  }> {
    const response = await this.client.get("/fs/list", {
      params: { path },
    });
    return response.data;
  }

  async deleteItem(path: string): Promise<{ success: boolean; message: string; error?: string }> {
    const response = await this.client.delete("/fs/delete", {
      params: { path },
    });
    return response.data;
  }

  async renameItem(oldPath: string, newPath: string): Promise<{ success: boolean; message: string; error?: string }> {
    const response = await this.client.post("/fs/rename", null, {
      params: { old_path: oldPath, new_path: newPath },
    });
    return response.data;
  }

  async createFolder(path: string): Promise<{ success: boolean; message: string; error?: string }> {
    const response = await this.client.post("/fs/mkdir", null, {
      params: { path },
    });
    return response.data;
  }

  async createFile(path: string): Promise<{ success: boolean; message: string; error?: string }> {
    const response = await this.client.post("/fs/touch", null, {
      params: { path },
    });
    return response.data;
  }

  async copyItem(srcPath: string, destPath: string): Promise<{ success: boolean; message: string; error?: string }> {
    const response = await this.client.post("/fs/copy", null, {
      params: { src: srcPath, dest: destPath },
    });
    return response.data;
  }

  async moveItem(srcPath: string, destPath: string): Promise<{ success: boolean; message: string; error?: string }> {
    const response = await this.client.post("/fs/move", null, {
      params: { src: srcPath, dest: destPath },
    });
    return response.data;
  }

  async checkFirstLaunch(signal?: AbortSignal): Promise<{ is_first_launch: boolean }> {
    const response = await this.client.get("/v1/xi/first-launch", {
      signal,
      timeout: 5000,
    });
    return response.data;
  }

  async completeFirstLaunch(): Promise<{ success: boolean }> {
    const response = await this.client.post("/v1/xi/complete-first-launch");
    return response.data;
  }

  async getCommandSchema(command: string): Promise<{
    command: string;
    description: string;
    available: boolean;
    unavailable_reason: string;
    tabs: Array<{
      name: string;
      label: string;
      available: boolean;
      unavailable_reason: string;
      parameters?: string[];
    }>;
    parameters: Array<{
      name: string;
      type: string;
      description: string;
      required?: boolean;
      default?: unknown;
      source?: string;
      source_type?: string;
      available?: boolean;
      unavailable_reason?: string;
      min?: number;
      max?: number;
      filter?: string;
      tab?: string;
      options?: string[];
    }>;
  }> {
    const response = await this.client.get(`/commands/${command}/schema`);
    return response.data;
  }

  async getParameterOptions(command: string, parameterName: string): Promise<{
    parameter: string;
    options: Array<{
      label: string;
      value: string;
    }>;
  }> {
    const response = await this.client.get(`/commands/${command}/parameters/${parameterName}/options`);
    return response.data;
  }

  async chatCompletion(request: {
    model: string;
    messages: Array<{ role: string; content: unknown }>;
    temperature?: number;
    max_tokens?: number;
  }): Promise<{
    id: string;
    choices: Array<{
      index: number;
      message: {
        role: string;
        content: string;
      };
      finish_reason: string;
    }>;
    usage: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
    };
  }> {
    const response = await this.client.post("/v1/chat/completions", request);
    return response.data;
  }

  async *streamChatCompletion(request: {
    model: string;
    messages: Array<{ role: string; content: unknown }>;
    temperature?: number;
    max_tokens?: number;
    stream?: boolean;
  }): AsyncGenerator<string, void, unknown> {
    const response = await fetch(`${API_BASE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(this._handshakeState.token && { Authorization: `Bearer ${this._handshakeState.token}` }),
      },
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.body) {
      throw new Error("No response body");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      yield chunk;
    }
  }
}

export const apiClient = new ApiClient();
