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

import type { RunInfo } from "@/types";

export type RunStatus = "pending" | "running" | "paused" | "completed" | "failed" | "cancelled";

export type RunsClientMessage =
  | { type: "get_runs" }
  | { type: "control"; run_id: string; action: "pause" | "resume" | "cancel" | "kill" };

export type RunsServerMessage =
  | { type: "runs_list"; runs: RunInfo[]; total: number }
  | { type: "runs_update"; runs: RunInfo[]; timestamp: string }
  | { type: "run_update"; run: RunInfo }
  | { type: "control_result"; run_id: string; action: string; success: boolean }
  | { type: "error"; run_id?: string; message: string };

type RunsMessageHandler = (message: RunsServerMessage) => void;
type ConnectionHandler = () => void;

export class PiscesL1RunsWS {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: Map<string, Set<RunsMessageHandler>> = new Map();
  private onConnectHandlers: Set<ConnectionHandler> = new Set();
  private onDisconnectHandlers: Set<ConnectionHandler> = new Set();
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectDelay: number = 1000;
  private shouldReconnect: boolean = true;

  constructor(url?: string) {
    if (url) {
      this.url = url;
    } else {
      // Connect directly to backend WebSocket server
      this.url = "ws://127.0.0.1:3140/ws/runs";
    }
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log(`[Runs WS] Connecting to WebSocket at: ${this.url}`);
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = "arraybuffer";

        this.ws.onopen = () => {
          this.reconnectAttempts = 0;
          console.log(`[Runs WS] Connected successfully`);
          this.onConnectHandlers.forEach((handler) => handler());
          resolve();
        };

        this.ws.onclose = (event) => {
          console.log(`[Runs WS] Disconnected. Code: ${event.code}, Reason: ${event.reason}`);
          this.onDisconnectHandlers.forEach((handler) => handler());
          if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`[Runs WS] Reconnecting... Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectAttempts);
          }
        };

        this.ws.onerror = (error) => {
          console.error(`[Runs WS] Connection error:`, error);
          const errorMsg = error instanceof Event ? "Failed to connect to WebSocket server" : String(error);
          console.error(`[Runs WS] Error message: ${errorMsg}`);
          reject(new Error(errorMsg));
        };

        this.ws.onmessage = (event) => {
          try {
            const message: RunsServerMessage = JSON.parse(event.data);
            const handlers = this.handlers.get(message.type);
            if (handlers) {
              handlers.forEach((handler) => handler(message));
            }
            const allHandlers = this.handlers.get("*");
            if (allHandlers) {
              allHandlers.forEach((handler) => handler(message));
            }
          } catch (e) {
            console.error("Failed to parse Runs WebSocket message:", e);
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  disconnect(): void {
    this.shouldReconnect = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(message: RunsClientMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.error("Runs WebSocket is not connected");
    }
  }

  on(type: RunsServerMessage["type"] | "*", handler: RunsMessageHandler): void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
  }

  off(type: RunsServerMessage["type"] | "*", handler: RunsMessageHandler): void {
    const handlers = this.handlers.get(type);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  onConnect(handler: ConnectionHandler): void {
    this.onConnectHandlers.add(handler);
  }

  offConnect(handler: ConnectionHandler): void {
    this.onConnectHandlers.delete(handler);
  }

  onDisconnect(handler: ConnectionHandler): void {
    this.onDisconnectHandlers.add(handler);
  }

  offDisconnect(handler: ConnectionHandler): void {
    this.onDisconnectHandlers.delete(handler);
  }

  getRuns(): void {
    this.send({ type: "get_runs" });
  }

  control(runId: string, action: "pause" | "resume" | "cancel" | "kill"): void {
    this.send({ type: "control", run_id: runId, action });
  }

  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

export const runsWS = new PiscesL1RunsWS();
