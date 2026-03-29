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

export type RunStatus = "pending" | "running" | "paused" | "completed" | "failed" | "cancelled";

export interface RunInfo {
  run_id: string;
  run_dir: string;
  status: RunStatus;
  phase: string;
  command?: string;
  created_at: string;
  updated_at: string;
  pid: number | null;
}

export type ClientMessage =
  | { type: "start_training"; command: string; args: Record<string, unknown>; run_name?: string; run_id?: string }
  | { type: "control"; run_id: string; action: "pause" | "resume" | "cancel" | "kill" }
  | { type: "get_runs" }
  | { type: "get_schema"; command: string }
  | { type: "subscribe"; run_id: string }
  | { type: "unsubscribe"; run_id: string };

export type ServerMessage =
  | { type: "training_started"; run_id: string; command: string; argv: string[]; message: string }
  | { type: "status"; run_id: string; status: RunStatus; phase?: string }
  | { type: "output"; run_id: string; line: string; source: string; timestamp: string }
  | { type: "metrics"; run_id: string; line: string; [key: string]: unknown }
  | { type: "completed"; run_id: string; exit_code: number; status: RunStatus; timestamp: string }
  | { type: "error"; run_id?: string; command?: string; message: string }
  | { type: "runs_list"; runs: RunInfo[]; total: number }
  | { type: "control_acknowledged"; run_id: string; action: string; status: RunStatus }
  | { type: "subscribed"; run_id: string }
  | { type: "unsubscribed"; run_id: string }
  | { type: "schema"; command: string; description: string; available: boolean; unavailable_reason: string; tabs: unknown[]; parameters: unknown[] };

type MessageHandler = (message: ServerMessage) => void;
type ConnectionHandler = () => void;

export class PiscesL1TrainingWS {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: Map<string, Set<MessageHandler>> = new Map();
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
      this.url = "ws://127.0.0.1:3140/ws/training";
    }
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log(`[Training WS] Connecting to WebSocket at: ${this.url}`);
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = "arraybuffer";

        // Set connection timeout to 5 seconds
        const connectionTimeout = setTimeout(() => {
          console.error(`[Training WS] Connection timeout after 5s`);
          if (this.ws) {
            this.ws.close();
            this.ws = null;
          }
          this.onDisconnectHandlers.forEach((handler) => handler());
          reject(new Error("WebSocket connection timeout"));
        }, 5000);

        this.ws.onopen = () => {
          clearTimeout(connectionTimeout);
          this.reconnectAttempts = 0;
          console.log(`[Training WS] Connected successfully`);
          this.onConnectHandlers.forEach((handler) => handler());
          resolve();
        };

        this.ws.onclose = (event) => {
          clearTimeout(connectionTimeout);
          console.log(`[Training WS] Disconnected. Code: ${event.code}, Reason: ${event.reason}`);
          this.onDisconnectHandlers.forEach((handler) => handler());
          if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`[Training WS] Reconnecting... Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectAttempts);
          }
        };

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout);
          console.error(`[Training WS] Connection error:`, error);
          const errorMsg = error instanceof Event ? "Failed to connect to WebSocket server" : String(error);
          console.error(`[Training WS] Error message: ${errorMsg}`);
          this.onDisconnectHandlers.forEach((handler) => handler());
          if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`[Training WS] Reconnecting after error... Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectAttempts);
          }
          reject(new Error(errorMsg));
        };

        this.ws.onmessage = (event) => {
          try {
            const message: ServerMessage = JSON.parse(event.data);
            const handlers = this.handlers.get(message.type);
            if (handlers) {
              handlers.forEach((handler) => handler(message));
            }
            const allHandlers = this.handlers.get("*");
            if (allHandlers) {
              allHandlers.forEach((handler) => handler(message));
            }
          } catch (e) {
            console.error("Failed to parse WebSocket message:", e);
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

  send(message: ClientMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.error("WebSocket is not connected");
    }
  }

  on(type: ServerMessage["type"] | "*", handler: MessageHandler): void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
  }

  off(type: ServerMessage["type"] | "*", handler: MessageHandler): void {
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

  startTraining(
    command: string,
    args: Record<string, unknown> = {},
    runName?: string,
    runId?: string
  ): void {
    this.send({
      type: "start_training",
      command,
      args,
      run_name: runName,
      run_id: runId,
    });
  }

  control(runId: string, action: "pause" | "resume" | "cancel" | "kill"): void {
    this.send({
      type: "control",
      run_id: runId,
      action,
    });
  }

  getRuns(): void {
    this.send({ type: "get_runs" });
  }

  getSchema(command: string = "train"): void {
    this.send({ type: "get_schema", command });
  }

  subscribe(runId: string): void {
    this.send({ type: "subscribe", run_id: runId });
  }

  unsubscribe(runId: string): void {
    this.send({ type: "unsubscribe", run_id: runId });
  }

  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

export const trainingWS = new PiscesL1TrainingWS();
