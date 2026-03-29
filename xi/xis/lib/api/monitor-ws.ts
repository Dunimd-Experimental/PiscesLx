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

import type { SystemStats } from "@/types";

export type MonitorClientMessage =
  | { type: "get_stats" };

export type MonitorServerMessage =
  | { type: "stats"; data: SystemStats; timestamp: string }
  | { type: "error"; message: string };

type MonitorMessageHandler = (message: MonitorServerMessage) => void;
type ConnectionHandler = () => void;

export class PiscesL1MonitorWS {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: Map<string, Set<MonitorMessageHandler>> = new Map();
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
      this.url = "ws://127.0.0.1:3140/ws/monitor";
    }
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log(`[Monitor WS] Connecting to WebSocket at: ${this.url}`);
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = "arraybuffer";

        this.ws.onopen = () => {
          this.reconnectAttempts = 0;
          console.log(`[Monitor WS] Connected successfully`);
          this.onConnectHandlers.forEach((handler) => handler());
          resolve();
        };

        this.ws.onclose = (event) => {
          console.log(`[Monitor WS] Disconnected. Code: ${event.code}, Reason: ${event.reason}`);
          this.onDisconnectHandlers.forEach((handler) => handler());
          if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`[Monitor WS] Reconnecting... Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectAttempts);
          }
        };

        this.ws.onerror = (error) => {
          console.error(`[Monitor WS] Connection error:`, error);
          const errorMsg = error instanceof Event ? "Failed to connect to WebSocket server" : String(error);
          console.error(`[Monitor WS] Error message: ${errorMsg}`);
          reject(new Error(errorMsg));
        };

        this.ws.onmessage = (event) => {
          try {
            const message: MonitorServerMessage = JSON.parse(event.data);
            const handlers = this.handlers.get(message.type);
            if (handlers) {
              handlers.forEach((handler) => handler(message));
            }
            const allHandlers = this.handlers.get("*");
            if (allHandlers) {
              allHandlers.forEach((handler) => handler(message));
            }
          } catch (e) {
            console.error("Failed to parse Monitor WebSocket message:", e);
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

  send(message: MonitorClientMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.error("Monitor WebSocket is not connected");
    }
  }

  on(type: MonitorServerMessage["type"] | "*", handler: MonitorMessageHandler): void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
  }

  off(type: MonitorServerMessage["type"] | "*", handler: MonitorMessageHandler): void {
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

  getStats(): void {
    this.send({ type: "get_stats" });
  }

  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

export const monitorWS = new PiscesL1MonitorWS();
