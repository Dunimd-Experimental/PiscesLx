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

export type WelcomeClientMessage =
  | { type: "get_agreement" }
  | { type: "validate_config" }
  | { type: "setup_environment" }
  | { type: "complete_first_launch" }
  | { type: "pong" };

export type WelcomeServerMessage =
  | { type: "agreement"; content: string }
  | { type: "checking"; step: string; message: string }
  | { type: "result"; step: string; valid: boolean; error: string | null; data: Record<string, unknown> | null; warnings?: string[] }
  | { type: "done"; valid: boolean }
  | { type: "first_launch_completed"; success: boolean }
  | { type: "error"; message: string }
  | { type: "ping" };

type MessageHandler = (message: WelcomeServerMessage) => void;
type ConnectionHandler = () => void;

export class PiscesL1WelcomeWS {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private onConnectHandlers: Set<ConnectionHandler> = new Set();
  private onDisconnectHandlers: Set<ConnectionHandler> = new Set();
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 3;
  private reconnectDelay: number = 1000;
  private shouldReconnect: boolean = true;
  private isDestroyed: boolean = false;

  constructor(url?: string) {
    if (url) {
      this.url = url;
    } else {
      this.url = "ws://127.0.0.1:3140/ws/welcome";
    }
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.isDestroyed) {
        reject(new Error("WebSocket has been destroyed"));
        return;
      }

      try {
        console.log(`[Welcome WS] Connecting to WebSocket at: ${this.url}`);
        this.ws = new WebSocket(this.url);

        const connectionTimeout = setTimeout(() => {
          if (this.ws && this.ws.readyState !== WebSocket.OPEN) {
            console.error("[Welcome WS] Connection timeout");
            this.ws.close();
            reject(new Error("Connection timeout"));
          }
        }, 5000);

        this.ws.onopen = () => {
          clearTimeout(connectionTimeout);
          console.log("[Welcome WS] Connected successfully");
          this.reconnectAttempts = 0;
          this.onConnectHandlers.forEach((handler) => handler());
          resolve();
        };

        this.ws.onclose = () => {
          clearTimeout(connectionTimeout);
          console.log("[Welcome WS] Disconnected");
          this.onDisconnectHandlers.forEach((handler) => handler());

          if (!this.isDestroyed && this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`[Welcome WS] Reconnecting (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => {
              if (!this.isDestroyed) {
                this.connect().catch(console.error);
              }
            }, this.reconnectDelay * this.reconnectAttempts);
          }
        };

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout);
          console.error("[Welcome WS] Error:", error);
          reject(error);
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data) as WelcomeServerMessage;
            
            if (data.type === "ping") {
              this.send({ type: "pong" });
              return;
            }
            
            this.handleMessage(data);
          } catch (e) {
            console.error("[Welcome WS] Failed to parse message:", e);
          }
        };
      } catch (error) {
        console.error("[Welcome WS] Connection error:", error);
        reject(error);
      }
    });
  }

  private handleMessage(message: WelcomeServerMessage): void {
    const handlers = this.handlers.get(message.type);
    if (handlers) {
      handlers.forEach((handler) => handler(message));
    }

    const allHandlers = this.handlers.get("*");
    if (allHandlers) {
      allHandlers.forEach((handler) => handler(message));
    }
  }

  on(event: string, handler: MessageHandler): void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);
  }

  off(event: string, handler: MessageHandler): void {
    const handlers = this.handlers.get(event);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  onConnect(handler: ConnectionHandler): void {
    this.onConnectHandlers.add(handler);
  }

  onDisconnect(handler: ConnectionHandler): void {
    this.onDisconnectHandlers.add(handler);
  }

  send(message: WelcomeClientMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.error("[Welcome WS] Cannot send: WebSocket not connected");
    }
  }

  getAgreement(): void {
    this.send({ type: "get_agreement" });
  }

  validateConfig(): void {
    this.send({ type: "validate_config" });
  }

  setupEnvironment(): void {
    this.send({ type: "setup_environment" });
  }

  completeFirstLaunch(): void {
    this.send({ type: "complete_first_launch" });
  }

  disconnect(): void {
    this.shouldReconnect = false;
    this.isDestroyed = true;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.handlers.clear();
    this.onConnectHandlers.clear();
    this.onDisconnectHandlers.clear();
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}
