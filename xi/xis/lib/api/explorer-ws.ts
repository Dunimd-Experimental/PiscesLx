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

export interface FileItem {
  name: string;
  path: string;
  type: "file" | "directory" | "unknown";
  size?: number;
  modified?: string;
  error?: string;
}

export interface DiskInfo {
  total: number;
  used: number;
  free: number;
  is_windows: boolean;
}

export type ExplorerClientMessage =
  | { type: "browse"; path: string }
  | { type: "create_folder"; path: string }
  | { type: "create_file"; path: string; content: string }
  | { type: "delete"; path: string }
  | { type: "rename"; old_path: string; new_path: string }
  | { type: "copy"; source: string; destination: string }
  | { type: "move"; source: string; destination: string };

export type ExplorerServerMessage =
  | { type: "directory"; path: string; items: FileItem[]; disk?: DiskInfo }
  | { type: "operation_result"; success: boolean; operation: string; path?: string; error?: string }
  | { type: "error"; message: string };

type ExplorerMessageHandler = (message: ExplorerServerMessage) => void;
type ConnectionHandler = () => void;

export class PiscesL1ExplorerWS {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: Map<string, Set<ExplorerMessageHandler>> = new Map();
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
      this.url = "ws://127.0.0.1:3140/ws/fs";
    }
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log(`[Explorer WS] Connecting to WebSocket at: ${this.url}`);
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = "arraybuffer";

        this.ws.onopen = () => {
          this.reconnectAttempts = 0;
          console.log(`[Explorer WS] Connected successfully`);
          this.onConnectHandlers.forEach((handler) => handler());
          resolve();
        };

        this.ws.onclose = (event) => {
          console.log(`[Explorer WS] Disconnected. Code: ${event.code}, Reason: ${event.reason}`);
          this.onDisconnectHandlers.forEach((handler) => handler());
          if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`[Explorer WS] Reconnecting... Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectAttempts);
          }
        };

        this.ws.onerror = (error) => {
          console.error(`[Explorer WS] Connection error:`, error);
          const errorMsg = error instanceof Event ? "Failed to connect to WebSocket server" : String(error);
          console.error(`[Explorer WS] Error message: ${errorMsg}`);
          reject(new Error(errorMsg));
        };

        this.ws.onmessage = (event) => {
          try {
            const message: ExplorerServerMessage = JSON.parse(event.data);
            const handlers = this.handlers.get(message.type);
            if (handlers) {
              handlers.forEach((handler) => handler(message));
            }
            const allHandlers = this.handlers.get("*");
            if (allHandlers) {
              allHandlers.forEach((handler) => handler(message));
            }
          } catch (e) {
            console.error("Failed to parse Explorer WebSocket message:", e);
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

  send(message: ExplorerClientMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.error("Explorer WebSocket is not connected");
    }
  }

  on(type: ExplorerServerMessage["type"] | "*", handler: ExplorerMessageHandler): void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
  }

  off(type: ExplorerServerMessage["type"] | "*", handler: ExplorerMessageHandler): void {
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

  browse(path: string): void {
    this.send({ type: "browse", path });
  }

  createFolder(path: string): void {
    this.send({ type: "create_folder", path });
  }

  createFile(path: string, content: string): void {
    this.send({ type: "create_file", path, content });
  }

  deleteItem(path: string): void {
    this.send({ type: "delete", path });
  }

  renameItem(oldPath: string, newPath: string): void {
    this.send({ type: "rename", old_path: oldPath, new_path: newPath });
  }

  copyItem(source: string, destination: string): void {
    this.send({ type: "copy", source, destination });
  }

  moveItem(source: string, destination: string): void {
    this.send({ type: "move", source, destination });
  }

  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

export const explorerWS = new PiscesL1ExplorerWS();
