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

/**
 * Configuration Types for Xi Studio Frontend.
 *
 * These types correspond to the Python configuration dataclasses
 * in xi/xsc/config.py and are used for type-safe configuration
 * management in the frontend.
 */

/**
 * Project metadata configuration.
 */
export interface XiProjectConfig {
  name: string;
  version: string;
  backend: string;
  description?: string;
  author?: string;
}

/**
 * Path configuration for project directories.
 */
export interface XiPathsConfig {
  root: string;
  models: string;
  checkpoints: string;
  data: string;
  outputs: string;
  logs: string;
  cache: string;
  temp: string;
  configs: string;
}

/**
 * API server configuration.
 */
export interface XiApiConfig {
  host: string;
  port: number;
  cors_origins: string[];
  timeout: number;
  max_workers: number;
}

/**
 * UI configuration for frontend.
 */
export interface XiUiConfig {
  theme: "light" | "dark" | "system";
  language: string;
  sidebar_collapsed: boolean;
}

/**
 * Notification system configuration.
 */
export interface XiNotificationConfig {
  enabled: boolean;
  retention_days: number;
  max_count: number;
  sound: boolean;
}

/**
 * Command definition configuration.
 */
export interface XiCommandConfig {
  executable: string;
  script: string;
  args: string[];
  env: Record<string, string>;
  cwd: string;
  timeout: number;
  background: boolean;
  defaults: Record<string, unknown>;
}

/**
 * Main configuration container.
 */
export interface XiConfig {
  project: XiProjectConfig;
  paths: XiPathsConfig;
  api: XiApiConfig;
  ui: XiUiConfig;
  notifications: XiNotificationConfig;
  commands: Record<string, XiCommandConfig>;
}

/**
 * Notification type for the notification system.
 */
export type NotificationType = "info" | "success" | "warning" | "error";

/**
 * Notification object returned from the API.
 */
export interface XiNotification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  time: string;
  read: boolean;
  metadata?: Record<string, unknown>;
}

/**
 * Notification list response.
 */
export interface NotificationListResponse {
  notifications: XiNotification[];
  total: number;
}

/**
 * Create notification request.
 */
export interface CreateNotificationRequest {
  type: NotificationType;
  title: string;
  message: string;
  metadata?: Record<string, unknown>;
}

/**
 * Configuration API response types.
 */
export interface XiConfigResponse {
  project: XiProjectConfig;
  paths: XiPathsConfig;
  api: XiApiConfig;
  ui: XiUiConfig;
  notifications: XiNotificationConfig;
  commands: Record<string, XiCommandConfig>;
}

export interface XiPathsResponse {
  root: string;
  models: string;
  checkpoints: string;
  data: string;
  outputs: string;
  logs: string;
  cache: string;
  temp: string;
  configs: string;
}

export interface XiCommandsResponse {
  commands: Record<string, XiCommandConfig>;
  total: number;
}

export interface XiCommandResponse {
  name: string;
  executable: string;
  script: string;
  args: string[];
  env: Record<string, string>;
  cwd: string;
  timeout: number;
  background: boolean;
  defaults: Record<string, unknown>;
}
