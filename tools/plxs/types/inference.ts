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

export type MessageRole = "system" | "user" | "assistant" | "tool";

export interface ChatMessage {
  role: MessageRole;
  content: string | ContentPart[];
  name?: string;
  tool_call_id?: string;
}

export interface ContentPart {
  type: "text" | "image_url" | "audio_url" | "video_url";
  text?: string;
  image_url?: { url: string };
  audio_url?: { url: string };
  video_url?: { url: string };
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  stream?: boolean;
  stop?: string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  tools?: ToolDefinition[];
  tool_choice?: "none" | "auto" | "required";
}

export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface ChatCompletionChoice {
  index: number;
  message: ChatMessage;
  finish_reason: string;
}

export interface ChatCompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: ChatCompletionUsage;
}

export interface ChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    delta: {
      role?: MessageRole;
      content?: string;
    };
    finish_reason: string | null;
  }[];
}

export interface EmbeddingRequest {
  model: string;
  input: string | string[];
  encoding_format?: "float" | "base64";
}

export interface EmbeddingResponse {
  object: string;
  data: {
    object: string;
    embedding: number[];
    index: number;
  }[];
  model: string;
  usage: ChatCompletionUsage;
}

export interface ImageGenerationRequest {
  model: string;
  prompt: string;
  n?: number;
  size?: string;
  response_format?: "url" | "b64_json";
}

export interface ImageGenerationResponse {
  created: number;
  data: {
    b64_json?: string;
    url?: string;
  }[];
}

export interface VideoGenerationRequest {
  model: string;
  prompt: string;
  duration?: number;
  fps?: number;
}

export interface VideoGenerationResponse {
  data: {
    b64_json: string;
    duration: number;
  }[];
}

export interface AgentExecuteRequest {
  agent: string;
  task: string;
  tools?: string[];
  context?: Record<string, unknown>;
}

export interface AgentExecuteResponse {
  agent: string;
  success: boolean;
  result?: string;
  error?: string;
  metadata?: Record<string, unknown>;
}

export interface ToolInfo {
  name: string;
  description: string;
  category?: string;
  parameters?: Record<string, unknown>;
}

export interface ToolListResponse {
  tools: ToolInfo[];
  total: number;
}

export interface ToolExecuteRequest {
  tool_name: string;
  arguments: Record<string, unknown>;
}

export interface ToolExecuteResponse {
  tool_name: string;
  success: boolean;
  result?: unknown;
  error?: string;
}
