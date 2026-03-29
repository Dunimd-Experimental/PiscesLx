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

import { create } from "zustand";
import type { ChatMessage, ChatCompletionChunk } from "@/types";
import { apiClient } from "@/lib/api/client";

interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  currentModel: string;
  error: string | null;

  addMessage: (message: ChatMessage) => void;
  clearMessages: () => void;
  setModel: (model: string) => void;
  sendMessage: (content: string) => Promise<void>;
  sendMessageStream: (content: string) => Promise<void>;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isLoading: false,
  currentModel: "piscesl1-7b",
  error: null,

  addMessage: (message) => {
    set((state) => ({ messages: [...state.messages, message] }));
  },

  clearMessages: () => {
    set({ messages: [], error: null });
  },

  setModel: (model) => {
    set({ currentModel: model });
  },

  sendMessage: async (content) => {
    const { messages, currentModel, addMessage } = get();

    const userMessage: ChatMessage = { role: "user", content };
    addMessage(userMessage);

    set({ isLoading: true, error: null });

    try {
      const response = await apiClient.chatCompletion({
        model: currentModel,
        messages: [...messages, userMessage],
        temperature: 0.7,
        max_tokens: 2048,
      });

      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: response.choices[0]?.message?.content || "",
      };
      addMessage(assistantMessage);
    } catch (error) {
      set({ error: String(error) });
    } finally {
      set({ isLoading: false });
    }
  },

  sendMessageStream: async (content) => {
    const { messages, currentModel, addMessage } = get();

    const userMessage: ChatMessage = { role: "user", content };
    addMessage(userMessage);

    set({ isLoading: true, error: null });

    const assistantMessage: ChatMessage = { role: "assistant", content: "" };
    addMessage(assistantMessage);

    try {
      const stream = apiClient.streamChatCompletion({
        model: currentModel,
        messages: [...messages, userMessage],
        temperature: 0.7,
        max_tokens: 2048,
        stream: true,
      });

      let fullContent = "";

      for await (const chunk of stream) {
        try {
          const lines = chunk.split("\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6);
              if (data === "[DONE]") continue;
              const parsed = JSON.parse(data);
              const delta = parsed.choices?.[0]?.delta?.content || "";
              fullContent += delta;

              set((state) => ({
                messages: state.messages.map((msg, idx) =>
                  idx === state.messages.length - 1
                    ? { ...msg, content: fullContent }
                    : msg
                ),
              }));
            }
          }
        } catch {
          // Skip invalid JSON chunks
        }
      }
    } catch (error) {
      set({ error: String(error) });
    } finally {
      set({ isLoading: false });
    }
  },
}));
