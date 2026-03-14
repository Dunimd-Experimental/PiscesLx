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
        const delta = chunk.choices[0]?.delta?.content || "";
        fullContent += delta;

        set((state) => ({
          messages: state.messages.map((msg, idx) =>
            idx === state.messages.length - 1
              ? { ...msg, content: fullContent }
              : msg
          ),
        }));
      }
    } catch (error) {
      set({ error: String(error) });
    } finally {
      set({ isLoading: false });
    }
  },
}));
