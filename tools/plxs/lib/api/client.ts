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
} from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:3140";

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 120000,
      headers: {
        "Content-Type": "application/json",
      },
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        const message = error.response?.data
          ? JSON.stringify(error.response.data)
          : error.message;
        console.error("API Error:", message);
        return Promise.reject(error);
      }
    );
  }

  setApiKey(apiKey: string) {
    this.client.defaults.headers.common["Authorization"] = `Bearer ${apiKey}`;
  }

  async getModels(): Promise<ModelListResponse> {
    const response = await this.client.get<ModelListResponse>("/v1/models");
    return response.data;
  }

  async chatCompletion(
    request: ChatCompletionRequest
  ): Promise<ChatCompletionResponse> {
    const response = await this.client.post<ChatCompletionResponse>(
      "/v1/chat/completions",
      request
    );
    return response.data;
  }

  async *streamChatCompletion(
    request: ChatCompletionRequest
  ): AsyncGenerator<ChatCompletionChunk, void, unknown> {
    const response = await fetch(`${API_BASE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n").filter((line) => line.startsWith("data: "));

      for (const line of lines) {
        const data = line.slice(6);
        if (data === "[DONE]") return;
        try {
          yield JSON.parse(data);
        } catch {
          continue;
        }
      }
    }
  }

  async createEmbedding(request: EmbeddingRequest): Promise<EmbeddingResponse> {
    const response = await this.client.post<EmbeddingResponse>(
      "/v1/embeddings",
      request
    );
    return response.data;
  }

  async generateImage(
    request: ImageGenerationRequest
  ): Promise<ImageGenerationResponse> {
    const response = await this.client.post<ImageGenerationResponse>(
      "/v1/images/generations",
      request
    );
    return response.data;
  }

  async executeAgent(request: AgentExecuteRequest): Promise<AgentExecuteResponse> {
    const response = await this.client.post<AgentExecuteResponse>(
      "/v1/agents/execute",
      request
    );
    return response.data;
  }

  async listTools(category?: string): Promise<ToolListResponse> {
    const params = category ? { category } : {};
    const response = await this.client.get<ToolListResponse>("/v1/tools/list", {
      params,
    });
    return response.data;
  }

  async executeTool(request: ToolExecuteRequest): Promise<ToolExecuteResponse> {
    const response = await this.client.post<ToolExecuteResponse>(
      "/v1/tools/execute",
      request
    );
    return response.data;
  }

  async listRuns(): Promise<RunListResponse> {
    const response = await this.client.get<RunListResponse>("/v1/runs");
    return response.data;
  }

  async controlRun(
    runId: string,
    request: RunControlRequest
  ): Promise<RunControlResponse> {
    const response = await this.client.post<RunControlResponse>(
      `/v1/runs/${runId}/control`,
      request
    );
    return response.data;
  }

  async getStats(): Promise<Record<string, unknown>> {
    const response = await this.client.get("/stats");
    return response.data;
  }

  async healthCheck(): Promise<{ status: string }> {
    const response = await this.client.get("/healthz");
    return response.data;
  }
}

export const apiClient = new ApiClient();
