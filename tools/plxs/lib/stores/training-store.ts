import { create } from "zustand";
import type { RunInfo, RunStatus, TrainingConfig, TrainingStage } from "@/types";
import { apiClient } from "@/lib/api/client";

interface TrainingState {
  runs: RunInfo[];
  currentRun: RunInfo | null;
  config: Partial<TrainingConfig>;
  isLoading: boolean;
  error: string | null;

  fetchRuns: () => Promise<void>;
  setConfig: (config: Partial<TrainingConfig>) => void;
  setStage: (stage: TrainingStage) => void;
  controlRun: (runId: string, action: "pause" | "resume" | "cancel" | "kill") => Promise<void>;
  selectRun: (runId: string) => void;
}

const DEFAULT_CONFIG: Partial<TrainingConfig> = {
  model_name: "piscesl1-base",
  model_size: "7B",
  output_dir: ".pisceslx/ckpt",
  max_steps: 100000,
  save_steps: 1000,
  eval_steps: 500,
  log_steps: 10,
  device: "cuda",
  mixed_precision: "bf16",
  gradient_checkpointing: true,
  flash_attention: true,
  distributed: false,
  world_size: 1,
  gradient_accumulation_steps: 1,
  stage: "pretrain",
  optimizer: {
    name: "adamw",
    learning_rate: 2e-4,
    weight_decay: 0.01,
    betas: [0.9, 0.999],
    eps: 1e-8,
    max_grad_norm: 1.0,
    use_galore: false,
    galore_rank: 128,
    galore_update_proj_gap: 200,
    use_fp4: false,
    fp4_block_size: 16,
  },
  scheduler: {
    name: "cosine",
    warmup_steps: 1000,
    warmup_ratio: 0.1,
    min_lr_ratio: 0.1,
    decay_steps: null,
  },
  data: {
    batch_size: 32,
    sequence_length: 2048,
    num_workers: 4,
    pin_memory: true,
    prefetch_factor: 2,
    datasets: [],
  },
};

export const useTrainingStore = create<TrainingState>((set, get) => ({
  runs: [],
  currentRun: null,
  config: DEFAULT_CONFIG,
  isLoading: false,
  error: null,

  fetchRuns: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await apiClient.listRuns();
      set({ runs: response.runs, isLoading: false });
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  setConfig: (config) => {
    set((state) => ({
      config: { ...state.config, ...config },
    }));
  },

  setStage: (stage) => {
    set((state) => ({
      config: { ...state.config, stage },
    }));
  },

  controlRun: async (runId, action) => {
    set({ isLoading: true, error: null });
    try {
      await apiClient.controlRun(runId, { action });
      await get().fetchRuns();
    } catch (error) {
      set({ error: String(error), isLoading: false });
    }
  },

  selectRun: (runId) => {
    const run = get().runs.find((r) => r.run_id === runId);
    set({ currentRun: run || null });
  },
}));
