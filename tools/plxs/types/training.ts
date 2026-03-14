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

export type ModelSize = 
  | "0.5B" 
  | "1.5B" 
  | "7B" 
  | "32B" 
  | "64B" 
  | "70B" 
  | "128B" 
  | "314B" 
  | "671B" 
  | "1T";

export type TrainingStage = 
  | "pretrain" 
  | "continued_pretrain" 
  | "sft" 
  | "alignment_dpo" 
  | "alignment_ppo" 
  | "alignment_orpo" 
  | "specialized";

export type RunStatus = 
  | "pending" 
  | "running" 
  | "paused" 
  | "completed" 
  | "failed" 
  | "cancelled";

export interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

export interface ModelListResponse {
  data: ModelInfo[];
}

export interface OptimizerConfig {
  name: string;
  learning_rate: number;
  weight_decay: number;
  betas: [number, number];
  eps: number;
  max_grad_norm: number;
  use_galore: boolean;
  galore_rank: number;
  galore_update_proj_gap: number;
  use_fp4: boolean;
  fp4_block_size: number;
}

export interface SchedulerConfig {
  name: string;
  warmup_steps: number;
  warmup_ratio: number;
  min_lr_ratio: number;
  decay_steps: number | null;
}

export interface DataConfig {
  batch_size: number;
  sequence_length: number;
  num_workers: number;
  pin_memory: boolean;
  prefetch_factor: number;
  datasets: DatasetConfig[];
}

export interface DatasetConfig {
  name: string;
  path: string;
  weight: number;
  split: string;
}

export interface QuantizationConfig {
  enable_quantization: boolean;
  quant_method: "int4" | "int8" | "fp8" | "nf4";
  bits: number;
  group_size: number;
  symmetric: boolean;
}

export interface LoRAConfig {
  enabled: boolean;
  r: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string[] | null;
  bias: string;
  task_type: string;
}

export interface TrainingConfig {
  model_name: string;
  model_size: ModelSize;
  output_dir: string;
  max_steps: number;
  save_steps: number;
  eval_steps: number;
  log_steps: number;
  device: string;
  mixed_precision: "fp32" | "fp16" | "bf16";
  gradient_checkpointing: boolean;
  flash_attention: boolean;
  distributed: boolean;
  world_size: number;
  gradient_accumulation_steps: number;
  optimizer: OptimizerConfig;
  scheduler: SchedulerConfig;
  data: DataConfig;
  quantization: QuantizationConfig;
  lora: LoRAConfig;
  stage: TrainingStage | null;
  loss_type: string;
  response_only_loss: boolean;
  beta: number;
  reference_free: boolean;
  ppo_epochs: number;
  clip_range: number;
  kl_coef: number;
  lambda_orpo: number;
  packing: boolean;
  moe_gradient: Record<string, unknown>;
  moe: Record<string, unknown>;
  kfac: Record<string, unknown>;
  multitask: Record<string, unknown>;
  watermark: Record<string, unknown>;
  modality_scheduler: Record<string, unknown>;
  enable_dpo: boolean;
  enable_sft: boolean;
  enable_pref_align: boolean;
  enable_multitask: boolean;
  parallel_3d: Record<string, unknown>;
}

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

export interface RunListResponse {
  runs: RunInfo[];
  total: number;
}

export interface RunControlRequest {
  action: "pause" | "resume" | "cancel" | "kill";
}

export interface RunControlResponse {
  success: boolean;
  run_id: string;
  action: string;
  message: string;
}
