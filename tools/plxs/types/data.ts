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

export interface Dataset {
  id: string;
  name: string;
  description?: string;
  type: "text" | "image-text" | "audio" | "video" | "multimodal";
  format: "jsonl" | "parquet" | "arrow" | "csv";
  size_bytes: number;
  num_samples: number;
  created_at: string;
  updated_at: string;
  tags: string[];
  metadata?: Record<string, unknown>;
}

export interface DatasetListResponse {
  datasets: Dataset[];
  total: number;
}

export interface DatasetPreview {
  columns: string[];
  rows: Record<string, unknown>[];
  total_rows: number;
}

export interface DataQualityReport {
  total_samples: number;
  valid_samples: number;
  invalid_samples: number;
  duplicate_samples: number;
  avg_length: number;
  max_length: number;
  min_length: number;
  issues: DataQualityIssue[];
}

export interface DataQualityIssue {
  type: string;
  count: number;
  examples: string[];
}

export interface UploadProgress {
  file_name: string;
  total_bytes: number;
  uploaded_bytes: number;
  percent: number;
  status: "pending" | "uploading" | "processing" | "completed" | "failed";
  error?: string;
}
