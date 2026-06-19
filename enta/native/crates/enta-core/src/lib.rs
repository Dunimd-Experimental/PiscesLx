//! Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
//!
//! This file is part of EnTA.
//! The EnTA project belongs to the Dunimd Team.
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! You may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!     http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.
//! 
//! DISCLAIMER: Users must comply with applicable AI regulations.
//! Non-compliance may result in service termination or legal liability.

pub mod diff;
pub mod embedding;
pub mod fs;
pub mod indexer;
pub mod landlock;
pub mod lsp_proto;
pub mod permission;
pub mod sandbox;
pub mod search;
pub mod shell;
pub mod simd_search;
pub mod tokenizer;

pub use diff::{apply_diff, compute_diff};
pub use embedding::{cosine_similarity, text_similarity};
pub use fs::{read_file, write_file};
pub use indexer::Bm25Index;
pub use landlock::{
    landlock_abi_version, landlock_available, landlock_full_sandbox,
    landlock_restrict_network, landlock_restrict_read_only,
    landlock_workspace_sandbox,
};
pub use lsp_proto::{
    build_content_length_header, build_lsp_message, build_lsp_notification,
    build_lsp_request, parse_diagnostics, parse_lsp_message, LspDiagnostic,
    LspMessage, LspPosition, LspRange,
};
pub use permission::{
    permission_check, permission_get_policies, permission_record_decision,
    permission_set_policies, PolicyState, PolicyValue, PermissionDecision,
};
pub use sandbox::{sandbox_execute, sandbox_read_file, sandbox_write_file};
pub use search::{glob, grep, search_codebase, SearchResult};
pub use shell::execute_shell;
pub use simd_search::{simd_contains, simd_find_all, simd_memmem};
pub use tokenizer::count_tokens;

#[cfg(feature = "embedding")]
pub use embedding::TextEmbedder;
